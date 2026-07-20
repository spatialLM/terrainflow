"""
Keyline Analysis — Yeomans Keyline Design computation.

Provides:
  - DrainageLineAnalysis  (was KeylineAnalysis): drainage-line detection, pond sites,
    cultivation elevation list.  ``KeylineAnalysis`` is a deprecated alias.
  - YeomansKeylineAnalysis: true Yeomans keyline design — extracts the primary
    thalweg, detects the steep-to-gentle inflection (keypoint), and generates
    cultivation runs with a configurable cross-grade (default 1:500).

All geometry returned as shapely objects (or plain dicts with x/y for points).
Display is handled by terrain_flow.py.
"""

import warnings

import numpy as np
import rasterio
from shapely.geometry import LineString, Point


class DrainageLineAnalysis:

    def __init__(self, dem_path, acc_path):
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype("float32")
            self.transform = src.transform
            self.crs = src.crs
            nodata = src.nodata
            if nodata is not None:
                self.dem[self.dem == nodata] = np.nan

        with rasterio.open(acc_path) as src:
            self.acc = src.read(1).astype("float32")

        self.cell_w = abs(self.transform.a)
        self.cell_h = abs(self.transform.e)
        self.cell_size = (self.cell_w + self.cell_h) / 2
        self._slope_deg = None  # computed lazily

    # ---------------------------------------------------------------------- helpers

    def _rc_to_xy(self, row, col):
        x = self.transform.c + col * self.transform.a + self.cell_w / 2
        y = self.transform.f + row * self.transform.e + self.cell_h / 2
        return float(x), float(y)

    def _compute_slope_deg(self):
        """Slope in degrees, computed once and cached."""
        if self._slope_deg is None:
            dem_safe = np.where(np.isnan(self.dem), 0.0, self.dem)
            dy, dx = np.gradient(dem_safe, self.cell_h, self.cell_w)
            self._slope_deg = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
        return self._slope_deg

    def _order_pixels(self, rc_list):
        """
        Greedily walk a set of skeleton pixels into a connected polyline.
        Returns ordered list of (row, col) tuples.
        """
        if len(rc_list) < 2:
            return rc_list

        coord_set = {(int(r), int(c)) for r, c in rc_list}

        def nbrs(r, c):
            return [
                (r + dr, c + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr or dc) and (r + dr, c + dc) in coord_set
            ]

        # Find an endpoint (≤1 neighbour) to start from; fall back to first cell
        endpoints = [rc for rc in coord_set if len(nbrs(*rc)) <= 1]
        start = endpoints[0] if endpoints else next(iter(coord_set))

        ordered = [start]
        visited = {start}
        cur = start
        while True:
            nexts = [n for n in nbrs(*cur) if n not in visited]
            if not nexts:
                break
            cur = nexts[0]
            ordered.append(cur)
            visited.add(cur)
        return ordered

    # ---------------------------------------------------------------------- keypoints

    def find_keypoints(self, min_acc_cells=500, n_keypoints=5, boundary_mask=None):
        """
        Find Yeomans keypoints — valley inflection points where slope transitions
        from steep to gentle.  The keypoint marks where the effective water-retention
        zone begins along a drainage line.

        Algorithm
        ---------
        1.  Identify valley cells (accumulation >= min_acc_cells).
        2.  Smooth slope to remove pixel-level noise.
        3.  Within valleys, score each cell by high accumulation (large catchment) /
            low smoothed slope (gentle angle). The highest score = best keypoint.
        4.  Iteratively select keypoints with a minimum spatial separation so they
            span the full elevation range of the site.

        Returns list of dicts: {x, y, elevation, slope_deg, catchment_ha, label,
                                 _row, _col}
        """
        from scipy.ndimage import uniform_filter

        slope = self._compute_slope_deg()
        acc = self.acc
        rows, cols = acc.shape

        # Smooth slope over ~10 m neighbourhood (min 3×3 window)
        win = max(3, int(10.0 / self.cell_size) | 1)  # keep odd
        slope_smooth = uniform_filter(slope.astype("float32"), size=win)

        # Valley cells: significant upstream area
        valley = acc >= min_acc_cells
        if not valley.any():
            return []

        # Candidate mask: valley cells that are below the 75th-percentile accumulation
        # (avoids placing keypoints at the very bottom of the watershed where flow
        # has already converged and there is no upslope area left to redirect).
        acc_p75 = float(np.percentile(acc[valley], 75))
        candidate = valley & (acc <= acc_p75)
        if not candidate.any():
            candidate = valley

        # Apply boundary mask — ignore cells outside the site boundary
        if boundary_mask is not None:
            candidate &= boundary_mask
            if not candidate.any():
                # Fallback: all valley cells within boundary
                candidate = valley & boundary_mask
                if not candidate.any():
                    return []

        # Score: large catchment AND gentle slope
        slope_safe = np.where(candidate, slope_smooth, np.inf)
        score = np.where(candidate, acc / (slope_safe + 1.0), 0.0)

        # Spatial separation: at least 10 % of the shorter grid dimension
        min_sep = max(10, int(min(rows, cols) * 0.10))

        kept = []
        score_work = score.copy()
        candidate_work = candidate.copy()

        while len(kept) < n_keypoints:
            if not candidate_work.any():
                break
            r, c = np.unravel_index(np.argmax(score_work), score_work.shape)
            if score_work[r, c] <= 0:
                break

            # Check spatial separation from previously selected keypoints
            too_close = any(
                (r - kr) ** 2 + (c - kc) ** 2 < min_sep ** 2
                for kr, kc in kept
            )
            if not too_close:
                kept.append((r, c))

            # Zero out a neighbourhood around this candidate regardless
            r0, c0 = max(0, r - min_sep), max(0, c - min_sep)
            r1, c1 = min(rows, r + min_sep), min(cols, c + min_sep)
            score_work[r0:r1, c0:c1] = 0
            candidate_work[r0:r1, c0:c1] = False

        results = []
        for i, (r, c) in enumerate(kept):
            x, y = self._rc_to_xy(r, c)
            elev = float(self.dem[r, c]) if not np.isnan(self.dem[r, c]) else 0.0
            slp = float(slope[r, c])
            catchment_ha = float(acc[r, c]) * self.cell_w * self.cell_h / 10_000
            results.append({
                "x": x,
                "y": y,
                "elevation": round(elev, 1),
                "slope_deg": round(slp, 1),
                "catchment_ha": round(catchment_ha, 1),
                "label": (
                    f"Keypoint {i + 1} — {elev:.0f} m, "
                    f"{catchment_ha:.0f} ha upslope, {slp:.1f}° slope"
                ),
                "_row": r,
                "_col": c,
            })

        return results

    # ---------------------------------------------------------------------- ridgelines

    def find_ridgelines(self, tpi_window=15, min_tpi_m=1.5, min_length_m=100.0, boundary_mask=None):
        """
        Find watershed divides (ridgelines) using the Topographic Position Index.

        TPI = cell elevation − neighbourhood mean elevation.
        Cells with high TPI and very low flow accumulation (acc ≤ 2) are ridge cells.
        These are thinned morphologically and vectorised into polylines.

        Parameters
        ----------
        tpi_window   : int   — neighbourhood window size (cells) for TPI
        min_tpi_m    : float — minimum TPI (m) for a cell to count as a ridge
        min_length_m : float — minimum ridge segment length to keep

        Returns list of dicts: {geometry (LineString), length_m, mean_elevation, label}
        """
        from scipy.ndimage import uniform_filter, label as nd_label, binary_erosion

        dem_safe = np.where(np.isnan(self.dem), float(np.nanmean(self.dem)), self.dem)

        neighbourhood_mean = uniform_filter(dem_safe.astype("float64"),
                                            size=tpi_window).astype("float32")
        tpi = dem_safe - neighbourhood_mean

        valid = ~np.isnan(self.dem)
        ridge_raw = (tpi > min_tpi_m) & (self.acc <= 2) & valid

        # Remove 1-cell border (often artefacts)
        ridge_raw[[0, -1], :] = False
        ridge_raw[:, [0, -1]] = False

        # Apply boundary mask — only detect ridges inside the site boundary
        if boundary_mask is not None:
            ridge_raw &= boundary_mask

        if not ridge_raw.any():
            return []

        # Thin by repeated erosion (3 passes maximum; stop if nothing left)
        skeleton = ridge_raw.copy()
        for _ in range(3):
            eroded = binary_erosion(skeleton, structure=np.ones((3, 3)))
            if not eroded.any():
                break
            skeleton = eroded

        if not skeleton.any():
            skeleton = ridge_raw

        labeled, n_regions = nd_label(skeleton)
        min_cells = max(3, int(min_length_m / self.cell_size))
        lines = []

        for region_id in range(1, n_regions + 1):
            rc = np.argwhere(labeled == region_id)
            if len(rc) < min_cells:
                continue

            ordered = self._order_pixels(rc.tolist())
            if len(ordered) < 2:
                continue

            xy = [self._rc_to_xy(r, c) for r, c in ordered]
            try:
                geom = LineString(xy)
            except Exception:
                continue

            elev_vals = [
                self.dem[r, c] for r, c in ordered if not np.isnan(self.dem[r, c])
            ]
            mean_elev = float(np.mean(elev_vals)) if elev_vals else 0.0

            lines.append({
                "geometry": geom,
                "length_m": round(geom.length, 0),
                "mean_elevation": round(mean_elev, 1),
                "label": f"Ridge — {geom.length:.0f} m",
            })

        lines.sort(key=lambda l: l["length_m"], reverse=True)
        return lines[:30]  # cap: keep only the 30 longest segments

    # ---------------------------------------------------------------------- pond sites

    def recommend_pond_sites(self, keypoints, boundary_mask=None):
        """
        For each keypoint, recommend a dam/pond location just downstream where the
        valley is at its narrowest (smallest cross-sectional width at dam crest).

        The dam crest elevation is set at keypoint elevation + 2 m.

        Returns list of dicts: {x, y, elevation, catchment_ha, dam_width_m,
                                 keypoint, label}
        """
        rows, cols = self.dem.shape
        results = []

        for i, kp in enumerate(keypoints):
            r0, c0 = kp["_row"], kp["_col"]
            kp_acc = float(self.acc[r0, c0])

            search_r = max(5, int(30.0 / self.cell_size))
            best = None
            best_score = -1.0

            for dr in range(-search_r, search_r + 1):
                for dc in range(-search_r, search_r + 1):
                    nr, nc = r0 + dr, c0 + dc
                    if not (1 <= nr < rows - 1 and 1 <= nc < cols - 1):
                        continue
                    if np.isnan(self.dem[nr, nc]):
                        continue
                    cell_acc = float(self.acc[nr, nc])
                    # Must be strictly downstream (higher acc) but not too far
                    if cell_acc <= kp_acc or cell_acc > kp_acc * 4:
                        continue

                    # Must be inside the site boundary
                    if boundary_mask is not None and not boundary_mask[nr, nc]:
                        continue

                    dam_crest = float(self.dem[nr, nc]) + 2.0
                    width = self._valley_cross_width(nr, nc, dam_crest)
                    if width <= 0:
                        continue

                    # Score: more upstream area is better; narrower valley is better
                    score = cell_acc / (width + 1.0)
                    if score > best_score:
                        best_score = score
                        best = (nr, nc, width)

            if best is None:
                # Fallback: use the keypoint itself
                best = (r0, c0, self.cell_w * 10)

            nr, nc, width = best
            x, y = self._rc_to_xy(nr, nc)
            elev = float(self.dem[nr, nc]) if not np.isnan(self.dem[nr, nc]) else kp["elevation"]
            catchment_ha = float(self.acc[nr, nc]) * self.cell_w * self.cell_h / 10_000

            results.append({
                "x": x,
                "y": y,
                "elevation": round(elev, 1),
                "catchment_ha": round(catchment_ha, 1),
                "dam_width_m": round(width, 0),
                "keypoint": i + 1,
                "label": (
                    f"Pond site {i + 1} — {catchment_ha:.0f} ha catchment, "
                    f"~{width:.0f} m dam wall"
                ),
            })

        return results

    def _valley_cross_width(self, row, col, fill_elev):
        """
        Estimate valley width at (row, col) by scanning left and right along the
        same row and counting cells at or below fill_elev.
        """
        _, cols = self.dem.shape
        count = 0
        for dc in range(-200, 201):
            nc = col + dc
            if nc < 0 or nc >= cols:
                break
            if np.isnan(self.dem[row, nc]):
                continue
            if self.dem[row, nc] <= fill_elev:
                count += 1
        return count * self.cell_w

    # ---------------------------------------------------------------------- cultivation elevations

    def get_cultivation_elevations(self, keypoints, n_each_side=2, spacing_m=5.0):
        """
        Return the list of target elevations for keylines and cultivation lines.

        For each keypoint:
          - offset 0   → the keyline (at keypoint elevation)
          - offset +1,+2,… → cultivation lines above (upslope)
          - offset -1,-2,… → cultivation lines below (downslope)

        Returns list of dicts:
            {elevation, line_type ("keyline"|"cultivation_upper"|"cultivation_lower"),
             label, keypoint_idx}
        """
        results = []
        seen_elevs = set()

        for kp_idx, kp in enumerate(keypoints):
            base = kp["elevation"]

            for offset in range(-n_each_side, n_each_side + 1):
                elev = round(base + offset * spacing_m, 2)

                # Avoid duplicate elevations across keypoints
                if elev in seen_elevs:
                    continue
                seen_elevs.add(elev)

                if offset == 0:
                    line_type = "keyline"
                    label = f"Keyline {kp_idx + 1} — {elev:.1f} m"
                elif offset > 0:
                    line_type = "cultivation_upper"
                    label = f"Cultivation {kp_idx + 1} (+{offset}) — {elev:.1f} m"
                else:
                    line_type = "cultivation_lower"
                    label = f"Cultivation {kp_idx + 1} ({offset}) — {elev:.1f} m"

                results.append({
                    "elevation": elev,
                    "line_type": line_type,
                    "label": label,
                    "keypoint_idx": kp_idx + 1,
                })

        return results


# ---------------------------------------------------------------------------
# Backward-compat alias — emit a DeprecationWarning on first instantiation
# ---------------------------------------------------------------------------

class _DeprecatedKeylineAliasMeta(type):
    def __call__(cls, *args, **kwargs):
        warnings.warn(
            "KeylineAnalysis has been renamed to DrainageLineAnalysis.  "
            "Please update your code.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__call__(*args, **kwargs)


class KeylineAnalysis(DrainageLineAnalysis, metaclass=_DeprecatedKeylineAliasMeta):
    """Deprecated alias for DrainageLineAnalysis."""
    pass


# ---------------------------------------------------------------------------
# True Yeomans keyline design
# ---------------------------------------------------------------------------

class YeomansKeylineAnalysis:
    """
    True Yeomans keyline design.

    1. Extract the primary thalweg (highest-accumulation D8 path from source
       to outlet).
    2. Sample DEM elevations along the thalweg at even spacing.
    3. Smooth the long-profile with a Savitzky–Golay filter.
    4. Detect the keypoint = location of maximum positive second derivative
       (where the slope magnitude decreases most rapidly — the steep-to-gentle
       inflection).
    5. Generate cultivation runs: contour-parallel lines with a deliberate
       ``cross_grade`` (default 1/500) so water is gently directed across
       the slope rather than flowing straight downhill.

    Parameters
    ----------
    dem_path : str
        Path to a projected DEM GeoTIFF.
    fdir_path : str or None
        Pre-computed D8 flow-direction raster (pysheds ESRI encoding).
        If None the flow direction is computed internally from the DEM.
    acc_path : str or None
        Pre-computed flow accumulation raster.  If None it is computed
        together with the flow direction.
    """

    def __init__(self, dem_path, fdir_path=None, acc_path=None):
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype("float32")
            self.transform = src.transform
            self.crs = src.crs
            nodata = src.nodata
            if nodata is not None:
                self.dem[self.dem == nodata] = np.nan

        self.cell_w = abs(self.transform.a)
        self.cell_h = abs(self.transform.e)
        self.cell_size = (self.cell_w + self.cell_h) / 2.0
        self._dem_path = dem_path
        self._fdir_path = fdir_path
        self._acc_path = acc_path
        self._fdir_arr = None
        self._acc_arr = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_keypoint(self):
        """
        Detect the Yeomans keypoint on the primary thalweg.

        Returns a dict with keys ``x``, ``y``, ``elevation``, ``row``,
        ``col``, ``arc_length_m``.  Returns *None* if the DEM is too
        small to compute a meaningful profile.
        """
        from scipy.signal import savgol_filter

        fdir_arr, acc_arr = self._ensure_flow_data()

        # Primary thalweg = D8 path traced upstream from the outlet
        # (the cell with maximum accumulation).
        outlet_r, outlet_c = np.unravel_index(
            int(np.argmax(acc_arr)), acc_arr.shape
        )
        thalweg = self._trace_thalweg(fdir_arr, acc_arr, outlet_r, outlet_c)
        if len(thalweg) < 5:
            return None

        # Elevation profile and arc-length vector along thalweg
        elevs = [
            float(self.dem[r, c]) if not np.isnan(self.dem[r, c]) else 0.0
            for r, c in thalweg
        ]
        arc = [0.0]
        for i in range(1, len(thalweg)):
            dr = thalweg[i][0] - thalweg[i - 1][0]
            dc = thalweg[i][1] - thalweg[i - 1][1]
            arc.append(arc[-1] + (dr ** 2 + dc ** 2) ** 0.5 * self.cell_size)
        total_len = arc[-1]

        # Resample to regular spacing: min(5×cell_size, 10 m)
        spacing = min(5.0 * self.cell_size, 10.0)
        n_samp = max(5, int(total_len / spacing))
        s_uni = np.linspace(0.0, total_len, n_samp)
        elev_uni = np.interp(s_uni, arc, elevs)

        # Savitzky–Golay smoothing — window scaled to ~20 % of profile length
        win = max(5, int(n_samp * 0.20) | 1)
        win = min(win, n_samp - (1 if n_samp % 2 == 0 else 0))
        if win % 2 == 0:
            win += 1
        elev_smooth = savgol_filter(elev_uni, window_length=win, polyorder=3)

        # Second derivative along arc length
        ds = total_len / max(n_samp - 1, 1)
        d2 = np.gradient(np.gradient(elev_smooth, ds), ds)

        # Keypoint = maximum positive d²E/ds² (slope easing most rapidly)
        kp_idx = int(np.argmax(d2))
        kp_s = float(s_uni[kp_idx])

        # Map back to the nearest thalweg cell
        kp_thalweg_idx = int(np.argmin(np.abs(np.array(arc) - kp_s)))
        kr, kc = thalweg[kp_thalweg_idx]
        x, y = self._rc_to_xy(kr, kc)
        elev = float(self.dem[kr, kc]) if not np.isnan(self.dem[kr, kc]) else 0.0

        return {
            "x": x,
            "y": y,
            "elevation": round(elev, 1),
            "row": kr,
            "col": kc,
            "arc_length_m": round(kp_s, 1),
        }

    def get_cultivation_runs(self, keypoint, n_runs=3, cross_grade=1 / 500):
        """
        Generate cultivation run polylines centred on *keypoint*.

        Each run is a line through the keypoint's map position that follows
        the local contour direction but slopes at *cross_grade* (rise over
        run) so water is guided gently across the hillside.

        Parameters
        ----------
        keypoint : dict
            Output from :meth:`find_keypoint`.
        n_runs : int
            Number of runs above/below the keypoint elevation (total runs
            returned = 2 × n_runs + 1).
        cross_grade : float
            Desired cross-grade (e.g. 1/500 = 0.002).

        Returns
        -------
        list of dict
            Each dict has keys ``elevation``, ``geometry`` (shapely
            LineString with Z coordinates), ``cross_grade``,
            ``line_type`` ("keyline" | "cultivation_upper" |
            "cultivation_lower").
        """
        spacing_m = 5.0 * self.cell_size  # vertical spacing between runs
        kr, kc = keypoint["row"], keypoint["col"]
        rows, cols = self.dem.shape

        # Contour direction = perpendicular to the steepest-descent vector.
        # Approximate steepest descent from DEM gradient at keypoint.
        r0, r1 = max(0, kr - 1), min(rows - 1, kr + 1)
        c0, c1 = max(0, kc - 1), min(cols - 1, kc + 1)
        dz_dr = (float(self.dem[r1, kc]) - float(self.dem[r0, kc])) / (
            (r1 - r0) * self.cell_h
        ) if r1 > r0 else 0.0
        dz_dc = (float(self.dem[kr, c1]) - float(self.dem[kr, c0])) / (
            (c1 - c0) * self.cell_w
        ) if c1 > c0 else 0.0

        # Contour direction (perpendicular to gradient): (-dz_dc, dz_dr)
        cx, cy = -dz_dc, dz_dr
        mag = (cx ** 2 + cy ** 2) ** 0.5 or 1.0
        cx, cy = cx / mag, cy / mag  # unit vector along contour

        kx, ky = self._rc_to_xy(kr, kc)
        base_elev = keypoint["elevation"]
        half_len = min(cols, rows) * self.cell_size / 2.0

        results = []
        offsets = list(range(-n_runs, n_runs + 1))
        for offset in offsets:
            elev = round(base_elev + offset * spacing_m, 2)

            # Build run line: extend half_len in each contour direction,
            # applying cross_grade drop along the run.
            # The grade drops in the direction of increasing contour distance
            # from the centre point.
            pts = []
            for frac in np.linspace(-1.0, 1.0, max(10, int(half_len))):
                dist = frac * half_len
                px = kx + cx * dist
                py = ky + cy * dist
                pz = elev - abs(dist) * cross_grade  # symmetric drop from centre
                pts.append((px, py, pz))

            geom = LineString(pts)

            if offset == 0:
                line_type = "keyline"
            elif offset > 0:
                line_type = "cultivation_upper"
            else:
                line_type = "cultivation_lower"

            results.append({
                "elevation": elev,
                "geometry": geom,
                "cross_grade": cross_grade,
                "line_type": line_type,
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rc_to_xy(self, row, col):
        x = self.transform.c + (col + 0.5) * self.transform.a
        y = self.transform.f + (row + 0.5) * self.transform.e
        return float(x), float(y)

    def _ensure_flow_data(self):
        """Return (fdir_arr, acc_arr), computing them if not yet available."""
        if self._fdir_arr is not None and self._acc_arr is not None:
            return self._fdir_arr, self._acc_arr

        import os
        import tempfile
        from pysheds.grid import Grid

        if self._fdir_path and self._acc_path:
            with rasterio.open(self._fdir_path) as src:
                self._fdir_arr = src.read(1).astype("int32")
            with rasterio.open(self._acc_path) as src:
                self._acc_arr = src.read(1).astype("float32")
            return self._fdir_arr, self._acc_arr

        # Compute from DEM
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)
        try:
            with rasterio.open(
                tmp_path, "w", driver="GTiff", dtype="float32",
                crs=self.crs, transform=self.transform,
                width=self.dem.shape[1], height=self.dem.shape[0],
                count=1, nodata=-9999.0,
            ) as dst:
                data = np.where(np.isnan(self.dem), -9999.0, self.dem)
                dst.write(data.astype("float32"), 1)

            grid = Grid.from_raster(tmp_path)
            dem_r = grid.read_raster(tmp_path)
            pit_filled = grid.fill_pits(dem_r)
            try:
                breached = grid.breach_depressions(pit_filled)
            except AttributeError:
                breached = grid.fill_depressions(pit_filled)
            inflated = grid.resolve_flats(breached)
            try:
                fdir = grid.flowdir(inflated, routing="dinf")
                _routing = "dinf"
            except TypeError:
                fdir = grid.flowdir(inflated)
                _routing = None
            try:
                acc = (grid.accumulation(fdir, routing=_routing)
                       if _routing else grid.accumulation(fdir))
            except (TypeError, AttributeError):
                acc = grid.accumulation(fdir)

            self._fdir_arr = np.array(fdir, dtype="int32")
            self._acc_arr = np.array(acc, dtype="float32")
        finally:
            os.unlink(tmp_path)

        return self._fdir_arr, self._acc_arr

    def _trace_thalweg(self, fdir_arr, acc_arr, outlet_r, outlet_c,
                       max_steps=10_000):
        """
        Walk upstream from (outlet_r, outlet_c) following the main stem.

        At each step, consider only the 8-connected neighbours that have
        HIGHER elevation than the current cell (strictly uphill = genuinely
        upstream).  Among those, pick the one with the highest accumulation
        value, which identifies the main-stem tributary.

        Using elevation as the upstream filter (rather than a lower-
        accumulation check) avoids the non-monotone accumulation artefacts
        that D-infinity routing produces in nearly-flat or convergent zones.

        Returns ordered list of (row, col) from source to outlet.
        """
        rows, cols = acc_arr.shape
        dem = self.dem
        NBRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        path = [(outlet_r, outlet_c)]
        visited = {(outlet_r, outlet_c)}
        r, c = outlet_r, outlet_c

        for _ in range(max_steps):
            best = None
            best_acc = -1.0
            cur_elev = float(dem[r, c]) if not np.isnan(dem[r, c]) else -np.inf
            for dr, dc in NBRS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in visited):
                    nbr_elev = float(dem[nr, nc]) if not np.isnan(dem[nr, nc]) else -np.inf
                    nbr_acc = float(acc_arr[nr, nc])
                    # Require the neighbour to be strictly uphill; among uphill
                    # candidates pick the one with greatest accumulation (main
                    # stem carries most water).
                    if nbr_elev > cur_elev and nbr_acc > best_acc:
                        best_acc = nbr_acc
                        best = (nr, nc)
            if best is None:
                break
            r, c = best
            visited.add((r, c))
            path.append((r, c))

        path.reverse()  # source first, outlet last
        return path
