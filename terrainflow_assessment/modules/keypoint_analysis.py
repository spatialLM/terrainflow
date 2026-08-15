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
from shapely.affinity import translate
from shapely.geometry import LineString

from terrainflow_assessment.modules.footprint import xy_to_rc


def _thin_to_centreline(mask):
    """Reduce a boolean ridge mask to single-cell-wide centrelines.

    Skeletonisation, not erosion. Erosion shrinks a shape from every side at once, so
    three passes with a 3×3 structure delete any ridge six cells or fewer across
    outright and retreat the ends of the survivors by about three cells — and what is
    left is a thinner blob, not a centreline. On a 1 m DEM that silently discards every
    ridge under ~6 m wide, which is most of them. ``skeletonize`` preserves each
    component's topology and length instead.

    Falls back to the historical erosion only when scikit-image is unavailable.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    try:
        from skimage.morphology import skeletonize
    except Exception:
        from scipy.ndimage import binary_erosion
        thinned = mask.copy()
        for _ in range(3):
            eroded = binary_erosion(thinned, structure=np.ones((3, 3)))
            if not eroded.any():
                break
            thinned = eroded
        return thinned
    return np.asarray(skeletonize(mask), dtype=bool)


class DrainageLineAnalysis:

    def __init__(self, dem_path, acc_path, pond_path=None):
        """Read the terrain and the flow field this analysis reasons over.

        ``pond_path`` is the pond throughflow raster the analysis writes beside the
        accumulation. **Pass it whenever it exists.** Every use of ``self.acc`` in this class
        reads accumulation as *contributing area* — a keypoint's catchment is
        ``acc x cell area``, a pond site is found by walking to cells of higher ``acc``, and
        a ridge is ``acc <= 2``. Inside a contracted pond the accumulation stops meaning
        that: the pond holds its inflow and sheds it along its crest rather than threading a
        channel through itself, so the median pool cell reads **1.0 where it used to read
        22.3**. Substituting the pond's own throughput restores the reading for every use at
        once, which is why it is done here rather than masked at each of them.

        **On the ridge test it changes nothing, and that was worth measuring.** A pool is a
        hollow, so its TPI is negative and ``tpi > min_tpi_m`` excludes it whatever the
        accumulation says: on Quail Island the ridge set moves by **0 cells inside a pond**,
        against 9,325 pool cells that newly satisfy ``acc <= 2`` on their own. The two
        conditions are not independent, and reading only the second one predicts a fault
        that does not exist. Kept here because the *reading* is still wrong without it and
        the next thing to consult ``self.acc`` would inherit that.

        Values are per-cell lookups of the pond's whole throughput, so every cell of a pool
        carries the same figure. **Do not sum it over an area** — it is a lookup, not a
        distributable quantity.
        """
        with rasterio.open(dem_path) as src:
            self.dem = src.read(1).astype("float32")
            self.transform = src.transform
            self.crs = src.crs
            nodata = src.nodata
            if nodata is not None:
                self.dem[self.dem == nodata] = np.nan

        with rasterio.open(acc_path) as src:
            self.acc = src.read(1).astype("float32")

        self.pond = None
        if pond_path:
            with rasterio.open(pond_path) as src:
                pond = src.read(1).astype("float32")
            if pond.shape == self.acc.shape:
                self.pond = pond > 0
                self.acc = np.where(self.pond, pond, self.acc)

        self.cell_w = abs(self.transform.a)
        self.cell_h = abs(self.transform.e)
        self.cell_size = (self.cell_w + self.cell_h) / 2
        self._slope_deg = None  # computed lazily

    # ---------------------------------------------------------------------- helpers

    def _rc_to_xy(self, row, col):
        """Map coordinates of the CENTRE of cell (row, col).

        ``transform.e`` is negative, so the half-cell offset that centres the sample
        must follow it downward — ``(row + 0.5) * e``. Adding ``+cell_h/2`` instead put
        every keypoint, ridgeline and pond site one full cell north of its own cell,
        and disagreed with the identical conversion used by the thalweg/keyline path.
        """
        x = self.transform.c + (col + 0.5) * self.transform.a
        y = self.transform.f + (row + 0.5) * self.transform.e
        return float(x), float(y)

    def _compute_slope_deg(self):
        """Slope in degrees, computed once and cached.

        Uses the shared Horn's-method helper so contour, keypoint and slope-raster
        tools all agree numerically (was a bespoke 2-cell gradient here)."""
        if self._slope_deg is None:
            from .dem_loader import slope_degrees
            self._slope_deg = slope_degrees(self.dem, self.cell_w, self.cell_h)
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
        Find candidate valley water-retention points — gentle valley cells carrying a
        large catchment, spread across the site.

        NOTE: this is a *heuristic proxy*, not the strict Yeomans keypoint. It scores
        "big catchment on gentle ground" rather than the steep→gentle inflection of a
        valley long-profile. It is well suited to seeding pond/dam-site candidates
        (see :meth:`recommend_pond_sites`). The strict single Yeomans keypoint (the
        long-profile inflection) is computed by
        :meth:`YeomansKeylineAnalysis.find_keypoint` and drives the keyline feature.

        Algorithm
        ---------
        1.  Identify valley cells (accumulation >= min_acc_cells).
        2.  Smooth slope to remove pixel-level noise.
        3.  Within valleys, score each cell by high accumulation (large catchment) /
            low smoothed slope (gentle angle). The highest score = best candidate.
        4.  Iteratively select points with a minimum spatial separation so they
            span the full elevation range of the site.

        Returns list of dicts: {x, y, elevation, slope_deg, catchment_ha, label,
                                 _row, _col}
        """
        from scipy.ndimage import uniform_filter

        slope = self._compute_slope_deg()
        acc = self.acc
        rows, cols = acc.shape

        # Smooth slope over ~10 m neighbourhood (min 3×3 window). NaN-aware: slope is
        # NaN over nodata, and a plain uniform_filter would smear that NaN across the
        # whole window and from there into every keypoint score.
        win = max(3, int(10.0 / self.cell_size) | 1)  # keep odd
        valid_slope = np.isfinite(slope)
        slope_sum = uniform_filter(
            np.where(valid_slope, slope, 0.0).astype("float32"), size=win)
        slope_cnt = uniform_filter(valid_slope.astype("float32"), size=win)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope_smooth = np.where(slope_cnt > 0, slope_sum / slope_cnt, np.nan)

        # Valley cells: significant upstream area, and terrain we actually know about
        valley = (acc >= min_acc_cells) & np.isfinite(slope_smooth)
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
        These are thinned to centrelines and vectorised into polylines.

        ``tpi_window`` is a **cell** count, so the landform scale it responds to
        depends on the DEM's resolution — 15 cells is 15 m on a 1 m grid and 75 m on a
        5 m grid. That is deliberate (it keeps the cost fixed) but it means the ridge
        set is not comparable between DEMs of different resolution.

        Parameters
        ----------
        tpi_window   : int   — neighbourhood window size (cells) for TPI
        min_tpi_m    : float — minimum TPI (m) for a cell to count as a ridge
        min_length_m : float — minimum ridge segment length to keep

        Returns list of dicts: {geometry (LineString), length_m, mean_elevation, label}
        """
        from scipy.ndimage import label as nd_label
        from scipy.ndimage import uniform_filter

        # Neighbourhood mean over the VALID cells only. Substituting the whole-DEM mean
        # for nodata dragged the local mean toward it for every cell within half a window
        # of a hole, fabricating a ridge line all the way around the data boundary — the
        # one place users most often clip to (a property edge).
        valid = np.isfinite(self.dem)
        filled = np.where(valid, self.dem, 0.0).astype("float64")
        sum_filter = uniform_filter(filled, size=tpi_window)
        count_filter = uniform_filter(valid.astype("float64"), size=tpi_window)
        with np.errstate(invalid="ignore", divide="ignore"):
            neighbourhood_mean = np.where(count_filter > 0,
                                          sum_filter / count_filter, np.nan)
        tpi = np.where(valid, filled - neighbourhood_mean, np.nan)

        with np.errstate(invalid="ignore"):
            ridge_raw = (tpi > min_tpi_m) & (self.acc <= 2) & valid

        # Remove 1-cell border (often artefacts)
        ridge_raw[[0, -1], :] = False
        ridge_raw[:, [0, -1]] = False

        # Apply boundary mask — only detect ridges inside the site boundary
        if boundary_mask is not None:
            ridge_raw &= boundary_mask

        if not ridge_raw.any():
            return []

        skeleton = _thin_to_centreline(ridge_raw)
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

        lines.sort(key=lambda ln: ln["length_m"], reverse=True)
        return lines[:30]  # cap: keep only the 30 longest segments

    # ---------------------------------------------------------------------- pond sites

    def recommend_pond_sites(self, keypoints, boundary_mask=None):
        """
        For each keypoint, recommend a dam/pond location just downstream where the
        valley is at its narrowest (smallest cross-sectional width at dam crest).

        The trial dam crest used to measure valley width is set 2 m above the
        *candidate cell's own* elevation — not above the keypoint's, as this once
        claimed. The two differ by the fall between keypoint and dam site.

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

    #: How far either side of a candidate the valley is measured, in **metres**.
    #:
    #: It was 200 cells, which is 400 m on a 2 m DEM and 50 m on a 0.25 m one — the
    #: same constant meaning two entirely different questions depending on the
    #: survey. A valley is a physical width, so the reach is one too.
    _CROSS_SCAN_M = 400.0

    def _valley_cross_width(self, row, col, fill_elev):
        """
        Estimate valley width at (row, col) by scanning left and right along the
        same row and counting cells at or below fill_elev.

        The clamp matters and used to be a ``break``: the scan starts 200 columns to the
        *left*, so stopping there ended it on its first iteration for any candidate within
        200 columns of the west edge — width 0 m, which maximises the ``acc/(width+1)`` dam
        score and pulled every pond-site recommendation to the raster's left edge.

        Counted with numpy rather than by walking 401 cells in Python. This is called once
        per candidate — ``n_keypoints x (2 x search_r + 1)^2`` times, which is tens of
        thousands on a fine grid — and it was the entire cost of ``recommend_pond_sites``
        (0.047 s of 0.054 s profiled, in only 120 calls; 0.001 s after). NaN compares
        False without warning, so nodata is excluded exactly as the explicit ``isnan``
        test did — asserted against the original loop in
        ``TestValleyCrossWidthEquivalence``.
        """
        _, cols = self.dem.shape
        reach = max(1, int(round(self._CROSS_SCAN_M / max(self.cell_w, 1e-9))))
        lo = max(0, col - reach)
        hi = min(cols, col + reach + 1)
        # No guard for hi <= lo: an inverted slice is empty and already counts zero.
        band = self.dem[row, lo:hi] <= fill_elev

        # **Contiguous with the candidate**, not every below-fill cell in the window.
        # Counting the whole window folded a separate gully two hundred cells away
        # into this dam's wall, and the score is `acc / (width + 1)` — so an
        # unrelated hollow made a good site look like a bad one. Walk out from the
        # candidate in both directions and stop at the first cell that stands above
        # the crest, which is where the wall would actually end.
        here = col - lo
        if here < 0 or here >= band.size or not band[here]:
            return 0.0
        left = band[:here][::-1]
        right = band[here + 1:]
        stop_l = int(np.argmin(left)) if (~left).any() else left.size
        stop_r = int(np.argmin(right)) if (~right).any() else right.size
        return float(1 + stop_l + stop_r) * self.cell_w

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

    1. Extract the primary thalweg (highest-accumulation path from source to
       outlet, walked on elevation + accumulation rather than on the flow-direction
       codes — see :meth:`_trace_thalweg`).
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
        Pre-computed flow-direction raster. Requested from pysheds as D-infinity
        (continuous radians), so it is NOT the ESRI D8 code set this once claimed;
        the thalweg walk does not read it, and any future consumer must handle the
        radian encoding rather than casting it to integer codes.
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

        # Primary thalweg = drainage path traced upstream from the outlet
        # (the cell with maximum accumulation).
        outlet_r, outlet_c = np.unravel_index(
            int(np.argmax(acc_arr)), acc_arr.shape
        )
        thalweg = self._trace_thalweg(fdir_arr, acc_arr, outlet_r, outlet_c)
        if len(thalweg) < 5:
            return None

        # Elevation profile and arc-length vector along thalweg. Nodata cells are
        # dropped and bridged by interpolation, never substituted with 0.0 m: a single
        # sea-level stand-in on a 300 m hillside is a 300 m cliff in the profile, and
        # the keypoint is the argmax of its *second* derivative — so one nodata cell
        # could capture the answer outright.
        arc_all = [0.0]
        for i in range(1, len(thalweg)):
            dr = thalweg[i][0] - thalweg[i - 1][0]
            dc = thalweg[i][1] - thalweg[i - 1][1]
            arc_all.append(arc_all[-1] + (dr ** 2 + dc ** 2) ** 0.5 * self.cell_size)
        total_len = arc_all[-1]

        arc, elevs = [], []
        for s, (r, c) in zip(arc_all, thalweg):
            z = float(self.dem[r, c])
            if np.isfinite(z):
                arc.append(s)
                elevs.append(z)
        if len(arc) < 5:
            return None            # too little real ground along the thalweg to read

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

    def get_cultivation_runs(self, keypoint, n_runs=3, cross_grade=1 / 500,
                             spacing_m=None):
        """
        Generate the Yeomans keyline + parallel cultivation guides.

        Per Yeomans' method (``The Keyline Plan``, 1954) the **keyline** is the
        on-contour line through the keypoint — it follows the valley shape at the
        keypoint elevation. **Cultivation guides** are geometric *parallel offsets*
        of that keyline, above and below. Because a parallel offset of a curved
        valley contour is not itself a contour, the guides drift off-contour
        automatically, moving water from the wet valley floor toward the drier
        ridge (the purpose of keyline cultivation). The drift is emergent from
        parallelism, so no artificial cross-grade is imposed on the geometry;
        ``cross_grade`` is retained as advisory metadata (the intended slight
        irrigation guide-grade) and echoed on every run.

        Parameters
        ----------
        keypoint : dict
            Output from :meth:`find_keypoint`.
        n_runs : int
            Number of guides above/below the keyline (total = 2 × n_runs + 1).
        cross_grade : float
            Advisory guide-grade metadata (e.g. 1/500 = 0.002).
        spacing_m : float or None
            Horizontal spacing between parallel guides (implement/plough width).
            Defaults to ``max(3.0, 5·cell_size)``.

        Returns
        -------
        list of dict
            Each dict has keys ``elevation`` (mean DEM elevation along the guide),
            ``geometry`` (shapely 3D LineString, Z sampled from the DEM),
            ``cross_grade``, ``line_type`` ("keyline" | "cultivation_upper" |
            "cultivation_lower").
        """
        kr, kc = keypoint["row"], keypoint["col"]
        base_elev = keypoint["elevation"]
        if spacing_m is None:
            spacing_m = max(3.0, 5.0 * self.cell_size)

        # Master keyline = the contour at the keypoint elevation (valley shape).
        keyline_xy = self._trace_keyline(base_elev, kr, kc)
        if keyline_xy is None or len(keyline_xy) < 2:
            keyline_xy = self._fallback_keyline(kr, kc)
        base_line = LineString(keyline_xy)

        results = []
        for offset in range(-n_runs, n_runs + 1):
            if offset == 0:
                line2d = base_line
            else:
                line2d = self._offset_line(base_line, offset * spacing_m, kr, kc)
            if line2d is None or line2d.is_empty or line2d.length <= 0:
                continue

            pts3d, mean_elev = self._sample_z(line2d, base_elev)
            if len(pts3d) < 2:
                continue

            if offset == 0:
                line_type = "keyline"
            elif offset > 0:
                line_type = "cultivation_upper"
            else:
                line_type = "cultivation_lower"

            results.append({
                "elevation": round(mean_elev, 2),
                "geometry": LineString(pts3d),
                "cross_grade": cross_grade,
                "line_type": line_type,
            })

        return results

    # ------------------------------------------------------------------
    # Keyline geometry helpers
    # ------------------------------------------------------------------

    def _trace_keyline(self, elev, kr, kc):
        """Trace the DEM contour at *elev* and return the polyline (list of x,y)
        passing closest to the keypoint (kr, kc), or None if unavailable."""
        try:
            from skimage import measure
        except Exception:
            return None
        # find_contours excludes NaN natively, leaving contours open where the data
        # stops. Pre-filling nodata with elev−1e6 instead put a synthetic cliff around
        # the hole, so marching squares closed every contour along the data boundary —
        # and the keyline could then snap to that artefact instead of the real contour.
        dem = np.asarray(self.dem, dtype="float64")
        try:
            contours = measure.find_contours(dem, float(elev))
        except Exception:
            return None
        best, best_d = None, float("inf")
        for path in contours:
            if len(path) < 2:
                continue
            # Distance in METRES: rows and columns are not interchangeable units on a
            # non-square grid, so comparing raw row²+col² picked the wrong component.
            dy = (path[:, 0] - kr) * self.cell_h
            dx = (path[:, 1] - kc) * self.cell_w
            d = float(np.min(dy ** 2 + dx ** 2))
            if d < best_d:
                best_d, best = d, path
        if best is None:
            return None
        return [self._rc_to_xy(r, c) for r, c in best]

    def _fallback_keyline(self, kr, kc):
        """Straight contour-direction segment through the keypoint — used only if
        contour tracing is unavailable (e.g. skimage missing or tiny DEM)."""
        rows, cols = self.dem.shape
        r0, r1 = max(0, kr - 1), min(rows - 1, kr + 1)
        c0, c1 = max(0, kc - 1), min(cols - 1, kc + 1)
        dz_dr = ((float(self.dem[r1, kc]) - float(self.dem[r0, kc]))
                 / ((r1 - r0) * self.cell_h)) if r1 > r0 else 0.0
        dz_dc = ((float(self.dem[kr, c1]) - float(self.dem[kr, c0]))
                 / ((c1 - c0) * self.cell_w)) if c1 > c0 else 0.0
        # Map space, not index space: x grows with column but y grows with *decreasing*
        # row (transform.e < 0), so ∇z = (dz_dc, −dz_dr) and the contour runs
        # perpendicular to it, (−∂z/∂y, ∂z/∂x) = (dz_dr, dz_dc). The old (−dz_dc, dz_dr)
        # was −∇z — the fall line, i.e. 90° off the contour, so on a south-facing slope
        # the fallback keyline was drawn N-S where it should run E-W.
        cx, cy = dz_dr, dz_dc
        mag = (cx ** 2 + cy ** 2) ** 0.5 or 1.0
        cx, cy = cx / mag, cy / mag
        kx, ky = self._rc_to_xy(kr, kc)
        half_len = min(cols, rows) * self.cell_size / 2.0
        return [(kx - cx * half_len, ky - cy * half_len),
                (kx + cx * half_len, ky + cy * half_len)]

    def _offset_line(self, line, signed_dist, kr, kc):
        """Parallel offset of *line* by *signed_dist* (metres). Prefers shapely's
        offset_curve (constant perpendicular spacing); falls back to translating
        the line along the keypoint gradient so a guide is always produced."""
        off = None
        try:
            off = line.offset_curve(signed_dist)
        except Exception:
            try:
                side = "left" if signed_dist >= 0 else "right"
                off = line.parallel_offset(abs(signed_dist), side)
            except Exception:
                off = None
        if off is not None and not off.is_empty:
            if off.geom_type == "MultiLineString":
                off = max(off.geoms, key=lambda g: g.length)
            if off.length > 0:
                return off
        # Fallback: translate along the (down-slope) gradient direction.
        rows, cols = self.dem.shape
        r0, r1 = max(0, kr - 1), min(rows - 1, kr + 1)
        c0, c1 = max(0, kc - 1), min(cols - 1, kc + 1)
        dz_dr = ((float(self.dem[r1, kc]) - float(self.dem[r0, kc]))
                 / ((r1 - r0) * self.cell_h)) if r1 > r0 else 0.0
        dz_dc = ((float(self.dem[kr, c1]) - float(self.dem[kr, c0]))
                 / ((c1 - c0) * self.cell_w)) if c1 > c0 else 1.0
        gmag = (dz_dr ** 2 + dz_dc ** 2) ** 0.5 or 1.0
        # map dx/dy from column/row gradient; row increases downward (transform.e<0)
        ux, uy = dz_dc / gmag, -dz_dr / gmag
        return translate(line, xoff=ux * signed_dist, yoff=uy * signed_dist)

    def _sample_z(self, line, base_elev):
        """Sample DEM elevation along *line* → list of (x, y, z) plus mean z."""
        total = line.length
        n = max(2, int(total / max(self.cell_size, 1e-6)))
        pts, elevs = [], []
        for i in range(n + 1):
            p = line.interpolate(total * i / n)
            z = self._sample_dem(p.x, p.y, base_elev)
            pts.append((p.x, p.y, z))
            elevs.append(z)
        return pts, (float(np.mean(elevs)) if elevs else base_elev)

    def _sample_dem(self, x, y, default):
        row, col = xy_to_rc(self.transform, x, y)
        if 0 <= row < self.dem.shape[0] and 0 <= col < self.dem.shape[1]:
            v = self.dem[row, col]
            if not np.isnan(v):
                return float(v)
        return float(default)

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
            # float32, not int32: a supplied raster may be dinf radians as easily as
            # D8 codes, and truncating the former loses the direction entirely.
            with rasterio.open(self._fdir_path) as src:
                self._fdir_arr = src.read(1).astype("float32")
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
            # A fill, not a breach: ``breach_depressions`` does not exist in pysheds 0.5,
            # so the except branch is the one that has always run (Round 14).
            try:
                filled = grid.breach_depressions(pit_filled)
            except AttributeError:
                filled = grid.fill_depressions(pit_filled)
            inflated = grid.resolve_flats(filled)
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

            # Keep the flow direction in its native encoding. Under "dinf" this is a
            # continuous angle in radians, and casting it to int32 collapsed every
            # direction to 0–6 while turning NaN into a garbage integer (a warning at
            # every call). Nothing reads it today — _trace_thalweg walks elevation and
            # accumulation — but a future consumer must get the real values, and must
            # check `dinf_routing` before treating them as D8 codes.
            self._fdir_arr = np.asarray(fdir, dtype="float32")
            self.dinf_routing = _routing == "dinf"
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
