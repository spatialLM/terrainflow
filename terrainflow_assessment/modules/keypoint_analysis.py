"""
Keyline Analysis — Yeomans Keyline Design computation.

Provides:
  - DrainageLineAnalysis  (was KeylineAnalysis): drainage-line detection, pond sites,
    cultivation elevation list.  ``KeylineAnalysis`` is a deprecated alias.
  - YeomansKeylineAnalysis: true Yeomans keyline design — traces each primary
    valley from its divide, fits the two slopes Yeomans describes to find the
    keypoint at their change, and generates cultivation runs with the drift measured.

All geometry returned as shapely objects (or plain dicts with x/y for points).
Display is handled by the QGIS tier (`qgis/controllers/contour.py`).
"""

import warnings

import numpy as np
import rasterio
from shapely.geometry import LineString

from terrainflow_assessment.modules.footprint import xy_to_rc


def _two_slope_break(s, z, min_reach):
    """Least-squares break of a continuous two-slope fit to the profile ``(s, z)``.

    Fits ``z = a + b·s + c·max(0, s − s_k)`` for every candidate ``k`` with at least
    *min_reach* samples strictly above it and strictly below it, and returns
    ``(k, grade_above, grade_below)`` for the ``k`` with the smallest residual — grades
    as fall per metre, positive downhill. Ties go to the uppermost candidate.

    Closed form through the normal equations with suffix sums, so every candidate costs
    O(1) after one pass and the whole search is one batched 3×3 solve rather than a
    least-squares call per cell. A 300-cell profile is a few hundred microseconds, which
    matters because every valley on the DEM is fitted, not only the ones that pass.
    """
    s = np.asarray(s, dtype="float64")
    z = np.asarray(z, dtype="float64")
    n = s.size
    ks = np.arange(min_reach, n - min_reach)

    # Whole-profile sums, then sums over the cells strictly below each candidate.
    s_sum, ss_sum = s.sum(), (s * s).sum()
    z_sum, sz_sum, zz_sum = z.sum(), (s * z).sum(), (z * z).sum()
    below_s = s_sum - np.cumsum(s)[ks]
    below_ss = ss_sum - np.cumsum(s * s)[ks]
    below_z = z_sum - np.cumsum(z)[ks]
    below_sz = sz_sum - np.cumsum(s * z)[ks]
    m = (n - 1 - ks).astype("float64")
    sk = s[ks]

    # The hinge column h = max(0, s - s_k) is s - s_k on the cells below and 0 above.
    h_sum = below_s - m * sk
    hh_sum = below_ss - 2.0 * sk * below_s + m * sk * sk
    hs_sum = below_ss - sk * below_s
    hz_sum = below_sz - sk * below_z

    count = ks.size
    xtx = np.empty((count, 3, 3), dtype="float64")
    xtx[:, 0, 0] = float(n)
    xtx[:, 0, 1] = xtx[:, 1, 0] = s_sum
    xtx[:, 0, 2] = xtx[:, 2, 0] = h_sum
    xtx[:, 1, 1] = ss_sum
    xtx[:, 1, 2] = xtx[:, 2, 1] = hs_sum
    xtx[:, 2, 2] = hh_sum
    xtz = np.column_stack([np.full(count, z_sum), np.full(count, sz_sum), hz_sum])

    try:
        beta = np.linalg.solve(xtx, xtz[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # A degenerate profile (repeated arc lengths) — fall back to the slow exact form.
        beta = np.empty((count, 3), dtype="float64")
        for i, k in enumerate(ks):
            design = np.column_stack([np.ones(n), s, np.maximum(0.0, s - s[k])])
            beta[i] = np.linalg.lstsq(design, z, rcond=None)[0]

    residual = zz_sum - (beta * xtz).sum(axis=1)
    best = int(np.argmin(residual))
    _a, b, c = beta[best]
    return int(ks[best]), float(-b), float(-(b + c))


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
        a ridge is a cell whose ``acc x cell area`` is under ``max_catchment_m2``. Inside a
        contracted pond the accumulation stops meaning
        that: the pond holds its inflow and sheds it along its crest rather than threading a
        channel through itself, so the median pool cell reads **1.0 where it used to read
        22.3**. Substituting the pond's own throughput restores the reading for every use at
        once, which is why it is done here rather than masked at each of them.

        **On the ridge test it changes nothing, and that was worth measuring.** A pool is a
        hollow, so its TPI is negative and ``tpi > min_tpi_m`` excludes it whatever the
        accumulation says: on Quail Island the ridge set moves by **0 cells inside a pond**,
        against 9,325 pool cells that newly satisfy the catchment bar on their own. The two
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

    def find_ridgelines(self, tpi_window_m=15.0, min_tpi_sd=1.0, min_tpi_m=None,
                        min_length_m=50.0, max_catchment_m2=20.0,
                        boundary_mask=None):
        """
        Find watershed divides (ridgelines) using the Topographic Position Index.

        TPI = cell elevation − neighbourhood mean elevation.
        Cells with high TPI and a catchment no larger than *max_catchment_m2* are
        ridge cells. These are thinned to centrelines and vectorised into polylines.

        **The TPI itself comes from ``terrain_indices.landform_tpi``, and the ridge cut
        from ``terrain_indices.landform_classes``.** This method used to carry its own
        copy of both — a line-for-line duplicate of the neighbourhood-mean code, down to
        the comment about the data boundary — and the copy had never picked up two fixes
        the library version had:

        * **The window is metres, not cells.** A cell count makes the landform scale
          silently resolution-dependent, which is the fault ``landform_tpi``'s docstring
          records as already fixed *there*. Measured consequence of the copy: on a 2 m
          DEM ``tpi_window=15`` asked a 30 m question and cleared the old 1.5 m bar, while
          the identical call on a 1 m DEM asked a 15 m question and could not — so
          ridgelines worked on the synthetic test surface and had **never** fired on a 1 m
          DEM, which is the resolution most farm LiDAR arrives at.
        * **The cut is in standard deviations of the site's own TPI**, per Weiss, not in
          absolute metres. A metre bar cannot mean the same thing on a scarp and on
          rolling pasture: the real 16 ha fixture tops out at 1.20 m of TPI, so the old
          1.5 m default excluded every cell on it before any ridge was traced.

        ``min_tpi_m`` overrides the standard-deviation rule with an absolute bar when a
        caller genuinely wants one. It defaults to ``None``, which means "use Weiss".

        **The catchment bar is an area, not a cell count.** It read ``acc <= 2`` — a
        bare count — which is the same resolution-dependence the window and the cut
        above were each already fixed for, and the third instance of it in this one
        function: 2 cells is 2 m² of contributing area on a 1 m DEM and 8 m² on a 2 m
        one, so the same ground answered differently at every resolution. The bar
        floors at one cell, because a cell contributes itself and any bar under that
        excludes every cell on earth — on a 10 m DEM 20 m² is a fifth of a cell, and
        without the floor ridgelines would go silent on every grid coarser than ~4 m.

        **20 m², measured rather than chosen.** On the reference design, with the
        connectivity below fixed, the number of drawn ridgelines and their total length
        rise with the bar to about 20 m² and then stop: 5 m² draws 7 lines / 150 m,
        10 m² draws 14 / 337 m, 20 m² draws 19 / 587 m, and past it 30 m² and 50 m²
        add only fragments — 22 / 695 m and 25 / 689 m, with the median line *falling*
        from 26.6 m to 17.7 m. 20 m² is where the longest ridge reaches its full
        157 m extent and the median peaks. ``min_length_m`` is deliberately **not**
        lowered from 50 m: the bar was never the problem.

        **This is a threshold, not a definition.** What counts as a ridge here is still
        "convex ground that sheds nearly all its own water", and that is a proxy chosen
        for being computable off a DEM rather than for being right. Revisiting it — a
        divide traced from the flow field, or a multi-scale TPI — is open work; see
        `tests_qgis/probes/p_ridgelines.py` for the evidence this default rests on.

        Parameters
        ----------
        tpi_window_m     : float — neighbourhood window for TPI, in **metres**
        min_tpi_sd       : float — ridge cut, in standard deviations of this site's TPI
        min_tpi_m        : float or None — absolute TPI bar (m); overrides *min_tpi_sd*
        min_length_m     : float — minimum ridge segment length to keep
        max_catchment_m2 : float — largest contributing area a ridge cell may carry,
                           in **square metres**; floored at one cell

        Returns list of dicts: {geometry (LineString), length_m, mean_elevation, label}
        """
        from scipy.ndimage import find_objects
        from scipy.ndimage import label as nd_label

        from terrainflow_assessment.modules.terrain_indices import (
            landform_classes,
            landform_tpi,
        )

        valid = np.isfinite(self.dem)
        tpi = landform_tpi(self.dem, self.cell_w, self.cell_h,
                           window_m=tpi_window_m)

        with np.errstate(invalid="ignore"):
            if min_tpi_m is not None:
                above = tpi > float(min_tpi_m)
            else:
                # Weiss's cut, over this site's own relief. `landform_classes` returns
                # +1 for ridge; the valley class it also finds is not wanted here.
                above = landform_classes(tpi, sd=min_tpi_sd,
                                         mask=boundary_mask) == 1
            # `self.acc` is a **count of cells**, so the bar converts. Floored at one
            # cell: a cell contributes itself, so anything under 1 selects nothing.
            acc_bar = max(1.0, float(max_catchment_m2) / (self.cell_w * self.cell_h))
            ridge_raw = above & (self.acc <= acc_bar) & valid

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

        # **Eight-connected, which is how `_order_pixels` below traverses it.**
        # `nd_label`'s default structure is the 4-connected cross, and a skeleton runs
        # diagonally wherever the ridge is not aligned to the grid — so a diagonal run
        # of N cells was labelled as N components of one cell each, and a staircase as
        # one component per tread. `min_cells` then discarded all of them. The two
        # rules sat twenty lines apart in this function and disagreed: the walker would
        # have joined what the labeller had already severed.
        #
        # It was a ceiling, not a contribution. On the reference design the count of
        # 4-connected components reaching 50 cells was **zero at every accumulation
        # bar** measured, up to and including dropping the term altogether — so no
        # setting of the thresholds this function exposes could have drawn a single
        # ridgeline. 1,462 components against 816 relabelled. See
        # `tests_qgis/probes/p_ridgelines.py`.
        labeled, n_regions = nd_label(skeleton, structure=np.ones((3, 3), dtype=int))
        min_cells = max(3, int(min_length_m / self.cell_size))
        lines = []

        # Size first, and off one pass rather than one per component. ``labeled ==
        # region_id`` is a compare over the whole grid, and the component it finds is
        # then usually thrown away: on the owner's 1139x1016 design **all 1,462** of
        # them are shorter than the 50-cell minimum, so the button spent 5.8 s
        # measuring things it discarded and returned nothing. ``bincount`` sizes every
        # component in one pass, and ``find_objects`` bounds the survivors so their
        # own cost is proportional to the component rather than to the grid.
        # ``sizes[region_id]`` *is* the old ``len(rc)``, so the filter is unchanged;
        # C-order within a bounding box plus a constant offset gives the same ``rc``
        # in the same order, so ``_order_pixels`` sees exactly what it saw before.
        sizes = np.bincount(labeled.ravel(), minlength=n_regions + 1)
        boxes = find_objects(labeled)

        for region_id in range(1, n_regions + 1):
            if sizes[region_id] < min_cells:
                continue
            box = boxes[region_id - 1]
            if box is None:
                continue
            rc = np.argwhere(labeled[box] == region_id)
            rc += (box[0].start, box[1].start)

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

        **Superseded — no longer the production path.** The "Recommend Pond Sites"
        button now goes through ``modules/impoundment_sites``, which ranks candidates by
        storage held per cubic metre of embankment: both terms measured off the DEM,
        dimensionless, and comparable between sites. The score below is
        ``acc / (width + 1)`` — a cell count over a length, flagged HAZ as KPA-20 — and
        the width feeding it is measured *along the raster row*, so an east–west valley
        has its own length read as its width. Kept for now only because its tests
        exercise several branches of this class; it should go with them.

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

    1. Find every **primary valley** — :meth:`primary_valleys`. One steepest-descent
       graph on the conditioned DEM, thresholded on its own accumulation, cut into
       order-1 links, and each link walked back up its main stem to the divide, because
       a primary valley "starts as a more or less sudden steepening of the side slope of
       a main ridge" (*Water for Every Farm*, p58), not at a channel head.
    2. Along each, fit Yeomans' **two slopes** — :meth:`keypoint_on_path_with_reason`.
       The keypoint is "the point of change in the two slopes of the primary valley"
       (p60–61): the break of a continuous two-reach straight-line fit, accepted when
       the grade above exceeds the grade below by :data:`MIN_SLOPE_EASE`.
    3. Generate cultivation runs parallel to the keyline, with the drift measured
       rather than imposed — :meth:`get_cultivation_runs`.

    :meth:`find_keypoint` is the older single-stem walk (highest-accumulation path
    from the outlet, see :meth:`_trace_thalweg`) and is kept as the controller's
    fallback; it shares the criterion.

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
        together with the flow direction. Ranks the valleys and labels the
        "N ha above" figures; the valley *network* comes from the graph below.
    routing : str
        Only consulted when no accumulation is supplied.
    conditioned_path : str or None
        The baseline's hydrologically conditioned DEM (``conditioned_dem`` in the
        baseline result — float64, carrying the DEM's own nodata). The steepest-
        descent pointer graph every valley is traced on is built from this surface.
        If None, or unreadable, or the wrong shape, the DEM is conditioned here the
        same way (`fill_pits`, `fill_depressions`, `resolve_flats_safely`) — see
        :meth:`_ensure_conditioned`, and ``conditioned_source`` afterwards.
    """

    def __init__(self, dem_path, fdir_path=None, acc_path=None, routing="dinf",
                 conditioned_path=None):
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
        self._nodata = nodata
        self._fdir_path = fdir_path
        self._acc_path = acc_path
        self._conditioned_path = conditioned_path
        # Only consulted when no accumulation is supplied — see `_ensure_flow_data`.
        # When one is, its routing is already baked into the raster and this is ignored.
        self._routing = routing
        self._fdir_arr = None
        self._acc_arr = None
        #: The conditioned surface the pointer graph is built on (float64, NaN where
        #: the DEM is NaN) and where it came from: "supplied" or "recomputed".
        self._conditioned = None
        self.conditioned_source = None
        self._graph = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    #: Minimum easing of the valley floor, in metres of fall per metre, for a break to
    #: count as a keypoint: the grade of the reach **above** the break must exceed the
    #: grade of the reach **below** it by at least this much.
    #:
    #: A least-squares fit always returns a best break, just as an ``argmax`` always
    #: returned a peak. On a uniform-gradient valley there is *no* keypoint — Yeomans'
    #: break from steeper above to gentler below simply is not there — and on a valley
    #: that steepens downhill (a "nosed over" ridge, *Water for Every Farm* p44) the two
    #: slopes are the wrong way round. Both fit with an easing at or below zero and are
    #: refused. The 2 % value is a TerrainFlow convention (`KPA-49`), unchanged when the
    #: criterion moved from a local second derivative to the two-slope fit on 2026-09-11.
    MIN_SLOPE_EASE = 0.02          # 2 % — a 1:50 change in grade

    #: Fewest cells a reach may have on either side of a candidate break. A one-cell
    #: reach has no slope to fit; three is the smallest count that gives the line a
    #: residual to be judged by. A TerrainFlow convention (ANALYSIS_DEFECTS §0.3), not a
    #: Yeomans number.
    MIN_REACH_CELLS = 3

    #: Why a valley was refused, as constants so a caller can match on them rather than
    #: parse a sentence. `KPA-39` measured the one message the old loop emitted as false
    #: 126 times out of 126, because every guard shared it.
    REFUSED_TOO_FEW_CELLS = "too few cells to fit two slopes to"
    REFUSED_ZERO_LENGTH = "zero-length path"
    REFUSED_TOO_LITTLE_GROUND = "too little finite ground along the valley"
    REFUSED_NO_BREAK = ("no break in the floor clearing {ease:.0%} of grade change "
                        "(above {above:.1%}, below {below:.1%})")
    #: Not a refusal: the valley was never looked at because `max_valleys` was reached.
    NOT_EXAMINED = "not examined"

    def find_keypoint(self):
        """The Yeomans keypoint on the largest stream, or ``None``.

        Preserved as it was so existing callers and tests keep their meaning;
        :meth:`find_keypoints` is the per-primary-valley form and is what the keyline
        path now uses.
        """
        fdir_arr, acc_arr = self._ensure_flow_data()
        outlet_r, outlet_c = np.unravel_index(
            int(np.argmax(acc_arr)), acc_arr.shape
        )
        thalweg = self._trace_thalweg(fdir_arr, acc_arr, outlet_r, outlet_c)
        return self.keypoint_on_path(thalweg)

    def keypoint_on_path(self, thalweg, require_prominence=False):
        """The keypoint on one ordered run of valley-floor cells, or ``None``.

        The wrapper every existing caller uses. The criterion, and the reason a valley
        is refused, live in :meth:`keypoint_on_path_with_reason`.
        """
        kp, _reason = self.keypoint_on_path_with_reason(
            thalweg, require_prominence=require_prominence)
        return kp

    def keypoint_on_path_with_reason(self, thalweg, require_prominence=False):
        """Run the keypoint criterion along one ordered run of valley-floor cells.

        Returns ``(keypoint, None)`` or ``(None, reason)``. A refusal always names the
        guard that refused, because `KPA-39` measured the old single message as false
        126 times out of 126: every guard shared it.

        **The criterion is Yeomans' two slopes, fitted.** "The primary valley has two
        slopes; the upper slope is steep and changes to a much flatter slope at the
        Keyline of the valley" (*Water for Every Farm*, p44). The keypoint is "the point
        of change in the two slopes of the primary valley" (p60–61), and on a contour map
        "the contour lines are closer together above it, and further apart below it"
        (*Keyline Design Mark IV*). So the profile is fitted with one break and two
        straight reaches — ``z = a + b·s + c·max(0, s − s_k)``, continuous at the break —
        at every candidate cell with at least :data:`MIN_REACH_CELLS` cells on each side,
        and the break with the least residual is the keypoint. The grade above it is
        ``−b``, the grade below is ``−(b + c)``, and ``slope_ease`` is their difference.

        This replaced the argmax of a smoothed second derivative on 2026-09-11. That
        found the *sharpest* local easing rather than *the* change between two slopes,
        and once valleys were profiled from the divide (`KPA-52`) it put 3 of 15
        keypoints at its own smoothing guard, 10–16 m below the divide on under 3 m of
        fall, and 4 more on valleys that steepen downhill overall — the "nosed over"
        shape p44 says is a ridge, not a valley. A whole-valley fit can do neither: a
        valley that steepens downhill fits with the grade above *less* than the grade
        below and is refused, and there is no filter for a guard to sit at. On the
        synthetic test valleys it lands on the built-in break exactly. `MATHS_AUDIT`
        §9.10 has the measurement; the resampling, the Savitzky–Golay window and the
        35 m profile floor went with the derivative.

        With *require_prominence*, a break whose grade above does not exceed the grade
        below by :data:`MIN_SLOPE_EASE` is refused rather than returned as the best
        available noise.

        Nodata cells are dropped from the profile, never substituted with 0.0 m: a
        single sea-level stand-in on a 300 m hillside would be a cliff the fit has to
        explain, and it would explain it with the break.

        Returns a dict with ``x``, ``y``, ``elevation``, ``row``, ``col``,
        ``arc_length_m``, ``slope_ease``, ``grade_above`` and ``grade_below``.
        """
        reach = self.MIN_REACH_CELLS
        if thalweg is None or len(thalweg) < 2 * reach + 1:
            return None, self.REFUSED_TOO_FEW_CELLS

        arc_all = [0.0]
        for i in range(1, len(thalweg)):
            dr = thalweg[i][0] - thalweg[i - 1][0]
            dc = thalweg[i][1] - thalweg[i - 1][1]
            arc_all.append(arc_all[-1] + (dr ** 2 + dc ** 2) ** 0.5 * self.cell_size)
        if arc_all[-1] <= 0:
            return None, self.REFUSED_ZERO_LENGTH

        arc, elevs, kept = [], [], []
        for i, (s, (r, c)) in enumerate(zip(arc_all, thalweg)):
            z = float(self.dem[r, c])
            if np.isfinite(z):
                arc.append(s)
                elevs.append(z)
                kept.append(i)
        if len(arc) < 2 * reach + 1:
            return None, self.REFUSED_TOO_LITTLE_GROUND

        k, grade_above, grade_below = _two_slope_break(
            np.asarray(arc, dtype="float64"), np.asarray(elevs, dtype="float64"), reach)
        slope_ease = grade_above - grade_below
        if require_prominence and slope_ease < self.MIN_SLOPE_EASE:
            return None, self.REFUSED_NO_BREAK.format(
                ease=self.MIN_SLOPE_EASE, above=grade_above, below=grade_below)

        kr, kc = thalweg[kept[k]]
        x, y = self._rc_to_xy(kr, kc)
        return {
            "x": x,
            "y": y,
            "elevation": round(float(self.dem[kr, kc]), 1),
            "row": kr,
            "col": kc,
            "arc_length_m": round(float(arc[k]), 1),
            "slope_ease": round(slope_ease, 4),
            "grade_above": round(grade_above, 4),
            "grade_below": round(grade_below, 4),
        }, None

    def primary_valleys(self, stream_threshold_cells=None, max_order=1,
                        boundary_mask=None):
        """Every primary valley on the DEM, each as an ordered run of cells, divide to foot.

        **What a primary valley is.** Yeomans: valleys that "form into the side of the
        main ridge… the smallest of the three shapes of land… the first valley and the
        only true valley shape in the landscape" (*Water for Every Farm*, p40). One
        "generally starts as a more or less sudden steepening of the side slope of a
        main ridge. Further down, the valley changes to a flatter sloping floor which
        continues more or less uniformly to the stream course below it" (p58), and it
        "does not usually have a washed out or channelled water course down the middle"
        (p58). Its floor is where runoff goes "by the steepest path and the fastest
        route" (p45), "at right angles to the contours" (p43).

        **How that is read off a DEM here.**

        * The *line* is single-successor steepest descent — :func:`flow_graph.d8_from_dem`
          on the **conditioned** surface (:meth:`_ensure_conditioned`). D-infinity is an
          area partition, not a line, and the raw DEM has pits (525 on the fixture) that
          end a walk where the valley does not (`KPA-38`).
        * The *network* is that graph's **own** accumulation (:func:`flow_graph.accumulate`)
          thresholded at *stream_threshold_cells*, so no pointer can leave the mask. The
          old mask came from pysheds' D-infinity while the pointers were D8, and links
          ended wherever the two disagreed — 3 m fragments, 1 keypoint where there are
          many (`KPA-52`). The threshold does **not** say where a valley starts; it is
          only the smallest catchment that counts as a valley at all.
        * A *primary* valley is an order-1 link (:func:`flow_graph.stream_links`), which
          runs to "the creek (or valley junction) below" (p40) — **extended upstream**
          along its main stem to the divide (:func:`flow_graph.main_stem_to_divide`),
          because the short steep reach the keypoint is defined against lies above the
          channel head, and a link that starts at the channel head has dropped it.
        * **The data edge.** A boundary row has no outside for ``d8_from_dem`` to route
          into, so the pointers run *along* it and fabricate a channel there — 455 cells
          on the fixture, and the largest valley's keypoint sat on that run. The valley
          left the site where it reached the edge ("the creek is the lower boundary",
          p45; ``flow_graph.LABEL_EXIT``), so it is **cut** at the first grid-edge cell
          below the divide. A divide that sits on the edge is kept and **flagged**
          (``head_on_boundary``) rather than refused: the break may still be on the map,
          and a designer with a truncated sheet notes it rather than discarding the
          valley. See :meth:`_edge_rule`.

        **Ranking.** Largest catchment first, on the *supplied* accumulation — the
        baseline's, pond-corrected — which is also what the "N ha" labels quote, so the
        list order and the numbers agree. ``own_catchment_cells`` carries this graph's
        count beside it so the two can be compared.

        Returns a list of dicts: ``cells`` (row, col) top-down, ``channel_cells``,
        ``extension_cells``, ``length_m``, ``head_rc`` (the channel head), ``divide_rc``,
        ``outlet_rc``, ``head_on_boundary``, ``runs_off_dem_m``, ``channel_on_map``,
        ``catchment_cells``, ``own_catchment_cells``. Empty when nothing reaches the
        threshold. A valley with ``channel_on_map`` False is returned so the caller can
        say why it was refused; :meth:`find_keypoints` does not key it.
        """
        from terrainflow_assessment.modules.flow_graph import (
            data_boundary_mask,
            main_stem_to_divide,
            strahler_order,
            stream_links,
        )

        _fdir, acc_arr = self._ensure_flow_data()
        next_flat, own_acc = self._primary_graph()
        rows, cols = self.dem.shape

        if stream_threshold_cells is None:
            # A modest default in CELLS derived from the grid, not a bare constant:
            # KPA-05 is the standing lesson that a cell threshold means a different
            # catchment on every resolution.
            stream_threshold_cells = max(20, int(round(2_000.0 / (self.cell_w * self.cell_h))))

        finite = np.isfinite(self.dem)
        stream = (own_acc.reshape(rows, cols) >= stream_threshold_cells) & finite
        allowed = None
        if boundary_mask is not None:
            allowed2d = np.asarray(boundary_mask, dtype=bool)
            stream &= allowed2d
            allowed = allowed2d.ravel()
        if not stream.any():
            return []

        order = strahler_order(next_flat, stream.ravel())
        links = stream_links(next_flat, stream.ravel(), order, cols,
                             max_order=max_order)
        boundary = data_boundary_mask(finite)

        valleys = []
        for link in links:
            head_flat = link[0][0] * cols + link[0][1]
            above = main_stem_to_divide(next_flat, own_acc, head_flat, cols,
                                        allowed_flat=allowed)
            cells = ([(int(i // cols), int(i % cols)) for i in above]
                     + [(int(r), int(c)) for r, c in link])
            kept, start, end, runs_off_m = self._edge_rule(cells, boundary)
            n_above = len(above)
            extension_cells = max(0, min(end, n_above) - start)
            channel_cells = max(0, end - max(start, n_above))
            foot = kept[-1]
            catchment = float(acc_arr[foot])
            valleys.append({
                "cells": kept,
                "channel_cells": channel_cells,
                "extension_cells": extension_cells,
                "length_m": self._arc_length_m(kept),
                "head_rc": (int(link[0][0]), int(link[0][1])),
                "divide_rc": kept[0],
                "outlet_rc": foot,
                "head_on_boundary": bool(boundary[kept[0]]),
                "runs_off_dem_m": runs_off_m,
                # A channel that only reaches the threshold *on* the boundary row is
                # the row's own collecting artefact, not a channel the map shows: the
                # cut leaves nothing of it but the edge cell. Such a valley has no
                # channel on the map and cannot be vouched for.
                "channel_on_map": not (runs_off_m > 0 and channel_cells <= 1),
                "catchment_cells": catchment if np.isfinite(catchment) else 0.0,
                "own_catchment_cells": int(own_acc[foot[0] * cols + foot[1]]),
            })

        # Largest catchment first — if only some valleys get a keyline, they should be
        # the ones carrying the most water.
        valleys.sort(key=lambda v: (v["catchment_cells"], v["own_catchment_cells"]),
                     reverse=True)
        return valleys

    def _edge_rule(self, cells, boundary):
        """Trim a valley to the ground the DEM can vouch for.

        *boundary* is :func:`flow_graph.data_boundary_mask`: the grid edge **and** the
        cells beside nodata. Both are the edge of the data, and both fabricate channels
        the same way — a boundary cell has no outside for ``d8_from_dem`` to route into,
        so the pointers run *along* it. Measured on the fixture bounded by a nodata
        ellipse: three valleys' channels ran entirely along the rim, 6 of 6, 24 of 24
        and 8 of 8 cells beside nodata, exactly as they run along a grid row.

        One pass from the top. Find the first cell that is not on the boundary and keep
        at most one boundary cell above it — a divide on the edge stays as the first
        cell, and a run *along* the edge is trimmed to that one cell. From there, cut at
        the first boundary cell (kept, as the foot): the valley left the site.

        Returns ``(kept, start, end, runs_off_m)`` — the kept cells, their slice of
        *cells*, and how much valley was discarded below the cut.
        """

        def on_edge(rc):
            return bool(boundary[rc])

        first_inside = next((i for i, rc in enumerate(cells) if not on_edge(rc)), None)
        if first_inside is None:
            # Nothing inside at all: keep the divide alone so the caller can refuse it
            # with the length it lost, rather than silently dropping the valley.
            return cells[:1], 0, 1, self._arc_length_m(cells)
        start = max(0, first_inside - 1)
        cut = next((i for i in range(first_inside, len(cells)) if on_edge(cells[i])),
                   None)
        end = len(cells) if cut is None else cut + 1
        runs_off_m = 0.0 if cut is None else self._arc_length_m(cells[cut:])
        return cells[start:end], start, end, runs_off_m

    def _arc_length_m(self, cells):
        """Map-space length of an ordered run of cells, centre to centre."""
        total = 0.0
        for (r0, c0), (r1, c1) in zip(cells[:-1], cells[1:]):
            total += ((r1 - r0) ** 2 + (c1 - c0) ** 2) ** 0.5 * self.cell_size
        return total

    def find_keypoints(self, max_valleys=8, stream_threshold_cells=None,
                       max_order=1, boundary_mask=None):
        """One keypoint per **primary valley**, which is what Yeomans' method asks for.

        ``find_keypoint`` walks the single largest stream. The trunk of a catchment is
        not a primary valley, so that answer was the right criterion applied to the
        wrong feature, once. :meth:`primary_valleys` says what a primary valley is and
        how it is traced; :meth:`keypoint_on_path_with_reason` says how the keypoint is
        located on it.

        Valleys are ranked by contributing area and capped at *max_valleys*: a fine DEM
        with a low stream threshold has thousands of order-1 links, and drawing a keyline
        set on every one of them is neither useful nor affordable.

        Returns ``(keypoints, skipped)``. Each skipped valley carries its own reason —
        the house style is to say what was refused and why, not to return a shorter list.

        **Two catchment figures, and they are not interchangeable.** ``catchment_ha`` is
        the *valley's* — accumulation at its foot — which is what ranks the valleys and
        what the map layer's attribute of that name has always held.
        ``keypoint_catchment_ha`` is the ground above the keypoint itself, which sits
        partway down the valley and therefore commands less. The label quotes the second
        and names the first, because "N ha above" attached to a point means the ground
        above *that point*.
        """
        _fdir, acc_arr = self._ensure_flow_data()
        valleys = self.primary_valleys(stream_threshold_cells=stream_threshold_cells,
                                       max_order=max_order, boundary_mask=boundary_mask)
        if not valleys:
            return [], ["no channel network at this threshold"]

        cell_area = self.cell_w * self.cell_h
        keypoints, skipped = [], []

        def _where(valley):
            return (f"valley from row {valley['divide_rc'][0]}, "
                    f"col {valley['divide_rc'][1]} ({valley['length_m']:.0f} m: "
                    f"{valley['channel_cells']} channel cells, "
                    f"{valley['extension_cells']} above the channel head)")

        first_unexamined = len(valleys)
        for index, valley in enumerate(valleys):
            if len(keypoints) >= max_valleys:
                first_unexamined = index
                break
            cells = valley["cells"]
            where = _where(valley)
            if not valley["channel_on_map"]:
                skipped.append(
                    f"{where}: runs off the DEM after {valley['length_m']:.0f} m — "
                    "its channel begins on the data edge, so no channel is on the map")
                continue
            if (valley["runs_off_dem_m"] > 0
                    and len(cells) < 2 * self.MIN_REACH_CELLS + 1):
                skipped.append(
                    f"{where}: runs off the DEM after {valley['length_m']:.0f} m")
                continue
            kp, reason = self.keypoint_on_path_with_reason(cells, require_prominence=True)
            if kp is None:
                skipped.append(f"{where}: {reason}")
                continue
            kp["valley_cells"] = len(cells)
            kp["channel_cells"] = valley["channel_cells"]
            kp["extension_cells"] = valley["extension_cells"]
            kp["head_on_boundary"] = valley["head_on_boundary"]
            kp["runs_off_dem_m"] = round(valley["runs_off_dem_m"], 1)
            # The valley's catchment, measured at its **foot**. This is the ranking basis
            # and what the map layer's `catchment_ha` attribute has always carried, so it
            # keeps the name.
            kp["catchment_ha"] = valley["catchment_cells"] * cell_area / 10_000.0
            kp["_row"], kp["_col"] = kp["row"], kp["col"]
            # The ground above the keypoint **itself**, which is a different and usually
            # much smaller number — the keypoint sits partway down the valley, not at its
            # foot. On the Quail Island fixture the rank-1 keypoint once read 2.1 ha here
            # against 5.9 ha for its valley, so a label saying "5.9 ha above" over-stated
            # what that point commands by 177%. That was `KPA-54`.
            kp["keypoint_catchment_ha"] = (
                float(acc_arr[kp["_row"], kp["_col"]]) * cell_area / 10_000.0)
            kp["label"] = (
                f"Keypoint at {kp['elevation']:.1f} m — "
                f"{kp['keypoint_catchment_ha']:.1f} ha above "
                f"({kp['catchment_ha']:.1f} ha in the valley)"
                + (" (valley head at data edge)" if kp["head_on_boundary"] else ""))
            keypoints.append(kp)

        # The cap used to drop every remaining valley from *both* lists, so the panel
        # said "5 had no keypoint" over a network of 23 (`KPA-43`). A valley the cap
        # stopped short of is reported as exactly that, and `keypoints + skipped ==
        # valleys` holds at every cap. `NOT_EXAMINED` is the marker a caller counts on.
        for valley in valleys[first_unexamined:]:
            skipped.append(
                f"{_where(valley)}: {self.NOT_EXAMINED} — the cap of {max_valleys} "
                f"keypoint(s) was reached")

        return keypoints, skipped

    def get_cultivation_runs(self, keypoint, n_runs=3, max_grade_n=500,
                             spacing_m=None, pattern="both", cross_grade=None):
        """The Yeomans keyline and its cultivation guides, with the drift **measured**.

        Per Yeomans (*The Keyline Plan*, 1954) the **keyline** is the on-contour line
        through the keypoint. **Cultivation guides** are geometric parallel offsets of
        it. Because a parallel offset of a curved valley contour is not itself a
        contour, the guides drift off-contour on their own, moving water from the wet
        valley floor toward the drier ridge — the drift is emergent from parallelism,
        and no artificial grade is imposed on the geometry.

        **Two patterns, because Yeomans specifies two**, and he stresses that most of a
        landscape is the second:

        * ``"valley"`` — guides parallel to and **below** the keyline, which spread
          runoff out of the valley floor toward the flanking ridges.
        * ``"ridge"`` — guides parallel to and **above** a contour guide taken on the
          ridge, which drift water off the ridge nose out toward the valleys.

        The default is ``"both"``, which is what a whole ridge-and-valley pair wants —
        and what keeps a keypoint in a valley from producing guides on only one side of
        itself.

        **The grade is a limit, not a generator.** ``cross_grade`` used to be echoed
        onto every run and read by nothing, so a user could set 1:50 or 1:5000 and get
        byte-identical lines — while the map carried a column asserting the grade the
        geometry did not have. It is gone. ``max_grade_n`` is a **threshold**: every run
        reports the drift it actually achieves, and guides steeper than 1:``max_grade_n``
        are flagged. That is the number nobody had ever measured, and it is what the
        "does the drift make sense" question was really asking.

        Returns a list of dicts with ``elevation``, ``geometry`` (3D LineString, Z
        sampled from the ground — a plough guide sits *on* the ground, and a line set
        out to a designed invert is a diversion drain, which exists), ``line_type``
        (``keyline`` | ``valley_guide`` | ``ridge_guide``), ``offset_m``,
        ``drift_1_in_n``, ``drift_fall_m``, ``over_limit``.
        """
        if cross_grade is not None:
            warnings.warn(
                "cross_grade is gone: it never reached the geometry. Use max_grade_n, "
                "which flags guides whose *measured* drift is steeper than 1:N.",
                DeprecationWarning, stacklevel=2)

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
        keyline_pts, keyline_elev = self._sample_z(base_line, base_elev)
        if len(keyline_pts) >= 2:
            results.append(self._run_record(
                LineString(keyline_pts), keyline_elev, "keyline", 0.0, max_grade_n))

        wants_valley = pattern in ("valley", "both", "auto")
        wants_ridge = pattern in ("ridge", "both", "auto")

        for step in range(1, n_runs + 1):
            distance = step * spacing_m
            for signed in (distance, -distance):
                for part in self.offset_parts(base_line, signed):
                    pts3d, mean_elev = self._sample_z(part, base_elev)
                    if len(pts3d) < 2:
                        continue

                    # **Labelled by measured elevation, not by the sign of the offset.**
                    # ``offset_curve``'s sign means "left of the direction of travel",
                    # and the traced contour's winding comes from find_contours without
                    # being normalised — so upper and lower were being assigned by an
                    # arbitrary sign. Measuring is this repo's own lesson.
                    above = mean_elev >= keyline_elev
                    if above and not wants_ridge:
                        continue
                    if not above and not wants_valley:
                        continue
                    line_type = "ridge_guide" if above else "valley_guide"

                    results.append(self._run_record(
                        LineString(pts3d), mean_elev, line_type,
                        distance if above else -distance, max_grade_n))

        return results

    #: Window over which a guide's drift is judged, in metres. **A TerrainFlow
    #: convention, not a Yeomans figure** — the texts were fetched and read on
    #: 2026-09-11 and publish no drift tolerance at all (`MATHS_AUDIT` §9.8).
    #:
    #: Sized rather than chosen, by the roughness sweep in
    #: `tests_qgis/probes/p_keypoints.py`: it is the shortest window whose steepest-grade
    #: reading stays inside `MIN_SLOPE_EASE` across synthetic correlated roughness of
    #: 0.00 -> 0.10 m, which is the band `_box_blur`'s docstring puts real LiDAR noise in.
    #: 10 m and 20 m read 2.18x and 2.44x `MIN_SLOPE_EASE` on the same ground and are
    #: measuring the DEM's noise rather than the guide's drift.
    DRIFT_WINDOW_M = 50.0

    def _steepest_drift(self, coords, window_m=None):
        """Steepest sustained fall over any *window_m* of a guide — ``(grade, 1:N)``.

        **Net end-to-end fall is not drift**, which is `KPA-41`. A guide wanders up and
        down its own length by construction, so the two ends can sit at nearly the same
        height while the middle runs steeply: measured on the fixture, a ridge guide
        reporting **1:544.8** and `over_limit` **False** runs **1:6.3** over its steepest
        20 m — an 86x understatement of the thing the docstring promises to flag.

        Windows are taken between vertices at least *window_m* apart along the line, so a
        guide shorter than the window reduces to its own end-to-end fall rather than
        returning nothing.

        **A vertex is "real ground" only if its own cell is**, and that cannot be decided
        from Z. `_sample_dem` hands back the *keypoint elevation* for any sample off the
        grid or on a nodata cell (`KPA-42`), which is a perfectly finite number — so
        testing `isfinite(z)` would call a fabricated vertex real and let it flatten every
        window it falls inside, under-reading exactly the drift this is here to find. The
        mask is therefore recomputed from the geometry against the raster, the same way
        `_sample_dem` decided it, and a window is measured only when every vertex in it is
        real.

        Returns ``(0.0, None)`` when there is no real span to measure.
        """
        window_m = self.DRIFT_WINDOW_M if window_m is None else float(window_m)
        if len(coords) < 2:
            return 0.0, None

        xs = np.array([c[0] for c in coords], dtype="float64")
        ys = np.array([c[1] for c in coords], dtype="float64")
        zs = np.array([c[2] if len(c) > 2 else np.nan for c in coords], dtype="float64")

        rows, cols = self.dem.shape
        real = np.zeros(len(coords), dtype=bool)
        for i, (x, y) in enumerate(zip(xs, ys)):
            if not np.isfinite(zs[i]):
                continue
            r, c = xy_to_rc(self.transform, float(x), float(y))
            real[i] = (0 <= r < rows and 0 <= c < cols
                       and not np.isnan(self.dem[r, c]))
        if real.sum() < 2:
            return 0.0, None

        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        # O(1) test for "every vertex between i and j inclusive is real ground".
        real_prefix = np.concatenate([[0], np.cumsum(real.astype("int64"))])

        best = 0.0
        j = 0
        for i in range(len(s)):
            if not real[i]:
                continue
            while j < len(s) and s[j] - s[i] < window_m:
                j += 1
            if j >= len(s):
                break
            if not real[j]:
                continue
            if real_prefix[j + 1] - real_prefix[i] != (j - i + 1):
                continue
            run = s[j] - s[i]
            if run <= 0:
                continue
            grade = abs(zs[j] - zs[i]) / run
            if grade > best:
                best = grade

        if best == 0.0:
            # Shorter than the window, or no all-real window in it: fall back to the
            # widest real span there is, which for a short guide is its whole length.
            idx = np.flatnonzero(real)
            i, j = int(idx[0]), int(idx[-1])
            run = s[j] - s[i]
            if run > 0 and real_prefix[j + 1] - real_prefix[i] == (j - i + 1):
                best = abs(zs[j] - zs[i]) / run

        return best, ((1.0 / best) if best > 0 else None)

    def _run_record(self, geometry, mean_elev, line_type, offset_m, max_grade_n):
        """One cultivation run, with its achieved drift attached.

        The drift is the net fall from one end of the guide to the other over its
        length, reported as ``1:N``. The claim that "the drift is emergent from
        parallelism" has stood in a docstring since this feature was written and has
        never been measured anywhere; this is the measurement.
        """
        coords = list(geometry.coords)
        fall = float(coords[0][2] - coords[-1][2]) if len(coords[0]) > 2 else 0.0
        length = geometry.length
        grade = abs(fall) / length if length > 0 else 0.0
        one_in_n = (1.0 / grade) if grade > 0 else None

        # `over_limit` is judged on the steepest sustained fall, not the net one. The
        # docstring promises to flag guides "whose measured drift is steeper than 1:N",
        # and a net figure cannot keep that promise on a line that undulates — KPA-41.
        steep_grade, steep_one_in_n = self._steepest_drift(coords)
        over = bool(max_grade_n and steep_one_in_n is not None
                    and steep_one_in_n < max_grade_n)
        return {
            "elevation": round(mean_elev, 2),
            "geometry": geometry,
            "line_type": line_type,
            "offset_m": round(offset_m, 2),
            "drift_fall_m": round(fall, 3),
            # Net end-to-end, kept under its own name: it is what the map layer's
            # attribute of that name has always carried, and it is still the right
            # answer to "where does this guide start and finish".
            "drift_1_in_n": round(one_in_n, 1) if one_in_n is not None else None,
            # What the limit is actually judged on.
            "steepest_1_in_n": (round(steep_one_in_n, 1)
                                if steep_one_in_n is not None else None),
            "steepest_window_m": self.DRIFT_WINDOW_M,
            "over_limit": over,
        }

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

    #: How far an offset part may sit from its source, as a fraction of the offset
    #: distance, before it is treated as a fold rather than a guide.
    _OFFSET_TOLERANCE = 0.25

    def offset_parts(self, line, signed_dist):
        """Every usable parallel offset of *line*, as a list of LineStrings.

        Two faults in one place, both fixed here.

        **Folding.** ``offset_curve`` self-intersects wherever the offset exceeds the
        local radius of curvature — which is every tight valley head, which is exactly
        where a keyline is. The old code took ``max(off.geoms, key=length)`` and
        returned it, so a folded lobe came back looking like a guide. Parts are now
        kept only where their distance back to the source stays within
        :data:`_OFFSET_TOLERANCE` of ``|signed_dist|``; a fold comes back *closer* than
        the offset, which is what makes it identifiable.

        **Discarding real limbs.** Taking only the longest part also threw away the
        second limb of a guide that legitimately splits around a spur. Both are real
        plough runs, so both are returned.

        Returns ``[]`` when nothing survives — the caller refuses the guide and says
        why, rather than drawing a line a plough cannot follow.
        """
        from shapely.ops import linemerge, unary_union

        off = None
        try:
            off = line.offset_curve(signed_dist)
        except Exception:
            try:
                side = "left" if signed_dist >= 0 else "right"
                off = line.parallel_offset(abs(signed_dist), side)
            except Exception:
                off = None

        if off is None or off.is_empty:
            return []

        try:
            merged = linemerge(unary_union(off))
        except Exception:
            merged = off

        parts = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]

        target = abs(signed_dist)
        tol = max(self.cell_size, target * self._OFFSET_TOLERANCE)
        kept = []
        for part in parts:
            if part.geom_type != "LineString" or part.length <= 0:
                continue
            # Sample the part rather than trusting a single distance: a fold is only
            # close to the source along the folded section.
            n = max(2, min(24, int(part.length / max(self.cell_size, 1e-6))))
            dists = [part.interpolate(part.length * i / n).distance(line)
                     for i in range(n + 1)]
            if max(abs(d - target) for d in dists) <= tol:
                kept.append(part)
        return kept

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
        """Return ``(fdir_arr, acc_arr)``, reading or computing them as needed.

        **Accumulation is the cache key, and the gate, because it is the only half
        anybody reads.** ``fdir_arr`` is returned and threaded through
        :meth:`find_keypoint` into :meth:`_trace_thalweg`, which takes it as a parameter
        and never touches it — that walk goes by elevation and accumulation on purpose
        (see its docstring). So a supplied accumulation is enough to answer with, and
        ``fdir_arr`` may legitimately come back ``None``. Anything that starts reading it
        must check for that and for ``dinf_routing`` before treating it as D8 codes.

        Both the gate and the cache test used to be ``and`` over the two, which is
        `KPA-48`: ``contour.py`` supplies ``acc_path`` alone, so the gate never opened and
        every keyline press paid **0.48 s** to recompute a field it had been handed
        (0.022 s to read) — and got a *different* one, differing by up to 65,086 cells,
        because the supplied field is crest-split and the recompute is not. The keyline
        was the only tool in the plugin drawing on a pond-uncorrected accumulation.

        Honouring the supplied field closes `KPA-53` with it: that raster came from
        ``FlowAnalysis.run(routing=panel.routing)``, so the user's routing choice now
        reaches the keyline tier by carrying it in the data rather than by threading a
        second copy of the setting down. ``routing`` below is only for the no-baseline
        case, where there is nothing to inherit.
        """
        if self._acc_arr is not None:
            return self._fdir_arr, self._acc_arr

        if self._acc_path:
            # float32, not int32: a supplied raster may be dinf radians as easily as
            # D8 codes, and truncating the former loses the direction entirely.
            with rasterio.open(self._acc_path) as src:
                self._acc_arr = src.read(1).astype("float32")
            if self._fdir_path:
                with rasterio.open(self._fdir_path) as src:
                    self._fdir_arr = src.read(1).astype("float32")
            return self._fdir_arr, self._acc_arr

        # Compute from DEM — and keep the conditioned surface, which the valley graph
        # is built on (`_primary_graph`), rather than throwing it away as this used to.
        grid, inflated = self._condition_surface()
        try:
            # `self._routing`, not a literal "dinf". This branch only runs when no
            # accumulation was supplied — no baseline, nothing to inherit — and a
            # user who chose D8 should not silently get D-infinity here either.
            # That literal was half of `KPA-53`.
            fdir = grid.flowdir(inflated, routing=self._routing)
            _routing = self._routing
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

        return self._fdir_arr, self._acc_arr

    def _condition_surface(self):
        """Condition the DEM the way the baseline does, and keep the surface.

        ``fill_pits`` → ``fill_depressions`` → ``resolve_flats_safely``, on a temporary
        GeoTIFF because pysheds reads from a file. (A fill, not a breach:
        ``breach_depressions`` does not exist in pysheds 0.5, so the except branch is
        the one that has always run — Round 14. The flat step is derived from this
        surface, not pysheds' fixed default: on a big flat the default lifts cells over
        neighbours that were genuinely lower — see ``flow_analysis.safe_flat_epsilon``.)

        The temp raster is written float64. Measured, that changes nothing here — the
        input is already float32 and ``resolve_flats_safely`` builds its surface in
        memory — but the surface is now *kept*, as float64, and is what the valley
        pointers are built on, so it is not going through a float32 hop on the way.

        Returns ``(grid, inflated)`` for a caller that goes on to route on it, and sets
        ``self._conditioned`` (float64, NaN where the DEM is NaN).
        """
        import os
        import tempfile

        from terrainflow_assessment.modules.flow_analysis import resolve_flats_safely
        from terrainflow_assessment.modules.pysheds_compat import Grid

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)
        try:
            with rasterio.open(
                tmp_path, "w", driver="GTiff", dtype="float64",
                crs=self.crs, transform=self.transform,
                width=self.dem.shape[1], height=self.dem.shape[0],
                count=1, nodata=-9999.0,
            ) as dst:
                data = np.where(np.isnan(self.dem), -9999.0, self.dem)
                dst.write(data.astype("float64"), 1)

            grid = Grid.from_raster(tmp_path)
            dem_r = grid.read_raster(tmp_path)
            pit_filled = grid.fill_pits(dem_r)
            try:
                filled = grid.breach_depressions(pit_filled)
            except AttributeError:
                filled = grid.fill_depressions(pit_filled)
            inflated, _eps, _inv = resolve_flats_safely(grid, filled)
        finally:
            os.unlink(tmp_path)

        surface = np.array(inflated, dtype="float64")
        surface[~np.isfinite(self.dem)] = np.nan
        self._conditioned = surface
        self.conditioned_source = "recomputed"
        return grid, inflated

    def _ensure_conditioned(self):
        """The conditioned surface the valley graph is traced on, float64.

        The supplied raster when there is one and it fits — read the way
        ``earthworks._ensure_flow_graph`` reads it: as float64, because it carries
        ``resolve_flats``' synthetic gradient in multiples of 1e-5 m, and with the
        DEM's own nodata sentinel as the fallback for an untagged file, because a hole
        read as ground at -9999 is a ten-kilometre pit. Otherwise conditioned here.
        """
        if self._conditioned is not None:
            return self._conditioned
        if self._conditioned_path:
            surface = self._read_conditioned(self._conditioned_path)
            if surface is not None:
                self._conditioned = surface
                self.conditioned_source = "supplied"
                return surface
        self._condition_surface()
        return self._conditioned

    def _read_conditioned(self, path):
        """A supplied conditioned raster as float64 with NaN holes, or ``None``."""
        try:
            with rasterio.open(path) as src:
                surface = src.read(1).astype("float64")
                nodata = src.nodata
        except OSError:
            return None                 # rasterio's IO errors are OSErrors
        if surface.shape != self.dem.shape:
            return None
        if nodata is None:
            nodata = self._nodata
        if nodata is not None and np.isfinite(nodata):
            surface[surface == nodata] = np.nan
        surface[~np.isfinite(self.dem)] = np.nan
        return surface

    def _primary_graph(self):
        """``(next_flat, own_acc)`` — the one graph every valley is traced on. Cached."""
        if self._graph is None:
            from terrainflow_assessment.modules.flow_graph import (
                accumulate,
                d8_from_dem,
            )

            surface = self._ensure_conditioned()
            next_flat, _sink = d8_from_dem(surface, self.cell_w, self.cell_h)
            own_acc = accumulate(next_flat, np.isfinite(self.dem).ravel())
            self._graph = (next_flat, own_acc)
        return self._graph

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
