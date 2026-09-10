"""
impoundment_sites.py — ranking dam sites by what they store per cubic metre of wall.

Replaces the proxy that shipped before it. ``recommend_pond_sites`` scored candidates
``acc / (width + 1)`` — a cell count divided by a length, flagged HAZ as KPA-20 in the
maths audit — and answered no question a designer asks.

**The index here is storage ÷ embankment: cubic metres held per cubic metre moved.**
Dimensionless, measured off the DEM at both ends, and the standard farm-dam siting
figure (USDA-NRCS *Ponds — Planning, Design, Construction*, Agriculture Handbook 590,
which directs siting to where that ratio is greatest — a narrow neck below a wide
basin). It is the cost of the water, which is what actually decides between two saddles.

**Everything else is a column, not a term in the score.** Catchment yield, wall height
and length, and the freeboard to the lowest saddle outside the wall are all reported
beside the ratio and never folded into it. This repo already has the cautionary tale:
"Design storage" was dropped from the report because *"a figure from a rule of thumb,
sitting first in a run of columns meant to be compared against each other, invited
exactly the comparison it could not support"*. A blended dam score is that figure.

**The transect runs perpendicular to the flow, which the old one did not.** The width
that fed the proxy was measured by scanning *along the raster row* — its own docstring
said "left and right along the same row" — so it was a cross-section only where the
valley happened to run north–south. An east–west valley had its *length* measured as
its width, and scored accordingly.

**It is a screening rank, not a measurement.** The design tier burns a real dam and
measures the pond it makes; this floods a bounded window on the bare DEM to compare
candidates. The two are allowed to disagree because they answer different questions,
and the UI says which is which — the same ``CALCULATED`` / ``MEASURED`` distinction the
report already draws.

Pure numpy + scipy: no QGIS, no rasterio.
"""

from __future__ import annotations

import math

import numpy as np

from terrainflow_assessment.core.sizing.primitives import (
    prismatic_volume,
    trapezoid_section,
)

#: Trial wall heights swept per candidate, in metres.
#:
#: The old code used a single hard-wired ``+2.0 m`` crest (KPA-19) and reported the
#: width at it. Sweeping turns that constant into a design decision and costs one
#: bounded flood per height; the winning height is reported so the user can see which
#: wall the ratio belongs to.
DEFAULT_CREST_HEIGHTS_M = (1.0, 1.5, 2.0, 3.0, 4.0)

#: Embankment geometry for the screening estimate — crest width and batter (z:1).
#: Deliberately coarse: this ranks sites, and the dam dialog sizes the one chosen.
DEFAULT_CREST_WIDTH_M = 3.0
DEFAULT_BATTER = 2.0

#: How far either side of the candidate the wall may run before the site is refused.
#: A wall longer than this is not a neck.
DEFAULT_MAX_WALL_M = 200.0

#: How far upstream a pool may reach and still count as a local impoundment, in metres.
#:
#: This bounds the flood window, and the bound is a *statement*, not an optimisation: a
#: pool that runs further than this up the valley is not the pond the candidate
#: describes, and the site is refused saying so rather than the window quietly grown
#: until something fits. It also fixes the cost — the window is the same size whatever
#: the DEM is, so a sweep over twenty candidates has a predictable price.
DEFAULT_POOL_REACH_M = 250.0


def flow_bearing(dz_dx, dz_dy, row, col):
    """Downslope direction at one cell, as a map-space unit vector ``(ex, ny)``.

    ``dz_dy`` from ``horn_gradient`` rises toward increasing *row*, which is southward
    on a north-up grid, so the northward rise is its negation. Water runs down the
    negated gradient.

    Returns ``None`` on genuinely flat ground, where there is no direction to be
    perpendicular to.
    """
    rise_east = float(dz_dx[row, col])
    rise_north = -float(dz_dy[row, col])
    mag = math.hypot(rise_east, rise_north)
    if not math.isfinite(mag) or mag <= 0.0:
        return None
    return (-rise_east / mag, -rise_north / mag)


def transect_cells(shape, row, col, direction, cell_w, cell_h, reach_m):
    """Cells along the line through ``(row, col)`` **perpendicular** to *direction*.

    *direction* is the downslope unit vector; the wall crosses the flow, so the crest
    runs square across it. Marched outward in both directions at roughly one cell per
    step, in map space, so a 2 m x 2 m grid and a 0.25 m grid ask the same physical
    question.

    Returns ``(cells, step_m)`` where *cells* is a list of ``(row, col)`` ordered from
    one side to the other through the candidate.
    """
    rows, cols = shape
    ex, ny = direction
    # Perpendicular in map space: rotate the downslope vector a quarter turn.
    px, py = -ny, ex

    step = min(abs(cell_w), abs(cell_h))
    n_steps = max(1, int(round(reach_m / max(step, 1e-9))))

    out = []
    seen = set()
    for i in range(-n_steps, n_steps + 1):
        dx = px * i * step
        dy = py * i * step
        # Map metres → cells. Row increases southward, so a northward offset is negative.
        r = int(round(row - dy / abs(cell_h)))
        c = int(round(col + dx / abs(cell_w)))
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if (r, c) in seen:
            continue
        seen.add((r, c))
        out.append((r, c))
    return out, step


def wall_run(dem, cells, crest, candidate_index):
    """The contiguous below-crest run of *cells* containing the candidate.

    That run is where the wall would actually stand: it starts and ends where the
    ground already reaches the crest, which is where the abutments are. Walking out
    from the candidate rather than counting every below-crest cell in the window is the
    difference between one wall and a separate gully two hundred metres away folded
    into the same figure.

    Returns the list of ``(row, col, height_m)`` under the wall, or ``[]`` when the
    candidate itself is not below the crest.
    """
    heights = []
    for r, c in cells:
        z = dem[r, c]
        heights.append(crest - z if np.isfinite(z) else np.nan)

    if candidate_index >= len(heights):
        return []
    here = heights[candidate_index]
    if not (np.isfinite(here) and here > 0):
        return []

    lo = candidate_index
    while lo - 1 >= 0 and np.isfinite(heights[lo - 1]) and heights[lo - 1] > 0:
        lo -= 1
    hi = candidate_index
    while hi + 1 < len(heights) and np.isfinite(heights[hi + 1]) and heights[hi + 1] > 0:
        hi += 1

    return [(cells[i][0], cells[i][1], float(heights[i])) for i in range(lo, hi + 1)]


def embankment_volume(run, step_m, crest_width_m=DEFAULT_CREST_WIDTH_M,
                      batter=DEFAULT_BATTER):
    """Fill in the wall, m³ — a trapezoidal section integrated along the crest.

    Section per station is a trapezoid of crest width ``W`` at the top, battered out at
    ``z:1``: bottom width ``W + 2·z·h``, area ``h·(W + z·h)``. Not a rectangle, which
    would under-state a 4 m wall by more than half.

    Goes through ``core.sizing.primitives`` rather than re-deriving the trapezoid here
    — and gives ``prismatic_volume`` its first production caller.
    """
    total = 0.0
    for _r, _c, h in run:
        if h <= 0:
            continue
        section = trapezoid_section(
            top_width=crest_width_m + 2.0 * batter * h,
            bottom_width=crest_width_m,
            depth=h,
        )
        total += prismatic_volume(section.area, step_m).volume
    return total


def _upstream_seed(dem, acc, row, col):
    """The neighbour the water arrives from — where the pond will actually form.

    Flooding from the candidate itself would spill straight through the wall line, so
    the fill is seeded one step upstream. Upstream on a stream means *less* accumulated
    area, so the seed is the highest-accumulation neighbour that still carries less
    than this cell; failing that, simply the highest ground.
    """
    rows, cols = dem.shape
    here_acc = float(acc[row, col])
    best = None
    best_acc = -1.0
    highest = None
    highest_z = -np.inf
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            z = dem[r, c]
            if not np.isfinite(z):
                continue
            a = float(acc[r, c])
            if a < here_acc and a > best_acc:
                best_acc, best = a, (r, c)
            if z > highest_z:
                highest_z, highest = z, (r, c)
    return best or highest


def impounded_volume(dem, acc, row, col, crest, wall_cells, cell_area_m2,
                     window_cells):
    """Volume held behind a wall at *crest*, and whether the pond is actually enclosed.

    The wall is stamped into the mask as a barrier and the below-crest region is flooded
    from the cell upstream of it. A component that reaches the window edge is **not an
    impoundment** — the water goes round, and the window simply was not big enough to
    see where. It is refused with that reason rather than grown, which is the rule
    ``_cut_spillway``'s ``notch_pool`` already applies to a notch discharging into open
    ground.

    Returns ``(volume_m3, area_m2, reason)`` — *reason* is ``None`` on success.
    """
    from scipy.ndimage import label as nd_label

    rows, cols = dem.shape
    r0 = max(0, row - window_cells)
    r1 = min(rows, row + window_cells + 1)
    c0 = max(0, col - window_cells)
    c1 = min(cols, col + window_cells + 1)

    sub = dem[r0:r1, c0:c1]
    below = np.isfinite(sub) & (sub <= crest)

    for wr, wc, _h in wall_cells:
        if r0 <= wr < r1 and c0 <= wc < c1:
            below[wr - r0, wc - c0] = False

    seed = _upstream_seed(dem, acc, row, col)
    if seed is None:
        return 0.0, 0.0, "no upstream cell to flood from"
    sr, sc = seed
    if not (r0 <= sr < r1 and c0 <= sc < c1) or not below[sr - r0, sc - c0]:
        return 0.0, 0.0, "the ground upstream of the wall is already above the crest"

    labelled, _n = nd_label(below)
    tag = labelled[sr - r0, sc - c0]
    if tag == 0:
        return 0.0, 0.0, "no pool forms behind this wall"

    pool = labelled == tag
    touches_edge = (pool[0, :].any() or pool[-1, :].any()
                    or pool[:, 0].any() or pool[:, -1].any())
    if touches_edge:
        return 0.0, 0.0, ("not enclosed within the pond window — the pool either runs "
                          "round the wall or further upstream than a pond of this "
                          "height should")

    depths = np.where(pool, crest - sub, 0.0)
    volume = float(np.nansum(depths)) * cell_area_m2
    area = float(pool.sum()) * cell_area_m2
    return volume, area, None


def rank_impoundment_sites(dem, acc, candidates, cell_w, cell_h,
                           rc_to_xy,
                           crest_heights_m=DEFAULT_CREST_HEIGHTS_M,
                           runoff_mm=None,
                           crest_width_m=DEFAULT_CREST_WIDTH_M,
                           batter=DEFAULT_BATTER,
                           max_wall_m=DEFAULT_MAX_WALL_M,
                           pool_reach_m=DEFAULT_POOL_REACH_M,
                           progress=None):
    """Rank *candidates* by storage held per cubic metre of embankment.

    *candidates* are ``(row, col)`` cells — the caller decides which ones are worth
    measuring, because that screen is where the cost lives (see the cost note below).
    *rc_to_xy* converts a cell to map coordinates; passed in so this module stays free
    of any transform convention of its own.

    Returns a list of dicts sorted by ``storage_ratio`` descending. Every site carries
    the winning ``wall_height_m`` and its ``notes``, and a candidate that never formed
    an enclosed pool at any trial height is returned with ``storage_ratio`` of 0 and the
    reason, rather than dropped — a site the user can see was considered and refused is
    worth more than a shorter list.

    **Cost.** One bounded flood per candidate per trial height, each a
    ``scipy.ndimage.label`` over a window sized to the wall. Five heights over twenty
    candidates is a hundred labellings on windows of a few tens of thousands of cells —
    about a second. Sweeping every cell in a search box instead would be thousands, and
    it is why the caller screens first.
    """
    from terrainflow_assessment.modules.dem_loader import horn_gradient

    dz_dx, dz_dy, _invalid = horn_gradient(dem, cell_w, cell_h)
    cell_area = abs(cell_w) * abs(cell_h)

    results = []
    total = max(1, len(candidates))
    for i, (row, col) in enumerate(candidates):
        if progress:
            progress(int(100 * i / total), f"Measuring site {i + 1}/{total}…")

        direction = flow_bearing(dz_dx, dz_dy, row, col)
        if direction is None:
            results.append(_refused(row, col, rc_to_xy, dem, acc, cell_area,
                                    "the ground here is level — no valley to dam"))
            continue

        cells, step = transect_cells(dem.shape, row, col, direction,
                                     cell_w, cell_h, max_wall_m)
        try:
            here = cells.index((row, col))
        except ValueError:
            results.append(_refused(row, col, rc_to_xy, dem, acc, cell_area,
                                    "could not lay a crest line across the valley"))
            continue

        base = float(dem[row, col])
        best = None
        reason = "no enclosed pool at any trial wall height"

        for height in crest_heights_m:
            crest = base + height
            run = wall_run(dem, cells, crest, here)
            if not run:
                continue
            wall_len = len(run) * step
            if wall_len > max_wall_m:
                reason = f"the wall would run over {max_wall_m:.0f} m — not a neck"
                continue

            # Sized from how far upstream a pool may reach and still be the pond this
            # candidate describes — not from the wall, which says nothing about how far
            # back the water goes on a gentle valley floor.
            window = max(8, int(math.ceil(pool_reach_m / max(step, 1e-9))))
            volume, area, why = impounded_volume(
                dem, acc, row, col, crest, run, cell_area, window)
            if why:
                reason = why
                continue

            fill = embankment_volume(run, step, crest_width_m, batter)
            if fill <= 0 or volume <= 0:
                continue

            ratio = volume / fill
            if best is None or ratio > best["storage_ratio"]:
                best = {
                    "storage_ratio": ratio,
                    "storage_m3": volume,
                    "pond_area_m2": area,
                    "fill_m3": fill,
                    "wall_height_m": height,
                    "wall_length_m": wall_len,
                }

        if best is None:
            results.append(_refused(row, col, rc_to_xy, dem, acc, cell_area, reason))
            continue

        site = _site(row, col, rc_to_xy, dem, acc, cell_area)
        site.update(best)
        if runoff_mm:
            # A superb ratio on a site with no catchment is a hole in the ground, so
            # the yield is reported beside it — never folded into the rank.
            yield_m3 = (float(acc[row, col]) + 1.0) * cell_area * (runoff_mm / 1000.0)
            site["event_yield_m3"] = yield_m3
            site["fills_in_events"] = (
                best["storage_m3"] / yield_m3 if yield_m3 > 0 else None)
        site["notes"] = None
        site["label"] = (
            f"{best['storage_ratio']:.1f} m³ held per m³ of fill — "
            f"{best['storage_m3']:,.0f} m³ behind a {best['wall_height_m']:.1f} m × "
            f"{best['wall_length_m']:.0f} m wall"
        )
        results.append(site)

    results.sort(key=lambda s: s.get("storage_ratio", 0.0), reverse=True)
    for rank, site in enumerate(results, start=1):
        site["rank"] = rank
    return results


def _site(row, col, rc_to_xy, dem, acc, cell_area):
    x, y = rc_to_xy(row, col)
    z = float(dem[row, col]) if np.isfinite(dem[row, col]) else None
    return {
        "row": row, "col": col, "x": x, "y": y,
        "elevation": z,
        # Contributing area is a CELL COUNT, and it stays one here — the +1 is the
        # cell's own footprint, the same convention terrain_indices uses.
        "catchment_ha": (float(acc[row, col]) + 1.0) * cell_area / 10_000.0,
        "storage_ratio": 0.0,
        "storage_m3": 0.0,
        "fill_m3": 0.0,
        "pond_area_m2": 0.0,
        "wall_height_m": None,
        "wall_length_m": 0.0,
        "event_yield_m3": None,
        "fills_in_events": None,
        "notes": None,
        "label": "",
    }


def _refused(row, col, rc_to_xy, dem, acc, cell_area, reason):
    site = _site(row, col, rc_to_xy, dem, acc, cell_area)
    site["notes"] = reason
    site["label"] = f"No usable wall here — {reason}"
    return site
