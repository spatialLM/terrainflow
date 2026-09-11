"""p_topographic_valleys — can a keypoint be found without routing at all?

The owner's position, and it is well founded: **a keypoint is a topographic point.** Yeomans
locates it on a contour map and on the ground — it is the break where a primary valley's
steep upper floor eases into its gentler lower floor. Nothing about that definition mentions
a flow algorithm, yet `find_keypoints` currently answers a different keypoint depending on
whether the panel says D-infinity or D8 (`KPA-52`, §9.5): the two share **none** of their
results on the fixture.

This probe asks whether the routing can be removed rather than chosen between. It extracts
valley floors the way `find_ridgelines` extracts ridges — from **TPI**, which is
``z - mean(z)`` over a neighbourhood and is a pure landform measure — and then runs the
**unchanged** `keypoint_on_path` along them. No accumulation, no pointers, no threshold in
cells.

**What "primary valley" has to mean if flow goes away.** Today it means a Strahler order-1
link, which is a flow concept: a channel with no other channel draining into it. The
landform analogue is exact and needs no water — skeletonise the valley class, cut the
skeleton at its junctions, and a **leaf branch** (one whose end is a free end, not a
junction) is a valley nothing else joins from above. That is the same idea expressed in the
graph of the land rather than the graph of the water.

**Ordering matters and is easy to get wrong.** `keypoint_on_path` computes
``slope_ease = slope[hi] - slope[lo]`` and requires it positive, which holds only when the
path runs **downhill**. `stream_links` guarantees that ordering; a skeleton branch does not,
so each is sorted by elevation before it is offered.

This is a **measurement, not a proposal**. It answers: does a routing-free extraction find
keypoints at all, how many, and do they agree with either routing answer?

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_topographic_valleys.py
"""

import _probe

import numpy as np

TPI_WINDOW_M = 15.0
TPI_SD = 1.0
MIN_BRANCH_M = 50.0

#: Cells nearer the clip edge than this are not trusted to be valley floor.
#:
#: `landform_tpi`'s own docstring records the failure this prevents: a neighbourhood mean
#: taken over a window that hangs off the data edge is computed from fewer cells and drifts,
#: which *"fabricates a ridge line all the way round the data boundary"*. The same drift
#: fabricates valleys. The first run of this probe returned 7 keypoints of which **4** sat
#: on rows 1 and 397 or column 1 of a 400x400 clip — the edge, not the ground. Half a TPI
#: window is the distance at which the window is fully inside the data.
BOUNDARY_GUARD_CELLS = None   # derived per DEM: half the TPI window, in cells


# ------------------------------------------------------------------ extraction


def valley_mask(dem, cell_w, cell_h, window_m=TPI_WINDOW_M, sd=TPI_SD):
    """Cells the landform classifier calls valley — no flow of any kind involved."""
    from terrainflow_assessment.modules.terrain_indices import (
        landform_classes,
        landform_tpi,
    )

    tpi = landform_tpi(dem, cell_w, cell_h, window_m=window_m)
    valley = landform_classes(tpi, sd=sd) == -1

    # Drop the rim. Half a window is where the neighbourhood first sits entirely inside
    # the data, so it is the first row and column whose TPI means what it says.
    cell = (float(cell_w) + float(cell_h)) / 2.0
    guard = max(1, int(round((window_m / 2.0) / max(cell, 1e-9))))
    interior = np.zeros_like(valley, dtype=bool)
    interior[guard:-guard, guard:-guard] = True
    return valley & interior & np.isfinite(dem), tpi


def skeleton_branches(mask):
    """Split a thinned mask into branches at its junctions.

    A junction is a skeleton cell with three or more skeleton neighbours. Removing them
    leaves the branches as separate connected components, which is the landform analogue
    of a Strahler order-1 link: a run of valley floor that nothing else joins along its
    length.
    """
    from scipy.ndimage import label

    from terrainflow_assessment.modules.keypoint_analysis import _thin_to_centreline

    thin = _thin_to_centreline(mask)
    padded = np.pad(thin, 1, mode="constant", constant_values=False)
    neighbours = np.zeros_like(thin, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbours += padded[1 + dr:1 + dr + thin.shape[0],
                                 1 + dc:1 + dc + thin.shape[1]].astype(np.int16)
    junction = thin & (neighbours >= 3)
    branch_cells = thin & ~junction

    labelled, n = label(branch_cells, structure=np.ones((3, 3), dtype=int))
    out = []
    for tag in range(1, n + 1):
        rs, cs = np.nonzero(labelled == tag)
        if rs.size:
            out.append(list(zip(rs.tolist(), cs.tolist())))
    return out, thin, junction


def order_downhill(cells, dem, cell_w, cell_h):
    """Order a branch's cells into a polyline running downhill.

    Walked as a path rather than sorted by elevation alone: sorting would let the line
    jump between two limbs that happen to interleave in height. The walk starts at the
    highest endpoint and always steps to the nearest unvisited neighbour, which on a
    one-cell-wide skeleton is unambiguous.
    """
    if len(cells) < 2:
        return cells
    coord_set = set(cells)

    def nbrs(rc):
        r, c = rc
        return [(r + dr, c + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and (r + dr, c + dc) in coord_set]

    ends = [rc for rc in coord_set if len(nbrs(rc)) <= 1]
    if ends:
        start = max(ends, key=lambda rc: (dem[rc[0], rc[1]]
                                          if np.isfinite(dem[rc[0], rc[1]]) else -np.inf))
    else:
        start = max(coord_set, key=lambda rc: (dem[rc[0], rc[1]]
                                               if np.isfinite(dem[rc[0], rc[1]]) else -np.inf))

    ordered, visited, cur = [start], {start}, start
    while True:
        nxt = [n for n in nbrs(cur) if n not in visited]
        if not nxt:
            break
        cur = min(nxt, key=lambda rc: (abs(rc[0] - cur[0]) + abs(rc[1] - cur[1])))
        ordered.append(cur)
        visited.add(cur)

    # If the walk ended uphill of where it started, the branch was entered from its foot.
    z0 = dem[ordered[0][0], ordered[0][1]]
    z1 = dem[ordered[-1][0], ordered[-1][1]]
    if np.isfinite(z0) and np.isfinite(z1) and z1 > z0:
        ordered.reverse()
    return ordered


def branch_length_m(cells, cell_w, cell_h):
    total = 0.0
    for (r0, c0), (r1, c1) in zip(cells[:-1], cells[1:]):
        total += float(np.hypot((r1 - r0) * cell_h, (c1 - c0) * cell_w))
    return total


# --------------------------------------------------------------------- stages


def stage_topographic(ev, ya):
    with ev.stage("topographic_valleys") as rec:
        dem = ya.dem
        cw, ch = ya.cell_w, ya.cell_h
        mask, tpi = valley_mask(dem, cw, ch)
        rec["tpi_window_m"] = TPI_WINDOW_M
        rec["tpi_sd"] = TPI_SD
        rec["boundary_guard_cells"] = max(
            1, int(round((TPI_WINDOW_M / 2.0) / max((cw + ch) / 2.0, 1e-9))))
        rec["valley_cells"] = int(mask.sum())

        branches, thin, junction = skeleton_branches(mask)
        rec["thinned_cells"] = int(thin.sum())
        rec["junction_cells"] = int(junction.sum())
        rec["branches"] = len(branches)

        ordered = [order_downhill(b, dem, cw, ch) for b in branches]
        lengths = [branch_length_m(b, cw, ch) for b in ordered]
        keep = [(b, L) for b, L in zip(ordered, lengths) if L >= MIN_BRANCH_M]
        rec["min_branch_m"] = MIN_BRANCH_M
        rec["branches_over_min_length"] = len(keep)
        if lengths:
            rec["branch_length_m"] = {
                "median": round(float(np.median(lengths)), 2),
                "max": round(float(max(lengths)), 1),
            }

        keypoints = []
        for b, L in keep:
            kp = ya.keypoint_on_path(b, require_prominence=True)
            if kp is not None:
                kp["branch_length_m"] = round(L, 1)
                keypoints.append(kp)
        rec["keypoints"] = len(keypoints)
        rec["keypoint_rc"] = sorted((int(k["row"]), int(k["col"])) for k in keypoints)
        rec["keypoint_detail"] = [
            {"rc": [int(k["row"]), int(k["col"])],
             "elevation": k["elevation"],
             "slope_ease": k["slope_ease"],
             "branch_length_m": k["branch_length_m"]}
            for k in keypoints]

        ev.note(
            f"Topographic valleys (TPI {TPI_WINDOW_M:.0f} m, {TPI_SD:.1f} sd): "
            f"{rec['valley_cells']:,} valley cells thin to {rec['thinned_cells']:,}, "
            f"cut at {rec['junction_cells']} junctions into {rec['branches']} branches, "
            f"{rec['branches_over_min_length']} of them over {MIN_BRANCH_M:.0f} m. "
            f"Running the unchanged keypoint_on_path along those gives "
            f"**{rec['keypoints']} keypoints** at {rec['keypoint_rc']}. No accumulation, "
            f"no pointers, no routing — so this answer cannot depend on a routing setting.")
        return rec["keypoint_rc"]


def stage_compare(ev, ya, topo_rc, dem_path):
    """Against the two routing answers, so the comparison is like for like."""
    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    with ev.stage("against_the_routing_answers") as rec:
        work = _probe.workdir("topovalleys")
        rec["topographic"] = topo_rc
        for routing in ("dinf", "d8"):
            fa = FlowAnalysis()
            fa.load_dem(str(dem_path))
            fa.run(routing=routing)
            acc_path = str(work / f"acc_{routing}.tif")
            fa.save_result(fa.acc, acc_path, nodata=np.nan)
            other = YeomansKeylineAnalysis(str(dem_path), acc_path=acc_path,
                                           routing=routing)
            kps, _sk = other.find_keypoints(max_valleys=500)
            rc = sorted((int(k["row"]), int(k["col"])) for k in kps)
            rec[routing] = rc

            def near(a, b, tol=3):
                return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol

            shared = [x for x in topo_rc if any(near(x, y) for y in rc)]
            rec[f"shared_with_{routing}_within_3_cells"] = len(shared)

        ev.note(
            f"Keypoints: topographic {len(topo_rc)}, dinf {len(rec['dinf'])}, "
            f"d8 {len(rec['d8'])}. Of the topographic set, "
            f"{rec['shared_with_dinf_within_3_cells']} are within 3 cells of a dinf "
            f"keypoint and {rec['shared_with_d8_within_3_cells']} of a d8 one.")


def main():
    _probe.start_qgis()
    dem = _probe.fixture_path()
    _probe.banner("p_topographic_valleys — a keypoint without routing", dem)

    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    ev = _probe.Evidence("p_topographic_valleys", ["KPA-52"], dem)
    ev["dem_stats"] = _probe.dem_stats(dem)

    ya = YeomansKeylineAnalysis(str(dem))
    topo_rc = stage_topographic(ev, ya)
    if topo_rc is not None:
        stage_compare(ev, ya, topo_rc, dem)

    ev.write()


if __name__ == "__main__":
    main()
