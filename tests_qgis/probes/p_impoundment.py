"""p_impoundment — how much of a dam wall the transect forgets to count.

Step A1 of the defect-documentation campaign. Supports **IMP-01**, and records the site
list IMP-02/03/04 are about.

`transect_cells` lays a crest line across the valley by stepping `step = min(cell_w,
cell_h)` metres at a time along a bearing and rounding each step to a cell — then
**dedupes** repeated cells (`impoundment_sites.py:119-123`). On an axis-aligned bearing
one map-space step is one new cell. On a 45 deg bearing, consecutive cells are `sqrt(2) *
cell` apart, so one new cell covers `sqrt(2)` map-space steps and the surviving list is a
factor of `1/sqrt(2)` too short for the ground it spans.

Two consumers then count cells as if each were one `step` long:

* `wall_len = len(run) * step` (`:334`) — the crest length that `max_wall_m` refuses on,
  and the length printed in the site label.
* `embankment_volume` integrates one prismatic section per surviving cell (`:161-183`).

So a diagonal neck reports a shorter wall **and** a smaller fill than it has, which
inflates `storage_ratio = volume / fill` and biases the rank toward diagonal sites. The
existing rotation test compares north-south against east-west — both axis-aligned — so it
cannot see this.

**The candidates are production's, not raw `find_keypoints` output.** `_rank_pond_sites` is
seeded from `_state.found_keypoints` (`contour.py:1282`), which is the
*"Find Keypoints + Ridgelines"* heuristic — `DrainageLineAnalysis.find_keypoints`, a
different function from the Yeomans `YeomansKeylineAnalysis.find_keypoints` the rest of
this campaign measures. This probe therefore drives the harness: baseline, then the
find-keypoints signal, then the recommend-ponds signal, and reads `_state`.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_impoundment.py
"""

import math
import sys

import _probe

import numpy as np
import rasterio


def polyline_length_m(cells, cell_w, cell_h):
    """True map-space length of a crest laid through *cells*, centre to centre."""
    total = 0.0
    for i in range(1, len(cells)):
        dr = cells[i][0] - cells[i - 1][0]
        dc = cells[i][1] - cells[i - 1][1]
        total += math.hypot(dr * abs(cell_h), dc * abs(cell_w))
    return total


def drive_harness(ev, dem_path):
    """Baseline, find keypoints, recommend ponds — through the real signals."""
    _harness = _probe.start_qgis()
    _harness.make_workers_synchronous()

    with _harness.PluginHarness(str(dem_path)) as h:
        h.run_baseline()
        h.panel.run_keypoint_analysis_requested.emit()
        h.panel.recommend_ponds_requested.emit()

        found = list(getattr(h.state, "found_keypoints", []) or [])
        sites = list(getattr(h.state, "pond_sites", []) or [])
        bar = {
            "criticals": [m for m in h.bar.criticals],
            "warnings": [m for m in h.bar.warnings],
            "info_count": len(h.bar.messages) if hasattr(h.bar, "messages") else None,
        }
        acc_path = (h.state.baseline_result or {}).get("flow_accumulation")
        runoff_mm = (h.state.baseline_result or {}).get("runoff_mm")
        layers = h.layer_names()

    return {
        "found_keypoints": found,
        "pond_sites": sites,
        "acc_path": acc_path,
        "runoff_mm": runoff_mm,
        "layers": layers,
        "message_bar": bar,
    }


def stage_transect_undercount(ev, dem_path, driven):
    """Per site: the crest the code counted against the crest that is there."""
    from terrainflow_assessment.modules.dem_loader import horn_gradient
    from terrainflow_assessment.modules.impoundment_sites import (
        DEFAULT_BATTER,
        DEFAULT_CREST_WIDTH_M,
        DEFAULT_MAX_WALL_M,
        embankment_volume,
        flow_bearing,
        transect_cells,
        wall_run,
    )

    with ev.stage("transect_undercount") as rec:
        with rasterio.open(str(dem_path)) as src:
            dem = src.read(1).astype("float64")
            nodata = src.nodata
            cell_w = abs(src.transform.a)
            cell_h = abs(src.transform.e)
        if nodata is not None:
            dem[dem == nodata] = np.nan

        dz_dx, dz_dy, _invalid = horn_gradient(dem, cell_w, cell_h)
        step = min(cell_w, cell_h)

        rec["cell_w_m"] = cell_w
        rec["cell_h_m"] = cell_h
        rec["step_m"] = step
        rec["sites"] = []

        sites = driven["pond_sites"]
        rec["sites_returned"] = len(sites)
        rec["sites_ranked"] = sum(1 for s in sites if not s.get("notes"))
        rec["sites_refused"] = sum(1 for s in sites if s.get("notes"))
        rec["refusal_reasons"] = sorted({
            s["notes"] for s in sites if s.get("notes")})

        for site in sites:
            row, col = site["row"], site["col"]
            entry = {
                "rank_as_shipped": site.get("rank"),
                "row": row, "col": col,
                "notes": site.get("notes"),
                "wall_height_m": site.get("wall_height_m"),
                "wall_length_m_as_reported": site.get("wall_length_m"),
                "fill_m3_as_reported": site.get("fill_m3"),
                "storage_m3": site.get("storage_m3"),
                "storage_ratio_as_reported": site.get("storage_ratio"),
            }

            direction = flow_bearing(dz_dx, dz_dy, row, col)
            entry["bearing_vector"] = (
                None if direction is None else [float(direction[0]), float(direction[1])])
            if direction is None:
                entry["skipped"] = "level ground — flow_bearing returned None"
                rec["sites"].append(entry)
                continue

            # The bearing the crest is laid on is perpendicular to the downslope vector.
            ex, ny = direction
            px, py = -ny, ex
            bearing_deg = math.degrees(math.atan2(px, py)) % 180.0
            entry["crest_bearing_deg"] = bearing_deg
            entry["degrees_off_axis"] = min(
                abs(bearing_deg - a) for a in (0.0, 90.0, 180.0))

            cells, step_back = transect_cells(
                dem.shape, row, col, direction, cell_w, cell_h, DEFAULT_MAX_WALL_M)
            entry["transect_cells"] = len(cells)
            entry["transect_steps_requested"] = 2 * max(
                1, int(round(DEFAULT_MAX_WALL_M / max(step, 1e-9)))) + 1
            entry["cells_per_step"] = (
                len(cells) / entry["transect_steps_requested"])
            entry["transect_span_m_as_counted"] = len(cells) * step_back
            entry["transect_span_m_true"] = polyline_length_m(cells, cell_w, cell_h)

            height = site.get("wall_height_m")
            if height is None or not np.isfinite(dem[row, col]):
                entry["skipped"] = "refused site — no winning wall height to reproduce"
                rec["sites"].append(entry)
                continue

            try:
                here = cells.index((row, col))
            except ValueError:
                entry["skipped"] = "candidate not on its own transect"
                rec["sites"].append(entry)
                continue

            crest = float(dem[row, col]) + float(height)
            run = wall_run(dem, cells, crest, here)
            if not run:
                entry["skipped"] = "wall_run empty at the winning height"
                rec["sites"].append(entry)
                continue

            run_cells = [(r, c) for r, c, _h in run]
            counted = len(run) * step_back
            # Centre-to-centre through the run, plus one cell so a single-cell wall is
            # one cell long rather than zero — the same convention `len(run) * step` uses.
            true = polyline_length_m(run_cells, cell_w, cell_h) + step_back
            entry["wall_cells"] = len(run)
            entry["wall_length_m_counted"] = counted
            entry["wall_length_m_true"] = true
            entry["wall_length_understatement"] = (
                (true - counted) / true if true > 0 else None)

            fill_counted = embankment_volume(
                run, step_back, DEFAULT_CREST_WIDTH_M, DEFAULT_BATTER)
            scale = (true / counted) if counted > 0 else 1.0
            fill_true = embankment_volume(
                run, step_back * scale, DEFAULT_CREST_WIDTH_M, DEFAULT_BATTER)
            entry["fill_m3_counted"] = fill_counted
            entry["fill_m3_true"] = fill_true
            entry["storage_ratio_corrected"] = (
                site["storage_m3"] / fill_true if fill_true > 0 else None)
            entry["ratio_inflation"] = (
                site["storage_ratio"] / entry["storage_ratio_corrected"]
                if entry["storage_ratio_corrected"] else None)
            entry["crosses_max_wall_when_corrected"] = true > DEFAULT_MAX_WALL_M

            rec["sites"].append(entry)

        # Would correcting the fill change the order the user is shown?
        corrected = [
            (s["storage_ratio_corrected"], s["rank_as_shipped"])
            for s in rec["sites"] if s.get("storage_ratio_corrected") is not None]
        corrected.sort(key=lambda t: t[0], reverse=True)
        rec["rank_as_shipped_in_corrected_order"] = [r for _v, r in corrected]
        rec["rank_order_changes"] = (
            rec["rank_as_shipped_in_corrected_order"]
            != sorted(rec["rank_as_shipped_in_corrected_order"]))

        measured = [s for s in rec["sites"]
                    if s.get("wall_length_understatement") is not None]
        if measured:
            worst = max(measured, key=lambda s: s["wall_length_understatement"])
            ev.note(
                f"IMP-01, measured on {len(measured)} ranked site(s): the worst "
                f"under-statement of crest length is "
                f"{worst['wall_length_understatement']:.1%} "
                f"({worst['wall_length_m_counted']:.1f} m counted against "
                f"{worst['wall_length_m_true']:.1f} m on the ground) at a crest bearing "
                f"{worst['crest_bearing_deg']:.1f} deg, "
                f"{worst['degrees_off_axis']:.1f} deg off axis. Its fill is understated "
                f"in the same proportion, so storage_ratio is inflated "
                f"{worst['ratio_inflation']:.2f}x. Rank order changes when corrected: "
                f"{rec['rank_order_changes']}.")
        else:
            ev.note(
                "IMP-01: no site on this fixture produced a wall at any trial height, so "
                "the under-count has no ranked site to bite on here. The transect "
                "geometry is still measured above — see 'cells_per_step', which is the "
                "defect in its purest form.")

        per_step = [s["cells_per_step"] for s in rec["sites"]
                    if s.get("cells_per_step") is not None]
        if per_step:
            rec["cells_per_step_range"] = [min(per_step), max(per_step)]
            ev.note(
                "IMP-01 geometry, independent of whether a wall was found: "
                f"transect_cells returns between {min(per_step):.3f} and "
                f"{max(per_step):.3f} distinct cells per map-space step across "
                f"{len(per_step)} candidate(s). 1.000 is axis-aligned; 0.707 is 45 deg. "
                "Everything below 1.000 is crest length and fill silently lost.")


def main():
    dem = _probe.fixture_path()
    _probe.start_qgis()
    _probe.banner("p_impoundment — transect dedup against map-space crest length", dem)

    ev = _probe.Evidence("p_impoundment", findings=["IMP-01", "IMP-02", "IMP-03",
                                                    "IMP-04"], dem=dem)
    ev["dem_stats"] = _probe.dem_stats(dem)

    with ev.stage("drive_production_path") as rec:
        driven = drive_harness(ev, dem)
        rec["found_keypoints"] = len(driven["found_keypoints"])
        rec["pond_sites"] = len(driven["pond_sites"])
        rec["runoff_mm"] = driven["runoff_mm"]
        rec["layers"] = driven["layers"]
        rec["message_bar"] = driven["message_bar"]
        rec["site_labels"] = [s.get("label") for s in driven["pond_sites"]]
        ev.note(
            f"Production path: {len(driven['found_keypoints'])} heuristic keypoints "
            f"seeded {len(driven['pond_sites'])} pond candidate(s). The seed is "
            "_state.found_keypoints (DrainageLineAnalysis), not the Yeomans "
            "find_keypoints this campaign measures elsewhere.")

    if driven["pond_sites"]:
        stage_transect_undercount(ev, dem, driven)
    else:
        ev.note(
            "No pond candidates were produced, so IMP-01 has nothing to measure on this "
            "fixture through the production path.")

    ev.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
