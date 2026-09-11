"""Numerical regression against a fixed piece of real terrain.

Every other check asks whether the code *runs*. This one asks whether it still gives the
same **answers**. It runs the pipeline over a committed clip of the Quail Island survey
and compares the results against values recorded when the fixture was made.

Why a real DEM rather than the synthetic one the rest of the suite uses: the synthetic
terrain is smooth and noise-free — no pits, no flats, perfectly regular contours — so it
cannot tell you whether the sizing maths behaves on ground that has all three. The
fixture is 16 ha of actual hillside carrying an 8 ha catchment, which is the scale the
plugin is designed for.

**A moved number here is information, not necessarily a bug.** Level-pool swale
segmentation and the regional-CN work will both change these figures deliberately. The
point is that you find out, and then decide. When a change is intended, re-record the
constants below in the same commit that causes them to move.

Tolerances are proportional and deliberately tight (0.5%). Rounding differences across
platforms land far inside that; a real change to the sizing or routing maths lands well
outside it.
"""

import os

from _harness import PluginHarness
from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

FIXTURE_DEM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "quail_island_catchment.tif",
)

# Clip bounds (EPSG:2193), from the fixture's own header. Earthworks are placed at fixed
# offsets inside these so the geometry is identical on every run and on every machine.
X0, Y0 = 1574561.0, 5169232.0
SIZE = 400.0

# Storm held constant so a changed panel default cannot silently move these numbers.
STORM = {
    "rainfall_mm": 120.0,
    "duration_hr": 24.0,
    "cn": 61,
    "stream_threshold_ha": 1.0,
    "sizing_basis": "coefficient",
    "runoff_coefficient": 0.50,
}

# Recorded 2026-08-05 against fixture quail_island_catchment.tif; the two capacities
# re-recorded 2026-08-18 when the blanket 20% freeboard deduction was removed from
# `calculate_capacity` — both moved by exactly 1/0.8, which is the whole of the change.
#
# The three catchment figures and uncaptured_cells come off the conditioned flow graph —
# they are the terrain-dependent ones, and what the regional-CN work will move. The two
# capacities come from the sizing primitives, which is what level-pool segmentation will
# move. runoff_mm and catchment_area_ha are input-derived sanity anchors: if either drifts
# without the storm changing, something upstream of the terrain maths has broken.
EXPECTED = {
    "runoff_mm": 60.0000,
    "catchment_area_ha": 12.9600,
    "exit_points": 4.0000,
    "swale_a_catchment_ha": 1.6551,
    "swale_b_catchment_ha": 2.1829,
    "basin_c_catchment_ha": 0.1884,
    "uncaptured_cells": 89336.0000,
    "swale_capacity_m3": 252.0000,
    "basin_capacity_m3": 1170.0000,
}
TOLERANCE = 0.005


def _swale_line(northing_offset):
    """A straight line across the clip at a fixed northing — deterministic by construction."""
    y = Y0 + northing_offset
    return QgsGeometry.fromWkt(
        f"LINESTRING ({X0 + 60} {y}, {X0 + 200} {y}, {X0 + 340} {y})")


def _basin_polygon(east_offset, north_offset, side=30.0):
    x, y = X0 + east_offset, Y0 + north_offset
    return QgsGeometry.fromWkt(
        f"POLYGON (({x} {y}, {x + side} {y}, {x + side} {y + side}, {x} {y + side}, "
        f"{x} {y}))")


def _apply_storm(panel):
    """Pin the storm inputs through the panel's own restore path."""
    inputs = dict(panel.collect_inputs())
    inputs.update(STORM)
    panel.apply_inputs(inputs)


def _boundary_layer(inset_m=20.0):
    """A site boundary inset from the fixture's own bounds.

    The harness's `site_boundary_layer` is shaped around the synthetic DEM, so the
    fixture needs its own. Without a boundary the exit-point search has no perimeter to
    test against and reports none at all.
    """
    x0, y0 = X0 + inset_m, Y0 + inset_m
    x1, y1 = X0 + SIZE - inset_m, Y0 + SIZE - inset_m
    layer = QgsVectorLayer("Polygon?crs=EPSG:2193", "Fixture boundary", "memory")
    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromWkt(
        f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"))
    layer.dataProvider().addFeatures([feature])
    layer.updateExtents()
    return layer


def _size(earthwork):
    """Compute and store capacity the way the properties dialog does.

    `add_earthwork` puts a feature straight into the manager, bypassing the dialog — so
    without this every capacity stays at its 0.0 default and the check asserts nothing.
    """
    from terrainflow_assessment.modules.earthwork_design import calculate_capacity

    m3, litres = calculate_capacity(
        earthwork.type,
        earthwork.geometry,
        earthwork.depth,
        earthwork.width,
        companion_berm=getattr(earthwork, "companion_berm", False),
        bottom_width=getattr(earthwork, "bottom_width_m", None),
        batter_run=getattr(earthwork, "batter_run_m", None),
    )
    earthwork.capacity_m3 = m3
    earthwork.capacity_l = litres
    return earthwork


def _build_design(harness):
    """Two swales across the slope and one basin, at fixed coordinates."""
    swale = harness.add_earthwork("swale", geometry=_swale_line(140.0), name="Swale A")
    swale.depth = 0.6
    swale.top_width_m = 2.0
    swale.bottom_width_m = 1.0

    swale_b = harness.add_earthwork("swale", geometry=_swale_line(220.0), name="Swale B")
    swale_b.depth = 0.6
    swale_b.top_width_m = 2.0
    swale_b.bottom_width_m = 1.0

    basin = harness.add_earthwork("basin", geometry=_basin_polygon(170.0, 60.0),
                                  name="Basin C")
    basin.depth = 1.5
    basin.batter_run_m = 2.0

    return [_size(ew) for ew in (swale, swale_b, basin)]


def _relative_gap(actual, expected):
    if expected == 0:
        return abs(actual)
    return abs(actual - expected) / abs(expected)


def _compare(observed, failures):
    if not EXPECTED:
        print("    NO RECORDED VALUES — copy the block below into EXPECTED:")
        print("    EXPECTED = {")
        for key, value in observed.items():
            print(f'        "{key}": {value:.4f},')
        print("    }")
        failures.append(
            "EXPECTED is empty, so nothing was actually asserted. Record the values above."
        )
        return

    for key, expected in EXPECTED.items():
        actual = observed.get(key)
        if actual is None:
            failures.append(f"{key}: not produced by this run")
            continue
        gap = _relative_gap(actual, expected)
        marker = "ok " if gap <= TOLERANCE else "MOVED"
        print(f"    {marker} {key:24} {actual:14.3f}  (recorded {expected:.3f}, "
              f"{gap * 100:.2f}%)")
        if gap > TOLERANCE:
            failures.append(
                f"{key}: {actual:.3f} vs recorded {expected:.3f} "
                f"({gap * 100:.2f}% > {TOLERANCE * 100:.1f}%)"
            )


def check_fixture_dem_is_present_and_sane(dem_path):
    """The fixture must be committed, projected, and actual land rather than sea."""
    import numpy as np
    import rasterio

    assert os.path.exists(FIXTURE_DEM), (
        f"fixture DEM missing: {FIXTURE_DEM} — it is committed to the repo, not generated"
    )
    with rasterio.open(FIXTURE_DEM) as src:
        assert src.crs and src.crs.is_projected, "fixture DEM must be in a projected CRS"
        assert src.width == src.height == 400, f"unexpected grid {src.width}x{src.height}"
        assert abs(abs(src.transform.a) - 1.0) < 1e-6, "expected 1 m cells"
        data = src.read(1)

    # The survey encodes sea as a flat -0.1 m surface rather than nodata, so a fixture cut
    # from it can look valid while being mostly water. It happened on the first attempt.
    land = float((data > 1.0).mean())
    assert land > 0.95, f"fixture is only {land * 100:.0f}% land — re-cut it"
    assert np.ptp(data) > 20.0, "fixture has too little relief to route flow"


def check_real_terrain_numbers_have_not_moved(dem_path):
    """Run the pipeline over the fixture and compare against recorded values."""
    from qgis.core import QgsProject

    with PluginHarness(FIXTURE_DEM, load_boundary=False) as h:
        boundary = _boundary_layer()
        QgsProject.instance().addMapLayer(boundary)
        h.panel.boundary_changed.emit(boundary)

        _apply_storm(h.panel)
        _build_design(h)

        h.run_baseline()
        h.assert_no_errors("baseline over the fixture DEM")

        result = h.state.baseline_result
        assert result, "baseline produced no result over the fixture DEM"

        manager = h.state.earthwork_manager
        by_name = {ew.name: ew for ew in manager.get_all()}

        # Terrain-dependent: which cells drain to which feature. Computed off the flow
        # graph, so this is the figure the regional-CN work will move. It also confirms
        # the baseline_finished -> live-assessment wiring fired, since nothing else here
        # asks for catchments.
        counts = h.state.catchment_counts or {}
        cell_area = (h.state.flow_grid_meta or {}).get("cell_area_m2", 1.0)

        def catchment_ha(name):
            return float(counts.get(by_name[name].id, 0)) * cell_area / 1e4

        observed = {
            "runoff_mm": float(result.get("runoff_mm", 0.0)),
            "catchment_area_ha": float(result.get("catchment_area_m2", 0.0)) / 1e4,
            "exit_points": float(len(result.get("exit_points", []))),
            "swale_a_catchment_ha": catchment_ha("Swale A"),
            "swale_b_catchment_ha": catchment_ha("Swale B"),
            "basin_c_catchment_ha": catchment_ha("Basin C"),
            "uncaptured_cells": float(h.state.catchment_exit_cells
                                      + h.state.catchment_sink_cells),
            "swale_capacity_m3": float(by_name["Swale A"].capacity_m3 or 0.0),
            "basin_capacity_m3": float(by_name["Basin C"].capacity_m3 or 0.0),
        }

        print("\n    --- fixture numbers ---")
        failures = []
        _compare(observed, failures)
        print(f"    (exit points: {len(result.get('exit_points', []))})")

        assert not failures, (
            "numbers moved against the recorded fixture:\n      "
            + "\n      ".join(failures)
            + "\n    If the change was intended, re-record EXPECTED in this file."
        )


# ---------------------------------------------------------------------------
# Keyline network — the production path, pinned as integers
# ---------------------------------------------------------------------------
#
# Recorded 2026-09-11 against the same fixture, from the probes under
# `tests_qgis/probes/`. Every number here is measured **twice**: once by
# `p_flow_graph.py` / `p_keypoints.py`, and once by this check. If the two disagree, one
# of them is wrong and the disagreement is the finding.
#
# **These pin currently broken behaviour**, and that is deliberate. Each key names the
# entry in `CLudeDocs/ANALYSIS_DEFECTS.md` it belongs to, so the first real fix moves the
# number and the reviewer knows which finding moved it. The file's own rule applies:
# re-record the constants in the same commit that causes them to move.
#
# **Production path only.** `contour.py:1397` calls `find_keypoints(max_valleys=...)` and
# passes no threshold, so `stream_threshold_cells` takes its
# `max(20, round(2000 / cell_area))` default — 2,000 cells = 0.20 ha on this 1 m fixture.
# The 0.5 ha and 1.0 ha figures in the probe evidence are **sensitivity**, not production,
# and are not pinned. Neither are the conditioned-surface numbers: they are unreachable
# from production, which is the whole of KPA-38, so pinning them would pin probe code.
#
# **Integers, compared exactly.** `TOLERANCE` is 0.5 % relative, which is the right
# instrument for a capacity in m³ and the wrong one for a count of links — 0.5 % of 127 is
# 0.6, so a single link appearing or disappearing would pass. These are counts; they are
# compared with `==`.
#
# The 35.0 m profile floor is **not** pinned as a value. It is analytic — it falls out of
# `7 · min(5·cell, 10)` — not a property of this fixture, so a fixture check is the wrong
# home for it. What is pinned is how many links clear it.
EXPECTED_KEYLINE = {
    # FLG-18 — links emitted at min_cells=3 against links long enough to profile.
    "order1_links": 127,
    "links_clearing_profile_floor": 1,
    # KPA-39 — the guard histogram, derived here from link geometry alone (no settrace),
    # and equal to what p_keypoints.py traced. Two independent routes to 78 / 48 / 0.
    "links_refused_too_few_cells": 78,
    "links_refused_profile_too_short": 48,
    # KPA-43 — keypoints, skipped, and whether they account for the input.
    "keypoints_at_max_valleys_8": 1,
    "skipped_at_max_valleys_8": 126,
    "links_unaccounted_at_max_valleys_8": 0,
    # FLG-19 — the raw surface production actually walks, and what survives on it.
    "raw_sinks_on_finite_ground": 525,
    "raw_stream_cells_that_are_sinks": 42,
    "stream_cells": 1688,
}

# Recorded 2026-09-11, after KPA-12 closed (MATHS_AUDIT §9.9). Pinned because the number
# this replaced was **zero** and nothing noticed for as long as the feature has existed:
# ridgeline detection asked its TPI question in cells rather than metres and cut at an
# absolute 1.5 m, so it cleared its own bar on the 2 m synthetic surface the suite runs on
# and could not clear it on any 1 m DEM. A count taken on real 1 m ground is the only thing
# that would have caught that, so here it is.
#
# One is not many. The fixture is a 16 ha clip whose longest connected, thinned ridge run
# is 77 m, because `acc <= 2` fragments real ridges at every saddle. But one is the
# difference between a feature that works and a feature that never has, and if this returns
# to zero something has regressed.
EXPECTED_RIDGELINES = {
    "ridgelines": 1,
    "longest_ridgeline_m": 77.0,
}

#: Cells of the channel network at the production threshold, for context in the printout.
KEYLINE_THRESHOLD_CELLS = 2000


def _arc_length_m(link, cell_size):
    """Map-space length of an ordered run of cells, centre to centre."""
    total = 0.0
    for i in range(1, len(link)):
        dr = link[i][0] - link[i - 1][0]
        dc = link[i][1] - link[i - 1][1]
        total += (dr * dr + dc * dc) ** 0.5 * cell_size
    return total


def check_keyline_network_numbers_have_not_moved(dem_path):
    """The keyline network on real terrain, as integers, on the production path.

    `check_fixture_numbers_have_not_moved` asks whether the *sizing* answers still hold.
    This asks the same question of the **keyline network** — how many primary valleys the
    analysis finds, how many it refuses, and why — because that tier had no numeric
    regression at all and every finding in `ANALYSIS_DEFECTS.md` rests on these counts.

    Runs `YeomansKeylineAnalysis` directly rather than through the panel: the controller
    adds a worker, a progress callback and four map layers, none of which changes a
    number, and going straight at the module makes a moved figure point at the maths
    rather than at the plumbing. The **arguments** are production's — `max_valleys=8` is
    the panel default (`project_io.INPUT_FIELDS`), and no threshold is passed, exactly as
    `contour.py:1397` does it.
    """
    import numpy as np

    from terrainflow_assessment.modules.flow_graph import (
        d8_from_dem,
        strahler_order,
        stream_links,
    )
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    ya = YeomansKeylineAnalysis(FIXTURE_DEM)
    cell_area = ya.cell_w * ya.cell_h
    threshold = max(20, int(round(2_000.0 / cell_area)))

    # The floor `keypoint_on_path` cannot return a keypoint below: `spacing = min(5·cell,
    # 10)`, `win >= 5` so `guard >= 2`, and `n_samp - 2·guard >= 3` first holds at
    # `n_samp = 7`. Derived here rather than hard-coded so the arithmetic stays visible
    # next to the count it produces, and so the 2 m arm gets 70 m without an edit.
    spacing = min(5.0 * ya.cell_size, 10.0)
    profile_floor_m = 7.0 * spacing

    _fdir, acc_arr = ya._ensure_flow_data()
    stream = (acc_arr >= threshold) & np.isfinite(ya.dem)

    # d8 on the RAW DEM — which is what `find_keypoints:789` does, and is KPA-38.
    next_flat, is_sink = d8_from_dem(ya.dem, ya.cell_w, ya.cell_h)
    order = strahler_order(next_flat, stream.ravel())
    links = stream_links(next_flat, stream.ravel(), order, ya.dem.shape[1], max_order=1)

    lengths = [_arc_length_m(lk, ya.cell_size) for lk in links]
    finite_flat = np.isfinite(ya.dem).ravel()
    stream_flat = stream.ravel()

    keypoints, skipped = ya.find_keypoints(max_valleys=8)

    observed = {
        "order1_links": len(links),
        "links_clearing_profile_floor": sum(1 for m in lengths if m >= profile_floor_m),
        "links_refused_too_few_cells": sum(1 for lk in links if len(lk) < 5),
        "links_refused_profile_too_short": sum(
            1 for lk, m in zip(links, lengths)
            if len(lk) >= 5 and m < profile_floor_m),
        "keypoints_at_max_valleys_8": len(keypoints),
        "skipped_at_max_valleys_8": len(skipped),
        "links_unaccounted_at_max_valleys_8": (
            len(links) - (len(keypoints) + len(skipped))),
        "raw_sinks_on_finite_ground": int(np.count_nonzero(is_sink & finite_flat)),
        "raw_stream_cells_that_are_sinks": int(
            np.count_nonzero(stream_flat & is_sink)),
        "stream_cells": int(stream.sum()),
    }

    print("\n    --- keyline network (production path) ---")
    print(f"    threshold {threshold} cells = {threshold * cell_area / 1e4:.2f} ha; "
          f"profile floor {profile_floor_m:.1f} m")

    failures = []
    for key, expected in EXPECTED_KEYLINE.items():
        actual = observed[key]
        marker = "ok " if actual == expected else "MOVED"
        print(f"    {marker} {key:38s} {actual!s:>8}  (recorded {expected})")
        if actual != expected:
            failures.append(f"{key}: {actual} against a recorded {expected}")

    # Two things the arithmetic must satisfy whatever the recorded numbers are, so that a
    # re-recorded EXPECTED_KEYLINE cannot quietly encode nonsense.
    guard_total = (observed["links_refused_too_few_cells"]
                   + observed["links_refused_profile_too_short"]
                   + observed["links_clearing_profile_floor"])
    if guard_total != observed["order1_links"]:
        failures.append(
            f"the three link classes sum to {guard_total}, not "
            f"{observed['order1_links']} — they are meant to partition the link set")

    if observed["links_clearing_profile_floor"] < len(keypoints):
        failures.append(
            f"{len(keypoints)} keypoint(s) were found on only "
            f"{observed['links_clearing_profile_floor']} link(s) long enough to profile")

    assert not failures, (
        "the keyline network moved against the recorded fixture:\n      "
        + "\n      ".join(failures)
        + "\n    These pin CURRENT, PARTLY BROKEN behaviour — see "
          "CLudeDocs/ANALYSIS_DEFECTS.md. A fix is expected to move them; re-record "
          "EXPECTED_KEYLINE in the same commit, and say which finding moved it."
    )


def check_ridgelines_still_fire_on_real_ground(dem_path):
    """Ridgeline detection finds something on a real 1 m DEM.

    Guards the fix recorded in `MATHS_AUDIT` §9.9. Before it, this number was **0** on
    every 1 m DEM and the suite could not tell, because the suite runs on a 2 m synthetic
    surface where a cell-count TPI window happened to ask a question twice as large and
    cleared the absolute 1.5 m bar. The whole defect lived in the gap between those two
    resolutions, so the check has to be on the real fixture or it is worthless.

    Driven through the production signals rather than the module, because the thresholds
    are not exposed anywhere a user can reach and the defaults are the entire subject.
    """
    from terrainflow_assessment.modules.keypoint_analysis import DrainageLineAnalysis

    with PluginHarness(FIXTURE_DEM) as h:
        h.run_baseline()
        acc_path = (h.state.baseline_result or {}).get("flow_accumulation")
        assert acc_path, "no accumulation raster from the baseline"

        ka = DrainageLineAnalysis(FIXTURE_DEM, acc_path)
        ridgelines = ka.find_ridgelines()

    observed = {
        "ridgelines": len(ridgelines),
        "longest_ridgeline_m": (
            max(r["length_m"] for r in ridgelines) if ridgelines else 0.0),
    }

    print("\n    --- ridgelines on real ground ---")
    failures = []
    for key, expected in EXPECTED_RIDGELINES.items():
        actual = observed[key]
        ok = (actual == expected if isinstance(expected, int)
              else abs(actual - expected) <= 1.0)
        print(f"    {'ok ' if ok else 'MOVED'} {key:26s} {actual!s:>8}  "
              f"(recorded {expected})")
        if not ok:
            failures.append(f"{key}: {actual} against a recorded {expected}")

    assert observed["ridgelines"] > 0, (
        "ridgeline detection found nothing on the real fixture. That was the state "
        "before MATHS_AUDIT §9.9, and it held for every 1 m DEM while the synthetic "
        "suite stayed green. Check the TPI window units and the standard-deviation cut "
        "in DrainageLineAnalysis.find_ridgelines before re-recording anything."
    )
    assert not failures, (
        "ridgelines moved against the recorded fixture:\n      "
        + "\n      ".join(failures)
        + "\n    If the change was intended, re-record EXPECTED_RIDGELINES here and say "
          "which finding moved it."
    )
