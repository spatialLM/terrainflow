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
    # Re-recorded 2026-09-12: 89,336 -> 52,839 when `_build_design` gained Berm D and
    # Diversion E. The two new features intercept ground that previously reached the
    # boundary uncaptured, which is what a berm across the slope is for. Everything
    # else in this block held to 0.00% — direct catchments are mutually exclusive, so
    # the new features take from the uncaptured remainder rather than from Swale A,
    # Swale B or Basin C, and the two capacities are analytic and never saw the burn.
    "uncaptured_cells": 52839.0000,
    "swale_capacity_m3": 252.0000,
    "basin_capacity_m3": 1170.0000,
}
TOLERANCE = 0.005


def _swale_line(northing_offset):
    """A straight line across the clip at a fixed northing — deterministic by construction."""
    y = Y0 + northing_offset
    return QgsGeometry.fromWkt(
        f"LINESTRING ({X0 + 60} {y}, {X0 + 200} {y}, {X0 + 340} {y})")


def _diversion_line():
    """A graded drain on ground that falls the way a drain needs — measured, not guessed.

    `_burn_diversion` grades from the **first vertex** at `gradient_pct` and keeps
    the lower of that invert and the ground, so the alignment decides whether this
    pins a drain or a canyon. The first alignment tried here ran east-west, which
    on this clip **rises 17.3 m** along its length: the graded invert then sat some
    20 m below ground for most of the line and the burn cut a trench to match. A
    real code path, and a worthless thing to hang a site total on.

    This one falls **1.90 m over 126 m (-1.51%)** against the 1.0% the drain is
    graded at, so the invert tracks the ground and cuts the shallow trench a
    diversion actually is. The slope on this clip runs east-west — the berm
    alignment rises 23 m along it — which is why this runs across that.

    Off the half-metre on purpose: a band whose edges land exactly on cell centres
    is the one alignment where centre-based rasterising is ambiguous, and a pin
    taken there would pin the ambiguity.
    """
    return QgsGeometry.fromWkt(
        f"LINESTRING ({X0 + 260.3} {Y0 + 300.3}, {X0 + 200.3} {Y0 + 280.3}, "
        f"{X0 + 140.3} {Y0 + 260.3})")


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
    """One of every burn method, at fixed coordinates.

    It was two swales and a basin, and that left a hole worth naming: `_burn_swale`
    and `_burn_basin` were the two volumetric burns already rasterising on cell
    centres, so **no pinned number here touched `_burn_berm`, `_burn_diversion` or
    `_key_berm_into_banks`**. When those three were fixed (M-1) and the berm's
    section was changed from a 1:1 triangle to a trapezoid (M-2), every pin in this
    module held — not because the burns were unchanged, but because nothing pinned
    reached them.

    So: a berm, a diversion, and a companion berm on Swale B, which is the only way
    `_key_berm_into_banks` runs (`_companion_berm` calls it when `key_into_banks` is
    set, and that defaults True for a swale). Together with the burn quantities
    pinned in `check_the_burn_quantities_have_not_moved`, every burn method the
    plugin has is now behind a recorded number.
    """
    swale = harness.add_earthwork("swale", geometry=_swale_line(140.0), name="Swale A")
    swale.depth = 0.6
    swale.top_width_m = 2.0
    swale.bottom_width_m = 1.0

    swale_b = harness.add_earthwork("swale", geometry=_swale_line(220.0), name="Swale B")
    swale_b.depth = 0.6
    swale_b.top_width_m = 2.0
    swale_b.bottom_width_m = 1.0
    # The keyed-berm path. `key_into_banks` is already True by default for a swale,
    # so this one flag is what reaches `_key_berm_into_banks` — and it also puts a
    # companion berm's spoil into the fill total below.
    swale_b.companion_berm = True

    basin = harness.add_earthwork("basin", geometry=_basin_polygon(170.0, 60.0),
                                  name="Basin C")
    basin.depth = 1.5
    basin.batter_run_m = 2.0

    berm = harness.add_earthwork("berm", geometry=_swale_line(300.0), name="Berm D")
    berm.depth = 0.5
    berm.top_width_m = 2.0

    drain = harness.add_earthwork("diversion", geometry=_diversion_line(),
                                  name="Diversion E")
    drain.depth = 0.5
    drain.top_width_m = 2.0
    drain.bottom_width_m = 1.0
    drain.gradient_pct = 1.0

    return [_size(ew) for ew in (swale, swale_b, basin, berm, drain)]


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
# **Re-recorded 2026-09-11 when KPA-52 closed** (ANALYSIS_DEFECTS §10). Until then these
# pinned deliberately broken behaviour — 127 links of median 3.4 m, one keypoint — because
# the stream mask came from pysheds' D-infinity accumulation while the pointers came from
# `d8_from_dem` on the raw DEM, and links ended wherever the two disagreed. Now the mask,
# the pointers and the accumulation are one graph on the conditioned surface, every valley
# is walked back up to its divide and cut at the data edge, and the keypoint is the break
# of a two-slope fit. Each key still names the finding it belongs to. The file's own rule
# applies: re-record the constants in the same commit that causes them to move.
#
# **Production path only.** `contour.py` calls `find_keypoints(max_valleys=...)` and
# passes no threshold, so `stream_threshold_cells` takes its
# `max(20, round(2000 / cell_area))` default — 2,000 cells = 0.20 ha on this 1 m fixture.
# The 0.5 ha and 1.0 ha figures in the probe evidence are **sensitivity**, not production,
# and are not pinned.
#
# **Integers, compared exactly.** `TOLERANCE` is 0.5 % relative, which is the right
# instrument for a capacity in m³ and the wrong one for a count of links — 0.5 % of 23 is
# 0.1, so a single link appearing or disappearing would pass. These are counts; they are
# compared with `==`.
#
# There is no longer a 35 m profile floor to pin against: the two-slope fit needs
# `2 · MIN_REACH_CELLS + 1` cells and nothing else, and every valley on this fixture has
# at least 74 once it starts at its divide.
EXPECTED_KEYLINE = {
    # KPA-52 — the network, on one graph. 23 primary valleys, not 127 fragments.
    "order1_links": 23,
    "stream_cells": 1571,
    "conditioned_sinks_on_finite_ground": 65,
    # KPA-52 — the divide and the data edge. Half the cells of a typical valley lie
    # above the old channel head; 7 valleys reach the grid edge and are cut there; 3
    # have their divide on it and are keyed with a flag.
    "median_extension_cells": 91,
    "links_cut_at_the_data_edge": 7,
    "links_with_divide_on_the_data_edge": 3,
    # KPA-39 — the refusal classes, now read off the reasons `find_keypoints` gives.
    # Five valleys are refused before the fit because their channel exists only on the
    # boundary row (the row's own collecting artefact — no channel is on the map); the
    # fit refuses 8 for having no two-slope break.
    "links_refused_before_the_fit": 5,
    "links_refused_no_two_slope_break": 8,
    "keypoints_at_every_valley": 10,
    # KPA-43 — at the panel default the loop stops at 8 keypoints; the 10 valleys it
    # never reached are reported as "not examined", so keypoints + skipped == valleys
    # at every cap (asserted below as a property).
    "keypoints_at_max_valleys_8": 8,
    "skipped_at_max_valleys_8": 15,
    "unexamined_at_max_valleys_8": 10,
}

# Recorded 2026-09-11, after KPA-12 closed (MATHS_AUDIT §9.9). Pinned because the number
# this replaced was **zero** and nothing noticed for as long as the feature has existed:
# ridgeline detection asked its TPI question in cells rather than metres and cut at an
# absolute 1.5 m, so it cleared its own bar on the 2 m synthetic surface the suite runs on
# and could not clear it on any 1 m DEM. A count taken on real 1 m ground is the only thing
# that would have caught that, so here it is.
#
# Re-recorded 2026-09-12, 1 -> 7 lines and 77 -> 79 m, by the two faults `p_ridgelines`
# found: `nd_label` grouped the skeleton 4-connected while `_order_pixels` walked it
# 8-connected, so every diagonal run was labelled one component per cell; and the
# accumulation bar was a bare `acc <= 2` rather than a catchment area, which asked a
# different question at every resolution. On the owner's 1139x1016 design the first of
# those was a ceiling — no setting of any exposed threshold could produce one 50-cell
# component — which is why this clip managed one line and the real site managed none.
#
# Seven is not many either. The fixture is a 16 ha clip and this is still a proxy for a
# ridge rather than a definition of one. But if it returns to zero something has regressed:
# zero is where the feature sat for its whole life before MATHS_AUDIT §9.9.
EXPECTED_RIDGELINES = {
    "ridgelines": 7,
    "longest_ridgeline_m": 79.0,
    # A count and a maximum both survive the middle of the distribution changing
    # completely — seven lines could each be replaced and leave both fixed. The total
    # is the term that moves when they do.
    "total_ridgeline_m": 216.0,
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
    """The keyline network on real terrain, as integers, with no baseline to inherit.

    **This pins the no-baseline branch**, which is what it has always pinned and is no
    longer the whole story. Since `KPA-48`/`KPA-53` were fixed, a keyline press *after a
    baseline* hands `_ensure_flow_data` the baseline's crest-split accumulation and gets a
    different, better answer — 122 valleys refused rather than 126, in 0.02 s rather than
    0.50 s. That branch is pinned by `check_keyline_network_with_a_baseline_has_not_moved`
    below. This one still matters: a user can press Keyline before Baseline, and then this
    is exactly what they get.

    `check_fixture_numbers_have_not_moved` asks whether the *sizing* answers still hold.
    This asks the same question of the **keyline network** — how many primary valleys the
    analysis finds, how many it refuses, and why — because that tier had no numeric
    regression at all and every finding in `ANALYSIS_DEFECTS.md` rests on these counts.

    Runs `YeomansKeylineAnalysis` directly rather than through the panel: the controller
    adds a worker, a progress callback and four map layers, none of which changes a
    number, and going straight at the module makes a moved figure point at the maths
    rather than at the plumbing. The **arguments** are production's — `max_valleys=8` is
    the panel default (`project_io.INPUT_FIELDS`), and no threshold is passed, exactly as
    `contour.py` does it. The valleys come from `primary_valleys()` itself rather than a
    hand-built copy of its construction, so this cannot drift from what production walks.

    Two properties are asserted whatever the recorded numbers say: **no stream cell's
    pointer leaves the stream mask** (the mask and the pointers are one graph — the whole
    of KPA-52), and **keypoints + skipped == valleys** when the cap is lifted (KPA-43's
    identity, which only the cap breaks).
    """
    import numpy as np

    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    ya = YeomansKeylineAnalysis(FIXTURE_DEM)
    cell_area = ya.cell_w * ya.cell_h
    threshold = max(20, int(round(2_000.0 / cell_area)))

    valleys = ya.primary_valleys()
    next_flat, own_acc = ya._primary_graph()
    finite_flat = np.isfinite(ya.dem).ravel()
    stream_flat = (own_acc >= threshold) & finite_flat
    is_sink = next_flat == np.arange(next_flat.size)
    leaving = int(sum(1 for i in np.flatnonzero(stream_flat)
                      if next_flat[i] != i and not stream_flat[next_flat[i]]))

    keypoints, skipped = ya.find_keypoints(max_valleys=8)
    all_keypoints, all_skipped = ya.find_keypoints(max_valleys=len(valleys) + 1)

    observed = {
        "order1_links": len(valleys),
        "stream_cells": int(stream_flat.sum()),
        "conditioned_sinks_on_finite_ground": int(np.count_nonzero(is_sink & finite_flat)),
        "median_extension_cells": int(np.median([v["extension_cells"] for v in valleys])),
        "links_cut_at_the_data_edge": sum(1 for v in valleys if v["runs_off_dem_m"] > 0),
        "links_with_divide_on_the_data_edge": sum(
            1 for v in valleys if v["head_on_boundary"]),
        "links_refused_before_the_fit": sum(
            1 for s in all_skipped if "grade change" not in s
            and YeomansKeylineAnalysis.NOT_EXAMINED not in s),
        "links_refused_no_two_slope_break": sum(
            1 for s in all_skipped if "grade change" in s),
        "keypoints_at_every_valley": len(all_keypoints),
        "keypoints_at_max_valleys_8": len(keypoints),
        "skipped_at_max_valleys_8": len(skipped),
        "unexamined_at_max_valleys_8": sum(
            1 for s in skipped if YeomansKeylineAnalysis.NOT_EXAMINED in s),
    }

    print("\n    --- keyline network (production path) ---")
    print(f"    threshold {threshold} cells = {threshold * cell_area / 1e4:.2f} ha; "
          f"conditioned surface {ya.conditioned_source}; "
          f"{leaving} pointer(s) leave the mask")

    failures = []
    for key, expected in EXPECTED_KEYLINE.items():
        actual = observed[key]
        marker = "ok " if actual == expected else "MOVED"
        print(f"    {marker} {key:38s} {actual!s:>8}  (recorded {expected})")
        if actual != expected:
            failures.append(f"{key}: {actual} against a recorded {expected}")

    # Properties, not recorded numbers: these hold on any terrain, so a failure here is a
    # defect and never a re-record.
    if leaving != 0:
        failures.append(
            f"{leaving} stream cell(s) have a pointer that leaves the stream mask — the "
            "mask and the pointers are no longer one graph (KPA-52)")
    partition = (observed["links_refused_before_the_fit"]
                 + observed["links_refused_no_two_slope_break"]
                 + observed["keypoints_at_every_valley"])
    if partition != observed["order1_links"]:
        failures.append(
            f"the refusal classes and the keypoints sum to {partition}, not "
            f"{observed['order1_links']} — they are meant to partition the valley set")
    for label, kps, sk in (("uncapped", all_keypoints, all_skipped),
                           ("at the cap of 8", keypoints, skipped)):
        if len(kps) + len(sk) != len(valleys):
            failures.append(
                f"{label}, keypoints + skipped = {len(kps) + len(sk)} against "
                f"{len(valleys)} valleys (KPA-43's identity)")
    if len(keypoints) > 8:
        failures.append(f"{len(keypoints)} keypoints returned against a cap of 8")

    assert not failures, (
        "the keyline network moved against the recorded fixture:\n      "
        + "\n      ".join(failures)
        + "\n    See CLudeDocs/ANALYSIS_DEFECTS.md §10. If the maths changed on purpose, "
          "re-record EXPECTED_KEYLINE in the same commit, and say which finding moved it."
    )


#: The same network, reached the way a user reaches it: Baseline, then Keyline. Recorded
#: 2026-09-11 against the fixture, after `KPA-48`/`KPA-53` were fixed and re-recorded the
#: same day when `KPA-52` closed. The valley *network* is now the same on both branches —
#: it comes from the conditioned surface, which the baseline hands over and the recompute
#: reproduces cell for cell — but the supplied accumulation is **crest-split** and the
#: recompute is not, so the valleys *rank* differently and the eighth keypoint at the cap
#: can differ. Uncapped, the two keypoint sets are identical.
EXPECTED_KEYLINE_WITH_BASELINE = {
    "keypoints": 8,
    "skipped": 15,                  # 5 refused + 10 not examined at the cap (KPA-43)
    "keypoints_at_every_valley": 10,
    "valleys": 23,
    # The supplied raster must be used *as supplied*. A single differing cell means the
    # gate has closed again and the tier is back on its own recompute.
    "cells_differing_from_the_supplied_raster": 0,
    # And the supplied conditioned surface must be the one the graph is built on.
    "conditioned_source": "supplied",
}


def check_keyline_network_with_a_baseline_has_not_moved(dem_path):
    """Keyline after Baseline — the sequence a user actually performs.

    `contour.py:1390` hands `YeomansKeylineAnalysis` the baseline's accumulation raster.
    Until `KPA-48` was fixed the gate in `_ensure_flow_data` required *both* a flow-
    direction path and an accumulation path, so that argument was inert: every keyline
    press threw the supplied field away and spent 0.48 s recomputing a different one,
    uncorrected for ponds. This pins the corrected behaviour.

    Four things are asserted and each fails differently:

    1. **The supplied raster is used as supplied** — zero differing cells. This is the
       direct regression guard on the gate. If someone restores the `and`, this is the
       assertion that says so, rather than a count drifting for no visible reason.
    2. **The supplied conditioned surface is the one the valley graph is built on**
       (`conditioned_source == "supplied"`), and it yields the same pointers as the
       recompute would — zero pointer disagreements — so the two branches cannot drift
       into two networks.
    3. **The counts**, at the cap and uncapped.
    4. **`keypoints + skipped == valleys`** uncapped, so `KPA-43`'s accounting identity
       is checked against the field production actually uses.

    Not asserted: wall time. A keyline press with a baseline is now 0.14 s against 0.81 s
    without one, and that is the *reason* the surface is handed over, but a timing
    assertion on a shared machine is a flake generator.
    """
    import numpy as np
    import rasterio

    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    import tempfile

    with tempfile.TemporaryDirectory(prefix="tfa_keyline_baseline_") as tmp:
        import os as _os

        # Crest-split, as production's baseline is — that is what makes the supplied
        # field differ from a plain recompute. The conditioned surface is written the way
        # the analysis worker writes it: float64, with the DEM's own nodata.
        fa = FlowAnalysis()
        fa.load_dem(FIXTURE_DEM)
        fa.run(routing="dinf")
        acc_path = _os.path.join(tmp, "baseline_acc.tif")
        fa.save_result(fa.acc, acc_path, nodata=np.nan)
        cond_path = _os.path.join(tmp, "baseline_conditioned.tif")
        fa.save_result(np.array(fa.conditioned, dtype="float64"), cond_path,
                       "hydrologically conditioned DEM", dtype="float64",
                       nodata=fa.nodata)

        ya = YeomansKeylineAnalysis(FIXTURE_DEM, acc_path=acc_path,
                                    conditioned_path=cond_path)
        _fdir, acc_arr = ya._ensure_flow_data()

        with rasterio.open(acc_path) as src:
            on_disk = src.read(1).astype("float64")
        used = np.asarray(acc_arr, dtype="float64")
        both = np.isfinite(used) & np.isfinite(on_disk)
        differing = int((used[both] != on_disk[both]).sum())

        valleys = ya.primary_valleys()
        keypoints, skipped = ya.find_keypoints(max_valleys=8)
        all_keypoints, all_skipped = ya.find_keypoints(max_valleys=len(valleys) + 1)

        # The graph the supplied surface gives against the graph a recompute gives.
        supplied_next, _acc = ya._primary_graph()
        recomputed = YeomansKeylineAnalysis(FIXTURE_DEM)
        recomputed_next, _acc = recomputed._primary_graph()
        pointer_disagreements = int((supplied_next != recomputed_next).sum())

    observed = {
        "keypoints": len(keypoints),
        "skipped": len(skipped),
        "keypoints_at_every_valley": len(all_keypoints),
        "valleys": len(valleys),
        "cells_differing_from_the_supplied_raster": differing,
        "conditioned_source": ya.conditioned_source,
    }

    print("\n    --- keyline after baseline (the supplied field) ---")
    failures = []
    for key, expected in EXPECTED_KEYLINE_WITH_BASELINE.items():
        actual = observed[key]
        print(f"    {'ok ' if actual == expected else 'MOVED'} {key:44s} "
              f"{actual!s:>6}  (recorded {expected})")
        if actual != expected:
            failures.append(f"{key}: {actual} against a recorded {expected}")

    assert differing == 0, (
        f"{differing:,} cells of the accumulation raster handed to "
        "YeomansKeylineAnalysis were not the ones it used.\n"
        "    `acc_path` is being ignored again — check the gate in "
        "`_ensure_flow_data`, which must key on accumulation alone.\n"
        "    That gate was an `and` over two paths and the controller supplies one, "
        "which is KPA-48."
    )
    for label, kps, sk in (("uncapped", all_keypoints, all_skipped),
                           ("at the cap of 8", keypoints, skipped)):
        if len(kps) + len(sk) != len(valleys):
            failures.append(
                f"{label}, keypoints + skipped = {len(kps) + len(sk)} against "
                f"{len(valleys)} valleys (KPA-43's identity, on the supplied field)")
    if pointer_disagreements:
        failures.append(
            f"{pointer_disagreements} pointer(s) differ between the supplied conditioned "
            "surface and a recompute — the two branches are walking two graphs")

    assert not failures, (
        "the keyline network moved on the branch production actually takes:\n      "
        + "\n      ".join(failures)
        + "\n    These are pinned separately from EXPECTED_KEYLINE because the supplied "
          "field is crest-split and a recompute is not.\n    Re-record "
          "EXPECTED_KEYLINE_WITH_BASELINE in the same commit that moves them, and say "
          "which finding did it."
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
        "total_ridgeline_m": float(sum(r["length_m"] for r in ridgelines)),
    }

    print("\n    --- ridgelines on real ground ---")
    failures = []
    # Walks `observed`, not `EXPECTED_RIDGELINES`. Walking the recorded dict means a
    # newly observed quantity is silently never asserted — it is simply absent from
    # the loop — so an added measurement looks recorded while pinning nothing.
    for key, actual in observed.items():
        if key not in EXPECTED_RIDGELINES:
            failures.append(f"{key}: {actual} observed with nothing recorded for it")
            print(f"    NEW   {key:26s} {actual!s:>8}  (nothing recorded)")
            continue
        expected = EXPECTED_RIDGELINES[key]
        ok = (actual == expected if isinstance(expected, int)
              else abs(actual - expected) <= 1.0)
        print(f"    {'ok ' if ok else 'MOVED'} {key:26s} {actual!s:>8}  "
              f"(recorded {expected})")
        if not ok:
            failures.append(f"{key}: {actual} against a recorded {expected}")
    for key in EXPECTED_RIDGELINES:
        if key not in observed:
            failures.append(f"{key}: recorded but no longer observed")

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


# ---------------------------------------------------------------------------
# Step C / D / F — invariants and cross-checks, rather than recorded numbers
# ---------------------------------------------------------------------------
#
# Everything above this line pins a *measurement* and has to be re-recorded when the
# maths deliberately changes. Everything below pins a *property*: a relation that holds
# on any terrain at any elevation, so a failure is always a defect and none of it ever
# needs re-recording. That difference is why they can share a file without adding to the
# maintenance burden the constants above carry.
#
# Measured first by `tests_qgis/probes/p_invariance.py`, `p_battery.py` and
# `p_crosscheck.py`; every number quoted in an assertion message came from those runs on
# 2026-09-11. See `CLudeDocs/ANALYSIS_DEFECTS.md` §8.

#: Elevation offset for the invariance arm. Chosen because float32's spacing crosses
#: `resolve_flats`' 1e-5 m inflation step at roughly 600 m — the regime `save_result`'s
#: docstring, `earthwork_design.py:2137` and `KPA-38`'s fix note all reason about and nothing
#: else in either suite reaches.
Z_OFFSET_M = 600.0

#: Signed-witness bars, from `p_battery.py`. Deliberately far below what correct code
#: achieves (measured: +0.489 raw, 100% for aspect) so the check fails on a flipped sign
#: rather than on a different DEM.
MIN_POINT_BISERIAL = 0.25
MIN_ASPECT_AGREEMENT = 0.90

#: `haul_regions` and `burn_quantities` reduce the same two surfaces to a volume by
#: different routes. 0.1% is the plan's bar; measured agreement is exact.
HAUL_TOLERANCE = 0.001


def _fixture_array():
    """`(z, transform, cell_w, cell_h)` for the fixture, nodata already NaN."""
    import numpy as np
    import rasterio

    with rasterio.open(FIXTURE_DEM) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
        return z, src.transform, abs(src.transform.a), abs(src.transform.e)


def _write_dem(path, z, transform):
    import numpy as np
    import rasterio

    with rasterio.open(FIXTURE_DEM) as src:
        crs = src.crs
    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32", crs=crs, transform=transform,
        width=z.shape[1], height=z.shape[0], count=1, nodata=-9999.0,
    ) as dst:
        dst.write(np.where(np.isfinite(z), z, -9999.0).astype("float32"), 1)
    return path


def _accumulate(z, next_flat):
    """Cells draining through each cell, by one high-to-low pass over the pointers.

    Exact on an acyclic graph and independent of pysheds, so the arms below measure
    TerrainFlow's own flow graph rather than a library's conditioning.
    """
    import numpy as np

    finite = np.isfinite(z).ravel()
    acc = np.ones(z.size, dtype=np.int64)
    acc[~finite] = 0
    flat_z = z.ravel()
    for i in np.argsort(np.where(finite, -flat_z, np.inf), kind="stable"):
        if not finite[i]:
            continue
        j = int(next_flat[i])
        if j != i:
            acc[j] += acc[i]
    return acc


def _flow_bits(z, cell_w, cell_h):
    """`(next_flat, is_sink, acc, links)` at the production stream threshold."""
    import numpy as np

    from terrainflow_assessment.modules.flow_graph import (
        accumulate,
        d8_from_dem,
        strahler_order,
        stream_links,
    )

    next_flat, is_sink = d8_from_dem(z, cell_w, cell_h)
    acc = _accumulate(z, next_flat)
    # Two routes to one count: the elevation-sorted oracle above against the level-
    # synchronous pass production uses. They must agree cell for cell.
    assert np.array_equal(acc, accumulate(next_flat, np.isfinite(z).ravel())), (
        "flow_graph.accumulate disagrees with the elevation-sorted oracle")
    threshold = max(20, int(round(2_000.0 / (cell_w * cell_h))))
    stream = (acc >= threshold) & np.isfinite(z).ravel()
    order = strahler_order(next_flat, stream)
    links = stream_links(next_flat, stream, order, z.shape[1], max_order=1)
    return next_flat, is_sink, acc, links


def _worst_delta(a, b):
    import numpy as np

    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0


def check_terrain_answers_do_not_depend_on_absolute_elevation(dem_path):
    """Add 600 m to every cell. Nothing but the elevations may move.

    The exact oracle no amount of resampling gives: slope, curvature, TPI, the D8 pointer
    graph, the order-1 link population and the *positions* of every keypoint are all
    properties of shape, and shape does not care how high the hill is.

    This is the only arm in either suite that reaches the float32 regime. At the fixture's
    own 84.78 m float32 spacing is 7.63e-06 m and `resolve_flats`' 1e-05 m inflation step
    survives it; at +600 m the spacing is 6.10e-05 m and it does not, which is exactly the
    failure `flow_analysis.save_result`'s docstring documents and steers the conditioned
    surface to float64 to avoid. Measured 2026-09-11: the answers hold anyway, because
    `safe_flat_epsilon` derives its step from the surface's own elevation instead of
    taking pysheds' fixed default — so this also guards that derivation, which is the
    thing actually doing the work.
    """
    import os as _os
    import tempfile

    from terrainflow_assessment.modules.dem_loader import slope_degrees
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis
    from terrainflow_assessment.modules.terrain_indices import curvature, landform_tpi

    z, transform, cell_w, cell_h = _fixture_array()
    lifted = z + Z_OFFSET_M
    failures = []

    print("\n    --- Z + 600 m invariance ---")
    # float64 throughout: the question is whether the *maths* reads absolute elevation.
    # What the float32 cast costs is a different question, measured by p_invariance.py.
    for name, fn in (
        ("slope", lambda s: slope_degrees(s, cell_w, cell_h)),
        ("plan curvature", lambda s: curvature(s, cell_w, cell_h)[0]),
        ("profile curvature", lambda s: curvature(s, cell_w, cell_h)[1]),
        ("TPI", lambda s: landform_tpi(s, cell_w, cell_h)),
    ):
        delta = _worst_delta(fn(z), fn(lifted))
        ok = delta < 1e-6
        print(f"    {'ok ' if ok else 'MOVED'} {name:20s} max |delta| {delta:.3e}")
        if not ok:
            failures.append(f"{name} moved by {delta:.3e} under a {Z_OFFSET_M:g} m lift")

    base_next, base_sink, _base_acc, base_links = _flow_bits(z, cell_w, cell_h)
    lift_next, lift_sink, _lift_acc, lift_links = _flow_bits(lifted, cell_w, cell_h)
    pointer_moves = int((base_next != lift_next).sum())
    print(f"    {'ok ' if pointer_moves == 0 else 'MOVED'} "
          f"{'D8 pointers':20s} {pointer_moves} disagreements")
    print(f"    sinks {int(base_sink.sum())} vs {int(lift_sink.sum())}; "
          f"order-1 links {len(base_links)} vs {len(lift_links)}")
    if pointer_moves:
        failures.append(
            f"{pointer_moves} D8 pointers moved under a {Z_OFFSET_M:g} m lift")
    if int(base_sink.sum()) != int(lift_sink.sum()):
        failures.append(
            f"sink count moved: {int(base_sink.sum())} -> {int(lift_sink.sum())}")
    if len(base_links) != len(lift_links):
        failures.append(
            f"order-1 link count moved: {len(base_links)} -> {len(lift_links)}")

    with tempfile.TemporaryDirectory(prefix="tfa_zshift_") as tmp:
        def keypoints(path):
            ya = YeomansKeylineAnalysis(path)
            kps, skipped = ya.find_keypoints(max_valleys=8)
            ordered = sorted(kps, key=lambda k: (k["row"], k["col"]))
            return ([(int(k["row"]), int(k["col"])) for k in ordered],
                    [float(k["elevation"]) for k in ordered],
                    len(skipped))

        base_rc, base_z, base_skipped = keypoints(
            _write_dem(_os.path.join(tmp, "base.tif"), z, transform))
        lift_rc, lift_z, lift_skipped = keypoints(
            _write_dem(_os.path.join(tmp, "lifted.tif"), lifted, transform))

    print(f"    {'ok ' if base_rc == lift_rc else 'MOVED'} "
          f"{'keypoint positions':20s} {base_rc} vs {lift_rc}")
    if base_rc != lift_rc:
        failures.append(f"keypoint positions moved: {base_rc} -> {lift_rc}")
    if base_skipped != lift_skipped:
        failures.append(
            f"skipped-valley count moved: {base_skipped} -> {lift_skipped}")
    for before, after in zip(base_z, lift_z):
        if abs((after - before) - Z_OFFSET_M) > 1e-2:
            failures.append(
                f"a keypoint elevation rose by {after - before:.4f} m, not "
                f"{Z_OFFSET_M:g} m")

    assert not failures, (
        "the analysis tier's answers depend on absolute elevation:\n      "
        + "\n      ".join(failures)
        + f"\n    Every quantity above is a property of shape, and a site {Z_OFFSET_M:g} "
          "m higher is the same shape.\n    Suspect float32 somewhere the conditioned "
          "surface passes through — read `flow_analysis.save_result`'s docstring and "
          "`safe_flat_epsilon`, which is what currently makes this pass."
    )


def check_terrain_answers_mirror_under_a_horizontal_flip(dem_path):
    """Flip the DEM left-right. Every answer must flip with it.

    One arm for the whole sign-and-axis class: `KPA-33`'s row-to-y flip, `CTA-08`'s
    rc-to-map asymmetry, `IMP-01`'s diagonal count and `UNI-15`'s rc/xy convention were
    each argued check by check, and a mirror tests all of them at once. `UNI-15` and its
    `CTA-08` component are fixed (`MATHS_AUDIT` §9.2), so this guards shipped fixes.

    Scalars are unchanged, the east-west derivative negates, and aspect — a compass
    bearing — reflects to `(360 - a) mod 360`, with the `-1` flat sentinel carried through
    as itself rather than becoming 361.

    **D8 pointer disagreement is expected and is not a failure by itself.**
    `d8_from_dem` breaks ties by scan position (`flow_graph.py:106-113`) and a flip
    reverses that order, so a cell with two equally steep neighbours may legitimately pick
    the other one. What must be zero is disagreement where the two candidates are *not*
    equally steep. Measured 2026-09-11: 31 disagreements, all 31 exact ties.
    """
    import numpy as np

    from terrainflow_assessment.modules.dem_loader import (
        aspect_degrees,
        horn_gradient,
        slope_degrees,
    )
    from terrainflow_assessment.modules.terrain_indices import curvature, landform_tpi

    z, _transform, cell_w, cell_h = _fixture_array()
    zm = np.fliplr(z).copy()
    cols = z.shape[1]
    failures = []

    print("\n    --- mirror invariance ---")
    dzdx, dzdy, _inv = horn_gradient(z, cell_w, cell_h)
    mdzdx, mdzdy, _minv = horn_gradient(zm, cell_w, cell_h)
    for name, expected, observed in (
        ("slope", np.fliplr(slope_degrees(z, cell_w, cell_h)),
         slope_degrees(zm, cell_w, cell_h)),
        ("plan curvature", np.fliplr(curvature(z, cell_w, cell_h)[0]),
         curvature(zm, cell_w, cell_h)[0]),
        ("TPI", np.fliplr(landform_tpi(z, cell_w, cell_h)),
         landform_tpi(zm, cell_w, cell_h)),
        ("dz_dx (negates)", -np.fliplr(dzdx), mdzdx),
        ("dz_dy (unchanged)", np.fliplr(dzdy), mdzdy),
    ):
        delta = _worst_delta(expected, observed)
        ok = delta < 1e-6
        print(f"    {'ok ' if ok else 'MOVED'} {name:20s} max |delta| {delta:.3e}")
        if not ok:
            failures.append(f"{name} did not mirror: max |delta| {delta:.3e}")

    asp = np.fliplr(aspect_degrees(z, cell_w, cell_h)).astype("float64")
    reflected = np.where(asp < 0, asp, np.mod(360.0 - asp, 360.0))
    observed = aspect_degrees(zm, cell_w, cell_h).astype("float64")
    bearings = (np.isfinite(reflected) & np.isfinite(observed)
                & (reflected >= 0) & (observed >= 0))
    # A compass wraps and subtraction does not: without this, 359.999 and 0.001 read as a
    # 360-degree disagreement between two rasters that agree.
    d = np.abs(reflected[bearings] - observed[bearings])
    d = np.minimum(d, 360.0 - d)
    aspect_delta = float(d.max()) if d.size else 0.0
    flat_flips = int(((reflected < 0) != (observed < 0)).sum())
    print(f"    {'ok ' if aspect_delta < 1e-3 else 'MOVED'} "
          f"{'aspect (reflects)':20s} max |delta| {aspect_delta:.3e} deg, "
          f"{flat_flips} flat-sentinel flips")
    if aspect_delta >= 1e-3:
        failures.append(f"aspect did not reflect: max |delta| {aspect_delta:.3e} deg")
    if flat_flips:
        failures.append(f"{flat_flips} cells changed flat-sentinel state under a flip")

    base_next, base_sink, _base_acc, base_links = _flow_bits(z, cell_w, cell_h)
    mir_next, mir_sink, _mir_acc, mir_links = _flow_bits(zm, cell_w, cell_h)

    idx = np.arange(z.size, dtype=np.int64)

    def mirror_index(flat):
        r, c = divmod(np.asarray(flat, dtype=np.int64), cols)
        return r * cols + (cols - 1 - c)

    expected_next = mirror_index(base_next)[mirror_index(idx)]
    differing = expected_next != mir_next
    flat = zm.ravel()

    def drop(src, dst):
        sr, sc = divmod(src, cols)
        dr, dc = divmod(dst, cols)
        dist = np.hypot((dr - sr) * cell_h, (dc - sc) * cell_w)
        safe = np.where(dist > 0, dist, 1.0)
        with np.errstate(invalid="ignore"):
            return np.where(dist > 0, (flat[src] - flat[dst]) / safe, 0.0)

    src = idx[differing]
    tied = (drop(src, mir_next[differing].astype(np.int64))
            == drop(src, expected_next[differing].astype(np.int64)))
    real_moves = int((~tied).sum())
    print(f"    {'ok ' if real_moves == 0 else 'MOVED'} "
          f"{'D8 pointers':20s} {int(differing.sum())} disagreements, "
          f"{int(tied.sum())} on exact ties, {real_moves} not")
    print(f"    sinks {int(base_sink.sum())} vs {int(mir_sink.sum())}; "
          f"order-1 links {len(base_links)} vs {len(mir_links)}")
    if real_moves:
        failures.append(
            f"{real_moves} D8 pointers disagree under a mirror on cells where the two "
            "candidate neighbours are NOT equally steep")
    if int(base_sink.sum()) != int(mir_sink.sum()):
        failures.append(
            f"sink count moved under a mirror: {int(base_sink.sum())} -> "
            f"{int(mir_sink.sum())}")
    if len(base_links) != len(mir_links):
        failures.append(
            f"order-1 link count moved under a mirror: {len(base_links)} -> "
            f"{len(mir_links)}")

    assert not failures, (
        "the analysis tier does not mirror:\n      " + "\n      ".join(failures)
        + "\n    A horizontal flip is an exact oracle — no resampling, no tolerance to "
          "argue about.\n    Suspect a row/column or an x/y convention used one way in "
          "one place and the other way in another; that is the class this arm exists to "
          "catch."
    )


def check_signed_rasters_point_the_right_way(dem_path):
    """A flipped sign has identical percentiles. This is what catches one.

    Every other statistical guard in this suite — finite fraction, NaN count, quantile
    spread — passes unchanged on a raster whose sign has been inverted, which is the
    single most consequential thing that can go quietly wrong in a terrain index. These
    witnesses take the sign of a signed raster's mean over a reference set identified
    **independently of that raster**.

    The curvature-versus-TPI bar is deliberately not "agreement above N%": plan curvature
    is a 3x3 second derivative and TPI a 15 m neighbourhood mean, so correct code
    disagrees cell by cell about high-frequency detail and measures 73.2% here. The test
    only wrong code fails is the sign of the two means, plus a correlation that is
    positive and not trivially so. A **negative** correlation is the unambiguous failure.
    """
    import numpy as np

    from terrainflow_assessment.modules.dem_loader import (
        aspect_degrees,
        horn_gradient,
        slope_degrees,
    )
    from terrainflow_assessment.modules.terrain_indices import (
        curvature,
        landform_classes,
        landform_tpi,
    )

    z, _transform, cell_w, cell_h = _fixture_array()
    failures = []

    plan, _profile = curvature(z, cell_w, cell_h)
    tpi = landform_tpi(z, cell_w, cell_h)
    next_flat, _sink, acc_flat, _links = _flow_bits(z, cell_w, cell_h)
    acc = acc_flat.reshape(z.shape)

    print("\n    --- signed witnesses ---")
    valid = np.isfinite(plan) & np.isfinite(z) & np.isfinite(tpi)
    network = valid & (acc >= np.quantile(acc[valid], 0.95))
    noses = valid & (tpi >= np.quantile(tpi[valid], 0.95))
    mean_network = float(plan[network].mean())
    mean_noses = float(plan[noses].mean())
    print(f"    plan curvature over the top-5% flow network {mean_network:+.3e} /m "
          "(must be negative — hollows converge)")
    print(f"    plan curvature over the top-5% TPI          {mean_noses:+.3e} /m "
          "(must be positive — noses diverge)")
    if not mean_network < 0:
        failures.append(
            f"plan curvature averages {mean_network:+.3e} over the flow network; a "
            "converging hollow must be negative in plan")
    if not mean_noses > 0:
        failures.append(
            f"plan curvature averages {mean_noses:+.3e} over the highest TPI; a "
            "diverging nose must be positive in plan")

    classes = landform_classes(tpi)
    ridge, valley = classes == 1, classes == -1
    mean_ridge = float(np.nanmean(plan[ridge]))
    mean_valley = float(np.nanmean(plan[valley]))
    classed = ridge | valley
    values = plan[classed].astype("float64")
    groups = ridge[classed].astype("float64")
    r = (float(np.corrcoef(values, groups)[0, 1])
         if values.std() and groups.std() else 0.0)
    print(f"    plan on TPI ridge {mean_ridge:+.3e}, in TPI valley {mean_valley:+.3e}; "
          f"point-biserial {r:+.3f} (bar {MIN_POINT_BISERIAL:+.2f})")
    if not mean_valley < 0 < mean_ridge:
        failures.append(
            f"curvature and TPI disagree about which way is up: plan averages "
            f"{mean_ridge:+.3e} on TPI ridges and {mean_valley:+.3e} in TPI valleys")
    if r < MIN_POINT_BISERIAL:
        failures.append(
            f"plan curvature correlates {r:+.3f} with the TPI ridge class, under the "
            f"{MIN_POINT_BISERIAL:+.2f} bar; a negative value means one of the two has "
            "had its sign inverted")

    _dz_dx, dz_dy, _invalid = horn_gradient(z, cell_w, cell_h)
    aspect = aspect_degrees(z, cell_w, cell_h)
    slope = slope_degrees(z, cell_w, cell_h)
    real = np.isfinite(aspect) & (aspect >= 0) & np.isfinite(slope) & (slope > 2.0)
    southward = (aspect > 90.0) & (aspect < 270.0)
    # `horn_gradient`'s dz_dy rises toward increasing row and row increases southward, so
    # ground falling to the south has dz_dy < 0 — and aspect, which points downslope,
    # lands in (90, 270) exactly there. `aspect = atan2(-dz_dx, dz_dy)` has dz_dy as its
    # northward term, which is the same statement. Stated with the sign the other way
    # round this witness reports 0.01% on correct code, which is how it was first written.
    agreement = float((southward == (dz_dy < 0))[real].sum() / max(int(real.sum()), 1))
    print(f"    aspect in (90,270) coincides with dz_dy < 0 for {agreement:.2%} of "
          f"{int(real.sum()):,} cells over 2 deg (bar {MIN_ASPECT_AGREEMENT:.0%})")
    if agreement < MIN_ASPECT_AGREEMENT:
        failures.append(
            f"aspect and dz_dy agree on only {agreement:.2%} of real slopes. Near 0% "
            "means aspect's north/south sense is inverted relative to the gradient it "
            "is computed from")

    assert not failures, (
        "a signed raster points the wrong way:\n      " + "\n      ".join(failures)
        + "\n    Percentile and finite-fraction checks cannot see this — an inverted "
          "raster has identical magnitudes.\n    Read `terrain_indices.curvature`'s "
          "sign-convention docstring and `dem_loader.aspect_degrees` before changing "
          "anything here."
    )


def check_haul_volumes_agree_with_the_burn(dem_path):
    """`haul_regions` and `burn_quantities` reduce the same surfaces. They must agree.

    One sums the difference between two elevation surfaces flat; the other labels it into
    connected components and sums each. Any gap beyond `min_region_m3` is an accounting
    error in a figure that goes to a contractor — the cheapest high-value cross-check
    available, and neither function had one.

    The burn is synthetic on purpose: the identity is arithmetic over two surfaces and
    does not care how the second arose, whereas a full earthwork run would make a failure
    ambiguous between the burn and the accounting.
    """
    import numpy as np

    from terrainflow_assessment.modules.earthwork_design import burn_quantities
    from terrainflow_assessment.modules.mass_haul import (
        DEFAULT_MIN_REGION_M3,
        haul_regions,
    )

    z, transform, cell_w, cell_h = _fixture_array()
    cell_area = cell_w * cell_h
    rows, cols = z.shape
    rr, cc = np.mgrid[0:rows, 0:cols]

    burned = z.copy()
    burned[(np.abs(rr - rows // 3) <= 2) & (cc > cols // 4)
           & (cc < 3 * cols // 4)] -= 0.5
    burned[(np.abs(rr - 2 * rows // 3) <= 2) & (cc > cols // 4)
           & (cc < 3 * cols // 4)] += 0.4
    for i in range(8):                    # dabs under the 5 m3 floor, deliberately
        burned[10 + i * 7, 10] += 0.1
    burned[~np.isfinite(z)] = np.nan

    quantities = burn_quantities(z, burned, cell_area)
    cuts, fills = haul_regions(z, burned, transform, cell_area, min_region_m3=0.0)
    totals = {
        "cut": float(sum(r["volume_m3"] for r in cuts)),
        "fill": float(sum(r["volume_m3"] for r in fills)),
    }

    print("\n    --- haul volumes against the burn ---")
    failures = []
    for name, regions in (("cut", cuts), ("fill", fills)):
        total = totals[name]
        expected = quantities[f"{name}_m3"]
        error = abs(total - expected) / max(abs(expected), 1e-12)
        ok = error < HAUL_TOLERANCE
        print(f"    {'ok ' if ok else 'MOVED'} {name:5s} {total:12,.2f} m3 over "
              f"{len(regions):3d} regions against burn_quantities' {expected:12,.2f} m3 "
              f"(relative error {error:.2e})")
        if not ok:
            failures.append(
                f"{name}: haul_regions sums {total:,.2f} m3 against burn_quantities' "
                f"{expected:,.2f} m3, a relative error of {error:.2e}")

    _floored_cuts, floored_fills = haul_regions(z, burned, transform, cell_area)
    dropped = totals["fill"] - float(sum(r["volume_m3"] for r in floored_fills))
    print(f"    (at the shipped min_region_m3={DEFAULT_MIN_REGION_M3:g} m3 the haul plan "
          f"omits {dropped:.2f} m3 of fill across {len(fills) - len(floored_fills)} "
          "regions — expected, and still counted by the site total)")

    assert not failures, (
        "the haul plan and the site total disagree about how much earth moves:\n      "
        + "\n      ".join(failures)
        + "\n    With min_region_m3=0 these are two routes to one sum and must agree.\n"
          "    Suspect a nansum, a cell-area factor, or a sign on one side."
    )


def check_d8_routing_runs_and_differs_from_dinf(dem_path):
    """Both routing schemes the panel offers actually run, and they disagree.

    **Inverted from `check_d8_routing_is_still_the_documented_crash`**, which asserted the
    `FLA-26` crash on purpose and carried instructions to become this the day it was fixed.
    `ANALYSIS_DEFECTS.md` §9.1 is that day.

    The defect: pysheds 0.5 calls `np.in1d` at nine sites in `sgrid.py`, NumPy removed the
    name in 2.0, and every one of those sites is on a **D8** code path — so selecting "D8"
    in the Routing combo (`panel.py:773`) took the baseline run down with an unhandled
    `AttributeError`. `modules/pysheds_compat.py` restores the alias and re-exports `Grid`;
    everything that builds a pysheds grid imports it from there.

    Two assertions, because the fix has two ways to be wrong and only one is obvious:

    1. **d8 runs.** The regression guard.
    2. **d8 and dinf disagree.** The guard against a fix that appears to work by quietly
       giving everyone D-infinity. That failure mode is not hypothetical — it is what
       `MATHS_AUDIT`'s `FLA-02` was opened about, and `flow_analysis.py:504-507` still
       contains a routing downgrade of exactly that shape for a different cause. If this
       check only asserted "d8 runs", a silent downgrade would pass it.

    Measured 2026-09-11 on the fixture: accumulation maxima 67,198.9 (dinf) against
    105,167.0 (d8) — D8 concentrates flow into one successor where D-infinity divides it —
    differing on 152,647 of 160,000 cells. The conditioned surfaces are **identical**, and
    must be: they are built before the routing branch (`flow_analysis.py:503`), so a
    non-zero difference there would mean the conditioning had become routing-dependent.

    No number here is recorded as an expected value. The assertions are "runs",
    "differs somewhere" and "conditioning is identical", all of which hold on any terrain.
    """
    import numpy as np

    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

    print("\n    --- routing: dinf against d8 ---")
    results = {}
    for routing in ("dinf", "d8"):
        fa = FlowAnalysis()
        fa.load_dem(FIXTURE_DEM)
        try:
            out = fa.run(routing=routing, crest_split=False)
        except AttributeError as exc:
            raise AssertionError(
                f"routing='{routing}' raises AttributeError: {exc}\n"
                "    If this names np.in1d, the pysheds compatibility shim is not being "
                "applied — check that every `Grid` import in modules/ comes from\n"
                "    terrainflow_assessment.modules.pysheds_compat, not from "
                "pysheds.grid. See ANALYSIS_DEFECTS.md §9.1."
            ) from exc
        results[routing] = {
            "acc": np.asarray(out["flow_accumulation"], dtype="float64"),
            "conditioned": np.asarray(out["conditioned_dem"], dtype="float64"),
            "routing_used": fa.routing,
        }
        print(f"    ok  {routing:4s} ran; accumulation max "
              f"{np.nanmax(results[routing]['acc']):12,.1f}; "
              f"FlowAnalysis.routing reports {fa.routing!r}")

    failures = []
    for routing, rec in results.items():
        if rec["routing_used"] != routing:
            failures.append(
                f"asked for routing={routing!r} and FlowAnalysis reports "
                f"{rec['routing_used']!r} — a silent downgrade, which is the failure "
                "mode MATHS_AUDIT FLA-02 was opened about")

    a, b = results["dinf"]["acc"], results["d8"]["acc"]
    both = np.isfinite(a) & np.isfinite(b)
    differing = int((a[both] != b[both]).sum())
    print(f"    ok  accumulation differs on {differing:,} of {int(both.sum()):,} cells")
    if differing == 0:
        failures.append(
            "dinf and d8 produce identical accumulation on every cell. Either one is "
            "being silently substituted for the other, or the routing argument is no "
            "longer reaching pysheds")

    ca, cb = results["dinf"]["conditioned"], results["d8"]["conditioned"]
    ok = np.isfinite(ca) & np.isfinite(cb)
    conditioned_delta = float(np.abs(ca[ok] - cb[ok]).max()) if ok.any() else 0.0
    print(f"    ok  conditioned surfaces differ by {conditioned_delta:.3e} m "
          f"(must be 0 — conditioning precedes the routing branch)")
    if conditioned_delta != 0.0:
        failures.append(
            f"the conditioned surfaces differ by {conditioned_delta:.3e} m between the "
            "two routing schemes. Pit-filling, depression-filling and flat resolution all "
            "happen before the routing branch, so this must be exactly 0")

    assert not failures, (
        "the two routing schemes are not both working as distinct schemes:\n      "
        + "\n      ".join(failures)
        + "\n    Both are offered in the panel's Routing combo, so both have to mean "
          "something.\n    See ANALYSIS_DEFECTS.md §9.1 and modules/pysheds_compat.py."
    )




# Recorded 2026-09-12, when `_build_design` gained a berm, a diversion and a companion
# berm so that every burn method is exercised.
#
# **Per feature, each burned alone, and the site total beside them.** A single site
# total does not close the coverage gap it was added to close: the two swales and the
# basin already account for 13,869.8 m3 of the 13,914.2 m3 cut, so the diversion's own
# 48.8 m3 is 0.35% of the figure. A change that moved `_burn_diversion` by a third
# would shift the total by 0.1% and pass the 0.5% tolerance without a murmur.
#
# Burning each feature alone is well defined and order-independent — it is the same
# device `_isolated_burn` uses for `feature_storage`, and it sidesteps the attribution
# problem `burn_quantities`' own docstring gives as the reason it reports site totals
# only. Nothing is attributed here; each figure is what one feature does to this
# hillside with nothing else on it.
#
# What each line covers:
#   swale_a      `_burn_swale`, level invert
#   swale_b      the same plus `_add_companion_berm` -> `_key_berm_into_banks` (fill)
#   basin_c      `_burn_basin`, level floor
#   berm_d       `_burn_berm` — the trapezoid section and the centreline seal (M-1, M-2)
#   diversion_e  `_burn_diversion` — graded invert, tapered section (M-1)
EXPECTED_BURN = {
    "swale_a_cut_m3": 6120.7970,
    "swale_b_cut_m3": 5010.0210,
    "swale_b_fill_m3": 3757.5160,
    "basin_c_cut_m3": 2739.0170,
    "berm_d_fill_m3": 562.0000,
    "diversion_e_cut_m3": 48.8260,
    "site_cut_m3": 13914.1630,
    "site_fill_m3": 4315.1177,
    # The keyed companion berm, pinned on the two things a conserved volume cannot
    # hide. `level_crest_from_spoil` spreads a fixed quantity of spoil, so changing
    # `_key_berm_into_banks`' end-cap footprint moves the crest and the cell count
    # in *opposite* directions and leaves `swale_b_fill_m3` untouched — which is
    # exactly what it did when M-1 was reverted, at 0.00%.
    # 6.28 m mean is a real figure and an absurd bank, and it comes from the
    # fixture's pre-existing geometry rather than from anything added here: the
    # two swales run along the fall line (Swale B rises 11.75 m end to end) and
    # are cut to a *level* invert, so the trench yields 3,757 m3 of spoil and
    # `level_crest_from_spoil` has to put it somewhere. The cut was 5,010 m3
    # before a companion berm was ever added. Recorded as a regression pin, not
    # endorsed as a design — see the note on `_build_design`.
    "swale_b_berm_height_mean_m": 6.2835,
    "swale_b_berm_cells": 598.0000,
}


def check_the_burn_quantities_have_not_moved(dem_path):
    """Every burn method, pinned on what it does to this hillside alone.

    The gap this closes, stated plainly: `_build_design` was two swales and a basin,
    which are the burns that were *already* rasterising on cell centres. So when
    M-1 fixed the other three and M-2 changed the berm's section from a 1:1 triangle
    to a trapezoid, every pin in this module held — not because the burns were
    unchanged, but because nothing here reached them. The only other burn figure in
    the module, `check_haul_volumes_agree_with_the_burn`, reduces a deliberately
    *synthetic* surface and never touched a real burn either.
    """
    from terrainflow_assessment.modules.earthwork_design import (
        DEMBurner,
        burn_quantities,
    )

    with PluginHarness(FIXTURE_DEM, load_boundary=False, load_dem=False) as h:
        design = _build_design(h)
        by_name = {ew.name: ew for ew in design}

        present = sorted({ew.type for ew in design})
        assert present == ["basin", "berm", "diversion", "swale"], (
            f"the design no longer covers every burn method: {present}")
        bermed = [ew for ew in design if getattr(ew, "companion_berm", False)]
        assert bermed and all(getattr(ew, "key_into_banks", False) for ew in bermed), (
            "no keyed companion berm in the design, so `_key_berm_into_banks` is "
            "unpinned again")

        def alone(name):
            """Cut and fill for one feature burned into fresh ground."""
            burner = DEMBurner(FIXTURE_DEM)
            burned = burner.burn_earthworks([by_name[name]])
            return burn_quantities(burner.original, burned, burner.cell_area)

        def keyed_berm(name):
            """`(crest_elevation, footprint_cells)` for a keyed companion berm.

            **Volume cannot pin this path, and that is a property of the design
            rather than an oversight.** `level_crest_from_spoil` conserves the
            spoil the trench yields, so when `_key_berm_into_banks` changes the
            end-cap footprint the crest simply moves to absorb it: narrow the cap
            and the same earth stands higher. `swale_b_fill_m3` therefore held at
            0.00% when M-1 was reverted, even though that burn had changed.

            The two quantities that *do* move are how tall the spoil stands and how
            many cells it is spread over — and they move in **opposite** directions,
            which is what makes the pair a constraint rather than two views of one
            number.

            Height above ground, not the crest elevation. Both move, but the crest
            is an absolute datum near 69 m here, so the 0.11 m the M-1 revert moves
            it is 0.16% — inside this module's 0.5% tolerance, and it read "ok"
            while the footprint underneath it had changed. The same 0.11 m on a
            bank about a third of a metre tall is a third of the quantity. Pin the
            small number; a relative tolerance on a large datum asserts almost
            nothing.
            """
            ew = by_name[name]
            burner = DEMBurner(FIXTURE_DEM)
            burner.burn_earthworks([ew])
            raised = (burner.burned_raised or {}).get(getattr(ew, "id", None))
            assert raised is not None and raised.any(), (
                f"{name} recorded no raised ground — its companion berm did not "
                f"burn, so the keyed path is unpinned")
            height = getattr(ew, "berm_height_m", None)
            assert height and len(height) == 3, (
                f"{name} has no (min, mean, max) berm height after its burn: "
                f"{height!r}")
            return float(height[1]), float(raised.sum())

        site_burner = DEMBurner(FIXTURE_DEM)
        site = burn_quantities(
            site_burner.original,
            site_burner.burn_earthworks(design),
            site_burner.cell_area,
        )

        observed = {
            "swale_a_cut_m3": float(alone("Swale A")["cut_m3"]),
            "swale_b_cut_m3": float(alone("Swale B")["cut_m3"]),
            "swale_b_fill_m3": float(alone("Swale B")["fill_m3"]),
            "basin_c_cut_m3": float(alone("Basin C")["cut_m3"]),
            "berm_d_fill_m3": float(alone("Berm D")["fill_m3"]),
            "diversion_e_cut_m3": float(alone("Diversion E")["cut_m3"]),
            "site_cut_m3": float(site["cut_m3"]),
            "site_fill_m3": float(site["fill_m3"]),
        }
        berm_height_m, berm_cells = keyed_berm("Swale B")
        observed["swale_b_berm_height_mean_m"] = berm_height_m
        observed["swale_b_berm_cells"] = berm_cells

        # Each per-method figure has to be a real quantity, or its pin is a pin on
        # zero and the method could stop burning entirely without moving it.
        for key, value in observed.items():
            assert value > 0.0, f"{key} is {value} — that burn produced nothing"

        print("\n    --- burned cut and fill, per feature and site ---")
        failures = []
        if not EXPECTED_BURN:
            print("    NO RECORDED VALUES — copy the block below into EXPECTED_BURN:")
            print("    EXPECTED_BURN = {")
            for key, value in observed.items():
                print(f'        "{key}": {value:.4f},')
            print("    }")
            failures.append("EXPECTED_BURN is empty, so nothing was asserted.")
        else:
            # Driven from *observed*, not from EXPECTED_BURN. Walking the recorded
            # dict means a newly observed quantity is computed and then silently
            # never compared — which is how a pin gets added that asserts nothing.
            for key, actual in observed.items():
                expected = EXPECTED_BURN.get(key)
                if expected is None:
                    print(f"    NEW {key:22} {actual:13.4f}  <- record this")
                    failures.append(
                        f"{key}: observed but not in EXPECTED_BURN, so nothing "
                        f"asserted it. Record {actual:.4f}.")
                    continue
                gap = _relative_gap(actual, expected)
                marker = "ok " if gap <= TOLERANCE else "MOVED"
                print(f"    {marker} {key:22} {actual:13.3f}  "
                      f"(recorded {expected:.3f}, {gap * 100:.2f}%)")
                if gap > TOLERANCE:
                    failures.append(
                        f"{key}: {actual:.3f} vs recorded {expected:.3f} "
                        f"({gap * 100:.2f}% > {TOLERANCE * 100:.1f}%)")
            for key in EXPECTED_BURN:
                if key not in observed:
                    failures.append(f"{key}: recorded but not produced by this run")

        assert not failures, (
            "the burn moved against the recorded fixture:\n      "
            + "\n      ".join(failures)
            + "\n    If the change was intended, re-record EXPECTED_BURN in this file."
        )
