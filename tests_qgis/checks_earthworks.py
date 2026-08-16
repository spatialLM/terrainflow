"""Earthwork design: burn → re-analyse → compare, plus persistence."""

import os

from _harness import PluginHarness, line_across_valley


def check_earthworks_requires_a_feature(dem_path):
    """Re-analysing with nothing drawn warns instead of burning an empty design."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks with no features")
        assert h.bar.warnings, "expected a warning when no earthworks are enabled"


def check_swale_burns_and_reanalyses(dem_path):
    """The core before/after loop: burn a swale into the DEM and re-run analysis."""
    with PluginHarness(dem_path) as h:
        baseline = h.run_baseline()
        h.assert_no_errors("baseline run")

        h.add_earthwork("swale")
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")

        assert h.state.modified_dem_path, "no modified DEM path recorded"
        assert os.path.exists(h.state.modified_dem_path), (
            f"burned DEM not written: {h.state.modified_dem_path}"
        )
        assert h.state.earthworks_result is not None, "earthworks analysis produced no result"
        assert h.state.earthworks_layer_ids, "no earthworks result layers registered"

        # Burning a swale must actually change the terrain, or the whole
        # before/after comparison is comparing a DEM against itself.
        import rasterio

        with rasterio.open(dem_path) as src:
            original = src.read(1)
        with rasterio.open(h.state.modified_dem_path) as src:
            burned = src.read(1)
        assert (original != burned).any(), "burning the swale left the DEM unchanged"

        assert baseline is not None


def check_event_pond_nests_inside_the_capacity_pond(dem_path):
    """The two pond layers, the line over them, and the invariant between them.

    "Pond Capacity (full)" is a depression-fill — every hollow to its spill point,
    drawn identically at 5% full and at 100%. "Pond Capacity (event)" is what the storm
    actually delivers, so it has to lie *inside* it: never a wet cell the full pond does
    not have, never deeper in any cell, never more volume overall. That containment is
    the entire claim the pair makes to a reader comparing them, and solving a level per
    pool is exactly the kind of thing that can quietly break it.

    A basin, not a swale: an excavated hole ponds on any terrain, so the check rests on
    the relationship between the layers rather than on the fixture happening to trap
    water behind a bank.
    """
    import numpy as np
    import rasterio
    from _harness import CELL_M, ORIGIN_Y, centreline_x
    from qgis.core import QgsGeometry, QgsProject, QgsRectangle

    from terrainflow_assessment.qgis.controllers import _groups as G

    def _band(layer):
        with rasterio.open(layer.source()) as src:
            arr = src.read(1).astype("float64")
            if src.nodata is not None:
                arr[arr == src.nodata] = 0.0
        return np.clip(arr, 0.0, None)

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        cx, y = centreline_x(), ORIGIN_Y - 60 * CELL_M
        geom = QgsGeometry.fromRect(
            QgsRectangle(cx - 25.0, y - 25.0, cx + 25.0, y + 25.0))
        h.add_earthwork("basin", geometry=geom)
        # ``add_earthwork`` bypasses the dialog, and the dialog is what sizes a feature:
        # left alone the basin carries capacity 0, which drops it out of the
        # verification pass and so out of the event pond too. Settling the geometry is
        # the controller's own "exact tier" and fills both capacities the way an edit
        # on the canvas does.
        h.plugin._earthworks._on_vertex_edit_finished(0, geom)
        ew = h.state.earthwork_manager.get(0)
        assert ew.capacity_m3 > 0, "the basin was not sized — the check needs a pond"

        # Populates state.balance. The event pond is filled from its per-feature stored
        # volumes, so without a live recompute first there is nothing to fill it with.
        h.panel.analysis_inputs_changed.emit()
        assert h.state.balance is not None, "no live assessment to fill the pond from"
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")
        assert h.state.pond_context is not None, (
            "the verification pass did not run, so it handed nothing forward")

        layers = list(QgsProject.instance().mapLayers().values())
        named = {}
        for fragment in ("Pond Capacity (full)", "Pond Capacity (event)",
                         "Event Water Line"):
            hit = [lyr for lyr in layers
                   if fragment in lyr.name() and lyr.name().startswith("Earthworks")]
            assert hit, (f"no {fragment!r} layer after re-analysis: "
                         f"{sorted(lyr.name() for lyr in layers)}")
            named[fragment] = hit[0]

        assert named["Event Water Line"].featureCount() > 0, (
            "the water line layer was added with no rings in it")

        full = _band(named["Pond Capacity (full)"])
        event = _band(named["Pond Capacity (event)"])
        assert event.shape == full.shape, "the two pond rasters are on different grids"
        assert event.any(), "the event pond is empty — nothing was actually compared"
        deeper = event > full + 1e-9
        assert not deeper.any(), (
            f"the event pond stands deeper than the pool can hold in "
            f"{int(deeper.sum())} cell(s) — max excess "
            f"{float((event - full).max()):.3f} m")
        assert event.sum() <= full.sum() + 1e-6, (
            f"the event pond holds more than the capacity pond: "
            f"{float(event.sum()):.1f} vs {float(full.sum()):.1f} (depth-sum)")

        # The line has to sit above the fill, and the fill above the capacity, or the
        # comparison the layers exist for is hidden underneath itself.
        group = G.group(h.plugin._project, G.RERUN,
                        site_name=h.panel.site_name, tag=h.state.run_tag)
        order = [n.layer().name() for n in group.findLayers() if n.layer() is not None]

        def rank(fragment):
            return next(i for i, name in enumerate(order) if fragment in name)

        assert rank("Event Water Line") < rank("Pond Capacity (event)") \
            < rank("Pond Capacity (full)"), f"pond layers stacked wrong: {order}"


def check_a_dam_that_pours_over_its_own_crest_is_flagged(dem_path):
    """A dam across the valley with no spillway must warn, and draw the run of crest.

    The band is the point: D8 sends the whole overflow through one cell, so the stream
    layer draws a single thread over the wall. A level crest spills along all of itself
    at once, and the layer has to say so — the length is what discharge per metre, and
    therefore whether the face erodes, is figured against.
    """
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        geom = line_across_valley(row=60)
        ew = h.add_earthwork("dam", geometry=geom)
        # A crest above the natural channel, keyed, and deliberately no spillway.
        ew.crest_elevation = float(h.plugin._earthworks._feature_elevation(geom) or 0) + 2.0
        ew.key_into_banks = True
        h.plugin._earthworks._on_vertex_edit_finished(0, geom)
        h.panel.analysis_inputs_changed.emit()
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")

        warnings = " ".join(str(w) for w in h.bar.warnings)
        assert "leaves over its own crest" in warnings, (
            f"no overtopping advisory was raised: {h.bar.warnings}")

        all_layers = list(QgsProject.instance().mapLayers().values())
        layers = [lyr for lyr in all_layers if "Overtopping" in lyr.name()]
        assert layers, ("no overtopping layer: "
                        f"{sorted(lyr.name() for lyr in all_layers)}")
        layer = layers[0]
        assert layer.featureCount() >= 1, "the overtopping layer carries no band"

        feat = next(layer.getFeatures())
        length = feat["length_m"]
        assert length > 0, f"a spill with no length: {length}"
        # It spills along the crest, not at one cell — and never along more wall
        # than there is.
        cell = h.state.dem_info.cell_size_m
        assert length > cell, (
            f"the band is a single cell ({length} m) — that is the D8 artefact this "
            f"layer exists to contradict")
        assert length <= geom.length() + 1e-6, (
            f"spill length {length} m exceeds the {geom.length():.1f} m wall")
        assert feat["spillway"] == "none", feat["spillway"]


def check_multiple_earthwork_types_burn(dem_path):
    """Every registry type must survive being burned and re-analysed."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        for row, ew_type in ((40, "swale"), (60, "berm"), (80, "diversion")):
            h.add_earthwork(ew_type, geometry=line_across_valley(row=row))

        assert len(h.state.earthwork_manager) == 3, "not all earthworks were added"

        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("multi-type earthworks re-analysis")
        assert h.state.earthworks_result is not None, "multi-type analysis produced no result"


def check_catchment_and_assessment_recompute(dem_path):
    """Flow-graph build, catchment labelling, and the live assessment recompute."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale")

        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("live assessment recompute")
        assert h.state.catchment_labels is not None, "no cells were labelled"

        h.panel.toggle_catchment_layer_requested.emit(True)
        h.assert_no_errors("catchment layer on")
        # Without this the check passes when the toggle quietly makes no layer —
        # "nothing errored" and "nothing happened" look identical otherwise.
        assert h.state.catchment_labels_layer_id, "toggling on created no layer"

        h.panel.toggle_catchment_layer_requested.emit(False)
        h.assert_no_errors("catchment layer off")
        assert not h.state.catchment_labels_layer_id, "toggling off left the layer behind"


def check_properties_dialog_path(dem_path):
    """Exercise _on_geometry_drawn, including the properties dialog it opens.

    The dialog is modal, so exec() is stubbed to "accepted" — everything else
    (provisional catchment, spillway datums, dialog construction against real
    terrain, layer refresh) runs for real. This is the path a user takes when
    they finish drawing a swale on the canvas.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    original_exec = EarthworkPropertiesDialog.exec
    EarthworkPropertiesDialog.exec = lambda self: 1
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.assert_no_errors("baseline run")

            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("geometry drawn → properties dialog")

            assert len(h.state.earthwork_manager) == 1, (
                "accepting the properties dialog did not add the earthwork"
            )
    finally:
        EarthworkPropertiesDialog.exec = original_exec


def check_project_persistence_roundtrip(dem_path):
    """Earthworks written into the project must come back identically."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", geometry=line_across_valley(row=50))
        h.add_earthwork("dam", geometry=line_across_valley(row=70))
        before = [(ew.type, ew.name) for ew in h.state.earthwork_manager.get_all()]

        h.plugin._earthworks.save_to_project()
        h.assert_no_errors("save to project")

        h.state.earthwork_manager.clear()
        assert len(h.state.earthwork_manager) == 0

        h.plugin._earthworks.load_from_project()
        h.assert_no_errors("load from project")

        after = [(ew.type, ew.name) for ew in h.state.earthwork_manager.get_all()]
        assert after == before, f"persistence roundtrip changed the design: {before} → {after}"


def check_spillway_review_populates(dem_path):
    """Every water-holding feature gets a review row, designed or not.

    Asserts the rows are non-empty rather than merely that nothing errored: the live
    assessment swallows its exceptions to a console print, so "no criticals" and "the
    review never ran" look identical from the message bar.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", geometry=line_across_valley(row=40))
        h.add_earthwork("berm", geometry=line_across_valley(row=70))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("spillway review")

        rows = h.panel._spillway_table._rows
        assert rows, "no spillway review rows were built"
        names = {r["name"] for r in rows}
        assert any(n.startswith("Swale") for n in names), names
        # A berm holds nothing, so it has nothing to spill and must not appear.
        assert not any(n.startswith("Berm") for n in names), (
            f"a berm should not get a spillway row: {names}")

        swale = next(r for r in rows if r["name"].startswith("Swale"))
        assert swale["required_width_m"], (
            "an undesigned feature must still be sized — that is what makes the "
            "auto-design visible before the user commits to it")
        assert swale["designed"] is False


def check_a_feature_upslope_re_splits_the_downstream_flow(dem_path):
    """Interposing a swale moves flow from own to upstream — and conserves the total.

    Worth pinning because it is the opposite of the intuition. ``cascade_peak_flows``
    routes on the assumption that a full feature passes its whole peak on, so a swale
    placed directly above another does not reduce the *rate* arriving below it: the
    cells it intercepts leave the lower feature's own catchment and come straight back
    as its upstream contribution. The sill below therefore does not shrink, and a review
    that showed it shrinking would be wrong.

    What genuinely re-sizes a downstream spillway is flow being redirected *away* —
    routed to a different feature or off the site — not merely intercepted en route.
    """
    from terrainflow_assessment.earthwork_properties_dialog import (
        EarthworkPropertiesDialog,
    )

    # Drawn through the real add path, not the harness shortcut: only that path
    # re-labels the catchments, and without a re-label the second swale never gets one
    # — so the cascade would have nothing to carry and this check would pass vacuously
    # in the one direction that matters.
    original_exec = EarthworkPropertiesDialog.exec
    EarthworkPropertiesDialog.exec = lambda self: 1
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn(
                "swale", line_across_valley(row=90))
            h.assert_no_errors("first swale drawn")
            lower = h.state.earthwork_manager.get(0)

            rows = {r["id"]: r for r in h.panel._spillway_table._rows}
            before_total = rows[lower.id]["peak_flow_m3s"]
            before_upstream = rows[lower.id]["upstream_m3s"]
            assert before_total, "the lone swale was never given a design flow"
            assert before_upstream == 0, "nothing is upslope of it yet"

            # Row 40 is uphill of row 90: the synthetic DEM drains south, elevation
            # falling as the row index rises.
            h.plugin._earthworks._on_geometry_drawn(
                "swale", line_across_valley(row=40))
            h.assert_no_errors("second swale drawn")

            rows = {r["id"]: r for r in h.panel._spillway_table._rows}
            after_total = rows[lower.id]["peak_flow_m3s"]
            after_upstream = rows[lower.id]["upstream_m3s"]

            assert after_upstream > 0, (
                f"{lower.name} shows no upstream contribution after a swale was drawn "
                f"above it — the peak-flow cascade is not reaching the review"
            )
            drift = abs(after_total - before_total) / before_total
            assert drift <= 0.02, (
                f"total design flow moved from {before_total} to {after_total} "
                f"({drift:.1%}); a feature interposed on the same flow path passes its "
                f"whole peak on, so the rate below it should be unchanged"
            )
    finally:
        EarthworkPropertiesDialog.exec = original_exec


def check_auto_width_survives_leaving_the_dialog(dem_path):
    """The defect this pass exists to fix.

    ``width_auto`` promised the built width tracked the requirement, but the only code
    that recomputed it lived inside the modal properties dialog. The stored figure — the
    one on the map label and in any review — went stale the moment the dialog closed,
    while the undersized-spillway check skipped auto widths on the grounds that they
    tracked by definition.
    """
    from terrainflow_assessment.modules.earthwork_design import Spillway

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=90))
        ew.spillway = Spillway(crest_elevation=None, width_m=0.0, width_auto=True)

        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("auto width refresh")

        assert ew.spillway.width_m > 0, (
            "an auto width never left its 0.0 default — nothing outside the dialog "
            "wrote it, which is exactly the defect"
        )
        first = ew.spillway.width_m

        # Doubling the design intensity doubles the peak flow, and the weir width is
        # linear in flow, so the stored width must follow without reopening anything.
        h.panel.set_peak_intensity(h.panel.peak_intensity_mm_hr * 2)
        h.assert_no_errors("auto width after intensity change")
        assert ew.spillway.width_m > first * 1.5, (
            f"width stayed at {ew.spillway.width_m} m after the intensity doubled "
            f"(was {first} m)"
        )


def check_auto_width_is_not_zeroed_without_an_intensity(dem_path):
    """``calculate_spillway_width`` returns 0.0 for invalid input, meaning "these inputs
    say nothing" — never a spillway zero metres wide. Writing it through would persist
    that fiction on every project without a design intensity."""
    from terrainflow_assessment.modules.earthwork_design import Spillway

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=90))
        ew.spillway = Spillway(width_m=2.5, width_auto=True)

        h.panel.set_peak_intensity(h.panel._peak_intensity_spin.minimum())
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("auto width with a negligible intensity")

        assert ew.spillway.width_m > 0, (
            f"stored width collapsed to {ew.spillway.width_m} — a 0.0 from the weir "
            f"formula means invalid input, not a designed width"
        )


def check_placing_from_the_review_targets_the_right_feature(dem_path):
    """The Spillways list names the feature by row, so placement must not go through
    whichever feature the flow network happened to be showing."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        first = h.add_earthwork("swale", geometry=line_across_valley(row=40))
        second = h.add_earthwork("swale", geometry=line_across_valley(row=90))
        h.panel.analysis_inputs_changed.emit()

        rows = h.panel._spillway_table._rows
        target = next(r for r in rows if r["id"] == second.id)
        h.panel.place_spillway_for_requested.emit(target["index"], "outflow")
        h.assert_no_errors("place spillway from the review")

        tool = h.canvas.mapTool()
        assert tool is not None, "no map tool was armed for placement"
        # The selection has to have moved with it, or the action bar and the map tool
        # would be pointed at different features.
        assert h.panel.get_selected_earthwork_index() in (None, target["index"]), (
            "placement armed against a different feature than the row it came from"
        )
        assert first.spillway is None, "the wrong feature was touched"


def check_spillway_survives_a_project_roundtrip(dem_path):
    """Both spillway kinds, and a freeboard override, through save and reload.

    The field-test log lists save/reopen for spillways as never run, and this pass makes
    that state considerably more load-bearing.
    """
    from terrainflow_assessment.modules.earthwork_design import Spillway

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=60))
        ew.spillway = Spillway(crest_elevation=41.5, head_m=0.15, width_m=3.0,
                               width_auto=False, point_wkt="POINT (100 100)",
                               freeboard_m=0.05)
        ew.inflow_spillway = Spillway(point_wkt="POINT (120 100)")

        h.plugin._earthworks.save_to_project()
        h.state.earthwork_manager.clear()
        h.plugin._earthworks.load_from_project()
        h.assert_no_errors("spillway project roundtrip")

        restored = h.state.earthwork_manager.get(0)
        assert restored.spillway is not None, "the outflow did not survive"
        assert restored.spillway.point_wkt == "POINT (100 100)"
        assert restored.spillway.freeboard_m == 0.05, (
            "a deliberate freeboard override must not be lost — it is the only record "
            "that the reduction was a decision")
        assert restored.spillway.width_auto is False
        assert restored.spillway.width_m == 3.0, (
            "a committed width was rewritten; only auto widths may be re-derived")
        assert restored.inflow_spillway is not None, "the inlet did not survive"


def check_simulation_runs(dem_path):
    """Fill simulation over the burned DEM, then frame stepping."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("basin", geometry=line_across_valley(row=60, half_width_m=15.0))
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")

        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        if h.state.sim_result is not None:
            h.panel.sim_frame_changed.emit(0)
            h.assert_no_errors("simulation frame step")


def check_two_features_sharing_a_name_are_verified_apart(dem_path):
    """Verification is keyed by id, so a duplicate name cannot merge two features.

    The default name is f"{type} {len(manager)+1}", counted over every earthwork, so
    deleting one and drawing another reproduces a name already in use — the same
    collision `_build_network_nodes` documents and was fixed for. Under name keying
    the first of the pair disappeared from `analytic_by_name`, `min_dims`,
    `breakdowns` and the footprint list, so its water read as zero and the surviving
    row scored one feature's pond against the other's capacity.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()

        a = h.add_earthwork("swale", line_across_valley(row=60), name="Swale 1")
        b = h.add_earthwork("swale", line_across_valley(row=120), name="Swale 1")
        assert a.name == b.name, "the fixture must actually collide"
        assert a.id != b.id, "ids must still be distinct"
        # Verification only considers features with a design capacity; the harness
        # builds bare earthworks, so give them distinguishable ones. Different
        # values on purpose — if the two rows were ever to merge, the survivor
        # would carry one of these and the loss would be visible.
        a.capacity_m3 = 40.0
        b.capacity_m3 = 90.0

        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("re-analysis with a duplicated name")

        v = h.state.verification
        assert v is not None, "no verification was produced"

        rows = [r for r in v.per_feature if r["name"] == "Swale 1"]
        assert len(rows) == 2, (
            f"expected both features to be verified, got {len(rows)} row(s) — "
            f"the collision collapsed them")

        # Each was measured against its own capacity, not the other's.
        analytic = sorted(r["analytic_m3"] for r in rows)
        assert analytic == [40.0, 90.0], (
            f"each row must carry its own feature's capacity, got {analytic}")


def check_reanalysis_exit_markers_rescale_with_the_baseline(dem_path):
    """The re-analysis renders through the plugin's own BaselineController.

    It used to build a throwaway one per run. That copy connects `scaleChanged` in
    its constructor and is then dropped, so every re-analysis left another connection
    to a dead controller — and the exit-marker layer ids it collected went with it,
    which is why earthworks exit markers never resized on zoom while the baseline's
    did.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        baseline_ctl = h.plugin._baseline
        before = set(baseline_ctl._exit_layer_ids)
        assert before, "the baseline produced no exit-marker layer to track"

        ew = h.add_earthwork("swale", line_across_valley())
        ew.capacity_m3 = 50.0
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("re-analysis")

        after = set(baseline_ctl._exit_layer_ids)
        assert after > before, (
            "the re-analysis's exit markers were not registered for rescaling — "
            f"{len(before)} before, {len(after)} after")

        # And the rescale itself still runs over all of them.
        h.canvas.setExtent(h.dem_layer.extent())
        baseline_ctl.on_map_scale_changed()
        h.assert_no_errors("rescale after a re-analysis")
        assert set(baseline_ctl._exit_layer_ids) == after, (
            "a rescale dropped layers that are still on the map")


# ---------------------------------------------------------------- edge cases
# Item 39. Three paths the controllers take when the design is incomplete. Each
# has a `return None` or an `or {}` in it, which is how a missing input is meant
# to be handled — and also how a *broken* one looks, right up until something
# downstream unpacks the None.

def check_verification_is_none_with_nothing_enabled(dem_path):
    """No enabled storage feature, no verification — and no exception either.

    An earthwork with zero capacity is skipped by the loop that builds the
    footprints, so a design of nothing but disabled features reaches the maths
    with empty inputs. Returning None is right; raising, or returning a result
    whose totals are all zero, are both wrong — a zero total reads on the panel
    as "measured, and it holds nothing".
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", line_across_valley())
        # Two things the control below insisted on, in turn. Verification is
        # measured off the burned DEM's ponding raster, so it needs a full
        # re-analysis rather than a design-tier refresh; and it only considers
        # features that carry a design capacity, which a harness-built earthwork
        # does not. Without either, this check passed while asserting nothing.
        ew.capacity_m3 = 40.0
        h.panel.run_earthworks_requested.emit()

        assert h.plugin._earthworks._compute_verification() is not None, (
            "the fixture cannot produce a verification even with the feature "
            "enabled, so the disabled case below proves nothing")

        ew.enabled = False
        result = h.plugin._earthworks._compute_verification()
        assert result is None, (
            f"expected no verification with nothing enabled, got {result!r}")
        h.assert_no_errors("verification with nothing enabled")


def check_the_design_tier_survives_having_no_balance(dem_path):
    """Everything that reads `state.balance` must cope with it being None.

    It is None before the first design edit and after a baseline re-run clears
    it, and the report, the scorecard and the network view all read it. This
    drops it deliberately and drives the readouts that consume it.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()
        assert h.state.balance is not None, "fixture never produced a balance"

        h.state.balance = None
        h.state.balance_stores = None
        # The two readouts that consume it, driven directly.
        h.plugin._earthworks._recompute_live_assessment()
        h.plugin._reporting._collect(h.plugin._reporting._site_name())
        h.assert_no_errors("readouts with no balance")


def check_spillway_context_without_an_idf_table(dem_path):
    """`has_idf` is False rather than absent when no HIRDS table is loaded.

    The report's spillway footer reads this key to decide whether to say the
    intensity was looked up at Tc or assumed. A missing key and a False one read
    the same way in Python and differently to anyone maintaining it, and the
    default-intensity flag beside it has to stay independent of the table.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.state.idf_table = None

        context = h.plugin._earthworks._spillway_context()
        assert context["has_idf"] is False, (
            f"has_idf should be False with no table, got {context['has_idf']!r}")
        assert "intensity_mm_hr" in context, "the footer still needs an intensity"
        assert isinstance(context["intensity_is_default"], bool)
        h.assert_no_errors("spillway context with no IDF table")
