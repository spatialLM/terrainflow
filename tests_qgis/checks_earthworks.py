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
