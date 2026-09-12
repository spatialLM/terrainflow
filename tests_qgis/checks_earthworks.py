"""Earthwork design: burn → re-analyse → compare, plus persistence."""

import math
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


def check_matching_before_and_after_layers_share_one_ramp(dem_path):
    """A Baseline layer and its Earthworks counterpart are drawn on one scale.

    The only reason to draw both is to read one against the other, and that works
    only while a colour means one quantity on both. Each layer used to stretch its
    stops over its own band maximum, so the same depth was mid-blue before the
    design and navy after it because the deepest pond on the site had moved — a
    difference that belonged to the ramp and not to the earthworks.

    Checked on the renderers rather than on the state dict: the shared top is only
    worth anything if it reached the pixels.
    """
    from qgis.core import QgsProject

    def ramp_items(layer):
        shader = layer.renderer().shader().rasterShaderFunction()
        return [(item.value, item.label) for item in shader.colorRampItemList()]

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")
        h.add_earthwork("swale")
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")

        layers = list(QgsProject.instance().mapLayers().values())

        def named(prefix, fragment):
            hit = [lyr for lyr in layers
                   if lyr.name().startswith(prefix) and fragment in lyr.name()]
            assert hit, (f"no {prefix} {fragment!r} layer: "
                         f"{sorted(lyr.name() for lyr in layers)}")
            return hit[0]

        for fragment in ("Pond Capacity (full)", "Surface Runoff", "Streams"):
            before = ramp_items(named("Baseline", fragment))
            after = ramp_items(named("Earthworks", fragment))
            assert before == after, (
                f"{fragment!r} is drawn on two different ramps — before {before}, "
                f"after {after}. A colour has to mean one quantity on both or the "
                f"pair cannot be compared.")

        # The event pond rides the same scale, which is the case the argument was
        # first made for: a part-full pond scaled to itself would say "deepest" in
        # the same navy as a brim-full one.
        event = [lyr for lyr in layers if "Pond Capacity (event)" in lyr.name()]
        if event:
            assert ramp_items(event[0]) == ramp_items(
                named("Baseline", "Pond Capacity (full)")), (
                "the event pond is on its own scale again")


def check_surface_runoff_fades_in_over_the_first_cubic_metres(dem_path):
    """Nothing at 0 m³, half at 1 m³, solid from 2 m³ up — and solid above that.

    Every cell on the site has runoff, so drawing them all solid painted the map
    with "it rained here"; that wash is what two rounds of transparency were
    trying to fix. A hard floor answers it but draws an edge, and an edge on this
    layer reads as water *stopping* there. The renderer is checked rather than the
    palette because the interpolated shader is what actually produces the half.
    """
    from qgis.core import QgsProject

    from terrainflow_assessment.core.registry.map_palette import (
        SURFACE_RUNOFF_FADE_TOP_M3,
    )

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                  if lyr.name().startswith("Baseline") and "Surface Runoff" in lyr.name()]
        assert layers, "no Baseline Surface Runoff layer"
        layer = layers[0]

        assert layer.opacity() == 1.0, (
            f"the runoff layer is washed out at {layer.opacity()} — the fade at the "
            f"bottom is what thins this layer now, not whole-layer opacity")

        shader = layer.renderer().shader().rasterShaderFunction()
        items = shader.colorRampItemList()
        assert items[0].value == 0.0, (
            f"the fade does not start at nothing: {items[0].value}")
        assert items[0].color.alpha() == 0, "0 m³ is drawn"
        assert items[1].value == SURFACE_RUNOFF_FADE_TOP_M3, (
            f"the colour ramp starts at {items[1].value}, not at "
            f"{SURFACE_RUNOFF_FADE_TOP_M3} m³")
        assert items[0].color.rgb() == items[1].color.rgb(), (
            "the fade changes hue on the way up, so it is encoding magnitude in "
            "alpha rather than fading one colour in")
        assert all(item.color.alpha() == 255 for item in items[1:]), (
            "a stop above the fade top is translucent again")

        # The half at the midpoint is the shader's, not the palette's — this is the
        # line that would fail if the ramp type stopped being interpolated.
        ok, r, g, b, a = shader.shade(SURFACE_RUNOFF_FADE_TOP_M3 / 2.0)
        assert ok, "the shader refused the midpoint of its own fade"
        assert abs(a - 128) <= 2, f"1 m³ renders at alpha {a}, not half"
        diffuse = items[1].color
        assert (r, g, b) == (diffuse.red(), diffuse.green(), diffuse.blue()), (
            f"the fade renders {(r, g, b)} where the ramp opens on "
            f"{(diffuse.red(), diffuse.green(), diffuse.blue())}")


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
        # The capacity answer, always drawn where a barrier pours over itself. The
        # "(event)" layer beside it is the subset this storm reaches, and is absent
        # when that subset is empty — so this is the one that must exist.
        layers = [lyr for lyr in all_layers
                  if "Overtopping (full)" in lyr.name()]
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

        # Which storm the band is about. It is measured on the full pond, so
        # unqualified it read as a claim about the event just routed — beside an
        # event water line drawn a metre below the crest. The event pond is built
        # immediately before this layer, so "unknown" here means the two stopped
        # being handed to each other.
        assert feat["state"] in ("event", "capacity"), (
            f"the band does not say which storm it is about: {feat['state']!r}")
        assert feat["event_m"] is not None, "no event level was measured"
        reaches = feat["event_m"] >= feat["level_m"] - 1e-3
        assert (feat["state"] == "event") == reaches, (
            f"state {feat['state']!r} disagrees with the levels it was derived "
            f"from: event {feat['event_m']} vs pour {feat['level_m']}")

        # And the split into two layers, which is what makes the distinction usable:
        # the event bands can be judged against the storm with the capacity ones
        # ticked off, and back on to ask about freeboard.
        event_layers = [lyr for lyr in all_layers
                        if "Overtopping (event)" in lyr.name()]
        if reaches:
            assert event_layers, (
                "this event reaches the crest and no '(event)' layer was drawn: "
                f"{sorted(lyr.name() for lyr in all_layers)}")
            assert all(f["state"] == "event"
                       for f in event_layers[0].getFeatures()), (
                "a band that does not overtop this event is in the event layer")
        else:
            assert not event_layers, (
                "nothing overtops this event, so an '(event)' layer asserts a no "
                "the map did not measure")


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

        # The area readout is deliberately NOT gated on the checkbox: it describes
        # the design, not the layer. Asserted with the layer OFF so a later change
        # that ties the two together fails here rather than in someone's field run.
        lbl = h.panel._catchment_coverage_lbl
        # isHidden(), not isVisible(): the dock window is never shown in this harness.
        assert not lbl.isHidden(), "coverage readout hid when the layer went off"
        text = lbl.text()
        for want in ("Catchment worked", "of site area", "drains into a feature"):
            assert want in text, f"expected {want!r} in the readout, got: {text}"

        cov = h.plugin._earthworks.compute_catchment_coverage()
        assert cov is not None, "no coverage computed with a labelled design"
        # It must reconcile with the labelling the map draws, not merely look sane.
        expected = (sum(h.state.catchment_counts.values())
                    / int(h.state.flow_domain_mask.sum()) * 100.0)
        assert abs(cov["managed_pct"] - expected) < 1e-6, (
            f"readout {cov['managed_pct']} vs labelling {expected}")
        # And with the same denominator the capture score uses.
        assert abs(cov["catchment_m2"]
                   - int(h.state.flow_domain_mask.sum()) * h.state.flow_grid_meta[
                       "cell_area_m2"]) < 1e-6
        parts = (cov["managed_m2"] + cov["exit_m2"] + cov["sink_m2"] + cov["other_m2"])
        assert abs(parts - cov["catchment_m2"]) < 1e-6, "buckets are not a partition"


def check_catchment_coverage_survives_disabling_everything(dem_path):
    """Every feature disabled is 0%, not a blank.

    The live assessment's `have_flow` goes False here (no counts), so a readout wired
    to that flag would vanish at exactly the moment the user is asking what the
    feature was doing. Deleting the last feature *is* different, and does clear it.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale")
        h.panel.analysis_inputs_changed.emit()
        assert not h.panel._catchment_coverage_lbl.isHidden()

        for ew in h.state.earthwork_manager.get_all():
            ew.enabled = False
        h.plugin._earthworks.recompute_catchments()
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("recompute with everything disabled")

        cov = h.plugin._earthworks.compute_catchment_coverage()
        assert cov is not None, "coverage went blank when features were disabled"
        assert cov["managed_pct"] == 0.0, f"expected 0%, got {cov['managed_pct']}"
        assert not h.panel._catchment_coverage_lbl.isHidden(), "readout hid at 0%"


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
        first = ew.spillway.width_required_m
        assert first, "the requirement was never recorded beside the built width"

        # Doubling the design intensity doubles the peak flow, and the weir width is
        # linear in flow, so the requirement must follow without reopening anything.
        #
        # Asserted on the **requirement**, not on the built width. The built width is
        # the requirement rounded up to whole DEM cells, so on a 1 m grid it moves in
        # 1 m steps and a real doubling can land inside one — which is a fact about the
        # grid, not a width that went stale. That distinction is the whole reason the
        # two figures are kept apart.
        h.panel.set_peak_intensity(h.panel.peak_intensity_mm_hr * 2)
        h.assert_no_errors("auto width after intensity change")
        second = ew.spillway.width_required_m
        assert second > first * 1.5, (
            f"required width stayed at {second} m after the intensity doubled "
            f"(was {first} m)"
        )
        # And the built width is that requirement, rounded up to a whole cell — which
        # is the width the burn will cut and therefore the width the map must label.
        cell = h.plugin._earthworks._dem_cell_size_m()
        expected = math.ceil(second / cell - 1e-9) * cell
        assert abs(ew.spillway.width_m - expected) < 1e-6, (
            f"built width {ew.spillway.width_m} m is not the {second:.2f} m "
            f"requirement rounded up to the {cell} m grid ({expected} m)"
        )


def check_a_placed_spillway_is_cut_into_the_burned_dem(dem_path):
    """The whole of Stage B, end to end: a sited spillway moves the terrain.

    Until this landed the burn was spillway-blind — a placed spillway changed no raster,
    no routing and no pond — so the three things asserted here are the ones that could
    not previously be true at once: the burn records a notch, the burned surface is
    lowered to the designed crest along it, and the `Spillways (burned)` layer that says
    so lands under **Verify** through ``_groups`` rather than loose at the top of the
    legend.

    Drawn across the valley, where the ground below the sill falls away — a notch that
    cannot daylight is refused on purpose, and this check is about the case that works.
    """
    import numpy as np
    import rasterio
    from qgis.core import QgsProject

    from terrainflow_assessment.qgis.controllers import _groups as G

    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        geom = line_across_valley(row=60)
        ew = h.add_earthwork("swale", geometry=geom)
        # ``add_earthwork`` bypasses the dialog, and the dialog is what sizes a feature.
        # Left at capacity 0 it drops out of the verification pass, and with it out of
        # the pond attribution the measured spill level is read from — so the check
        # would assert the notch and quietly skip half of what it is here for.
        h.plugin._earthworks._on_vertex_edit_finished(0, geom)
        point = ew.geometry.interpolate(ew.geometry.length() / 2.0).asPoint()

        # Placed at the level the feature is held to, which the band then clamps down to
        # `containment - head - freeboard` — the highest sill this feature can offer, and
        # the one the headless before/after measured. Passing None here seeds no crest at
        # all: `_on_spillway_placed` only reads the ground when it is given one.
        _lip, _invert, containment, _src = controller._spillway_datums(
            ew.geometry, ew.type, top_width_m=ew.top_width_m, depth=ew.depth, ew=ew)
        assert containment is not None, "no datum — the check cannot site anything"
        controller._on_spillway_placed(ew.id, point, containment, kind="outflow")
        h.assert_no_errors("spillway placed")
        assert ew.spillway is not None and ew.spillway.point_wkt, "the sill was not sited"
        crest = ew.spillway.crest_elevation
        assert crest is not None, "no crest to cut to"

        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis with a sited spillway")

        notches = getattr(h.state.burner, "burned_notches", None) or {}
        assert ew.id in notches, (
            f"no notch was cut for {ew.name} — warnings: "
            f"{list(getattr(h.state.burner, 'warnings', []))}")
        mask = notches[ew.id]

        with rasterio.open(h.state.modified_dem_path) as src:
            burned = src.read(1).astype("float64")
        assert float(np.nanmax(burned[mask])) <= crest + 1e-3, (
            f"the notch was recorded but the surface still stands at "
            f"{float(np.nanmax(burned[mask])):.2f} m against a {crest:.2f} m crest")

        # And the elevations the review reads, measured off that burn rather than
        # echoed back from the design. On a clean cut the as-burned sill *is* the
        # designed one; the value of the figure is that it can disagree.
        assert ew.burned_sill_elevation_m is not None, (
            "the as-burned sill was never recorded, so the review has nothing to "
            "compare the designed sill against")
        assert abs(ew.burned_sill_elevation_m - crest) < 0.01, (
            f"the notch cut to {ew.burned_sill_elevation_m:.2f} m against a designed "
            f"{crest:.2f} m sill")

        layer = next(
            (lyr for lyr in QgsProject.instance().mapLayers().values()
             if "Spillways (burned)" in lyr.name()), None)
        assert layer is not None, (
            "no Spillways (burned) layer after a notch was cut: "
            + str(sorted(lyr.name() for lyr in
                         QgsProject.instance().mapLayers().values())))
        assert layer.featureCount() > 0, "the burned spillway layer is empty"

        group = G.group(h.plugin._project, G.VERIFY,
                        site_name=h.panel.site_name, tag=h.state.run_tag)
        assert group is not None, "no Verify group to place it under"
        under = [n.layer().name() for n in group.findLayers() if n.layer() is not None]
        assert any("Spillways (burned)" in name for name in under), (
            f"the burned spillway layer is not under Verify — that group holds {under}")


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


def check_unticking_a_sited_spillway_asks_first(dem_path):
    """Clearing the Spillway tick must not silently delete a sited overflow.

    ``get_spillway()`` returns None when the group is unticked, and that used to be
    assigned straight onto the feature — taking the location, the crest and the width
    with it, from one small gesture, with no undo stack anywhere in this plugin to get
    them back. Harmless while a spillway changed no terrain; from the moment one cuts a
    notch, an accidental untick un-cuts a hole in a dam.

    The harness answers ``question()`` with No unless told otherwise, which is the case
    that matters: declining has to keep everything.
    """
    from qgis.PyQt.QtWidgets import QMessageBox

    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog
    from terrainflow_assessment.modules.earthwork_design import Spillway

    original_exec = EarthworkPropertiesDialog.exec
    original_get = EarthworkPropertiesDialog.get_spillway
    EarthworkPropertiesDialog.exec = lambda self: 1
    EarthworkPropertiesDialog.get_spillway = lambda self: None      # the tick, cleared
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            ew = h.add_earthwork("swale", geometry=line_across_valley(row=60))
            ew.spillway = Spillway(crest_elevation=41.5, width_m=3.0,
                                   width_auto=False, point_wkt="POINT (100 100)")

            h.plugin._earthworks.edit_earthwork_at(0)
            h.assert_no_errors("edit with the spillway group unticked")

            asked = h.dialogs.of("question")
            assert asked, "unticking a sited spillway was accepted without asking"
            assert "41.50" in asked[-1][2], (
                f"the question must name what is being lost: {asked[-1][2]!r}")
            assert ew.spillway is not None, (
                "declining the question still removed the spillway")
            assert ew.spillway.point_wkt == "POINT (100 100)", (
                "the sited location was lost even though the removal was declined")

            # And accepting really does remove it — otherwise the tick means nothing.
            h.dialogs.answer = QMessageBox.Yes
            h.plugin._earthworks.edit_earthwork_at(0)
            h.assert_no_errors("edit with the removal accepted")
            assert ew.spillway is None, "accepting the question did not remove it"
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        EarthworkPropertiesDialog.get_spillway = original_get


def check_an_unsited_spillway_is_cleared_without_asking(dem_path):
    """The confirmation is about losing a *placed* structure, not about the tick itself.

    A spillway that was never sited carries nothing the user cannot retype, so asking
    about it would be a dialog on an ordinary edit — which is how a confirmation stops
    being read.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog
    from terrainflow_assessment.modules.earthwork_design import Spillway

    original_exec = EarthworkPropertiesDialog.exec
    original_get = EarthworkPropertiesDialog.get_spillway
    EarthworkPropertiesDialog.exec = lambda self: 1
    EarthworkPropertiesDialog.get_spillway = lambda self: None
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            ew = h.add_earthwork("swale", geometry=line_across_valley(row=60))
            ew.spillway = Spillway(crest_elevation=41.5, width_m=3.0)   # never placed

            h.plugin._earthworks.edit_earthwork_at(0)
            h.assert_no_errors("edit clearing an unsited spillway")

            assert not h.dialogs.of("question"), (
                "an unsited spillway should be cleared without a confirmation")
            assert ew.spillway is None
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        EarthworkPropertiesDialog.get_spillway = original_get


def check_a_map_placed_crest_is_held_inside_its_band(dem_path):
    """A click records a place, not a licence to put the crest anywhere.

    ``_on_spillway_placed`` bound the crest without a band, so a click on ground above
    the level that contains the feature was stored verbatim — a crest that cannot pass
    its design head with any freeboard at all, and once the notch is cut, a hole at
    whatever elevation happened to be under the cursor.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=60))
        point = ew.geometry.interpolate(ew.geometry.length() / 2.0).asPoint()

        _lip, invert, containment, _src = controller._spillway_datums(
            ew.geometry, ew.type, top_width_m=ew.top_width_m, depth=ew.depth, ew=ew)
        assert containment is not None, "no datum — the check cannot say anything"

        # A click a clear metre above anything this feature can contain.
        controller._on_spillway_placed(ew.id, point, containment + 1.0, kind="outflow")
        h.assert_no_errors("spillway placed above the containing ground")

        assert ew.spillway is not None and ew.spillway.crest_elevation is not None
        # The ceiling is read back through the controller's own helper, with the
        # spillway that was actually created: head and freeboard are per-type policy,
        # so recomputing it here with the embankment defaults would test a different
        # feature from the one that was placed.
        _lo, ceiling = controller._crest_band_for(ew, ew.spillway, containment, invert)
        assert ew.spillway.crest_elevation <= ceiling + 1e-6, (
            f"a clicked crest of {containment + 1.0:.2f} m was stored as "
            f"{ew.spillway.crest_elevation:.2f} m, above the {ceiling:.2f} m ceiling "
            f"this feature can offer")
        assert ew.spillway.crest_elevation < containment + 1.0 - 1e-6, (
            "the clicked elevation was stored verbatim — the band was never applied")
        # And the partners have to have come with it, or the dialog opens disagreeing
        # with the value it is showing.
        assert abs(ew.spillway.drop_below_rim_m
                   - (containment - ew.spillway.crest_elevation)) < 1e-6, (
            "the crest was clamped but its drop was left describing the unclamped value")

        # An inlet is not a weir and must NOT be held under a weir's ceiling — clamping
        # one would move the recorded entry point off the ground the user clicked.
        controller._on_spillway_placed(ew.id, point, containment + 1.0, kind="inflow")
        h.assert_no_errors("inflow spillway placed")
        assert ew.inflow_spillway.crest_elevation == containment + 1.0, (
            f"the inlet was clamped to {ew.inflow_spillway.crest_elevation:.2f} m; it "
            f"records where water enters, not a weir crest")


def _configure_spillway(controller, index, configure):
    """Open the properties dialog on a feature, run *configure* on it, accept.

    The dialog is modal, so ``exec`` is stubbed the way
    ``check_properties_dialog_path`` stubs it — but here the stub also *uses* the
    dialog before returning accepted, which is the only way to exercise a control the
    user would have typed into. Everything else (construction against real terrain, the
    bindings, ``get_spillway``, the controller's write-back) runs for real.

    Returns the dialog, so a check can assert on what it was showing at the moment it
    was accepted.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    seen = {}

    def _exec(dlg):
        seen["dlg"] = dlg
        configure(dlg)
        return 1

    original = EarthworkPropertiesDialog.exec
    EarthworkPropertiesDialog.exec = _exec
    try:
        controller.edit_earthwork_at(index)
    finally:
        EarthworkPropertiesDialog.exec = original
    return seen.get("dlg")


def check_the_sill_depth_is_the_control_and_the_elevation_follows(dem_path):
    """Log 1: depth in, overflow elevation out — and the elevation is not typeable.

    The dialog used to offer the crest, the drop and the height as three editable ways
    into the same sill, plus a head and a freeboard that both moved the band the crest
    was clamped into. Five controls for one decision. The depth below the containing
    ground is now the decision, and everything else on those rows reports.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))

        def configure(dlg):
            assert dlg.spin_spillway_crest.isReadOnly(), (
                "the overflow elevation is still typeable — it is derived from the "
                "depth and cannot also be an input")
            assert not dlg.spin_spillway_drop.isReadOnly(), (
                "the sill depth is read-only; it is the one control the user has")
            dlg.grp_spillway.setChecked(True)
            dlg.spin_spillway_drop.setValue(0.45)

        dlg = _configure_spillway(controller, 0, configure)
        h.assert_no_errors("sill depth set through the dialog")

        containment = dlg._containment_elevation
        assert containment is not None, "no datum — the check cannot say anything"
        sp = ew.spillway
        assert sp is not None, "accepting a ticked spillway group stored nothing"
        # To the centimetre the design is expressed in, not to the millimetre: every
        # control on this dialog rounds to two decimals, so a crest derived from a
        # datum with more precision than that lands within half a displayed step. What
        # must be exact is the *relationship* — asserted below.
        assert abs(sp.crest_elevation - (containment - 0.45)) < 0.005, (
            f"overflow elevation {sp.crest_elevation:.3f} m does not sit 0.45 m under "
            f"the {containment:.3f} m this feature is held to — the two rows disagree")
        assert abs(sp.drop_below_rim_m
                   - (containment - sp.crest_elevation)) < 1e-9, (
            f"the stored depth ({sp.drop_below_rim_m:.4f} m) is not the stored crest "
            f"measured against the stored datum — the two fields describe different "
            f"containing ground")


def check_a_sill_depth_of_zero_is_drawable_and_called_out(dem_path):
    """Log 1: 0.00 m is allowed, and is not allowed to look fine.

    The depth control was floored at the type's head plus freeboard by the band clamp,
    so a bank that simply overtops at its lowest point could not be represented at all.
    It can now be typed — and the freeboard readout goes negative and the warning block
    fills, because it is a thing to draw and not a thing to build.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=54))

        def configure(dlg):
            assert dlg.spin_spillway_drop.minimum() == 0.0, (
                f"the sill depth floors at {dlg.spin_spillway_drop.minimum()} m")
            dlg.grp_spillway.setChecked(True)
            dlg.spin_spillway_drop.setValue(0.0)

        dlg = _configure_spillway(controller, 0, configure)
        h.assert_no_errors("zero sill depth")

        containment = dlg._containment_elevation
        assert abs(ew.spillway.drop_below_rim_m) < 0.005, (
            f"a depth of 0.00 m was clamped back up to "
            f"{ew.spillway.drop_below_rim_m:.2f} m")
        assert abs(ew.spillway.crest_elevation - containment) < 0.005, (
            "a zero-depth sill must sit at the containing ground, not below it")
        assert dlg.spin_spillway_freeboard.value() < 0, (
            f"freeboard read {dlg.spin_spillway_freeboard.value():.2f} m on a sill with "
            f"no depth at all — the one row that says the design is unsafe said it was "
            f"fine")
        assert dlg.lbl_spillway_warn.text(), (
            "a sill level with the containing ground drew no warning")


def check_a_cut_feature_sets_its_sill_out_from_its_floor(dem_path):
    """Every feature with a floor may set the sill out from it — not swales only.

    The carve-out from "depth and width are the only inputs" is about having a floor to
    stand a staff on, so it belongs to the cut features: a basin has one for the same
    reason a swale does. A dam does not — its invert is the ground the wall stands on,
    and a height above that is the wall height, not a sill level — so it must not offer
    the row at all rather than offer it greyed.

    Also pins the binding both ways round: a height typed in has to move the depth and
    the elevation with it, or the three rows are three separate numbers again.
    """
    from _harness import CELL_M, ORIGIN_Y, centreline_x
    from qgis.core import QgsGeometry, QgsRectangle

    cx = centreline_x()

    def basin_at(row):
        y = ORIGIN_Y - row * CELL_M
        return QgsGeometry.fromRect(
            QgsRectangle(cx - 25.0, y - 25.0, cx + 25.0, y + 25.0))

    cases = (
        ("swale", line_across_valley(row=58)),
        ("basin", basin_at(72)),
    )
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()

        for i, (ew_type, geom) in enumerate(cases):
            h.add_earthwork(ew_type, geometry=geom)
            # `add_earthwork` bypasses the dialog, and the dialog is what sizes a
            # feature; the controller's settle pass is what gives one a measured floor
            # to set out from. Without it the invert is unknown and the row this check
            # is about would be legitimately disabled.
            controller._on_vertex_edit_finished(i, geom)

            def configure(dlg, _t=ew_type):
                assert dlg.spin_spillway_height is not None, (
                    f"a {_t} has a cut floor but was offered no way to set out from it")
                # Ticked first: Qt disables every child of an unchecked checkable group,
                # so `isEnabled` before this reports the group's state and not the row's.
                dlg.grp_spillway.setChecked(True)
                assert not dlg.spin_spillway_height.isReadOnly(), (
                    f"'Height above floor' is read-only on a {_t} — a cut feature must "
                    f"be settable from its floor, which is the one measurement a "
                    f"builder can take standing in it")
                assert dlg.spin_spillway_height.isEnabled(), (
                    f"a {_t} reached the dialog with no invert to measure against")
                dlg.spin_spillway_height.setValue(0.30)

            dlg = _configure_spillway(controller, i, configure)
            h.assert_no_errors(f"{ew_type} sill set out from the floor")

            ew = h.state.earthwork_manager.get(i)
            sp = ew.spillway
            assert sp is not None, f"the {ew_type}'s spillway was not stored"
            # Typed into the height row, and the other two came with it — to the
            # centimetre the dialog rounds every one of them to.
            assert abs(sp.height_above_floor_m - 0.30) < 0.005, (
                f"the {ew_type} stored {sp.height_above_floor_m} m above its floor, "
                f"not the 0.30 m set")
            assert abs(sp.crest_elevation
                       - (dlg._invert_elevation + 0.30)) < 0.005, (
                f"the {ew_type}'s overflow elevation did not follow the height typed "
                f"into the row above it")
            assert abs(sp.drop_below_rim_m
                       - (dlg._containment_elevation - sp.crest_elevation)) < 1e-9, (
                f"the {ew_type}'s depth and elevation describe different sills")

        # And a dam, which has no cut floor, must not carry the row at all.
        h.add_earthwork("dam", geometry=line_across_valley(row=86))
        dam_dlg = _configure_spillway(
            controller, len(cases), lambda dlg: dlg.grp_spillway.setChecked(True))
        h.assert_no_errors("dam sill")
        assert dam_dlg.spin_spillway_height is None, (
            "a dam was offered 'Height above floor' — its invert is the ground under "
            "the wall, so that row would read the wall height, not a sill level")


def check_a_configured_sill_is_not_reset_when_it_is_placed(dem_path):
    """Log 2: siting a spillway records *where*, never *how deep*.

    ``_on_spillway_placed`` re-seeds the crest from the ground under the click while
    ``Spillway.auto`` is set, and nothing in the dialog ever cleared it — so every sill
    configured there was overwritten the moment it was placed. On Dam 1 of the reported
    design a 71.79 m sill came back at 70.16 m, 1.93 m below the spill level instead of
    0.30 m, and the feature went from full to holding nothing.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=56))

        def configure(dlg):
            dlg.grp_spillway.setChecked(True)
            dlg.spin_spillway_drop.setValue(0.40)

        _configure_spillway(controller, 0, configure)
        h.assert_no_errors("sill configured")
        assert ew.spillway.auto is False, (
            "a sill the user set is still flagged auto, so placing it will re-read the "
            "crest off the ground — this is the defect itself")

        chosen = ew.spillway.crest_elevation
        point = ew.geometry.interpolate(ew.geometry.length() / 2.0).asPoint()
        # A click on ground a clear metre below the sill that was configured: under the
        # old behaviour this is exactly what replaced it.
        controller._on_spillway_placed(ew.id, point, chosen - 1.0, kind="outflow")
        h.assert_no_errors("configured spillway placed")

        assert ew.spillway.point_wkt, "placing the sill did not record its location"
        assert abs(ew.spillway.crest_elevation - chosen) < 1e-6, (
            f"placing the sill moved it from {chosen:.2f} m to "
            f"{ew.spillway.crest_elevation:.2f} m — the user's value was reset")
        assert abs(ew.spillway.drop_below_rim_m - 0.40) < 0.005, (
            "the crest survived but its depth did not — the two now disagree")


def check_the_review_says_what_each_sill_gives_up(dem_path):
    """The Spillways review carries the storage figures, or says nothing at all.

    Two states, and the second is the one worth pinning: before anything is measured
    there is no stage-storage curve, and the row must be blank rather than reporting
    that this sill gives up zero cubic metres.
    """
    from terrainflow_assessment.modules.earthwork_design import Spillway

    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("basin", geometry=line_across_valley(
            row=60, half_width_m=15.0))
        ew.spillway = Spillway(crest_elevation=None, width_m=2.0)
        h.panel.analysis_inputs_changed.emit()
        controller._build_spillway_rows()
        h.assert_no_errors("spillway review, unmeasured")

        row = next(r for r in h.panel._spillway_table._rows if r["id"] == ew.id)
        assert "lip_elevation" in row and "given_up_m3" in row, (
            "the review row lost the datum keys the report reads")
        assert row["given_up_m3"] is None, (
            "an unmeasured feature reported a give-up figure — a blank is the honest "
            "answer, and a zero reads as 'this sill costs nothing'")

        # A row that never gets past the flow gate returns before the datums are read,
        # so the rest of this check would be asserting against an early return.
        if row["state"] in ("no_flow", "disabled", "no_datum"):
            raise AssertionError(
                f"the review row for {ew.name} is {row['state']!r}, so the storage "
                f"figures were never reached — the fixture stopped producing flow")

        # Now measure it, and set a sill part way down the pond.
        controller._refresh_terrain_capacity(ew, quiet=True)
        curve = getattr(ew, "stage_storage", None)
        level = ew.terrain_spill_level_m
        assert curve is not None and level is not None, (
            f"{ew.name} was measured at {ew.terrain_capacity_m3} m3 but produced no "
            f"stage-storage curve — the readout has nothing to show")

        floor = float(curve.levels_m[0])
        ew.spillway.crest_elevation = floor + 0.5 * (level - floor)
        controller._build_spillway_rows()
        h.assert_no_errors("spillway review, measured")

        row = next(r for r in h.panel._spillway_table._rows if r["id"] == ew.id)
        assert row["containment_storage_m3"] is not None, (
            f"no full-pond figure on a measured feature (row state {row['state']!r}, "
            f"held to {row['rim_elevation']})")
        assert row["sill_storage_m3"] is not None, "no at-the-sill figure"
        assert row["sill_storage_m3"] <= row["containment_storage_m3"] + 1e-6, (
            f"a sill below the spill level holds {row['sill_storage_m3']} m3 against a "
            f"full pond of {row['containment_storage_m3']} m3")
        assert row["given_up_m3"] > 0, (
            "a sill half way down the pond gives up nothing at all")


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


def check_linking_a_drain_to_a_spillway_grades_it_from_the_crest(dem_path):
    """Stage C end to end: two clicks, and the drain is cut from the crest.

    Driven with **real synthetic clicks** rather than by emitting the tool's signal,
    because the novel thing about this tool is that its first click picks an *endpoint*
    and that endpoint decides which way the drain falls. Grading from the wrong end
    produces a drain running uphill from an entirely plausible-looking level — no
    exception, no warning, no odd number — so the click-to-endpoint resolution is the
    part worth exercising through the whole conversion chain.

    Four things are asserted, and only the first is about the tool: the link records the
    end that was clicked; the level it resolves to is the source's crest; the burn cuts
    the drain from that level rather than from the ground under its own line; and
    repeating the same gesture removes the link.
    """
    import numpy as np
    import rasterio
    from _mouse import click_map
    from qgis.core import QgsGeometry, QgsPointXY

    from terrainflow_assessment.map_tools.link_spillway_tool import LinkSpillwayTool
    from terrainflow_assessment.modules.earthwork_design import parse_spillway_link

    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        h.prepare_canvas_for_input()

        # The source: a swale across the valley with a sited outflow spillway, sized
        # through the vertex-edit path so it has a capacity and a datum to work from.
        source_geom = line_across_valley(row=60)
        source = h.add_earthwork("swale", geometry=source_geom, name="Source swale")
        controller._on_vertex_edit_finished(0, source_geom)
        sill = source.geometry.interpolate(source.geometry.length() / 2.0).asPoint()
        _lip, _invert, containment, _src = controller._spillway_datums(
            source.geometry, source.type, top_width_m=source.top_width_m,
            depth=source.depth, ew=source)
        assert containment is not None, "no datum — the check cannot site a spillway"
        controller._on_spillway_placed(source.id, sill, containment, kind="outflow")
        crest = source.spillway.crest_elevation
        assert crest is not None, "no crest to link to"

        # The drain: drawn well clear of the source, so a click near one of its ends
        # cannot be nearer to anything else.
        west = (sill.x() - 90.0, sill.y() - 60.0)
        east = (sill.x() + 90.0, sill.y() - 60.0)
        drain = h.add_earthwork(
            "diversion", name="Drain 9",
            geometry=QgsGeometry.fromPolylineXY(
                [QgsPointXY(*west), QgsPointXY(*east)]))

        h.panel.link_drain_to_spillway_requested.emit()
        h.assert_no_errors("activate the drain-link tool")
        tool = h.canvas.mapTool()
        assert isinstance(tool, LinkSpillwayTool), (
            f"expected LinkSpillwayTool on the canvas, got {type(tool).__name__}")

        # Click the drain's EAST end, then the source. East is the drain's last vertex,
        # so the link must record "end" — not "start", which is what a tool that only
        # picked whole features would have had to assume.
        click_map(h.canvas, *east)

        # The second click hit-tests the **sill**, not the feature carrying it. A click
        # on the swale but well away from the sill must therefore do nothing and leave
        # the tool armed — otherwise a feature carrying both an outflow and an inlet
        # could not be told apart, and a click aimed at one would be right by accident.
        far_end = source.geometry.asPolyline()[0]
        click_map(h.canvas, far_end.x(), far_end.y())
        assert drain.spillway_link_id is None, (
            "a click on the feature away from the sill linked anyway — the second "
            "click is still hit-testing the feature rather than the spillway")

        click_map(h.canvas, sill.x(), sill.y())
        h.assert_no_errors("linking the drain to the spillway")

        link = parse_spillway_link(drain.spillway_link_id)
        assert link is not None, f"no link was recorded — messages:\n{h.bar.render()}"
        assert link[0] == source.id, "the link names the wrong feature"
        assert link[2] == "end", (
            f"clicked the drain's last vertex and the link recorded {link[2]!r} — "
            f"the drain would be graded from the wrong end")
        assert drain.invert_start_m == crest, (
            f"the link resolved to {drain.invert_start_m!r} against a {crest!r} crest")

        # And the burn uses it. The drain is well clear of the source's own cut, so
        # anything that happens along it is the link's doing.
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("re-analysis with a linked drain")
        with rasterio.open(h.state.modified_dem_path) as src:
            burned = src.read(1).astype("float64")
        band = h.state.burner.burned_masks.get(drain.id)
        assert band is not None and band.any(), "the drain burned nothing at all"
        deepest = float(np.nanmin(burned[band]))
        assert deepest <= crest + 1e-6, (
            f"the linked drain bottoms out at {deepest:.2f} m, which is above the "
            f"{crest:.2f} m crest it was supposed to start from")

        # Repeating the gesture unlinks — the only way back, since there is no undo
        # stack anywhere in this plugin.
        h.panel.link_drain_to_spillway_requested.emit()
        click_map(h.canvas, *east)
        click_map(h.canvas, sill.x(), sill.y())
        h.assert_no_errors("unlinking the drain")
        assert drain.spillway_link_id is None, (
            "repeating the same link did not remove it")
        assert drain.invert_start_m is None, (
            "the link went but the level it resolved to stayed on the feature")


def check_a_drain_link_survives_a_project_roundtrip(dem_path):
    """The link is stored; the level it resolves to is derived on the way back in.

    A link is a decision and is recoverable from nothing, which is what took
    ``SCHEMA_VERSION`` to 3. The level is another feature's crest, so it is re-derived
    by the restore path rather than trusted from the file — ``_refresh_spillway_link_
    inverts`` runs in the same step list as ``_refresh_auto_spillway_widths``, and for
    the same reason.
    """
    from terrainflow_assessment.modules.earthwork_design import format_spillway_link

    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()

        source_geom = line_across_valley(row=60)
        source = h.add_earthwork("swale", geometry=source_geom, name="Source swale")
        controller._on_vertex_edit_finished(0, source_geom)
        sill = source.geometry.interpolate(source.geometry.length() / 2.0).asPoint()
        _lip, _invert, containment, _src = controller._spillway_datums(
            source.geometry, source.type, top_width_m=source.top_width_m,
            depth=source.depth, ew=source)
        controller._on_spillway_placed(source.id, sill, containment, kind="outflow")
        crest = source.spillway.crest_elevation

        drain = h.add_earthwork("diversion", name="Drain 9")
        drain.spillway_link_id = format_spillway_link(source.id, "outflow", "end")

        text = h.state.earthwork_manager.to_json()
        restored = controller.restore_earthworks_from_json(text)
        h.assert_no_errors("restoring a design with a linked drain")
        assert restored == 2, f"restored {restored} features, expected 2"

        back = [e for e in h.state.earthwork_manager.get_all()
                if e.name == "Drain 9"][0]
        assert back.spillway_link_id, "the link did not survive the round trip"
        assert back.invert_start_m == crest, (
            f"the restored drain resolved to {back.invert_start_m!r} against a "
            f"{crest!r} crest — the restore path did not re-derive it")


def check_removing_a_spillway_names_the_drains_that_lose_their_level(dem_path):
    """The removal confirmation has to state the second thing it costs.

    Removing a spillway does not clear the links pointing at it — those resolve at read
    time and simply stop resolving, which is deliberate. So without this sentence the
    user answers a smaller question than the one being asked: the drains go on looking
    linked, and go back to grading from the ground under their own alignment.
    """
    from terrainflow_assessment.earthwork_properties_dialog import (
        EarthworkPropertiesDialog,
    )
    from terrainflow_assessment.modules.earthwork_design import (
        Spillway,
        format_spillway_link,
    )

    original_exec = EarthworkPropertiesDialog.exec
    original_get = EarthworkPropertiesDialog.get_spillway
    EarthworkPropertiesDialog.exec = lambda self: 1
    EarthworkPropertiesDialog.get_spillway = lambda self: None
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            source = h.add_earthwork("swale", geometry=line_across_valley(row=60),
                                     name="Source swale")
            source.spillway = Spillway(crest_elevation=41.5, width_m=3.0,
                                       width_auto=False, point_wkt="POINT (100 100)")
            drain = h.add_earthwork("diversion", name="Drain 9")
            drain.spillway_link_id = format_spillway_link(source.id)

            h.plugin._earthworks.edit_earthwork_at(0)
            h.assert_no_errors("removing a spillway that feeds a drain")

            asked = h.dialogs.of("question")
            assert asked, "removing a sited spillway was accepted without asking"
            text = asked[-1][2]
            assert "Drain 9" in text, (
                f"the question must name the drain that loses its level: {text!r}")
            assert "41.50" in text, "the question still has to name the crest"
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        EarthworkPropertiesDialog.get_spillway = original_get


# ---------------------------------------------------------------------------
# Setting the sill depth from the review table
# ---------------------------------------------------------------------------

def _review_row(h, ew):
    """The review row for *ew*, rebuilt fresh.

    The recompute first, because a row with no peak flow returns before it reads any
    datums -- so without it every feature reads `no_flow` and every assertion below
    would be about the wrong state.
    """
    h.panel.analysis_inputs_changed.emit()
    h.plugin._earthworks._build_spillway_rows()
    return next(r for r in h.panel._spillway_table._rows if r["id"] == ew.id)


def check_a_depth_typed_in_the_review_designs_a_spillway(dem_path):
    """The headline: a feature with no spillway gets one, at this type's design depth.

    Eleven of the thirty-one features on the reported design are in this state, and each
    one used to cost a modal round trip. Everything asserted here is something the dialog
    would have got right and a table path can quietly get wrong: the type's head rather
    than the constructor's, a freeboard left on policy rather than frozen at today's
    figure, a width still on auto, and a sill that is designed without claiming to be
    placed.
    """
    from terrainflow_assessment.modules.earthwork_design import spillway_policy

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        row = _review_row(h, ew)
        assert row["state"] == "undesigned", (
            f"fixture is not in the state this check is about: {row['state']!r}")
        containment = row["rim_elevation"]
        assert containment is not None, "no datum — the check cannot say anything"

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.45)
        h.assert_no_errors("sill depth typed on an undesigned feature")

        sp = ew.spillway
        assert sp is not None, "typing a depth did not design a spillway"
        assert abs(sp.crest_elevation - (containment - 0.45)) < 1e-6, (
            f"crest {sp.crest_elevation:.4f} m does not sit 0.45 m under the "
            f"{containment:.4f} m this feature is held to")
        assert abs(sp.drop_below_rim_m - 0.45) < 1e-9, (
            f"stored depth is {sp.drop_below_rim_m!r}, not the 0.45 m typed")
        assert sp.head_m == spillway_policy("swale")[1], (
            f"head is {sp.head_m}, not the swale policy — a spillway created here must "
            f"not carry the Spillway() constructor default")
        assert sp.freeboard_m is None, (
            "freeboard was written in as a number; None is what means 'the type's "
            "policy', and a figure frozen here would stop tracking a policy change")
        assert sp.width_auto is True, "the width should still be tracking"
        assert sp.point_wkt is None, (
            "designing a sill sited it — the review must not claim a location the user "
            "has not chosen")
        assert sp.auto is False, (
            "auto is still set, so placing this sill would re-seed the crest off the "
            "ground and throw the typed depth away")

        row = _review_row(h, ew)
        assert row["designed"] is True and row["state"] == "unsited"
        assert abs(row["sill_depth_m"] - 0.45) < 1e-6, row["sill_depth_m"]
        assert row["built_width_m"], (
            "the auto width was never derived — a designed sill with no width is what "
            "_refresh_auto_spillway_widths exists to prevent")


def check_a_typed_depth_is_the_depth_stored(dem_path):
    """No band on the way in: type 0.10 m and 0.10 m is what is kept.

    The placement path clamps a crest into the band the type's head and freeboard leave,
    because a map click lands on arbitrary ground. A typed number is not arbitrary, and
    clamping it would show the user a corrected figure where theirs had just been.
    A shallow sill is reported in words instead.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=54))
        row = _review_row(h, ew)
        containment = row["rim_elevation"]

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.10)
        h.assert_no_errors("shallow sill depth typed")

        assert abs(ew.spillway.drop_below_rim_m - 0.10) < 1e-9, (
            f"0.10 m was typed and {ew.spillway.drop_below_rim_m:.2f} m stored — a band "
            f"is clamping the value on its way in")
        assert abs(ew.spillway.crest_elevation - (containment - 0.10)) < 1e-6
        row = _review_row(h, ew)
        assert row["problems"], (
            "a sill far too shallow for its head passed without comment; the clamp was "
            "removed on the understanding that spillway_validity would say so instead")


def check_retyping_the_depth_on_screen_does_not_move_the_sill(dem_path):
    """The round trip is a no-op, which is only true because the cell derives its figure.

    `drop_below_rim_m` is re-based when a design is restored and not afterwards, while
    the review recomputes its containment on every build — so after an earthworks
    re-analysis the two can be measured against different rims. A cell rendering the
    stored figure would move the crest when the user typed back the number in front of
    them, which is the worst thing an editable cell can do.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("basin", geometry=line_across_valley(
            row=60, half_width_m=15.0))
        row = _review_row(h, ew)
        h.panel.set_spillway_depth_requested.emit(row["index"], 0.55)
        h.assert_no_errors("first depth")

        row = _review_row(h, ew)
        shown = row["sill_depth_m"]
        before = ew.spillway.crest_elevation
        h.panel.set_spillway_depth_requested.emit(row["index"], shown)
        h.assert_no_errors("depth retyped unchanged")

        assert abs(ew.spillway.crest_elevation - before) < 1e-9, (
            f"retyping the {shown:.4f} m the table was showing moved the crest from "
            f"{before:.4f} m to {ew.spillway.crest_elevation:.4f} m")


def check_a_typed_depth_keeps_everything_else_about_the_sill(dem_path):
    """Only the level moves. The spillway is mutated, never rebuilt.

    A freeboard override, a committed width and a placed location are all things the
    dialog is careful to carry through untouched. Constructing a fresh Spillway on edit
    would drop every one of them, and the freeboard override silently — it has no column.
    """
    from terrainflow_assessment.modules.earthwork_design import Spillway

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        row = _review_row(h, ew)
        containment = row["rim_elevation"]
        assert containment is not None, "no datum — the check cannot say anything"
        ew.spillway = Spillway(
            crest_elevation=containment - 0.40, drop_below_rim_m=0.40,
            head_m=0.20, width_m=3.0, width_auto=False, freeboard_m=0.05,
            point_wkt="POINT(0 0)", auto=False)

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.30)
        h.assert_no_errors("depth changed on a configured sill")

        sp = ew.spillway
        assert abs(sp.drop_below_rim_m - 0.30) < 1e-9, "the depth did not move"
        assert sp.freeboard_m == 0.05, (
            f"the freeboard override became {sp.freeboard_m!r} — the spillway was "
            f"rebuilt rather than mutated, and the override has no column to notice it")
        assert sp.head_m == 0.20, f"head became {sp.head_m}"
        assert sp.width_m == 3.0 and sp.width_auto is False, "the built width was lost"
        assert sp.point_wkt == "POINT(0 0)", "the sill was un-sited by a depth change"


def check_a_typed_depth_does_not_reflood_the_feature(dem_path):
    """Capacity follows the crest off the cached curve, with no depression fill.

    The flood behind the curve is deliberately brim-full with the notch not cut, so a
    crest move cannot invalidate it. This is the check that catches someone tidying
    `_reapply_sill_capacity` back into a `_refresh_terrain_capacity` call — which would
    buy identical numbers at a fill apiece, and would return silently mid-burn.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("basin", geometry=line_across_valley(
            row=60, half_width_m=15.0))
        row = _review_row(h, ew)
        h.panel.set_spillway_depth_requested.emit(row["index"], 0.50)
        controller._refresh_terrain_capacity(ew, quiet=True)
        curve = getattr(ew, "stage_storage", None)
        level = ew.terrain_spill_level_m
        brim = ew.containment_capacity_m3
        assert curve is not None and level is not None and brim, (
            "the fixture was never measured, so the claim cannot be tested")

        # Site it, or _sill_limited_capacity correctly declines to cut anything back.
        ew.spillway.point_wkt = "POINT(0 0)"
        controller._reapply_sill_capacity(ew)
        deep_capacity = ew.terrain_capacity_m3

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.05)
        h.assert_no_errors("sill raised")

        assert ew.stage_storage is curve, (
            "the stage-storage curve was rebuilt — the feature was re-flooded for a "
            "crest move that cannot change the flood")
        assert ew.terrain_spill_level_m == level, "the measured spill level moved"
        assert ew.containment_capacity_m3 == brim, "the brim volume moved"
        assert ew.terrain_capacity_m3 != deep_capacity, (
            "what the feature holds to its sill did not follow the sill")


def check_a_dams_first_sill_is_the_one_case_that_re_floods(dem_path):
    """The single exception to "a depth edit never floods".

    `_refresh_dam_stage_storage` clears the measured levels and returns for a dam with no
    spillway, on the stated grounds that nothing would read the answer. Designing one from
    the review is the moment that stops being true, so that one edit has to pay for a
    fill. Every other edit must not, and the pair is asserted together — a change that
    made the cheap path unconditional would break the first half, and one that made the
    flood unconditional would break the second.

    The branch is asserted rather than the resulting curve: whether the fixture's terrain
    impounds anything behind a wall is a fact about the synthetic DEM, and the decision
    under test is which path the controller takes.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        dam = h.add_earthwork("dam", geometry=line_across_valley(row=86))
        swale = h.add_earthwork("swale", geometry=line_across_valley(row=52))

        floods = []
        original = controller._refresh_terrain_capacity
        controller._refresh_terrain_capacity = (
            lambda ew, *a, **kw: (floods.append(ew.id), original(ew, *a, **kw))[1])
        try:
            row = _review_row(h, dam)
            h.panel.set_spillway_depth_requested.emit(row["index"], 0.60)
            h.assert_no_errors("first sill on a dam")
            assert floods == [dam.id], (
                f"designing a dam's first sill did not re-measure it (floods={floods}) — "
                f"it has no stage-storage curve until this happens, so the Storage "
                f"column would have nothing to show and never would")

            floods.clear()
            row = _review_row(h, dam)
            h.panel.set_spillway_depth_requested.emit(row["index"], 0.40)
            h.assert_no_errors("second depth on the same dam")
            assert floods == [], (
                "moving a sill that already has a curve re-flooded the feature; the "
                "flood is brim-full with the notch not cut, so a crest cannot change it")

            row = _review_row(h, swale)
            h.panel.set_spillway_depth_requested.emit(row["index"], 0.35)
            h.assert_no_errors("first sill on a cut feature")
            assert floods == [], (
                "designing a cut feature's first sill re-flooded it — only a dam has no "
                "curve until it carries a spillway")
        finally:
            controller._refresh_terrain_capacity = original

        assert dam.spillway is not None
        assert dam.spillway.height_above_floor_m is None, (
            "a dam was given a height above floor — its invert is the ground under the "
            "wall rather than a cut floor, and that is a figure the dialog refuses to "
            "offer at all")


def check_a_typed_depth_of_zero_is_not_a_deletion(dem_path):
    """0.00 m is a drawable design that gets flagged, not a way to remove a spillway.

    Deleting one is unrecoverable and is confirmed in exactly one place. Nothing in this
    column may reach that outcome.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=58))
        row = _review_row(h, ew)
        h.panel.set_spillway_depth_requested.emit(row["index"], 0.35)
        row = _review_row(h, ew)

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.0)
        h.assert_no_errors("zero depth typed")

        assert ew.spillway is not None, "typing 0.00 m deleted the spillway"
        assert abs(ew.spillway.drop_below_rim_m) < 1e-9
        row = _review_row(h, ew)
        assert row["state"] == "fail" and row["problems"], (
            "a sill flush with the containing ground passed without comment")


def check_a_typed_depth_marks_the_design_edited(dem_path):
    """A moved sill is a design change, so the last burn is one edit staler."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=50))
        row = _review_row(h, ew)
        # A design that has never been verified has nothing to go stale, so the counter
        # stays None by design. Seed it as a burnt design would.
        h.state.edits_since_verify = 0
        before = h.state.edits_since_verify

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.35)
        h.assert_no_errors("depth typed")

        assert (h.state.edits_since_verify or 0) > before, (
            "setting a sill from the review left the design claiming to be verified")


def check_the_review_index_survives_a_filtered_row_list(dem_path):
    """A berm gets no row, so table position and earthwork index are different numbers.

    `_build_spillway_rows` keeps only the types that can spill and *then* stamps each row
    with its manager index. Anything that treats the row list as index-aligned edits the
    wrong feature the moment a berm sits above a swale.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        berm = h.add_earthwork("berm", geometry=line_across_valley(row=44))
        swale = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        row = _review_row(h, swale)
        assert row["index"] == 1, (
            f"fixture is not exercising the offset: the swale is at index "
            f"{row['index']}")

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.35)
        h.assert_no_errors("depth typed past a berm")

        assert swale.spillway is not None, "the depth went somewhere other than the swale"
        assert getattr(berm, "spillway", None) is None, (
            "the berm was given a spillway — the row list was read as index-aligned")


def check_only_the_sill_depth_column_can_be_typed_in(dem_path):
    """Requirement 3, asserted structurally rather than trusted.

    QTableWidgetItem is editable by *default*, so opening the edit triggers made every
    column editable until _fill_row started clearing the flag. Edits to the others would
    have been accepted and silently discarded on the next rebuild, which from the outside
    is indistinguishable from having worked.
    """
    from qgis.PyQt.QtCore import Qt

    from terrainflow_assessment.qgis.widgets.spillway_table import _COL_DEPTH, _HEADERS

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        live = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        dead = h.add_earthwork("swale", geometry=line_across_valley(row=68))
        dead.enabled = False
        h.plugin._earthworks._build_spillway_rows()

        table = h.panel._spillway_table
        assert table.table.columnCount() == len(_HEADERS), (
            "the table and its headers disagree about how many columns there are")
        assert table.table.horizontalHeaderItem(_COL_DEPTH).text() == "Sill depth", (
            "the editable column is not where _COL_DEPTH says it is")

        rows = table._rows
        assert rows, "nothing to inspect"
        for r, data in enumerate(rows):
            for c in range(table.table.columnCount()):
                item = table.table.item(r, c)
                editable = bool(item.flags() & Qt.ItemFlag.ItemIsEditable)
                wants = (c == _COL_DEPTH
                         and data.get("state") != "disabled"
                         and data.get("rim_elevation") is not None)
                assert editable == wants, (
                    f"{data['name']} column {_HEADERS[c]!r}: editable={editable}, "
                    f"expected {wants}")
        assert any(r["id"] == dead.id for r in rows), (
            "the disabled feature dropped out of the review, so the read-only half of "
            "this check asserted nothing")
        assert any(r["id"] == live.id for r in rows)


def check_the_depth_editor_writes_through(dem_path):
    """The delegate end to end, without simulating a mouse.

    createEditor -> setEditorData -> setModelData is the whole path the user drives, and
    it is where the seed value and the deferred commit both live.
    """
    from qgis.PyQt.QtCore import QCoreApplication

    from terrainflow_assessment.modules.earthwork_design import default_sill_depth_m
    from terrainflow_assessment.qgis.widgets.spillway_table import _COL_DEPTH

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        h.plugin._earthworks._build_spillway_rows()

        table = h.panel._spillway_table
        r = next(i for i, row in enumerate(table._rows) if row["id"] == ew.id)
        index = table.table.model().index(r, _COL_DEPTH)
        delegate = table.table.itemDelegateForColumn(_COL_DEPTH)
        assert delegate is not None, "no delegate on the sill depth column"

        editor = delegate.createEditor(table.table, None, index)
        delegate.setEditorData(editor, index)
        assert abs(editor.value() - default_sill_depth_m("swale")) < 1e-9, (
            f"an undesigned swale seeded at {editor.value()}, not at the head plus "
            f"freeboard its type is designed to")

        editor.setValue(0.42)
        delegate.setModelData(editor, table.table.model(), index)
        # The commit is handed to the event loop on purpose, so the rebuild it triggers
        # does not run inside the editor teardown.
        for _ in range(3):
            QCoreApplication.processEvents()

        assert ew.spillway is not None, "the editor did not reach the controller"
        assert abs(ew.spillway.drop_below_rim_m - 0.42) < 1e-9


def check_clicking_out_of_an_untouched_editor_creates_nothing(dem_path):
    """The silent-creation trap: Qt commits on focus-out, and this column must not.

    Opening the editor on a blank row and clicking elsewhere would otherwise design a
    spillway the user never asked for, in a plugin with no undo anywhere. A value they
    did change still commits on the way out — dropping a deliberate keystroke without
    saying so is the opposite mistake, not a safer one.
    """
    from qgis.PyQt.QtCore import QCoreApplication, QEvent
    from qgis.PyQt.QtGui import QFocusEvent

    from terrainflow_assessment.qgis.widgets.spillway_table import _COL_DEPTH

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        h.plugin._earthworks._build_spillway_rows()

        table = h.panel._spillway_table
        r = next(i for i, row in enumerate(table._rows) if row["id"] == ew.id)
        index = table.table.model().index(r, _COL_DEPTH)
        delegate = table.table.itemDelegateForColumn(_COL_DEPTH)

        editor = delegate.createEditor(table.table, None, index)
        delegate.setEditorData(editor, index)
        handled = delegate.eventFilter(editor, QFocusEvent(QEvent.Type.FocusOut))
        for _ in range(3):
            QCoreApplication.processEvents()

        assert handled is True, (
            "the untouched editor was allowed through to Qt's default focus-out commit")
        assert getattr(ew, "spillway", None) is None, (
            "clicking away from an editor nobody typed in designed a spillway")

        editor.setValue(0.33)
        delegate.eventFilter(editor, QFocusEvent(QEvent.Type.FocusOut))
        delegate.setModelData(editor, table.table.model(), index)
        for _ in range(3):
            QCoreApplication.processEvents()
        assert ew.spillway is not None and abs(
            ew.spillway.drop_below_rim_m - 0.33) < 1e-9, (
            "a value the user did change was dropped on focus-out")


def check_double_clicking_the_sill_depth_does_not_open_the_dialog(dem_path):
    """Double-click already meant "open properties". On this one column the editor wins.

    Both firing would raise the modal over the editor, and the dialog's OK would then
    write its own spillway over whatever had been typed.

    Driven against a standalone table rather than the panel's: emitting `edit_requested`
    through the live plugin opens the real properties dialog, which offscreen is a modal
    nothing dismisses.
    """
    from terrainflow_assessment.qgis.widgets.spillway_table import (
        _COL_DEPTH,
        SpillwayTable,
    )

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        row = _review_row(h, ew)
        assert row["rim_elevation"] is not None, "the fixture row is not editable"

        table = SpillwayTable()
        try:
            table.set_rows([row])
            seen = []
            table.edit_requested.connect(seen.append)

            table._on_cell_double_clicked(0, _COL_DEPTH)
            assert not seen, (
                "double-clicking the sill depth also asked for the properties dialog, "
                "which would come up over the editor the same gesture just opened")

            table._on_cell_double_clicked(0, 0)
            assert seen == [row["index"]], (
                f"double-clicking another cell stopped opening the properties dialog "
                f"(saw {seen!r})")
        finally:
            table.deleteLater()


def check_the_review_keeps_your_place_across_a_rebuild(dem_path):
    """Thirty-one features, eight visible rows, and a rebuild after every edit.

    Without this the user is thrown back to the top of the list each time they commit a
    depth, which on its own would make a pass down the column not worth doing. The anchor
    is the feature id, because a feature added above shifts every row number below it.
    """
    from terrainflow_assessment.qgis.widgets.spillway_table import _COL_DEPTH

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row_y in (44, 48, 52, 56):
            h.add_earthwork("swale", geometry=line_across_valley(row=row_y))
        controller = h.plugin._earthworks
        controller._build_spillway_rows()

        table = h.panel._spillway_table
        target = table._rows[2]["id"]
        table.table.setCurrentCell(2, _COL_DEPTH)

        # A feature inserted above pushes the anchor down a row; matching on position
        # would land on its neighbour.
        h.add_earthwork("swale", geometry=line_across_valley(row=40))
        controller._build_spillway_rows()

        current = table.table.currentRow()
        assert 0 <= current < len(table._rows), "the current cell was lost entirely"
        assert table._rows[current]["id"] == target, (
            f"the rebuild moved the cursor from {target} to "
            f"{table._rows[current]['id']} — it followed the row number, not the feature")
        assert table.table.currentColumn() == _COL_DEPTH


def check_a_rebuild_waits_for_an_open_editor(dem_path):
    """A background refresh must not take a half-typed depth with it.

    Any design change, and every storm or soil control, reaches `set_rows`. Replacing the
    items would destroy the open editor and the value in it, on a table where the user is
    working down a column. The rows are held instead and drawn when the editor closes.

    The view is asked to *report* that it is editing rather than being made to edit: an
    offscreen QTableWidget will not open an editor for a synthetic `edit()`, and the
    branch under test is what `set_rows` does with the answer. That the real editor opens
    and writes through is `check_the_depth_editor_writes_through`'s job.
    """
    from qgis.PyQt.QtWidgets import QAbstractItemView

    from terrainflow_assessment.qgis.widgets.spillway_table import _COL_DEPTH

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        row = _review_row(h, ew)
        h.panel.set_spillway_depth_requested.emit(row["index"], 0.45)
        _review_row(h, ew)

        table = h.panel._spillway_table
        r = next(i for i, data in enumerate(table._rows) if data["id"] == ew.id)
        before = table.table.item(r, _COL_DEPTH).text()
        assert before.startswith("0.45"), f"the fixture cell reads {before!r}"

        original_state = table.table.state
        table.table.state = lambda: QAbstractItemView.EditingState
        try:
            h.panel.set_spillway_depth_requested.emit(row["index"], 0.20)
            h.assert_no_errors("depth changed while an editor is open")
            assert table._pending is not None, (
                "a rebuild landed while an editor was open instead of being held")
            assert table.table.item(r, _COL_DEPTH).text() == before, (
                "the cells were rewritten under the open editor, which would have taken "
                "the editor and anything half-typed in it")
        finally:
            table.table.state = original_state

        table._drain_pending()
        assert table._pending is None, "the held rows were never drawn"
        assert table.table.item(r, _COL_DEPTH).text().startswith("0.20"), (
            f"the held rows were drawn but did not carry the change "
            f"({table.table.item(r, _COL_DEPTH).text()!r})")


# ---------------------------------------------------------------------------
# One sizing rule: the width a sill needs is solved at the head the sill can pass
# ---------------------------------------------------------------------------

def _shallow_swale(h, row_index=52, depth_m=0.05):
    """A swale carrying a sill deliberately shallower than its type's design head.

    0.05 m against a swale's 0.15 m policy head, so the sill is what limits the flow
    depth and the width has to be solved against it. Everything in this group needs that
    condition and none of it is interesting without it.
    """
    ew = h.add_earthwork("swale", geometry=line_across_valley(row=row_index))
    row = _review_row(h, ew)
    h.panel.set_spillway_depth_requested.emit(row["index"], depth_m)
    return ew, _review_row(h, ew)


def _dialog_for(h, index):
    """Open the properties dialog on *index*, capture it, and reject it.

    Rejecting rather than accepting: this reads what the dialog *says*, and an accepted
    dialog would write its own spillway back and destroy the state under test. The draw
    path's dialog is deliberately not used -- it is built with `_provisional_peak_flow`,
    a feature's own catchment before any routing, so its width would differ from the
    review's for reasons that have nothing to do with the sizing rule.
    """
    from terrainflow_assessment.earthwork_properties_dialog import (
        EarthworkPropertiesDialog,
    )

    seen = []
    original = EarthworkPropertiesDialog.exec
    EarthworkPropertiesDialog.exec = lambda self: (seen.append(self), 0)[1]
    try:
        h.plugin._earthworks.edit_selected_earthwork(index=index)
    finally:
        EarthworkPropertiesDialog.exec = original
    assert seen, "the properties dialog was never constructed"
    return seen[0]


def _metres(text):
    """The leading figure out of a width label such as '2.01 m - at the 0.05 m ...'."""
    head = text.strip().split(" m")[0]
    return float(head)


def check_the_table_and_the_dialog_size_one_weir(dem_path):
    """The headline. Two surfaces set a sill depth; they must cost it the same.

    The properties dialog has always solved the width at ``min(design head, sill depth)``
    -- water standing deeper than the notch is over the containing ground, not over the
    weir. The review row solved it at the design head, so on a shallow sill the dialog
    said "2.01 m" and the table said "1.4 m needed" for one structure. Now that a depth
    can be typed straight into the table, the two are the same control and cannot be
    allowed to disagree about what it costs.

    Compared to the centimetre rather than exactly: the dialog caps against a spin box
    rounded to two decimals and the row against an unrounded ``containment - crest``, so
    bit-equality is not achievable by construction and asserting it would be a flaky
    test rather than a stricter one.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, row = _shallow_swale(h)
        assert row["required_width_m"], "the fixture sill was never sized"
        assert row["sizing_head_m"] < row["target_head_m"], (
            f"the fixture sill is not shallow enough to limit anything "
            f"(sizing {row['sizing_head_m']}, target {row['target_head_m']})")

        dlg = _dialog_for(h, row["index"])
        h.assert_no_errors("properties dialog on a shallow sill")
        shown = _metres(dlg.lbl_spillway_width.text())
        assert abs(shown - row["required_width_m"]) <= 0.02, (
            f"the dialog quotes {shown:.2f} m and the review row {row['required_width_m']:.2f} m "
            f"for one sill at one depth -- the two surfaces are solving different equations")


def check_a_shallower_sill_needs_a_wider_weir(dem_path):
    """The depth control moves the one number it should move.

    Sizing the width off the design head left the requirement frozen: the same figure at
    a 1.00 m sill and at a 0.05 m one, when the shallow notch has a fraction of the depth
    to pass the same flow through.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, deep = _shallow_swale(h, depth_m=0.40)
        h.panel.set_spillway_depth_requested.emit(deep["index"], 0.05)
        shallow = _review_row(h, ew)

        assert deep["required_width_m"] and shallow["required_width_m"]
        assert shallow["required_width_m"] > deep["required_width_m"], (
            f"taking the sill from 0.40 m to 0.05 m left the requirement at "
            f"{shallow['required_width_m']:.2f} m (was {deep['required_width_m']:.2f} m)")
        assert deep["sizing_head_m"] == deep["target_head_m"], (
            "a sill deeper than the design head must not be capped at all")


def check_the_sill_cap_moves_the_width_and_nothing_else(dem_path):
    """The line the cap must not cross.

    Only the width is sized against the sill. The head, the freeboard reading and the
    validity sentences all stay on the depth the type designs for, so a sill too shallow
    for its storm still reads as one. Capping them too would improve the freeboard on
    exactly the sills that are worst -- a bad design redefining its way into compliance,
    silently, and in the direction that looks like an improvement.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, deep = _shallow_swale(h, depth_m=0.40)
        head_before = ew.spillway.head_m
        h.panel.set_spillway_depth_requested.emit(deep["index"], 0.05)
        shallow = _review_row(h, ew)

        assert ew.spillway.head_m == head_before, (
            "the stored design head moved when the sill was made shallower")
        assert shallow["target_head_m"] == deep["target_head_m"], (
            "the target head followed the sill depth")
        assert shallow["actual_head_m"] == deep["actual_head_m"], (
            f"the achieved head followed the sill depth "
            f"({deep['actual_head_m']} -> {shallow['actual_head_m']}) -- on an auto width "
            f"it is the design head by construction and has nothing to follow")
        assert abs(shallow["freeboard_m"]
                   - (shallow["sill_depth_m"] - shallow["actual_head_m"])) < 1e-9, (
            "freeboard is no longer what the notch has left once the flow has run its "
            "depth, which is the reading that says the sill is too shallow")
        assert shallow["freeboard_m"] < 0, (
            "a 0.05 m sill on a swale wanting 0.15 m of head plus freeboard must read "
            "negative -- that is the whole point of holding the head fixed")


def check_the_shortfall_warning_quotes_the_reviews_figure(dem_path):
    """The message bar and the list must not cost one sill two ways.

    `_check_spillway_capacity` fires in the same recompute that builds the review, one
    line after it. Sizing its own requirement at the design head put "needs 1.40 m" in
    the message bar while the list beside it said "needs 2.01 m" -- the divergence this
    channel exists to report, appearing inside it.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, row = _shallow_swale(h)
        # Commit the width: the warning skips auto sills, on the grounds that they track
        # the requirement by definition.
        ew.spillway.width_auto = False
        ew.spillway.width_m = 0.5
        h.iface.messageBar().messages.clear()
        row = _review_row(h, ew)
        h.plugin._earthworks._check_spillway_capacity()

        pushed = [m for m in h.iface.messageBar().messages
                  if "needs" in m[2] and ew.name in m[2]]
        assert pushed, (
            f"a 0.5 m weir on a sill needing {row['required_width_m']:.2f} m was not "
            f"reported at all: {h.iface.messageBar().messages}")
        text = pushed[-1][2]
        assert f"needs {row['required_width_m']:.2f} m" in text, (
            f"the warning quotes a different requirement from the list "
            f"(list says {row['required_width_m']:.2f} m): {text}")
        assert "this sill can pass" in text, (
            f"the warning names a head it did not size against: {text}")


def check_the_shortfall_warning_does_not_nag(dem_path):
    """It reports a change, so it must stop once it has been said.

    This runs on every settled recompute -- every add, edit, storm change and typed
    depth. Re-pushing the same features every time turns a notification into wallpaper,
    and a message bar that repaints on every edit is not a warning.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, _row = _shallow_swale(h)
        ew.spillway.width_auto = False
        ew.spillway.width_m = 0.5
        _review_row(h, ew)

        controller = h.plugin._earthworks
        controller._check_spillway_capacity()
        h.iface.messageBar().messages.clear()
        controller._check_spillway_capacity()
        assert not h.iface.messageBar().messages, (
            f"the same shortfall was pushed again with nothing changed: "
            f"{h.iface.messageBar().messages}")

        # A feature joining the set is news again.
        controller._short_spillways_reported = frozenset()
        controller._check_spillway_capacity()
        assert h.iface.messageBar().messages, (
            "a shortfall that has not been reported yet was suppressed")


def check_a_sill_with_no_depth_says_so_in_the_width_column(dem_path):
    """The most undersized sill possible must not render as a faint dash.

    At zero sill depth the weir equation has no answer -- ``L = Q / (C*H^1.5)`` diverges
    -- so `calculate_spillway_width` returns 0.0 and the row's requirement is None. Both
    of `spillway_validity`'s width checks are then skipped, because there is no
    requirement for them to test against, which leaves this cell as the only thing that
    can name the condition.
    """
    from terrainflow_assessment.qgis.widgets.spillway_table import SpillwayTable

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, _row = _shallow_swale(h, depth_m=0.40)
        h.panel.set_spillway_depth_requested.emit(_row["index"], 0.0)
        row = _review_row(h, ew)

        assert row["sizing_head_m"] == 0.0, row["sizing_head_m"]
        assert row["required_width_m"] is None, (
            "a zero-head weir has no width that passes anything; 0.0 must not reach the "
            "row as a designed figure")
        assert ew.spillway is not None, "a depth of zero is a depth, not a deletion"

        table = SpillwayTable()
        try:
            table.set_rows([row])
            text = table._width_cell(row, False)[0]
        finally:
            table.deleteLater()
        assert "no depth" in text, (
            f"the width cell reads {text!r} on a sill with nothing to spill through, "
            f"which is indistinguishable from a feature that simply has no flow")


def check_an_unbuildable_width_is_not_adopted(dem_path):
    """A requirement wider than the feature must not become the width that gets burned.

    Solving at the sill depth makes the requirement grow without bound as the notch gets
    shallow -- 2.01 m at 0.10 m, 22.43 m at 0.02 m, 63.45 m at 0.01 m -- and nothing
    downstream clamps it: `_scaled_bar` re-cuts the sill bar to whatever width it is
    handed. Until this change the figure was contained only by the uncapped auto-width
    pass overwriting it, and that overwrite is exactly what is being removed.

    Refused rather than clamped: the requirement is still reported in full and
    `spillway_validity` still says in words that the feature cannot carry it, while the
    stored width stays the last one that could actually be built.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, row = _shallow_swale(h, depth_m=0.40)
        buildable = ew.spillway.width_m
        assert buildable, "the fixture sill never got a width to keep"

        h.panel.set_spillway_depth_requested.emit(row["index"], 0.005)
        row = _review_row(h, ew)

        assert row["required_width_m"] > (ew.length_m or 0.0), (
            f"the fixture is not extreme enough: it needs "
            f"{row['required_width_m']:.2f} m on a {ew.length_m:.1f} m feature")
        assert abs(ew.spillway.width_m - buildable) < 1e-9, (
            f"a width of {ew.spillway.width_m:.2f} m was adopted on a "
            f"{ew.length_m:.1f} m feature -- the burn would cut the notch straight "
            f"through the ground holding the water in")
        assert any("cannot pass its own" in p or "long" in p for p in row["problems"]), (
            f"nothing told the user the weir does not fit: {row['problems']}")


def check_the_stored_width_matches_the_one_on_screen(dem_path):
    """The Width column, the map label, the summary and the burn read one number.

    `_refresh_auto_spillway_widths` runs earlier in the same recompute and does no DEM
    work, so it caps against whatever the previous settled build measured. Without the
    write-back that follows the review, the stored width lags the displayed one by a
    recompute -- the column right and everything drawn from the model wrong, which is
    this change's own divergence relocated rather than fixed.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, row = _shallow_swale(h, depth_m=0.40)
        assert row["width_auto"], "the fixture sill is not on auto"
        assert abs(ew.spillway.width_m - row["built_width_m"]) < 1e-9, (
            f"stored {ew.spillway.width_m} vs displayed {row['built_width_m']}")

        # The case that actually needs the write-back: read the state the typed depth
        # left behind, with **no** further recompute. `_refresh_auto_spillway_widths`
        # runs first inside that one recompute and caps against the depth the *previous*
        # build measured, so it writes the width the old sill needed; only the write-back
        # that follows the review corrects it. Calling `_review_row` here instead would
        # run a second recompute -- by which time the cache is fresh and the lag has
        # closed itself, which is how this check first passed with the write-back
        # deleted.
        h.panel.set_spillway_depth_requested.emit(row["index"], 0.08)
        row = next(r for r in h.state.spillway_rows if r["id"] == ew.id)
        assert abs(ew.spillway.width_m - row["built_width_m"]) < 1e-9, (
            f"after a typed depth the model holds {ew.spillway.width_m} while the list "
            f"shows {row['built_width_m']} -- the write-back did not settle it, so the "
            f"map label, the summary and the burn are a recompute behind the column")
        assert abs(ew.spillway.width_required_m - row["required_width_m"]) < 1e-9


def check_the_review_survives_features_it_cannot_size(dem_path):
    """Undesigned, disabled and flow-less features all still build a row.

    `_spillway_row` exists to serve features with no `Spillway` at all -- showing what
    one would need is the whole of "features auto-size their spillways as you add them".
    Those rows return before any datum is read, so every figure derived from a datum is
    None on them, and the freeboard line reaching for one unguarded is a TypeError.
    `_build_spillway_rows` swallows to a console print, so the symptom is not a crash:
    it is the entire review going stale, silently, for the whole design.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        undesigned = h.add_earthwork("swale", geometry=line_across_valley(row=40))
        disabled = h.add_earthwork("basin", geometry=line_across_valley(
            row=110, half_width_m=15.0))
        disabled.enabled = False
        designed, _row = _shallow_swale(h, row_index=70)

        h.panel.analysis_inputs_changed.emit()
        h.plugin._earthworks._build_spillway_rows()
        h.assert_no_errors("review over unsizeable features")

        rows = {r["id"]: r for r in h.panel._spillway_table._rows}
        assert undesigned.id in rows and disabled.id in rows and designed.id in rows, (
            f"the review dropped features it could not size: {sorted(rows)}")
        assert rows[disabled.id]["state"] == "disabled"
        assert rows[undesigned.id]["designed"] is False
        assert rows[undesigned.id]["required_width_m"], (
            "an undesigned feature must still be sized")
        for row in rows.values():
            assert "sizing_head_m" in row, "the row contract lost sizing_head_m"


def check_a_feature_that_cannot_be_measured_caps_nothing(dem_path):
    """The cached depth must never outlive the crest it was measured against.

    Two width solves cannot call `_spillway_datums` for themselves -- one runs per drag
    frame and the other has no footprint mask -- so they read a measurement cached on the
    feature. A cache that survived a feature being switched off, or its spillway being
    replaced wholesale by the properties dialog, would go on capping the width against a
    sill that no longer exists. `None` means "not measured", which correctly means "do
    not cap", and that is the same thing the review row does with the same absent datum.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew, row = _shallow_swale(h, depth_m=0.40)
        assert ew.measured_sill_depth_m is not None, "nothing was ever measured"

        ew.enabled = False
        _rows = h.plugin._earthworks._build_spillway_rows()
        assert ew.measured_sill_depth_m is None, (
            "a disabled feature kept a measured depth, which the auto-width pass would "
            "go on sizing against")

        ew.enabled = True
        _review_row(h, ew)
        assert ew.measured_sill_depth_m is not None, "it was never measured again"

        h.plugin._earthworks._clear_measured_levels(ew)
        assert ew.measured_sill_depth_m is None, (
            "the measured depth is not cleared with the other three figures measured "
            "off the same flood, so it can outlive them")


def check_a_column_widens_for_content_it_has_never_held(dem_path):
    """A column that gains a number for the first time must not elide it.

    `set_rows` re-measures the columns only when the row count changed, so that they do
    not shimmy under the cursor as a dash becomes "0.60 m" while the user is typing down
    the Sill depth column. The cost of that guard is that a column can never *grow*
    either -- and this table opens on a design whose features have no spillway yet, so
    Sill is laid out for an em dash. Typing the first depth fills it with an elevation on
    a rebuild with the same row count, which is exactly the rebuild the guard skips: the
    figure the column exists to show renders as "84...", indefinitely.

    Caught by looking at a screenshot rather than by any assertion, which is what the
    visual checks are for. Pinned here because it is a behaviour, not a rendering.
    """
    from terrainflow_assessment.qgis.widgets.spillway_table import _HEADERS, SpillwayTable

    col_sill = _HEADERS.index("Sill")

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=52))
        undesigned = _review_row(h, ew)
        assert undesigned["crest_elevation"] is None, "the fixture starts designed"

        table = SpillwayTable()
        try:
            table.window().show()
            table.show()
            table.resize(660, 320)
            table.set_rows([undesigned])
            narrow = table.table.columnWidth(col_sill)

            h.panel.set_spillway_depth_requested.emit(undesigned["index"], 0.40)
            designed = _review_row(h, ew)
            assert designed["crest_elevation"] is not None

            # The same row count, so the full re-measure is skipped -- which is the whole
            # point of the fixture.
            table.set_rows([designed])
            assert table.table.rowCount() == 1
            grown = table.table.columnWidth(col_sill)
            assert grown >= table.table.sizeHintForColumn(col_sill), (
                f"the Sill column stayed {grown}px against a content width of "
                f"{table.table.sizeHintForColumn(col_sill)}px, so the crest elevation is "
                f"rendered elided")
            assert grown > narrow, (
                f"the column did not widen at all ({narrow}px -> {grown}px) when it went "
                f"from an em dash to an elevation")

            # And it must not give the space back on the way down, which is what the
            # anti-shimmy guard is protecting.
            table.set_rows([undesigned])
            assert table.table.columnWidth(col_sill) == grown, (
                "the column narrowed again when the elevation went away, which is the "
                "shimmy the row-count guard exists to prevent")
        finally:
            table.deleteLater()


# ---------------------------------------------------------------------------
# The user's standard earthwork dimensions
# ---------------------------------------------------------------------------

def _standard_key():
    from terrainflow_assessment.qgis.controllers.earthworks import EarthworksController
    return EarthworksController._EARTHWORK_DEFAULTS_KEY


def _set_standard(text):
    """Write the stored standard directly, returning the previous value.

    QgsSettings is process-wide. The harness points the org/app name at
    TerrainFlowTests so a run cannot touch the real QGIS profile, but that does
    nothing to isolate one check from the next in the same subprocess — so every
    check here restores what it found in a finally, or every later check in this
    module draws swales at the test's dimensions.
    """
    from qgis.core import QgsSettings
    settings = QgsSettings()
    previous = settings.value(_standard_key(), "")
    settings.setValue(_standard_key(), text)
    return previous


def check_a_drawn_swale_uses_the_users_standard(dem_path):
    """A stored standard must reach the feature the user ends up with."""
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    original_exec = EarthworkPropertiesDialog.exec
    EarthworkPropertiesDialog.exec = lambda self: 1
    previous = _set_standard(
        '{"swale": {"depth": 0.35, "top_width_m": 1.6, "bottom_width_m": 0.9}}')
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.assert_no_errors("baseline run")

            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("geometry drawn with a standard set")

            ew = h.state.earthwork_manager.get(0)
            assert abs(ew.depth - 0.35) < 1e-6, f"depth was {ew.depth}, expected 0.35"
            assert abs(ew.top_width_m - 1.6) < 1e-6, f"top width was {ew.top_width_m}"
            assert abs(ew.bottom_width_m - 0.9) < 1e-6, (
                f"bottom width was {ew.bottom_width_m}, expected 0.9"
            )
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        _set_standard(previous)


def check_the_properties_dialog_opens_at_the_standard(dem_path):
    """The dialog must show the standard, and offer to keep it.

    Also pins that the dialog seeds from the constructed Earthwork rather than from
    the registry directly: if it read ``_cfg.default_*`` the spin boxes would show
    the shipped 0.50 / 2.00 here.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    seen = {}
    original_exec = EarthworkPropertiesDialog.exec

    def capture(dialog):
        seen["depth"] = dialog.spin_depth.value()
        seen["width"] = dialog.spin_width.value()
        seen["bottom"] = dialog.spin_bottom_width.value()
        seen["toggle_present"] = dialog.chk_save_standard is not None
        seen["toggle_on"] = dialog.get_save_as_standard()
        seen["label"] = dialog.chk_save_standard.text()
        # Departing from the standard must clear the toggle, so changing the size is
        # never a silent rewrite of what the user built to last week.
        dialog.spin_depth.setValue(0.80)
        seen["toggle_after_change"] = dialog.get_save_as_standard()
        return 0  # rejected — this check is only about what the dialog offered

    EarthworkPropertiesDialog.exec = capture
    previous = _set_standard(
        '{"swale": {"depth": 0.35, "top_width_m": 1.6, "bottom_width_m": 0.9}}')
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("properties dialog with a standard set")
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        _set_standard(previous)

    assert seen, "the properties dialog never opened"
    assert abs(seen["depth"] - 0.35) < 1e-6, "dialog depth was {}".format(seen["depth"])
    assert abs(seen["width"] - 1.6) < 1e-6, "dialog width was {}".format(seen["width"])
    assert abs(seen["bottom"] - 0.9) < 1e-6, "dialog bottom was {}".format(seen["bottom"])
    assert seen["toggle_present"], "a drawn feature must offer the standard toggle"
    assert seen["toggle_on"], "matching the standard should leave the toggle ticked"
    assert not seen["toggle_after_change"], (
        "departing from the standard must clear the toggle"
    )
    assert "1.60" in seen["label"] and "0.35" in seen["label"], (
        "the toggle should name the current standard, got {!r}".format(seen["label"])
    )


def check_the_toggle_is_on_for_a_first_feature(dem_path):
    """With no standard stored the toggle starts ticked, so the first one sets it."""
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    seen = {}
    original_exec = EarthworkPropertiesDialog.exec

    def capture(dialog):
        seen["on"] = dialog.get_save_as_standard()
        seen["label"] = dialog.chk_save_standard.text()
        return 0

    EarthworkPropertiesDialog.exec = capture
    previous = _set_standard("")
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("properties dialog with no standard set")
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        _set_standard(previous)

    assert seen.get("on"), "the first feature drawn should offer to set the standard"
    assert "now" not in seen["label"], (
        "with no standard stored the label should not quote one: {!r}".format(seen["label"])
    )


def check_saving_a_standard_round_trips(dem_path):
    """Accepting with the toggle on must persist, and be read back on next load."""
    from qgis.core import QgsSettings

    from terrainflow_assessment.core.registry.earthwork_defaults import decode
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    original_exec = EarthworkPropertiesDialog.exec

    def accept_with_new_size(dialog):
        dialog.spin_depth.setValue(0.42)
        dialog.spin_width.setValue(1.80)
        dialog.spin_bottom_width.setValue(1.10)
        dialog.chk_save_standard.setChecked(True)
        return 1

    EarthworkPropertiesDialog.exec = accept_with_new_size
    previous = _set_standard("")
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("accepting with save-as-standard ticked")

            stored = decode(QgsSettings().value(_standard_key(), ""))
            assert "swale" in stored, f"nothing was stored: {stored!r}"
            assert abs(stored["swale"].depth - 0.42) < 1e-6
            assert abs(stored["swale"].top_width_m - 1.80) < 1e-6
            assert abs(stored["swale"].bottom_width_m - 1.10) < 1e-6

            # A reload must read the same thing back.
            h.plugin._earthworks.load_earthwork_defaults()
            dims = h.plugin._earthworks._resolved_dims("swale")
            assert abs(dims.depth - 0.42) < 1e-6, f"reloaded depth was {dims.depth}"
    finally:
        EarthworkPropertiesDialog.exec = original_exec
        _set_standard(previous)


def check_a_saved_design_ignores_the_users_standard(dem_path):
    """A stored design must reload at its own size, not at the reader's standard.

    The guarantee the constructor parameter buys: ``from_dict`` passes no dims, so a
    payload written before a field existed comes back at the shipped default rather
    than at whatever the person opening it happens to prefer.
    """
    import json

    previous = _set_standard(
        '{"swale": {"depth": 0.35, "top_width_m": 1.6, "bottom_width_m": 0.9}}')
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.add_earthwork("swale")
            payload = json.loads(h.state.earthwork_manager.to_json())
            items = payload if isinstance(payload, list) else payload.get("earthworks", [])
            # Strip the dimensions, as a design written by an older schema would be.
            for item in items:
                for field in ("depth", "top_width_m", "bottom_width_m"):
                    item.pop(field, None)

            h.plugin._earthworks.restore_earthworks_from_json(json.dumps(payload))
            h.assert_no_errors("restoring a design with no stored dimensions")

            ew = h.state.earthwork_manager.get(0)
            assert abs(ew.depth - 0.5) < 1e-6, (
                f"a restored swale loaded at {ew.depth} — the reader's standard leaked into a "
                "saved design"
            )
    finally:
        _set_standard(previous)


def check_the_provisional_catchment_uses_the_standard_width(dem_path):
    """The inflow shown on screen must describe the swale actually being built."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        ew_ctl = h.plugin._earthworks
        geometry = line_across_valley()
        narrow = ew_ctl._provisional_catchment("swale", geometry, top_width_m=0.5)
        wide = ew_ctl._provisional_catchment("swale", geometry, top_width_m=8.0)
        assert narrow != wide, (
            "footprint width does not affect the provisional catchment, so this check "
            "cannot prove the two call sites agree"
        )

        # What _on_geometry_drawn passes must be what the constructor seeds.
        dims = ew_ctl._resolved_dims("swale")
        assert (ew_ctl._provisional_catchment("swale", geometry)
                == ew_ctl._provisional_catchment("swale", geometry,
                                                 top_width_m=dims.top_width_m)), (
            "the provisional catchment and the seeded feature disagree on width"
        )


def check_a_basin_ignores_a_width_standard(dem_path):
    """A basin's footprint is the drawn polygon, so a stored width is not a preference."""
    from terrainflow_assessment.core.registry.earthwork_defaults import decode

    previous = _set_standard('{"basin": {"depth": 2.2, "top_width_m": 3.0}}')
    try:
        stored = decode('{"basin": {"depth": 2.2, "top_width_m": 3.0}}')
        assert stored["basin"].top_width_m is None, (
            "a basin has no width row, so a width entry must be dropped"
        )
        assert abs(stored["basin"].depth - 2.2) < 1e-6
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            dims = h.plugin._earthworks._resolved_dims("basin")
            assert abs(dims.depth - 2.2) < 1e-6, f"basin depth was {dims.depth}"
            assert abs(dims.top_width_m - 0.0) < 1e-6, (
                f"basin top width should stay 0.0, was {dims.top_width_m}"
            )
    finally:
        _set_standard(previous)


def check_a_dem_swap_drops_the_measured_containment_level(dem_path):
    """Per-feature terrain measurements live on the `Earthwork` objects, not on
    `PluginState`, so `invalidate_results()` never reached them — it clears state
    attributes and nothing else. The design-file Open path re-measured afterwards
    and was covered by accident; the DEM picker did not and was not.

    The consequence is not merely a stale number. `_spillway_datums` *prefers*
    `terrain_spill_level_m` as the containment ceiling, so after a swap the lip and
    the invert come off the new terrain while the ceiling comes off the old one —
    and the dialog labels that mixture "measured".

    The clip overlaps the fixture exactly, so a level that survives is surviving
    because nothing cleared it, not because the two terrains agree. Sits beside the
    in-session checks above, which pin the opposite: a crest move must *not* discard
    these.
    """
    from qgis.core import QgsProject, QgsRasterLayer

    from _harness import cropped_dem
    from terrainflow_assessment.modules.earthwork_design import (
        CONTAINMENT_LIP,
        CONTAINMENT_MEASURED,
    )

    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        # A swale with a companion berm, because the containment preference only
        # bites on a feature that holds water *above* natural ground: a plain basin
        # spills at its own ring minimum, so the measured level equals the lip and
        # `_spillway_datums` correctly reports the lip either way. The bermed swale
        # is the case the method's docstring is written about.
        ew = h.add_earthwork("swale", geometry=line_across_valley())
        ew.companion_berm = True
        controller._refresh_terrain_capacity(ew, quiet=True)
        h.assert_no_errors("terrain capacity")

        assert ew.terrain_spill_level_m is not None, (
            "the fixture was never measured, so the claim cannot be tested")
        assert ew.terrain_capacity_m3, "no measured capacity either"
        before = controller._spillway_datums(
            ew.geometry, ew.type, top_width_m=ew.width, depth=ew.depth, ew=ew)
        assert before[3] == CONTAINMENT_MEASURED, (
            f"the containment source is {before[3]!r}, not the measured level — "
            f"the preference under test is not being exercised")

        clip = cropped_dem(dem_path, os.path.join(h.state.output_dir, "clipped_dem.tif"))
        layer = QgsRasterLayer(clip, "Clipped DEM")
        assert layer.isValid(), "the clipped DEM did not load"
        QgsProject.instance().addMapLayer(layer)
        h.panel.dem_changed.emit(layer)
        h.assert_no_errors("DEM swap")

        assert h.state.dem_info.width == 200, (
            "the picker did not adopt the clip, so no grid move happened")
        assert ew.terrain_spill_level_m is None, (
            "a spill level measured on the previous terrain survived the swap and is "
            "still the containment ceiling")
        assert ew.terrain_capacity_m3 is None, (
            "the measured capacity survived the swap")
        assert ew.excavation_m3 is None, "the measured excavation survived the swap"
        assert ew.stage_storage is None, "the stage-storage curve survived the swap"

        after = controller._spillway_datums(
            ew.geometry, ew.type, top_width_m=ew.width, depth=ew.depth, ew=ew)
        assert after[3] == CONTAINMENT_LIP, (
            f"the containment source is {after[3]!r} after the terrain changed; with "
            f"nothing measured against the new grid it has to fall back to the lip")


def check_two_barriers_sharing_a_name_are_overtopped_apart(dem_path):
    """The overtopping layer keyed its barriers by display *name*.

    `key = ew.id` was already being computed two lines above for the mask lookups
    while `barriers` and `by_name` went on using `ew.name`. The default name
    counter reproduces a deleted feature's name — delete "Dam 1" and draw another
    and you have two "Dam 2"s — so the freeboard advisory reported `has_spillway`
    off whichever one the dict happened to keep, and the "(full)" layer's
    `spillway` attribute was wrong on that row. One of these dams has a spillway
    and one does not, which is the difference the column exists to show.
    """
    from qgis.core import QgsProject

    from terrainflow_assessment.modules.earthwork_design import Spillway

    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()

        dams = []
        for row in (60, 110):
            geom = line_across_valley(row=row)
            ew = h.add_earthwork("dam", geometry=geom, name="Dam 2")
            ew.crest_elevation = float(
                controller._feature_elevation(geom) or 0) + 2.0
            ew.key_into_banks = True
            controller._on_vertex_edit_finished(
                len(h.state.earthwork_manager) - 1, geom)
            dams.append(ew)

        assert dams[0].name == dams[1].name == "Dam 2", "the fixture is not the case"
        assert dams[0].id != dams[1].id, "two features shared an id, which is a worse bug"

        # One of them gets a spillway; the other must not be credited with it.
        dams[1].spillway = Spillway()

        h.panel.analysis_inputs_changed.emit()
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")

        layers = [lyr for lyr in QgsProject.instance().mapLayers().values()
                  if "Overtopping (full)" in lyr.name()]
        assert layers, "no overtopping layer was built"
        rows = list(layers[0].getFeatures())
        assert len(rows) == 2, (
            f"two barriers, {len(rows)} band(s) — one overwrote the other's key")
        assert sorted(f["spillway"] for f in rows) == ["designed", "none"], (
            f"both bands report the same spillway state "
            f"{[f['spillway'] for f in rows]} — the layer is reading one feature "
            f"for both rows")
        assert {f["feature"] for f in rows} == {"Dam 2"}, (
            "the id leaked into the layer's feature column instead of the label")


def check_the_ponding_query_reports_the_storm_fill_verdict(dem_path):
    """`PondingQueryTool` has always taken `earthwork_inflows` and documented the
    shape it wants; the one construction site passed none, so
    `_find_nearest_inflow` returned -1.0 on every click, `fill_fraction` was
    always -1.0, and the design-storm block in `_on_ponding_selected` never
    rendered. Silent in both directions — no verdict, and no word that there was
    not going to be one.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        ew = h.add_earthwork("basin", geometry=line_across_valley(
            row=60, half_width_m=15.0))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("design tier")

        controller.activate_ponding_query()
        tool = h.canvas.mapTool()
        assert getattr(tool, "earthwork_inflows", None), (
            "the tool was built with no inflows, so the verdict cannot be reached")

        inflow, name = tool._find_nearest_inflow(*_centroid_xy(ew))
        assert name == ew.name, (
            f"the nearest feature came back as {name!r}, not {ew.name!r}")
        assert inflow >= 0.0, (
            f"no design-storm inflow for the nearest feature: {inflow}")


def _centroid_xy(ew):
    """(x, y) of an earthwork's centroid, for pointing the ponding tool at it."""
    point = ew.geometry.centroid().asPoint()
    return point.x(), point.y()


def check_a_drag_frame_does_not_rebuild_the_panel(dem_path):
    """M-5a / G-7. Six things sat above `_recompute_live_assessment`'s own
    `if geometry_settled:` guard, whose comment reads "doing that per feature at
    12.5 Hz is exactly the cost this method's docstring promises to avoid":
    `set_network` (which deletes every child widget and reconstructs a
    `_NodeCard` and a connector per feature), `_refresh_connections_layer` (a map
    layer rebuild), the area subtotals, the coverage readout, the report summary,
    and `refresh_stress_points_layer` — a full-raster pass plus up to 20,000 GEOS
    `project` calls per linear feature.

    Measured on the real fixture at 12 features: **526 ms per drag frame against
    the 80 ms a 12.5 Hz throttle allows**, and 17.9 ms after. `tests_qgis/probes/
    p_drag_cost.py` is that measurement.

    The text readouts stay live, and that is the point of the split rather than
    an oversight: a designer dragging a vertex is watching those numbers move.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        for row in (40, 70, 100):
            h.add_earthwork("swale", geometry=line_across_valley(row=row))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("design tier")

        called = []
        for name in ("refresh_stress_points_layer", "_refresh_connections_layer",
                     "_build_spillway_rows"):
            original = getattr(controller, name)

            def spy(*a, _n=name, _o=original, **kw):
                called.append(_n)
                return _o(*a, **kw)

            setattr(controller, name, spy)

        live = []
        original_live = h.panel.set_live_assessment

        def live_spy(*a, **kw):
            live.append(1)
            return original_live(*a, **kw)

        h.panel.set_live_assessment = live_spy
        try:
            called.clear()
            live.clear()
            controller._recompute_live_assessment(geometry_settled=False)
            assert not called, (
                f"a drag frame still ran the heavy rebuilds: {sorted(set(called))}")
            assert live, (
                "the live readout stopped updating mid-drag — the text is the half "
                "that has to stay live")

            called.clear()
            controller._recompute_live_assessment(geometry_settled=True)
            assert set(called) == {"refresh_stress_points_layer",
                                   "_refresh_connections_layer",
                                   "_build_spillway_rows"}, (
                f"a settled edit skipped work it must still do: {sorted(set(called))}")
        finally:
            h.panel.set_live_assessment = original_live


def check_a_drag_frame_does_not_re_sum_the_domain_mask(dem_path):
    """Q-12. `int(self._state.flow_domain_mask.sum())` reduced a 2.85 M-element
    mask on every frame, against the `meta["domain_cells"]` cache whose own
    comment names this method as the reason it exists — and which
    `compute_catchment_coverage` was already reading.
    """
    with PluginHarness(dem_path) as h:
        controller = h.plugin._earthworks
        h.run_baseline()
        h.add_earthwork("swale", geometry=line_across_valley(row=60))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("design tier")

        meta = h.state.flow_grid_meta
        if meta is None or meta.get("domain_cells") is None:
            return      # no cached count on this fixture; nothing to read instead

        # Poison the mask: if the frame still reduces it, the answer changes.
        cached = meta["domain_cells"]
        meta["domain_cells"] = cached + 1_000_000
        try:
            controller._recompute_live_assessment(geometry_settled=False)
            balance = h.state.balance
            assert balance is not None, "no balance was produced"
            poisoned = balance.total_inflow_m3
        finally:
            meta["domain_cells"] = cached

        controller._recompute_live_assessment(geometry_settled=False)
        honest = h.state.balance.total_inflow_m3
        assert poisoned != honest, (
            "the cached domain count made no difference to the frame, so the mask "
            "is still being re-summed")


def check_the_dam_dialog_reads_the_dem_once(dem_path):
    """Q-11. `_calc_dam_wall_metrics` did `src.read(1)` — the whole band — to sample
    about sixty cells along the wall, and it is wired to `spin_crest_elev`,
    `chk_key_banks` and `spin_width`. Holding the up arrow on Crest elevation
    auto-repeats at ~30 Hz, on the GUI thread, inside a modal: 15.3 ms of the 18.4 ms
    the dialog spends answering each tick, on the owner's 1139x1016 DEM.

    The dialog is modal, short-lived and its `_dem_path` never changes, so the band
    can simply be held. Ten ticks, one open.
    """
    import rasterio

    from terrainflow_assessment.earthwork_properties_dialog import (
        EarthworkPropertiesDialog,
    )

    seen = {}
    original_exec = EarthworkPropertiesDialog.exec
    real_open = rasterio.open

    def capture(dialog):
        # One tick first: the band is fetched when it is first needed, and what this
        # check is about is the *second* tick onwards.
        dialog.spin_crest_elev.setValue(dialog.spin_crest_elev.value() + 0.01)
        opens = {"n": 0}

        def counting_open(*args, **kwargs):
            opens["n"] += 1
            return real_open(*args, **kwargs)

        rasterio.open = counting_open
        try:
            base = dialog.spin_crest_elev.value()
            for i in range(10):
                dialog.spin_crest_elev.setValue(base + (i + 1) * 0.01)
        finally:
            rasterio.open = real_open
        seen["opens"] = opens["n"]
        seen["wall_volume"] = dialog.lbl_wall_volume.text()
        return 0

    EarthworkPropertiesDialog.exec = capture
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn(
                "dam", line_across_valley(row=70))
            h.assert_no_errors("dam properties dialog")
    finally:
        EarthworkPropertiesDialog.exec = original_exec

    assert seen, "the dam properties dialog never opened"
    assert seen["opens"] == 0, (
        f"ten crest-elevation ticks reopened the DEM {seen['opens']} times — the "
        f"band is being re-read per tick"
    )
    assert seen["wall_volume"] and seen["wall_volume"] != "—", (
        f"the wall-volume readout stopped working: {seen['wall_volume']!r}"
    )


def check_a_zero_nodata_dem_does_not_erase_the_dam_wall(dem_path):
    """Q-11, second half. `abs(ground - nodata) < 1.0` called any cell within a
    metre of the sentinel nodata. At -9999 that is harmless; on a DEM that declares
    `nodata=0` it silently discards every cell between -1 m and +1 m — real ground
    on any site surveyed to a local datum or to mean sea level. The wall then gets
    its height from whatever survived, or from nothing at all.

    Equality is what "is this cell nodata" means, so that is what it asks now.
    """
    import numpy as np
    import rasterio

    from terrainflow_assessment.earthwork_properties_dialog import (
        EarthworkPropertiesDialog,
    )

    with PluginHarness(dem_path, load_dem=False) as h:
        # A DEM declaring nodata=0, with real ground either side of it.
        with rasterio.open(dem_path) as src:
            profile = src.profile.copy()
            transform = src.transform
        ground = np.full((40, 40), 0.4, dtype="float32")   # 0.4 m above the datum
        profile.update(height=40, width=40, nodata=0.0, dtype="float32",
                       count=1, transform=transform)
        near_datum = os.path.join(
            h.state.output_dir or os.path.dirname(dem_path), "near_datum.tif")
        with rasterio.open(near_datum, "w", **profile) as dst:
            dst.write(ground, 1)

        from qgis.core import QgsGeometry, QgsPointXY

        x0, y0 = transform.c, transform.f
        geom = QgsGeometry.fromPolylineXY(
            [QgsPointXY(x0 + 5.0, y0 - 20.0), QgsPointXY(x0 + 35.0, y0 - 20.0)])

        dlg = EarthworkPropertiesDialog(
            ew_type="dam", geometry=geom, parent=h.main_window,
            dem_path=near_datum,
        )
        try:
            # Crest 3 m above ground that sits 0.4 m above a zero datum.
            max_h, wall_vol = dlg._calc_dam_wall_metrics(3.4, 2.0)
        finally:
            dlg.deleteLater()

    assert abs(max_h - 3.0) < 0.11, (
        f"ground 0.4 m above a zero nodata value was treated as nodata: the wall "
        f"measured {max_h} m against a 3.0 m crest height"
    )
    assert wall_vol > 0, "a wall standing 3 m over real ground priced at nothing"
