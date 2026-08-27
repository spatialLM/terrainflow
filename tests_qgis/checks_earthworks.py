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
