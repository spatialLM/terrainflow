"""
Visual checks — render the real panel, dialogs and map canvas, assert something
was drawn, and leave the PNGs in tests_qgis/_shots/ to be looked at.

These answer the question state assertions cannot: *did it draw?* A renderer that
builds without error but paints nothing, a layer added below an opaque raster, a
dock that collapses to zero width — all pass Tier 1 and fail here.
"""

from _harness import PluginHarness, line_across_valley
from _shots import assert_rendered, describe, save_canvas, save_widget
from qgis.PyQt.QtWidgets import QPushButton

PANEL_SIZE = (460, 1400)


def check_panel_renders(dem_path):
    """The whole panel draws — the stage stepper, scorecard and disclosure sections."""
    with PluginHarness(dem_path) as h:
        path = save_widget(h.panel, "panel_initial", size=PANEL_SIZE)
        assert_rendered(path, "panel (initial)", min_colours=12)


def check_panel_renders_after_baseline(dem_path):
    """Post-run panel state: verified chip, summary text, populated results."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        path = save_widget(h.panel, "panel_after_baseline", size=PANEL_SIZE)
        assert_rendered(path, "panel (after baseline)", min_colours=12)


def check_water_leaving_readout(dem_path):
    """The Baseline water readout: total vs shown-exits, plus water captured.

    Rendered on its own because the panel screenshot lands on the Design step after a
    run, so this box never appears there. The two figures must stay distinguishable —
    reporting only the exit-filtered subtotal is what made it read as a site total.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        lbl = h.panel._area_outflow_lbl
        # isHidden(), not isVisible(): the dock's window is not shown until
        # save_widget() shows it, so isVisible() is False for every widget here.
        assert not lbl.isHidden(), "water-leaving readout stayed hidden after baseline"
        text = lbl.text()
        assert "Water leaving" in text, text
        assert "in total" in text and "of which" in text, (
            f"expected both the total and the shown-exits figure, got: {text}")

        path = save_widget(lbl, "panel_water_leaving", size=(520, 150))
        assert_rendered(path, "water leaving readout", min_colours=3)


def check_panel_renders_with_design(dem_path):
    """Panel with an earthwork table populated and an assessment computed."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row, ew_type in ((40, "swale"), (60, "dam"), (80, "basin")):
            h.add_earthwork(ew_type, geometry=line_across_valley(row=row))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("live assessment")

        path = save_widget(h.panel, "panel_with_design", size=PANEL_SIZE)
        assert_rendered(path, "panel (with design)", min_colours=12)


def check_spillway_review_renders(dem_path):
    """The Spillways review with real rows, expanded.

    Ships collapsed, so the shot has to open it — a closed disclosure arrow renders
    perfectly well and says nothing about the table underneath it.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row in (40, 65, 90):
            h.add_earthwork("swale", geometry=line_across_valley(row=row))
        h.add_earthwork("basin", geometry=line_across_valley(row=110, half_width_m=15.0))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("spillway review render")

        assert h.panel._spillway_table._rows, "nothing to render — no rows were built"

        h.panel._show_stage("design")
        for header in h.panel.findChildren(QPushButton, "tfSectionHeader"):
            header.setChecked(True)

        path = save_widget(h.panel._spillway_table, "panel_spillway_review",
                           size=(460, 320))
        assert_rendered(path, "spillway review table", min_colours=6)


def check_contour_panel_renders(dem_path):
    """The Contours tab with results: ticked list, inflow-band legend, controls.

    The band legend prints the run's actual natural-breaks boundaries, so this is
    where you see whether they read as sensible numbers rather than just whether
    the widget exists.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.panel.find_segments_requested.emit()
        h.assert_no_errors("contour + segment analysis")

        h.panel._show_stage("analysis")
        # The Contour & Keypoint section ships collapsed; open it and give the
        # results room, or the shot is of a closed disclosure arrow.
        for header in h.panel.findChildren(QPushButton, "tfSectionHeader"):
            header.setChecked(True)

        path = save_widget(h.panel, "panel_contours", size=(460, 1700))
        assert_rendered(path, "panel (contour results)", min_colours=12)


def check_canvas_renders_dem(dem_path):
    """Baseline: the DEM alone must reach the canvas."""
    with PluginHarness(dem_path) as h:
        h.sync_canvas()
        path = save_canvas(h.canvas, "canvas_dem")
        assert_rendered(path, "canvas (DEM only)")


def check_canvas_renders_baseline_layers(dem_path):
    """Stream ramp, exit points and labels drawn over the DEM."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        h.sync_canvas()
        path = save_canvas(h.canvas, "canvas_baseline")
        assert_rendered(path, "canvas (baseline results)")


def check_canvas_renders_earthworks(dem_path):
    """Earthwork symbology — the true real-world-width rendering.

    A width expressed in metres that silently falls back to millimetres still
    produces a valid layer and a clean Tier 1 pass; here it shows up as a hairline.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row, ew_type in ((40, "swale"), (60, "dam"), (80, "diversion")):
            h.add_earthwork(ew_type, geometry=line_across_valley(row=row))
        h.plugin._earthworks._refresh_ew_layer()
        h.assert_no_errors("earthwork layer refresh")

        h.sync_canvas()
        path = save_canvas(h.canvas, "canvas_earthworks")
        assert_rendered(path, "canvas (earthworks)")


def check_canvas_renders_catchments(dem_path):
    """One colour per earthwork — the direct-catchment layer.

    add_earthwork() reaches straight into the manager, so nothing has labelled the
    cells yet and refresh_catchment_layer() would return early with no layer at
    all. Emitting analysis_inputs_changed is what a real edit does, and it is what
    calls recompute_catchments(). Without it this check passed on the DEM alone.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", geometry=line_across_valley(row=50))
        h.add_earthwork("swale", geometry=line_across_valley(row=80))
        h.panel.analysis_inputs_changed.emit()
        h.panel.toggle_catchment_layer_requested.emit(True)
        h.assert_no_errors("catchment layer")

        assert h.state.catchment_labels is not None, (
            "no cells were labelled — the catchment layer has nothing to draw"
        )
        layer_id = h.state.catchment_labels_layer_id
        assert layer_id, "toggling the catchment layer on created no layer"

        h.sync_canvas()
        assert any(lyr.id() == layer_id for lyr in h.canvas.layers()), (
            "the catchment layer is not on the canvas — this shot would be the DEM"
        )
        path = save_canvas(h.canvas, "canvas_catchments")
        assert_rendered(path, "canvas (catchments)")


def check_canvas_renders_slope_overlays(dem_path):
    """Slope classes plus hachures — the tapered stroke symbol over the class ramp."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.toggle_slope_class_requested.emit(True)
        h.panel.toggle_slope_vectors_requested.emit(True)
        h.assert_no_errors("slope overlays")

        h.sync_canvas()
        path = save_canvas(h.canvas, "canvas_slope")
        assert_rendered(path, "canvas (slope overlays)")


def check_canvas_renders_contours(dem_path):
    """Candidate contour swales, ranked and styled."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.panel.select_top5_contours_requested.emit()
        h.assert_no_errors("contour analysis")

        h.sync_canvas()
        path = save_canvas(h.canvas, "canvas_contours")
        assert_rendered(path, "canvas (contour swales)")


def check_each_baseline_layer_renders(dem_path):
    """Render each baseline result layer on its own.

    A combined canvas shot only proves *something* drew. One image per layer says
    which layer drew what — so an invisible layer hidden under an opaque raster,
    or a symbol that paints nothing, is attributable instead of guessed at.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        from qgis.core import QgsProject

        project = QgsProject.instance()
        blank = []
        for lid in h.state.baseline_layer_ids:
            layer = project.mapLayer(lid)
            if layer is None:
                continue
            safe = "".join(c if c.isalnum() else "_" for c in layer.name()).strip("_").lower()
            h.canvas.setLayers([layer])
            extent = layer.extent()
            if extent.isEmpty():
                extent = h.dem_layer.extent()
            extent.scale(1.05)
            h.canvas.setExtent(extent)
            path = save_canvas(h.canvas, f"layer_{safe}")

            info = describe(path)
            print(
                f" [{info['colours']} colours, {info['ink_share']:.4%} ink]",
                end="",
            )
            if info["colours"] < 3:
                blank.append(layer.name())

        # Emptiness is not automatically wrong — a ponding layer with no earthworks
        # in the design has nothing to draw. So this reports rather than fails, and
        # only insists that the run drew *something*.
        if blank:
            print(f"\n    NOTE: drew nothing on their own: {', '.join(blank)}", end="")
        assert len(blank) < len(h.state.baseline_layer_ids), (
            "every baseline layer rendered blank — nothing reached the canvas"
        )


def check_properties_dialog_renders(dem_path):
    """The restyled earthwork properties dialog, built with real arguments.

    exec() is replaced with a grab-and-reject so the dialog is constructed by the
    controller exactly as a user's drawing action would construct it — real
    catchment figures, real spillway datums, real overflow options.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    captured = {}
    original_exec = EarthworkPropertiesDialog.exec

    def grab_instead_of_exec(dialog):
        dialog.resize(560, 760)
        captured["path"] = save_widget(dialog, "dialog_earthwork_properties")
        return 0   # rejected — nothing added, this check is only about drawing

    EarthworkPropertiesDialog.exec = grab_instead_of_exec
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("properties dialog")
    finally:
        EarthworkPropertiesDialog.exec = original_exec

    assert "path" in captured, "the properties dialog was never opened"
    assert_rendered(captured["path"], "earthwork properties dialog", min_colours=12)
