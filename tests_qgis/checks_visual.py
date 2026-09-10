"""
Visual checks — render the real panel, dialogs and map canvas, assert something
was drawn, and leave the PNGs in tests_qgis/_shots/ to be looked at.

These answer the question state assertions cannot: *did it draw?* A renderer that
builds without error but paints nothing, a layer added below an opaque raster, a
dock that collapses to zero width — all pass Tier 1 and fail here.
"""

from _harness import PluginHarness, build_synthetic_dem, line_across_valley
from _shots import _settle, assert_rendered, describe, save_canvas, save_widget
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

        # Wide enough for all nine columns. Narrower and the table falls back on a
        # horizontal scrollbar with the Feature column squeezed to an ellipsis, which
        # renders perfectly and shows nothing about the layout under review.
        #
        # Sized and settled before the grab, not only by save_widget: the table sits in
        # the panel's layout, which pulls it back to the dock width on the first settle
        # after a bare resize.
        table = h.panel._spillway_table
        table.window().show()
        table.show()
        table.resize(660, 320)
        _settle()
        path = save_widget(table, "panel_spillway_review", size=(660, 320))
        assert_rendered(path, "spillway review table", min_colours=6)


def check_a_sill_limited_width_renders(dem_path):
    """The Width column on sills too shallow to pass their design head.

    The only shot that can see this at all. Every other spillway fixture is built with
    `add_earthwork`, which makes a bare feature with no `Spillway` -- so its crest is
    None, no depth is ever measured, nothing is capped, and the Width column renders
    exactly as it did before this change. The sills here are designed and deliberately
    shallow.

    Three rows, three states the column has to tell apart at a glance: a sill deep enough
    to be sized at its design head, one shallow enough that the width is solved against
    the notch instead, and one with no depth left to spill through at all -- which is the
    most undersized sill it is possible to draw and used to render as the same faint dash
    as a feature with no flow. What this catches is the widened figures overflowing a
    column laid out for "0.45 m", and the third state disappearing into the second.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row in (40, 65, 90):
            h.add_earthwork("swale", geometry=line_across_valley(row=row))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("sill-limited width render")

        table = h.panel._spillway_table
        assert table._rows, "nothing to render - no rows were built"
        for depth, data in zip((0.40, 0.05, 0.0), list(table._rows)):
            h.panel.set_spillway_depth_requested.emit(data["index"], depth)
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("sill depths applied")

        heads = [(r.get("sizing_head_m"), r.get("target_head_m")) for r in table._rows]
        assert any(s is not None and t is not None and s < t for s, t in heads), (
            f"no row in the fixture is sill-limited, so the shot shows nothing new: "
            f"{heads}")

        h.panel._show_stage("design")
        for header in h.panel.findChildren(QPushButton, "tfSectionHeader"):
            header.setChecked(True)
        table.window().show()
        table.show()
        table.resize(660, 320)
        _settle()
        path = save_widget(table, "panel_spillway_sill_limited_width", size=(660, 320))
        assert_rendered(path, "sill-limited spillway widths", min_colours=6)


def check_spillway_depth_editor_renders(dem_path):
    """The Sill depth column with an editor open in it.

    The only check that can see the two ways a spin box in a cell goes wrong: overflowing
    a column sized to "0.45 m", and — being framed, and so taller than a text row —
    growing its row inside a table whose height is fixed to eight of them, pushing the
    rest out of view mid-edit. Both render perfectly and pass every assertion elsewhere.
    """
    from terrainflow_assessment.qgis.widgets.spillway_table import _COL_DEPTH

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row in (40, 65, 90):
            h.add_earthwork("swale", geometry=line_across_valley(row=row))
        h.add_earthwork("basin", geometry=line_across_valley(row=110, half_width_m=15.0))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("spillway depth editor render")

        table = h.panel._spillway_table
        assert table._rows, "nothing to render — no rows were built"
        editable = [i for i, row in enumerate(table._rows)
                    if table._depth_editable(row)]
        assert editable, "no row in the fixture can be typed in, so there is no editor"
        r = editable[0]

        h.panel._show_stage("design")
        for header in h.panel.findChildren(QPushButton, "tfSectionHeader"):
            header.setChecked(True)

        # Sized before the editor is placed, because save_widget resizes on the way in
        # and the column the editor was measured against would have moved out from under
        # it -- which is exactly the wrong-cell shot this check exists to notice.
        table.window().show()
        table.show()
        table.resize(660, 320)
        _settle()

        # Placed by hand rather than through the view: an offscreen table will not open
        # an editor for a synthetic edit(), and what is being photographed is the
        # editor's geometry inside the cell.
        index = table.table.model().index(r, _COL_DEPTH)
        delegate = table.table.itemDelegateForColumn(_COL_DEPTH)
        editor = delegate.createEditor(table.table.viewport(), None, index)
        delegate.setEditorData(editor, index)
        rect = table.table.visualRect(index)
        assert rect.isValid() and rect.width() > 0, (
            "the sill depth cell has no visible rectangle, so the editor cannot be "
            "photographed where it would actually appear")
        editor.setGeometry(rect)
        editor.show()
        try:
            path = save_widget(table, "panel_spillway_depth_editor", size=(660, 320))
            assert_rendered(path, "spillway depth editor", min_colours=6)
        finally:
            editor.deleteLater()


def check_verification_table_renders(dem_path):
    """Drawn vs measured, with every row state in one image.

    Driven from a synthetic result rather than a real burn: the point is the styling
    of the five states side by side — flagged, clean, dam, sub-cell, shared-pool — and
    a fixture that happens to produce them all at once is not something to rely on.

    What to look at: the flagged row's Δ must be blue with a dagger and NOT green
    (that green on the wrong comparison is the bug this table had), the clean row's Δ
    must still be green, the shared-pool row must read "shared" and "—" rather than a
    share of someone else's water, the dam row must carry its impounded volume under
    Geometric, the subhead must wrap inside the dock width without pushing the table
    off, and the Δ column must be wide enough for "+0% †".
    """
    from terrainflow_assessment.modules.reporting import VerificationResult

    result = VerificationResult(
        analytic_total_m3=2741.0, terrain_total_m3=2645.0,
        per_feature=[
            # Too narrow for the grid to hold its section: At grid sits 24% over the
            # drawn trench, so Δ +0% is agreement with the grid and not with the design.
            # Geometric also carries a companion berm, which the trench columns do not.
            {"name": "Swale 33", "analytic_m3": 364.0, "geometric_m3": 364.0,
             "section_m3": 208.0, "berm_credit_m3": 156.0,
             "rasterisable_m3": 259.0, "terrain_m3": 259.0, "delta_pct": 0.0,
             "resolution_penalty_m3": 51.0,
             "section_overstated": True, "section_gap_pct": 24.5,
             "existing_m3": 0.0, "total_m3": 259.0},
            # The grid holds the section: green means something here.
            {"name": "Basin 39", "analytic_m3": 765.0, "geometric_m3": 765.0,
             "section_m3": 765.0, "berm_credit_m3": 0.0,
             "rasterisable_m3": 772.0, "terrain_m3": 759.0, "delta_pct": -1.7,
             "resolution_penalty_m3": 7.0,
             "section_overstated": False, "section_gap_pct": 0.9,
             "existing_m3": 0.0, "total_m3": 759.0},
            # A dam: one figure, no cross-section to overstate.
            {"name": "Dam 40", "analytic_m3": 1838.0, "geometric_m3": 1838.0,
             "section_m3": 1838.0, "berm_credit_m3": 0.0,
             "rasterisable_m3": 1838.0, "terrain_m3": 1433.0, "delta_pct": -22.0,
             "barrier_impounded": True, "section_overstated": False,
             "existing_m3": 555.0, "total_m3": 1988.0},
            # Narrower than a cell: no measured claim at all.
            {"name": "Swale 41", "analytic_m3": 80.0, "geometric_m3": 100.0,
             "routing_only": True, "section_overstated": False},
            # Its pool runs into Basin 39's: measured, but not measured of it.
            {"name": "Swale 12", "analytic_m3": 240.0, "geometric_m3": 300.0,
             "section_m3": 300.0, "berm_credit_m3": 0.0,
             "rasterisable_m3": 305.0, "terrain_m3": None, "delta_pct": None,
             "merged_with": ["Basin 39"], "section_overstated": False,
             "existing_m3": 0.0, "total_m3": None},
        ],
        merged_groups=[{"names": ["Basin 39", "Swale 12"],
                        "rasterisable_m3": 1077.0, "terrain_m3": 1064.0,
                        "delta_pct": -1.2, "existing_m3": 0.0}])

    with PluginHarness(dem_path) as h:
        h.panel.set_verification(result, cell_size_m=1.0)
        h.assert_no_errors("verification table populate")

        table = h.panel._verification_table
        assert not table.isHidden(), "the table hid itself with four rows to show"
        assert h.panel._verification_empty.isHidden(), (
            "the 'run an analysis' placeholder is still showing above four real rows")

        # Columns are read by name, not by a number that silently means the wrong one
        # after a column is added or dropped — which is what happened when "Design" went
        # and every index in this check was off by one.
        headers = [table.table.horizontalHeaderItem(c).text()
                   for c in range(table.table.columnCount())]
        assert "Design" not in headers, (
            f"the freeboard-derived Design column is back: {headers}")
        col = {name: i for i, name in enumerate(headers)}
        d, meas, geom = col["Δ"], col["Measured"], col["Geometric"]

        deltas = [table.table.item(r, d).text() for r in range(table.table.rowCount())]
        assert deltas[0] == "+0% †", f"flagged row lost its marker: {deltas[0]}"
        assert "†" not in deltas[1], f"clean row was marked: {deltas[1]}"

        flagged = table.table.item(0, d).foreground().color().name()
        clean = table.table.item(1, d).foreground().color().name()
        assert flagged == "#1273b5", f"flagged delta should be informational: {flagged}"
        assert clean == "#1e8449", f"clean delta should stay green: {clean}"

        # The dam has no drawn cross-section, so Geometric carries its impounded volume
        # rather than the bare word "impounded". Before Design was dropped that figure
        # lived in Design and the row would now have no volume on it at all.
        dam_row = next(r for r in range(table.table.rowCount())
                       if table.table.item(r, 0).text() == "Dam 40")
        dam_geom = table.table.item(dam_row, geom).text()
        assert "1,838" in dam_geom and "impounded" in dam_geom, (
            f"the dam row lost its only volume figure: {dam_geom!r}")

        merged_row = table.table.rowCount() - 1
        assert table.table.item(merged_row, meas).text() == "shared", (
            "a feature sharing a pool still claims a measured volume of its own")
        assert table.table.item(merged_row, d).text() == "—", (
            "a feature sharing a pool still carries a Δ it cannot have earned")

        footer = table.footer.text()
        assert "cannot hold its section" in footer, footer
        # Freeboard is not a column and no longer a figure at all: the blanket 20%
        # is gone from `calculate_capacity`, so there is no allowance left to state.
        # The footer stating one again would mean the deduction had come back.
        assert "freeboard" not in footer.lower(), (
            f"the blanket freeboard allowance is being reported again: {footer}")
        assert "between Design and" not in footer, (
            "the footer still tells the reader to subtract two columns, one of which "
            "is gone")
        assert "flat-floored trench" not in footer, (
            "the footer still describes the pre-taper burn, which squared every drawn "
            "channel off to a rectangle regardless of its cross-section")
        assert "Basin 39 + Swale 12" in footer, (
            "the shared pool is measured but never reported, so two rows are blank "
            "with nothing to explain them")

        path = save_widget(table, "panel_verification_table", size=(460, 320))
        assert_rendered(path, "verification table", min_colours=6)

        # The dock is ~460 px and the Δ column just grew a marker. Compare the summed
        # section widths against the viewport rather than asking the scrollbar whether
        # it is visible — offscreen, a widget whose window was never shown reports
        # hidden either way, so that question always answers "fine".
        width = table.table.horizontalHeader().length()
        viewport = table.table.viewport().width()
        assert width <= viewport, (
            f"columns need {width}px inside a {viewport}px viewport — the table "
            f"scrolls sideways in the dock")

        # And in situ, because a word-wrapped subhead can widen the whole dock even
        # when the table itself fits.
        h.panel._show_stage("verify")
        panel_shot = save_widget(h.panel, "panel_verification", size=PANEL_SIZE)
        assert_rendered(panel_shot, "panel (verification populated)", min_colours=12)


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


def check_pond_layer_renders(dem_path):
    """A pond in the channel: the crest split fires, and the reservoir draws as water.

    The standard synthetic DEM is written to be depression-free, so **nothing** on it
    reaches the crest split — a green run over it says nothing about ponds, which is why
    this check builds its own surface with a basin cut into the valley floor.

    Two things a pure test cannot reach. That the pond raster survives the trip through the
    worker onto disk and back into a valid layer, and that it *paints* — a reservoir now
    genuinely stops the channel, since a pond holds its inflow and sheds it along its whole
    crest instead of threading a line through itself, so if this layer draws nothing the map
    has a hole where the water is.

    The layer read is **Pond Capacity (full)**, not the "Ponds (routed)" layer this check
    was written against. That one is gone — it painted a cell count through a depth ramp —
    and Pond Capacity covers a strict superset of the same pools, so it is what now fills
    the gap in Streams. ``pond_flow`` is still asserted on disk, because keypoint analysis
    reads that file and the layer going away must not take the raster with it.
    """
    import os
    from pathlib import Path

    ponded = build_synthetic_dem(Path(dem_path).with_name("ponded_dem.tif"), pond=True)
    with PluginHarness(str(ponded)) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run over a ponded DEM")

        result = h.state.baseline_result or {}
        assert result.get("crest_ponds", 0) > 0, (
            "the basin did not register as a pond — this check is no longer testing "
            "what it exists to test"
        )
        pond_path = result.get("pond_flow")
        assert pond_path and os.path.exists(pond_path), (
            f"no pond raster written ({pond_path!r})"
        )

        from qgis.core import QgsProject

        project = QgsProject.instance()
        names = [project.mapLayer(lid).name()
                 for lid in h.state.baseline_layer_ids
                 if project.mapLayer(lid) is not None]
        assert not [n for n in names if "Ponds (routed)" in n], (
            f"the Ponds (routed) layer is back in the baseline group: {names}"
        )
        pond_layers = [n for n in names if "Pond Capacity (full)" in n]
        assert pond_layers, f"no Pond Capacity layer in the baseline group: {names}"

        layer = next(project.mapLayer(lid) for lid in h.state.baseline_layer_ids
                     if project.mapLayer(lid) is not None
                     and "Pond Capacity (full)" in project.mapLayer(lid).name())
        # On its own, not synced to the project: the question is whether *this* layer
        # paints, and a shot with the DEM under it cannot answer that. Framed on the water
        # rather than on the raster, which spans the whole DEM and is zero nearly
        # everywhere — at full extent a reservoir is a speck and the shot proves nothing.
        import rasterio
        from qgis.core import QgsRectangle

        with rasterio.open(result.get("ponding") or pond_path) as src:
            band = src.read(1)
            rows, cols = (band > 0).nonzero()
            left, top = src.xy(rows.min(), cols.min(), offset="ul")
            right, bottom = src.xy(rows.max(), cols.max(), offset="lr")

        h.canvas.setLayers([layer])
        extent = QgsRectangle(left, bottom, right, top)
        extent.scale(1.4)
        h.canvas.setExtent(extent)
        path = save_canvas(h.canvas, "layer_baseline___pond_capacity")
        describe(path)
        # Depth on nothing. Two colours is the floor — one means "the layer exists but
        # paints nothing", which is the failure this shot is here to catch.
        assert_rendered(path, "Pond Capacity (full) layer", min_colours=2)


def check_earthworks_hillshade_renders(dem_path):
    """The shaded relief over the burned DEM — the view that shows the design as earth.

    On the elevation ramp a 1 m swale inside tens of metres of relief is one grey
    against another; shaded, it is a cut line with a bank beside it. This asserts three
    things a pure test cannot reach: that the layer exists and is ticked, that it renders
    something, and — the part that has already been got wrong once — that it sits **above**
    the Burned DEM in the tree. ``place`` appends, and appended means painted underneath,
    so a hillshade added after the raster it shades is invisible and the feature silently
    does nothing.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale")
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")

        from qgis.core import QgsProject

        project = QgsProject.instance()
        names = {}
        for lid in h.state.earthworks_layer_ids:
            layer = project.mapLayer(lid)
            if layer is not None:
                names[layer.name()] = layer
        shade = names.get("Earthworks — Hillshade")
        assert shade is not None, (
            f"no earthworks hillshade registered; got {sorted(names)}")

        root = project.layerTreeRoot()
        node = root.findLayer(shade.id())
        assert node is not None, "hillshade is not in the layer tree"
        assert node.isVisible(), "the earthworks hillshade arrived unticked"

        burned = names.get("Earthworks — Burned DEM")
        if burned is not None:
            siblings = [n.layerId() for n in node.parent().findLayers()]
            assert siblings.index(shade.id()) < siblings.index(burned.id()), (
                "hillshade sits below the Burned DEM, so it renders underneath it "
                "and nothing on screen changes"
            )

        h.canvas.setLayers([shade])
        extent = shade.extent()
        if extent.isEmpty():
            extent = h.dem_layer.extent()
        extent.scale(1.05)
        h.canvas.setExtent(extent)
        path = save_canvas(h.canvas, "canvas_earthworks_hillshade")
        info = describe(path)
        print(f" [{info['colours']} colours, {info['ink_share']:.4%} ink]", end="")
        assert info["colours"] > 3, "the hillshade rendered flat — nothing to see"


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


def check_spillway_section_renders(dem_path):
    """The overflow section with a sill designed on it — one input, four readouts.

    Nothing drew this before: the only dialog shot in the suite has the Spillway group
    unticked, so the whole section collapsed to its title row and every change to it went
    unreviewed.

    What to look at: the sill is set by 'Spillway depth', or on a cut feature from the
    floor instead — those two carry spin arrows. The overflow elevation and the
    freeboard beneath them sit on a grey field with no arrows, because they are
    consequences rather than five ways of saying the same thing. Freeboard is coloured
    by how much margin is left, and the storage line says what the sill costs in cubic
    metres.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    captured = {}
    original_exec = EarthworkPropertiesDialog.exec

    def grab_instead_of_exec(dialog):
        dialog.grp_spillway.setChecked(True)
        dialog.spin_spillway_drop.setValue(0.45)
        captured["depth_editable"] = not dialog.spin_spillway_drop.isReadOnly()
        captured["crest_readonly"] = dialog.spin_spillway_crest.isReadOnly()
        captured["freeboard"] = dialog.spin_spillway_freeboard.value()
        captured["crest"] = dialog.spin_spillway_crest.value()
        dialog.resize(560, 900)
        captured["path"] = save_widget(dialog, "dialog_spillway_section")
        return 0

    EarthworkPropertiesDialog.exec = grab_instead_of_exec
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("spillway section")
    finally:
        EarthworkPropertiesDialog.exec = original_exec

    assert "path" in captured, "the properties dialog was never opened"
    assert captured["depth_editable"], "the sill depth is not editable"
    assert captured["crest_readonly"], "the overflow elevation is still typeable"
    # Freeboard is the depth less the design head, so on a 0.45 m sill at the swale's
    # policy head it is a real number that moves — not a constant the row could have
    # printed without computing anything.
    assert captured["freeboard"] < 0.45, (
        f"freeboard came back as {captured['freeboard']:.2f} m on a 0.45 m sill — the "
        f"design flow has to take some of that depth")
    assert_rendered(captured["path"], "spillway section", min_colours=12)


def check_companion_berm_readout_renders(dem_path):
    """What you are told about the berm while drawing, and after analysing.

    The berm is the one part of a swale whose size is not something you type — it is
    whatever the trench yields — so the dialog has to predict it, and the prediction has
    to be the bank the burner actually lays down. It was not: this line quoted
    ``√(0.75 × section)``, the height of a 1:1 triangular ridge, while the burn spread
    the same spoil across a band as wide as the swale to a level crest. For the registry
    default that is 1.22 m against 0.50 m — the same earth, 2.4× the height.

    What to look at: the predicted height, its band width and the spoil per metre are
    one sentence under Calculated Capacity; the crest the last burn built sits beside
    the key-in checkbox that changes it; and neither appears when there is no berm.
    """
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    captured = {}
    original_exec = EarthworkPropertiesDialog.exec

    def grab_instead_of_exec(dialog):
        # Tick the berm and hand it the crest a re-analysed design would carry. The
        # registry-default swale (0.5 m deep, 2.0 / 1.0 m) yields 0.5625 m³/m of spoil
        # over a 2.0 m band, so the burn builds 0.28 m — deliberately the same figure
        # the line above predicts, because the whole point of this pair is that they
        # agree. If they ever stop agreeing, the prediction is describing a bank the
        # burner does not build, which is the defect this check exists to catch.
        dialog.chk_companion.setChecked(True)
        dialog._earthwork.berm_crest_elevation = 79.40
        # (min, mean, max) height along the run. Deliberately uneven, because that is
        # the case a single figure hides: a level crest over ground that falls gives a
        # bank of very different heights at its two ends.
        dialog._earthwork.berm_height_m = (0.19, 0.28, 0.44)
        dialog._update_capacity()
        captured["height"] = dialog.lbl_berm_height.text()
        captured["crest"] = dialog.lbl_berm_crest.text()
        dialog.resize(560, 760)
        captured["path"] = save_widget(dialog, "dialog_companion_berm")
        return 0

    EarthworkPropertiesDialog.exec = grab_instead_of_exec
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            ew = h.add_earthwork("swale", geometry=line_across_valley())
            ew.companion_berm = True
            h.plugin._earthworks._on_geometry_drawn("swale", line_across_valley())
            h.assert_no_errors("companion berm readout")
    finally:
        EarthworkPropertiesDialog.exec = original_exec

    assert "path" in captured, "the properties dialog was never opened"
    height, crest = captured["height"], captured["crest"]
    assert "level-topped" in height, height
    assert "spoil per metre" in height, height
    assert "0.19–0.44 m tall" in crest, (
        f"the crest must be stated as the range the bank actually stands, not as a bare "
        f"datum and not as a mean that describes neither end: {crest!r}")
    assert "mean 0.28" in crest, crest
    assert "79.40" in crest, crest
    assert "0.28 m high" in height, (
        f"the predicted height and the built bank's mean must agree — predicted line "
        f"reads {height!r} against a built {crest!r}")
    assert_rendered(captured["path"], "companion berm readout", min_colours=12)


def check_catchment_coverage_readout(dem_path):
    """The area readout under the catchment toggle, shot on its own.

    Rendered separately for the same reason the water-leaving box is: inside a
    460x1400 panel image a two-line block asserts nothing but a colour count, and
    the thing worth seeing here is whether the indent still reads as belonging to
    the checkbox above it and whether the second line has wrapped to three.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        for row, ew_type in ((40, "swale"), (60, "dam")):
            h.add_earthwork(ew_type, geometry=line_across_valley(row=row))
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("live assessment")

        lbl = h.panel._catchment_coverage_lbl
        assert not lbl.isHidden(), "coverage readout stayed hidden after a design"
        text = lbl.text()
        assert "Catchment worked" in text, text
        assert "of site area" in text, (
            f"the area/volume distinction must sit on the number's own line: {text}")
        assert "analysed" in text, (
            f"the denominator has to be printed so the share is checkable: {text}")

        # assert_rendered treats anything under 50px tall as "never laid out" — a
        # sound guard for an empty widget, but this readout is a legitimately short
        # two-line block that grabs at ~46px. Give it headroom for the shot only, so
        # the image shows the text with whitespace rather than tripping the heuristic.
        lbl.setMinimumHeight(110)
        try:
            path = save_widget(lbl, "panel_catchment_coverage", size=(520, 120))
            assert_rendered(path, "catchment coverage readout", min_colours=3)
        finally:
            lbl.setMinimumHeight(0)
