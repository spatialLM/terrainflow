"""
Map tools driven by real synthetic mouse input.

Everything else in this suite hands geometry to the controllers directly. These
checks instead click on the canvas, so the tools' own event handling and
coordinate conversion are exercised: QgsMapCanvas builds the QgsMapMouseEvent and
DrawLineTool converts it back with toMapCoordinates(). A tool that mis-reads a
click position fails here and nowhere else.
"""

import os

from _harness import PluginHarness, centreline_x
from _mouse import RIGHT, click_map, dclick_map, move_map, pixel_size_m, press_key
from _shots import assert_rendered, pixels, save_widget

# Three points across the valley, at 36 ha fixture coordinates.
ROW_Y = 5_900_000.0 - 120 * 2.0     # 120 rows down from the top edge


def _line_points():
    cx = centreline_x()
    return [(cx - 60.0, ROW_Y), (cx, ROW_Y), (cx + 60.0, ROW_Y)]


def _saturated_blue_share(path):
    """Fraction of pixels that are distinctly blue — the DEM behind is greyscale."""
    import numpy as np

    arr = pixels(path)
    if arr is None:
        return 0.0
    red, green, blue = (arr[:, :, i].astype(np.int16) for i in range(3))
    is_blue = (blue - red > 40) & (blue - green > 20)
    return float(is_blue.mean())


def _accept_properties_dialog():
    """Stub the modal dialog to 'accepted'; returns a restore callable."""
    from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

    original = EarthworkPropertiesDialog.exec
    EarthworkPropertiesDialog.exec = lambda self: 1
    return lambda: setattr(EarthworkPropertiesDialog, "exec", original)


def check_draw_tool_activates(dem_path):
    """The controller's draw dispatch installs a live DrawLineTool on the canvas.

    This is the check that catches the tools' dependency on the *global*
    qgis.utils.iface: DrawLineTool.__init__ calls iface.mainWindow().statusBar(),
    so if that global is not populated, construction raises before any click.
    """
    from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool

    with PluginHarness(dem_path) as h:
        h.prepare_canvas_for_input()
        h.plugin._earthworks.activate_draw_earthwork("diversion")
        h.assert_no_errors("activate draw tool")

        tool = h.canvas.mapTool()
        assert isinstance(tool, DrawLineTool), (
            f"expected DrawLineTool on the canvas, got {type(tool).__name__}"
        )


def check_click_positions_round_trip(dem_path):
    """A synthetic click must land where the check aimed it.

    Clicks are delivered in pixels, so this pins down the whole conversion chain
    (map -> pixel -> Qt event -> toMapCoordinates) before any check trusts it.
    """
    from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool

    with PluginHarness(dem_path) as h:
        h.prepare_canvas_for_input()
        h.plugin._earthworks.activate_draw_earthwork("diversion")
        tool = h.canvas.mapTool()
        assert isinstance(tool, DrawLineTool)

        points = _line_points()
        for x, y in points:
            click_map(h.canvas, x, y)

        assert len(tool.points) == len(points), (
            f"clicked {len(points)} times but the tool recorded "
            f"{len(tool.points)} vertices - events are not reaching it"
        )

        tolerance = 2.0 * pixel_size_m(h.canvas)   # pixel rounding, both directions
        for (want_x, want_y), got in zip(points, tool.points):
            dx, dy = abs(got.x() - want_x), abs(got.y() - want_y)
            assert dx <= tolerance and dy <= tolerance, (
                f"click landed at ({got.x():.2f}, {got.y():.2f}), "
                f"expected ({want_x:.2f}, {want_y:.2f}); "
                f"off by ({dx:.2f}, {dy:.2f}) m, tolerance {tolerance:.2f} m"
            )


def check_rubber_band_renders_mid_gesture(dem_path):
    """Half-drawn geometry must be visible on the canvas.

    The rubber band is the only feedback while drawing, and it is invisible to every
    other check in this suite because they never enter a gesture.
    """
    with PluginHarness(dem_path) as h:
        h.prepare_canvas_for_input()
        h.plugin._earthworks.activate_draw_earthwork("diversion")
        tool = h.canvas.mapTool()

        points = _line_points()
        click_map(h.canvas, *points[0])
        click_map(h.canvas, *points[1])
        assert tool.rubber_band.numberOfVertices() == 2, (
            f"two clicks gave {tool.rubber_band.numberOfVertices()} band vertices"
        )

        move_map(h.canvas, *points[2])          # preview segment to the cursor

        # 3, not ">= 2": the move must add the preview vertex. Asserting >= 2 passed
        # even when the move was never delivered at all.
        assert tool.rubber_band.numberOfVertices() == 3, (
            f"after moving the cursor the band has "
            f"{tool.rubber_band.numberOfVertices()} vertices, expected 3 "
            f"(2 clicked + 1 preview) - canvasMoveEvent is not firing"
        )

        # save_widget, NOT save_canvas: canvas.saveAsImage() renders the map layers
        # only. A rubber band is a QgsMapCanvasItem living in the QGraphicsView
        # scene, so it is absent from that render and only a widget grab catches it.
        path = save_widget(h.canvas, "maptool_rubber_band")
        assert_rendered(path, "rubber band mid-gesture")

        # The band is drawn in blue over a greyscale DEM, so saturated pixels are
        # proof it actually painted rather than merely existing in the scene.
        blue = _saturated_blue_share(path)
        assert blue > 0.0001, (
            f"only {blue:.4%} of pixels are saturated blue - the rubber band "
            f"exists in the scene but did not paint ({path})"
        )


def check_right_click_finishes_line(dem_path):
    """Left-click vertices, right-click to finish -> an earthwork is created."""
    restore = _accept_properties_dialog()
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.prepare_canvas_for_input()
            h.plugin._earthworks.activate_draw_earthwork("diversion")

            points = _line_points()
            for x, y in points:
                click_map(h.canvas, x, y)
            click_map(h.canvas, *points[-1], button=RIGHT)

            h.assert_no_errors("draw by mouse, right-click finish")
            assert len(h.state.earthwork_manager) == 1, (
                "right-click did not finish the line into an earthwork"
            )

            ew = h.state.earthwork_manager.get(0)
            vertices = [v for v in ew.geometry.vertices()]
            assert len(vertices) == len(points), (
                f"expected {len(points)} vertices, got {len(vertices)}"
            )
    finally:
        restore()


def check_double_click_finishes_line(dem_path):
    """Double-click also finishes, and does not keep the double-clicked point.

    A real double click arrives as press, release, DblClick: the press lands a
    vertex and canvasDoubleClickEvent pops it straight back off. So the position
    that is double-clicked is deliberately NOT a vertex -- three clicks then a
    double click gives a three-vertex line. Encoded so a change is visible.
    """
    restore = _accept_properties_dialog()
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.prepare_canvas_for_input()
            h.plugin._earthworks.activate_draw_earthwork("diversion")

            points = _line_points()
            for x, y in points:
                click_map(h.canvas, x, y)
            dclick_map(h.canvas, *points[-1])

            h.assert_no_errors("draw by mouse, double-click finish")
            assert len(h.state.earthwork_manager) == 1, (
                "double-click did not finish the line into an earthwork"
            )

            ew = h.state.earthwork_manager.get(0)
            vertices = [v for v in ew.geometry.vertices()]
            assert len(vertices) == len(points), (
                f"expected {len(points)} vertices after the double-click pop, "
                f"got {len(vertices)}"
            )
    finally:
        restore()


def check_escape_cancels_line(dem_path):
    """Escape mid-gesture must discard the line and add nothing."""
    from qgis.PyQt.QtCore import Qt

    restore = _accept_properties_dialog()
    try:
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.prepare_canvas_for_input()
            h.plugin._earthworks.activate_draw_earthwork("diversion")
            tool = h.canvas.mapTool()

            points = _line_points()
            click_map(h.canvas, *points[0])
            click_map(h.canvas, *points[1])
            assert len(tool.points) == 2

            press_key(h.canvas, Qt.Key.Key_Escape)

            h.assert_no_errors("escape mid-draw")
            assert tool.points == [], (
                f"escape left {len(tool.points)} vertices on the tool"
            )
            assert len(h.state.earthwork_manager) == 0, (
                "escape still created an earthwork"
            )
    finally:
        restore()


def check_arming_a_draw_tool_does_not_reread_the_slope_band(dem_path):
    """G-9. Both draw tools did a full ``src.read(1)`` in ``__init__``, and a fresh
    tool is built for every draw action — 16.2 ms of the 18.2 ms it takes to arm one
    on the owner's 1139x1016 site, paid again on every click of every draw button.

    The slope raster is a fixed-path product of the baseline, so the band is read
    once and held on `_state`. Footprint: one float32 copy of the grid, 4.6 MB on
    that design, released with the rest of the terrain-derived state when the DEM
    changes.
    """
    import rasterio

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        if not h.state.slope_raster_path:
            return          # no slope product on this fixture; nothing to hold

        controller = h.plugin._earthworks
        controller.activate_draw_line("swale")       # first arm: the read happens

        real_open = rasterio.open
        opens = {"n": 0}

        def counting_open(*args, **kwargs):
            opens["n"] += 1
            return real_open(*args, **kwargs)

        rasterio.open = counting_open
        try:
            for _ in range(5):
                controller.activate_draw_line("swale")
            controller.activate_draw_polygon("basin")
        finally:
            rasterio.open = real_open

        h.assert_no_errors("arming draw tools")
        assert opens["n"] == 0, (
            f"arming six more draw tools reopened a raster {opens['n']} times — "
            f"the slope band is still read per tool"
        )

        tool = h.canvas.mapTool()
        assert getattr(tool, "_slope_array", None) is not None, (
            "the tool has no slope band, so the readout is dead"
        )


def check_a_slope_readout_stays_correct_when_the_band_is_shared(dem_path):
    """The held band must be the same ground the tool would have read itself.

    A cache that hands back a different array is worse than the read it replaced,
    so this pins the readout against a direct read of the same file.
    """
    import rasterio

    from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        slope_path = h.state.slope_raster_path
        if not slope_path:
            return

        h.plugin._earthworks.activate_draw_line("swale")
        tool = h.canvas.mapTool()
        assert isinstance(tool, DrawLineTool)

        held = h.state.slope_band()
        assert held is not None, "the state holds no slope band"
        assert tool._slope_array is held[0], (
            "the tool read its own band instead of taking the one the state holds, "
            "so the rest of this check compares two direct reads and proves nothing"
        )

        with rasterio.open(slope_path) as src:
            t, width, height = src.transform, src.width, src.height

        direct = DrawLineTool(h.canvas, slope_raster_path=slope_path,
                              tool_label="swale")
        try:
            probes = [(10, 10), (height // 2, width // 2), (height - 11, width - 11)]
            for row, col in probes:
                x = t.c + t.a * (col + 0.5)
                y = t.f + t.e * (row + 0.5)
                assert tool._get_slope_text(x, y) == direct._get_slope_text(x, y), (
                    f"the shared band reports a different slope at ({row}, {col}) "
                    f"than a direct read"
                )
        finally:
            direct.deactivate()


def check_the_draw_hint_is_not_resent_unchanged(dem_path):
    """G-9, third part. ``_show_hint`` pushed an identical status-bar message on
    every ``canvasMoveEvent`` — a Qt signal and a repaint per mouse move to say what
    the bar already said. Only a changed message is worth sending.
    """
    from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool

    with PluginHarness(dem_path) as h:
        h.prepare_canvas_for_input()
        h.plugin._earthworks.activate_draw_earthwork("diversion")
        tool = h.canvas.mapTool()
        assert isinstance(tool, DrawLineTool)

        sent = []
        bar = h.iface.mainWindow().statusBar()
        real_show = bar.showMessage
        bar.showMessage = lambda text, *a, **kw: sent.append(text)
        try:
            x, y = _line_points()[0]
            for _ in range(6):
                move_map(h.canvas, x, y)     # the same place, so the same message
        finally:
            bar.showMessage = real_show

        h.assert_no_errors("hovering with the draw tool")
        assert len(sent) <= 1, (
            f"six mouse moves over one spot sent {len(sent)} identical status "
            f"messages: {sent[:2]}"
        )


def check_a_broken_slope_raster_is_logged_not_swallowed(dem_path):
    """G-9, second part. ``except Exception: pass`` meant a slope raster that had
    been deleted, truncated or locked killed the live slope readout in silence —
    the tool still drew, the hint simply never mentioned the grade again, and
    nothing anywhere said why.
    """
    import logging

    from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool

    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Catch()
    logger = logging.getLogger("terrainflow_assessment.map_tools.draw_line_tool")
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with PluginHarness(dem_path) as h:
            h.prepare_canvas_for_input()
            tool = DrawLineTool(h.canvas,
                                slope_raster_path="/no/such/slope.tif",
                                tool_label="swale")
            try:
                assert tool._slope_array is None, (
                    "a missing slope raster somehow produced a band"
                )
            finally:
                tool.deactivate()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert records, (
        "a slope raster that could not be read was swallowed silently — the live "
        "readout dies and nothing says why"
    )


# ---------------------------------------------------------------- scene items

def check_a_deactivated_draw_tool_leaves_nothing_in_the_scene(dem_path):
    """G-10. Four tools only ever `reset()` their rubber band on deactivate — which
    empties the geometry and leaves the QGraphicsItem parented to the canvas scene.
    `use_tool` then drops the tool, the only reference to that band, so every draw
    action orphaned one invisible item for the canvas's lifetime. They survive
    unload. Their four siblings (`connect_earthworks_tool`, `link_spillway_tool`,
    `place_point_tool`, `edit_earthwork_tool`) all call `scene().removeItem`.
    """
    with PluginHarness(dem_path) as h:
        h.prepare_canvas_for_input()
        controller = h.plugin._earthworks
        scene = h.canvas.scene()

        # One round first: the canvas adds its own furniture on first use, and that
        # is not what this is counting.
        controller.activate_draw_line("swale")
        h.canvas.unsetMapTool(h.canvas.mapTool())
        baseline = len(scene.items())

        for _ in range(5):
            controller.activate_draw_line("swale")
            h.canvas.unsetMapTool(h.canvas.mapTool())
            controller.activate_draw_polygon("basin")
            h.canvas.unsetMapTool(h.canvas.mapTool())

        h.assert_no_errors("arming and dropping draw tools")
        after = len(scene.items())
        assert after == baseline, (
            f"ten draw tools left {after - baseline} items in the canvas scene "
            f"({baseline} -> {after}) — each one is an orphaned rubber band"
        )


def check_a_deactivated_query_tool_leaves_nothing_in_the_scene(dem_path):
    """G-10, the other two tools: the ponding query and the contour segment picker.

    The segment tool is the worst of the four — two rubber bands and a vertex
    marker, and its `_cleanup` only hides the marker rather than removing it.
    """
    with PluginHarness(dem_path) as h:
        h.prepare_canvas_for_input()
        h.run_baseline()
        scene = h.canvas.scene()

        from terrainflow_assessment.map_tools.contour_segment_tool import (
            ContourSegmentTool,
        )
        from terrainflow_assessment.map_tools.ponding_query_tool import (
            PondingQueryTool,
        )

        ponding = (h.state.baseline_result or {}).get("ponding")

        def _round():
            if ponding and os.path.exists(ponding):
                tool = PondingQueryTool(h.canvas, ponding)
                h.canvas.setMapTool(tool)
                h.canvas.unsetMapTool(tool)
            seg = ContourSegmentTool(h.canvas, [])
            h.canvas.setMapTool(seg)
            h.canvas.unsetMapTool(seg)

        _round()
        baseline = len(scene.items())
        for _ in range(5):
            _round()

        h.assert_no_errors("arming and dropping query tools")
        after = len(scene.items())
        assert after == baseline, (
            f"five rounds left {after - baseline} items in the canvas scene "
            f"({baseline} -> {after})"
        )
