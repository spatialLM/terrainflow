"""
Map tools driven by real synthetic mouse input.

Everything else in this suite hands geometry to the controllers directly. These
checks instead click on the canvas, so the tools' own event handling and
coordinate conversion are exercised: QgsMapCanvas builds the QgsMapMouseEvent and
DrawLineTool converts it back with toMapCoordinates(). A tool that mis-reads a
click position fails here and nowhere else.
"""

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
