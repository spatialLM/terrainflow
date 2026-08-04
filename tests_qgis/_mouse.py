"""
_mouse.py — drive the map canvas with synthetic mouse and keyboard input.

Clicks are delivered as real Qt events to the canvas viewport, so they travel the
genuine path: QgsMapCanvas builds a QgsMapMouseEvent and dispatches it to the
active QgsMapTool, which converts it back with toMapCoordinates(). That means the
tools' own coordinate handling is under test, not bypassed.

Positions are given in *map* coordinates (EPSG:2193 metres for the test fixture)
and converted to canvas pixels here, so a check reads in the units of the terrain
rather than in pixels.

The canvas must be sized and visible first — see PluginHarness.prepare_canvas_for_input().
"""

from __future__ import annotations

from qgis.core import QgsPointXY
from qgis.PyQt.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from qgis.PyQt.QtGui import QMouseEvent
from qgis.PyQt.QtTest import QTest

LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton
NO_MOD = Qt.KeyboardModifier.NoModifier


def to_pixel(canvas, x, y):
    """Map coordinate -> canvas pixel QPoint."""
    device = canvas.getCoordinateTransform().transform(QgsPointXY(x, y))
    return QPoint(int(round(device.x())), int(round(device.y())))


def to_map(canvas, point):
    """Canvas pixel -> map coordinate, for asserting round-trips."""
    return canvas.getCoordinateTransform().toMapCoordinates(point.x(), point.y())


def _settle(passes=2):
    for _ in range(passes):
        QCoreApplication.processEvents()


def click_map(canvas, x, y, button=LEFT, modifier=NO_MOD):
    """Single click at a map coordinate."""
    QTest.mouseClick(canvas.viewport(), button, modifier, to_pixel(canvas, x, y))
    _settle()


def dclick_map(canvas, x, y, button=LEFT, modifier=NO_MOD):
    """Double click at a map coordinate, matching what a real double click delivers.

    Deliberately NOT QTest.mouseDClick(): that sends a bare MouseButtonDblClick with
    no preceding press. A real double click is press, release, DblClick, release --
    the first press lands a vertex, which draw_line_tool then pops in
    canvasDoubleClickEvent. Using QTest's version made a 3-vertex line come out with
    2, because the pop had nothing to pop.
    """
    position = to_pixel(canvas, x, y)
    viewport = canvas.viewport()

    QTest.mouseClick(viewport, button, modifier, position)   # press + release
    _settle()

    event = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(position),
        button,
        button,
        modifier,
    )
    QCoreApplication.sendEvent(viewport, event)
    _settle()

    QTest.mouseRelease(viewport, button, modifier, position)
    _settle()


def move_map(canvas, x, y, modifier=NO_MOD):
    """Move the cursor to a map coordinate (drives rubber-band previews).

    Deliberately NOT QTest.mouseMove(): that warps the real cursor, which does
    nothing under the offscreen platform, so canvasMoveEvent never fires and the
    preview segment silently never appears. Posting the event directly works.
    """
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(to_pixel(canvas, x, y)),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        modifier,
    )
    QCoreApplication.sendEvent(canvas.viewport(), event)
    _settle()


def press_key(canvas, key, modifier=NO_MOD):
    """Send a key to the canvas so the active tool's keyPressEvent sees it."""
    canvas.setFocus()
    _settle()
    QTest.keyClick(canvas, key, modifier)
    _settle()


def pixel_size_m(canvas):
    """Map units per pixel — the tolerance floor for click round-trip assertions."""
    return canvas.getCoordinateTransform().mapUnitsPerPixel()
