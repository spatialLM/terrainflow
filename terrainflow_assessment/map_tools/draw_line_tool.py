import logging

from qgis.core import QgsGeometry, QgsPointXY, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

_log = logging.getLogger(__name__)


class DrawLineTool(QgsMapTool):
    """
    Map tool for drawing a polyline on the canvas.
    Left-click adds vertices. Right-click or double-click finishes.
    Escape cancels.
    Emits line_drawn(QgsGeometry) on completion.
    """
    line_drawn = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, canvas, color=None, slope_raster_path=None, tool_label="line",
                 slope_band=None):
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        self._double_click_pending = False
        self._tool_label = tool_label  # e.g. "swale", "berm", "dam"
        self._last_hint = None

        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        c = color or QColor(0, 120, 220, 200)
        self.rubber_band.setColor(c)
        self.rubber_band.setWidth(3)

        # *slope_band* is ``(array, transform)`` already in hand — `PluginState.
        # slope_band()`, which reads the raster once per terrain rather than once per
        # tool. A fresh tool is built for every draw action, and that read was 16.2 ms
        # of the 18.2 ms it takes to arm one. The path is still accepted, so a tool
        # built outside the plugin (a check, a script) still works on its own.
        self._slope_array = None
        self._slope_transform = None
        if slope_band is not None:
            self._slope_array, self._slope_transform = slope_band
        elif slope_raster_path:
            try:
                import rasterio
                with rasterio.open(slope_raster_path) as src:
                    self._slope_array = src.read(1).astype("float32")
                    self._slope_transform = src.transform
            except Exception:
                # Was `pass`. A slope raster that has been deleted, truncated or
                # locked then killed the live grade readout in silence: the tool
                # still drew, the hint simply never mentioned the grade again, and
                # nothing anywhere said why.
                _log.debug("slope raster could not be read: %s",
                           slope_raster_path, exc_info=True)

        self._show_hint()

    def _show_hint(self, slope_text=""):
        """Update the QGIS status bar with drawing instructions + optional slope."""
        base = (
            f"✏  Drawing {self._tool_label}  —  "
            "Left-click: add point  |  Right-click / Double-click: finish  |  Esc: cancel"
        )
        if slope_text:
            msg = f"{base}  |  {slope_text}"
        else:
            msg = base
        # Called from `canvasMoveEvent`, where the message is usually the one already
        # showing: the hint only changes when the cursor crosses into a different
        # slope band. Re-sending it is a signal and a status-bar repaint per mouse
        # move to say what the bar already says.
        if msg == self._last_hint:
            return
        self._last_hint = msg
        iface.mainWindow().statusBar().showMessage(msg)

    def canvasPressEvent(self, event):
        if self.rubber_band is None:
            return              # deactivated; a queued event is not a gesture
        if self._double_click_pending:
            self._double_click_pending = False
            return
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self.toMapCoordinates(event.pos())
            self.points.append(QgsPointXY(pt))
            self.rubber_band.addPoint(pt, True)
        elif event.button() == Qt.MouseButton.RightButton:
            self._finish()

    def canvasDoubleClickEvent(self, event):
        if self.rubber_band is None:
            return
        self._double_click_pending = True
        if len(self.points) >= 1:
            self.points.pop()
            self.rubber_band.removeLastPoint()
        self._finish()

    def canvasMoveEvent(self, event):
        if self.rubber_band is None:
            return
        pt = self.toMapCoordinates(event.pos())
        slope_text = ""
        if self._slope_array is not None:
            slope_text = self._get_slope_text(pt.x(), pt.y())
        self._show_hint(slope_text)

        if not self.points:
            return
        # Show preview segment to cursor
        if self.rubber_band.numberOfVertices() > len(self.points):
            self.rubber_band.removeLastPoint()
        self.rubber_band.addPoint(pt, True)

    def _get_slope_text(self, x, y):
        t = self._slope_transform
        col = int((x - t.c) / t.a)
        row = int((y - t.f) / t.e)
        row = max(0, min(self._slope_array.shape[0] - 1, row))
        col = max(0, min(self._slope_array.shape[1] - 1, col))
        slope = float(self._slope_array[row, col])
        if slope < 0 or slope > 90:
            return ""
        if slope <= 5:
            label = "Gentle ✓"
        elif slope <= 15:
            label = "Moderate ⚠"
        else:
            label = "Steep ⛔"
        return f"Slope: {slope:.1f}°  {label}"

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()

    def _finish(self):
        if len(self.points) >= 2:
            geom = QgsGeometry.fromPolylineXY(self.points)
            self._reset()
            self.line_drawn.emit(geom)
        else:
            self._cancel()

    def _cancel(self):
        self._reset()
        self.cancelled.emit()

    def _reset(self):
        self.points = []
        if self.rubber_band is not None:
            self.rubber_band.reset(QgsWkbTypes.LineGeometry)

    def deactivate(self):
        self._reset()
        # `reset()` empties the geometry; the QGraphicsItem stays parented to the
        # canvas scene. `use_tool` then drops this tool, the band's only reference,
        # so every draw action left one invisible item in the scene for the canvas's
        # lifetime — they survive unload. The four sibling tools that do this right
        # (`connect_earthworks_tool`, `link_spillway_tool`, `place_point_tool`,
        # `edit_earthwork_tool`) all call `scene().removeItem`.
        if self.rubber_band is not None:
            try:
                self.canvas.scene().removeItem(self.rubber_band)
            except Exception:
                pass
            self.rubber_band = None
        if iface:
            iface.mainWindow().statusBar().clearMessage()
        # The bar is now empty, so the next hint is a change however it reads.
        self._last_hint = None
        super().deactivate()
