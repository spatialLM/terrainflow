import logging

from qgis.core import QgsGeometry, QgsPointXY, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

_log = logging.getLogger(__name__)


class DrawPolygonTool(QgsMapTool):
    """
    Map tool for drawing a polygon on the canvas.
    Left-click adds vertices. Right-click or double-click closes and finishes.
    Escape cancels.
    Emits polygon_drawn(QgsGeometry) on completion.
    """
    polygon_drawn = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, canvas, color=None, slope_raster_path=None,
                 tool_label="polygon", slope_band=None):
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        # No `_double_click_pending` flag: it guarded the press *after* a double
        # click, and there is never one — all five completion handlers unset the map
        # tool and every activation builds a fresh instance.
        self._tool_label = tool_label
        self._last_hint = None

        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        c = color or QColor(0, 180, 80, 160)
        self.rubber_band.setColor(c)
        self.rubber_band.setFillColor(QColor(c.red(), c.green(), c.blue(), 60))
        self.rubber_band.setWidth(3)

        # See `DrawLineTool.__init__`: the band is read once per terrain by
        # `PluginState.slope_band()`, not once per tool, and a tool is built for
        # every draw action. The path is still accepted for standalone use.
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
                _log.debug("slope raster could not be read: %s",
                           slope_raster_path, exc_info=True)

        self._show_hint()

    def _show_hint(self, slope_text=""):
        base = (
            f"✏  Drawing {self._tool_label}  —  "
            "Left-click: add point  |  Right-click / Double-click: finish (≥ 3 points)  |  Esc: cancel"
        )
        msg = f"{base}  |  {slope_text}" if slope_text else base
        # Only a changed hint is worth a signal and a repaint — see DrawLineTool.
        if msg == self._last_hint:
            return
        self._last_hint = msg
        iface.mainWindow().statusBar().showMessage(msg)

    def canvasPressEvent(self, event):
        if self.rubber_band is None:
            return              # deactivated; a queued event is not a gesture
        if event.button() == Qt.MouseButton.LeftButton:
            pt = self.toMapCoordinates(event.pos())
            self.points.append(QgsPointXY(pt))
            self.rubber_band.addPoint(pt, True)
        elif event.button() == Qt.MouseButton.RightButton:
            self._finish()

    def canvasDoubleClickEvent(self, event):
        if self.rubber_band is None:
            return
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
        if len(self.points) >= 3:
            closed = self.points + [self.points[0]]
            geom = QgsGeometry.fromPolygonXY([closed])
            self._reset()
            self.polygon_drawn.emit(geom)
        else:
            self._cancel()

    def _cancel(self):
        self._reset()
        self.cancelled.emit()

    def _reset(self):
        self.points = []
        if self.rubber_band is not None:
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)

    def deactivate(self):
        self._reset()
        # See `DrawLineTool.deactivate`: `reset()` empties the geometry and leaves
        # the QGraphicsItem parented to the scene, so every draw action orphaned one.
        if self.rubber_band is not None:
            try:
                self.canvas.scene().removeItem(self.rubber_band)
            except Exception:
                pass
            self.rubber_band = None
        if iface:
            iface.mainWindow().statusBar().clearMessage()
        self._last_hint = None
        super().deactivate()
