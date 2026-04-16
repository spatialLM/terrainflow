from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor, QPixmap
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.core import QgsWkbTypes, QgsGeometry, QgsPointXY
from qgis.utils import iface


class DrawLineTool(QgsMapTool):
    """
    Map tool for drawing a polyline on the canvas.
    Left-click adds vertices. Right-click or double-click finishes.
    Escape cancels.
    Emits line_drawn(QgsGeometry) on completion.
    """
    line_drawn = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, canvas, color=None, slope_raster_path=None, tool_label="line"):
        super().__init__(canvas)
        self.canvas = canvas
        self.points = []
        self._double_click_pending = False
        self._tool_label = tool_label  # e.g. "swale", "berm", "dam"

        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        c = color or QColor(0, 120, 220, 200)
        self.rubber_band.setColor(c)
        self.rubber_band.setWidth(3)

        self._slope_array = None
        self._slope_transform = None
        if slope_raster_path:
            try:
                import rasterio
                with rasterio.open(slope_raster_path) as src:
                    self._slope_array = src.read(1).astype("float32")
                    self._slope_transform = src.transform
            except Exception:
                pass

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
        iface.mainWindow().statusBar().showMessage(msg)

    def canvasPressEvent(self, event):
        if self._double_click_pending:
            self._double_click_pending = False
            return
        if event.button() == Qt.LeftButton:
            pt = self.toMapCoordinates(event.pos())
            self.points.append(QgsPointXY(pt))
            self.rubber_band.addPoint(pt, True)
        elif event.button() == Qt.RightButton:
            self._finish()

    def canvasDoubleClickEvent(self, event):
        self._double_click_pending = True
        if len(self.points) >= 1:
            self.points.pop()
            self.rubber_band.removeLastPoint()
        self._finish()

    def canvasMoveEvent(self, event):
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
        if event.key() == Qt.Key_Escape:
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
        self.rubber_band.reset(QgsWkbTypes.LineGeometry)

    def deactivate(self):
        self._reset()
        if iface:
            iface.mainWindow().statusBar().clearMessage()
        super().deactivate()
