from qgis.core import QgsPointXY
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor


class PlacePointTool(QgsMapTool):
    """
    Single-click map tool for placing a point (e.g. a spillway location).

    Emits ``point_placed(QgsPointXY, elevation)`` then deactivates. *elevation* is
    sampled from *snap_raster_path* when one is given, and ``None`` otherwise — a
    spillway sited without an elevation is a location with no crest behind it, so
    the caller needs to know which it got rather than being handed a silent 0.0.
    """
    point_placed = pyqtSignal(object, object)   # QgsPointXY, float | None
    cancelled = pyqtSignal()

    def __init__(self, canvas, snap_raster_path=None):
        super().__init__(canvas)
        self.canvas = canvas
        self._snap_raster_path = snap_raster_path
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pt = QgsPointXY(self.toMapCoordinates(event.pos()))
            self.point_placed.emit(pt, self._elevation_at(pt))
            self.canvas.unsetMapTool(self)
        elif event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()
            self.canvas.unsetMapTool(self)

    def _elevation_at(self, pt):
        """DEM elevation under *pt*, or None when unavailable.

        Uses the same sampler as the rest of the plugin so a placed point and the
        analysis cannot disagree about the ground they are standing on.
        """
        if not self._snap_raster_path:
            return None
        try:
            from terrainflow_assessment.modules.swale_design import (
                snap_point_to_contour_elevation,
            )
            elev = snap_point_to_contour_elevation(
                (pt.x(), pt.y()), self._snap_raster_path)
            return None if elev is None else float(elev)
        except Exception:
            return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.canvas.unsetMapTool(self)
