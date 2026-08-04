"""
connect_earthworks_tool.py — draw an overflow link between two earthworks.

Two clicks: pick the feature that overflows, then the feature it overflows into.
A rubber band follows the cursor between them so the link is visible before it is
committed. Emits ``connection_made(from_id, to_id)``.

The controller supplies the candidate features as ``(id, name, QgsGeometry)``
tuples rather than a layer, so hit-testing works against the earthworks the model
actually holds instead of whatever a mirror layer happens to carry.
"""

from qgis.core import QgsGeometry, QgsPointXY, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

_PREVIEW = QColor(20, 90, 160, 200)
_SEARCH_PIXELS = 14


class ConnectEarthworksTool(QgsMapTool):
    """Pick a source earthwork, then a target, to route overflow between them."""

    connection_made = pyqtSignal(str, str)   # from_id, to_id
    source_picked = pyqtSignal(str)          # for a status message while mid-link
    cancelled = pyqtSignal()

    def __init__(self, canvas, features):
        super().__init__(canvas)
        self._canvas = canvas
        self._features = list(features or [])
        self._source = None          # (id, name, anchor QgsPointXY)
        self._band = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # ------------------------------------------------------------------ events

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._abort()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        hit = self._feature_at(self.toMapCoordinates(event.pos()))
        if hit is None:
            return
        ew_id, name, anchor = hit

        if self._source is None:
            self._source = hit
            self._start_band(anchor)
            self.source_picked.emit(name)
            return

        if ew_id == self._source[0]:
            return                    # a feature cannot overflow into itself
        self._clear_band()
        source_id = self._source[0]
        self._source = None
        self.connection_made.emit(source_id, ew_id)
        self._canvas.unsetMapTool(self)

    def canvasMoveEvent(self, event):
        if self._band is None or self._source is None:
            return
        self._band.movePoint(1, self.toMapCoordinates(event.pos()))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._abort()

    def deactivate(self):
        self._clear_band()
        self._source = None
        super().deactivate()

    # ------------------------------------------------------------------ internals

    def _abort(self):
        """Escape backs out of a half-drawn link before it abandons the tool.

        Mis-picking the source is the common slip, and making that cost a trip back
        to the menu would be a poor trade for one keystroke.
        """
        if self._source is not None:
            self._clear_band()
            self._source = None
            return
        self.cancelled.emit()
        self._canvas.unsetMapTool(self)

    def _feature_at(self, point):
        """Nearest candidate within the search radius → (id, name, anchor)."""
        radius = self._canvas.mapUnitsPerPixel() * _SEARCH_PIXELS
        click = QgsGeometry.fromPointXY(QgsPointXY(point))
        best, best_dist = None, float("inf")
        for ew_id, name, geom in self._features:
            if geom is None:
                continue
            try:
                dist = geom.distance(click)
            except Exception:
                continue
            if dist < best_dist:
                best, best_dist = (ew_id, name, geom), dist
        if best is None or best_dist > radius:
            return None
        ew_id, name, geom = best
        try:
            anchor = geom.centroid().asPoint()
        except Exception:
            anchor = QgsPointXY(point)
        return (ew_id, name, anchor)

    def _start_band(self, anchor):
        self._clear_band()
        self._band = QgsRubberBand(self._canvas, QgsWkbTypes.LineGeometry)
        self._band.setColor(_PREVIEW)
        self._band.setWidth(2)
        self._band.addPoint(QgsPointXY(anchor))
        self._band.addPoint(QgsPointXY(anchor))

    def _clear_band(self):
        if self._band is not None:
            try:
                self._canvas.scene().removeItem(self._band)
            except Exception:
                pass
            self._band = None
