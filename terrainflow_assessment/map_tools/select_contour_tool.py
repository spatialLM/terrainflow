from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QCursor

from terrainflow_assessment.map_tools._contour_pick import (
    as_layers,
    elevation_of,
    nearest_contour,
)


class SelectContourTool(QgsMapTool):
    """
    Map tool that lets the user click on a contour line to select it.
    Emits contour_selected(QgsGeometry, float elevation, list contour_coords)
    on click — the coords are the full contour polyline [(x, y), ...] used as
    reshape provenance. Picks the geometrically nearest feature within the
    search radius so it works correctly on flat single-layer ranked contour
    outputs.

    Takes one contour layer or a list of them, so a contour is pickable whether
    or not the analysis ranked it. A single layer is still accepted.
    """

    contour_selected = pyqtSignal(object, float, object)   # QgsGeometry, elevation, coords
    cancelled = pyqtSignal()

    def __init__(self, canvas, contour_layers):
        super().__init__(canvas)
        self._canvas = canvas
        self._layers = as_layers(contour_layers)
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()
            return

        point = self.toMapCoordinates(event.pos())
        # Search radius: 12 pixels in map units
        radius = self._canvas.mapUnitsPerPixel() * 12
        best_feature, alive = nearest_contour(self._layers, point, radius)
        if not alive:
            # Every contour layer was deleted/swapped while this tool was active.
            self.cancelled.emit()
            return
        if best_feature is None:
            return

        geom = best_feature.geometry()
        elev = elevation_of(best_feature)
        contour_coords = self._geometry_coords(geom)
        self.contour_selected.emit(geom, elev, contour_coords)

    @staticmethod
    def _geometry_coords(geom):
        """Full contour polyline coords (longest part of a multi-line), or None."""
        try:
            import json

            from shapely.geometry import shape as _shape
            shp = _shape(json.loads(geom.asJson()))
            if shp.geom_type == "MultiLineString":
                shp = max(shp.geoms, key=lambda g: g.length)
            return [(float(x), float(y)) for x, y in shp.coords]
        except Exception:
            return None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
