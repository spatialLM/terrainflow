from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.PyQt.QtGui import QCursor
from qgis.gui import QgsMapTool
from qgis.core import QgsFeatureRequest, QgsRectangle, QgsGeometry


class SelectContourTool(QgsMapTool):
    """
    Map tool that lets the user click on a contour line to select it.
    Emits contour_selected(QgsGeometry, float elevation) on click.
    Picks the geometrically nearest feature within the search radius so it
    works correctly on flat single-layer ranked contour outputs.
    """

    contour_selected = pyqtSignal(object, float)   # QgsGeometry, elevation
    cancelled = pyqtSignal()

    def __init__(self, canvas, contour_layer):
        super().__init__(canvas)
        self._canvas = canvas
        self._layer = contour_layer
        self.setCursor(QCursor(Qt.CrossCursor))

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.cancelled.emit()
            return

        point = self.toMapCoordinates(event.pos())
        # Search radius: 12 pixels in map units
        radius = self._canvas.mapUnitsPerPixel() * 12
        rect = QgsRectangle(
            point.x() - radius, point.y() - radius,
            point.x() + radius, point.y() + radius,
        )
        click_geom = QgsGeometry.fromPointXY(point)
        request = QgsFeatureRequest().setFilterRect(rect)

        best_feature = None
        best_dist = float("inf")
        for feature in self._layer.getFeatures(request):
            dist = feature.geometry().distance(click_geom)
            if dist < best_dist:
                best_dist = dist
                best_feature = feature

        if best_feature is None:
            return

        geom = best_feature.geometry()
        elev = 0.0
        for fname in ("ELEV", "elev", "elevation", "Elevation", "HEIGHT", "height"):
            if fname in best_feature.fields().names():
                try:
                    elev = float(best_feature[fname])
                except (ValueError, TypeError):
                    pass
                break
        self.contour_selected.emit(geom, elev)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
