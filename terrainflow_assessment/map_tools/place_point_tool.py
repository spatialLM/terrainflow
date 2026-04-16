from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.gui import QgsMapTool
from qgis.core import QgsPointXY


class PlacePointTool(QgsMapTool):
    """
    Single-click map tool for placing a point (e.g. spillway location).
    Emits point_placed(QgsPointXY) then deactivates.
    """
    point_placed = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas

    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pt = QgsPointXY(self.toMapCoordinates(event.pos()))
            self.point_placed.emit(pt)
            self.canvas.unsetMapTool(self)
        elif event.button() == Qt.RightButton:
            self.cancelled.emit()
            self.canvas.unsetMapTool(self)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.canvas.unsetMapTool(self)
