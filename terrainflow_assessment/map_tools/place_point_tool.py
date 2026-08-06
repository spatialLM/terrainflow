from qgis.core import QgsPointXY
from qgis.gui import QgsMapTool, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

# Same pixel tolerance the reshape tool uses for grabbing a vertex. Sharing the
# number keeps "near enough to count" meaning one thing across every tool that
# asks the question.
_HIT_TOLERANCE_PX = 12


class PlacePointTool(QgsMapTool):
    """
    Single-click map tool for placing a point (e.g. a spillway location).

    Emits ``point_placed(QgsPointXY, elevation)`` then deactivates. *elevation* is
    sampled from *snap_raster_path* when one is given, and ``None`` otherwise — a
    spillway sited without an elevation is a location with no crest behind it, so
    the caller needs to know which it got rather than being handed a silent 0.0.

    Pass *constrain_to* (a ``QgsGeometry``) to require the click to land on that
    feature. A click within tolerance is **snapped onto** the geometry; one
    outside is refused via ``rejected(distance_m)`` and the tool stays armed.

    The constraint is not cosmetic. The caller samples the crest elevation at the
    point this tool emits, so an unconstrained click seeds crest, freeboard and
    required weir width from whatever ground happened to be under the cursor —
    a spillway sited in a paddock 200 m away is sized from that paddock.
    """
    point_placed = pyqtSignal(object, object)   # QgsPointXY, float | None
    rejected = pyqtSignal(float)                # metres from the target feature
    cancelled = pyqtSignal()

    def __init__(self, canvas, snap_raster_path=None, constrain_to=None,
                 tolerance_px=_HIT_TOLERANCE_PX):
        super().__init__(canvas)
        self.canvas = canvas
        self._snap_raster_path = snap_raster_path
        self._constrain_to = constrain_to
        self._tolerance_px = tolerance_px
        self._marker = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # ------------------------------------------------------------------ input

    def canvasMoveEvent(self, event):
        """Show where the click will actually land.

        Snapping that only happens on release is snapping the user cannot see;
        the marker makes the constraint visible before it is committed.
        """
        if self._constrain_to is None:
            return
        pt = QgsPointXY(self.toMapCoordinates(event.pos()))
        snapped, distance = self._snap(pt)
        if snapped is not None and distance <= self._tolerance_m():
            self._show_marker(snapped)
        else:
            self._hide_marker()

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._hide_marker()
            self.cancelled.emit()
            self.canvas.unsetMapTool(self)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        pt = QgsPointXY(self.toMapCoordinates(event.pos()))

        if self._constrain_to is not None:
            snapped, distance = self._snap(pt)
            if snapped is None or distance > self._tolerance_m():
                # Stay armed. A mis-click should cost one more click, not a trip
                # back through the menu — the same call connect_earthworks_tool
                # makes when a click lands on nothing.
                self.rejected.emit(float(distance))
                return
            pt = snapped

        self._hide_marker()
        self.point_placed.emit(pt, self._elevation_at(pt))
        self.canvas.unsetMapTool(self)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._hide_marker()
            self.cancelled.emit()
            self.canvas.unsetMapTool(self)

    def deactivate(self):
        self._hide_marker()
        super().deactivate()

    # ------------------------------------------------------------------ snapping

    def _tolerance_m(self):
        return self.canvas.mapUnitsPerPixel() * self._tolerance_px

    def _snap(self, pt):
        """Nearest point on the constraining geometry, and how far the click was.

        Returns ``(None, inf)`` when the geometry cannot answer, so a failure
        reads as "not on the feature" rather than as a snap to the origin.
        """
        if self._constrain_to is None:
            return None, float("inf")
        try:
            distance_sq, closest, _after, _side = \
                self._constrain_to.closestSegmentWithContext(pt)
            if closest is None:
                return None, float("inf")
            return QgsPointXY(closest), float(distance_sq) ** 0.5
        except Exception:
            return None, float("inf")

    # ------------------------------------------------------------------ marker

    def _show_marker(self, pt):
        if self._marker is None:
            self._marker = QgsVertexMarker(self.canvas)
            self._marker.setIconType(QgsVertexMarker.ICON_CROSS)
            self._marker.setColor(QColor(20, 120, 200))
            self._marker.setIconSize(14)
            self._marker.setPenWidth(3)
        self._marker.setCenter(pt)
        self._marker.show()

    def _hide_marker(self):
        if self._marker is not None:
            try:
                self.canvas.scene().removeItem(self._marker)
            except Exception:
                pass
            self._marker = None

    # ------------------------------------------------------------------ elevation

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
