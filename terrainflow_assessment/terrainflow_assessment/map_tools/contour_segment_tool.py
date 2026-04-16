"""
contour_segment_tool.py — Three-click tool for selecting a segment of a contour line.

Flow
----
Phase 0  Click near a contour → contour is highlighted.
Phase 1  Click to place the start point (snapped to contour).
Phase 2  Move to preview the segment; click to place the end point → emits segment.

Right-click or Escape cancels at any phase.
"""

import json

from qgis.PyQt.QtCore import pyqtSignal, Qt
from qgis.PyQt.QtGui import QCursor, QColor
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.core import (
    QgsFeatureRequest, QgsRectangle, QgsWkbTypes,
    QgsGeometry, QgsPointXY,
)


class ContourSegmentTool(QgsMapTool):
    """
    Three-click contour-segment selector.

    Signals
    -------
    segment_selected(QgsGeometry, float)
        Emitted with the sub-line geometry and the contour's elevation.
    cancelled()
    """

    segment_selected = pyqtSignal(object, float)
    cancelled = pyqtSignal()

    _HINT = [
        "Click a contour line to select it",
        "Click the START point of the swale on the contour",
        "Click the END point of the swale on the contour",
    ]

    def __init__(self, canvas, contour_layer, status_bar=None):
        super().__init__(canvas)
        self._canvas = canvas
        self._layer = contour_layer
        self._status_bar = status_bar   # optional QStatusBar for hints
        self.setCursor(QCursor(Qt.CrossCursor))

        # State
        self._phase = 0
        self._contour_shp = None        # shapely geometry of selected contour
        self._contour_geom = None       # QgsGeometry of selected contour
        self._elevation = 0.0
        self._start_dist = None         # distance along contour of start point

        # Rubber bands
        self._rb_contour = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        self._rb_contour.setColor(QColor(255, 165, 0, 200))
        self._rb_contour.setWidth(3)

        self._rb_segment = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        self._rb_segment.setColor(QColor(0, 180, 80, 220))
        self._rb_segment.setWidth(3)

        # Start-point marker
        self._marker = QgsVertexMarker(canvas)
        self._marker.setColor(QColor(0, 180, 80))
        self._marker.setIconType(QgsVertexMarker.ICON_CROSS)
        self._marker.setIconSize(12)
        self._marker.setPenWidth(2)
        self._marker.setVisible(False)

        self._show_hint(0)

    # ---------------------------------------------------------------- events

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._cleanup()
            self.cancelled.emit()
            return

        pt = self.toMapCoordinates(event.pos())

        if self._phase == 0:
            self._try_select_contour(pt)
        elif self._phase == 1:
            self._place_start(pt)
        elif self._phase == 2:
            self._place_end(pt)

    def canvasMoveEvent(self, event):
        if self._phase != 2 or self._contour_shp is None:
            return
        pt = self.toMapCoordinates(event.pos())
        end_dist = self._project_onto_contour(pt.x(), pt.y())
        self._update_segment_preview(self._start_dist, end_dist)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._cleanup()
            self.cancelled.emit()

    def deactivate(self):
        self._cleanup()
        super().deactivate()

    # ---------------------------------------------------------------- phases

    def _try_select_contour(self, pt):
        radius = self._canvas.mapUnitsPerPixel() * 12
        rect = QgsRectangle(
            pt.x() - radius, pt.y() - radius,
            pt.x() + radius, pt.y() + radius,
        )
        # Find all candidates, pick geometrically nearest
        click_geom = QgsGeometry.fromPointXY(pt)
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

        # Convert to shapely — handle multi-geometry by taking longest part
        try:
            from shapely.geometry import shape as _shape, MultiLineString
            shp = _shape(json.loads(geom.asJson()))
            if shp.geom_type == "MultiLineString":
                shp = max(shp.geoms, key=lambda g: g.length)
        except Exception:
            return

        self._contour_shp = shp
        self._contour_geom = geom
        self._elevation = elev

        # Highlight the full contour
        self._rb_contour.setToGeometry(geom, None)
        self._phase = 1
        self._show_hint(1)

    def _place_start(self, pt):
        dist = self._project_onto_contour(pt.x(), pt.y())
        snapped = self._contour_shp.interpolate(dist)
        self._start_dist = dist

        self._marker.setCenter(QgsPointXY(snapped.x, snapped.y))
        self._marker.setVisible(True)
        self._phase = 2
        self._show_hint(2)

    def _place_end(self, pt):
        end_dist = self._project_onto_contour(pt.x(), pt.y())
        if end_dist == self._start_dist:
            return  # zero-length segment — ignore

        segment_shp = self._extract_segment(self._start_dist, end_dist)
        if segment_shp is None or segment_shp.is_empty:
            return

        # Convert segment to QgsGeometry
        qgs_geom = QgsGeometry.fromWkt(segment_shp.wkt)
        elev = self._elevation

        self._cleanup()
        self.segment_selected.emit(qgs_geom, elev)

    # ---------------------------------------------------------------- helpers

    def _project_onto_contour(self, x, y):
        """Return the distance along _contour_shp nearest to (x, y)."""
        from shapely.geometry import Point
        return self._contour_shp.project(Point(x, y))

    def _extract_segment(self, d0, d1):
        """Return the shapely LineString substring between distances d0 and d1."""
        try:
            from shapely.ops import substring
            length = self._contour_shp.length
            d0 = max(0.0, min(d0, length))
            d1 = max(0.0, min(d1, length))
            if d0 > d1:
                d0, d1 = d1, d0
            return substring(self._contour_shp, d0, d1)
        except Exception:
            return None

    def _update_segment_preview(self, d0, d1):
        seg = self._extract_segment(d0, d1)
        if seg and not seg.is_empty:
            self._rb_segment.setToGeometry(QgsGeometry.fromWkt(seg.wkt), None)
        else:
            self._rb_segment.reset(QgsWkbTypes.LineGeometry)

    def _show_hint(self, phase):
        msg = self._HINT[phase]
        if self._status_bar:
            self._status_bar.showMessage(msg)
        # Also push to QGIS main status bar
        try:
            from qgis.utils import iface
            iface.mainWindow().statusBar().showMessage(
                f"TerrainFlow — {msg}  (right-click or Esc to cancel)"
            )
        except Exception:
            pass

    def _cleanup(self):
        self._rb_contour.reset(QgsWkbTypes.LineGeometry)
        self._rb_segment.reset(QgsWkbTypes.LineGeometry)
        self._marker.setVisible(False)
        try:
            from qgis.utils import iface
            iface.mainWindow().statusBar().clearMessage()
        except Exception:
            pass
        self._phase = 0
        self._contour_shp = None
        self._start_dist = None
