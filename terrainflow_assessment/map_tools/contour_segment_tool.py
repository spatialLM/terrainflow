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

from qgis.core import (
    QgsGeometry,
    QgsPointXY,
    QgsWkbTypes,
)
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

from terrainflow_assessment.map_tools._contour_pick import (
    as_layers,
    elevation_of,
    nearest_contour,
)


class ContourSegmentTool(QgsMapTool):
    """
    Three-click contour-segment selector.

    Takes one contour layer or a list of them, so a segment can be cut from any
    contour on screen rather than only from one the analysis ranked.

    Signals
    -------
    segment_selected(QgsGeometry, float, list)
        Emitted with the sub-line geometry, the contour's elevation, and the
        full contour polyline coords [(x, y), ...] (reshape provenance).
    cancelled()
    """

    segment_selected = pyqtSignal(object, float, object)  # geometry, elevation, contour coords
    cancelled = pyqtSignal()

    _HINT = [
        "Click a contour line to select it",
        "Click the START point of the swale on the contour",
        "Click the END point of the swale on the contour",
    ]

    def __init__(self, canvas, contour_layers, status_bar=None):
        super().__init__(canvas)
        self._canvas = canvas
        self._layers = as_layers(contour_layers)
        self._status_bar = status_bar   # optional QStatusBar for hints
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

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
        if event.button() == Qt.MouseButton.RightButton:
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
        if event.key() == Qt.Key.Key_Escape:
            self._cleanup()
            self.cancelled.emit()

    def deactivate(self):
        self._cleanup()
        self._discard_scene_items()
        super().deactivate()

    def _discard_scene_items(self):
        """Take the two rubber bands and the marker out of the canvas scene.

        `_cleanup` resets the bands and *hides* the marker, which leaves all three
        QGraphicsItems parented to the scene. `use_tool` then drops this tool, their
        only reference, so every segment pick orphaned three invisible items for the
        canvas's lifetime — and they survive unload. Guarded for re-entry: deactivate
        can arrive twice, and a tool with no items is simply already clean.
        """
        for attr in ("_rb_contour", "_rb_segment", "_marker"):
            item = getattr(self, attr, None)
            if item is None:
                continue
            try:
                self._canvas.scene().removeItem(item)
            except Exception:
                pass
            setattr(self, attr, None)

    # ---------------------------------------------------------------- phases

    def _try_select_contour(self, pt):
        radius = self._canvas.mapUnitsPerPixel() * 12
        best_feature, alive = nearest_contour(self._layers, pt, radius)
        if not alive:
            # Every contour layer was deleted/swapped while this tool was active.
            self._cleanup()
            self.cancelled.emit()
            return
        if best_feature is None:
            return

        geom = best_feature.geometry()
        elev = elevation_of(best_feature)

        # Convert to shapely — handle multi-geometry by taking longest part
        try:
            from shapely.geometry import shape as _shape
            shp = _shape(json.loads(geom.asJson()))
            if shp.geom_type == "MultiLineString":
                shp = max(shp.geoms, key=lambda g: g.length)
        except Exception:
            return

        self._contour_shp = shp
        self._contour_geom = geom
        self._elevation = elev

        # Highlight the full contour
        if self._rb_contour is not None:
            self._rb_contour.setToGeometry(geom, None)
        self._phase = 1
        self._show_hint(1)

    def _place_start(self, pt):
        dist = self._project_onto_contour(pt.x(), pt.y())
        snapped = self._contour_shp.interpolate(dist)
        self._start_dist = dist

        if self._marker is not None:
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
        # Full contour provenance (before cleanup clears it) — lets the reshape
        # tool slide the swale's endpoints along this contour later.
        contour_coords = [(float(x), float(y)) for x, y in self._contour_shp.coords]

        self._cleanup()
        self.segment_selected.emit(qgs_geom, elev, contour_coords)

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
        if self._rb_segment is None:
            return
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
        if self._rb_contour is not None:
            self._rb_contour.reset(QgsWkbTypes.LineGeometry)
        if self._rb_segment is not None:
            self._rb_segment.reset(QgsWkbTypes.LineGeometry)
        if self._marker is not None:
            self._marker.setVisible(False)
        try:
            from qgis.utils import iface
            iface.mainWindow().statusBar().clearMessage()
        except Exception:
            pass
        self._phase = 0
        self._contour_shp = None
        self._start_dist = None
        # Cleared too. Nothing reads these before `_try_select_contour` writes all
        # three together, so this is latent rather than live — but leaving one
        # gesture's elevation and QgsGeometry lying about for the next one makes a
        # reset that resets most of the state, which is worse than none because it
        # reads as complete.
        self._contour_geom = None
        self._elevation = 0.0
