from qgis.core import QgsGeometry, QgsPointXY, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand, QgsVertexMarker
from qgis.PyQt.QtCore import Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.utils import iface

from terrainflow_assessment.modules.swale_design import contour_section


class EditEarthworkTool(QgsMapTool):
    """
    Map tool for reshaping earthworks by dragging their vertices.

    Left-press near a vertex grabs it; dragging moves it with a rubber-band
    preview, emitting a throttled geometry_edited so the Live Assessment
    updates *during* the drag (design record: throttle only the heavy
    flow-dependent recompute, never the visual). Release emits edit_finished
    for the exact recompute. Double-click a segment inserts a vertex;
    Delete/Backspace removes the highlighted vertex. Esc cancels an active
    drag (restores the grabbed geometry); Esc or right-click otherwise ends
    the session.

    Contour-locked features (created from a contour, carrying the full
    contour polyline) behave differently: only their two ENDPOINTS are
    grabbable, and dragging one slides it ALONG the contour — the geometry
    re-derives as the contour section between the endpoints, so the swale
    always keeps following its contour. Insert/delete are disabled for them.
    """

    geometry_edited = pyqtSignal(int, object)   # throttled, during drag (live tier)
    edit_finished = pyqtSignal(int, object)     # on release / insert / delete (exact tier)
    session_ended = pyqtSignal()

    _THROTTLE_MS = 80        # live-recompute cadence during a drag
    _HIT_TOLERANCE_PX = 12   # vertex grab radius in screen pixels

    def __init__(self, canvas, earthworks):
        """earthworks: list of (index, QgsGeometry, ew_type, contour_coords_or_None)."""
        super().__init__(canvas)
        self.canvas = canvas
        # Working copies — edits accumulate here across the session; the
        # controller owns the real Earthwork objects and hears about changes
        # only via the signals above.
        self._items = [
            (idx, QgsGeometry(geom), ew_type, contour)
            for idx, geom, ew_type, contour in earthworks
        ]

        self._dragging = False
        self._drag_item = None       # position in self._items
        self._drag_vertex = None     # vertex index within the geometry
        self._drag_backup = None     # geometry before the grab (Esc restore)
        self._drag_other_xy = None   # contour-locked: the fixed endpoint (x, y)
        self._hover_item = None
        self._hover_vertex = None

        self._rubber_band = None
        self._vertex_markers = []

        self._throttle_timer = QTimer()
        self._throttle_timer.setSingleShot(True)
        self._throttle_timer.setInterval(self._THROTTLE_MS)
        self._throttle_timer.timeout.connect(self._flush_pending_edit)
        self._pending_emit = False

        self._show_hint()

    # ------------------------------------------------------------------ hints

    def _show_hint(self):
        iface.mainWindow().statusBar().showMessage(
            "✥  Reshape earthworks  —  Drag a vertex: move "
            "(contour swales: endpoints slide along the contour)  |  "
            "Double-click a segment: insert vertex  |  Del: delete vertex  |  "
            "Right-click / Esc: finish"
        )

    # ------------------------------------------------------------------ hit-testing

    def _tolerance(self):
        return self.canvas.mapUnitsPerPixel() * self._HIT_TOLERANCE_PX

    @staticmethod
    def _vertex_points(geom):
        """All vertices of a geometry as [(vertex_idx, x, y)]."""
        return [(i, v.x(), v.y()) for i, v in enumerate(geom.vertices())]

    def _grabbable_vertices(self, geom, contour):
        """Vertices the user may grab: all for freehand, endpoints only when locked."""
        pts = self._vertex_points(geom)
        if contour is not None and len(pts) > 2:
            return [pts[0], pts[-1]]
        return pts

    def _nearest_vertex(self, map_pt):
        """(item_pos, vertex_idx) of the closest grabbable vertex within tolerance."""
        tol2 = self._tolerance() ** 2
        best = (None, None, tol2)
        for pos, (_, geom, _, contour) in enumerate(self._items):
            for vidx, x, y in self._grabbable_vertices(geom, contour):
                sq_dist = (x - map_pt.x()) ** 2 + (y - map_pt.y()) ** 2
                if sq_dist <= best[2]:
                    best = (pos, vidx, sq_dist)
        return best[0], best[1]

    def _nearest_segment(self, map_pt):
        """(item_pos, after_vertex, point) of the closest freehand segment within tolerance."""
        tol2 = self._tolerance() ** 2
        best = (None, None, None, tol2)
        for pos, (_, geom, _, contour) in enumerate(self._items):
            if contour is not None:
                continue  # contour-locked shapes derive their vertices — no inserts
            try:
                sq_dist, min_pt, after_vertex, _ = geom.closestSegmentWithContext(
                    QgsPointXY(map_pt)
                )
            except Exception:
                continue
            if after_vertex >= 0 and sq_dist <= best[3]:
                best = (pos, after_vertex, min_pt, sq_dist)
        return best[0], best[1], best[2]

    # ------------------------------------------------------------------ visuals

    def _clear_markers(self):
        for m in self._vertex_markers:
            self.canvas.scene().removeItem(m)
        self._vertex_markers = []

    def _show_vertex_markers(self, item_pos, active_vertex=None):
        """Mark the item's grabbable vertices; the active one highlighted."""
        self._clear_markers()
        _, geom, _, contour = self._items[item_pos]
        for i, x, y in self._grabbable_vertices(geom, contour):
            marker = QgsVertexMarker(self.canvas)
            marker.setCenter(QgsPointXY(x, y))
            marker.setIconType(QgsVertexMarker.ICON_BOX)
            if i == active_vertex:
                marker.setColor(QColor(230, 60, 40))
                marker.setIconSize(12)
                marker.setPenWidth(3)
            else:
                marker.setColor(QColor(0, 120, 220))
                marker.setIconSize(9)
                marker.setPenWidth(2)
            self._vertex_markers.append(marker)

    def _ensure_rubber_band(self, geom):
        geom_type = QgsWkbTypes.geometryType(geom.wkbType())
        if self._rubber_band is not None:
            self._rubber_band.reset(geom_type)
        else:
            self._rubber_band = QgsRubberBand(self.canvas, geom_type)
            self._rubber_band.setColor(QColor(230, 130, 30, 180))
            self._rubber_band.setWidth(3)
        self._rubber_band.setToGeometry(geom, None)

    def _clear_rubber_band(self):
        if self._rubber_band is not None:
            self._rubber_band.reset(QgsWkbTypes.LineGeometry)
            self.canvas.scene().removeItem(self._rubber_band)
            self._rubber_band = None

    # ------------------------------------------------------------------ throttle

    def _emit_live_edit(self):
        """Leading-edge throttle: first move emits instantly, then ≤1 per interval."""
        if not self._throttle_timer.isActive():
            self._do_emit_live()
            self._throttle_timer.start()
        else:
            self._pending_emit = True

    def _flush_pending_edit(self):
        if self._pending_emit and self._dragging:
            self._do_emit_live()
            self._pending_emit = False
            self._throttle_timer.start()

    def _do_emit_live(self):
        idx, geom, _, _ = self._items[self._drag_item]
        self.geometry_edited.emit(idx, QgsGeometry(geom))

    # ------------------------------------------------------------------ events

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._end_session()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        map_pt = self.toMapCoordinates(event.pos())
        item_pos, vertex_idx = self._nearest_vertex(map_pt)
        if item_pos is None:
            return
        _, geom, _, contour = self._items[item_pos]
        self._dragging = True
        self._drag_item = item_pos
        self._drag_vertex = vertex_idx
        self._drag_backup = QgsGeometry(geom)
        self._pending_emit = False
        self._drag_other_xy = None
        if contour is not None:
            # The endpoint NOT being dragged stays fixed; the section re-derives
            # between it and wherever the cursor projects onto the contour.
            pts = self._vertex_points(geom)
            other = pts[-1] if vertex_idx == 0 else pts[0]
            self._drag_other_xy = (other[1], other[2])
        self._ensure_rubber_band(geom)
        self._show_vertex_markers(item_pos, vertex_idx)

    def canvasMoveEvent(self, event):
        map_pt = self.toMapCoordinates(event.pos())
        if self._dragging:
            if self._drag_other_xy is not None:
                self._drag_along_contour(map_pt)
            else:
                self._drag_free_vertex(map_pt)
            return
        # Hover: highlight the grabbable vertex under the cursor.
        item_pos, vertex_idx = self._nearest_vertex(map_pt)
        if item_pos is None:
            self._hover_item = self._hover_vertex = None
            self._clear_markers()
            return
        if (item_pos, vertex_idx) != (self._hover_item, self._hover_vertex):
            self._hover_item, self._hover_vertex = item_pos, vertex_idx
            self._show_vertex_markers(item_pos, vertex_idx)

    def _drag_free_vertex(self, map_pt):
        idx, geom, ew_type, contour = self._items[self._drag_item]
        if geom.moveVertex(map_pt.x(), map_pt.y(), self._drag_vertex):
            self._items[self._drag_item] = (idx, geom, ew_type, contour)
            self._rubber_band.setToGeometry(geom, None)   # instant visual
            self._show_vertex_markers(self._drag_item, self._drag_vertex)
            self._emit_live_edit()                        # throttled recompute

    def _drag_along_contour(self, map_pt):
        """Contour-locked drag: re-derive the section between cursor and fixed end."""
        idx, _, ew_type, contour = self._items[self._drag_item]
        section = contour_section(
            contour, (map_pt.x(), map_pt.y()), self._drag_other_xy
        )
        if not section or len(section) < 2:
            return
        new_geom = QgsGeometry.fromPolylineXY([QgsPointXY(x, y) for x, y in section])
        self._items[self._drag_item] = (idx, new_geom, ew_type, contour)
        # The dragged endpoint is whichever end of the new section is nearer the cursor.
        d_first = (section[0][0] - map_pt.x()) ** 2 + (section[0][1] - map_pt.y()) ** 2
        d_last = (section[-1][0] - map_pt.x()) ** 2 + (section[-1][1] - map_pt.y()) ** 2
        self._drag_vertex = 0 if d_first <= d_last else len(section) - 1
        self._rubber_band.setToGeometry(new_geom, None)   # instant visual
        self._show_vertex_markers(self._drag_item, self._drag_vertex)
        self._emit_live_edit()                            # throttled recompute

    def canvasReleaseEvent(self, event):
        if not self._dragging or event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = False
        self._throttle_timer.stop()
        self._pending_emit = False
        idx, geom, _, _ = self._items[self._drag_item]
        self._clear_rubber_band()
        self._show_vertex_markers(self._drag_item)
        self.edit_finished.emit(idx, QgsGeometry(geom))
        self._drag_item = self._drag_vertex = self._drag_backup = None
        self._drag_other_xy = None

    def canvasDoubleClickEvent(self, event):
        """Insert a vertex on the nearest freehand segment (not near an existing vertex)."""
        if self._dragging:
            return
        map_pt = self.toMapCoordinates(event.pos())
        near_item, _ = self._nearest_vertex(map_pt)
        if near_item is not None:
            return  # double-clicked a vertex, not a segment — nothing to insert
        item_pos, after_vertex, seg_pt = self._nearest_segment(map_pt)
        if item_pos is None:
            return
        idx, geom, ew_type, contour = self._items[item_pos]
        if geom.insertVertex(seg_pt.x(), seg_pt.y(), after_vertex):
            self._items[item_pos] = (idx, geom, ew_type, contour)
            self._show_vertex_markers(item_pos, after_vertex)
            self.edit_finished.emit(idx, QgsGeometry(geom))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self._dragging:
                self._cancel_drag()
            else:
                self._end_session()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_hovered_vertex()

    # ------------------------------------------------------------------ actions

    def _delete_hovered_vertex(self):
        if self._dragging or self._hover_item is None:
            return
        idx, geom, ew_type, contour = self._items[self._hover_item]
        if contour is not None:
            iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Contour swale vertices are derived from the contour — slide the "
                "endpoints instead.",
            )
            return
        trial = QgsGeometry(geom)
        if not trial.deleteVertex(self._hover_vertex):
            return
        if trial.isEmpty() or not trial.isGeosValid():
            iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Can't delete that vertex — the shape needs it.",
            )
            return
        self._items[self._hover_item] = (idx, trial, ew_type, contour)
        self._show_vertex_markers(self._hover_item)
        self._hover_vertex = None
        self.edit_finished.emit(idx, QgsGeometry(trial))

    def _cancel_drag(self):
        """Esc mid-drag: restore the grabbed geometry (undoes throttled live edits)."""
        self._dragging = False
        self._throttle_timer.stop()
        self._pending_emit = False
        idx, _, ew_type, contour = self._items[self._drag_item]
        self._items[self._drag_item] = (idx, QgsGeometry(self._drag_backup), ew_type, contour)
        self._clear_rubber_band()
        self._show_vertex_markers(self._drag_item)
        self.edit_finished.emit(idx, QgsGeometry(self._drag_backup))
        self._drag_item = self._drag_vertex = self._drag_backup = None
        self._drag_other_xy = None

    def _end_session(self):
        self._cleanup()
        self.session_ended.emit()

    def _cleanup(self):
        self._throttle_timer.stop()
        self._pending_emit = False
        self._dragging = False
        self._clear_rubber_band()
        self._clear_markers()

    def deactivate(self):
        self._cleanup()
        if iface:
            iface.mainWindow().statusBar().clearMessage()
        super().deactivate()
