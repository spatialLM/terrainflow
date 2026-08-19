"""
link_spillway_tool.py — give a diversion drain its start level from a spillway.

Two clicks: the **end of the drain** that attaches, then the feature whose spillway
supplies the level. Emits ``link_made(drain_id, end, source_id)`` where *end* is
``"start"`` or ``"end"`` — which of the drain's own vertices the first click landed
nearest.

**The first click picks an end, not a feature**, and that is the whole reason this tool
exists rather than reusing ``ConnectEarthworksTool``. ``_burn_diversion`` runs its grade
down from the linked end, so which end was clicked decides which way the drain falls. A
drain graded from the wrong end runs uphill from an entirely plausible-looking level —
there is no error, no warning and no obviously wrong number, just a channel that does not
carry water. Recording the end here, and drawing the rubber band from that exact vertex,
is what makes the choice visible while it is being made.

The candidate features arrive as ``(id, name, QgsGeometry, ...)`` tuples from the
controller rather than as a layer, so hit-testing runs against the earthworks the model
actually holds — the same arrangement ``connect_earthworks_tool`` uses.
"""

from qgis.core import QgsGeometry, QgsPointXY, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

_PREVIEW = QColor(18, 115, 181, 200)
_LINKED = QColor(95, 113, 118, 160)
_SEARCH_PIXELS = 14
_END_MARKER_PIXELS = 7


class LinkSpillwayTool(QgsMapTool):
    """Pick the end of a diversion drain, then the spillway it starts at."""

    link_made = pyqtSignal(str, str, str)    # drain_id, end ('start'|'end'), source_id
    drain_picked = pyqtSignal(str, str)      # name, end — for the mid-link status message
    cancelled = pyqtSignal()

    def __init__(self, canvas, drains, sources):
        """*drains* are ``(id, name, geometry, already_linked)``; *sources*
        ``(id, name, geometry)`` for features carrying a sited outflow spillway."""
        super().__init__(canvas)
        self._canvas = canvas
        self._drains = list(drains or [])
        self._sources = list(sources or [])
        self._picked = None          # (drain_id, name, end, anchor QgsPointXY)
        self._band = None
        self._ends = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    # ------------------------------------------------------------------ events

    def activate(self):
        super().activate()
        self._show_drain_ends()

    def canvasPressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._abort()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        point = self.toMapCoordinates(event.pos())
        if self._picked is None:
            hit = self._drain_end_at(point)
            if hit is None:
                return
            self._picked = hit
            self._clear_ends()
            self._start_band(hit[3])
            self.drain_picked.emit(hit[1], hit[2])
            return

        source = self._source_at(point)
        if source is None:
            return
        drain_id, _name, end, _anchor = self._picked
        self._picked = None
        self._clear_band()
        self.link_made.emit(drain_id, end, source)
        self._canvas.unsetMapTool(self)

    def canvasMoveEvent(self, event):
        if self._band is None or self._picked is None:
            return
        self._band.movePoint(1, self.toMapCoordinates(event.pos()))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._abort()

    def deactivate(self):
        self._clear_band()
        self._clear_ends()
        self._picked = None
        super().deactivate()

    # ------------------------------------------------------------------ internals

    def _abort(self):
        """Escape backs out of a half-made link before it abandons the tool.

        Picking the wrong end is the likely slip here, and it is the one that matters,
        so one keystroke has to undo it without a trip back through the menu.
        """
        if self._picked is not None:
            self._clear_band()
            self._picked = None
            self._show_drain_ends()
            return
        self.cancelled.emit()
        self._canvas.unsetMapTool(self)

    @staticmethod
    def _endpoints(geom):
        """``(first, last)`` vertices of a line geometry, or ``None``."""
        try:
            line = geom.asPolyline()
            if not line:
                parts = geom.asMultiPolyline()
                line = parts[0] if parts else []
            if len(line) < 2:
                return None
            return QgsPointXY(line[0]), QgsPointXY(line[-1])
        except Exception:
            return None

    def _drain_end_at(self, point):
        """Nearest drain **endpoint** within the search radius.

        Measured to the endpoints rather than to the alignment: the click is choosing
        which end attaches, so a click nearer the middle of a long drain is not a
        useful answer to that question and is better ignored than rounded to one end.
        """
        radius = self._canvas.mapUnitsPerPixel() * _SEARCH_PIXELS
        click = QgsPointXY(point)
        best, best_dist = None, float("inf")
        for drain_id, name, geom, _linked in self._drains:
            if geom is None:
                continue
            ends = self._endpoints(geom)
            if ends is None:
                continue
            for label, vertex in (("start", ends[0]), ("end", ends[1])):
                dist = click.distance(vertex)
                if dist < best_dist:
                    best, best_dist = (drain_id, name, label, vertex), dist
        return best if best is not None and best_dist <= radius else None

    def _source_at(self, point):
        """Nearest spillway-carrying feature within the search radius → its id."""
        radius = self._canvas.mapUnitsPerPixel() * _SEARCH_PIXELS
        click = QgsGeometry.fromPointXY(QgsPointXY(point))
        best, best_dist = None, float("inf")
        for source_id, _name, geom in self._sources:
            if geom is None:
                continue
            try:
                dist = geom.distance(click)
            except Exception:
                continue
            if dist < best_dist:
                best, best_dist = source_id, dist
        return best if best is not None and best_dist <= radius else None

    def _show_drain_ends(self):
        """Mark every drain end that can be picked, so the target is not invisible.

        A click radius of fourteen pixels around a vertex is not discoverable on its
        own, and the difference between the two ends is the decision this tool is for.
        Ends of a drain that is already linked are drawn muted rather than hidden —
        re-picking one is how the link is moved or removed.
        """
        self._clear_ends()
        self._ends = []
        for _id, _name, geom, linked in self._drains:
            ends = self._endpoints(geom) if geom is not None else None
            if ends is None:
                continue
            for vertex in ends:
                band = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
                band.setColor(_LINKED if linked else _PREVIEW)
                band.setWidth(2)
                band.setIconSize(_END_MARKER_PIXELS)
                band.addPoint(QgsPointXY(vertex))
                self._ends.append(band)

    def _clear_ends(self):
        for band in self._ends or ():
            try:
                self._canvas.scene().removeItem(band)
            except Exception:
                pass
        self._ends = None

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
