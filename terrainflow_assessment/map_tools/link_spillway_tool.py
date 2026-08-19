"""
link_spillway_tool.py — give a diversion drain its start level from a spillway.

Two clicks: the **end of the drain** that attaches, then the feature whose spillway
supplies the level. Emits ``link_made(drain_id, end, source_id)`` where *end* is
``"start"`` or ``"end"`` — which of the drain's own vertices the first click landed
nearest.

**Neither click picks a feature**, and that is the whole reason this tool exists rather
than reusing ``ConnectEarthworksTool``.

The **first** click picks a drain *endpoint*. ``_burn_diversion`` runs its grade down
from the linked end, so which end was clicked decides which way the drain falls. A drain
graded from the wrong end runs uphill from an entirely plausible-looking level — no
error, no warning and no obviously wrong number, just a channel that does not carry
water.

The **second** click picks a *spillway*, not the feature carrying it. A feature carries
two — an outflow and an inlet — and they sit a few metres apart on the same bank, so
hit-testing the feature could not tell them apart and would be right only by accident.
It also means "click the outflow spillway" is literally what happens.

Both are marked on the canvas while the tool is armed, one step at a time: a click radius
of fourteen pixels around a point is not discoverable on its own, and showing only what
is pickable *now* says which half of the gesture the user is in.

Candidates arrive as tuples from the controller rather than as a layer, so hit-testing
runs against the earthworks the model actually holds — the same arrangement
``connect_earthworks_tool`` uses.
"""

from qgis.core import QgsPointXY, QgsWkbTypes
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QCursor

_PREVIEW = QColor(18, 115, 181, 200)
_LINKED = QColor(95, 113, 118, 160)
_SILL = QColor(18, 115, 181, 220)
_SEARCH_PIXELS = 14
_END_MARKER_PIXELS = 7
_SILL_MARKER_PIXELS = 9


class LinkSpillwayTool(QgsMapTool):
    """Pick the end of a diversion drain, then the spillway it starts at."""

    link_made = pyqtSignal(str, str, str)    # drain_id, end ('start'|'end'), source_id
    drain_picked = pyqtSignal(str, str)      # name, end — for the mid-link status message
    cancelled = pyqtSignal()

    def __init__(self, canvas, drains, sources):
        """*drains* are ``(id, name, geometry, already_linked)``; *sources* are
        ``(id, name, sill QgsPointXY, crest_elevation)`` — the spillway **point**, not
        the feature carrying it."""
        super().__init__(canvas)
        self._canvas = canvas
        self._drains = list(drains or [])
        self._sources = list(sources or [])
        self._picked = None          # (drain_id, name, end, anchor QgsPointXY)
        self._band = None
        self._markers = None
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
            self._start_band(hit[3])
            self._show_spillways()
            self.drain_picked.emit(hit[1], hit[2])
            return

        source = self._source_at(point)
        if source is None:
            return
        drain_id, _name, end, _anchor = self._picked
        self._picked = None
        self._clear_band()
        self._clear_markers()
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
        self._clear_markers()
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
        self._clear_markers()
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
        """Nearest **spillway point** within the search radius → its feature's id.

        Measured to the sill rather than to the feature it sits on. A feature carries an
        outflow and an inlet a few metres apart on the same bank, and only the outflow is
        a source of water — hit-testing the feature could not tell them apart, so a click
        aimed at the outflow would be right by accident and wrong as soon as the two were
        placed near each other.
        """
        radius = self._canvas.mapUnitsPerPixel() * _SEARCH_PIXELS
        click = QgsPointXY(point)
        best, best_dist = None, float("inf")
        for source_id, _name, sill, _crest in self._sources:
            if sill is None:
                continue
            dist = click.distance(QgsPointXY(sill))
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
        self._clear_markers()
        self._markers = [
            self._marker(vertex, _LINKED if linked else _PREVIEW, _END_MARKER_PIXELS)
            for _id, _name, geom, linked in self._drains
            for vertex in (self._endpoints(geom) if geom is not None else ()) or ()
        ]

    def _show_spillways(self):
        """Once an end is picked, mark the sills — the only things now pickable.

        Swapping the markers rather than showing both at once is what says which half of
        the gesture the user is in. Without it the canvas carries a field of dots and the
        second click is a guess.
        """
        self._clear_markers()
        self._markers = [
            self._marker(sill, _SILL, _SILL_MARKER_PIXELS)
            for _id, _name, sill, _crest in self._sources if sill is not None
        ]

    def _marker(self, point, colour, size):
        band = QgsRubberBand(self._canvas, QgsWkbTypes.PointGeometry)
        band.setColor(colour)
        band.setWidth(2)
        band.setIconSize(size)
        band.addPoint(QgsPointXY(point))
        return band

    def _clear_markers(self):
        for band in self._markers or ():
            try:
                self._canvas.scene().removeItem(band)
            except Exception:
                pass
        self._markers = None

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
