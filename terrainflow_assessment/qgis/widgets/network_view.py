"""
network_view.py — the Live Assessment flow network.

Replaces both the flat earthwork list and the HTML readout with one view: the
earthworks drawn as a water-flow network, ordered high → low, each node showing
its type colour, elevation, storage capacity (standard ink) and actual water
held (blue), with connector rows naming where each feature's overflow goes.

Selecting a node reports its index up to the panel, so the existing
Edit / Reshape / Enable-Disable / Delete action bar keeps working unchanged.

A feature captures water two ways — by **ponding** it (``stored_m3``, which fills the
bar) and by **soaking it away** (``soaked_m3``). Both count as captured, and for a
wide, shallow feature on free-draining soil the soakage can be all of it: such a
feature ponds nothing and reads 0% full while still taking every drop that reaches
it. Both are shown, because storage alone made that look like a broken feature.

Data contract — ``set_network(nodes, edges, exit_m3)``:
  nodes : list of dicts, each
    {index, id, name, ew_type, colour, elevation, crest_elevation, capacity_m3,
     stored_m3, soaked_m3, drain_hours, fill_pct, overflowed, overflow_m3,
     catchment_m2, is_terminal, enabled, has_water}
  ``elevation`` is the ground at the feature's centroid (it orders the network);
  ``crest_elevation`` is the dam's design crest, and None for everything else.
  edges : {from_id: (to_id_or_None, is_user_link)}
  exit_m3 : total water leaving the site
"""

import logging

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from terrainflow_assessment.qgis import help_text as H

_log = logging.getLogger(__name__)

_WATER = "#1273b5"
_SOAKED = "#79b8dd"   # same token the scorecard uses for infiltrated water
_WARN = "#b9770e"
_BAD = "#c0392b"
_INK = "#22302e"
_MUTED = "#5f7176"
_FAINT = "#8fa0a4"
_GLYPH = {"swale": "∿", "basin": "▢", "dam": "▮", "berm": "⌒", "diversion": "↘"}


def _capacity_text(node):
    """Capacity as the node prints it — the measured pond, with the drawn figure beside.

    Both, whenever they differ by more than rounding. The bar fills against the measured
    one, and that figure is routinely twice the drawn section on a swale whose companion
    berm is keyed into the banks, because the bank holds water above natural ground and
    further up the hill than the trench reaches. A number that size shown alone would look
    like a mistake; shown alone in the *other* direction it caused a real one — sizing
    against the drawn section reports a swale full while most of its pond is empty.
    """
    cap = node.get("capacity_m3", 0.0) or 0.0
    drawn = node.get("drawn_capacity_m3", cap) or 0.0
    if not node.get("capacity_is_measured") or abs(cap - drawn) < max(1.0, 0.01 * drawn):
        return f"{cap:,.0f} m³"
    return f"{cap:,.0f} m³ · {drawn:,.0f} drawn"


def _capacity_tooltip(node):
    cap = node.get("capacity_m3", 0.0) or 0.0
    drawn = node.get("drawn_capacity_m3", cap) or 0.0
    if not node.get("capacity_is_measured"):
        return H.NETWORK_CAPACITY_UNMEASURED
    tip = H.NETWORK_CAPACITY_MEASURED.format(cap=cap, drawn=drawn)
    if cap > drawn * 1.2:
        tip += H.NETWORK_CAPACITY_ABOVE_GROUND
    return tip


class _FillBar(QWidget):
    """A thin fill bar; amber when the feature is full/overflowing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pct = 0.0
        self._full = False
        self.setFixedSize(44, 5)

    def set(self, pct, full):
        self._pct = max(0.0, min(100.0, pct))
        self._full = full
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#dde4e5"))
        p.drawRoundedRect(0, 0, self.width(), self.height(), 2, 2)
        w = int(self.width() * self._pct / 100.0)
        if w > 0:
            p.setBrush(QColor(_WARN if self._full else _WATER))
            p.drawRoundedRect(0, 0, max(w, 2), self.height(), 2, 2)
        p.end()


class _NodeCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.index = node["index"]
        self._selected = False
        self._colour = node["colour"]
        self._enabled = node["enabled"]
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        h = QHBoxLayout(self)
        h.setContentsMargins(9, 6, 9, 6)
        h.setSpacing(8)

        chip = QLabel(_GLYPH.get(node["ew_type"], "●"))
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chip.setFixedSize(18, 18)
        chip.setStyleSheet(
            f"background: {node['colour']}; color: white; border-radius: 4px;"
            " font-size: 10px; font-weight: bold;"
        )
        h.addWidget(chip)

        name = QLabel(node["name"])
        name.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {_INK};")
        h.addWidget(name)
        elev = QLabel(self._elev_text(node))
        elev.setStyleSheet(f"font-size: 10.5px; color: {_FAINT};")
        h.addWidget(elev)
        h.addStretch(1)

        cap = QLabel(_capacity_text(node))
        cap.setStyleSheet(f"font-size: 11.5px; color: {_MUTED};")
        cap.setToolTip(_capacity_tooltip(node))
        h.addWidget(cap)

        if node["enabled"] and node["has_water"]:
            full = node["overflowed"] or node["fill_pct"] >= 100
            wcol = _WARN if full else _WATER
            wtxt = "full" if full else f"{node['fill_pct']:.0f}%"
            water = QLabel(f"💧 {node['stored_m3']:,.0f} · {wtxt}")
            water.setStyleSheet(f"font-size: 11.5px; font-weight: 600; color: {wcol};")
            water.setToolTip(H.NETWORK_WATER_HELD)
            h.addWidget(water)

            # Infiltration is capture too, and for a wide shallow feature it can be
            # ALL of the capture: a large basin on free-draining soil soaks away
            # everything that reaches it, so it ponds nothing and its fill bar sits
            # empty. Showing stored volume alone made that read as "does nothing".
            soaked = node.get("soaked_m3", 0.0)
            if soaked >= 1.0:
                soak = QLabel(f"↓ {soaked:,.0f}")
                soak.setStyleSheet(
                    f"font-size: 11.5px; font-weight: 600; color: {_SOAKED};")
                soak.setToolTip(H.NETWORK_SOAKED)
                h.addWidget(soak)

            # How long standing water takes to soak away. Lancaster sizes earthworks
            # so they "work, don't flood, and don't puddle" — this is the third
            # constraint. Water left standing breeds mosquitoes, drowns the plantings
            # the earthwork exists to support, and leaves nothing free for the next
            # storm. Conventional practice is full drawdown inside 24-48 hours.
            drain = node.get("drain_hours")
            if node["stored_m3"] >= 1.0:
                if drain is None:
                    txt, col, tip = ("⏱ never", _BAD, H.NETWORK_DRAIN_NEVER)
                elif drain > 48:
                    txt, col, tip = (f"⏱ {drain:,.0f} h", _BAD, H.NETWORK_DRAIN_SLOW)
                elif drain > 24:
                    txt, col, tip = (f"⏱ {drain:,.0f} h", _WARN,
                                     H.NETWORK_DRAIN_MARGINAL)
                else:
                    txt, col, tip = (f"⏱ {drain:,.0f} h", _MUTED,
                                     H.NETWORK_DRAIN_GOOD)
                lbl = QLabel(txt)
                lbl.setStyleSheet(f"font-size: 11px; color: {col};")
                lbl.setToolTip(tip)
                h.addWidget(lbl)

            bar = _FillBar()
            bar.set(node["fill_pct"], full)
            h.addWidget(bar)

        self._restyle()
        self.setToolTip(node.get("summary", ""))

    def _elev_text(self, node):
        """Ground under the feature — except a dam, which is described by its crest.

        ``elevation`` is a DEM sample at the geometry centroid, so labelling it "crest"
        for a dam printed the ground *under* the wall: Dam 15 read "crest 54 m" against
        a crest of 56.12 m. A dam drawn but not yet given a crest falls back to the
        ground, unlabelled, rather than naming a figure that is not the crest.
        """
        crest = node.get("crest_elevation") if node["ew_type"] == "dam" else None
        if crest is not None:
            return f"crest {crest:.0f} m"
        return f"{node['elevation']:.0f} m"

    def set_selected(self, sel):
        self._selected = sel
        self._restyle()

    def _restyle(self):
        if self._selected:
            border = "#2e7d55"
        else:
            border = "#dde4e5"
        opacity = "" if self._enabled else "color: #b6c0be;"
        self.setStyleSheet(
            f"_NodeCard {{ border: 1.5px solid {border}; border-left: 3px solid "
            f"{self._colour}; border-radius: 6px; background: #ffffff; {opacity} }} "
            "_NodeCard:hover { border-color: #8fa0a4; }"
        )

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)


class _FlowChart(QWidget):
    """The network as a chart: chips by rank, water flowing top to bottom.

    The list view answers "what have I got"; this answers "how does it connect".
    Rank comes from :func:`~terrainflow_assessment.modules.simulation.layer_nodes`,
    which puts each feature below everything that spills into it — the graph maths
    stays out of the widget so it can be tested without Qt.

    Edge thickness scales with the volume actually routed, so a heavily loaded link
    reads heavier than a nominal one.
    """

    clicked = pyqtSignal(int)

    _CHIP_W, _CHIP_H = 104, 34
    _GAP_X, _GAP_Y = 14, 40
    _MARGIN = 10
    _AXIS_W = 40          # room for the metre scale in elevation mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nodes = []
        self._edges = {}
        self._layout = {}
        self._exit_m3 = 0.0
        self._selected_index = None
        self._boxes = {}          # id → QRect
        self._ticks = []          # (y, elevation) for the elevation scale
        self._axis = "rank"
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(80)

    def set_network(self, nodes, edges, exit_m3, layout):
        self._nodes = [n for n in nodes if n.get("enabled", True)]
        self._edges = edges or {}
        self._layout = layout or {}
        self._exit_m3 = exit_m3
        self._relayout()
        self.update()

    def set_selected(self, index):
        self._selected_index = index
        self.update()

    # ------------------------------------------------------------------ layout

    def set_axis_mode(self, mode):
        """``"rank"`` (spill order) or ``"elevation"`` (true height)."""
        self._axis = "elevation" if mode == "elevation" else "rank"
        self._relayout()
        self.update()

    def _relayout(self):
        self._boxes = {}
        self._ticks = []
        if not self._nodes:
            self.setFixedHeight(80)
            return
        if getattr(self, "_axis", "rank") == "elevation":
            self._relayout_by_elevation()
        else:
            self._relayout_by_rank()

    def _relayout_by_rank(self):
        by_rank = {}
        for node in self._nodes:
            rank, order = self._layout.get(node["id"], (0, 0))
            by_rank.setdefault(rank, []).append((order, node))

        width = max(self.width(), 240)
        # How many chips fit across before a rank has to wrap. A rank used to be
        # laid out as one unwrapped row — `x0 + i * (chip + gap)`, with only
        # `setFixedHeight` called and the panel's scroll area holding its
        # horizontal bar off — so on a ~360 px dock the 4th chip onward sat
        # off-widget and invisible, and `mousePressEvent` hit-tests the same boxes,
        # so it could not be clicked either. `layer_nodes` gives rank 0 to every
        # feature nothing spills into, which is the ordinary design of independent
        # swales: every chip on one row.
        per_row = max(1, (width - 2 * self._MARGIN + self._GAP_X)
                      // (self._CHIP_W + self._GAP_X))

        band = 0
        for rank in sorted(by_rank):
            entries = sorted(by_rank[rank], key=lambda e: e[0])
            for start in range(0, len(entries), per_row):
                chunk = entries[start:start + per_row]
                n = len(chunk)
                span = n * self._CHIP_W + (n - 1) * self._GAP_X
                x0 = max(self._MARGIN, (width - span) // 2)
                y = self._MARGIN + band * (self._CHIP_H + self._GAP_Y)
                for i, (_order, node) in enumerate(chunk):
                    self._boxes[node["id"]] = (
                        x0 + i * (self._CHIP_W + self._GAP_X), y,
                        self._CHIP_W, self._CHIP_H)
                band += 1

        # Sub-rows, not ranks. A wrapped rank is taller than one band, and a height
        # that does not know it clips the chips this method just placed.
        depth = max(band, 1)
        # One extra band for the "leaves site" sink at the foot of the chart.
        self.setFixedHeight(
            self._MARGIN * 2 + depth * self._CHIP_H + depth * self._GAP_Y + 18)

    def _relayout_by_elevation(self):
        """Place chips at their true height, with a metre scale down the left.

        Spill rank tells you the order things fill in; elevation tells you what is
        *physically possible*. A feature sitting above another is a candidate route
        even if nothing currently connects them — which is the question the rank
        layout cannot answer, because it only draws links that already exist.
        """
        elevations = [float(n.get("elevation") or 0.0) for n in self._nodes]
        lo, hi = min(elevations), max(elevations)
        if hi - lo < 1e-6:
            self._relayout_by_rank()
            return

        plot_h = max(180, min(520, int((hi - lo) * 12)))
        left = self._AXIS_W
        width = max(self.width(), 300)

        # Nudge chips that would overlap sideways rather than moving them off their
        # true height — the vertical position is the information here.
        placed = []
        for node, elevation in sorted(
                zip(self._nodes, elevations), key=lambda p: -p[1]):
            frac = (hi - elevation) / (hi - lo)
            y = self._MARGIN + int(frac * (plot_h - self._CHIP_H))
            column = 0
            while any(abs(py - y) < self._CHIP_H + 2 and pc == column
                      for _pid, py, pc in placed):
                column += 1
            x = left + column * (self._CHIP_W + self._GAP_X)
            if x + self._CHIP_W > width - self._MARGIN and column > 0:
                # Drop a band rather than slamming back to column 0, which is the
                # column the nudge loop above has just proved occupied — so the
                # overflow guard put the chip straight back on top of another one.
                # Cosmetic, and only reachable once a row is full.
                y += self._CHIP_H + 2
                column = 0
                x = left
            placed.append((node["id"], y, column))
            self._boxes[node["id"]] = (x, y, self._CHIP_W, self._CHIP_H)

        step = _nice_step(hi - lo)
        tick = (int(lo / step)) * step
        while tick <= hi + step:
            if lo - step <= tick <= hi + step:
                frac = (hi - tick) / (hi - lo)
                self._ticks.append(
                    (self._MARGIN + int(frac * (plot_h - self._CHIP_H))
                     + self._CHIP_H // 2, tick))
            tick += step

        self.setFixedHeight(plot_h + self._MARGIN * 2 + 18)

    def resizeEvent(self, event):
        self._relayout()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ painting

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        by_id = {n["id"]: n for n in self._nodes}
        self._draw_axis(p)

        sink_y = self.height() - 16
        for node in self._nodes:
            src = self._boxes.get(node["id"])
            if src is None:
                continue
            target_id, is_user = self._edges.get(node["id"], (None, False))
            volume = float(node.get("overflow_m3", 0.0) or 0.0)
            dst = self._boxes.get(target_id) if target_id else None

            start = (src[0] + src[2] / 2.0, src[1] + src[3])
            if dst is not None and target_id in by_id:
                end = (dst[0] + dst[2] / 2.0, dst[1])
            else:
                end = (src[0] + src[2] / 2.0, sink_y)
            self._draw_edge(p, start, end, volume, is_user, dst is not None)

        for node in self._nodes:
            box = self._boxes.get(node["id"])
            if box is not None:
                self._draw_chip(p, node, box)

        if self._nodes:
            p.setPen(QColor(_FAINT))
            f = p.font()
            f.setPointSizeF(7.5)
            p.setFont(f)
            p.drawText(self._MARGIN, sink_y + 12,
                       f"●  {self._exit_m3:,.0f} m³ leaves the site")
        p.end()

    def _draw_axis(self, p):
        """Metre scale down the left, in elevation mode only."""
        if not self._ticks:
            return
        f = p.font()
        f.setPointSizeF(7.0)
        p.setFont(f)
        for y, value in self._ticks:
            p.setPen(QPen(QColor("#eef1f0"), 1))
            p.drawLine(self._AXIS_W - 4, y, self.width() - self._MARGIN, y)
            p.setPen(QColor(_FAINT))
            p.drawText(2, y + 3, f"{value:,.0f} m")

    def _draw_edge(self, p, start, end, volume, is_user, has_target):
        width = 1.0 + min(volume / 200.0, 3.0)
        colour = QColor(_WATER if has_target else _FAINT)
        colour.setAlpha(210 if is_user else 130)
        pen = QPen(colour, width)
        pen.setStyle(Qt.PenStyle.SolidLine if is_user else Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        path = QPainterPath()
        path.moveTo(start[0], start[1])
        mid = (start[1] + end[1]) / 2.0
        path.cubicTo(start[0], mid, end[0], mid, end[0], end[1])
        p.drawPath(path)

        # Arrowhead, so direction survives a crossing.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        head = QPainterPath()
        head.moveTo(end[0], end[1])
        head.lineTo(end[0] - 4, end[1] - 6)
        head.lineTo(end[0] + 4, end[1] - 6)
        head.closeSubpath()
        p.drawPath(head)

    def _draw_chip(self, p, node, box):
        x, y, w, h = box
        selected = node["index"] == self._selected_index
        full = node.get("overflowed") or node.get("fill_pct", 0) >= 100

        p.setPen(QPen(QColor("#2e7d55" if selected else "#dde4e5"), 1.5))
        p.setBrush(QColor("#ffffff"))
        p.drawRoundedRect(x, y, w, h, 6, 6)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(node["colour"]))
        p.drawRoundedRect(x, y, 4, h, 2, 2)

        f = p.font()
        f.setPointSizeF(8.0)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(_INK))
        p.drawText(x + 9, y + 14, _elide(node["name"], 15))

        f.setBold(False)
        f.setPointSizeF(7.5)
        p.setFont(f)
        if node.get("has_water"):
            p.setPen(QColor(_WARN if full else _WATER))
            label = "full" if full else f"{node.get('fill_pct', 0):.0f}%"
            p.drawText(x + 9, y + 27, f"💧 {node.get('stored_m3', 0):,.0f} · {label}")
        else:
            p.setPen(QColor(_MUTED))
            p.drawText(x + 9, y + 27, f"{node.get('capacity_m3', 0):,.0f} m³")

    # ------------------------------------------------------------------ input

    def mousePressEvent(self, event):
        pos = event.pos()
        for node in self._nodes:
            box = self._boxes.get(node["id"])
            if box is None:
                continue
            x, y, w, h = box
            if x <= pos.x() <= x + w and y <= pos.y() <= y + h:
                self.clicked.emit(node["index"])
                return


def _elide(text, limit):
    return text if len(text) <= limit else text[: limit - 1] + "…"


class NetworkView(QWidget):
    selection_changed = pyqtSignal(object)   # index (int) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(4)

        self._list_host = QWidget()
        self._lay = QVBoxLayout(self._list_host)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._outer.addWidget(self._list_host)

        self._chart = _FlowChart()
        self._chart.clicked.connect(self._on_card_clicked)
        self._chart.setVisible(False)
        self._outer.addWidget(self._chart)

        self._mode = "list"
        self._cards = {}
        self._selected_index = None
        self._empty = QLabel("Draw a swale, basin or dam to start a network.")
        self._empty.setStyleSheet(f"color: {_FAINT}; font-style: italic; padding: 8px 2px;")
        self._lay.addWidget(self._empty)

    # ------------------------------------------------------------------ API

    def selected_index(self):
        return self._selected_index

    def set_mode(self, mode):
        """``"list"``, ``"flow"`` (spill order) or ``"elevation"`` (true height)."""
        if mode in ("flow", "elevation"):
            self._mode = "flow"
            self._chart.set_axis_mode("elevation" if mode == "elevation" else "rank")
        else:
            self._mode = "list"
        self._apply_mode()

    def _apply_mode(self):
        has_nodes = bool(self._cards)
        flow = self._mode == "flow" and has_nodes
        self._chart.setVisible(flow)
        self._list_host.setVisible(not flow)

    def set_network(self, nodes, edges, exit_m3):
        self._clear()
        if not nodes:
            self._empty.setVisible(True)
            self._selected_index = None
            self._chart.set_network([], {}, 0.0, {})
            self._apply_mode()
            return
        self._empty.setVisible(False)

        # High → low; disabled sink to the bottom (kept, but out of the flow).
        ordered = sorted(
            nodes, key=lambda n: (n["enabled"] is False, -n["elevation"])
        )
        by_id = {n["id"]: n for n in nodes}

        for node in ordered:
            card = _NodeCard(node)
            card.clicked.connect(self._on_card_clicked)
            self._cards[node["index"]] = card
            self._lay.addWidget(card)

            if node["enabled"]:
                target_id, is_user = edges.get(node["id"], (None, False))
                target = by_id.get(target_id)
                self._lay.addWidget(self._connector(target, is_user))

        exit_lbl = QLabel(f"●  {exit_m3:,.0f} m³ leaves the site")
        exit_lbl.setStyleSheet(
            f"color: {_FAINT}; font-size: 11px; padding: 6px 2px 2px;"
        )
        exit_lbl.setToolTip(H.SITE_EXIT)
        self._lay.addWidget(exit_lbl)

        # The chart shares the list's data; only the arrangement differs. Rank comes
        # from the pure layer_nodes so the graph maths is testable without Qt.
        try:
            from terrainflow_assessment.modules.simulation import layer_nodes
            plain_edges = {k: v[0] for k, v in (edges or {}).items()}
            layout = layer_nodes([n["id"] for n in ordered if n["enabled"]],
                                 plain_edges)
        except Exception:
            # Logged, not silent. The degenerate result is every node at rank 0 —
            # one flat row — which is also a perfectly ordinary layout for a design
            # of independent features, so the failed case and the normal case draw
            # identically and nothing on screen says which this is.
            _log.debug("network layout failed; every chip falls to rank 0",
                       exc_info=True)
            layout = {}
        self._chart.set_network(nodes, edges, exit_m3, layout)

        # Re-apply selection if it still exists.
        if self._selected_index in self._cards:
            self._cards[self._selected_index].set_selected(True)
        else:
            self._selected_index = None
        self._chart.set_selected(self._selected_index)
        self._apply_mode()

    def set_selected(self, index):
        """Select *index* from outside, without re-announcing it.

        Deliberately not ``_on_card_clicked``: that emits ``selection_changed``, and the
        caller here is another view that has already reported the change. It also bails
        early when the index is unchanged, which would leave a re-click from that view
        doing nothing at all.
        """
        self._selected_index = index
        for i, c in self._cards.items():
            c.set_selected(i == index)
        self._chart.set_selected(index)

    def clear_selection(self):
        self._selected_index = None
        for c in self._cards.values():
            c.set_selected(False)
        self._chart.set_selected(None)

    # ------------------------------------------------------------------ internals

    def _connector(self, target, is_user):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(14, 1, 4, 1)
        h.setSpacing(6)
        if target is None:
            glyph, text, col = "↳", "leaves site", _FAINT
        elif is_user:
            glyph, text, col = "↳", f"overflows to {target['name']}", _MUTED
        else:
            glyph, text, col = "⤷", f"downslope to {target['name']}", _FAINT
        lbl = QLabel(f"{glyph}  {text}")
        weight = "font-weight:600;" if is_user else ""
        lbl.setStyleSheet(f"font-size: 10px; color: {col}; {weight}")
        h.addWidget(lbl)
        h.addStretch(1)
        return row

    def _on_card_clicked(self, index):
        if self._selected_index == index:
            return
        self._selected_index = index
        for i, c in self._cards.items():
            c.set_selected(i == index)
        self._chart.set_selected(index)
        self.selection_changed.emit(index)

    def _clear(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._empty:
                w.deleteLater()
        self._lay.addWidget(self._empty)
        self._cards = {}


def _nice_step(span):
    """A round metre interval giving roughly 4-8 gridlines across *span*."""
    if span <= 0:
        return 1.0
    for step in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0):
        if span / step <= 8:
            return step
    return 1000.0
