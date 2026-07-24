"""
network_view.py — the Live Assessment flow network.

Replaces both the flat earthwork list and the HTML readout with one view: the
earthworks drawn as a water-flow network, ordered high → low, each node showing
its type colour, elevation, storage capacity (standard ink) and actual water
held (blue), with connector rows naming where each feature's overflow goes.

Selecting a node reports its index up to the panel, so the existing
Edit / Reshape / Enable-Disable / Delete action bar keeps working unchanged.

Data contract — ``set_network(nodes, edges, exit_m3)``:
  nodes : list of dicts, each
    {index, id, name, ew_type, colour, elevation, capacity_m3, stored_m3,
     fill_pct, overflowed, enabled, has_water}
  edges : {from_id: (to_id_or_None, is_user_link)}
  exit_m3 : total water leaving the site
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_WATER = "#1273b5"
_WARN = "#b9770e"
_INK = "#22302e"
_MUTED = "#5f7176"
_FAINT = "#8fa0a4"
_GLYPH = {"swale": "∿", "basin": "▢", "dam": "▮", "berm": "⌒", "diversion": "↘"}


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

        cap = QLabel(f"{node['capacity_m3']:,.0f} m³")
        cap.setStyleSheet(f"font-size: 11.5px; color: {_MUTED};")
        cap.setToolTip("Storage capacity (potential)")
        h.addWidget(cap)

        if node["enabled"] and node["has_water"]:
            full = node["overflowed"] or node["fill_pct"] >= 100
            wcol = _WARN if full else _WATER
            wtxt = "full" if full else f"{node['fill_pct']:.0f}%"
            water = QLabel(f"💧 {node['stored_m3']:,.0f} · {wtxt}")
            water.setStyleSheet(f"font-size: 11.5px; font-weight: 600; color: {wcol};")
            water.setToolTip("Water held (stored) · fill")
            h.addWidget(water)
            bar = _FillBar()
            bar.set(node["fill_pct"], full)
            h.addWidget(bar)

        self._restyle()
        self.setToolTip(node.get("summary", ""))

    def _elev_text(self, node):
        if node["ew_type"] == "dam":
            return f"crest {node['elevation']:.0f} m"
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


class NetworkView(QWidget):
    selection_changed = pyqtSignal(object)   # index (int) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._cards = {}
        self._selected_index = None
        self._empty = QLabel("Draw a swale, basin or dam to start a network.")
        self._empty.setStyleSheet(f"color: {_FAINT}; font-style: italic; padding: 8px 2px;")
        self._lay.addWidget(self._empty)

    # ------------------------------------------------------------------ API

    def selected_index(self):
        return self._selected_index

    def set_network(self, nodes, edges, exit_m3):
        self._clear()
        if not nodes:
            self._empty.setVisible(True)
            self._selected_index = None
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
        exit_lbl.setToolTip("Runoff not held by any feature")
        self._lay.addWidget(exit_lbl)

        # Re-apply selection if it still exists.
        if self._selected_index in self._cards:
            self._cards[self._selected_index].set_selected(True)
        else:
            self._selected_index = None

    def clear_selection(self):
        self._selected_index = None
        for c in self._cards.values():
            c.set_selected(False)

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
        self.selection_changed.emit(index)

    def _clear(self):
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._empty:
                w.deleteLater()
        self._lay.addWidget(self._empty)
        self._cards = {}
