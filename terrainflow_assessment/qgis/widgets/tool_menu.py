"""
tool_menu.py — Felt-style earthwork tool menu.

Registry-driven rows (icon · name), grouped Storage / Flow control,
with the swale's three draw modes as an inline segmented control. Replaces the
grid of coloured buttons in the Earthwork Design stage.

Emits:
  draw_swale_requested(mode)      — 'contour' | 'full_contour' | 'freehand'
  draw_earthwork_requested(key)   — any non-swale registry key
  place_spillway_requested(kind)  — 'outflow' | 'inflow' spillway placement
  connect_earthworks_requested()  — route one feature's overflow into another
  link_drain_to_spillway_requested() — grade a diversion drain from a spillway crest
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from terrainflow_assessment.core.registry.earthwork_types import all_types
from terrainflow_assessment.qgis import help_text as H

# Small monochrome-on-colour glyphs per type (kept ASCII/BMP so Qt renders them
# everywhere). The chip carries the type's registry colour.
_GLYPH = {
    "swale": "∿", "basin": "▢", "dam": "▮", "berm": "⌒", "diversion": "↘",
}
_SWALE_MODES = (("contour", "Segment"), ("full_contour", "Contour"), ("freehand", "Free"))

# Overflow routing rows. Deliberately NOT registry types: `_ensure_ew_layers`
# iterates all_types(), so a registry entry here would conjure a map layer, a burn
# method and a capacity path for something that is not an earthwork at all.
_CONNECTION_ROWS = (
    ("outflow", "▽", "#1273b5", "Outflow Spillway", H.TOOL_OUTFLOW_SPILLWAY),
    ("inflow", "▲", "#2e7d55", "Inflow Spillway", H.TOOL_INFLOW_SPILLWAY),
    ("connect", "⇢", "#5f7176", "Route Overflow", H.TOOL_ROUTE_OVERFLOW),
    # The same grey as Route Overflow, deliberately: both rows link two features that
    # already exist, as against the two above them, which place a structure. A fourth
    # chip colour would say these are four unrelated things.
    ("link_drain", "⇥", "#5f7176", "Drain from Spillway", H.TOOL_LINK_DRAIN),
)


class EarthworkToolMenu(QWidget):
    draw_swale_requested = pyqtSignal(str)
    draw_earthwork_requested = pyqtSignal(str)
    place_spillway_requested = pyqtSignal(str)   # 'outflow' | 'inflow'
    connect_earthworks_requested = pyqtSignal()
    link_drain_to_spillway_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        types = all_types()
        storage = [k for k, c in types.items() if c.category == "storage"]
        control = [k for k, c in types.items() if c.category == "control"]
        other = [k for k in types if k not in storage and k not in control]

        lay.addWidget(self._group_label("STORAGE — HOLDS WATER"))
        lay.addWidget(self._tool_group(storage, types))
        if control:
            lay.addWidget(self._group_label("FLOW CONTROL — MOVES / BLOCKS WATER"))
            lay.addWidget(self._tool_group(control, types))
        if other:
            lay.addWidget(self._tool_group(other, types))

        # Not "ROUTE OVERFLOW" any more: the group holds two spillway placements, one
        # overflow link and one drain link, and only the third of those routes an
        # overflow. A group label that describes half its rows is worse than a general
        # one, because it reads as a promise about what is in the box.
        lay.addWidget(self._group_label("CONNECTIONS — SPILLWAYS AND ROUTING"))
        lay.addWidget(self._connection_group())

    # ------------------------------------------------------------------ build

    def _group_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 9px; font-weight: 700; letter-spacing: 1px; color: #8fa0a4;"
        )
        return lbl

    def _tool_group(self, keys, types):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #dde4e5; border-radius: 7px; background: #ffffff; }"
        )
        col = QVBoxLayout(frame)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        for i, key in enumerate(keys):
            col.addWidget(self._tool_row(key, types[key], last=(i == len(keys) - 1)))
        return frame

    def _tool_row(self, key, cfg, last):
        row = QFrame()
        border = "" if last else "border-bottom: 1px solid #eef1f0;"
        row.setStyleSheet(
            f"QFrame {{ {border} background: transparent; }} "
            "QFrame:hover { background: #f6f8f8; }"
        )
        h = QHBoxLayout(row)
        h.setContentsMargins(9, 5, 9, 5)
        h.setSpacing(9)

        chip = QLabel(_GLYPH.get(key, "●"))
        chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chip.setFixedSize(22, 22)
        chip.setStyleSheet(
            f"background: {cfg.style[1]}; color: white; border-radius: 5px;"
            " font-size: 12px; font-weight: bold;"
        )
        h.addWidget(chip)

        name = QLabel(cfg.label)
        name.setStyleSheet("font-size: 12.5px; font-weight: 600; color: #22302e;")
        name.setToolTip(cfg.tooltip
                        or H.TOOL_DRAW_FALLBACK.format(label=cfg.label.lower()))
        h.addWidget(name)
        h.addStretch(1)

        if key == "swale":
            h.addWidget(self._swale_modes())
        else:
            # Whole row is the click target for single-tool types.
            row.mousePressEvent = lambda _e, k=key: self.draw_earthwork_requested.emit(k)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
        return row

    def _connection_group(self):
        """Hand-built rows for the two overflow-routing tools.

        Same row anatomy as the registry-driven groups so the menu reads as one
        list, but built by hand because these emit parameterless signals rather
        than a type key.
        """
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { border: 1px solid #dde4e5; border-radius: 7px; background: #ffffff; }"
        )
        col = QVBoxLayout(frame)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        emitters = {
            "outflow": lambda: self.place_spillway_requested.emit("outflow"),
            "inflow": lambda: self.place_spillway_requested.emit("inflow"),
            "connect": self.connect_earthworks_requested.emit,
            "link_drain": self.link_drain_to_spillway_requested.emit,
        }
        for i, (key, glyph, colour, label, tip) in enumerate(_CONNECTION_ROWS):
            row = QFrame()
            border = "" if i == len(_CONNECTION_ROWS) - 1 else "border-bottom: 1px solid #eef1f0;"
            row.setStyleSheet(
                f"QFrame {{ {border} background: transparent; }} "
                "QFrame:hover { background: #f6f8f8; }"
            )
            h = QHBoxLayout(row)
            h.setContentsMargins(9, 5, 9, 5)
            h.setSpacing(9)

            chip = QLabel(glyph)
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setFixedSize(22, 22)
            chip.setStyleSheet(
                f"background: {colour}; color: white; border-radius: 5px;"
                " font-size: 12px; font-weight: bold;"
            )
            h.addWidget(chip)

            name = QLabel(label)
            name.setStyleSheet("font-size: 12.5px; font-weight: 600; color: #22302e;")
            name.setToolTip(tip)
            h.addWidget(name)
            h.addStretch(1)

            row.setToolTip(tip)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
            row.mousePressEvent = lambda _e, fn=emitters[key]: fn()
            col.addWidget(row)
        return frame

    def _swale_modes(self):
        seg = QFrame()
        seg.setStyleSheet(
            "QFrame { border: 1px solid #c6d1d3; border-radius: 5px; }"
        )
        hb = QHBoxLayout(seg)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        for i, (mode, label) in enumerate(_SWALE_MODES):
            b = QPushButton(label)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            left = "" if i == 0 else "border-left: 1px solid #c6d1d3;"
            b.setStyleSheet(
                "QPushButton { border: none; " + left + " background: transparent;"
                " color: #5f7176; font-size: 10.5px; padding: 3px 9px; } "
                "QPushButton:hover { background: #e9f3ee; color: #2e7d55; }"
            )
            b.clicked.connect(lambda _=False, m=mode: self.draw_swale_requested.emit(m))
            hb.addWidget(b)
        return seg
