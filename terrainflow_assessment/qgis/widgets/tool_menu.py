"""
tool_menu.py — Felt-style earthwork tool menu.

Registry-driven rows (icon · name · shortcut), grouped Storage / Flow control,
with the swale's three draw modes as an inline segmented control. Replaces the
grid of coloured buttons in the Earthwork Design stage.

Emits:
  draw_swale_requested(mode)      — 'contour' | 'full_contour' | 'freehand'
  draw_earthwork_requested(key)   — any non-swale registry key
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

# Small monochrome-on-colour glyphs per type (kept ASCII/BMP so Qt renders them
# everywhere). The chip carries the type's registry colour.
_GLYPH = {
    "swale": "∿", "basin": "▢", "dam": "▮", "berm": "⌒", "diversion": "↘",
}
_SHORTCUT = {"swale": "S", "basin": "B", "dam": "D", "berm": "M", "diversion": "V"}
_SWALE_MODES = (("contour", "Segment"), ("full_contour", "Contour"), ("freehand", "Free"))


class EarthworkToolMenu(QWidget):
    draw_swale_requested = pyqtSignal(str)
    draw_earthwork_requested = pyqtSignal(str)

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
        name.setToolTip(cfg.tooltip or f"Draw a {cfg.label.lower()}")
        h.addWidget(name)
        h.addStretch(1)

        if key == "swale":
            h.addWidget(self._swale_modes())
        else:
            h.addWidget(self._shortcut_badge(_SHORTCUT.get(key, "")))
            # Whole row is the click target for single-tool types.
            row.mousePressEvent = lambda _e, k=key: self.draw_earthwork_requested.emit(k)
            row.setCursor(Qt.CursorShape.PointingHandCursor)
        return row

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

    def _shortcut_badge(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "font-size: 10px; color: #8fa0a4; border: 1px solid #dde4e5;"
            " border-radius: 3px; padding: 0 5px;"
        )
        return lbl
