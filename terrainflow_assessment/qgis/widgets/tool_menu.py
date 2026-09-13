"""
tool_menu.py — Felt-style earthwork tool menu.

Registry-driven rows (icon · name), grouped Storage / Flow control, under one
"Draw along" control — Free / Segment / Contour — that applies to every line type.
Replaces the grid of coloured buttons in the Earthwork Design stage.

The draw mode is a setting the menu holds (``draw_mode``), not a signal: a row
click emits the type key alone and the controller reads the mode off the panel
when it arms the tool. It used to be three buttons on the swale row, so a berm —
whose whole job is to hold a pool on a contour — could only be drawn freehand.

Emits:
  draw_earthwork_requested(key)   — any registry key
  place_spillway_requested(kind)  — 'outflow' | 'inflow' spillway placement
  connect_earthworks_requested()  — route one feature's overflow into another
  link_drain_to_spillway_requested() — grade a diversion drain from a spillway crest
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from terrainflow_assessment.core.registry.earthwork_types import all_types
from terrainflow_assessment.qgis import _theme
from terrainflow_assessment.qgis import help_text as H

# Small monochrome-on-colour glyphs per type (kept ASCII/BMP so Qt renders them
# everywhere). The chip carries the type's registry colour. Shared with the network
# view, which carried a byte-identical copy.
_GLYPH = _theme.GLYPH
#: (mode key, button label, tooltip). The keys are what the controller's draw path
#: has always taken: 'contour' is a stretch of one, 'full_contour' the whole line.
_DRAW_MODES = (
    ("freehand", "Free", H.TOOL_DRAW_FREE),
    ("contour", "Segment", H.TOOL_DRAW_SEGMENT),
    ("full_contour", "Contour", H.TOOL_DRAW_CONTOUR),
)

# Overflow routing rows. Deliberately NOT registry types: `_ensure_ew_layers`
# iterates all_types(), so a registry entry here would conjure a map layer, a burn
# method and a capacity path for something that is not an earthwork at all.
_CONNECTION_ROWS = (
    ("outflow", *_theme.SPILLWAY_OUT, "Outflow Spillway", H.TOOL_OUTFLOW_SPILLWAY),
    ("inflow", *_theme.SPILLWAY_IN, "Inflow Spillway", H.TOOL_INFLOW_SPILLWAY),
    ("connect", "⇢", _theme.MUTED, "Route Overflow", H.TOOL_ROUTE_OVERFLOW),
    # The same grey as Route Overflow, deliberately: both rows link two features that
    # already exist, as against the two above them, which place a structure. A fourth
    # chip colour would say these are four unrelated things.
    ("link_drain", "⇥", _theme.MUTED, "Drain from Spillway", H.TOOL_LINK_DRAIN),
)


class EarthworkToolMenu(QWidget):
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

        self._draw_mode = "freehand"
        lay.addWidget(self._draw_mode_row())

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

        chip = QLabel(_GLYPH.get(key, _theme.GLYPH_UNKNOWN))
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

        # The whole row is the click target; how a line is drawn is the menu's
        # draw-mode setting, not a property of the row.
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

    # ------------------------------------------------------------- draw mode

    @property
    def draw_mode(self):
        """'freehand' | 'contour' (a stretch of one) | 'full_contour'."""
        return self._draw_mode

    def set_draw_mode(self, mode):
        """Select a draw mode exactly as a click on its button would."""
        button = self._mode_buttons.get(mode)
        if button is None:
            raise ValueError(f"unknown draw mode {mode!r}; one of "
                             f"{tuple(self._mode_buttons)}")
        button.setChecked(True)      # toggled → _on_mode_toggled → _draw_mode

    def _on_mode_toggled(self, checked, mode):
        if checked:
            self._draw_mode = mode

    def _draw_mode_row(self):
        """One draw-along setting for every line type: Free, Segment or Contour.

        Menu-wide rather than per row because the question — freehand, or on a
        contour — is the same for every line type, and a per-row answer left it
        available to one. The controller reads ``draw_mode`` when a row is clicked;
        a polygon type has no line to lay along a contour and ignores it.
        """
        row = QFrame()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(9)
        title = self._group_label("DRAW ALONG")
        title.setToolTip(H.TOOL_DRAW_MODE)
        h.addWidget(title)
        h.addStretch(1)

        seg = QFrame()
        seg.setStyleSheet(
            "QFrame { border: 1px solid #c6d1d3; border-radius: 5px; }"
        )
        hb = QHBoxLayout(seg)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons = {}
        for i, (mode, label, tip) in enumerate(_DRAW_MODES):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(tip)
            left = "" if i == 0 else "border-left: 1px solid #c6d1d3;"
            b.setStyleSheet(
                "QPushButton { border: none; " + left + " background: transparent;"
                " color: #5f7176; font-size: 10.5px; padding: 3px 9px; } "
                "QPushButton:hover { background: #e9f3ee; color: #2e7d55; } "
                "QPushButton:checked { background: #e9f3ee; color: #2e7d55;"
                " font-weight: 600; }"
            )
            b.toggled.connect(lambda checked, m=mode: self._on_mode_toggled(checked, m))
            self._mode_group.addButton(b)
            self._mode_buttons[mode] = b
            hb.addWidget(b)
        self._mode_buttons[self._draw_mode].setChecked(True)
        h.addWidget(seg)
        return row
