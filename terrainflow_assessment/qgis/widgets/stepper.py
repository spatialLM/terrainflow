"""
stepper.py — the Workbench pipeline stepper.

A horizontal row of stage tabs (Terrain → Baseline → Design → Verify → Report)
with per-stage state glyphs:

    todo   ·   quiet, nothing yet
    done   ✓   green — stage output exists
    active ✎   the stage currently shown
    stale  ⚠   amber — stage output exists but is out of date (e.g. the design
               changed since the last verify burn)

Emits ``stage_selected(str key)`` when a stage is clicked. The active stage is
whichever the user last selected; state is orthogonal (a stage can be done AND
active).
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QHBoxLayout, QPushButton, QWidget

_STATE_GLYPHS = {"todo": "·", "done": "✓", "active": "✎", "stale": "⚠"}
_STATE_COLOURS = {"todo": "#8fa0a4", "done": "#1e8449", "stale": "#b9770e"}


class StageStepper(QWidget):
    stage_selected = pyqtSignal(str)

    def __init__(self, stages, parent=None):
        """stages: list of (key, label)."""
        super().__init__(parent)
        self._buttons = {}
        self._states = {}
        self._current = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        for key, label in stages:
            btn = QPushButton()
            btn.setFlat(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.stage_selected.emit(k))
            lay.addWidget(btn, 1)
            self._buttons[key] = (btn, label)
            self._states[key] = "todo"
        self._restyle()

    # ------------------------------------------------------------------ API

    def set_state(self, key, state):
        """state: 'todo' | 'done' | 'stale'. Active is set via set_current."""
        if key in self._states and state in ("todo", "done", "stale"):
            self._states[key] = state
            self._restyle()

    def state(self, key):
        """A stage's current state, so a caller can react to what came before.

        A failed run reports differently depending on whether an earlier run
        left usable output behind: amber if it did, quiet if nothing has ever
        succeeded. Asking the stepper avoids keeping a second copy of that.
        """
        return self._states.get(key, "todo")

    def set_current(self, key):
        if key in self._buttons:
            self._current = key
            self._restyle()

    def current(self):
        return self._current

    # ------------------------------------------------------------------ paint

    def _restyle(self):
        for key, (btn, label) in self._buttons.items():
            state = self._states[key]
            is_active = key == self._current
            glyph = "✎" if is_active and state != "stale" else _STATE_GLYPHS[state]
            underline = "#2e7d55" if is_active else "transparent"
            text_colour = "#2e7d55" if is_active else (
                "#5f7176" if state == "done" else "#8fa0a4"
            )
            btn.setText(f"{glyph}\n{label}")
            btn.setStyleSheet(
                "QPushButton {"
                f"  color: {text_colour};"
                "  border: none;"
                f"  border-bottom: 2px solid {underline};"
                "  border-radius: 0;"
                "  padding: 5px 2px 4px;"
                "  font-size: 10px; font-weight: 600;"
                "  background: transparent;"
                "} "
                # The glyph is part of the button's text, so it takes the text
                # colour above — there is no separate label to colour.
                "QPushButton:hover { background: #eef1f0; color: #22302e; }"
            )
