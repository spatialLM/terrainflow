"""
stepper.py — the Workbench pipeline stepper.

A horizontal row of stage tabs (Terrain → Baseline → Design → Verify → Report)
with per-stage state glyphs:

    todo   ·   quiet, nothing yet
    done   ✓   the tick carries the state; the text stays a settled grey
    active ✎   the stage currently shown — green, and the only green here
    stale  ⚠   amber — stage output exists but is out of date (e.g. the design
               changed since the last verify burn)

Emits ``stage_selected(str key)`` when a stage is clicked. The active stage is
whichever the user last selected; state is orthogonal (a stage can be done AND
active).
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QHBoxLayout, QPushButton, QWidget

from terrainflow_assessment.qgis import _theme

_STATE_GLYPHS = {"todo": "·", "done": "✓", "active": "✎", "stale": "⚠"}

#: The one source for the stage text colours, and now actually read. Nothing read
#: it before: ``_restyle`` hard-coded two greys, so a stale stage rendered in the
#: same faint grey as a never-run one — the ⚠ takes the text colour, so the whole
#: distinction was invisible and the module docstring, ``panel.py``'s
#: "Amber when an earlier run left usable output behind" and six live
#: ``mark_stage(..., "stale")`` call sites all promised something that never
#: happened.
#:
#: ``done`` was ``#1e8449`` here, a green the live UI has never painted; it is now
#: the grey that is actually drawn, so reading the table is the same as reading the
#: screen. Green belongs to the *active* stage alone, which is not a state and so
#: is not in here. A lookup table nothing looks up is documentation, and this one
#: had drifted.
_STATE_COLOURS = {"todo": _theme.FAINT, "done": _theme.MUTED, "stale": _theme.WARN}
_ACTIVE_COLOUR = _theme.GROWTH


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
            underline = _ACTIVE_COLOUR if is_active else "transparent"
            text_colour = (_ACTIVE_COLOUR if is_active
                           else _STATE_COLOURS.get(state, _STATE_COLOURS["todo"]))
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
