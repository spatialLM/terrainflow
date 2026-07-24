"""
run_button.py — the analysis run button with a ghost → fill → done lifecycle.

One button that is also its own progress bar (Felt-style, from the design notes):

    idle      outlined "ghost" — clearly a button, not yet run
    running   the button fills left→right with live progress (replaces the
              separate QProgressBar)
    done      solid + ✓, still clickable to re-run

Colour comes from ``accent`` (default water-blue). The fill is a CSS gradient
whose stop tracks the progress percent.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QPushButton


class RunButton(QPushButton):
    def __init__(self, idle_text, accent="#15628f", accent_hover="#0f4f75",
                 ghost_bg="#eef4f8", parent=None):
        super().__init__(idle_text, parent)
        self._idle_text = idle_text
        self._accent = accent
        self._accent_hover = accent_hover
        self._ghost_bg = ghost_bg
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        self.set_idle()

    # ------------------------------------------------------------------ states

    def set_idle(self, text=None):
        self.setText(text or self._idle_text)
        self.setEnabled(True)
        self.setStyleSheet(
            "QPushButton {"
            f" border: 1.5px solid {self._accent}; border-radius: 5px;"
            f" background: {self._ghost_bg}; color: {self._accent};"
            " font-weight: 650; padding: 6px 10px; }"
            f" QPushButton:hover {{ background: {self._accent}; color: white; }}"
        )

    def set_progress(self, pct, text=None):
        pct = max(0, min(100, int(pct)))
        self.setText(text or f"Running… {pct}%")
        self.setEnabled(False)
        # The fill is a hard-stop gradient at the progress fraction.
        stop = pct / 100.0
        edge = min(1.0, stop + 0.0001)
        self.setStyleSheet(
            "QPushButton {"
            f" border: 1.5px solid {self._accent}; border-radius: 5px;"
            " color: white; font-weight: 650; padding: 6px 10px;"
            " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"  stop:0 {self._accent}, stop:{stop:.4f} {self._accent},"
            f"  stop:{edge:.4f} {self._ghost_bg}, stop:1 {self._ghost_bg}); }}"
        )

    def set_done(self, text=None):
        self.setText(text or f"✓ {self._idle_text}")
        self.setEnabled(True)
        self.setStyleSheet(
            "QPushButton {"
            f" border: 1.5px solid {self._accent}; border-radius: 5px;"
            f" background: {self._accent}; color: white;"
            " font-weight: 650; padding: 6px 10px; }"
            f" QPushButton:hover {{ background: {self._accent_hover};"
            f"  border-color: {self._accent_hover}; }}"
        )
