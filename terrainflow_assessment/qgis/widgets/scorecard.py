"""
scorecard.py — the persistent Workbench scorecard.

Always visible at the top of the panel: the capture score (the one number the
tool exists to move), a proportional water-budget band (stored / soaked in /
leaves site), and a legend. Colour grammar: blue = actual water, semantic
green/amber/red only for the score itself.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

_WATER = "#1273b5"
_SOAKED = "#79b8dd"
_LEAVES = "#c6d1d3"


def _score_colour(pct):
    if pct >= 80:
        return "#1e8449"
    if pct >= 40:
        return "#b9770e"
    return "#c0392b"


class _BudgetBand(QWidget):
    """Proportional stored / soaked / leaves band (custom paint)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parts = (0.0, 0.0, 1.0)  # stored, soaked, leaves fractions
        self.setFixedHeight(13)

    def set_parts(self, stored, soaked, leaves):
        total = max(1e-9, stored + soaked + leaves)
        self._parts = (stored / total, soaked / total, leaves / total)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        radius = 3.5
        p.setPen(Qt.PenStyle.NoPen)
        # background (leaves colour fills any rounding gaps)
        p.setBrush(QColor(_LEAVES))
        p.drawRoundedRect(0, 0, w, h, radius, radius)
        x = 0.0
        for frac, colour in zip(self._parts[:2], (_WATER, _SOAKED)):
            seg = frac * w
            if seg > 0.5:
                p.setBrush(QColor(colour))
                p.drawRoundedRect(int(x), 0, int(seg + radius), h, radius, radius)
                x += seg
        p.end()


class Scorecard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "Scorecard { background: #f6f8f8; border: 1px solid #dde4e5;"
            " border-radius: 8px; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(8)
        self._score_lbl = QLabel("–")
        self._score_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._caption_lbl = QLabel("")
        self._caption_lbl.setStyleSheet("color: #5f7176; font-size: 11px;")
        self._verified_lbl = QLabel("")  # phase 3: verified-vs-design chip
        self._verified_lbl.setVisible(False)
        top.addWidget(self._score_lbl)
        top.addWidget(self._caption_lbl, 1, Qt.AlignmentFlag.AlignBottom)
        top.addWidget(self._verified_lbl, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)

        self._band = _BudgetBand()
        lay.addWidget(self._band)

        self._legend_lbl = QLabel("")
        self._legend_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._legend_lbl.setStyleSheet("font-size: 10px; color: #5f7176;")
        lay.addWidget(self._legend_lbl)

        self.show_empty()

    # ------------------------------------------------------------------ API

    def show_empty(self, message="Draw a storage earthwork to see the live score."):
        self._score_lbl.setText(
            "<span style='font-size:22px;font-weight:700;color:#8fa0a4;'>–</span>"
        )
        self._caption_lbl.setText(message)
        self._band.set_parts(0.0, 0.0, 1.0)
        self._legend_lbl.setText("")

    def show_no_flow(self, capacity_m3):
        """Earthworks exist but no baseline yet — show capacity, prompt baseline."""
        self._score_lbl.setText(
            "<span style='font-size:22px;font-weight:700;color:#8fa0a4;'>–</span>"
        )
        self._caption_lbl.setText(
            f"Capacity {capacity_m3:,.0f} m³ — run Baseline to score the design."
        )
        self._band.set_parts(0.0, 0.0, 1.0)
        self._legend_lbl.setText("")

    def set_balance(self, capture_pct, stored_m3, soaked_m3, leaves_m3):
        colour = _score_colour(capture_pct)
        self._score_lbl.setText(
            f"<span style='font-size:24px;font-weight:700;color:{colour};'>"
            f"{capture_pct:.0f}%</span>"
        )
        self._caption_lbl.setText("of storm held on site")
        self._band.set_parts(max(0.0, stored_m3), max(0.0, soaked_m3),
                             max(0.0, leaves_m3))
        sw = ("<span style='font-size:12px;'>■</span>")
        self._legend_lbl.setText(
            f"<span style='color:{_WATER};'>{sw}</span> {stored_m3:,.0f} m³ stored"
            f"&nbsp;&nbsp;<span style='color:{_SOAKED};'>{sw}</span> "
            f"{soaked_m3:,.0f} m³ soaked in"
            f"&nbsp;&nbsp;<span style='color:{_LEAVES};'>{sw}</span> "
            f"{leaves_m3:,.0f} m³ leaves site"
        )

    def set_verified(self, text, fresh, tooltip=""):
        """The verified-vs-design chip.

        *tooltip* carries the sentence explaining what the delta compares — a bare
        "Δ −38%" is not self-explanatory, and the chip is too small to say more.
        """
        colour = "#1e8449" if fresh else "#b9770e"
        self._verified_lbl.setStyleSheet(
            f"color: {colour}; border: 1px solid {colour}; border-radius: 9px;"
            f" padding: 1px 8px; font-size: 10px;"
        )
        self._verified_lbl.setText(text)
        self._verified_lbl.setToolTip(tooltip or "")
        self._verified_lbl.setVisible(bool(text))
