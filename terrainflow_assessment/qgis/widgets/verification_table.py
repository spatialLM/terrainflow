"""
verification_table.py — per-earthwork design vs measured storage.

Replaces a single site-wide "Verified · Δ −38%" chip, which conflated three unrelated
gaps and so meant nothing. Four columns, each answering one question:

  Design       geometric x 0.8 freeboard        "What do I plan on?"
  Geometric    the drawn shape, exactly          "How big is the hole I drew?"
  At grid      that shape rasterised to the DEM  "How much of it can this grid hold?"
  Measured     ponding read off the burned DEM   "What did the burn actually produce?"

Only the last comparison is a correctness check. **Δ = Measured vs At-grid**, and it
should sit near zero — a non-zero Δ there is a burn bug and nothing else. The other
two gaps are expected and explainable: Geometric → At-grid is the resolution penalty
(often *positive*, because a 1 m cell cannot represent a battered side and burns the
trench square), and Design → Geometric is the freeboard allowance, a fixed −20%.

Rows for sub-cell features carry no measured figure at all. A feature narrower than
one DEM cell is validated for placement and routing only; claiming a storage volume
for it would be inventing precision the grid does not have.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_INK = "#22302e"
_MUTED = "#5f7176"
_FAINT = "#8fa0a4"
_HAIRLINE = "#dde4e5"
_HAIRLINE_STRONG = "#c6d1d3"
_GROUND = "#eef1f2"
_SURFACE = "#ffffff"
_GOOD = "#1e8449"
_WARN = "#b9770e"
_BAD = "#c0392b"

# Δ bands. Under 5% the burn reproduces what the grid can represent; past 15% it is
# not a rounding artefact and something in the burn is wrong.
_DELTA_GOOD = 5.0
_DELTA_WARN = 15.0

_HEADERS = ("Feature", "Design", "Geometric", "At grid", "Measured", "Δ")

_QSS = f"""
QTableWidget {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE};
    border-radius: 6px; gridline-color: {_HAIRLINE};
    selection-background-color: #e9f3ee; selection-color: {_INK};
}}
QHeaderView::section {{
    background: {_GROUND}; border: none;
    border-bottom: 1px solid {_HAIRLINE_STRONG};
    padding: 4px 5px; font-size: 10px; font-weight: 700; color: {_MUTED};
}}
"""


class VerificationTable(QWidget):
    """Per-feature verification, with a footer that explains the Δ in words."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(_QSS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        lay.addWidget(self.table)

        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
        lay.addWidget(self.footer)

        self.setVisible(False)

    # ------------------------------------------------------------------ API

    def set_result(self, result, cell_size_m=1.0):
        """Populate from a :class:`~terrainflow_assessment.modules.reporting.
        VerificationResult`, or hide when there is nothing to show."""
        rows = list(getattr(result, "per_feature", None) or []) if result else []
        if not rows:
            self.setVisible(False)
            self.table.setRowCount(0)
            self.footer.setText("")
            return

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._fill_row(r, row)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._size_to_contents(len(rows))
        self.footer.setText(self._footer_text(result, rows, cell_size_m))
        self.setVisible(True)

    # ------------------------------------------------------------------ internals

    def _fill_row(self, r, row):
        routing_only = bool(row.get("routing_only"))
        measured = row.get("terrain_m3")
        delta = row.get("delta_pct")

        # A dam has no drawn cross-section — the shape of the water is the shape of
        # the valley — so repeating one number across three columns would imply an
        # agreement that was never tested. Show it once, under Design.
        barrier = bool(row.get("barrier_impounded"))

        cells = [
            (row.get("name", "—"), _INK, Qt.AlignmentFlag.AlignLeft),
            (_m3(row.get("analytic_m3")), _MUTED, Qt.AlignmentFlag.AlignRight),
            ("impounded" if barrier else _m3(row.get("geometric_m3")),
             _FAINT if barrier else _MUTED, Qt.AlignmentFlag.AlignRight),
            ("—" if barrier else _m3(row.get("rasterisable_m3")),
             _FAINT if barrier else _INK, Qt.AlignmentFlag.AlignRight),
            ("sub-cell" if routing_only else _m3(measured),
             _FAINT if routing_only else _INK, Qt.AlignmentFlag.AlignRight),
            (self._delta_text(delta, routing_only),
             self._delta_colour(delta, routing_only), Qt.AlignmentFlag.AlignRight),
        ]
        for c, (text, colour, align) in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(_brush(colour))
            item.setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
            if c == 0:
                item.setToolTip(self._row_tooltip(row))
            self.table.setItem(r, c, item)

    @staticmethod
    def _delta_text(delta, routing_only):
        if routing_only:
            return "—"
        return "—" if delta is None else f"{delta:+.0f}%"

    @staticmethod
    def _delta_colour(delta, routing_only):
        if routing_only or delta is None:
            return _FAINT
        magnitude = abs(delta)
        if magnitude <= _DELTA_GOOD:
            return _GOOD
        return _WARN if magnitude <= _DELTA_WARN else _BAD

    def _row_tooltip(self, row):
        """Spell out this feature's three gaps, since only one of them is a defect."""
        name = row.get("name", "This feature")
        if row.get("barrier_impounded"):
            measured = row.get("terrain_m3")
            design = row.get("analytic_m3") or 0.0
            return (
                f"{name} impounds water against the terrain, so there is no drawn "
                f"cross-section to rasterise — the shape of the pool is the shape of "
                f"the valley. Its design figure is already a flooded-volume "
                f"calculation over this same grid.\n\n"
                f"Δ therefore compares the burn ({_m3(measured)} m³) against that "
                f"calculation ({design:,.0f} m³) directly. Near zero means the two "
                f"agree; there is no freeboard or resolution term to separate out."
            )
        if row.get("routing_only"):
            return (f"{name} is narrower than one DEM cell. It is verified for "
                    f"placement and routing only — a storage volume measured off the "
                    f"grid would be an artefact of the grid, not of the design.")

        design = row.get("analytic_m3") or 0.0
        geometric = row.get("geometric_m3") or 0.0
        at_grid = row.get("rasterisable_m3") or 0.0
        measured = row.get("terrain_m3")
        penalty = row.get("resolution_penalty_m3") or 0.0
        freeboard = row.get("freeboard_m3") or 0.0

        parts = [f"{name}"]
        if geometric > 0 and freeboard:
            parts.append(
                f"Design {design:,.0f} m³ is the drawn {geometric:,.0f} m³ less "
                f"{freeboard:,.0f} m³ of freeboard — your allowance, not an error.")
        if geometric > 0 and penalty:
            parts.append(
                f"A {at_grid:,.0f} m³ at-grid figure is {penalty / geometric * 100:+.0f}% "
                f"off the drawn shape. That is what the cell size can represent, and "
                f"is usually positive: a grid cannot cut a battered side, so it burns "
                f"the section square.")
        if measured is not None and at_grid > 0:
            parts.append(
                f"Measured {measured:,.0f} m³ against {at_grid:,.0f} m³ at grid is the "
                f"only comparison that tests the burn. Everything else is arithmetic "
                f"you chose.")
        return "\n\n".join(parts)

    def _footer_text(self, result, rows, cell_size_m):
        real = [r for r in rows if not r.get("routing_only")
                and r.get("delta_pct") is not None]
        sub_cell = len(rows) - len([r for r in rows if not r.get("routing_only")])

        bits = []
        if real:
            worst = max(real, key=lambda r: abs(r["delta_pct"]))
            worst_pct = worst["delta_pct"]
            if abs(worst_pct) <= _DELTA_GOOD:
                bits.append(
                    f"The burn reproduces what a {cell_size_m:.2f} m grid can hold, to "
                    f"within {abs(worst_pct):.0f}% on every feature.")
            elif worst.get("barrier_impounded"):
                # A dam has no at-grid figure to be "off" — the column reads "—". Its
                # reference is a flooded-volume calculation, so the channel wording
                # below would describe a comparison that was never made.
                bits.append(
                    f"{worst['name']} is {worst_pct:+.0f}% off its flooded-volume "
                    f"calculation — it impounds against the terrain, so there is no "
                    f"grid figure to compare it with. That gap is between the burn and "
                    f"the calculation, not resolution or freeboard.")
            else:
                bits.append(
                    f"{worst['name']} is {worst_pct:+.0f}% off what a "
                    f"{cell_size_m:.2f} m grid can hold — that gap is a burn issue, "
                    f"not resolution or freeboard.")

        penalties = [(r["name"], r["resolution_penalty_m3"] / r["geometric_m3"] * 100.0)
                     for r in rows
                     if r.get("geometric_m3") and r.get("resolution_penalty_m3")]
        if penalties:
            name, pct = max(penalties, key=lambda p: abs(p[1]))
            bits.append(
                f"Above that, {pct:+.0f}% on {name} is grid resolution, and the "
                f"site-wide −20% between Design and Geometric is your freeboard.")

        if sub_cell:
            bits.append(
                f"{sub_cell} feature{'s' if sub_cell != 1 else ''} narrower than one "
                f"cell {'are' if sub_cell != 1 else 'is'} checked for placement and "
                f"routing only.")
        return " ".join(bits)

    def _size_to_contents(self, n_rows):
        header = self.table.horizontalHeader().height()
        row_h = self.table.rowHeight(0) if n_rows else 20
        self.table.setFixedHeight(header + row_h * min(n_rows, 8) + 6)


def _m3(value):
    return "—" if value is None else f"{value:,.0f}"


def _brush(hex_colour):
    from qgis.PyQt.QtGui import QBrush, QColor
    return QBrush(QColor(hex_colour))
