"""
exit_table.py — where water crosses the boundary, and how much goes out each way.

The baseline already draws an Exit Points layer and already prints one site total, and
between the two there was nowhere to read the thing a land manager actually asks after a
run: *which* crossings, and how big is each. The markers carry it in their labels, so
answering it meant clicking round the map edge and holding six numbers in your head.

Nothing here is computed. ``FlowAnalysis.get_boundary_exit_points`` has produced these
exact figures all along and the report has printed them in this exact shape since it was
written; this is the same table on the panel, so the two documents cannot drift.

**One datum, shown as two columns.** ``volume_m3`` is ``flow_ls`` multiplied by the
event duration — the rate is derived from the volume and not measured separately, so a
reader who divides one by the other learns only the storm length. Both are shown anyway,
because a rate is what a culvert is sized by and a volume is what a dam is sized by, and
converting between them by hand at the map edge is exactly the friction this removes.

**The rate is an event average, not a design peak.** ``flow_ls`` is the event volume over
the event duration. The *peak* a structure is sized against comes from ``peak_flow.py``
at the time of concentration and is a different, larger number — so this column is
deliberately not headed "peak".

**These do not sum to the boundary flux, and the footer says so.** Each crossing is
collapsed to its busiest cell after a 3x3 max-pool, and crossings under the panel's
threshold are dropped entirely. The true total is measured cell by cell and is printed
directly above this table, which is what makes the discrepancy safe to show: the reader
can see both, so the subtotal cannot be mistaken for the total.

Data contract — ``set_rows(points, threshold_ls)``:
  points : the list on ``result["exit_points"]``, each
    {x, y, flow_ls, volume_m3, label}
  threshold_ls : the panel's "Show exits above (L/s)", named in the footer so a short
    table reads as a filter rather than as a finding.
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

from terrainflow_assessment.qgis import help_text as H

_INK = "#22302e"
_MUTED = "#5f7176"
_HAIRLINE = "#dde4e5"
_HAIRLINE_STRONG = "#c6d1d3"
_GROUND = "#eef1f2"
_SURFACE = "#ffffff"

# The exit-marker blue, so the table and the dots it describes read as one thing.
_FLOW = "#1273b5"

_HEADERS = ("Crossing", "Average rate", "Volume over event")

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


class ExitPointsTable(QWidget):
    """Boundary crossings after a baseline run, largest volume first."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(_QSS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setToolTip(H.EXIT_TABLE)
        lay.addWidget(self.table)

        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
        lay.addWidget(self.footer)

        self.setVisible(False)

    # ------------------------------------------------------------------ API

    def set_rows(self, points, threshold_ls=None):
        """Populate from ``result["exit_points"]`` (empty or None hides the widget)."""
        # Sorted by volume, matching the report's own ordering, so the same site reads
        # the same way in both. The source list is ordered by rate — the same order
        # while the two are proportional, but not something to rely on.
        rows = sorted((points or []),
                      key=lambda p: p.get("volume_m3") or 0.0, reverse=True)
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
        self.footer.setText(self._footer_text(rows, threshold_ls))
        self.setVisible(True)

    # ------------------------------------------------------------------ internals

    def _fill_row(self, r, row):
        # The map marker's own text is "Exit 3: 12.4 L/s (…)". Only the leading name is
        # wanted here — the rest of it is the other two columns — and it is split rather
        # than re-numbered so the row and the dot on the map carry the same name.
        name = (row.get("label") or "").split(":")[0].strip() or f"Exit {r + 1}"
        cells = [
            (name, _INK, Qt.AlignmentFlag.AlignLeft),
            (f"{row.get('flow_ls') or 0.0:,.1f} L/s", _FLOW,
             Qt.AlignmentFlag.AlignRight),
            (f"{row.get('volume_m3') or 0.0:,.0f} m³", _INK,
             Qt.AlignmentFlag.AlignRight),
        ]
        for c, (text, colour, align) in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(_brush(colour))
            item.setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
            self.table.setItem(r, c, item)

    @staticmethod
    def _footer_text(rows, threshold_ls):
        n = len(rows)
        bits = []
        if threshold_ls:
            bits.append(
                f"{n} crossing{'s' if n != 1 else ''} above your "
                f"{threshold_ls:,.1f} L/s threshold.")
        else:
            bits.append(f"{n} crossing{'s' if n != 1 else ''}.")
        # Said every time, not only when it happens to be large. A reader who adds the
        # column up and compares it against the total above will find a gap, and the gap
        # is the method rather than an error — but only if they are told before they
        # look, which is why this is not conditional on the size of it.
        bits.append(
            "Each is its busiest cell after a 3x3 max-pool, so they do not add up to "
            "the total above — that is measured across the whole boundary, cell by "
            "cell, and includes everything under the threshold.")
        return " ".join(bits)

    def _size_to_contents(self, n_rows):
        header = self.table.horizontalHeader().height()
        row_h = self.table.rowHeight(0) if n_rows else 22
        self.table.setFixedHeight(header + row_h * min(n_rows, 8) + 6)


def _brush(hex_colour):
    from qgis.PyQt.QtGui import QBrush, QColor
    return QBrush(QColor(hex_colour))
