"""
rainfall_data_dialog.py — enter the site's rainfall statistics.

The rational method needs the rainfall intensity for a storm lasting as long as the
catchment takes to respond, at a chosen return period. That is regional rainfall
statistics; no amount of terrain analysis produces it. In New Zealand it comes from
NIWA's High Intensity Rainfall Design System (HIRDS v4), free at
https://hirds.niwa.co.nz — enter a location, get a depth-duration-frequency table.

This dialog accepts that table, either pasted wholesale or typed cell by cell, and
does nothing clever with it. Entering the table also supplies the 2-year 24-hour depth
TR-55's sheet-flow term needs, so it is not a second question.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .modules.rainfall_idf import (
    HIRDS_ARI_YEARS,
    HIRDS_DURATIONS_MIN,
    IDFTable,
    parse_hirds_text,
)
from .qgis import help_text as H

_INK = "#22302e"
_MUTED = "#5f7176"
_HAIRLINE = "#dde4e5"
_HAIRLINE_STRONG = "#c6d1d3"
_GROUND = "#eef1f2"
_SURFACE = "#ffffff"
_ACCENT = "#2e7d55"
_ACCENT_HOVER = "#256645"
_WARN = "#b9770e"
_GOOD = "#1e8449"

_QSS = f"""
QDialog {{ background: {_GROUND}; }}
QLabel {{ color: {_INK}; font-size: 12px; }}
QLineEdit, QPlainTextEdit {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE_STRONG};
    border-radius: 5px; padding: 4px 7px; color: {_INK};
}}
QLineEdit:focus, QPlainTextEdit:focus {{ border: 1px solid {_ACCENT}; }}
QTableWidget {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE};
    border-radius: 6px; gridline-color: {_HAIRLINE};
}}
QHeaderView::section {{
    background: {_GROUND}; border: none;
    border-bottom: 1px solid {_HAIRLINE_STRONG};
    padding: 4px; font-size: 10.5px; font-weight: 700; color: {_MUTED};
}}
QPushButton {{
    background: {_SURFACE}; color: {_INK}; border: 1px solid {_HAIRLINE_STRONG};
    border-radius: 6px; padding: 5px 12px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {_ACCENT}; color: {_ACCENT}; }}
QDialogButtonBox QPushButton:default {{
    background: {_ACCENT}; color: #ffffff; border: 1px solid {_ACCENT};
}}
QDialogButtonBox QPushButton:default:hover {{ background: {_ACCENT_HOVER}; }}
"""


class RainfallDataDialog(QDialog):
    """Depth-duration-frequency entry. Returns an :class:`IDFTable`."""

    HIRDS_URL = "https://hirds.niwa.co.nz"

    def __init__(self, parent=None, table=None):
        super().__init__(parent)
        self._table = table if isinstance(table, IDFTable) else IDFTable()
        self.setWindowTitle("Rainfall data (depth-duration-frequency)")
        self.setMinimumWidth(560)
        self._build_ui()
        self._load(self._table)

    # ------------------------------------------------------------------ build

    def _build_ui(self):
        self.setStyleSheet(_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QLabel(
            'Rainfall statistics for this site. Look them up at '
            f'<a href="{self.HIRDS_URL}">hirds.niwa.co.nz</a> — enter your location, '
            'then copy the depth-duration table below. Depths in millimetres.'
        )
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        intro.setToolTip(H.RAINFALL_DATA)
        layout.addWidget(intro)

        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)
        meta_row.addWidget(QLabel("Site:"))
        self.edit_site = QLineEdit()
        self.edit_site.setPlaceholderText("place name, coordinates, or the date you looked it up")
        meta_row.addWidget(self.edit_site, 1)
        layout.addLayout(meta_row)

        self.table = QTableWidget(len(HIRDS_DURATIONS_MIN), len(HIRDS_ARI_YEARS))
        self.table.setHorizontalHeaderLabels([f"{a} yr" for a in HIRDS_ARI_YEARS])
        self.table.setVerticalHeaderLabels([_duration_label(d) for d in HIRDS_DURATIONS_MIN])
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setToolTip(H.RAINFALL_DATA_TABLE)
        layout.addWidget(self.table)

        paste_row = QHBoxLayout()
        paste_row.setSpacing(6)
        paste_row.addWidget(QLabel("Or paste the table:"))
        paste_row.addStretch(1)
        self.btn_parse = QPushButton("Read pasted table")
        self.btn_parse.clicked.connect(self._on_parse)
        paste_row.addWidget(self.btn_parse)
        layout.addLayout(paste_row)

        self.paste_box = QPlainTextEdit()
        self.paste_box.setPlaceholderText(
            "duration, 2, 5, 10, 20, 50, 100\n"
            "10m, 8.0, 11.0, 13.0, 15.0, 19.0, 22.0\n"
            "…"
        )
        self.paste_box.setFixedHeight(78)
        layout.addWidget(self.paste_box)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-size: 10.5px;")
        layout.addWidget(self.lbl_status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ behaviour

    def _load(self, table):
        """Fill the grid from an existing table."""
        self.edit_site.setText(table.site or "")
        for r, duration in enumerate(HIRDS_DURATIONS_MIN):
            for c, ari in enumerate(HIRDS_ARI_YEARS):
                value = (table.depths.get(ari) or {}).get(duration)
                item = QTableWidgetItem("" if value is None else f"{value:g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                      | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self._report(table, [])

    def _on_parse(self):
        """Parse the pasted text into the grid, reporting anything unreadable.

        Problems are shown rather than swallowed: a rainfall depth read wrongly
        propagates into every spillway on the site, and a paste that half-worked is
        the failure most likely to go unnoticed.
        """
        parsed, problems = parse_hirds_text(self.paste_box.toPlainText())
        if parsed.has_data():
            merged = self._current_table()
            for ari, rows in parsed.depths.items():
                merged.depths.setdefault(ari, {}).update(rows)
            self._load(merged)
            self.edit_site.setText(self.edit_site.text() or parsed.site)
        self._report(parsed, problems)

    def _current_table(self):
        """Read the grid back into an IDFTable."""
        depths = {}
        for r, duration in enumerate(HIRDS_DURATIONS_MIN):
            for c, ari in enumerate(HIRDS_ARI_YEARS):
                item = self.table.item(r, c)
                text = item.text().strip() if item is not None else ""
                if not text:
                    continue
                try:
                    value = float(text.replace(",", ""))
                except ValueError:
                    continue
                if value > 0:
                    depths.setdefault(ari, {})[duration] = value
        return IDFTable(depths=depths, source="HIRDS v4", site=self.edit_site.text().strip())

    def _report(self, table, problems):
        if problems:
            self.lbl_status.setStyleSheet(f"color: {_WARN}; font-size: 10.5px;")
            self.lbl_status.setText("\n".join(f"⚠ {p}" for p in problems))
            return
        current = self._current_table()
        if not current.has_data():
            self.lbl_status.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
            self.lbl_status.setText(
                "No data yet. Without it, overflow sizing falls back to the peak "
                "intensity typed on the Baseline tab."
            )
            return
        p2 = current.sheet_flow_p2_mm()
        bits = [f"{len(current.available_aris())} return periods entered"]
        if p2:
            bits.append(f"2-year 24-hour depth {p2:,.0f} mm (used for sheet-flow timing)")
        else:
            bits.append("no 2-year 24-hour depth — sheet-flow timing will be skipped")
        self.lbl_status.setStyleSheet(
            f"color: {_GOOD if p2 else _WARN}; font-size: 10.5px;")
        self.lbl_status.setText(" · ".join(bits))

    # ------------------------------------------------------------------ result

    def get_table(self):
        return self._current_table()


def _duration_label(minutes):
    if minutes >= 60 and minutes % 60 == 0:
        return f"{int(minutes // 60)} h"
    return f"{int(minutes)} min"
