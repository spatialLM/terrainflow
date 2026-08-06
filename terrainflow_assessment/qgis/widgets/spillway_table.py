"""
spillway_table.py — per-feature overflow review, and where the spillways get sited.

The properties dialog answers "is *this* spillway right" one feature at a time, modally,
and freezes its figures the moment it opens. That is the wrong shape for the question a
design actually raises, which is comparative and arrives late: having laid the earthworks
out, which of them can pass what now reaches them?

It arrives late because the numbers move on their own. A feature's design flow is a
function of the whole network above it, so drawing a swale upslope changes what reaches
everything below — twice over, since it both intercepts part of their catchment and, once
full, passes its own peak on. A sill sized last week can be undersized by a line drawn
today with nothing about it appearing to change.

Two decisions worth stating, because both look like omissions:

**Every water-holding feature gets a row, designed or not.** What a spillway needs is a
function of the terrain and the storm, not of whether the user has ticked a box. A
feature with no spillway yet still shows the width it would need — that is what makes
auto-sizing something you can see rather than something you are told about.

**The inlet gets a marker and no numbers.** An outflow is a weir sized to pass a peak; an
inlet is a protected entry that stops the incoming jet cutting the bank. They are
different structures with different jobs, and this module has no basis for sizing the
second. A width in that column would be confidently wrong.

Data contract — ``set_rows(rows, context)``:
  rows : list of dicts from ``EarthworksController._spillway_row``, each
    {index, id, name, ew_type, enabled, designed, sited, inlet_sited, width_auto,
     target_head_m, actual_head_m, freeboard_min_m, standard_freeboard_m,
     peak_flow_m3s, upstream_m3s, required_width_m, built_width_m, freeboard_m,
     crest_elevation, rim_elevation, problems, state}
  context : {intensity_mm_hr, intensity_is_default, has_idf, harvesting_coefficient}
"""

from qgis.PyQt.QtCore import Qt, pyqtSignal
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

# Glyph and colour per spillway kind — the same pair the tool menu and the map layer
# use, so the three surfaces read as one thing rather than three.
_OUT_GLYPH, _OUT_COLOUR = "▽", "#1273b5"
_IN_GLYPH, _IN_COLOUR = "▲", "#2e7d55"

# Head this far over target before the cell stops reading as "as designed". A centimetre
# is below setting-out resolution on a DEM, so tighter than this is noise.
_HEAD_TOLERANCE_M = 0.01

_HEADERS = ("Feature", "Peak flow", "Head", "Width", "Freeboard", "Sited")

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


class SpillwayTable(QWidget):
    """Per-feature overflow review; clicking the markers sites the structures."""

    feature_selected = pyqtSignal(int)          # row index into the earthwork manager
    place_requested = pyqtSignal(int, str)      # index, 'outflow' | 'inflow'
    edit_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        self._rows = []

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(_QSS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        lay.addWidget(self.table)

        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
        lay.addWidget(self.footer)

        self.setVisible(False)

    # ------------------------------------------------------------------ API

    def set_rows(self, rows, context=None):
        """Populate from the controller's row dicts (empty list hides the widget)."""
        rows = list(rows or [])
        self._rows = rows
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
        self.footer.setText(self._footer_text(rows, context or {}))
        self.setVisible(True)

    def summary(self):
        """One-line state for the section header — counts, worst first."""
        if not self._rows:
            return ""
        live = [r for r in self._rows if r.get("state") != "disabled"]
        if not live:
            return f"{len(self._rows)} features · all disabled"
        bad = sum(1 for r in live if r.get("state") == "fail")
        unsited = sum(1 for r in live
                      if r.get("state") in ("undesigned", "unsited", "ok")
                      and not r.get("sited"))
        bits = [f"{len(live)} feature{'s' if len(live) != 1 else ''}"]
        if bad:
            bits.append(f"{bad} needs attention")
        elif unsited:
            bits.append(f"{unsited} not sited")
        return " · ".join(bits)

    # ------------------------------------------------------------------ internals

    def _on_cell_clicked(self, row, col):
        if not (0 <= row < len(self._rows)):
            return
        data = self._rows[row]
        index = data.get("index")
        if index is None:
            return
        self.feature_selected.emit(index)
        # The markers double as the placement affordance. Siting a spillway used to
        # mean selecting the feature in a different widget first, then finding a row in
        # the tool menu — a coupling nobody discovers. Here the thing you click is the
        # thing you are placing.
        if col == len(_HEADERS) - 1 and data.get("state") != "disabled":
            self.place_requested.emit(index, "outflow")

    def _on_cell_double_clicked(self, row, _col):
        if 0 <= row < len(self._rows):
            index = self._rows[row].get("index")
            if index is not None:
                self.edit_requested.emit(index)

    def _fill_row(self, r, row):
        state = row.get("state")
        disabled = state == "disabled"

        cells = [
            (row.get("name", "—"), _FAINT if disabled else _INK,
             Qt.AlignmentFlag.AlignLeft),
            self._flow_cell(row, disabled),
            self._head_cell(row, disabled),
            self._width_cell(row, disabled),
            self._freeboard_cell(row, disabled),
            self._sited_cell(row, disabled),
        ]
        for c, (text, colour, align) in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(_brush(colour))
            item.setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
            item.setToolTip(self._row_tooltip(row))
            self.table.setItem(r, c, item)

    @staticmethod
    def _flow_cell(row, disabled):
        if disabled:
            return ("disabled", _FAINT, Qt.AlignmentFlag.AlignRight)
        flow = row.get("peak_flow_m3s")
        if not flow:
            return ("no flow", _FAINT, Qt.AlignmentFlag.AlignRight)
        return (f"{flow * 1000:,.0f} L/s", _MUTED, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _head_cell(row, disabled):
        """Target while the width is free; achieved once a width is committed.

        Never both. They are the same number whenever the width tracks, so showing two
        columns would imply a distinction that does not exist in that mode — and hide
        the one case where it does.
        """
        if disabled or not row.get("peak_flow_m3s"):
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        target = row.get("target_head_m")
        if row.get("width_auto") or row.get("actual_head_m") is None:
            return (f"{target:.2f} m target", _MUTED, Qt.AlignmentFlag.AlignRight)
        actual = row["actual_head_m"]
        over = actual - (target or 0.0)
        colour = _INK if over <= _HEAD_TOLERANCE_M else _BAD
        return (f"{actual:.2f} m actual", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _width_cell(row, disabled):
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        required = row.get("required_width_m")
        if not required:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        if not row.get("designed"):
            # Sized but not committed to: the point of showing it is that the number
            # exists before the decision does.
            return (f"{required:.1f} m needed", _WARN, Qt.AlignmentFlag.AlignRight)
        built = row.get("built_width_m") or 0.0
        if row.get("width_auto"):
            return (f"{built:.1f} m auto", _MUTED, Qt.AlignmentFlag.AlignRight)
        short = built + 0.01 < required
        return (f"{built:.1f} m built", _BAD if short else _INK,
                Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _freeboard_cell(row, disabled):
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        if row.get("rim_elevation") is None:
            return ("no DEM", _FAINT, Qt.AlignmentFlag.AlignRight)
        freeboard = row.get("freeboard_m")
        if freeboard is None:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignRight)
        minimum = row.get("freeboard_min_m") or 0.0
        if freeboard < 0:
            colour = _BAD
        elif freeboard + 1e-6 < minimum:
            colour = _BAD
        elif freeboard < minimum + 0.05:
            colour = _WARN
        else:
            colour = _GOOD
        return (f"{freeboard:.2f} m", colour, Qt.AlignmentFlag.AlignRight)

    @staticmethod
    def _sited_cell(row, disabled):
        if disabled:
            return ("—", _FAINT, Qt.AlignmentFlag.AlignCenter)
        out = _OUT_GLYPH if row.get("sited") else f"{_OUT_GLYPH}·"
        inlet = _IN_GLYPH if row.get("inlet_sited") else f"{_IN_GLYPH}·"
        colour = _MUTED if row.get("sited") else _WARN
        return (f"{out} {inlet}", colour, Qt.AlignmentFlag.AlignCenter)

    def _row_tooltip(self, row):
        """Everything the cells had to compress, in words."""
        name = row.get("name", "This feature")
        if row.get("state") == "disabled":
            return (f"{name} is disabled, so it takes no catchment and has nothing to "
                    f"pass. Re-enable it to size its overflow.")

        bits = []
        flow = row.get("peak_flow_m3s")
        if flow:
            upstream = row.get("upstream_m3s") or 0.0
            own = flow - upstream
            line = f"Peak design flow {flow * 1000:,.0f} L/s"
            if upstream > 0:
                line += (f" — {own * 1000:,.0f} from its own catchment, "
                         f"{upstream * 1000:,.0f} arriving from features upslope")
            bits.append(line + ".")
        else:
            bits.append(
                f"{name} has no design flow yet — run Baseline, and set a peak "
                f"intensity, before its overflow can be sized.")

        if not row.get("designed") and row.get("required_width_m"):
            bits.append(
                f"No spillway designed yet. At {row['target_head_m']:.2f} m of head it "
                f"would need a {row['required_width_m']:.1f} m sill.")

        if row.get("rim_elevation") is None and flow:
            bits.append(
                "No rim elevation — the crest and freeboard cannot be checked until a "
                "DEM is loaded and Baseline has run.")
        elif row.get("freeboard_m") is not None:
            minimum = row.get("freeboard_min_m") or 0.0
            bits.append(
                f"At the design flow the water surface sits {row['freeboard_m']:.2f} m "
                f"below the rim; this type is designed for {minimum:.2f} m.")

        if row.get("ew_type") == "dam":
            bits.append(
                "For a dam the rim is the wall crest you specified, so this margin is "
                "measured against the wall you intend to build rather than against "
                "existing ground.")

        for problem in row.get("problems", []):
            bits.append("⚠ " + problem)

        bits.append("Click the markers to site the overflow; double-click to edit.")
        return "\n\n".join(bits)

    def _footer_text(self, rows, context):
        live = [r for r in rows if r.get("state") != "disabled"]
        bits = []

        # Leads, because it qualifies every number above it: an intensity nobody chose
        # is not a design storm, and the widths are linear in it.
        if context.get("intensity_is_default") and not context.get("has_idf"):
            bits.append(
                f"Every flow here is at {context.get('intensity_mm_hr', 0):.0f} mm/hr — "
                f"the starting value, not a lookup for your site. Enter your HIRDS "
                f"table, or use Compare on the Baseline stage, before treating these "
                f"widths as sized."
            )
        if context.get("harvesting_coefficient"):
            bits.append(
                "The runoff coefficient is a water-harvesting figure, calibrated for "
                "ordinary rain rather than the extreme event on saturated ground. It "
                "will undersize every overflow below."
            )

        failing = [r for r in live if r.get("state") == "fail"]
        if failing:
            worst = failing[0]
            problem = (worst.get("problems") or ["needs attention"])[0]
            bits.append(f"{worst.get('name')}: {problem}")

        undesigned = [r for r in live if not r.get("designed")]
        if undesigned:
            names = ", ".join(r.get("name", "?") for r in undesigned[:3])
            more = "" if len(undesigned) <= 3 else f" (+{len(undesigned) - 3} more)"
            bits.append(
                f"{len(undesigned)} feature{'s' if len(undesigned) != 1 else ''} have no "
                f"spillway designed: {names}{more}. They will still overflow — at "
                f"whichever point of the rim happens to be lowest."
            )

        unsited = [r for r in live if r.get("designed") and not r.get("sited")]
        if unsited:
            bits.append(
                f"{len(unsited)} sized but not sited. A width without a location is not "
                f"yet a design — place it where you can armour it."
            )

        # Named once, quietly, because it is a real limitation of the reported storage
        # and the user will otherwise carry the capacity figure to site.
        if any(r.get("crest_elevation") is not None for r in live):
            bits.append(
                "Storage is reported to each feature's full depth, not down to its "
                "crest, so a crest set well below the rim holds less than the capacity "
                "figure states."
            )

        return "  ·  ".join(bits)

    def _size_to_contents(self, n_rows):
        header = self.table.horizontalHeader().height()
        row_h = self.table.rowHeight(0) if n_rows else 22
        self.table.setFixedHeight(header + row_h * min(n_rows, 8) + 6)


def _brush(hex_colour):
    from qgis.PyQt.QtGui import QBrush, QColor
    return QBrush(QColor(hex_colour))
