"""
design_intensity_dialog.py — choose the peak rainfall intensity that sizes spillways.

Storage is sized on how much rain falls; an overflow structure is sized on how fast it
falls. The plugin previously derived the second from the first by dividing event volume
by event duration, which yields the storm *average* — 5 mm/hr for a 120 mm / 24 h design
storm. That is a daily mean, not a peak, and it sized a 2.8 ha spillway at 7 cm.

There is no way to infer the peak from a depth and a duration, so this dialog asks,
and shows the consequence of each answer instead of burying the choice in a default.

Two deliberate constraints:

* **The intensity is selectable. The basis is not.** The other two runoff bases are
  shown so the current one can be judged against them, but choosing one here would
  size a spillway on a basis that contradicts the storage it protects. Changing basis
  is a Baseline decision with staleness consequences, so the dialog says that rather
  than offering a shortcut that could be clicked by accident.
* Everything is computed from cached values. Catchment area comes from the flow
  graph and is basis-independent; the basis enters only as a scalar fraction. No
  re-analysis is triggered by opening this.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .modules.earthwork_design import calculate_spillway_width
from .modules.peak_flow import (
    BASIS_COEFFICIENT,
    BASIS_RAINFALL,
    BASIS_SCS,
    peak_runoff_fraction,
    rational_peak_flow,
)
from .qgis import help_text as H

_INK = "#22302e"
_MUTED = "#5f7176"
_FAINT = "#8fa0a4"
_HAIRLINE = "#dde4e5"
_HAIRLINE_STRONG = "#c6d1d3"
_GROUND = "#eef1f2"
_SURFACE = "#ffffff"
_ACCENT = "#2e7d55"
_ACCENT_HOVER = "#256645"
_WARN = "#b9770e"

_BASIS_LABELS = {
    BASIS_COEFFICIENT: "Runoff coefficient (Lancaster)",
    BASIS_RAINFALL: "Total rainfall",
    BASIS_SCS: "Surface runoff (SCS-CN)",
}

# Short-duration design intensities. The first four are the storm presets already in
# SCSRunoff; the durations below them show what the user's own storm depth implies if
# it fell over a shorter window, which is the comparison that makes the storm-average
# default look as wrong as it is.
_PRESETS = (
    ("Light", 25.0), ("Moderate", 40.0), ("Heavy", 65.0), ("Extreme", 80.0),
)
_DURATIONS = (0.5, 1.0, 2.0, 6.0, 12.0, 24.0)

_QSS = f"""
QDialog {{ background: {_GROUND}; }}
QLabel {{ color: {_INK}; font-size: 12px; }}
QDoubleSpinBox {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE_STRONG};
    border-radius: 5px; padding: 4px 7px; color: {_INK};
}}
QDoubleSpinBox:focus {{ border: 1px solid {_ACCENT}; }}
QGroupBox {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE};
    border-radius: 7px; margin-top: 12px; padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 10px;
    padding: 0 5px; color: {_INK}; font-weight: 650; font-size: 11px;
    background: {_GROUND};
}}
QTableWidget {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE};
    border-radius: 6px; gridline-color: {_HAIRLINE};
    selection-background-color: #e9f3ee; selection-color: {_INK};
}}
QHeaderView::section {{
    background: {_GROUND}; border: none;
    border-bottom: 1px solid {_HAIRLINE_STRONG};
    padding: 5px; font-size: 10.5px; font-weight: 700; color: {_MUTED};
}}
QDialogButtonBox QPushButton {{
    min-width: 84px; padding: 6px 14px; border-radius: 6px; font-weight: 600;
}}
QDialogButtonBox QPushButton:default {{
    background: {_ACCENT}; color: #ffffff; border: 1px solid {_ACCENT};
}}
QDialogButtonBox QPushButton:default:hover {{ background: {_ACCENT_HOVER}; }}
QDialogButtonBox QPushButton:!default {{
    background: {_SURFACE}; color: {_INK}; border: 1px solid {_HAIRLINE_STRONG};
}}
"""


class DesignIntensityDialog(QDialog):
    """Pick the peak design intensity, with its consequences visible.

    *area_m2* is the reference catchment the table is costed against — the selected
    feature's, or the largest on site, so the widths shown are ones the user will
    actually meet. Everything else is the storm and basis context from the panel.

    *head_m* is the design head every width in the table is solved at, and it is the
    **caller's** to supply: this default is the embankment figure, and applying it to
    a swale — whose registry head is 0.15 m — understates every width by
    ``(0.30/0.15)^1.5`` ≈ 2.8x while printing "Spillway @ 0.30 m" as fact.
    *head_note* is where that figure came from, shown beside it, so a default is
    never read as a measurement.
    """

    def __init__(self, parent=None, area_m2=0.0, area_label="", rainfall_mm=120.0,
                 duration_hr=24.0, basis=BASIS_COEFFICIENT, coefficient=0.5,
                 cn=61.0, head_m=0.30, head_note="", current_intensity=None,
                 travel_time=None, idf_table=None):
        self._tc = travel_time
        self._idf = idf_table
        super().__init__(parent)
        self._area_m2 = float(area_m2 or 0.0)
        self._rainfall_mm = float(rainfall_mm or 0.0)
        self._duration_hr = float(duration_hr or 0.0)
        self._basis = basis
        self._coefficient = coefficient
        self._cn = cn
        self._head_m = float(head_m or 0.30)
        self._head_note = head_note or ""

        self.setWindowTitle("Design intensity for overflow sizing")
        self.setMinimumWidth(520)
        self._build_ui(area_label, current_intensity)
        self._refresh_comparison()

    # ------------------------------------------------------------------ build

    def _build_ui(self, area_label, current_intensity):
        self.setStyleSheet(_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QLabel(
            "Storage is sized on how much rain falls. An overflow is sized on how "
            "fast it falls, and a depth over a duration cannot tell you that — "
            "120 mm in 2 hours and 120 mm in 24 hours are the same storage and a "
            "12-fold difference in peak flow."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        layout.addWidget(intro)

        ctx = QGroupBox("Costed against")
        ctx_form = QFormLayout(ctx)
        if self._area_m2 > 0:
            ctx_form.addRow("Catchment:", self._value_label(
                f"{self._area_m2 / 10_000.0:,.2f} ha"
                + (f"  ({area_label})" if area_label else "")))
        else:
            # Without an area every flow and width below is 0, which reads like an
            # answer rather than a missing input.
            none_yet = QLabel(
                "No area yet — run the baseline so the flows and widths below are "
                "costed against real ground. The intensity itself can still be set."
            )
            none_yet.setWordWrap(True)
            none_yet.setStyleSheet(f"color: {_WARN}; font-size: 10.5px;")
            ctx_form.addRow("Catchment:", none_yet)
        ctx_form.addRow("Design storm:", self._value_label(
            f"{self._rainfall_mm:,.0f} mm over {self._duration_hr:,.0f} h"))
        head_text = f"{self._head_m:.2f} m"
        if self._head_note:
            # Attributed, because the same number means different things: the
            # feature type's registry head is a policy figure for that type, and
            # the whole-site fallback is the embankment default standing in for a
            # feature nobody has selected.
            head_text += f" ({self._head_note})"
        ctx_form.addRow("Head over crest:", self._value_label(head_text))

        # Time of concentration — the duration the rational method actually wants.
        if self._tc is not None:
            tc_label = QLabel(self._tc.summary())
            tc_label.setStyleSheet(f"color: {_INK}; font-weight: 600;")
            tc_label.setToolTip(H.TIME_OF_CONCENTRATION)
            ctx_form.addRow("Responds in (Tc):", tc_label)
            path = getattr(self._tc, "flow_path_m", None)
            if path:
                channel = getattr(self._tc, "channel_length_m", 0.0) or 0.0
                text = f"{path:,.0f} m measured down the flow path"
                if channel > 0:
                    text += f", last {channel:,.0f} m in a channel"
                path_label = self._value_label(text)
                # Why the channel leg matters, on the row that reports it: channel
                # flow is about twice the speed of shallow concentrated flow, so
                # leaving it at zero overstates Tc and undersizes the overflow.
                path_label.setToolTip(H.CHANNEL_LENGTH)
                ctx_form.addRow("Longest flow path:", path_label)

            legs = getattr(self._tc, "leg_slopes", None)
            if legs:
                # Per-leg rather than one average, because hillslopes are concave and
                # the sheet-flow leg — the slowest and most influential — sits on the
                # steepest part of the profile.
                ctx_form.addRow("Slope by leg:", self._value_label(
                    f"sheet {legs[0] * 100:,.1f}%  ·  shallow {legs[1] * 100:,.1f}%"
                    f"  ·  channel {legs[2] * 100:,.1f}%"))
            for warning in self._tc.warnings:
                warn = QLabel(f"⚠ {warning}")
                warn.setWordWrap(True)
                warn.setStyleSheet(f"color: {_WARN}; font-size: 10.5px;")
                ctx_form.addRow(warn)
        else:
            missing = QLabel(
                "Not computed — run the baseline and draw a feature, and the plugin "
                "will measure the flow path and work out how fast this catchment "
                "responds."
            )
            missing.setWordWrap(True)
            missing.setStyleSheet(f"color: {_FAINT}; font-size: 10.5px;")
            ctx_form.addRow("Responds in (Tc):", missing)

        if self._idf is None or not self._idf.has_data():
            no_idf = QLabel(
                "No site rainfall data entered, so every row below is a judgement "
                "rather than a lookup. Enter a HIRDS table on the Baseline tab to get "
                "the intensity for a storm of exactly this catchment's duration."
            )
            no_idf.setWordWrap(True)
            no_idf.setStyleSheet(f"color: {_WARN}; font-size: 10.5px;")
            ctx_form.addRow(no_idf)
        elif self._idf.site:
            ctx_form.addRow("Rainfall data:", self._value_label(
                f"{self._idf.source or 'IDF'} · {self._idf.site}"))

        layout.addWidget(ctx)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Design intensity", "Basis", "Peak flow", f"Spillway @ {self._head_m:.2f} m"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setToolTip(H.DESIGN_INTENSITY_TABLE)
        self._populate(current_intensity)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table)

        custom = QGroupBox("Or set it directly")
        custom_form = QFormLayout(custom)
        self.spin_intensity = QDoubleSpinBox()
        self.spin_intensity.setRange(0.1, 500.0)
        self.spin_intensity.setDecimals(1)
        self.spin_intensity.setSingleStep(5.0)
        self.spin_intensity.setSuffix(" mm/hr")
        self.spin_intensity.setValue(float(current_intensity or 40.0))
        self.spin_intensity.setToolTip(H.PEAK_INTENSITY)
        self.spin_intensity.valueChanged.connect(self._refresh_comparison)
        custom_form.addRow("Peak intensity:", self.spin_intensity)
        layout.addWidget(custom)

        self._comparison = QGroupBox("What the other runoff methods would give")
        comp_layout = QVBoxLayout(self._comparison)
        self._comp_rows = QLabel("")
        self._comp_rows.setTextFormat(Qt.TextFormat.RichText)
        comp_layout.addWidget(self._comp_rows)

        note = QLabel(
            "Shown for comparison only. Selecting one here would size the overflow "
            "on a method the rest of the assessment is not using — change it on the "
            "Baseline tab if you want it, and re-run."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {_WARN}; font-size: 10.5px; font-style: italic;")
        comp_layout.addWidget(note)
        layout.addWidget(self._comparison)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _value_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {_MUTED};")
        return lbl

    def _populate(self, current_intensity):
        """One row per candidate intensity, all costed on the *current* basis.

        The IDF rows come first when the site's rainfall data has been entered,
        because they are the only rows that are a lookup rather than a choice: the
        rational method asks for the intensity of a storm lasting exactly as long as
        the catchment takes to respond, and with Tc and an IDF table that is a
        determined number.
        """
        rows = []
        if self._idf is not None and self._idf.has_data() and self._tc is not None:
            duration = self._tc.design_min
            for ari in self._idf.available_aris():
                intensity = self._idf.intensity_mm_hr(duration, ari)
                if intensity is None:
                    continue
                hint = (f"HIRDS {ari}-year depth over {duration:,.0f} min — the "
                        f"duration this catchment responds at")
                if self._idf.is_extrapolated(duration):
                    hint += ". Outside the entered durations, so held at the nearest end."
                rows.append((f"◆ {ari}-year at Tc = {duration:,.0f} min", intensity, hint))

        rows += [(f"{name} storm", i, f"{i:.0f} mm/hr preset") for name, i in _PRESETS]
        if self._rainfall_mm > 0:
            for d in _DURATIONS:
                rows.append((
                    f"This storm in {d:g} h", self._rainfall_mm / d,
                    "storm average — only valid if the catchment responds this slowly"
                    if d > 1 else "storm average",
                ))

        self.table.setRowCount(len(rows))
        fraction = self._fraction(self._basis)
        basis_txt = _BASIS_LABELS.get(self._basis, self._basis)
        best_row, best_gap = 0, None

        for r, (label, intensity, hint) in enumerate(rows):
            q = rational_peak_flow(fraction, intensity, self._area_m2)
            width = calculate_spillway_width(q, self._head_m)
            cells = [
                f"{label}  ·  {intensity:,.1f} mm/hr",
                f"{basis_txt}  ({fraction:.2f})",
                f"{q * 1000:,.1f} L/s",
                f"{width:,.2f} m",
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setToolTip(hint)
                    item.setData(Qt.ItemDataRole.UserRole, float(intensity))
                if c == 1:
                    item.setForeground(_qcolor(_FAINT))
                self.table.setItem(r, c, item)

            if current_intensity is not None:
                gap = abs(intensity - float(current_intensity))
                if best_gap is None or gap < best_gap:
                    best_gap, best_row = gap, r

        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        if self.table.rowCount():
            self.table.selectRow(best_row)

    # ------------------------------------------------------------------ behaviour

    def _fraction(self, basis):
        return peak_runoff_fraction(
            basis, rainfall_mm=self._rainfall_mm,
            coefficient=self._coefficient, cn=self._cn,
        )

    def _on_row_selected(self):
        items = self.table.selectedItems()
        if not items:
            return
        value = self.table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        if value is None:
            return
        self.spin_intensity.blockSignals(True)
        self.spin_intensity.setValue(float(value))
        self.spin_intensity.blockSignals(False)
        self._refresh_comparison()

    def _refresh_comparison(self):
        """The other two bases at the chosen intensity — informational, never chosen.

        Honest because the basis enters as a scalar on a basis-independent catchment
        area, so these are exact, not estimates. They are the answer to "what am I
        exposed to if my method is the wrong one".
        """
        intensity = self.spin_intensity.value()
        lines = []
        for basis in (BASIS_COEFFICIENT, BASIS_RAINFALL, BASIS_SCS):
            fraction = self._fraction(basis)
            q = rational_peak_flow(fraction, intensity, self._area_m2)
            width = calculate_spillway_width(q, self._head_m)
            label = _BASIS_LABELS.get(basis, basis)
            if basis == self._basis:
                lines.append(
                    f'<div style="color:{_INK}; font-weight:600;">'
                    f"{label} &nbsp;·&nbsp; {fraction:.2f} &nbsp;·&nbsp; "
                    f"{q * 1000:,.1f} L/s &nbsp;·&nbsp; <b>{width:,.2f} m</b>"
                    f' &nbsp;<span style="color:{_ACCENT};">← current</span></div>'
                )
            else:
                lines.append(
                    f'<div style="color:{_FAINT};">'
                    f"{label} &nbsp;·&nbsp; {fraction:.2f} &nbsp;·&nbsp; "
                    f"{q * 1000:,.1f} L/s &nbsp;·&nbsp; {width:,.2f} m</div>"
                )
        self._comp_rows.setText("".join(lines))

    # ------------------------------------------------------------------ result

    def get_intensity(self):
        """Chosen peak design intensity in mm/hr."""
        return self.spin_intensity.value()


def _qcolor(hex_str):
    from qgis.PyQt.QtGui import QBrush, QColor
    return QBrush(QColor(hex_str))
