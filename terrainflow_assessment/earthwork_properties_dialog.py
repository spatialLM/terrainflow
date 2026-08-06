import math

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from .core.registry.earthwork_types import get_type
from .core.sizing import (
    basin_volume_battered,
    batter_advisory,
    grade_advisory,
    trapezoid_section,
)
from .modules.earthwork_design import (
    Spillway,
    berm_height_estimate,
    bind_crest,
    calculate_capacity,
    calculate_diversion_discharge,
    calculate_spillway_width,
    effective_freeboard_m,
    effective_head_m,
    spillway_datum,
    spillway_policy,
    spillway_validity,
)
from .qgis import help_text as H

# Design-language tokens (see the reference "TerrainFlow panel visual language").
# Kept local to this dialog for now; the eventual repo theme.py can absorb them.
_INK = "#22302e"           # primary text
_MUTED = "#5f7176"         # secondary / italic notes
_HAIRLINE = "#dde4e5"      # card borders
_HAIRLINE_STRONG = "#c6d1d3"  # input borders / separators
_GROUND = "#eef1f2"        # dialog background
_SURFACE = "#ffffff"       # card / input background
_ACCENT = "#2e7d55"        # actions, selection, focus
_ACCENT_HOVER = "#256645"
_WATER = "#1273b5"         # actual water quantities ONLY (stored m³)
_GOOD = "#1e8449"          # traffic-light green (within envelope / pass)
_WARN = "#b9770e"          # traffic-light amber (advisory / uphill / converge)
_BAD = "#c0392b"           # traffic-light red (fail / needs engineer)

# One cohesive stylesheet so the dialog reads as part of the Workbench panel:
# white card group-boxes, hairline-bordered inputs with a green focus ring, an
# accented OK button and a ghost Cancel.
_DIALOG_QSS = f"""
QDialog {{ background: {_GROUND}; }}
QLabel {{ color: {_INK}; font-size: 12px; }}
QLineEdit, QDoubleSpinBox, QComboBox {{
    background: {_SURFACE};
    border: 1px solid {_HAIRLINE_STRONG};
    border-radius: 5px;
    padding: 4px 7px;
    color: {_INK};
    selection-background-color: {_ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {_ACCENT};
}}
QCheckBox {{ color: {_INK}; font-size: 12px; spacing: 6px; }}
QGroupBox {{
    background: {_SURFACE};
    border: 1px solid {_HAIRLINE};
    border-radius: 7px;
    margin-top: 12px;
    padding: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {_INK};
    font-weight: 650;
    font-size: 11px;
    background: {_GROUND};
}}
QDialogButtonBox QPushButton {{
    min-width: 84px;
    padding: 6px 14px;
    border-radius: 6px;
    font-weight: 600;
}}
QDialogButtonBox QPushButton:default {{
    background: {_ACCENT}; color: #ffffff; border: 1px solid {_ACCENT};
}}
QDialogButtonBox QPushButton:default:hover {{ background: {_ACCENT_HOVER}; }}
QDialogButtonBox QPushButton:!default {{
    background: {_SURFACE}; color: {_INK}; border: 1px solid {_HAIRLINE_STRONG};
}}
QDialogButtonBox QPushButton:!default:hover {{ border-color: {_ACCENT}; color: {_ACCENT}; }}
"""


class _InflowSparkline(QWidget):
    """Where the catchment arrives along the alignment.

    A flat line means the lumped "total capacity vs total inflow" verdict is safe. A
    spike means it is not: a swale with adequate total capacity can still go over the
    side where a drainage line crosses it. The amber tick is the overtopping station.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = []
        self._overtop_frac = None
        self.setFixedHeight(34)
        self.setMinimumWidth(160)

    def set_profile(self, profile, overtop_station=None):
        self._values = list((profile or {}).get("inflow_m3") or [])
        length = None
        stations = (profile or {}).get("stations") or []
        if stations:
            step = (profile or {}).get("station_length_m") or 0.0
            length = stations[-1] + step / 2.0
        self._overtop_frac = (
            float(overtop_station) / length
            if overtop_station is not None and length else None)
        self.update()

    def paintEvent(self, event):
        from qgis.PyQt.QtGui import QPainter, QPen

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#f2f6f7"))
        p.drawRoundedRect(0, 0, w, h, 4, 4)

        if not self._values:
            p.end()
            return
        peak = max(self._values) or 1.0
        n = len(self._values)
        bar_w = max(1.0, (w - 6) / n)
        p.setBrush(QColor(_WATER))
        for i, value in enumerate(self._values):
            bar_h = max(1.0, (h - 8) * (value / peak))
            p.drawRect(int(3 + i * bar_w), int(h - 4 - bar_h),
                       max(1, int(bar_w - 1)), int(bar_h))

        if self._overtop_frac is not None:
            x = int(3 + self._overtop_frac * (w - 6))
            p.setPen(QPen(QColor(_WARN), 1.5))
            p.drawLine(x, 2, x, h - 2)
        p.end()


class EarthworkPropertiesDialog(QDialog):
    """
    Popup dialog shown after an earthwork is drawn.
    Lets the user set name, depth, width, and companion berm option.
    Displays calculated capacity in real time.
    """

    def __init__(self, ew_type, geometry, parent=None, earthwork=None,
                 peak_inflow_m3=None, crest_elevation=None, duration_hours=None,
                 dem_path=None, soil_name=None, cn=None, overflow_options=None,
                 own_elevation=None, catchment_m2=None, count_infiltration=False,
                 rim_elevation=None, invert_elevation=None,
                 peak_flow_m3s=None, upstream_flow_m3s=0.0,
                 harvesting_coefficient=False,
                 inflow_profile=None, overtop_station=None, overtop_surplus=0.0):
        super().__init__(parent)
        self.ew_type = ew_type
        self.geometry = geometry
        self._earthwork = earthwork
        self._editing = earthwork is not None
        self._peak_inflow_m3 = peak_inflow_m3   # None for freehand swales
        self._crest_elevation = crest_elevation  # pre-sampled for dam type
        self._duration_hours = duration_hours    # storm duration for spillway sizing
        self._dem_path = dem_path                # for dam wall height/volume
        self._soil_name = soil_name              # site earthwork soil → batter/grade advisory
        self._cn = cn                            # curve number (advisory cross-check context)
        self._overflow_options = overflow_options or []  # [(id, name[, elev])] of OTHERS
        self._own_elevation = own_elevation      # DEM at this feature's centroid, or None
        self._catchment_m2 = catchment_m2        # direct contributing area (flow_graph)
        self._count_infiltration = count_infiltration  # does soakage count as capture?
        self._overflow_elevations = {}           # id → elevation (None when unknown)
        # Spillway datums, sampled by the controller from footprint.pour_point:
        # the rim is the lowest containing ground (where it would spill unaided),
        # the invert the burned floor. Both None without a DEM — the crest is then
        # editable but unanchored, and the dialog says so.
        self._rim_elevation = rim_elevation
        self._invert_elevation = invert_elevation
        self._spillway_binding = False           # re-entrancy guard for crest ↔ drop
        # Per-type spillway policy: a swale overflows over a low sill in its own bank,
        # an embankment over a designed wall, and the published 0.30 m figure describes
        # only the second. Resolved once here so every row below reads the same source.
        (self._policy_freeboard, self._policy_head,
         self._policy_head_band) = spillway_policy(ew_type)
        # Peak flow (m³/s) from the rational method, with upstream overflow already
        # cascaded in; the controller owns that because only it knows the network.
        self._peak_flow_m3s = peak_flow_m3s
        self._upstream_flow_m3s = float(upstream_flow_m3s or 0.0)
        self._harvesting_coefficient = harvesting_coefficient
        # Where the catchment arrives along the alignment, and where (if anywhere)
        # arriving water outruns the storage upstream of it.
        self._inflow_profile = inflow_profile
        self._overtop_station = overtop_station
        self._overtop_surplus = float(overtop_surplus or 0.0)
        # Registry sizing policy for this type; None for unregistered types → the
        # historical hardcoded ranges/defaults apply as fallbacks throughout.
        try:
            self._cfg = get_type(ew_type)
        except KeyError:
            self._cfg = None

        type_labels = {"diversion": "Diversion Drain"}
        type_label = type_labels.get(ew_type, ew_type.capitalize())
        self.setWindowTitle(f"{'Edit' if self._editing else 'New'} {type_label} Properties")
        self.setMinimumWidth(360)
        self._build_ui(earthwork)
        self._update_capacity()

    def _build_ui(self, ew=None):
        self.setStyleSheet(_DIALOG_QSS)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setHorizontalSpacing(10)
        form.setContentsMargins(2, 2, 2, 4)

        # Name
        self.edit_name = QLineEdit(ew.name if ew else f"New {self.ew_type.capitalize()}")
        form.addRow("Name:", self.edit_name)

        # Type (read-only label)
        type_labels = {"diversion": "Diversion Drain"}
        form.addRow("Type:", QLabel(type_labels.get(self.ew_type, self.ew_type.capitalize())))

        # Dam: crest elevation instead of depth
        if self.ew_type == "dam":
            self.spin_crest_elev = QDoubleSpinBox()
            self.spin_crest_elev.setRange(-500, 9000)
            self.spin_crest_elev.setDecimals(2)
            self.spin_crest_elev.setSuffix(" m")
            if ew and ew.crest_elevation is not None:
                self.spin_crest_elev.setValue(ew.crest_elevation)
            elif self._crest_elevation is not None:
                self.spin_crest_elev.setValue(self._crest_elevation)
            self.spin_crest_elev.setToolTip(
                "Absolute elevation of the dam crest (top of the wall).\n\n"
                "Pre-filled from the highest ground the drawn line touches.\n"
                "All cells under the wall will be raised to this elevation,\n"
                "so the wall height varies with the valley shape beneath it.\n\n"
                "Water will pool behind the dam up to this level.\n"
                "Run Re-analyse with Earthworks to see retained volume."
            )
            self.spin_crest_elev.valueChanged.connect(self._update_capacity)
            form.addRow("Crest elevation:", self.spin_crest_elev)

            # Keying in is now real geometry, not a what-if: the wall is extended to
            # the natural abutments so the drawn dam, its capacity and the verification
            # burn all describe the same structure. It was previously an idealised
            # capacity estimate that left the short wall in place — which is how a dam
            # could report 69% full while water visibly ran over it on the map.
            self.chk_key_banks = QCheckBox("Extend wall into the banks (key in)")
            self.chk_key_banks.setChecked(
                bool(getattr(ew, "key_into_banks", False)) if ew else True
            )
            self.chk_key_banks.setToolTip(
                "On (default): extend each end of the wall along its own bearing\n"
                "until the ground rises to the crest elevation, so water cannot flow\n"
                "around the ends. The drawn line is replaced by the wall that would\n"
                "actually have to be built — often noticeably longer — and both the\n"
                "capacity and the verification burn use that wall.\n\n"
                "You are told how far each end moved. If an end finds no ground at\n"
                "crest height within 250 m you get a warning: the design does not\n"
                "impound as drawn, and the crest is too high for this location.\n\n"
                "Off: keep the wall exactly as drawn. Water escapes around the ends\n"
                "if it stops short of high ground, and the reported storage is what\n"
                "the short wall actually holds."
            )
            self.chk_key_banks.toggled.connect(self._update_capacity)
            form.addRow("", self.chk_key_banks)
            self.spin_depth = None
        else:
            self.spin_crest_elev = None
            # Depth — range/default from the registry sizing policy for the type.
            depth_lo, depth_hi = self._cfg.depth_range if self._cfg else (0.1, 10.0)
            depth_seed = ew.depth if ew else (self._cfg.default_depth if self._cfg else 0.5)
            self.spin_depth = QDoubleSpinBox()
            self.spin_depth.setRange(depth_lo, depth_hi)
            self.spin_depth.setValue(depth_seed)
            self.spin_depth.setDecimals(2)
            self.spin_depth.setSuffix(" m")
            self.spin_depth.valueChanged.connect(self._update_capacity)
            form.addRow("Depth:", self.spin_depth)

        # Width — only for types that take a top width (a basin's footprint comes
        # from the drawn polygon, so it gets no meaningless Width row).
        if self._cfg is None or "top_width" in self._cfg.independent_dims:
            width_lo, width_hi = self._cfg.top_width_range if self._cfg else (0.1, 100.0)
            width_seed = ew.width if ew else (
                self._cfg.default_top_width if self._cfg
                else (2.0 if self.ew_type == "dam" else 1.0)
            )
            self.spin_width = QDoubleSpinBox()
            self.spin_width.setRange(width_lo, width_hi)
            self.spin_width.setValue(width_seed)
            self.spin_width.setDecimals(2)
            self.spin_width.setSuffix(" m")
            self.spin_width.valueChanged.connect(self._update_capacity)
            lbl_width = "Wall thickness:" if self.ew_type == "dam" else "Width:"
            form.addRow(lbl_width, self.spin_width)
        else:
            self.spin_width = None

        # Bottom width (channels only) — the canonical cross-section input, centred
        # under the top width (symmetric trapezoid). The side batter is derived and
        # shown in degrees below. setValue before connect avoids an early fire.
        # Registry-driven: any type whose bottom_width is a derived dimension.
        if self._cfg is not None and "bottom_width" in self._cfg.derived_dims:
            self.spin_bottom_width = QDoubleSpinBox()
            self.spin_bottom_width.setRange(0.05, 100.0)
            self.spin_bottom_width.setDecimals(2)
            self.spin_bottom_width.setSingleStep(0.1)
            self.spin_bottom_width.setSuffix(" m")
            seed_bottom = (ew.bottom_width_m if ew
                           else max(0.05, self.spin_width.value() - 2 * 0.5))
            self.spin_bottom_width.setValue(min(seed_bottom, self.spin_width.value()))
            self.spin_bottom_width.setToolTip(
                "Width of the channel floor, centred under the top width.\n"
                "Together with depth and top width this sets the side batter\n"
                "(shown below). A narrower bottom → steeper batter."
            )
            self.spin_bottom_width.valueChanged.connect(self._update_capacity)
            form.addRow("Bottom width:", self.spin_bottom_width)

            self.lbl_side_slope = QLabel("—")
            self.lbl_side_slope.setToolTip(
                "Side batter angle from horizontal, derived from top/bottom width and\n"
                "depth. 45° = 1:1; a smaller angle is flatter/more stable; 90° = vertical."
            )
            form.addRow("Side slope:", self.lbl_side_slope)
        else:
            self.spin_bottom_width = None
            self.lbl_side_slope = None

        # Gradient — registry-driven (types with gradient_pct as an independent dim)
        if self._cfg is not None and "gradient_pct" in self._cfg.independent_dims:
            self.spin_gradient = QDoubleSpinBox()
            self.spin_gradient.setRange(0.1, 5.0)
            self.spin_gradient.setValue(ew.gradient_pct if ew else 1.0)
            self.spin_gradient.setDecimals(1)
            self.spin_gradient.setSuffix(" %")
            self.spin_gradient.setSingleStep(0.1)
            self.spin_gradient.setToolTip(
                "Channel gradient — the fall in elevation per 100 m of drain length.\n\n"
                "Recommended range: 0.5–2.0 %\n"
                "  0.5 % — minimum to maintain flow, suits gentle slopes\n"
                "  1.0 % — standard design gradient\n"
                "  2.0 % — steep; use erosion protection (rock mulch / vegetation)\n"
                "  >2.0 % — significant erosion risk; consider drop structures\n\n"
                "Higher gradient → higher discharge capacity but greater erosion risk."
            )
            self.spin_gradient.valueChanged.connect(self._update_capacity)
            form.addRow("Drain gradient:", self.spin_gradient)
        else:
            self.spin_gradient = None

        # Basin wall batter — polygon storage types. Analytic capacity honours the
        # batter (inset-prism model); the DEM burn stays a vertical drop this phase.
        if self._cfg is not None and self._cfg.geom_type == "Polygon" and self._cfg.has_storage:
            self.spin_wall_slope = QDoubleSpinBox()
            self.spin_wall_slope.setRange(0.0, 5.0)
            self.spin_wall_slope.setSingleStep(0.25)
            self.spin_wall_slope.setDecimals(2)
            self.spin_wall_slope.setSuffix(" : 1")
            self.spin_wall_slope.setValue(
                ew.wall_slope if ew else self._cfg.default_side_slope
            )
            self.spin_wall_slope.setToolTip(
                "Wall batter as horizontal run per unit of depth (H:V).\n"
                "0 : 1 = vertical walls; 1 : 1 = 45°; flatter is more stable.\n\n"
                "The stored capacity accounts for the sloped walls. Note the DEM\n"
                "burn (Re-analyse) still carves vertical walls this phase — the\n"
                "verification comparison will surface the difference."
            )
            self.spin_wall_slope.valueChanged.connect(self._update_capacity)
            form.addRow("Wall batter:", self.spin_wall_slope)

            self.lbl_basin_converge = QLabel("")
            self.lbl_basin_converge.setWordWrap(True)
            self.lbl_basin_converge.setStyleSheet("color: #b9770e; font-style: italic;")
            form.addRow("", self.lbl_basin_converge)
        else:
            self.spin_wall_slope = None
            self.lbl_basin_converge = None

        # Companion berm (swales only)
        self.chk_companion = QCheckBox("Build companion berm on downhill side")
        self.chk_companion.setChecked(ew.companion_berm if ew else False)
        self.chk_companion.setVisible(self.ew_type == "swale")
        if self.ew_type == "swale":
            self.chk_companion.setToolTip(
                "Excavated material is placed on the downhill side of the swale,\n"
                "forming a retaining berm. Volume is conserved — the berm height is\n"
                "calculated from the excavated volume at 75% compaction.\n\n"
                "The berm raises the effective water level above the original ground,\n"
                "significantly increasing total water retention capacity."
            )
            self.chk_companion.stateChanged.connect(self._update_capacity)
            form.addRow("", self.chk_companion)

        # Per-feature soil. Soil is rarely uniform across a farm and infiltration is
        # the term most sensitive to it, so a basin in a clay hollow can be sized on
        # clay without misrepresenting the loam everywhere else. Blank = site default.
        if self._cfg is not None and self._cfg.has_cut:
            self.combo_soil = QComboBox()
            self.combo_soil.addItem(
                f"Site default ({self._soil_name or 'Loam'})", None)
            for _name in ("Sand", "Sandy loam", "Loam", "Clay loam", "Clay"):
                self.combo_soil.addItem(_name, _name)
            own = getattr(ew, "soil_name", None) if ew else None
            if own:
                idx = self.combo_soil.findData(own)
                if idx >= 0:
                    self.combo_soil.setCurrentIndex(idx)
            self.combo_soil.setToolTip(H.FEATURE_SOIL)
            self.combo_soil.currentIndexChanged.connect(self._update_capacity)
            form.addRow("Soil here:", self.combo_soil)
        else:
            self.combo_soil = None

        # Overflow routing — storage-category types can name the feature their
        # overflow spills into (id-based, survives renames/reorders).
        if (self._cfg is not None and self._cfg.category == "storage"
                and self._overflow_options):
            self.combo_overflow = QComboBox()
            self.combo_overflow.addItem("Auto (downslope)", None)
            current_target = getattr(ew, "overflow_target_id", None) if ew else None
            for opt in self._overflow_options:
                opt_id, opt_name = opt[0], opt[1]
                self._overflow_elevations[opt_id] = opt[2] if len(opt) > 2 else None
                self.combo_overflow.addItem(opt_name, opt_id)
                if opt_id == current_target:
                    self.combo_overflow.setCurrentIndex(self.combo_overflow.count() - 1)
            self.combo_overflow.setToolTip(
                "Where this feature's overflow goes once it is full.\n\n"
                "Auto: the nearest feature downslope (elevation heuristic).\n"
                "A named target only receives water when it actually sits\n"
                "downslope of this feature — water can't flow uphill. An uphill\n"
                "choice is flagged below and its water goes downslope instead."
            )
            self.combo_overflow.currentIndexChanged.connect(self._update_overflow_warning)
            form.addRow("Overflows to:", self.combo_overflow)

            self.lbl_overflow_warning = QLabel("")
            self.lbl_overflow_warning.setWordWrap(True)
            self.lbl_overflow_warning.setStyleSheet("color: #b9770e; font-style: italic;")
            form.addRow("", self.lbl_overflow_warning)
            self._update_overflow_warning()
        else:
            self.combo_overflow = None
            self.lbl_overflow_warning = None

        layout.addLayout(form)

        # Capacity display
        cap_group_title = "Discharge Capacity" if self.ew_type == "diversion" else "Calculated Capacity"
        cap_group = QGroupBox(cap_group_title)
        cap_layout = QFormLayout(cap_group)

        if self.ew_type == "diversion":
            # Diversion drain: show Manning's discharge + length
            length_m = self.geometry.length()
            lbl_length = QLabel(f"{length_m:,.1f} m")
            cap_layout.addRow("Drain length:", lbl_length)

            self.lbl_capacity_m3 = QLabel("—")
            self.lbl_capacity_m3.setToolTip(
                "Peak discharge capacity using Manning's equation.\n"
                "Q = (1/n) × A × R^(2/3) × S^(1/2)\n"
                "Manning's n = 0.025 (compacted earthen channel)\n"
                "Trapezoidal cross-section, 1:1 side slopes."
            )
            cap_layout.addRow("Discharge capacity:", self.lbl_capacity_m3)
            self.lbl_capacity_l = QLabel("—")   # repurposed: capacity vs inflow status
            cap_layout.addRow("", self.lbl_capacity_l)
            self.lbl_berm_height = None
        else:
            # Swale length — shown so it can be compared against the recommended length
            if self.ew_type == "swale":
                length_m = self.geometry.length()
                lbl_length = QLabel(f"{length_m:,.1f} m")
                lbl_length.setToolTip(
                    "Total length of the swale as drawn on the map.\n"
                    "Compare with the Recommended length below — if this swale\n"
                    "is shorter, consider extending it or adjusting depth / width."
                )
                cap_layout.addRow("Swale length:", lbl_length)

            self.lbl_capacity_m3 = QLabel("—")
            self.lbl_capacity_l  = QLabel("—")
            # Stored water volume → the water-quantity blue (the only on-grammar use).
            for _lbl in (self.lbl_capacity_m3, self.lbl_capacity_l):
                _lbl.setStyleSheet("font-weight: 600; color: #1273b5;")
            cap_layout.addRow("Volume (m³):", self.lbl_capacity_m3)
            cap_layout.addRow("Volume (L):",  self.lbl_capacity_l)
            if self.ew_type == "swale":
                self.lbl_berm_height = QLabel("")
                self.lbl_berm_height.setStyleSheet("color: #5f7176; font-style: italic;")
                cap_layout.addRow(self.lbl_berm_height)
            else:
                self.lbl_berm_height = None
            if self.ew_type == "berm":
                cap_layout.addRow(QLabel("Berms are barriers — no storage capacity."))
            if self.ew_type == "dam":
                self.lbl_wall_volume = QLabel("—")
                self.lbl_wall_volume.setToolTip(
                    "Estimated volume of earthfill needed to construct the dam wall.\n\n"
                    "Calculated as: sum along the wall of (crest − ground) × wall thickness × segment length.\n"
                    "This is a rectangular cross-section approximation — add ~20% for side slopes."
                )
                cap_layout.addRow("Wall fill volume:", self.lbl_wall_volume)

                self.lbl_max_height = QLabel("—")
                self.lbl_max_height.setToolTip(
                    "Height of the tallest point of the dam wall above the ground beneath it.\n\n"
                    "Lower is better — a maximum height under 4–5 m is generally\n"
                    "considered feasible for a farm dam without engineering certification.\n"
                    "Higher walls require professional design and may need regulatory approval."
                )
                cap_layout.addRow("Max wall height:", self.lbl_max_height)

                lbl_note = QLabel(
                    "Retained water volume depends on valley shape.\n"
                    "Run Re-analyse with Earthworks to see ponded volume."
                )
                lbl_note.setStyleSheet("color: #5f7176; font-style: italic;")
                lbl_note.setWordWrap(True)
                cap_layout.addRow(lbl_note)
            else:
                self.lbl_wall_volume = None
                self.lbl_max_height = None

        # Does this swale, as drawn, hold its event? Deficit leads; recommended
        # length is demoted to a muted secondary line — see H.SWALE_DEFICIT for why.
        if self.ew_type == "swale" and self._peak_inflow_m3 is not None:
            sep = QLabel("─" * 30)
            sep.setStyleSheet("color: #c6d1d3;")
            cap_layout.addRow(sep)

            catch_txt = ""
            if self._catchment_m2:
                catch_txt = f"   (from {self._catchment_m2 / 10_000.0:,.1f} ha draining here)"
            lbl_inflow = QLabel(f"{self._peak_inflow_m3:,.1f} m³{catch_txt}")
            lbl_inflow.setToolTip(H.DIRECT_CATCHMENT)
            cap_layout.addRow("Event inflow:", lbl_inflow)

            self.lbl_holds = QLabel("—")
            self.lbl_holds.setToolTip(H.SWALE_HOLDS)
            cap_layout.addRow("Holds over the event:", self.lbl_holds)

            self.lbl_verdict = QLabel("—")
            self.lbl_verdict.setWordWrap(True)
            self.lbl_verdict.setToolTip(H.SWALE_DEFICIT)
            cap_layout.addRow(self.lbl_verdict)

            self.lbl_req_length = QLabel("—")
            self.lbl_req_length.setStyleSheet("color: #8fa0a4; font-size: 10.5px;")
            self.lbl_req_length.setToolTip(H.SWALE_RECOMMENDED_LENGTH)
            cap_layout.addRow(self.lbl_req_length)

            self._swale_length_m = self.geometry.length()
        else:
            self.lbl_req_length = None
            self.lbl_holds = None
            self.lbl_verdict = None

        # Where along the alignment the catchment actually arrives. The lumped verdict
        # above is only safe while inflow is reasonably even; this is what shows when
        # it is not.
        if self._inflow_profile:
            self.spark_inflow = _InflowSparkline()
            self.spark_inflow.set_profile(self._inflow_profile, self._overtop_station)
            self.spark_inflow.setToolTip(H.INFLOW_PROFILE)
            cap_layout.addRow("Inflow along it:", self.spark_inflow)

            uniformity = self._inflow_profile.get("uniformity", 1.0)
            if self._overtop_station is not None:
                text = (f"⚠ Overtops about {self._overtop_station:,.0f} m along, "
                        f"{self._overtop_surplus:,.0f} m³ over — a check-bank near the "
                        f"peak, or start it further upslope.")
                colour = _WARN
            elif uniformity < 0.5:
                text = (f"Inflow is concentrated (uniformity {uniformity:.2f}); the "
                        f"total-vs-total verdict above is optimistic.")
                colour = _MUTED
            else:
                text = f"Inflow is reasonably even (uniformity {uniformity:.2f})."
                colour = _MUTED
            self.lbl_inflow_note = QLabel(text)
            self.lbl_inflow_note.setWordWrap(True)
            self.lbl_inflow_note.setStyleSheet(f"color: {colour}; font-size: 10.5px;")
            cap_layout.addRow(self.lbl_inflow_note)
        else:
            self.spark_inflow = None
            self.lbl_inflow_note = None

        # Channel batter feedback: narrowest width (min_dimension) + live soil advisory.
        if self._cfg is not None and "bottom_width" in self._cfg.derived_dims:
            self.lbl_min_dim = QLabel("—")
            self.lbl_min_dim.setToolTip(
                "Narrowest dimension of the cross-section (the channel bottom width).\n"
                "If this falls below the DEM cell size the feature burns at 1-cell width\n"
                "(routing effect only) — you'll see a warning when you re-analyse."
            )
            cap_layout.addRow("Bottom width (min):", self.lbl_min_dim)

            self.lbl_advisory = QLabel("")
            self.lbl_advisory.setWordWrap(True)
            self.lbl_advisory.setStyleSheet("font-style: italic;")
            cap_layout.addRow(self.lbl_advisory)
        else:
            self.lbl_min_dim = None
            self.lbl_advisory = None

        layout.addWidget(cap_group)

        # Spillway — the designed overflow point.
        #
        # Shown for every type that holds water, regardless of whether the baseline
        # has run. Previously the whole group was gated on a known peak inflow, so
        # the crest — a decision about the feature's own geometry, not about any
        # storm — could not be set until after the analysis it feeds. Only the
        # *sizing* rows genuinely need the flow, so only they are conditional now.
        if self.ew_type in ("swale", "dam", "basin"):
            self.grp_spillway = QGroupBox("Spillway — designed overflow")
            self.grp_spillway.setCheckable(True)
            existing = getattr(ew, "spillway", None) if ew else None
            self.grp_spillway.setChecked(existing is not None)
            self.grp_spillway.setToolTip(H.SPILLWAY_GROUP)
            spill_layout = QFormLayout(self.grp_spillway)

            # Datum. Every other number here is relative to it, so it is stated
            # rather than assumed.
            if self._rim_elevation is not None:
                rim_txt = f"{self._rim_elevation:.2f} m"
                rim_style = f"color: {_MUTED};"
            else:
                rim_txt = "unknown — load a DEM to anchor the crest"
                rim_style = f"color: {_WARN}; font-style: italic;"
            if (self._rim_elevation is not None and self.ew_type == "swale"
                    and getattr(ew, "companion_berm", False)):
                # The burn raises the berm before taking its own pour point, so the real
                # spill level is higher than this. Said out loud rather than folded in:
                # the berm is one-sided, so where the low point is at an end it adds
                # nothing, and quietly crediting it would claim headroom that is not there.
                rim_txt += "  (excludes the companion berm)"
            lbl_rim = QLabel(rim_txt)
            lbl_rim.setStyleSheet(rim_style)
            lbl_rim.setToolTip(H.SPILLWAY_RIM)
            spill_layout.addRow("Rim (natural spill level):", lbl_rim)

            self.spin_spillway_crest = QDoubleSpinBox()
            self.spin_spillway_crest.setRange(-500, 9000)
            self.spin_spillway_crest.setDecimals(2)
            self.spin_spillway_crest.setSingleStep(0.05)
            self.spin_spillway_crest.setSuffix(" m")
            self.spin_spillway_crest.setToolTip(H.SPILLWAY_CREST)
            spill_layout.addRow("Crest elevation:", self.spin_spillway_crest)

            self.spin_spillway_drop = QDoubleSpinBox()
            self.spin_spillway_drop.setRange(0.0, 50.0)
            self.spin_spillway_drop.setDecimals(2)
            self.spin_spillway_drop.setSingleStep(0.05)
            self.spin_spillway_drop.setSuffix(" m")
            self.spin_spillway_drop.setEnabled(self._rim_elevation is not None)
            self.spin_spillway_drop.setToolTip(H.SPILLWAY_DROP)
            spill_layout.addRow("Below rim:", self.spin_spillway_drop)

            self.spin_spillway_head = QDoubleSpinBox()
            self.spin_spillway_head.setRange(0.05, 2.0)
            self.spin_spillway_head.setDecimals(2)
            self.spin_spillway_head.setSingleStep(0.05)
            self.spin_spillway_head.setSuffix(" m")
            # A fresh spillway takes its type's design head — a swale spills over a low
            # sill in its own bank and wants far less than an embankment does.
            self.spin_spillway_head.setValue(
                existing.head_m if existing is not None else self._policy_head)
            self.spin_spillway_head.setToolTip(H.SPILLWAY_HEAD)
            spill_layout.addRow("Design head (target):", self.spin_spillway_head)

            # Read-only counterpart, shown only once a width is committed: then the head
            # is the consequence rather than the choice, and it is the number the
            # freeboard is actually spent on.
            self.lbl_actual_head = QLabel("")
            self.lbl_actual_head.setToolTip(H.SPILLWAY_ACTUAL_HEAD)
            self.row_actual_head = QLabel("Head at that width:")
            spill_layout.addRow(self.row_actual_head, self.lbl_actual_head)

            self.spin_spillway_freeboard = QDoubleSpinBox()
            self.spin_spillway_freeboard.setRange(0.0, 1.0)
            self.spin_spillway_freeboard.setDecimals(2)
            self.spin_spillway_freeboard.setSingleStep(0.05)
            self.spin_spillway_freeboard.setSuffix(" m")
            self.spin_spillway_freeboard.setValue(
                effective_freeboard_m(existing, self.ew_type))
            self.spin_spillway_freeboard.setToolTip(H.SPILLWAY_FREEBOARD)
            spill_layout.addRow("Freeboard (min):", self.spin_spillway_freeboard)

            # Peak flow, supplied by the controller from the rational method with the
            # upstream cascade already added. It was previously derived here as event
            # volume / event duration — the storm *average*, which sized a 2.8 ha
            # spillway at 7 cm.
            if self._peak_flow_m3s is not None:
                bits = [f"{self._peak_flow_m3s * 1000:,.1f} L/s"]
                if self._upstream_flow_m3s:
                    bits.append(
                        f"({(self._peak_flow_m3s - self._upstream_flow_m3s) * 1000:,.1f}"
                        f" own + {self._upstream_flow_m3s * 1000:,.1f} from upslope)")
                lbl_qdesign = QLabel("  ".join(bits))
                lbl_qdesign.setToolTip(H.SPILLWAY_DESIGN_FLOW_PEAK)
                spill_layout.addRow("Peak design flow:", lbl_qdesign)

            self.lbl_spillway_width = QLabel("—")
            self.lbl_spillway_width.setStyleSheet(f"font-weight: bold; color: {_INK};")
            self.lbl_spillway_width.setToolTip(H.SPILLWAY_WIDTH)
            spill_layout.addRow("Min spillway width:", self.lbl_spillway_width)

            # Built width. Until now the stored width WAS the computed requirement, so
            # "is it big enough" could not be asked — spillway_validity carried the
            # check and nothing could reach it.
            width_row = QHBoxLayout()
            width_row.setContentsMargins(0, 0, 0, 0)
            width_row.setSpacing(6)
            self.spin_built_width = QDoubleSpinBox()
            self.spin_built_width.setRange(0.0, 200.0)
            self.spin_built_width.setDecimals(2)
            self.spin_built_width.setSingleStep(0.1)
            self.spin_built_width.setSuffix(" m")
            self.spin_built_width.setToolTip(H.SPILLWAY_BUILT_WIDTH)
            width_row.addWidget(self.spin_built_width, 1)
            self.chk_width_auto = QCheckBox("auto")
            self.chk_width_auto.setChecked(
                existing.width_auto if existing is not None else True)
            self.chk_width_auto.setToolTip(H.SPILLWAY_BUILT_WIDTH)
            width_row.addWidget(self.chk_width_auto)
            spill_layout.addRow("Built width:", width_row)

            if existing is not None and existing.width_m:
                self.spin_built_width.setValue(float(existing.width_m))
            self.spin_built_width.valueChanged.connect(self._update_spillway_sizing)
            self.chk_width_auto.toggled.connect(self._update_spillway_sizing)

            sited = existing.point_wkt if existing is not None else None
            self.lbl_spillway_site = QLabel(
                "placed on the map" if sited else "not sited — use Place Spillway on the map"
            )
            self.lbl_spillway_site.setStyleSheet(
                f"color: {_MUTED if sited else _WARN}; font-size: 10.5px; font-style: italic;")
            self.lbl_spillway_site.setToolTip(H.SPILLWAY_LOCATION)
            spill_layout.addRow("Location:", self.lbl_spillway_site)

            # The inlet, read-only. It is a separate structure with a separate job — a
            # protected entry, not a weir — so it carries no head or width here; showing
            # one would invent a design procedure this module does not implement. But
            # the dialog previously did not mention it at all, so a placed inlet was
            # invisible from the only screen that claims to describe the feature.
            inlet = getattr(ew, "inflow_spillway", None) if ew else None
            inlet_sited = inlet is not None and inlet.point_wkt
            if inlet_sited and inlet.crest_elevation is not None:
                inlet_txt = f"placed at {inlet.crest_elevation:.2f} m"
            elif inlet_sited:
                inlet_txt = "placed on the map"
            else:
                inlet_txt = "not sited — use Inflow Spillway on the map"
            self.lbl_inlet_site = QLabel(inlet_txt)
            self.lbl_inlet_site.setStyleSheet(
                f"color: {_MUTED}; font-size: 10.5px; font-style: italic;")
            self.lbl_inlet_site.setToolTip(H.SPILLWAY_INLET)
            spill_layout.addRow("Inlet:", self.lbl_inlet_site)

            self.lbl_spillway_warn = QLabel("")
            self.lbl_spillway_warn.setWordWrap(True)
            self.lbl_spillway_warn.setStyleSheet(
                f"color: {_BAD}; font-size: 10.5px;")
            spill_layout.addRow(self.lbl_spillway_warn)

            self._spillway_point_wkt = sited
            self._spillway_auto = existing.auto if existing is not None else True
            self._seed_spillway(existing)

            self.spin_spillway_crest.valueChanged.connect(self._on_spillway_crest_changed)
            self.spin_spillway_drop.valueChanged.connect(self._on_spillway_drop_changed)
            self.spin_spillway_head.valueChanged.connect(self._on_spillway_head_changed)
            # Freeboard moves the ceiling of the crest band exactly as head does, so it
            # has to re-bind the crest through the new band rather than only re-warn.
            self.spin_spillway_freeboard.valueChanged.connect(
                self._on_spillway_head_changed)
            self.grp_spillway.toggled.connect(self._update_spillway_sizing)
            self._update_spillway_sizing()

            layout.addWidget(self.grp_spillway)
        else:
            self.grp_spillway = None
            self.spin_spillway_crest = None
            self.spin_spillway_drop = None
            self.spin_spillway_head = None
            self.spin_spillway_freeboard = None
            self.lbl_actual_head = None
            self.row_actual_head = None
            self.spin_built_width = None
            self.chk_width_auto = None
            self.lbl_spillway_width = None
            self.lbl_spillway_warn = None
            self.lbl_spillway_site = None
            self.lbl_inlet_site = None
            self._spillway_point_wkt = None
            self._spillway_auto = True

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_capacity(self):
        if self.ew_type == "dam":
            self.lbl_capacity_m3.setText("—")
            self.lbl_capacity_l.setText("—")
            if self._dem_path and self.lbl_wall_volume is not None:
                crest = self.spin_crest_elev.value()
                thickness = self.spin_width.value()
                max_h, wall_vol = self._calc_dam_wall_metrics(crest, thickness)
                self.lbl_wall_volume.setText(f"{wall_vol:,.0f} m³")
                colour = "#c0392b" if max_h > 5 else "#b9770e" if max_h > 3 else "#1e8449"
                self.lbl_max_height.setText(
                    f'<span style="color:{colour}; font-weight:bold;">{max_h:.1f} m</span>'
                    + ("  ⚠ may need engineer" if max_h > 5 else "  ✓ feasible" if max_h <= 4 else "")
                )
                self.lbl_max_height.setTextFormat(1)  # Qt.RichText
            return

        if self.ew_type == "diversion":
            depth = self.spin_depth.value()
            width = self.spin_width.value()
            gradient = self.spin_gradient.value()
            side_slope = self._current_side_slope()
            # width is the top width; the bottom drives the batter → Manning's Q.
            bottom = self._current_bottom_width(width) or max(0.05, width - 2 * side_slope * depth)
            q = calculate_diversion_discharge(depth, width, gradient, bottom_width=bottom)
            self.lbl_capacity_m3.setText(f"{q:.4f} m³/s  ({q * 1000:.1f} L/s)")
            self._update_channel_feedback(width, bottom, depth, side_slope, gradient)
            # Compare against peak inflow rate if available
            if self._peak_inflow_m3 is not None:
                # peak_inflow_m3 is total volume — inflow rate is not directly derivable
                # here (we don't know storm duration), so show a note instead.
                self.lbl_capacity_l.setText(
                    "Peak inflow volume: see swale properties\n"
                    "for direct comparison."
                )
                self.lbl_capacity_l.setStyleSheet("color: #5f7176; font-style: italic;")
            else:
                self.lbl_capacity_l.setText(
                    f"Manning's n = 0.025 · trapezoidal {side_slope:.2g}:1 side slopes"
                )
                self.lbl_capacity_l.setStyleSheet("color: #5f7176; font-style: italic;")
            return

        depth = self.spin_depth.value()
        width = self.spin_width.value() if self.spin_width is not None else 0.0
        companion = self.chk_companion.isChecked() if self.ew_type == "swale" else False
        # Channels (swale) take the bottom width directly from the control; non-channel
        # storage (basin) falls back to the feature's stored bottom width.
        if self.spin_bottom_width is not None:
            side_slope = self._current_side_slope()
            bottom_width = self._current_bottom_width(width)
        else:
            side_slope = None
            bottom_width = getattr(self._earthwork, "bottom_width_m", None)
        batter_run = (
            self.spin_wall_slope.value() * depth
            if self.spin_wall_slope is not None else None
        )
        m3, litres = calculate_capacity(
            self.ew_type, self.geometry, depth, width, companion,
            bottom_width=bottom_width, batter_run=batter_run,
        )
        self.lbl_capacity_m3.setText(f"{m3:,.2f}")
        self.lbl_capacity_l.setText(f"{litres:,.0f}")
        self._update_basin_converge(depth)

        if side_slope is not None:
            self._update_channel_feedback(width, bottom_width, depth, side_slope, None)

        if self.lbl_berm_height is not None:
            if companion:
                h_b = berm_height_estimate(depth, width)
                self.lbl_berm_height.setText(
                    f"Berm height ≈ {h_b:.2f} m  · capacity is an estimate — actual\n"
                    f"backwater ponding depends on local slope and terrain."
                )
            else:
                self.lbl_berm_height.setText("")

        self._update_swale_verdict(depth, width, side_slope)

    def _update_swale_verdict(self, depth, width, side_slope):
        """Deficit-at-the-drawn-length readout for a swale.

        Uses the real trapezoidal section, the 0.8 freeboard allowance and event
        infiltration — i.e. the same model as the capacity the swale actually
        delivers. The old readout divided the inflow by a rectangular ``depth ×
        width``, which overstated capacity by ~⅓ for a 1:1 batter, and the result was
        labelled "recommended length" even though that figure scales with the drawn
        length and so could never be satisfied by extending the swale.
        """
        if self.lbl_verdict is None or self._peak_inflow_m3 is None:
            return
        length = getattr(self, "_swale_length_m", None)
        if not length or depth <= 0 or width <= 0:
            return

        from terrainflow_assessment.modules.swale_design import (
            get_infiltration_rate,
            required_storage_at_length,
        )

        soil = None
        if getattr(self, "combo_soil", None) is not None:
            soil = self.combo_soil.currentData()
        soil = soil or self._soil_name
        infil = get_infiltration_rate(soil) if soil else 0.0
        if not self._count_infiltration:
            infil = 0.0        # sizing on held volume alone — soakage is a bonus
        check = required_storage_at_length(
            self._peak_inflow_m3, length, depth, width,
            side_slope=side_slope if side_slope is not None else 1.0,
            freeboard=0.8,
            infiltration_mm_hr=infil,
            duration_hr=self._duration_hours or 0.0,
        )

        soak = (f"  ({check.storage_m3:,.0f} stored + {check.infiltration_m3:,.0f} soaked in)"
                if check.infiltration_m3 > 0.5 else "")
        self.lbl_holds.setText(f"{check.available_m3:,.0f} m³{soak}")
        self.lbl_holds.setStyleSheet("font-weight: bold; color: #22302e;")

        if check.holds:
            self.lbl_verdict.setText(
                f"✓ Holds the event — {check.available_m3 - check.inflow_m3:,.0f} m³ to spare."
            )
            self.lbl_verdict.setStyleSheet("font-weight: bold; color: #1e8449;")
        elif check.depth_reachable:
            self.lbl_verdict.setText(
                f"Short by {check.deficit_m3:,.0f} m³ — deepen to "
                f"{check.required_depth_m:.2f} m at this width, or route the surplus "
                f"to a downstream feature."
            )
            self.lbl_verdict.setStyleSheet("font-weight: bold; color: #c0392b;")
        else:
            self.lbl_verdict.setText(
                f"Short by {check.deficit_m3:,.0f} m³ — no depth at a {width:.1f} m top "
                f"width can hold this (the batters meet first). Widen it, extend it, or "
                f"add downstream storage."
            )
            self.lbl_verdict.setStyleSheet("font-weight: bold; color: #c0392b;")

        if self.lbl_req_length is not None:
            self.lbl_req_length.setText(
                f"Length that would hold it at these dimensions: "
                f"{check.recommended_length_m:,.0f} m  (drawn: {length:,.0f} m)"
            )

    def _update_overflow_warning(self):
        """Advise when an overflow target sits uphill of this feature.

        This is now **advice, not a veto**. Routing used to drop any target whose
        centroid was higher, but centroid elevation is a poor proxy once real flow
        paths are followed — a large tilted basin's centroid can sit above a swale
        that genuinely drains into it. A chosen target is honoured unless it would
        create a loop; this note just flags that water may need help to get there.
        """
        if self.lbl_overflow_warning is None or self.combo_overflow is None:
            return
        target_id = self.combo_overflow.currentData()
        if target_id is None or self._own_elevation is None:
            self.lbl_overflow_warning.setText("")
            return
        target_elev = self._overflow_elevations.get(target_id)
        if target_elev is None:
            self.lbl_overflow_warning.setText("")
            return
        if target_elev >= self._own_elevation:
            name = self.combo_overflow.currentText()
            self.lbl_overflow_warning.setText(
                f"⚠ {name} sits uphill of this feature's centroid "
                f"({target_elev:.1f} m vs {self._own_elevation:.1f} m). The link will "
                f"still be used, but check that water can actually reach it — you may "
                f"need a diversion drain to carry it there."
            )
        else:
            self.lbl_overflow_warning.setText("")

    def _update_basin_converge(self, depth):
        """Amber note when battered basin walls meet before the design depth."""
        if self.lbl_basin_converge is None or self.spin_wall_slope is None:
            return
        try:
            area = self.geometry.area()
            perimeter = self.geometry.length()
            r = basin_volume_battered(area, perimeter, depth,
                                      self.spin_wall_slope.value())
            if r.effective_depth < depth - 1e-9:
                self.lbl_basin_converge.setText(
                    f"⚠ Walls converge at ~{r.effective_depth:.2f} m — the basin "
                    f"bottoms out before the design depth ({depth:.2f} m). "
                    f"Widen the footprint or flatten the batter."
                )
            else:
                self.lbl_basin_converge.setText("")
        except Exception:
            self.lbl_basin_converge.setText("")

    def _current_bottom_width(self, top_width):
        """Bottom width from the control (never wider than the top), or None if absent."""
        if self.spin_bottom_width is None:
            return None
        return min(self.spin_bottom_width.value(), top_width)

    def _current_side_slope(self):
        """Side slope (z:1) derived from the current top/bottom width and depth."""
        if self.spin_bottom_width is None:
            return 1.0
        depth = self.spin_depth.value()
        top = self.spin_width.value()
        bottom = self._current_bottom_width(top)
        if depth <= 0:
            return 0.0
        return max(0.0, (top - bottom) / (2.0 * depth))

    def _update_channel_feedback(self, top_width, bottom_width, depth, side_slope, grade_pct):
        """Live batter/grade advisory + narrowest-dimension + degrees readout for channels."""
        if self.lbl_min_dim is None:
            return
        sec = trapezoid_section(top_width, bottom_width, depth)
        self.lbl_min_dim.setText(f"{sec.min_dimension:.2f} m")

        if self.lbl_side_slope is not None:
            # Batter angle from horizontal: atan(rise/run) = atan(1/z). z=0 → 90° (vertical).
            angle = math.degrees(math.atan2(1.0, side_slope))
            self.lbl_side_slope.setText(f"{angle:.1f}°  ({side_slope:.2g} : 1)")

        within, text = batter_advisory(self._soil_name, side_slope)
        if grade_pct is not None:
            g_within, g_text = grade_advisory(self._soil_name, grade_pct)
            within = within and g_within
            text = f"{text}\n{g_text}"
        self._set_advisory(within, text)

    def _set_advisory(self, within, text):
        """Colour the advisory label green (within envelope) / amber (outside)."""
        if self.lbl_advisory is None:
            return
        colour = "#1e8449" if within else "#b9770e"
        self.lbl_advisory.setStyleSheet(f"font-style: italic; color: {colour};")
        self.lbl_advisory.setText(text)
        self.lbl_advisory.setToolTip(text)

    # -- Spillway ------------------------------------------------------------

    def _current_freeboard(self):
        """The freeboard in force — the live control, else this type's policy."""
        if self.spin_spillway_freeboard is not None:
            return self.spin_spillway_freeboard.value()
        return self._policy_freeboard

    def _crest_band(self):
        """Crest elevations this feature can currently offer, at the chosen head.

        Both the head and the freeboard move the ceiling (``rim − head − freeboard``),
        so both have to be read live or the band and the warnings disagree.
        """
        head = (self.spin_spillway_head.value() if self.spin_spillway_head
                else self._policy_head)
        return spillway_datum(self._rim_elevation, self._invert_elevation,
                              head_m=head, min_freeboard_m=self._current_freeboard())

    def _seed_spillway(self, existing):
        """Initial crest/drop pair — the saved one, or the highest crest that fits.

        A fresh spillway starts as high as the head and freeboard allow, because
        that is the crest which stores the most water while still being a spillway.
        """
        crest = existing.crest_elevation if existing is not None else None
        drop = existing.drop_below_rim_m if existing is not None else None
        if crest is None and drop is None:
            _lo, hi = self._crest_band()
            crest = hi
        crest, drop = bind_crest(
            self._rim_elevation, crest=crest, drop=drop, band=self._crest_band())
        self._set_spillway_pair(crest, drop)

    def _set_spillway_pair(self, crest, drop):
        """Write both controls without re-entering the binding."""
        self._spillway_binding = True
        try:
            for widget, value in ((self.spin_spillway_crest, crest),
                                  (self.spin_spillway_drop, drop)):
                if widget is None or value is None:
                    continue
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)
        finally:
            self._spillway_binding = False

    def _on_spillway_crest_changed(self):
        if self._spillway_binding:
            return
        crest, drop = bind_crest(
            self._rim_elevation, crest=self.spin_spillway_crest.value(),
            band=self._crest_band())
        self._set_spillway_pair(crest, drop)
        self._update_spillway_sizing()

    def _on_spillway_drop_changed(self):
        if self._spillway_binding:
            return
        crest, drop = bind_crest(
            self._rim_elevation, drop=self.spin_spillway_drop.value(),
            band=self._crest_band())
        self._set_spillway_pair(crest, drop)
        self._update_spillway_sizing()

    def _on_spillway_head_changed(self):
        # Head moves the ceiling of the valid band (rim − head − freeboard), so the
        # crest may need to come down with it. Re-bind through the new band.
        if self._spillway_binding:
            return
        crest, drop = bind_crest(
            self._rim_elevation, crest=self.spin_spillway_crest.value(),
            band=self._crest_band())
        self._set_spillway_pair(crest, drop)
        self._update_spillway_sizing()

    def _required_width(self, head):
        if self._peak_flow_m3s is None:
            return None
        return calculate_spillway_width(self._peak_flow_m3s, head)

    def _feature_length_m(self):
        """Characteristic length of the drawn feature, for the does-the-weir-fit check.

        Shapely reports a polygon's perimeter as its length, so one call covers both a
        swale's run and a basin's rim without a special case.
        """
        try:
            import json

            from shapely.geometry import shape as shapely_shape

            return shapely_shape(json.loads(self.geometry.asJson())).length or None
        except Exception:
            return None

    def _update_spillway_sizing(self):
        """Refresh the required width, the auto-tracked built width, and the notes.

        The weir equation has one spare degree of freedom, so exactly one of head and
        width is chosen and the other follows. ``auto`` says which, and this method is
        where that shows: with it ticked the head is the target and the width is solved;
        with it unticked the width is the commitment and the head is the consequence.
        Only the second case reports a head, because in the first the two are the same
        number by construction.
        """
        if self.lbl_spillway_width is None:
            return
        enabled = self.grp_spillway is None or self.grp_spillway.isChecked()
        target_head = (self.spin_spillway_head.value() if self.spin_spillway_head
                       else self._policy_head)
        required = self._required_width(target_head)

        if required is not None:
            self.lbl_spillway_width.setText(f"{required:.2f} m")
        else:
            self.lbl_spillway_width.setText("— set a peak intensity on Baseline")

        # Auto keeps the built width on the requirement as head, catchment or
        # upstream routing change. Unticking it commits to a number, which is what
        # makes the shortfall check meaningful rather than tautological.
        auto = True
        if self.chk_width_auto is not None:
            auto = self.chk_width_auto.isChecked()
            self.spin_built_width.setEnabled(not auto)
            if auto and required is not None:
                self.spin_built_width.blockSignals(True)
                self.spin_built_width.setValue(required)
                self.spin_built_width.blockSignals(False)

        built = self.spin_built_width.value() if self.spin_built_width else None
        head = effective_head_m(target_head, peak_flow_m3s=self._peak_flow_m3s,
                                width_m=built, width_auto=auto)

        # The head control is the target only while the width is free; once a width is
        # committed it describes nothing, so it greys out and the achieved head takes
        # over the row below it.
        if self.spin_spillway_head is not None:
            self.spin_spillway_head.setEnabled(auto)
        if self.lbl_actual_head is not None:
            show_actual = (not auto) and head is not None
            self.lbl_actual_head.setVisible(show_actual)
            self.row_actual_head.setVisible(show_actual)
            if show_actual:
                over = head - target_head
                colour = _INK if over <= 0.005 else _WARN
                note = "" if over <= 0.005 else f"  ({over:+.2f} m on target)"
                self.lbl_actual_head.setText(f"{head:.2f} m{note}")
                self.lbl_actual_head.setStyleSheet(
                    f"color: {colour}; font-weight: 600;")

        if not enabled or self.lbl_spillway_warn is None:
            if self.lbl_spillway_warn is not None:
                self.lbl_spillway_warn.setText("")
            return

        problems = spillway_validity(
            self.spin_spillway_crest.value() if self.spin_spillway_crest else None,
            self._rim_elevation,
            invert_elevation=self._invert_elevation,
            head_m=head,
            min_freeboard_m=self._current_freeboard(),
            width_m=built,
            required_width_m=required,
            standard_freeboard_m=self._policy_freeboard,
            typical_head_m=self._policy_head_band,
            feature_length_m=self._feature_length_m(),
        )
        if self._harvesting_coefficient:
            problems.append(
                "The runoff coefficient is a water-harvesting figure, calibrated for "
                "ordinary rain rather than for the extreme event on saturated ground. "
                "It will undersize this overflow."
            )
        self.lbl_spillway_warn.setText("\n".join(f"⚠ {p}" for p in problems))
        self.lbl_spillway_warn.setToolTip(
            H.SPILLWAY_HARVESTING_C if self._harvesting_coefficient else "")
        self.lbl_spillway_warn.setVisible(bool(problems))

    # -- Result accessors --

    def get_name(self):
        return self.edit_name.text().strip() or f"New {self.ew_type.capitalize()}"

    def get_depth(self):
        return self.spin_depth.value() if self.spin_depth is not None else 0.5

    def get_crest_elevation(self):
        return self.spin_crest_elev.value() if self.spin_crest_elev is not None else None

    def get_soil_name(self):
        """Per-feature soil override, or None to inherit the site default."""
        combo = getattr(self, "combo_soil", None)
        return combo.currentData() if combo is not None else None

    def get_key_into_banks(self):
        chk = getattr(self, "chk_key_banks", None)
        return bool(chk.isChecked()) if chk is not None else False

    def get_width(self):
        # Basins have no width control (footprint comes from the polygon) —
        # keep whatever the feature already stores.
        if self.spin_width is None:
            return self._earthwork.width if self._earthwork else 2.0
        return self.spin_width.value()

    def get_wall_slope(self):
        """Basin wall batter (H:V), or 0.0 when the type has no batter control."""
        return self.spin_wall_slope.value() if self.spin_wall_slope is not None else 0.0

    def get_overflow_target_id(self):
        """Chosen overflow target earthwork id, or None for Auto (downslope)."""
        return self.combo_overflow.currentData() if self.combo_overflow is not None else None

    def get_companion_berm(self):
        return self.chk_companion.isChecked() if self.ew_type == "swale" else False

    def get_gradient_pct(self):
        return self.spin_gradient.value() if self.spin_gradient is not None else 1.0

    def get_spillway(self):
        """The configured :class:`Spillway`, or None when the group is unchecked.

        This accessor did not exist: the dialog computed a spillway width, showed
        it, and discarded everything on close. Nothing about a spillway survived
        the OK button.
        """
        if self.grp_spillway is None or not self.grp_spillway.isChecked():
            return None
        head = self.spin_spillway_head.value()
        # Store the freeboard only where it departs from the type's policy. Writing the
        # policy value back would freeze today's figure into the design, so a later
        # change to the standard would reach new features and silently skip saved ones.
        freeboard = self.spin_spillway_freeboard.value()
        override = (None if abs(freeboard - self._policy_freeboard) < 1e-9
                    else freeboard)
        return Spillway(
            crest_elevation=self.spin_spillway_crest.value(),
            drop_below_rim_m=(self.spin_spillway_drop.value()
                              if self._rim_elevation is not None else None),
            head_m=head,
            width_m=self.spin_built_width.value(),
            width_auto=self.chk_width_auto.isChecked(),
            point_wkt=self._spillway_point_wkt,
            auto=self._spillway_auto,
            freeboard_m=override,
        )

    def get_bottom_width(self):
        """Bottom width (m) from the control, or None when the type has no channel section."""
        return self._current_bottom_width(self.spin_width.value()) \
            if self.spin_bottom_width is not None else None

    def _calc_dam_wall_metrics(self, crest_elev, wall_thickness):
        """
        Sample the DEM under the dam line and return (max_wall_height_m, wall_fill_volume_m3).

        max_wall_height — tallest point from ground to crest (lower = easier to build).
        wall_fill_volume — approximate earthfill using rectangular cross-section.
        """
        try:
            import json

            import rasterio
            from shapely.geometry import shape as shapely_shape

            shp = shapely_shape(json.loads(self.geometry.asJson()))
            with rasterio.open(self._dem_path) as src:
                dem = src.read(1).astype("float32")
                t = src.transform
                cell_size = abs(t.a)
                nodata = src.nodata

            # Sample every cell_size along the wall (min 10 points)
            n_steps = max(10, int(shp.length / max(cell_size, 0.5)))
            heights = []
            for i in range(n_steps + 1):
                pt = shp.interpolate(i / n_steps, normalized=True)
                col = int((pt.x - t.c) / t.a)
                row = int((pt.y - t.f) / t.e)
                if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
                    continue
                ground = float(dem[row, col])
                if nodata is not None and abs(ground - nodata) < 1.0:
                    continue
                h = crest_elev - ground
                if h > 0:
                    heights.append(h)

            if not heights:
                return 0.0, 0.0

            max_h = max(heights)
            step_len = shp.length / n_steps
            wall_vol = sum(heights) * step_len * wall_thickness
            return round(max_h, 1), round(wall_vol, 0)

        except Exception:
            return 0.0, 0.0
