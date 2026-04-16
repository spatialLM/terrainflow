from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QLabel, QPushButton, QDialogButtonBox, QGroupBox
)
from qgis.PyQt.QtCore import Qt

from .modules.earthwork_design import (
    calculate_capacity, berm_height_estimate,
    calculate_diversion_discharge, calculate_spillway_width,
)


class EarthworkPropertiesDialog(QDialog):
    """
    Popup dialog shown after an earthwork is drawn.
    Lets the user set name, depth, width, and companion berm option.
    Displays calculated capacity in real time.
    """

    def __init__(self, ew_type, geometry, parent=None, earthwork=None,
                 peak_inflow_m3=None, crest_elevation=None, duration_hours=None,
                 dem_path=None):
        super().__init__(parent)
        self.ew_type = ew_type
        self.geometry = geometry
        self._editing = earthwork is not None
        self._peak_inflow_m3 = peak_inflow_m3   # None for freehand swales
        self._crest_elevation = crest_elevation  # pre-sampled for dam type
        self._duration_hours = duration_hours    # storm duration for spillway sizing
        self._dem_path = dem_path                # for dam wall height/volume

        type_labels = {"diversion": "Diversion Drain"}
        type_label = type_labels.get(ew_type, ew_type.capitalize())
        self.setWindowTitle(f"{'Edit' if self._editing else 'New'} {type_label} Properties")
        self.setMinimumWidth(360)
        self._build_ui(earthwork)
        self._update_capacity()

    def _build_ui(self, ew=None):
        layout = QVBoxLayout(self)

        form = QFormLayout()

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
                "Pre-filled from the higher of the two drawn endpoints.\n"
                "All cells under the wall will be raised to this elevation,\n"
                "so the wall height varies with the valley shape beneath it.\n\n"
                "Water will pool behind the dam up to this level.\n"
                "Run Re-analyse with Earthworks to see retained volume."
            )
            self.spin_crest_elev.valueChanged.connect(self._update_capacity)
            form.addRow("Crest elevation:", self.spin_crest_elev)
            self.spin_depth = None
        else:
            self.spin_crest_elev = None
            # Depth
            self.spin_depth = QDoubleSpinBox()
            self.spin_depth.setRange(0.1, 10.0)
            self.spin_depth.setValue(ew.depth if ew else 0.5)
            self.spin_depth.setDecimals(2)
            self.spin_depth.setSuffix(" m")
            self.spin_depth.valueChanged.connect(self._update_capacity)
            form.addRow("Depth:", self.spin_depth)

        # Width
        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(0.1, 100.0)
        self.spin_width.setValue(ew.width if ew else (2.0 if self.ew_type == "dam" else 1.0))
        self.spin_width.setDecimals(2)
        self.spin_width.setSuffix(" m")
        self.spin_width.valueChanged.connect(self._update_capacity)
        lbl_width = "Wall thickness:" if self.ew_type == "dam" else "Width:"
        form.addRow(lbl_width, self.spin_width)

        # Gradient (diversion drains only)
        if self.ew_type == "diversion":
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
            cap_layout.addRow("Volume (m³):", self.lbl_capacity_m3)
            cap_layout.addRow("Volume (L):",  self.lbl_capacity_l)
            if self.ew_type == "swale":
                self.lbl_berm_height = QLabel("")
                self.lbl_berm_height.setStyleSheet("color: #555555; font-style: italic;")
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
                lbl_note.setStyleSheet("color: #555555; font-style: italic;")
                lbl_note.setWordWrap(True)
                cap_layout.addRow(lbl_note)
            else:
                self.lbl_wall_volume = None
                self.lbl_max_height = None

        # Recommended length — only shown for contour swales where inflow is known
        if self.ew_type == "swale" and self._peak_inflow_m3 is not None:
            sep = QLabel("─" * 30)
            sep.setStyleSheet("color: #aaaaaa;")
            cap_layout.addRow(sep)

            lbl_inflow = QLabel(f"{self._peak_inflow_m3:,.1f} m³")
            lbl_inflow.setToolTip(
                "Total runoff volume flowing into this swale from the slope above\n"
                "for the current storm scenario (flow accumulation × runoff depth)."
            )
            cap_layout.addRow("Peak storm inflow:", lbl_inflow)

            self.lbl_req_length = QLabel("—")
            self.lbl_req_length.setStyleSheet("font-weight: bold; color: #003080;")
            self.lbl_req_length.setToolTip(
                "Minimum swale length needed to store the full storm inflow.\n\n"
                "Formula: inflow volume ÷ cross-section area\n"
                "Cross-section treated as rectangular (depth × width).\n\n"
                "This is a conservative estimate — a trapezoidal cross-section\n"
                "would require slightly less length. Add 10–20% safety margin\n"
                "for practical construction."
            )
            cap_layout.addRow("Recommended length:", self.lbl_req_length)

            self._swale_length_m = self.geometry.length()
            lbl_note = QLabel("Adjust depth / width above to see how\ndimensions affect required length.")
            lbl_note.setStyleSheet("color: #555555; font-style: italic;")
            cap_layout.addRow(lbl_note)
        else:
            self.lbl_req_length = None

        layout.addWidget(cap_group)

        # Spillway sizing — shown when peak inflow and storm duration are known
        if (self.ew_type in ("swale", "dam", "basin")
                and self._peak_inflow_m3 is not None
                and self._duration_hours is not None
                and self._duration_hours > 0):
            spill_size_group = QGroupBox("Spillway Sizing")
            spill_size_layout = QFormLayout(spill_size_group)

            duration_s = self._duration_hours * 3600.0
            peak_flow = self._peak_inflow_m3 / duration_s
            lbl_qdesign = QLabel(f"{peak_flow:.4f} m³/s  ({peak_flow * 1000:.1f} L/s)")
            lbl_qdesign.setToolTip(
                "Estimated peak design flow rate = total storm inflow ÷ storm duration.\n"
                "Conservative (average rate) — actual peak may be higher in short storms."
            )
            spill_size_layout.addRow("Design flow rate:", lbl_qdesign)

            head_row = QHBoxLayout()
            head_row.addWidget(QLabel("Head above crest (H):"))
            self.spin_spillway_head = QDoubleSpinBox()
            self.spin_spillway_head.setRange(0.05, 2.0)
            self.spin_spillway_head.setValue(0.3)
            self.spin_spillway_head.setDecimals(2)
            self.spin_spillway_head.setSuffix(" m")
            self.spin_spillway_head.setSingleStep(0.05)
            self.spin_spillway_head.setToolTip(
                "Depth of water flowing over the spillway crest at peak flow.\n"
                "Typical design values: 0.2–0.5 m\n\n"
                "Lower head → wider spillway needed.\n"
                "Higher head → narrower spillway, but less freeboard."
            )
            head_row.addWidget(self.spin_spillway_head)
            spill_size_layout.addRow(head_row)

            self.lbl_spillway_width = QLabel("—")
            self.lbl_spillway_width.setStyleSheet("font-weight: bold; color: #003080;")
            self.lbl_spillway_width.setToolTip(
                "Minimum spillway width — broad-crested weir formula:\n"
                "  Q = 1.7 × L × H^1.5\n"
                "  L = Q / (1.7 × H^1.5)\n\n"
                "Add 20–30% safety margin for design, and ensure the\n"
                "spillway outlet is protected against erosion."
            )
            spill_size_layout.addRow("Min spillway width:", self.lbl_spillway_width)

            self._peak_flow_m3s = peak_flow
            self.spin_spillway_head.valueChanged.connect(self._update_spillway_sizing)
            self._update_spillway_sizing()

            layout.addWidget(spill_size_group)
        else:
            self.spin_spillway_head = None
            self.lbl_spillway_width = None
            self._peak_flow_m3s = None

        # Spillway section (display only in properties, placement done via map tool)
        if self.ew_type != "berm" and ew and ew.spillway_point:
            spill_group = QGroupBox("Spillway")
            spill_layout = QFormLayout(spill_group)
            spill_layout.addRow("Point:", QLabel("Placed ✓"))
            self.spin_spillway_elev = QDoubleSpinBox()
            self.spin_spillway_elev.setRange(0, 5000)
            self.spin_spillway_elev.setValue(ew.spillway_elevation or 0)
            self.spin_spillway_elev.setSuffix(" m")
            spill_layout.addRow("Overflow elevation:", self.spin_spillway_elev)
            layout.addWidget(spill_group)
        else:
            self.spin_spillway_elev = None

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
                colour = "#cc0000" if max_h > 5 else "#cc6600" if max_h > 3 else "#006600"
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
            q = calculate_diversion_discharge(depth, width, gradient)
            self.lbl_capacity_m3.setText(f"{q:.4f} m³/s  ({q * 1000:.1f} L/s)")
            # Compare against peak inflow rate if available
            if self._peak_inflow_m3 is not None:
                # Approximate storm duration: assume 1 hour (3600 s) as a reference
                # peak_inflow_m3 is total volume — inflow rate is not directly derivable
                # Show a note instead of a comparison (we don't know storm duration here)
                self.lbl_capacity_l.setText(
                    "Peak inflow volume: see swale properties\n"
                    "for direct comparison."
                )
                self.lbl_capacity_l.setStyleSheet("color: #555555; font-style: italic;")
            else:
                self.lbl_capacity_l.setText(
                    "Manning's n = 0.025 · trapezoidal 1:1 side slopes"
                )
                self.lbl_capacity_l.setStyleSheet("color: #555555; font-style: italic;")
            return

        depth = self.spin_depth.value()
        width = self.spin_width.value()
        companion = self.chk_companion.isChecked() if self.ew_type == "swale" else False
        m3, l = calculate_capacity(self.ew_type, self.geometry, depth, width, companion)
        self.lbl_capacity_m3.setText(f"{m3:,.2f}")
        self.lbl_capacity_l.setText(f"{l:,.0f}")

        if self.lbl_berm_height is not None:
            if companion:
                h_b = berm_height_estimate(depth, width)
                self.lbl_berm_height.setText(
                    f"Berm height ≈ {h_b:.2f} m  · capacity is an estimate — actual\n"
                    f"backwater ponding depends on local slope and terrain."
                )
            else:
                self.lbl_berm_height.setText("")

        if self.lbl_req_length is not None and self._peak_inflow_m3 is not None:
            section = depth * width   # rectangular cross-section (m²)
            if section > 0:
                req_m = self._peak_inflow_m3 / section
                self.lbl_req_length.setText(f"{req_m:.0f} m")
                # Colour feedback: green = swale is long enough, red = too short
                actual_m = getattr(self, "_swale_length_m", None)
                if actual_m is not None:
                    if actual_m >= req_m:
                        self.lbl_req_length.setStyleSheet("font-weight: bold; color: #1a7a1a;")
                    else:
                        self.lbl_req_length.setStyleSheet("font-weight: bold; color: #cc0000;")
            else:
                self.lbl_req_length.setText("—")

    def _update_spillway_sizing(self):
        if self.spin_spillway_head is None or self._peak_flow_m3s is None:
            return
        head = self.spin_spillway_head.value()
        width = calculate_spillway_width(self._peak_flow_m3s, head)
        self.lbl_spillway_width.setText(f"{width:.2f} m")

    # -- Result accessors --

    def get_name(self):
        return self.edit_name.text().strip() or f"New {self.ew_type.capitalize()}"

    def get_depth(self):
        return self.spin_depth.value() if self.spin_depth is not None else 0.5

    def get_crest_elevation(self):
        return self.spin_crest_elev.value() if self.spin_crest_elev is not None else None

    def get_width(self):
        return self.spin_width.value()

    def get_companion_berm(self):
        return self.chk_companion.isChecked() if self.ew_type == "swale" else False

    def get_gradient_pct(self):
        return self.spin_gradient.value() if self.spin_gradient is not None else 1.0

    def get_spillway_elevation(self):
        if self.spin_spillway_elev:
            return self.spin_spillway_elev.value()
        return None

    def _calc_dam_wall_metrics(self, crest_elev, wall_thickness):
        """
        Sample the DEM under the dam line and return (max_wall_height_m, wall_fill_volume_m3).

        max_wall_height — tallest point from ground to crest (lower = easier to build).
        wall_fill_volume — approximate earthfill using rectangular cross-section.
        """
        try:
            import json
            import numpy as np
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
