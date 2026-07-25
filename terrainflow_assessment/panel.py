"""
panel.py — Docked Workbench panel UI for TerrainFlow Assessment.

AssessmentPanel is a QDockWidget structured as a pipeline workbench:

  Persistent chrome (always visible)
    · header — site name + DEM info, storm chip (the scenario the score is
      scored against; click → Baseline stage)
    · scorecard — capture % + proportional water-budget band
    · stage stepper — Baseline → Analysis → Design → Verify → Report, with
      per-stage state (· todo / ✓ done / ⚠ stale)

  Stages (one visible at a time, QStackedWidget)
    baseline — Data Input + Baseline Analysis (storm inputs + run button)
    analysis — Terrain Tools (ponding/slope/contours) + Contour & Keypoint
    design   — Earthwork Design + Live Assessment
    verify   — Fill Simulation (+ Re-analyse burn lives in design for now)
    report   — Report & Export
"""

from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class AssessmentPanel(QDockWidget):
    """
    Main control panel for TerrainFlow Assessment.

    All user interactions emit signals that the plugin controller (plugin.py)
    connects to.  No analysis logic lives here — the panel is pure UI.
    """

    # ---------------------------------------------------------------- signals
    # Data
    dem_changed = pyqtSignal(object)          # QgsRasterLayer or None
    boundary_changed = pyqtSignal(object)     # QgsVectorLayer or None
    analysis_area_changed = pyqtSignal(object)
    earthworks_area_changed = pyqtSignal(object)

    # Baseline
    run_baseline_requested = pyqtSignal()
    threshold_changed = pyqtSignal()
    query_ponding_requested = pyqtSignal()
    toggle_slope_class_requested = pyqtSignal(bool)
    toggle_slope_arrows_requested = pyqtSignal(bool)

    # Contour analysis
    run_contour_analysis_requested = pyqtSignal()
    select_top5_contours_requested = pyqtSignal()
    find_segments_requested = pyqtSignal()
    generate_simple_contours_requested = pyqtSignal()
    contour_layer_changed = pyqtSignal(object)
    run_keypoint_analysis_requested = pyqtSignal()
    recommend_ponds_requested = pyqtSignal()

    # Earthworks
    draw_swale_requested = pyqtSignal(str)      # mode: 'freehand' | 'contour' | 'full_contour'
    draw_earthwork_requested = pyqtSignal(str)  # registry type key (berm/basin/dam/…)
    usable_area_source_changed = pyqtSignal(str)   # "none" | "analysis" | "earthworks"
    run_earthworks_requested = pyqtSignal()
    reshape_earthworks_requested = pyqtSignal()   # vertex-drag tool with live readout
    before_after_toggled = pyqtSignal(bool)   # True = with earthworks
    analysis_inputs_changed = pyqtSignal()    # storm/soil input changed → live re-assess

    # Simulation
    run_simulation_requested = pyqtSignal()
    sim_frame_changed = pyqtSignal(int)
    sim_play_toggled = pyqtSignal(bool)

    # Report
    export_report_requested = pyqtSignal()

    # Workbench pipeline stages: (key, label) → which section builders live in it
    _STAGES = (
        ("baseline", "Baseline"),
        ("analysis", "Analysis"),
        ("design", "Design"),
        ("verify", "Verify"),
        ("report", "Report"),
    )

    def __init__(self, parent=None):
        super().__init__("TerrainFlow Assessment", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(340)

        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setSpacing(6)
        root_lay.setContentsMargins(8, 8, 8, 8)

        self._build_workbench_chrome(root_lay)
        self._build_ui()
        root_lay.addWidget(self._stack, 1)
        self.setWidget(root)

        # Default landing stage
        self._show_stage("design")

    # ---------------------------------------------------------------- Workbench chrome

    def _build_workbench_chrome(self, root_lay):
        """Persistent header: site line + storm chip, scorecard, pipeline stepper."""
        from terrainflow_assessment.qgis.widgets.scorecard import Scorecard
        from terrainflow_assessment.qgis.widgets.stepper import StageStepper

        head = QHBoxLayout()
        head.setSpacing(8)
        site_col = QVBoxLayout()
        site_col.setSpacing(0)
        self._head_site_lbl = QLabel("Unnamed Site")
        self._head_site_lbl.setStyleSheet("font-weight: 600; font-size: 13px; color: #22302e;")
        self._head_info_lbl = QLabel("Load a DEM to begin")
        self._head_info_lbl.setStyleSheet("font-size: 10px; color: #8fa0a4;")
        site_col.addWidget(self._head_site_lbl)
        site_col.addWidget(self._head_info_lbl)
        head.addLayout(site_col, 1)

        # Storm chip — the scenario the score is scored against; click → Baseline
        self._storm_chip = QPushButton("— set storm ▾")
        self._storm_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self._storm_chip.setStyleSheet(
            "QPushButton { border: 1px solid #c6d1d3; border-radius: 12px;"
            " padding: 3px 11px; font-size: 11px; color: #5f7176; background: transparent; }"
            "QPushButton:hover { border-color: #2e7d55; color: #22302e; }"
        )
        self._storm_chip.setToolTip(
            "The design storm the score is computed against.\nClick to edit in Baseline."
        )
        self._storm_chip.clicked.connect(lambda: self._show_stage("baseline"))
        head.addWidget(self._storm_chip)
        root_lay.addLayout(head)

        self._scorecard = Scorecard()
        root_lay.addWidget(self._scorecard)

        self._stepper = StageStepper(list(self._STAGES))
        self._stepper.stage_selected.connect(self._show_stage)
        root_lay.addWidget(self._stepper)

    def _make_stage_page(self):
        """A scrollable page hosting one stage's sections; returns (page, layout)."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setSpacing(8)
        lay.setContentsMargins(0, 4, 0, 4)
        scroll.setWidget(container)
        return scroll, lay

    def _show_stage(self, key):
        idx = [k for k, _ in self._STAGES].index(key)
        self._stack.setCurrentIndex(idx)
        self._stepper.set_current(key)

    def mark_stage(self, key, state):
        """Set a stage's pipeline state: 'todo' | 'done' | 'stale'."""
        self._stepper.set_state(key, state)

    # ---------------------------------------------------------------- UI construction

    def _build_ui(self):
        self._stack = QStackedWidget()
        self._stage_layouts = {}
        for key, _label in self._STAGES:
            page, lay = self._make_stage_page()
            self._stack.addWidget(page)
            self._stage_layouts[key] = lay

        # Each stage repoints self._layout (the target _section() appends to)
        # before building its sections.
        # Baseline = data input + storm inputs + the run button (everything up
        # to and including "Run Baseline Analysis").
        self._layout = self._stage_layouts["baseline"]
        self._build_section_data()
        self._build_section_baseline_inputs()

        # Analysis = post-baseline exploration: results tools + contour/keypoint.
        self._layout = self._stage_layouts["analysis"]
        self._build_section_baseline_results()
        self._build_section_contour_keypoint()

        self._layout = self._stage_layouts["design"]
        self._build_section_earthworks()
        self._build_section_live_assessment()

        self._layout = self._stage_layouts["verify"]
        self._build_section_simulation()

        self._layout = self._stage_layouts["report"]
        self._build_section_report()

        for lay in self._stage_layouts.values():
            lay.addStretch()

        self._wire_input_change_signals()

    def _section(self, title, collapsed=False):
        """A collapsible section with a disclosure-arrow header (▶ / ▼).

        Replaces the old checkable QGroupBox (the tick-box → dropdown-arrow change
        from the UI review). The stylesheet is scoped by objectName so it never
        cascades into the section's child frames / cards.
        """
        frame = QFrame()
        frame.setObjectName("tfSection")
        frame.setStyleSheet(
            "QFrame#tfSection { border: 1px solid #dde4e5; border-radius: 7px;"
            " background: #ffffff; }"
        )
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QPushButton()
        header.setObjectName("tfSectionHeader")
        header.setCheckable(True)
        header.setChecked(not collapsed)
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setStyleSheet(
            "QPushButton#tfSectionHeader { border: none; text-align: left;"
            " padding: 8px 10px; font-weight: 650; font-size: 12.5px;"
            " color: #22302e; background: transparent; border-radius: 7px; }"
            " QPushButton#tfSectionHeader:hover { background: #f6f8f8; }"
        )

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 0, 10, 10)
        body.setVisible(not collapsed)

        def _toggle(checked):
            body.setVisible(checked)
            header.setText(("▼  " if checked else "▶  ") + title)
        header.toggled.connect(_toggle)
        _toggle(not collapsed)

        outer.addWidget(header)
        outer.addWidget(body)
        self._layout.addWidget(frame)
        return layout

    def _label(self, text, small=False):
        lbl = QLabel(text)
        if small:
            lbl.setStyleSheet("color: #7f8c8d; font-size: 10px;")
        return lbl

    def _button(self, text, color=None):
        btn = QPushButton(text)
        if color:
            btn.setStyleSheet(
                f"QPushButton {{ background: {color}; color: white; "
                f"border-radius: 4px; padding: 5px 10px; font-weight: bold; }} "
                f"QPushButton:hover {{ opacity: 0.9; }}"
            )
        return btn

    # ---------------------------------------------------------------- Section 1: Data

    def _build_section_data(self):
        lay = self._section("Data Input")

        lay.addWidget(self._label("DEM Layer"))
        self._dem_combo = QgsMapLayerComboBox()
        self._dem_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self._dem_combo.setAllowEmptyLayer(True)
        self._dem_combo.setCurrentIndex(0)
        lay.addWidget(self._dem_combo)

        self._dem_info_lbl = self._label("", small=True)
        lay.addWidget(self._dem_info_lbl)

        lay.addWidget(self._label("Site Boundary (polygon)"))
        self._boundary_combo = QgsMapLayerComboBox()
        self._boundary_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self._boundary_combo.setAllowEmptyLayer(True)
        lay.addWidget(self._boundary_combo)

        self._site_name_edit = QLineEdit()
        self._site_name_edit.setPlaceholderText("Site name (for report)")
        lay.addWidget(self._site_name_edit)

        lay.addWidget(self._label("Analysis area (post-analysis tools)"))
        self._analysis_area_combo = QgsMapLayerComboBox()
        self._analysis_area_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self._analysis_area_combo.setAllowEmptyLayer(True)
        self._analysis_area_combo.setToolTip(
            "Restrict Contour Analysis and Keypoint Analysis to this polygon.\n\n"
            "Leave blank to use the full DEM (or the site boundary if set)."
        )
        lay.addWidget(self._analysis_area_combo)

        lay.addWidget(self._label("Earthworks area (polygon layer)"))
        self._earthworks_area_combo = QgsMapLayerComboBox()
        self._earthworks_area_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self._earthworks_area_combo.setAllowEmptyLayer(True)
        self._earthworks_area_combo.setToolTip(
            "Optional polygon defining where earthworks can be placed.\n\n"
            "When set, drawing tools will be constrained to this area and\n"
            "Optimal Swale Contours will only be generated within it.\n\n"
            "Useful for separating the project area from sensitive zones\n"
            "(wetlands, roads, existing structures) that must not be disturbed."
        )
        lay.addWidget(self._earthworks_area_combo)

        self._dem_combo.layerChanged.connect(lambda layer: self.dem_changed.emit(layer))
        self._boundary_combo.layerChanged.connect(lambda layer: self.boundary_changed.emit(layer))
        self._analysis_area_combo.layerChanged.connect(
            lambda layer: self.analysis_area_changed.emit(layer))
        self._earthworks_area_combo.layerChanged.connect(
            lambda layer: self.earthworks_area_changed.emit(layer))

    # ---------------------------------------------------------------- Section 2: Baseline

    def _build_section_baseline_inputs(self):
        lay = self._section("Baseline Analysis")

        # Rainfall
        rf_grid = QGridLayout()
        rf_grid.addWidget(self._label("Total rainfall (mm)"), 0, 0)
        self._rainfall_spin = QDoubleSpinBox()
        self._rainfall_spin.setRange(0, 1000)
        self._rainfall_spin.setValue(65)
        self._rainfall_spin.setSuffix(" mm")
        self._rainfall_spin.setToolTip(
            "Total rainfall depth for the storm event (mm).\n\n"
            "This is the cumulative rainfall over the full duration —\n"
            "the same figure reported in daily rainfall records or\n"
            "intensity-frequency-duration (IFD) tables as a daily total.\n\n"
            "Example: a 1-in-10-year, 24-hour storm in NZ hill country\n"
            "might be 80–120 mm total.\n\n"
            "The SCS model converts this total depth into runoff depth\n"
            "using the Curve Number and soil moisture condition."
        )
        rf_grid.addWidget(self._rainfall_spin, 0, 1)

        rf_grid.addWidget(self._label("Duration (hr)"), 1, 0)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.1, 72)
        self._duration_spin.setValue(24)
        self._duration_spin.setSuffix(" hr")
        self._duration_spin.setToolTip(
            "Storm duration in hours.\n\n"
            "Used to calculate peak flow rate at site exit points\n"
            "(volume ÷ duration = average flow rate).\n\n"
            "Set this to match the duration of your total rainfall figure —\n"
            "e.g. 24 hr if using a daily rainfall total from historical records."
        )
        rf_grid.addWidget(self._duration_spin, 1, 1)

        rf_grid.addWidget(self._label("Soil Type"), 2, 0)
        self._soil_combo = QComboBox()
        for name in ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]:
            self._soil_combo.addItem(name)
        self._soil_combo.setCurrentText("Loam")
        rf_grid.addWidget(self._soil_combo, 2, 1)

        rf_grid.addWidget(self._label("Curve Number (CN)"), 3, 0)
        self._cn_spin = QSpinBox()
        self._cn_spin.setRange(1, 100)
        self._cn_spin.setValue(61)
        self._cn_spin.setToolTip(
            "SCS Curve Number — soil runoff potential.\n"
            "Higher = more runoff.\n\n"
            "Typical values (Normal moisture):\n"
            "  Sand: 39  |  Sandy loam: 49\n"
            "  Loam: 61  |  Clay loam: 74  |  Clay: 80\n\n"
            "Auto-filled from Soil Type above. Override if you know the\n"
            "site-specific CN (e.g. from land-use or measured data)."
        )
        rf_grid.addWidget(self._cn_spin, 3, 1)

        rf_grid.addWidget(self._label("Moisture Condition"), 4, 0)
        self._moisture_combo = QComboBox()
        self._moisture_combo.addItems(["normal", "dry", "wet"])
        rf_grid.addWidget(self._moisture_combo, 4, 1)

        lay.addLayout(rf_grid)

        # Wire soil → CN auto-fill
        _SOIL_CN = {"Sand": 39, "Sandy loam": 49, "Loam": 61, "Clay loam": 74, "Clay": 80}
        self._soil_combo.currentTextChanged.connect(
            lambda name: self._cn_spin.setValue(_SOIL_CN.get(name, 61)))

        # Stream threshold
        thr_grid = QGridLayout()
        thr_grid.addWidget(self._label("Stream Threshold (ha)"), 0, 0)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.1, 10000)
        self._threshold_spin.setValue(5.0)
        self._threshold_spin.setSuffix(" ha")
        self._threshold_spin.setToolTip(
            "Minimum upstream catchment area for a flow path to be shown as a channel.\n\n"
            "Lower = more channels shown.  Higher = major watercourses only.\n\n"
            "Channel types by contributing area:\n"
            "  Rills / erosion paths:   < 0.5 ha\n"
            "  Ephemeral / seasonal:    0.5 – 5 ha\n"
            "  Permanent stream:        5 – 20 ha\n"
            "  River:                   > 20 ha"
        )
        self._threshold_spin.valueChanged.connect(self._update_channel_type_label)
        thr_grid.addWidget(self._threshold_spin, 0, 1)

        self._channel_type_lbl = self._label("", small=True)
        self._channel_type_lbl.setStyleSheet("color: #555555; font-style: italic;")
        thr_grid.addWidget(self._channel_type_lbl, 1, 0, 1, 2)

        thr_grid.addWidget(self._label("Routing"), 2, 0)
        self._routing_combo = QComboBox()
        self._routing_combo.addItems(["D-infinity (recommended)", "D8"])
        thr_grid.addWidget(self._routing_combo, 2, 1)
        lay.addLayout(thr_grid)

        from terrainflow_assessment.qgis.widgets.run_button import RunButton
        self._run_baseline_btn = RunButton("Run Baseline Analysis")
        lay.addWidget(self._run_baseline_btn)

        self._baseline_results_lbl = self._label("", small=True)
        self._baseline_results_lbl.setWordWrap(True)
        lay.addWidget(self._baseline_results_lbl)

        self._run_baseline_btn.clicked.connect(self.run_baseline_requested)

    def _build_section_baseline_results(self):
        """Post-baseline exploration tools (Analysis stage). Enabled after a run."""
        lay = self._section("Terrain Tools")

        lay.addWidget(self._label(
            "Run Baseline first, then explore ponding, slope and contours."
        ))
        self._query_ponding_btn = QPushButton("Query Depression / Ponding")
        self._query_ponding_btn.setEnabled(False)
        self._query_ponding_btn.setToolTip(
            "Click on a blue zone in the 'Water Captured' layer to select the\n"
            "entire connected pooling area and report its volume and surface area.\n\n"
            "Baseline: shows natural low spots where water collects.\n"
            "Earthworks: shows water captured by your swales/basins."
        )
        lay.addWidget(self._query_ponding_btn)

        self._toggle_slope_class_btn = QPushButton("Show Slope Classification")
        self._toggle_slope_class_btn.setCheckable(True)
        self._toggle_slope_class_btn.setEnabled(False)
        self._toggle_slope_class_btn.setToolTip(
            "Semi-transparent slope suitability overlay (calculated from DEM):\n"
            "  Green  (0–3°):   ideal — suitable for swales and basins\n"
            "  Yellow (3–8°):   moderate — suitable with care\n"
            "  Orange (8–15°):  challenging — consider companion berm\n"
            "  Red    (>15°):   steep — berms or diversion drains recommended"
        )
        lay.addWidget(self._toggle_slope_class_btn)

        # Inline slope legend
        slope_legend = QWidget()
        slope_legend_layout = QHBoxLayout(slope_legend)
        slope_legend_layout.setContentsMargins(4, 0, 4, 2)
        slope_legend_layout.setSpacing(4)
        for hex_colour, lbl_text in [
            ("#50C850", "0–3°"),
            ("#DCDC1E", "3–8°"),
            ("#FFA500", "8–13°"),
            ("#FF6600", "13–18°"),
            ("#CC2200", "18–25°"),
            ("#660000", ">25°"),
        ]:
            swatch = QLabel()
            swatch.setFixedSize(13, 13)
            swatch.setStyleSheet(
                f"background-color: {hex_colour}; border: 1px solid #888;"
            )
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet("font-size: 10px;")
            slope_legend_layout.addWidget(swatch)
            slope_legend_layout.addWidget(lbl)
        slope_legend_layout.addStretch()
        lay.addWidget(slope_legend)

        self._toggle_slope_arrows_btn = QPushButton("Show Slope Direction")
        self._toggle_slope_arrows_btn.setCheckable(True)
        self._toggle_slope_arrows_btn.setEnabled(False)
        self._toggle_slope_arrows_btn.setToolTip(
            "Overlay arrows showing the direction of steepest downslope at regular intervals.\n"
            "Generated from the DEM aspect — arrows point in the direction water would flow."
        )
        lay.addWidget(self._toggle_slope_arrows_btn)

        contour_row = QHBoxLayout()
        self._simple_contour_interval_spin = QDoubleSpinBox()
        self._simple_contour_interval_spin.setRange(0.1, 100.0)
        self._simple_contour_interval_spin.setValue(1.0)
        self._simple_contour_interval_spin.setSuffix(" m")
        self._simple_contour_interval_spin.setSingleStep(0.5)
        self._simple_contour_interval_spin.setDecimals(1)
        self._simple_contour_interval_spin.setToolTip("Contour interval (m)")
        self._simple_contour_interval_spin.setFixedWidth(75)
        contour_row.addWidget(self._simple_contour_interval_spin)

        self._generate_contours_btn = QPushButton("Generate Contours")
        self._generate_contours_btn.setEnabled(False)
        self._generate_contours_btn.setToolTip(
            "Generate simple elevation contours from the DEM at the chosen interval.\n\n"
            "Useful for visualising terrain alongside the slope classification."
        )
        contour_row.addWidget(self._generate_contours_btn)
        lay.addLayout(contour_row)

        self._generate_contours_btn.clicked.connect(self.generate_simple_contours_requested)
        self._query_ponding_btn.clicked.connect(self.query_ponding_requested)
        self._toggle_slope_class_btn.toggled.connect(self.toggle_slope_class_requested)
        self._toggle_slope_arrows_btn.toggled.connect(self.toggle_slope_arrows_requested)

        self._update_channel_type_label()

    # ---------------------------------------------------------------- Section 3: Contour & Keypoint

    def _build_section_contour_keypoint(self):
        lay = self._section("Contour & Keypoint Analysis", collapsed=True)

        tabs = QTabWidget()

        # --- Contour tab ---
        contour_w = QWidget()
        contour_lay = QVBoxLayout(contour_w)

        contour_lay.addWidget(self._label("Contour interval (m)"))
        self._contour_interval_spin = QDoubleSpinBox()
        self._contour_interval_spin.setRange(0.1, 100)
        self._contour_interval_spin.setValue(1.0)
        self._contour_interval_spin.setSuffix(" m")
        contour_lay.addWidget(self._contour_interval_spin)

        contour_lay.addWidget(self._label("Max slope (°) — filter"))
        self._max_slope_spin = QDoubleSpinBox()
        self._max_slope_spin.setRange(1, 45)
        self._max_slope_spin.setValue(18.0)
        self._max_slope_spin.setSuffix("°")
        contour_lay.addWidget(self._max_slope_spin)

        contour_lay.addWidget(self._label("Usable area (clip contours to)"))
        self._usable_area_combo = QComboBox()
        self._usable_area_combo.addItems(["None", "Analysis Area", "Earthworks Area"])
        self._usable_area_combo.setToolTip(
            "Optionally clip contour analysis to one of the polygon layers\n"
            "already selected in the Data section above.\n\n"
            "  None — analyse the full DEM extent\n"
            "  Analysis Area — use the Analysis Area polygon layer\n"
            "  Earthworks Area — use the Earthworks Area polygon layer"
        )
        contour_lay.addWidget(self._usable_area_combo)

        contour_lay.addWidget(self._label("Min contour length (m)"))
        self._min_contour_length_spin = QDoubleSpinBox()
        self._min_contour_length_spin.setRange(0, 5000)
        self._min_contour_length_spin.setValue(50.0)
        self._min_contour_length_spin.setSuffix(" m")
        self._min_contour_length_spin.setSingleStep(10)
        self._min_contour_length_spin.setToolTip(
            "Exclude contours shorter than this length.\n\n"
            "Short enclosed contours (from shallow dips or small knolls)\n"
            "can rank highly because their accumulation is concentrated,\n"
            "but they are too short to place a meaningful swale.\n\n"
            "Set to 0 to include all contours."
        )
        contour_lay.addWidget(self._min_contour_length_spin)

        self._run_contour_btn = self._button("Analyse Contours", "#27ae60")
        contour_lay.addWidget(self._run_contour_btn)

        self._contour_progress = QProgressBar()
        self._contour_progress.setVisible(False)
        contour_lay.addWidget(self._contour_progress)

        self._contour_list = QListWidget()
        self._contour_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._contour_list.setMaximumHeight(150)
        contour_lay.addWidget(self._contour_list)

        self._top5_contours_btn = self._button("Select Top 5 Swales", "#1a6b3a")
        self._top5_contours_btn.setEnabled(False)
        self._top5_contours_btn.setToolTip(
            "Create a separate layer containing the top 5 ranked candidate\n"
            "swale contours by peak inflow accumulation.\n\n"
            "Requires: Analyse Contours run first."
        )
        contour_lay.addWidget(self._top5_contours_btn)

        # --- Segment analysis ---
        contour_lay.addWidget(self._label("Min catchment above swale (ha)"))
        self._min_catchment_ha_spin = QDoubleSpinBox()
        self._min_catchment_ha_spin.setRange(0.0, 500.0)
        self._min_catchment_ha_spin.setValue(0.5)
        self._min_catchment_ha_spin.setSuffix(" ha")
        self._min_catchment_ha_spin.setSingleStep(0.5)
        self._min_catchment_ha_spin.setToolTip(
            "Minimum contributing area above a contour crossing to qualify\n"
            "as a swale placement zone.\n\n"
            "Only flow paths draining at least this many hectares will produce\n"
            "a recommended segment.  Raise this to focus on major drainage lines;\n"
            "lower it to pick up smaller catchments too.\n\n"
            "Default 0.5 ha."
        )
        contour_lay.addWidget(self._min_catchment_ha_spin)

        swale_dim_grid = QGridLayout()
        swale_dim_grid.addWidget(self._label("Swale depth (m)"), 0, 0)
        self._swale_depth_spin = QDoubleSpinBox()
        self._swale_depth_spin.setRange(0.1, 2.0)
        self._swale_depth_spin.setValue(0.3)
        self._swale_depth_spin.setSuffix(" m")
        self._swale_depth_spin.setSingleStep(0.05)
        self._swale_depth_spin.setDecimals(2)
        self._swale_depth_spin.setToolTip(
            "Design depth of the swale cross-section (m).\n"
            "Used to calculate required swale length:\n"
            "  length = inflow volume / (depth × width)"
        )
        swale_dim_grid.addWidget(self._swale_depth_spin, 0, 1)

        swale_dim_grid.addWidget(self._label("Swale width (m)"), 1, 0)
        self._swale_width_spin = QDoubleSpinBox()
        self._swale_width_spin.setRange(0.1, 10.0)
        self._swale_width_spin.setValue(0.6)
        self._swale_width_spin.setSuffix(" m")
        self._swale_width_spin.setSingleStep(0.1)
        self._swale_width_spin.setDecimals(2)
        self._swale_width_spin.setToolTip(
            "Design base width of the swale cross-section (m).\n"
            "Used to calculate required swale length:\n"
            "  length = inflow volume / (depth × width)"
        )
        swale_dim_grid.addWidget(self._swale_width_spin, 1, 1)
        contour_lay.addLayout(swale_dim_grid)

        self._find_segments_btn = self._button("Find Best Swale Segments", "#145a32")
        self._find_segments_btn.setEnabled(False)
        self._find_segments_btn.setToolTip(
            "Find swale placement zones on each candidate contour and size\n"
            "each segment to capture the full incoming runoff volume.\n\n"
            "Locates where drainage lines cross each contour (flow accumulation\n"
            "peaks), calculates inflow volume from the contributing catchment,\n"
            "then sets the swale length to store that volume:\n\n"
            "  required length = inflow m³ / (depth × width)\n\n"
            "The segment is centered on the crossing point.\n"
            "Results ranked globally by inflow volume (m³).\n\n"
            "Requires: Analyse Contours + Baseline Analysis run first."
        )
        contour_lay.addWidget(self._find_segments_btn)
        contour_lay.addStretch()

        tabs.addTab(contour_w, "Contours")

        # --- Keypoint tab ---
        keypoint_w = QWidget()
        keypoint_lay = QVBoxLayout(keypoint_w)

        keypoint_lay.addWidget(self._label("Number of keypoints"))
        self._keypoint_count_spin = QSpinBox()
        self._keypoint_count_spin.setRange(1, 20)
        self._keypoint_count_spin.setValue(5)
        self._keypoint_count_spin.setToolTip(
            "Number of keypoints to detect.\n"
            "Each keypoint is a valley inflection where slope eases from steep to gentle.\n"
            "Keypoints are spatially separated so they cover the full elevation range."
        )
        keypoint_lay.addWidget(self._keypoint_count_spin)

        self._run_keypoint_btn = self._button("Find Keypoints + Ridgelines", "#8e44ad")
        self._run_keypoint_btn.setToolTip(
            "Analyse the DEM and flow accumulation to locate:\n\n"
            "  Keypoints — valley inflection points where slope transitions\n"
            "  from steep to gentle. This is where Yeomans' keyline begins.\n\n"
            "  Ridgelines — watershed divides that separate drainage basins.\n\n"
            "Requires: baseline analysis run."
        )
        keypoint_lay.addWidget(self._run_keypoint_btn)

        self._recommend_ponds_btn = self._button("Recommend Pond Sites", "#6c3483")
        self._recommend_ponds_btn.setEnabled(False)
        self._recommend_ponds_btn.setToolTip(
            "For each keypoint, find the optimal dam/pond location:\n"
            "the narrowest valley cross-section just downstream.\n\n"
            "Requires: keypoints found first."
        )
        keypoint_lay.addWidget(self._recommend_ponds_btn)

        self._keypoint_progress = QProgressBar()
        self._keypoint_progress.setVisible(False)
        keypoint_lay.addWidget(self._keypoint_progress)

        self._keypoint_results_lbl = self._label("", small=True)
        self._keypoint_results_lbl.setWordWrap(True)
        keypoint_lay.addWidget(self._keypoint_results_lbl)
        keypoint_lay.addStretch()

        tabs.addTab(keypoint_w, "Keypoints")

        lay.addWidget(tabs)

        self._usable_area_combo.currentTextChanged.connect(
            lambda text: self.usable_area_source_changed.emit(
                "none" if text == "None"
                else "analysis" if text == "Analysis Area"
                else "earthworks"
            )
        )
        self._run_contour_btn.clicked.connect(self.run_contour_analysis_requested)
        self._top5_contours_btn.clicked.connect(self.select_top5_contours_requested)
        self._find_segments_btn.clicked.connect(self.find_segments_requested)
        self._run_keypoint_btn.clicked.connect(self.run_keypoint_analysis_requested)
        self._recommend_ponds_btn.clicked.connect(self.recommend_ponds_requested)

    # ---------------------------------------------------------------- Section 4: Earthworks

    def _build_section_earthworks(self):
        lay = self._section("Earthwork Design")

        lay.addWidget(self._label("Soil type (for earthwork sizing)"))
        self._ew_soil_combo = QComboBox()
        for name in ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]:
            self._ew_soil_combo.addItem(name)
        self._ew_soil_combo.setCurrentText("Loam")
        lay.addWidget(self._ew_soil_combo)

        # Felt-style tool menu (registry-driven) replaces the button grid.
        from terrainflow_assessment.qgis.widgets.tool_menu import EarthworkToolMenu
        self._tool_menu = EarthworkToolMenu()
        self._tool_menu.draw_swale_requested.connect(self.draw_swale_requested)
        self._tool_menu.draw_earthwork_requested.connect(self.draw_earthwork_requested)
        lay.addWidget(self._tool_menu)

        ew_actions = QHBoxLayout()
        self._ew_edit_btn = QPushButton("Edit")
        self._ew_reshape_btn = QPushButton("Reshape")
        self._ew_reshape_btn.setToolTip(
            "Drag earthwork vertices on the map — the Live Assessment updates\n"
            "as you drag. Double-click a segment to insert a vertex; Del removes\n"
            "the highlighted vertex; right-click or Esc finishes."
        )
        self._ew_delete_btn = QPushButton("Delete")
        self._ew_toggle_btn = QPushButton("Enable/Disable")
        ew_actions.addWidget(self._ew_edit_btn)
        ew_actions.addWidget(self._ew_reshape_btn)
        ew_actions.addWidget(self._ew_toggle_btn)
        ew_actions.addWidget(self._ew_delete_btn)
        lay.addLayout(ew_actions)

        from terrainflow_assessment.qgis.widgets.run_button import RunButton
        self._run_ew_btn = RunButton("Re-analyse with Earthworks")
        lay.addWidget(self._run_ew_btn)

        self._earthworks_results_lbl = self._label("", small=True)
        self._earthworks_results_lbl.setWordWrap(True)
        lay.addWidget(self._earthworks_results_lbl)

        self._before_after_check = QCheckBox("Show: with earthworks")
        lay.addWidget(self._before_after_check)

        # Connections
        self._run_ew_btn.clicked.connect(self.run_earthworks_requested)
        self._ew_reshape_btn.clicked.connect(self.reshape_earthworks_requested)
        self._before_after_check.toggled.connect(self.before_after_toggled)

    # ---------------------------------------------------------------- Section 5: Live Assessment (network)

    def _build_section_live_assessment(self):
        lay = self._section("Live Assessment")

        from terrainflow_assessment.qgis.widgets.network_view import NetworkView
        self._network = NetworkView()
        lay.addWidget(self._network)

        # Totals + disclaimer line under the network (capacity / cut / fill).
        self._live_assessment_lbl = QLabel("")
        self._live_assessment_lbl.setWordWrap(True)
        self._live_assessment_lbl.setStyleSheet("font-size: 10.5px; color: #7f8c8d;")
        lay.addWidget(self._live_assessment_lbl)

    def _wire_input_change_signals(self):
        """Emit analysis_inputs_changed on any storm/soil input change (drives the live
        analytical readout — the accumulation raster is storm-independent, so no re-analyse).
        Also keeps the header's storm chip and site name live."""
        for spin in (self._rainfall_spin, self._duration_spin, self._cn_spin):
            spin.valueChanged.connect(lambda *_: self.analysis_inputs_changed.emit())
            spin.valueChanged.connect(lambda *_: self._refresh_storm_chip())
        for combo in (self._soil_combo, self._moisture_combo, self._ew_soil_combo):
            combo.currentTextChanged.connect(lambda *_: self.analysis_inputs_changed.emit())
        self._site_name_edit.textChanged.connect(
            lambda text: self._head_site_lbl.setText(text.strip() or "Unnamed Site")
        )
        self._refresh_storm_chip()

    def _refresh_storm_chip(self):
        self._storm_chip.setText(
            f"{self._rainfall_spin.value():.0f} mm · "
            f"{self._duration_spin.value():.0f} h · CN {self._cn_spin.value()} ▾"
        )

    # ---------------------------------------------------------------- Section 6: Simulation

    def _build_section_simulation(self):
        lay = self._section("Fill Simulation", collapsed=False)

        lay.addWidget(self._label("Rainfall mode"))
        self._sim_mode_combo = QComboBox()
        self._sim_mode_combo.addItems(["Uniform event", "Hyetograph CSV"])
        lay.addWidget(self._sim_mode_combo)

        # Uniform event controls
        self._sim_uniform_w = QWidget()
        u_lay = QGridLayout(self._sim_uniform_w)
        u_lay.setContentsMargins(0, 0, 0, 0)
        u_lay.addWidget(self._label("Total rainfall (mm)"), 0, 0)
        self._sim_rain_spin = QDoubleSpinBox()
        self._sim_rain_spin.setRange(1, 2000)
        self._sim_rain_spin.setValue(80)
        self._sim_rain_spin.setSuffix(" mm")
        u_lay.addWidget(self._sim_rain_spin, 0, 1)
        u_lay.addWidget(self._label("Duration (hr)"), 1, 0)
        self._sim_dur_spin = QDoubleSpinBox()
        self._sim_dur_spin.setRange(0.5, 72)
        self._sim_dur_spin.setValue(12)
        self._sim_dur_spin.setSuffix(" hr")
        u_lay.addWidget(self._sim_dur_spin, 1, 1)
        u_lay.addWidget(self._label("Timestep (min)"), 2, 0)
        self._sim_step_spin = QSpinBox()
        self._sim_step_spin.setRange(1, 120)
        self._sim_step_spin.setValue(60)
        self._sim_step_spin.setSuffix(" min")
        u_lay.addWidget(self._sim_step_spin, 2, 1)
        lay.addWidget(self._sim_uniform_w)

        # CSV controls
        self._sim_csv_w = QWidget()
        c_lay = QHBoxLayout(self._sim_csv_w)
        c_lay.setContentsMargins(0, 0, 0, 0)
        self._sim_csv_path = QLineEdit()
        self._sim_csv_path.setPlaceholderText("hyetograph.csv")
        self._sim_csv_browse = QPushButton("Browse…")
        c_lay.addWidget(self._sim_csv_path)
        c_lay.addWidget(self._sim_csv_browse)
        lay.addWidget(self._sim_csv_w)
        self._sim_csv_w.setVisible(False)

        self._run_sim_btn = self._button("Run Simulation", "#8e44ad")
        lay.addWidget(self._run_sim_btn)

        self._sim_progress = QProgressBar()
        self._sim_progress.setVisible(False)
        lay.addWidget(self._sim_progress)

        # Playback controls
        self._sim_controls_w = QWidget()
        sim_c_lay = QVBoxLayout(self._sim_controls_w)
        sim_c_lay.setContentsMargins(0, 0, 0, 0)
        self._sim_slider = QSlider(Qt.Horizontal)
        self._sim_slider.setMinimum(0)
        sim_c_lay.addWidget(self._sim_slider)
        sim_play_row = QHBoxLayout()
        self._sim_play_btn = QPushButton("Play")
        self._sim_play_btn.setCheckable(True)
        self._sim_stop_btn = QPushButton("Stop")
        self._sim_mode_view = QComboBox()
        self._sim_mode_view.addItems(["Incremental", "Cumulative"])
        sim_play_row.addWidget(self._sim_play_btn)
        sim_play_row.addWidget(self._sim_stop_btn)
        sim_play_row.addWidget(self._sim_mode_view)
        sim_c_lay.addLayout(sim_play_row)
        self._sim_time_lbl = self._label("T = —", small=True)
        sim_c_lay.addWidget(self._sim_time_lbl)
        lay.addWidget(self._sim_controls_w)
        self._sim_controls_w.setVisible(False)

        # Fill table
        self._sim_table = QTableWidget(0, 4)
        self._sim_table.setHorizontalHeaderLabels(["Time", "Runoff (m³)", "Exit (L/s)", "Earthworks"])
        self._sim_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._sim_table.setMaximumHeight(180)
        self._sim_table.setVisible(False)
        lay.addWidget(self._sim_table)

        # Connections
        self._sim_mode_combo.currentIndexChanged.connect(self._on_sim_mode_changed)
        self._sim_csv_browse.clicked.connect(self._browse_csv)
        self._run_sim_btn.clicked.connect(self.run_simulation_requested)
        self._sim_slider.valueChanged.connect(self.sim_frame_changed)
        self._sim_play_btn.toggled.connect(self.sim_play_toggled)
        self._sim_stop_btn.clicked.connect(lambda: self._sim_play_btn.setChecked(False))

    def _update_channel_type_label(self, _=None):
        ha = self._threshold_spin.value()
        if ha < 0.5:
            label = "Rills / erosion paths"
        elif ha < 5:
            label = "Ephemeral / seasonal stream"
        elif ha < 20:
            label = "Permanent stream"
        else:
            label = "River"
        self._channel_type_lbl.setText(label)

    def _on_sim_mode_changed(self, idx):
        self._sim_uniform_w.setVisible(idx == 0)
        self._sim_csv_w.setVisible(idx == 1)

    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Hyetograph CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self._sim_csv_path.setText(path)

    # ---------------------------------------------------------------- Section 6: Report

    def _build_section_report(self):
        lay = self._section("Report", collapsed=False)

        self._report_summary_lbl = QLabel("Run baseline and simulation first.")
        self._report_summary_lbl.setWordWrap(True)
        self._report_summary_lbl.setStyleSheet(
            "background: #ecf0f1; padding: 10px; border-radius: 6px; font-size: 11px;"
        )
        lay.addWidget(self._report_summary_lbl)

        self._export_btn = self._button("Export HTML Report", "#1abc9c")
        self._export_btn.setEnabled(False)
        lay.addWidget(self._export_btn)

        self._export_btn.clicked.connect(self.export_report_requested)

    # ---------------------------------------------------------------- public setters

    def set_dem_info(self, info_str):
        self._dem_info_lbl.setText(info_str)
        self._head_info_lbl.setText(info_str or "Load a DEM to begin")

    def set_baseline_progress(self, pct, msg):
        self._run_baseline_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_baseline_complete(self, summary):
        self._run_baseline_btn.set_done()
        self._baseline_results_lbl.setText(summary)
        self.mark_stage("baseline", "done")
        # Enable results tools after first successful baseline
        self._query_ponding_btn.setEnabled(True)
        self._toggle_slope_class_btn.setEnabled(True)
        self._toggle_slope_arrows_btn.setEnabled(True)
        self._generate_contours_btn.setEnabled(True)

    def set_contour_progress(self, pct, msg):
        self._contour_progress.setVisible(True)
        self._contour_progress.setValue(pct)
        self._contour_progress.setFormat(f"{msg} ({pct}%)")

    def set_contour_complete(self):
        self._contour_progress.setVisible(False)
        self._top5_contours_btn.setEnabled(True)
        self._find_segments_btn.setEnabled(True)
        self.mark_stage("analysis", "done")

    def set_keypoint_progress(self, pct, msg):
        self._keypoint_progress.setVisible(True)
        self._keypoint_progress.setValue(pct)
        self._keypoint_progress.setFormat(f"{msg} ({pct}%)")

    def set_keypoint_complete(self, summary=""):
        self._keypoint_progress.setVisible(False)
        self._keypoint_results_lbl.setText(summary)
        self._recommend_ponds_btn.setEnabled(bool(summary))

    def set_earthworks_progress(self, pct, msg):
        self._run_ew_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_earthworks_complete(self, summary=""):
        self._run_ew_btn.set_done()
        self._earthworks_results_lbl.setText(summary)
        self.mark_stage("verify", "done")

    def set_verified_chip(self, text, fresh):
        """Scorecard's verified-vs-design chip (Workbench)."""
        self._scorecard.set_verified(text, fresh)

    def set_live_assessment(self, html):
        """Update the live analytical assessment readout (design-tier, no burn)."""
        self._live_assessment_lbl.setText(html)

    def update_scorecard(self, capture_pct, stored_m3, soaked_m3, leaves_m3):
        """Feed the persistent scorecard with the live balance."""
        self._scorecard.set_balance(capture_pct, stored_m3, soaked_m3, leaves_m3)

    def scorecard_empty(self, message=None):
        if message:
            self._scorecard.show_empty(message)
        else:
            self._scorecard.show_empty()

    def scorecard_no_flow(self, capacity_m3):
        self._scorecard.show_no_flow(capacity_m3)

    def set_contour_results(self, contours):
        self._contour_list.clear()
        for feat in contours[:50]:  # cap display at 50
            item = QListWidgetItem(feat.label)
            item.setCheckState(Qt.Checked)
            self._contour_list.addItem(item)

    def set_keypoint_results(self, summary):
        self._keypoint_results_lbl.setText(summary)

    # The earthwork list is now the flow network (driven by set_network on every
    # recompute). These legacy hooks are retained as no-ops so the controller's
    # mutation paths stay unchanged; the network re-renders via the recompute
    # that always follows each mutation.
    def add_earthwork_to_list(self, index, summary):
        pass

    def update_earthwork_in_list(self, index, summary):
        pass

    def refresh_earthwork_list(self, earthworks):
        pass

    def set_network(self, nodes, edges, exit_m3):
        """Render the earthwork flow network (Live Assessment)."""
        self._network.set_network(nodes, edges, exit_m3)

    def get_selected_earthwork_index(self):
        return self._network.selected_index()

    def set_simulation_progress(self, pct, msg):
        self._sim_progress.setVisible(True)
        self._sim_progress.setValue(pct)
        self._sim_progress.setFormat(f"{msg} ({pct}%)")

    def set_simulation_ready(self, result):
        self._sim_progress.setVisible(False)
        n = len(result.get("frames", []))
        self._sim_slider.setMaximum(max(0, n - 1))
        self._sim_controls_w.setVisible(True)
        self._populate_sim_table(result)

    def _populate_sim_table(self, result):
        table = result.get("timestep_table", [])
        ew_names = [s["name"] for s in result.get("earthwork_summary", [])]

        self._sim_table.setVisible(True)
        self._sim_table.setRowCount(len(table))

        for i, row in enumerate(table):
            self._sim_table.setItem(i, 0, QTableWidgetItem(f"{row['time_min']} min"))
            self._sim_table.setItem(i, 1, QTableWidgetItem(f"{row.get('runoff_m3', 0):,.0f}"))
            self._sim_table.setItem(i, 2, QTableWidgetItem(f"{row.get('outflow_ls', 0):,.1f}"))
            # Summarise earthwork fill in the last column
            ew_fills = " | ".join(
                f"{n}: {row.get(f'{n}_fill_pct', 0):.0f}%"
                for n in ew_names
            )
            self._sim_table.setItem(i, 3, QTableWidgetItem(ew_fills or "—"))

    def set_sim_time_label(self, time_label):
        self._sim_time_lbl.setText(f"T = {time_label}")

    def set_report_summary(self, comparison):
        """Update the report section with before/after headline metrics."""
        from .modules.reporting import ComparisonResult
        if not isinstance(comparison, ComparisonResult):
            return
        text = (
            f"<b>Water captured on-site:</b> {comparison.captured_pct:.0f}%<br>"
            f"<b>Peak flow reduction:</b> {comparison.peak_reduction_pct:.0f}%<br>"
            f"<b>Peak timing delay:</b> {comparison.peak_delay_hr:.1f} hr<br>"
            f"<b>Exit volume reduction:</b> {comparison.exit_reduction_pct:.0f}%<br>"
            f"<b>Net cut/fill balance:</b> {comparison.net_cut_fill_m3:+,.0f} m³"
        )
        self._report_summary_lbl.setText(text)
        self._export_btn.setEnabled(True)

    # ---------------------------------------------------------------- getters

    @property
    def dem_layer(self):
        return self._dem_combo.currentLayer()

    @property
    def boundary_layer(self):
        return self._boundary_combo.currentLayer()

    @property
    def analysis_area_layer(self):
        return self._analysis_area_combo.currentLayer()

    @property
    def earthworks_area_layer(self):
        return self._earthworks_area_combo.currentLayer()

    @property
    def site_name(self):
        return self._site_name_edit.text().strip() or "Unnamed Site"

    @property
    def rainfall_mm(self):
        return self._rainfall_spin.value()

    @property
    def duration_hr(self):
        return self._duration_spin.value()

    @property
    def soil_name(self):
        return self._soil_combo.currentText()

    @property
    def cn(self):
        """Direct CN value from spinner (auto-filled from soil type, but overridable)."""
        return self._cn_spin.value()

    @property
    def moisture(self):
        return self._moisture_combo.currentText()

    @property
    def routing(self):
        return "d8" if "D8" in self._routing_combo.currentText() else "dinf"

    @property
    def stream_threshold_ha(self):
        return self._threshold_spin.value()

    @property
    def contour_interval_m(self):
        return self._contour_interval_spin.value()

    @property
    def simple_contour_interval_m(self):
        return self._simple_contour_interval_spin.value()

    @property
    def max_slope_deg(self):
        return self._max_slope_spin.value()

    @property
    def min_contour_length_m(self):
        return self._min_contour_length_spin.value()

    @property
    def min_catchment_ha(self):
        return self._min_catchment_ha_spin.value()

    @property
    def swale_depth_m(self):
        return self._swale_depth_spin.value()

    @property
    def swale_width_m(self):
        return self._swale_width_spin.value()

    @property
    def keypoint_count(self):
        return self._keypoint_count_spin.value()

    @property
    def earthwork_soil_name(self):
        return self._ew_soil_combo.currentText()

    @property
    def sim_rainfall_mm(self):
        return self._sim_rain_spin.value()

    @property
    def sim_duration_hr(self):
        return self._sim_dur_spin.value()

    @property
    def sim_timestep_min(self):
        return self._sim_step_spin.value()

    @property
    def sim_csv_path(self):
        return self._sim_csv_path.text().strip()

    @property
    def sim_mode(self):
        return "uniform" if self._sim_mode_combo.currentIndex() == 0 else "csv"

    @property
    def sim_display_mode(self):
        return "inc" if self._sim_mode_view.currentIndex() == 0 else "cum"
