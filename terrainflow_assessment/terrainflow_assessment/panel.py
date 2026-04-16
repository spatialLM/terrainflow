"""
panel.py — Docked panel UI for TerrainFlow Assessment.

AssessmentPanel is a QDockWidget with six collapsible sections:
  1. Data (DEM + boundary + site info)
  2. Baseline Analysis (run + results)
  3. Contour & Keypoint Analysis
  4. Earthwork Design (draw tools + properties + soil selector)
  5. Simulation (rainfall input + timestep controls + fill table)
  6. Report (before/after stats + HTML export)
"""

from qgis.PyQt.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QCheckBox, QLineEdit, QSlider, QProgressBar,
    QListWidget, QListWidgetItem, QFileDialog, QTableWidget,
    QTableWidgetItem, QSizePolicy, QFrame, QScrollArea, QTabWidget,
    QAbstractItemView, QHeaderView,
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QFont, QColor

from qgis.core import QgsMapLayerProxyModel
from qgis.gui import QgsMapLayerComboBox


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
    draw_swale_requested = pyqtSignal(str)    # mode: 'freehand' or 'contour'
    draw_berm_requested = pyqtSignal()
    draw_basin_requested = pyqtSignal()
    draw_dam_requested = pyqtSignal()
    draw_diversion_requested = pyqtSignal()
    usable_area_source_changed = pyqtSignal(str)   # "none" | "analysis" | "earthworks"
    run_earthworks_requested = pyqtSignal()
    before_after_toggled = pyqtSignal(bool)   # True = with earthworks

    # Simulation
    run_simulation_requested = pyqtSignal()
    sim_frame_changed = pyqtSignal(int)
    sim_play_toggled = pyqtSignal(bool)

    # Report
    export_report_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("TerrainFlow Assessment", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(340)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(8, 8, 8, 8)

        self._build_ui()

        self._layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

    # ---------------------------------------------------------------- UI construction

    def _build_ui(self):
        self._build_section_data()
        self._build_section_baseline()
        self._build_section_contour_keypoint()
        self._build_section_earthworks()
        self._build_section_simulation()
        self._build_section_report()

    def _section(self, title, collapsed=False):
        """Create a collapsible QGroupBox section."""
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(not collapsed)
        box.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #bdc3c7; "
            "border-radius: 6px; margin-top: 8px; padding-top: 8px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 4px; color: #2c3e50; }"
        )
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 4, 8, 8)
        box.setLayout(QVBoxLayout())
        box.layout().addWidget(inner)
        box.layout().setContentsMargins(0, 12, 0, 0)

        def _toggle(checked):
            inner.setVisible(checked)
        box.toggled.connect(_toggle)
        inner.setVisible(not collapsed)

        self._layout.addWidget(box)
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
        lay = self._section("1 — Data Input")

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

    def _build_section_baseline(self):
        lay = self._section("2 — Baseline Analysis")

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

        self._run_baseline_btn = self._button("Run Baseline Analysis", "#2980b9")
        lay.addWidget(self._run_baseline_btn)

        self._baseline_progress = QProgressBar()
        self._baseline_progress.setVisible(False)
        lay.addWidget(self._baseline_progress)

        self._baseline_results_lbl = self._label("", small=True)
        self._baseline_results_lbl.setWordWrap(True)
        lay.addWidget(self._baseline_results_lbl)

        # Results tools (enabled after baseline runs)
        lay.addWidget(self._label("Results"))
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

        self._generate_contours_btn = QPushButton("Generate Contours")
        self._generate_contours_btn.setEnabled(False)
        self._generate_contours_btn.setToolTip(
            "Generate simple elevation contours from the DEM at the interval\n"
            "set in the Contour & Keypoint Analysis section.\n\n"
            "Useful for visualising terrain alongside the slope classification."
        )
        lay.addWidget(self._generate_contours_btn)

        self._generate_contours_btn.clicked.connect(self.generate_simple_contours_requested)
        self._run_baseline_btn.clicked.connect(self.run_baseline_requested)
        self._query_ponding_btn.clicked.connect(self.query_ponding_requested)
        self._toggle_slope_class_btn.toggled.connect(self.toggle_slope_class_requested)
        self._toggle_slope_arrows_btn.toggled.connect(self.toggle_slope_arrows_requested)

        self._update_channel_type_label()

    # ---------------------------------------------------------------- Section 3: Contour & Keypoint

    def _build_section_contour_keypoint(self):
        lay = self._section("3 — Contour & Keypoint Analysis", collapsed=True)

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
        lay = self._section("4 — Earthwork Design")

        lay.addWidget(self._label("Soil type (for earthwork sizing)"))
        self._ew_soil_combo = QComboBox()
        for name in ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]:
            self._ew_soil_combo.addItem(name)
        self._ew_soil_combo.setCurrentText("Loam")
        lay.addWidget(self._ew_soil_combo)

        # Swale drawing mode
        lay.addWidget(self._label("Draw Swale"))
        swale_row = QHBoxLayout()
        self._draw_swale_contour_btn = QPushButton("Pick Segment")
        self._draw_swale_contour_btn.setToolTip(
            "Click a contour, then pick a start and end point\n"
            "to place a swale along that segment."
        )
        self._draw_swale_fullcontour_btn = QPushButton("Full Contour")
        self._draw_swale_fullcontour_btn.setToolTip(
            "Click a contour line to place a swale along its entire length."
        )
        self._draw_swale_freehand_btn = QPushButton("Freehand")
        self._draw_swale_freehand_btn.setToolTip(
            "Draw a swale freehand — vertices snap to contour elevation."
        )
        swale_row.addWidget(self._draw_swale_contour_btn)
        swale_row.addWidget(self._draw_swale_fullcontour_btn)
        swale_row.addWidget(self._draw_swale_freehand_btn)
        lay.addLayout(swale_row)

        # Other earthwork draw buttons
        draw_row1 = QHBoxLayout()
        self._draw_berm_btn = self._button("Berm")
        self._draw_basin_btn = self._button("Basin")
        self._draw_dam_btn = self._button("Dam")
        draw_row1.addWidget(self._draw_berm_btn)
        draw_row1.addWidget(self._draw_basin_btn)
        draw_row1.addWidget(self._draw_dam_btn)
        lay.addLayout(draw_row1)

        self._draw_diversion_btn = self._button("Diversion Drain")
        lay.addWidget(self._draw_diversion_btn)

        # Earthwork list
        lay.addWidget(self._label("Earthworks"))
        self._ew_list = QListWidget()
        self._ew_list.setMaximumHeight(130)
        lay.addWidget(self._ew_list)

        ew_actions = QHBoxLayout()
        self._ew_edit_btn = QPushButton("Edit")
        self._ew_delete_btn = QPushButton("Delete")
        self._ew_toggle_btn = QPushButton("Enable/Disable")
        ew_actions.addWidget(self._ew_edit_btn)
        ew_actions.addWidget(self._ew_toggle_btn)
        ew_actions.addWidget(self._ew_delete_btn)
        lay.addLayout(ew_actions)

        self._run_ew_btn = self._button("Re-analyse with Earthworks", "#e67e22")
        lay.addWidget(self._run_ew_btn)

        self._earthworks_progress = QProgressBar()
        self._earthworks_progress.setVisible(False)
        lay.addWidget(self._earthworks_progress)

        self._earthworks_results_lbl = self._label("", small=True)
        self._earthworks_results_lbl.setWordWrap(True)
        lay.addWidget(self._earthworks_results_lbl)

        self._before_after_check = QCheckBox("Show: with earthworks")
        lay.addWidget(self._before_after_check)

        # Connections
        self._draw_swale_contour_btn.clicked.connect(
            lambda: self.draw_swale_requested.emit("contour"))
        self._draw_swale_fullcontour_btn.clicked.connect(
            lambda: self.draw_swale_requested.emit("full_contour"))
        self._draw_swale_freehand_btn.clicked.connect(
            lambda: self.draw_swale_requested.emit("freehand"))
        self._draw_berm_btn.clicked.connect(self.draw_berm_requested)
        self._draw_basin_btn.clicked.connect(self.draw_basin_requested)
        self._draw_dam_btn.clicked.connect(self.draw_dam_requested)
        self._draw_diversion_btn.clicked.connect(self.draw_diversion_requested)
        self._run_ew_btn.clicked.connect(self.run_earthworks_requested)
        self._before_after_check.toggled.connect(self.before_after_toggled)

    # ---------------------------------------------------------------- Section 5: Simulation

    def _build_section_simulation(self):
        lay = self._section("5 — Fill Simulation", collapsed=True)

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
        lay = self._section("6 — Report", collapsed=True)

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

    def set_baseline_progress(self, pct, msg):
        self._baseline_progress.setVisible(True)
        self._baseline_progress.setValue(pct)
        self._baseline_progress.setFormat(f"{msg} ({pct}%)")

    def set_baseline_complete(self, summary):
        self._baseline_progress.setVisible(False)
        self._baseline_results_lbl.setText(summary)
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

    def set_keypoint_progress(self, pct, msg):
        self._keypoint_progress.setVisible(True)
        self._keypoint_progress.setValue(pct)
        self._keypoint_progress.setFormat(f"{msg} ({pct}%)")

    def set_keypoint_complete(self, summary=""):
        self._keypoint_progress.setVisible(False)
        self._keypoint_results_lbl.setText(summary)
        self._recommend_ponds_btn.setEnabled(bool(summary))

    def set_earthworks_progress(self, pct, msg):
        self._earthworks_progress.setVisible(True)
        self._earthworks_progress.setValue(pct)
        self._earthworks_progress.setFormat(f"{msg} ({pct}%)")

    def set_earthworks_complete(self, summary=""):
        self._earthworks_progress.setVisible(False)
        self._earthworks_results_lbl.setText(summary)

    def set_contour_results(self, contours):
        self._contour_list.clear()
        for feat in contours[:50]:  # cap display at 50
            item = QListWidgetItem(feat.label)
            item.setCheckState(Qt.Checked)
            self._contour_list.addItem(item)

    def set_keypoint_results(self, summary):
        self._keypoint_results_lbl.setText(summary)

    def add_earthwork_to_list(self, index, summary):
        item = QListWidgetItem(summary)
        item.setCheckState(Qt.Checked)
        self._ew_list.addItem(item)

    def update_earthwork_in_list(self, index, summary):
        if 0 <= index < self._ew_list.count():
            self._ew_list.item(index).setText(summary)

    def refresh_earthwork_list(self, earthworks):
        self._ew_list.clear()
        for ew in earthworks:
            item = QListWidgetItem(ew.summary())
            item.setCheckState(Qt.Checked if ew.enabled else Qt.Unchecked)
            self._ew_list.addItem(item)

    def get_selected_earthwork_index(self):
        row = self._ew_list.currentRow()
        return row if row >= 0 else None

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
