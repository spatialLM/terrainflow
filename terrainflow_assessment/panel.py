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
    design   — Earthwork Design + Live Assessment + Spillways
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
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from terrainflow_assessment.modules.contour_analysis import (
    INFLOW_RAMP_HEX as _INFLOW_RAMP,
)
from terrainflow_assessment.modules.project_io import (
    INPUT_FIELDS,
    SIZING_BASIS_VALUES,
    normalise_inputs,
)
from terrainflow_assessment.qgis import help_text as H
from terrainflow_assessment.qgis.widgets.run_button import RunButton

# Block-drawing characters of rising height — the legend's stand-in for the line
# width the map draws each band at. Rich text in a QLabel cannot vary stroke
# weight, and a row of identical squares would say the bands differ only in
# colour, which is the reading that made the old gradient unreadable.
_BAND_GLYPHS = ("▂", "▄", "▆", "█")

# The report needs a baseline and nothing else. Saying "and simulation" here was
# the visible half of a gate that no longer exists.
_REPORT_SUMMARY_IDLE = "Run Baseline to enable the report."


def _ramp_swatches(labels=None):
    """The inflow ramp as coloured, thickening glyphs, optionally labelled."""
    out = []
    for i, hexcode in enumerate(_INFLOW_RAMP):
        glyph = _BAND_GLYPHS[min(i, len(_BAND_GLYPHS) - 1)]
        chip = f"<span style='color:{hexcode}'>{glyph}</span>"
        out.append(f"{chip}&nbsp;{labels[i]}" if labels else chip)
    return "&nbsp;&nbsp;".join(out) if labels else "".join(out)


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
    # The site name heads the output layer group, so a name typed after the first
    # run has to reach the layer tree (see _groups.rename_default_site).
    site_name_changed = pyqtSignal(str)
    # Draw-on-canvas requests for the three polygon area pickers (for QGIS novices)
    draw_boundary_requested = pyqtSignal()
    draw_analysis_area_requested = pyqtSignal()
    draw_earthworks_area_requested = pyqtSignal()
    # Portable design files — save the whole session, reopen it here or on another machine
    save_design_requested = pyqtSignal()
    open_design_requested = pyqtSignal()

    # Baseline
    run_baseline_requested = pyqtSignal()
    threshold_changed = pyqtSignal()
    query_ponding_requested = pyqtSignal()
    toggle_slope_class_requested = pyqtSignal(bool)
    toggle_slope_vectors_requested = pyqtSignal(bool)
    toggle_throughflow_requested = pyqtSignal(bool)   # blue per-cell water gradient
    throughflow_scale_changed = pyqtSignal(str)

    # Contour analysis
    run_contour_analysis_requested = pyqtSignal()
    select_top5_contours_requested = pyqtSignal()
    find_segments_requested = pyqtSignal()
    show_inflow_bands_requested = pyqtSignal(bool)   # colour contours by inflow share
    show_segment_gradient_requested = pyqtSignal(bool)  # peak inflow inside each segment
    # Which contours the user has ruled out → hide them and drop them from the
    # analysis. Carries the *unticked* rows, not the ticked ones: the list caps its
    # display at 50 rows, so "not in the ticked set" would silently rule out every
    # contour past the cap.
    contour_visibility_changed = pyqtSignal(object)  # list[int] of unticked row indices
    # Which contour rows are highlighted → select the matching features on the map
    contour_rows_selected = pyqtSignal(object)       # list[int] of selected row indices
    clear_analysis_requested = pyqtSignal()          # wipe analysis layers + state
    generate_simple_contours_requested = pyqtSignal()
    contour_layer_changed = pyqtSignal(object)
    run_keypoint_analysis_requested = pyqtSignal()
    recommend_ponds_requested = pyqtSignal()
    keypoint_result_activated = pyqtSignal(float, float)  # (x, y) → zoom canvas to it
    segment_activated = pyqtSignal(str)  # segment WKT → highlight + zoom to it
    run_keyline_requested = pyqtSignal()   # generate Yeomans keyline + guides
    draw_keyline_requested = pyqtSignal()  # draw a keyline plough guide freehand
    convert_keyline_to_swale_requested = pyqtSignal()  # master keyline → swale

    # Earthworks
    draw_swale_requested = pyqtSignal(str)      # mode: 'freehand' | 'contour' | 'full_contour'
    draw_earthwork_requested = pyqtSignal(str)  # registry type key (berm/basin/dam/…)
    usable_area_source_changed = pyqtSignal(str)   # "none" | "analysis" | "earthworks"
    run_earthworks_requested = pyqtSignal()
    reshape_earthworks_requested = pyqtSignal()   # vertex-drag tool with live readout
    place_spillway_requested = pyqtSignal(str)    # 'outflow' | 'inflow'
    # Same, but naming the feature by row — the Spillways list has its own rows and
    # should not have to reach through the flow network's selection to say which.
    place_spillway_for_requested = pyqtSignal(int, str)   # index, kind
    edit_earthwork_requested = pyqtSignal(int)            # index → properties dialog
    connect_earthworks_requested = pyqtSignal()   # route one feature's overflow to another
    choose_design_intensity_requested = pyqtSignal()  # open the peak-intensity comparison
    edit_rainfall_data_requested = pyqtSignal()       # enter the site's HIRDS table
    earthwork_selected = pyqtSignal(object)          # index, or None — highlight it
    before_after_toggled = pyqtSignal(bool)   # True = with earthworks
    toggle_catchment_layer_requested = pyqtSignal(bool)  # who catches what, by colour
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

        # True once a baseline run has completed — lets input changes mark the
        # analysis stale instead of leaving a tick over numbers that have moved.
        self._baseline_has_run = False

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
        self._storm_chip.setToolTip(H.STORM_CHIP)
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

    def _mark_stage_failed(self, key):
        """A run that errored leaves the stage un-ticked.

        Amber when an earlier run left usable output behind — that output is
        still there but no longer reflects what was just attempted — and quiet
        when nothing has ever succeeded. Never green.
        """
        previously_done = self._stepper.state(key) == "done"
        self.mark_stage(key, "stale" if previously_done else "todo")

    # ---------------------------------------------------------------- UI construction

    def _build_ui(self):
        self._stack = QStackedWidget()
        self._stage_layouts = {}
        # Section header state, so a collapsed section can still carry a live note.
        self._section_headers = {}
        self._section_titles = {}
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
        self._build_section_spillways()

        self._layout = self._stage_layouts["verify"]
        self._build_section_verification()
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

        # The header text is rebuilt on every expand/collapse, so a suffix written onto
        # the button directly would be wiped the first time the user closed the section.
        # Keeping it in a dict the closure reads means a collapsed section can still
        # report its state — which is the only way a collapsed section is worth having.
        self._section_titles[title] = title

        def _toggle(checked):
            body.setVisible(checked)
            header.setText(("▼  " if checked else "▶  ")
                           + self._section_titles.get(title, title))
        header.toggled.connect(_toggle)
        self._section_headers[title] = (header, _toggle)
        _toggle(not collapsed)

        outer.addWidget(header)
        outer.addWidget(body)
        self._layout.addWidget(frame)
        return layout

    def set_section_note(self, title, note=""):
        """Append a live state note to a section header, visible while collapsed."""
        if title not in self._section_headers:
            return
        self._section_titles[title] = f"{title} — {note}" if note else title
        header, toggle = self._section_headers[title]
        toggle(header.isChecked())

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

    def _area_row(self, combo, draw_signal, tooltip):
        """Wrap a layer combo with an inline '✏ Draw' button that emits *draw_signal*.

        Lets QGIS-novice users draw the polygon directly instead of selecting an
        existing layer (the drawn layer is then auto-selected in *combo*).
        """
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(combo, 1)
        draw_btn = QPushButton("✏ Draw")
        draw_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        draw_btn.setToolTip(tooltip)
        draw_btn.setStyleSheet(
            "QPushButton { border: 1px solid #c6d1d3; border-radius: 4px;"
            " padding: 4px 9px; font-size: 11px; color: #2e7d55; background: #ffffff; }"
            "QPushButton:hover { background: #e9f3ee; border-color: #2e7d55; }"
        )
        draw_btn.clicked.connect(lambda: draw_signal.emit())
        h.addWidget(draw_btn)
        return row

    _SECONDARY_BTN_STYLE = (
        "QPushButton { border: 1px solid #c6d1d3; border-radius: 4px;"
        " padding: 4px 9px; font-size: 11px; color: #2e7d55; background: #ffffff; }"
        "QPushButton:hover { background: #e9f3ee; border-color: #2e7d55; }"
    )

    def _design_file_row(self):
        """Save / Open buttons for a portable design file.

        Sits at the foot of Data Input because that is what a design file *is* — the whole
        set of inputs above it, plus the earthworks drawn from them, in one file that can
        move between machines.
        """
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 4, 0, 0)
        h.setSpacing(6)

        save_btn = QPushButton("💾 Save design")
        save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        save_btn.setToolTip(H.SAVE_DESIGN)
        save_btn.setStyleSheet(self._SECONDARY_BTN_STYLE)
        save_btn.clicked.connect(lambda: self.save_design_requested.emit())

        open_btn = QPushButton("📂 Open design")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setToolTip(H.OPEN_DESIGN)
        open_btn.setStyleSheet(self._SECONDARY_BTN_STYLE)
        open_btn.clicked.connect(lambda: self.open_design_requested.emit())

        h.addWidget(save_btn, 1)
        h.addWidget(open_btn, 1)
        return row

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
        lay.addWidget(self._area_row(
            self._boundary_combo, self.draw_boundary_requested,
            H.DRAW_BOUNDARY))

        self._site_name_edit = QLineEdit()
        self._site_name_edit.setPlaceholderText("Site name (for report)")
        lay.addWidget(self._site_name_edit)

        lay.addWidget(self._label("Analysis area (post-analysis tools)"))
        self._analysis_area_combo = QgsMapLayerComboBox()
        self._analysis_area_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self._analysis_area_combo.setAllowEmptyLayer(True)
        self._analysis_area_combo.setToolTip(H.ANALYSIS_AREA)
        lay.addWidget(self._area_row(
            self._analysis_area_combo, self.draw_analysis_area_requested,
            H.DRAW_ANALYSIS_AREA))

        lay.addWidget(self._label("Earthworks area (polygon layer)"))
        self._earthworks_area_combo = QgsMapLayerComboBox()
        self._earthworks_area_combo.setFilters(QgsMapLayerProxyModel.PolygonLayer)
        self._earthworks_area_combo.setAllowEmptyLayer(True)
        self._earthworks_area_combo.setToolTip(H.EARTHWORKS_AREA)
        lay.addWidget(self._area_row(
            self._earthworks_area_combo, self.draw_earthworks_area_requested,
            H.DRAW_EARTHWORKS_AREA))

        lay.addWidget(self._design_file_row())

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
        self._rainfall_spin.setToolTip(H.RAINFALL)
        rf_grid.addWidget(self._rainfall_spin, 0, 1)

        rf_grid.addWidget(self._label("Duration (hr)"), 1, 0)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.1, 72)
        self._duration_spin.setValue(24)
        self._duration_spin.setSuffix(" hr")
        self._duration_spin.setToolTip(H.DURATION)
        rf_grid.addWidget(self._duration_spin, 1, 1)

        # Asked immediately after the storm itself, and before anything it governs:
        # the three methods want different inputs, and the rows below are shown or
        # hidden to match (see _sync_basis_controls). Asking for a curve number and a
        # runoff coefficient side by side, only one of which is ever read, was the
        # single most confusing thing on this panel.
        #
        # Which depth of water everything downstream works from — the analysis
        # rasters as well as earthwork sizing. It belongs with the storm inputs
        # because it IS a statement about the storm, and having Analysis and Design
        # quote different depths for the same event would be indefensible.
        rf_grid.addWidget(self._label("Runoff Calculation Method"), 2, 0)
        self._sizing_basis_combo = QComboBox()
        self._sizing_basis_combo.addItems([
            "Runoff coefficient (Lancaster)",
            "Total rainfall (most conservative)",
            "Surface runoff (SCS-CN)",
        ])
        self._sizing_basis_combo.setToolTip(H.RUNOFF_METHOD)
        self._sizing_basis_combo.currentIndexChanged.connect(self._on_basis_changed)
        rf_grid.addWidget(self._sizing_basis_combo, 2, 1)

        # Lancaster's surface table drives the coefficient; the spin stays editable so
        # a measured or locally-derived value can be typed straight in.
        from terrainflow_assessment.modules.catchment import (
            DEFAULT_RUNOFF_COEFFICIENT,
            LANCASTER_COEFFICIENTS,
        )
        from terrainflow_assessment.modules.peak_flow import DEFAULT_PEAK_INTENSITY_MM_HR
        surface_lbl = self._label("Surface")
        rf_grid.addWidget(surface_lbl, 3, 0)
        self._runoff_surface_combo = QComboBox()
        for label, (typical, lo, hi, src) in LANCASTER_COEFFICIENTS.items():
            tag = "" if src == "lancaster" else "  ·  design default"
            self._runoff_surface_combo.addItem(
                f"{label} — {typical:.2f}  ({lo:.2f}–{hi:.2f}){tag}", typical)
        self._runoff_surface_combo.addItem("Custom", None)
        self._runoff_surface_combo.setToolTip(H.RUNOFF_SURFACE)
        rf_grid.addWidget(self._runoff_surface_combo, 3, 1)

        coeff_lbl = self._label("Runoff coefficient (C)")
        rf_grid.addWidget(coeff_lbl, 4, 0)
        self._runoff_coeff_spin = QDoubleSpinBox()
        self._runoff_coeff_spin.setRange(0.01, 1.0)
        self._runoff_coeff_spin.setSingleStep(0.05)
        self._runoff_coeff_spin.setDecimals(2)
        self._runoff_coeff_spin.setValue(DEFAULT_RUNOFF_COEFFICIENT)
        self._runoff_coeff_spin.setToolTip(H.RUNOFF_COEFFICIENT)
        rf_grid.addWidget(self._runoff_coeff_spin, 4, 1)

        soil_lbl = self._label("Soil Type")
        rf_grid.addWidget(soil_lbl, 5, 0)
        self._soil_combo = QComboBox()
        for name in ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]:
            self._soil_combo.addItem(name)
        self._soil_combo.setCurrentText("Loam")
        self._soil_combo.setToolTip(H.SOIL_TYPE)
        rf_grid.addWidget(self._soil_combo, 5, 1)

        # Ground condition moves the curve number further than soil texture does — on
        # sand, Good is CN 39 and Poor is 68 — and it was previously assumed to be Good
        # and never asked. Sits next to Soil Type because the pair is what selects a
        # row from TR-55 Table 2-2; neither means much without the other.
        from terrainflow_assessment.modules.catchment import (
            DEFAULT_GROUND_CONDITION,
            GROUND_CONDITIONS,
        )
        ground_lbl = self._label("Ground condition")
        rf_grid.addWidget(ground_lbl, 6, 0)
        self._ground_condition_combo = QComboBox()
        for key, (label, meaning) in GROUND_CONDITIONS.items():
            self._ground_condition_combo.addItem(label, key)
            self._ground_condition_combo.setItemData(
                self._ground_condition_combo.count() - 1,
                meaning, Qt.ItemDataRole.ToolTipRole)
        self._ground_condition_combo.setCurrentIndex(
            list(GROUND_CONDITIONS).index(DEFAULT_GROUND_CONDITION))
        self._ground_condition_combo.setToolTip(H.GROUND_CONDITION)
        rf_grid.addWidget(self._ground_condition_combo, 6, 1)

        cn_lbl = self._label("Curve Number (CN)")
        rf_grid.addWidget(cn_lbl, 7, 0)
        self._cn_spin = QSpinBox()
        self._cn_spin.setRange(1, 100)
        self._cn_spin.setValue(61)
        self._cn_spin.setToolTip(H.CURVE_NUMBER)
        rf_grid.addWidget(self._cn_spin, 7, 1)

        moisture_lbl = self._label("Moisture Condition")
        rf_grid.addWidget(moisture_lbl, 8, 0)
        self._moisture_combo = QComboBox()
        self._moisture_combo.addItems(["normal", "dry", "wet"])
        self._moisture_combo.setToolTip(H.MOISTURE)
        rf_grid.addWidget(self._moisture_combo, 8, 1)

        # Label and field together, because hiding a field but not its label leaves
        # a caption pointing at the row beneath it.
        self._basis_rows = {
            "coefficient": [(surface_lbl, self._runoff_surface_combo),
                            (coeff_lbl, self._runoff_coeff_spin)],
            "runoff": [(soil_lbl, self._soil_combo),
                       (ground_lbl, self._ground_condition_combo),
                       (cn_lbl, self._cn_spin),
                       (moisture_lbl, self._moisture_combo)],
        }

        # Peak intensity — sizes overflow structures, and cannot be derived from the
        # depth/duration above. 120 mm in 2 h and in 24 h are identical storage and a
        # twelve-fold difference in peak flow, so this is asked rather than assumed.
        rf_grid.addWidget(self._label("Peak intensity"), 9, 0)
        intensity_row = QHBoxLayout()
        intensity_row.setContentsMargins(0, 0, 0, 0)
        intensity_row.setSpacing(4)
        self._peak_intensity_spin = QDoubleSpinBox()
        self._peak_intensity_spin.setRange(0.1, 500.0)
        self._peak_intensity_spin.setDecimals(1)
        self._peak_intensity_spin.setSingleStep(5.0)
        self._peak_intensity_spin.setSuffix(" mm/hr")
        self._peak_intensity_spin.setValue(DEFAULT_PEAK_INTENSITY_MM_HR)
        self._peak_intensity_spin.setToolTip(H.PEAK_INTENSITY)
        self._peak_intensity_spin.valueChanged.connect(self.analysis_inputs_changed)
        intensity_row.addWidget(self._peak_intensity_spin, 1)

        self._intensity_choose_btn = QPushButton("Compare…")
        self._intensity_choose_btn.setToolTip(H.DESIGN_INTENSITY_TABLE)
        self._intensity_choose_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._intensity_choose_btn.clicked.connect(self.choose_design_intensity_requested)
        intensity_row.addWidget(self._intensity_choose_btn)
        rf_grid.addLayout(intensity_row, 9, 1)

        # Site rainfall statistics (HIRDS). Not derivable from terrain, so it is an
        # explicit input — and with it the intensity above becomes a lookup at the
        # catchment's own response time rather than a judgement.
        rf_grid.addWidget(self._label("Site rainfall data"), 10, 0)
        self._rainfall_data_btn = QPushButton("Enter HIRDS table…")
        self._rainfall_data_btn.setToolTip(H.RAINFALL_DATA)
        self._rainfall_data_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rainfall_data_btn.clicked.connect(self.edit_rainfall_data_requested)
        rf_grid.addWidget(self._rainfall_data_btn, 10, 1)

        self._runoff_surface_combo.currentIndexChanged.connect(self._on_surface_preset)
        self._runoff_coeff_spin.valueChanged.connect(self._on_basis_changed)
        self._sync_basis_controls()

        lay.addLayout(rf_grid)

        # Wire soil + ground condition → CN auto-fill. The CN spinner stays editable:
        # this fills in a defensible starting point, it does not take the decision away.
        #
        # The lookup comes from SCSRunoff rather than a table copied into this file.
        # There were three copies of the TR-55 curve numbers already, and the fourth
        # one here would have gone on answering "Loam is 61" after the module learned
        # that Loam is 61 only on well-covered ground.
        self._soil_combo.currentTextChanged.connect(
            lambda *_: self._refresh_cn_from_soil())
        self._ground_condition_combo.currentIndexChanged.connect(
            lambda *_: self._refresh_cn_from_soil())

        # Stream threshold
        thr_grid = QGridLayout()
        thr_grid.addWidget(self._label("Stream Threshold (ha)"), 0, 0)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.1, 10000)
        self._threshold_spin.setValue(5.0)
        self._threshold_spin.setSuffix(" ha")
        self._threshold_spin.setToolTip(H.STREAM_THRESHOLD)
        self._threshold_spin.valueChanged.connect(self._update_channel_type_label)
        thr_grid.addWidget(self._threshold_spin, 0, 1)

        self._channel_type_lbl = self._label("", small=True)
        self._channel_type_lbl.setStyleSheet("color: #555555; font-style: italic;")
        thr_grid.addWidget(self._channel_type_lbl, 1, 0, 1, 2)

        thr_grid.addWidget(self._label("Show exits above (L/s)"), 2, 0)
        self._exit_flow_spin = QDoubleSpinBox()
        self._exit_flow_spin.setRange(0.0, 10000.0)
        self._exit_flow_spin.setValue(0.5)
        self._exit_flow_spin.setSuffix(" L/s")
        self._exit_flow_spin.setDecimals(2)
        self._exit_flow_spin.setSingleStep(0.5)
        self._exit_flow_spin.setToolTip(H.EXIT_FLOW)
        thr_grid.addWidget(self._exit_flow_spin, 2, 1)

        thr_grid.addWidget(self._label("Routing"), 3, 0)
        self._routing_combo = QComboBox()
        self._routing_combo.addItems(["D-infinity (recommended)", "D8"])
        thr_grid.addWidget(self._routing_combo, 3, 1)
        lay.addLayout(thr_grid)

        self._run_baseline_btn = RunButton("Run Baseline Analysis")
        lay.addWidget(self._run_baseline_btn)

        self._baseline_results_lbl = self._label("", small=True)
        self._baseline_results_lbl.setWordWrap(True)
        lay.addWidget(self._baseline_results_lbl)

        # Per-area outflow readout (populated after baseline) — instant feedback on
        # how much water leaves each defined area.
        self._area_outflow_lbl = QLabel("")
        self._area_outflow_lbl.setWordWrap(True)
        self._area_outflow_lbl.setVisible(False)
        self._area_outflow_lbl.setStyleSheet(
            "background: #eef4f8; border: 1px solid #d3e0e6; border-radius: 4px;"
            " padding: 6px; font-size: 11px; color: #2c3e50;"
        )
        lay.addWidget(self._area_outflow_lbl)

        self._run_baseline_btn.clicked.connect(self.run_baseline_requested)

    def _build_section_baseline_results(self):
        """Post-baseline exploration tools (Analysis stage). Enabled after a run."""
        lay = self._section("Terrain Tools")

        lay.addWidget(self._label(
            "Run Baseline first, then explore ponding, slope and contours."
        ))
        self._query_ponding_btn = QPushButton("Query Depression / Ponding")
        self._query_ponding_btn.setEnabled(False)
        self._query_ponding_btn.setToolTip(H.QUERY_PONDING)
        lay.addWidget(self._query_ponding_btn)

        slope_class_row = QHBoxLayout()
        slope_class_row.setSpacing(6)
        self._toggle_slope_class_btn = QPushButton("Show Slope Classification")
        self._toggle_slope_class_btn.setCheckable(True)
        self._toggle_slope_class_btn.setEnabled(False)
        self._toggle_slope_class_btn.setToolTip(H.SLOPE_CLASS)
        slope_class_row.addWidget(self._toggle_slope_class_btn, 1)
        self._slope_class_info_btn = QPushButton("ⓘ")
        self._slope_class_info_btn.setFixedWidth(30)
        self._slope_class_info_btn.setToolTip(H.SLOPE_CLASS_INFO_BTN)
        self._slope_class_info_btn.clicked.connect(self._show_slope_class_info)
        slope_class_row.addWidget(self._slope_class_info_btn)
        lay.addLayout(slope_class_row)

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
            ("#660000", "25–50°"),
            ("#3A0000", "≥50°"),
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

        self._toggle_slope_vectors_btn = QPushButton("Show Slope Vectors")
        self._toggle_slope_vectors_btn.setCheckable(True)
        self._toggle_slope_vectors_btn.setEnabled(False)
        self._toggle_slope_vectors_btn.setToolTip(H.SLOPE_VECTORS)
        lay.addWidget(self._toggle_slope_vectors_btn)

        contour_row = QHBoxLayout()
        self._simple_contour_interval_spin = QDoubleSpinBox()
        self._simple_contour_interval_spin.setRange(0.1, 100.0)
        self._simple_contour_interval_spin.setValue(1.0)
        self._simple_contour_interval_spin.setSuffix(" m")
        self._simple_contour_interval_spin.setSingleStep(0.5)
        self._simple_contour_interval_spin.setDecimals(1)
        self._simple_contour_interval_spin.setToolTip(H.SIMPLE_CONTOUR_INTERVAL)
        self._simple_contour_interval_spin.setFixedWidth(75)
        contour_row.addWidget(self._simple_contour_interval_spin)

        self._generate_contours_btn = QPushButton("Generate Contours")
        self._generate_contours_btn.setEnabled(False)
        self._generate_contours_btn.setToolTip(H.GENERATE_CONTOURS)
        contour_row.addWidget(self._generate_contours_btn)
        lay.addLayout(contour_row)

        # Surface runoff: total event water through every cell, as a blue
        # gradient. Sits with the other terrain overlays; off by default (it
        # covers the map). Named for what it shows rather than for the
        # accumulation step that produces it — "throughflow" is a soil-science
        # term for subsurface flow, which is the opposite of this.
        flow_row = QHBoxLayout()
        self._toggle_throughflow_btn = QPushButton("Surface Runoff")
        self._toggle_throughflow_btn.setCheckable(True)
        self._toggle_throughflow_btn.setEnabled(False)
        self._toggle_throughflow_btn.setToolTip(H.THROUGHFLOW)
        flow_row.addWidget(self._toggle_throughflow_btn)

        self._throughflow_scale_combo = QComboBox()
        self._throughflow_scale_combo.addItems(["Log", "Linear", "Quantile"])
        self._throughflow_scale_combo.setToolTip(H.THROUGHFLOW_SCALE)
        self._throughflow_scale_combo.setFixedWidth(90)
        self._throughflow_scale_combo.setEnabled(False)
        flow_row.addWidget(self._throughflow_scale_combo)

        # Read from the ramp the layer is actually painted with, rather than
        # hand-copied from it — these four hexes used to live here as literals
        # under a comment asking whoever changed the renderer to remember to
        # change them here too. The renderer fades the low end out with alpha so
        # the map stays readable underneath; a key shows the hue, not the blend,
        # so the swatches carry a border to keep the near-white one visible. The
        # transparent "none" stop drops out of visible_stops by construction.
        from terrainflow_assessment.core.registry.map_palette import (
            surface_runoff_ramp,
            visible_stops,
        )

        for hex_colour, lbl_text in visible_stops(surface_runoff_ramp()):
            swatch = QLabel()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(
                f"background-color: {hex_colour}; border: 1px solid #888;"
            )
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet("font-size: 10px; color: #5f7176;")
            flow_row.addWidget(swatch)
            flow_row.addWidget(lbl)
        flow_row.addStretch()
        lay.addLayout(flow_row)

        self._toggle_throughflow_btn.toggled.connect(self.toggle_throughflow_requested)
        self._throughflow_scale_combo.currentIndexChanged.connect(
            lambda _i: self.throughflow_scale_changed.emit(self.throughflow_scale_mode))

        self._generate_contours_btn.clicked.connect(self.generate_simple_contours_requested)
        self._query_ponding_btn.clicked.connect(self.query_ponding_requested)
        self._toggle_slope_class_btn.toggled.connect(self.toggle_slope_class_requested)
        self._toggle_slope_vectors_btn.toggled.connect(self.toggle_slope_vectors_requested)

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
        self._contour_interval_spin.setToolTip(H.CONTOUR_INTERVAL)
        contour_lay.addWidget(self._contour_interval_spin)

        contour_lay.addWidget(self._label("Max slope (°) — filter"))
        self._max_slope_spin = QDoubleSpinBox()
        self._max_slope_spin.setRange(1, 45)
        self._max_slope_spin.setValue(18.0)
        self._max_slope_spin.setSuffix("°")
        self._max_slope_spin.setToolTip(H.MAX_SLOPE)
        contour_lay.addWidget(self._max_slope_spin)

        contour_lay.addWidget(self._label("Usable area (clip contours to)"))
        self._usable_area_combo = QComboBox()
        self._usable_area_combo.addItems(["None", "Analysis Area", "Earthworks Area"])
        self._usable_area_combo.setToolTip(H.USABLE_AREA)
        contour_lay.addWidget(self._usable_area_combo)

        contour_lay.addWidget(self._label("Min contour length (m)"))
        self._min_contour_length_spin = QDoubleSpinBox()
        self._min_contour_length_spin.setRange(0, 5000)
        self._min_contour_length_spin.setValue(50.0)
        self._min_contour_length_spin.setSuffix(" m")
        self._min_contour_length_spin.setSingleStep(10)
        self._min_contour_length_spin.setToolTip(H.MIN_CONTOUR_LENGTH)
        contour_lay.addWidget(self._min_contour_length_spin)

        self._run_contour_btn = RunButton(
            "Analyse Contours", accent="#27ae60", accent_hover="#1e8449",
            ghost_bg="#eafaf1")
        self._run_contour_btn.setToolTip(H.ANALYSE_CONTOURS)
        contour_lay.addWidget(self._run_contour_btn)

        # Results area as a vertical splitter so both lists can be dragged
        # taller/shorter to show more items (top = contours, bottom = segments,
        # controls in the middle).
        results_splitter = QSplitter(Qt.Vertical)
        results_splitter.setChildrenCollapsible(False)

        self._contour_list = QListWidget()
        self._contour_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._contour_list.setMinimumHeight(80)
        self._contour_list.setToolTip(H.CONTOUR_LIST)
        # Set while the map's selection is being mirrored onto the rows, so the
        # rows don't turn round and re-select the map features they came from.
        self._syncing_contour_rows = False
        self._contour_list.itemChanged.connect(self._on_contour_item_changed)
        self._contour_list.itemSelectionChanged.connect(self._on_contour_rows_selected)
        results_splitter.addWidget(self._contour_list)

        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)

        # Legend for the inflow bands the candidate contours are drawn in. Filled
        # with the run's actual break values by set_contour_legend().
        self._contour_legend = QLabel(H.CONTOUR_LEGEND_EMPTY)
        self._contour_legend.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        self._contour_legend.setToolTip(H.CONTOUR_LEGEND)
        mid_lay.addWidget(self._contour_legend)

        top_n_row = QHBoxLayout()
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 50)
        self._top_n_spin.setValue(5)
        self._top_n_spin.setFixedWidth(56)
        self._top_n_spin.setToolTip(H.TOP_N)
        top_n_row.addWidget(self._top_n_spin)
        self._top5_contours_btn = self._button("Select Top Swales", "#1a6b3a")
        self._top5_contours_btn.setEnabled(False)
        self._top5_contours_btn.setToolTip(H.TOP5_SWALES)
        top_n_row.addWidget(self._top5_contours_btn, 1)
        mid_lay.addLayout(top_n_row)

        inflow_row = QHBoxLayout()
        self._inflow_bands_btn = QPushButton("Show Inflow Gradient")
        self._inflow_bands_btn.setCheckable(True)
        self._inflow_bands_btn.setEnabled(False)
        self._inflow_bands_btn.setToolTip(H.INFLOW_BANDS)
        inflow_row.addWidget(self._inflow_bands_btn, 1)
        self._inflow_scale_combo = QComboBox()
        # "Natural" first, and therefore the default: it is what the candidate
        # contour bands use, so the list and the gradient agree out of the box.
        self._inflow_scale_combo.addItems(["Natural", "Log", "Linear", "Quantile"])
        self._inflow_scale_combo.setToolTip(H.INFLOW_SCALE)
        self._inflow_scale_combo.setFixedWidth(90)
        # Re-render live when the scale changes and either gradient is showing.
        self._inflow_scale_combo.currentIndexChanged.connect(self._on_inflow_scale_changed)
        inflow_row.addWidget(self._inflow_scale_combo)
        mid_lay.addLayout(inflow_row)

        # Inflow gradient legend — four bands, absolute m³, shared across the
        # contours in scope (see the layer's own legend for the m³ ranges). The
        # swatches thicken with the bands because on the map width is the primary
        # signal; a row of equal squares would misdescribe it.
        band_legend = QLabel("Inflow m³ (low→high): " + _ramp_swatches())
        band_legend.setStyleSheet("font-size: 10px; color: #7f8c8d;")
        band_legend.setToolTip(H.CONTOUR_LEGEND)
        mid_lay.addWidget(band_legend)

        # --- Segment analysis ---
        mid_lay.addWidget(self._label("Min catchment above swale (ha)"))
        self._min_catchment_ha_spin = QDoubleSpinBox()
        self._min_catchment_ha_spin.setRange(0.0, 500.0)
        self._min_catchment_ha_spin.setValue(0.5)
        self._min_catchment_ha_spin.setSuffix(" ha")
        self._min_catchment_ha_spin.setSingleStep(0.5)
        self._min_catchment_ha_spin.setToolTip(H.MIN_CATCHMENT)
        mid_lay.addWidget(self._min_catchment_ha_spin)

        seg_mode_row = QHBoxLayout()
        seg_mode_row.addWidget(self._label("Rank segments by"))
        self._seg_rank_combo = QComboBox()
        self._seg_rank_combo.addItems(["Min catchment above", "Largest inflow"])
        self._seg_rank_combo.setToolTip(H.SEG_RANK_MODE)
        seg_mode_row.addWidget(self._seg_rank_combo, 1)
        mid_lay.addLayout(seg_mode_row)

        seg_slope_row = QHBoxLayout()
        self._seg_slope_check = QCheckBox("Limit segment slope")
        self._seg_slope_check.setToolTip(H.SEG_MAX_SLOPE)
        seg_slope_row.addWidget(self._seg_slope_check)
        self._seg_max_slope_spin = QDoubleSpinBox()
        self._seg_max_slope_spin.setRange(1, 45)
        self._seg_max_slope_spin.setValue(10.0)
        self._seg_max_slope_spin.setSuffix("°")
        self._seg_max_slope_spin.setEnabled(False)
        self._seg_max_slope_spin.setToolTip(H.SEG_MAX_SLOPE)
        self._seg_slope_check.toggled.connect(self._seg_max_slope_spin.setEnabled)
        seg_slope_row.addWidget(self._seg_max_slope_spin)
        mid_lay.addLayout(seg_slope_row)

        swale_dim_grid = QGridLayout()
        swale_dim_grid.addWidget(self._label("Swale depth (m)"), 0, 0)
        self._swale_depth_spin = QDoubleSpinBox()
        self._swale_depth_spin.setRange(0.1, 2.0)
        self._swale_depth_spin.setValue(0.3)
        self._swale_depth_spin.setSuffix(" m")
        self._swale_depth_spin.setSingleStep(0.05)
        self._swale_depth_spin.setDecimals(2)
        self._swale_depth_spin.setToolTip(H.SWALE_DEPTH)
        swale_dim_grid.addWidget(self._swale_depth_spin, 0, 1)

        swale_dim_grid.addWidget(self._label("Swale width (m)"), 1, 0)
        self._swale_width_spin = QDoubleSpinBox()
        self._swale_width_spin.setRange(0.1, 10.0)
        self._swale_width_spin.setValue(0.6)
        self._swale_width_spin.setSuffix(" m")
        self._swale_width_spin.setSingleStep(0.1)
        self._swale_width_spin.setDecimals(2)
        self._swale_width_spin.setToolTip(H.SWALE_WIDTH)
        swale_dim_grid.addWidget(self._swale_width_spin, 1, 1)
        mid_lay.addLayout(swale_dim_grid)

        self._find_segments_btn = RunButton(
            "Find Best Swale Segments", accent="#145a32", accent_hover="#0e3d22",
            ghost_bg="#e8f5ee")
        self._find_segments_btn.setEnabled(False)
        self._find_segments_btn.setToolTip(H.FIND_SEGMENTS)
        mid_lay.addWidget(self._find_segments_btn)

        self._segment_gradient_check = QCheckBox("Show peak inflow inside segments")
        self._segment_gradient_check.setEnabled(False)
        self._segment_gradient_check.setToolTip(H.SEGMENT_GRADIENT)
        self._segment_gradient_check.setStyleSheet("font-size: 11px;")
        mid_lay.addWidget(self._segment_gradient_check)
        results_splitter.addWidget(mid)

        # Segment results — click a row to highlight+zoom; ✓ holds / ⚠ needs overflow.
        self._segment_list = QListWidget()
        self._segment_list.setMinimumHeight(80)
        self._segment_list.setToolTip(H.SEGMENT_LIST)
        self._segment_list.itemClicked.connect(self._on_segment_item_clicked)
        results_splitter.addWidget(self._segment_list)

        results_splitter.setStretchFactor(0, 3)   # contour list grows
        results_splitter.setStretchFactor(1, 0)   # middle controls stay compact
        results_splitter.setStretchFactor(2, 3)   # segment list grows
        results_splitter.setSizes([180, 300, 180])
        contour_lay.addWidget(results_splitter, 1)

        self._clear_analysis_btn = QPushButton("🗑 Clear Analysis Layers")
        self._clear_analysis_btn.setToolTip(H.CLEAR_ANALYSIS)
        contour_lay.addWidget(self._clear_analysis_btn)

        tabs.addTab(contour_w, "Contours")

        # --- Keypoint tab ---
        keypoint_w = QWidget()
        keypoint_lay = QVBoxLayout(keypoint_w)

        keypoint_lay.addWidget(self._label("Number of keypoints"))
        self._keypoint_count_spin = QSpinBox()
        self._keypoint_count_spin.setRange(1, 20)
        self._keypoint_count_spin.setValue(5)
        self._keypoint_count_spin.setToolTip(H.KEYPOINT_COUNT)
        keypoint_lay.addWidget(self._keypoint_count_spin)

        self._run_keypoint_btn = RunButton(
            "Find Keypoints + Ridgelines", accent="#8e44ad", accent_hover="#6c3483",
            ghost_bg="#f5eefa")
        self._run_keypoint_btn.setToolTip(H.RUN_KEYPOINT)
        keypoint_lay.addWidget(self._run_keypoint_btn)

        self._recommend_ponds_btn = RunButton(
            "Recommend Pond Sites", accent="#6c3483", accent_hover="#532567",
            ghost_bg="#f3eaf7")
        self._recommend_ponds_btn.setEnabled(False)
        self._recommend_ponds_btn.setToolTip(H.RECOMMEND_PONDS)
        keypoint_lay.addWidget(self._recommend_ponds_btn)

        # --- Yeomans keyline design ---
        keyline_grid = QGridLayout()
        keyline_grid.addWidget(self._label("Cultivation guides (each side)"), 0, 0)
        self._keyline_runs_spin = QSpinBox()
        self._keyline_runs_spin.setRange(0, 20)
        self._keyline_runs_spin.setValue(3)
        self._keyline_runs_spin.setToolTip(H.KEYLINE_RUNS)
        keyline_grid.addWidget(self._keyline_runs_spin, 0, 1)

        keyline_grid.addWidget(self._label("Guide spacing (m)"), 1, 0)
        self._keyline_spacing_spin = QDoubleSpinBox()
        self._keyline_spacing_spin.setRange(1.0, 100.0)
        self._keyline_spacing_spin.setValue(5.0)
        self._keyline_spacing_spin.setSuffix(" m")
        self._keyline_spacing_spin.setToolTip(H.KEYLINE_SPACING)
        keyline_grid.addWidget(self._keyline_spacing_spin, 1, 1)

        keyline_grid.addWidget(self._label("Guide grade (1 : N)"), 2, 0)
        self._keyline_grade_spin = QSpinBox()
        self._keyline_grade_spin.setRange(50, 5000)
        self._keyline_grade_spin.setValue(500)
        self._keyline_grade_spin.setSingleStep(50)
        self._keyline_grade_spin.setToolTip(H.KEYLINE_GRADE)
        keyline_grid.addWidget(self._keyline_grade_spin, 2, 1)
        keypoint_lay.addLayout(keyline_grid)

        self._run_keyline_btn = RunButton(
            "Generate Keylines", accent="#a0662a", accent_hover="#7d4e20",
            ghost_bg="#f7efe6")
        self._run_keyline_btn.setToolTip(H.RUN_KEYLINE)
        keypoint_lay.addWidget(self._run_keyline_btn)

        keyline_actions = QHBoxLayout()
        self._draw_keyline_btn = QPushButton("✏ Draw Keyline")
        self._draw_keyline_btn.setToolTip(H.DRAW_KEYLINE)
        keyline_actions.addWidget(self._draw_keyline_btn)
        self._convert_keyline_btn = QPushButton("Convert Keyline → Swale")
        self._convert_keyline_btn.setToolTip(H.CONVERT_KEYLINE)
        keyline_actions.addWidget(self._convert_keyline_btn)
        keypoint_lay.addLayout(keyline_actions)

        self._keypoint_status_lbl = self._label("", small=True)
        self._keypoint_status_lbl.setWordWrap(True)
        keypoint_lay.addWidget(self._keypoint_status_lbl)

        # Clickable result list — selecting a row zooms the canvas to that feature.
        self._keypoint_list = QListWidget()
        self._keypoint_list.setMaximumHeight(170)
        self._keypoint_list.setToolTip(H.KEYPOINT_LIST)
        self._keypoint_list.itemClicked.connect(self._on_keypoint_item_clicked)
        keypoint_lay.addWidget(self._keypoint_list)
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
        self._inflow_bands_btn.toggled.connect(self.show_inflow_bands_requested)
        self._segment_gradient_check.toggled.connect(self.show_segment_gradient_requested)
        self._clear_analysis_btn.clicked.connect(self.clear_analysis_requested)
        self._run_keypoint_btn.clicked.connect(self.run_keypoint_analysis_requested)
        self._recommend_ponds_btn.clicked.connect(self.recommend_ponds_requested)
        self._run_keyline_btn.clicked.connect(self.run_keyline_requested)
        self._draw_keyline_btn.clicked.connect(self.draw_keyline_requested)
        self._convert_keyline_btn.clicked.connect(self.convert_keyline_to_swale_requested)

    # ---------------------------------------------------------------- Section 4: Earthworks

    def _build_section_earthworks(self):
        lay = self._section("Earthwork Design")

        lay.addWidget(self._label("Soil type (for earthwork sizing)"))
        self._ew_soil_combo = QComboBox()
        for name in ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]:
            self._ew_soil_combo.addItem(name)
        self._ew_soil_combo.setCurrentText("Loam")
        self._ew_soil_combo.setToolTip(H.SITE_SOIL)
        lay.addWidget(self._ew_soil_combo)

        # Off by default: size on water the feature actually HOLDS, and treat
        # whatever soaks away as spare capacity rather than something to rely on.
        self._count_infiltration_check = QCheckBox("Count soakage toward capture")
        self._count_infiltration_check.setChecked(False)
        self._count_infiltration_check.setToolTip(H.COUNT_INFILTRATION)
        self._count_infiltration_check.toggled.connect(
            lambda _v: self.analysis_inputs_changed.emit())
        lay.addWidget(self._count_infiltration_check)

        # Felt-style tool menu (registry-driven) replaces the button grid.
        from terrainflow_assessment.qgis.widgets.tool_menu import EarthworkToolMenu
        self._tool_menu = EarthworkToolMenu()
        self._tool_menu.draw_swale_requested.connect(self.draw_swale_requested)
        self._tool_menu.draw_earthwork_requested.connect(self.draw_earthwork_requested)
        self._tool_menu.place_spillway_requested.connect(self.place_spillway_requested)
        self._tool_menu.connect_earthworks_requested.connect(
            self.connect_earthworks_requested)
        lay.addWidget(self._tool_menu)

        ew_actions = QHBoxLayout()
        self._ew_edit_btn = QPushButton("Edit")
        self._ew_reshape_btn = QPushButton("Reshape")
        self._ew_reshape_btn.setToolTip(H.RESHAPE_EARTHWORK)
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

        # The clearest answer to "why is my capture only 19%?" — grey is ground that
        # reaches no feature at all.
        self._catchment_layer_check = QCheckBox("Show: which earthwork catches what")
        self._catchment_layer_check.setToolTip(H.CATCHMENT_LAYER)
        lay.addWidget(self._catchment_layer_check)

        # Connections
        self._run_ew_btn.clicked.connect(self.run_earthworks_requested)
        self._ew_reshape_btn.clicked.connect(self.reshape_earthworks_requested)
        self._before_after_check.toggled.connect(self.before_after_toggled)
        self._catchment_layer_check.toggled.connect(self.toggle_catchment_layer_requested)

    # ---------------------------------------------------------------- Section 5: Live Assessment (network)

    def _build_section_live_assessment(self):
        lay = self._section("Live Assessment")

        # List / Flow — the same network read two ways. The list answers "what have
        # I got"; the chart answers "how does it connect", which a flat list cannot
        # show once features start spilling into one another.
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(0)
        mode_row.addStretch(1)
        self._network_mode_btns = {}
        modes = (("list", "List"), ("flow", "Flow"), ("elevation", "Elevation"))
        for i, (mode, label) in enumerate(modes):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(mode == "list")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            left = "" if i == 0 else "border-left: none;"
            if i == 0:
                radius = ("border-top-left-radius: 5px;"
                          " border-bottom-left-radius: 5px;")
            elif i == len(modes) - 1:
                radius = ("border-top-right-radius: 5px;"
                          " border-bottom-right-radius: 5px;")
            else:
                radius = ""
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #c6d1d3; " + left + radius
                + " background: transparent; color: #5f7176; font-size: 10.5px;"
                " padding: 2px 10px; } "
                "QPushButton:checked { background: #e9f3ee; color: #2e7d55;"
                " font-weight: 600; }"
            )
            btn.clicked.connect(lambda _c=False, m=mode: self._on_network_mode(m))
            self._network_mode_btns[mode] = btn
            mode_row.addWidget(btn)
        lay.addLayout(mode_row)

        from terrainflow_assessment.qgis.widgets.network_view import NetworkView
        self._network = NetworkView()
        # Selecting a card told nobody, so there was no way to tell which of five
        # swales the row referred to. Relay it so the map can highlight the feature.
        self._network.selection_changed.connect(self.earthwork_selected)
        lay.addWidget(self._network)

        # Per-area breakdown. One site-wide capture figure hides where the problem
        # is: 19% overall can be 80% on one catchment and nothing on the next, and
        # those need different work.
        self._area_subtotals_lbl = QLabel("")
        self._area_subtotals_lbl.setWordWrap(True)
        self._area_subtotals_lbl.setTextFormat(Qt.TextFormat.RichText)
        self._area_subtotals_lbl.setStyleSheet("font-size: 10.5px;")
        self._area_subtotals_lbl.setToolTip(H.AREA_SUBTOTALS)
        self._area_subtotals_lbl.setVisible(False)
        lay.addWidget(self._area_subtotals_lbl)

        # Totals + disclaimer line under the network (capacity / cut / fill).
        self._live_assessment_lbl = QLabel("")
        self._live_assessment_lbl.setWordWrap(True)
        self._live_assessment_lbl.setStyleSheet("font-size: 10.5px; color: #7f8c8d;")
        lay.addWidget(self._live_assessment_lbl)

    _SPILLWAY_SECTION = "Spillways"

    def _build_section_spillways(self):
        """Overflow review, and where the outflows and inlets get sited.

        Collapsed by default and in the Design stage rather than Verify: this is a
        design decision you make *before* burning, and it is the pass you do once the
        earthworks are laid out — not something to solve one feature at a time while
        drawing them. The header carries its own state so a collapsed section still says
        when something needs attention.
        """
        from terrainflow_assessment.qgis.widgets.spillway_table import SpillwayTable

        lay = self._section(self._SPILLWAY_SECTION, collapsed=True)
        self._spillway_empty = self._label(
            "Draw a swale, basin or dam — each one is sized for the overflow it needs, "
            "and updates as you add more.", small=True)
        self._spillway_empty.setWordWrap(True)
        self._spillway_empty.setStyleSheet("color: #8fa0a4; font-style: italic;")
        lay.addWidget(self._spillway_empty)

        self._spillway_table = SpillwayTable()
        self._spillway_table.setToolTip(H.SPILLWAY_REVIEW_TABLE)
        # Selecting here selects everywhere: activate_place_spillway and the action bar
        # both read the flow network's selection, so a row click has to move that too or
        # the two views quietly disagree about which feature is current.
        self._spillway_table.feature_selected.connect(self._on_spillway_row_selected)
        self._spillway_table.place_requested.connect(self.place_spillway_for_requested)
        self._spillway_table.edit_requested.connect(self.edit_earthwork_requested)
        lay.addWidget(self._spillway_table)

    def _on_spillway_row_selected(self, index):
        self.select_earthwork(index)

    def set_spillway_review(self, rows, context=None):
        """Populate the Spillways review (empty list clears it)."""
        self._spillway_table.set_rows(rows, context)
        self._spillway_empty.setVisible(not self._spillway_table.isVisible())
        self.set_section_note(self._SPILLWAY_SECTION, self._spillway_table.summary())

    def _wire_input_change_signals(self):
        """Emit analysis_inputs_changed on any storm/soil input change (drives the live
        analytical readout — the accumulation raster is storm-independent, so no re-analyse).
        Also keeps the header's storm chip and site name live.

        Inputs that participate in ``run_tag`` additionally mark the completed stages
        stale. Only the method combo used to, so changing the rainfall depth or the
        duration left a green tick and a "Verified" chip standing over rasters routed
        for a different storm — and the group name in the legend still carried the old
        one, which is the value the report re-derives.
        """
        for spin in (self._rainfall_spin, self._duration_spin, self._cn_spin):
            spin.valueChanged.connect(lambda *_: self._on_storm_input_changed())
            spin.valueChanged.connect(lambda *_: self._refresh_storm_chip())
        self._runoff_coeff_spin.valueChanged.connect(lambda *_: self._refresh_storm_chip())
        self._runoff_coeff_spin.valueChanged.connect(
            lambda *_: self._on_storm_input_changed())
        self._sizing_basis_combo.currentIndexChanged.connect(
            lambda *_: self._refresh_storm_chip())
        self._threshold_spin.valueChanged.connect(
            lambda *_: self._on_storm_input_changed())
        for combo in (self._soil_combo, self._moisture_combo, self._ew_soil_combo,
                      self._ground_condition_combo):
            combo.currentTextChanged.connect(lambda *_: self._on_storm_input_changed())
        self._site_name_edit.textChanged.connect(
            lambda text: self._head_site_lbl.setText(text.strip() or "Unnamed Site")
        )
        self._site_name_edit.editingFinished.connect(
            lambda: self.site_name_changed.emit(self.site_name))
        self._refresh_storm_chip()

    @property
    def runoff_basis_tag(self):
        """The one number the chosen method is calibrated by — ``C0.40`` / ``CN61``.

        Quoting a curve number while the method is Lancaster would advertise a
        value the assessment never reads and the user was never shown.
        """
        basis = self.sizing_basis
        if basis == "coefficient":
            return f"C{self._runoff_coeff_spin.value():.2f}"
        if basis == "runoff":
            return f"CN{self._cn_spin.value()}"
        return "all-rain"

    def _refresh_storm_chip(self):
        self._storm_chip.setText(
            f"{self._rainfall_spin.value():.0f} mm · "
            f"{self._duration_spin.value():.0f} h · {self.runoff_basis_tag} ▾"
        )

    # ---------------------------------------------------------------- Section 6: Simulation

    def _build_section_verification(self):
        """Design vs measured, per feature.

        The scorecard chip carries one site-wide Δ, which on its own conflates the
        freeboard allowance, what the terrain model made of the drawn section, and any
        actual burn error. This section separates them so a number that looks alarming
        can be read — and marks the rows where the last two columns are not a capacity.
        """
        from terrainflow_assessment.qgis.widgets.verification_table import (
            VerificationTable,
        )

        lay = self._section("Design vs Measured", collapsed=False)
        self._verification_empty = self._label(
            "Re-analyse with earthworks to compare the design against what the "
            "burned terrain actually holds.", small=True)
        self._verification_empty.setWordWrap(True)
        self._verification_empty.setStyleSheet("color: #8fa0a4; font-style: italic;")
        lay.addWidget(self._verification_empty)

        self._verification_table = VerificationTable()
        self._verification_table.setToolTip(H.VERIFICATION_TABLE)
        lay.addWidget(self._verification_table)

    def set_verification(self, result, cell_size_m=1.0):
        """Populate the per-feature verification table (None clears it)."""
        self._verification_table.set_result(result, cell_size_m=cell_size_m)
        # isHidden(), not isVisible(): a widget on a stage page that is not the active
        # one is not visible even though it was never hidden. Verification is computed
        # from the Design stage, so isVisible() answered False every time and the
        # "re-analyse to compare" placeholder stayed on screen above the populated
        # table until something else redrew the section.
        self._verification_empty.setVisible(self._verification_table.isHidden())

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
        # Disabled at click time, not when the first progress signal arrives. The
        # controller refuses a second run anyway, but a button that stays lit through
        # a minute of work reads as "nothing happened" and invites the second click.
        self._run_sim_btn.clicked.connect(self._on_run_sim_clicked)
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

        self._report_summary_lbl = QLabel(_REPORT_SUMMARY_IDLE)
        self._report_summary_lbl.setWordWrap(True)
        self._report_summary_lbl.setStyleSheet(
            "background: #ecf0f1; padding: 10px; border-radius: 6px; font-size: 11px;"
        )
        lay.addWidget(self._report_summary_lbl)

        self._export_btn = self._button("Export Report", "#1abc9c")
        self._export_btn.setEnabled(False)
        self._export_btn.setToolTip(H.EXPORT_REPORT)
        lay.addWidget(self._export_btn)

        self._export_btn.clicked.connect(self.export_report_requested)

    # ---------------------------------------------------------------- public setters

    def set_dem_info(self, info_str):
        self._dem_info_lbl.setText(info_str)
        self._head_info_lbl.setText(info_str or "Load a DEM to begin")

    def set_baseline_progress(self, pct, msg):
        self._run_baseline_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_baseline_complete(self, summary):
        self._baseline_has_run = True
        self._run_baseline_btn.set_done()
        self._baseline_results_lbl.setText(summary)
        self.mark_stage("baseline", "done")
        # Verify measured the previous baseline, so this one retires it. The
        # controller clears the matching state; without this the tick stays green
        # over a comparison that no longer has the run it was made against.
        if self._stepper.state("verify") == "done":
            self.mark_stage("verify", "stale")
        # Enable results tools after first successful baseline
        self._query_ponding_btn.setEnabled(True)
        self._toggle_slope_class_btn.setEnabled(True)
        self._toggle_slope_vectors_btn.setEnabled(True)
        self._toggle_throughflow_btn.setEnabled(True)
        self._throughflow_scale_combo.setEnabled(True)
        self._generate_contours_btn.setEnabled(True)

    def set_baseline_failed(self, summary):
        """A baseline that errored must not look like one that worked.

        The error handler used to call :meth:`set_baseline_complete`, which put
        the button in its Done state, ticked the stage green *and* switched on
        every downstream results tool — for a run that produced no data at all.
        The button goes back to idle so it reads as runnable, and the tools stay
        as they were: enabled only if an earlier run genuinely earned them.
        """
        self._run_baseline_btn.set_idle()
        self._baseline_results_lbl.setText(summary)
        self._mark_stage_failed("baseline")

    def set_usable_area_source(self, source):
        """Set the contour 'Usable area (clip)' selector (fires usable-area change)."""
        text = {"analysis": "Analysis Area", "earthworks": "Earthworks Area"}.get(
            source, "None")
        if self._usable_area_combo.currentText() != text:
            self._usable_area_combo.setCurrentText(text)

    def set_area_outflow(self, area_outflow, ponded_volume_m3=None):
        """Show how much water leaves each defined area after baseline.

        Each area reports two figures because they answer different questions: the
        **total** crossing the boundary, which is fixed for a given storm, and the part
        of it running through the exits currently drawn, which moves with the "Show
        exits above (L/s)" threshold. Only ever showing the second made a filtered
        subtotal look like a site total.
        """
        if not area_outflow:
            self._area_outflow_lbl.setVisible(False)
            return
        labels = {"site": "Site boundary", "analysis": "Analysis area",
                  "earthworks": "Earthworks area"}
        rows = []
        for key in ("site", "analysis", "earthworks"):
            d = area_outflow.get(key)
            if not d:
                continue
            shown = (
                f"{d['flow_ls']:,.1f} L/s ({d['volume_m3']:,.0f} m³) "
                f"via {d['n_exits']} shown exit{'s' if d['n_exits'] != 1 else ''}"
            )
            total = d.get("total_volume_m3")
            if total is None:
                rows.append(f"<b>{labels[key]}:</b> {shown}")
            else:
                rows.append(
                    f"<b>{labels[key]}:</b> {d.get('total_flow_ls', 0.0):,.1f} L/s "
                    f"({total:,.0f} m³ over event) in total"
                    f"<br><span style='color:#5b6b78;'>&nbsp;&nbsp;of which {shown}</span>"
                )
        if not rows:
            self._area_outflow_lbl.setVisible(False)
            return
        text = "Water leaving —<br>" + "<br>".join(rows)
        if ponded_volume_m3:
            text += (
                f"<br><b>Water captured:</b> {ponded_volume_m3:,.0f} m³ "
                f"ponding naturally on site"
            )
        self._area_outflow_lbl.setText(text)
        self._area_outflow_lbl.setToolTip(H.AREA_OUTFLOW)
        self._area_outflow_lbl.setVisible(True)

    def set_contour_progress(self, pct, msg):
        self._run_contour_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_contour_complete(self):
        self._run_contour_btn.set_done()
        self._top5_contours_btn.setEnabled(True)
        self._find_segments_btn.setEnabled(True)
        self._inflow_bands_btn.setEnabled(True)
        self.mark_stage("analysis", "done")

    def clear_analysis_ui(self):
        """Reset the Analysis-tab widgets after the controller wipes the layers."""
        self._contour_list.blockSignals(True)
        self._contour_list.clear()
        self._contour_list.blockSignals(False)
        self._segment_list.clear()
        self._keypoint_list.clear()
        self._keypoint_status_lbl.setText("")
        self._contour_legend.setText(H.CONTOUR_LEGEND_EMPTY)
        self._run_contour_btn.set_idle()
        self._find_segments_btn.set_idle()
        self._find_segments_btn.setEnabled(False)
        self._top5_contours_btn.setEnabled(False)
        self._inflow_bands_btn.setChecked(False)
        self._inflow_bands_btn.setEnabled(False)
        self._segment_gradient_check.setChecked(False)
        self._segment_gradient_check.setEnabled(False)
        self._run_keypoint_btn.set_idle()
        self._recommend_ponds_btn.set_idle()
        self._recommend_ponds_btn.setEnabled(False)
        self._run_keyline_btn.set_idle()

    def set_segment_progress(self, pct, msg):
        self._find_segments_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_segment_complete(self):
        self._find_segments_btn.set_done()
        self._segment_gradient_check.setEnabled(True)

    def set_keypoint_progress(self, pct, msg):
        self._run_keypoint_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_keypoint_complete(self, summary=""):
        self._run_keypoint_btn.set_done()
        self._keypoint_status_lbl.setText(summary)
        self._recommend_ponds_btn.setEnabled(bool(summary))

    def set_ponds_progress(self, pct, msg):
        self._recommend_ponds_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_ponds_complete(self, summary=""):
        self._recommend_ponds_btn.set_done()
        if summary:
            self._keypoint_status_lbl.setText(summary)

    def set_keyline_progress(self, pct, msg):
        self._run_keyline_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_keyline_complete(self, summary=""):
        self._run_keyline_btn.set_done()
        if summary:
            self._keypoint_status_lbl.setText(summary)

    def set_earthworks_progress(self, pct, msg):
        self._run_ew_btn.set_progress(pct, f"{msg} ({pct}%)")

    def set_earthworks_complete(self, summary=""):
        self._run_ew_btn.set_done()
        self._earthworks_results_lbl.setText(summary)
        self.mark_stage("verify", "done")

    def set_earthworks_failed(self, summary=""):
        """As :meth:`set_baseline_failed`, for Re-analyse with Earthworks.

        A failed burn leaves no verification, so ticking Verify green would
        claim a measurement that was never taken.
        """
        self._run_ew_btn.set_idle()
        self._earthworks_results_lbl.setText(summary)
        self._mark_stage_failed("verify")

    def set_verified_chip(self, text, fresh, tooltip=""):
        """Scorecard's verified-vs-design chip (Workbench)."""
        self._scorecard.set_verified(text, fresh, tooltip)

    def set_live_assessment(self, html):
        """Update the live analytical assessment readout (design-tier, no burn)."""
        self._live_assessment_lbl.setText(html)

    def update_scorecard(self, capture_pct, stored_m3, soaked_m3, leaves_m3,
                         natural_ponding_m3=None, capacity_note=None):
        """Feed the persistent scorecard with the live balance.

        *natural_ponding_m3* is context rather than part of the balance — it comes from
        a depression-fill of the bare terrain, not the event routing the band shows, so
        the scorecard prints it below the band instead of inside it. *capacity_note* is
        the same kind of thing: a sentence about whether storage is the binding
        constraint, which is a statement about the score rather than part of it.
        """
        self._scorecard.set_balance(capture_pct, stored_m3, soaked_m3, leaves_m3)
        self._scorecard.set_natural_ponding(natural_ponding_m3)
        self._scorecard.set_capacity_note(capacity_note)

    def scorecard_empty(self, message=None):
        if message:
            self._scorecard.show_empty(message)
        else:
            self._scorecard.show_empty()

    def scorecard_no_flow(self, capacity_m3):
        self._scorecard.show_no_flow(capacity_m3)

    def set_contour_results(self, contours):
        # Blocked while filling: addItem fires itemChanged per row, and fifty
        # "visibility changed" round-trips to the map is not what populating a
        # list means.
        self._contour_list.blockSignals(True)
        try:
            self._contour_list.clear()
            for i, feat in enumerate(contours[:50]):  # cap display at 50
                item = QListWidgetItem(feat.label)
                item.setCheckState(Qt.Checked)
                # Row → contour index, so a row survives the map round-trip even
                # if the list is ever sorted or filtered.
                item.setData(Qt.UserRole, i)
                self._contour_list.addItem(item)
        finally:
            self._contour_list.blockSignals(False)

    def set_contour_legend(self, breaks, unit="m³"):
        """Show the inflow bands the candidate contours are actually drawn in.

        *breaks* is the ascending boundary list from ``natural_breaks`` — the same
        numbers the map is banded on, so the legend cannot describe a scheme the
        canvas is not using.
        """
        if not breaks or len(breaks) < 3:
            self._contour_legend.setText(H.CONTOUR_LEGEND_EMPTY)
            return

        def _short(v):
            v = float(v)
            if v >= 1_000_000:
                return f"{v / 1_000_000:,.1f}M"
            if v >= 10_000:
                return f"{v / 1000:,.0f}k"
            if v >= 1000:
                return f"{v / 1000:,.1f}k"
            return f"{v:,.0f}"

        labels = [f"≤{_short(breaks[i + 1])}" for i in range(len(breaks) - 1)]
        self._contour_legend.setText(f"Inflow {unit}: " + _ramp_swatches(labels))

    def _on_contour_item_changed(self, _item):
        """A tick changed — hand the controller the rows the user has ruled out."""
        self.contour_visibility_changed.emit(self.unchecked_contour_rows())

    def unchecked_contour_rows(self):
        """Contour indices the user has unticked (see contour_visibility_changed)."""
        return [
            self._contour_list.item(i).data(Qt.UserRole)
            for i in range(self._contour_list.count())
            if self._contour_list.item(i).checkState() != Qt.Checked
        ]

    def _on_contour_rows_selected(self):
        if self._syncing_contour_rows:
            return
        self.contour_rows_selected.emit([
            item.data(Qt.UserRole) for item in self._contour_list.selectedItems()
        ])

    def select_contour_rows(self, indices):
        """Highlight the rows for *indices*, without echoing the selection back.

        Driven by the map side: picking a contour on the canvas should land on its
        row, and the row highlight is the same one clicking the row produces.
        """
        wanted = set(indices or [])
        self._syncing_contour_rows = True
        try:
            self._contour_list.clearSelection()
            first = None
            for i in range(self._contour_list.count()):
                item = self._contour_list.item(i)
                if item.data(Qt.UserRole) in wanted:
                    item.setSelected(True)
                    if first is None:
                        first = item
            if first is not None:
                self._contour_list.scrollToItem(first)
        finally:
            self._syncing_contour_rows = False

    def set_keypoint_results(self, items):
        """Populate the clickable keypoint/pond result list.

        *items* is a list of dicts: {label, x, y, kind}. Rows with x/y set to None
        are treated as non-clickable headers/summaries.
        """
        self._keypoint_list.clear()
        for it in items:
            row = QListWidgetItem(it.get("label", ""))
            x, y = it.get("x"), it.get("y")
            if x is not None and y is not None:
                row.setData(Qt.UserRole, (float(x), float(y)))
            else:
                # Header/summary row — not clickable, visually muted.
                row.setFlags(Qt.ItemIsEnabled)
                row.setForeground(Qt.gray)
            self._keypoint_list.addItem(row)

    def _on_keypoint_item_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data:
            self.keypoint_result_activated.emit(data[0], data[1])

    def set_segment_results(self, segments):
        """Populate the swale-segment list: ✓ holds / ⚠ needs overflow, storing
        each segment's geometry so a click highlights + zooms to that exact swale."""
        self._segment_list.clear()
        for seg in segments:
            mark = "⚠" if seg.capped else "✓"
            item = QListWidgetItem(f"{mark} {seg.label}")
            try:
                item.setData(Qt.UserRole, seg.geometry.wkt)
            except Exception:
                pass
            self._segment_list.addItem(item)

    def _on_segment_item_clicked(self, item):
        wkt = item.data(Qt.UserRole)
        if wkt:
            self.segment_activated.emit(wkt)

    def _show_slope_class_info(self):
        from qgis.PyQt.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Slope classes & earthworks")
        box.setTextFormat(Qt.RichText)
        box.setText(H.SLOPE_CLASS_INFO)
        box.setStandardButtons(QMessageBox.Ok)
        box.exec()

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

    def set_area_subtotals(self, rows):
        """Per-catchment capture, worst first — the row worth acting on leads."""
        if not rows:
            self._area_subtotals_lbl.setVisible(False)
            self._area_subtotals_lbl.setText("")
            return
        ordered = sorted(rows, key=lambda r: r["capture_pct"])
        lines = ['<div style="color:#5f7176; font-weight:600;">By catchment</div>']
        for row in ordered[:6]:
            pct = row["capture_pct"]
            colour = "#c0392b" if pct < 20 else "#b9770e" if pct < 50 else "#1e8449"
            lines.append(
                f'<div style="color:#5f7176;">{row["name"]} — '
                f'<span style="color:{colour}; font-weight:600;">{pct:.0f}%</span> held'
                f' · {row["runoff_m3"]:,.0f} m³ generated'
                f' · {row["exit_m3"]:,.0f} m³ leaves</div>'
            )
        if len(ordered) > 6:
            lines.append(f'<div style="color:#8fa0a4;">+{len(ordered) - 6} more</div>')
        self._area_subtotals_lbl.setText("".join(lines))
        self._area_subtotals_lbl.setVisible(True)

    def _on_network_mode(self, mode):
        for key, btn in self._network_mode_btns.items():
            btn.setChecked(key == mode)
        self._network.set_mode(mode)

    def set_network(self, nodes, edges, exit_m3):
        """Render the earthwork flow network (Live Assessment)."""
        self._network.set_network(nodes, edges, exit_m3)

    def get_selected_earthwork_index(self):
        return self._network.selected_index()

    def select_earthwork(self, index):
        """Make *index* the current feature everywhere, from any view.

        The flow network owns the selection that ``get_selected_earthwork_index`` reads,
        and both the action bar and spillway placement go through that. A second view
        that set only its own selection would send those to whichever feature the
        network happened to be on.
        """
        self._network.set_selected(index)
        self.earthwork_selected.emit(index)

    def _on_run_sim_clicked(self):
        self._run_sim_btn.setEnabled(False)
        self.run_simulation_requested.emit()

    def set_simulation_idle(self):
        """Re-arm the Run Simulation button — on success, failure or refusal."""
        self._run_sim_btn.setEnabled(True)
        self._sim_progress.setVisible(False)

    def set_simulation_progress(self, pct, msg):
        self._sim_progress.setVisible(True)
        self._sim_progress.setValue(pct)
        self._sim_progress.setFormat(f"{msg} ({pct}%)")

    def set_simulation_ready(self, result):
        self.set_simulation_idle()
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

    def clear_report_summary(self):
        """Drop the headline metrics — the design they described is gone."""
        self._report_summary_lbl.setText(_REPORT_SUMMARY_IDLE)

    def set_report_ready(self, ready):
        """Enable/disable the export button.

        The single owner of that button's state. It used to be switched on inside
        :meth:`set_report_summary`, whose only caller is the simulation — which is
        what made a report impossible without one, and what left the button lit
        after a design file was opened and every derived result cleared.
        """
        self._export_btn.setEnabled(bool(ready))

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

    def set_dem_layer(self, layer):
        """Select *layer* in the DEM picker, so the session's terrain is what is shown.

        Setting currentLayer re-fires ``dem_changed``, which is what rebuilds the burner,
        the DEM info label and the slope raster. Callers that must guarantee the rebuild
        emit the signal themselves as well — ``setLayer`` is a no-op when the layer is
        already current, and a silent no-op here means the burner keeps describing a
        different DEM from the one the session is analysing.
        """
        self._dem_combo.setLayer(layer)

    def set_area_layer(self, kind, layer):
        """Select *layer* in the picker named by *kind* ('boundary' | 'analysis' |
        'earthworks'). Setting currentLayer re-fires the matching *_changed signal."""
        combo = {
            "boundary": self._boundary_combo,
            "analysis": self._analysis_area_combo,
            "earthworks": self._earthworks_area_combo,
        }.get(kind)
        if combo is not None:
            combo.setLayer(layer)

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
    def ground_condition(self):
        """TR-55 hydrologic condition key — ``"good"`` / ``"fair"`` / ``"poor"``."""
        from terrainflow_assessment.modules.catchment import DEFAULT_GROUND_CONDITION
        return self._ground_condition_combo.currentData() or DEFAULT_GROUND_CONDITION

    def _refresh_cn_from_soil(self):
        """Re-fill the CN spinner from the current soil texture and ground condition."""
        from terrainflow_assessment.modules.catchment import SCSRunoff
        self._cn_spin.setValue(
            SCSRunoff.soil_reference_cn(self.soil_name, self.ground_condition))

    @property
    def cn(self):
        """Direct CN value from the spinner.

        Auto-filled from soil texture + ground condition, and overridable — a measured
        or land-use-derived CN beats any table lookup.
        """
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
    def exit_flow_ls(self):
        return self._exit_flow_spin.value()

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
    def top_n(self):
        return self._top_n_spin.value()

    @property
    def segment_rank_mode(self):
        return "inflow" if self._seg_rank_combo.currentIndex() == 1 else "catchment"

    @property
    def seg_max_slope_deg(self):
        """Segment slope limit in degrees, or None when the filter is off."""
        return self._seg_max_slope_spin.value() if self._seg_slope_check.isChecked() else None

    def _on_inflow_scale_changed(self, _index=0):
        """Re-render whichever inflow view is on — both read the same scale."""
        if self._inflow_bands_btn.isChecked():
            self.show_inflow_bands_requested.emit(True)
        if self._segment_gradient_check.isChecked():
            self.show_segment_gradient_requested.emit(True)

    @property
    def inflow_scale_mode(self):
        return {"Natural": "natural", "Log": "log", "Linear": "linear",
                "Quantile": "quantile"}.get(
            self._inflow_scale_combo.currentText(), "natural")

    @property
    def inflow_bands_active(self):
        """True while the contour inflow gradient is on the map."""
        return self._inflow_bands_btn.isChecked()

    @property
    def segment_gradient_active(self):
        """True while the peak-inflow overlay inside the swale segments is on."""
        return self._segment_gradient_check.isChecked()

    @property
    def throughflow_scale_mode(self):
        return {"Log": "log", "Linear": "linear", "Quantile": "quantile"}.get(
            self._throughflow_scale_combo.currentText(), "log")

    @property
    def throughflow_visible(self):
        return self._toggle_throughflow_btn.isChecked()

    @property
    def keypoint_count(self):
        return self._keypoint_count_spin.value()

    @property
    def keyline_runs(self):
        return self._keyline_runs_spin.value()

    @property
    def keyline_spacing_m(self):
        return self._keyline_spacing_spin.value()

    @property
    def keyline_cross_grade(self):
        n = self._keyline_grade_spin.value()
        return 1.0 / n if n else 0.0

    @property
    def earthwork_soil_name(self):
        return self._ew_soil_combo.currentText()

    def _on_surface_preset(self, _index=None):
        """Applying a surface preset sets C; 'Custom' leaves whatever is typed."""
        value = self._runoff_surface_combo.currentData()
        if value is not None:
            self._runoff_coeff_spin.blockSignals(True)
            self._runoff_coeff_spin.setValue(float(value))
            self._runoff_coeff_spin.blockSignals(False)
        self._on_basis_changed()

    def _sync_basis_controls(self):
        """Show only the inputs the chosen method actually reads.

        Hidden rather than greyed out: a disabled Curve Number still reads as a
        number this assessment depends on. Total rainfall asks for neither set —
        every millimetre that falls is routed, so there is nothing to calibrate.
        """
        basis = self.sizing_basis
        for key, rows in self._basis_rows.items():
            visible = key == basis
            for label, widget in rows:
                label.setVisible(visible)
                widget.setVisible(visible)

    def _on_storm_input_changed(self):
        """A storm/soil input moved: re-assess live, and mark what it invalidated.

        Same treatment `_on_basis_changed` already gave the method combo. The live
        readout is analytical and recomputes immediately; the *baseline rasters* do
        not, so leaving Baseline ticked green claims a run that described a different
        storm.
        """
        self.analysis_inputs_changed.emit()
        if self._baseline_has_run:
            self.mark_stage("baseline", "stale")
            self.mark_stage("analysis", "stale")

    def _on_basis_changed(self, _index=None):
        """The basis drives the baseline rasters too, so a run made under the old
        one no longer describes this storm — mark it stale rather than leaving a
        green tick over numbers that have silently changed meaning."""
        self._sync_basis_controls()
        self.analysis_inputs_changed.emit()
        if self._baseline_has_run:
            self.mark_stage("baseline", "stale")
            self.mark_stage("analysis", "stale")

    @property
    def sizing_basis(self):
        """'coefficient' (rational method), 'rainfall', or 'runoff' (SCS-CN)."""
        return ("coefficient", "rainfall", "runoff")[
            self._sizing_basis_combo.currentIndex()]

    @property
    def runoff_coefficient(self):
        """Rational-method runoff coefficient C, used when the basis is 'coefficient'."""
        return self._runoff_coeff_spin.value()

    @property
    def peak_intensity_mm_hr(self):
        """Peak design rainfall intensity — sizes overflow structures.

        Separate from the storm depth/duration because it cannot be derived from
        them: dividing depth by duration gives the event *average*, which for a
        24-hour design storm is a daily mean rather than anything a spillway will
        ever see.
        """
        return self._peak_intensity_spin.value()

    def set_peak_intensity(self, value):
        """Apply an intensity chosen in the comparison dialog."""
        self._peak_intensity_spin.setValue(float(value))

    @property
    def count_infiltration(self):
        """Whether soakage counts as capture, or is only reported as a buffer."""
        return self._count_infiltration_check.isChecked()

    # ------------------------------------------------------------------ Design file I/O

    # Input name → (widget attribute, kind). Kinds name the widget API, not the meaning:
    # "value" is any spin box, "text" a line edit, "checked" a check box, "combo_text" a
    # combo matched on item text, "combo_data" a combo matched on item *data* — for
    # combos whose stored key differs from the label the user reads.
    #
    # `routing` and `sizing_basis` are deliberately absent: their properties *derive* a
    # value from combo position rather than reading it back verbatim, so restoring them
    # means inverting that mapping. Both are handled explicitly in apply_inputs.
    _INPUT_WIDGETS = {
        "site_name": ("_site_name_edit", "text"),
        "rainfall_mm": ("_rainfall_spin", "value"),
        "duration_hr": ("_duration_spin", "value"),
        "soil_name": ("_soil_combo", "combo_text"),
        "ground_condition": ("_ground_condition_combo", "combo_data"),
        "cn": ("_cn_spin", "value"),
        "moisture": ("_moisture_combo", "combo_text"),
        "stream_threshold_ha": ("_threshold_spin", "value"),
        "exit_flow_ls": ("_exit_flow_spin", "value"),
        "runoff_coefficient": ("_runoff_coeff_spin", "value"),
        "earthwork_soil_name": ("_ew_soil_combo", "combo_text"),
        "peak_intensity_mm_hr": ("_peak_intensity_spin", "value"),
        "count_infiltration": ("_count_infiltration_check", "checked"),
        "contour_interval_m": ("_contour_interval_spin", "value"),
        "simple_contour_interval_m": ("_simple_contour_interval_spin", "value"),
        "max_slope_deg": ("_max_slope_spin", "value"),
        "min_contour_length_m": ("_min_contour_length_spin", "value"),
        "min_catchment_ha": ("_min_catchment_ha_spin", "value"),
        "swale_depth_m": ("_swale_depth_spin", "value"),
        "swale_width_m": ("_swale_width_spin", "value"),
    }

    def collect_inputs(self):
        """Every analysis input, keyed as ``project_io.INPUT_FIELDS`` expects.

        Reads through the existing public properties rather than the widgets, so there is
        exactly one definition of what each input *means* and this cannot drift from what
        an analysis run actually receives.
        """
        return {name: getattr(self, name) for name in INPUT_FIELDS}

    def apply_inputs(self, values):
        """Push a restored input set back into the widgets.

        Signals are blocked across the whole apply and ``analysis_inputs_changed`` is
        emitted once at the end. Setting twenty widgets individually would otherwise fire
        the live re-assessment twenty times, each on a half-restored input set — slow, and
        briefly scoring the design against a storm that is part old and part new.
        """
        values = normalise_inputs(values)
        widgets = [w for w in (
            [getattr(self, attr, None) for attr, _kind in self._INPUT_WIDGETS.values()]
            + [self._routing_combo, self._sizing_basis_combo]
        ) if w is not None]

        try:
            for widget in widgets:
                widget.blockSignals(True)

            for name, (attr, kind) in self._INPUT_WIDGETS.items():
                widget = getattr(self, attr, None)
                if widget is None:
                    continue
                value = values[name]
                if kind == "value":
                    widget.setValue(value)
                elif kind == "text":
                    widget.setText(value)
                elif kind == "checked":
                    widget.setChecked(bool(value))
                elif kind == "combo_text":
                    # A soil or moisture label the current build doesn't offer leaves the
                    # combo alone: keeping a valid selection beats blanking it to nothing.
                    index = widget.findText(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif kind == "combo_data":
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)

            self._apply_routing(values["routing"])
            self._apply_sizing_basis(values["sizing_basis"])
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        # The basis governs whether the coefficient controls are live, and it was set
        # with signals blocked, so its own handler never ran.
        self._sync_basis_controls()
        self.analysis_inputs_changed.emit()

    def _apply_routing(self, routing):
        """Select the combo entry the :attr:`routing` property would read back as *routing*.

        Matched on item text the same way the getter is, rather than by index, so
        relabelling or reordering the combo cannot silently invert the choice.
        """
        want_d8 = routing == "d8"
        for index in range(self._routing_combo.count()):
            if ("D8" in self._routing_combo.itemText(index)) == want_d8:
                self._routing_combo.setCurrentIndex(index)
                return

    def _apply_sizing_basis(self, basis):
        """Select the combo position matching *basis*, whose order SIZING_BASIS_VALUES mirrors."""
        try:
            index = SIZING_BASIS_VALUES.index(basis)
        except ValueError:
            return
        if index < self._sizing_basis_combo.count():
            self._sizing_basis_combo.setCurrentIndex(index)

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
