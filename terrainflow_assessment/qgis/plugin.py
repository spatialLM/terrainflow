"""
qgis/plugin.py — TerrainFlowAssessmentPlugin

Thin dispatcher: creates PluginState, instantiates all five controllers,
and wires panel signals to controller methods.  No analysis logic lives here.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QAction

from terrainflow_assessment.modules.earthwork_design import EarthworkManager
from terrainflow_assessment.panel import AssessmentPanel
from terrainflow_assessment.qgis.adapters.project import ProjectAdapter
from terrainflow_assessment.qgis.controllers._state import PluginState
from terrainflow_assessment.qgis.controllers.baseline import BaselineController
from terrainflow_assessment.qgis.controllers.contour import ContourController
from terrainflow_assessment.qgis.controllers.earthworks import EarthworksController
from terrainflow_assessment.qgis.controllers.reporting import ReportingController
from terrainflow_assessment.qgis.controllers.simulation import SimulationController


class TerrainFlowAssessmentPlugin:
    """
    QGIS plugin entry point (new architecture).

    Registered via classFactory() in __init__.py.
    """

    def __init__(self, iface):
        self._iface = iface
        self._canvas = iface.mapCanvas()
        self._action = None
        self.panel = None

        self._project = ProjectAdapter()
        self._state = PluginState()
        self._state.earthwork_manager = EarthworkManager()

    # ---------------------------------------------------------------- QGIS lifecycle

    def initGui(self):
        self._action = QAction("TerrainFlow Assessment", self._iface.mainWindow())
        self._action.triggered.connect(self.toggle_panel)
        self._iface.addToolBarIcon(self._action)
        self._iface.addPluginToMenu("TerrainFlow", self._action)
        self._create_panel()

    def unload(self):
        self._iface.removeToolBarIcon(self._action)
        self._iface.removePluginMenu("TerrainFlow", self._action)
        if self.panel:
            self._iface.removeDockWidget(self.panel)
            self.panel = None

    def toggle_panel(self):
        if self.panel:
            self.panel.setVisible(not self.panel.isVisible())

    # ---------------------------------------------------------------- Panel + controllers

    def _create_panel(self):
        self.panel = AssessmentPanel(self._iface.mainWindow())
        self._iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.panel)

        args = (self._state, self.panel, self._project, self._iface, self._canvas)

        self._baseline = BaselineController(*args)
        self._contour = ContourController(*args)
        self._earthworks = EarthworksController(*args)
        self._simulation = SimulationController(*args)
        self._reporting = ReportingController(*args)

        self._wire_signals()

    def _wire_signals(self):
        p = self.panel
        bl = self._baseline
        ct = self._contour
        ew = self._earthworks
        sim = self._simulation
        rep = self._reporting

        # DEM / boundary
        p.dem_changed.connect(bl.on_dem_changed)
        p.boundary_changed.connect(bl.on_boundary_changed)
        p.analysis_area_changed.connect(bl.on_analysis_area_changed)
        p.earthworks_area_changed.connect(bl.on_earthworks_area_changed)

        # Baseline
        p.run_baseline_requested.connect(bl.run_baseline)
        p.before_after_toggled.connect(bl.toggle_before_after)

        # Slope / ponding query
        p.query_ponding_requested.connect(ew.activate_ponding_query)
        p.toggle_slope_class_requested.connect(ew.toggle_slope_class)
        p.toggle_slope_arrows_requested.connect(ew.toggle_slope_arrows)

        # Contour
        p.run_contour_analysis_requested.connect(ct.run_contour_analysis)
        p.select_top5_contours_requested.connect(ct.select_top5_contours)
        p.find_segments_requested.connect(ct.run_segment_analysis)
        p.generate_simple_contours_requested.connect(ct.generate_simple_contours)
        p.run_keypoint_analysis_requested.connect(ct.run_keypoint_analysis)
        p.recommend_ponds_requested.connect(ct.run_recommend_ponds)

        # Earthworks drawing
        p.draw_swale_requested.connect(ew.activate_draw_swale)
        p.draw_berm_requested.connect(lambda: ew.activate_draw_line("berm"))
        p.draw_basin_requested.connect(ew.activate_draw_basin)
        p.draw_dam_requested.connect(lambda: ew.activate_draw_line("dam"))
        p.draw_diversion_requested.connect(lambda: ew.activate_draw_line("diversion"))
        p.usable_area_source_changed.connect(ew.on_usable_area_source_changed)
        p.run_earthworks_requested.connect(ew.run_with_earthworks)

        # Earthworks table buttons
        p._ew_edit_btn.clicked.connect(ew.edit_selected_earthwork)
        p._ew_delete_btn.clicked.connect(ew.delete_selected_earthwork)
        p._ew_toggle_btn.clicked.connect(ew.toggle_selected_earthwork)

        # Simulation
        p.run_simulation_requested.connect(sim.run_simulation)
        p.sim_frame_changed.connect(sim.show_sim_frame)
        p.sim_play_toggled.connect(sim.on_sim_play_toggled)

        # Reporting
        p.export_report_requested.connect(rep.export_report)
