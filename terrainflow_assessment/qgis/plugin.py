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
from terrainflow_assessment.qgis.controllers.design_file import DesignFileController
from terrainflow_assessment.qgis.controllers.earthworks import EarthworksController
from terrainflow_assessment.qgis.controllers.reporting import ReportingController
from terrainflow_assessment.qgis.controllers.simulation import SimulationController
from terrainflow_assessment.qgis.workers._lifecycle import join_workers


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
        """Take the plugin apart in the one order that is safe.

        Reloading from the Plugin Manager calls this and then builds a second
        plugin against the same QGIS. Anything left attached — a running thread, a
        map tool, a signal on the canvas or the layer tree, the panel itself —
        outlives the object it calls back into, and the failure shows up later as
        a click doing nothing or QGIS aborting outright.

        The order matters and is the reason this reads as a list:

        1. stop the workers *first*, because everything below removes things they
           are using, and step 5 deletes the directory they are writing into;
        2. take our map tools off the canvas before the controllers holding them
           go, or the next click on the map reaches a dead controller;
        3. disconnect from objects QGIS owns and we do not — the canvas, the layer
           tree, the project;
        4. drop the panel and its ~60 signal connections;
        5. only then remove the scratch directory.
        """
        self._iface.removeToolBarIcon(self._action)
        self._iface.removePluginMenu("TerrainFlow", self._action)

        # 1. Workers. Stragglers are workers that ignored the abort and are still
        #    running; the directory removal below is skipped rather than racing them.
        stragglers = join_workers(self._state)

        # 2 + 3. Per-controller teardown: active map tool off the canvas, and the
        #        canvas/layer-tree connections undone.
        for name in ("_baseline", "_contour", "_earthworks",
                     "_simulation", "_reporting", "_design_file"):
            controller = getattr(self, name, None)
            teardown = getattr(controller, "teardown", None)
            if teardown is None:
                continue
            try:
                teardown()
            except Exception:
                # One controller failing to tidy up must not strand the rest.
                pass

        if getattr(self, "_persistence_wired", False):
            try:
                instance = self._project.instance()
                instance.writeProject.disconnect(self._on_write_project)
                instance.readProject.disconnect(self._on_read_project)
            except Exception:
                pass
            self._persistence_wired = False

        # 4. The panel. `removeDockWidget` only un-docks it — the widget, and every
        #    connection from its signals into the controllers, stays alive and
        #    reachable without the `deleteLater`.
        if self.panel:
            self._iface.removeDockWidget(self.panel)
            try:
                self.panel.deleteLater()
            except RuntimeError:
                pass
            self.panel = None

        # 5. The per-session scratch directory is never reused; leaving it behind
        #    grows the temp folder by a DEM's worth of rasters every run. Skipped
        #    while anything is still writing into it — a leaked temp directory is a
        #    far smaller problem than pulling files out from under a live thread.
        if stragglers:
            return
        try:
            import shutil
            shutil.rmtree(self._state.output_dir, ignore_errors=True)
        except Exception:
            pass

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
        self._design_file = DesignFileController(*args)

        self._wire_signals()
        self._wire_project_persistence()

    def _wire_project_persistence(self):
        """Save/restore the earthwork design with the QGIS project.

        Earthworks were previously session-only — closing QGIS discarded the whole
        design with no warning and no way to recover it.
        """
        try:
            instance = self._project.instance()
            instance.writeProject.connect(self._on_write_project)
            instance.readProject.connect(self._on_read_project)
            self._persistence_wired = True
        except Exception as exc:
            self._persistence_wired = False
            print(f"TerrainFlow Assessment — project persistence unavailable: {exc}")

    def _on_write_project(self, *_args):
        self._earthworks.save_to_project()
        self._earthworks.save_rainfall_data()

    def _on_read_project(self, *_args):
        # Rainfall data first: the earthworks' time-of-concentration figures read the
        # 2-year depth from it, so restoring it afterwards would leave the first
        # assessment computed without it.
        self._earthworks.load_rainfall_data()
        self._earthworks.load_from_project()

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
        p.site_name_changed.connect(bl.on_site_name_changed)
        p.draw_boundary_requested.connect(lambda: bl.draw_area("boundary"))
        p.draw_analysis_area_requested.connect(lambda: bl.draw_area("analysis"))
        p.draw_earthworks_area_requested.connect(lambda: bl.draw_area("earthworks"))

        # Baseline
        p.run_baseline_requested.connect(bl.run_baseline)
        p.before_after_toggled.connect(bl.toggle_before_after)

        # Slope / ponding query
        p.query_ponding_requested.connect(ew.activate_ponding_query)
        p.toggle_slope_class_requested.connect(ew.toggle_slope_class)
        p.toggle_slope_vectors_requested.connect(ew.toggle_slope_vectors)

        # Contour
        p.run_contour_analysis_requested.connect(ct.run_contour_analysis)
        p.select_top5_contours_requested.connect(ct.select_top5_contours)
        p.find_segments_requested.connect(ct.run_segment_analysis)
        p.show_inflow_bands_requested.connect(ct.show_inflow_bands)
        p.show_segment_gradient_requested.connect(ct.show_segment_gradient)
        p.contour_visibility_changed.connect(ct.set_contour_visibility)
        p.contour_rows_selected.connect(ct.highlight_contour_rows)
        p.clear_analysis_requested.connect(ct.clear_analysis)
        p.generate_simple_contours_requested.connect(ct.generate_simple_contours)
        p.run_keypoint_analysis_requested.connect(ct.run_keypoint_analysis)
        p.recommend_ponds_requested.connect(ct.run_recommend_ponds)
        p.keypoint_result_activated.connect(ct.zoom_to_point)
        p.segment_activated.connect(ct.highlight_segment)
        p.run_keyline_requested.connect(ct.run_keyline_analysis)
        p.draw_keyline_requested.connect(ct.activate_draw_keyline)
        p.convert_keyline_to_swale_requested.connect(ew.create_swale_from_keyline)

        # Earthworks drawing — registry-driven: the panel emits the type key and
        # the controller resolves the right map tool from the type's geometry.
        p.draw_swale_requested.connect(ew.activate_draw_swale)
        p.draw_earthwork_requested.connect(ew.activate_draw_earthwork)
        p.usable_area_source_changed.connect(ew.on_usable_area_source_changed)
        p.run_earthworks_requested.connect(ew.run_with_earthworks)
        p.reshape_earthworks_requested.connect(ew.activate_edit_earthwork_vertices)

        # Overflow routing: where a feature spills, and what it spills into.
        p.place_spillway_requested.connect(ew.activate_place_spillway)
        p.place_spillway_for_requested.connect(ew.place_spillway_for)
        p.edit_earthwork_requested.connect(ew.edit_earthwork_at)
        p.connect_earthworks_requested.connect(ew.activate_connect_earthworks)
        p.choose_design_intensity_requested.connect(ew.choose_design_intensity)
        p.edit_rainfall_data_requested.connect(ew.edit_rainfall_data)
        p.earthwork_selected.connect(ew.highlight_selected_earthwork)

        # Earthworks table buttons
        p._ew_edit_btn.clicked.connect(ew.edit_selected_earthwork)
        p._ew_delete_btn.clicked.connect(ew.delete_selected_earthwork)
        p._ew_toggle_btn.clicked.connect(ew.toggle_selected_earthwork)

        # Direct-catchment layer (one colour per earthwork)
        p.toggle_catchment_layer_requested.connect(ew.toggle_catchment_layer)

        # Throughflow gradient (blue per-cell water volume)
        p.toggle_throughflow_requested.connect(bl.set_throughflow_visible)
        p.throughflow_scale_changed.connect(bl.set_throughflow_scale)

        # Live analytical assessment — recompute when storm/soil inputs change
        p.analysis_inputs_changed.connect(ew._recompute_live_assessment)

        # A successful baseline re-scores whatever earthworks already exist. Normally
        # they are drawn after a baseline, but a restored design file reverses that order,
        # leaving features with no catchment labels until terrain results arrive.
        bl.baseline_finished.connect(ew._recompute_live_assessment)

        # Portable design files. The design-file controller hands restored payloads to
        # whichever controller owns them rather than calling across directly.
        df = self._design_file
        p.save_design_requested.connect(df.save_design)
        p.open_design_requested.connect(df.open_design)
        df.earthworks_payload_ready.connect(ew.restore_earthworks_from_json)
        df.idf_payload_ready.connect(ew.restore_rainfall_from_json)
        df.baseline_rerun_requested.connect(bl.run_baseline)

        # Simulation
        p.run_simulation_requested.connect(sim.run_simulation)
        p.sim_frame_changed.connect(sim.show_sim_frame)
        p.sim_play_toggled.connect(sim.on_sim_play_toggled)

        # Reporting. The selection highlight is a rubber band owned by the
        # earthworks controller, so it is outside every render flag and outside
        # anything the reporting controller can reach — clear it first, in
        # connection order, or the operator's last click prints as a green line
        # through one swale on the design map.
        p.export_report_requested.connect(ew.clear_selection_highlight)
        p.export_report_requested.connect(rep.export_report)
