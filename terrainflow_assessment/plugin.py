"""
plugin.py — TerrainFlow Assessment plugin controller.

Wires the AssessmentPanel UI to the analysis modules.  All analysis runs in
background QThreads (AnalysisWorker, SimulationWorker) to keep QGIS responsive.
"""

import os
import tempfile
import traceback

from qgis.PyQt.QtCore import Qt, QTimer, QObject
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QMessageBox, QDockWidget
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsWkbTypes, QgsField, QgsFields, QgsCoordinateReferenceSystem,
    QgsSymbol, QgsSingleSymbolRenderer, QgsRendererRange,
    QgsColorRampShader,
    QgsRasterShader, QgsSingleBandPseudoColorRenderer,
    QgsRasterBandStats, QgsMarkerSymbol, QgsSymbolLayer, QgsProperty,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat,
)
from qgis.gui import QgsMapCanvas

from .panel import AssessmentPanel
from .modules.dem_loader import load_dem, compute_slope_raster
from .modules.catchment import SCSRunoff
from .modules.flow_analysis import AnalysisWorker
from .modules.earthwork_design import (
    Earthwork, EarthworkManager, DEMBurner,
    calculate_capacity, calculate_cut_volume, calculate_fill_volume,
)
from .modules.swale_design import (
    SOIL_REFERENCE, recommend_swale_length, sample_peak_inflow,
    contour_to_swale_geometry, snap_point_to_contour_elevation,
)
from .map_tools.draw_line_tool import DrawLineTool
from .map_tools.draw_polygon_tool import DrawPolygonTool
from .map_tools.place_point_tool import PlacePointTool
from .map_tools.ponding_query_tool import PondingQueryTool
from .map_tools.select_contour_tool import SelectContourTool
from .map_tools.contour_segment_tool import ContourSegmentTool


def _crs_label(crs):
    """Return a human-readable CRS string from a rasterio.crs.CRS."""
    if crs is None:
        return "unknown CRS"
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    return crs.to_string()


class TerrainFlowAssessment:
    """
    Main plugin class — registered with QGIS via classFactory().

    Manages:
      - Panel creation and signal wiring
      - DEM/boundary loading
      - Baseline analysis (AnalysisWorker)
      - Contour analysis
      - Keypoint analysis
      - Earthwork management and DEM burning
      - Fill simulation (SimulationWorker)
      - Before/after reporting and HTML export
    """

    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.panel = None
        self._action = None

        # Analysis state
        self._dem_path = None
        self._dem_info = None
        self._boundary_path = None
        self._analysis_area_path = None
        self._earthworks_area_path = None
        self._modified_dem_path = None
        self._slope_raster_path = None
        self._output_dir = tempfile.mkdtemp(prefix="tfa_")

        # Raster paths from most recent analysis runs
        self._baseline_result = None     # dict from AnalysisWorker.finished
        self._earthworks_result = None

        # Simulation
        self._sim_result = None
        self._sim_global_max_inc = 1.0
        self._sim_global_max_cum = 1.0
        self._sim_timer = QTimer()
        self._sim_timer.timeout.connect(self._advance_sim_frame)
        self._sim_fill_layer = None     # live fill-status point layer
        self._sim_ew_centroids = {}     # store_name → QgsPointXY
        self._sim_ponding_capacity = None   # numpy array — max ponding depth
        self._sim_ponding_masks = {}        # store_name → bool mask array
        self._sim_ponding_meta = None       # rasterio meta dict for temp rasters
        self._sim_ponding_frame_layer = None  # live partial-fill raster layer
        self._sim_ponding_outline_layer = None  # static full-capacity outline

        # Earthworks
        self._burner = None
        self._earthwork_manager = EarthworkManager()
        self._ew_layers = {}    # type → QgsVectorLayer
        self._ew_group = None   # QgsLayerTreeGroup
        self._ponding_raster_path = None

        # Contour analysis
        self._contour_features = []
        self._usable_polygon = None
        self._contour_layer = None
        self._top5_layer = None
        self._segment_layer = None
        self._simple_contour_layer = None

        # Reporting
        self._baseline_report = None
        self._post_report = None

        # QGIS layer IDs for before/after toggling
        self._baseline_layer_ids = []
        self._earthworks_layer_ids = []

        # Toggleable result layer IDs
        self._accumulation_layer_id = None
        self._slope_class_layer_id = None
        self._slope_arrows_layer_id = None

        # Workers (kept to avoid GC)
        self._analysis_worker = None
        self._sim_worker = None
        self._keypoint_worker = None

    # ---------------------------------------------------------------- QGIS lifecycle

    def initGui(self):
        self._action = QAction("TerrainFlow Assessment", self.iface.mainWindow())
        self._action.triggered.connect(self.toggle_panel)
        self.iface.addToolBarIcon(self._action)
        self.iface.addPluginToMenu("TerrainFlow", self._action)
        self._create_panel()

    def unload(self):
        self.iface.removeToolBarIcon(self._action)
        self.iface.removePluginMenu("TerrainFlow", self._action)
        if self.panel:
            self.iface.removeDockWidget(self.panel)
            self.panel = None

    def toggle_panel(self):
        if self.panel:
            self.panel.setVisible(not self.panel.isVisible())

    # ---------------------------------------------------------------- Panel setup

    def _create_panel(self):
        self.panel = AssessmentPanel(self.iface.mainWindow())
        self.iface.addDockWidget(Qt.RightDockWidgetArea, self.panel)

        # Wire signals
        self.panel.dem_changed.connect(self._on_dem_changed)
        self.panel.boundary_changed.connect(self._on_boundary_changed)
        self.panel.analysis_area_changed.connect(self._on_analysis_area_changed)
        self.panel.earthworks_area_changed.connect(self._on_earthworks_area_changed)
        self.panel.run_baseline_requested.connect(self._run_baseline)
        self.panel.query_ponding_requested.connect(self._activate_ponding_query)
        self.panel.toggle_slope_class_requested.connect(self._toggle_slope_class)
        self.panel.toggle_slope_arrows_requested.connect(self._toggle_slope_arrows)
        self.panel.run_contour_analysis_requested.connect(self._run_contour_analysis)
        self.panel.select_top5_contours_requested.connect(self._select_top5_contours)
        self.panel.find_segments_requested.connect(self._run_segment_analysis)
        self.panel.generate_simple_contours_requested.connect(self._generate_simple_contours)
        self.panel.run_keypoint_analysis_requested.connect(self._run_keypoint_analysis)
        self.panel.recommend_ponds_requested.connect(self._run_recommend_ponds)
        self.panel.draw_swale_requested.connect(self._activate_draw_swale)
        self.panel.draw_berm_requested.connect(lambda: self._activate_draw_line("berm"))
        self.panel.draw_basin_requested.connect(self._activate_draw_basin)
        self.panel.draw_dam_requested.connect(lambda: self._activate_draw_line("dam"))
        self.panel.draw_diversion_requested.connect(
            lambda: self._activate_draw_line("diversion"))
        self.panel.usable_area_source_changed.connect(self._on_usable_area_source_changed)
        self.panel.run_earthworks_requested.connect(self._run_with_earthworks)
        self.panel.before_after_toggled.connect(self._toggle_before_after)
        self.panel._ew_edit_btn.clicked.connect(self._edit_selected_earthwork)
        self.panel._ew_delete_btn.clicked.connect(self._delete_selected_earthwork)
        self.panel._ew_toggle_btn.clicked.connect(self._toggle_selected_earthwork)
        self.panel.run_simulation_requested.connect(self._run_simulation)
        self.panel.sim_frame_changed.connect(self._show_sim_frame)
        self.panel.sim_play_toggled.connect(self._on_sim_play_toggled)
        self.panel.export_report_requested.connect(self._export_report)

    # ---------------------------------------------------------------- DEM & boundary

    def _on_dem_changed(self, layer):
        if layer is None:
            self._dem_path = None
            self._dem_info = None
            return
        try:
            path = layer.source()
            info = load_dem(path)
            self._dem_path = path
            self._dem_info = info
            self._burner = DEMBurner(path)
            self.panel.set_dem_info(
                f"Cell: {info.cell_size_m:.2f} m | Area: {info.area_ha:.1f} ha | {_crs_label(info.crs)}"
            )
            # Pre-compute slope raster for drawing tools
            slope_path = os.path.join(self._output_dir, "slope.tif")
            compute_slope_raster(path, slope_path)
            self._slope_raster_path = slope_path
        except Exception as exc:
            self.iface.messageBar().pushWarning("TerrainFlow Assessment",
                                                 f"DEM load error: {exc}")

    def _on_boundary_changed(self, layer):
        if layer is None:
            self._boundary_path = None
            return
        try:
            self._boundary_path = self._layer_to_path(layer)
        except Exception:
            self._boundary_path = None

    def _on_analysis_area_changed(self, layer):
        self._analysis_area_path = self._layer_to_path(layer) if layer else None

    def _on_earthworks_area_changed(self, layer):
        self._earthworks_area_path = self._layer_to_path(layer) if layer else None

    def _layer_to_path(self, layer):
        if layer is None:
            return None
        src = layer.source()
        if os.path.exists(src.split("|")[0]):
            return src.split("|")[0]
        # Memory layer — write to temp file
        import tempfile
        path = tempfile.mktemp(suffix=".gpkg")
        from qgis.core import QgsVectorFileWriter
        QgsVectorFileWriter.writeAsVectorFormat(layer, path, "UTF-8")
        return path

    # ---------------------------------------------------------------- Baseline analysis

    def _run_baseline(self):
        if not self._dem_path:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Please select a DEM first."
            )
            return

        cell_area_m2 = self._dem_info.cell_area_m2 if self._dem_info else 1.0
        # Convert ha threshold to cell count
        ha_threshold = self.panel.stream_threshold_ha
        threshold_cells = int(ha_threshold * 10_000 / cell_area_m2) if cell_area_m2 > 0 else 1000

        self._analysis_worker = AnalysisWorker(
            dem_path=self._dem_path,
            output_dir=self._output_dir,
            stream_threshold=threshold_cells,
            cn=self.panel.cn,
            moisture=self.panel.moisture,
            rainfall_mm=self.panel.rainfall_mm,
            duration_hours=self.panel.duration_hr,
            boundary_path=self._boundary_path,
            label="baseline",
            run_catchments=True,
            threshold_mode="cells",
            routing=self.panel.routing,
        )
        self._analysis_worker.progress.connect(self.panel.set_baseline_progress)
        self._analysis_worker.finished.connect(self._on_baseline_complete)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def _on_baseline_complete(self, result):
        self._baseline_result = result
        self._load_result_layers(result, is_earthworks=False)

        # Populate baseline report
        from .modules.reporting import BaselineReport
        catchment_ha = result.get("catchment_area_m2", 0) / 10_000.0
        self._baseline_report = BaselineReport(
            site_name=self.panel.site_name,
            dem_path=self._dem_path,
            crs=_crs_label(self._dem_info.crs) if self._dem_info else "",
            cell_size_m=self._dem_info.cell_size_m if self._dem_info else 1.0,
            catchment_area_ha=catchment_ha,
            rainfall_mm=self.panel.rainfall_mm,
            duration_hr=self.panel.duration_hr,
            cn=result.get("effective_cn", 70),
            runoff_mm=result.get("runoff_mm", 0),
            total_runoff_m3=result.get("runoff_volume_m3", 0),
            exit_volume_m3=sum(ep.get("volume_m3", 0) for ep in result.get("exit_points", [])),
            exit_points=result.get("exit_points", []),
        )

        summary = (
            f"Runoff: {result.get('runoff_mm', 0):.1f} mm | "
            f"CN: {result.get('effective_cn', 0):.0f} | "
            f"Exit points: {len(result.get('exit_points', []))}"
        )
        self.panel.set_baseline_complete(summary)

    def _on_analysis_error(self, tb):
        self.panel.set_baseline_complete("Analysis failed — see Python console for details.")
        print("TerrainFlow Assessment — Analysis error:\n" + tb)
        self.iface.messageBar().pushCritical("TerrainFlow Assessment",
                                              "Analysis failed. See Python console.")

    # ---------------------------------------------------------------- Layer loading

    def _load_result_layers(self, result, is_earthworks=False):
        label = "Earthworks" if is_earthworks else "Baseline"
        layer_ids = []

        # Stream network raster
        stream_path = result.get("stream_network")
        if stream_path and os.path.exists(stream_path):
            layer = QgsRasterLayer(stream_path, f"{label} — Streams")
            if layer.isValid():
                self._apply_stream_ramp(layer, result.get("stream_acc_max", 1))
                QgsProject.instance().addMapLayer(layer)
                layer_ids.append(layer.id())

        # Ponding raster
        ponding_path = result.get("ponding")
        if ponding_path and os.path.exists(ponding_path):
            self._ponding_raster_path = ponding_path
            layer = QgsRasterLayer(ponding_path, f"{label} — Water Captured")
            if layer.isValid():
                self._apply_ponding_ramp(layer)
                QgsProject.instance().addMapLayer(layer)
                layer_ids.append(layer.id())

        # Exit points vector
        exit_points = result.get("exit_points", [])
        if exit_points:
            ep_layer = self._create_exit_points_layer(exit_points, label)
            if ep_layer:
                QgsProject.instance().addMapLayer(ep_layer)
                layer_ids.append(ep_layer.id())

        if is_earthworks:
            self._earthworks_layer_ids = layer_ids
        else:
            self._baseline_layer_ids = layer_ids

    def _create_exit_points_layer(self, exit_points, label):
        from qgis.PyQt.QtCore import QVariant

        layer = QgsVectorLayer("Point?crs=" + (self._dem_info.crs_wkt or "EPSG:4326"),
                               f"{label} — Exit Points", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label", QVariant.String),
            QgsField("volume_m3", QVariant.Double),
            QgsField("flow_ls", QVariant.Double),
        ])
        layer.updateFields()

        feats = []
        for ep in exit_points:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(ep["x"], ep["y"])))
            f.setAttributes([ep.get("label", ""), ep.get("volume_m3", 0), ep.get("flow_ls", 0)])
            feats.append(f)
        pr.addFeatures(feats)

        # Red circle marker
        symbol = QgsMarkerSymbol.createSimple({
            "name": "circle", "color": "220,0,0,200",
            "outline_color": "140,0,0", "size": "5",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        # Map labels — bold dark-red text with white halo, placed above the dot
        from qgis.PyQt.QtGui import QFont
        from qgis.core import QgsTextBufferSettings

        text_fmt = QgsTextFormat()
        text_fmt.setColor(QColor(160, 0, 0))
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        text_fmt.setFont(font)

        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor(255, 255, 255))
        buf.setSize(1.5)
        text_fmt.setBuffer(buf)

        label_settings = QgsPalLayerSettings()
        label_settings.fieldName = "label"
        label_settings.enabled = True
        # Place label above the point
        try:
            label_settings.placement = QgsPalLayerSettings.OverPoint
            label_settings.quadOffset = QgsPalLayerSettings.QuadrantAbove
        except Exception:
            pass
        label_settings.yOffset = 2.0
        label_settings.setFormat(text_fmt)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
        layer.setLabelsEnabled(True)

        return layer

    # ---------------------------------------------------------------- Contour analysis

    def _run_contour_analysis(self):
        if not self._dem_path:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM first."
            )
            return
        acc_path = (self._baseline_result or {}).get("flow_accumulation")
        if not acc_path or not os.path.exists(acc_path):
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first to get flow accumulation."
            )
            return

        from .modules.contour_analysis import analyse_contours

        def _progress(pct, msg):
            self.panel.set_contour_progress(pct, msg)

        try:
            contours = analyse_contours(
                dem_path=self._dem_path,
                acc_path=acc_path,
                interval_m=self.panel.contour_interval_m,
                max_slope_deg=self.panel.max_slope_deg,
                usable_polygon=self._usable_polygon,
                progress_callback=_progress,
                cell_area_m2=self._dem_info.cell_area_m2 if self._dem_info else None,
                runoff_mm=(self._baseline_result or {}).get("runoff_mm"),
                min_length_m=self.panel.min_contour_length_m,
            )
            self._contour_features = contours
            self.panel.set_contour_complete()
            self.panel.set_contour_results(contours)
            self._display_contour_layer(contours)
        except Exception as exc:
            self.panel.set_contour_complete()
            self.iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Contour analysis failed: {exc}"
            )

    def _display_contour_layer(self, contours):
        from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry
        from qgis.PyQt.QtCore import QVariant
        from qgis.core import QgsField

        crs_str = self._dem_info.crs_wkt if self._dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Candidate Contour Swales", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("elevation", QVariant.Double),
            QgsField("rank", QVariant.Int),
            QgsField("peak_acc", QVariant.Double),
            QgsField("mean_slope", QVariant.Double),
        ])
        layer.updateFields()

        feats = []
        for feat in contours:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(feat.geometry.wkt))
            f.setAttributes([feat.elevation, feat.rank or 0, feat.peak_acc, feat.mean_slope_deg])
            feats.append(f)
        pr.addFeatures(feats)

        # Data-defined blue ramp by peak_acc (flat layer, no sub-groups)
        max_acc = max((c.peak_acc or 0) for c in contours) if contours else 1.0
        self._apply_rank_style(layer, max_acc=max_acc)
        if self._contour_layer:
            try:
                QgsProject.instance().removeMapLayer(self._contour_layer)
            except Exception:
                pass
        QgsProject.instance().addMapLayer(layer)
        self._contour_layer = layer

    def _apply_rank_style(self, layer, max_acc=None):
        """Single-symbol renderer with distinct colours by rank group.
        Rank 1        → gold   #FFD700  (thickest — the best)
        Rank 2–5      → orange #FF6B00
        Rank 6–10     → steel  #4A90D9
        Rank 11+      → grey   #9E9E9E
        Produces a flat (non-grouped) layer entry so click tools see all features."""
        from qgis.core import QgsLineSymbol
        # Colour by rank band using a CASE expression
        color_expr = (
            "CASE"
            " WHEN \"rank\" = 1      THEN color_rgb(255,215,  0)"   # gold
            " WHEN \"rank\" <= 5     THEN color_rgb(255,107,  0)"   # orange
            " WHEN \"rank\" <= 10    THEN color_rgb( 74,144,217)"   # steel blue
            " ELSE                       color_rgb(158,158,158)"    # grey
            " END"
        )
        # Width also varies by rank group — top contour is boldest
        width_expr = (
            "CASE"
            " WHEN \"rank\" = 1   THEN 2.2"
            " WHEN \"rank\" <= 5  THEN 1.6"
            " WHEN \"rank\" <= 10 THEN 1.1"
            " ELSE                     0.7"
            " END"
        )
        symbol = QgsLineSymbol.createSimple({"width": "1.0", "capstyle": "round"})
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor,
            QgsProperty.fromExpression(color_expr),
        )
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth,
            QgsProperty.fromExpression(width_expr),
        )
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    def _select_top5_contours(self):
        """Create a separate layer with the top 5 ranked candidate swale contours."""
        if not self._contour_features:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run contour analysis first."
            )
            return

        from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry, QgsField
        from qgis.PyQt.QtCore import QVariant

        ranked = sorted(
            self._contour_features,
            key=lambda f: f.peak_acc if f.peak_acc is not None else 0,
            reverse=True,
        )[:5]

        crs_str = self._dem_info.crs_wkt if self._dem_info else "EPSG:4326"

        # Remove previous top-5 layer if it exists
        if hasattr(self, "_top5_layer") and self._top5_layer:
            try:
                QgsProject.instance().removeMapLayer(self._top5_layer)
            except Exception:
                pass

        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Top 5 Swale Contours", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("elevation", QVariant.Double),
            QgsField("rank", QVariant.Int),
            QgsField("peak_acc", QVariant.Double),
            QgsField("mean_slope", QVariant.Double),
        ])
        layer.updateFields()

        feats = []
        for i, feat in enumerate(ranked):
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(feat.geometry.wkt))
            f.setAttributes([feat.elevation, i + 1, feat.peak_acc, feat.mean_slope_deg])
            feats.append(f)
        pr.addFeatures(feats)

        # Bright distinct style — thick orange lines
        from qgis.core import QgsLineSymbol
        symbol = QgsLineSymbol.createSimple({
            "color": "255,140,0", "width": "1.2", "capstyle": "round",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        QgsProject.instance().addMapLayer(layer)
        self._top5_layer = layer

    def _run_segment_analysis(self):
        if not self._contour_features:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run contour analysis first."
            )
            return
        acc_path = (self._baseline_result or {}).get("flow_accumulation")
        if not acc_path or not os.path.exists(acc_path):
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first to get flow accumulation."
            )
            return

        from .modules.contour_analysis import find_swale_segments

        def _progress(pct, msg):
            self.panel.set_contour_progress(pct, msg)

        try:
            segments = find_swale_segments(
                contours=self._contour_features,
                acc_path=acc_path,
                cell_area_m2=self._dem_info.cell_area_m2 if self._dem_info else 1.0,
                runoff_mm=(self._baseline_result or {}).get("runoff_mm"),
                min_acc_ha=self.panel.min_catchment_ha,
                swale_depth_m=self.panel.swale_depth_m,
                swale_width_m=self.panel.swale_width_m,
                progress_callback=_progress,
            )
            self.panel.set_contour_complete()
            self._display_swale_segments(segments)
        except Exception as exc:
            self.panel.set_contour_complete()
            self.iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Segment analysis failed: {exc}"
            )

    def _display_swale_segments(self, segments):
        from qgis.core import QgsVectorLayer, QgsFeature, QgsGeometry, QgsField, QgsLineSymbol
        from qgis.PyQt.QtCore import QVariant

        if not segments:
            self.iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No swale segments found — try lowering 'Min catchment above swale' "
                "or run contour analysis with a smaller contour interval."
            )
            return

        # Remove previous layer
        if self._segment_layer:
            try:
                QgsProject.instance().removeMapLayer(self._segment_layer)
            except Exception:
                pass

        crs_str = self._dem_info.crs_wkt if self._dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Recommended Swale Segments", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",             QVariant.String),
            QgsField("elevation",         QVariant.Double),
            QgsField("contributing_ha",   QVariant.Double),
            QgsField("inflow_m3",         QVariant.Double),
            QgsField("required_length_m", QVariant.Double),
            QgsField("rank",              QVariant.Int),
        ])
        layer.updateFields()

        feats = []
        for seg in segments:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(seg.geometry.wkt))
            f.setAttributes([
                seg.label, seg.elevation, seg.contributing_ha,
                seg.inflow_m3, seg.required_length_m, seg.segment_rank,
            ])
            feats.append(f)
        pr.addFeatures(feats)

        # Distinct colours by segment rank group — matches contour layer scheme
        from qgis.core import QgsLineSymbol
        color_expr = (
            "CASE"
            " WHEN \"rank\" = 1      THEN color_rgb(255,215,  0)"   # gold
            " WHEN \"rank\" <= 5     THEN color_rgb(255,107,  0)"   # orange
            " WHEN \"rank\" <= 10    THEN color_rgb( 74,144,217)"   # steel blue
            " ELSE                       color_rgb(158,158,158)"    # grey
            " END"
        )
        width_expr = (
            "CASE"
            " WHEN \"rank\" = 1   THEN 2.2"
            " WHEN \"rank\" <= 5  THEN 1.6"
            " WHEN \"rank\" <= 10 THEN 1.1"
            " ELSE                     0.7"
            " END"
        )
        symbol = QgsLineSymbol.createSimple({"width": "1.0", "capstyle": "round"})
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor,
            QgsProperty.fromExpression(color_expr),
        )
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth,
            QgsProperty.fromExpression(width_expr),
        )
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        # Labels showing rank + inflow
        from qgis.PyQt.QtGui import QFont
        from qgis.core import QgsTextBufferSettings
        text_fmt = QgsTextFormat()
        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        text_fmt.setFont(font)
        text_fmt.setColor(QColor(20, 90, 40))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor(255, 255, 255))
        buf.setSize(1.0)
        text_fmt.setBuffer(buf)

        lbl = QgsPalLayerSettings()
        lbl.fieldName = "label"
        lbl.enabled = True
        lbl.setFormat(text_fmt)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
        layer.setLabelsEnabled(True)

        QgsProject.instance().addMapLayer(layer)
        self._segment_layer = layer

        self.iface.messageBar().pushSuccess(
            "TerrainFlow Assessment",
            f"{len(segments)} swale segment(s) found. Top segment: {segments[0].label}"
        )

    def _generate_simple_contours(self):
        """Generate plain elevation contours from the DEM using GDAL."""
        if not self._dem_path:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM first."
            )
            return
        try:
            import processing
            interval = self.panel.simple_contour_interval_m
            result = processing.run("gdal:contour", {
                "INPUT": self._dem_path,
                "BAND": 1,
                "INTERVAL": interval,
                "FIELD_NAME": "ELEV",
                "OUTPUT": "TEMPORARY_OUTPUT",
            })
            out = result.get("OUTPUT")
            if hasattr(out, "source"):
                out = out.source()

            # Remove previous simple contour layer if present
            if hasattr(self, "_simple_contour_layer") and self._simple_contour_layer:
                try:
                    QgsProject.instance().removeMapLayer(self._simple_contour_layer)
                except Exception:
                    pass

            layer = QgsVectorLayer(out, f"Contours ({interval} m)", "ogr")
            if not layer.isValid():
                self.iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", "Contour generation produced no output."
                )
                return

            # Thin grey lines
            from qgis.core import QgsLineSymbol
            symbol = QgsLineSymbol.createSimple({
                "color": "100,100,100", "width": "0.3",
            })
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))

            QgsProject.instance().addMapLayer(layer)
            self._simple_contour_layer = layer

        except Exception as exc:
            self.iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Contour generation failed: {exc}"
            )

    # ---------------------------------------------------------------- Keypoint analysis

    def _get_keypoint_boundary_mask(self, dem_path):
        """
        Return a boolean numpy mask (same shape as the DEM) restricting
        keypoint/ridgeline search to the earthworks area, or failing that
        the analysis area, or the site boundary.  Returns None if none set.
        """
        import json
        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rasterize
        from shapely.geometry import shape as _shape
        from shapely.ops import unary_union

        # Priority: earthworks area → analysis area → boundary
        layer = (
            self.panel.earthworks_area_layer
            or self.panel.analysis_area_layer
        )
        if layer is None and not self._boundary_path:
            return None

        try:
            with rasterio.open(dem_path) as src:
                shape = (src.height, src.width)
                transform = src.transform
                crs_wkt = src.crs.to_wkt() if src.crs else None

            if layer is not None:
                import geopandas as gpd
                gdf = gpd.GeoDataFrame.from_features(
                    [f.__geo_interface__ for f in layer.getFeatures()],
                    crs=layer.crs().toWkt(),
                )
                if crs_wkt:
                    gdf = gdf.to_crs(crs_wkt)
                polys = list(gdf.geometry)
            else:
                import geopandas as gpd
                gdf = gpd.read_file(self._boundary_path)
                if crs_wkt:
                    gdf = gdf.to_crs(crs_wkt)
                polys = list(gdf.geometry)

            if not polys:
                return None

            mask = _rasterize(
                [(geom, 1) for geom in polys if geom is not None],
                out_shape=shape,
                transform=transform,
                fill=0, all_touched=True, dtype="uint8",
            ).astype(bool)
            return mask if mask.any() else None

        except Exception:
            return None

    def _run_keypoint_analysis(self):
        if not self._dem_path:
            return
        acc_path = (self._baseline_result or {}).get("flow_accumulation")
        if not acc_path:
            QMessageBox.warning(self.panel, "No Analysis",
                                "Run baseline analysis first.")
            return

        from .modules.keypoint_analysis import KeylineAnalysis

        self.panel.set_keypoint_progress(5, "Loading DEM…")
        try:
            # Scale min_acc to cell size so threshold = ≥1 ha drainage area
            cell_m2 = self._dem_info.cell_area_m2 if self._dem_info else 100.0
            min_acc = max(50, int(1.0 * 10_000 / cell_m2))

            self.panel.set_keypoint_progress(15, "Building boundary mask…")
            boundary_mask = self._get_keypoint_boundary_mask(self._dem_path)

            self.panel.set_keypoint_progress(25, "Finding keypoints…")
            ka = KeylineAnalysis(self._dem_path, acc_path)
            self._keyline_analysis = ka

            keypoints = ka.find_keypoints(
                min_acc_cells=min_acc,
                n_keypoints=self.panel.keypoint_count,
                boundary_mask=boundary_mask,
            )
            self._found_keypoints = keypoints

            self.panel.set_keypoint_progress(70, "Finding ridgelines…")
            ridgelines = ka.find_ridgelines(boundary_mask=boundary_mask)

            self._display_keypoints(keypoints)
            self._display_ridgelines(ridgelines)

            if keypoints:
                self.panel.set_keypoint_complete(
                    f"{len(keypoints)} keypoints | {len(ridgelines)} ridgeline segment(s)\n"
                    "Click 'Recommend Pond Sites' to find optimal dam locations."
                )
            else:
                self.panel.set_keypoint_complete(
                    "No keypoints found — try a larger DEM area or lower the\n"
                    "number of keypoints requested."
                )

            kp_lines = "\n".join(f"  {kp['label']}" for kp in keypoints)
            ridge_summary = (
                f"{len(ridgelines)} ridge segment(s)" if ridgelines
                else "No ridgelines found"
            )
            self.panel.set_keypoint_results(
                f"Keypoints ({len(keypoints)}):\n{kp_lines}\n\n{ridge_summary}"
            )

        except Exception:
            import traceback
            self.panel.set_keypoint_complete("")
            QMessageBox.critical(self.panel, "Keypoint Analysis Error",
                                 traceback.format_exc())

    def _run_recommend_ponds(self):
        if not getattr(self, "_found_keypoints", None):
            QMessageBox.warning(self.panel, "No Keypoints",
                                "Run 'Find Keypoints + Ridgelines' first.")
            return

        self.panel.set_keypoint_progress(5, "Finding pond sites…")
        try:
            ka = getattr(self, "_keyline_analysis", None)
            if ka is None:
                from .modules.keypoint_analysis import KeylineAnalysis
                acc_path = (self._baseline_result or {}).get("flow_accumulation")
                ka = KeylineAnalysis(self._dem_path, acc_path)

            self.panel.set_keypoint_progress(50, "Finding pond sites…")
            boundary_mask = self._get_keypoint_boundary_mask(self._dem_path)
            pond_sites = ka.recommend_pond_sites(self._found_keypoints,
                                                 boundary_mask=boundary_mask)
            self._display_pond_sites(pond_sites)

            lines = "\n".join(f"  {s['label']}" for s in pond_sites)
            existing = self.panel._keypoint_results_lbl.text()
            self.panel.set_keypoint_results(
                existing + f"\n\nPond sites:\n{lines}"
            )
            self.panel.set_keypoint_complete(
                f"{len(self._found_keypoints)} keypoints | {len(pond_sites)} pond site(s) found."
            )
        except Exception:
            import traceback
            self.panel.set_keypoint_complete("")
            QMessageBox.critical(self.panel, "Pond Site Error",
                                 traceback.format_exc())

    def _display_keypoints(self, keypoints):
        from qgis.PyQt.QtCore import QVariant
        for lyr in QgsProject.instance().mapLayersByName("Keypoints"):
            QgsProject.instance().removeMapLayer(lyr)

        layer = QgsVectorLayer("Point", "Keypoints", "memory")
        layer.setCrs(QgsProject.instance().crs())
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",        QVariant.String),
            QgsField("elevation",    QVariant.Double),
            QgsField("slope_deg",    QVariant.Double),
            QgsField("catchment_ha", QVariant.Double),
        ])
        layer.updateFields()
        feats = []
        for kp in keypoints:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(kp["x"], kp["y"])))
            f.setAttributes([kp["label"], kp["elevation"],
                             kp.get("slope_deg", 0.0), kp["catchment_ha"]])
            feats.append(f)
        pr.addFeatures(feats)

        symbol = QgsMarkerSymbol.createSimple({
            "name": "diamond", "color": "220,100,0,230",
            "outline_color": "100,40,0,230", "size": "8",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        lbl = QgsPalLayerSettings()
        lbl.fieldName = "label"
        lbl.enabled = True
        tf = QgsTextFormat()
        tf.setColor(QColor(120, 50, 0))
        lbl.setFormat(tf)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
        layer.setLabelsEnabled(True)
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)

    def _display_ridgelines(self, ridgelines):
        from qgis.core import QgsLineSymbol
        from qgis.PyQt.QtCore import QVariant
        for lyr in QgsProject.instance().mapLayersByName("Ridgelines (Water Divides)"):
            QgsProject.instance().removeMapLayer(lyr)

        layer = QgsVectorLayer("LineString", "Ridgelines (Water Divides)", "memory")
        layer.setCrs(QgsProject.instance().crs())
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",          QVariant.String),
            QgsField("length_m",       QVariant.Double),
            QgsField("mean_elevation", QVariant.Double),
        ])
        layer.updateFields()
        feats = []
        for rl in ridgelines:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(rl["geometry"].wkt))
            f.setAttributes([rl["label"], rl["length_m"], rl.get("mean_elevation", 0.0)])
            feats.append(f)
        pr.addFeatures(feats)

        sym = QgsLineSymbol.createSimple({
            "color": "140,60,180,200", "width": "1.0", "line_style": "dash",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(sym))
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)

    def _display_pond_sites(self, sites):
        from qgis.PyQt.QtCore import QVariant
        for lyr in QgsProject.instance().mapLayersByName("Recommended Pond Sites"):
            QgsProject.instance().removeMapLayer(lyr)

        layer = QgsVectorLayer("Point", "Recommended Pond Sites", "memory")
        layer.setCrs(QgsProject.instance().crs())
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",        QVariant.String),
            QgsField("elevation",    QVariant.Double),
            QgsField("catchment_ha", QVariant.Double),
            QgsField("dam_width_m",  QVariant.Double),
            QgsField("keypoint",     QVariant.Int),
        ])
        layer.updateFields()
        feats = []
        for s in sites:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(s["x"], s["y"])))
            f.setAttributes([s["label"], s["elevation"], s["catchment_ha"],
                             s["dam_width_m"], s["keypoint"]])
            feats.append(f)
        pr.addFeatures(feats)

        symbol = QgsMarkerSymbol.createSimple({
            "name": "circle", "color": "0,120,200,230",
            "outline_color": "0,60,120,255", "size": "10",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        lbl = QgsPalLayerSettings()
        lbl.fieldName = "label"
        lbl.enabled = True
        tf = QgsTextFormat()
        tf.setColor(QColor(0, 60, 140))
        lbl.setFormat(tf)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
        layer.setLabelsEnabled(True)
        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)

    # ---------------------------------------------------------------- Earthwork drawing

    def _activate_draw_swale(self, mode):
        if mode == "contour":
            if not self._contour_layer:
                self.iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "Run contour analysis first, then pick a segment on a contour."
                )
                return
            tool = ContourSegmentTool(self.canvas, self._contour_layer)
            tool.segment_selected.connect(
                lambda geom, elev: self._on_contour_selected_for_swale(geom, elev)
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self.canvas.setMapTool(tool)
        elif mode == "full_contour":
            if not self._contour_layer:
                self.iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "Run contour analysis first, then click a contour."
                )
                return
            tool = SelectContourTool(self.canvas, self._contour_layer)
            tool.contour_selected.connect(
                lambda geom, elev: self._on_contour_selected_for_swale(geom, elev)
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self.canvas.setMapTool(tool)
        else:
            tool = DrawLineTool(self.canvas,
                                slope_raster_path=self._slope_raster_path,
                                tool_label="swale")
            tool.line_drawn.connect(lambda geom: self._on_geometry_drawn("swale", geom))
            tool.cancelled.connect(self._on_draw_cancelled)
            self.canvas.setMapTool(tool)

    def _on_contour_selected_for_swale(self, geom, elevation):
        swale_geom = contour_to_swale_geometry(geom)
        self._on_geometry_drawn("swale", swale_geom)

    def _activate_draw_line(self, ew_type):
        tool = DrawLineTool(self.canvas,
                            slope_raster_path=self._slope_raster_path,
                            tool_label=ew_type)
        tool.line_drawn.connect(lambda geom: self._on_geometry_drawn(ew_type, geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self.canvas.setMapTool(tool)

    def _activate_draw_basin(self):
        tool = DrawPolygonTool(self.canvas,
                               slope_raster_path=self._slope_raster_path,
                               tool_label="basin")
        tool.polygon_drawn.connect(lambda geom: self._on_geometry_drawn("basin", geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self.canvas.setMapTool(tool)

    def _on_usable_area_source_changed(self, source):
        """Set _usable_polygon from the analysis or earthworks area layer."""
        import json
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union

        if source == "none":
            self._usable_polygon = None
            self.iface.messageBar().pushInfo(
                "TerrainFlow Assessment", "Usable area cleared — full DEM will be used."
            )
            return

        layer = (
            self.panel.analysis_area_layer if source == "analysis"
            else self.panel.earthworks_area_layer
        )
        if layer is None:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "No layer selected for that area — set it in the Data section first."
            )
            return

        try:
            polys = []
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom and not geom.isEmpty():
                    polys.append(shapely_shape(json.loads(geom.asJson())))
            if not polys:
                raise ValueError("Layer has no valid polygon features.")
            self._usable_polygon = unary_union(polys)
            label = "Analysis Area" if source == "analysis" else "Earthworks Area"
            self.iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"Usable area set from '{label}'. Run contour analysis to apply."
            )
        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not read usable area layer: {exc}"
            )

    def _on_geometry_drawn(self, ew_type, geometry):
        from .earthwork_properties_dialog import EarthworkPropertiesDialog

        # Sample peak inflow at this geometry
        peak_inflow = 0.0
        if self._baseline_result:
            acc_path = self._baseline_result.get("flow_accumulation")
            if acc_path:
                acc_cells = sample_peak_inflow(geometry, acc_path)
                cell_area = self._dem_info.cell_area_m2 if self._dem_info else 1.0
                runoff_mm = self._baseline_result.get("runoff_mm", 0)
                peak_inflow = acc_cells * cell_area * runoff_mm / 1000.0

        # Crest elevation for dams
        crest_elev = None
        if ew_type == "dam" and self._dem_path:
            try:
                import json, rasterio
                from shapely.geometry import shape as shapely_shape
                shp = shapely_shape(json.loads(geometry.asJson()))
                centroid = shp.centroid
                with rasterio.open(self._dem_path) as src:
                    t = src.transform
                    col = int((centroid.x - t.c) / t.a)
                    row = int((centroid.y - t.f) / t.e)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        crest_elev = float(src.read(1)[row, col]) + 2.0
            except Exception:
                pass

        n = len(self._earthwork_manager) + 1
        ew_name = f"{ew_type.capitalize()} {n}"
        ew = Earthwork(ew_type, geometry, ew_name)

        dlg = EarthworkPropertiesDialog(
            ew_type=ew_type,
            geometry=geometry,
            parent=self.iface.mainWindow(),
            earthwork=ew,
            peak_inflow_m3=peak_inflow,
            crest_elevation=crest_elev,
            duration_hours=self.panel.duration_hr,
            dem_path=self._dem_path if ew_type == "dam" else None,
        )

        if dlg.exec_():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            ew.width = getattr(dlg, "get_width", lambda: ew.width)()
            if ew_type == "dam":
                ew.crest_elevation = dlg.get_crest_elevation()
            elif ew_type == "swale":
                ew.companion_berm = getattr(dlg, "get_companion_berm", lambda: False)()

            ew.capacity_m3, ew.capacity_l = calculate_capacity(
                ew_type, geometry, ew.depth, ew.width,
                getattr(ew, "companion_berm", False)
            )
            self._earthwork_manager.add(ew)
            self.panel.add_earthwork_to_list(
                len(self._earthwork_manager) - 1, ew.summary()
            )
            self._refresh_ew_layer()
        self.canvas.unsetMapTool(self.canvas.mapTool())

    def _on_draw_cancelled(self):
        self.canvas.unsetMapTool(self.canvas.mapTool())

    def _edit_selected_earthwork(self):
        idx = self.panel.get_selected_earthwork_index()
        if idx is None:
            return
        ew = self._earthwork_manager.get(idx)
        from .earthwork_properties_dialog import EarthworkPropertiesDialog
        dlg = EarthworkPropertiesDialog(
            ew_type=ew.type,
            geometry=ew.geometry,
            parent=self.iface.mainWindow(),
            earthwork=ew,
            duration_hours=self.panel.duration_hr,
            dem_path=self._dem_path if ew.type == "dam" else None,
        )
        if dlg.exec_():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            ew.capacity_m3, ew.capacity_l = calculate_capacity(
                ew.type, ew.geometry, ew.depth, ew.width,
                getattr(ew, "companion_berm", False)
            )
            self.panel.update_earthwork_in_list(idx, ew.summary())
            self._refresh_ew_layer()

    def _delete_selected_earthwork(self):
        idx = self.panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._earthwork_manager.remove(idx)
        self.panel.refresh_earthwork_list(self._earthwork_manager.get_all())
        self._refresh_ew_layer()

    def _toggle_selected_earthwork(self):
        idx = self.panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._earthwork_manager.toggle(idx)
        self.panel.refresh_earthwork_list(self._earthwork_manager.get_all())

    # Per-type styles: (geometry_type, display_name, line/outline colour, fill colour or None, line width)
    _EW_STYLES = {
        "swale":     ("LineString", "Swales",     "#00BCD4", None,      "2.5"),
        "berm":      ("LineString", "Berms",      "#FF6D00", None,      "2.5"),
        "dam":       ("LineString", "Dams",       "#E53935", None,      "3.0"),
        "diversion": ("LineString", "Diversions", "#AB47BC", None,      "2.0"),
        "basin":     ("Polygon",    "Basins",     "#1565C0", "#1565C0", "1.5"),
    }

    def _ensure_ew_layers(self):
        """Create (or re-create) the Earthworks layer group and per-type layers."""
        from qgis.core import (QgsVectorLayer, QgsField, QgsLineSymbol,
                                QgsFillSymbol, QgsLayerTreeGroup)
        from qgis.PyQt.QtCore import QVariant
        from qgis.PyQt.QtGui import QColor, QFont

        crs_str = self._dem_info.crs_wkt if self._dem_info else "EPSG:4326"
        root = QgsProject.instance().layerTreeRoot()

        # Re-use existing group by name, or insert a new one at the top
        self._ew_group = root.findGroup("Earthworks") or root.insertGroup(0, "Earthworks")

        for ew_type, (geom_type, display_name, color_hex, fill_hex, width) in self._EW_STYLES.items():
            existing = self._ew_layers.get(ew_type)
            if existing and QgsProject.instance().mapLayer(existing.id()):
                continue  # still valid — leave it

            layer = QgsVectorLayer(f"{geom_type}?crs={crs_str}", display_name, "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name",        QVariant.String),
                QgsField("type",        QVariant.String),
                QgsField("capacity_m3", QVariant.Double),
                QgsField("enabled",     QVariant.Int),
            ])
            layer.updateFields()

            # --- Symbol ---
            if geom_type == "LineString":
                sym = QgsLineSymbol.createSimple({
                    "color": color_hex, "width": width,
                    "capstyle": "round", "joinstyle": "round",
                })
                layer.setRenderer(QgsSingleSymbolRenderer(sym))
            else:
                # Polygon — no fill, solid outline only so ponding is visible beneath
                outline_color = QColor(color_hex)
                sym = QgsFillSymbol.createSimple({
                    "style": "no",           # transparent fill
                    "outline_style": "solid",
                    "outline_width": width,
                    "outline_color": color_hex,
                })
                layer.setRenderer(QgsSingleSymbolRenderer(sym))

            # --- Labels (name, white halo, coloured text) ---
            text_fmt = QgsTextFormat()
            font = QFont()
            font.setBold(True)
            font.setPointSize(8)
            text_fmt.setFont(font)
            text_fmt.setColor(QColor(color_hex))
            from qgis.core import QgsTextBufferSettings
            buf = QgsTextBufferSettings()
            buf.setEnabled(True)
            buf.setColor(QColor(255, 255, 255))
            buf.setSize(1.2)
            text_fmt.setBuffer(buf)
            lbl = QgsPalLayerSettings()
            lbl.fieldName = "name"
            lbl.enabled = True
            lbl.setFormat(text_fmt)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
            layer.setLabelsEnabled(True)

            QgsProject.instance().addMapLayer(layer, False)  # don't auto-add to root
            self._ew_group.addLayer(layer)
            self._ew_layers[ew_type] = layer

    def _refresh_ew_layer(self):
        self._ensure_ew_layers()

        # Clear all layers
        for layer in self._ew_layers.values():
            if layer and QgsProject.instance().mapLayer(layer.id()):
                layer.dataProvider().truncate()

        # Repopulate per type
        for ew in self._earthwork_manager.get_all():
            layer = self._ew_layers.get(ew.type)
            if not layer or not QgsProject.instance().mapLayer(layer.id()):
                continue
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(ew.geometry.asWkt()))
            f.setAttributes([ew.name, ew.type, ew.capacity_m3,
                              1 if getattr(ew, "enabled", True) else 0])
            layer.dataProvider().addFeature(f)

        for layer in self._ew_layers.values():
            if layer and QgsProject.instance().mapLayer(layer.id()):
                layer.triggerRepaint()

    # ---------------------------------------------------------------- Earthworks analysis

    def _run_with_earthworks(self):
        if not self._dem_path or not self._burner:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM and run baseline first."
            )
            return

        enabled = self._earthwork_manager.get_enabled()
        if not enabled:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "No enabled earthworks to re-analyse with."
            )
            return

        modified_dem = self._burner.burn_earthworks(enabled)
        mod_path = os.path.join(self._output_dir, "modified_dem.tif")
        self._burner.save(modified_dem, mod_path)
        self._modified_dem_path = mod_path

        cell_area_m2 = self._dem_info.cell_area_m2 if self._dem_info else 1.0
        threshold_cells = int(
            self.panel.stream_threshold_ha * 10_000 / cell_area_m2
        ) if cell_area_m2 > 0 else 1000

        self._analysis_worker = AnalysisWorker(
            dem_path=mod_path,
            output_dir=self._output_dir,
            stream_threshold=threshold_cells,
            cn=self.panel.cn,
            moisture=self.panel.moisture,
            rainfall_mm=self.panel.rainfall_mm,
            duration_hours=self.panel.duration_hr,
            boundary_path=self._boundary_path,
            label="earthworks",
            run_catchments=False,
            routing=self.panel.routing,
        )
        self._analysis_worker.progress.connect(self.panel.set_earthworks_progress)
        self._analysis_worker.finished.connect(self._on_earthworks_complete)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def _on_earthworks_complete(self, result):
        self._earthworks_result = result
        self._load_result_layers(result, is_earthworks=True)
        if result.get("ponding"):
            self._ponding_raster_path = result["ponding"]
        self.panel.set_earthworks_complete(
            "Earthworks analysis complete. Toggle 'Show: with earthworks' to compare."
        )

    def _activate_ponding_query(self):
        if not self._ponding_raster_path:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first."
            )
            return
        tool = PondingQueryTool(self.canvas, self._ponding_raster_path)
        tool.ponding_selected.connect(self._on_ponding_selected)
        tool.no_ponding.connect(self._on_no_ponding)
        self.canvas.setMapTool(tool)

    def _on_ponding_selected(self, volume_m3, volume_l, cell_count, area_m2,
                              outline_geom, inflow_m3, fill_fraction):
        from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
        from qgis.PyQt.QtGui import QFont

        dlg = QDialog(self.iface.mainWindow())
        dlg.setWindowTitle("Depression / Ponding Results")
        dlg.setMinimumWidth(340)
        layout = QVBoxLayout(dlg)

        def _row(bold_text, value_text, tooltip=None):
            lbl = QLabel(f"<b>{bold_text}</b>  {value_text}")
            if tooltip:
                lbl.setToolTip(tooltip)
            layout.addWidget(lbl)

        _row("Volume held:",
             f"{volume_m3:,.1f} m³  ({volume_l:,.0f} L)",
             "Total water volume in the connected depression area.")

        area_ha = area_m2 / 10_000
        if area_ha >= 0.1:
            area_str = f"{area_ha:.2f} ha  ({area_m2:,.0f} m²)"
        else:
            area_str = f"{area_m2:,.0f} m²"
        _row("Surface area:", area_str)

        # Approximate mean depth
        if area_m2 > 0:
            mean_depth = volume_m3 / area_m2
            _row("Mean depth:", f"{mean_depth:.2f} m")

        _row("Cell count:", f"{cell_count:,} cells")

        # Fill fraction for the design storm
        if fill_fraction >= 0:
            fill_pct = fill_fraction * 100.0
            if fill_fraction <= 1.0:
                fill_str = (
                    f"{fill_pct:.0f}% of design-storm inflow captured  ✓"
                )
                colour = "#006600"
            else:
                fill_str = (
                    f"{fill_pct:.0f}% of design-storm inflow — depression overflows  ⚠"
                )
                colour = "#cc4400"
            lbl_fill = QLabel(f"<b>Storm fill:</b>  "
                              f'<span style="color:{colour}">{fill_str}</span>')
            lbl_fill.setToolTip(
                "Ratio of design-storm inflow volume to depression capacity.\n"
                "Below 100%: depression absorbs the full storm event.\n"
                "Above 100%: overflow will occur — consider enlarging the earthwork."
            )
            layout.addWidget(lbl_fill)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec_()

    def _on_no_ponding(self):
        self.iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "No ponding at that location — click a blue zone in the Water Captured layer."
        )

    def _toggle_slope_class(self, checked):
        if not self._slope_raster_path or not os.path.exists(self._slope_raster_path):
            return
        lid = self._slope_class_layer_id
        if lid:
            node = QgsProject.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(checked)
                self.canvas.refresh()
                return
        layer = QgsRasterLayer(self._slope_raster_path, "Slope Classification")
        if layer.isValid():
            self._apply_slope_class_ramp(layer)
            layer.setOpacity(0.6)
            QgsProject.instance().addMapLayer(layer)
            self._slope_class_layer_id = layer.id()
            node = QgsProject.instance().layerTreeRoot().findLayer(layer)
            if node:
                node.setItemVisibilityChecked(checked)
            self.canvas.refresh()

    def _apply_slope_class_ramp(self, layer):
        shader = QgsColorRampShader()
        shader.setColorRampType(QgsColorRampShader.Interpolated)
        shader.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(3,  QColor("#50C850"), "0–3°"),
            QgsColorRampShader.ColorRampItem(8,  QColor("#DCDC1E"), "3–8°"),
            QgsColorRampShader.ColorRampItem(13, QColor("#FFA500"), "8–13°"),
            QgsColorRampShader.ColorRampItem(18, QColor("#FF6600"), "13–18°"),
            QgsColorRampShader.ColorRampItem(25, QColor("#CC2200"), "18–25°"),
            QgsColorRampShader.ColorRampItem(90, QColor("#660000"), ">25°"),
        ])
        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, raster_shader)
        layer.setRenderer(renderer)

    def _toggle_slope_arrows(self, checked):
        if checked:
            existing = (self._slope_arrows_layer_id and
                        QgsProject.instance().mapLayer(self._slope_arrows_layer_id))
            if existing:
                node = QgsProject.instance().layerTreeRoot().findLayer(self._slope_arrows_layer_id)
                if node:
                    node.setItemVisibilityChecked(True)
            else:
                self._generate_slope_arrows()
        else:
            if self._slope_arrows_layer_id:
                node = QgsProject.instance().layerTreeRoot().findLayer(self._slope_arrows_layer_id)
                if node:
                    node.setItemVisibilityChecked(False)
            self.canvas.refresh()

    def _generate_slope_arrows(self):
        if not self._dem_path:
            return
        try:
            import processing
            import rasterio
            import numpy as np

            # Compute aspect using GDAL (0–360, clockwise from North)
            result = processing.run("gdal:aspect", {
                "INPUT": self._dem_path,
                "BAND": 1,
                "TRIG_ANGLE": False,
                "ZERO_FOR_FLAT": True,
                "COMPUTE_EDGES": True,
                "ZEVENBERGEN": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            })
            aspect_path = result["OUTPUT"]
            if hasattr(aspect_path, "source"):
                aspect_path = aspect_path.source()

            with rasterio.open(self._dem_path) as src:
                cell_size = abs(src.transform.a)
                transform = src.transform

            with rasterio.open(aspect_path) as src:
                aspect = src.read(1).astype("float32")
                rows, cols = aspect.shape

            # One arrow per ~50 m on the ground
            step = max(1, int(50.0 / cell_size))

            layer = QgsVectorLayer("Point", "Slope Direction", "memory")
            layer.setCrs(QgsProject.instance().crs())
            from qgis.PyQt.QtCore import QVariant
            pr = layer.dataProvider()
            pr.addAttributes([QgsField("angle", QVariant.Double)])
            layer.updateFields()

            features = []
            for r in range(0, rows, step):
                for c in range(0, cols, step):
                    val = float(aspect[r, c])
                    if val < 0 or val > 360:   # nodata or flat
                        continue
                    x = transform.c + c * transform.a + cell_size / 2
                    y = transform.f + r * transform.e + cell_size / 2
                    f = QgsFeature()
                    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                    f.setAttributes([val])
                    features.append(f)

            pr.addFeatures(features)
            layer.updateExtents()

            # Arrow marker symbol with data-driven rotation
            symbol = QgsMarkerSymbol.createSimple({
                "name": "arrow",
                "color": "60,60,200,200",
                "outline_color": "20,20,120,200",
                "size": "5",
                "angle": "0",
            })
            symbol.symbolLayer(0).setDataDefinedProperty(
                QgsSymbolLayer.PropertyAngle,
                QgsProperty.fromField("angle")
            )
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))

            QgsProject.instance().addMapLayer(layer)
            self._slope_arrows_layer_id = layer.id()
            self.canvas.refresh()

        except Exception as exc:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not generate slope direction arrows: {exc}"
            )

    def _toggle_before_after(self, show_earthworks):
        for lid in self._baseline_layer_ids:
            node = QgsProject.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(not show_earthworks)
        for lid in self._earthworks_layer_ids:
            node = QgsProject.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(show_earthworks)
        self.canvas.refresh()

    # ---------------------------------------------------------------- Simulation

    def _run_simulation(self):
        # Use earthworks DEM + flow direction when available, fall back to baseline
        if self._modified_dem_path and self._earthworks_result:
            dem_path = self._modified_dem_path
            fdir_path = self._earthworks_result.get("flow_direction")
        else:
            dem_path = self._dem_path
            fdir_path = (self._baseline_result or {}).get("flow_direction")

        if not dem_path or not fdir_path or not os.path.exists(fdir_path):
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run baseline analysis first to generate flow direction. "
                "For earthworks simulation, run 'Re-analyse with Earthworks' first."
            )
            return

        # Build rainfall data
        if self.panel.sim_mode == "uniform":
            rain_mm = self.panel.sim_rainfall_mm
            dur_hr = self.panel.sim_duration_hr
            step_min = self.panel.sim_timestep_min
            n_steps = max(1, int(dur_hr * 60 / step_min))
            rainfall_data = [(0, 0.0)]
            for i in range(1, n_steps + 1):
                t_min = i * step_min
                cum_rain = rain_mm * (t_min / (dur_hr * 60))
                rainfall_data.append((t_min, cum_rain))
        else:
            csv_path = self.panel.sim_csv_path
            if not csv_path or not os.path.exists(csv_path):
                self.iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", "Select a valid CSV file."
                )
                return
            try:
                rainfall_data = SCSRunoff.parse_hyetograph_csv(csv_path)
            except Exception as exc:
                self.iface.messageBar().pushCritical(
                    "TerrainFlow Assessment", f"CSV parse error: {exc}"
                )
                return

        # Build earthwork stores
        from .modules.simulation import SimulationWorker, build_stores_from_earthworks
        stores = build_stores_from_earthworks(
            self._earthwork_manager.get_enabled(),
            soil_name=self.panel.earthwork_soil_name,
            dem_path=dem_path,
        )

        scs = SCSRunoff()
        soil_cn = SOIL_REFERENCE.get(self.panel.soil_name, 70)

        self._sim_worker = SimulationWorker(
            dem_path=dem_path,
            fdir_path=fdir_path,
            output_dir=self._output_dir,
            cn=soil_cn,
            moisture=self.panel.moisture,
            rainfall_data=rainfall_data,
            routing=self.panel.routing,
            earthwork_stores=stores,
            soil_name=self.panel.earthwork_soil_name,
        )
        self._sim_worker.progress.connect(self.panel.set_simulation_progress)
        self._sim_worker.finished.connect(self._on_simulation_complete)
        self._sim_worker.error.connect(lambda tb: self.iface.messageBar().pushCritical(
            "TerrainFlow Assessment", "Simulation failed — see Python console."))
        self._sim_worker.start()

    def _on_simulation_complete(self, result):
        self._sim_result = result
        self.panel.set_simulation_ready(result)

        # Pre-compute global colour-scale maxima so every frame uses the same
        # gradient — otherwise each frame normalises to its own max and the
        # animation looks inconsistent as values grow over time.
        self._sim_global_max_inc = 1.0
        self._sim_global_max_cum = 1.0
        try:
            import rasterio
            # Inc max — use the peak_path raster which is already the per-pixel
            # maximum of all incremental frames.
            peak_path = result.get("peak_path")
            if peak_path and os.path.exists(peak_path):
                with rasterio.open(peak_path) as src:
                    data = src.read(1).astype("float64")
                    data[data == src.nodata] = 0.0
                    v = float(data.max())
                    self._sim_global_max_inc = max(v, 1.0)
            # Cum max — last frame cumulative raster is the highest by definition.
            frames = result.get("frames", [])
            if frames:
                last_cum = frames[-1].get("cum")
                if last_cum and os.path.exists(last_cum):
                    with rasterio.open(last_cum) as src:
                        data = src.read(1).astype("float64")
                        data[data == src.nodata] = 0.0
                        v = float(data.max())
                        self._sim_global_max_cum = max(v, 1.0)
        except Exception:
            pass  # fallback values already set above

        # Build earthwork centroid map for the fill-status layer
        self._sim_ew_centroids = {}
        try:
            import json
            from shapely.geometry import shape as _shp
            for ew in self._earthwork_manager.get_enabled():
                shp = _shp(json.loads(ew.geometry.asJson()))
                c = shp.centroid
                self._sim_ew_centroids[ew.name] = QgsPointXY(c.x, c.y)
        except Exception:
            pass

        # Create (or recreate) the live fill-status layer
        self._create_sim_fill_layer(result)

        # Set up gradual ponding display
        self._setup_sim_ponding()

        # Backfill baseline hydrograph from simulation data.
        # The baseline (no earthworks) outflow at each step = all runoff exits
        # immediately (no storage).  This is stored as outflow_ls_baseline in
        # each timestep row and gives us a proper before/after comparison.
        sim_ts = result.get("timestep_table", [])
        if sim_ts and self._baseline_report:
            self._baseline_report.timestep_table = [
                {"time_hr": r["time_hr"],
                 "outflow_ls": r.get("outflow_ls_baseline", 0.0)}
                for r in sim_ts
            ]
            peak_bl = max((r["outflow_ls"] for r in self._baseline_report.timestep_table),
                          default=0.0)
            if not self._baseline_report.peak_outflow_ls:
                self._baseline_report.peak_outflow_ls = peak_bl

        # Build post-intervention report
        from .modules.reporting import PostInterventionReport
        self._post_report = PostInterventionReport(
            exit_volume_m3=result.get("total_outflow_m3", 0),
            peak_outflow_ls=result.get("peak_outflow_ls", 0),
            peak_outflow_time_hr=result.get("peak_outflow_time_hr", 0),
            earthwork_summary=result.get("earthwork_summary", []),
            timestep_table=result.get("timestep_table", []),
            exit_points=(self._earthworks_result or {}).get("exit_points", []),
        )

        if self._baseline_report and self._post_report:
            from .modules.reporting import compare
            comparison = compare(self._baseline_report, self._post_report)
            self.panel.set_report_summary(comparison)
            self._comparison = comparison

    def _setup_sim_ponding(self):
        """
        Load the ponding capacity raster and build per-earthwork spatial masks.
        Creates a static outline layer (full capacity boundary) and prepares
        for per-frame partial-fill raster updates.
        """
        import numpy as np
        import rasterio
        import json
        from shapely.geometry import shape as _shp
        from rasterio.features import rasterize as _rasterize

        ponding_path = self._ponding_raster_path
        if not ponding_path or not os.path.exists(ponding_path):
            return

        # Load capacity raster
        try:
            with rasterio.open(ponding_path) as src:
                capacity = src.read(1).astype("float32")
                nodata = src.nodata or -9999.0
                capacity[capacity == nodata] = 0.0
                self._sim_ponding_capacity = capacity
                self._sim_ponding_meta = dict(src.meta)
                transform = src.transform
                shape = capacity.shape
        except Exception:
            return

        # Build a spatial mask for each earthwork by rasterizing its geometry
        # buffered by the swale/berm width so ponding cells are captured
        self._sim_ponding_masks = {}
        for ew in self._earthwork_manager.get_enabled():
            try:
                shp = _shp(json.loads(ew.geometry.asJson()))
                buf = getattr(ew, "width", 2.0) or 2.0
                shp_buf = shp.buffer(buf * 5)   # generous buffer to catch depression
                mask = _rasterize(
                    [(shp_buf, 1)],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                )
                # Only keep cells that actually have ponding capacity
                self._sim_ponding_masks[ew.name] = (mask == 1) & (capacity > 0.001)
            except Exception:
                pass

        # Remove any previous frame layer
        for attr in ("_sim_ponding_frame_layer", "_sim_ponding_outline_layer"):
            lyr = getattr(self, attr, None)
            if lyr:
                try:
                    QgsProject.instance().removeMapLayer(lyr)
                except Exception:
                    pass
            setattr(self, attr, None)

        # Create the static full-capacity outline layer (dashed white/teal border)
        crs_str = self._dem_info.crs_wkt if self._dem_info else "EPSG:4326"
        try:
            import tempfile
            outline_path = os.path.join(self._output_dir, "ponding_outline.tif")
            # Write the capacity raster as a binary presence mask
            out_meta = dict(self._sim_ponding_meta)
            out_meta.update(dtype="float32", nodata=-9999.0, count=1)
            outline_data = np.where(capacity > 0.001, 1.0, -9999.0).astype("float32")
            with rasterio.open(outline_path, "w", **out_meta) as dst:
                dst.write(outline_data, 1)

            outline_layer = QgsRasterLayer(outline_path, "Ponding Capacity Outline")
            if outline_layer.isValid():
                from qgis.core import (QgsRasterShader, QgsColorRampShader,
                                       QgsSingleBandPseudoColorRenderer)
                shader = QgsRasterShader()
                cr = QgsColorRampShader()
                cr.setColorRampType(QgsColorRampShader.Exact)
                cr.setColorRampItemList([
                    QgsColorRampShader.ColorRampItem(1.0, QColor(0, 230, 200, 120), "Max capacity"),
                ])
                shader.setRasterShaderFunction(cr)
                renderer = QgsSingleBandPseudoColorRenderer(
                    outline_layer.dataProvider(), 1, shader)
                outline_layer.setRenderer(renderer)
                QgsProject.instance().addMapLayer(outline_layer)
                self._sim_ponding_outline_layer = outline_layer
        except Exception:
            pass

        # Create the live partial-fill raster layer (starts empty)
        self._update_sim_ponding_frame({})

    def _update_sim_ponding_frame(self, fills):
        """
        Write a new partial-fill ponding raster based on current fill fractions
        and refresh the live layer.

        fills : dict  store_name → {"fill_pct": float, "overflowed": bool}
        """
        import numpy as np
        import rasterio

        capacity = self._sim_ponding_capacity
        meta = self._sim_ponding_meta
        masks = self._sim_ponding_masks
        if capacity is None or meta is None:
            return

        # Build partial-fill array: each cell = capacity × fill_fraction for its earthwork
        partial = np.zeros_like(capacity, dtype="float32")
        for name, mask in masks.items():
            if not np.any(mask):
                continue
            fd = fills.get(name)
            fraction = min((fd["fill_pct"] / 100.0), 1.0) if fd else 0.0
            partial[mask] = capacity[mask] * fraction

        # Write to temp raster
        frame_path = os.path.join(self._output_dir, "sim_ponding_frame.tif")
        out_meta = dict(meta)
        out_meta.update(dtype="float32", nodata=-9999.0, count=1)
        data = np.where(partial > 0.001, partial, -9999.0).astype("float32")
        try:
            with rasterio.open(frame_path, "w", **out_meta) as dst:
                dst.write(data, 1)
        except Exception:
            return

        # Remove old frame layer and add fresh one
        if self._sim_ponding_frame_layer:
            try:
                QgsProject.instance().removeMapLayer(self._sim_ponding_frame_layer)
            except Exception:
                pass
            self._sim_ponding_frame_layer = None

        layer = QgsRasterLayer(frame_path, "Ponding Fill")
        if layer.isValid():
            self._apply_ponding_ramp(layer)
            QgsProject.instance().addMapLayer(layer)
            self._sim_ponding_frame_layer = layer

    def _create_sim_fill_layer(self, result):
        """Create the live earthwork fill-status point layer."""
        from qgis.core import QgsVectorLayer, QgsField, QgsMarkerSymbol
        from qgis.PyQt.QtCore import QVariant

        # Remove any previous fill layer
        if self._sim_fill_layer:
            try:
                QgsProject.instance().removeMapLayer(self._sim_fill_layer)
            except Exception:
                pass
            self._sim_fill_layer = None

        if not self._sim_ew_centroids:
            return  # no earthworks with geometry

        crs_str = self._dem_info.crs_wkt if self._dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"Point?crs={crs_str}", "Earthwork Fill Status", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("name",      QVariant.String),
            QgsField("fill_pct",  QVariant.Double),
            QgsField("overflowed", QVariant.Int),
        ])
        layer.updateFields()

        # Add one feature per earthwork (geometry only — attributes updated per frame)
        feats = []
        for name, pt in self._sim_ew_centroids.items():
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(pt))
            f.setAttributes([name, 0.0, 0])
            feats.append(f)
        pr.addFeatures(feats)

        # Data-defined symbol: circle colour green→yellow→red by fill_pct;
        # stroke turns bold red when overflowed.
        sym = QgsMarkerSymbol.createSimple({
            "name": "circle", "size": "7", "color": "0,180,0,220",
            "outline_color": "50,50,50,200", "outline_width": "0.5",
        })
        fill_color_expr = (
            "color_hsv("
            "  scale_linear(\"fill_pct\", 0, 100, 120, 0),"  # hue 120°(green)→0°(red)
            "  90, 85"
            ")"
        )
        outline_expr = (
            "CASE WHEN \"overflowed\" = 1 "
            "THEN color_rgb(220,0,0) "
            "ELSE color_rgb(50,50,50) END"
        )
        outline_width_expr = (
            "CASE WHEN \"overflowed\" = 1 THEN 2.5 ELSE 0.5 END"
        )
        sym.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyFillColor, QgsProperty.fromExpression(fill_color_expr))
        sym.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(outline_expr))
        sym.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth, QgsProperty.fromExpression(outline_width_expr))
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

        # Labels: "Name\n42%" — bold, white halo
        from qgis.PyQt.QtGui import QFont
        from qgis.core import QgsTextBufferSettings
        text_fmt = QgsTextFormat()
        font = QFont(); font.setBold(True); font.setPointSize(8)
        text_fmt.setFont(font)
        text_fmt.setColor(QColor(20, 20, 20))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True); buf.setColor(QColor(255, 255, 255)); buf.setSize(1.2)
        text_fmt.setBuffer(buf)
        lbl = QgsPalLayerSettings()
        lbl.fieldName = "concat(\"name\", '\\n', round(\"fill_pct\", 0), '%')"
        lbl.isExpression = True
        lbl.enabled = True
        lbl.setFormat(text_fmt)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
        layer.setLabelsEnabled(True)

        QgsProject.instance().addMapLayer(layer)
        self._sim_fill_layer = layer

    def _show_sim_frame(self, idx):
        if not self._sim_result:
            return
        frames = self._sim_result.get("frames", [])
        if not frames or idx >= len(frames):
            return
        frame = frames[idx]
        mode = self.panel.sim_display_mode

        # --- Raster frame ---
        path = frame.get("inc" if mode == "inc" else "cum")
        if path and os.path.exists(path):
            existing = QgsProject.instance().mapLayersByName("Simulation Frame")
            for lyr in existing:
                QgsProject.instance().removeMapLayer(lyr)
            layer = QgsRasterLayer(path, "Simulation Frame")
            if layer.isValid():
                global_max = (
                    getattr(self, "_sim_global_max_inc", None)
                    if mode == "inc"
                    else getattr(self, "_sim_global_max_cum", None)
                )
                self._apply_stream_ramp(layer, global_max)
                QgsProject.instance().addMapLayer(layer)

        # --- Time label + overflow annotation ---
        time_labels = self._sim_result.get("time_labels", [])
        time_str = time_labels[idx] if idx < len(time_labels) else "—"

        fills = frame.get("fills", {})
        new_overflows = [name for name, fd in fills.items() if fd.get("first_overflow_this_step")]
        if new_overflows:
            overflow_str = "  ⚠ OVERFLOW: " + ", ".join(new_overflows)
        else:
            overflow_str = ""
        self.panel.set_sim_time_label(time_str + overflow_str)

        # --- Update fill-status layer ---
        fill_layer = getattr(self, "_sim_fill_layer", None)
        centroids = getattr(self, "_sim_ew_centroids", {})
        if fill_layer and QgsProject.instance().mapLayer(fill_layer.id()) and centroids:
            pr = fill_layer.dataProvider()
            pr.truncate()
            feats = []
            for name, pt in centroids.items():
                fd = fills.get(name, {"fill_pct": 0.0, "overflowed": False})
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPointXY(pt))
                f.setAttributes([name, fd["fill_pct"], 1 if fd["overflowed"] else 0])
                feats.append(f)
            pr.addFeatures(feats)
            fill_layer.triggerRepaint()

        # --- Update gradual ponding raster ---
        if getattr(self, "_sim_ponding_capacity", None) is not None:
            self._update_sim_ponding_frame(fills)

    def _on_sim_play_toggled(self, playing):
        if playing:
            self._sim_timer.start(500)
        else:
            self._sim_timer.stop()

    def _advance_sim_frame(self):
        if not self._sim_result:
            return
        n = len(self._sim_result.get("frames", []))
        slider = self.panel._sim_slider
        next_idx = slider.value() + 1
        if next_idx >= n:
            next_idx = 0
        slider.setValue(next_idx)

    # ---------------------------------------------------------------- Report export

    def _export_report(self):
        if not hasattr(self, "_comparison") or self._comparison is None:
            self.iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run baseline + simulation first to generate a comparison."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(),
            "Save Report",
            os.path.expanduser("~/terrainflow_report.html"),
            "HTML Files (*.html)",
        )
        if not path:
            return

        from .modules.reporting import export_html
        try:
            export_html(self._comparison, path)
            self.iface.messageBar().pushSuccess(
                "TerrainFlow Assessment",
                f"Report saved to {path}"
            )
            import subprocess, sys
            if sys.platform == "win32":
                os.startfile(path)
        except Exception as exc:
            self.iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Export failed: {exc}"
            )

    # ---------------------------------------------------------------- Layer styling

    def _apply_stream_ramp(self, layer, max_acc=None):
        from qgis.core import (QgsRasterShader, QgsColorRampShader,
                               QgsSingleBandPseudoColorRenderer)
        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        if max_acc is None:
            try:
                stats = layer.dataProvider().bandStatistics(1)
                max_acc = stats.maximumValue or 1.0
            except Exception:
                max_acc = 1.0
        color_ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(0, QColor(220, 235, 255, 0), "0"),
            QgsColorRampShader.ColorRampItem(max_acc * 0.3, QColor(100, 160, 230), "low"),
            QgsColorRampShader.ColorRampItem(max_acc, QColor(20, 60, 150), "high"),
        ])
        shader.setRasterShaderFunction(color_ramp)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        layer.setRenderer(renderer)

    def _apply_ponding_ramp(self, layer):
        from qgis.core import (QgsRasterShader, QgsColorRampShader,
                               QgsSingleBandPseudoColorRenderer)
        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        try:
            stats = layer.dataProvider().bandStatistics(1)
            max_v = stats.maximumValue or 1.0
        except Exception:
            max_v = 1.0
        color_ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(0, QColor(180, 220, 255, 0), "0"),
            QgsColorRampShader.ColorRampItem(max_v * 0.5, QColor(80, 160, 240, 160), "mid"),
            QgsColorRampShader.ColorRampItem(max_v, QColor(0, 40, 180, 220), "max"),
        ])
        shader.setRasterShaderFunction(color_ramp)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        layer.setRenderer(renderer)
