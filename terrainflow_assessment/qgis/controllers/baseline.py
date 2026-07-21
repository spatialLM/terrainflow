"""
baseline.py — BaselineController

Handles: DEM/boundary loading, baseline analysis (AnalysisWorker), result
layer creation, before/after toggling, and shared raster styling helpers.
"""

from __future__ import annotations

import os

from qgis.core import (
    QgsColorRampShader,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.modules.dem_loader import compute_slope_raster, load_dem
from terrainflow_assessment.modules.earthwork_design import DEMBurner
from terrainflow_assessment.modules.reporting import BaselineReport
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker


def _crs_label(crs):
    if crs is None:
        return "unknown CRS"
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    return crs.to_string()


class BaselineController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    # ---------------------------------------------------------------- DEM / boundary

    def on_dem_changed(self, layer):
        if layer is None:
            self._state.dem_path = None
            self._state.dem_info = None
            return
        try:
            path = layer.source()
            info = load_dem(path)
            self._state.dem_path = path
            self._state.dem_info = info
            self._state.burner = DEMBurner(path)
            self._panel.set_dem_info(
                f"Cell: {info.cell_size_m:.2f} m | Area: {info.area_ha:.1f} ha | "
                f"{_crs_label(info.crs)}"
            )
            slope_path = os.path.join(self._state.output_dir, "slope.tif")
            compute_slope_raster(path, slope_path)
            self._state.slope_raster_path = slope_path
        except Exception as exc:
            self._iface.messageBar().pushWarning("TerrainFlow Assessment",
                                                  f"DEM load error: {exc}")

    def on_boundary_changed(self, layer):
        if layer is None:
            self._state.boundary_path = None
            return
        try:
            self._state.boundary_path = self._layer_to_path(layer)
        except Exception:
            self._state.boundary_path = None

    def on_analysis_area_changed(self, layer):
        self._state.analysis_area_path = self._layer_to_path(layer) if layer else None

    def on_earthworks_area_changed(self, layer):
        self._state.earthworks_area_path = self._layer_to_path(layer) if layer else None

    def _layer_to_path(self, layer):
        if layer is None:
            return None
        src = layer.source()
        if os.path.exists(src.split("|")[0]):
            return src.split("|")[0]
        import tempfile
        path = tempfile.mktemp(suffix=".gpkg")
        from qgis.core import QgsVectorFileWriter
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "GPKG"
        save_options.fileEncoding = "UTF-8"
        QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, path,
            self._project.transform_context(),
            save_options,
        )
        return path

    # ---------------------------------------------------------------- Baseline analysis

    def run_baseline(self):
        if not self._state.dem_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Please select a DEM first."
            )
            return

        cell_area_m2 = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
        ha_threshold = self._panel.stream_threshold_ha
        threshold_cells = int(ha_threshold * 10_000 / cell_area_m2) if cell_area_m2 > 0 else 1000

        self._state.analysis_worker = AnalysisWorker(
            dem_path=self._state.dem_path,
            output_dir=self._state.output_dir,
            stream_threshold=threshold_cells,
            cn=self._panel.cn,
            moisture=self._panel.moisture,
            rainfall_mm=self._panel.rainfall_mm,
            duration_hours=self._panel.duration_hr,
            boundary_path=self._state.boundary_path,
            label="baseline",
            run_catchments=True,
            threshold_mode="cells",
            routing=self._panel.routing,
        )
        self._state.analysis_worker.progress.connect(self._panel.set_baseline_progress)
        self._state.analysis_worker.finished.connect(self._on_baseline_complete)
        self._state.analysis_worker.error.connect(self._on_analysis_error)
        self._state.analysis_worker.start()

    def _on_baseline_complete(self, result):
        self._state.baseline_result = result
        self._load_result_layers(result, is_earthworks=False)

        catchment_ha = result.get("catchment_area_m2", 0) / 10_000.0
        self._state.baseline_report = BaselineReport(
            site_name=self._panel.site_name,
            dem_path=self._state.dem_path,
            crs=_crs_label(self._state.dem_info.crs) if self._state.dem_info else "",
            cell_size_m=self._state.dem_info.cell_size_m if self._state.dem_info else 1.0,
            catchment_area_ha=catchment_ha,
            rainfall_mm=self._panel.rainfall_mm,
            duration_hr=self._panel.duration_hr,
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
        self._panel.set_baseline_complete(summary)

    def _on_analysis_error(self, tb):
        self._panel.set_baseline_complete("Analysis failed — see Python console for details.")
        print("TerrainFlow Assessment — Analysis error:\n" + tb)
        self._iface.messageBar().pushCritical("TerrainFlow Assessment",
                                               "Analysis failed. See Python console.")

    # ---------------------------------------------------------------- Layer loading

    def _load_result_layers(self, result, is_earthworks=False):
        label = "Earthworks" if is_earthworks else "Baseline"
        layer_ids = []

        stream_path = result.get("stream_network")
        if stream_path and os.path.exists(stream_path):
            layer = QgsRasterLayer(stream_path, f"{label} — Streams")
            if layer.isValid():
                self.apply_stream_ramp(layer, result.get("stream_acc_max", 1))
                self._project.instance().addMapLayer(layer)
                layer_ids.append(layer.id())

        ponding_path = result.get("ponding")
        if ponding_path and os.path.exists(ponding_path):
            self._state.ponding_raster_path = ponding_path
            layer = QgsRasterLayer(ponding_path, f"{label} — Water Captured")
            if layer.isValid():
                self.apply_ponding_ramp(layer)
                self._project.instance().addMapLayer(layer)
                layer_ids.append(layer.id())

        exit_points = result.get("exit_points", [])
        if exit_points:
            ep_layer = self._create_exit_points_layer(exit_points, label)
            if ep_layer:
                self._project.instance().addMapLayer(ep_layer)
                layer_ids.append(ep_layer.id())

        if is_earthworks:
            self._state.earthworks_layer_ids = layer_ids
        else:
            self._state.baseline_layer_ids = layer_ids

    def _create_exit_points_layer(self, exit_points, label):
        from qgis.core import QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont

        layer = QgsVectorLayer(
            "Point?crs=" + (self._state.dem_info.crs_wkt or "EPSG:4326"),
            f"{label} — Exit Points", "memory",
        )
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label", QMetaType.QString),
            QgsField("volume_m3", QMetaType.Double),
            QgsField("flow_ls", QMetaType.Double),
        ])
        layer.updateFields()

        feats = []
        for ep in exit_points:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(ep["x"], ep["y"])))
            f.setAttributes([ep.get("label", ""), ep.get("volume_m3", 0), ep.get("flow_ls", 0)])
            feats.append(f)
        pr.addFeatures(feats)

        symbol = QgsMarkerSymbol.createSimple({
            "name": "circle", "color": "220,0,0,200",
            "outline_color": "140,0,0", "size": "5",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

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

    # ---------------------------------------------------------------- Before/after toggle

    def toggle_before_after(self, show_earthworks):
        for lid in self._state.baseline_layer_ids:
            node = self._project.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(not show_earthworks)
        for lid in self._state.earthworks_layer_ids:
            node = self._project.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(show_earthworks)
        self._canvas.refresh()

    # ---------------------------------------------------------------- Shared raster styling

    def apply_stream_ramp(self, layer, max_acc=None):
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

    def apply_ponding_ramp(self, layer):
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
