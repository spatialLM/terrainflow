"""
simulation.py — SimulationController

Handles: simulation run, per-frame display, ponding visualisation, fill-status
layer, play/pause timer, and post-intervention report building.
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
    QgsProperty,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsSymbolLayer,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType, QTimer
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.modules.catchment import SCSRunoff
from terrainflow_assessment.modules.swale_design import SOIL_REFERENCE
from terrainflow_assessment.qgis.workers.simulation_worker import SimulationWorker


class SimulationController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

        self._sim_timer = QTimer()
        self._sim_timer.timeout.connect(self._advance_sim_frame)

    # ---------------------------------------------------------------- Run

    def run_simulation(self):
        # Use earthworks DEM + flow direction when available, fall back to baseline
        if self._state.modified_dem_path and self._state.earthworks_result:
            dem_path = self._state.modified_dem_path
            fdir_path = self._state.earthworks_result.get("flow_direction")
        else:
            dem_path = self._state.dem_path
            fdir_path = (self._state.baseline_result or {}).get("flow_direction")

        if not dem_path or not fdir_path or not os.path.exists(fdir_path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run baseline analysis first to generate flow direction. "
                "For earthworks simulation, run 'Re-analyse with Earthworks' first."
            )
            return

        # Build rainfall data
        if self._panel.sim_mode == "uniform":
            rain_mm = self._panel.sim_rainfall_mm
            dur_hr = self._panel.sim_duration_hr
            step_min = self._panel.sim_timestep_min
            n_steps = max(1, int(dur_hr * 60 / step_min))
            rainfall_data = [(0, 0.0)]
            for i in range(1, n_steps + 1):
                t_min = i * step_min
                cum_rain = rain_mm * (t_min / (dur_hr * 60))
                rainfall_data.append((t_min, cum_rain))
        else:
            csv_path = self._panel.sim_csv_path
            if not csv_path or not os.path.exists(csv_path):
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", "Select a valid CSV file."
                )
                return
            try:
                rainfall_data = SCSRunoff.parse_hyetograph_csv(csv_path)
            except Exception as exc:
                self._iface.messageBar().pushCritical(
                    "TerrainFlow Assessment", f"CSV parse error: {exc}"
                )
                return

        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        stores = build_stores_from_earthworks(
            self._state.earthwork_manager.get_enabled(),
            soil_name=self._panel.earthwork_soil_name,
            dem_path=dem_path,
        )

        soil_cn = SOIL_REFERENCE.get(self._panel.soil_name, 70)

        self._state.sim_worker = SimulationWorker(
            dem_path=dem_path,
            fdir_path=fdir_path,
            output_dir=self._state.output_dir,
            cn=soil_cn,
            moisture=self._panel.moisture,
            rainfall_data=rainfall_data,
            routing=self._panel.routing,
            earthwork_stores=stores,
            soil_name=self._panel.earthwork_soil_name,
        )
        self._state.sim_worker.progress.connect(self._panel.set_simulation_progress)
        self._state.sim_worker.finished.connect(self._on_simulation_complete)
        self._state.sim_worker.error.connect(
            lambda tb: self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", "Simulation failed — see Python console."
            )
        )
        self._state.sim_worker.start()

    # ---------------------------------------------------------------- Completion

    def _on_simulation_complete(self, result):
        self._state.sim_result = result
        self._panel.set_simulation_ready(result)

        # Pre-compute global colour-scale maxima
        self._state.sim_global_max_inc = 1.0
        self._state.sim_global_max_cum = 1.0
        try:
            import rasterio
            peak_path = result.get("peak_path")
            if peak_path and os.path.exists(peak_path):
                with rasterio.open(peak_path) as src:
                    data = src.read(1).astype("float64")
                    data[data == src.nodata] = 0.0
                    v = float(data.max())
                    self._state.sim_global_max_inc = max(v, 1.0)
            frames = result.get("frames", [])
            if frames:
                last_cum = frames[-1].get("cum")
                if last_cum and os.path.exists(last_cum):
                    with rasterio.open(last_cum) as src:
                        data = src.read(1).astype("float64")
                        data[data == src.nodata] = 0.0
                        v = float(data.max())
                        self._state.sim_global_max_cum = max(v, 1.0)
        except Exception:
            pass

        # Build earthwork centroid map for the fill-status layer
        self._state.sim_ew_centroids = {}
        try:
            import json

            from shapely.geometry import shape as _shp
            for ew in self._state.earthwork_manager.get_enabled():
                shp = _shp(json.loads(ew.geometry.asJson()))
                c = shp.centroid
                self._state.sim_ew_centroids[ew.name] = QgsPointXY(c.x, c.y)
        except Exception:
            pass

        self._create_sim_fill_layer(result)
        self._setup_sim_ponding()

        # Backfill baseline hydrograph
        sim_ts = result.get("timestep_table", [])
        if sim_ts and self._state.baseline_report:
            self._state.baseline_report.timestep_table = [
                {"time_hr": r["time_hr"],
                 "outflow_ls": r.get("outflow_ls_baseline", 0.0)}
                for r in sim_ts
            ]
            peak_bl = max(
                (r["outflow_ls"] for r in self._state.baseline_report.timestep_table),
                default=0.0,
            )
            if not self._state.baseline_report.peak_outflow_ls:
                self._state.baseline_report.peak_outflow_ls = peak_bl

        from terrainflow_assessment.modules.reporting import PostInterventionReport, compare
        self._state.post_report = PostInterventionReport(
            exit_volume_m3=result.get("total_outflow_m3", 0),
            peak_outflow_ls=result.get("peak_outflow_ls", 0),
            peak_outflow_time_hr=result.get("peak_outflow_time_hr", 0),
            earthwork_summary=result.get("earthwork_summary", []),
            timestep_table=result.get("timestep_table", []),
            exit_points=(self._state.earthworks_result or {}).get("exit_points", []),
        )

        if self._state.baseline_report and self._state.post_report:
            comparison = compare(self._state.baseline_report, self._state.post_report)
            self._panel.set_report_summary(comparison)
            self._state.comparison = comparison

    # ---------------------------------------------------------------- Ponding

    def _setup_sim_ponding(self):
        import json

        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rasterize
        from shapely.geometry import shape as _shp

        ponding_path = self._state.ponding_raster_path
        if not ponding_path or not os.path.exists(ponding_path):
            return

        try:
            with rasterio.open(ponding_path) as src:
                capacity = src.read(1).astype("float32")
                nodata = src.nodata or -9999.0
                capacity[capacity == nodata] = 0.0
                self._state.sim_ponding_capacity = capacity
                self._state.sim_ponding_meta = dict(src.meta)
                transform = src.transform
                shape = capacity.shape
        except Exception:
            return

        self._state.sim_ponding_masks = {}
        for ew in self._state.earthwork_manager.get_enabled():
            try:
                shp = _shp(json.loads(ew.geometry.asJson()))
                buf = getattr(ew, "width", 2.0) or 2.0
                shp_buf = shp.buffer(buf * 5)
                mask = _rasterize(
                    [(shp_buf, 1)],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                )
                self._state.sim_ponding_masks[ew.name] = (mask == 1) & (capacity > 0.001)
            except Exception:
                pass

        # Remove any previous frame/outline layers
        for attr in ("sim_ponding_frame_layer", "sim_ponding_outline_layer"):
            lyr = getattr(self._state, attr, None)
            if lyr:
                try:
                    self._project.instance().removeMapLayer(lyr)
                except Exception:
                    pass
            setattr(self._state, attr, None)

        # Create static full-capacity outline layer
        try:
            outline_path = os.path.join(self._state.output_dir, "ponding_outline.tif")
            out_meta = dict(self._state.sim_ponding_meta)
            out_meta.update(dtype="float32", nodata=-9999.0, count=1)
            import numpy as np
            outline_data = np.where(capacity > 0.001, 1.0, -9999.0).astype("float32")
            with rasterio.open(outline_path, "w", **out_meta) as dst:
                dst.write(outline_data, 1)

            outline_layer = QgsRasterLayer(outline_path, "Ponding Capacity Outline")
            if outline_layer.isValid():
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
                self._project.instance().addMapLayer(outline_layer)
                self._state.sim_ponding_outline_layer = outline_layer
        except Exception:
            pass

        self._update_sim_ponding_frame({})

    def _update_sim_ponding_frame(self, fills):
        import numpy as np
        import rasterio

        capacity = self._state.sim_ponding_capacity
        meta = self._state.sim_ponding_meta
        masks = self._state.sim_ponding_masks
        if capacity is None or meta is None:
            return

        partial = np.zeros_like(capacity, dtype="float32")
        for name, mask in masks.items():
            if not np.any(mask):
                continue
            fd = fills.get(name)
            fraction = min((fd["fill_pct"] / 100.0), 1.0) if fd else 0.0
            partial[mask] = capacity[mask] * fraction

        frame_path = os.path.join(self._state.output_dir, "sim_ponding_frame.tif")
        out_meta = dict(meta)
        out_meta.update(dtype="float32", nodata=-9999.0, count=1)
        data = np.where(partial > 0.001, partial, -9999.0).astype("float32")
        try:
            with rasterio.open(frame_path, "w", **out_meta) as dst:
                dst.write(data, 1)
        except Exception:
            return

        if self._state.sim_ponding_frame_layer:
            try:
                self._project.instance().removeMapLayer(self._state.sim_ponding_frame_layer)
            except Exception:
                pass
            self._state.sim_ponding_frame_layer = None

        layer = QgsRasterLayer(frame_path, "Ponding Fill")
        if layer.isValid():
            self._apply_ponding_ramp(layer)
            self._project.instance().addMapLayer(layer)
            self._state.sim_ponding_frame_layer = layer

    def _apply_ponding_ramp(self, layer):
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

    # ---------------------------------------------------------------- Fill layer

    def _create_sim_fill_layer(self, result):
        if self._state.sim_fill_layer:
            try:
                self._project.instance().removeMapLayer(self._state.sim_fill_layer)
            except Exception:
                pass
            self._state.sim_fill_layer = None

        if not self._state.sim_ew_centroids:
            return

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"Point?crs={crs_str}", "Earthwork Fill Status", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("name",       QMetaType.QString),
            QgsField("fill_pct",   QMetaType.Double),
            QgsField("overflowed", QMetaType.Int),
        ])
        layer.updateFields()

        feats = []
        for name, pt in self._state.sim_ew_centroids.items():
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(pt))
            f.setAttributes([name, 0.0, 0])
            feats.append(f)
        pr.addFeatures(feats)

        sym = QgsMarkerSymbol.createSimple({
            "name": "circle", "size": "7", "color": "0,180,0,220",
            "outline_color": "50,50,50,200", "outline_width": "0.5",
        })
        fill_color_expr = (
            "color_hsv("
            "  scale_linear(\"fill_pct\", 0, 100, 120, 0),"
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

        from qgis.core import QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont
        text_fmt = QgsTextFormat()
        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        text_fmt.setFont(font)
        text_fmt.setColor(QColor(20, 20, 20))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor(255, 255, 255))
        buf.setSize(1.2)
        text_fmt.setBuffer(buf)
        lbl = QgsPalLayerSettings()
        lbl.fieldName = "concat(\"name\", '\\n', round(\"fill_pct\", 0), '%')"
        lbl.isExpression = True
        lbl.enabled = True
        lbl.setFormat(text_fmt)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
        layer.setLabelsEnabled(True)

        self._project.instance().addMapLayer(layer)
        self._state.sim_fill_layer = layer

    # ---------------------------------------------------------------- Frame display

    def show_sim_frame(self, idx):
        if not self._state.sim_result:
            return
        frames = self._state.sim_result.get("frames", [])
        if not frames or idx >= len(frames):
            return
        frame = frames[idx]
        mode = self._panel.sim_display_mode

        path = frame.get("inc" if mode == "inc" else "cum")
        if path and os.path.exists(path):
            existing = self._project.instance().mapLayersByName("Simulation Frame")
            for lyr in existing:
                self._project.instance().removeMapLayer(lyr)
            layer = QgsRasterLayer(path, "Simulation Frame")
            if layer.isValid():
                global_max = (
                    self._state.sim_global_max_inc
                    if mode == "inc"
                    else self._state.sim_global_max_cum
                )
                self._apply_stream_ramp(layer, global_max)
                self._project.instance().addMapLayer(layer)

        time_labels = self._state.sim_result.get("time_labels", [])
        time_str = time_labels[idx] if idx < len(time_labels) else "—"

        fills = frame.get("fills", {})
        new_overflows = [name for name, fd in fills.items() if fd.get("first_overflow_this_step")]
        overflow_str = ("  ⚠ OVERFLOW: " + ", ".join(new_overflows)) if new_overflows else ""
        self._panel.set_sim_time_label(time_str + overflow_str)

        fill_layer = self._state.sim_fill_layer
        centroids = self._state.sim_ew_centroids
        if fill_layer and self._project.instance().mapLayer(fill_layer.id()) and centroids:
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

        if self._state.sim_ponding_capacity is not None:
            self._update_sim_ponding_frame(fills)

    def _apply_stream_ramp(self, layer, max_acc=None):
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

    # ---------------------------------------------------------------- Timer

    def on_sim_play_toggled(self, playing):
        if playing:
            self._sim_timer.start(500)
        else:
            self._sim_timer.stop()

    def _advance_sim_frame(self):
        if not self._state.sim_result:
            return
        n = len(self._state.sim_result.get("frames", []))
        slider = self._panel._sim_slider
        next_idx = slider.value() + 1
        if next_idx >= n:
            next_idx = 0
        slider.setValue(next_idx)
