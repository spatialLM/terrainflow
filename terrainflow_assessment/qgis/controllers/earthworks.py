"""
earthworks.py — EarthworksController

Handles: earthwork drawing tools, properties dialog, layer management,
DEM re-analysis with earthworks, ponding query, slope visualisation.
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
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.map_tools.contour_segment_tool import ContourSegmentTool
from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool
from terrainflow_assessment.map_tools.draw_polygon_tool import DrawPolygonTool
from terrainflow_assessment.map_tools.ponding_query_tool import PondingQueryTool
from terrainflow_assessment.map_tools.select_contour_tool import SelectContourTool
from terrainflow_assessment.modules.earthwork_design import (
    Earthwork,
    calculate_capacity,
)
from terrainflow_assessment.modules.swale_design import (
    contour_to_swale_geometry,
    sample_peak_inflow,
)
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker

# Per-type styles: (geometry_type, display_name, colour, fill or None, line width)
_EW_STYLES = {
    "swale":     ("LineString", "Swales",     "#00BCD4", None,      "2.5"),
    "berm":      ("LineString", "Berms",      "#FF6D00", None,      "2.5"),
    "dam":       ("LineString", "Dams",       "#E53935", None,      "3.0"),
    "diversion": ("LineString", "Diversions", "#AB47BC", None,      "2.0"),
    "basin":     ("Polygon",    "Basins",     "#1565C0", "#1565C0", "1.5"),
}


class EarthworksController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    # ---------------------------------------------------------------- Drawing tools

    def activate_draw_swale(self, mode):
        if mode == "contour":
            if not self._state.contour_layer:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "Run contour analysis first, then pick a segment on a contour.",
                )
                return
            tool = ContourSegmentTool(self._canvas, self._state.contour_layer)
            tool.segment_selected.connect(
                lambda geom, elev: self._on_contour_selected_for_swale(geom, elev)
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)
        elif mode == "full_contour":
            if not self._state.contour_layer:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "Run contour analysis first, then click a contour.",
                )
                return
            tool = SelectContourTool(self._canvas, self._state.contour_layer)
            tool.contour_selected.connect(
                lambda geom, elev: self._on_contour_selected_for_swale(geom, elev)
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)
        else:
            tool = DrawLineTool(self._canvas,
                                slope_raster_path=self._state.slope_raster_path,
                                tool_label="swale")
            tool.line_drawn.connect(lambda geom: self._on_geometry_drawn("swale", geom))
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)

    def _on_contour_selected_for_swale(self, geom, elevation):
        swale_geom = contour_to_swale_geometry(geom)
        self._on_geometry_drawn("swale", swale_geom)

    def activate_draw_line(self, ew_type):
        tool = DrawLineTool(self._canvas,
                            slope_raster_path=self._state.slope_raster_path,
                            tool_label=ew_type)
        tool.line_drawn.connect(lambda geom: self._on_geometry_drawn(ew_type, geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)

    def activate_draw_basin(self):
        tool = DrawPolygonTool(self._canvas,
                               slope_raster_path=self._state.slope_raster_path,
                               tool_label="basin")
        tool.polygon_drawn.connect(lambda geom: self._on_geometry_drawn("basin", geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)

    def on_usable_area_source_changed(self, source):
        import json

        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union

        if source == "none":
            self._state.usable_polygon = None
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment", "Usable area cleared — full DEM will be used."
            )
            return

        layer = (
            self._panel.analysis_area_layer if source == "analysis"
            else self._panel.earthworks_area_layer
        )
        if layer is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "No layer selected for that area — set it in the Data section first.",
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
            self._state.usable_polygon = unary_union(polys)
            label = "Analysis Area" if source == "analysis" else "Earthworks Area"
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"Usable area set from '{label}'. Run contour analysis to apply.",
            )
        except Exception as exc:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not read usable area layer: {exc}"
            )

    def _on_geometry_drawn(self, ew_type, geometry):
        from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

        peak_inflow = 0.0
        if self._state.baseline_result:
            acc_path = self._state.baseline_result.get("flow_accumulation")
            if acc_path:
                acc_cells = sample_peak_inflow(geometry, acc_path)
                cell_area = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
                runoff_mm = self._state.baseline_result.get("runoff_mm", 0)
                peak_inflow = acc_cells * cell_area * runoff_mm / 1000.0

        crest_elev = None
        if ew_type == "dam" and self._state.dem_path:
            try:
                import json

                import rasterio
                from shapely.geometry import shape as shapely_shape
                shp = shapely_shape(json.loads(geometry.asJson()))
                centroid = shp.centroid
                with rasterio.open(self._state.dem_path) as src:
                    t = src.transform
                    col = int((centroid.x - t.c) / t.a)
                    row = int((centroid.y - t.f) / t.e)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        crest_elev = float(src.read(1)[row, col]) + 2.0
            except Exception:
                pass

        n = len(self._state.earthwork_manager) + 1
        ew_name = f"{ew_type.capitalize()} {n}"
        ew = Earthwork(ew_type, geometry, ew_name)

        dlg = EarthworkPropertiesDialog(
            ew_type=ew_type,
            geometry=geometry,
            parent=self._iface.mainWindow(),
            earthwork=ew,
            peak_inflow_m3=peak_inflow,
            crest_elevation=crest_elev,
            duration_hours=self._panel.duration_hr,
            dem_path=self._state.dem_path if ew_type == "dam" else None,
            soil_name=self._panel.earthwork_soil_name,
            cn=self._panel.cn,
        )

        if dlg.exec():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            ew.width = getattr(dlg, "get_width", lambda: ew.width)()
            if ew_type == "dam":
                ew.crest_elevation = dlg.get_crest_elevation()
                ew.key_into_banks = getattr(dlg, "get_key_into_banks", lambda: False)()
            elif ew_type == "swale":
                ew.companion_berm = getattr(dlg, "get_companion_berm", lambda: False)()
            # Apply the bottom width (channels only; None otherwise) — the canonical
            # cross-section field that drives capacity and the burn footprint.
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw

            if ew_type == "dam":
                ew.capacity_m3 = self._compute_dam_capacity(ew)
                ew.capacity_l = ew.capacity_m3 * 1000.0
            else:
                ew.capacity_m3, ew.capacity_l = calculate_capacity(
                    ew_type, geometry, ew.depth, ew.width,
                    getattr(ew, "companion_berm", False),
                    bottom_width=getattr(ew, "bottom_width_m", None),
                )
            self._state.earthwork_manager.add(ew)
            self._panel.add_earthwork_to_list(
                len(self._state.earthwork_manager) - 1, ew.summary()
            )
            self._refresh_ew_layer()
            self._recompute_live_assessment()
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def _on_draw_cancelled(self):
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def edit_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        ew = self._state.earthwork_manager.get(idx)
        from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog
        dlg = EarthworkPropertiesDialog(
            ew_type=ew.type,
            geometry=ew.geometry,
            parent=self._iface.mainWindow(),
            earthwork=ew,
            duration_hours=self._panel.duration_hr,
            dem_path=self._state.dem_path if ew.type == "dam" else None,
            soil_name=self._panel.earthwork_soil_name,
            cn=self._panel.cn,
        )
        if dlg.exec():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            # Re-read every edited dimension (the edit path previously dropped
            # width / companion / gradient / side-slope changes silently).
            ew.width = getattr(dlg, "get_width", lambda: ew.width)()
            if ew.type == "dam":
                ew.crest_elevation = dlg.get_crest_elevation()
                ew.key_into_banks = getattr(dlg, "get_key_into_banks", lambda: False)()
            elif ew.type == "swale":
                ew.companion_berm = getattr(dlg, "get_companion_berm", lambda: False)()
            elif ew.type == "diversion":
                ew.gradient_pct = getattr(dlg, "get_gradient_pct", lambda: ew.gradient_pct)()
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw
            if ew.type == "dam":
                ew.capacity_m3 = self._compute_dam_capacity(ew)
                ew.capacity_l = ew.capacity_m3 * 1000.0
            else:
                ew.capacity_m3, ew.capacity_l = calculate_capacity(
                    ew.type, ew.geometry, ew.depth, ew.width,
                    getattr(ew, "companion_berm", False),
                    bottom_width=getattr(ew, "bottom_width_m", None),
                )
            self._panel.update_earthwork_in_list(idx, ew.summary())
            self._refresh_ew_layer()
            self._recompute_live_assessment()

    def delete_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.remove(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())
        self._refresh_ew_layer()
        self._recompute_live_assessment()

    def toggle_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.toggle(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())
        self._recompute_live_assessment()

    # ---------------------------------------------------------------- Dam analytical capacity

    def _compute_dam_capacity(self, ew):
        """One-time DEM flood → dam impounded volume (m³), cached on ``ew.capacity_m3``.

        Heavy (a depression-fill), so run only at dam draw/edit — the live readout reads
        the cached value. Returns 0 without a DEM or a crest; failures degrade to 0.
        """
        if not self._state.dem_path or getattr(ew, "crest_elevation", None) is None:
            return 0.0
        try:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment", "Computing dam storage…"
            )
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            burner = self._state.burner or DEMBurner(self._state.dem_path)
            baseline_ponding = self._cached_baseline_ponding(burner.shape)
            volume = round(
                burner.dam_stage_storage(
                    ew, baseline_ponding=baseline_ponding,
                    key_into_banks=getattr(ew, "key_into_banks", False),
                ),
                2,
            )
            for msg in getattr(burner, "warnings", []):
                self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)
            return volume
        except Exception as exc:
            print(f"TerrainFlow Assessment — dam storage error: {exc}")
            return 0.0

    def _cached_baseline_ponding(self, shape):
        """Baseline ponding array (saves a flood pass), or None if unavailable/mismatched."""
        path = (self._state.baseline_result or {}).get("ponding")
        if not path or not os.path.exists(path):
            return None
        try:
            import numpy as np
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                nodata = src.nodata
            if arr.shape != shape:
                return None
            if nodata is not None:
                arr[arr == nodata] = 0.0
            return np.clip(arr, 0.0, None)
        except Exception:
            return None

    # ---------------------------------------------------------------- Live analytical assessment

    def _recompute_live_assessment(self):
        """Design-tier: recompute the live analytical water balance → panel readout.

        Fast, no burn. Geometry metrics (capacity/cut/fill) always; storm capture %
        once a baseline (flow accumulation) exists. Runoff is recomputed live from the
        current storm/soil inputs, so the score reacts without re-running baseline.
        Fires on every earthwork edit and storm/soil change; failures are swallowed so
        the readout never breaks the edit flow.
        """
        try:
            from terrainflow_assessment.modules.catchment import SCSRunoff
            from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
            from terrainflow_assessment.modules.swale_design import sample_peak_inflow
            from terrainflow_assessment.modules.water_balance import run_water_balance

            enabled = [
                ew for ew in self._state.earthwork_manager.get_enabled()
                if getattr(ew, "capacity_m3", 0.0) > 0
            ]
            if not enabled:
                self._panel.set_live_assessment(
                    "<i style='color:#7f8c8d;'>No storage earthworks yet — "
                    "draw a swale or basin.</i>"
                )
                return

            scs = SCSRunoff()
            runoff_mm = scs.runoff_depth(
                self._panel.rainfall_mm,
                scs.adjust_cn(self._panel.cn, self._panel.moisture),
            )
            duration_hr = self._panel.duration_hr or 1.0
            soil = self._panel.earthwork_soil_name

            baseline = self._state.baseline_result or {}
            acc_path = baseline.get("flow_accumulation")
            have_flow = bool(acc_path)
            cell_area = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
            total_runoff_m3 = (
                runoff_mm / 1000.0 * baseline.get("catchment_area_m2", 0.0)
                if have_flow else 0.0
            )

            stores = build_stores_from_earthworks(
                enabled, soil_name=soil, dem_path=self._state.dem_path
            )
            if have_flow:
                by_name = {s.name: s for s in stores}
                for ew in enabled:
                    store = by_name.get(ew.name)
                    if store is None:
                        continue
                    try:
                        acc_cells = sample_peak_inflow(ew.geometry, acc_path)
                        store.inflow_m3 = acc_cells * cell_area * runoff_mm / 1000.0
                    except Exception:
                        store.inflow_m3 = 0.0

            result = run_water_balance(stores, duration_hr, total_runoff_m3)
            self._panel.set_live_assessment(self._format_live_assessment(result, have_flow))
        except Exception as exc:  # never let the readout break the edit flow
            print(f"TerrainFlow Assessment — live assessment error: {exc}")

    def _format_live_assessment(self, r, have_flow):
        """Build the HTML summary for the Live Assessment panel readout."""
        lines = [
            f"<b>Storage capacity:</b> {r.total_capacity_m3:,.0f} m³",
            f"<b>Earthworks (cut / fill):</b> {r.total_cut_m3:,.0f} / {r.total_fill_m3:,.0f} m³",
        ]
        if have_flow:
            colour = (
                "#1a7a1a" if r.capture_pct >= 80
                else "#cc6600" if r.capture_pct >= 40 else "#cc0000"
            )
            lines.append(
                f"<b>Storm capture:</b> <span style='color:{colour};font-weight:bold;'>"
                f"{r.capture_pct:.0f}%</span> "
                f"({r.total_captured_m3:,.0f} of {r.total_inflow_m3:,.0f} m³ runoff; "
                f"{r.site_exit_m3:,.0f} m³ spills)"
            )
            overflowing = [f["name"] for f in r.per_feature if f["overflowed"]]
            if overflowing:
                lines.append(
                    f"<span style='color:#cc6600;'>Overflowing: "
                    f"{', '.join(overflowing)}</span>"
                )
        else:
            lines.append(
                "<i style='color:#7f8c8d;'>Run baseline analysis to see storm capture %.</i>"
            )
        lines.append(
            "<span style='color:#7f8c8d;font-size:10px;'>Analytical estimate — "
            "confirm with Re-analyse with Earthworks.</span>"
        )
        return "<br>".join(lines)

    # ---------------------------------------------------------------- Earthwork layers

    def _ensure_ew_layers(self):
        from qgis.core import QgsFillSymbol, QgsLineSymbol, QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        root = self._project.instance().layerTreeRoot()
        self._state.ew_group = (
            root.findGroup("Earthworks") or root.insertGroup(0, "Earthworks")
        )

        for ew_type, (geom_type, display_name, color_hex, fill_hex, width) in _EW_STYLES.items():
            existing = self._state.ew_layers.get(ew_type)
            if existing and self._project.instance().mapLayer(existing.id()):
                continue

            layer = QgsVectorLayer(f"{geom_type}?crs={crs_str}", display_name, "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name",        QMetaType.QString),
                QgsField("type",        QMetaType.QString),
                QgsField("capacity_m3", QMetaType.Double),
                QgsField("enabled",     QMetaType.Int),
            ])
            layer.updateFields()

            if geom_type == "LineString":
                sym = QgsLineSymbol.createSimple({
                    "color": color_hex, "width": width,
                    "capstyle": "round", "joinstyle": "round",
                })
                layer.setRenderer(QgsSingleSymbolRenderer(sym))
            else:
                sym = QgsFillSymbol.createSimple({
                    "style": "no",
                    "outline_style": "solid",
                    "outline_width": width,
                    "outline_color": color_hex,
                })
                layer.setRenderer(QgsSingleSymbolRenderer(sym))

            text_fmt = QgsTextFormat()
            font = QFont()
            font.setBold(True)
            font.setPointSize(8)
            text_fmt.setFont(font)
            text_fmt.setColor(QColor(color_hex))
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

            self._project.instance().addMapLayer(layer, False)
            self._state.ew_group.addLayer(layer)
            self._state.ew_layers[ew_type] = layer

    def _refresh_ew_layer(self):
        self._ensure_ew_layers()
        for layer in self._state.ew_layers.values():
            if layer and self._project.instance().mapLayer(layer.id()):
                layer.dataProvider().truncate()

        for ew in self._state.earthwork_manager.get_all():
            layer = self._state.ew_layers.get(ew.type)
            if not layer or not self._project.instance().mapLayer(layer.id()):
                continue
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(ew.geometry.asWkt()))
            f.setAttributes([ew.name, ew.type, ew.capacity_m3,
                              1 if getattr(ew, "enabled", True) else 0])
            layer.dataProvider().addFeature(f)

        for layer in self._state.ew_layers.values():
            if layer and self._project.instance().mapLayer(layer.id()):
                layer.triggerRepaint()

    # ---------------------------------------------------------------- Earthworks analysis

    def run_with_earthworks(self):
        if not self._state.dem_path or not self._state.burner:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM and run baseline first."
            )
            return

        enabled = self._state.earthwork_manager.get_enabled()
        if not enabled:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "No enabled earthworks to re-analyse with."
            )
            return

        modified_dem = self._state.burner.burn_earthworks(enabled)
        # Surface Strategy-C honesty warnings (sub-cell features, resolution-cap
        # degrade) so a 1-cell routing approximation is never silent.
        for msg in getattr(self._state.burner, "warnings", []):
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)
        mod_path = os.path.join(self._state.output_dir, "modified_dem.tif")
        self._state.burner.save(modified_dem, mod_path)
        self._state.modified_dem_path = mod_path

        cell_area_m2 = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
        threshold_cells = int(
            self._panel.stream_threshold_ha * 10_000 / cell_area_m2
        ) if cell_area_m2 > 0 else 1000

        self._state.analysis_worker = AnalysisWorker(
            dem_path=mod_path,
            output_dir=self._state.output_dir,
            stream_threshold=threshold_cells,
            cn=self._panel.cn,
            moisture=self._panel.moisture,
            rainfall_mm=self._panel.rainfall_mm,
            duration_hours=self._panel.duration_hr,
            boundary_path=self._state.boundary_path,
            label="earthworks",
            run_catchments=False,
            routing=self._panel.routing,
        )
        self._state.analysis_worker.progress.connect(self._panel.set_earthworks_progress)
        self._state.analysis_worker.finished.connect(self._on_earthworks_complete)
        self._state.analysis_worker.error.connect(self._on_analysis_error)
        self._state.analysis_worker.start()

    def _on_earthworks_complete(self, result):
        self._state.earthworks_result = result
        # Delegate layer loading to baseline controller's helper via the shared project
        from terrainflow_assessment.qgis.controllers.baseline import BaselineController
        bl = BaselineController(self._state, self._panel, self._project,
                                self._iface, self._canvas)
        bl._load_result_layers(result, is_earthworks=True)
        self._load_burned_dem_layer()
        if result.get("ponding"):
            self._state.ponding_raster_path = result["ponding"]

        # Non-circular check: terrain-derived ponding vs analytic capacity (§4).
        self._state.verification = self._compute_verification()
        msg = "Earthworks analysis complete. Toggle 'Show: with earthworks' to compare."
        v = self._state.verification
        if v is not None:
            msg += (
                f"\nTerrain-derived storage {v.terrain_total_m3:,.0f} m³ vs analytic "
                f"{v.analytic_total_m3:,.0f} m³ (Δ {v.delta_pct:+.0f}%)."
            )
        self._panel.set_earthworks_complete(msg)

    def _compute_verification(self):
        """Reconcile terrain-derived ponding (burned DEM) against analytic capacity (§4).

        Returns a reporting.VerificationResult, or None if the ponding raster or any
        storage feature is unavailable. Pure maths lives in modules/reporting.py; this
        only reads the rasters + builds per-feature footprint masks.
        """
        import json

        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rasterize
        from shapely.geometry import shape as _shp

        from terrainflow_assessment.modules.reporting import (
            attribute_ponding_volume,
            build_verification,
            raster_ponding_volume,
        )

        ew_result = self._state.earthworks_result or {}
        ew_pond_path = ew_result.get("ponding")
        if not ew_pond_path or not os.path.exists(ew_pond_path):
            return None

        try:
            with rasterio.open(ew_pond_path) as src:
                ew_pond = src.read(1).astype("float64")
                nodata = src.nodata
                transform = src.transform
                shape = ew_pond.shape
            cell_area = abs(transform.a * transform.e)
            cell_size = abs(transform.a)
            if nodata is not None:
                ew_pond[ew_pond == nodata] = 0.0
            ew_pond = np.clip(ew_pond, 0.0, None)
        except Exception:
            return None

        # Baseline ponding (absent → zeros; only used if it aligns to the same grid).
        bl_pond = np.zeros(shape, dtype="float64")
        bl_pond_path = (self._state.baseline_result or {}).get("ponding")
        if bl_pond_path and os.path.exists(bl_pond_path):
            try:
                with rasterio.open(bl_pond_path) as src:
                    arr = src.read(1).astype("float64")
                    bnd = src.nodata
                if arr.shape == shape:
                    if bnd is not None:
                        arr[arr == bnd] = 0.0
                    bl_pond = np.clip(arr, 0.0, None)
            except Exception:
                pass

        diff = np.clip(ew_pond - bl_pond, 0.0, None)

        analytic_by_name = {}
        min_dims = {}
        footprints = []
        for ew in self._state.earthwork_manager.get_enabled():
            if getattr(ew, "capacity_m3", 0.0) <= 0:
                continue
            analytic_by_name[ew.name] = ew.capacity_m3
            # Sub-cell check keys off the channel bottom width; polygons never sub-cell.
            min_dims[ew.name] = (
                getattr(ew, "bottom_width_m", None) if ew.type == "swale" else None
            )
            try:
                geom = _shp(json.loads(ew.geometry.asJson()))
                if geom.geom_type in ("LineString", "MultiLineString"):
                    geom = geom.buffer(max(getattr(ew, "width", 2.0) / 2.0, cell_size))
                mask = _rasterize(
                    [(geom, 1)], out_shape=shape, transform=transform,
                    fill=0, dtype="uint8",
                ).astype(bool)
                footprints.append((ew.name, mask))
            except Exception:
                footprints.append((ew.name, np.zeros(shape, dtype=bool)))

        if not analytic_by_name:
            return None

        terrain_by_name, unattributed = attribute_ponding_volume(diff, cell_area, footprints)
        baseline_total = raster_ponding_volume(bl_pond, cell_area)
        earthworks_total = raster_ponding_volume(ew_pond, cell_area)

        result = build_verification(
            analytic_by_name, terrain_by_name, baseline_total, earthworks_total,
            min_dims, cell_size,
        )
        result.unattributed_m3 = unattributed
        return result

    def _load_burned_dem_layer(self):
        """Add the burned (Strategy-C) DEM to the layer panel so the carve/ridge is visible.

        Placed at the bottom of the layer tree (it is a backdrop, not a result overlay)
        and registered under the earthworks layer group so it shows/hides with the
        'with earthworks' toggle. Silently skips if the burn produced no valid raster.
        """
        path = self._state.modified_dem_path
        if not path or not os.path.exists(path):
            return
        from qgis.core import QgsRasterLayer
        layer = QgsRasterLayer(path, "Earthworks — Burned DEM")
        if not layer.isValid():
            return
        project = self._project.instance()
        project.addMapLayer(layer, False)          # don't auto-add to legend top
        project.layerTreeRoot().addLayer(layer)     # append at the bottom instead
        ids = list(getattr(self._state, "earthworks_layer_ids", None) or [])
        ids.append(layer.id())
        self._state.earthworks_layer_ids = ids

    def _on_analysis_error(self, tb):
        self._panel.set_earthworks_complete("Analysis failed — see Python console for details.")
        print("TerrainFlow Assessment — Analysis error:\n" + tb)
        self._iface.messageBar().pushCritical("TerrainFlow Assessment",
                                               "Analysis failed. See Python console.")

    # ---------------------------------------------------------------- Ponding query

    def activate_ponding_query(self):
        if not self._state.ponding_raster_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first."
            )
            return
        tool = PondingQueryTool(self._canvas, self._state.ponding_raster_path)
        tool.ponding_selected.connect(self._on_ponding_selected)
        tool.no_ponding.connect(self._on_no_ponding)
        self._canvas.setMapTool(tool)

    def _on_ponding_selected(self, volume_m3, volume_l, cell_count, area_m2,
                              outline_geom, inflow_m3, fill_fraction):
        from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

        dlg = QDialog(self._iface.mainWindow())
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
        area_str = f"{area_ha:.2f} ha  ({area_m2:,.0f} m²)" if area_ha >= 0.1 else f"{area_m2:,.0f} m²"
        _row("Surface area:", area_str)

        if area_m2 > 0:
            _row("Mean depth:", f"{volume_m3 / area_m2:.2f} m")

        _row("Cell count:", f"{cell_count:,} cells")

        if fill_fraction >= 0:
            fill_pct = fill_fraction * 100.0
            if fill_fraction <= 1.0:
                fill_str = f"{fill_pct:.0f}% of design-storm inflow captured  ✓"
                colour = "#006600"
            else:
                fill_str = f"{fill_pct:.0f}% of design-storm inflow — depression overflows  ⚠"
                colour = "#cc4400"
            lbl_fill = QLabel(
                f"<b>Storm fill:</b>  <span style=\"color:{colour}\">{fill_str}</span>"
            )
            lbl_fill.setToolTip(
                "Ratio of design-storm inflow volume to depression capacity.\n"
                "Below 100%: depression absorbs the full storm event.\n"
                "Above 100%: overflow will occur — consider enlarging the earthwork."
            )
            layout.addWidget(lbl_fill)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _on_no_ponding(self):
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "No ponding at that location — click a blue zone in the Water Captured layer.",
        )

    # ---------------------------------------------------------------- Slope visualisation

    def toggle_slope_class(self, checked):
        if not self._state.slope_raster_path or not os.path.exists(self._state.slope_raster_path):
            return
        lid = self._state.slope_class_layer_id
        if lid:
            node = self._project.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(checked)
                self._canvas.refresh()
                return
        layer = QgsRasterLayer(self._state.slope_raster_path, "Slope Classification")
        if layer.isValid():
            self._apply_slope_class_ramp(layer)
            layer.setOpacity(0.6)
            self._project.instance().addMapLayer(layer)
            self._state.slope_class_layer_id = layer.id()
            node = self._project.instance().layerTreeRoot().findLayer(layer)
            if node:
                node.setItemVisibilityChecked(checked)
            self._canvas.refresh()

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

    def toggle_slope_arrows(self, checked):
        if checked:
            existing = (self._state.slope_arrows_layer_id and
                        self._project.instance().mapLayer(self._state.slope_arrows_layer_id))
            if existing:
                node = self._project.instance().layerTreeRoot().findLayer(
                    self._state.slope_arrows_layer_id)
                if node:
                    node.setItemVisibilityChecked(True)
            else:
                self._generate_slope_arrows()
        else:
            if self._state.slope_arrows_layer_id:
                node = self._project.instance().layerTreeRoot().findLayer(
                    self._state.slope_arrows_layer_id)
                if node:
                    node.setItemVisibilityChecked(False)
            self._canvas.refresh()

    def _generate_slope_arrows(self):
        if not self._state.dem_path:
            return
        try:
            import processing
            import rasterio

            result = processing.run("gdal:aspect", {
                "INPUT": self._state.dem_path,
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

            with rasterio.open(self._state.dem_path) as src:
                cell_size = abs(src.transform.a)
                transform = src.transform

            with rasterio.open(aspect_path) as src:
                aspect = src.read(1).astype("float32")
                rows, cols = aspect.shape

            step = max(1, int(50.0 / cell_size))

            layer = QgsVectorLayer("Point", "Slope Direction", "memory")
            layer.setCrs(self._project.instance().crs())
            pr = layer.dataProvider()
            pr.addAttributes([QgsField("angle", QMetaType.Double)])
            layer.updateFields()

            features = []
            for r in range(0, rows, step):
                for c in range(0, cols, step):
                    val = float(aspect[r, c])
                    if val < 0 or val > 360:
                        continue
                    x = transform.c + c * transform.a + cell_size / 2
                    y = transform.f + r * transform.e + cell_size / 2
                    f = QgsFeature()
                    from qgis.core import QgsGeometry, QgsPointXY
                    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                    f.setAttributes([val])
                    features.append(f)

            pr.addFeatures(features)
            layer.updateExtents()

            symbol = QgsMarkerSymbol.createSimple({
                "name": "arrow", "color": "60,60,200,200",
                "outline_color": "20,20,120,200", "size": "5", "angle": "0",
            })
            symbol.symbolLayer(0).setDataDefinedProperty(
                QgsSymbolLayer.PropertyAngle, QgsProperty.fromField("angle"))
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))

            self._project.instance().addMapLayer(layer)
            self._state.slope_arrows_layer_id = layer.id()
            self._canvas.refresh()

        except Exception as exc:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not generate slope direction arrows: {exc}"
            )
