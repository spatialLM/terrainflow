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
        )

        if dlg.exec():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            ew.width = getattr(dlg, "get_width", lambda: ew.width)()
            if ew_type == "dam":
                ew.crest_elevation = dlg.get_crest_elevation()
            elif ew_type == "swale":
                ew.companion_berm = getattr(dlg, "get_companion_berm", lambda: False)()

            ew.capacity_m3, ew.capacity_l = calculate_capacity(
                ew_type, geometry, ew.depth, ew.width,
                getattr(ew, "companion_berm", False),
            )
            self._state.earthwork_manager.add(ew)
            self._panel.add_earthwork_to_list(
                len(self._state.earthwork_manager) - 1, ew.summary()
            )
            self._refresh_ew_layer()
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
        )
        if dlg.exec():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            ew.capacity_m3, ew.capacity_l = calculate_capacity(
                ew.type, ew.geometry, ew.depth, ew.width,
                getattr(ew, "companion_berm", False),
            )
            self._panel.update_earthwork_in_list(idx, ew.summary())
            self._refresh_ew_layer()

    def delete_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.remove(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())
        self._refresh_ew_layer()

    def toggle_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.toggle(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())

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
        if result.get("ponding"):
            self._state.ponding_raster_path = result["ponding"]
        self._panel.set_earthworks_complete(
            "Earthworks analysis complete. Toggle 'Show: with earthworks' to compare."
        )

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
