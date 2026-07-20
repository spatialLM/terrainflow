"""
contour.py — ContourController

Handles: contour analysis, swale segment detection, simple contour generation,
keypoint analysis, ridgeline display, and pond-site recommendations.
"""

from __future__ import annotations

import os

from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, QgsField,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat,
    QgsMarkerSymbol, QgsSingleSymbolRenderer,
    QgsSymbolLayer, QgsProperty,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QMessageBox


class ContourController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    # ---------------------------------------------------------------- Contour analysis

    def run_contour_analysis(self):
        if not self._state.dem_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM first."
            )
            return
        acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
        if not acc_path or not os.path.exists(acc_path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first."
            )
            return

        from terrainflow_assessment.modules.contour_analysis import analyse_contours

        def _progress(pct, msg):
            self._panel.set_contour_progress(pct, msg)

        try:
            contours = analyse_contours(
                dem_path=self._state.dem_path,
                acc_path=acc_path,
                interval_m=self._panel.contour_interval_m,
                max_slope_deg=self._panel.max_slope_deg,
                usable_polygon=self._state.usable_polygon,
                progress_callback=_progress,
                cell_area_m2=self._state.dem_info.cell_area_m2 if self._state.dem_info else None,
                runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
                min_length_m=self._panel.min_contour_length_m,
            )
            self._state.contour_features = contours
            self._panel.set_contour_complete()
            self._panel.set_contour_results(contours)
            self._display_contour_layer(contours)
        except Exception as exc:
            self._panel.set_contour_complete()
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Contour analysis failed: {exc}"
            )

    def _display_contour_layer(self, contours):
        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Candidate Contour Swales", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("elevation", QMetaType.Double),
            QgsField("rank", QMetaType.Int),
            QgsField("peak_acc", QMetaType.Double),
            QgsField("mean_slope", QMetaType.Double),
        ])
        layer.updateFields()

        feats = []
        for feat in contours:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(feat.geometry.wkt))
            f.setAttributes([feat.elevation, feat.rank or 0, feat.peak_acc, feat.mean_slope_deg])
            feats.append(f)
        pr.addFeatures(feats)

        max_acc = max((c.peak_acc or 0) for c in contours) if contours else 1.0
        self._apply_rank_style(layer, max_acc=max_acc)
        if self._state.contour_layer:
            try:
                self._project.instance().removeMapLayer(self._state.contour_layer)
            except Exception:
                pass
        self._project.instance().addMapLayer(layer)
        self._state.contour_layer = layer

    def _apply_rank_style(self, layer, max_acc=None):
        from qgis.core import QgsLineSymbol
        color_expr = (
            "CASE"
            " WHEN \"rank\" = 1      THEN color_rgb(255,215,  0)"
            " WHEN \"rank\" <= 5     THEN color_rgb(255,107,  0)"
            " WHEN \"rank\" <= 10    THEN color_rgb( 74,144,217)"
            " ELSE                       color_rgb(158,158,158)"
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

    def select_top5_contours(self):
        if not self._state.contour_features:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run contour analysis first."
            )
            return

        ranked = sorted(
            self._state.contour_features,
            key=lambda f: f.peak_acc if f.peak_acc is not None else 0,
            reverse=True,
        )[:5]

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        if self._state.top5_layer:
            try:
                self._project.instance().removeMapLayer(self._state.top5_layer)
            except Exception:
                pass

        layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Top 5 Swale Contours", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("elevation", QMetaType.Double),
            QgsField("rank", QMetaType.Int),
            QgsField("peak_acc", QMetaType.Double),
            QgsField("mean_slope", QMetaType.Double),
        ])
        layer.updateFields()

        feats = []
        for i, feat in enumerate(ranked):
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(feat.geometry.wkt))
            f.setAttributes([feat.elevation, i + 1, feat.peak_acc, feat.mean_slope_deg])
            feats.append(f)
        pr.addFeatures(feats)

        from qgis.core import QgsLineSymbol
        symbol = QgsLineSymbol.createSimple({
            "color": "255,140,0", "width": "1.2", "capstyle": "round",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        self._project.instance().addMapLayer(layer)
        self._state.top5_layer = layer

    def run_segment_analysis(self):
        if not self._state.contour_features:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run contour analysis first."
            )
            return
        acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
        if not acc_path or not os.path.exists(acc_path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first."
            )
            return

        from terrainflow_assessment.modules.contour_analysis import find_swale_segments

        def _progress(pct, msg):
            self._panel.set_contour_progress(pct, msg)

        try:
            segments = find_swale_segments(
                contours=self._state.contour_features,
                acc_path=acc_path,
                cell_area_m2=self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0,
                runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
                min_acc_ha=self._panel.min_catchment_ha,
                swale_depth_m=self._panel.swale_depth_m,
                swale_width_m=self._panel.swale_width_m,
                progress_callback=_progress,
            )
            self._panel.set_contour_complete()
            self._display_swale_segments(segments)
        except Exception as exc:
            self._panel.set_contour_complete()
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Segment analysis failed: {exc}"
            )

    def _display_swale_segments(self, segments):
        from qgis.core import QgsLineSymbol
        from qgis.core import QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont

        if not segments:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No swale segments found — try lowering 'Min catchment above swale' "
                "or run contour analysis with a smaller contour interval.",
            )
            return

        if self._state.segment_layer:
            try:
                self._project.instance().removeMapLayer(self._state.segment_layer)
            except Exception:
                pass

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Recommended Swale Segments", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",             QMetaType.QString),
            QgsField("elevation",         QMetaType.Double),
            QgsField("contributing_ha",   QMetaType.Double),
            QgsField("inflow_m3",         QMetaType.Double),
            QgsField("required_length_m", QMetaType.Double),
            QgsField("rank",              QMetaType.Int),
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

        color_expr = (
            "CASE"
            " WHEN \"rank\" = 1      THEN color_rgb(255,215,  0)"
            " WHEN \"rank\" <= 5     THEN color_rgb(255,107,  0)"
            " WHEN \"rank\" <= 10    THEN color_rgb( 74,144,217)"
            " ELSE                       color_rgb(158,158,158)"
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
            QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(color_expr))
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth, QgsProperty.fromExpression(width_expr))
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

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

        self._project.instance().addMapLayer(layer)
        self._state.segment_layer = layer
        self._iface.messageBar().pushSuccess(
            "TerrainFlow Assessment",
            f"{len(segments)} swale segment(s) found. Top segment: {segments[0].label}",
        )

    def generate_simple_contours(self):
        if not self._state.dem_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM first."
            )
            return
        try:
            import processing
            interval = self._panel.simple_contour_interval_m
            result = processing.run("gdal:contour", {
                "INPUT": self._state.dem_path,
                "BAND": 1,
                "INTERVAL": interval,
                "FIELD_NAME": "ELEV",
                "OUTPUT": "TEMPORARY_OUTPUT",
            })
            out = result.get("OUTPUT")
            if hasattr(out, "source"):
                out = out.source()

            if self._state.simple_contour_layer:
                try:
                    self._project.instance().removeMapLayer(self._state.simple_contour_layer)
                except Exception:
                    pass

            layer = QgsVectorLayer(out, f"Contours ({interval} m)", "ogr")
            if not layer.isValid():
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", "Contour generation produced no output."
                )
                return

            from qgis.core import QgsLineSymbol
            symbol = QgsLineSymbol.createSimple({"color": "100,100,100", "width": "0.3"})
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            self._project.instance().addMapLayer(layer)
            self._state.simple_contour_layer = layer

        except Exception as exc:
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Contour generation failed: {exc}"
            )

    # ---------------------------------------------------------------- Keypoint analysis

    def _get_keypoint_boundary_mask(self, dem_path):
        import json
        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rasterize

        layer = (
            self._panel.earthworks_area_layer
            or self._panel.analysis_area_layer
        )
        if layer is None and not self._state.boundary_path:
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
                gdf = gpd.read_file(self._state.boundary_path)
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

    def run_keypoint_analysis(self):
        if not self._state.dem_path:
            return
        acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
        if not acc_path:
            QMessageBox.warning(self._panel, "No Analysis",
                                "Run baseline analysis first.")
            return

        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis

        self._panel.set_keypoint_progress(5, "Loading DEM…")
        try:
            cell_m2 = self._state.dem_info.cell_area_m2 if self._state.dem_info else 100.0
            min_acc = max(50, int(1.0 * 10_000 / cell_m2))

            self._panel.set_keypoint_progress(15, "Building boundary mask…")
            boundary_mask = self._get_keypoint_boundary_mask(self._state.dem_path)

            self._panel.set_keypoint_progress(25, "Finding keypoints…")
            ka = KeylineAnalysis(self._state.dem_path, acc_path)
            self._state.keyline_analysis = ka

            keypoints = ka.find_keypoints(
                min_acc_cells=min_acc,
                n_keypoints=self._panel.keypoint_count,
                boundary_mask=boundary_mask,
            )
            self._state.found_keypoints = keypoints

            self._panel.set_keypoint_progress(70, "Finding ridgelines…")
            ridgelines = ka.find_ridgelines(boundary_mask=boundary_mask)

            self._display_keypoints(keypoints)
            self._display_ridgelines(ridgelines)

            if keypoints:
                self._panel.set_keypoint_complete(
                    f"{len(keypoints)} keypoints | {len(ridgelines)} ridgeline segment(s)\n"
                    "Click 'Recommend Pond Sites' to find optimal dam locations."
                )
            else:
                self._panel.set_keypoint_complete(
                    "No keypoints found — try a larger DEM area or lower the\n"
                    "number of keypoints requested."
                )

            kp_lines = "\n".join(f"  {kp['label']}" for kp in keypoints)
            ridge_summary = (
                f"{len(ridgelines)} ridge segment(s)" if ridgelines else "No ridgelines found"
            )
            self._panel.set_keypoint_results(
                f"Keypoints ({len(keypoints)}):\n{kp_lines}\n\n{ridge_summary}"
            )

        except Exception:
            import traceback
            self._panel.set_keypoint_complete("")
            QMessageBox.critical(self._panel, "Keypoint Analysis Error",
                                 traceback.format_exc())

    def run_recommend_ponds(self):
        if not self._state.found_keypoints:
            QMessageBox.warning(self._panel, "No Keypoints",
                                "Run 'Find Keypoints + Ridgelines' first.")
            return

        self._panel.set_keypoint_progress(5, "Finding pond sites…")
        try:
            ka = self._state.keyline_analysis
            if ka is None:
                from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
                acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
                ka = KeylineAnalysis(self._state.dem_path, acc_path)

            self._panel.set_keypoint_progress(50, "Finding pond sites…")
            boundary_mask = self._get_keypoint_boundary_mask(self._state.dem_path)
            pond_sites = ka.recommend_pond_sites(self._state.found_keypoints,
                                                 boundary_mask=boundary_mask)
            self._display_pond_sites(pond_sites)

            lines = "\n".join(f"  {s['label']}" for s in pond_sites)
            existing = self._panel._keypoint_results_lbl.text()
            self._panel.set_keypoint_results(existing + f"\n\nPond sites:\n{lines}")
            self._panel.set_keypoint_complete(
                f"{len(self._state.found_keypoints)} keypoints | "
                f"{len(pond_sites)} pond site(s) found."
            )
        except Exception:
            import traceback
            self._panel.set_keypoint_complete("")
            QMessageBox.critical(self._panel, "Pond Site Error", traceback.format_exc())

    def _display_keypoints(self, keypoints):
        for lyr in self._project.instance().mapLayersByName("Keypoints"):
            self._project.instance().removeMapLayer(lyr)

        layer = QgsVectorLayer("Point", "Keypoints", "memory")
        layer.setCrs(self._project.instance().crs())
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",        QMetaType.QString),
            QgsField("elevation",    QMetaType.Double),
            QgsField("slope_deg",    QMetaType.Double),
            QgsField("catchment_ha", QMetaType.Double),
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
        self._project.instance().addMapLayer(layer)

    def _display_ridgelines(self, ridgelines):
        from qgis.core import QgsLineSymbol
        for lyr in self._project.instance().mapLayersByName("Ridgelines (Water Divides)"):
            self._project.instance().removeMapLayer(lyr)

        layer = QgsVectorLayer("LineString", "Ridgelines (Water Divides)", "memory")
        layer.setCrs(self._project.instance().crs())
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",          QMetaType.QString),
            QgsField("length_m",       QMetaType.Double),
            QgsField("mean_elevation", QMetaType.Double),
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
        self._project.instance().addMapLayer(layer)

    def _display_pond_sites(self, sites):
        for lyr in self._project.instance().mapLayersByName("Recommended Pond Sites"):
            self._project.instance().removeMapLayer(lyr)

        layer = QgsVectorLayer("Point", "Recommended Pond Sites", "memory")
        layer.setCrs(self._project.instance().crs())
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("label",        QMetaType.QString),
            QgsField("elevation",    QMetaType.Double),
            QgsField("catchment_ha", QMetaType.Double),
            QgsField("dam_width_m",  QMetaType.Double),
            QgsField("keypoint",     QMetaType.Int),
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
        self._project.instance().addMapLayer(layer)
