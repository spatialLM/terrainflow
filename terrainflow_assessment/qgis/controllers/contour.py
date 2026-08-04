"""
contour.py — ContourController

Handles: contour analysis, swale segment detection, simple contour generation,
keypoint analysis, ridgeline display, and pond-site recommendations.
"""

from __future__ import annotations

import os

from qgis.core import (
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProperty,
    QgsSingleSymbolRenderer,
    QgsSymbolLayer,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QMessageBox

from terrainflow_assessment.qgis.controllers._layers import remove_layer, resolve_layer


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

        # Deactivate any contour-picking map tool so it can't outlive the layer
        # swap (its captured layer would become a dead reference → crash).
        self._reset_contour_map_tool()

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
        remove_layer(self._project, self._state.contour_layer_id)
        self._project.instance().addMapLayer(layer)
        self._state.contour_layer_id = layer.id()

    def _reset_contour_map_tool(self):
        """Deactivate the active map tool if it is a contour-picking tool, so a
        swapped/deleted contour layer can't be dereferenced by a stale tool."""
        active = self._canvas.mapTool()
        if active is not None and active.__class__.__name__ in (
            "ContourSegmentTool", "SelectContourTool",
        ):
            self._canvas.unsetMapTool(active)

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

        top_n = self._panel.top_n
        ranked = sorted(
            self._state.contour_features,
            key=lambda f: f.peak_acc if f.peak_acc is not None else 0,
            reverse=True,
        )[:top_n]

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        remove_layer(self._project, self._state.top5_layer_id)

        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               f"Top {top_n} Swale Contours", "memory")
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
        self._state.top5_layer_id = layer.id()

    # ---------------------------------------------------------------- Clear / inflow bands

    def clear_analysis(self):
        """Remove all analysis layers and reset state so the user can re-run cleanly."""
        self._reset_contour_map_tool()
        proj = self._project.instance()
        for attr in ("contour_layer_id", "top5_layer_id", "segment_layer_id",
                     "simple_contour_layer_id", "keyline_layer_id",
                     "drawn_keyline_layer_id", "inflow_bands_layer_id"):
            remove_layer(self._project, getattr(self._state, attr, None))
            setattr(self._state, attr, None)
        for name in ("Keypoints", "Ridgelines (Water Divides)", "Recommended Pond Sites",
                     "Keyline Design", "Keyline Keypoint"):
            for lyr in proj.mapLayersByName(name):
                proj.removeMapLayer(lyr)
        self._state.contour_features = []
        self._state.found_keypoints = None
        self._state.keyline_master_geom = None
        self._state.keyline_master_coords = None
        self._panel.clear_analysis_ui()
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment", "Analysis layers cleared."
        )
        self._canvas.refresh()

    def show_inflow_bands(self, checked):
        """Toggle the along-contour inflow-share classification layer."""
        remove_layer(self._project, self._state.inflow_bands_layer_id)
        self._state.inflow_bands_layer_id = None
        if not checked:
            self._canvas.refresh()
            return
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
        from terrainflow_assessment.modules.contour_analysis import classify_contour_inflow
        try:
            stretches = classify_contour_inflow(
                self._state.contour_features, acc_path,
                cell_area_m2=self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0,
                runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
                duration_hr=self._panel.duration_hr,
            )
            self._display_inflow_gradient(stretches, self._panel.inflow_scale_mode)
        except Exception as exc:
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Inflow classification failed: {exc}"
            )

    def _display_inflow_gradient(self, stretches, scale="log"):
        """Render contour stretches on one ramp keyed to the global inflow range,
        so stretches are comparable across the whole map.

        *scale* controls how colour maps to value:
          "log"      — colour by log(inflow); compresses a skewed range so one
                       extreme point doesn't flatten everything (default).
          "linear"   — straight 0→max absolute scale.
          "quantile" — equal count per colour (rank-based separation).
        """
        import math

        from qgis.core import (
            QgsGradientColorRamp,
            QgsGraduatedSymbolRenderer,
            QgsLineSymbol,
        )
        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Contour Inflow (m³)", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("inflow_m3", QMetaType.Double),
            QgsField("flow_ls", QMetaType.Double),
            QgsField("inflow_log", QMetaType.Double),
        ])
        layer.updateFields()
        feats = []
        for s in stretches:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(s["geometry"].wkt))
            # Clamp before the log: a negative inflow would raise a math domain error
            # here rather than anywhere the cause would be visible.
            inflow = max(0.0, float(s["inflow_m3"]))
            f.setAttributes([inflow, float(s["flow_ls"]), math.log10(inflow + 1.0)])
            feats.append(f)

        if not feats:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No contour inflow to show — run contour analysis first, and check "
                "the stream threshold is not above everything on the site.",
            )
            return
        pr.addFeatures(feats)

        base = QgsLineSymbol.createSimple({"width": "1.6", "capstyle": "round"})
        # Continuous low→high ramp: blue → cyan → green → amber → red.
        ramp = QgsGradientColorRamp(QColor(44, 123, 182), QColor(215, 25, 28))
        try:
            from qgis.core import QgsGradientStop
            ramp.setStops([
                QgsGradientStop(0.25, QColor(0, 170, 200)),
                QgsGradientStop(0.50, QColor(120, 195, 70)),
                QgsGradientStop(0.75, QColor(253, 174, 97)),
            ])
        except Exception:
            pass

        if scale == "quantile":
            attr, mode = "inflow_m3", QgsGraduatedSymbolRenderer.Quantile
        elif scale == "linear":
            attr, mode = "inflow_m3", QgsGraduatedSymbolRenderer.EqualInterval
        else:  # log (default): even classes in log space, relabelled to m³
            attr, mode = "inflow_log", QgsGraduatedSymbolRenderer.EqualInterval

        # createRenderer cannot classify a degenerate range — every stretch carrying
        # the same inflow, or a single stretch — and raises rather than returning
        # something usable. That took out the whole Show Inflow Gradient action, and
        # the traceback named the renderer rather than the data behind it.
        values = {f.attribute(attr) for f in layer.getFeatures()}
        renderer = None
        if len(values) > 1:
            try:
                n_classes = min(16, len(values))
                renderer = QgsGraduatedSymbolRenderer.createRenderer(
                    layer, attr, n_classes, mode, base, ramp)
            except Exception as exc:
                print(f"TerrainFlow Assessment — inflow gradient classify failed: {exc}")
                renderer = None
        if renderer is None or not renderer.ranges():
            # One colour is honest for one value; a 16-class ramp over it would
            # invent a gradient that is not in the data.
            from qgis.core import QgsSingleSymbolRenderer
            layer.setRenderer(QgsSingleSymbolRenderer(base))
            self._project.instance().addMapLayer(layer)
            self._state.inflow_bands_layer_id = layer.id()
            self._canvas.refresh()
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Contour inflow is uniform across the site, so there is no gradient "
                "to band — shown in a single colour.",
            )
            return

        if scale not in ("quantile", "linear"):
            # Relabel the log-space classes back to their m³ bounds so the legend
            # stays readable in real units.
            for i, rng in enumerate(renderer.ranges()):
                lo = max(10 ** rng.lowerValue() - 1.0, 0.0)
                hi = 10 ** rng.upperValue() - 1.0
                try:
                    renderer.updateRangeLabel(i, f"{lo:,.0f} – {hi:,.0f} m³")
                except Exception:
                    break

        layer.setRenderer(renderer)
        self._project.instance().addMapLayer(layer)
        self._state.inflow_bands_layer_id = layer.id()
        self._canvas.refresh()

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
        from terrainflow_assessment.modules.swale_design import get_infiltration_rate

        def _progress(pct, msg):
            self._panel.set_segment_progress(pct, msg)

        try:
            segments = find_swale_segments(
                contours=self._state.contour_features,
                acc_path=acc_path,
                cell_area_m2=self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0,
                runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
                min_acc_ha=self._panel.min_catchment_ha,
                swale_depth_m=self._panel.swale_depth_m,
                swale_width_m=self._panel.swale_width_m,
                infiltration_mm_hr=get_infiltration_rate(self._panel.earthwork_soil_name),
                duration_hr=self._panel.duration_hr,
                rank_mode=self._panel.segment_rank_mode,
                slope_path=self._state.slope_raster_path,
                seg_max_slope_deg=self._panel.seg_max_slope_deg,
                progress_callback=_progress,
            )
            self._panel.set_segment_complete()
            self._panel.set_segment_results(segments)
            self._display_swale_segments(segments)
        except Exception as exc:
            self._panel.set_segment_complete()
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Segment analysis failed: {exc}"
            )

    def _display_swale_segments(self, segments):
        from qgis.core import QgsLineSymbol, QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont

        if not segments:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No swale segments found — try lowering 'Min catchment above swale' "
                "or run contour analysis with a smaller contour interval.",
            )
            return

        remove_layer(self._project, self._state.segment_layer_id)

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
            QgsField("capped",            QMetaType.Int),
        ])
        layer.updateFields()

        feats = []
        for seg in segments:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(seg.geometry.wkt))
            f.setAttributes([
                seg.label, seg.elevation, seg.contributing_ha,
                seg.inflow_m3, seg.required_length_m, seg.segment_rank,
                1 if seg.capped else 0,
            ])
            feats.append(f)
        pr.addFeatures(feats)

        # Distinct "swale" style: a white casing under a bold core coloured by
        # whether the swale holds its inflow (green) or needs an overflow (amber),
        # so it reads clearly as the recommended swale — separate from the thin
        # ranked candidate contours and the slope-coloured flow lines.
        from qgis.core import QgsSimpleLineSymbolLayer
        from qgis.PyQt.QtCore import Qt as _Qt
        color_expr = (
            "CASE WHEN \"capped\" = 1 THEN color_rgb(230,126, 34)"
            " ELSE color_rgb( 39,174, 96) END"
        )
        symbol = QgsLineSymbol.createSimple({"width": "1.8", "capstyle": "round"})
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(color_expr))
        casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 160))
        casing.setWidth(3.6)
        casing.setPenCapStyle(_Qt.RoundCap)
        symbol.insertSymbolLayer(0, casing)  # draw casing beneath the core
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
        self._state.segment_layer_id = layer.id()
        n_capped = sum(1 for s in segments if s.capped)
        msg = f"{len(segments)} swale segment(s) found. Top segment: {segments[0].label}"
        if n_capped:
            msg += (
                f"  ⚠ {n_capped} contour(s) too short to hold the design inflow — "
                "consider a deeper/wider swale, a basin, or splitting the catchment."
            )
        self._iface.messageBar().pushSuccess("TerrainFlow Assessment", msg)

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

            remove_layer(self._project, self._state.simple_contour_layer_id)

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
            self._state.simple_contour_layer_id = layer.id()

        except Exception as exc:
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Contour generation failed: {exc}"
            )

    # ---------------------------------------------------------------- Keypoint analysis

    def _get_keypoint_boundary_mask(self, dem_path):
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

        from terrainflow_assessment.modules.keypoint_analysis import DrainageLineAnalysis

        self._panel.set_keypoint_progress(5, "Loading DEM…")
        try:
            cell_m2 = self._state.dem_info.cell_area_m2 if self._state.dem_info else 100.0
            min_acc = max(50, int(1.0 * 10_000 / cell_m2))

            self._panel.set_keypoint_progress(15, "Building boundary mask…")
            boundary_mask = self._get_keypoint_boundary_mask(self._state.dem_path)

            self._panel.set_keypoint_progress(25, "Finding keypoints…")
            ka = DrainageLineAnalysis(self._state.dem_path, acc_path)
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
                    f"{len(keypoints)} valley points | {len(ridgelines)} ridgeline segment(s)\n"
                    "Click a row to zoom, or 'Recommend Pond Sites' for dam locations."
                )
            else:
                self._panel.set_keypoint_complete(
                    "No valley points found — try a larger DEM area or lower the\n"
                    "number requested."
                )

            self._panel.set_keypoint_results(
                self._keypoint_result_items(keypoints, ridgelines)
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

        self._panel.set_ponds_progress(5, "Finding pond sites…")
        try:
            ka = self._state.keyline_analysis
            if ka is None:
                from terrainflow_assessment.modules.keypoint_analysis import (
                    DrainageLineAnalysis,
                )
                acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
                ka = DrainageLineAnalysis(self._state.dem_path, acc_path)

            self._panel.set_ponds_progress(50, "Finding pond sites…")
            boundary_mask = self._get_keypoint_boundary_mask(self._state.dem_path)
            pond_sites = ka.recommend_pond_sites(self._state.found_keypoints,
                                                 boundary_mask=boundary_mask)
            self._display_pond_sites(pond_sites)

            self._panel.set_keypoint_results(
                self._keypoint_result_items(
                    self._state.found_keypoints, pond_sites=pond_sites)
            )
            self._panel.set_ponds_complete(
                f"{len(self._state.found_keypoints)} valley points | "
                f"{len(pond_sites)} pond site(s) found."
            )
        except Exception:
            import traceback
            self._panel.set_ponds_complete("")
            QMessageBox.critical(self._panel, "Pond Site Error", traceback.format_exc())

    @staticmethod
    def _keypoint_result_items(keypoints, ridgelines=None, pond_sites=None):
        """Build the clickable result rows for the keypoint list. Header rows have
        x/y = None (non-clickable); feature rows carry their map coordinates."""
        items = [{"label": f"Valley points ({len(keypoints)})", "x": None, "y": None}]
        for kp in keypoints:
            items.append({"label": f"  {kp['label']}", "x": kp["x"], "y": kp["y"]})
        if ridgelines is not None:
            n = len(ridgelines)
            items.append({
                "label": f"{n} ridge segment(s)" if n else "No ridgelines found",
                "x": None, "y": None,
            })
        if pond_sites is not None:
            items.append({"label": f"Pond sites ({len(pond_sites)})", "x": None, "y": None})
            for s in pond_sites:
                items.append({"label": f"  {s['label']}", "x": s["x"], "y": s["y"]})
        return items

    def zoom_to_point(self, x, y, half_extent_m=80.0):
        """Centre and zoom the canvas on a result point, then flash it."""
        from qgis.core import QgsPointXY, QgsRectangle

        rect = QgsRectangle(
            x - half_extent_m, y - half_extent_m,
            x + half_extent_m, y + half_extent_m,
        )
        self._canvas.setExtent(rect)
        self._canvas.refresh()
        try:
            self._canvas.flashGeometries(
                [QgsGeometry.fromPointXY(QgsPointXY(x, y))]
            )
        except Exception:
            pass

    def highlight_segment(self, wkt):
        """Zoom to a swale segment and flash its exact geometry so the user can
        see precisely which swale a result row refers to."""
        geom = QgsGeometry.fromWkt(wkt)
        if geom is None or geom.isEmpty():
            return
        rect = geom.boundingBox()
        rect.grow(max(rect.width(), rect.height()) * 0.4 + 20.0)
        self._canvas.setExtent(rect)
        self._canvas.refresh()
        try:
            self._canvas.flashGeometries([geom])
        except Exception:
            pass

    # ---------------------------------------------------------------- Keyline design

    def run_keyline_analysis(self):
        """Generate the Yeomans keyline + parallel cultivation guides from the DEM
        and display them as a styled line layer (Phase 3a)."""
        if not self._state.dem_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM first."
            )
            return
        acc_path = (self._state.baseline_result or {}).get("flow_accumulation")

        from terrainflow_assessment.modules.keypoint_analysis import (
            YeomansKeylineAnalysis,
        )

        self._panel.set_keyline_progress(10, "Extracting primary valley…")
        try:
            ya = YeomansKeylineAnalysis(self._state.dem_path, acc_path=acc_path)
            self._panel.set_keyline_progress(45, "Locating keypoint…")
            keypoint = ya.find_keypoint()
            if keypoint is None:
                self._panel.set_keyline_complete("")
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "No keypoint found — the DEM area may be too small or too flat.",
                )
                return

            self._panel.set_keyline_progress(70, "Generating cultivation guides…")
            runs = ya.get_cultivation_runs(
                keypoint,
                n_runs=self._panel.keyline_runs,
                cross_grade=self._panel.keyline_cross_grade,
                spacing_m=self._panel.keyline_spacing_m,
            )
            # Clip to the usable area (analysis/earthworks polygon) when set.
            runs = self._clip_runs_to_usable(runs)
            self._display_keylines(runs, keypoint)
            self._panel.set_keyline_complete(
                f"Keyline at {keypoint['elevation']:.1f} m + "
                f"{len(runs) - 1} cultivation guide(s)."
            )
        except Exception:
            import traceback
            self._panel.set_keyline_complete("")
            QMessageBox.critical(self._panel, "Keyline Analysis Error",
                                 traceback.format_exc())

    def activate_draw_keyline(self):
        """Let the user draw a keyline plough guide freehand, with the live slope
        read-out from DrawLineTool as the grade cue. The drawn line is added to a
        'Drawn Keylines' layer and becomes the master for 'Convert to Swale'."""
        from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool

        tool = DrawLineTool(
            self._canvas,
            slope_raster_path=self._state.slope_raster_path,
            tool_label="keyline plough guide",
        )
        tool.line_drawn.connect(self._on_keyline_drawn)
        tool.cancelled.connect(
            lambda: self._canvas.unsetMapTool(self._canvas.mapTool())
        )
        self._draw_keyline_tool = tool
        self._canvas.setMapTool(tool)

    def _on_keyline_drawn(self, geom):
        self._canvas.unsetMapTool(self._canvas.mapTool())
        self._draw_keyline_tool = None

        from qgis.core import QgsLineSymbol

        layer = resolve_layer(self._project, self._state.drawn_keyline_layer_id)
        if layer is None:
            crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
            layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Drawn Keylines", "memory")
            sym = QgsLineSymbol.createSimple({
                "color": "150,90,30", "width": "1.6", "capstyle": "round",
            })
            layer.setRenderer(QgsSingleSymbolRenderer(sym))
            self._project.instance().addMapLayer(layer)
            self._state.drawn_keyline_layer_id = layer.id()

        f = QgsFeature()
        f.setGeometry(geom)
        layer.dataProvider().addFeatures([f])
        layer.updateExtents()
        layer.triggerRepaint()

        self._state.keyline_master_geom = geom
        self._state.keyline_master_coords = [(p.x(), p.y()) for p in geom.asPolyline()]
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "Keyline drawn — use 'Convert Keyline → Swale' to add it to the design.",
        )

    def _clip_runs_to_usable(self, runs):
        poly = getattr(self._state, "usable_polygon", None)
        if poly is None:
            return runs
        out = []
        for run in runs:
            try:
                clipped = run["geometry"].intersection(poly)
            except Exception:
                out.append(run)
                continue
            if clipped.is_empty:
                continue
            if clipped.geom_type == "MultiLineString":
                clipped = max(clipped.geoms, key=lambda g: g.length)
            if clipped.geom_type != "LineString" or clipped.length <= 0:
                continue
            run = dict(run)
            run["geometry"] = clipped
            out.append(run)
        return out

    def _display_keylines(self, runs, keypoint):
        from qgis.core import QgsLineSymbol

        for name in ("Keyline Design", "Keyline Keypoint"):
            for lyr in self._project.instance().mapLayersByName(name):
                self._project.instance().removeMapLayer(lyr)

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Keyline Design", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("line_type",   QMetaType.QString),
            QgsField("elevation",   QMetaType.Double),
            QgsField("cross_grade", QMetaType.Double),
        ])
        layer.updateFields()

        feats = []
        for run in runs:
            xy = [QgsPointXY(x, y) for x, y, *_ in run["geometry"].coords]
            if len(xy) < 2:
                continue
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPolylineXY(xy))
            f.setAttributes([run["line_type"], run["elevation"], run["cross_grade"]])
            feats.append(f)
        pr.addFeatures(feats)

        # Keyline solid brown-gold; cultivation guides dashed grey-green.
        color_expr = (
            "CASE WHEN \"line_type\" = 'keyline' THEN color_rgb(150,90,30)"
            " ELSE color_rgb(90,140,90) END"
        )
        width_expr = "CASE WHEN \"line_type\" = 'keyline' THEN 1.8 ELSE 0.9 END"
        style_expr = "CASE WHEN \"line_type\" = 'keyline' THEN 'solid' ELSE 'dash' END"
        symbol = QgsLineSymbol.createSimple({"width": "1.0", "capstyle": "round"})
        sl = symbol.symbolLayer(0)
        sl.setDataDefinedProperty(QgsSymbolLayer.PropertyStrokeColor,
                                  QgsProperty.fromExpression(color_expr))
        sl.setDataDefinedProperty(QgsSymbolLayer.PropertyStrokeWidth,
                                  QgsProperty.fromExpression(width_expr))
        sl.setDataDefinedProperty(QgsSymbolLayer.PropertyStrokeStyle,
                                  QgsProperty.fromExpression(style_expr))
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        self._project.instance().addMapLayer(layer)
        self._state.keyline_layer_id = layer.id()

        # Remember the master keyline so it can be converted to a swale.
        for run in runs:
            if run["line_type"] == "keyline":
                xy = [(x, y) for x, y, *_ in run["geometry"].coords]
                self._state.keyline_master_coords = xy
                self._state.keyline_master_geom = QgsGeometry.fromPolylineXY(
                    [QgsPointXY(x, y) for x, y in xy]
                )
                break

        # Keypoint marker.
        kp_layer = QgsVectorLayer("Point", "Keyline Keypoint", "memory")
        kp_layer.setCrs(self._project.instance().crs())
        kpr = kp_layer.dataProvider()
        kpr.addAttributes([QgsField("elevation", QMetaType.Double)])
        kp_layer.updateFields()
        kf = QgsFeature()
        kf.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(keypoint["x"], keypoint["y"])))
        kf.setAttributes([keypoint["elevation"]])
        kpr.addFeatures([kf])
        kp_sym = QgsMarkerSymbol.createSimple({
            "name": "star", "color": "200,140,0,240",
            "outline_color": "120,70,0", "size": "9",
        })
        kp_layer.setRenderer(QgsSingleSymbolRenderer(kp_sym))
        kp_layer.updateExtents()
        self._project.instance().addMapLayer(kp_layer)

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
