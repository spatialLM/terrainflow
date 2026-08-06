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
    QgsUnitTypes,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType, QObject, pyqtSignal
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.modules.dem_loader import compute_slope_raster, load_dem
from terrainflow_assessment.modules.earthwork_design import DEMBurner
from terrainflow_assessment.modules.reporting import BaselineReport
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers import _layers as L
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker

# --------------------------------------------------------------------------- Exit labels
# Exit markers and their labels are sized against the ground rather than the screen, so
# they keep a constant relationship to the terrain instead of swamping it as you zoom out,
# and are clamped at both ends so they never become illegible or overbearing.
#
# The ground sizes reproduce the previous fixed-screen look at 1:1184 (the scale the old
# styling was judged correct at): 1 mm on screen is 1.184 m on the ground there, so the old
# 10 pt text is ~4.2 m and the old 5 mm marker is ~5.9 m.
#
# The size is computed here in Python and re-applied on scale change, rather than handed to
# QGIS as a map-unit size with a QgsMapUnitScale min/max. That looks like the natural fit
# and is not: on QGIS 3.40 the mm clamp bounds the glyph size but *not* the text layout, so
# the advances stay frozen at the unclamped width and the label renders stretched to ~3x —
# measured at ratio w/h 10 (correct) against 25-40 (clamped map units). Points and
# millimetres both lay out correctly, so the arithmetic lives here instead.
_EXIT_LABEL_GROUND_M = 4.2      # text cap height in metres on the ground
_EXIT_LABEL_MIN_PT = 8.0        # floor — legible against imagery with the buffer
_EXIT_LABEL_MAX_PT = 14.0       # ceiling when zoomed right in
_EXIT_MARKER_GROUND_M = 6.0     # marker diameter in metres on the ground
_EXIT_MARKER_MIN_MM = 1.5
_EXIT_MARKER_MAX_MM = 7.0

# A ground length of G metres subtends G * 1000 / scale millimetres on the page, and a
# point is 25.4/72 mm.
_MM_PER_M = 1000.0
_PT_PER_MM = 72.0 / 25.4


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def exit_label_size_pt(map_scale):
    """Label point size for a map scale (1:*map_scale*), clamped at both ends."""
    if not map_scale or map_scale <= 0:
        return _EXIT_LABEL_MAX_PT
    ground_pt = _EXIT_LABEL_GROUND_M * _MM_PER_M * _PT_PER_MM / map_scale
    return round(_clamp(ground_pt, _EXIT_LABEL_MIN_PT, _EXIT_LABEL_MAX_PT), 1)


def exit_marker_size_mm(map_scale):
    """Marker diameter in millimetres for a map scale, clamped at both ends."""
    if not map_scale or map_scale <= 0:
        return _EXIT_MARKER_MAX_MM
    ground_mm = _EXIT_MARKER_GROUND_M * _MM_PER_M / map_scale
    return round(_clamp(ground_mm, _EXIT_MARKER_MIN_MM, _EXIT_MARKER_MAX_MM), 2)


def _crs_label(crs):
    if crs is None:
        return "unknown CRS"
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    return crs.to_string()


class BaselineController(G.LayerTreeMixin, QObject):
    # Emitted only after a baseline run *succeeds*. Earthworks that were restored before
    # the run — as happens when a design file is opened — hold no catchment labels yet, so
    # something has to re-score them once terrain results exist. Deliberately not emitted
    # on the error path: a failed run has nothing to score against.
    baseline_finished = pyqtSignal()

    def __init__(self, state, panel, project, iface, canvas):
        super().__init__()
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas
        # Exit-points layers whose marker/label size tracks the map scale. Ids, not
        # layers: the user can delete these from the legend at any time.
        self._exit_layer_ids = []
        try:
            self._canvas.scaleChanged.connect(self.on_map_scale_changed)
        except Exception:
            pass

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
        # Auto-apply the Analysis Area as the contour/keypoint clip so analysis
        # stays inside the boundary (matches the field's tooltip).
        if layer is not None:
            self._panel.set_usable_area_source("analysis")

    def on_earthworks_area_changed(self, layer):
        self._state.earthworks_area_path = self._layer_to_path(layer) if layer else None

    def on_site_name_changed(self, name):
        """A name typed after the first run adopts the auto-named output group."""
        try:
            G.rename_default_site(self._project, name)
        except Exception:
            pass

    # ---------------------------------------------------------------- Draw area on canvas

    _AREA_LABELS = {
        "boundary": "Site Boundary",
        "analysis": "Analysis Area",
        "earthworks": "Earthworks Area",
    }

    # Outline-only render colours (RGB) — distinct, high-contrast, and different
    # from the red parcel outline so each drawn area stands out over the map.
    _AREA_OUTLINE = {
        "boundary": "0,162,232",     # bright blue
        "analysis": "255,127,14",    # orange
        "earthworks": "148,103,189",  # purple
    }

    def draw_area(self, kind):
        """Let the user draw a polygon on the canvas for one of the three area
        pickers (boundary / analysis / earthworks). The drawn polygon becomes a
        memory layer that is added to the project and auto-selected in its combo.
        """
        from terrainflow_assessment.map_tools.draw_polygon_tool import DrawPolygonTool

        label = self._AREA_LABELS.get(kind, "Area")
        tool = DrawPolygonTool(
            self._canvas,
            slope_raster_path=self._state.slope_raster_path,
            tool_label=label,
        )
        tool.polygon_drawn.connect(lambda geom: self._on_area_drawn(kind, geom))
        tool.cancelled.connect(self._on_area_draw_cancelled)
        # Keep a reference so the tool is not garbage-collected while active.
        self._draw_area_tool = tool
        self._canvas.setMapTool(tool)

    def _on_area_drawn(self, kind, geometry):
        label = self._AREA_LABELS.get(kind, "Area")
        crs = None
        if self._state.dem_info is not None:
            crs = self._state.dem_info.crs_wkt
        if not crs:
            crs = self._project.instance().crs().toWkt()

        layer = QgsVectorLayer(f"Polygon?crs={crs}", f"Drawn {label}", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([QgsField("name", QMetaType.QString)])
        layer.updateFields()
        feat = QgsFeature()
        feat.setGeometry(geometry)
        feat.setAttributes([label])
        pr.addFeatures([feat])
        layer.updateExtents()

        # Outline-only: a bold coloured boundary with no fill, so the map beneath
        # stays visible (like the parcel layer, but in a standout colour).
        from qgis.core import QgsFillSymbol
        symbol = QgsFillSymbol.createSimple({
            "style": "no",
            "outline_color": self._AREA_OUTLINE.get(kind, "0,162,232"),
            "outline_width": "0.6",
            "outline_style": "solid",
        })
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        # Directly under the site group, below the stage groups: it is an input the
        # user drew, not an output of any one stage.
        self.place(layer, G.SITE)

        self._canvas.unsetMapTool(self._canvas.mapTool())
        self._draw_area_tool = None
        # Selecting the layer re-fires the matching *_changed signal → sets the path.
        self._panel.set_area_layer(kind, layer)
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment", f"{label} drawn and selected."
        )

    def _on_area_draw_cancelled(self):
        self._canvas.unsetMapTool(self._canvas.mapTool())
        self._draw_area_tool = None

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

        # Every stage group is tagged with the storm this run routed, so the tag is
        # frozen here rather than read live — the Analysis and Design layers made
        # afterwards belong to *these* numbers, whatever the spinners say later.
        self._state.run_tag = self._param_tag()

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
            exit_flow_ls=self._panel.exit_flow_ls,
            analysis_area_path=self._state.analysis_area_path,
            earthworks_area_path=self._state.earthworks_area_path,
            sizing_basis=self._panel.sizing_basis,
            runoff_coefficient=self._panel.runoff_coefficient,
        )
        self._state.analysis_worker.progress.connect(self._panel.set_baseline_progress)
        self._state.analysis_worker.finished.connect(self._on_baseline_complete)
        self._state.analysis_worker.error.connect(self._on_analysis_error)
        self._state.analysis_worker.start()

    def _on_baseline_complete(self, result):
        self._state.baseline_result = result
        # New terrain conditioning → the cached flow pointers and catchment labels
        # describe the previous run and must not be reused.
        self._state.invalidate_flow_cache()
        self._load_result_layers(result, is_earthworks=False)

        catchment_ha = result.get("catchment_area_m2", 0) / 10_000.0
        # The comparison report subtracts this from the simulation's routed outflow, so it
        # has to be the true boundary flux. Summing the exit points instead compared a
        # threshold-filtered sum of per-crossing peak cells against a time-integrated
        # total — two different measurements — and flattered the exit-reduction figure.
        # The sum is kept only as a fallback for a run that produced no area totals.
        site_outflow = (result.get("area_outflow", {}).get("site") or {})
        exit_volume_m3 = site_outflow.get("total_volume_m3")
        if exit_volume_m3 is None:
            exit_volume_m3 = sum(
                ep.get("volume_m3", 0) for ep in result.get("exit_points", []))

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
            exit_volume_m3=exit_volume_m3,
            exit_points=result.get("exit_points", []),
        )

        summary = (
            f"Runoff: {result.get('runoff_mm', 0):.1f} mm | "
            f"CN: {result.get('effective_cn', 0):.0f} | "
            f"Exit points: {len(result.get('exit_points', []))}"
        )
        self._panel.set_baseline_complete(summary)
        self._panel.set_area_outflow(result.get("area_outflow", {}),
                                     result.get("ponded_volume_m3"))
        self.baseline_finished.emit()

    def _on_analysis_error(self, tb):
        self._panel.set_baseline_complete("Analysis failed — see Python console for details.")
        print("TerrainFlow Assessment — Analysis error:\n" + tb)
        self._iface.messageBar().pushCritical("TerrainFlow Assessment",
                                               "Analysis failed. See Python console.")

    # ---------------------------------------------------------------- Layer loading

    def _param_tag(self):
        """Abbreviated run parameters for layer-group naming, e.g.
        ``120mm·24h·C0.40·5ha`` — so multiple baseline runs stay distinguishable.

        The middle term follows the Runoff Calculation Method (``runoff_basis_tag``)
        because that is the only calibration number in play: quoting ``CN61`` on a
        Lancaster run would advertise a value the run never read.
        """
        p = self._panel
        try:
            return (f"{p.rainfall_mm:.0f}mm·{p.duration_hr:.0f}h·"
                    f"{p.runoff_basis_tag}·{p.stream_threshold_ha:g}ha")
        except Exception:
            return ""

    def _load_result_layers(self, result, is_earthworks=False):
        label = "Earthworks" if is_earthworks else "Baseline"
        # Re-running replaces this stage's rasters; the group itself is kept and
        # renamed so the rest of the tree stays where the user left it.
        path = G.RERUN if is_earthworks else G.BASELINE
        group = G.clear_group(
            self._project, path,
            site_name=self._panel.site_name, tag=self._state.run_tag,
        )
        layer_ids = []

        def _add(layer, visible=True):
            # Add to the project without the flat legend, then place in the group.
            self._project.instance().addMapLayer(layer, False)
            node = group.addLayer(layer)
            if node is not None:
                # Collapsed: an expanded raster ramp is ~150 px of legend each.
                node.setExpanded(False)
                if not visible:
                    node.setItemVisibilityChecked(False)
            layer_ids.append(layer.id())

        stream_path = result.get("stream_network")
        if stream_path and os.path.exists(stream_path):
            layer = QgsRasterLayer(stream_path, f"{label} — Streams")
            if layer.isValid():
                self.apply_stream_ramp(layer, result.get("stream_acc_max", 1))
                _add(layer)

        # Total event water passing through each cell — the whole surface, not just
        # the cells that pass the stream threshold, so water is visible gathering
        # before it becomes a defined channel. Off by default: it covers the map.
        throughflow_path = result.get("throughflow")
        if throughflow_path and os.path.exists(throughflow_path):
            layer = QgsRasterLayer(throughflow_path, f"{label} — Throughflow (m³)")
            if layer.isValid():
                self.apply_throughflow_ramp(layer, self._panel.throughflow_scale_mode)
                _add(layer, visible=self._panel.throughflow_visible)
                if not is_earthworks:
                    self._state.throughflow_layer_id = layer.id()

        ponding_path = result.get("ponding")
        if ponding_path and os.path.exists(ponding_path):
            self._state.ponding_raster_path = ponding_path
            layer = QgsRasterLayer(ponding_path, f"{label} — Water Captured")
            if layer.isValid():
                self.apply_ponding_ramp(layer)
                _add(layer)

        exit_points = result.get("exit_points", [])
        if exit_points:
            ep_layer = self._create_exit_points_layer(exit_points, label)
            if ep_layer:
                _add(ep_layer)

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
            "outline_color": "140,0,0",
        })
        symbol.setSizeUnit(QgsUnitTypes.RenderMillimeters)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        text_fmt = QgsTextFormat()
        text_fmt.setColor(QColor(160, 0, 0))
        font = QFont()
        font.setBold(True)
        text_fmt.setFont(font)
        # QgsTextFormat.setFont() ignores the QFont's own point size, so without this the
        # format silently keeps its 10 pt default and never responds to the map scale.
        text_fmt.setSizeUnit(QgsUnitTypes.RenderPoints)

        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor(255, 255, 255))
        # Left in millimetres: the halo is there to lift text off aerial imagery, and at the
        # minimum text size a ground-referenced buffer would swamp the glyphs it outlines.
        buf.setSize(1.0)
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

        self._exit_layer_ids.append(layer.id())
        self._apply_exit_scale(layer, self._map_scale())
        return layer

    # ---------------------------------------------------------------- Exit point scaling

    def _map_scale(self):
        try:
            return float(self._canvas.scale())
        except Exception:
            return 0.0

    @staticmethod
    def _apply_exit_scale(layer, map_scale):
        """Resize one exit-points layer's marker and label for *map_scale*.

        Returns True when something changed, so the caller can skip the repaint on the
        many scale-change signals that do not move a clamped size.
        """
        changed = False
        marker_mm = exit_marker_size_mm(map_scale)
        symbol = layer.renderer().symbol() if layer.renderer() else None
        if symbol is not None and symbol.size() != marker_mm:
            symbol.setSize(marker_mm)
            changed = True

        labeling = layer.labeling()
        if labeling is not None:
            settings = labeling.settings()
            fmt = settings.format()
            label_pt = exit_label_size_pt(map_scale)
            if fmt.size() != label_pt:
                fmt.setSize(label_pt)
                settings.setFormat(fmt)
                layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
                changed = True
        return changed

    def on_map_scale_changed(self, *_):
        """Keep exit markers/labels ground-referenced as the user zooms.

        Wired to the canvas rather than expressed as a QGIS map-unit size because the
        map-unit path mis-lays out the text — see the note by ``_EXIT_LABEL_GROUND_M``.
        """
        scale = self._map_scale()
        live = []
        repaint = False
        for lid in self._exit_layer_ids:
            layer = L.resolve_layer(self._project, lid)
            if layer is None:
                continue
            live.append(lid)
            try:
                repaint |= self._apply_exit_scale(layer, scale)
            except Exception:
                pass
        self._exit_layer_ids = live
        if repaint:
            for lid in live:
                layer = L.resolve_layer(self._project, lid)
                if layer is not None:
                    layer.triggerRepaint()

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
        # Stream cells are the only non-zero cells (all exceed the threshold), so
        # jump to a solid, saturated blue immediately above zero — otherwise the
        # thin low-accumulation threads render near-transparent and are hard to see.
        color_ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(0, QColor(0, 0, 0, 0), "none"),
            QgsColorRampShader.ColorRampItem(max_acc * 0.001, QColor(60, 130, 220, 255), "stream"),
            QgsColorRampShader.ColorRampItem(max_acc * 0.4, QColor(25, 85, 190, 255), "channel"),
            QgsColorRampShader.ColorRampItem(max_acc, QColor(8, 32, 110, 255), "main"),
        ])
        shader.setRasterShaderFunction(color_ramp)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        layer.setRenderer(renderer)

    def apply_throughflow_ramp(self, layer, scale="log"):
        """Blue gradient over the whole site: total event water through each cell.

        Off-white where flow is diffuse through to dark blue where it concentrates —
        a different colour family from the green→red slope ramp, keeping the shared
        rule that blue means actual water.

        Flow accumulation is heavily skewed: a handful of channel cells carry orders
        of magnitude more than the hillsides feeding them, so a linear stretch renders
        everything but the main channels as near-white. ``log`` (the default) places
        the stops at decades of the maximum so the minor flow paths stay legible;
        ``linear`` and ``quantile`` mirror the contour-inflow gradient's options.
        """
        try:
            stats = layer.dataProvider().bandStatistics(1)
            max_v = stats.maximumValue or 1.0
        except Exception:
            max_v = 1.0
        if max_v <= 0:
            max_v = 1.0

        if scale == "linear":
            stops = (0.0, 0.15, 0.4, 0.7, 1.0)
        elif scale == "quantile":
            # Even visual weight per band: bunch the stops toward the low end, where
            # the overwhelming majority of cells actually sit.
            stops = (0.0, 0.02, 0.08, 0.25, 1.0)
        else:  # log — decades below the maximum
            stops = (0.0, 1e-4, 1e-3, 1e-2, 1.0)

        # Alpha rises far faster than the colour does. Diffuse sheet flow covers
        # nearly the whole site, so at any weight it reads as a wash over the map
        # rather than an overlay on it — the layer underneath has to stay legible
        # through it, since the point is to see where that flow is going. The
        # gathering and channel stops keep their weight; those are the answer.
        # Mirrored by the inline key in panel.py — change a hue here and change it
        # there, or the key stops describing the map.
        colours = [
            QColor(255, 255, 255, 0),      # nothing flows here — fully transparent
            QColor(226, 240, 250, 40),     # off-white: diffuse sheet flow, ~16%
            QColor(144, 196, 232, 140),
            QColor(48, 122, 190, 225),
            QColor(8, 36, 110, 245),       # dark blue: concentrated channel
        ]
        labels = ["none", "diffuse", "gathering", "concentrated", "channel"]

        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        color_ramp.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(max_v * f, c, lbl)
            for f, c, lbl in zip(stops, colours, labels)
        ])
        shader.setRasterShaderFunction(color_ramp)
        layer.setRenderer(
            QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))

    def set_throughflow_visible(self, visible):
        """Show/hide the throughflow raster without re-running the analysis."""
        from terrainflow_assessment.qgis.controllers._layers import resolve_layer
        layer = resolve_layer(self._project, self._state.throughflow_layer_id)
        if layer is None:
            return
        node = self._project.instance().layerTreeRoot().findLayer(layer.id())
        if node is not None:
            node.setItemVisibilityChecked(bool(visible))
        self._canvas.refresh()

    def set_throughflow_scale(self, _mode=None):
        """Re-stretch the throughflow ramp to the panel's current scale mode."""
        from terrainflow_assessment.qgis.controllers._layers import resolve_layer
        layer = resolve_layer(self._project, self._state.throughflow_layer_id)
        if layer is None:
            return
        self.apply_throughflow_ramp(layer, self._panel.throughflow_scale_mode)
        layer.triggerRepaint()
        self._canvas.refresh()

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
