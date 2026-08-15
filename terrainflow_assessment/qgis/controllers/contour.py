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

from terrainflow_assessment.modules.contour_analysis import INFLOW_RAMP_HEX
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers._layers import remove_layer, resolve_layer
from terrainflow_assessment.qgis.controllers._tools import MapToolMixin

# Width, in mm, for the four inflow bands — the primary signal, not decoration.
# Over aerial imagery width is the one channel the background cannot destroy: a
# 4 mm line is obviously fatter than a 0.7 mm one whatever is underneath, whereas
# a colour step can be wiped out by a sunlit paddock or a tree shadow. The ~6x
# spread is deliberate; a subtle one is no better than colour alone.
_INFLOW_BAND_WIDTHS = (0.7, 1.4, 2.5, 4.0)

# Narrower ramp for the overlay drawn *inside* the swale segments, so the green
# verdict outline still shows as a rim around the widest band.
_SEGMENT_BAND_WIDTHS = (0.5, 1.0, 1.7, 2.5)

# White halo under every banded line. Does the work colour cannot: it separates
# the pale bands from sunlit grass and the dark bands from tree shadow, so one
# style stays legible across a whole aerial photo.
_INFLOW_CASING = QColor(255, 255, 255, 200)


def _ramp_colour(i):
    """Band *i* of the shared inflow ramp as a QColor (clamped to the ramp length)."""
    return QColor(INFLOW_RAMP_HEX[max(0, min(i, len(INFLOW_RAMP_HEX) - 1))])


class ContourController(G.LayerTreeMixin, MapToolMixin):
    def teardown(self):
        """Take this controller's tool off the canvas. Nothing else to undo."""
        self.release_tool()

    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas
        # Guards the table↔map selection loop: selecting rows selects features,
        # which fires selectionChanged, which would select rows again.
        self._syncing_selection = False

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
            # A fresh run replaces the candidates, so anything derived from the old
            # ones (the top-N subset, the segments the overlay grades, and the two
            # gradients drawn from them) is stale. Dropping the gradient layers
            # matters more than it looks: the gradient switches the candidate
            # contours off while it is up, so a stale one left behind would keep
            # the new candidates hidden.
            self._state.top_contour_features = []
            self._state.segment_features = []
            for attr in ("inflow_bands_layer_id", "segment_gradient_layer_id"):
                remove_layer(self._project, getattr(self._state, attr))
                setattr(self._state, attr, None)
            self._panel.set_contour_complete()
            self._panel.set_contour_results(contours)
            self._display_contour_layer(contours)
            self._panel.set_contour_legend(self._state.contour_breaks,
                                           self._contour_value_unit())
        except Exception as exc:
            self._panel.set_contour_complete()
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Contour analysis failed: {exc}"
            )

    def _contour_value(self, feat):
        """The quantity the candidate contours are banded and ranked on.

        Inflow volume when the storm and cell size are known — that is the number
        on the panel row, so the map and the list band on the same figure. Raw
        accumulation otherwise; the two are proportional, so the bands land in the
        same places, only the legend units change.
        """
        inflow = feat.inflow_m3
        return float(inflow) if inflow is not None else float(feat.peak_acc or 0.0)

    def _contour_value_unit(self):
        feats = self._state.contour_features
        return "m³" if feats and feats[0].inflow_m3 is not None else "cells"

    def _display_contour_layer(self, contours):
        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Candidate Contour Swales", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            # cid ties a map feature back to its row in the panel list (and back
            # again), which is what makes the two selections one selection.
            QgsField("cid", QMetaType.Int),
            QgsField("elevation", QMetaType.Double),
            QgsField("rank", QMetaType.Int),
            QgsField("peak_acc", QMetaType.Double),
            QgsField("inflow_m3", QMetaType.Double),
            QgsField("mean_slope", QMetaType.Double),
        ])
        layer.updateFields()

        feats = []
        for i, feat in enumerate(contours):
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(feat.geometry.wkt))
            f.setAttributes([i, feat.elevation, feat.rank or 0, feat.peak_acc,
                             self._contour_value(feat), feat.mean_slope_deg])
            feats.append(f)
        pr.addFeatures(feats)

        self._apply_inflow_tier_style(layer, contours)
        remove_layer(self._project, self._state.contour_layer_id)
        self.place(layer, G.CONTOUR)
        self._state.contour_layer_id = layer.id()
        self._connect_contour_selection(layer)
        self._apply_contour_filter()

    def _reset_contour_map_tool(self):
        """Deactivate the active map tool if it is a contour-picking tool, so a
        swapped/deleted contour layer can't be dereferenced by a stale tool."""
        active = self._canvas.mapTool()
        if active is not None and active.__class__.__name__ in (
            "ContourSegmentTool", "SelectContourTool",
        ):
            self._canvas.unsetMapTool(active)

    def _apply_inflow_tier_style(self, layer, contours):
        """Band the candidate contours by inflow **value**, not by rank position.

        Natural breaks over the inflow figures, coloured on the shared low→high
        ramp. The old "#1 / top 5 / top 10 / rest" scheme banded by queue position:
        it drew a fixed five contours orange whether the fifth carried nearly as
        much water as the first or a twentieth of it, which is the opposite of what
        the colour was being read to mean.
        """
        from terrainflow_assessment.modules.contour_analysis import natural_breaks

        values = [self._contour_value(c) for c in contours]
        breaks = natural_breaks(values, n_classes=len(INFLOW_RAMP_HEX))
        self._state.contour_breaks = list(breaks)
        unit = "m³" if contours and contours[0].inflow_m3 is not None else "cells"
        self._apply_banded_renderer(layer, "inflow_m3", breaks, unit=unit)

    @staticmethod
    def _band_symbol(i, widths, casing):
        """One band's line symbol: a coloured core over an optional white halo."""
        from qgis.core import QgsLineSymbol, QgsSimpleLineSymbolLayer
        from qgis.PyQt.QtCore import Qt as _Qt

        width = widths[min(i, len(widths) - 1)]
        symbol = QgsLineSymbol.createSimple({"capstyle": "round", "joinstyle": "round"})
        symbol.setColor(_ramp_colour(i))
        symbol.setWidth(width)
        if casing:
            halo = QgsSimpleLineSymbolLayer(_INFLOW_CASING)
            halo.setWidth(width + casing)
            halo.setPenCapStyle(_Qt.RoundCap)
            halo.setPenJoinStyle(_Qt.RoundJoin)
            symbol.insertSymbolLayer(0, halo)   # beneath the core
        return symbol

    @classmethod
    def _apply_banded_renderer(cls, layer, attr, breaks, unit="m³",
                               widths=_INFLOW_BAND_WIDTHS, casing=1.1):
        """Band *layer* on *attr* using the shared ramp: colour **and** width.

        Ranges are built explicitly rather than through ``createRenderer``, which
        would re-derive its own classes from the layer — the point of passing
        breaks in is that the candidate contours, the panel legend, the gradient
        and the segment overlay are all drawn on the same boundaries.

        Four bands, not sixteen. A sixteen-class ramp over a line on an aerial
        photo is not readable: adjacent classes differ by a few percent of
        lightness and the background varies far more than that between one metre
        and the next. Four bands, each a distinct width, can be read at a glance.
        """
        from qgis.core import QgsGraduatedSymbolRenderer, QgsRendererRange

        # Nothing to band (no features, or every one carrying the same figure) —
        # one colour is what the data actually says.
        if len(breaks) < 3:
            layer.setRenderer(QgsSingleSymbolRenderer(
                cls._band_symbol(len(INFLOW_RAMP_HEX) - 1, widths, casing)))
            return

        ranges = []
        for i in range(len(breaks) - 1):
            # Nudge every lower bound but the first so the ranges do not overlap on
            # a shared boundary value (QGIS tests lower <= v <= upper per range).
            lower = breaks[i] if i == 0 else breaks[i] + 1e-9
            ranges.append(QgsRendererRange(
                lower, breaks[i + 1], cls._band_symbol(i, widths, casing),
                f"{breaks[i]:,.0f} – {breaks[i + 1]:,.0f} {unit}",
            ))
        layer.setRenderer(QgsGraduatedSymbolRenderer(attr, ranges))

    # ------------------------------------------------------- Row ↔ map wiring

    def _connect_contour_selection(self, layer):
        """Mirror a map-side feature selection back onto the panel's contour rows."""
        try:
            layer.selectionChanged.connect(self._on_contour_layer_selection)
        except Exception:
            pass

    def _on_contour_layer_selection(self, *_args):
        if self._syncing_selection:
            return
        layer = resolve_layer(self._project, self._state.contour_layer_id)
        if layer is None:
            return
        try:
            cids = sorted(int(f["cid"]) for f in layer.selectedFeatures())
        except Exception:
            return
        self._syncing_selection = True
        try:
            self._panel.select_contour_rows(cids)
        finally:
            self._syncing_selection = False

    def set_contour_visibility(self, unticked_indices):
        """Hide the contours the user has unticked — the checkboxes, made real.

        The tick also decides what "Select Top Swales" and the inflow gradient work
        from, so unticking a contour takes it out of the analysis, not just off
        the screen.

        Takes the *unticked* rows rather than the ticked ones because the panel
        list caps its display at 50: a contour past the cap has no checkbox to
        read, and inferring "unticked" from its absence would rule out every
        contour the user never had the chance to see.
        """
        unticked = set(int(i) for i in unticked_indices)
        for i, feat in enumerate(self._state.contour_features):
            feat.selected = i not in unticked
        self._apply_contour_filter()

    def _visible_contour_filter(self, field):
        """A QGIS filter expression on *field* hiding unticked contours ("" = all)."""
        hidden = [i for i, f in enumerate(self._state.contour_features)
                  if not getattr(f, "selected", True)]
        if not hidden:
            return ""
        return f'"{field}" NOT IN ({",".join(str(i) for i in hidden)})'

    def _apply_contour_filter(self):
        """Push the tick state onto every layer that draws per-contour geometry.

        The gradient carries the contour it came from, so it can be filtered by the
        same expression instead of reclassified — unticking a contour should not
        cost a pass over the accumulation raster.
        """
        for layer_id, field in (
            (self._state.contour_layer_id, "cid"),
            (self._state.inflow_bands_layer_id, "source_id"),
        ):
            layer = resolve_layer(self._project, layer_id)
            if layer is None:
                continue
            try:
                layer.setSubsetString(self._visible_contour_filter(field))
            except Exception as exc:
                print(f"TerrainFlow Assessment — contour visibility filter failed: {exc}")
        self._canvas.refresh()

    def highlight_contour_rows(self, indices):
        """Select the map features for the rows the user picked in the list.

        Uses the layer's own selection rather than a private rubber band, so the
        highlight is the same object in both directions: clicking the contour on
        the map with QGIS's Select Features tool ticks the row back.
        """
        layer = resolve_layer(self._project, self._state.contour_layer_id)
        if layer is None:
            return
        wanted = set(int(i) for i in indices)
        try:
            ids, geoms = [], []
            for f in layer.getFeatures():
                if int(f["cid"]) in wanted:
                    ids.append(f.id())
                    geoms.append(QgsGeometry(f.geometry()))
        except Exception:
            return
        self._syncing_selection = True
        try:
            layer.selectByIds(ids)
        finally:
            self._syncing_selection = False
        # Flash a single pick — on a multi-row selection the flashes overlap into
        # noise and the selection colour already carries it.
        if len(geoms) == 1:
            try:
                self._canvas.flashGeometries(geoms)
            except Exception:
                pass
        self._canvas.refresh()

    def select_top5_contours(self):
        if not self._state.contour_features:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run contour analysis first."
            )
            return

        # Unticked contours are out of the running: "best 5" has to mean best five
        # of the ones still on the table, or the tick achieves nothing.
        candidates = [f for f in self._state.contour_features
                      if getattr(f, "selected", True)]
        if not candidates:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Every contour is unticked — tick at least one to select from.",
            )
            return

        top_n = self._panel.top_n
        ranked = sorted(
            candidates,
            key=lambda f: f.peak_acc if f.peak_acc is not None else 0,
            reverse=True,
        )[:top_n]
        self._state.top_contour_features = ranked

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
        self.place(layer, G.CONTOUR)
        self._state.top5_layer_id = layer.id()

        skipped = len(self._state.contour_features) - len(candidates)
        msg = f"Top {len(ranked)} of {len(candidates)} candidate contour(s) selected."
        if skipped:
            msg += f" {skipped} unticked contour(s) excluded."
        self._iface.messageBar().pushInfo("TerrainFlow Assessment", msg)

        # The gradient scopes itself to this subset, so a narrower pick has to
        # redraw it rather than leave the previous scope on screen.
        if self._panel.inflow_bands_active:
            self.show_inflow_bands(True)

    # ---------------------------------------------------------------- Clear / inflow bands

    def clear_analysis(self):
        """Remove all analysis layers and reset state so the user can re-run cleanly."""
        self._reset_contour_map_tool()
        proj = self._project.instance()
        for attr in ("contour_layer_id", "top5_layer_id", "segment_layer_id",
                     "segment_gradient_layer_id",
                     "simple_contour_layer_id", "keyline_layer_id",
                     "drawn_keyline_layer_id", "inflow_bands_layer_id"):
            remove_layer(self._project, getattr(self._state, attr, None))
            setattr(self._state, attr, None)
        for name in ("Keypoints", "Ridgelines (Water Divides)", "Recommended Pond Sites",
                     "Keyline Design", "Keyline Keypoint"):
            for lyr in proj.mapLayersByName(name):
                proj.removeMapLayer(lyr)
        self._state.contour_features = []
        self._state.top_contour_features = []
        self._state.segment_features = []
        self._state.contour_breaks = []
        self._state.found_keypoints = None
        self._state.keyline_master_geom = None
        self._state.keyline_master_coords = None
        self._panel.clear_analysis_ui()
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment", "Analysis layers cleared."
        )
        self._canvas.refresh()

    def _gradient_scope(self):
        """Which contours the inflow gradient grades, and how to describe that.

        The top-N pick when the user has made one: once they have narrowed to the
        swales they are actually considering, grading every contour on the site
        paints over the answer with a hundred lines nobody asked about. Falls back
        to the ticked candidates before any pick has been made.

        Returns (features, source_ids, description). *source_ids* are positions in
        ``contour_features`` so the tick filter still reaches the stretches.
        """
        by_identity = {id(f): i for i, f in enumerate(self._state.contour_features)}
        top = [f for f in self._state.top_contour_features
               if getattr(f, "selected", True)]
        if top:
            return (top, [by_identity.get(id(f), -1) for f in top],
                    f"top {len(top)} selected swale(s)")
        ticked = [(i, f) for i, f in enumerate(self._state.contour_features)
                  if getattr(f, "selected", True)]
        return ([f for _, f in ticked], [i for i, _ in ticked],
                f"{len(ticked)} ticked contour(s)")

    def show_inflow_bands(self, checked):
        """Toggle the along-contour inflow-share classification layer."""
        remove_layer(self._project, self._state.inflow_bands_layer_id)
        self._state.inflow_bands_layer_id = None
        if not checked:
            self._show_candidate_contours(True)
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
        features, source_ids, scope = self._gradient_scope()
        if not features:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Nothing in scope — tick a contour, or run 'Select Top Swales'.",
            )
            return
        from terrainflow_assessment.modules.contour_analysis import classify_contour_inflow
        try:
            stretches = classify_contour_inflow(
                features, acc_path,
                cell_area_m2=self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0,
                runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
                duration_hr=self._panel.duration_hr,
                source_ids=source_ids,
            )
            self._display_inflow_gradient(stretches, self._panel.inflow_scale_mode,
                                          scope=scope)
        except Exception as exc:
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Inflow classification failed: {exc}"
            )

    def _show_candidate_contours(self, visible):
        """Check/uncheck the candidate contour layer in the tree.

        The gradient is drawn on exactly the same lines, so with both on you get
        two renderings of one geometry — the wider candidate line fringing out from
        under the gradient, and neither readable. The gradient answers the same
        question in more detail, so while it is up the candidates step aside.
        """
        layer = resolve_layer(self._project, self._state.contour_layer_id)
        if layer is None:
            return
        try:
            node = self._project.instance().layerTreeRoot().findLayer(layer.id())
        except Exception:
            return
        if node is not None:
            node.setItemVisibilityChecked(bool(visible))

    def _display_inflow_gradient(self, stretches, scale="natural", scope=""):
        """Render contour stretches in four inflow bands keyed to the global range,
        so stretches are comparable across the whole map.

        *scale* only decides where the four boundaries fall — see
        :func:`contour_analysis.class_breaks`. The bands themselves are the shared
        colour+width grammar, so this reads the same way as the candidate contours
        and the segment overlay.
        """
        from terrainflow_assessment.modules.contour_analysis import class_breaks

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Contour Inflow (m³)", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("source_id", QMetaType.Int),
            QgsField("inflow_m3", QMetaType.Double),
            QgsField("flow_ls", QMetaType.Double),
        ])
        layer.updateFields()
        feats = []
        values = []
        for s in stretches:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(s["geometry"].wkt))
            # Clamp: a negative inflow is not a thing, and it would drag the
            # bottom boundary below zero where no band could reach it.
            inflow = max(0.0, float(s["inflow_m3"]))
            values.append(inflow)
            f.setAttributes([int(s.get("source_id", -1)), inflow, float(s["flow_ls"])])
            feats.append(f)

        if not feats:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No contour inflow to show — run contour analysis first, and check "
                "the stream threshold is not above everything on the site.",
            )
            return
        pr.addFeatures(feats)

        breaks = class_breaks(values, mode=scale, n_classes=len(INFLOW_RAMP_HEX))
        self._apply_banded_renderer(layer, "inflow_m3", breaks)

        # Above the candidate contours in the group; the candidates themselves are
        # switched off, since this is the same geometry read in more detail.
        self.place(layer, G.CONTOUR, at_top=True)
        self._state.inflow_bands_layer_id = layer.id()
        self._show_candidate_contours(False)
        # Unticked contours stay hidden in the gradient too — it is the same
        # geometry wearing a different colour.
        self._apply_contour_filter()

        if len(breaks) < 3:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Contour inflow is uniform across the site, so there is no gradient "
                "to band — shown in a single colour.",
            )
        elif scope:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment", f"Inflow gradient shown for the {scope}."
            )

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
            self._state.segment_features = segments
            self._panel.set_segment_complete()
            self._panel.set_segment_results(segments)
            self._display_swale_segments(segments)
            # A new set of segments needs a new overlay, not the previous one
            # sitting over geometry that has moved.
            if self._panel.segment_gradient_active:
                self.show_segment_gradient(True)
        except Exception as exc:
            self._panel.set_segment_complete()
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Segment analysis failed: {exc}"
            )

    def _display_swale_segments(self, segments):
        from qgis.core import QgsTextBufferSettings
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

        self._style_swale_segments(layer, self._panel.segment_gradient_active)

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

        self.place(layer, G.CONTOUR)
        self._state.segment_layer_id = layer.id()
        n_capped = sum(1 for s in segments if s.capped)
        msg = f"{len(segments)} swale segment(s) found. Top segment: {segments[0].label}"
        if n_capped:
            msg += (
                f"  ⚠ {n_capped} contour(s) too short to hold the design inflow — "
                "consider a deeper/wider swale, a basin, or splitting the catchment."
            )
        self._iface.messageBar().pushSuccess("TerrainFlow Assessment", msg)

    def _style_swale_segments(self, layer, gradient_on=False):
        """Distinct "swale" style: a white casing under a bold core coloured by
        whether the swale holds its inflow (green) or needs an overflow (amber),
        so it reads clearly as the recommended swale — separate from the thin
        ranked candidate contours and the slope-coloured flow lines.

        With the peak-inflow overlay on, the core widens to 4.0 mm so the widest
        gradient band (2.5 mm) still leaves ~0.75 mm of green showing either side.
        The green/amber verdict is the point of this layer — where the water
        concentrates is extra information about the same swale, so it is drawn
        within the outline rather than in place of it. The green also acts as the
        overlay's backdrop, which is why the gradient carries no white halo here.
        """
        from qgis.core import QgsLineSymbol, QgsSimpleLineSymbolLayer
        from qgis.PyQt.QtCore import Qt as _Qt
        color_expr = (
            "CASE WHEN \"capped\" = 1 THEN color_rgb(230,126, 34)"
            " ELSE color_rgb( 39,174, 96) END"
        )
        core_w, casing_w = (4.0, 5.6) if gradient_on else (1.8, 3.6)
        symbol = QgsLineSymbol.createSimple({"width": str(core_w), "capstyle": "round"})
        symbol.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(color_expr))
        casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 190))
        casing.setWidth(casing_w)
        casing.setPenCapStyle(_Qt.RoundCap)
        symbol.insertSymbolLayer(0, casing)  # draw casing beneath the core
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()

    def show_segment_gradient(self, checked):
        """Toggle the peak-inflow gradient drawn inside the recommended segments.

        A plain green outline says a swale belongs on this stretch of contour but
        not *where along it* the water actually arrives — which is where the
        crossing, the deepest section and any overflow want to go. Same ramp and
        same numbers as the contour gradient, so the two read as one scheme.
        """
        remove_layer(self._project, self._state.segment_gradient_layer_id)
        self._state.segment_gradient_layer_id = None

        seg_layer = resolve_layer(self._project, self._state.segment_layer_id)
        if seg_layer is not None:
            self._style_swale_segments(seg_layer, checked)

        if not checked:
            self._canvas.refresh()
            return
        if not self._state.segment_features:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run 'Find Best Swale Segments' first."
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
                self._state.segment_features, acc_path,
                cell_area_m2=self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0,
                runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
                duration_hr=self._panel.duration_hr,
                # A swale segment is short, so the 3-sample window that suits a whole
                # contour would reduce it to a couple of blocks.
                window=2,
            )
        except Exception as exc:
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Segment inflow classification failed: {exc}"
            )
            return
        self._display_segment_gradient(stretches)

    def _display_segment_gradient(self, stretches):
        from terrainflow_assessment.modules.contour_analysis import class_breaks

        if not stretches:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No inflow found along the recommended segments to grade.",
            )
            return

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        layer = QgsVectorLayer(f"LineString?crs={crs_str}",
                               "Swale Segment Inflow (m³)", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("seg_rank",  QMetaType.Int),
            QgsField("inflow_m3", QMetaType.Double),
            QgsField("flow_ls",   QMetaType.Double),
        ])
        layer.updateFields()
        feats = []
        values = []
        for s in stretches:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(s["geometry"].wkt))
            inflow = max(0.0, float(s["inflow_m3"]))
            values.append(inflow)
            f.setAttributes([int(s.get("source_id", -1)) + 1, inflow,
                             float(s["flow_ls"])])
            feats.append(f)
        pr.addFeatures(feats)

        # Same bands, same colours and the same width ordering as the contour
        # gradient — the overlay is that scheme read at segment scale, not a second
        # one. Narrower widths and no halo: it has to fit inside the green core,
        # which is already doing the halo's job of separating it from the ground.
        # Boundaries follow the same scale control as the contour gradient, so both
        # views answer "how much" the same way.
        self._apply_banded_renderer(
            layer, "inflow_m3",
            class_breaks(values, mode=self._panel.inflow_scale_mode,
                         n_classes=len(INFLOW_RAMP_HEX)),
            widths=_SEGMENT_BAND_WIDTHS, casing=None)

        # Above the segments in the group, or the green core it belongs inside
        # would be painted over it.
        self.place(layer, G.CONTOUR, at_top=True)
        self._state.segment_gradient_layer_id = layer.id()
        self._canvas.refresh()

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
            self.place(layer, G.ANALYSIS)
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
            # The pond raster goes with the accumulation: inside a contracted pond the
            # accumulation is no longer contributing area, and every test in here reads it
            # as if it were. Without it a reservoir floor comes back as a ridgeline.
            ka = DrainageLineAnalysis(
                self._state.dem_path, acc_path,
                (self._state.baseline_result or {}).get("pond_flow"),
            )
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
                ka = DrainageLineAnalysis(
                    self._state.dem_path, acc_path,
                    (self._state.baseline_result or {}).get("pond_flow"),
                )

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
        self.use_tool(tool)

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
            self.place(layer, G.KEYPOINT)
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
        self.place(layer, G.KEYPOINT)
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
        self.place(kp_layer, G.KEYPOINT)

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
        self.place(layer, G.KEYPOINT)

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
        self.place(layer, G.KEYPOINT)

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
        self.place(layer, G.KEYPOINT)
