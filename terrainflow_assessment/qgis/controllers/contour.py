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

from terrainflow_assessment.modules.contour_analysis import (
    INFLOW_RAMP_HEX,
    SEGMENT_INFLOW_RAMP_HEX,
)
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers._layers import (
    crs_object,
    dem_crs,
    remove_layer,
    resolve_layer,
)
from terrainflow_assessment.qgis.controllers._tools import MapToolMixin
from terrainflow_assessment.qgis.workers._lifecycle import worker_is_running
from terrainflow_assessment.qgis.workers.task_worker import TaskWorker

# Width, in mm, for the four inflow bands — the primary signal, not decoration.
# Over aerial imagery width is the one channel the background cannot destroy: a
# 4 mm line is obviously fatter than a 0.7 mm one whatever is underneath, whereas
# a colour step can be wiped out by a sunlit paddock or a tree shadow. The ~6x
# spread is deliberate; a subtle one is no better than colour alone.
_INFLOW_BAND_WIDTHS = (0.7, 1.4, 2.5, 4.0)

# Narrower ramp for the overlay drawn over the swale segments. It used to be sized so
# the green verdict outline showed as a rim around the widest band — that stopped being
# expressible when the segment core moved to metres-in-map-units: a rim in millimetres
# around a core in metres holds at exactly one scale. The bands stay in millimetres on
# purpose. They rank where inflow concentrates along an alignment, and `_symbols`' rule
# is that magnitude and identity are drawn in millimetres while structure is drawn in
# metres. Narrower than `_INFLOW_BAND_WIDTHS` so the two overlays remain distinguishable.
_SEGMENT_BAND_WIDTHS = (0.5, 1.0, 1.7, 2.5)

# White halo under every banded line. Does the work colour cannot: it separates
# the pale bands from sunlit grass and the dark bands from tree shadow, so one
# style stays legible across a whole aerial photo.
_INFLOW_CASING = QColor(255, 255, 255, 200)


def _ramp_colour(i, ramp=INFLOW_RAMP_HEX):
    """Band *i* of an inflow ramp as a QColor (clamped to the ramp length)."""
    return QColor(ramp[max(0, min(i, len(ramp) - 1))])


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

    # ---------------------------------------------------------------- Threaded tasks

    def _claim_worker(self, what):
        """True if this controller's worker slot is free, having said so if not.

        The four analyses here share one slot. They all read the same DEM and the
        same accumulation raster and each takes seconds, so running two at once
        buys nothing and costs the memory of two full grids — and the second would
        overwrite ``contour_worker``, which is the only reference keeping the first
        thread alive.
        """
        if worker_is_running(self._state, "contour_worker"):
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{what} is waiting — another analysis is still running.")
            return False
        return True

    def _start_task(self, work, label, on_progress, on_done, on_failed, failure_title):
        """Run *work* on this controller's worker and wire up its three outcomes.

        *on_failed* re-arms the button; it is called on the error path only, since
        the success path re-arms inside *on_done* where the summary text is known.
        """
        worker = TaskWorker(work, label=label)
        worker.progress.connect(on_progress)
        worker.completed.connect(on_done)

        def _failed(tb):
            on_failed()
            # The console gets the traceback; the bar gets the headline. A message
            # that says "see the console" with nothing in it is worse than silence.
            print(f"TerrainFlow Assessment — {failure_title}:")
            print(tb)
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment",
                f"{failure_title} — see the Python console.")

        worker.error.connect(_failed)
        self._state.contour_worker = worker
        # Disabled at click time, not when the first progress arrives: the gap
        # between them is a real window and a second click lands in it.
        on_progress(1, "Starting…")
        worker.start()

    # ---------------------------------------------------------------- Contour analysis

    def suggest_spacing(self):
        """Derive a contour interval from the terrain instead of asking for a guess.

        Reads the slope raster over the usable area, then asks both spacing rules —
        the terrace vertical interval for erosion control, and how wide an upslope
        strip the drawn section can actually hold — and reports the one that governs.

        Cheap enough to run inline: a percentile over the slope array is tens of
        milliseconds, so there is no worker and no progress to report.
        """
        # The precondition is a loaded DEM, not a baseline: the slope raster is written
        # by ``on_dem_changed``. The erosion rule needs only slope, so the advice is
        # available immediately; the capture rule needs a storm depth and degrades to
        # "erosion governs" until Baseline has produced one.
        slope_path = self._state.slope_raster_path
        if not slope_path or not os.path.exists(slope_path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Load a DEM first — the advice is read off its slope raster.")
            return

        import numpy as np
        import rasterio

        from terrainflow_assessment.core.sizing import spacing_advisory
        from terrainflow_assessment.modules.swale_design import capacity_per_metre
        from terrainflow_assessment.modules.terrain_indices import slope_statistics

        with rasterio.open(slope_path) as src:
            slope = src.read(1).astype("float64")
            nodata = src.nodata
        if nodata is not None:
            slope = np.where(slope == nodata, np.nan, slope)

        stats = slope_statistics(slope)
        if stats is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "No usable slope values — the DEM may be entirely nodata here.")
            return

        panel = self._panel
        capacity = capacity_per_metre(
            panel.swale_depth_m, panel.swale_width_m,
            side_slope=panel.swale_side_slope,
            duration_hr=panel.duration_hr,
        )
        runoff_mm = (self._state.baseline_result or {}).get("runoff_mm")

        # Advised at the MEDIAN slope, with the quartiles printed beside it. A farm is
        # not one slope, and a single figure is the answer that hides the paddock the
        # advice is wrong for.
        advice = spacing_advisory(
            stats["p50_grade"] * 100.0,
            soil_name=panel.earthwork_soil_name,
            runoff_mm=runoff_mm,
            capacity_m3_per_m=capacity or None,
        )

        vi = advice["vertical_interval_m"]
        spread = (f"Ground runs {stats['p25']:.1f}° / {stats['p50']:.1f}° / "
                  f"{stats['p75']:.1f}° (quartiles). ")
        if advice["recommended_spacing_m"] is None:
            self._panel.set_spacing_advice(spread + advice["text"])
            return

        text = (f"{spread}{advice['text']} Interval set to {vi:.1f} m; "
                f"features about {advice['recommended_spacing_m']:.0f} m apart.")
        self._panel.set_spacing_advice(text, interval_m=round(vi, 1))

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

        if not self._claim_worker("Contour analysis"):
            return

        # Deactivate any contour-picking map tool so it can't outlive the layer
        # swap (its captured layer would become a dead reference → crash).
        self._reset_contour_map_tool()

        from terrainflow_assessment.modules.contour_analysis import analyse_contours

        # Every setting is read here, on the GUI thread, and captured. Reaching back
        # into the panel from inside the work would be reading Qt state off the
        # worker thread — the fault this whole change exists to remove.
        dem_path = self._state.dem_path
        interval_m = self._panel.contour_interval_m
        max_slope_deg = self._panel.max_slope_deg
        usable_polygon = self._state.usable_polygon
        cell_area_m2 = (self._state.dem_info.cell_area_m2
                        if self._state.dem_info else None)
        runoff_mm = (self._state.baseline_result or {}).get("runoff_mm")
        min_length_m = self._panel.min_contour_length_m

        # Collected on the worker thread, pushed on the GUI thread in _on_contours_ready.
        # A clip that quietly took every contour used to be indistinguishable from an
        # analysis that legitimately found none.
        self._contour_warnings = []

        def work(report):
            return analyse_contours(
                dem_path=dem_path,
                acc_path=acc_path,
                interval_m=interval_m,
                max_slope_deg=max_slope_deg,
                usable_polygon=usable_polygon,
                progress_callback=report,
                cell_area_m2=cell_area_m2,
                runoff_mm=runoff_mm,
                min_length_m=min_length_m,
                on_warning=self._contour_warnings.append,
            )

        self._start_task(work, "contours",
                         self._panel.set_contour_progress,
                         self._on_contours_ready,
                         self._panel.set_contour_complete,
                         "Contour analysis failed")

    def _on_contours_ready(self, contours):
        """Back on the GUI thread — every layer below belongs to it."""
        for message in getattr(self, "_contour_warnings", []):
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", message)
        self._contour_warnings = []
        # Number them as they arrive. Everything downstream that has to say
        # *which* contour it means — the tick filter, the inflow gradient's
        # scope — needs a name for one, and position in this list is the only
        # thing a ContourFeature has that is unique: two of them can share an
        # elevation, a rank and a length.
        for i, feature in enumerate(contours):
            feature.index = i
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
        crs_str = dem_crs(self._state)
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
    def _band_symbol(i, widths, casing, ramp=INFLOW_RAMP_HEX):
        """One band's line symbol: a coloured core over an optional white halo."""
        from qgis.core import QgsLineSymbol, QgsSimpleLineSymbolLayer
        from qgis.PyQt.QtCore import Qt as _Qt

        width = widths[min(i, len(widths) - 1)]
        symbol = QgsLineSymbol.createSimple({"capstyle": "round", "joinstyle": "round"})
        symbol.setColor(_ramp_colour(i, ramp))
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
                               widths=_INFLOW_BAND_WIDTHS, casing=1.1,
                               ramp=INFLOW_RAMP_HEX):
        """Band *layer* on *attr* using an inflow ramp: colour **and** width.

        Ranges are built explicitly rather than through ``createRenderer``, which
        would re-derive its own classes from the layer — the point of passing
        breaks in is that the candidate contours, the panel legend, the gradient
        and the segment overlay are all drawn on the same boundaries.

        Four bands, not sixteen. A sixteen-class ramp over a line on an aerial
        photo is not readable: adjacent classes differ by a few percent of
        lightness and the background varies far more than that between one metre
        and the next. Four bands, each a distinct width, can be read at a glance.

        ``ramp`` is a parameter because the segment overlay is drawn inside the
        swale's own green core rather than on ground, and takes the contrasting
        ``SEGMENT_INFLOW_RAMP_HEX`` for it. Breaks, widths and band count stay
        shared — only the hues differ, so the two views are one scheme.
        """
        from qgis.core import QgsGraduatedSymbolRenderer, QgsRendererRange

        # Nothing to band (no features, or every one carrying the same figure) —
        # one colour is what the data actually says.
        if len(breaks) < 3:
            layer.setRenderer(QgsSingleSymbolRenderer(
                cls._band_symbol(len(ramp) - 1, widths, casing, ramp)))
            return

        ranges = []
        for i in range(len(breaks) - 1):
            # Nudge every lower bound but the first so the ranges do not overlap on
            # a shared boundary value (QGIS tests lower <= v <= upper per range).
            lower = breaks[i] if i == 0 else breaks[i] + 1e-9
            ranges.append(QgsRendererRange(
                lower, breaks[i + 1], cls._band_symbol(i, widths, casing, ramp),
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

        crs_str = dem_crs(self._state)
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
        for attr in ("contour_layer_id", "top5_layer_id", "segment_layer_id",
                     "segment_gradient_layer_id",
                     "simple_contour_layer_id", "keyline_layer_id",
                     "drawn_keyline_layer_id", "inflow_bands_layer_id",
                     "keypoints_layer_id", "ridgelines_layer_id",
                     "pond_sites_layer_id", "keyline_keypoint_layer_id"):
            remove_layer(self._project, getattr(self._state, attr, None))
            setattr(self._state, attr, None)
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
        # `f.index`, not `id(f)`. Object identity is exact only while the two
        # lists hold the same objects, and says -1 the moment anything hands
        # back a copy — a reload, a round-trip through the worker — which reads
        # as "not a candidate" and quietly drops that contour from the scope.
        top = [f for f in self._state.top_contour_features
               if getattr(f, "selected", True)]
        if top:
            return (top, [getattr(f, "index", -1) for f in top],
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

        crs_str = dem_crs(self._state)
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

        if not self._claim_worker("Segment analysis"):
            return

        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        from terrainflow_assessment.modules.swale_design import get_infiltration_rate

        # Read on the GUI thread, captured for the worker. See run_contour_analysis.
        kwargs = dict(
            contours=self._state.contour_features,
            acc_path=acc_path,
            cell_area_m2=(self._state.dem_info.cell_area_m2
                          if self._state.dem_info else 1.0),
            runoff_mm=(self._state.baseline_result or {}).get("runoff_mm"),
            min_acc_ha=self._panel.min_catchment_ha,
            swale_depth_m=self._panel.swale_depth_m,
            swale_width_m=self._panel.swale_width_m,
            # Derived on the panel from the three entered dimensions, not typed in.
            # `find_swale_segments` has accepted this since the trapezoid went in and
            # nothing ever passed it, so its own 1.0 default was the site's batter
            # whatever the design said. The sizing was trapezoidal all along; the third
            # dimension of the trapezoid was simply unreachable from the criteria box.
            side_slope=self._panel.swale_side_slope,
            infiltration_mm_hr=get_infiltration_rate(self._panel.earthwork_soil_name),
            duration_hr=self._panel.duration_hr,
            rank_mode=self._panel.segment_rank_mode,
            slope_path=self._state.slope_raster_path,
            seg_max_slope_deg=self._panel.seg_max_slope_deg,
        )

        def work(report):
            return find_swale_segments(progress_callback=report, **kwargs)

        self._start_task(work, "segments",
                         self._panel.set_segment_progress,
                         self._on_segments_ready,
                         self._panel.set_segment_complete,
                         "Segment analysis failed")

    def _on_segments_ready(self, segments):
        self._state.segment_features = segments
        self._panel.set_segment_complete()
        self._panel.set_segment_results(segments)
        self._display_swale_segments(segments)
        # A new set of segments needs a new overlay, not the previous one
        # sitting over geometry that has moved.
        if self._panel.segment_gradient_active:
            self.show_segment_gradient(True)

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

        crs_str = dem_crs(self._state)
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
            # The top width these lengths were sized against, carried on the feature so
            # the symbol can draw the segment at the width it means. `_symbols.BAND_EXPR`
            # reads this field by name, which is what lets the recommendation be measured
            # off the map instead of being a constant-thickness ribbon.
            QgsField("width_m",           QMetaType.Double),
        ])
        layer.updateFields()

        # Read once, on the GUI thread, from the same criteria box that sized the
        # segments — so the line cannot drift from the number that produced it.
        top_width_m = float(self._panel.swale_width_m or 0.0)

        feats = []
        for seg in segments:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(seg.geometry.wkt))
            f.setAttributes([
                seg.label, seg.elevation, seg.contributing_ha,
                seg.inflow_m3, seg.required_length_m, seg.segment_rank,
                1 if seg.capped else 0, top_width_m,
            ])
            feats.append(f)
        pr.addFeatures(feats)

        self._style_swale_segments(layer)

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

    def _style_swale_segments(self, layer):
        """Distinct "swale" style: a white casing under a bold core coloured by
        whether the swale holds its inflow (green) or needs an overflow (amber),
        so it reads clearly as the recommended swale — separate from the thin
        ranked candidate contours and the slope-coloured flow lines.

        **Drawn at the top width it was sized for.** This was the last layer in the
        plugin still styled in millimetres, so a recommendation zoomed in looked like a
        farm track and the same recommendation zoomed out looked like the same farm
        track — the one thing a proposed swale drawn over a hillside has to say is how
        much of that hillside it takes, and the line said nothing about it. It now
        follows ``_symbols``' rule and its constants: structure in metres, data-defined
        off ``width_m``, floored by ``minSizeMM`` so it survives a zoom-out rather than
        vanishing (at 0.6 m it goes sub-pixel somewhere above 1:3000).

        The ``gradient_on`` parameter is gone with it. It widened the core to 4.0 mm so
        the widest 2.5 mm inflow band left ~0.75 mm of green showing either side; against
        a core that is now a real width, that relationship holds at one scale and at no
        other — at 1:100 the core swallows the band, at 1:2000 the band swallows the
        core. The overlay stays in millimetres deliberately: it ranks where water
        arrives, and a ranking is identity, not structure.
        """
        from qgis.core import (
            QgsLineSymbol,
            QgsSimpleLineSymbolLayer,
            QgsUnitTypes,
        )
        from qgis.PyQt.QtCore import Qt as _Qt

        from ._symbols import (
            BAND_EXPR,
            BAND_MIN_MM,
            CASING_EXPR,
            CASING_MIN_MM,
            _clamped,
        )
        color_expr = (
            "CASE WHEN \"capped\" = 1 THEN color_rgb(230,126, 34)"
            " ELSE color_rgb( 39,174, 96) END"
        )
        # Static fallbacks in metres, used only if `width_m` is missing — which it is
        # not, `_display_swale_segments` writes it — so `coalesce(...,0)` can never
        # collapse the whole layer onto `minSizeMM` and quietly restore the old look.
        fallback_m = self._panel.swale_width_m or 1.0

        casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 190))
        casing.setWidth(fallback_m + 0.6)
        casing.setWidthUnit(QgsUnitTypes.RenderMetersInMapUnits)
        casing.setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth, QgsProperty.fromExpression(CASING_EXPR))
        casing.setPenCapStyle(_Qt.RoundCap)
        _clamped(casing, CASING_MIN_MM)

        core = QgsSimpleLineSymbolLayer(QColor(39, 174, 96))
        core.setWidth(fallback_m)
        core.setWidthUnit(QgsUnitTypes.RenderMetersInMapUnits)
        core.setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth, QgsProperty.fromExpression(BAND_EXPR))
        core.setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(color_expr))
        core.setPenCapStyle(_Qt.RoundCap)
        _clamped(core, BAND_MIN_MM)

        symbol = QgsLineSymbol()
        symbol.changeSymbolLayer(0, casing)   # casing beneath
        symbol.appendSymbolLayer(core)
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

        # The segment core no longer changes with the overlay — it is the swale's real
        # top width either way — so there is nothing to restyle here.

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

        crs_str = dem_crs(self._state)
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

        # Same bands and the same width ordering as the contour gradient — the
        # overlay is that scheme read at segment scale, not a second one. Narrower
        # widths and no halo: it has to fit inside the green core, which is already
        # doing the halo's job of separating it from the ground. Boundaries follow
        # the same scale control as the contour gradient, so both views answer "how
        # much" the same way.
        #
        # The hues are the one thing that does differ, and for the same reason the
        # halo is dropped: this ramp is read against the core rather than against
        # the ground, and cyan→navy inside a green band is a hue step small enough
        # that the thin low bands disappeared into it.
        self._apply_banded_renderer(
            layer, "inflow_m3",
            class_breaks(values, mode=self._panel.inflow_scale_mode,
                         n_classes=len(SEGMENT_INFLOW_RAMP_HEX)),
            widths=_SEGMENT_BAND_WIDTHS, casing=None,
            ramp=SEGMENT_INFLOW_RAMP_HEX)

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
            import rasterio
            interval = self._panel.simple_contour_interval_m
            # EXTRA and NODATA for the same reason as
            # contour_analysis.extract_contours: gdal_contour takes the range of
            # levels it emits from the band statistics and accepts the approximate
            # ones QGIS caches in a PAM sidecar, which are computed from a decimated
            # sample and so read a summit lower than it is — the contours above that
            # under-read maximum are simply never drawn. PAM off makes it read the
            # real pixels; the nodata it would have found there is passed on instead.
            with rasterio.open(self._state.dem_path) as src:
                nodata = src.nodata
            params = {
                "INPUT": self._state.dem_path,
                "BAND": 1,
                "INTERVAL": interval,
                "FIELD_NAME": "ELEV",
                "EXTRA": "--config GDAL_PAM_ENABLED NO",
                "OUTPUT": "TEMPORARY_OUTPUT",
            }
            if nodata is not None:
                params["NODATA"] = float(nodata)
            result = processing.run("gdal:contour", params)
            out = result.get("OUTPUT")
            if hasattr(out, "source"):
                out = out.source()

            # A picking tool may be holding this layer — the swale tools draw from
            # it now, not just from the analysed candidates — and regenerating
            # would leave it pointing at a layer the project no longer owns.
            self._reset_contour_map_tool()
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
                # Same conversion as the usable area's, through the same adapter —
                # this path was the one that had it right, and keeping two of them is
                # how the other one came to be missing its reprojection.
                from terrainflow_assessment.qgis.adapters.geom import (
                    polygons_in_dem_crs,
                )
                polys = polygons_in_dem_crs(layer, crs_wkt)
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

        if not self._claim_worker("Keypoint analysis"):
            return

        from terrainflow_assessment.modules.keypoint_analysis import DrainageLineAnalysis

        # The boundary mask is built here rather than in the worker: it rasterises a
        # project layer, and the vector layers belong to the GUI thread.
        self._panel.set_keypoint_progress(5, "Building boundary mask…")
        dem_path = self._state.dem_path
        boundary_mask = self._get_keypoint_boundary_mask(dem_path)
        cell_m2 = self._state.dem_info.cell_area_m2 if self._state.dem_info else 100.0
        min_acc = max(50, int(1.0 * 10_000 / cell_m2))
        n_keypoints = self._panel.keypoint_count
        pond_path = (self._state.baseline_result or {}).get("pond_flow")

        def work(report):
            report(25, "Finding keypoints…")
            # The pond raster goes with the accumulation: inside a contracted pond the
            # accumulation is no longer contributing area, and every test in here reads
            # it as if it were. Without it a reservoir floor comes back as a ridgeline.
            ka = DrainageLineAnalysis(dem_path, acc_path, pond_path)
            keypoints = ka.find_keypoints(
                min_acc_cells=min_acc,
                n_keypoints=n_keypoints,
                boundary_mask=boundary_mask,
            )
            report(70, "Finding ridgelines…")
            ridgelines = ka.find_ridgelines(boundary_mask=boundary_mask)
            return ka, keypoints, ridgelines

        self._start_task(work, "keypoints",
                         self._panel.set_keypoint_progress,
                         self._on_keypoints_ready,
                         lambda: self._panel.set_keypoint_complete(""),
                         "Keypoint analysis failed")

    def _on_keypoints_ready(self, result):
        ka, keypoints, ridgelines = result
        # Held for "Recommend Pond Sites", which reuses the loaded rasters.
        self._state.keyline_analysis = ka
        self._state.found_keypoints = keypoints

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

    def run_recommend_ponds(self):
        """Rank impoundment sites by storage held per cubic metre of embankment.

        **On the worker, not the GUI thread.** This used to run inline with a manual
        progress poke, which was affordable only while it was a proxy score over a
        vectorised width. It now floods a bounded window per candidate per trial wall
        height, so it costs seconds — and this is the very function that once spent
        300 s parked on a modal dialog nobody could click offscreen.
        """
        if not self._state.found_keypoints:
            QMessageBox.warning(self._panel, "No Keypoints",
                                "Run 'Find Keypoints + Ridgelines' first.")
            return

        if not self._claim_worker("Pond site ranking"):
            return

        # Every input read here, on the GUI thread.
        dem_path = self._state.dem_path
        acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
        runoff_mm = (self._state.baseline_result or {}).get("runoff_mm")
        boundary_mask = self._get_keypoint_boundary_mask(dem_path)
        keypoints = list(self._state.found_keypoints)
        max_sites = max(1, int(self._panel.keypoint_count))

        def work(report):
            return _rank_pond_sites(dem_path, acc_path, keypoints, boundary_mask,
                                    runoff_mm, max_sites, report)

        self._start_task(work, "pond sites",
                         self._panel.set_ponds_progress,
                         self._on_pond_sites_ready,
                         lambda: self._panel.set_ponds_complete(""),
                         "Pond site ranking failed")

    def _on_pond_sites_ready(self, pond_sites):
        self._state.pond_sites = pond_sites
        self._display_pond_sites(pond_sites)
        self._panel.set_keypoint_results(
            self._keypoint_result_items(
                self._state.found_keypoints, pond_sites=pond_sites)
        )
        usable = [s for s in pond_sites if not s.get("notes")]
        refused = len(pond_sites) - len(usable)
        summary = (f"{len(self._state.found_keypoints)} valley points | "
                   f"{len(usable)} site(s) ranked")
        if refused:
            # Named, not dropped: a candidate the user can see was considered and
            # refused is worth more than a quietly shorter list.
            summary += f", {refused} refused"
        self._panel.set_ponds_complete(summary + ".")

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

        if not self._claim_worker("Keyline analysis"):
            return

        from terrainflow_assessment.modules.keypoint_analysis import (
            YeomansKeylineAnalysis,
        )

        dem_path = self._state.dem_path
        n_runs = self._panel.keyline_runs
        max_grade_n = self._panel.keyline_max_grade_n
        spacing_m = self._panel.keyline_spacing_m
        max_valleys = self._panel.keyline_max_valleys
        # Only used when no baseline has been run, so there is no accumulation raster to
        # inherit a routing scheme from — see `_ensure_flow_data`. KPA-53.
        routing = self._panel.routing

        def work(report):
            report(10, "Finding primary valleys…")
            ya = YeomansKeylineAnalysis(dem_path, acc_path=acc_path, routing=routing)

            # One keypoint per PRIMARY valley — a Strahler order-1 link, which is what
            # Yeomans means by a primary valley. The single largest stream is the trunk
            # of the catchment and is not one, so the old pass applied the right
            # criterion to the wrong feature.
            report(40, "Locating keypoints…")
            keypoints, skipped = ya.find_keypoints(max_valleys=max_valleys)
            if not keypoints:
                # Fall back to the historical single-stem answer rather than returning
                # nothing: on a small or single-valley DEM there may be no order-1 link
                # long enough to profile, and the old answer is still an answer.
                one = ya.find_keypoint()
                if one is None:
                    return [], [], skipped
                one.setdefault("_row", one["row"])
                one.setdefault("_col", one["col"])
                one.setdefault("label", f"Keypoint at {one['elevation']:.1f} m")
                keypoints = [one]

            report(70, "Generating cultivation guides…")
            runs = []
            for index, keypoint in enumerate(keypoints, start=1):
                for run in ya.get_cultivation_runs(
                        keypoint, n_runs=n_runs, max_grade_n=max_grade_n,
                        spacing_m=spacing_m):
                    run["valley"] = index
                    runs.append(run)
            return keypoints, runs, skipped

        self._start_task(work, "keyline",
                         self._panel.set_keyline_progress,
                         self._on_keyline_ready,
                         lambda: self._panel.set_keyline_complete(""),
                         "Keyline analysis failed")

    def _on_keyline_ready(self, result):
        keypoints, runs, skipped = result
        if not keypoints:
            self._panel.set_keyline_complete("")
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "No keypoint found — the DEM area may be too small or too flat.",
            )
            return

        # Clip to the usable area (analysis/earthworks polygon) when set. Left on
        # this side because it reads `state.usable_polygon`, which the user can
        # change while the analysis runs.
        runs = self._clip_runs_to_usable(runs)
        self._state.keyline_keypoints = keypoints
        self._display_keylines(runs, keypoints[0])

        guides = [r for r in runs if r["line_type"] != "keyline"]
        flagged = [r for r in guides if r.get("over_limit")]
        # The steepest sustained fall, not the net one: a guide can start and finish at
        # nearly the same height while running steeply in the middle, and the net figure
        # reads 86x gentler than the ground does on this fixture (KPA-41). `over_limit`
        # is judged on this, so the readout quotes the same thing the flag counts.
        drifts = [r["steepest_1_in_n"] for r in guides
                  if r.get("steepest_1_in_n") is not None]
        window = next((r.get("steepest_window_m") for r in guides
                       if r.get("steepest_window_m")), None)

        summary = (f"{len(keypoints)} primary valley(s) | "
                   f"{len(guides)} cultivation guide(s)")
        if drifts:
            # The number nobody had ever measured. "The drift is emergent from
            # parallelism" had stood in a docstring since this feature was written;
            # this is the measurement of it.
            summary += (f" | steepest drift 1:{min(drifts):.0f}–1:{max(drifts):.0f}"
                        + (f" over {window:.0f} m" if window else ""))
        if flagged:
            summary += f" | {len(flagged)} steeper than the limit"
        self._panel.set_keyline_complete(summary + ".")

        if skipped:
            # Refusals are named, not swallowed — the house style everywhere else here.
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{len(skipped)} valley(s) had no keypoint: {skipped[0]}")

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
            crs_str = dem_crs(self._state)
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

        for attr in ("keyline_layer_id", "keyline_keypoint_layer_id"):
            remove_layer(self._project, getattr(self._state, attr, None))
            setattr(self._state, attr, None)

        crs_str = dem_crs(self._state)
        layer = QgsVectorLayer(f"LineString?crs={crs_str}", "Keyline Design", "memory")
        pr = layer.dataProvider()
        # ``cross_grade`` is gone. It carried the value of a spin box that never
        # reached the geometry, so the attribute table asserted a grade the lines did
        # not have — a sharper fault than the control merely doing nothing. What
        # replaces it is measured: the drift each guide actually achieves.
        pr.addAttributes([
            QgsField("line_type",    QMetaType.QString),
            QgsField("valley",       QMetaType.Int),
            QgsField("elevation",    QMetaType.Double),
            QgsField("offset_m",     QMetaType.Double),
            QgsField("drift_1_in_n", QMetaType.Double),
            QgsField("steepest_1_in_n", QMetaType.Double),
            QgsField("drift_fall_m", QMetaType.Double),
            QgsField("over_limit",   QMetaType.Bool),
        ])
        layer.updateFields()

        feats = []
        for run in runs:
            xy = [QgsPointXY(x, y) for x, y, *_ in run["geometry"].coords]
            if len(xy) < 2:
                continue
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPolylineXY(xy))
            f.setAttributes([
                run["line_type"], run.get("valley", 1), run["elevation"],
                run.get("offset_m"), run.get("drift_1_in_n"),
                run.get("steepest_1_in_n"),
                run.get("drift_fall_m"), bool(run.get("over_limit")),
            ])
            feats.append(f)
        pr.addFeatures(feats)

        # Keyline solid brown-gold; guides dashed, and a guide whose measured drift
        # exceeds the limit is drawn in a warning red so the flag is on the map and
        # not only in the table.
        color_expr = (
            "CASE WHEN \"over_limit\" THEN color_rgb(190,60,40)"
            " WHEN \"line_type\" = 'keyline' THEN color_rgb(150,90,30)"
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
        kp_layer.setCrs(crs_object(dem_crs(self._state)))
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
        self._state.keyline_keypoint_layer_id = kp_layer.id()

    def _display_keypoints(self, keypoints):
        remove_layer(self._project, self._state.keypoints_layer_id)
        self._state.keypoints_layer_id = None

        layer = QgsVectorLayer("Point", "Keypoints", "memory")
        layer.setCrs(crs_object(dem_crs(self._state)))
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
        self._state.keypoints_layer_id = layer.id()

    def _display_ridgelines(self, ridgelines):
        from qgis.core import QgsLineSymbol
        remove_layer(self._project, self._state.ridgelines_layer_id)
        self._state.ridgelines_layer_id = None

        layer = QgsVectorLayer("LineString", "Ridgelines (Water Divides)", "memory")
        layer.setCrs(crs_object(dem_crs(self._state)))
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
        self._state.ridgelines_layer_id = layer.id()

    def _display_pond_sites(self, sites):
        remove_layer(self._project, self._state.pond_sites_layer_id)
        self._state.pond_sites_layer_id = None

        layer = QgsVectorLayer("Point", "Ranked Pond Sites", "memory")
        layer.setCrs(crs_object(dem_crs(self._state)))
        pr = layer.dataProvider()
        # Every column beside the ratio is a trade-off the user reads, not a term in
        # the rank. A blended score would hide which of them drove the ordering, which
        # is the failure KPA-20 already demonstrated.
        pr.addAttributes([
            QgsField("label",          QMetaType.QString),
            QgsField("rank",           QMetaType.Int),
            QgsField("storage_ratio",  QMetaType.Double),
            QgsField("storage_m3",     QMetaType.Double),
            QgsField("fill_m3",        QMetaType.Double),
            QgsField("wall_height_m",  QMetaType.Double),
            QgsField("wall_length_m",  QMetaType.Double),
            QgsField("elevation",      QMetaType.Double),
            QgsField("catchment_ha",   QMetaType.Double),
            QgsField("fills_in",       QMetaType.Double),
            QgsField("refused",        QMetaType.QString),
        ])
        layer.updateFields()
        feats = []
        for s in sites:
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(s["x"], s["y"])))
            f.setAttributes([
                s.get("label", ""), s.get("rank"), s.get("storage_ratio"),
                s.get("storage_m3"), s.get("fill_m3"),
                s.get("wall_height_m"), s.get("wall_length_m"),
                s.get("elevation"), s.get("catchment_ha"),
                s.get("fills_in_events"), s.get("notes"),
            ])
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
        self._state.pond_sites_layer_id = layer.id()


def _rank_pond_sites(dem_path, acc_path, keypoints, boundary_mask,
                     runoff_mm, max_sites, report):
    """Worker body for the impoundment ranking. No Qt beyond the progress callable.

    Candidates are screened before they are measured. The old code scanned a
    ``(2·search_r + 1)²`` box around every keypoint — thousands of cells — because each
    was a cheap proxy; measuring storage is not cheap, so this takes a handful of cells
    down the stream below each keypoint and measures only those. The screen is the whole
    reason the sweep is affordable, and removing it as "an approximation" would put the
    run back into minutes.
    """
    import numpy as np
    import rasterio

    from terrainflow_assessment.modules.impoundment_sites import rank_impoundment_sites

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata
        cell_w = abs(transform.a)
        cell_h = abs(transform.e)
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    if acc_path and os.path.exists(acc_path):
        with rasterio.open(acc_path) as src:
            acc = src.read(1).astype("float64")
            acc_nodata = src.nodata
        if acc_nodata is not None:
            acc = np.where(acc == acc_nodata, 0.0, acc)
    else:
        acc = np.zeros_like(dem)

    def rc_to_xy(row, col):
        # GDAL-consistent cell-centre form, matching keypoint_analysis._rc_to_xy.
        return (transform.c + (col + 0.5) * transform.a,
                transform.f + (row + 0.5) * transform.e)

    rows, cols = dem.shape
    candidates = []
    seen = set()
    for kp in keypoints[:max_sites]:
        r0, c0 = kp.get("_row"), kp.get("_col")
        if r0 is None or c0 is None:
            continue
        # Walk a short way downstream: a dam sits below the keypoint, not on it.
        r, c = int(r0), int(c0)
        for _step in range(12):
            best, best_acc = None, float(acc[r, c])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr < rows - 1 and 1 <= nc < cols - 1):
                        continue
                    if not np.isfinite(dem[nr, nc]):
                        continue
                    if boundary_mask is not None and not boundary_mask[nr, nc]:
                        continue
                    if float(acc[nr, nc]) > best_acc:
                        best, best_acc = (nr, nc), float(acc[nr, nc])
            if best is None:
                break
            r, c = best
            if (r, c) not in seen:
                seen.add((r, c))
                candidates.append((r, c))

    if not candidates:
        return []

    return rank_impoundment_sites(
        dem, acc, candidates, cell_w, cell_h, rc_to_xy,
        runoff_mm=runoff_mm, progress=report)
