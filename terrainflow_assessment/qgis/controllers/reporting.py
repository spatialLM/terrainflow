"""
reporting.py — ReportingController

Exports the Site Water Plan as a PDF (or the legacy HTML).

**A baseline is the only precondition.** This used to gate on
``state.comparison``, which is written in exactly one place — the end of a fill
simulation — so a landowner had to run a simulation before the plugin would
produce any document at all. Everything the report actually needs is computed by
the design tier and is now retained on the state: ``balance`` (capture %,
per-feature water, cut/fill, the mass-balance flag), ``balance_stores``,
``spillway_rows`` and ``spillway_context``. A simulation, where one has been
run, is passed through as optional enrichment.

This controller assembles :class:`ReportData` and resolves map layers. Every
decision about *what the report says* lives in
:mod:`terrainflow_assessment.modules.report_model`, which is pure.
"""

from __future__ import annotations

import datetime
import os
import re
import sys

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QApplication, QFileDialog

from terrainflow_assessment.qgis.controllers._layers import resolve_layer

# Characters Windows will not accept in a filename. A site called
# "Smith 1/2 Block" would otherwise produce an unopenable path.
_UNSAFE = re.compile(r'[\\/:*?"<>|]+')


class ReportingController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    # ------------------------------------------------------------------ entry

    def export_report(self):
        if self._state.baseline_report is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run Baseline before exporting a report.")
            return

        site = self._site_name()
        default = os.path.join(
            os.path.expanduser("~"),
            f"{_safe(site)}_WaterPlan_{datetime.date.today():%Y-%m-%d}.pdf")
        path, selected = QFileDialog.getSaveFileName(
            self._iface.mainWindow(), "Save Report", default,
            "PDF (*.pdf);;HTML (*.html)")
        if not path:
            return
        # The dialog already asked which format; a panel combo would only ask
        # the same question twice.
        path, as_html = resolve_target(path, selected)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._export(path, site, as_html)
            self._state.report_last_path = path
            self._panel.mark_stage("report", "done")
            self._iface.messageBar().pushSuccess(
                "TerrainFlow Assessment", f"Report saved to {path}")
            if site == "Unnamed Site":
                self._iface.messageBar().pushInfo(
                    "TerrainFlow Assessment",
                    "The report is titled 'Unnamed Site' — set Site Name on the "
                    "Baseline stage and export again to name it.")
            self._open(path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Export failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------ pdf

    def _export(self, path, site, as_html):
        """Build the document once, then hand it to whichever renderer.

        Both formats consume the same :class:`Report`, the same chart PNGs and
        the same maps, so the only thing they can differ about is paper size.
        """
        from terrainflow_assessment.modules.report_model import build_report

        self._clear_interactive_state()
        data = self._collect(site)
        report = build_report(data)

        work = os.path.join(self._state.output_dir, "report_assets")
        os.makedirs(work, exist_ok=True)
        try:
            images = self._render_charts(data, work)
            if as_html:
                from terrainflow_assessment.modules.report_html import write_html
                write_html(path, report, images=images,
                           maps=self._render_maps(data, work))
            else:
                from terrainflow_assessment.qgis.adapters.layout_pdf import (
                    export_pdf,
                )
                export_pdf(path, self._project.instance(), report,
                           images=images, maps=self._map_specs(data))
        finally:
            # Consumed synchronously, so nothing outlives the export. Both
            # outputs embed what they need and survive output_dir being cleared.
            _rmtree(work)

    def _clear_interactive_state(self):
        """Drop feature selections before the maps are drawn.

        A report figure showing a selected feature is showing whatever the
        operator happened to be clicking on when they hit Export. It means
        nothing to the person reading the document, and a highlighted swale
        among forty reads as a finding. ``contour.highlight_contour_rows``
        selects and never deselects, so this is the only place it comes off.

        The off-screen renderer also refuses to draw selections at all
        (``map_image.render_map_image``); this belt-and-braces pass covers the
        live ``QgsLayoutItemMap`` the PDF path uses. The canvas rubber band is
        a third mechanism and is cleared by the plugin's export wiring, because
        it belongs to the earthworks controller.
        """
        from qgis.core import QgsMapLayer

        for layer in self._project.instance().mapLayers().values():
            try:
                if (layer.type() == QgsMapLayer.VectorLayer
                        and layer.selectedFeatureCount()):
                    layer.removeSelection()
            except Exception:
                continue

    def _render_charts(self, data, work):
        """Every chart the model can reference, as PNG paths.

        Paths rather than base64 so one dict serves both renderers: the layout
        places the file, the HTML inlines it as a data URI.
        """
        from terrainflow_assessment.modules.report_charts import (
            render_fill_timeline,
            render_flow_network,
            render_hydrograph,
        )
        from terrainflow_assessment.modules.report_model import build_flow_graph

        images = {}
        if data.balance is not None and data.balance.per_feature:
            png = os.path.join(work, "network.png")
            if render_flow_network(build_flow_graph(data.balance), path=png):
                images["network"] = png

        comparison = data.comparison
        if comparison is not None:
            png = os.path.join(work, "hydrograph.png")
            if render_hydrograph(comparison.baseline, comparison.post, path=png):
                images["hydrograph"] = png
            png = os.path.join(work, "fill_timeline.png")
            if render_fill_timeline(comparison.post, path=png):
                images["fill_timeline"] = png
        return images

    def _render_maps(self, data, work):
        """Maps as PNGs, for the renderer that cannot host a live map item."""
        from terrainflow_assessment.qgis.adapters.map_image import save_map_png

        out = {}
        for key, spec in self._map_specs(data).items():
            path = os.path.join(work, f"map_{key}.png")
            try:
                if save_map_png(path, spec.layers, extent=spec.extent,
                                crs=spec.crs, size_px=(1500, 900), dpi=150):
                    out[key] = path
            except Exception as exc:
                print(f"TerrainFlow Assessment — map '{key}' failed: {exc}")
        return out

    # ------------------------------------------------------------------ data

    def _collect(self, site):
        from terrainflow_assessment.modules.report_model import ReportData

        state = self._state
        return ReportData(
            site_name=site,
            generated_at=f"{datetime.datetime.now():%Y-%m-%d %H:%M}",
            plugin_version=_plugin_version(),
            run_tag=state.run_tag or "",
            current_tag=self._current_tag(),
            baseline=state.baseline_report,
            baseline_result=state.baseline_result,
            balance=state.balance,
            balance_stores=state.balance_stores,
            earthworks=self._earthworks(),
            verification=state.verification,
            comparison=state.comparison,
            spillway_rows=state.spillway_rows,
            spillway_context=state.spillway_context,
            edits_since_verify=state.edits_since_verify,
            natural_ponding_m3=(state.baseline_result or {}).get(
                "ponded_volume_m3"),
            burn_quantities=state.burn_quantities,
            inputs=self._inputs(),
            dem=self._dem_provenance(),
            maps=self._map_reasons(),
        )

    def _site_name(self):
        """Fall back to something meaningful rather than shipping "Unnamed Site".

        The panel returns that string as its own default, so a blank field puts
        it on the cover of a document the owner is about to hand to someone.
        """
        name = (self._panel.site_name or "").strip()
        if name and name != "Unnamed Site":
            return name
        title = (self._project.instance().title() or "").strip()
        if title:
            return title
        dem = self._dem_basename()
        return dem or "Unnamed Site"

    def _dem_basename(self):
        """The DEM's name, unless it is one the plugin generated itself.

        Anything under ``output_dir`` is a working file — a restored design
        writes ``restored_dem.tif``, a clip writes ``design_dem_clip.tif`` — and
        putting that on the cover is worse than admitting the site is unnamed.
        """
        path = self._state.dem_path
        if not path:
            return ""
        try:
            work = os.path.normcase(os.path.abspath(self._state.output_dir))
            if os.path.normcase(os.path.abspath(path)).startswith(work):
                return ""
        except Exception:
            pass
        return os.path.splitext(os.path.basename(path))[0]

    def _current_tag(self):
        """Re-derive the storm tag now, to catch inputs changed since Baseline."""
        from terrainflow_assessment.qgis.controllers.baseline import param_tag
        return param_tag(self._panel)

    def _earthworks(self):
        manager = getattr(self._state, "earthwork_manager", None)
        try:
            return list(manager.get_all()) if manager else None
        except Exception:
            return None

    def _inputs(self):
        from terrainflow_assessment.modules.project_io import INPUT_FIELDS

        out = {}
        for field in INPUT_FIELDS:
            try:
                value = getattr(self._panel, field)
            except Exception:
                continue
            if value not in (None, ""):
                out[field] = value
        return out

    def _dem_provenance(self):
        state = self._state
        info, out = state.dem_info, {}
        if state.dem_path:
            out["source"] = state.dem_path
            out["fingerprint"] = _digest(state.dem_path)
        if info is not None:
            out["cell size"] = f"{info.cell_size_m:.2f} m"
            crs = getattr(info, "crs", None)
            if crs:
                out["crs"] = str(crs)
        return out

    # ------------------------------------------------------------------ maps

    def _map_reasons(self):
        """Why a map cannot be drawn, keyed the same way as the specs."""
        reasons = {}
        if not self._layers_for("design"):
            reasons["design"] = (
                "No earthwork layers are on the map to draw a design plan from.")
        if not self._layers_for("flow"):
            reasons["flow"] = (
                "The baseline flow layers are no longer on the map. Re-run "
                "Baseline to draw this.")
        before, after = self._ponding_pair()
        if not (before and after):
            reasons["ponding"] = (
                "A before-and-after comparison needs the water-captured layers "
                "from both Baseline and Re-analyse with Earthworks.")
        elif not self._ponding_tags_agree():
            reasons["ponding"] = (
                "The before and after layers come from different runs, so they "
                "cannot be compared. Re-run Baseline and Re-analyse together.")
        return reasons

    def _map_specs(self, data):
        from terrainflow_assessment.qgis.adapters.layout_pdf import MapSpec
        from terrainflow_assessment.qgis.adapters.map_image import (
            layers_extent,
            usable_layers,
        )

        crs = self._project.crs()
        frame = self._frame_extent(crs)
        specs = {}
        for key in ("design", "flow"):
            layers = usable_layers(self._layers_for(key))
            if layers:
                specs[key] = MapSpec(
                    layers, frame or layers_extent(layers, crs), crs)
        before, after = self._ponding_pair()
        if before and after and self._ponding_tags_agree():
            layers = usable_layers(
                [self._boundary_layer(), after, before]
                + self._terrain_layers())
            if layers:
                specs["ponding"] = MapSpec(
                    layers, frame or layers_extent(layers, crs), crs,
                    height_mm=110.0)
        return specs

    def _frame_extent(self, crs):
        """One frame for every map: the site boundary, where there is one.

        Each map used to take the union of its own layer extents, which is
        dominated by whichever raster happens to be in the list — so the three
        maps came out at three different scales and, on a design whose DEM did
        not resolve, the design map zoomed to the drawn features and printed
        them at whatever size their spread happened to produce. Framing them
        all on the boundary makes them comparable and keeps the site in the
        same place on the page from one figure to the next.
        """
        from terrainflow_assessment.qgis.adapters.map_image import (
            layers_extent,
            usable_layers,
        )

        boundary = usable_layers([self._boundary_layer()])
        return layers_extent(boundary, crs) if boundary else None

    def _boundary_layer(self):
        """The drawn site boundary, or the analysis area standing in for it."""
        for fragment in ("Drawn Site Boundary", "Drawn Analysis Area",
                         "Drawn Earthworks Area"):
            layer = self._named_layer(fragment)
            if layer is not None:
                return layer
        return None

    def _terrain_layers(self):
        """What every map is drawn over, bottom-last.

        The hillshade is what stops the design map printing on blank white; the
        DEM behind it is the fallback for a project that predates the hillshade
        or whose source raster the user has styled themselves.
        """
        return [self._hillshade_layer(), self._dem_layer()]

    def _layers_for(self, key):
        """Layers for a report map, top-first — the reverse of the layer tree.

        The stacking is the whole of the map's legibility, so it is written out
        rather than assembled: **markers and labels, then the design, then the
        boundary, then the water rasters, then terrain.** The boundary has to
        sit above the rasters — the surface-runoff ramp reaches full opacity at
        its top stop and would bury a line drawn under it — and the thresholded
        stream network has to sit above the diffuse runoff wash it is the
        distilled version of.
        """
        state = self._state
        design = [resolve_layer(self._project, i) for i in
                  ([state.spillway_layer_id, state.connections_layer_id]
                   + list(state.ew_layer_ids.values())) if i]
        catchments = resolve_layer(self._project,
                                   state.catchment_labels_layer_id)
        if key == "design":
            top = design
            # catchment_labels is a Design-tier layer behind a panel checkbox
            # that defaults off, so it can only ever be an optional overlay.
            water = [self._named_layer("Streams"), catchments]
        elif key == "flow":
            top = [self._named_layer("Exit Points")]
            water = [self._named_layer("Streams"),
                     resolve_layer(self._project, state.throughflow_layer_id),
                     catchments]
        else:
            return []
        resolved = (top + [self._boundary_layer()] + water
                    + self._terrain_layers())
        return [layer for layer in resolved if layer is not None]

    def _ponding_pair(self):
        """The two capacity rasters, before and after the earthworks.

        The full-capacity layer specifically. The event layer beside it is one storm's
        answer on the design side with no "before" to pair it against, and a map
        captioned before/after has to compare like with like.
        """
        return (self._named_layer("Pond Capacity (full)", group="baseline"),
                self._named_layer("Pond Capacity (full)", group="earthworks"))

    def _ponding_tags_agree(self):
        """A Baseline re-run rebuilds only the 'before' layer.

        Pairing a fresh before against a stale after would show a difference
        that never existed on any one design.
        """
        before = self._named_layer("Pond Capacity (full)", group="baseline")
        after = self._named_layer("Pond Capacity (full)", group="earthworks")
        if not (before and after):
            return False
        return _group_tag(self._project, before) == _group_tag(self._project, after)

    _STAGE_PREFIXES = ("Baseline — ", "Earthworks — ")

    def _named_layer(self, fragment, group=None):
        """A layer by name fragment, optionally restricted to one stage.

        An ungrouped lookup prefers an unprefixed layer and *falls back* to a
        stage-prefixed one, rather than rejecting it. It used to reject: every
        stream raster the plugin makes is called "Baseline — Streams" or
        "Earthworks — Streams", so ``_named_layer("Streams")`` could only ever
        return None and the stream network was silently missing from every
        report map. Baseline wins the tie — it is the run the flow map is about.
        """
        prefix = {"baseline": self._STAGE_PREFIXES[0],
                  "earthworks": self._STAGE_PREFIXES[1]}.get(group)
        fallback = None
        for layer in self._project.instance().mapLayers().values():
            name = layer.name()
            if fragment not in name:
                continue
            if prefix:
                if name.startswith(prefix):
                    return layer
                continue
            if not name.startswith(self._STAGE_PREFIXES):
                return layer
            if fallback is None or name.startswith(self._STAGE_PREFIXES[0]):
                fallback = layer
        return fallback

    def _hillshade_layer(self):
        """The shaded-relief backdrop for the report's maps.

        There are two once an earthworks run has happened — one over the source DEM and
        one over the burned one — and ``_named_layer`` breaks the tie toward Baseline,
        which is the right way round: every map in this report draws the design *on top*
        of the terrain, so shading the burned surface underneath would print each swale
        twice, once as a line and once as the trench it cut.
        """
        return self._named_layer("Hillshade", group="baseline") or \
            self._named_layer("Hillshade")

    def _dem_layer(self):
        """Whatever raster can stand as the terrain behind the maps.

        Widened deliberately. This used to accept only a layer named
        "…Burned DEM" or one whose source string matched ``state.dem_path``
        exactly — and on a project whose DEM was loaded from a different path,
        or renamed, or clipped into ``output_dir``, it matched nothing. The map
        then had no backdrop at all and printed the design on blank white,
        which is where item 20 of the change list came from.
        """
        layers = list(self._project.instance().mapLayers().values())
        for layer in layers:
            if layer.name().endswith("Burned DEM"):
                return layer
        dem_path = self._state.dem_path or ""
        if dem_path:
            for layer in layers:
                if layer.source() == dem_path:
                    return layer
            base = os.path.basename(dem_path)
            for layer in layers:
                if base and os.path.basename(layer.source() or "") == base:
                    return layer
        # Last resort: any raster that is not one of our own overlays. Better a
        # plain DEM behind the design than nothing behind it.
        for layer in layers:
            if self._is_plain_raster(layer):
                return layer
        return None

    #: Overlay layers that are rasters but are emphatically not terrain.
    _OVERLAY_FRAGMENTS = ("Streams", "Throughflow", "Surface Runoff",
                          "Pond Capacity", "Ponding", "Slope", "Catchment",
                          "Inflow", "Hillshade", "Simulation")

    def _is_plain_raster(self, layer):
        try:
            from qgis.core import QgsMapLayer

            if layer.type() != QgsMapLayer.RasterLayer or not layer.isValid():
                return False
        except Exception:
            return False
        name = layer.name()
        return not any(f in name for f in self._OVERLAY_FRAGMENTS)

    # ------------------------------------------------------------------ misc

    def _open(self, path):
        """Open the finished file, unless a headless check is driving us.

        On Windows an open viewer holds a lock on the PDF, so re-exporting to
        the same name fails; the checks would also pop a viewer per run.
        """
        if os.environ.get("TERRAINFLOW_NO_OPEN"):
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not open report: {exc}")


_HTML_SUFFIXES = (".html", ".htm")


def resolve_target(path, selected_filter):
    """Settle the format and make the file name match it.

    Returns ``(path, as_html)``.

    The save dialog offers the format twice over — a filter dropdown and a file
    name — and they disagree easily: the default name ends ``.pdf``, so picking
    "HTML (*.html)" without retyping the name would otherwise write HTML into a
    file called ``.pdf``, which opens in a PDF reader and fails. Typing an
    explicit ``.html`` wins over the dropdown; otherwise the dropdown decides;
    either way the suffix is rewritten so the name never lies about the content.
    """
    lowered = str(path).lower()
    as_html = (lowered.endswith(_HTML_SUFFIXES)
               or str(selected_filter or "").strip().upper().startswith("HTML"))
    # ".htm" is as valid as ".html"; renaming a suffix the user typed on purpose
    # would be its own small surprise.
    accepted = _HTML_SUFFIXES if as_html else (".pdf",)
    if not lowered.endswith(accepted):
        # Replace a known report suffix, otherwise append — a name with a dot in
        # it ("Smith 1.2 Block") must not lose its tail.
        base, ext = os.path.splitext(path)
        keep = base if ext.lower() in _HTML_SUFFIXES + (".pdf",) else path
        path = keep + accepted[0]
    return path, as_html


def _safe(name):
    return _UNSAFE.sub("-", str(name)).strip() or "Site"


def _plugin_version():
    """The version from metadata.txt.

    Nothing else in the plugin reads it — ``project_io.SCHEMA_VERSION`` is the
    design-file schema, not the plugin — but a report that cannot say which
    version produced it is not much of a record.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "metadata.txt")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def _rmtree(path):
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _digest(dem_path):
    try:
        from terrainflow_assessment.modules.dem_loader import dem_content_digest
        return dem_content_digest(dem_path)
    except Exception:
        return ""


def _group_tag(project, layer):
    """The run tag stamped on the stage group a layer sits in."""
    try:
        node = project.layer_tree_root().findLayer(layer.id())
        parent = node.parent() if node else None
        return parent.name() if parent else ""
    except Exception:
        return ""
