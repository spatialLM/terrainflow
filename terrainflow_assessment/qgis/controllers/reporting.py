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

#: How far the overview map is pulled back beyond the site boundary, as a
#: fraction of the boundary's own width and height on each side.
_OVERVIEW_MARGIN = 0.25

#: How far the design map is pulled back beyond the earthworks themselves. Small
#: — the map is meant to be as close in as it can get and still show every
#: feature — but not nothing: each feature carries a name label placed outside
#: its geometry, and framed on the geometry alone the outermost labels are cut
#: in half by the neat line.
_DESIGN_MARGIN = 0.08

#: Providers that mean "a tile basemap". XYZ, WMS and WMTS layers all arrive
#: through the ``wms`` provider, and nothing this plugin creates is anything but
#: a local file, so a raster answering to this is the operator's own imagery.
_BASEMAP_PROVIDERS = ("wms",)

#: Outline colour for a catchment whose earthwork the registry cannot colour —
#: a feature deleted between the re-analysis and the export. Grey, matching
#: ``report_charts.type_colour``'s own fallback.
CATCHMENT_OUTLINE_FALLBACK = "#7f8c8d"


def _padded(extent, fraction):
    """*extent* padded on every side by *fraction* of its **larger** dimension.

    Not by each side's own size, which is what :func:`_grown` does. A design can
    be perfectly flat in one axis — one straight swale, or a row of them on the
    same contour — and its extent is then a line with zero height. Scaling that
    proportionally pads it by zero and leaves a rectangle a map item cannot
    frame, so the design map silently fell back to the whole block: the one
    thing the zoom exists to stop.
    """
    if extent is None:
        return None
    try:
        from qgis.core import QgsRectangle

        span = max(extent.width(), extent.height())
        if span <= 0:
            return None            # a single point has no scale to pad by
        pad = span * fraction
        return QgsRectangle(extent.xMinimum() - pad, extent.yMinimum() - pad,
                            extent.xMaximum() + pad, extent.yMaximum() + pad)
    except Exception:
        return None


def _grown(extent, fraction):
    """A rectangle expanded by *fraction* of its own size on every side.

    Returns None for a None or empty extent rather than a degenerate rectangle:
    a map item handed a zero-width extent renders at whatever scale QGIS falls
    back to, which is not a failure anyone would notice on the page.
    """
    if extent is None:
        return None
    try:
        from qgis.core import QgsRectangle

        width, height = extent.width(), extent.height()
        if width <= 0 or height <= 0:
            return None
        return QgsRectangle(extent.xMinimum() - width * fraction,
                            extent.yMinimum() - height * fraction,
                            extent.xMaximum() + width * fraction,
                            extent.yMaximum() + height * fraction)
    except Exception:
        return None


class ReportingController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas
        # Layers built for one document and registered outside the layer tree.
        # See :meth:`_open_transients`.
        self._transients = {}

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

        held_selections = self._clear_interactive_state()
        work = os.path.join(self._state.output_dir, "report_assets")
        os.makedirs(work, exist_ok=True)
        try:
            # Before the data is collected, not after: whether the catchment
            # labelling traced to anything is a fact about this document, and
            # the model builds the summary map's key out of it.
            self._open_transients(work)
            data = self._collect(site)
            report = build_report(data)
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
            # Before the working directory goes: a raster layer holds its file
            # open, and on Windows a tree with an open handle in it does not
            # delete.
            self._close_transients()
            # Consumed synchronously, so nothing outlives the export. Both
            # outputs embed what they need and survive output_dir being cleared.
            _rmtree(work)
            self._restore_selections(held_selections)

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

        Every vector layer in the project is cleared, including the operator's own
        cadastre and asset layers — the live ``QgsLayoutItemMap`` draws whatever it
        is given and a selection is not a plugin concept. So the ids are returned
        and :meth:`_restore_selections` puts them back: exporting a document is not
        a reason to destroy a selection the operator spent time building.
        """
        from qgis.core import QgsMapLayer

        held = {}
        for layer in self._project.instance().mapLayers().values():
            try:
                if (layer.type() == QgsMapLayer.VectorLayer
                        and layer.selectedFeatureCount()):
                    held[layer.id()] = list(layer.selectedFeatureIds())
                    layer.removeSelection()
            except Exception:
                continue
        return held

    def _restore_selections(self, held):
        """Put back what :meth:`_clear_interactive_state` took, where it still exists.

        A layer removed during the export simply drops out; ids that no longer match
        a feature are ignored by ``selectByIds``.
        """
        for layer_id, fids in (held or {}).items():
            layer = resolve_layer(self._project, layer_id)
            if layer is None:
                continue
            try:
                layer.selectByIds(fids)
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
            graph = build_flow_graph(
                data.balance, getattr(data, "display_names", None))
            if render_flow_network(graph, path=png):
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
            catchment_outline="catchment_outline" in (self._transients or {}),
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
        if not self._layers_for("overview"):
            reasons["overview"] = (
                "There is nothing on the map to draw the site from yet — load a "
                "DEM and run Baseline.")
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
            usable_layers,
        )

        crs = self._map_crs()
        frame = self._frame_extent(crs)
        specs = {}
        # The overview is the one map deliberately *not* on the common frame: it
        # exists to place the block in its surroundings, and a frame drawn tight
        # to the boundary shows the property with nothing around it, which is the
        # one thing an orientation figure must not do.
        overview = usable_layers(self._layers_for("overview"))
        if overview:
            extent = _grown(frame if frame is not None
                            else self._content_extent(overview, crs),
                            _OVERVIEW_MARGIN)
            if extent is not None:
                specs["overview"] = MapSpec(overview, extent, crs,
                                            height_mm=95.0)
        # The design map is the one that is *not* framed on the boundary. It is
        # read to build from, so it wants to be as close in as it can get and
        # still show every feature; on a block whose earthworks sit in one
        # corner, the common frame spent most of the page on ground the design
        # never touches.
        design = usable_layers(self._layers_for("design"))
        if design:
            extent = (_padded(self._earthworks_extent(crs), _DESIGN_MARGIN)
                      or frame or self._content_extent(design, crs))
            if extent is not None:
                specs["design"] = MapSpec(design, extent, crs)
        layers = usable_layers(self._layers_for("flow"))
        if layers:
            specs["flow"] = MapSpec(
                layers, frame or self._content_extent(layers, crs), crs)
        before, after = self._ponding_pair()
        if before and after and self._ponding_tags_agree():
            layers = usable_layers(
                [self._boundary_layer(), after, before]
                + self._terrain_layers())
            if layers:
                specs["ponding"] = MapSpec(
                    layers, frame or self._content_extent(layers, crs), crs,
                    height_mm=110.0)
        return specs

    def _content_extent(self, layers, crs):
        """The extent of *layers*, ignoring anything that covers the world.

        A tile basemap's extent is the whole globe. Left in the union it does not
        widen the fallback frame so much as replace it, and a map that had merely
        lost its boundary would come out showing the Pacific.
        """
        from terrainflow_assessment.qgis.adapters.map_image import layers_extent

        basemap = self._basemap_layer()
        return layers_extent([layer for layer in layers if layer is not basemap],
                             crs)

    def _earthworks_extent(self, crs):
        """The bounding extent of every drawn earthwork, or None if there are none.

        Deliberately not ``map_image.layers_extent``, which drops any rectangle
        ``isEmpty()`` calls empty — and ``QgsRectangle.isEmpty()`` is true of a
        rectangle with **zero height**, not just of one with no area. A design
        of one straight swale, or of several sitting on the same contour, has
        exactly that extent, so the whole design was discarded and the map fell
        back to the block. A flat design is a real design; it is
        :func:`_padded`'s job to give it a rectangle, and it can only do that if
        it is handed one.
        """
        from qgis.core import QgsCoordinateTransform, QgsRectangle

        from terrainflow_assessment.qgis.adapters.map_image import usable_layers

        combined = None
        for layer in usable_layers([resolve_layer(self._project, i)
                                    for i in
                                    self._state.ew_layer_ids.values() if i]):
            try:
                rect = layer.extent()
                # Null is an empty layer; inverted is QGIS's "minimal" seed. A
                # zero-width or zero-height rectangle is neither — it is a line.
                if rect.isNull() or rect.width() < 0 or rect.height() < 0:
                    continue
                if crs is not None and layer.crs() != crs:
                    rect = QgsCoordinateTransform(
                        layer.crs(), crs,
                        self._project.instance()).transformBoundingBox(rect)
                if combined is None:
                    combined = QgsRectangle(rect)
                else:
                    combined.combineExtentWith(rect)
            except Exception as exc:
                print(f"TerrainFlow Assessment — earthwork extent: {exc}")
        return combined

    def _map_crs(self):
        """The CRS the report's maps are drawn in — the DEM's, not the project's.

        Every metre in this document is measured on the DEM's grid, and both
        renderers stamp a scale bar in metres off the rendered extent. Render in a
        geographic project CRS and that extent is in degrees, so "metres per pixel"
        is out by about five orders of magnitude and the bar comes out plausible
        and wrong. The project CRS is the operator's display choice; it is not a
        property of anything this document measures.
        """
        from qgis.core import QgsCoordinateReferenceSystem

        info = self._state.dem_info
        wkt = getattr(info, "crs_wkt", None) if info is not None else None
        if wkt:
            crs = QgsCoordinateReferenceSystem(wkt)
            if crs.isValid():
                return crs
        return self._project.crs()

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

    #: What the operator chose, most specific first. The boundary is the site;
    #: the other two are what stands in for it when nobody set one. Each is a
    #: ``_state`` id and the matching panel property, which answer the same
    #: question from the two ends of the same signal.
    _AREA_SOURCES = (("boundary_layer_id", "boundary_layer"),
                     ("analysis_area_layer_id", "analysis_area_layer"),
                     ("earthworks_area_layer_id", "earthworks_area_layer"))

    def _boundary_layer(self):
        """The site boundary, or the area standing in for it.

        What the operator actually chose — recorded as a layer id when the
        picker fired, and read back off the panel if it was not — with the layer
        names only as a last fallback. This used to search the project for a layer called "Drawn Site Boundary"
        and nothing else — which is the name the *draw-on-canvas* tool gives its
        output, and nothing else has it. An operator who instead **picked** an
        existing polygon in the boundary combo, which is the ordinary way to use
        a cadastral parcel or a title boundary, had no layer of that name
        anywhere in the project. So ``_frame_extent`` came back None and every
        figure in the document fell back to the extent of whichever raster
        happened to be in its layer list — usually the whole DEM tile.

        Nothing said so, and nothing could: a map drawn to the wrong extent
        renders exactly as well as one drawn to the right one. It is also why
        the report's own test for this rule passed throughout — the harness
        picks its boundary rather than drawing it, so the rule was never
        actually exercised.

        Drawing still works, and reaches the same place: the draw tool calls
        ``panel.set_area_layer``, which selects the new layer in the combo.
        """
        for state_attr, panel_attr in self._AREA_SOURCES:
            layer = resolve_layer(self._project,
                                  getattr(self._state, state_attr, None))
            if layer is None:
                try:
                    layer = getattr(self._panel, panel_attr, None)
                except (AttributeError, RuntimeError):
                    layer = None
            try:
                if layer is not None and layer.isValid():
                    return layer
            except RuntimeError:
                continue
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

    def _basemap_layer(self):
        """The operator's own aerial photograph, where they have one loaded.

        The report has never had a basemap — the hillshade exists precisely
        because it did not — and it still does not fetch one: a LINZ aerial
        needs an API key, and the operator who wants the photograph behind their
        scheme already has it on the canvas. So this finds it rather than making
        it. Every raster this plugin creates is a local file read through
        ``gdal``; a raster on the ``wms`` provider is a tile service and is
        therefore not ours.

        Bottom-most first, and visible only. A project can hold several tile
        layers — an aerial, a topo, a cadastral overlay — and the one at the
        bottom of the tree with its box ticked is the one the operator is using
        as their ground, which is the job being filled here.
        """
        from qgis.core import QgsMapLayer

        found = None
        try:
            nodes = self._project.instance().layerTreeRoot().findLayers()
        except Exception:
            return None
        for node in nodes:
            layer = node.layer()
            if layer is None or not node.isVisible():
                continue
            try:
                if layer.type() != QgsMapLayer.RasterLayer:
                    continue
                if layer.dataProvider().name() not in _BASEMAP_PROVIDERS:
                    continue
            except Exception:
                continue
            found = layer
        return found

    def _backdrop(self):
        """What a map is drawn over: the operator's aerial, or the terrain.

        Never nothing. A summary map composed of a boundary and two ponds over
        blank white is the failure the hillshade was introduced to prevent, and
        a project with no tile layer loaded must not fall into it just because
        this report now prefers a photograph.
        """
        basemap = self._basemap_layer()
        return [basemap] if basemap is not None else self._terrain_layers()

    def _layers_for(self, key):
        """Layers for a report map, top-first — the reverse of the layer tree.

        The stacking is the whole of the map's legibility, so it is written out
        rather than assembled: **markers and labels, then the design, then the
        boundary, then the water rasters, then the backdrop.** The boundary has
        to sit above the rasters — the surface-runoff ramp reaches full opacity
        at its top stop and would bury a line drawn under it — and the
        thresholded stream network has to sit above the diffuse runoff wash it
        is the distilled version of.

        Each of the three answers one question, and carries only what that
        question needs. They used to differ by a layer or two around a common
        core, which is how the summary page came to be a shaded terrain model
        with forty feature labels on it — a figure a reader has to decode before
        it has told them where they are.
        """
        state = self._state
        if key == "overview":
            # Where the block is, and what the scheme does to the water on it.
            # No terrain model and no feature labels: those are the design map's
            # job, and repeating them here only buries the photograph that makes
            # this figure worth having. The catchment outline is the one
            # analysis product that belongs — it is the answer to "how much of
            # my land does this actually catch", drawn as a line so the ground
            # underneath it stays visible.
            resolved = [
                self._transients.get("catchment_outline"),
                self._boundary_layer(),
                self._named_layer("Pond Capacity (event)", group="earthworks"),
                self._stage_layer("Pond Capacity (full)"),
                self._stage_layer("Streams"),
            ] + self._backdrop()
        elif key == "design":
            # Every feature as drawn, over the ground it is to be dug in, as
            # close in as the page allows. Nothing analytical: this is the sheet
            # somebody stands in a paddock holding.
            resolved = ([resolve_layer(self._project, i)
                         for i in state.ew_layer_ids.values() if i]
                        + [self._boundary_layer()] + self._backdrop())
        elif key == "flow":
            # The baseline, before anything was dug. The runoff wash is the
            # clipped copy where there is one: unclipped it covers the whole DEM
            # tile, and on a coastal tile the block ends up sitting in a fan of
            # blue streaks running off every edge of the page, none of which is
            # ground the owner can do anything about.
            runoff = (self._transients.get("runoff_clipped")
                      or resolve_layer(self._project, state.throughflow_layer_id))
            resolved = [
                self._named_layer("Exit Points", group="baseline"),
                self._boundary_layer(),
                self._named_layer("Streams", group="baseline"),
                runoff,
                self._named_layer("Pond Capacity (full)", group="baseline"),
            ] + self._terrain_layers()
        else:
            return []
        return [layer for layer in resolved if layer is not None]

    def _stage_layer(self, fragment):
        """An earthworks-stage layer, falling back to the baseline's.

        The summary map is about the scheme, so it wants the re-analysed answer.
        A design that has been drawn but not re-analysed has no such layer, and
        showing the baseline's streams is closer to the truth than showing the
        reader an empty photograph.
        """
        return (self._named_layer(fragment, group="earthworks")
                or self._named_layer(fragment, group="baseline"))

    # ------------------------------------------------------ document layers

    def _open_transients(self, work):
        """Build the layers that exist only for this document.

        Both are derived products the canvas has no use for: a clipped copy of
        the runoff raster, and the catchment labelling traced as a line. They go
        into the project because the PDF renderer's ``QgsLayoutItemMap`` resolves
        its layers by id at render time and cannot see one that is not there —
        and they stay out of the layer tree because a legend that grows two
        layers every time anyone presses Export is not a legend.
        """
        from terrainflow_assessment.qgis.controllers._groups import (
            register_render_only,
        )

        self._transients = {}
        for key, build in (("catchment_outline", self._catchment_outline_layer),
                           ("runoff_clipped",
                            lambda: self._clipped_runoff_layer(work))):
            try:
                layer = build()
            except Exception as exc:
                print(f"TerrainFlow Assessment — report layer '{key}': {exc}")
                continue
            if layer is not None:
                self._transients[key] = register_render_only(
                    self._project, layer)

    def _close_transients(self):
        """Take them out again. Paired with :meth:`_open_transients`."""
        from terrainflow_assessment.qgis.controllers._groups import discard

        for layer in (self._transients or {}).values():
            discard(self._project, layer)
        self._transients = {}

    def _boundary_geometries(self, crs=None):
        """The drawn boundary as shapely polygons, in *crs* if given."""
        from qgis.core import QgsCoordinateTransform, QgsGeometry

        from terrainflow_assessment.qgis.adapters.geom import qgs_to_shapely

        layer = self._boundary_layer()
        if layer is None:
            return []
        transform = None
        if crs is not None and crs.isValid() and layer.crs() != crs:
            transform = QgsCoordinateTransform(layer.crs(), crs,
                                               self._project.instance())
        out = []
        for feature in layer.getFeatures():
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                continue
            if transform is not None:
                geometry = QgsGeometry(geometry)
                if geometry.transform(transform) != 0:
                    continue
            shape = qgs_to_shapely(geometry)
            if shape is not None and not shape.is_empty:
                out.append(shape)
        return out

    def _catchment_outline_layer(self):
        """The ground each earthwork catches, traced as a line.

        The filled version of this already exists — the paletted "Catchment by
        earthwork" raster — and it is drawn at alpha 150 over everything beneath
        it, which on an aerial photograph is most of the point of having the
        photograph. An outline answers the same question and leaves the ground
        visible.

        Built here rather than kept on the canvas because the canvas one is only
        ever refreshed by a panel checkbox: a re-analysis rebuilds the labelling
        underneath it and leaves the layer showing the run before last. The
        labelling itself is recomputed on every geometry edit, so reading it
        directly is the version that cannot be stale.
        """
        from qgis.core import (
            QgsCategorizedSymbolRenderer,
            QgsFeature,
            QgsField,
            QgsGeometry,
            QgsLineSymbol,
            QgsPointXY,
            QgsRendererCategory,
            QgsVectorLayer,
        )
        from qgis.PyQt.QtCore import QMetaType

        from terrainflow_assessment.core.registry.earthwork_types import get_type
        from terrainflow_assessment.modules.catchment import label_outlines
        from terrainflow_assessment.qgis.controllers._layers import (
            MissingDemCrs,
            dem_crs,
        )

        state = self._state
        labels = getattr(state, "catchment_labels", None)
        meta = getattr(state, "flow_grid_meta", None)
        if labels is None or not meta or meta.get("transform") is None:
            return None
        try:
            # These are DEM grid coordinates in metres, so the project's CRS is
            # not a fallback — drawn as degrees they land off the coast of
            # Ghana. No DEM, no outline, and nothing to report about it: a map
            # that has lost its DEM has larger problems, and says so elsewhere.
            crs = dem_crs(state)
        except MissingDemCrs:
            return None
        rings = label_outlines(labels, meta["transform"],
                               list(state.catchment_label_ids or []))
        if not rings:
            return None

        manager = getattr(state, "earthwork_manager", None)
        by_id = {ew.id: ew for ew in (manager.get_all() if manager else [])}
        layer = QgsVectorLayer(f"LineString?crs={crs}",
                               "Catchment outline", "memory")
        if not layer.isValid():
            return None
        layer.dataProvider().addAttributes(
            [QgsField("ew_type", QMetaType.QString)])
        layer.updateFields()

        feats = []
        types = []
        for ew_id, ring_list in rings.items():
            ew = by_id.get(ew_id)
            ew_type = getattr(ew, "type", "") or ""
            if ew_type and ew_type not in types:
                types.append(ew_type)
            for ring in ring_list:
                points = [QgsPointXY(x, y) for x, y in ring]
                feature = QgsFeature(layer.fields())
                feature.setGeometry(QgsGeometry.fromPolylineXY(points))
                feature.setAttribute("ew_type", ew_type)
                feats.append(feature)
        if not feats:
            return None
        layer.dataProvider().addFeatures(feats)
        layer.updateExtents()

        # Categorised on the type, so a catchment is drawn in the colour of the
        # thing that catches it and the map needs no second key to be read.
        categories = []
        for ew_type in types:
            try:
                colour = get_type(ew_type).style[1]
            except Exception:
                colour = CATCHMENT_OUTLINE_FALLBACK
            categories.append(QgsRendererCategory(
                ew_type,
                QgsLineSymbol.createSimple({"color": colour, "width": "0.4"}),
                ew_type))
        categories.append(QgsRendererCategory(
            "", QgsLineSymbol.createSimple(
                {"color": CATCHMENT_OUTLINE_FALLBACK, "width": "0.4"}), ""))
        layer.setRenderer(QgsCategorizedSymbolRenderer("ew_type", categories))
        return layer

    def _clipped_runoff_layer(self, work):
        """The Surface Runoff raster, masked to the drawn boundary.

        A copy, styled with the source layer's own renderer rather than a
        rebuilt one: the ramp top is shared across a family through
        ``_symbols.apply_shared_ramp``, and a second derivation of it here would
        be a second place for the pair to drift apart.
        """
        from qgis.core import QgsRasterLayer

        from terrainflow_assessment.modules.footprint import (
            clip_raster_to_polygons,
        )

        source = resolve_layer(self._project, self._state.throughflow_layer_id)
        if source is None or not source.isValid():
            return None
        geometries = self._boundary_geometries(source.crs())
        if not geometries:
            return None
        clipped = clip_raster_to_polygons(
            source.source(), geometries,
            os.path.join(work, "runoff_clipped.tif"))
        if clipped is None:
            return None
        layer = QgsRasterLayer(clipped, source.name())
        if not layer.isValid():
            return None
        try:
            layer.setRenderer(source.renderer().clone())
        except Exception:
            pass
        return layer

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

    def _analysis_layers(self):
        """The analysis outputs, resolved from the ids the plugin recorded.

        "Streams", "Exit Points" and "Pond Capacity (full)" are ordinary words.
        An operator whose project holds a layer of that name would have it drawn
        on the report's maps in place of the analysis, and nothing would say so
        — a map with the wrong stream network renders exactly as well as one
        with the right one. Ids are the plugin's own record of what it made, so
        a search inside them cannot reach anything it did not.
        """
        ids = (list(self._state.baseline_layer_ids or [])
               + list(self._state.earthworks_layer_ids or []))
        found = (resolve_layer(self._project, i) for i in ids)
        return [layer for layer in found if layer is not None]

    def _named_layer(self, fragment, group=None):
        """A layer by name fragment, optionally restricted to one stage.

        An ungrouped lookup prefers an unprefixed layer and *falls back* to a
        stage-prefixed one, rather than rejecting it. It used to reject: every
        stream raster the plugin makes is called "Baseline — Streams" or
        "Earthworks — Streams", so ``_named_layer("Streams")`` could only ever
        return None and the stream network was silently missing from every
        report map. Baseline wins the tie — it is the run the flow map is about.

        The plugin's own analysis layers are searched first and the rest of the
        project only if that finds nothing. Preferring rather than restricting,
        because not everything these maps need is an analysis output: the site
        boundary and the analysis area are drawn by the operator, and the
        hillshade may predate any run. Those are found by name, as they have to
        be — but only once the layers the plugin can identify have had their say.
        """
        prefix = {"baseline": self._STAGE_PREFIXES[0],
                  "earthworks": self._STAGE_PREFIXES[1]}.get(group)

        def _search(layers):
            fallback = None
            for layer in layers:
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

        return (_search(self._analysis_layers())
                or _search(self._project.instance().mapLayers().values()))

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
