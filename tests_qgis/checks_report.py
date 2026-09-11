"""
Report export — the PDF path, driven through real panel signals.

`pytest tests/` covers what the report *says* (`modules/report_model.py` is pure
and inside the coverage gate). What it cannot cover is everything here: the
QGIS print layout, off-screen map rendering, layer resolution, and the wiring
that decides when the export button is live.

The headline check is `check_report_pdf_without_simulation`: a baseline, one
earthwork, no simulation anywhere, and a real PDF on disk. That gate is the
whole reason this work exists, so it is the first thing asserted.

`TERRAINFLOW_NO_OPEN` stops the controller launching a PDF viewer per check —
which would both flood the desktop and, on Windows, lock the file so the next
export to the same name fails.
"""

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from _harness import PluginHarness, line_across_valley
from _shots import save_qimage


@contextmanager
def workdir():
    """A scratch directory per check, removed afterwards."""
    path = Path(tempfile.mkdtemp(prefix="tfa_report_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

os.environ.setdefault("TERRAINFLOW_NO_OPEN", "1")

PDF_MAGIC = b"%PDF-"
# A PDF whose maps or tables silently failed to draw is dramatically smaller
# than one where they worked, so this is a real signal rather than a formality.
MIN_PDF_BYTES = 20_000


class ScriptedSaveDialog:
    """Answer the controller's save dialog without a user."""

    def __init__(self, path, selected="PDF (*.pdf)"):
        self.path = path
        self.selected = selected

    def __enter__(self):
        from terrainflow_assessment.qgis.controllers import reporting

        self._module = reporting
        self._prev = reporting.QFileDialog
        scripted = self

        class _FileDialog:
            @staticmethod
            def getSaveFileName(*_a, **_k):
                return (scripted.path, scripted.selected)

        reporting.QFileDialog = _FileDialog
        return self

    def __exit__(self, *_exc):
        self._module.QFileDialog = self._prev
        return False


def _export(h, path, selected="PDF (*.pdf)"):
    with ScriptedSaveDialog(path, selected):
        h.panel.export_report_requested.emit()
    return path


def _assert_pdf(path, context):
    assert os.path.exists(path), f"{context}: no file written to {path}"
    with open(path, "rb") as handle:
        assert handle.read(5) == PDF_MAGIC, f"{context}: not a PDF"
    size = os.path.getsize(path)
    assert size >= MIN_PDF_BYTES, (
        f"{context}: PDF is only {size:,} bytes — content probably failed to draw")
    return size


# ---------------------------------------------------------------- the gate

def check_report_pdf_without_simulation(dem_path):
    """The whole point: a baseline and a design produce a PDF. No simulation."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        assert h.state.comparison is None, "this check must not run a simulation"
        assert h.state.balance is not None, (
            "the design-tier balance must be retained on the state — it used to "
            "fall out of scope, which is why a report needed a simulation")

        out = _export(h, str(tmp / "no_sim.pdf"))
        h.assert_no_errors("report export without simulation")
        _assert_pdf(out, "no-simulation export")


def check_report_button_enabled_by_baseline_alone(dem_path):
    """Baseline is the only precondition — no design, no simulation."""
    with PluginHarness(dem_path) as h:
        assert not h.panel._export_btn.isEnabled(), (
            "the export must start disabled")
        h.run_baseline()
        assert h.panel._export_btn.isEnabled(), (
            "a baseline alone is a reportable site assessment")


def check_report_button_survives_disabling_every_earthwork(dem_path):
    """Regression: wiring the button to the balance switched it off here.

    run_water_balance returns None when no *enabled* feature exists, so keying
    the button off that result took the export away with a valid baseline behind
    it.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()
        assert h.panel._export_btn.isEnabled()

        ew.enabled = False
        h.panel.analysis_inputs_changed.emit()
        assert h.panel._export_btn.isEnabled(), (
            "disabling every earthwork must not disable the report")


def check_report_baseline_only_export(dem_path):
    """No earthworks at all still produces a document that says so."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        out = _export(h, str(tmp / "baseline_only.pdf"))
        h.assert_no_errors("baseline-only report")
        _assert_pdf(out, "baseline-only export")


def check_report_blocked_before_baseline(dem_path):
    """Without a baseline there is nothing to report, and it says so."""
    with PluginHarness(dem_path) as h:
        h.panel.export_report_requested.emit()
        warnings = h.bar.warnings
        assert any("Baseline" in text for _, _, text in warnings), (
            f"expected a warning naming Baseline, got {warnings}")


def check_report_button_disabled_after_design_open(dem_path):
    """Regression: _clear_derived_results left the button lit and dead.

    Opening a design cleared every derived result but never touched the button,
    so it stayed enabled and the click dead-ended on a warning.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        assert h.panel._export_btn.isEnabled()

        h.plugin._design_file._clear_derived_results()
        assert not h.panel._export_btn.isEnabled(), (
            "clearing derived results must disable the export")
        assert h.state.balance is None
        assert h.state.spillway_rows is None


# ---------------------------------------------------------------- content

def check_report_names_the_site(dem_path):
    """The shipped sample was headed 'Unnamed Site'. It should not have been."""
    from terrainflow_assessment.modules.report_model import build_report

    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.panel._site_name_edit.setText("Quail Island")
        h.run_baseline()
        data = h.plugin._reporting._collect(h.plugin._reporting._site_name())
        report = build_report(data)
        assert report.title.startswith("Quail Island")
        assert "Unnamed Site" not in report.title

        out = _export(h, str(tmp / "named.pdf"))
        _assert_pdf(out, "named-site export")


def check_report_falls_back_when_site_unnamed(dem_path):
    """A blank field must not put 'Unnamed Site' on a document being handed on."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.project.setTitle("Hillside Block")
        assert h.plugin._reporting._site_name() == "Hillside Block"


def check_report_survives_a_second_export_same_day(dem_path):
    """The default filename is date-stamped, so re-export is the normal case."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        target = str(tmp / "twice.pdf")
        _export(h, target)
        first = _assert_pdf(target, "first export")
        _export(h, target)
        h.assert_no_errors("second export to the same path")
        assert _assert_pdf(target, "second export") > 0 and first > 0


def check_report_survives_missing_map_layers(dem_path):
    """Layers whose source vanished resolve live-but-broken and render blank."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        for layer_id in list(h.project.mapLayers()):
            h.project.removeMapLayer(layer_id)

        out = _export(h, str(tmp / "no_layers.pdf"))
        h.assert_no_errors("report with every layer removed")
        assert os.path.exists(out), "a map with no layers must not stop the export"


def check_report_pdf_is_self_contained(dem_path):
    """output_dir is rmtree'd on unload; the PDF must not depend on it."""
    with workdir() as tmp:
        out = str(tmp / "standalone.pdf")
        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.add_earthwork("swale", line_across_valley())
            h.panel.analysis_inputs_changed.emit()
            _export(h, out)
            before = _assert_pdf(out, "self-contained export")
            shutil.rmtree(h.state.output_dir, ignore_errors=True)

        assert os.path.getsize(out) == before, (
            "the PDF changed when the scratch directory went away")
        with open(out, "rb") as handle:
            assert handle.read(5) == PDF_MAGIC


def check_report_format_matches_the_file_name(dem_path):
    """The dialog asks for the format twice; the two must not disagree.

    Picking "HTML (*.html)" while the pre-filled name still ends .pdf would
    otherwise write HTML into a .pdf, which opens in a PDF reader and fails.
    """
    from terrainflow_assessment.qgis.controllers.reporting import resolve_target

    cases = [
        # (typed name,        chosen filter,     expected name,     expected html)
        ("plan.pdf", "PDF (*.pdf)", "plan.pdf", False),
        ("plan.html", "HTML (*.html)", "plan.html", True),
        # the trap: filter switched, name left alone
        ("plan.pdf", "HTML (*.html)", "plan.html", True),
        # typed an explicit .html but left the filter on PDF
        ("plan.html", "PDF (*.pdf)", "plan.html", True),
        ("plan.htm", "PDF (*.pdf)", "plan.htm", True),
        # no extension at all - some dialogs do not append one
        ("plan", "PDF (*.pdf)", "plan.pdf", False),
        ("plan", "HTML (*.html)", "plan.html", True),
        # a dot in the site name must not be eaten
        ("Smith 1.2 Block", "PDF (*.pdf)", "Smith 1.2 Block.pdf", False),
        ("Smith 1.2 Block", "HTML (*.html)", "Smith 1.2 Block.html", True),
        # no filter reported at all (non-native dialogs sometimes return "")
        ("plan.html", "", "plan.html", True),
        ("plan.pdf", "", "plan.pdf", False),
    ]
    for typed, chosen, expect_path, expect_html in cases:
        got_path, got_html = resolve_target(typed, chosen)
        assert (got_path, got_html) == (expect_path, expect_html), (
            f"{typed!r} + {chosen!r} -> {(got_path, got_html)}, "
            f"expected {(expect_path, expect_html)}")


def check_report_html_filter_writes_html_content(dem_path):
    """End to end: the HTML filter with a .pdf name lands real HTML on disk."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        # Exactly what the dialog hands back when the format is switched but
        # the pre-filled name is left alone.
        _export(h, str(tmp / "plan.pdf"), selected="HTML (*.html)")
        h.assert_no_errors("html filter with a pdf name")

        assert not (tmp / "plan.pdf").exists(), (
            "HTML content must not be written to a .pdf name")
        out = tmp / "plan.html"
        assert out.exists(), "expected the suffix to be corrected to .html"
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
        assert h.state.report_last_path == str(out)


# ---------------------------------------------------------------- both formats

def check_report_html_without_simulation(dem_path):
    """HTML used to need a simulation too — it rendered a different document."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()
        assert h.state.comparison is None

        out = _export(h, str(tmp / "plan.html"), selected="HTML (*.html)")
        h.assert_no_errors("html export without simulation")
        with open(out, encoding="utf-8") as handle:
            text = handle.read()
        assert text.startswith("<!DOCTYPE html>")
        assert "Your scheme in one page" in text


def check_report_html_is_self_contained(dem_path):
    """It gets emailed on its own, so nothing may be fetched from elsewhere."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()
        out = _export(h, str(tmp / "plan.html"), selected="HTML (*.html)")
        with open(out, encoding="utf-8") as handle:
            text = handle.read()

        assert "data:image/png;base64," in text, (
            "maps and charts must be embedded, not linked")
        for remote in ("http://", "https://", "file://", "<script"):
            assert remote not in text, f"{remote} leaked into the HTML"


def check_report_html_and_pdf_agree(dem_path):
    """The point of the shared model: one design, two formats, same figures."""
    from terrainflow_assessment.modules.report_model import (
        DataTable,
        Hero,
        build_report,
    )

    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.panel._site_name_edit.setText("Quail Island")
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        report = build_report(controller._collect(controller._site_name()))

        pdf = _export(h, str(tmp / "plan.pdf"))
        html_path = _export(h, str(tmp / "plan.html"),
                            selected="HTML (*.html)")
        h.assert_no_errors("both formats from one state")
        _assert_pdf(pdf, "pdf half of the pair")

        with open(html_path, encoding="utf-8") as handle:
            text = handle.read()
        # Every figure the model produced has to be in the HTML; the PDF's side
        # is covered by rendering its pages into the shot diff.
        import html as _h
        for section in report.sections:
            if isinstance(section, Hero):
                assert _h.escape(section.value) in text, section.value
            elif isinstance(section, DataTable):
                for row in section.rows:
                    for cell in row:
                        cell = str(cell).strip()
                        if cell and cell != "—":
                            assert _h.escape(cell) in text, cell
        assert "Quail Island" in text


def check_report_both_formats_carry_the_same_maps(dem_path):
    """A map drawn in one format and missing from the other is a disagreement."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        data = controller._collect(controller._site_name())
        specs = controller._map_specs(data)
        assert specs, "expected at least one drawable map"

        work = str(tmp / "assets")
        os.makedirs(work, exist_ok=True)
        rendered = controller._render_maps(data, work)
        assert set(rendered) == set(specs), (
            f"PDF would draw {sorted(specs)} but HTML has {sorted(rendered)}")
        for path in rendered.values():
            assert os.path.getsize(path) > 1000


# ---------------------------------------------------------------- rendering

def check_report_maps_include_the_stream_network(dem_path):
    """Regression: the stream layer was silently absent from every map.

    `_named_layer("Streams")` rejected any name starting with "Baseline — ",
    which is the only name the stream raster ever has, so it always resolved to
    None and was filtered out — with no warning anywhere, because a map with one
    fewer layer still renders.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        specs = controller._map_specs(controller._collect("Site"))
        # The summary and flow maps are the two that carry water. The design map
        # deliberately does not: it is the sheet somebody stands in a paddock
        # holding, and it shows the features and the ground and nothing else.
        for key in ("overview", "flow"):
            names = [layer.name() for layer in specs[key].layers]
            assert any("Streams" in name for name in names), (
                f"the {key} map has no stream network: {names}")
        design = [layer.name() for layer in specs["design"].layers]
        assert not any("Streams" in name for name in design), (
            f"the design map is carrying analysis layers again: {design}")


def check_report_maps_have_terrain_behind_them(dem_path):
    """Item 20: the design printed as coloured lines on blank white paper.

    The backdrop is now the operator's own aerial photograph where the project
    has one loaded, and the shaded relief where it does not. The harness project
    has no tile layer, so this exercises the fallback — which is the branch that
    matters: a project without a basemap must not fall back into the blank white
    page the hillshade was introduced to prevent.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        assert controller._basemap_layer() is None, (
            "the harness project has no tile layer; something matched anyway")
        specs = controller._map_specs(controller._collect("Site"))
        for key in ("overview", "design", "flow"):
            names = [layer.name() for layer in specs[key].layers]
            assert any("Hillshade" in name for name in names), (
                f"no shaded relief behind the {key} map: {names}")
            # Bottom-last: the layer list is top-first, so the backdrop must not
            # be drawn over the thing it is meant to sit behind.
            assert "Hillshade" in names[-1] or "DEM" in names[-1], (key, names)


def check_report_map_stacking_keeps_everything_visible(dem_path):
    """Order is the whole of a map's legibility, and every rule here is one a
    render silently breaks rather than failing on.

    The surface-runoff ramp reaches full opacity at its top stop, so anything
    drawn under it is buried: the boundary and the thresholded stream network
    both have to sit above it. Terrain has to sit under everything.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.plugin._earthworks._refresh_ew_layer()
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        specs = controller._map_specs(controller._collect("Site"))

        def order(key):
            # Layer lists are top-first, so a lower index draws in front.
            return {layer.name(): i
                    for i, layer in enumerate(specs[key].layers)}

        def find(names, fragment):
            for name, i in names.items():
                if fragment in name:
                    return i
            return None

        for key in specs:
            names = order(key)
            terrain = find(names, "Hillshade")
            boundary = find(names, "Drawn ")
            if terrain is not None:
                assert terrain >= max(names.values()) - 1, (
                    f"{key}: terrain is not at the back: {names}")
            if boundary is not None:
                for raster in ("Surface Runoff", "Pond Capacity"):
                    under = find(names, raster)
                    assert under is None or boundary < under, (
                        f"{key}: the boundary is buried under {raster}")

        flow = order("flow")
        streams, runoff = find(flow, "Streams"), find(flow, "Surface Runoff")
        if streams is not None and runoff is not None:
            assert streams < runoff, (
                f"the stream network is under the runoff wash: {flow}")


def check_report_maps_are_framed_on_the_boundary(dem_path):
    """Every figure at one scale, with the site in the same place on the page.

    Each map used to take the union of its own layers' extents, so the three
    figures came out at three different scales — and on a design whose DEM did
    not resolve, the design map zoomed to whatever the features happened to
    span.

    Two maps are exceptions, and both are excluded by name rather than by
    loosening the rule. The overview on page 1 exists to place the block in its
    surroundings, so it is deliberately pulled back — but it still has to be the
    *same* frame enlarged, concentric rather than a different view, or it stops
    being the figure the others are a zoom into. The design map is framed on the
    earthworks instead: it is read to build from, so it goes as close in as it
    can and still show every feature, which on a block whose design sits in one
    corner is a great deal closer than the boundary.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        # The drawn layer has to exist for the design map to have anything to
        # frame itself on; add_earthwork alone only reaches the store.
        h.plugin._earthworks._refresh_ew_layer()
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        specs = controller._map_specs(controller._collect("Site"))
        extents = {k: spec.extent for k, spec in specs.items()}
        comparable = {k: e for k, e in extents.items()
                      if k not in ("overview", "design")}
        assert len(comparable) >= 1, f"expected several maps, got {list(extents)}"
        first = next(iter(comparable.values()))
        for key, extent in comparable.items():
            assert extent == first, (
                f"the {key} map is framed differently: {extent} vs {first}")

        design = extents.get("design")
        assert design is not None, "page 3 lost its design map"
        assert design.width() < first.width(), (
            f"the design map is not zoomed to its earthworks: {design} vs {first}")
        assert first.contains(design.center()), (
            f"the design map has wandered off the block: {design}")

        overview = extents.get("overview")
        assert overview is not None, "page 1 lost its overview map"
        assert overview.width() > first.width(), (
            f"the overview is not pulled back: {overview} vs {first}")
        for axis in ("x", "y"):
            centre = getattr(overview, f"{axis}Minimum")() + \
                getattr(overview, f"{axis}Maximum")()
            same = getattr(first, f"{axis}Minimum")() + \
                getattr(first, f"{axis}Maximum")()
            assert abs(centre - same) < 1.0, (
                f"the overview is off-centre on {axis}: {overview} vs {first}")


def check_summary_map_is_the_scheme_over_the_ground(dem_path):
    """Log #1/#2: page one is a photograph of the block with its water on it.

    It used to be a shaded terrain model carrying every earthwork and its name
    label — a figure a reader has to decode before it has told them where they
    are, and the one page they look at first. What it draws now is the boundary,
    what the scheme holds, where the water runs, and the ground each feature
    catches; the earthworks and their labels moved to the design map.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.plugin._earthworks._refresh_ew_layer()
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        specs = controller._map_specs(controller._collect("Site"))
        layers = specs["overview"].layers
        names = [layer.name() for layer in layers]

        assert controller._boundary_layer() in layers, names
        assert any("Pond Capacity" in name for name in names), names
        assert any("Streams" in name for name in names), names
        # The drawn earthwork layers are named for their type — "Swales",
        # "Basins". None of them belongs on this map.
        assert not any(name in ("Swales", "Basins", "Dams", "Diversions",
                                "Berms") for name in names), (
            f"the summary map is carrying the design again: {names}")


def check_summary_map_outlines_what_the_scheme_catches(dem_path):
    """Log #2: the catchment traced as a line, not the filled raster.

    The filled version draws at alpha 150 over everything beneath it, which on
    an aerial photograph is most of the point of having the photograph. It is
    also only ever refreshed by a panel checkbox, so it shows the run before
    last; this is built from the labelling itself, which is recomputed on every
    edit.
    """
    from qgis.core import QgsWkbTypes

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        state = h.plugin._state
        assert state.catchment_labels is not None, (
            "nothing was labelled, so there is no outline to check")

        work = os.path.join(state.output_dir, "outline_check")
        os.makedirs(work, exist_ok=True)
        controller._open_transients(work)
        try:
            outline = controller._transients.get("catchment_outline")
            assert outline is not None, "the labelling traced to nothing"
            assert outline.featureCount() > 0, "an outline with no rings"
            assert outline.geometryType() == QgsWkbTypes.LineGeometry, (
                "the outline is not a line — the filled version already exists")
            # In the project so the layout can resolve it, out of the tree so the
            # operator's legend does not grow a layer per export.
            root = h.project.instance().layerTreeRoot()
            assert h.project.instance().mapLayer(outline.id()) is not None
            assert root.findLayer(outline.id()) is None, (
                "a report-only layer reached the layer tree")

            outline_id = outline.id()
            assert outline in controller._map_specs(
                controller._collect("S"))["overview"].layers, (
                "the summary map is not drawing the catchment outline")
        finally:
            controller._close_transients()
        assert h.project.instance().mapLayer(outline_id) is None, (
            "the report-only layer outlived the export")


def check_flow_map_runoff_stops_at_the_boundary(dem_path):
    """Log #3: the wash used to run off every edge of the page.

    The Surface Runoff raster covers the whole DEM tile, so the figure captioned
    "where the water goes" showed the block sitting in a fan of blue streaks
    over ground its owner can do nothing about. The clipped copy is a copy on
    purpose — the layer on the canvas is the analysis output, and the numbers
    behind it are measured over the whole grid.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()

        controller = h.plugin._reporting
        state = h.plugin._state
        source = controller._named_layer("Surface Runoff", group="baseline")
        assert source is not None, "the baseline drew no runoff layer"

        work = os.path.join(state.output_dir, "clip_check")
        os.makedirs(work, exist_ok=True)
        controller._open_transients(work)
        try:
            clipped = controller._transients.get("runoff_clipped")
            assert clipped is not None, "the runoff raster was not clipped"
            assert clipped.isValid(), "the clipped copy did not open"
            assert clipped.source() != source.source(), (
                "the source raster was clipped in place")
            # Same ramp, by cloning rather than by re-deriving it.
            assert type(clipped.renderer()) is type(source.renderer()), (
                "the clipped copy was restyled instead of inheriting")
            layers = controller._map_specs(controller._collect("S"))["flow"].layers
            assert clipped in layers, (
                "the flow map is still drawing the unclipped raster")
            assert source not in layers, [ly.name() for ly in layers]
        finally:
            controller._close_transients()


def check_report_map_never_draws_a_selection(dem_path):
    """A highlighted feature among forty reads as a finding.

    DrawSelection is on by QGIS's default flag set, and nothing in the plugin
    ever deselects — `contour.highlight_contour_rows` selects and leaves it —
    so whatever the operator last clicked printed into the report.
    """
    from terrainflow_assessment.qgis.adapters.map_image import render_map_image

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        spec = controller._map_specs(controller._collect("Site"))["design"]
        vectors = [layer for layer in spec.layers
                   if hasattr(layer, "selectAll")]
        assert vectors, "expected a vector layer on the design map"

        def render():
            return render_map_image(spec.layers, extent=spec.extent,
                                    crs=spec.crs, size_px=(400, 260), dpi=96,
                                    decorations=False)

        clean = render()
        for layer in vectors:
            layer.selectAll()
        selected = render()
        for layer in vectors:
            layer.removeSelection()

        assert clean is not None and selected is not None
        assert _image_bytes(clean) == _image_bytes(selected), (
            "selecting every feature changed the rendered map")


def _image_bytes(image):
    from qgis.PyQt.QtCore import QBuffer, QByteArray

    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QBuffer.WriteOnly)
    image.save(buf, "PNG")
    buf.close()
    return bytes(data)


def check_report_map_png_carries_a_scale_bar(dem_path):
    """The PDF has had one; the HTML's maps were bare rasters.

    Compared against the same render with decorations off, so this asserts that
    something was painted rather than that a particular pixel is a particular
    colour.
    """
    from terrainflow_assessment.qgis.adapters.map_image import render_map_image

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        controller = h.plugin._reporting
        spec = controller._map_specs(controller._collect("Site"))["flow"]

        def render(decorations):
            return render_map_image(spec.layers, extent=spec.extent,
                                    crs=spec.crs, size_px=(600, 400), dpi=96,
                                    decorations=decorations)

        plain, decorated = render(False), render(True)
        assert plain is not None and decorated is not None
        assert _image_bytes(plain) != _image_bytes(decorated), (
            "the decorated render is identical — no scale bar was drawn")

        # The bar sits bottom-left and the arrow top-right, so both corners must
        # have moved. Compared as corner *regions*, not single pixels: the exact
        # placement is a layout detail and a pixel probe just encodes it into a
        # check that is meant to be about whether the thing is there.
        assert _corner_changed(plain, decorated, "bottomleft"), "no scale bar"
        assert _corner_changed(plain, decorated, "topright"), "no north arrow"


def _corner_changed(before, after, corner, fraction=0.25):
    """True when a quarter-size corner of the image differs between renders."""
    w, h = after.width(), after.height()
    cw, ch = int(w * fraction), int(h * fraction)
    x0 = 0 if corner.endswith("left") else w - cw
    y0 = 0 if corner.startswith("top") else h - ch
    for x in range(x0, x0 + cw, 2):
        for y in range(y0, y0 + ch, 2):
            if before.pixel(x, y) != after.pixel(x, y):
                return True
    return False


def check_scale_bar_length_is_a_round_number(dem_path):
    """A bar reading "137 m" is not a scale bar — you cannot count it off."""
    from terrainflow_assessment.qgis.adapters.map_image import nice_bar_length

    for target, expected in [(137.0, 100), (999.0, 500), (1000.0, 1000),
                             (23.0, 20), (0.4, 1), (60000.0, 50000)]:
        assert nice_bar_length(target) == expected, target


def check_selection_highlight_clears_when_its_layer_is_hidden(dem_path):
    """Item 7/19: the green highlight could not be got rid of.

    The band is a canvas item held by the controller, deliberately independent
    of the layers so it survives them being rebuilt on every edit — which also
    meant unticking the swale layer, the obvious way to make it go away, did
    nothing at all.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()

        ew = h.plugin._earthworks
        # The harness adds straight to the manager, so the map layers only
        # exist once the controller has drawn them.
        ew._refresh_ew_layer()
        h.panel.earthwork_selected.emit(0)
        band = getattr(ew, "_selection_band", None)
        assert band is not None and band.numberOfVertices() > 0, (
            "selecting an earthwork should highlight it")

        hidden = 0
        for layer_id in ew._state.ew_layer_ids.values():
            node = h.project.layerTreeRoot().findLayer(layer_id)
            if node is not None:
                node.setItemVisibilityChecked(False)
                hidden += 1
        assert hidden, "there was no earthwork layer node to hide"
        assert band.numberOfVertices() == 0, (
            "hiding the layer must take its highlight with it")


def check_export_clears_the_selection_highlight(dem_path):
    """Whatever the operator last clicked must not print as a finding."""
    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.panel.analysis_inputs_changed.emit()
        h.panel.earthwork_selected.emit(0)

        band = getattr(h.plugin._earthworks, "_selection_band", None)
        assert band is not None and band.numberOfVertices() > 0

        _export(h, str(tmp / "clean.pdf"))
        h.assert_no_errors("export with a feature highlighted")
        assert band.numberOfVertices() == 0, (
            "the export left the highlight on the canvas")


def check_map_image_renders_non_blank(dem_path):
    from terrainflow_assessment.qgis.adapters.map_image import (
        layers_extent,
        render_map_image,
        usable_layers,
    )

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        layers = usable_layers([h.dem_layer])
        assert layers, "the DEM should be usable"
        image = render_map_image(
            layers, extent=layers_extent(layers, h.project.crs()),
            size_px=(600, 400), dpi=96, crs=h.project.crs())
        assert image is not None and not image.isNull()

        colours = {image.pixel(x, y)
                   for x in range(0, image.width(), 23)
                   for y in range(0, image.height(), 23)}
        assert len(colours) > 1, "the rendered map is a flat colour"


def check_map_image_ignores_dead_layers(dem_path):
    from terrainflow_assessment.qgis.adapters.map_image import render_map_image

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        assert render_map_image([None, None], size_px=(200, 200)) is None


def check_report_page_shots(dem_path):
    """Render every PDF page into the shot-diff workflow.

    A layout change then reports as a changed image with a pixel percentage —
    you look at it and accept it. No PDF parser, no new tooling.
    """
    from terrainflow_assessment.modules.report_charts import render_flow_network
    from terrainflow_assessment.modules.report_model import (
        build_flow_graph,
        build_report,
    )
    from terrainflow_assessment.qgis.adapters.layout_pdf import (
        build_layout,
        render_page_image,
    )

    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.panel._site_name_edit.setText("Quail Island")
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        # The drawn layer, and the layers the exporter builds for the document
        # itself. Without them these shots were of a report missing three of the
        # things it now draws — the design map framed on the block instead of on
        # its earthworks, no catchment outline, and the runoff wash unclipped —
        # so the one visual record of the map work showed none of it.
        h.plugin._earthworks._refresh_ew_layer()
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._reporting
        controller._open_transients(str(tmp))
        data = controller._collect(controller._site_name())
        # These shots test layout, not which run produced them. The footer
        # timestamp and the temp DEM path change every run, so left alone they
        # move every page by ~0.01% forever — the exact noise a real regression
        # would then hide behind.
        data.generated_at = "2026-01-01 00:00"
        data.dem = {"source": "<fixed for the shot diff>",
                    "fingerprint": "sha256:0000000000",
                    "cell size": "2.00 m", "crs": "EPSG:2193"}
        # This check does not re-analyse, so no burn has happened and the measured
        # earthmoving would be absent — and its table is the one place the document
        # prints a calculated and a measured figure side by side. Fixed here like the
        # timestamp above, for the same reason: the shot is about layout.
        # Scaled to this fixture's single small swale, not the field design's
        # totals: a ratio of 334x in the shot reads as a fault in the code
        # rather than as the layout it is there to show.
        data.burn_quantities = {"cut_m3": 101.0, "fill_m3": 12.0}
        images = {}
        if data.balance is not None and data.balance.per_feature:
            png = str(tmp / "network.png")
            if render_flow_network(build_flow_graph(data.balance), path=png):
                images["network"] = png
                from qgis.PyQt.QtGui import QImage
                save_qimage(QImage(png), "report_flow_network")

        try:
            layout = build_layout(h.project, build_report(data), images=images,
                                  maps=controller._map_specs(data))
            pages = layout.pageCollection().pageCount()
            assert pages >= 4, f"expected a multi-page report, got {pages}"

            for page in range(pages):
                image = render_page_image(layout, page, dpi=72)
                assert not image.isNull(), f"page {page + 1} rendered null"
                save_qimage(image, f"report_page{page + 1}")
        finally:
            controller._close_transients()


def check_report_cascade_is_laid_out_by_height(dem_path):
    """The flow diagram on the one shape it exists for: a chain of features.

    Every other check in this module builds a single swale, so the diagram is one
    box and `_layout` falls back to ranked columns — which left the elevation
    layout, the orthogonal edge routing and the channel staggering with no
    coverage here at all, resting entirely on unit tests over hand-made node
    dicts. This drives them through the real pipeline: real centroid elevations
    sampled off the DEM, real routing targets resolved downhill, real overflow.

    Four swales down the valley rather than across one. The synthetic surface
    falls ~15% at the top easing to ~3%, so features at different rows sit at
    genuinely different heights and each one drains into the next — which is
    exactly the arrangement the cascade is meant to make legible and the one a
    single feature can say nothing about.
    """
    from terrainflow_assessment.modules import report_charts
    from terrainflow_assessment.modules.report_model import build_flow_graph

    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()
        for row in (40, 90, 140, 190):
            h.add_earthwork("swale", line_across_valley(row=row))
        h.panel.analysis_inputs_changed.emit()

        data = h.plugin._reporting._collect("Site")
        assert data.balance is not None, "no balance to draw a cascade from"
        graph = build_flow_graph(data.balance, data.display_names)
        nodes = graph["nodes"]
        assert len(nodes) == 4, f"expected four features, got {len(nodes)}"

        # The DEM has to have supplied real heights, or everything below is
        # testing the fallback and quietly passing.
        known = [n for n in nodes if n.get("elevation_known")]
        assert len(known) == 4, (
            f"only {len(known)} of 4 features got an elevation off the DEM")
        heights = {n["id"]: float(n["elevation"]) for n in known}
        assert max(heights.values()) - min(heights.values()) >= 1.0, (
            f"features are too close in height to test a cascade: {heights}")

        positions = report_charts._layout(nodes)
        assert positions != report_charts._layout_by_rank(nodes), (
            "fell back to ranked columns despite four known elevations")

        # Downhill is down the page. That is the whole claim the caption makes.
        by_height = sorted(heights, key=lambda i: -heights[i])
        page_y = [positions[i][1] for i in by_height]
        assert page_y == sorted(page_y), (
            f"height order does not match page order: {list(zip(by_height, page_y))}")

        # No link may cross a box it does not connect to — the misreading that
        # made a link appear to run into a feature and out the other side.
        stagger = report_charts._channel_stagger(nodes, positions)
        w, hgt = report_charts._NODE_W, report_charts._NODE_H
        for node in nodes:
            target = node.get("target_id")
            if not target or target not in positions:
                continue
            sx, sy = positions[node["id"]]
            tx, ty = positions[target]
            others = [p for nid, p in positions.items()
                      if nid not in (node["id"], target)]
            route = report_charts._edge_route(
                (sx + w, sy + hgt / 2.0), (tx, ty + hgt / 2.0), others,
                stagger.get(node["id"], 0.0))
            for (x1, y1), (x2, y2) in zip(route, route[1:]):
                for bx, by in others:
                    crosses = (min(x1, x2) < bx + w and max(x1, x2) > bx
                               and min(y1, y2) < by + hgt and max(y1, y2) > by)
                    assert not crosses, (
                        f"{node['name']} -> {target} crosses the box at "
                        f"({bx:.1f}, {by:.1f}); route {route}")

        png = str(tmp / "cascade.png")
        assert report_charts.render_flow_network(graph, path=png), \
            "the cascade did not render"
        from qgis.PyQt.QtGui import QImage
        save_qimage(QImage(png), "report_flow_network_cascade")


def check_report_verification_page_renders(dem_path):
    """The volume ladder on paper, with a flagged row.

    Its own page rather than a change to check_report_page_shots: that fixture never
    runs a verification, so the ladder is a "nothing measured" callout there, and
    making it measure would move every report page image for one table.

    What to look at: the two renamed headers must not have starved the volume columns
    (column widths are shared out in proportion to the longest cell, so a long header
    takes width from the numbers), and the dagger must print as a dagger — a glyph
    missing from DejaVu Sans comes out as a box.
    """
    from terrainflow_assessment.modules.report_model import (
        Report,
        ReportData,
        _page_verification,
    )
    from terrainflow_assessment.modules.reporting import VerificationResult
    from terrainflow_assessment.qgis.adapters.layout_pdf import (
        build_layout,
        render_page_image,
    )

    data = ReportData(site_name="Quail Island", verification=VerificationResult(
        analytic_total_m3=2741.0, terrain_total_m3=2645.0,
        caveats=["Terrain volume is attributed by connected depression."],
        per_feature=[
            {"name": "Swale 33", "analytic_m3": 291.0, "geometric_m3": 364.0,
             "rasterisable_m3": 453.0, "terrain_m3": 453.0, "delta_pct": 0.0,
             "section_overstated": True, "section_gap_pct": 24.5,
             "existing_m3": 0.0, "total_m3": 453.0},
            {"name": "Basin 39", "analytic_m3": 612.0, "geometric_m3": 765.0,
             "rasterisable_m3": 772.0, "terrain_m3": 759.0, "delta_pct": -1.7,
             "section_overstated": False, "section_gap_pct": 0.9},
            {"name": "Dam 40", "analytic_m3": 1838.0, "terrain_m3": 1433.0,
             "delta_pct": -22.0, "barrier_impounded": True},
            {"name": "Swale 41", "analytic_m3": 80.0, "geometric_m3": 100.0,
             "routing_only": True},
        ]))

    with PluginHarness(dem_path) as h:
        report = Report(title="Verification page", sections=_page_verification(data))
        layout = build_layout(h.project, report)
        image = render_page_image(layout, 0, dpi=72)
        assert not image.isNull(), "the verification page rendered null"
        save_qimage(image, "report_page_verification")


# ---------------------------------------------------------------- failure UI

def check_failed_baseline_does_not_tick_green(dem_path):
    """A run that errored must not present as one that worked.

    The error handler called set_baseline_complete(), so a failure put the
    button in its Done state, ticked the stage green, and switched on every
    downstream results tool - for a run that produced no data at all.
    """
    with PluginHarness(dem_path) as h:
        h.plugin._baseline._on_analysis_error("boom")

        assert h.panel._stepper.state("baseline") == "todo", (
            "a failed first run must not tick the stage")
        assert not h.panel._query_ponding_btn.isEnabled(), (
            "results tools must stay off after a failed baseline")
        assert not h.panel._generate_contours_btn.isEnabled()
        assert not h.panel._toggle_throughflow_btn.isEnabled()
        assert "failed" in h.panel._baseline_results_lbl.text().lower()
        assert any("failed" in text.lower() for _, _, text in h.bar.criticals)


def check_failed_baseline_after_a_good_one_goes_amber(dem_path):
    """Earlier output still exists, but no longer reflects what was attempted."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        assert h.panel._stepper.state("baseline") == "done"
        assert h.panel._query_ponding_btn.isEnabled()

        h.plugin._baseline._on_analysis_error("boom")
        assert h.panel._stepper.state("baseline") == "stale", (
            "a failure over existing output should warn, not claim success")
        # The earlier run genuinely earned these; a later failure does not
        # retract it.
        assert h.panel._query_ponding_btn.isEnabled()


def check_failed_earthworks_does_not_tick_verify(dem_path):
    """A failed burn measures nothing, so Verify must not go green."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.plugin._earthworks._on_analysis_error("boom")

        assert h.panel._stepper.state("verify") == "todo"
        assert "failed" in h.panel._earthworks_results_lbl.text().lower()


def check_successful_runs_still_tick(dem_path):
    """The guard must not have broken the happy path."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        assert h.panel._stepper.state("baseline") == "done"
        assert h.panel._query_ponding_btn.isEnabled()

        h.add_earthwork("swale", line_across_valley())
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("earthworks re-analysis")
        assert h.panel._stepper.state("verify") == "done"


def check_section_after_a_spilled_table_is_not_drawn_on_the_cover(dem_path):
    """A multi-page table must leave the cursor under its *last* frame.

    `_advance_past` used to read the page index with
    `pageNumberForPoint(last.pagePos())`. Those are different coordinate spaces —
    `pagePos()` is relative to the frame's own page, `pageNumberForPoint` wants an
    absolute layout coordinate — so any spilled table sent the cursor back to page
    0 and everything after it was drawn on top of the cover.

    Asserted directly on `_advance_past` rather than end-to-end, and that is not
    laziness. A layout built in one pass rarely reaches the multi-frame branch:
    QGIS creates continuation frames when it recalculates frame sizes, which for
    most of these tables happens after the section was placed. And when the branch
    *is* reached, two things downstream repair the damage by accident — a level-1
    heading starts its own page, and `_room_for` calls `_new_page` whenever the
    cursor sits low — so an end-to-end assertion passes with the bug in place. It
    was written that way first and did exactly that.
    """
    from qgis.core import QgsProject

    from terrainflow_assessment.modules.report_model import DataTable, Report
    from terrainflow_assessment.qgis.adapters.layout_pdf import ReportLayoutBuilder

    rows = [[f"Swale {i}", f"{i * 3} m", f"{i * 11} m3", "0.5 m"] for i in range(90)]
    report = Report(
        title="Spill test",
        sections=[DataTable(title="Build schedule",
                            headers=["Feature", "Length", "Volume", "Depth"],
                            rows=rows,
                            note="Every row above is one feature.")],
    )

    builder = ReportLayoutBuilder(QgsProject.instance(), report, None, None, 200)
    layout = builder.render()

    spilled = [m for m in layout.multiFrames()
               if hasattr(m, "frames") and len(m.frames()) > 1]
    assert spilled, "the table did not spill — raise the row count"
    table = spilled[0]
    last = max(table.frames(), key=lambda f: f.page())
    assert last.page() > 0, "the table never left page 0"

    builder._advance_past(table, 0.0)

    assert builder._page == last.page(), (
        f"cursor left on page {builder._page}, but the table's last frame is on "
        f"page {last.page()} — a page-relative point was read as an absolute one")
    expected_y = last.pagePos().y() + last.rect().height() + 2.0
    assert abs(builder._y - expected_y) < 0.01, (
        f"cursor y is {builder._y:.1f}, expected {expected_y:.1f} — measured from "
        f"the top margin rather than from where the frame actually sits")



def check_export_puts_the_operators_selection_back(dem_path):
    """Selections are cleared so they cannot print — then restored.

    The clear runs over every vector layer in the project, including the operator's
    own cadastre and asset layers, because the live layout map draws whatever it is
    handed. Destroying a selection somebody built by hand is not an acceptable price
    for exporting a document.
    """
    from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

    with PluginHarness(dem_path) as h, workdir() as tmp:
        h.run_baseline()

        own = QgsVectorLayer(
            f"Point?crs={h.dem_layer.crs().authid()}", "Operator assets", "memory")
        centre = h.dem_layer.extent().center()
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(centre))
        own.dataProvider().addFeatures([feature])
        own.updateExtents()
        h.project.instance().addMapLayer(own)
        own.selectAll()
        held = list(own.selectedFeatureIds())
        assert held, "nothing was selected to begin with"

        _export(h, str(tmp / "selection.pdf"))
        h.assert_no_errors("export with an operator selection")

        assert list(own.selectedFeatureIds()) == held, (
            "the export destroyed a selection that was not its to clear")


def check_the_report_summary_appears_without_a_simulation(dem_path):
    """It came only from a ComparisonResult, which only a fill simulation makes.

    So the Report stage's on-screen headline stayed blank for a document that has
    needed nothing but a baseline since it was rebuilt on the design tier — and
    when it did appear it printed the signed cut/fill figure `cut_fill_sentence`
    exists to replace.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", line_across_valley())
        ew.capacity_m3 = 50.0
        # add_earthwork goes straight to the manager, so nothing fires the live
        # assessment the way an edit on the canvas does. This is that edit.
        h.plugin._earthworks._recompute_live_assessment()
        h.assert_no_errors("design analysis")
        assert h.state.comparison is None, "a simulation ran; this checks the tier alone"

        text = h.panel._report_summary_lbl.text()
        assert "Run Baseline" not in text, "the summary is still idle after a design"
        assert "captured on-site" in text, f"no capture headline: {text!r}"
        assert "Earthmoving" in text, f"no earthmoving line: {text!r}"
        assert "m³" not in text.split("Earthmoving")[1].split("<br>")[0][:2], (
            "the earthmoving line leads with a bare signed volume")
        assert "Peak flow reduction" not in text, (
            "a timing claim with no simulation behind it")


def check_a_simulation_adds_the_timing_lines(dem_path):
    """Peak reduction and delay are timing claims, and only a time-stepped run
    can support one — so they appear when there is a simulation and not before."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", line_across_valley())
        ew.capacity_m3 = 50.0
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        text = h.panel._report_summary_lbl.text()
        assert "Peak flow reduction" in text, f"no timing line after a run: {text!r}"
        assert "from the simulation" in text, (
            "the timing line does not say where it came from")


def check_no_table_title_is_orphaned_at_the_foot_of_a_page(dem_path):
    """`_table` drew a DataTable's title with `_label`, which reserves room only
    for itself and commits `_y` as it goes — so the title always fitted and the
    overflow guard below it then decided, too late, to start a new page, leaving
    a 10 pt bold heading at the foot of one page and its table overleaf.
    `_render_heading` already carries one section of lookahead for this class,
    and a DataTable's title is drawn in `_table`, not by a Heading, so it was not
    covered.

    Driven at the page boundary rather than hoping a real report lands on one.
    A fixture that happens to put a title 250 mm down an A4 page is a fixture
    hunt, and one that stops doing so after an unrelated wording change is a
    check that quietly stops checking — this sets `_y` to the sliver directly,
    which is the only state the guard is about.
    """
    from qgis.core import (
        QgsLayoutItemLabel,
        QgsLayoutItemTextTable,
        QgsPrintLayout,
    )
    from terrainflow_assessment.modules.report_model import Report
    from terrainflow_assessment.qgis.adapters.layout_pdf import ReportLayoutBuilder

    with PluginHarness(dem_path) as h:
        builder = ReportLayoutBuilder(h.project, Report(title="T", subtitle="", footer=""))
        # `render()` builds the layout and then walks every section; this check
        # wants one section at one cursor position, so it does the first half.
        builder.layout = QgsPrintLayout(builder.project)
        builder.layout.initializeDefaults()
        builder.layout.pageCollection().clear()
        builder._new_page(landscape=False)
        # A sliver: enough for the 10 pt title, nowhere near the 30 mm the table
        # itself asks for. This is exactly the state that produced the orphan.
        builder._y = builder._page_bottom() - 12.0
        before = builder._page

        builder._table("Before and after the earthworks",
                       ["", "As it is now", "With your design"],
                       [["Water leaving the block", "1,000 m³", "1,300 m³"],
                        ["Fastest flow at the boundary", "40 L/s", "52 L/s"],
                        ["When that peak arrives", "6.0 hr", "4.0 hr"]])

        layout = builder.layout
        titles = [item for item in layout.items()
                  if isinstance(item, QgsLayoutItemLabel)
                  and item.text().strip() == "Before and after the earthworks"]
        assert len(titles) == 1, f"the title was not drawn once: {len(titles)}"

        tables = [m for m in layout.multiFrames()
                  if isinstance(m, QgsLayoutItemTextTable) and m.frames()]
        assert tables, "no table was placed"

        title_page = titles[0].page()
        table_page = tables[0].frames()[0].page()
        assert title_page == table_page, (
            f"the title landed on page {title_page} and its table on page "
            f"{table_page} — an orphaned heading over nothing")
        assert title_page > before, (
            "neither the title nor the table moved off the full page, so the "
            "fixture never reached the boundary this guard is about")
