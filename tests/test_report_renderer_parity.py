"""The PDF and the HTML must not be able to drift apart.

Both renderers consume the same ``Report`` from ``build_report``. That makes
agreement possible; these tests make it enforced. A section type added to one
renderer and not the other, or a number that reaches one output and not the
other, fails here rather than in a printed document six months later.

``layout_pdf`` is importable under the mocked QGIS in ``tests/`` because every
``qgis`` import in it is inside a function — only its dispatch table is read.
"""

import inspect

import pytest

from terrainflow_assessment.modules import report_html, report_model
from terrainflow_assessment.modules.report_model import (
    Callout,
    DataTable,
    Hero,
    KeyValueTable,
    MapRef,
    Report,
    ReportData,
    Section,
    StatGrid,
    build_report,
)
from terrainflow_assessment.modules.reporting import (
    BaselineReport,
    ComparisonResult,
    PostInterventionReport,
    VerificationResult,
)
from terrainflow_assessment.modules.water_balance import BalanceResult
from terrainflow_assessment.qgis.adapters import layout_pdf


def _concrete_sections():
    """Every Section subclass the model can emit."""
    return {
        obj for _, obj in inspect.getmembers(report_model, inspect.isclass)
        if issubclass(obj, Section) and obj is not Section
    }


def _feature(fid="a", name="Swale 1", **kw):
    row = {
        "id": fid, "name": name, "ew_type": "swale",
        "direct_catchment_m2": 34000.0, "direct_inflow_m3": 900.0,
        "upstream_inflow_m3": 300.0, "total_inflow_m3": 1200.0,
        "stored_m3": 1100.0, "infiltration_m3": 60.0,
        "infiltration_buffer_m3": 90.0, "drain_hours": 18.0,
        "overflow_m3": 0.0, "capacity_m3": 1250.0, "fill_pct": 88.0,
        "overflowed": False, "target_id": None, "is_user_link": False,
        "is_terminal": True,
    }
    row.update(kw)
    return row


class _Earthwork:
    id, name, type, enabled = "a", "Swale 1", "swale", True
    length_m, top_width_m, bottom_width_m, depth = 148.2, 2.4, 1.2, 0.45
    side_slope, soil_name, capacity_m3 = 1.5, "Loam", 96.0
    crest_elevation, key_into_banks = None, False
    gradient_pct, companion_berm = 1.0, False


def _full_data():
    """Everything switched on, including the simulation enrichment."""
    return ReportData(
        site_name="Quail Island",
        generated_at="2026-08-09 14:22",
        plugin_version="0.2.0",
        run_tag="120mm·24h·C0.50·0.5ha",
        baseline=BaselineReport(
            site_name="Quail Island", crs="EPSG:2193", cell_size_m=1.0,
            catchment_area_ha=36.2, rainfall_mm=120.0, duration_hr=24.0,
            cn=61, runoff_mm=52.3, total_runoff_m3=21715.0,
            exit_volume_m3=10767.0, peak_outflow_ls=2012.0,
            exit_points=[{"label": "Exit 1: x", "flow_ls": 42.3,
                          "volume_m3": 3656.0}]),
        balance=BalanceResult(
            capture_pct=78.0, total_inflow_m3=2521.0, total_captured_m3=1966.0,
            site_exit_m3=555.0, total_capacity_m3=2500.0,
            total_infiltration_m3=120.0, infiltration_buffer_m3=180.0,
            total_cut_m3=7944.0, total_fill_m3=3533.0,
            terminal_deficit_m3=430.0, mass_balance_ok=True,
            routing_warnings=["A routing warning."],
            per_feature=[_feature()]),
        earthworks=[_Earthwork()],
        verification=VerificationResult(
            analytic_total_m3=1000.0, terrain_total_m3=950.0,
            caveats=["A caveat."],
            # Flagged: the sweep must see a row whose Δ carries the marker and whose
            # explanation lands under the table, since print has no tooltip to hide in.
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                          "terrain_m3": 950.0, "delta_pct": -13.6,
                          "section_overstated": True, "section_gap_pct": -12.0,
                          "existing_m3": 42.0, "total_m3": 992.0}]),
        comparison=ComparisonResult(
            captured_pct=78.0, exit_reduction_pct=40.0, peak_reduction_pct=35.0,
            peak_delay_hr=1.5,
            baseline=BaselineReport(exit_volume_m3=10767.0,
                                    peak_outflow_ls=2012.0),
            post=PostInterventionReport(exit_volume_m3=6460.0,
                                        peak_outflow_ls=1308.0)),
        spillway_rows=[{
            "name": "Dam 4", "state": "fail", "peak_flow_m3s": 0.393,
            "required_width_m": 1.65, "built_width_m": 1.20,
            "actual_head_m": 0.37, "freeboard_m": 0.33,
            "problems": ["A spillway problem."]}],
        spillway_context={"intensity_is_default": True,
                          "intensity_mm_hr": 40.0},
        inputs={"sizing_basis": "coefficient", "runoff_coefficient": 0.5},
        dem={"fingerprint": "sha256-sampled:4f3a9c1"},
    )


# ---------------------------------------------------------------- dispatch

class TestDispatchParity:
    def test_both_renderers_cover_every_section_type(self):
        """The guard against adding a section type to only one renderer."""
        expected = _concrete_sections()
        assert set(report_html.SECTION_HANDLERS) == expected, (
            "HTML renderer is missing: "
            f"{sorted(c.__name__ for c in expected - set(report_html.SECTION_HANDLERS))}")
        assert set(layout_pdf.SECTION_HANDLERS) == expected, (
            "PDF renderer is missing: "
            f"{sorted(c.__name__ for c in expected - set(layout_pdf.SECTION_HANDLERS))}")

    def test_the_two_tables_agree(self):
        assert set(report_html.SECTION_HANDLERS) == set(layout_pdf.SECTION_HANDLERS)

    def test_every_pdf_handler_exists_on_the_builder(self):
        for section_cls, method in layout_pdf.SECTION_HANDLERS.items():
            assert hasattr(layout_pdf.ReportLayoutBuilder, method), (
                f"{section_cls.__name__} maps to a missing method {method!r}")

    def test_every_html_handler_is_callable(self):
        for section_cls, fn in report_html.SECTION_HANDLERS.items():
            assert callable(fn), f"{section_cls.__name__} has no HTML handler"

    def test_both_renderers_draw_the_map_key(self):
        """The legend is model data, so a renderer that ignores it prints a map
        of five colours with nothing saying what any of them mean."""
        assert hasattr(report_html, "_legend")
        assert hasattr(layout_pdf.ReportLayoutBuilder, "_map_key")
        assert hasattr(layout_pdf.ReportLayoutBuilder, "_north_arrow")


class TestMapKeyRendering:
    def _map_ref(self):
        report = build_report(_full_data())
        return [s for s in report.sections
                if isinstance(s, MapRef) and s.key == "design"][0]

    def test_html_names_every_legend_entry(self):
        ref = self._map_ref()
        assert ref.legend, "the design map should carry a key"
        html = report_html._legend(ref.legend)
        for entry in ref.legend:
            assert _escaped(entry.label) in html

    def test_html_paints_the_swatch_in_the_model_s_colour(self):
        ref = self._map_ref()
        html = report_html._legend(ref.legend)
        for entry in ref.legend:
            if entry.kind != "ramp":
                assert entry.colour in html

    def test_a_ramp_renders_as_a_gradient_not_two_ends(self):
        """The reader is matching a wash on the map against the key, so the
        middle stops are the ones doing the work."""
        report = build_report(_full_data())
        flow = [s for s in report.sections
                if isinstance(s, MapRef) and s.key == "flow"][0]
        ramp = [e for e in flow.legend if e.kind == "ramp"][0]
        html = report_html._legend(flow.legend)
        assert "linear-gradient" in html
        for colour in ramp.colours:
            assert colour in html

    def test_no_key_markup_when_there_is_no_key(self):
        assert report_html._legend([]) == ""
        assert report_html._legend(None) == ""


# ---------------------------------------------------------------- content

def _rendered_html(data=None):
    return report_html.render_html(build_report(data or _full_data()))


class TestContentAgreement:
    """The HTML must carry every value the shared model produces.

    The PDF's side of this is covered by tests_qgis rendering its pages; what
    matters here is that neither renderer silently drops model content.
    """

    def test_every_table_cell_reaches_the_html(self):
        report = build_report(_full_data())
        html = report_html.render_html(report)
        for section in report.sections:
            if not isinstance(section, (DataTable, KeyValueTable)):
                continue
            for row in section.rows:
                for cell in row:
                    text = str(cell).strip()
                    if not text or text == "—":
                        continue
                    assert _escaped(text) in html, (
                        f"{text!r} from {section.title!r} is missing from the HTML")

    def test_every_headline_reaches_the_html(self):
        report = build_report(_full_data())
        html = report_html.render_html(report)
        for section in report.sections:
            if isinstance(section, Hero):
                assert _escaped(section.value) in html
                assert _escaped(section.label) in html
            elif isinstance(section, Callout) and section.title:
                assert _escaped(section.title) in html
            elif isinstance(section, StatGrid):
                for card in section.cards:
                    assert _escaped(str(card[1])) in html

    def test_no_markup_leaks_into_the_prose(self):
        """Neither renderer interprets markdown.

        The PDF draws section text straight into a layout label and the HTML
        escapes it, so `*emphasis*` printed as literal asterisks in both. If
        emphasis is ever wanted it has to be a section attribute, not markup
        smuggled through the copy.
        """
        report = build_report(_full_data())
        for section in report.sections:
            for attr in ("text", "title", "label", "sub", "note", "caption"):
                value = getattr(section, attr, None)
                if not isinstance(value, str):
                    continue
                for marker in ("*", "_ ", "`", "<b>", "<i>", "<em>"):
                    assert marker not in value, (
                        f"{marker!r} in {type(section).__name__}.{attr}: "
                        f"{value[:80]!r}")

    def test_headings_appear_in_order(self):
        report = build_report(_full_data())
        html = report_html.render_html(report)
        from terrainflow_assessment.modules.report_model import Heading

        at = -1
        for heading in [s for s in report.sections if isinstance(s, Heading)]:
            found = html.find(_escaped(heading.text), at + 1)
            assert found > at, f"{heading.text!r} out of order or missing"
            at = found


def _escaped(text):
    import html as _h
    return _h.escape(str(text))


# ---------------------------------------------------------------- html shape

class TestHtmlRenderer:
    def test_is_self_contained(self, tmp_path):
        html = _rendered_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "<style>" in html
        # Nothing may be fetched from elsewhere: the file gets emailed alone.
        for remote in ("http://", "https://", "<script"):
            assert remote not in html

    def test_writes_a_file(self, tmp_path):
        out = str(tmp_path / "report.html")
        assert report_html.write_html(out, build_report(_full_data())) == out
        assert open(out, encoding="utf-8").read().startswith("<!DOCTYPE")

    def test_site_name_is_the_title(self):
        assert "<title>Quail Island — Site Water Plan</title>" in _rendered_html()

    def test_empty_data_renders(self):
        html = report_html.render_html(build_report(ReportData()))
        assert html.startswith("<!DOCTYPE html>")
        assert report_model.REASON_NO_BASELINE in html

    def test_content_is_escaped(self):
        data = _full_data()
        data.site_name = 'Smith <script>alert("x")</script> Block'
        html = report_html.render_html(build_report(data))
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_missing_map_states_its_reason(self):
        data = _full_data()
        data.maps = {"design": "the layer was deleted"}
        html = report_html.render_html(build_report(data))
        assert "the layer was deleted" in html

    def test_missing_chart_falls_back_to_the_table(self):
        """A chart is never the sole carrier of a number, in either format."""
        html = report_html.render_html(build_report(_full_data()), images={})
        assert "Overflow chains" in html

    def test_embedded_image_becomes_a_data_uri(self, tmp_path):
        png = tmp_path / "network.png"
        png.write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)   # not a valid image, just bytes
        html = report_html.render_html(build_report(_full_data()),
                                       images={"network": str(png)})
        assert "data:image/png;base64," in html

    def test_unknown_section_type_is_skipped_not_fatal(self):
        class Exotic(Section):
            pass

        report = Report(title="t", sections=[Exotic()])
        assert report_html.render_html(report).startswith("<!DOCTYPE")

    def test_unreadable_image_falls_back(self, tmp_path):
        html = report_html.render_html(
            build_report(_full_data()),
            images={"network": str(tmp_path / "gone.png")})
        assert "Overflow chains" in html

    def test_unreadable_map_states_a_reason(self, tmp_path):
        html = report_html.render_html(
            build_report(_full_data()),
            maps={"design": str(tmp_path / "gone.png")})
        assert "not available" in html

    def test_data_uri_survives_an_unreadable_file(self, tmp_path):
        assert report_html._data_uri(str(tmp_path / "missing.png")) == ""

    def test_empty_collections_render_nothing(self):
        ctx = {"images": {}, "maps": {}}
        assert report_html.render_section(StatGrid(cards=[]), ctx) == ""
        assert report_html.render_section(
            KeyValueTable(title="t", rows=[]), ctx) == ""
        assert report_html.render_section(
            DataTable(title="t", headers=["a"], rows=[]), ctx) == ""

    def test_headerless_table_omits_the_header_row(self):
        ctx = {"images": {}, "maps": {}}
        out = report_html.render_section(
            DataTable(headers=["", ""], rows=[["1", "a note"]]), ctx)
        assert "<thead>" not in out and "a note" in out


# ---------------------------------------------------------------- enrichment

class TestSimulationEnrichment:
    def test_absent_without_a_comparison(self):
        data = _full_data()
        data.comparison = None
        html = report_html.render_html(build_report(data))
        assert "How the storm plays out over time" not in html

    def test_present_with_one(self):
        html = _rendered_html()
        assert "How the storm plays out over time" in html
        assert "Fastest flow at the boundary" in html

    def test_timing_claims_are_attributed_to_the_simulation(self):
        """Timing is the one thing the event-total balance cannot claim."""
        report = build_report(_full_data())
        callouts = [s for s in report.sections
                    if isinstance(s, Callout)
                    and "come from the simulation" in s.title]
        assert callouts

    @pytest.mark.parametrize("field", ["baseline", "post"])
    def test_half_a_comparison_does_not_crash(self, field):
        data = _full_data()
        setattr(data.comparison, field, None)
        assert report_html.render_html(build_report(data))
