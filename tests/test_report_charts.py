"""Tests for the flow network diagram.

Pixel-perfect output is not the point — that is what the tests_qgis screenshot
diff is for. What matters here is that it draws every node without raising,
survives the shapes the real data can take (rings, terminal-only designs, large
schemes), and returns None rather than exploding when matplotlib is absent.
"""

import pytest

from terrainflow_assessment.modules import report_charts
from terrainflow_assessment.modules.report_model import build_flow_graph
from terrainflow_assessment.modules.water_balance import BalanceResult


def _feature(fid, name, target=None, terminal=True, **kw):
    row = {
        "id": fid, "name": name, "ew_type": "swale", "capacity_m3": 1250.0,
        "stored_m3": 1100.0, "fill_pct": 88.0, "overflowed": False,
        "overflow_m3": 0.0, "target_id": target, "is_user_link": False,
        "is_terminal": terminal,
    }
    row.update(kw)
    return row


def _graph(features):
    return build_flow_graph(BalanceResult(per_feature=features))


def _png(graph, **kw):
    return report_charts.render_flow_network(graph, **kw)


class TestRenderFlowNetwork:
    def test_returns_png_bytes(self):
        out = _png(_graph([_feature("a", "Swale 1")]))
        assert out is not None
        assert out[:8] == b"\x89PNG\r\n\x1a\n"

    def test_writes_to_path(self, tmp_path):
        target = tmp_path / "network.png"
        got = _png(_graph([_feature("a", "Swale 1")]), path=str(target))
        assert got == str(target)
        assert target.exists() and target.stat().st_size > 0

    def test_empty_graph_draws_nothing(self):
        assert _png(_graph([])) is None

    def test_chain_of_features(self):
        g = _graph([
            _feature("a", "Swale 1", target="b", terminal=False),
            _feature("b", "Basin 2", target="c", terminal=False),
            _feature("c", "Dam 3"),
        ])
        assert _png(g) is not None

    def test_ring_does_not_hang(self):
        """resolve_targets warns about rings rather than rejecting them."""
        g = _graph([
            _feature("a", "A", target="b", terminal=False),
            _feature("b", "B", target="a", terminal=False),
        ])
        assert _png(g) is not None

    def test_overflowing_feature_draws_its_volume(self):
        g = _graph([
            _feature("a", "Swale 1", target="b", terminal=False,
                     overflowed=True, overflow_m3=430.0, fill_pct=100.0),
            _feature("b", "Basin 2"),
        ])
        assert _png(g) is not None

    def test_every_earthwork_type_has_a_colour(self):
        feats = [_feature(str(i), f"F{i}", ew_type=t)
                 for i, t in enumerate(
                     ("swale", "basin", "dam", "berm", "diversion"))]
        for f in feats:
            assert report_charts.type_colour(f["ew_type"]) !=                 report_charts._FALLBACK_TYPE_COLOUR
        assert _png(_graph(feats)) is not None

    def test_node_colour_is_the_registry_colour(self):
        """The diagram used to keep its own five-colour table, and it drifted:
        a swale was cyan on the map and blue here, a dam brown there and green
        here. One source of truth or the printed page contradicts the screen."""
        from terrainflow_assessment.core.registry.earthwork_types import (
            all_types,
            get_type,
        )

        for key in all_types():
            assert report_charts.type_colour(key) == get_type(key).style[1]

    def test_unknown_type_colour_is_the_fallback(self):
        assert report_charts.type_colour("keyline") ==             report_charts._FALLBACK_TYPE_COLOUR
        assert report_charts.type_colour(None) ==             report_charts._FALLBACK_TYPE_COLOUR

    def test_every_glyph_exists_in_the_chart_font(self):
        """A glyph DejaVu lacks prints as a tofu box, silently.

        The panel can use anything the system fonts have; matplotlib defaults to
        DejaVu Sans and warns rather than failing, so without this the first
        anyone would know is a box in a printed report.
        """
        from fontTools.ttLib import TTFont
        from matplotlib.font_manager import FontProperties, findfont

        font = TTFont(findfont(FontProperties(family="DejaVu Sans")))
        cmap = set()
        for table in font["cmap"].tables:
            cmap |= set(table.cmap.keys())
        for ew_type, glyph in report_charts._GLYPH.items():
            for ch in glyph:
                assert ord(ch) in cmap, (
                    f"{ew_type} glyph {ch!r} (U+{ord(ch):04X}) is missing from "
                    "DejaVu Sans and will print as a box")

    def test_rendering_emits_no_missing_glyph_warning(self):
        """Belt and braces: catches a glyph added to a label, not just _GLYPH."""
        import warnings

        feats = [_feature(str(i), f"F{i}", ew_type=t, overflowed=True,
                          overflow_m3=120.0)
                 for i, t in enumerate(
                     ("swale", "basin", "dam", "berm", "diversion"))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _png(_graph(feats))
        missing = [str(w.message) for w in caught
                   if "missing from font" in str(w.message)]
        assert not missing, missing

    def test_unknown_type_falls_back(self):
        assert _png(_graph([_feature("a", "X", ew_type="keyline")])) is not None

    def test_large_scheme_keeps_its_detail(self):
        """A 42-feature design is exactly where a reader needs to know which
        features are full, so the detail no longer degrades with size.

        It used to fall back to name-only above 25 nodes, which left the page
        answering none of the questions it exists to answer."""
        feats = [_feature(str(i), f"Swale {i}") for i in range(42)]
        assert _png(_graph(feats)) is not None

    def test_wide_rank_wraps_into_sub_columns(self):
        feats = [_feature(str(i), f"Swale {i}") for i in range(20)]
        positions = report_charts._layout(_graph(feats)["nodes"])
        assert len({x for x, _ in positions.values()}) > 1

    @pytest.mark.parametrize("fill", [0.0, 50.0, 100.0, 140.0, -5.0])
    def test_fill_bar_clamps(self, fill):
        g = _graph([_feature("a", "Swale 1", fill_pct=fill)])
        assert _png(g) is not None

    def test_missing_matplotlib_returns_none(self, monkeypatch):
        monkeypatch.setattr(report_charts, "_pyplot", lambda: None)
        assert _png(_graph([_feature("a", "Swale 1")])) is None


class TestSimulationCharts:
    """The hydrograph and fill timeline reuse the builders in reporting.py.

    Both renderers want a file path, so these wrap the existing base64 builders
    rather than reimplementing two charts that are already tested.
    """

    def _post(self):
        from terrainflow_assessment.modules.reporting import (
            PostInterventionReport,
        )
        return PostInterventionReport(
            earthwork_summary=[{"name": "Swale 1", "capacity_m3": 100.0}],
            timestep_table=[{"time_hr": h / 2.0, "outflow_ls": 10.0 * h,
                             "outflow_ls_baseline": 14.0 * h,
                             "Swale 1_fill_pct": min(100.0, 12.0 * h),
                             "Swale 1_overflow": h > 6}
                            for h in range(10)])

    def _baseline(self):
        from terrainflow_assessment.modules.reporting import BaselineReport
        return BaselineReport(
            timestep_table=[{"time_hr": h / 2.0, "outflow_ls": 14.0 * h}
                            for h in range(10)])

    def test_hydrograph_writes_a_png(self, tmp_path):
        out = str(tmp_path / "hydro.png")
        assert report_charts.render_hydrograph(
            self._baseline(), self._post(), out) == out
        assert open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n"

    def test_fill_timeline_writes_a_png(self, tmp_path):
        out = str(tmp_path / "fill.png")
        assert report_charts.render_fill_timeline(self._post(), out) == out
        assert open(out, "rb").read(8) == b"\x89PNG\r\n\x1a\n"

    @pytest.mark.parametrize("baseline,post", [(None, "post"), ("base", None)])
    def test_hydrograph_needs_both_halves(self, tmp_path, baseline, post):
        out = str(tmp_path / "h.png")
        assert report_charts.render_hydrograph(
            self._baseline() if baseline else None,
            self._post() if post else None, out) is None

    def test_fill_timeline_without_a_simulation(self, tmp_path):
        assert report_charts.render_fill_timeline(
            None, str(tmp_path / "f.png")) is None

    def test_empty_timeline_writes_nothing(self, tmp_path):
        from terrainflow_assessment.modules.reporting import (
            PostInterventionReport,
        )
        assert report_charts.render_fill_timeline(
            PostInterventionReport(), str(tmp_path / "f.png")) is None

    def test_write_b64_rejects_empty(self, tmp_path):
        assert report_charts._write_b64(None, str(tmp_path / "x.png")) is None
        assert report_charts._write_b64("", str(tmp_path / "x.png")) is None
