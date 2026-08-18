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


_NODE_W = report_charts._NODE_W
_NODE_H = report_charts._NODE_H


def _node(nid, elev=None, known=True, rank=0, order=0, target=None):
    return {"id": nid, "name": nid, "rank": rank, "order": order,
            "target_id": target, "elevation": elev,
            "elevation_known": known and elev is not None}


class TestElevationLayout:
    """Water runs downhill, and a reader tracing a cascade is asking which
    feature is above which. Rank columns answer a different question — how many
    links from the top — and where two chains interleave in height the two
    answers look identical and are not."""

    def test_height_order_becomes_page_order(self):
        pos = report_charts._layout([
            _node("high", 220.0, rank=0), _node("mid", 210.0, rank=0),
            _node("low", 200.0, rank=0)])
        assert pos["high"][1] < pos["mid"][1] < pos["low"][1]

    def test_rank_still_drives_the_across_axis(self):
        pos = report_charts._layout([
            _node("a", 220.0, rank=0), _node("b", 200.0, rank=1)])
        assert pos["b"][0] > pos["a"][0]

    def test_boxes_at_the_same_height_do_not_stack_on_each_other(self):
        pos = report_charts._layout([
            _node("a", 220.00, rank=0), _node("b", 219.99, rank=0),
            _node("c", 200.00, rank=0)])
        ys = sorted(p[1] for p in pos.values())
        assert all(b - a >= _NODE_H for a, b in zip(ys, ys[1:])), ys

    def test_a_flat_scheme_falls_back_to_ranks(self):
        """Within a few centimetres there is nothing to draw to scale, and the
        chain says more than a flat line does."""
        nodes = [_node("a", 200.00, rank=0), _node("b", 200.05, rank=1)]
        assert report_charts._layout(nodes) == report_charts._layout_by_rank(nodes)

    def test_one_known_elevation_is_not_enough(self):
        nodes = [_node("a", 220.0, rank=0), _node("b", None, known=False, rank=1)]
        assert report_charts._layout(nodes) == report_charts._layout_by_rank(nodes)

    def test_an_unknown_elevation_is_parked_not_invented(self):
        """A centroid on nodata must never be placed by a stand-in value — it
        would sit on the wrong contour rather than merely out of order."""
        pos = report_charts._layout([
            _node("high", 220.0, rank=0), _node("low", 200.0, rank=0),
            _node("nodata", None, known=False, rank=0)])
        assert pos["nodata"][1] > pos["low"][1] > pos["high"][1]


def _chain(prefix, heights, rank_from=0):
    """A cascade: each feature spilling into the next, the last off the block."""
    out = []
    for i, elev in enumerate(heights):
        nid = f"{prefix}{i}"
        nxt = f"{prefix}{i + 1}" if i + 1 < len(heights) else None
        out.append(_node(nid, elev, rank=rank_from + i, target=nxt))
    return out


def _bounds(positions, ids):
    xs = [positions[i][0] for i in ids]
    ys = [positions[i][1] for i in ids]
    return (min(xs), max(xs) + _NODE_W, min(ys), max(ys) + _NODE_H)


def _overlap(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


class TestCascadeGrouping:
    """Features that flow into each other are drawn near each other.

    The old layout put every feature on one grid, so two chains interleaved in
    height had their boxes interleaved too and the connectors crossed a dozen
    links they had nothing to do with. Each cascade now gets a block of its own.
    """

    def test_two_cascades_do_not_overlap_on_the_page(self):
        nodes = _chain("a", [240.0, 238.0, 236.0, 234.0]) + \
            _chain("b", [239.0, 237.0, 235.0, 233.0])
        pos = report_charts._layout(nodes)
        a = _bounds(pos, ["a0", "a1", "a2", "a3"])
        b = _bounds(pos, ["b0", "b1", "b2", "b3"])
        assert not _overlap(a, b), (a, b)

    def test_height_order_survives_inside_a_cascade(self):
        """The trade the grouping makes: heights compare within a chain, not
        between them. Within one, the page must still run downhill."""
        nodes = _chain("a", [240.0, 238.0, 236.0]) + \
            _chain("b", [239.0, 231.0, 230.0])
        pos = report_charts._layout(nodes)
        assert pos["a0"][1] < pos["a1"][1] < pos["a2"][1]
        assert pos["b0"][1] < pos["b1"][1] < pos["b2"][1]

    def test_a_single_cascade_is_laid_out_exactly_as_before(self):
        """Grouping must be invisible on a scheme with nothing to group."""
        nodes = _chain("a", [240.0, 238.0, 236.0, 234.0])
        scale = report_charts._elevation_scale(nodes)
        assert report_charts._layout(nodes) == \
            report_charts._layout_by_elevation(nodes, scale)

    def test_unlinked_features_share_one_block(self):
        """A feature with no link either way is not a cascade. Giving each its
        own block would scatter a scheme that has not been wired up yet."""
        nodes = [_node("a", 240.0), _node("b", 230.0), _node("c", 220.0)]
        chains, loose = report_charts._components(nodes)
        assert chains == []
        assert [n["id"] for n in loose] == ["a", "b", "c"]
        scale = report_charts._elevation_scale(nodes)
        assert report_charts._layout(nodes) == \
            report_charts._layout_by_elevation(nodes, scale)

    def test_a_cascade_and_a_loose_feature_are_separate_blocks(self):
        nodes = _chain("a", [240.0, 238.0, 236.0]) + [_node("solo", 237.0)]
        keys = report_charts._component_keys(nodes)
        assert keys["a0"] == keys["a1"] == keys["a2"]
        assert keys["solo"] == report_charts._LOOSE
        pos = report_charts._layout(nodes)
        assert not _overlap(_bounds(pos, ["a0", "a1", "a2"]),
                            _bounds(pos, ["solo"]))

    def test_a_ring_is_one_component_and_still_terminates(self):
        """``resolve_targets`` breaks rings before the report sees them, but the
        renderer is handed hand-built data too, so it must not loop here."""
        nodes = [_node("a", 240.0, rank=0, target="b"),
                 _node("b", 230.0, rank=1, target="a")]
        keys = report_charts._component_keys(nodes)
        assert keys["a"] == keys["b"] != report_charts._LOOSE
        assert set(report_charts._layout(nodes)) == {"a", "b"}

    def test_a_self_link_does_not_hang(self):
        nodes = [_node("a", 240.0, target="a"), _node("b", 230.0)]
        assert set(report_charts._layout(nodes)) == {"a", "b"}

    def test_the_exaggeration_is_spent_where_it_buys_printed_size(self):
        """Banding costs height, and height is what the page shrinks the figure
        by. The ladder trades exaggeration for printed size, so a many-chain
        scheme must come out larger than the same grouping drawn at the full
        exaggeration would."""
        nodes = []
        for c in range(8):
            nodes += _chain(f"c{c}_",
                            [240.0 - c * 6.0 - i * 1.8 for i in range(6)])

        def fit_of(positions):
            width = max(x for x, _ in positions.values()) + _NODE_W
            height = max(y for _, y in positions.values()) + _NODE_H
            return min(report_charts._PAGE_W_MM / width,
                       report_charts._PAGE_H_MM / height)

        laddered = fit_of(report_charts._layout(nodes))
        ladder = report_charts._EXAGGERATIONS
        try:
            report_charts._EXAGGERATIONS = (1.0,)
            unspent = fit_of(report_charts._layout(nodes))
        finally:
            report_charts._EXAGGERATIONS = ladder
        assert laddered > unspent, (laddered, unspent)

    def test_the_exaggeration_is_not_spent_when_it_buys_nothing(self):
        """Two short chains fit either way, so the fall stays drawn to scale
        rather than being flattened for a page that had room for it."""
        nodes = _chain("a", [240.0, 230.0]) + _chain("b", [220.0, 200.0])
        pos = report_charts._layout(nodes)
        assert pos["a1"][1] - pos["a0"][1] > _NODE_H
        assert pos["b1"][1] - pos["b0"][1] > _NODE_H


class TestEdgeRouting:
    """A straight line between boxes at different heights cuts across whatever
    lies between, and the boxes draw over the top of it — so a link ran into one
    feature and out the other side of another it had nothing to do with."""

    def _crosses(self, route, box):
        """Does any segment of the route pass through this box?"""
        bx, by = box
        for (x1, y1), (x2, y2) in zip(route, route[1:]):
            lo_x, hi_x = min(x1, x2), max(x1, x2)
            lo_y, hi_y = min(y1, y2), max(y1, y2)
            if (lo_x < bx + _NODE_W and hi_x > bx
                    and lo_y < by + _NODE_H and hi_y > by):
                return True
        return False

    def test_a_level_clear_link_stays_straight(self):
        """Corners would imply a detour that is not there."""
        assert report_charts._edge_route((38.0, 7.0), (50.0, 7.0), ()) == [
            (38.0, 7.0), (50.0, 7.0)]

    def test_a_link_across_heights_turns_in_the_channel(self):
        route = report_charts._edge_route((38.0, 7.0), (50.0, 60.0), ())
        assert len(route) == 4
        # One vertical run, and it sits in the gap between the two columns.
        assert route[1][0] == route[2][0]
        assert 38.0 < route[1][0] < 50.0

    def test_it_does_not_pass_through_a_box_in_the_way(self):
        blocker = (0.0, 30.0)          # sits on the source's own row, to the left
        route = report_charts._edge_route((38.0, 33.0), (50.0, 90.0), (blocker,))
        assert not self._crosses(route, blocker)

    def test_a_skip_with_both_channels_blocked_goes_below(self):
        blockers = ((50.0, 5.0), (50.0, 40.0), (50.0, 75.0))
        route = report_charts._edge_route((38.0, 10.0), (100.0, 80.0), blockers)
        assert all(not self._crosses(route, b) for b in blockers), route

    def test_a_cycle_still_produces_a_route(self):
        """resolve_targets warns about rings rather than rejecting them."""
        route = report_charts._edge_route((88.0, 60.0), (0.0, 10.0), ())
        assert route[0] == (88.0, 60.0) and route[-1] == (0.0, 10.0)


class TestSharedChannels:
    def test_two_links_into_one_column_are_separated(self):
        """Superimposed, they print as a single line — so the diagram shows one
        inflow where there are two."""
        nodes = [_node("a", 220.0, rank=0, target="c"),
                 _node("b", 210.0, rank=0, target="c"),
                 _node("c", 200.0, rank=1)]
        pos = report_charts._layout(nodes)
        stagger = report_charts._channel_stagger(nodes, pos)
        assert stagger["a"] != stagger["b"]
        assert report_charts._edge_route((38.0, 7.0), (50.0, 60.0), (),
                                         stagger["a"])[1][0] != \
            report_charts._edge_route((38.0, 20.0), (50.0, 60.0), (),
                                      stagger["b"])[1][0]

    def test_a_lone_link_is_not_nudged_off_centre(self):
        nodes = [_node("a", 220.0, rank=0, target="b"), _node("b", 200.0, rank=1)]
        pos = report_charts._layout(nodes)
        assert report_charts._channel_stagger(nodes, pos)["a"] == 0.0

    def test_the_stagger_stays_inside_the_channel(self):
        """Nudged too far it would put a line through the box it is avoiding."""
        assert 2 * report_charts._CHANNEL_STAGGER_MM < report_charts._COL_GAP


class TestVolumeLabels:
    def test_a_label_avoids_a_row_that_is_already_busy(self):
        route = [(0.0, 0.0), (10.0, 0.0), (10.0, 100.0), (20.0, 100.0)]
        busy = (0.0, 50.0 - _NODE_H / 2.0)     # straddles the vertical's midpoint
        x, y = report_charts._label_point(route, (busy,))
        assert x == 10.0
        assert not (busy[1] <= y <= busy[1] + _NODE_H)

    def test_it_labels_the_vertical_not_the_corner(self):
        """The corner is the one part of the line the reader is following."""
        route = [(0.0, 0.0), (10.0, 0.0), (10.0, 100.0), (20.0, 100.0)]
        x, y = report_charts._label_point(route, ())
        assert (x, y) == (10.0, 50.0)


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
