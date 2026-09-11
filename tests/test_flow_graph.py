"""Tests for modules/flow_graph — flow pointers, catchment labelling, cascade order."""

import math

import numpy as np
import pytest

from terrainflow_assessment.modules.flow_graph import (
    LABEL_EXIT,
    LABEL_NONE,
    LABEL_UNRESOLVED,
    d8_from_dem,
    label_direct_catchments,
    longest_flow_path,
    resolve_terminals,
    topological_order,
    walk_downslope,
)


def _tilted(rows=10, cols=8, fall=1.0):
    """Plane falling south: z decreases as row increases, so flow is straight down."""
    return np.fromfunction(lambda r, c: 100.0 - r * fall, (rows, cols)).astype("float64")


class TestD8FromDem:
    def test_tilted_plane_flows_straight_downhill(self):
        z = _tilted(6, 5)
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        nxt2d = nxt.reshape(z.shape)
        # Every interior cell drains to the cell directly below it.
        for r in range(5):
            for c in range(5):
                assert nxt2d[r, c] == (r + 1) * 5 + c
        # The bottom row has nowhere lower to go.
        assert is_sink.reshape(z.shape)[5].all()
        assert not is_sink.reshape(z.shape)[:5].any()

    def test_diagonals_are_distance_weighted(self):
        # Straight neighbour drops 1.0 over 1 m; diagonal drops 1.2 over 1.414 m.
        # Raw drop favours the diagonal, slope favours the straight cell.
        z = np.array([
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [8.8, 9.0, 10.0],
        ])
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        assert nxt.reshape(z.shape)[1, 1] == 2 * 3 + 1  # (2,1), the straight drop

    def test_single_pit_is_a_sink(self):
        z = np.full((5, 5), 10.0)
        z[2, 2] = 5.0
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        assert is_sink.reshape(z.shape)[2, 2]
        assert nxt.reshape(z.shape)[2, 1] == 2 * 5 + 2

    def test_nodata_cells_neither_drain_nor_receive(self):
        z = _tilted(5, 5)
        z[2, 2] = -9999.0
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0, nodata=-9999.0)
        nxt2d = nxt.reshape(z.shape)
        assert is_sink.reshape(z.shape)[2, 2]           # nodata drains nowhere
        assert nxt2d[1, 2] != 2 * 5 + 2                 # and nothing drains into it

    def test_without_nodata_a_hole_becomes_a_capturing_pit(self):
        """Why a caller must supply nodata when the raster declares none.

        Unmasked, the sentinel is just an elevation — ten kilometres down. The hole
        stops being excluded and starts collecting the catchment around it, which
        is how an interior gap in a clipped tile silently steals its neighbours'
        drainage. Counterexample to the test above; the two differ only in the
        ``nodata=`` argument.
        """
        z = _tilted(5, 5)
        z[2, 2] = -9999.0
        nxt, _ = d8_from_dem(z, 1.0, 1.0)               # nodata not passed
        assert nxt.reshape(z.shape)[1, 2] == 2 * 5 + 2  # neighbour drains into it

    def test_flat_dem_is_all_sinks(self):
        nxt, is_sink = d8_from_dem(np.full((4, 4), 7.0), 1.0, 1.0)
        assert is_sink.all()
        assert (nxt == np.arange(16)).all()

    def test_every_path_strictly_descends_and_terminates(self):
        """Property test: steepest descent can never cycle, on any surface."""
        rng = np.random.default_rng(20260728)
        z = rng.normal(50.0, 5.0, size=(64, 64))
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        flat = z.ravel()

        for start in rng.integers(0, flat.size, size=500):
            cur = int(start)
            for _ in range(flat.size + 1):
                nxt_cell = int(nxt[cur])
                if nxt_cell == cur:
                    break
                assert flat[nxt_cell] < flat[cur]   # strictly downhill, so no revisits
                cur = nxt_cell
            else:
                raise AssertionError("walk did not terminate — the graph has a cycle")


class TestResolveTerminals:
    def test_chain_resolves_to_its_terminal(self):
        # 0 → 1 → 2 → 3(terminal, label 7)
        nxt = np.array([1, 2, 3, 3], dtype=np.int32)
        term = np.array([LABEL_NONE, LABEL_NONE, LABEL_NONE, 7], dtype=np.int32)
        labels, iters = resolve_terminals(nxt, term)
        assert labels.tolist() == [7, 7, 7, 7]
        assert iters >= 1

    def test_nearest_terminal_wins_in_series(self):
        # 0 → 1(terminal 5) → 2 → 3(terminal 9): cell 0 stops at the first one.
        nxt = np.array([1, 2, 3, 3], dtype=np.int32)
        term = np.array([LABEL_NONE, 5, LABEL_NONE, 9], dtype=np.int32)
        labels, _ = resolve_terminals(nxt, term)
        assert labels.tolist() == [5, 5, 9, 9]

    def test_cycle_reports_unresolved_without_hanging(self):
        # 0 ↔ 1 is a cycle; 2 → 3 → terminal.
        nxt = np.array([1, 0, 3, 3], dtype=np.int32)
        term = np.array([LABEL_NONE, LABEL_NONE, LABEL_NONE, 0], dtype=np.int32)
        labels, _ = resolve_terminals(nxt, term)
        assert labels[0] == LABEL_UNRESOLVED
        assert labels[1] == LABEL_UNRESOLVED
        assert labels[2] == 0


class TestLabelDirectCatchments:
    def _setup(self, interceptor_rows, rows=10, cols=8):
        z = _tilted(rows, cols)
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((rows, cols), LABEL_NONE, dtype=np.int32)
        for label, r in enumerate(interceptor_rows):
            inter[r, :] = label
        domain = np.ones((rows, cols), dtype=bool)
        return label_direct_catchments(nxt, inter, domain, is_sink=is_sink)

    def test_single_interceptor_takes_everything_above_it(self):
        res = self._setup([5])
        assert res.counts.tolist() == [6 * 8]        # rows 0-5 inclusive
        assert res.exit_cells == 4 * 8               # rows 6-9 run off the bottom
        assert res.sink_cells == 0
        assert res.unresolved_cells == 0

    def test_stacked_features_split_the_hillside(self):
        """The lower feature gets only the band between them — never the shared upslope."""
        res = self._setup([3, 7])
        assert res.counts.tolist() == [4 * 8, 4 * 8]  # rows 0-3 and rows 4-7
        assert res.exit_cells == 2 * 8

    def test_exhaustive_and_mutually_exclusive(self):
        res = self._setup([2, 6])
        assert res.is_exhaustive()
        assert res.accounted == res.domain_cells == 80
        # Every domain cell carries exactly one label, and labels never overlap.
        assert set(np.unique(res.labels).tolist()) <= {0, 1, LABEL_EXIT}

    def test_random_surface_still_closes(self):
        rng = np.random.default_rng(7)
        z = rng.normal(50.0, 3.0, size=(48, 48))
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((48, 48), LABEL_NONE, dtype=np.int32)
        inter[12, 5:40] = 0
        inter[30, 5:40] = 1
        inter[20:24, 20:24] = 2
        domain = np.ones((48, 48), dtype=bool)
        res = label_direct_catchments(nxt, inter, domain, is_sink=is_sink)
        assert res.is_exhaustive()
        assert res.unresolved_cells == 0

    def test_cells_outside_the_domain_are_not_counted(self):
        rows, cols = 10, 8
        z = _tilted(rows, cols)
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((rows, cols), LABEL_NONE, dtype=np.int32)
        inter[5, :] = 0
        domain = np.zeros((rows, cols), dtype=bool)
        domain[:, 2:6] = True                        # a 4-column strip of site
        res = label_direct_catchments(nxt, inter, domain, is_sink=is_sink)
        assert res.domain_cells == 10 * 4
        assert res.is_exhaustive()
        assert (res.labels[:, 0] == LABEL_NONE).all()

    def test_interior_pit_is_reported_separately_from_exit(self):
        z = np.full((9, 9), 20.0)
        z[4, 4] = 5.0                                # an interior nodata-style hole
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((9, 9), LABEL_NONE, dtype=np.int32)
        domain = np.ones((9, 9), dtype=bool)
        res = label_direct_catchments(nxt, inter, domain, is_sink=is_sink)
        assert res.sink_cells >= 1                   # the pit and what drains to it
        assert res.is_exhaustive()

    def test_rain_on_the_feature_counts_to_that_feature(self):
        res = self._setup([0])                       # interceptor on the top row
        assert res.counts.tolist() == [8]            # exactly its own cells


class TestWalkDownslope:
    def test_finds_the_next_feature_downslope(self):
        rows, cols = 10, 6
        z = _tilted(rows, cols)
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((rows, cols), LABEL_NONE, dtype=np.int32)
        inter[2, :] = 0
        inter[7, :] = 1
        label, steps = walk_downslope(nxt, 2 * cols + 3, inter.ravel(), skip_label=0)
        assert label == 1
        assert steps == 5

    def test_returns_none_when_water_leaves_the_site(self):
        rows, cols = 6, 5
        z = _tilted(rows, cols)
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((rows, cols), LABEL_NONE, dtype=np.int32)
        inter[1, :] = 0
        label, _ = walk_downslope(nxt, 1 * cols + 2, inter.ravel(), skip_label=0)
        assert label is None

    def test_a_cycle_is_bounded_not_infinite(self):
        nxt = np.array([1, 0], dtype=np.int32)
        inter = np.array([LABEL_NONE, LABEL_NONE], dtype=np.int32)
        label, steps = walk_downslope(nxt, 0, inter)
        assert label is None
        assert steps <= 2


class TestTopologicalOrder:
    def test_chain_orders_upstream_first(self):
        order, broken = topological_order({"a": "b", "b": "c", "c": None})
        assert order == ["a", "b", "c"]
        assert broken == []

    def test_fan_in_places_the_receiver_last(self):
        order, broken = topological_order({"a": "c", "b": "c", "c": None})
        assert order.index("c") == 2
        assert broken == []

    def test_cycle_is_reported_and_still_ordered(self):
        order, broken = topological_order({"a": "b", "b": "a", "c": None})
        assert set(broken) == {"a", "b"}
        assert set(order) == {"a", "b", "c"}
        assert order.index("c") < order.index("a")   # acyclic work drains first

    def test_self_link_is_not_a_cycle(self):
        order, broken = topological_order({"a": "a"})
        assert order == ["a"]
        assert broken == []

    def test_target_outside_the_graph_is_ignored(self):
        order, broken = topological_order({"a": "ghost"})
        assert order == ["a"]
        assert broken == []


class TestLongestFlowPath:
    """Travel distance sets the time of concentration, and therefore the design
    intensity, and therefore every spillway width downstream of it. It is the flow
    path that counts, not the straight line: a long shallow draw takes far longer to
    drain than its crow-flies length suggests."""

    @staticmethod
    def _plane(rows=20, cols=20):
        """A plane falling south — the longest path is one column, top to bottom."""
        return np.tile(np.arange(rows, 0, -1, dtype="float32").reshape(-1, 1), (1, cols))

    def test_a_tilted_plane_measures_its_full_fall(self):
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        length, visited = longest_flow_path(nxt, np.ones(dem.shape, bool), 20, 1.0, 1.0)
        assert length == pytest.approx(19.0)
        assert visited == 400                      # every cell was reached

    def test_diagonal_steps_cost_root_two(self):
        """A staircase path must not be measured as if it ran along the axes."""
        yy, xx = np.mgrid[0:20, 0:20]
        dem = (40 - (yy + xx)).astype("float32")
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        length, _ = longest_flow_path(nxt, np.ones(dem.shape, bool), 20, 1.0, 1.0)
        assert length == pytest.approx(19.0 * math.sqrt(2.0))

    def test_non_square_cells_are_measured_in_metres(self):
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        length, _ = longest_flow_path(nxt, np.ones(dem.shape, bool), 20, 2.0, 5.0)
        assert length == pytest.approx(19.0 * 5.0)

    def test_only_cells_inside_the_mask_are_counted(self):
        """A feature's Tc depends on its own catchment, not the whole site."""
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        mask = np.zeros(dem.shape, bool)
        mask[5:11, :] = True                       # six rows → five steps
        length, visited = longest_flow_path(nxt, mask, 20, 1.0, 1.0)
        assert length == pytest.approx(5.0)
        assert visited == mask.sum()

    def test_an_empty_mask_is_zero_not_an_error(self):
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        assert longest_flow_path(nxt, np.zeros(dem.shape, bool), 20) == (0.0, 0)

    def test_a_single_cell_has_no_travel_distance(self):
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        mask = np.zeros(dem.shape, bool)
        mask[3, 3] = True
        assert longest_flow_path(nxt, mask, 20, 1.0, 1.0) == (0.0, 1)

    def test_the_longest_branch_wins_not_the_last_one(self):
        """Two tributaries joining: the answer is the longer, not whichever the
        traversal happened to reach last."""
        dem = np.full((10, 10), 100.0, dtype="float32")
        dem[:, 5] = np.arange(10, 0, -1)           # long spine down column 5
        dem[8, :5] = np.linspace(3.0, 2.0, 5)      # short side branch into it
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        mask = np.isfinite(dem)
        length, _ = longest_flow_path(nxt, mask, 10, 1.0, 1.0)
        spine, _ = longest_flow_path(nxt, _column_mask(dem.shape, 5), 10, 1.0, 1.0)
        assert length >= spine

    def test_trace_returns_the_path_itself(self):
        """Needed so each TR-55 leg can take its slope from the real elevation
        profile rather than one average over the whole catchment."""
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        length, _visited, path = longest_flow_path(
            nxt, np.ones(dem.shape, bool), 20, 1.0, 1.0, trace=True)
        assert len(path) == 20                     # 20 cells, 19 steps
        assert length == pytest.approx(19.0)

    def test_the_path_runs_ridge_to_outlet(self):
        """Direction matters: the sheet-flow leg is the first 30 m *from the ridge*,
        so a reversed path would put it in the valley."""
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        _l, _v, path = longest_flow_path(
            nxt, np.ones(dem.shape, bool), 20, 1.0, 1.0, trace=True)
        flat = dem.ravel()
        assert flat[path[0]] > flat[path[-1]]
        assert all(flat[a] > flat[b] for a, b in zip(path, path[1:]))

    def test_the_traced_path_length_matches_the_reported_length(self):
        yy, xx = np.mgrid[0:20, 0:20]
        dem = (40 - (yy + xx)).astype("float32")
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        length, _v, path = longest_flow_path(
            nxt, np.ones(dem.shape, bool), 20, 1.0, 1.0, trace=True)
        walked = 0.0
        for a, b in zip(path, path[1:]):
            dr, dc = abs(b // 20 - a // 20), abs(b % 20 - a % 20)
            walked += math.sqrt(2.0) if (dr == 1 and dc == 1) else 1.0
        assert walked == pytest.approx(length)

    def test_trace_stays_inside_the_mask(self):
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        mask = np.zeros(dem.shape, bool)
        mask[5:11, :] = True
        _l, _v, path = longest_flow_path(nxt, mask, 20, 1.0, 1.0, trace=True)
        flat_mask = mask.ravel()
        assert all(flat_mask[p] for p in path)

    def test_trace_on_an_empty_mask_is_an_empty_path(self):
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        assert longest_flow_path(
            nxt, np.zeros(dem.shape, bool), 20, trace=True) == (0.0, 0, [])

    def test_untraced_calls_keep_the_two_value_shape(self):
        """Existing callers must not have to change."""
        dem = self._plane()
        nxt, _ = d8_from_dem(dem, 1.0, 1.0)
        assert len(longest_flow_path(nxt, np.ones(dem.shape, bool), 20, 1.0, 1.0)) == 2

    def test_a_cycle_under_reports_rather_than_hanging(self):
        """Conditioned DEMs are acyclic, but a malformed pointer array must cost a
        bounded traversal — the visited count is how the caller detects it."""
        nxt = np.array([1, 0, 3, 2], dtype=np.int64)   # two 2-cycles
        mask = np.ones(4, dtype=bool)
        length, visited = longest_flow_path(nxt, mask, 2, 1.0, 1.0)
        assert visited < 4
        assert length >= 0.0


def _column_mask(shape, col):
    mask = np.zeros(shape, bool)
    mask[:, col] = True
    return mask


class TestOverflowLinkAcceptance:
    """The rule the Route Overflow tool applies to a user's two clicks.

    A proposed link is accepted unless adding it closes a loop. That is a physical
    constraint rather than a solver limitation: water cannot overflow back into
    what is already feeding it. Only user-set links appear in this graph —
    auto-resolved targets follow real downslope paths on a conditioned DEM and are
    acyclic by construction.
    """

    @staticmethod
    def _accepts(existing, source, target):
        edges = dict(existing)
        edges[source] = target
        _order, broken = topological_order(edges)
        return not broken

    def test_a_link_further_downstream_is_accepted(self):
        assert self._accepts({"a": "b"}, "b", "c")

    def test_closing_a_two_feature_loop_is_refused(self):
        assert not self._accepts({"a": "b"}, "b", "a")

    def test_closing_a_longer_loop_is_refused(self):
        assert not self._accepts({"a": "b", "b": "c"}, "c", "a")

    def test_retargeting_an_existing_link_is_accepted(self):
        """Re-linking replaces the old target rather than adding a second edge,
        so a feature that used to point back is no longer a loop."""
        assert self._accepts({"a": "b", "b": "a"}, "b", "c")

    def test_two_features_may_share_one_receiver(self):
        assert self._accepts({"a": "c"}, "b", "c")


class TestGuards:
    """Defensive branches: bad input must fail loudly or degrade safely, never silently."""

    def test_non_2d_dem_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="2-D"):
            d8_from_dem(np.arange(10.0), 1.0, 1.0)

    def test_zero_cell_size_yields_no_flow(self):
        # Degenerate geotransform: no distance means no computable slope.
        _, is_sink = d8_from_dem(_tilted(4, 4), 0.0, 0.0)
        assert is_sink.all()

    def test_shape_mismatch_rejected(self):
        import pytest
        nxt, _ = d8_from_dem(_tilted(5, 5), 1.0, 1.0)
        inter = np.full((5, 5), LABEL_NONE, dtype=np.int32)
        with pytest.raises(ValueError, match="same shape"):
            label_direct_catchments(nxt, inter, np.ones((4, 4), dtype=bool))

    def test_sink_mask_is_derived_when_not_supplied(self):
        z = _tilted(8, 6)
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        inter = np.full((8, 6), LABEL_NONE, dtype=np.int32)
        inter[3, :] = 0
        domain = np.ones((8, 6), dtype=bool)
        derived = label_direct_catchments(nxt, inter, domain)
        supplied = label_direct_catchments(nxt, inter, domain, is_sink=is_sink)
        assert derived.counts.tolist() == supplied.counts.tolist()
        assert derived.exit_cells == supplied.exit_cells

    def test_walk_from_an_out_of_range_start_is_a_no_op(self):
        nxt = np.array([1, 1], dtype=np.int32)
        assert walk_downslope(nxt, 99, np.array([LABEL_NONE, LABEL_NONE])) == (None, 0)

    def test_walk_respects_max_steps(self):
        # A 50-cell chain, cut short at 5 steps.
        nxt = np.arange(1, 51, dtype=np.int32)
        nxt = np.append(nxt, 50).astype(np.int32)
        inter = np.full(51, LABEL_NONE, dtype=np.int32)
        inter[40] = 3
        label, steps = walk_downslope(nxt, 0, inter, max_steps=5)
        assert label is None
        assert steps == 5

    def test_unresolved_cells_are_counted_not_hidden(self):
        # A hand-built pointer graph with a cycle that no terminal can drain.
        nxt = np.array([1, 0, 3, 3], dtype=np.int32)
        inter = np.array([[LABEL_NONE, LABEL_NONE], [LABEL_NONE, 0]], dtype=np.int32)
        domain = np.ones((2, 2), dtype=bool)
        res = label_direct_catchments(nxt, inter, domain,
                                      is_sink=np.array([False, False, False, True]))
        assert res.unresolved_cells == 2
        assert res.is_exhaustive()


class TestTieBreak:
    """The tie-break was undocumented and untested, and it decides level-crest routing.

    ``best`` starts at 0.0 with a strict ``>``, so a level neighbour is never chosen and,
    among equal slopes, the earliest offset in ``_OFFSETS`` wins. On a resolved flat the
    synthetic gradient is an integer multiple of a small epsilon, so exact ties are the
    normal case rather than a corner one — and before Round 14 either the scan order or the
    comparison operator could have been changed with the whole suite staying green.
    """

    def test_an_exact_tie_goes_to_the_first_offset_scanned(self):
        # A symmetric pit: the centre cell's eight neighbours are all exactly 1.0 lower,
        # so every cardinal is tied and every diagonal is tied. _OFFSETS is scanned
        # NW, N, NE, W, E, SW, S, SE, and slope de-weights diagonals by sqrt(2), so the
        # first *cardinal* wins outright: N.
        z = np.full((3, 3), 10.0)
        z[1, 1] = 11.0
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        assert nxt.reshape(3, 3)[1, 1] == 0 * 3 + 1        # (0, 1) — due north

    def test_a_diagonal_only_tie_goes_to_north_west(self):
        # Knock out every cardinal so only the four diagonals are candidates. NW is first.
        z = np.full((3, 3), 12.0)
        z[1, 1] = 11.0
        for r, c in ((0, 0), (0, 2), (2, 0), (2, 2)):
            z[r, c] = 10.0
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        assert nxt.reshape(3, 3)[1, 1] == 0 * 3 + 0        # (0, 0) — north-west

    def test_a_level_neighbour_is_never_chosen(self):
        # The `> 0.0` rule, stated as a test: equal ground is not downhill, which is why a
        # flat has to be resolved upstream before these pointers mean anything.
        z = np.array([[5.0, 5.0, 5.0],
                      [5.0, 5.0, 5.0],
                      [5.0, 5.0, 4.0]])
        nxt, is_sink = d8_from_dem(z, 1.0, 1.0)
        # Only the cells touching the one low corner drain; the rest are sinks.
        assert is_sink.reshape(3, 3)[0, 0]
        assert not is_sink.reshape(3, 3)[1, 1]
        assert nxt.reshape(3, 3)[1, 1] == 2 * 3 + 2

    def test_the_scan_order_is_the_documented_one(self):
        from terrainflow_assessment.modules.flow_graph import _OFFSETS

        assert _OFFSETS == ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                            (0, 1), (1, -1), (1, 0), (1, 1))


# ---------------------------------------------------------------------------
# Strahler ordering and link extraction
# ---------------------------------------------------------------------------

class TestStrahlerOrder:
    @staticmethod
    def _chain(cells, cols):
        """Pointers for a hand-built network: cells is {flat_idx: downstream_flat_idx}."""
        n = max(max(cells), max(cells.values())) + 1
        nxt = np.arange(n, dtype=np.int32)
        for src, dst in cells.items():
            nxt[src] = dst
        mask = np.zeros(n, dtype=bool)
        mask[list(cells)] = True
        mask[list(cells.values())] = True
        return nxt, mask

    def test_a_single_chain_is_order_one_throughout(self):
        from terrainflow_assessment.modules.flow_graph import strahler_order

        nxt, mask = self._chain({0: 1, 1: 2, 2: 3}, cols=1)
        order = strahler_order(nxt, mask)
        assert list(order[:4]) == [1, 1, 1, 1]

    def test_two_equal_orders_promote_where_they_meet(self):
        """Strahler's rule: 1 + 1 = 2, and it stays 2 downstream."""
        from terrainflow_assessment.modules.flow_graph import strahler_order

        #  0 -> 2 <- 1 ;  2 -> 3
        nxt, mask = self._chain({0: 2, 1: 2, 2: 3}, cols=1)
        order = strahler_order(nxt, mask)
        assert order[0] == 1 and order[1] == 1
        assert order[2] == 2, "two order-1 links meeting must promote"
        assert order[3] == 2

    def test_unequal_orders_do_not_promote(self):
        from terrainflow_assessment.modules.flow_graph import strahler_order

        # A 2 (from 0,1 meeting at 2) joined by a 1 (from 4) at 3.
        nxt, mask = self._chain({0: 2, 1: 2, 2: 3, 4: 3, 3: 5}, cols=1)
        order = strahler_order(nxt, mask)
        assert order[2] == 2 and order[4] == 1
        assert order[3] == 2, "a smaller tributary must not promote the trunk"

    def test_non_channel_cells_are_zero(self):
        from terrainflow_assessment.modules.flow_graph import strahler_order

        nxt = np.arange(6, dtype=np.int32)
        nxt[0] = 1
        mask = np.zeros(6, dtype=bool)
        mask[[0, 1]] = True
        order = strahler_order(nxt, mask)
        assert list(order[2:]) == [0, 0, 0, 0]

    def test_an_empty_network_is_all_zero(self):
        from terrainflow_assessment.modules.flow_graph import strahler_order

        order = strahler_order(np.arange(5, dtype=np.int32), np.zeros(5, dtype=bool))
        assert not order.any()


class TestStreamLinks:
    def test_a_headwater_link_comes_back_ordered_top_down(self):
        """The profile needs the cells in order; that is half of why links exist."""
        from terrainflow_assessment.modules.flow_graph import (
            strahler_order,
            stream_links,
        )

        cols = 1
        nxt = np.arange(6, dtype=np.int32)
        for src, dst in {0: 1, 1: 2, 2: 3, 3: 4}.items():
            nxt[src] = dst
        mask = np.zeros(6, dtype=bool)
        mask[:5] = True

        order = strahler_order(nxt, mask)
        links = stream_links(nxt, mask, order, cols, max_order=1, min_cells=3)
        assert len(links) == 1
        rows = [r for r, _c in links[0]]
        assert rows == sorted(rows), "the link must run from the top down"

    def test_a_junction_splits_two_primary_valleys(self):
        """Two order-1 tributaries are two valleys, not one."""
        from terrainflow_assessment.modules.flow_graph import (
            strahler_order,
            stream_links,
        )

        # Two chains of three meeting at 6, which drains to 7.
        nxt = np.arange(9, dtype=np.int32)
        for src, dst in {0: 1, 1: 2, 2: 6, 3: 4, 4: 5, 5: 6, 6: 7}.items():
            nxt[src] = dst
        mask = np.zeros(9, dtype=bool)
        mask[:8] = True

        order = strahler_order(nxt, mask)
        links = stream_links(nxt, mask, order, cols=1, max_order=1, min_cells=3)
        assert len(links) == 2, f"expected two primary valleys, got {len(links)}"

    def test_short_stubs_are_dropped(self):
        """A two-cell link has no profile to take a second derivative of."""
        from terrainflow_assessment.modules.flow_graph import (
            strahler_order,
            stream_links,
        )

        nxt = np.arange(4, dtype=np.int32)
        nxt[0] = 1
        mask = np.zeros(4, dtype=bool)
        mask[[0, 1]] = True
        order = strahler_order(nxt, mask)
        assert stream_links(nxt, mask, order, cols=1, min_cells=3) == []

    def test_the_trunk_is_not_a_primary_valley(self):
        """max_order=1 excludes what two tributaries make when they meet."""
        from terrainflow_assessment.modules.flow_graph import (
            strahler_order,
            stream_links,
        )

        nxt = np.arange(12, dtype=np.int32)
        for src, dst in {0: 1, 1: 2, 2: 6, 3: 4, 4: 5, 5: 6,
                         6: 7, 7: 8, 8: 9}.items():
            nxt[src] = dst
        mask = np.zeros(12, dtype=bool)
        mask[:10] = True

        order = strahler_order(nxt, mask)
        assert order[7] == 2
        links = stream_links(nxt, mask, order, cols=1, max_order=1, min_cells=3)
        flat = {r for link in links for r, _c in link}
        assert 7 not in flat, "the order-2 trunk was returned as a primary valley"


# ---------------------------------------------------------------------------
# The graph's own accumulation, the walk to the divide, and the data edge (KPA-52)
# ---------------------------------------------------------------------------

class TestAccumulate:
    def test_a_tilted_plane_counts_the_rows_above(self):
        from terrainflow_assessment.modules.flow_graph import accumulate

        z = _tilted(6, 4)
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        acc = accumulate(nxt).reshape(z.shape)
        for r in range(6):
            assert (acc[r] == r + 1).all(), acc

    def test_a_y_network_sums_its_tributaries(self):
        from terrainflow_assessment.modules.flow_graph import accumulate

        nxt = np.arange(4, dtype=np.int32)
        nxt[0], nxt[1], nxt[2] = 2, 2, 3
        assert list(accumulate(nxt)) == [1, 1, 3, 4]

    def test_a_cycle_keeps_a_partial_count_rather_than_hanging(self):
        from terrainflow_assessment.modules.flow_graph import accumulate

        nxt = np.array([1, 0, 0], dtype=np.int32)   # 0 <-> 1, fed by 2
        acc = accumulate(nxt)
        assert acc[2] == 1
        # 2 hands its count to 0; 0 and 1 never see their last inflow arrive.
        assert acc[0] == 2 and acc[1] == 1

    def test_invalid_cells_weigh_nothing(self):
        from terrainflow_assessment.modules.flow_graph import accumulate

        nxt = np.array([1, 2, 2], dtype=np.int32)
        valid = np.array([False, True, True])
        assert list(accumulate(nxt, valid)) == [0, 1, 2]

    def test_agrees_with_an_elevation_sorted_pass_on_real_pointers(self):
        """Two routes to one number: Kahn's levels against a high-to-low sweep."""
        from terrainflow_assessment.modules.flow_graph import accumulate

        rng = np.random.default_rng(7)
        z = _tilted(30, 25, fall=0.5) + rng.normal(0.0, 0.05, (30, 25))
        z += 0.3 * np.abs(np.arange(25) - 12)          # a valley down the middle
        nxt, _ = d8_from_dem(z, 1.0, 1.0)
        # Valid because a pointer always goes strictly lower.
        expect = np.ones(z.size, dtype=np.int64)
        for i in np.argsort(-z.ravel(), kind="stable"):
            j = int(nxt[i])
            if j != i:
                expect[j] += expect[i]
        assert np.array_equal(accumulate(nxt), expect)


class TestMainStemToDivide:
    def test_a_single_chain_returns_everything_above_the_head_top_down(self):
        from terrainflow_assessment.modules.flow_graph import main_stem_to_divide

        nxt = np.array([1, 2, 3, 4, 4], dtype=np.int32)      # a 5x1 column
        acc = np.array([1, 2, 3, 4, 5])
        assert main_stem_to_divide(nxt, acc, head_flat=3, cols=1) == [0, 1, 2]

    def test_the_larger_tributary_is_the_stem(self):
        from terrainflow_assessment.modules.flow_graph import (
            accumulate,
            main_stem_to_divide,
        )

        # 3x3: (0,0)=0 and (0,2)=2 both drain into the head (1,1)=4; 0 is fed by (1,0)=3.
        nxt = np.arange(9, dtype=np.int32)
        nxt[0], nxt[2], nxt[3] = 4, 4, 0
        acc = accumulate(nxt)
        assert acc[0] == 2 and acc[2] == 1
        assert main_stem_to_divide(nxt, acc, 4, cols=3) == [3, 0]

    def test_an_exact_tie_goes_to_the_first_offset_scanned(self):
        """Pinned like d8_from_dem's tie-break: NW is scanned before NE."""
        from terrainflow_assessment.modules.flow_graph import (
            accumulate,
            main_stem_to_divide,
        )

        nxt = np.arange(9, dtype=np.int32)
        nxt[0], nxt[2] = 4, 4
        assert main_stem_to_divide(nxt, accumulate(nxt), 4, cols=3) == [0]

    def test_a_head_that_is_a_divide_gives_nothing(self):
        from terrainflow_assessment.modules.flow_graph import (
            accumulate,
            main_stem_to_divide,
        )

        nxt = np.arange(9, dtype=np.int32)
        nxt[4] = 7
        assert main_stem_to_divide(nxt, accumulate(nxt), 4, cols=3) == []

    def test_the_walk_stops_at_the_allowed_mask(self):
        from terrainflow_assessment.modules.flow_graph import (
            accumulate,
            main_stem_to_divide,
        )

        nxt = np.array([1, 2, 3, 4, 4], dtype=np.int32)
        allowed = np.array([False, True, True, True, True])
        assert main_stem_to_divide(nxt, accumulate(nxt), 3, cols=1,
                                   allowed_flat=allowed) == [1, 2]

    def test_a_cycle_terminates(self):
        from terrainflow_assessment.modules.flow_graph import (
            accumulate,
            main_stem_to_divide,
        )

        nxt = np.array([1, 0, 0], dtype=np.int32)             # 0 <-> 1 in a 3x1 column
        path = main_stem_to_divide(nxt, accumulate(nxt), 0, cols=1)
        assert path == [1]


class TestDataBoundaryMask:
    def test_the_outer_ring_is_the_boundary_of_a_full_grid(self):
        from terrainflow_assessment.modules.flow_graph import data_boundary_mask

        m = data_boundary_mask(np.ones((4, 5), dtype=bool))
        assert m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all()
        assert not m[1:-1, 1:-1].any()

    def test_ground_beside_a_hole_is_boundary_and_the_hole_is_not(self):
        from terrainflow_assessment.modules.flow_graph import data_boundary_mask

        valid = np.ones((7, 7), dtype=bool)
        valid[3, 3] = False
        m = data_boundary_mask(valid)
        assert m[2:5, 2:5].sum() == 8 and not m[3, 3]
        assert not m[1, 1] and not m[5, 5]
