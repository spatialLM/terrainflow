"""
Tests for terrainflow_assessment/modules/crest_routing.py.

The module is pure — it takes the caller's accumulation as a callback — so everything here
runs against a fifteen-line D8 accumulator defined below rather than against pysheds. That
is only honest if the toy engine behaves like the real one on the cases under test, so
``TestToyAccumulator`` pins the two properties the module actually leans on: linearity, and
that a self-looping cell absorbs its whole upstream and passes nothing on.

The end-to-end behaviour on real pysheds fields lives in
``tests/test_flow_analysis_assessment.py::TestCrestSplit``.
"""
import numpy as np
import pytest

from terrainflow_assessment.modules.crest_routing import (
    MIN_POND_CELLS,
    CrestPlan,
    Impoundment,
    find_impoundments,
    plan_crest_absorption,
    spread_crests,
)

# ---------------------------------------------------------------------------
# A toy routing engine: D8 pointers, and accumulation over them
# ---------------------------------------------------------------------------

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _d8(z):
    """Steepest-descent pointers over *z*. A cell with no lower neighbour points at itself."""
    z = np.asarray(z, dtype="float64")
    rows, cols = z.shape
    idx = np.arange(rows * cols).reshape(rows, cols)
    nxt = idx.copy()
    best = np.zeros_like(z)
    for dr, dc in _OFFSETS:
        dist = float(np.hypot(dr, dc))
        r0, r1 = max(0, -dr), rows - max(0, dr)
        c0, c1 = max(0, -dc), cols - max(0, dc)
        src = (slice(r0, r1), slice(c0, c1))
        nbr = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
        drop = (z[src] - z[nbr]) / dist
        better = drop > best[src]
        best[src] = np.where(better, drop, best[src])
        nxt[src] = np.where(better, idx[nbr], nxt[src])
    flat = nxt.ravel()
    return flat, flat == np.arange(rows * cols)


def _accumulator(next_flat, is_sink, absorb, shape):
    """``accumulate(weights)`` over the pointers, with *absorb* cells self-looping.

    Topological, so it is exact rather than iterative: a cell is released once every cell
    pointing at it has been released. A self-looping cell is never released, which is what
    makes it a terminal — everything upstream arrives and nothing leaves.
    """
    n = shape[0] * shape[1]
    idx = np.arange(n)
    tgt = np.where(np.asarray(is_sink).ravel() | np.asarray(absorb).ravel(),
                   idx, np.asarray(next_flat).ravel())

    def accumulate(weights):
        acc = (np.ones(n, dtype="float64") if weights is None
               else np.asarray(weights, dtype="float64").ravel().copy())
        indeg = np.bincount(tgt[tgt != idx], minlength=n)
        ready = [int(i) for i in np.flatnonzero(indeg == 0)]
        while ready:
            i = ready.pop()
            d = int(tgt[i])
            if d == i:
                continue
            acc[d] += acc[i]
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
        return acc.reshape(shape)

    return accumulate


def _terminal_total(next_flat, is_sink, absorb, acc):
    """Weight that has reached a **real** terminal — a cell that passes nothing on.

    Pond cells are deliberately left out even though they also self-loop. Their value is the
    running total of everything that ever arrived, and on a chain the same water arrives at
    several ponds in turn, so counting them would count it more than once. The ponds' share
    of the ledger is the residual instead.
    """
    flat = np.asarray(acc, dtype="float64").ravel()
    idx = np.arange(flat.size)
    held = np.asarray(absorb).ravel()
    tgt = np.where(np.asarray(is_sink).ravel() | held, idx, np.asarray(next_flat).ravel())
    return float(flat[(tgt == idx) & ~held].sum())


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------

WALL = 100.0


def _walled(*bands, width=3):
    """A walled trough: each band is one row of *width* interior cells, outlet row at 0."""
    out = [[WALL] * (width + 2)]
    out += [[WALL] + [float(v)] * width + [WALL] for v in bands]
    out += [[0.0] * (width + 2)]
    return np.array(out, dtype="float64")


def _staircase():
    """Two ponds chained **through open ground** — the surface the parked version lost on.

    Pond A's pool (12) fills to its crest at 15; below the crest sits a row of ordinary
    hillside at 14, and only *then* pond B's pool at 8. So no cell of A discharges into a
    cell of B, and a pond graph built from the immediate D8 neighbour of an exit sees no edge
    between them at all — while in fact every drop A sheds arrives in B.
    """
    return _walled(20, 12, 12, 12, 15, 14, 8, 8, 8, 10, 5)


def _touching():
    """The same chain with the gap row removed, so A's crest sits straight on B's pool."""
    return _walled(20, 12, 12, 12, 15, 8, 8, 8, 10, 5)


def _condition(z):
    """Fill the hollows in a walled trough, the way a priority flood would.

    Small and explicit rather than a call into pysheds. Every interior row of a trough is
    uniform, so the surface is a one-dimensional profile and the fill level at each row is
    the lower of the two highest barriers it would have to cross to escape — upward past the
    wall, or downward to the outlet.
    """
    filled = np.array(z, dtype="float64", copy=True)
    profile = filled[:, 1]
    up = np.maximum.accumulate(profile)
    down = np.maximum.accumulate(profile[::-1])[::-1]
    level = np.maximum(profile, np.minimum(up, down))
    filled[1:-1, 1:-1] = level[1:-1].reshape(-1, 1)
    return filled


def _plan_for(z, **kw):
    ground = np.array(z, dtype="float64", copy=True)
    filled = _condition(z)
    # A pond's rim is a flat, so the pointers need a surface where the flat still drains.
    # Tilting the filled surface by a whisper per row is what resolve_flats does for real.
    tilt = filled - np.arange(filled.shape[0], dtype="float64").reshape(-1, 1) * 1e-4
    next_flat, is_sink = _d8(tilt)
    imps, skipped = find_impoundments(filled, ground, **kw)
    plan = plan_crest_absorption(imps, next_flat, is_sink, filled.shape, skipped=skipped)
    return plan, next_flat, is_sink, filled, ground


def _spread(z, max_passes=None, weights=None, capacities=None, **kw):
    plan, next_flat, is_sink, filled, _ = _plan_for(z, **kw)
    accumulate = _accumulator(next_flat, is_sink, plan.absorb, filled.shape)
    out = spread_crests(plan, accumulate, base_weights=weights, max_passes=max_passes,
                        capacities=capacities)
    return plan, out, accumulate, (next_flat, is_sink)


# ---------------------------------------------------------------------------


class TestToyAccumulator:
    """The pure tests are only worth anything if the toy engine behaves like the real one."""

    def test_a_self_looping_cell_holds_its_upstream_and_passes_nothing_on(self):
        z = np.fromfunction(lambda r, c: 100.0 - r, (5, 4))   # a plane falling south
        next_flat, is_sink = _d8(z)
        absorb = np.zeros(z.shape, dtype=bool)
        absorb[3, :] = True
        acc = _accumulator(next_flat, is_sink, absorb, z.shape)(None)
        # Rows 0-2 drain straight into row 3, which absorbs: four cells of four rows each.
        assert acc[3].sum() == pytest.approx(16.0)
        assert acc[4].sum() == pytest.approx(4.0)

    def test_accumulation_is_linear_in_its_weights(self):
        z = _walled(5, 4, 3, 2, 1)
        next_flat, is_sink = _d8(z)
        acc = _accumulator(next_flat, is_sink, np.zeros(z.shape, bool), z.shape)
        rng = np.random.default_rng(3)
        w1 = rng.random(z.shape)
        w2 = rng.random(z.shape)
        assert np.allclose(acc(w1 + w2), acc(w1) + acc(w2))

    def test_every_unit_of_weight_ends_on_a_terminal(self):
        z = _staircase()
        next_flat, is_sink = _d8(_condition(z))
        absorb = np.zeros(z.shape, dtype=bool)
        acc = _accumulator(next_flat, is_sink, absorb, z.shape)(None)
        assert _terminal_total(next_flat, is_sink, absorb, acc) == pytest.approx(z.size)


class TestFindImpoundments:
    """Which hollows count as ponds, and how far up their wall the routing region reaches."""

    def test_a_dry_surface_has_no_ponds(self):
        z = _walled(5, 4, 3, 2, 1)
        assert find_impoundments(z, z) == ([], [])

    def test_the_staircase_finds_both_pools_and_their_crests(self):
        filled = _condition(_staircase())
        imps, skipped = find_impoundments(filled, _staircase())
        assert skipped == []
        assert [imp.pour_level_m for imp in imps] == [15.0, 10.0]
        # Three pool rows plus the one crest row standing at the pour level.
        assert [imp.n_cells for imp in imps] == [12, 12]

    def test_the_whole_level_band_joins_the_region_not_just_the_pool_facing_row(self):
        """A wall two cells thick is two rows of crest, and only the far one can discharge.

        Round 14 measured this on Dam 15: the pool-facing row is 24 cells of which 8 can pass
        water, while the full level band is 65 of which 49 can. Growing the rim through the
        wall is what makes an even spread reachable at all.
        """
        z = _walled(20, 12, 12, 12, 15, 15, 5)
        filled = _condition(z)
        imps, _ = find_impoundments(filled, z)
        assert len(imps) == 1
        assert imps[0].n_cells == 3 * 3 + 2 * 3   # pool rows plus both rows of the wall

    def test_a_pool_below_the_floor_keeps_the_default_routing(self):
        """Below a 3x3 pool there is nothing to spread along, and the median hollow on a real
        DEM is 3 cells of fill_depressions noise — see MIN_POND_CELLS."""
        for cells, expected in ((MIN_POND_CELLS - 1, 0), (MIN_POND_CELLS, 1)):
            ground = np.full((7, 4 + MIN_POND_CELLS), 20.0)
            ground[3, 2:2 + cells] = 10.0
            filled = ground.copy()
            filled[3, 2:2 + cells] = 20.0
            imps, _ = find_impoundments(filled, ground)
            assert len(imps) == expected

    def test_a_pond_over_the_size_cap_is_skipped_and_says_so(self):
        filled = _condition(_staircase())
        imps, skipped = find_impoundments(filled, _staircase(), max_cells=4)
        assert imps == []
        assert len(skipped) == 2
        assert "keeps the default routing" in skipped[0]

    def test_the_built_gate_keeps_only_ponds_against_raised_ground(self):
        z = _staircase()
        filled = _condition(z)
        built = np.zeros(z.shape, dtype=bool)
        built[5, 1:4] = True            # pond A's crest row only
        imps, _ = find_impoundments(filled, z, built=built)
        assert [imp.pour_level_m for imp in imps] == [15.0]

    def test_a_wrong_shaped_built_mask_is_ignored_rather_than_trusted(self):
        z = _staircase()
        filled = _condition(z)
        imps, _ = find_impoundments(filled, z, built=np.zeros((3, 3), dtype=bool))
        assert len(imps) == 2

    def test_a_puddle_shallower_than_min_depth_is_not_a_pond(self):
        ground = np.full((7, 14), 20.0)
        ground[3, 2:11] = 20.0 - 1e-6
        filled = np.full((7, 14), 20.0)
        assert find_impoundments(filled, ground) == ([], [])

    def test_the_rim_grows_past_a_window_that_first_looks_big_enough(self):
        """The rim is worked out inside a box around the pool and the box doubles while the
        rim still runs off its edge. A level band far longer than the pool exercises that."""
        ground = np.full((5, 40), 30.0)
        ground[2, :] = 20.0
        ground[2, 18:28] = 10.0          # a 10-cell pool in a 40-cell level band
        filled = np.full((5, 40), 30.0)
        filled[2, :] = 20.0
        imps, _ = find_impoundments(filled, ground)
        assert len(imps) == 1
        # The whole 20.0 band is one region with the pool, right across the grid — many
        # times wider than the box the rim search starts from.
        assert imps[0].n_cells == 40


class TestPlanCrestAbsorption:
    """Which cells absorb, which discharge, and the shape of the pond cascade."""

    def test_no_ponds_means_nothing_absorbs(self):
        plan = plan_crest_absorption([], np.zeros(9, int), np.ones(9, bool), (3, 3))
        assert plan.ponds == 0
        assert not plan.changed
        assert not plan.absorb.any()

    def test_each_pond_discharges_over_its_own_crest(self):
        plan, *_ = _plan_for(_staircase())
        assert plan.ponds == 2
        assert [int(e.size) for e in plan.exits] == [3, 3]
        assert plan.outlet_cells == 6
        assert plan.changed

    def test_the_hand_off_is_invisible_to_the_pointer_graph(self):
        """The bug, stated as an assertion.

        Not one of pond A's exits points at a cell of pond B, so a pond-to-pond graph read
        off ``rid[tgt[exit]]`` — which is what the parked version used — contains no edge
        between them. The water gets there all the same.
        """
        plan, *_ = _plan_for(_staircase())
        assert all(plan.rid[t] == 0 for t in plan.targets[0])

    def test_a_pond_with_nowhere_to_discharge_keeps_the_default_routing(self):
        """A bowl whose level band reaches the edge of the tile has no cell outside itself to
        spill into. Contracting it would swallow the whole domain, so it is left alone and
        the reason is recorded."""
        ground = np.full((9, 9), 50.0)
        ground[2:7, 2:7] = 10.0
        filled = np.full((9, 9), 50.0)
        next_flat, is_sink = _d8(filled)
        imps, skipped = find_impoundments(filled, ground)
        assert len(imps) == 1
        plan = plan_crest_absorption(imps, next_flat, is_sink, filled.shape, skipped=skipped)
        assert plan.ponds == 0
        assert not plan.absorb.any()
        assert "nothing discharges" in plan.skipped[0]

    def test_the_cascade_depth_counts_the_links_not_the_ponds(self):
        plan, *_ = _plan_for(_staircase())
        assert plan.cascade_depth >= 1
        assert plan.default_passes() >= 16

    def test_two_ponds_that_drain_into_each_other_do_not_recurse_forever(self):
        """Terrain cannot produce this — a spill leaves at the pour level and can only reach
        a pond below it — but the guard is what stops a malformed pointer graph hanging the
        analysis thread, so it is exercised directly."""
        from terrainflow_assessment.modules.crest_routing import _cascade_depth

        # Two one-cell ponds pointing at each other.
        rid = np.array([1, 2], dtype=np.int32)
        tgt = np.array([1, 0], dtype=np.int64)
        depth = _cascade_depth(rid, tgt, [np.array([0]), np.array([1])],
                              [np.array([1]), np.array([0])])
        assert depth >= 0

    def test_the_pass_budget_grows_with_the_cascade(self):
        def _plan(depth):
            return CrestPlan(np.zeros((2, 2), bool), np.zeros((2, 2), bool),
                             np.zeros(4, np.int32), [np.array([0])], [np.array([1])],
                             (2, 2), depth, [])

        assert _plan(1).default_passes() == 16
        assert _plan(40).default_passes() == 64      # capped, not 2 * 40 + 4

    def test_the_pool_is_kept_apart_from_the_wall(self):
        """A wall is not a water body, and the two masks are used for different things."""
        plan, *_ = _plan_for(_staircase())
        assert plan.pools.sum() < plan.absorb.sum()
        assert not (plan.pools & ~plan.absorb).any()   # every pool cell is in a region
        # Pond A: three rows of pool at 12, and one crest row at 15 that is not water.
        assert int(plan.pools.sum()) == 2 * 9
        assert int(plan.absorb.sum()) == 2 * 12


class TestSpreadCrests:
    """The contraction itself: an even crest, and every unit of water still accounted for."""

    def test_a_crest_sheds_the_same_flux_at_every_cell(self):
        plan, out, *_ = _spread(_staircase())
        crest_a = out.outlets[5, 1:4]
        crest_b = out.outlets[10, 1:4]
        assert crest_a.min() == crest_a.max()
        assert crest_b.min() == crest_b.max()
        assert out.outlet_cells == 6

    def test_the_downstream_row_is_even_where_it_used_to_be_concentrated(self):
        """Nearest-outlet routing hands the inflow channel to whichever crest cell is closest
        to where it arrives; contracting the pond makes the wall a weir.

        How lopsided "before" is depends on the tie-break, so only the direction is asserted
        here — the real ratio (36,781x down to 1.00x on Dam 15) is measured against pysheds
        in ``TestCrestSplit``.
        """
        plan, next_flat, is_sink, filled, _ = _plan_for(_staircase())
        plain = _accumulator(next_flat, is_sink, np.zeros(filled.shape, bool), filled.shape)
        before = plain(None)[-1, 1:4]
        _, out, *_ = _spread(_staircase())
        after = out.accumulation[-1, 1:4]
        assert before.max() > before.min()                # concentrated
        assert after.min() == pytest.approx(after.max())  # a weir

    def test_water_is_conserved_through_a_chain_the_pointer_graph_cannot_see(self):
        z = _staircase()
        plan, out, accumulate, (next_flat, is_sink) = _spread(z)
        assert out.residual == pytest.approx(0.0)
        total = _terminal_total(next_flat, is_sink, plan.absorb, out.accumulation)
        assert total == pytest.approx(float(z.size))

    def test_one_emission_round_is_not_enough_and_says_how_much_it_is_short(self):
        """The parked version was this loop stopped after a single round. On a chain it
        strands the upstream pond's whole inflow — and reports it, which is the difference."""
        z = _staircase()
        plan, out, _, (next_flat, is_sink) = _spread(z, max_passes=2)
        assert out.residual > 0
        total = _terminal_total(next_flat, is_sink, plan.absorb, out.accumulation)
        assert total + out.residual == pytest.approx(float(z.size))
        assert total < float(z.size)      # short, and short in the safe direction

    def test_a_direct_hand_off_is_carried_too(self):
        z = _touching()
        plan, out, _, (next_flat, is_sink) = _spread(z)
        assert any(plan.rid[t] != 0 for t in plan.targets[0])   # A spills onto B's pool
        assert out.residual == pytest.approx(0.0)
        total = _terminal_total(next_flat, is_sink, plan.absorb, out.accumulation)
        assert total == pytest.approx(float(z.size))

    def test_the_ledger_closes_at_every_truncation_point(self):
        """The conservation identity is not contingent on convergence — it holds at each cap,
        because the loop only ever under-emits."""
        z = _staircase()
        for cap in (2, 3, 4, 8):
            plan, out, _, (next_flat, is_sink) = _spread(z, max_passes=cap)
            total = _terminal_total(next_flat, is_sink, plan.absorb, out.accumulation)
            assert total + out.residual == pytest.approx(float(z.size)), cap

    def test_nothing_ever_goes_negative(self):
        """What killed the attempt before contraction: redistributing among the cells already
        leaving is not a clean cut, and it drove 64 cells negative on the real site."""
        _, out, *_ = _spread(_staircase())
        assert out.accumulation.min() >= 0.0

    def test_no_ponds_leaves_the_field_exactly_as_it_was(self):
        z = _walled(5, 4, 3, 2, 1)
        plan, out, accumulate, _ = _spread(z)
        assert out.ponds == 0
        assert not out.changed
        assert out.passes == 1
        assert np.array_equal(out.accumulation, accumulate(None))

    def test_doubling_the_rain_doubles_every_crest_share(self):
        z = _staircase()
        _, single, *_ = _spread(z, weights=np.ones(z.shape))
        _, double, *_ = _spread(z, weights=np.full(z.shape, 2.0))
        assert np.allclose(double.outlets, 2.0 * single.outlets)

    def test_a_pond_that_swallows_its_own_emission_stops_rather_than_hanging(self):
        """A surface that recirculates cannot arise from terrain — a spill leaves at the pour
        level and can only reach a pond below it — but the budget must still bite."""
        plan, next_flat, is_sink, filled, _ = _plan_for(_staircase())

        def never_leaves(weights):
            """Everything injected lands straight back in the ponds."""
            out = np.zeros(filled.shape, dtype="float64")
            total = 1.0 if weights is None else float(np.sum(weights))
            out[plan.absorb] = total / max(int(plan.absorb.sum()), 1)
            return out

        out = spread_crests(plan, never_leaves, max_passes=5)
        assert out.passes == 5
        assert out.residual > 0

    def test_the_pool_carries_the_pond_throughput_and_the_wall_carries_nothing(self):
        """Inside a pool the accumulation stops meaning contributing area — it is what
        arrived at that cell and stopped. Anything reading it as catchment size gets this
        raster instead, so a reservoir floor is not mistaken for a hilltop."""
        z = _staircase()
        plan, out, *_ = _spread(z)
        pool_a = out.pond_flow[2:5, 1:4]
        assert (pool_a == pool_a[0, 0]).all()          # one value across the whole pool
        assert pool_a[0, 0] > 0
        assert (out.pond_flow[5, 1:4] == 0).all()      # the crest row is wall, not water
        # Downstream pond carries its own catchment plus everything the upstream one sheds.
        assert out.pond_flow[7, 1] > pool_a[0, 0]
        assert np.array_equal(out.pond_mask, plan.pools)

    def test_no_ponds_means_an_empty_pond_raster(self):
        _, out, *_ = _spread(_walled(5, 4, 3, 2, 1))
        assert not out.pond_flow.any()
        assert not out.pond_mask.any()

    def test_the_skipped_reasons_reach_the_caller(self):
        plan, out, *_ = _spread(_staircase(), max_cells=4)
        assert out.ponds == 0
        assert len(out.skipped) == 2


class TestPondStorage:
    """What a hollow holds before it spills — measured off the surface, not off a design.

    This is the whole reason retention needs no idea what an earthwork is: the volume comes
    from Σ(filled − ground) over the pool, exactly as ``reporting.raster_ponding_volume``
    measures it and exactly as the panel's At-grid column reports it. A swale, a dam's pool
    and a natural hollow are the same object here.
    """

    def test_a_pool_holds_its_own_depth_times_its_own_area(self):
        z = _staircase()
        imps, _ = find_impoundments(_condition(z), z)
        # Pond A: three rows of three cells standing at 15 over ground at 12.
        assert imps[0].storage_m3 == pytest.approx(3 * 3 * 3.0)
        # Pond B: three rows of three at 10 over ground at 8.
        assert imps[1].storage_m3 == pytest.approx(3 * 3 * 2.0)

    def test_storage_scales_with_the_cell(self):
        z = _staircase()
        imps, _ = find_impoundments(_condition(z), z, cell_area_m2=4.0)
        assert imps[0].storage_m3 == pytest.approx(27.0 * 4.0)

    def test_the_rim_is_not_storage(self):
        """The level band through the wall routes water; it does not hold any. Counting it
        would credit the pond with the volume of its own bank."""
        z = _walled(20, 12, 12, 12, 15, 15, 5)
        imps, _ = find_impoundments(_condition(z), z)
        assert imps[0].n_cells == 15                     # pool plus two rows of wall
        assert imps[0].storage_m3 == pytest.approx(3 * 3 * 3.0)   # pool only

    def test_water_above_the_pour_level_is_not_counted(self):
        """Two hollows touching on a diagonal are one component but two pour levels, and the
        pond is treated as spilling at the lower. Storage measured against each cell's own
        filled value would credit it with water it has already released."""
        ground = np.full((9, 9), 20.0)
        ground[2:5, 2:5] = 10.0
        ground[5:8, 5:8] = 10.0
        filled = ground.copy()
        filled[2:5, 2:5] = 16.0        # upper sub-pool
        filled[5:8, 5:8] = 14.0        # lower sub-pool, and the level this pond spills at
        imps, skipped = find_impoundments(filled, ground, min_cells=1)
        assert len(imps) == 1
        assert imps[0].pour_level_m == pytest.approx(14.0)
        assert any("pour level" in s for s in skipped)
        # 18 cells of 4 m each, not 9 of 6 plus 9 of 4.
        assert imps[0].storage_m3 == pytest.approx(18 * 4.0)

    def test_the_plan_carries_them_aligned_with_the_exits(self):
        plan, *_ = _plan_for(_staircase())
        assert len(plan.capacities) == plan.ponds
        assert list(plan.capacities) == pytest.approx([27.0, 18.0])

    def test_a_pond_that_drops_out_takes_its_capacity_with_it(self):
        """Ids are rebuilt contiguously when a pond keeps the default routing, so the
        capacities must be rebuilt with them or every pond after it is capped by its
        neighbour's volume."""
        z = _walled(20, 12, 12, 12, 15, 14, 8, 8, 8, 10, 5)
        plan, *_ = _plan_for(z, min_cells=1)
        assert len(plan.capacities) == plan.ponds
        assert all(c > 0 for c in plan.capacities)


class TestRetention:
    """A pond fills before it spills, so the field downstream is the water that gets there.

    Without this the raster showed what would pass each cell if every hollow held nothing —
    a swale reading 46% full still drew a full channel leaving its pour point, which is the
    map contradicting the panel beside it.
    """

    def _ledger(self, z, **kw):
        plan, out, _, (next_flat, is_sink) = _spread(z, **kw)
        total = _terminal_total(next_flat, is_sink, plan.absorb, out.accumulation)
        return plan, out, total

    def test_retention_is_off_unless_asked_for(self):
        """The engine's default field is a cell count, and capping a count with a volume is
        a category error — so a caller that passes nothing gets exactly the old behaviour."""
        _, was, *_ = _spread(_staircase())
        _, now, *_ = _spread(_staircase(), capacities=None)
        assert np.array_equal(was.accumulation, now.accumulation)
        assert now.retained == 0.0

    def test_a_pond_that_can_hold_its_inflow_passes_nothing_on(self):
        z = _staircase()
        plan, out, total = self._ledger(z, capacities=[1e9, 1e9])
        assert not out.outlets.any()                  # nothing crossed either crest
        assert out.retained > 0
        # Only the ground below pond B reaches a terminal — everything above is held.
        assert total < float(z.size)
        assert total + out.retained + out.residual == pytest.approx(float(z.size))

    def test_a_full_pond_passes_the_excess_and_only_the_excess(self):
        z = _staircase()
        _, unheld, *_ = _spread(z)
        _, out, *_ = _spread(z, capacities=[5.0, 0.0])
        # Pond A kept 5 units; its crest sheds 5 fewer, and pond B is uncapped so all of
        # that shortfall carries straight through to the bottom.
        assert out.outlets.sum() == pytest.approx(unheld.outlets.sum() - 5.0 * 2)
        assert out.retained == pytest.approx(5.0)

    def test_the_ledger_closes_with_retention_in_it(self):
        """Retained water is a terminal like any other. It is *not* residual — that is water
        the loop ran out of passes to place, and conflating them would hide a truncation."""
        z = _staircase()
        for caps in ([0.0, 0.0], [5.0, 3.0], [1e9, 0.0], [0.0, 1e9], [1e9, 1e9]):
            plan, out, total = self._ledger(z, capacities=caps)
            assert total + out.retained + out.residual == pytest.approx(float(z.size)), caps
            assert out.accumulation.min() >= 0.0, caps

    def test_the_ledger_closes_at_every_truncation_point_too(self):
        z = _staircase()
        for cap in (2, 3, 4, 8):
            plan, out, total = self._ledger(z, capacities=[5.0, 3.0], max_passes=cap)
            assert total + out.retained + out.residual == pytest.approx(float(z.size)), cap

    def test_a_pond_holds_only_up_to_its_capacity_across_the_whole_cascade(self):
        """Headroom is drawn down pass by pass, so a pond filled by an early arrival must
        pass on what reaches it later. Resolving it up front would let a pond on a chain
        keep its capacity once per pass."""
        z = _staircase()
        _, out, *_ = _spread(z, capacities=[3.0, 4.0])
        assert out.retained == pytest.approx(7.0)
        assert out.retained_by_pond == pytest.approx([3.0, 4.0])

    def test_an_upstream_pond_filling_starves_the_one_below_it(self):
        z = _staircase()
        _, unheld, *_ = _spread(z)
        _, held, *_ = _spread(z, capacities=[1e9, 0.0])
        # Pond B is uncapped in both runs, so all of the difference is water pond A kept.
        assert held.pond_flow[7, 1] < unheld.pond_flow[7, 1]

    def test_the_pool_still_reports_what_arrived_not_what_left(self):
        """``pond_flow`` is read as a catchment-size proxy (``keypoint_analysis``), so it has
        to stay the arriving figure. Switching it to the surplus would make a pond that
        captures everything look like it drains nothing."""
        z = _staircase()
        _, unheld, *_ = _spread(z)
        _, held, *_ = _spread(z, capacities=[1e9, 1e9])
        assert held.pond_flow[2, 1] == pytest.approx(unheld.pond_flow[2, 1])

    def test_doubling_the_rain_does_not_double_what_a_pond_can_hold(self):
        """The retention is a volume, not a fraction — which is the whole point of it. A
        storm twice the size overflows a pond by more than twice as much."""
        z = _staircase()
        _, single, *_ = _spread(z, weights=np.ones(z.shape), capacities=[5.0, 1e9])
        _, double, *_ = _spread(z, weights=np.full(z.shape, 2.0), capacities=[5.0, 1e9])
        assert double.outlets.sum() > 2.0 * single.outlets.sum()
        # Pond A's own 5 m³, not the total: pond B is uncapped here and swallows whatever
        # reaches it, so its share of the retention does scale with the storm.
        assert single.retained_by_pond[0] == pytest.approx(double.retained_by_pond[0])
        assert single.retained_by_pond[0] == pytest.approx(5.0)

    def test_spreading_the_same_plan_twice_does_not_drain_its_headroom(self):
        """``FlowAnalysis.run`` spreads one plan twice — once plain, once weighted — so a
        ``remaining`` that aliased ``plan.capacities`` would leave the second pass with
        ponds already full, and the field would silently stop retaining."""
        plan, next_flat, is_sink, filled, _ = _plan_for(_staircase())
        accumulate = _accumulator(next_flat, is_sink, plan.absorb, filled.shape)
        first = spread_crests(plan, accumulate, capacities=plan.capacities)
        second = spread_crests(plan, accumulate, capacities=plan.capacities)
        assert first.retained == pytest.approx(second.retained)
        assert first.retained > 0
        assert list(plan.capacities) == pytest.approx([27.0, 18.0])

    def test_a_capacity_per_pond_is_required(self):
        z = _staircase()
        with pytest.raises(ValueError, match="capacities"):
            _spread(z, capacities=[1.0])

    def test_a_negative_capacity_is_read_as_no_storage(self):
        z = _staircase()
        _, out, *_ = _spread(z, capacities=[-5.0, -5.0])
        _, plain, *_ = _spread(z)
        assert out.retained == 0.0
        assert np.allclose(out.accumulation, plain.accumulation)

    def test_no_ponds_means_nothing_to_retain(self):
        _, out, *_ = _spread(_walled(5, 4, 3, 2, 1), capacities=[])
        assert out.retained == 0.0
        assert not out.changed


class TestImpoundment:
    def test_it_counts_its_own_cells(self):
        region = np.zeros((4, 4), dtype=bool)
        region[1:3, 1:3] = True
        imp = Impoundment(region, 12.5)
        assert imp.n_cells == 4
        assert imp.pour_level_m == 12.5
        assert imp.storage_m3 == 0.0


class TestMergedPoolPourLevel:
    """Two depressions touching on a diagonal are one 8-connected component here
    while remaining two ponds on a filled surface, each flat at its own level."""

    def _two_pools_touching_diagonally(self):
        import numpy as np

        # Ground: a plateau with two hollows meeting at a corner.
        ground = np.full((7, 7), 20.0)
        ground[2, 2] = 8.0
        ground[3, 3] = 5.0
        # Filled: each hollow flat at its own pour level.
        filled = ground.copy()
        filled[2, 2] = 10.0
        filled[3, 3] = 12.0
        return ground, filled

    def test_the_pour_level_is_the_lower_of_the_two(self):
        from terrainflow_assessment.modules.crest_routing import find_impoundments

        ground, filled = self._two_pools_touching_diagonally()
        ponds, skipped = find_impoundments(filled, ground, min_cells=1)
        assert ponds, "no impoundment found"
        pour = ponds[0].pour_level_m
        assert pour == 10.0, (
            f"pour level {pour} — the maximum sizes the rim for a level the "
            "lower sub-pool never reaches")

    def test_the_spread_is_reported_rather_than_hidden(self):
        from terrainflow_assessment.modules.crest_routing import find_impoundments

        ground, filled = self._two_pools_touching_diagonally()
        _, skipped = find_impoundments(filled, ground, min_cells=1)
        assert any("pour level" in note for note in skipped), (
            f"a two-level pond was merged silently: {skipped}")

    def test_one_level_pond_says_nothing(self):
        import numpy as np

        from terrainflow_assessment.modules.crest_routing import find_impoundments

        ground = np.full((7, 7), 20.0)
        ground[3, 3] = 5.0
        filled = ground.copy()
        filled[3, 3] = 12.0
        ponds, skipped = find_impoundments(filled, ground, min_cells=1)
        assert ponds and ponds[0].pour_level_m == 12.0
        assert not any("pour level" in note for note in skipped)
