"""Tests for modules/mass_haul — the balance, and the cost of moving.

The two claims worth pinning: a balance is not `cut - fill` (they are measured in
different states), and the greedy fallback is never better than the exact solve.
"""

import math

import numpy as np
import pytest
from rasterio.transform import from_origin

from terrainflow_assessment.modules.mass_haul import (
    _solve_greedy,
    _solve_lp,
    allocate_haul,
    earthwork_balance,
    haul_regions,
)

CELL = 1.0
N = 60
TRANSFORM = from_origin(0.0, N * CELL, CELL, CELL)


class TestEarthworkBalance:
    def test_equal_cut_and_fill_is_not_balanced(self):
        """The whole point. A compacted fill swallows more in-situ soil than its own
        volume, so `cut == fill` leaves the design short."""
        r = earthwork_balance(1000.0, 1000.0, "Loam")
        assert r["deficit_m3"] > 0
        assert r["surplus_m3"] == 0.0
        assert r["bank_needed_for_fill_m3"] > r["bank_fill_m3"]

    def test_surplus_and_deficit_are_mutually_exclusive(self):
        """Reporting both is how a reader ends up adding them together."""
        for cut, fill in ((2000.0, 100.0), (100.0, 2000.0), (0.0, 0.0)):
            r = earthwork_balance(cut, fill, "Loam")
            assert r["surplus_m3"] == 0.0 or r["deficit_m3"] == 0.0

    def test_loose_volume_exceeds_bank(self):
        """What a truck carries is not what came out of the hole."""
        r = earthwork_balance(1000.0, 0.0, "Clay")
        assert r["loose_from_cut_m3"] > r["bank_cut_m3"]

    def test_erodible_and_cohesive_soils_differ(self):
        sand = earthwork_balance(1000.0, 1000.0, "Sand")
        clay = earthwork_balance(1000.0, 1000.0, "Clay")
        assert sand["bank_needed_for_fill_m3"] != clay["bank_needed_for_fill_m3"]

    def test_factors_can_be_supplied_directly(self):
        r = earthwork_balance(100.0, 90.0, bulking=1.0, compaction=1.0)
        assert r["bank_needed_for_fill_m3"] == pytest.approx(90.0)
        assert r["surplus_m3"] == pytest.approx(10.0)

    def test_a_design_that_moves_nothing_is_balanced(self):
        r = earthwork_balance(0.0, 0.0, "Loam")
        assert r["surplus_m3"] == 0.0 and r["deficit_m3"] == 0.0

    def test_the_named_factor_replaces_two_anonymous_ones(self):
        """`berm_spoil_per_metre` and the burner both carried a bare 0.75, unsourced.

        The historical value stays available under a name so existing spoil figures do
        not move; the soil-keyed table is what the balance uses.
        """
        from terrainflow_assessment.core.sizing.advisories import (
            DEFAULT_COMPACTION,
            compaction_factor,
        )

        assert DEFAULT_COMPACTION == 0.75
        assert compaction_factor("Loam") != DEFAULT_COMPACTION


class TestHaulRegions:
    @staticmethod
    def _surfaces(cut_at, fill_at, depth=2.0, size=6):
        original = np.full((N, N), 50.0)
        burned = original.copy()
        r, c = cut_at
        burned[r:r + size, c:c + size] -= depth       # a cut
        r, c = fill_at
        burned[r:r + size, c:c + size] += depth       # a fill
        return original, burned

    def test_a_cut_and_a_fill_are_found_with_their_volumes(self):
        original, burned = self._surfaces((5, 5), (5, 40))
        cuts, fills = haul_regions(original, burned, TRANSFORM, CELL * CELL)
        assert len(cuts) == 1 and len(fills) == 1
        assert cuts[0]["volume_m3"] == pytest.approx(6 * 6 * 2.0)
        assert fills[0]["volume_m3"] == pytest.approx(6 * 6 * 2.0)

    def test_centroids_sit_where_the_earth_is(self):
        original, burned = self._surfaces((5, 5), (5, 40))
        cuts, fills = haul_regions(original, burned, TRANSFORM, CELL * CELL)
        assert fills[0]["x"] > cuts[0]["x"], "the fill is east of the cut"

    def test_nodata_holes_contribute_nothing_rather_than_nan(self):
        """A DEM clipped to a boundary always has holes, and one NaN used to turn a
        site total into `nan` and reach the report as a crash."""
        original, burned = self._surfaces((5, 5), (5, 40))
        burned[20, 20] = np.nan
        cuts, fills = haul_regions(original, burned, TRANSFORM, CELL * CELL)
        assert all(math.isfinite(c["volume_m3"]) for c in cuts + fills)

    def test_burn_noise_below_the_floor_is_dropped(self):
        original = np.full((N, N), 50.0)
        burned = original.copy()
        burned[3, 3] -= 0.01           # a single shallow cell
        cuts, _fills = haul_regions(original, burned, TRANSFORM, CELL * CELL)
        assert cuts == []

    def test_no_difference_gives_no_regions(self):
        flat = np.full((N, N), 50.0)
        assert haul_regions(flat, flat.copy(), TRANSFORM, CELL * CELL) == ([], [])


class TestAllocateHaul:
    def test_one_cut_to_one_fill_is_volume_times_distance(self):
        cuts = [{"x": 0.0, "y": 0.0, "volume_m3": 100.0}]
        fills = [{"x": 100.0, "y": 0.0, "volume_m3": 100.0}]
        plan = allocate_haul(cuts, fills)
        assert plan["matched_m3"] == pytest.approx(100.0)
        assert plan["haul_moment_m3m"] == pytest.approx(100.0 * 100.0)
        assert plan["mean_haul_m"] == pytest.approx(100.0)

    def test_the_exact_solve_is_never_beaten_by_greedy(self):
        """The claim that lets greedy be an honest fallback rather than a downgrade."""
        cuts = [{"x": 0.0, "y": 0.0, "volume_m3": 100.0},
                {"x": 90.0, "y": 0.0, "volume_m3": 100.0}]
        fills = [{"x": 10.0, "y": 0.0, "volume_m3": 100.0},
                 {"x": 100.0, "y": 0.0, "volume_m3": 100.0}]

        dist = [[math.hypot(c["x"] - f["x"], c["y"] - f["y"]) for f in fills]
                for c in cuts]
        supply = [c["volume_m3"] for c in cuts]
        demand = [f["volume_m3"] for f in fills]

        lp = _solve_lp(supply, demand, dist)
        greedy = _solve_greedy(supply, demand, dist)
        assert lp is not None, "scipy linprog should be available in this environment"

        lp_moment = sum(m["volume_m3"] * m["distance_m"] for m in lp)
        greedy_moment = sum(m["volume_m3"] * m["distance_m"] for m in greedy)
        assert lp_moment <= greedy_moment + 1e-6

    def test_greedy_is_used_and_named_when_asked_for(self):
        cuts = [{"x": 0.0, "y": 0.0, "volume_m3": 50.0}]
        fills = [{"x": 20.0, "y": 0.0, "volume_m3": 50.0}]
        plan = allocate_haul(cuts, fills, prefer_exact=False)
        assert "greedy" in plan["method"]

    def test_unmatched_volume_is_reported_not_dropped(self):
        cuts = [{"x": 0.0, "y": 0.0, "volume_m3": 500.0}]
        fills = [{"x": 50.0, "y": 0.0, "volume_m3": 100.0}]
        plan = allocate_haul(cuts, fills)
        assert plan["matched_m3"] == pytest.approx(100.0)
        assert plan["unmatched_cut_m3"] == pytest.approx(400.0)
        assert plan["unmatched_fill_m3"] == pytest.approx(0.0)

    def test_free_haul_and_overhaul_split_at_the_stated_distance(self):
        """A contract term, not a physical one — so it is an input."""
        cuts = [{"x": 0.0, "y": 0.0, "volume_m3": 100.0},
                {"x": 0.0, "y": 0.0, "volume_m3": 100.0}]
        fills = [{"x": 50.0, "y": 0.0, "volume_m3": 100.0},
                 {"x": 400.0, "y": 0.0, "volume_m3": 100.0}]
        plan = allocate_haul(cuts, fills, free_haul_m=150.0)
        assert plan["free_haul_m3"] == pytest.approx(100.0)
        assert plan["overhaul_m3m"] == pytest.approx(100.0 * (400.0 - 150.0))

    def test_nothing_to_move_is_not_a_division_by_zero(self):
        plan = allocate_haul([], [])
        assert plan["haul_moment_m3m"] == 0.0
        assert plan["mean_haul_m"] == 0.0
        assert plan["method"] == "nothing to move"

    def test_cut_with_no_fill_is_all_unmatched(self):
        plan = allocate_haul([{"x": 0.0, "y": 0.0, "volume_m3": 80.0}], [])
        assert plan["unmatched_cut_m3"] == pytest.approx(80.0)
        assert plan["moves"] == []


class TestEndToEnd:
    def test_a_cut_beside_a_fill_produces_a_short_haul(self):
        original = np.full((N, N), 50.0)
        burned = original.copy()
        burned[10:16, 10:16] -= 2.0
        burned[10:16, 20:26] += 2.0

        cuts, fills = haul_regions(original, burned, TRANSFORM, CELL * CELL)
        plan = allocate_haul(cuts, fills)
        assert plan["matched_m3"] == pytest.approx(72.0)
        # Centroids are ten metres apart.
        assert plan["mean_haul_m"] == pytest.approx(10.0, rel=0.2)
        assert plan["free_haul_m3"] == pytest.approx(72.0)
        assert plan["overhaul_m3m"] == 0.0
