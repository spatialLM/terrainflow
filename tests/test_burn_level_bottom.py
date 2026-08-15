"""Level-bottom burn regression — the basin/swale "says full, ponds a puddle" bug.

The old burn was ``dem[mask] -= depth``: a translation that preserved the original
ground slope, so the depression-filled pond was a wedge spilling at the lowest rim
rather than the prism the analytic capacity claimed. This suite pins the exact
gradient of that failure and asserts the level floor removes it.

Measured on a 400 m² × 1.5 m basin (600 m³ nominal prism):

    ground slope   translated burn   level floor
        0 %            600 m³           600 m³
        2 %            516 m³           600 m³
        5 %            390 m³           600 m³
       10 %            210 m³           600 m³

That gradient is what the Verify chip was reporting as a single confusing "Δ −38%".
"""

import numpy as np
import pytest

from terrainflow_assessment.modules.burn_strategy import (
    battered_invert,
    impoundment_warning,
    level_invert,
    rasterisable_capacity,
    steep_ground_warning,
)
from terrainflow_assessment.modules.footprint import pour_point

CELL = 1.0
ROWS = COLS = 60
BASIN = (slice(20, 40), slice(20, 40))     # 20 × 20 m = 400 m²
AREA_M2 = 400.0
DEPTH = 1.5
NOMINAL_M3 = AREA_M2 * DEPTH                # 600 m³


def _sloped(pct):
    """Plane falling south at *pct* percent."""
    return np.fromfunction(
        lambda r, c: 100.0 - r * (pct / 100.0) * CELL, (ROWS, COLS)
    ).astype("float64")


def _basin_mask():
    m = np.zeros((ROWS, COLS), dtype=bool)
    m[BASIN] = True
    return m


def _fill_depressions(dem):
    """Priority-flood depression fill — water level each cell can hold.

    Small, dependency-free implementation so the regression is pinned by this test
    rather than by whatever pysheds happens to do. Boundary cells drain freely; every
    interior cell rises to the lowest maximum along any path to the boundary.
    """
    import heapq

    rows, cols = dem.shape
    filled = np.full(dem.shape, np.inf)
    heap = []
    for r in range(rows):
        for c in (0, cols - 1):
            filled[r, c] = dem[r, c]
            heapq.heappush(heap, (dem[r, c], r, c))
    for c in range(cols):
        for r in (0, rows - 1):
            if filled[r, c] == np.inf:
                filled[r, c] = dem[r, c]
                heapq.heappush(heap, (dem[r, c], r, c))

    while heap:
        level, r, c = heapq.heappop(heap)
        if level > filled[r, c]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            new = max(dem[nr, nc], level)
            if new < filled[nr, nc]:
                filled[nr, nc] = new
                heapq.heappush(heap, (new, nr, nc))
    return filled


def _ponded_m3(dem):
    """Volume held once depressions fill, in m³."""
    depth = np.clip(_fill_depressions(dem) - dem, 0.0, None)
    return float(depth.sum()) * CELL * CELL


def _translated_burn(dem, mask, depth):
    """The old behaviour, kept here purely so the regression stays pinned."""
    out = dem.copy()
    out[mask] -= depth
    return out


def _level_burn(dem, mask, depth):
    spill, _ = pour_point(dem, mask)
    return level_invert(dem, mask, depth, spill)


@pytest.mark.parametrize("slope_pct", [0, 2, 5, 10])
class TestLevelFloorHoldsItsVolumeAtAnySlope:
    def test_level_floor_holds_the_design_prism(self, slope_pct):
        dem = _sloped(slope_pct)
        ponded = _ponded_m3(_level_burn(dem, _basin_mask(), DEPTH))
        assert ponded == pytest.approx(NOMINAL_M3, rel=0.02)

    def test_translated_burn_falls_short_on_slope(self, slope_pct):
        """The bug, pinned: a translated cut loses storage as the ground steepens."""
        dem = _sloped(slope_pct)
        ponded = _ponded_m3(_translated_burn(dem, _basin_mask(), DEPTH))
        if slope_pct == 0:
            assert ponded == pytest.approx(NOMINAL_M3, rel=0.02)
        else:
            assert ponded < NOMINAL_M3 * 0.95


class TestTheGradientOfTheOldFailure:
    def test_shortfall_worsens_monotonically_with_slope(self):
        held = [_ponded_m3(_translated_burn(_sloped(p), _basin_mask(), DEPTH))
                for p in (0, 2, 5, 10)]
        assert held == sorted(held, reverse=True)
        assert held[3] < held[0] * 0.5          # 10% ground loses over half the storage

    def test_level_floor_is_flat_regardless_of_slope(self):
        held = [_ponded_m3(_level_burn(_sloped(p), _basin_mask(), DEPTH))
                for p in (0, 2, 5, 10)]
        assert max(held) - min(held) < NOMINAL_M3 * 0.02


class TestLevelInvert:
    def test_never_raises_a_cell(self):
        dem = _sloped(5)
        out = _level_burn(dem, _basin_mask(), DEPTH)
        assert (out <= dem + 1e-9).all()

    def test_only_the_mask_is_touched(self):
        dem = _sloped(5)
        out = _level_burn(dem, _basin_mask(), DEPTH)
        outside = ~_basin_mask()
        assert np.array_equal(out[outside], dem[outside])

    def test_floor_is_flat(self):
        dem = _sloped(10)
        out = _level_burn(dem, _basin_mask(), DEPTH)
        floor = out[_basin_mask()]
        assert floor.max() - floor.min() < 1e-6

    def test_floor_sits_one_depth_below_the_spill_level(self):
        dem = _sloped(5)
        mask = _basin_mask()
        spill, _ = pour_point(dem, mask)
        out = level_invert(dem, mask, DEPTH, spill)
        assert out[mask].max() == pytest.approx(spill - DEPTH)

    def test_ground_already_below_the_floor_is_left_alone(self):
        dem = _sloped(0)
        dem[30, 30] = 90.0                       # a deep hole inside the footprint
        out = _level_burn(dem, _basin_mask(), DEPTH)
        assert out[30, 30] == pytest.approx(90.0)

    def test_empty_mask_is_a_no_op(self):
        dem = _sloped(5)
        out = level_invert(dem, np.zeros_like(_basin_mask()), DEPTH, 100.0)
        assert np.array_equal(out, dem)


class TestBatteredInvert:
    def test_steps_deepen_inward(self):
        dem = np.full((30, 30), 50.0)
        outer = np.zeros((30, 30), dtype=bool)
        outer[10:20, 10:20] = True
        inner = np.zeros((30, 30), dtype=bool)
        inner[12:18, 12:18] = True
        out = battered_invert(dem, [(outer, 0.5), (inner, 1.5)], 50.0)
        assert out[10, 10] == pytest.approx(49.5)    # outer step
        assert out[15, 15] == pytest.approx(48.5)    # inner step, deeper

    def test_holds_less_than_a_vertical_cut_of_the_same_depth(self):
        dem = np.full((30, 30), 50.0)
        outer = np.zeros((30, 30), dtype=bool)
        outer[10:20, 10:20] = True
        inner = np.zeros((30, 30), dtype=bool)
        inner[12:18, 12:18] = True
        battered = _ponded_m3(battered_invert(dem, [(outer, 0.5), (inner, 1.5)], 50.0))
        vertical = _ponded_m3(level_invert(dem, outer, 1.5, 50.0))
        assert battered < vertical


class TestRasterisableCapacity:
    def test_a_narrow_channel_collapses_to_a_rectangle(self):
        """Below ~3 cells across, the grid cannot hold the batter at all.

        The batter run is deliberately non-zero: it is the *width* that defeats the
        grid here, and the assertion is worthless if the batter gate short-circuits
        it first.
        """
        v = rasterisable_capacity(200, 1.0, 0.5, top_width=2.0,
                                  bottom_width=1.0, cell_size=1.0, batter_run=0.5)
        assert v == pytest.approx(200 * 1.0 * 0.5)     # full-depth trench

    def test_a_wide_feature_keeps_its_trapezoid(self):
        v = rasterisable_capacity(1000, 1.0, 0.5, top_width=10.0,
                                  bottom_width=8.0, cell_size=1.0, batter_run=1.0)
        assert v < 1000 * 1.0 * 0.5                     # batter is represented
        assert v == pytest.approx(1000 * 0.5 * (9.0 / 10.0))

    def test_an_unbattered_feature_stays_rectangular_however_wide(self):
        """No batter run means the burner levelled a flat floor — a rectangle.

        Regression for the field-test finding: a 3.00 m swale on a 1.00 m DEM cleared
        the three-cell width test and was discounted by ``mean_width/top_width`` (2/3),
        so ``Measured`` read a flat +50% against ``At grid`` on every such swale while
        the burn was in fact correct. ``batter_run_m`` is 0 on a drawn swale, and
        ``DEMBurner`` only calls ``battered_invert`` when it is positive.
        """
        v = rasterisable_capacity(500, 1.0, 1.0, top_width=3.0,
                                  bottom_width=1.0, cell_size=1.0, batter_run=0.0)
        assert v == pytest.approx(500 * 1.0 * 1.0)      # not 2/3 of it
        assert v != pytest.approx(500 * 1.0 * 1.0 * (2.0 / 3.0))

    def test_the_resolution_penalty_is_positive_for_a_narrow_swale(self):
        """A 2 m 1:1 swale burns as a trench holding a third more than its trapezoid.

        Over 100 m the swale covers 2 cells per metre = 200 cells, each cut to the
        full 0.5 m depth: 100 m³, against a true trapezoidal 75 m³. That +33% is the
        resolution penalty the Verify table reports rather than blaming on the burn.
        """
        geometric = (2.0 + 1.0) / 2.0 * 0.5 * 100       # 0.75 m² × 100 m = 75 m³
        raster = rasterisable_capacity(200, 1.0, 0.5, 2.0, 1.0, 1.0, batter_run=0.5)
        assert raster > geometric
        assert raster == pytest.approx(100.0)
        assert raster / geometric == pytest.approx(4 / 3, rel=0.01)

    def test_degenerate_inputs_are_zero(self):
        assert rasterisable_capacity(0, 1.0, 0.5, 2.0, 1.0, 1.0) == 0.0
        assert rasterisable_capacity(10, 1.0, 0.0, 2.0, 1.0, 1.0) == 0.0
        assert rasterisable_capacity(10, 0.0, 0.5, 2.0, 1.0, 1.0) == 0.0


class TestSteepGroundWarning:
    def test_fires_only_when_relief_exceeds_the_depth(self):
        assert steep_ground_warning("Basin 1", 2.0, 1.5) is not None
        assert steep_ground_warning("Basin 1", 1.0, 1.5) is None
        assert steep_ground_warning("Basin 1", 1.5, 1.5) is None

    def test_names_both_numbers_and_the_cost(self):
        msg = steep_ground_warning("Basin 1", 2.0, 1.5, cut_m3=1020.0, storage_m3=600.0)
        assert "2.0 m" in msg and "1.5 m" in msg
        assert "1,020" in msg and "600" in msg
        assert "Basin 1" in msg

    def test_degenerate_inputs_are_silent(self):
        assert steep_ground_warning("x", None, 1.5) is None
        assert steep_ground_warning("x", 2.0, 0.0) is None


class TestImpoundmentWarning:
    """When a companion berm has stopped being a bank and become a dam wall.

    Both thresholds are set from the 36 storage features of the Quail Island design,
    whose ponds were flooded individually: retained depth runs 0.00–3.13 m with a median
    of 0.77 m. Retaining *something* is what a berm is for, so a threshold near that
    median fires on nearly everything (0.5 m catches 31 of 36) and says nothing. Past a
    metre the population thins to 4 of 36 and the consequence changes.
    """

    def test_an_ordinary_swale_berm_is_silent(self):
        # Swale 22's measured figures: 0.44 m of head, 108 m³ above natural ground.
        assert impoundment_warning("Swale 22", 0.44, 108.0) is None

    def test_deep_head_fires(self):
        # Basin 39: 3.13 m standing above natural ground.
        msg = impoundment_warning("Basin 39", 3.13, 1158.0)
        assert msg is not None
        assert "3.1 m" in msg and "1,158" in msg and "Basin 39" in msg

    def test_a_large_volume_behind_a_shallower_bank_also_fires(self):
        """Swale 5: 0.91 m — under the depth arm — but 603 m³ held above ground.

        The risk has two shapes and one arm catches only the first. What runs downhill
        if the bank fails is the volume, not the head.
        """
        msg = impoundment_warning("Swale 5", 0.91, 603.0)
        assert msg is not None
        assert "603" in msg

    def test_it_asks_for_a_spillway_when_there_is_none(self):
        msg = impoundment_warning("Swale 5", 0.91, 603.0, has_spillway=False)
        assert "spillway" in msg and "council" in msg

    def test_with_a_spillway_it_still_names_the_structure(self):
        msg = impoundment_warning("Swale 5", 0.91, 603.0, has_spillway=True)
        assert "spillway" in msg
        assert "council" not in msg
        assert "compacted" in msg

    def test_degenerate_inputs_are_silent(self):
        assert impoundment_warning("x", None, None) is None
        assert impoundment_warning("x", 0.0, 0.0) is None


# ---------------------------------------------------------------------------
# DEMBurner integration — the level-bottom burn through the real burner
# ---------------------------------------------------------------------------

from terrainflow_assessment.modules.earthwork_design import (  # noqa: E402
    DEMBurner,
    _as_float,
    capacity_breakdown,
)
from tests.conftest import make_mock_line_geom, make_mock_polygon_geom  # noqa: E402
from tests.test_dem_burner import _mock_ew, _write_dem  # noqa: E402


def _burner(tmp_path, data, name="dem.tif"):
    return DEMBurner(_write_dem(str(tmp_path / name), data))


class TestBurnerBasinLevelFloor:
    def test_basin_floor_is_flat_on_sloping_ground(self, tmp_path):
        data = np.fromfunction(lambda r, c: 100.0 - r * 0.1, (30, 30)).astype("float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((10.0, 10.0, 20.0, 20.0)),
                      depth=1.5)
        ew.batter_run_m = 0.0
        out = b.burn_earthworks([ew])

        cut = out < data - 1e-6
        assert cut.any()
        floor = out[cut]
        assert floor.max() - floor.min() < 1e-4      # genuinely level

    def test_basin_holds_its_prism_where_the_old_burn_lost_half(self, tmp_path):
        from shapely.geometry import box

        data = np.fromfunction(lambda r, c: 100.0 - r * 0.1, (60, 60)).astype("float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((20.0, 20.0, 40.0, 40.0)),
                      depth=1.5)
        ew.batter_run_m = 0.0

        level = _ponded_m3(b.burn_earthworks([ew]).astype("float64"))
        mask = b._rasterize(box(20.0, 20.0, 40.0, 40.0))
        translated = _ponded_m3(_translated_burn(data.astype("float64"), mask, 1.5))

        assert level > translated * 1.5    # 10% ground: the old burn lost over a third

    def test_battered_basin_burns_stepped_walls(self, tmp_path):
        data = np.full((40, 40), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((10.0, 10.0, 30.0, 30.0)),
                      depth=2.0)
        ew.batter_run_m = 3.0
        out = b.burn_earthworks([ew])
        cut_depths = np.unique(np.round(data - out, 3))
        assert len(cut_depths) > 2                   # several step levels, not one wall
        assert out.min() == pytest.approx(48.0, abs=1e-3)

    def test_sub_cell_basin_burns_a_cell_and_warns(self, tmp_path):
        """A polygon smaller than a cell used to burn nothing at all, silently."""
        data = np.full((20, 20), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((10.1, 10.1, 10.4, 10.4)),
                      depth=1.0)
        ew.batter_run_m = 0.0
        out = b.burn_earthworks([ew])
        assert out.min() < 50.0
        assert any("1-cell width" in w for w in b.warnings)

    def test_steep_ground_warning_reaches_the_burner(self, tmp_path):
        data = np.fromfunction(lambda r, c: 100.0 - r * 0.2, (40, 40)).astype("float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((10.0, 10.0, 30.0, 30.0)),
                      depth=1.0, name="Basin 1")
        ew.batter_run_m = 0.0
        b.burn_earthworks([ew])
        assert any("falls" in w and "design depth" in w for w in b.warnings)

    def test_gentle_ground_raises_no_steep_warning(self, tmp_path):
        data = np.full((30, 30), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((10.0, 10.0, 20.0, 20.0)),
                      depth=2.0)
        ew.batter_run_m = 0.0
        b.burn_earthworks([ew])
        assert not any("falls" in w for w in b.warnings)


class TestBurnerSwaleLevelInvert:
    def test_swale_invert_is_level_not_breached(self, tmp_path):
        """The breach that used to drain every swale is gone."""
        data = np.fromfunction(lambda r, c: 100.0 - c * 0.02, (30, 40)).astype("float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("swale", make_mock_line_geom([(5.0, 15.0), (35.0, 15.0)]),
                      depth=0.5, width=2.0)
        ew.companion_berm = False
        out = b.burn_earthworks([ew])
        cut = out < data - 1e-6
        floor = out[cut]
        assert floor.max() - floor.min() < 1e-4

    def test_swale_now_ponds_instead_of_draining(self, tmp_path):
        data = np.fromfunction(lambda r, c: 100.0 - c * 0.02, (30, 40)).astype("float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("swale", make_mock_line_geom([(5.0, 15.0), (35.0, 15.0)]),
                      depth=0.5, width=2.0)
        ew.companion_berm = False
        assert _ponded_m3(b.burn_earthworks([ew]).astype("float64")) > 0.0

    def test_companion_berm_is_built_before_the_invert_is_referenced(self, tmp_path):
        data = np.full((30, 40), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("swale", make_mock_line_geom([(5.0, 15.0), (35.0, 15.0)]),
                      depth=0.5, width=2.0)
        ew.companion_berm = True
        out = b.burn_earthworks([ew])
        assert out.max() > 50.0        # the berm was raised
        assert out.min() < 50.0        # and the trench still cut


class TestCapacityBreakdown:
    def _swale(self):
        from unittest.mock import MagicMock
        ew = MagicMock()
        ew.type = "swale"
        ew.geometry = make_mock_line_geom([(0.0, 0.0), (100.0, 0.0)])
        ew.depth = 0.5
        ew.width = 2.0
        ew.top_width_m = 2.0
        ew.bottom_width_m = 1.0
        ew.batter_run_m = 0.0
        ew.companion_berm = False
        ew.capacity_m3 = 60.0          # 0.75 m² × 100 m × 0.8 freeboard
        return ew

    def test_splits_freeboard_from_geometry(self):
        b = capacity_breakdown(self._swale(), cell_size=1.0)
        assert b["design"] == pytest.approx(60.0)
        assert b["geometric"] == pytest.approx(75.0)
        assert b["freeboard_m3"] == pytest.approx(15.0)

    def test_resolution_penalty_measures_the_trench_the_burn_cut(self):
        """The penalty is ``cut − section``: did the grid hold the section you drew?

        It used to be ``rasterisable − geometric``, which worked only while
        ``rasterisable`` meant "the drawn shape at this cell size". It no longer does —
        it is the pond, and on a swale with a keyed companion berm the pond is roughly
        double the drawn section because the bank holds water above natural ground.
        Left pointed at that figure this key would have reported every working berm as a
        resolution failure, at +80% and upward. The narrow grid-fidelity question is
        still worth asking, so it keeps the key and gets the right operands.
        """
        b = capacity_breakdown(self._swale(), cell_size=1.0, n_cells=200, cut_m3=100.0)
        assert b["cut_m3"] == pytest.approx(100.0)
        assert b["section_m3"] == pytest.approx(75.0)
        assert b["resolution_penalty_m3"] == pytest.approx(25.0)

    def test_impoundment_is_what_the_pond_adds_over_the_drawn_trench(self):
        """The new figure, and the one a designer needs: what berm and hillside add.

        No cross-section predicts it — it depends on the ground the bank stands on — so
        it cannot live inside ``geometric`` and it must not be folded into the grid
        penalty. Its own key, measured against the drawn trench.
        """
        b = capacity_breakdown(self._swale(), cell_size=1.0, n_cells=200,
                               terrain_storage_m3=180.0, cut_m3=100.0)
        assert b["rasterisable"] == pytest.approx(180.0)
        assert b["impoundment_m3"] == pytest.approx(105.0)   # 180 pond − 75 trench
        assert b["resolution_penalty_m3"] == pytest.approx(25.0)  # unaffected

    def test_a_measurement_beats_the_model(self):
        """Given both, the flood wins — a model beside a burn has gone wrong twice."""
        b = capacity_breakdown(self._swale(), cell_size=1.0, n_cells=200,
                               terrain_storage_m3=42.0)
        assert b["rasterisable"] == pytest.approx(42.0)

    def test_without_a_cell_count_no_resolution_claim_is_made(self):
        b = capacity_breakdown(self._swale(), cell_size=1.0)
        assert b["rasterisable"] == b["geometric"]
        assert b["cut_m3"] is None
        assert b["resolution_penalty_m3"] == 0.0


class TestAsFloat:
    def test_coerces_numbers_and_strings(self):
        assert _as_float(2.5) == 2.5
        assert _as_float("3") == 3.0

    def test_falls_back_on_none_and_junk(self):
        assert _as_float(None) == 0.0
        assert _as_float("abc") == 0.0
        assert _as_float(object()) == 0.0
        assert _as_float(None, 1.5) == 1.5


class TestBurnerDefensivePaths:
    def test_basin_entirely_off_the_dem_is_a_no_op(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((500.0, 500.0, 510.0, 510.0)),
                      depth=1.0)
        ew.batter_run_m = 0.0
        assert np.array_equal(b.burn_earthworks([ew]), data)

    def test_basin_over_nodata_is_a_no_op(self, tmp_path):
        """No usable elevation under the footprint means no datum to level to.

        Asserted against ``b.original`` rather than the raw file contents: the
        burner masks the sentinel to NaN on load, so "unchanged" now means "still
        every bit as much a hole", not "still literally -9999". ``save`` puts the
        sentinel back — covered separately.
        """
        data = np.full((20, 20), -9999.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((5.0, 5.0, 15.0, 15.0)),
                      depth=1.0)
        ew.batter_run_m = 0.0
        out = b.burn_earthworks([ew])
        assert np.isnan(out).all()
        assert np.array_equal(np.isnan(out), np.isnan(b.original))

    def test_batter_wider_than_the_basin_degrades_to_a_single_cut(self, tmp_path):
        """An inset that erases the polygon must not erase the excavation."""
        data = np.full((20, 20), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("basin", make_mock_polygon_geom((8.0, 8.0, 12.0, 12.0)),
                      depth=1.0)
        ew.batter_run_m = 50.0                    # far wider than the 4 m footprint
        out = b.burn_earthworks([ew])
        assert out.min() < 50.0

    def test_swale_over_nodata_is_a_no_op(self, tmp_path):
        data = np.full((20, 20), -9999.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("swale", make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)]),
                      depth=0.5, width=2.0)
        ew.companion_berm = False
        out = b.burn_earthworks([ew])
        assert np.isnan(out).all()

    def test_saving_puts_the_sentinel_back(self, tmp_path):
        """Holes are NaN in memory and the declared sentinel on disk.

        The two have to agree: NaN written under a ``nodata=-9999`` tag is masked by
        neither convention, and the next reader sees a hole as ordinary ground.
        """
        data = np.full((6, 6), 50.0, dtype="float32")
        data[2, 2] = -9999.0
        b = _burner(tmp_path, data)
        assert np.isnan(b.original[2, 2])

        import rasterio

        out_path = str(tmp_path / "saved.tif")
        b.save(b.original, out_path)
        with rasterio.open(out_path) as src:
            back = src.read(1)
            assert src.nodata == -9999.0
        assert back[2, 2] == -9999.0
        assert back[0, 0] == pytest.approx(50.0)

class TestBurnerPartialHole:
    """A hole *inside* an otherwise valid site — the case the all-nodata tests miss.

    Every earlier nodata test made the whole tile nodata, where the burn is a no-op
    and nothing gets measured. The damage from a raw sentinel happens when there is
    real terrain beside it to contaminate: one -9999 cell is an elevation ten
    kilometres down, and it dominates any mean, range or interpolation it reaches.
    """

    @staticmethod
    def _sloping_with_hole(rows=30, cols=30, hole=(slice(4, 7), slice(4, 7))):
        data = np.fromfunction(lambda r, c: 100.0 - r * 0.1, (rows, cols))
        data = data.astype("float32")
        data[hole] = -9999.0
        return data

    def test_hole_is_nan_not_an_elevation(self, tmp_path):
        b = _burner(tmp_path, self._sloping_with_hole())
        assert np.isnan(b.original[5, 5])
        assert b.original[20, 20] == pytest.approx(98.0)
        assert not (b.original < -1000).any(), "a sentinel survived the load"

    def test_relief_ignores_the_hole(self, tmp_path):
        """`internal_relief` over a footprint touching a hole is real relief.

        Unmasked it reported ~10,049 m — the drop to the sentinel — and fired the
        steep-ground advisory on flat paddock.
        """
        from terrainflow_assessment.modules.footprint import internal_relief
        b = _burner(tmp_path, self._sloping_with_hole())
        mask = np.zeros(b.shape, dtype=bool)
        mask[3:9, 3:9] = True                       # straddles the hole

        # Deliberately without nodata=: the masked array must be safe on its own,
        # since that is how the call site read it for as long as the bug existed.
        assert internal_relief(b.original, mask) < 1.0
        assert internal_relief(b.original, mask, nodata=b.nodata) < 1.0

    def test_side_selection_is_not_dragged_by_a_hole(self, tmp_path):
        """The lower side is the lower *ground*, not the side containing a hole."""
        data = np.full((20, 20), 50.0, dtype="float32")
        data[2:5, 2:5] = -9999.0                    # hole on the otherwise-higher side
        data[:, 12:] = 45.0                         # genuinely lower ground, right
        b = _burner(tmp_path, data)

        left = np.zeros(b.shape, dtype=bool)
        left[0:8, 0:8] = True
        right = np.zeros(b.shape, dtype=bool)
        right[0:8, 12:20] = True

        assert b._ground_mean(left) == pytest.approx(50.0)
        assert b._ground_mean(right) == pytest.approx(45.0)
        assert b._ground_mean(right) < b._ground_mean(left)

    def test_an_all_hole_side_is_never_the_lower_one(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        data[0:8, 0:8] = -9999.0
        b = _burner(tmp_path, data)
        blind = np.zeros(b.shape, dtype=bool)
        blind[0:8, 0:8] = True
        seen = np.zeros(b.shape, dtype=bool)
        seen[10:18, 10:18] = True

        assert b._ground_mean(blind) == np.inf
        assert not b._ground_mean(blind) < b._ground_mean(seen)

    def test_ponding_stays_bounded_by_real_relief(self, tmp_path, monkeypatch):
        """The A-C2 path: a hole must not become a kilometre-deep phantom pit.

        Forced through the downsample by dropping the cell cap rather than
        allocating four million cells. Bilinear interpolation of a raw -9999 used to
        emit values like -4974.5 that matched no declared nodata, so
        `fill_depressions` filled to them.
        """
        from terrainflow_assessment.modules import earthwork_design as ed
        monkeypatch.setattr(ed, "_MAX_PONDING_CELLS", 400)

        b = _burner(tmp_path, self._sloping_with_hole(60, 60,
                                                      (slice(20, 30), slice(20, 30))))
        ponding = b.get_ponding_layer(b.original)

        assert np.isfinite(ponding).all(), "NaN leaked into the ponding layer"
        # Total fall across the tile is 60 * 0.1 = 6 m, so nothing can pond deeper.
        assert ponding.max() < 10.0, f"ponded {ponding.max():.1f} m on a 6 m tile"

    def test_diversion_starting_in_a_hole_is_not_burned_to_the_sentinel(self, tmp_path):
        """No datum anywhere on the line means no burn, and a warning saying so."""
        data = np.full((20, 20), -9999.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("diversion", make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)]),
                      depth=0.5, width=2.0)
        ew.gradient_pct = 1.0

        out = b.burn_earthworks([ew])

        assert np.isnan(out).all(), "a channel was graded from a sentinel"
        assert any("elevation to grade from" in w for w in b.warnings), b.warnings


class TestBurnerDefensivePathsExtra:
    def test_steep_warning_survives_an_unmeasurable_cut(self, tmp_path):
        """The advisory still fires when the volumes can't be quantified."""
        from terrainflow_assessment.modules.burn_strategy import steep_ground_warning
        msg = steep_ground_warning("Basin 9", 3.0, 1.0, cut_m3=None, storage_m3=None)
        assert msg is not None and "excavation" not in msg

    def test_degenerate_zero_area_basin_claims_its_centroid_cell(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        # A collapsed box rasterises to nothing; the centroid fallback still cuts.
        ew = _mock_ew("basin", make_mock_polygon_geom((10.0, 10.0, 10.0, 10.0)),
                      depth=1.0)
        ew.batter_run_m = 0.0
        out = b.burn_earthworks([ew])
        assert out.min() == pytest.approx(49.0, abs=1e-3)

    def test_sub_cell_swale_falls_back_to_its_path_cells(self, tmp_path):
        """A buffer too small to claim a cell still carves the line it follows."""
        data = np.full((20, 20), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("swale", make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)]),
                      depth=0.6, width=0.0)
        ew.buffer_radius_m = 0.0
        ew.bottom_width_m = 0.0
        ew.companion_berm = False
        out = b.burn_earthworks([ew])
        assert out.min() == pytest.approx(49.4, abs=1e-3)

    def test_warn_steep_survives_a_bad_depth(self, tmp_path):
        data = np.fromfunction(lambda r, c: 100.0 - r * 0.2, (30, 30)).astype("float32")
        b = _burner(tmp_path, data)
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True
        bad = _mock_ew("basin", make_mock_polygon_geom((0.0, 0.0, 1.0, 1.0)))
        bad.depth = "not a number"
        b._warn_steep(bad, mask, relief=3.0, burned_dem=data)   # must not raise

    def test_line_path_cells_is_empty_for_a_polygon(self, tmp_path):
        """Polygons have no .coords; the path helper must return empty, not raise."""
        from shapely.geometry import box
        b = _burner(tmp_path, np.full((10, 10), 50.0, dtype="float32"))
        assert b._line_path_cells(box(1.0, 1.0, 5.0, 5.0)) == []

    def test_dam_off_the_dem_leaves_the_terrain_alone(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        b = _burner(tmp_path, data)
        ew = _mock_ew("dam", make_mock_line_geom([(500.0, 500.0), (510.0, 500.0)]),
                      width=2.0, crest_elevation=60.0)
        assert np.array_equal(b.burn_earthworks([ew]), data)
