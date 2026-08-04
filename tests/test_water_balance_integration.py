"""End-to-end design-tier balance on a synthetic hillside.

Exercises the whole chain the Design tab now runs — conditioned DEM → D8 pointers →
direct catchments → routing → water balance — without QGIS. This is the regression
test for the reported bug: "100% of storm held on site, 0 m³ leaves" on a design that
plainly did not capture everything.
"""

import numpy as np
import pytest

from terrainflow_assessment.modules.flow_graph import (
    LABEL_NONE,
    d8_from_dem,
    label_direct_catchments,
    walk_downslope,
)
from terrainflow_assessment.modules.simulation import EarthworkStore, resolve_targets
from terrainflow_assessment.modules.water_balance import run_water_balance

ROWS, COLS = 100, 100
CELL_AREA = 1.0
RUNOFF_MM = 24.0                      # ~120 mm storm at CN 61
RUNOFF_M = RUNOFF_MM / 1000.0
SITE_RUNOFF_M3 = ROWS * COLS * CELL_AREA * RUNOFF_M   # 240 m³ over 1 ha


def _hillside():
    """A smooth 5% slope falling south — conditioned by construction (no pits)."""
    return np.fromfunction(lambda r, c: 100.0 - r * 0.05, (ROWS, COLS)).astype("float64")


class _Site:
    """A labelled hillside with some swales across it."""

    def __init__(self, swale_rows, capacities, widths=3):
        dem = _hillside()
        self.next_flat, self.is_sink = d8_from_dem(dem, 1.0, 1.0)
        self.interceptors = np.full((ROWS, COLS), LABEL_NONE, dtype=np.int32)
        self.ids = []
        for label, row in enumerate(swale_rows):
            self.interceptors[row:row + widths, :] = label
            self.ids.append(f"SW{label}")
        self.domain = np.ones((ROWS, COLS), dtype=bool)
        self.labels = label_direct_catchments(
            self.next_flat, self.interceptors, self.domain, is_sink=self.is_sink)
        self.capacities = capacities
        self.swale_rows = swale_rows
        self.widths = widths

    def stores(self, runoff_m=RUNOFF_M):
        out = []
        for i, ew_id in enumerate(self.ids):
            s = EarthworkStore(
                name=ew_id, ew_type="swale", capacity_m3=self.capacities[i],
                area_m2=COLS * self.widths, infiltration_rate_mm_hr=0.0,
                elevation=100.0 - self.swale_rows[i] * 0.05, id=ew_id,
            )
            cells = int(self.labels.counts[i])
            s.direct_catchment_m2 = cells * CELL_AREA
            s.inflow_m3 = cells * CELL_AREA * runoff_m
            # Outlet = the downslope edge of the footprint, mid-width.
            s.outlet_flat = (self.swale_rows[i] + self.widths - 1) * COLS + COLS // 2
            out.append(s)
        return out

    def walker(self):
        inter_flat = self.interceptors.ravel()
        index_of = {ew_id: i for i, ew_id in enumerate(self.ids)}
        by_index = {i: ew_id for ew_id, i in index_of.items()}

        def _walk(store):
            label, _ = walk_downslope(self.next_flat, store.outlet_flat, inter_flat,
                                      skip_label=index_of[store.id])
            return by_index.get(label) if label is not None else None

        return _walk

    def balance(self, runoff_m=RUNOFF_M, total=SITE_RUNOFF_M3):
        stores = self.stores(runoff_m)
        routing = resolve_targets(stores, walker=self.walker())
        uncaptured = ((self.labels.exit_cells + self.labels.sink_cells)
                      * CELL_AREA * runoff_m)
        return run_water_balance(stores, 24.0, total, uncaptured_m3=uncaptured,
                                 routing=routing), stores, routing


class TestOneSwaleOnAHillside:
    def test_capture_is_plausible_not_a_hundred_percent(self):
        """The headline regression: a single mid-slope swale cannot hold the site."""
        site = _Site([50], [40.0])
        result, _, _ = site.balance()

        assert result.capture_pct < 100.0
        assert result.site_exit_m3 > 0.0
        assert result.mass_balance_ok
        # It catches everything upslope of it (53 rows) but stores only 40 m³ of the
        # ~127 m³ that arrives, and the 47 rows below it never reach the swale at all.
        assert 15.0 < result.capture_pct < 20.0

    def test_mass_balance_closes(self):
        result, _, _ = _Site([50], [40.0]).balance()
        assert result.total_captured_m3 + result.site_exit_m3 == pytest.approx(
            SITE_RUNOFF_M3, rel=1e-6)

    def test_site_exit_splits_into_uncaptured_and_routed(self):
        result, _, _ = _Site([50], [40.0]).balance()
        assert result.uncaptured_m3 > 0      # the ground below the swale
        assert result.routed_exit_m3 > 0     # the swale's own overflow
        assert result.site_exit_m3 == pytest.approx(
            result.uncaptured_m3 + result.routed_exit_m3)

    def test_a_swale_big_enough_to_hold_its_catchment_still_is_not_100_percent(self):
        """Even an enormous swale can't capture ground that drains below it."""
        result, _, _ = _Site([50], [1e6]).balance()
        assert result.capture_pct < 100.0
        assert result.uncaptured_m3 > 0

    def test_a_swale_at_the_bottom_captures_nearly_everything(self):
        result, _, _ = _Site([96], [1e6], widths=4).balance()
        assert result.capture_pct == pytest.approx(100.0, abs=0.5)


class TestStackedSwales:
    def test_lower_swale_gets_only_the_band_between(self):
        """No double counting: the upper swale's catchment is not also the lower's."""
        site = _Site([20, 60], [40.0, 40.0])
        result, _, _ = site.balance()
        by_id = {f["id"]: f for f in result.per_feature}

        # Upper takes rows 0-22 (23 rows); lower takes rows 23-62 (40 rows).
        assert by_id["SW0"]["direct_catchment_m2"] == pytest.approx(23 * COLS)
        assert by_id["SW1"]["direct_catchment_m2"] == pytest.approx(40 * COLS)
        # Together they are the ground above the lower swale — counted exactly once.
        assert (by_id["SW0"]["direct_catchment_m2"]
                + by_id["SW1"]["direct_catchment_m2"]) == pytest.approx(63 * COLS)

    def test_upper_overflow_cascades_into_the_lower_swale(self):
        site = _Site([20, 60], [10.0, 500.0])
        result, _, routing = site.balance()
        by_id = {f["id"]: f for f in result.per_feature}

        assert routing.edges["SW0"] == "SW1"       # walked down the real flow path
        assert by_id["SW0"]["overflow_m3"] > 0
        assert by_id["SW1"]["upstream_inflow_m3"] == pytest.approx(
            by_id["SW0"]["overflow_m3"])

    def test_adding_upslope_storage_reduces_the_lower_swale_load(self):
        alone, _, _ = _Site([60], [500.0]).balance()
        pair, _, _ = _Site([20, 60], [500.0, 500.0]).balance()
        lower_alone = alone.per_feature[0]["total_inflow_m3"]
        lower_paired = {f["id"]: f for f in pair.per_feature}["SW1"]["total_inflow_m3"]
        assert lower_paired < lower_alone

    def test_terminal_deficit_is_the_bottom_swale_overflow(self):
        site = _Site([20, 60], [10.0, 10.0])
        result, _, _ = site.balance()
        by_id = {f["id"]: f for f in result.per_feature}
        assert by_id["SW1"]["is_terminal"]
        assert result.terminal_deficit_m3 == pytest.approx(by_id["SW1"]["overflow_m3"])
        assert result.terminal_deficit_m3 > 0

    def test_every_domain_cell_is_accounted_for(self):
        site = _Site([20, 50, 80], [40.0, 40.0, 40.0])
        assert site.labels.is_exhaustive()
        assert site.labels.domain_cells == ROWS * COLS
        assert site.labels.unresolved_cells == 0


class TestStormResponse:
    def test_a_bigger_storm_spills_more_than_proportionally(self):
        """Storage is fixed, so doubling the rain more than doubles what escapes."""
        site = _Site([50], [40.0])
        small, _, _ = site.balance(runoff_m=RUNOFF_M, total=SITE_RUNOFF_M3)
        big, _, _ = site.balance(runoff_m=RUNOFF_M * 2, total=SITE_RUNOFF_M3 * 2)

        # The swale is full in both cases, so capture is pinned at its 40 m³.
        assert small.total_captured_m3 == pytest.approx(40.0)
        assert big.total_captured_m3 == pytest.approx(40.0)
        assert big.site_exit_m3 == pytest.approx(SITE_RUNOFF_M3 * 2 - 40.0)
        assert big.site_exit_m3 > small.site_exit_m3 * 2
        assert big.capture_pct < small.capture_pct     # same storage, more water
        assert big.mass_balance_ok

    def test_storm_scaling_is_linear_when_nothing_is_full(self):
        site = _Site([50], [1e6])                      # never fills
        small, _, _ = site.balance(runoff_m=RUNOFF_M, total=SITE_RUNOFF_M3)
        big, _, _ = site.balance(runoff_m=RUNOFF_M * 2, total=SITE_RUNOFF_M3 * 2)
        assert big.total_captured_m3 == pytest.approx(small.total_captured_m3 * 2)
        assert big.capture_pct == pytest.approx(small.capture_pct)

    def test_capture_never_needs_clamping(self):
        for capacity in (0.0, 1.0, 40.0, 1e3, 1e7):
            result, _, _ = _Site([50], [capacity]).balance()
            assert 0.0 <= result.capture_pct <= 100.0
            assert result.mass_balance_ok


class TestSizingBasis:
    """Rainfall vs SCS-CN runoff as the depth features are sized against.

    The gap is a factor of ~4 at typical settings and it is all downside risk, so the
    basis is an explicit choice rather than an assumption buried in the model.
    """

    def _capture_at(self, runoff_m, capacity=40.0):
        site = _Site([50], [capacity])
        total = ROWS * COLS * CELL_AREA * runoff_m
        result, _, _ = site.balance(runoff_m=runoff_m, total=total)
        return result

    def test_rainfall_basis_demands_far_more_storage(self):
        from terrainflow_assessment.modules.catchment import SCSRunoff
        scs = SCSRunoff()
        rain_mm = 120.0
        runoff_mm = scs.runoff_depth(rain_mm, scs.adjust_cn(61, "normal"))

        on_rain = self._capture_at(rain_mm / 1000.0, capacity=1e9)
        on_runoff = self._capture_at(runoff_mm / 1000.0, capacity=1e9)

        # Same site, same feature — the water it must hold differs ~4×.
        ratio = (on_rain.per_feature[0]["total_inflow_m3"]
                 / on_runoff.per_feature[0]["total_inflow_m3"])
        assert 3.5 < ratio < 4.5

    def test_a_feature_that_copes_on_runoff_can_fail_on_rainfall(self):
        from terrainflow_assessment.modules.catchment import SCSRunoff
        scs = SCSRunoff()
        runoff_m = scs.runoff_depth(120.0, scs.adjust_cn(61, "normal")) / 1000.0

        # Sized to just hold the runoff-basis inflow.
        sized = self._capture_at(runoff_m, capacity=1e9)
        needed = sized.per_feature[0]["total_inflow_m3"]

        ok = self._capture_at(runoff_m, capacity=needed * 1.01)
        assert ok.per_feature[0]["overflowed"] is False

        under = self._capture_at(120.0 / 1000.0, capacity=needed * 1.01)
        assert under.per_feature[0]["overflowed"] is True

    def test_wet_antecedent_conditions_roughly_double_the_runoff(self):
        from terrainflow_assessment.modules.catchment import SCSRunoff
        scs = SCSRunoff()
        normal = scs.runoff_depth(120.0, scs.adjust_cn(61, "normal"))
        wet = scs.runoff_depth(120.0, scs.adjust_cn(61, "wet"))
        assert wet > normal * 1.9

    def test_mass_balance_closes_on_either_basis(self):
        for runoff_m in (0.0307, 0.120):
            r = self._capture_at(runoff_m)
            assert r.mass_balance_ok
            assert 0.0 <= r.capture_pct <= 100.0
