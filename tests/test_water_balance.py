"""Tests for terrainflow_assessment/modules/water_balance.py — analytical routed balance."""
import pytest

from terrainflow_assessment.modules.simulation import EarthworkStore
from terrainflow_assessment.modules.water_balance import run_water_balance


def _store(name, capacity, elevation, inflow, area=100.0, infil=0.0, target=None):
    s = EarthworkStore(
        name=name, ew_type="swale", capacity_m3=capacity, area_m2=area,
        infiltration_rate_mm_hr=infil, elevation=elevation,
        id=name, overflow_target_id=target,
    )
    s.inflow_m3 = inflow
    return s


class TestRunWaterBalance:
    def test_single_feature_capture_pct(self):
        # inflow 50 into a 100 m³ store, no infiltration; runoff denominator 100 → 50%
        r = run_water_balance([_store("S1", 100.0, 10.0, 50.0)], 1.0, total_runoff_m3=100.0)
        assert r.total_captured_m3 == pytest.approx(50.0)
        assert r.capture_pct == pytest.approx(50.0)
        assert r.site_exit_m3 == pytest.approx(50.0)
        assert r.total_capacity_m3 == pytest.approx(100.0)

    def test_infiltration_scales_with_duration(self):
        # rate 10 mm/hr × 100 m² × 2 hr = 2 m³ infiltrated
        r = run_water_balance(
            [_store("S1", 1000.0, 10.0, 500.0, area=100.0, infil=10.0)],
            2.0, total_runoff_m3=500.0,
        )
        assert r.total_infiltration_m3 == pytest.approx(2.0)
        assert r.total_captured_m3 == pytest.approx(500.0)  # 498 stored + 2 infiltrated

    def test_interception_reduces_downstream_and_overflow_cascades(self):
        # S1 (upslope) intercepts part of the shared catchment; S2's inflow drops, and
        # S1's overflow cascades into S2.
        s1 = _store("S1", 30.0, 20.0, 50.0)   # own catchment 50
        s2 = _store("S2", 100.0, 10.0, 80.0)  # cumulative (includes S1's 50)
        r = run_water_balance([s1, s2], 1.0, total_runoff_m3=80.0)

        by_name = {f["name"]: f for f in r.per_feature}
        # S2's corrected inflow = 80 − 30 (captured by S1) = 50 (< raw 80 → interception)
        assert by_name["S2"]["inflow_m3"] == pytest.approx(50.0)
        assert by_name["S1"]["overflowed"] is True         # 50 into a 30 store
        # S1 stores 30; S2 stores its 50 + S1's 20 overflow = 70
        assert by_name["S1"]["stored_m3"] == pytest.approx(30.0)
        assert by_name["S2"]["stored_m3"] == pytest.approx(70.0)
        assert r.total_captured_m3 == pytest.approx(100.0)

    def test_no_runoff_gives_zero_capture(self):
        # geometry-only (no baseline): capture % is 0 but capacity still reported
        r = run_water_balance([_store("S1", 100.0, 10.0, 0.0)], 1.0, total_runoff_m3=0.0)
        assert r.capture_pct == 0.0
        assert r.site_exit_m3 == 0.0
        assert r.total_capacity_m3 == pytest.approx(100.0)

    def test_empty_stores_all_zero(self):
        r = run_water_balance([], 1.0, total_runoff_m3=100.0)
        assert r.capture_pct == 0.0
        assert r.total_capacity_m3 == 0.0
        assert r.per_feature == []

    def test_capture_pct_capped_at_100(self):
        # captured could exceed a tiny runoff denominator → clamp to 100%
        r = run_water_balance([_store("S1", 100.0, 10.0, 50.0)], 1.0, total_runoff_m3=10.0)
        assert r.capture_pct == 100.0

    def test_zero_capacity_feature_fill_pct_safe(self):
        # capacity 0 must not divide by zero in the per-feature fill %
        r = run_water_balance([_store("S0", 0.0, 10.0, 5.0)], 1.0, total_runoff_m3=5.0)
        assert r.per_feature[0]["fill_pct"] == 0.0
