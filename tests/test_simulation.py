"""
Tests for terrainflow_assessment/modules/simulation.py

Covers pure-Python logic plus SimulationWorker __init__ and run().
"""
import shutil

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from terrainflow_assessment.modules.simulation import (
    EarthworkStore,
    _find_downslope_store,
    cascade_overflow,
)


# ---------------------------------------------------------------------------
# EarthworkStore
# ---------------------------------------------------------------------------

class TestEarthworkStore:
    def _store(self, name="S1", capacity=100.0, area=20.0,
               infil=4.0, elev=50.0):
        return EarthworkStore(
            name=name,
            ew_type="swale",
            capacity_m3=capacity,
            area_m2=area,
            infiltration_rate_mm_hr=infil,
            elevation=elev,
        )

    def test_defaults(self):
        s = self._store()
        assert s.stored_m3 == 0.0
        assert s.overflowed is False
        assert s.first_overflow_hr is None
        assert s.total_overflow_m3 == 0.0
        assert s.peak_fill_pct == 0.0
        assert s.inflow_m3 == 0.0

    def test_step_infiltration_basic(self):
        s = self._store(area=1000.0, infil=4.0)
        # 4 mm/hr * 1000 m² * 1 hr / 1000 = 4 m³
        result = s.step_infiltration(dt_hr=1.0)
        assert result == pytest.approx(4.0, rel=1e-6)

    def test_step_infiltration_half_hour(self):
        s = self._store(area=1000.0, infil=4.0)
        result = s.step_infiltration(dt_hr=0.5)
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_step_infiltration_zero_area(self):
        s = self._store(area=0.0, infil=4.0)
        assert s.step_infiltration(1.0) == 0.0

    def test_step_infiltration_zero_rate(self):
        s = self._store(area=1000.0, infil=0.0)
        assert s.step_infiltration(1.0) == 0.0

    def test_cut_fill_defaults_zero(self):
        s = self._store()
        assert s.cut_vol_m3 == 0.0
        assert s.fill_vol_m3 == 0.0

    def test_centroid_defaults_none(self):
        s = self._store()
        assert s.centroid_row is None
        assert s.centroid_col is None


# ---------------------------------------------------------------------------
# _find_downslope_store
# ---------------------------------------------------------------------------

class TestFindDownslopeStore:
    def _store(self, name, elevation):
        return EarthworkStore(
            name=name, ew_type="swale",
            capacity_m3=100.0, area_m2=20.0,
            elevation=elevation,
        )

    def test_returns_nearest_lower(self):
        high = self._store("high", 80.0)
        mid = self._store("mid", 60.0)
        low = self._store("low", 40.0)
        # nearest lower from high should be mid (60 > 40)
        result = _find_downslope_store(high, [high, mid, low])
        assert result is mid

    def test_returns_none_when_no_lower(self):
        low = self._store("low", 10.0)
        mid = self._store("mid", 30.0)
        result = _find_downslope_store(low, [low, mid])
        assert result is None

    def test_excludes_self(self):
        s = self._store("only", 50.0)
        result = _find_downslope_store(s, [s])
        assert result is None

    def test_two_candidates_picks_higher_of_lower(self):
        source = self._store("source", 70.0)
        close = self._store("close", 65.0)   # just below source
        far = self._store("far", 30.0)       # much lower
        result = _find_downslope_store(source, [source, close, far])
        assert result is close  # highest elevation still below source

    def test_equal_elevation_not_selected(self):
        a = self._store("A", 50.0)
        b = self._store("B", 50.0)
        result = _find_downslope_store(a, [a, b])
        assert result is None  # equal elevation not strictly lower


# ---------------------------------------------------------------------------
# cascade_overflow
# ---------------------------------------------------------------------------

class TestCascadeOverflow:
    def _store(self, name, elev, capacity=100.0, area=10.0, infil=0.0):
        """Zero infiltration by default to simplify arithmetic."""
        return EarthworkStore(
            name=name, ew_type="swale",
            capacity_m3=capacity, area_m2=area,
            infiltration_rate_mm_hr=infil,
            elevation=elev,
        )

    def test_single_store_below_capacity_no_overflow(self):
        s = self._store("S", elev=50.0, capacity=100.0)
        s.inflow_m3 = 50.0
        exit_m3 = cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert exit_m3 == 0.0
        assert s.stored_m3 == 50.0
        assert s.overflowed is False

    def test_single_store_overflow_exits_site(self):
        s = self._store("S", elev=50.0, capacity=100.0)
        s.inflow_m3 = 150.0
        exit_m3 = cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert exit_m3 == pytest.approx(50.0, rel=1e-6)
        assert s.stored_m3 == pytest.approx(100.0)
        assert s.overflowed is True

    def test_first_overflow_time_recorded(self):
        s = self._store("S", elev=50.0, capacity=50.0)
        s.inflow_m3 = 60.0
        cascade_overflow([s], time_hr=2.0, dt_hr=0.5)
        assert s.first_overflow_hr == 2.0

    def test_first_overflow_not_overwritten_on_second_event(self):
        s = self._store("S", elev=50.0, capacity=50.0)
        s.inflow_m3 = 60.0
        cascade_overflow([s], time_hr=1.0, dt_hr=0.5)
        s.inflow_m3 = 60.0
        cascade_overflow([s], time_hr=2.0, dt_hr=0.5)
        assert s.first_overflow_hr == 1.0  # first event preserved

    def test_overflow_cascades_to_downstream(self):
        high = self._store("high", elev=80.0, capacity=50.0)
        low = self._store("low", elev=40.0, capacity=200.0)
        high.inflow_m3 = 100.0  # 50 m³ overflow
        exit_m3 = cascade_overflow([high, low], time_hr=1.0, dt_hr=1.0)
        # 50 m³ overflows high → goes to low (lower elevation)
        assert exit_m3 == 0.0  # nothing exits since low has enough capacity
        assert low.stored_m3 == pytest.approx(50.0, abs=0.1)

    def test_cascade_chain(self):
        """Three stores in a chain: overflow propagates all the way out."""
        top = self._store("top", elev=100.0, capacity=50.0)
        mid = self._store("mid", elev=60.0, capacity=50.0)
        bot = self._store("bot", elev=20.0, capacity=50.0)

        top.inflow_m3 = 200.0   # 150 overflows from top
        # mid receives 150, overflows 100 to bot
        # bot receives 100, overflows 50 to exit
        exit_m3 = cascade_overflow([top, mid, bot], time_hr=1.0, dt_hr=1.0)
        assert exit_m3 == pytest.approx(50.0, abs=1.0)

    def test_stores_processed_highest_elevation_first(self):
        """Processing order: highest elevation first (top-down)."""
        low = self._store("low", elev=10.0, capacity=200.0)
        high = self._store("high", elev=90.0, capacity=50.0)
        high.inflow_m3 = 80.0  # 30 overflow → should go to low
        cascade_overflow([low, high], time_hr=1.0, dt_hr=1.0)
        assert high.stored_m3 == 50.0
        assert low.stored_m3 == pytest.approx(30.0, abs=0.1)

    def test_inflow_reset_after_step(self):
        s = self._store("S", elev=50.0, capacity=100.0)
        s.inflow_m3 = 20.0
        cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert s.inflow_m3 == 0.0  # reset for next step

    def test_total_inflow_accumulated(self):
        s = self._store("S", elev=50.0, capacity=200.0)
        for _ in range(3):
            s.inflow_m3 = 10.0
            cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert s.total_inflow_m3 == pytest.approx(30.0, rel=1e-6)

    def test_total_overflow_accumulated(self):
        s = self._store("S", elev=50.0, capacity=20.0)
        for _ in range(3):
            s.inflow_m3 = 30.0
            cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert s.total_overflow_m3 > 0.0

    def test_peak_fill_pct_updated(self):
        s = self._store("S", elev=50.0, capacity=100.0)
        s.inflow_m3 = 70.0
        cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert s.peak_fill_pct == pytest.approx(70.0, rel=1e-3)


# ---------------------------------------------------------------------------
# SimulationWorker.__init__ and run()
# ---------------------------------------------------------------------------

def _make_minimal_dem(tmp_path, name="dem.tif"):
    data = np.full((5, 5), 50.0, dtype="float32")
    h, w = data.shape
    transform = from_bounds(0, 0, 5, 5, w, h)
    path = str(tmp_path / name)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs="EPSG:32632",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


class TestSimulationWorker:
    def _make_worker(self, tmp_path):
        from terrainflow_assessment.modules.simulation import SimulationWorker
        dem = _make_minimal_dem(tmp_path)
        fdir = str(tmp_path / "fdir.tif")
        shutil.copy(dem, fdir)
        return SimulationWorker(
            dem_path=dem,
            fdir_path=fdir,
            output_dir=str(tmp_path / "out"),
            cn=75,
            moisture="normal",
            rainfall_data=[(0, 0.0), (30, 20.0)],
        )

    def test_init_cn(self, tmp_path):
        w = self._make_worker(tmp_path)
        assert w.cn == 75

    def test_init_moisture(self, tmp_path):
        w = self._make_worker(tmp_path)
        assert w.moisture == "normal"

    def test_init_default_routing(self, tmp_path):
        w = self._make_worker(tmp_path)
        assert w.routing == "dinf"

    def test_init_default_cn_zones_empty(self, tmp_path):
        w = self._make_worker(tmp_path)
        assert w.cn_zones_data == []

    def test_init_default_stores_empty(self, tmp_path):
        w = self._make_worker(tmp_path)
        assert w.earthwork_stores == []

    def test_init_default_soil_loam(self, tmp_path):
        w = self._make_worker(tmp_path)
        assert w.soil_name == "Loam"

    def test_init_custom_routing(self, tmp_path):
        from terrainflow_assessment.modules.simulation import SimulationWorker
        dem = _make_minimal_dem(tmp_path, "dem2.tif")
        w = SimulationWorker(
            dem_path=dem, fdir_path=dem,
            output_dir=str(tmp_path),
            cn=80, moisture="dry",
            rainfall_data=[(0, 0.0), (60, 50.0)],
            routing="d8",
            cn_zones_data=[{"wkt": "POLYGON((0 0, 1 0, 1 1, 0 0))", "cn": 85}],
        )
        assert w.routing == "d8"
        assert len(w.cn_zones_data) == 1

    def test_run_does_not_raise(self, tmp_path):
        """run() catches any internal error and does not propagate it."""
        w = self._make_worker(tmp_path)
        w.run()  # pysheds may fail but error is caught by try/except in run()

    def test_run_with_stores_does_not_raise(self, tmp_path):
        from terrainflow_assessment.modules.simulation import SimulationWorker
        dem = _make_minimal_dem(tmp_path, "dem3.tif")
        store = EarthworkStore(
            name="S1", ew_type="swale",
            capacity_m3=100.0, area_m2=20.0, elevation=50.0,
        )
        w = SimulationWorker(
            dem_path=dem, fdir_path=dem,
            output_dir=str(tmp_path / "out2"),
            cn=75, moisture="normal",
            rainfall_data=[(0, 0.0), (30, 25.0)],
            earthwork_stores=[store],
        )
        w.run()


# ---------------------------------------------------------------------------
# Remaining TestCascadeOverflow tests (accidentally orphaned — restored here)
# ---------------------------------------------------------------------------

class TestCascadeOverflowExtended:
    def _store(self, name, elev, capacity=100.0, area=10.0, infil=0.0):
        return EarthworkStore(
            name=name, ew_type="swale",
            capacity_m3=capacity, area_m2=area,
            infiltration_rate_mm_hr=infil,
            elevation=elev,
        )

    def test_peak_fill_pct_not_decreased(self):
        s = self._store("S", elev=50.0, capacity=100.0)
        s.inflow_m3 = 80.0
        cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        s.inflow_m3 = 0.0
        cascade_overflow([s], time_hr=2.0, dt_hr=1.0)
        assert s.peak_fill_pct == pytest.approx(80.0, rel=1e-3)

    def test_infiltration_reduces_stored(self):
        s = self._store("S", elev=50.0, capacity=100.0, area=1000.0, infil=4.0)
        s.inflow_m3 = 10.0
        # infiltration over 1 hr = 4mm/hr * 1000m² / 1000 = 4 m³
        cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        # stored = 10 - 4 = 6 m³
        assert s.stored_m3 == pytest.approx(6.0, abs=0.1)

    def test_infiltration_capped_at_available_water(self):
        """Infiltration can't remove more water than is present."""
        s = self._store("S", elev=50.0, capacity=100.0, area=100_000.0, infil=50.0)
        s.inflow_m3 = 1.0  # tiny inflow, huge infiltration
        cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert s.stored_m3 >= 0.0  # never negative

    def test_zero_capacity_store_all_exits(self):
        s = self._store("S", elev=50.0, capacity=0.0)
        s.inflow_m3 = 30.0
        exit_m3 = cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert exit_m3 == pytest.approx(30.0, abs=0.1)

    def test_total_infiltration_accumulated(self):
        s = self._store("S", elev=50.0, capacity=200.0, area=500.0, infil=4.0)
        for _ in range(4):
            s.inflow_m3 = 5.0
            cascade_overflow([s], time_hr=1.0, dt_hr=1.0)
        assert s.total_infiltration_m3 > 0.0
