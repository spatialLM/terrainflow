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
    layer_nodes,
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
# _find_downslope_store — user-specified overflow links
# ---------------------------------------------------------------------------

class TestFindDownslopeStoreLinked:
    def _store(self, name, elevation, target=None):
        return EarthworkStore(
            name=name, ew_type="swale", capacity_m3=100.0, area_m2=20.0,
            elevation=elevation, id=name, overflow_target_id=target,
        )

    def test_downhill_link_overrides_nearest(self):
        # source links to 'far' (a lower store) → routes there, not the nearest-lower 'close'
        source = self._store("source", 70.0, target="far")
        close = self._store("close", 65.0)
        far = self._store("far", 30.0)
        assert _find_downslope_store(source, [source, close, far]) is far

    def test_uphill_link_ignored_falls_back_to_elevation(self):
        # link points uphill (invalid for a single-pass cascade) → elevation heuristic
        source = self._store("source", 30.0, target="high")
        high = self._store("high", 70.0)
        low = self._store("low", 10.0)
        assert _find_downslope_store(source, [source, high, low]) is low

    def test_missing_link_target_falls_back_to_elevation(self):
        source = self._store("source", 70.0, target="ghost")
        mid = self._store("mid", 50.0)
        assert _find_downslope_store(source, [source, mid]) is mid


class TestOverflowGraph:
    """Routing now follows the flow path, not centroid elevation.

    The auto target comes from an injected *walker* that traces where water actually
    goes from a feature's outlet cell. Without one there is no auto target at all —
    deliberately: the old "highest feature below this one" heuristic happily linked
    features on opposite sides of a ridge, which is why it was removed rather than
    kept as a fallback.
    """

    def _store(self, name, elevation, target=None):
        return EarthworkStore(
            name=name, ew_type="swale", capacity_m3=100.0, area_m2=20.0,
            elevation=elevation, id=name, overflow_target_id=target,
        )

    def test_auto_edge_follows_the_walked_flow_path(self):
        from terrainflow_assessment.modules.simulation import overflow_graph
        hi = self._store("hi", 70.0)
        lo = self._store("lo", 40.0)
        g = overflow_graph([hi, lo], walker=lambda s: "lo" if s.id == "hi" else None)
        assert g["hi"] == ("lo", False)      # auto edge, from the flow path
        assert g["lo"] == (None, False)      # nothing downstream → exits site

    def test_no_walker_means_no_guessed_edges(self):
        from terrainflow_assessment.modules.simulation import overflow_graph
        g = overflow_graph([self._store("hi", 70.0), self._store("lo", 40.0)])
        assert g["hi"] == (None, False)      # elevation alone is not evidence of a path

    def test_user_link_flagged_and_overrides_the_walk(self):
        from terrainflow_assessment.modules.simulation import overflow_graph
        src = self._store("src", 70.0, target="far")
        close = self._store("close", 65.0)
        far = self._store("far", 30.0)
        walker = {"src": "close", "close": "far"}.get
        g = overflow_graph([src, close, far], walker=lambda s: walker(s.id))
        assert g["src"] == ("far", True)     # honoured user link, beating the walk
        assert g["close"] == ("far", False)  # auto

    def test_uphill_user_link_is_honoured(self):
        """The old "target must be lower" test is gone — see resolve_targets.

        Centroid elevation is the wrong proxy once real flow paths are used: a large
        tilted basin's centroid can sit above a swale that genuinely drains into it.
        """
        from terrainflow_assessment.modules.simulation import overflow_graph
        src = self._store("src", 30.0, target="high")
        high = self._store("high", 70.0)
        low = self._store("low", 10.0)
        g = overflow_graph([src, high, low], walker=lambda s: "low" if s.id == "src" else None)
        assert g["src"] == ("high", True)

    def test_missing_link_target_falls_back_to_the_walk(self):
        from terrainflow_assessment.modules.simulation import overflow_graph
        src = self._store("src", 70.0, target="ghost")
        mid = self._store("mid", 50.0)
        g = overflow_graph([src, mid], walker=lambda s: "mid" if s.id == "src" else None)
        assert g["src"] == ("mid", False)

    def test_store_without_id_skipped(self):
        from terrainflow_assessment.modules.simulation import overflow_graph
        s = EarthworkStore(name="x", ew_type="swale", capacity_m3=1.0,
                           area_m2=1.0, elevation=10.0, id=None)
        assert overflow_graph([s]) == {}


class TestResolveTargets:
    def _store(self, name, target=None):
        return EarthworkStore(
            name=name, ew_type="swale", capacity_m3=100.0, area_m2=20.0,
            elevation=10.0, id=name, overflow_target_id=target,
        )

    def test_order_puts_upstream_first(self):
        from terrainflow_assessment.modules.simulation import resolve_targets
        a, b, c = self._store("a"), self._store("b"), self._store("c")
        links = {"a": "b", "b": "c"}
        r = resolve_targets([c, b, a], walker=lambda s: links.get(s.id))
        assert r.order.index("a") < r.order.index("b") < r.order.index("c")
        assert r.warnings == []

    def test_cyclic_user_link_is_demoted_and_warned(self):
        from terrainflow_assessment.modules.simulation import resolve_targets
        a = self._store("a", target="b")
        b = self._store("b", target="a")
        r = resolve_targets([a, b], walker=lambda s: None)
        assert r.warnings and "loop" in r.warnings[0].lower()
        # Only the offender is undone; the graph is acyclic again.
        assert not (r.edges["a"] == "b" and r.edges["b"] == "a")
        assert r.is_user["b"] is False or r.is_user["a"] is False

    def test_self_link_is_ignored(self):
        from terrainflow_assessment.modules.simulation import resolve_targets
        r = resolve_targets([self._store("a", target="a")], walker=lambda s: None)
        assert r.edges["a"] is None

    def test_a_walker_that_raises_degrades_to_no_edge(self):
        from terrainflow_assessment.modules.simulation import resolve_targets

        def boom(_store):
            raise RuntimeError("raster gone")

        r = resolve_targets([self._store("a")], walker=boom)
        assert r.edges["a"] is None


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


# ---------------------------------------------------------------------------
# _run_simulation — direct invocation for branch coverage
# ---------------------------------------------------------------------------

def _make_sloped_dem(tmp_path, name="dem.tif", shape=(10, 10)):
    data = np.fromfunction(
        lambda r, c: 100.0 - r * 0.5 - c * 0.2, shape
    ).astype("float32")
    h, w = data.shape
    transform = from_bounds(0, 0, w, h, w, h)
    path = str(tmp_path / name)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs="EPSG:32632",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


class TestFdirNodata:
    """How the simulation decides what "no direction" means when it reads a grid.

    Never by leaving it to pysheds, which assumes 0 — the D-infinity angle for
    due east — and would retire every east-flowing cell in the catchment.
    """

    def test_prefers_what_the_file_declares(self, tmp_path):
        from terrainflow_assessment.modules.simulation import _fdir_nodata
        path = _make_sloped_dem(tmp_path, "tagged.tif")   # written with nodata=-9999
        assert _fdir_nodata(path, "dinf") == -9999.0

    def test_falls_back_to_the_routing_rule_for_an_untagged_file(self, tmp_path):
        from terrainflow_assessment.modules.simulation import _fdir_nodata
        path = str(tmp_path / "untagged.tif")
        with rasterio.open(
            path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="float32", crs="EPSG:32632",
            transform=from_bounds(0, 0, 4, 4, 4, 4),
        ) as dst:
            dst.write(np.zeros((4, 4), dtype="float32"), 1)
        assert np.isnan(_fdir_nodata(path, "dinf"))
        assert _fdir_nodata(path, "d8") == 0

    def test_falls_back_when_the_file_cannot_be_opened(self, tmp_path):
        from terrainflow_assessment.modules.simulation import _fdir_nodata
        missing = str(tmp_path / "does_not_exist.tif")
        assert np.isnan(_fdir_nodata(missing, "dinf"))
        assert _fdir_nodata(missing, "d8") == 0


class TestRunSimulationBranches:
    def test_rainfall_too_short_raises_value_error(self, tmp_path):
        """Line 282: ValueError when rainfall_data has <2 entries."""
        from terrainflow_assessment.modules.simulation import _run_simulation
        dem = _make_sloped_dem(tmp_path)
        fdir = str(tmp_path / "fdir.tif")
        shutil.copy(dem, fdir)
        with pytest.raises(ValueError, match="rainfall_data"):
            _run_simulation(
                dem_path=dem, fdir_path=fdir,
                output_dir=str(tmp_path / "out"),
                cn=75, moisture="normal",
                rainfall_data=[(0, 0.0)],
            )

    def test_cn_zones_processed_including_bad_wkt(self, tmp_path):
        """Lines 256-265: cn_zones wkt loads + exception branch."""
        from terrainflow_assessment.modules.simulation import _run_simulation
        dem = _make_sloped_dem(tmp_path, "zone_dem.tif")
        fdir = str(tmp_path / "zone_fdir.tif")
        shutil.copy(dem, fdir)
        cn_zones = [
            {"wkt": "POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))", "cn": 85},
            {"wkt": "NOT_VALID_WKT", "cn": 80},  # triggers except at 263-264
        ]
        try:
            _run_simulation(
                dem_path=dem, fdir_path=fdir,
                output_dir=str(tmp_path / "out_z"),
                cn=75, moisture="normal",
                rainfall_data=[(0, 0.0)],  # short-circuit at ValueError after CN build
                cn_zones_data=cn_zones,
            )
        except ValueError:
            pass  # Expected — rainfall too short, but cn_zones already processed

    def test_stale_sim_dir_files_cleaned(self, tmp_path):
        """Lines 274-278: os.listdir + os.remove loop for stale files."""
        from terrainflow_assessment.modules.simulation import _run_simulation
        dem = _make_sloped_dem(tmp_path, "clean_dem.tif")
        fdir = str(tmp_path / "clean_fdir.tif")
        shutil.copy(dem, fdir)
        out_dir = str(tmp_path / "out_clean")
        sim_dir = os.path.join(out_dir, "simulation")
        os.makedirs(sim_dir, exist_ok=True)
        # Pre-existing junk file that the loop should clean
        stale = os.path.join(sim_dir, "junk.tif")
        with open(stale, "w") as f:
            f.write("old")
        try:
            _run_simulation(
                dem_path=dem, fdir_path=fdir,
                output_dir=out_dir,
                cn=75, moisture="normal",
                rainfall_data=[(0, 0.0)],
            )
        except ValueError:
            pass
        assert not os.path.exists(stale)


# ---------------------------------------------------------------------------
# SimulationWorker.run() ValueError path (lines 214-215)
# ---------------------------------------------------------------------------

class TestSimulationWorkerRunErrorBranch:
    def test_run_emits_error_on_value_error(self, tmp_path):
        """run() should catch the ValueError from _run_simulation."""
        from terrainflow_assessment.modules.simulation import SimulationWorker
        dem = _make_sloped_dem(tmp_path, "errdem.tif")
        fdir = str(tmp_path / "errfdir.tif")
        shutil.copy(dem, fdir)
        w = SimulationWorker(
            dem_path=dem, fdir_path=fdir,
            output_dir=str(tmp_path / "err_out"),
            cn=75, moisture="normal",
            rainfall_data=[(0, 0.0)],  # Too short → ValueError → caught in run()
        )
        errs = []
        w.error.connect(lambda e: errs.append(e))
        w.run()
        assert len(errs) == 1
        assert "rainfall_data" in errs[0] or "ValueError" in errs[0]


# ---------------------------------------------------------------------------
# build_stores_from_earthworks branches (lines 503, 520-524)
# ---------------------------------------------------------------------------

import os  # noqa: E402 — used by tests above


class TestBuildStoresFromEarthworks:
    def _line_ew(self, ew_type="swale"):
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        from tests.conftest import make_mock_line_geom
        g = make_mock_line_geom([(1.0, 2.5), (4.0, 2.5)])
        ew = Earthwork(ew_type, g, "S1")
        ew.capacity_m3 = 50.0
        return ew

    def _poly_ew(self, ew_type="basin"):
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        from tests.conftest import make_mock_polygon_geom
        g = make_mock_polygon_geom((1.0, 1.0, 4.0, 4.0))
        ew = Earthwork(ew_type, g, "B1")
        ew.capacity_m3 = 30.0
        return ew

    def test_empty_list(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        assert build_stores_from_earthworks([]) == []

    def test_overflow_linkage_carried_into_store(self):
        """id + overflow_target_id flow from Earthwork onto its EarthworkStore."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        upstream = self._line_ew()
        downstream = self._line_ew()
        upstream.overflow_target_id = downstream.id
        stores = build_stores_from_earthworks([upstream], soil_name="Loam")
        assert stores[0].id == upstream.id
        assert stores[0].overflow_target_id == downstream.id

    def test_disabled_earthworks_skipped(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        ew = self._line_ew()
        ew.enabled = False
        assert build_stores_from_earthworks([ew]) == []

    def test_zero_capacity_kept_as_a_routing_node(self):
        """Berms and diversions store nothing but still route — they must not be dropped."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        ew = self._line_ew()
        ew.capacity_m3 = 0.0
        stores = build_stores_from_earthworks([ew])
        assert len(stores) == 1
        assert stores[0].capacity_m3 == 0.0

    def test_disabled_earthwork_is_skipped(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        ew = self._line_ew()
        ew.enabled = False
        assert build_stores_from_earthworks([ew]) == []

    def test_line_geom_area_from_polygon_buffer(self):
        """Polygon branch — uses shapely_geom.area."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        stores = build_stores_from_earthworks([self._poly_ew()], soil_name="Loam")
        assert len(stores) == 1
        assert stores[0].area_m2 > 0

    def test_with_dem_path_sets_centroid(self, tmp_path):
        """Lines 520-524: dem_path sets elevation and centroid row/col."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        dem = _make_sloped_dem(tmp_path, "centroid_dem.tif")
        stores = build_stores_from_earthworks(
            [self._line_ew()], soil_name="Loam", dem_path=dem,
        )
        assert len(stores) == 1
        assert stores[0].centroid_row is not None
        assert stores[0].centroid_col is not None
        assert stores[0].elevation != 0.0  # sampled from DEM
        assert stores[0].elevation_known is True

    def test_line_feature_gets_a_real_wetted_area(self):
        """SIM-26: a LineString's `.area` is 0.0, so testing for the attribute made
        the length × width branch unreachable. Swales, diversions and berms are all
        polylines, so every one of them was credited zero infiltration."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        ew = self._line_ew("swale")            # 3 m long
        stores = build_stores_from_earthworks([ew], soil_name="Loam")
        assert stores[0].area_m2 == pytest.approx(3.0 * ew.width)
        assert stores[0].step_infiltration(1.0) > 0

    def test_nodata_centroid_elevation_is_not_treated_as_known(self, tmp_path):
        """SIM-27: a −9999 elevation sorts below every real feature, so that store
        silently became the receiver for the whole site's overflow."""
        from terrainflow_assessment.modules.simulation import (
            _find_downslope_store,
            build_stores_from_earthworks,
        )
        data = np.full((10, 10), -9999.0, dtype="float32")
        path = str(tmp_path / "allnodata.tif")
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="float32", crs="EPSG:32632",
            transform=from_bounds(0, 0, 10, 10, 10, 10), nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        stores = build_stores_from_earthworks(
            [self._line_ew()], soil_name="Loam", dem_path=path,
        )
        blind = stores[0]
        assert blind.elevation_known is False

        real = build_stores_from_earthworks(
            [self._line_ew()], soil_name="Loam",
            dem_path=_make_sloped_dem(tmp_path, "real.tif"),
        )[0]
        # The unknown-elevation store neither receives nor routes by elevation.
        assert _find_downslope_store(real, [real, blind]) is None
        assert _find_downslope_store(blind, [real, blind]) is None

    def test_with_dem_path_outside_bounds(self, tmp_path):
        """Centroid outside DEM extent → centroid_row/col stay None."""
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        from tests.conftest import make_mock_line_geom
        dem = _make_sloped_dem(tmp_path, "oob_dem.tif")
        # Line far outside DEM bounds
        g = make_mock_line_geom([(500.0, 500.0), (510.0, 500.0)])
        ew = Earthwork("swale", g, "Far")
        ew.capacity_m3 = 10.0
        stores = build_stores_from_earthworks([ew], soil_name="Loam", dem_path=dem)
        assert len(stores) == 1
        assert stores[0].centroid_row is None
        assert stores[0].centroid_col is None

    def test_with_invalid_dem_path_except_branch(self):
        """Unreadable dem_path triggers except branch; row/col remain None."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        stores = build_stores_from_earthworks(
            [self._line_ew()], soil_name="Loam", dem_path="/nonexistent.tif",
        )
        assert stores[0].centroid_row is None

    def test_bad_geometry_falls_back_to_100(self):
        """asJson raising triggers except → area_m2 = 100.0 fallback."""
        from unittest.mock import MagicMock

        from terrainflow_assessment.modules.earthwork_design import Earthwork
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        g = MagicMock()
        g.asJson.side_effect = RuntimeError("bad geom")
        ew = Earthwork("swale", g, "Bad")
        ew.capacity_m3 = 10.0
        stores = build_stores_from_earthworks([ew], soil_name="Loam")
        assert len(stores) == 1
        assert stores[0].area_m2 == 100.0


# ---------------------------------------------------------------------------
# Phase 1 regression — item 5: stacked stores must conserve mass
# ---------------------------------------------------------------------------

class TestStackedStoresConserveMass:
    """Phase 1 item 5: cascade_overflow must not create water from nothing."""

    def _store(self, name, elev, capacity=100.0):
        return EarthworkStore(
            name=name, ew_type="swale",
            capacity_m3=capacity, area_m2=1.0,
            infiltration_rate_mm_hr=0.0,  # no infiltration → pure mass check
            elevation=elev,
        )

    def test_stacked_stores_conserve_mass(self):
        """
        Two stacked stores: upper (high elev) gets 150 m³, lower gets 0.
        After cascade: total (stored + site exit) == 150 m³.
        """
        upper = self._store("Upper", elev=70.0, capacity=100.0)
        lower = self._store("Lower", elev=50.0, capacity=100.0)

        total_inflow = 150.0
        upper.inflow_m3 = total_inflow
        lower.inflow_m3 = 0.0  # only receives overflow from upper

        exit_m3 = cascade_overflow(
            stores=[upper, lower], time_hr=1.0, dt_hr=1.0
        )

        total_out = upper.stored_m3 + lower.stored_m3 + exit_m3
        assert total_out == pytest.approx(total_inflow, rel=1e-6), (
            f"Mass not conserved: in={total_inflow}, "
            f"stored_upper={upper.stored_m3}, stored_lower={lower.stored_m3}, "
            f"exit={exit_m3}, total_out={total_out}"
        )

    def test_stacked_stores_correct_cascade_split(self):
        """Upper fills to capacity, overflow routes correctly to lower."""
        upper = self._store("Upper", elev=70.0, capacity=100.0)
        lower = self._store("Lower", elev=50.0, capacity=100.0)

        upper.inflow_m3 = 150.0   # 50 m³ overflows to lower
        exit_m3 = cascade_overflow([upper, lower], time_hr=1.0, dt_hr=1.0)

        assert upper.stored_m3 == pytest.approx(100.0)
        assert lower.stored_m3 == pytest.approx(50.0)
        assert exit_m3 == pytest.approx(0.0)

    def test_three_stacked_stores_conserve_mass(self):
        """Three stacked stores: mass flows through cascade without creation."""
        top = self._store("Top", elev=100.0, capacity=50.0)
        mid = self._store("Mid", elev=70.0, capacity=50.0)
        bot = self._store("Bot", elev=40.0, capacity=50.0)

        total_inflow = 200.0
        top.inflow_m3 = total_inflow

        exit_m3 = cascade_overflow([top, mid, bot], time_hr=1.0, dt_hr=1.0)
        total_out = top.stored_m3 + mid.stored_m3 + bot.stored_m3 + exit_m3
        assert total_out == pytest.approx(total_inflow, rel=1e-6)


class TestLayerNodes:
    """Rank/order for the Flow chart. The layout maths lives in the module rather
    than the widget so it can be checked without Qt — and because getting it wrong
    draws water running uphill."""

    def test_a_chain_ranks_in_order(self):
        assert layer_nodes(["a", "b", "c"], {"a": "b", "b": "c", "c": None}) == {
            "a": (0, 0), "b": (1, 0), "c": (2, 0)}

    def test_two_sources_share_the_top_rank(self):
        layout = layer_nodes(["a", "b", "c"], {"a": "c", "b": "c", "c": None})
        assert layout["a"][0] == layout["b"][0] == 0
        assert layout["c"][0] == 1

    def test_order_separates_nodes_sharing_a_rank(self):
        layout = layer_nodes(["a", "b", "c"], {"a": "c", "b": "c", "c": None})
        assert {layout["a"][1], layout["b"][1]} == {0, 1}

    def test_rank_is_the_longest_path_not_the_shortest(self):
        """A feature fed by both a rank-0 swale and a rank-1 dam belongs below the
        dam. Ranking by shortest path would draw water flowing upwards into it."""
        layout = layer_nodes(["a", "b", "c", "d"],
                             {"a": "b", "b": "d", "c": "d", "d": None})
        assert layout["d"][0] == 2
        assert layout["c"][0] == 0

    def test_a_terminal_feature_still_gets_a_rank(self):
        assert layer_nodes(["a"], {"a": None}) == {"a": (0, 0)}

    def test_order_follows_the_callers_sequence(self):
        """So the chart does not reshuffle between refreshes when nothing changed."""
        ids = ["x", "y", "z"]
        first = layer_nodes(ids, {"x": None, "y": None, "z": None})
        again = layer_nodes(ids, {"x": None, "y": None, "z": None})
        assert first == again
        assert [first[i][1] for i in ids] == [0, 1, 2]

    def test_a_cycle_does_not_hang_and_sits_below_the_rest(self):
        layout = layer_nodes(["a", "b", "c"], {"a": "b", "b": "a", "c": None})
        assert layout["c"][0] == 0
        assert layout["a"][0] == layout["b"][0] > 0

    def test_a_target_outside_the_node_set_is_ignored(self):
        assert layer_nodes(["a"], {"a": "ghost"}) == {"a": (0, 0)}

    def test_a_self_link_does_not_promote_a_node(self):
        assert layer_nodes(["a"], {"a": "a"}) == {"a": (0, 0)}

    def test_an_empty_network_lays_out_to_nothing(self):
        assert layer_nodes([], {}) == {}

    def test_a_missing_edge_entry_is_treated_as_terminal(self):
        layout = layer_nodes(["a", "b"], {"a": "b"})
        assert layout["b"][0] == 1
