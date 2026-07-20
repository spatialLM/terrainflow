"""
Comprehensive tests for terrainflow_assessment/modules/flow_analysis.py.

Covers FlowAnalysis (all public methods + private helpers) and AnalysisWorker
(the QThread wrapper is exercised through _do_analysis, which doesn't require
a real Qt event loop because the conftest stubs out QThread).
"""
import os
from unittest.mock import patch, MagicMock

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon, mapping


# ---------------------------------------------------------------------------
# Raster + boundary fixtures
# ---------------------------------------------------------------------------

def _write_raster(path, data, cell_size=1.0, crs="EPSG:32632", nodata=-9999.0):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


@pytest.fixture
def sloped_dem(tmp_path):
    """25×25 DEM sloping smoothly south with a valley column."""
    r_idx = np.arange(25).reshape(-1, 1)
    c_idx = np.arange(25).reshape(1, -1)
    data = (100.0 - r_idx * 2.0 + np.abs(c_idx - 12) * 0.5).astype("float32")
    return _write_raster(str(tmp_path / "dem.tif"), data)


@pytest.fixture
def boundary_gpkg(tmp_path):
    """A site boundary polygon covering most of the sloped DEM."""
    poly = Polygon([(2, 2), (23, 2), (23, 23), (2, 23), (2, 2)])
    gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:32632")
    path = str(tmp_path / "boundary.gpkg")
    gdf.to_file(path, driver="GPKG")
    return path


# ---------------------------------------------------------------------------
# FlowAnalysis.__init__ / load_dem
# ---------------------------------------------------------------------------

class TestInit:
    def test_defaults(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.grid is None
        assert fa.dem is None
        assert fa.fdir is None
        assert fa.acc is None
        assert fa.routing == "dinf"

    def test_load_dem_returns_true(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        assert fa.load_dem(sloped_dem) is True

    def test_load_dem_sets_grid_and_dem(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        assert fa.grid is not None
        assert fa.dem is not None
        assert fa.crs is not None
        assert fa.transform is not None


# ---------------------------------------------------------------------------
# FlowAnalysis.run
# ---------------------------------------------------------------------------

class TestRun:
    def test_run_raises_without_load(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="DEM not loaded"):
            fa.run()

    def test_run_default_dinf(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        result = fa.run()
        assert "flow_direction" in result
        assert "flow_accumulation" in result
        assert fa.fdir is not None
        assert fa.acc is not None

    def test_run_with_runoff_weights(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        weights = np.ones((25, 25), dtype="float32") * 0.5
        result = fa.run(runoff_weights=weights)
        assert "runoff_accumulation" in result

    def test_run_flowdir_typeerror_fallback(self, sloped_dem):
        """When grid.flowdir rejects routing kwarg, fall back without it.

        Both flowdir and accumulation are wrapped so they raise TypeError on
        the routing-kwarg call and return a safe dinf result otherwise — this
        exercises both except branches without invoking broken pysheds d8.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        orig_flowdir = fa.grid.flowdir
        orig_acc = fa.grid.accumulation

        def _flowdir_no_routing(*args, **kwargs):
            if "routing" in kwargs:
                raise TypeError("routing kwarg not supported")
            return orig_flowdir(*args, routing="dinf")

        def _acc_no_routing(*args, **kwargs):
            # Strip routing kwarg entirely so we stay on dinf (d8 is broken)
            kwargs.pop("routing", None)
            return orig_acc(*args, **kwargs, routing="dinf")

        fa.grid.flowdir = _flowdir_no_routing
        fa.grid.accumulation = _acc_no_routing
        fa.run()
        # After the TypeError branch, self.routing is set to 'd8'
        assert fa.routing == "d8"

    def test_run_accumulation_typeerror_fallback(self, sloped_dem):
        """When grid.accumulation rejects routing kwarg, fall back without it."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        orig_acc = fa.grid.accumulation
        calls = {"n": 0}

        def _acc_picky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1 and "routing" in kwargs:
                raise TypeError("routing kwarg rejected")
            # Second call: force dinf so we bypass pysheds d8 bug
            kwargs["routing"] = "dinf"
            return orig_acc(*args, **kwargs)

        fa.grid.accumulation = _acc_picky
        result = fa.run()
        assert "flow_accumulation" in result

    def test_run_breach_fallback_to_fill(self, sloped_dem):
        """When breach_depressions is missing, fall back to fill_depressions."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)

        def _raise_attr(*args, **kwargs):
            raise AttributeError("breach_depressions not available")

        fa.grid.breach_depressions = _raise_attr
        result = fa.run()
        assert "flow_direction" in result

    def test_run_with_weights_typeerror_fallback(self, sloped_dem):
        """Weighted accumulation falls back when routing kwarg rejected."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)

        orig_acc = fa.grid.accumulation
        calls = {"n": 0}

        def _acc_picky(*args, **kwargs):
            calls["n"] += 1
            # Let the unweighted first call succeed. Reject the weighted call
            # with routing, then accept it without routing (forced to dinf).
            if kwargs.get("weights") is not None and "routing" in kwargs:
                raise TypeError("routing+weights rejected")
            if kwargs.get("weights") is not None:
                kwargs["routing"] = "dinf"
            return orig_acc(*args, **kwargs)

        fa.grid.accumulation = _acc_picky
        weights = np.ones((25, 25), dtype="float32")
        result = fa.run(runoff_weights=weights)
        assert "runoff_accumulation" in result

    def test_run_weights_raster_ctor_fails(self, sloped_dem):
        """When pysheds Raster wrapper can't be built, raw array is passed through."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)

        # If runoff_weights has no .astype attribute, the Raster wrapper
        # construction raises AttributeError and the except branch falls back
        # to the raw weights. We stub accumulation so it accepts the raw list.
        orig_acc = fa.grid.accumulation

        def _acc_accept_any(*args, **kwargs):
            if not isinstance(kwargs.get("weights"), type(fa.fdir)):
                return np.zeros((25, 25), dtype="float32")
            return orig_acc(*args, **kwargs)

        fa.grid.accumulation = _acc_accept_any

        class _NoAstype:
            """Object without .astype — forces the Raster ctor into except."""
        result = fa.run(runoff_weights=_NoAstype())
        assert "runoff_accumulation" in result


# ---------------------------------------------------------------------------
# delineate_catchment / get_stream_network / runoff volume
# ---------------------------------------------------------------------------

class TestDelineateCatchment:
    def test_raises_before_run(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        with pytest.raises(RuntimeError, match="Run flow analysis first"):
            fa.delineate_catchment(10.0, 10.0)

    def test_returns_mask(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        catch = fa.delineate_catchment(12.0, 22.0)
        assert catch is not None

    def test_typeerror_fallback(self, sloped_dem):
        """delineate_catchment's except-TypeError branch."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()

        orig = fa.grid.catchment

        def _catch_picky(*args, **kwargs):
            if "routing" in kwargs:
                raise TypeError("routing rejected")
            # Re-dispatch with dinf routing (real pysheds d8 is buggy on NumPy 2.4)
            kwargs["routing"] = "dinf"
            return orig(*args, **kwargs)

        fa.grid.catchment = _catch_picky
        catch = fa.delineate_catchment(12.0, 22.0)
        assert catch is not None


class TestGetStreamNetwork:
    def test_raises_before_run(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="Run flow analysis first"):
            fa.get_stream_network()

    def test_returns_bool_mask(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        mask = fa.get_stream_network(accumulation_threshold=50)
        arr = np.array(mask)
        assert arr.dtype == bool


class TestGetRunoffVolumeRaster:
    def test_raises_before_run(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="Run flow analysis first"):
            fa.get_runoff_volume_raster(10.0, 1.0)

    def test_scales_correctly(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        vol = fa.get_runoff_volume_raster(runoff_mm=10.0, cell_area_m2=1.0)
        # Max vol should equal max acc × 0.01 × 1.0
        assert vol.max() == pytest.approx(np.array(fa.acc).max() * 0.01, rel=1e-3)


# ---------------------------------------------------------------------------
# get_boundary_exit_points
# ---------------------------------------------------------------------------

class TestBoundaryExitPoints:
    def test_raises_before_run(self, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="Run flow analysis first"):
            fa.get_boundary_exit_points(boundary_gpkg, 50, 10.0, 1.0)

    def test_returns_list(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(boundary_gpkg, 5, 10.0, 1.0)
        assert isinstance(pts, list)

    def test_points_have_required_keys(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(boundary_gpkg, 5, 10.0, 1.0)
        for p in pts:
            for key in ("x", "y", "accumulation", "volume_m3", "flow_ls", "label"):
                assert key in p

    def test_empty_when_threshold_too_high(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(boundary_gpkg, 10**9, 10.0, 1.0)
        assert pts == []

    def test_no_polygon_boundary_returns_empty(self, sloped_dem, tmp_path):
        """Boundary file with only a linestring (no exterior) → empty."""
        from shapely.geometry import LineString
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        line_gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (10, 10)])]},
            crs="EPSG:32632",
        )
        line_path = str(tmp_path / "line_only.gpkg")
        line_gdf.to_file(line_path, driver="GPKG")

        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(line_path, 5, 10.0, 1.0)
        assert pts == []

    def test_zero_duration_gives_zero_flow_ls(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(boundary_gpkg, 5, 10.0, 0.0)
        for p in pts:
            assert p["flow_ls"] == 0


# ---------------------------------------------------------------------------
# get_catchment_polygons + _find_boundary_outlets
# ---------------------------------------------------------------------------

class TestCatchmentPolygons:
    def test_raises_before_run(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="Run flow analysis first"):
            fa.get_catchment_polygons()

    def test_returns_list_dinf(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run(routing="dinf")
        result = fa.get_catchment_polygons(stream_threshold=50)
        assert isinstance(result, list)

    def test_d8_bearing_branch(self, sloped_dem):
        """Exercise the d8 bearing map branch of get_catchment_polygons.

        pysheds' real d8 accumulation is broken on NumPy 2.x (removed in1d),
        so we run dinf then flip fa.routing = 'd8' to hit the d8 bearing path
        in get_catchment_polygons without running d8 accumulation.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run(routing="dinf")
        fa.routing = "d8"
        # grid.catchment with routing='d8' is still okay since it only scans
        # fdir, which we computed as dinf angles — catchment may be empty but
        # not error out; the d8_map branch executes during bearing computation.
        result = fa.get_catchment_polygons(stream_threshold=50)
        assert isinstance(result, list)

    def test_with_explicit_outlet_points(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        # Outlet near bottom boundary (likely valid for south-sloping DEM)
        result = fa.get_catchment_polygons(outlet_points=[(12.5, 1.5)])
        assert isinstance(result, list)

    def test_no_boundary_outlets_fallback_to_max_acc(self, sloped_dem):
        """With impossibly high stream_threshold, falls back to max-acc cell."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        result = fa.get_catchment_polygons(stream_threshold=10**9)
        assert isinstance(result, list)

    def test_catchment_result_has_keys(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        result = fa.get_catchment_polygons(stream_threshold=30)
        for r in result:
            for key in ("id", "geometry", "area_m2", "area_ha",
                         "flow_bearing", "label"):
                assert key in r

    def test_catchment_exception_continues(self, sloped_dem):
        """If grid.catchment raises for one outlet, others still processed."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()

        orig = fa.grid.catchment
        calls = {"n": 0}

        def _flaky_catchment(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first call fails")
            return orig(*args, **kwargs)

        fa.grid.catchment = _flaky_catchment
        # Two outlets: first will fail, second should succeed
        result = fa.get_catchment_polygons(
            outlet_points=[(5.0, 5.0), (12.5, 2.5)]
        )
        assert isinstance(result, list)

    def test_catchment_typeerror_fallback(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()

        orig = fa.grid.catchment

        def _catch_picky(*args, **kwargs):
            if "routing" in kwargs:
                raise TypeError("routing rejected")
            return orig(*args, **kwargs)

        fa.grid.catchment = _catch_picky
        result = fa.get_catchment_polygons(outlet_points=[(12.5, 2.5)])
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_fdir_description / get_profile / save_result
# ---------------------------------------------------------------------------

class TestFdirDescription:
    def test_dinf_label(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.routing = "dinf"
        assert "D-infinity" in fa.get_fdir_description()

    def test_d8_label(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.routing = "d8"
        assert "D8" in fa.get_fdir_description()


class TestGetProfileAndSave:
    def test_get_profile_keys(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        profile = fa.get_profile()
        for key in ("driver", "dtype", "crs", "transform", "width", "height",
                     "count", "compress"):
            assert key in profile

    def test_save_result_creates_file(self, sloped_dem, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        out = str(tmp_path / "result.tif")
        fa.save_result(np.array(fa.acc).astype("float32"), out, "flow acc")
        assert os.path.exists(out)

    def test_save_result_without_description(self, sloped_dem, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        out = str(tmp_path / "result_nodesc.tif")
        fa.save_result(np.array(fa.acc).astype("float32"), out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# AnalysisWorker — exercises _do_analysis (QThread stubbed by conftest)
# ---------------------------------------------------------------------------

class TestAnalysisWorker:
    def _collect_signals(self, worker):
        """Connect simple collectors to progress/finished/error signals."""
        progress_log = []
        finished_log = []
        error_log = []
        worker.progress.connect(lambda p, m: progress_log.append((p, m)))
        worker.finished.connect(lambda r: finished_log.append(r))
        worker.error.connect(lambda e: error_log.append(e))
        return progress_log, finished_log, error_log

    def test_init_stores_config(self, sloped_dem, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path),
            stream_threshold=50, cn=70, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            label="test",
        )
        assert w.stream_threshold == 50
        assert w.cn == 70
        assert w.routing == "dinf"
        assert w.cn_zones_data == []

    def test_do_analysis_basic(self, sloped_dem, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=30, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            label="base",
        )
        progress, finished, errors = self._collect_signals(w)
        w._do_analysis()
        assert len(errors) == 0
        assert len(finished) == 1
        result = finished[0]
        assert os.path.exists(result["flow_accumulation"])
        assert os.path.exists(result["flow_direction"])
        assert os.path.exists(result["stream_network"])
        assert os.path.exists(result["runoff_volume"])
        assert result["effective_cn"] > 0

    def test_do_analysis_with_boundary(self, sloped_dem, boundary_gpkg, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=20, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            boundary_path=boundary_gpkg, label="bnd",
        )
        _, finished, errors = self._collect_signals(w)
        w._do_analysis()
        assert len(errors) == 0
        assert len(finished) == 1
        assert isinstance(finished[0]["exit_points"], list)

    def test_do_analysis_with_catchments(self, sloped_dem, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=30, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            run_catchments=True, label="catch",
        )
        _, finished, _ = self._collect_signals(w)
        w._do_analysis()
        assert isinstance(finished[0]["catchments"], list)

    def test_do_analysis_volume_threshold_mode(self, sloped_dem, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=30, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            threshold_mode="volume", volume_threshold=10.0,
            label="vol",
        )
        _, finished, _ = self._collect_signals(w)
        w._do_analysis()
        assert os.path.exists(finished[0]["stream_network"])

    def test_do_analysis_cn_zones(self, sloped_dem, tmp_path):
        """CN zones branch: provide WKT polygons, triggers weighted accumulation."""
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        zone_wkt = "POLYGON((5 5, 20 5, 20 20, 5 20, 5 5))"
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=30, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            threshold_mode="volume", volume_threshold=10.0,
            cn_zones_data=[{"wkt": zone_wkt, "cn": 85}],
            label="zones",
        )
        _, finished, errors = self._collect_signals(w)
        w._do_analysis()
        assert len(errors) == 0
        assert finished[0]["effective_cn"] > 0

    def test_do_analysis_cn_zones_bad_wkt_tolerated(self, sloped_dem, tmp_path):
        """Bad WKT in cn_zones_data is swallowed."""
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=30, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            cn_zones_data=[{"wkt": "THIS IS NOT WKT", "cn": 85}],
            label="badwkt",
        )
        _, finished, errors = self._collect_signals(w)
        w._do_analysis()
        assert len(errors) == 0

    def test_run_catches_exception(self, tmp_path):
        """AnalysisWorker.run() with a bad DEM path emits error, not raise."""
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path="nonexistent_dem.tif",
            output_dir=str(tmp_path),
            stream_threshold=30, cn=70, moisture="normal",
            rainfall_mm=10.0, duration_hours=1.0, label="bad",
        )
        _, _, errors = self._collect_signals(w)
        w.run()
        assert len(errors) == 1

    def test_do_analysis_exit_points_exception_swallowed(self, sloped_dem, tmp_path):
        """Boundary exit point extraction failure doesn't abort the worker."""
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path=sloped_dem, output_dir=str(tmp_path / "out"),
            stream_threshold=30, cn=72, moisture="normal",
            rainfall_mm=30.0, duration_hours=2.0,
            boundary_path="nonexistent_boundary.gpkg",  # triggers except
            label="err",
        )
        _, finished, errors = self._collect_signals(w)
        w._do_analysis()
        assert len(errors) == 0
        assert finished[0]["exit_points"] == []

    def test_flow_dir_description_dinf(self):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path="x", output_dir="x", stream_threshold=1, cn=1,
            moisture="normal", rainfall_mm=1.0, duration_hours=1.0,
            routing="dinf",
        )
        assert "D-infinity" in w._flow_dir_description()

    def test_flow_dir_description_d8(self):
        from terrainflow_assessment.modules.flow_analysis import AnalysisWorker
        w = AnalysisWorker(
            dem_path="x", output_dir="x", stream_threshold=1, cn=1,
            moisture="normal", rainfall_mm=1.0, duration_hours=1.0,
            routing="d8",
        )
        assert "D8" in w._flow_dir_description()
