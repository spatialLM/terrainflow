"""
Comprehensive tests for terrainflow_assessment/modules/flow_analysis.py.

Covers FlowAnalysis (all public methods + private helpers) and AnalysisWorker
(the QThread wrapper is exercised through _do_analysis, which doesn't require
a real Qt event loop because the conftest stubs out QThread).
"""
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

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
            for key in ("x", "y", "flow_ls", "volume_m3", "label"):
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

    def test_zero_duration_returns_empty(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        # Zero duration → flow rate undefined → no exits.
        assert fa.get_boundary_exit_points(boundary_gpkg, 5, 10.0, 0.0) == []

    def test_lower_threshold_reveals_at_least_as_many(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        low = fa.get_boundary_exit_points(boundary_gpkg, 0.0, 10.0, 1.0)
        high = fa.get_boundary_exit_points(boundary_gpkg, 10**9, 10.0, 1.0)
        assert len(low) >= len(high)
        assert high == []

    def test_all_exits_meet_flow_threshold(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(boundary_gpkg, 0.1, 10.0, 1.0)
        for p in pts:
            assert p["flow_ls"] >= 0.1

    def test_volume_raster_override(self, sloped_dem, boundary_gpkg):
        import numpy as np

        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        vol = np.array(fa.acc, dtype="float64") * 0.5  # arbitrary per-cell volume (m³)
        pts = fa.get_boundary_exit_points(
            boundary_gpkg, 0.0, 10.0, 1.0, volume_raster=vol)
        assert isinstance(pts, list)


# ---------------------------------------------------------------------------
# boundary_outflow_total
# ---------------------------------------------------------------------------

class TestBoundaryOutflowTotal:
    def test_raises_before_run(self, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        with pytest.raises(RuntimeError, match="Run flow analysis first"):
            fa.boundary_outflow_total(boundary_gpkg, 10.0, 1.0)

    def test_returns_both_keys(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        total = fa.boundary_outflow_total(boundary_gpkg, 10.0, 1.0)
        assert set(total) == {"volume_m3", "flow_ls"}
        assert total["volume_m3"] > 0

    def test_zero_duration_returns_zero(self, sloped_dem, boundary_gpkg):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        assert fa.boundary_outflow_total(boundary_gpkg, 10.0, 0.0) == {
            "volume_m3": 0.0, "flow_ls": 0.0}

    def test_no_polygon_returns_zero(self, sloped_dem, tmp_path):
        from shapely.geometry import LineString

        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        line_gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (10, 10)])]}, crs="EPSG:32632")
        line_path = str(tmp_path / "line_only.gpkg")
        line_gdf.to_file(line_path, driver="GPKG")

        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        assert fa.boundary_outflow_total(line_path, 10.0, 1.0)["volume_m3"] == 0.0

    def test_counts_each_flow_path_once(self, sloped_dem, boundary_gpkg):
        """Only the last inside-cell of a path contributes, not every boundary cell.

        ``sloped_dem`` drops 2 m per row southward against 0.5 m laterally, so every
        cell drains due south. ``boundary_gpkg`` covers a 21x21 block of cell centres
        (rows 2-22, cols 2-22), so exactly the 21 cells of its bottom row drain to a
        cell outside it. With 1 m³ on every cell the total is therefore 21 m³ — a
        per-boundary-cell sum would instead count the whole southern edge repeatedly.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        ones = np.ones((25, 25), dtype="float64")
        total = fa.boundary_outflow_total(
            boundary_gpkg, 10.0, 1.0, volume_raster=ones)
        assert total["volume_m3"] == pytest.approx(21.0)
        # 21 m³ over one hour, in L/s.
        assert total["flow_ls"] == pytest.approx(21.0 * 1000.0 / 3600.0, rel=1e-3)

    def test_independent_of_exit_threshold(self, sloped_dem, boundary_gpkg):
        """The total is a property of the storm; the exit sum is a display artefact."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        total = fa.boundary_outflow_total(boundary_gpkg, 10.0, 1.0)["volume_m3"]

        def exit_sum(threshold):
            pts = fa.get_boundary_exit_points(boundary_gpkg, threshold, 10.0, 1.0)
            return sum(p["volume_m3"] for p in pts)

        assert exit_sum(0.0) > exit_sum(10 ** 9) == 0.0
        assert fa.boundary_outflow_total(
            boundary_gpkg, 10.0, 1.0)["volume_m3"] == total

    def test_total_exceeds_thresholded_exit_sum(self, sloped_dem, boundary_gpkg):
        """The regression this method exists for: the exit sum under-reports.

        Every crossing contributes only its single peak cell, so summing the markers
        loses the rest of the flux even with the threshold wide open.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        pts = fa.get_boundary_exit_points(boundary_gpkg, 0.0, 10.0, 1.0)
        exit_sum = sum(p["volume_m3"] for p in pts)
        total = fa.boundary_outflow_total(boundary_gpkg, 10.0, 1.0)["volume_m3"]
        assert total > exit_sum


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


class TestOutletCellCentres:
    """row -> y must be the cell *centre*: ``f + (row + 0.5) * e``, e negative.

    Adding ``+cell_h/2`` instead puts the point one full cell north of its own
    cell — the same defect ``keypoint_analysis._rc_to_xy`` documents as fixed
    there. On the top edge that lands outside the raster altogether, so the
    outlet seeds nothing and the catchment is silently missing.

    Note the round-trip check deliberately avoids row 0: ``_acc_at`` converts
    back with ``int()``, which truncates -0.5 to 0 and would mask the fault for
    that one row. Row 0 is covered by the in-raster check instead.
    """

    @staticmethod
    def _loaded(dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(dem)
        return fa

    def test_top_edge_outlet_lands_inside_the_raster(self, sloped_dem):
        fa = self._loaded(sloped_dem)
        acc = np.zeros((25, 25))
        acc[0, 10] = 500.0          # only cell over threshold, on the top edge

        outlets = fa._find_boundary_outlets(acc, 100)

        assert len(outlets) == 1
        _, y = outlets[0]
        top = fa.transform.f
        bottom = top + 25 * fa.transform.e
        assert bottom < y < top, f"y={y} falls outside the raster (top={top})"
        assert y == pytest.approx(top + 0.5 * fa.transform.e)

    @pytest.mark.parametrize("row,col", [(24, 3), (7, 0), (7, 24), (24, 24)])
    def test_boundary_outlet_round_trips_to_its_own_cell(self, sloped_dem, row, col):
        fa = self._loaded(sloped_dem)
        acc = np.zeros((25, 25))
        acc[row, col] = 500.0

        (x, y), = fa._find_boundary_outlets(acc, 100)

        back_col = int((x - fa.transform.c) / fa.transform.a)
        back_row = int((y - fa.transform.f) / fa.transform.e)
        assert (back_row, back_col) == (row, col)

    def test_bottom_edge_outlet_still_delineates_a_catchment(self, tmp_path):
        """The other half of the same defect: how pysheds reads the seed back.

        ``grid.catchment`` defaults to ``snap="corner"``, the nearest grid
        intersection, resolved with ``np.around``. A cell centre sits exactly half
        a cell from a corner, so every seed is a ``.5`` index and banker's rounding
        picks by parity. On an even-height raster the last row — 19.5 here — rounds
        *up* to 20, off the raster, and the catchment comes back empty: the site's
        real outlet is the one seed guaranteed to be discarded.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        r = np.arange(20).reshape(-1, 1)
        c = np.arange(21).reshape(1, -1)
        data = (100.0 - r * 1.0 + np.abs(c - 10) * 0.5).astype("float32")
        dem = _write_raster(str(tmp_path / "valley.tif"), data)

        fa = FlowAnalysis()
        fa.load_dem(dem)
        fa.run(routing="dinf")

        outlets = fa._find_boundary_outlets(np.array(fa.acc), 30)
        assert outlets, "no boundary outlet found — fixture is not draining south"
        bottom = fa.transform.f + 19.5 * fa.transform.e
        assert any(y == pytest.approx(bottom) for _, y in outlets), (
            "expected an outlet on the last row")

        polys = fa.get_catchment_polygons(stream_threshold=30)
        assert polys, "the bottom-edge outlet delineated no catchment"
        assert max(p["area_m2"] for p in polys) > 0

    def test_max_acc_fallback_seeds_the_max_acc_cell(self, sloped_dem):
        fa = self._loaded(sloped_dem)
        fa.run()

        seen = {}
        orig = fa.grid.catchment

        def _spy(*args, **kwargs):
            seen.update(x=kwargs.get("x"), y=kwargs.get("y"))
            return orig(*args, **kwargs)

        fa.grid.catchment = _spy
        # No boundary cell clears this, so it falls back to the max-acc cell.
        fa.get_catchment_polygons(stream_threshold=10**9)

        row, col = np.unravel_index(np.argmax(np.array(fa.acc)), fa.acc.shape)
        assert seen["x"] == pytest.approx(
            fa.transform.c + (col + 0.5) * fa.transform.a)
        assert seen["y"] == pytest.approx(
            fa.transform.f + (row + 0.5) * fa.transform.e)


# ---------------------------------------------------------------------------
# get_fdir_description / get_profile / save_result
# ---------------------------------------------------------------------------

class TestSeedIsNeverOnTheRim:
    """pysheds must never be handed a pour point on the edge of the array it walks.

    ``_dinf_catchment_iter_numba`` walks the flattened flow-direction array and
    reads all eight neighbours of a cell — ``parent ± {1, ncols, ncols ± 1}`` —
    before asking whether they exist, with numba's bounds checking off. Seeded on
    row 0 the northern neighbours are negative flat indices; on the last row the
    southern ones are past the end. On a 25-cell fixture that reads adjacent heap
    and returns; on a real DEM it leaves the mapped page and QGIS dies with an
    access violation, which is what a 3,000-cell-square raster did while these
    checks stayed green.

    Every boundary outlet is a rim cell by construction, so the fix cannot be a
    bounds test on the seed — the seeds are on the grid, just on its edge. It is
    ``pad_for_seeded_walk``: the walk runs one cell in from the edge of a padded
    copy. What this asserts is the property that matters, whatever the mechanism:
    the cell pysheds resolves the seed to is interior to the array it is given.
    """

    @staticmethod
    def _seed_cell(fdir, x, y, snap):
        from pysheds.sview import View
        col, row = View.nearest_cell(x, y, affine=fdir.viewfinder.affine, snap=snap)
        return int(row), int(col)

    def test_boundary_outlets_are_walked_one_cell_in(self, sloped_dem, monkeypatch):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run(routing="dinf")

        seen = []
        original = fa.grid.catchment

        def _recording(*args, **kwargs):
            fdir = kwargs["fdir"]
            seen.append((
                self._seed_cell(fdir, kwargs["x"], kwargs["y"],
                                kwargs.get("snap", "corner")),
                np.asarray(fdir).shape,
            ))
            return original(*args, **kwargs)

        monkeypatch.setattr(fa.grid, "catchment", _recording)
        fa.get_catchment_polygons(stream_threshold=30)

        assert seen, "no outlet was delineated — the fixture is not draining"
        for (row, col), (rows, cols) in seen:
            assert 0 < row < rows - 1 and 0 < col < cols - 1, (
                f"pysheds was seeded at ({row}, {col}) of a {(rows, cols)} array — "
                "on the rim, where its kernel reads outside the allocation"
            )

    def test_padding_leaves_an_interior_catchment_untouched(self, sloped_dem):
        """The margin is for the kernel to read into, not a change of answer."""
        from terrainflow_assessment.modules.flow_analysis import (
            FlowAnalysis, catchment_from_seed,
        )
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run(routing="dinf")

        x = fa.transform.c + 12.5 * fa.transform.a
        y = fa.transform.f + 12.5 * fa.transform.e

        direct = np.asarray(fa.grid.catchment(
            x=x, y=y, fdir=fa.fdir, xytype="coordinate",
            routing=fa.routing, snap="center",
        )).astype(bool)
        padded = catchment_from_seed(
            fa.grid, fa.fdir, x, y, routing=fa.routing, snap="center",
        ).astype(bool)

        assert padded.shape == direct.shape
        assert np.array_equal(padded, direct)

    def test_a_seed_off_the_raster_is_still_refused(self, sloped_dem):
        """Padding is a margin for the kernel, not licence to seed off the grid."""
        from terrainflow_assessment.modules.flow_analysis import (
            FlowAnalysis, catchment_from_seed,
        )
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run(routing="dinf")

        off = fa.transform.c + 40.0 * fa.transform.a
        y = fa.transform.f + 12.5 * fa.transform.e
        with pytest.raises(ValueError, match="outside a raster"):
            catchment_from_seed(fa.grid, fa.fdir, off, y,
                                routing=fa.routing, snap="center")


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


class TestResultRasterNodata:
    """A result raster has to declare what "no data" means in it.

    ``pysheds.io.read_raster`` defaults to 0 for an untagged file. Under
    D-infinity a direction of 0.0 rad is "flows due east", so on an east-facing
    slope the round trip through disk marks most of the catchment as no-data and
    routing through it collapses — the simulation then disagrees with the flow map
    it was built from, with nothing to say so.
    """

    @pytest.fixture
    def east_plane(self, tmp_path):
        """A plane draining due east: every D-infinity angle is 0.0 rad."""
        col = np.arange(25, dtype="float32").reshape(1, -1)
        data = np.repeat(100.0 - col, 25, axis=0)
        return _write_raster(str(tmp_path / "east.tif"), data)

    def test_fdir_nodata_is_nan_under_dinf_and_zero_under_d8(self):
        from terrainflow_assessment.modules.flow_analysis import fdir_nodata
        assert np.isnan(fdir_nodata("dinf"))
        assert fdir_nodata("d8") == 0

    def test_profile_carries_the_nodata_it_is_given(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        assert "nodata" in fa.get_profile()
        assert fa.get_profile(nodata=-9999.0)["nodata"] == -9999.0

    def test_saved_fdir_declares_its_nodata_on_disk(self, east_plane, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(east_plane)
        fa.run(routing="dinf")
        out = str(tmp_path / "fdir.tif")
        fa.save_result(np.array(fa.fdir), out, fa.get_fdir_description(),
                       nodata=fa.get_fdir_nodata())
        with rasterio.open(out) as src:
            assert src.nodata is not None and np.isnan(src.nodata)

    def test_accumulation_survives_the_round_trip(self, east_plane, tmp_path):
        """Write the direction grid, read it back, accumulate: same answer."""
        from pysheds.grid import Grid

        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(east_plane)
        fa.run(routing="dinf")
        in_memory = float(np.nansum(np.array(fa.acc)))

        out = str(tmp_path / "fdir.tif")
        fa.save_result(np.array(fa.fdir), out, fa.get_fdir_description(),
                       nodata=fa.get_fdir_nodata())

        grid = Grid.from_raster(east_plane)
        # No nodata= here on purpose: it has to come off the file.
        reread = grid.read_raster(out)
        acc = grid.accumulation(reread, routing="dinf")

        assert float(np.nansum(np.array(acc))) == pytest.approx(in_memory, rel=1e-6)

    def test_untagged_fdir_loses_the_catchment(self, east_plane, tmp_path):
        """The counterexample the fix exists for — an untagged file still breaks.

        Kept so the guard above has something to be a guard against: if pysheds
        ever stops defaulting to 0 this fails, and the fix can be reconsidered
        rather than cargo-culted.
        """
        from pysheds.grid import Grid

        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(east_plane)
        fa.run(routing="dinf")
        in_memory = float(np.nansum(np.array(fa.acc)))

        out = str(tmp_path / "untagged.tif")
        fa.save_result(np.array(fa.fdir), out, fa.get_fdir_description())
        with rasterio.open(out) as src:
            assert src.nodata is None

        grid = Grid.from_raster(east_plane)
        acc = grid.accumulation(grid.read_raster(out), routing="dinf")

        assert float(np.nansum(np.array(acc))) < in_memory


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
        worker.completed.connect(lambda r: finished_log.append(r))
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


class TestUnroutedFlow:
    """Water the routing could not place has to be counted, not quietly dropped.

    A cell pysheds marks flat (-1) or pit (-2) is rewritten to a self-loop before
    accumulation, so it absorbs everything upstream and reports none of it downstream.
    Round 14 measured 36 such cells on the Quail Island design holding ~1.4% of the site's
    runoff, with nothing anywhere saying so.
    """

    def test_a_clean_dem_has_nothing_unrouted(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        result = fa.run()
        assert result["unrouted_cells"] == 0
        assert result["unrouted_flow"] == 0.0

    def test_before_a_run_it_reports_nothing_rather_than_failing(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        assert FlowAnalysis().unrouted_flow() == (0, 0.0)

    def test_a_flat_cell_is_counted_with_the_flow_it_holds(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        fdir = np.array(fa.fdir, dtype="float64")
        acc = np.array(fa.acc, dtype="float64")
        # Well inside the grid: a cell on the border, or against nodata, is the site's
        # edge rather than a place water gets stuck.
        fdir[10, 10] = FlowAnalysis.FDIR_FLAT
        fdir[12, 12] = FlowAnalysis.FDIR_PIT
        fa.fdir, fa.acc = fdir, acc
        n, flow = fa.unrouted_flow()
        assert n == 2
        assert flow == pytest.approx(acc[10, 10] + acc[12, 12])

    def test_water_stopping_at_the_edge_of_the_data_is_not_a_loss(self, sloped_dem):
        """Quail Island is an island: 35 of its 36 stuck cells are the coastline.

        Water reaching the end of the elevation data has left the analysed area, not gone
        missing. Counting those would fire this warning on every clipped or coastal DEM
        for an entirely benign reason.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        fdir = np.array(fa.fdir, dtype="float64")
        dem = np.array(fa.dem, dtype="float64")
        # One stuck cell on the border, one against a hole punched in the data.
        fdir[0, 5] = FlowAnalysis.FDIR_PIT
        dem[15, 15] = np.nan
        fdir[15, 16] = FlowAnalysis.FDIR_FLAT
        fa.fdir, fa.dem = fdir, dem
        assert fa.unrouted_flow() == (0, 0.0)

    def test_a_pit_outside_the_domain_is_not_the_domains_problem(self, sloped_dem):
        """The numerator has to be measured over the same ground as the denominator.

        ``unrouted_flow`` masked only by the data edge, while the caller divided by the
        *site's* cell count — so a pit in the DEM buffer outside the drawn boundary
        contributed its whole upstream to a ratio it was not part of. On a tile much
        larger than the site that alone can push the share past 100%, and a field run
        duly reported 105.0%.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        fdir = np.array(fa.fdir, dtype="float64")
        fdir[10, 10] = FlowAnalysis.FDIR_PIT     # inside the site
        fdir[22, 22] = FlowAnalysis.FDIR_PIT     # out in the buffer
        fa.fdir = fdir

        domain = np.zeros(fdir.shape, dtype=bool)
        domain[5:20, 5:20] = True

        assert fa.unrouted_flow()[0] == 2, "both pits are on the tile"
        n, _held = fa.unrouted_flow(domain=domain)
        assert n == 1, "the pit outside the site is still being counted against it"

    def test_the_held_load_can_be_measured_in_the_field_you_name(self, sloped_dem):
        """``acc`` is a cell count; calling a share of it "% of runoff" is a category
        error. Passing the m³ field makes the sentence literally true."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        fdir = np.array(fa.fdir, dtype="float64")
        fdir[10, 10] = FlowAnalysis.FDIR_PIT
        fa.fdir = fdir

        volumes = np.asarray(fa.acc, dtype="float64") * 7.0
        _n, counted = fa.unrouted_flow()
        _n, in_m3 = fa.unrouted_flow(field=volumes)
        assert in_m3 == pytest.approx(counted * 7.0)


class TestUnroutedDiagnostics:
    """The counts say how many; these say why, and the causes want different fixes."""

    def _stuck(self, sloped_dem, cells):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = FlowAnalysis()
        fa.load_dem(sloped_dem)
        fa.run()
        fdir = np.array(fa.fdir, dtype="float64")
        for r, c in cells:
            fdir[r, c] = FlowAnalysis.FDIR_PIT
        fa.fdir = fdir
        return fa

    def test_it_splits_flats_from_pits(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        fa = self._stuck(sloped_dem, [(10, 10), (12, 12)])
        fdir = np.array(fa.fdir, dtype="float64")
        fdir[14, 14] = FlowAnalysis.FDIR_FLAT
        fa.fdir = fdir
        d = fa.unrouted_diagnostics()
        assert d["totals"]["stuck_pit"] == 2
        assert d["totals"]["stuck_flat"] == 1

    def test_it_separates_inside_the_site_from_outside_it(self, sloped_dem):
        fa = self._stuck(sloped_dem, [(10, 10), (22, 22)])
        domain = np.zeros(np.asarray(fa.fdir).shape, dtype=bool)
        domain[5:20, 5:20] = True
        d = fa.unrouted_diagnostics(domain=domain)
        assert d["location"]["outside_domain"] == 1
        assert d["location"]["interior_in_domain"] == 1

    def test_scattered_spikes_do_not_look_like_one_plateau(self, sloped_dem):
        """516 singletons and three blobs of 170 are the same count and different bugs."""
        fa = self._stuck(sloped_dem, [(8, 8), (12, 12), (16, 16)])
        d = fa.unrouted_diagnostics()
        assert d["components"]["count"] == 3
        assert d["components"]["singletons"] == 3
        assert d["components"]["largest"] == 1

    def test_it_reports_whether_a_lower_neighbour_existed(self, sloped_dem):
        """On a uniform slope every cell has one, so a router calling it a pit is a
        comparison bug rather than terrain — which is the point of measuring it."""
        fa = self._stuck(sloped_dem, [(10, 10)])
        d = fa.unrouted_diagnostics()
        assert d["neighbour_drop"]["conditioned"]["has_lower"] == 1

    def test_the_m3_ratio_divides_by_the_rain_that_fell(self, sloped_dem):
        """The share must be over runoff, not over the sum of the throughflow field.

        Summing throughflow totals every cubic metre once per cell it passes — a number
        with no physical meaning and, on the reported run, 225x the site's runoff. The
        file written to check the warning read 0.000596 against a warning that correctly
        said 13.4%.
        """
        fa = self._stuck(sloped_dem, [(10, 10)])
        field = np.ones(np.asarray(fa.fdir).shape, dtype="float64")
        domain = np.ones(field.shape, dtype=bool)

        d = fa.unrouted_diagnostics(domain=domain, field=field, runoff_volume_m3=4.0)
        # One stuck cell holding 1 m3, against 4 m3 of rain.
        assert d["ratios"]["m3_over_m3_domain_masked"] == pytest.approx(0.25)
        assert d["ratios"]["runoff_volume_m3"] == 4.0
        # And the field's own sum is still reported, under a name that says what it is.
        assert d["field_ink"]["total_throughflow_m3"] == pytest.approx(float(field.size))

    def test_the_m3_ratio_is_omitted_rather_than_guessed(self, sloped_dem):
        """No runoff volume passed means no denominator — not the old wrong one."""
        fa = self._stuck(sloped_dem, [(10, 10)])
        field = np.ones(np.asarray(fa.fdir).shape, dtype="float64")
        d = fa.unrouted_diagnostics(field=field)
        assert "m3_over_m3_domain_masked" not in d["ratios"]

    def test_it_reports_the_flat_step_that_was_used(self, sloped_dem):
        """The 516 came from this step, so the run states it rather than implying it."""
        fa = self._stuck(sloped_dem, [(10, 10)])
        flat = fa.unrouted_diagnostics()["flat_resolution"]
        assert flat["eps"] == fa.flat_eps
        assert flat["pysheds_default_eps"] == 1e-5

    def test_it_states_which_conditioning_branch_ran(self, sloped_dem):
        fa = self._stuck(sloped_dem, [(10, 10)])
        assert fa.unrouted_diagnostics()["conditioning"] in ("fill", "breach")

    def test_it_says_so_rather_than_raising_before_a_run(self):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        assert "error" in FlowAnalysis().unrouted_diagnostics()

    def test_the_formatter_names_every_measurement_it_was_given(self, sloped_dem):
        """A diagnostic nobody reads until something is wrong must not silently drop a
        field, so the formatter walks the dict instead of naming its keys."""
        from terrainflow_assessment.modules.flow_analysis import (
            format_unrouted_diagnostics,
        )
        fa = self._stuck(sloped_dem, [(10, 10)])
        d = fa.unrouted_diagnostics()
        text = format_unrouted_diagnostics(d)
        for key in d:
            assert key in text, key
        for key in d["totals"]:
            assert key in text, key


class TestUnroutedFlowWarning:
    def test_it_says_nothing_when_the_share_is_trivial(self):
        from terrainflow_assessment.modules.flow_analysis import unrouted_flow_warning
        # One cell out of a 10,000-cell domain is a nodata speck, not a finding.
        assert unrouted_flow_warning(1, 1.0, 10_000) is None

    def test_it_reports_a_material_share_with_the_number(self):
        from terrainflow_assessment.modules.flow_analysis import unrouted_flow_warning
        msg = unrouted_flow_warning(36, 4152.0, 292_288)
        assert msg is not None
        assert "36 cells inside the site" in msg
        assert "1.4%" in msg

    def test_nothing_unrouted_is_not_a_warning(self):
        from terrainflow_assessment.modules.flow_analysis import unrouted_flow_warning
        assert unrouted_flow_warning(0, 0.0, 1000) is None
        assert unrouted_flow_warning(5, 0.0, 1000) is None
        assert unrouted_flow_warning(5, 10.0, 0) is None

    def test_one_cell_is_singular(self):
        from terrainflow_assessment.modules.flow_analysis import unrouted_flow_warning
        msg = unrouted_flow_warning(1, 500.0, 1000)
        assert "1 cell inside the site" in msg

    def test_an_impossible_share_is_printed_and_flagged_rather_than_clamped(self):
        """A field run reported 105.0%, and that is what got this looked at.

        ``min(share, 1.0)`` would have turned an obviously broken instrument into a
        plausible reading — the worst available outcome, because 100% of the site's
        water going missing is alarming but believable, whereas 105% is self-evidently
        a measurement fault and sends the reader to the diagnostic.
        """
        from terrainflow_assessment.modules.flow_analysis import unrouted_flow_warning
        msg = unrouted_flow_warning(516, 1050.0, 1000.0)
        assert "105.0%" in msg
        assert "not possible" in msg
        assert "diagnostic" in msg

    def test_a_believable_share_carries_no_such_note(self):
        from terrainflow_assessment.modules.flow_analysis import unrouted_flow_warning
        msg = unrouted_flow_warning(36, 140.0, 1000.0)
        assert "14.0%" in msg
        assert "not possible" not in msg


class TestConditionedSurfacePrecision:
    """The conditioned surface must survive the round-trip to disk.

    ``resolve_flats`` inflates a flat by integer multiples of a step that is at most
    ``FLAT_EPS_CEILING`` (1e-5 m) and, since the flat-inversion fix, often far smaller.
    float32's spacing is ~6.1e-5 m at 1000 m elevation, so above roughly 600 m that whole
    gradient is quantised away on save — and ``flow_graph.d8_from_dem``, which requires a *strictly*
    positive drop, then reads a genuine flat and calls every cell of it a sink. A site near
    sea level never shows it, which is why this fixture is deliberately in hill country.
    """

    def _flat_topped_hill(self, path, base):
        """A plateau at *base* draining off one edge — a flat that must be resolved."""
        data = np.full((20, 20), base, dtype="float64")
        data[:, 15:] = base - np.arange(1, 6, dtype="float64")   # fall away to the east
        with rasterio.open(
            path, "w", driver="GTiff", height=20, width=20, count=1,
            dtype="float64", crs="EPSG:2193",
            transform=rasterio.transform.from_origin(0, 20, 1, 1),
        ) as dst:
            dst.write(data, 1)
        return path

    def test_float32_destroys_the_flat_gradient_in_hill_country(self, tmp_path):
        """The fault itself, so the fix is pinned to a demonstrated failure."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        from terrainflow_assessment.modules.flow_graph import d8_from_dem

        fa = FlowAnalysis()
        fa.load_dem(self._flat_topped_hill(str(tmp_path / "hill.tif"), 1000.0))
        result = fa.run(routing="dinf")   # pysheds d8 accumulation needs np.in1d (gone in NumPy 2)
        conditioned = np.asarray(result["conditioned_dem"], dtype="float64")

        # In memory the plateau has a gradient and drains.
        _, sink64 = d8_from_dem(conditioned, 1.0, 1.0)
        # Rounded to float32 it does not: the inflation is below the representable step.
        _, sink32 = d8_from_dem(conditioned.astype("float32").astype("float64"), 1.0, 1.0)
        assert int(sink32.sum()) > int(sink64.sum()), (
            "float32 should flatten the resolved gradient at this elevation — if it no "
            "longer does, this fixture has stopped testing anything"
        )

    def test_saving_the_conditioned_surface_as_float64_keeps_it(self, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        from terrainflow_assessment.modules.flow_graph import d8_from_dem

        fa = FlowAnalysis()
        fa.load_dem(self._flat_topped_hill(str(tmp_path / "hill.tif"), 1000.0))
        result = fa.run(routing="dinf")   # pysheds d8 accumulation needs np.in1d (gone in NumPy 2)
        conditioned = np.asarray(result["conditioned_dem"], dtype="float64")

        out = str(tmp_path / "conditioned.tif")
        fa.save_result(conditioned, out, "conditioned", dtype="float64")
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "float64"
            round_tripped = src.read(1).astype("float64")

        _, sink_mem = d8_from_dem(conditioned, 1.0, 1.0)
        _, sink_disk = d8_from_dem(round_tripped, 1.0, 1.0)
        assert int(sink_disk.sum()) == int(sink_mem.sum())

    def test_the_default_is_still_float32_for_every_other_raster(self, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(self._flat_topped_hill(str(tmp_path / "hill.tif"), 1000.0))
        fa.run(routing="dinf")   # pysheds d8 accumulation needs np.in1d (gone in NumPy 2)
        out = str(tmp_path / "acc.tif")
        fa.save_result(np.asarray(fa.acc, dtype="float64"), out, "accumulation")
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "float32"


class TestCrestSplit:
    """A pond spills along its whole level crest, and the site loses no water doing it.

    ``resolve_flats`` is multi-outlet but routes each flat cell to its *nearest* way out, so
    a pool hands its inflow channel to whichever crest cell it happens to arrive beside. The
    fix contracts each pond to a mixing node and sheds its whole inflow evenly over the cells
    that discharge — measured on Quail Island's Dam 15, a level band of 65 cells reading
    ``min 1 / median 1 / max 36,782`` becomes a uniform 962.1, a ratio of exactly 1.00x.

    Every number below was measured, not derived. Each test runs ``routing="dinf"`` because
    pysheds' d8 accumulation calls ``np.in1d``, which NumPy 2 removed.
    """

    WALL = 100.0

    def _trough(self, path, *bands, width=3):
        """A walled trough: one uniform row per band, wall around it, outlet row at 0."""
        rows = [[self.WALL] * (width + 2)]
        rows += [[self.WALL] + [float(v)] * width + [self.WALL] for v in bands]
        rows += [[0.0] * (width + 2)]
        return _write_raster(path, np.array(rows, dtype="float32"))

    def _staircase(self, tmp_path):
        """Two ponds chained **through open ground** — what the parked version lost on.

        Pond A's pool (12) fills to its crest at 15; a row of ordinary hillside at 14 sits
        below that crest, and only then pond B's pool at 8. No cell of A discharges into a
        cell of B, so a pond graph read off the immediate D8 neighbour of an exit holds no
        edge between them — and yet every drop A sheds arrives in B.
        """
        return self._trough(str(tmp_path / "stair.tif"),
                            20, 12, 12, 12, 15, 14, 8, 8, 8, 10, 5)

    def test_the_chain_conserves_every_unit_of_water(self, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(self._staircase(tmp_path))
        result = fa.run(routing="dinf")
        acc = np.asarray(fa.acc, dtype="float64")

        assert result["crest_ponds"] == 2
        assert result["crest_cells"] == 6
        assert result["crest_passes"] == 3
        assert result["crest_residual"] == pytest.approx(0.0)
        # 33 interior cells; the border is nodata in a D-infinity direction field.
        assert acc[-1].sum() == pytest.approx(33.0)

    def test_the_outlet_row_stops_being_a_single_thread(self, tmp_path):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        path = self._staircase(tmp_path)
        before = FlowAnalysis()
        before.load_dem(path)
        before.run(routing="dinf", crest_split=False)
        was = np.asarray(before.acc, dtype="float64")[-1, 1:4]

        after = FlowAnalysis()
        after.load_dem(path)
        after.run(routing="dinf")
        now = np.asarray(after.acc, dtype="float64")[-1, 1:4]

        # Measured: 4.21 / 24.59 / 4.21 becomes a flat 11 / 11 / 11.
        assert was.max() / was.min() == pytest.approx(5.846, rel=1e-3)
        assert now == pytest.approx([11.0, 11.0, 11.0])
        assert was.sum() == pytest.approx(now.sum())

    def test_a_lopsided_inflow_still_leaves_over_the_whole_crest(self, tmp_path):
        """The case the user reported: one channel arrives, and the wall spills at one end.

        A nine-cell pool fed through a single-column inlet at its west end, against a level
        crest nine cells long.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        wall = self.WALL

        def band(v):
            return [wall] + [v] * 9 + [wall]

        data = np.array(
            [[wall] * 11]
            + [[wall, 32.0] + [wall] * 9,
               [wall, 31.0] + [wall] * 9,
               [wall, 30.0] + [wall] * 9]
            + [band(12.0)] * 3 + [band(15.0)] + [band(5.0)] + [[0.0] * 11],
            dtype="float32")
        path = _write_raster(str(tmp_path / "weir.tif"), data)

        before = FlowAnalysis()
        before.load_dem(path)
        before.run(routing="dinf", crest_split=False)
        was = np.asarray(before.acc, dtype="float64")[7, 1:10]

        after = FlowAnalysis()
        after.load_dem(path)
        after.run(routing="dinf")
        now = np.asarray(after.acc, dtype="float64")[7, 1:10]

        assert was.max() / np.median(was) == pytest.approx(1.535, rel=1e-3)
        assert now.max() / now.min() == pytest.approx(1.0)
        assert now == pytest.approx(np.full(9, 55.0 / 9.0))
        assert (np.asarray(after.acc, dtype="float64")[-1].sum()
                == pytest.approx(np.asarray(before.acc, dtype="float64")[-1].sum()))

    def test_the_flow_directions_are_not_touched(self, tmp_path):
        """The guarantee everything downstream rests on.

        One pointer per cell cannot divide the load of one cell, so the split is done in the
        flux field and the directions are left exactly alone — which is what keeps
        ``feature_inflow_m3``, the catchment labelling and capture % out of it.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        path = self._staircase(tmp_path)
        plain = FlowAnalysis()
        plain.load_dem(path)
        plain.run(routing="dinf", crest_split=False)

        split = FlowAnalysis()
        split.load_dem(path)
        split.run(routing="dinf")

        assert np.array_equal(np.asarray(plain.fdir, dtype="float64"),
                              np.asarray(split.fdir, dtype="float64"),
                              equal_nan=True)

    def test_a_pond_cell_is_never_counted_as_unrouted(self, tmp_path):
        """A pond cell self-loops *inside* the split and nowhere else.

        Were the absorbing map stored on ``self.fdir`` instead of kept local, every pond cell
        would read as ``FDIR_FLAT`` and ``unrouted_flow`` would report the whole pool as
        water the routing could not place.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(self._staircase(tmp_path))
        result = fa.run(routing="dinf")
        assert result["crest_ponds"] == 2
        assert fa.unrouted_flow() == (0, 0.0)

    def test_a_dem_with_no_ponds_is_left_exactly_as_it_was(self, sloped_dem):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        plain = FlowAnalysis()
        plain.load_dem(sloped_dem)
        plain.run(routing="dinf", crest_split=False)

        split = FlowAnalysis()
        split.load_dem(sloped_dem)
        result = split.run(routing="dinf")

        assert result["crest_ponds"] == 0
        assert np.array_equal(np.asarray(plain.acc, dtype="float64"),
                              np.asarray(split.acc, dtype="float64"))

    def test_a_bowl_with_no_way_out_keeps_the_default_routing_and_says_so(self, tmp_path):
        """Its level band reaches every cell around it, so there is nowhere to discharge to.

        Contracting it would swallow the whole domain and hand back water nobody could place.
        Left alone it stays with the mechanism that already exists for water with nowhere to
        go — and the reason is recorded rather than left to be inferred.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        data = np.full((9, 9), 50.0, dtype="float32")
        data[2:7, 2:7] = 10.0
        path = _write_raster(str(tmp_path / "bowl.tif"), data)

        plain = FlowAnalysis()
        plain.load_dem(path)
        plain.run(routing="dinf", crest_split=False)

        split = FlowAnalysis()
        split.load_dem(path)
        result = split.run(routing="dinf")

        assert result["crest_ponds"] == 0
        assert "nothing discharges" in result["crest_skipped"][0]
        assert np.array_equal(np.asarray(plain.acc, dtype="float64"),
                              np.asarray(split.acc, dtype="float64"))

    def test_the_weighted_field_is_spread_the_same_way(self, tmp_path):
        """``runoff_accumulation`` feeds throughflow and the exit volumes.

        Spreading one field and not the other would have the two rasters beside each other
        disagree by the whole of the crest correction.
        """
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        path = self._staircase(tmp_path)
        fa = FlowAnalysis()
        fa.load_dem(path)
        one = np.asarray(
            fa.run(routing="dinf", runoff_weights=np.full((13, 5), 10.0))
            ["runoff_accumulation"], dtype="float64")

        # Spread, not left as the raw routing would have it: the outlet row is a weir.
        assert one[-1, 1:4] == pytest.approx([one[-1, 1]] * 3)


class TestPondRetention:
    """A pond keeps what it can hold, so the runoff raster is the water that actually gets
    downstream — the fix for a swale reading 46% full while the map drew a full channel
    leaving its pour point.

    The two pools of the staircase measure 9 cells x 3 m and 9 cells x 2 m on a 1 m grid,
    so the site's storage is 27 + 18 = 45 m3 and every figure below follows from that.
    """

    WALL = 100.0

    def _staircase(self, tmp_path):
        rows = [[self.WALL] * 5]
        rows += [[self.WALL] + [float(v)] * 3 + [self.WALL]
                 for v in (20, 12, 12, 12, 15, 14, 8, 8, 8, 10, 5)]
        rows += [[0.0] * 5]
        return _write_raster(str(tmp_path / "stair.tif"), np.array(rows, dtype="float32"))

    def _run(self, path, depth_m3_per_cell, crest_split=True):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(path)
        result = fa.run(routing="dinf", crest_split=crest_split,
                        runoff_weights=np.full((13, 5), float(depth_m3_per_cell)))
        return fa, result, np.asarray(result["runoff_accumulation"], dtype="float64")

    def _outlet(self, path, depth, **kw):
        return float(np.nansum(self._run(path, depth, **kw)[2][-1]))

    def test_the_cell_count_field_is_left_alone(self, tmp_path):
        """The guarantee streams, keypoint catchment sizing and time-of-concentration rest
        on. ``acc`` is contributing area, not a volume, and capping a cell count with cubic
        metres would be a category error — so retention touches the weighted field only."""
        path = self._staircase(tmp_path)
        fa, result, _ = self._run(path, 10.0)
        # The unchanged invariant from TestCrestSplit: 33 interior cells reach the outlet.
        assert np.asarray(fa.acc, dtype="float64")[-1].sum() == pytest.approx(33.0)
        assert result["crest_ponds"] == 2

    def test_what_the_outlet_loses_is_exactly_what_the_ponds_kept(self, tmp_path):
        """The conservation statement, against the same site with retention off. Water is
        not deleted from the map — it is accounted to the hollow holding it."""
        path = self._staircase(tmp_path)
        for depth in (1.0, 10.0, 20.0):
            _, result, runoff = self._run(path, depth)
            unretained = self._outlet(path, depth, crest_split=False)
            assert (float(np.nansum(runoff[-1])) + result["crest_retained_m3"]
                    == pytest.approx(unretained)), depth

    def test_a_storm_the_ponds_can_swallow_leaves_only_the_ground_below_them(self, tmp_path):
        """At 1 m3 a cell the site generates less than its 45 m3 of storage, so nothing from
        above the lower pond reaches the bottom — the outlet carries the two rows beneath it
        and nothing else. This is the case that used to draw a full channel anyway."""
        path = self._staircase(tmp_path)
        _, result, runoff = self._run(path, 1.0)
        assert result["crest_retained_m3"] == pytest.approx(30.0)
        # Its own cell plus the row above it, and no more.
        assert runoff[-1, 1:4] == pytest.approx([2.0, 2.0, 2.0])

    def test_a_pond_never_keeps_more_than_it_can_hold(self, tmp_path):
        """27 m3 and 18 m3 of pool, measured off the surface — however hard it rains."""
        path = self._staircase(tmp_path)
        for depth in (10.0, 20.0, 100.0):
            _, result, _ = self._run(path, depth)
            assert result["crest_retained_m3"] == pytest.approx(45.0), depth

    def test_the_storage_is_a_volume_not_a_fraction(self, tmp_path):
        """Which is the whole point: twice the rain overflows by more than twice as much,
        so this field is deliberately no longer linear in its weights. The unweighted
        accumulation still is — see ``test_the_cell_count_field_is_left_alone``."""
        path = self._staircase(tmp_path)
        one = self._outlet(path, 10.0)
        two = self._outlet(path, 20.0)
        assert two > 2.0 * one
        assert two - 2.0 * one == pytest.approx(45.0)

    def test_the_surplus_still_leaves_over_the_whole_crest(self, tmp_path):
        """Retention subtracts from what a pond sheds; it must not undo the even spread,
        or the fix for one wrong picture would reintroduce the other."""
        path = self._staircase(tmp_path)
        _, _, runoff = self._run(path, 10.0)
        assert runoff[-1, 1:4] == pytest.approx([runoff[-1, 1]] * 3)

    def test_a_dem_with_no_ponds_retains_nothing(self, sloped_dem):
        path = sloped_dem
        _, result, runoff = self._run_shaped(path, 2.0)
        assert result["crest_ponds"] == 0
        assert result.get("crest_retained_m3", 0.0) == pytest.approx(0.0)
        assert runoff.sum() > 0

    def _run_shaped(self, path, depth):
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(path)
        weights = np.full(np.asarray(fa.dem).shape, float(depth))
        result = fa.run(routing="dinf", runoff_weights=weights)
        return fa, result, np.asarray(result["runoff_accumulation"], dtype="float64")


class TestCrestSpreadWarning:
    def test_silent_below_the_threshold(self):
        from terrainflow_assessment.modules.flow_analysis import crest_spread_warning
        assert crest_spread_warning(1.0, 1000) is None

    def test_silent_when_nothing_is_held(self):
        from terrainflow_assessment.modules.flow_analysis import crest_spread_warning
        assert crest_spread_warning(0.0, 1000) is None
        assert crest_spread_warning(10.0, 0) is None

    def test_says_the_share_and_which_way_it_is_wrong(self):
        from terrainflow_assessment.modules.flow_analysis import crest_spread_warning
        msg = crest_spread_warning(50.0, 1000)
        assert "5.0%" in msg
        assert "under-reported" in msg
        assert "nothing is over-reported" in msg


class TestFlatInflationCannotInvertADrop:
    """The conditioning must not take away a way downhill the terrain already had.

    ``resolve_flats`` returns ``filled + eps * drainage_gradient``, and the gradient is an
    integer BFS distance to the flat's outlet — so on a large flat it reaches the
    thousands and pysheds' fixed ``eps = 1e-5`` lifts cells by centimetres. A cell that
    stood a few millimetres above such a flat ends up *under* it, with no lower neighbour
    left, and becomes a self-loop that swallows its whole upstream in silence.

    Measured on the reported Quail Island run: 516 cells, absorbing 12.5% of the tile's
    drainage, on a DEM 70% of which is one harbour plane. Every one of them had a strictly
    lower neighbour on the depression-filled surface and none on the conditioned one.
    """

    def _flat_with_a_low_bump(self, path, length=60, proud=1e-6):
        """A long flat draining off its east edge, with one cell standing *proud* of it.

        The bump is west of the outlet, so the flat beside it carries a large drainage
        gradient — the inflation there is many multiples of eps, while the bump's real
        height above the flat is a micrometre. That is the whole of the fault, at the
        smallest size that shows it.
        """
        data = np.full((9, length), 100.0, dtype="float64")
        data[:, -1] = 99.0                       # the flat's outlet, at the east edge
        data[4, 3] = 100.0 + proud               # a bump, its only way down onto the flat
        with rasterio.open(
            path, "w", driver="GTiff", height=9, width=length, count=1,
            dtype="float64", crs="EPSG:2193",
            transform=rasterio.transform.from_origin(0, 9, 1, 1),
        ) as dst:
            dst.write(data, 1)
        return path

    def _interior_pits(self, surface):
        """Cells that are their own 3x3 minimum, away from the grid border."""
        from scipy.ndimage import minimum_filter

        s = np.asarray(surface, dtype="float64")
        own = (s - minimum_filter(s, size=3, mode="nearest")) == 0
        own[0, :] = own[-1, :] = own[:, 0] = own[:, -1] = False
        return int(own.sum())

    def test_pysheds_default_eps_buries_the_bump(self, tmp_path):
        """The fault itself, so the fix is pinned to a demonstrated failure."""
        from pysheds.grid import Grid

        path = self._flat_with_a_low_bump(str(tmp_path / "flat.tif"))
        grid = Grid.from_raster(path)
        filled = grid.fill_depressions(grid.fill_pits(grid.read_raster(path)))
        inflated = np.asarray(grid.resolve_flats(filled), dtype="float64")
        assert self._interior_pits(inflated) > 0

    def test_the_derived_step_leaves_it_a_way_down(self, tmp_path):
        """Same surface, same flat resolution, a step scaled to what the drop can bear."""
        from pysheds.grid import Grid

        from terrainflow_assessment.modules.flow_analysis import resolve_flats_safely

        path = self._flat_with_a_low_bump(str(tmp_path / "flat.tif"))
        grid = Grid.from_raster(path)
        filled = grid.fill_depressions(grid.fill_pits(grid.read_raster(path)))
        inflated, eps, residual = resolve_flats_safely(grid, filled)

        assert self._interior_pits(np.asarray(inflated, dtype="float64")) == 0
        assert residual == 0
        assert eps < 1e-5

    def test_the_flats_are_still_resolved(self, tmp_path):
        """A gentler step is worthless if it stops draining the flat it inflates."""
        from pysheds.grid import Grid

        from terrainflow_assessment.modules.flow_analysis import resolve_flats_safely

        path = self._flat_with_a_low_bump(str(tmp_path / "flat.tif"))
        grid = Grid.from_raster(path)
        filled = grid.fill_depressions(grid.fill_pits(grid.read_raster(path)))
        inflated, _eps, _residual = resolve_flats_safely(grid, filled)

        z = np.asarray(inflated, dtype="float64")
        # Every cell of the flat sits strictly above the one east of it, so the synthetic
        # gradient survived being scaled down rather than being rounded away.
        row = z[4, 5:-1]
        assert np.all(np.diff(row) < 0)

    def test_run_reports_no_unrouted_cells(self, tmp_path):
        """End to end: the surface the fault was reported on no longer strands anything."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(self._flat_with_a_low_bump(str(tmp_path / "flat.tif")))
        result = fa.run(routing="dinf")

        assert result["unrouted_cells"] == 0
        assert fa.flat_inversions == 0
        assert 0 < fa.flat_eps <= 1e-5

    def test_diagnostics_report_the_step_that_was_used(self, tmp_path):
        """The next reader is told which step ran, not left to assume the default."""
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        fa = FlowAnalysis()
        fa.load_dem(self._flat_with_a_low_bump(str(tmp_path / "flat.tif")))
        fa.run(routing="dinf")

        diag = fa.unrouted_diagnostics()
        assert diag["flat_resolution"]["eps"] == fa.flat_eps
        assert diag["flat_resolution"]["pysheds_default_eps"] == 1e-5
        assert diag["flat_resolution"]["inversions_remaining"] == 0


class TestSafeFlatEpsilon:
    """The bound itself, away from pysheds."""

    def test_no_flats_keeps_the_ceiling(self):
        from terrainflow_assessment.modules.flow_analysis import safe_flat_epsilon

        z = np.arange(25, dtype="float64").reshape(5, 5)
        eps, residual = safe_flat_epsilon(z, np.zeros_like(z))
        assert eps == 1e-5
        assert residual == 0

    def test_a_generous_drop_keeps_the_ceiling(self):
        """Metres of relief between the flat and its neighbours: nothing to protect."""
        from terrainflow_assessment.modules.flow_analysis import safe_flat_epsilon

        z = np.full((5, 5), 10.0)
        z[2, 2] = 20.0                       # a neighbour a whole 10 m above the flat
        grad = np.zeros((5, 5))
        grad[2, 3] = 500.0                   # inflated by 5 mm at eps = 1e-5
        eps, residual = safe_flat_epsilon(z, grad)
        assert eps == 1e-5
        assert residual == 0

    def test_a_sub_eps_drop_pulls_the_step_down(self):
        """1 mm above a flat carrying 500 gradient units: 1e-5 would bury it."""
        from terrainflow_assessment.modules.flow_analysis import safe_flat_epsilon

        z = np.full((5, 5), 10.0)
        z[2, 2] = 10.001
        grad = np.zeros((5, 5))
        grad[2, 3] = 500.0
        eps, residual = safe_flat_epsilon(z, grad)
        assert eps == pytest.approx(0.5 * 0.001 / 500.0)
        assert eps * 500.0 < 0.001            # the neighbour stays below the bump
        assert residual == 0

    def test_the_floor_binds_and_says_so(self):
        """A drop below float64's own resolution cannot be protected — and is counted.

        Reported rather than absorbed: the count is exactly the residue the bound exists
        to remove, and a silent zero would read as "nothing was buried".
        """
        from terrainflow_assessment.modules.flow_analysis import safe_flat_epsilon

        z = np.full((5, 5), 1000.0)
        z[2, 2] = np.nextafter(1000.0, np.inf)   # one ULP proud of the flat
        grad = np.zeros((5, 5))
        grad[2, 3] = 1000.0
        eps, residual = safe_flat_epsilon(z, grad)
        assert eps > 0
        assert residual > 0

    def test_opposite_grid_edges_are_not_neighbours(self):
        """A wrapped pair would bind the bound with a drop that spans the whole tile.

        ``np.roll`` pairs row 0 with row -1 as though they touched. Here the only pair
        that could ever bind is exactly that one, so a rolling implementation returns a
        step 1000x smaller than the terrain calls for.
        """
        from terrainflow_assessment.modules.flow_analysis import safe_flat_epsilon

        z = np.full((5, 5), 10.0)
        z[0, :] = 10.000001          # top row a micrometre proud of everything
        grad = np.zeros((5, 5))
        grad[-1, :] = 1000.0         # ...and only the bottom row is inflated
        eps, residual = safe_flat_epsilon(z, grad)
        assert eps == 1e-5
        assert residual == 0

    def test_shape_mismatch_is_an_error(self):
        from terrainflow_assessment.modules.flow_analysis import safe_flat_epsilon

        with pytest.raises(ValueError):
            safe_flat_epsilon(np.zeros((4, 4)), np.zeros((5, 5)))
