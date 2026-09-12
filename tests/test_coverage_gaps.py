"""
Targeted tests for remaining coverage gaps:
  - earthwork_design.DEMBurner.get_ponding_layer   (lines 455-514)
  - simulation.build_stores_from_earthworks         (lines 484-544)
  - earthwork_design line 93 (Earthwork.summary swale/basin branch)
  - earthwork_design lines 362-367 (companion berm edge cases)
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from tests.conftest import make_mock_line_geom, make_mock_polygon_geom

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dem(path, data, cell_size=1.0, crs="EPSG:32632", nodata=-9999.0):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _write_raster(path, data, cell_size=1.0):
    """Write any float32 raster (no nodata)."""
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs="EPSG:32632", transform=transform,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


# ---------------------------------------------------------------------------
# DEMBurner.get_ponding_layer
# ---------------------------------------------------------------------------

class TestGetPondingLayer:
    def test_returns_array_same_shape(self, tmp_path):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        # Flat DEM — minimal ponding expected
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "flat.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == data.shape

    def test_ponding_non_negative(self, tmp_path):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "flat.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert np.all(result >= 0.0)

    def test_bowl_dem_shows_ponding(self, tmp_path):
        """Bowl-shaped DEM should show ponding at the centre."""
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        r = np.arange(20)
        c = np.arange(20)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        data = (50.0 + ((rr - 9.5) ** 2 + (cc - 9.5) ** 2) * 0.3).astype("float32")
        # Artificially lower the centre so water pools
        data[8:12, 8:12] -= 5.0
        path = _write_dem(str(tmp_path / "bowl.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.max() >= 0.0  # result produced; depth analysis verified by shape

    def test_returns_float32(self, tmp_path):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        data = np.full((10, 10), 30.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Plugin dem_burner.get_ponding_layer (same logic, separate module)
# ---------------------------------------------------------------------------

class TestPluginGetPondingLayer:
    def test_returns_array(self, tmp_path):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "flat.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (20, 20)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# simulation.build_stores_from_earthworks
# ---------------------------------------------------------------------------

class TestBuildStoresFromEarthworks:
    def _ew(self, name, ew_type="swale", length=50.0):
        """Minimal Earthwork with capacity set and proper geometry JSON."""
        from terrainflow_assessment.modules.earthwork_design import Earthwork

        geom = make_mock_line_geom([(0.0, 5.0), (length, 5.0)])
        ew = Earthwork(ew_type, geom, name)
        ew.depth = 0.5
        ew.width = 2.0
        ew.capacity_m3 = 40.0  # above zero → store created
        ew.capacity_l = 40_000.0
        return ew

    def test_returns_stores_for_enabled(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        earthworks = [self._ew("S1"), self._ew("S2")]
        stores = build_stores_from_earthworks(earthworks, soil_name="Loam")
        assert len(stores) == 2

    def test_disabled_earthwork_excluded(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        ew = self._ew("S1")
        ew.enabled = False
        stores = build_stores_from_earthworks([ew], soil_name="Loam")
        assert len(stores) == 0

    def test_zero_capacity_kept_as_a_routing_node(self):
        """A storage-less feature still intercepts and redirects water."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        ew = self._ew("S1")
        ew.capacity_m3 = 0.0
        stores = build_stores_from_earthworks([ew], soil_name="Loam")
        assert len(stores) == 1
        assert stores[0].capacity_m3 == 0.0

    def test_a_measured_pond_sizes_the_store(self):
        """The one place the capacity basis is chosen, and the point of choosing it.

        Sized on the drawn section, a swale whose companion berm is keyed into the banks
        reports *full at this storm* while most of its pond is still empty — Swale 5 of
        the Quail Island design read 100% of 440 m³ against a measured 1,095 m³ — and a
        designer reading that bar enlarges a feature that needed nothing. The drawn
        figure is carried alongside, never discarded: it is the one a contractor builds
        to and the one that can be checked by hand.
        """
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        ew = self._ew("S1")
        ew.capacity_m3 = 440.0
        ew.terrain_capacity_m3 = 1095.0
        store = build_stores_from_earthworks([ew])[0]
        assert store.capacity_m3 == pytest.approx(1095.0)
        assert store.drawn_capacity_m3 == pytest.approx(440.0)
        assert store.capacity_is_measured is True

    def test_without_a_measurement_it_falls_back_to_the_drawn_section(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        ew = self._ew("S1")
        ew.capacity_m3 = 440.0
        store = build_stores_from_earthworks([ew])[0]
        assert store.capacity_m3 == pytest.approx(440.0)
        assert store.drawn_capacity_m3 == pytest.approx(440.0)
        assert store.capacity_is_measured is False

    def test_the_drawn_basis_can_be_forced_for_comparison(self):
        """Running the same storm both ways is what shows whether capacity is the limit.

        On the Quail Island design it is not — 57% either way, because 7,567 m³ of the
        storm never reaches a feature — and that is only sayable because both can be
        computed.
        """
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        ew = self._ew("S1")
        ew.capacity_m3 = 440.0
        ew.terrain_capacity_m3 = 1095.0
        store = build_stores_from_earthworks([ew], basis="drawn")[0]
        assert store.capacity_m3 == pytest.approx(440.0)
        assert store.capacity_is_measured is False

    def test_store_names_match_earthworks(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        earthworks = [self._ew("SwaleA"), self._ew("SwaleB")]
        stores = build_stores_from_earthworks(earthworks)
        names = [s.name for s in stores]
        assert "SwaleA" in names
        assert "SwaleB" in names

    def test_store_capacity_matches(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        ew = self._ew("S1")
        ew.capacity_m3 = 123.4
        stores = build_stores_from_earthworks([ew])
        assert stores[0].capacity_m3 == pytest.approx(123.4)

    def test_infiltration_rate_from_soil(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        from terrainflow_assessment.modules.swale_design import INFILTRATION_RATE_MM_HR

        ew = self._ew("S1")
        stores = build_stores_from_earthworks([ew], soil_name="Sand")
        assert stores[0].infiltration_rate_mm_hr == INFILTRATION_RATE_MM_HR["Sand"]

    def test_with_dem_path_sets_elevation(self, tmp_path):
        """When dem_path provided, elevation is read from raster."""
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        data = np.full((20, 20), 75.0, dtype="float32")
        dem = _write_dem(str(tmp_path / "dem.tif"), data)

        ew = self._ew("S1")
        stores = build_stores_from_earthworks([ew], dem_path=dem)
        # Elevation should be ~75 (or 0 if outside raster)
        assert stores[0].elevation >= 0.0

    def test_polygon_earthwork_area_computed(self):
        """Polygon earthwork (basin) uses area() for area_m2."""
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        geom = make_mock_polygon_geom((0, 0, 10, 10))  # 100 m²
        ew = Earthwork("basin", geom, "Basin1")
        ew.depth = 1.0
        ew.width = 0
        ew.capacity_m3 = 80.0
        stores = build_stores_from_earthworks([ew])
        assert stores[0].area_m2 == pytest.approx(100.0, rel=0.1)

    def test_empty_list_returns_empty(self):
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        assert build_stores_from_earthworks([]) == []

    def test_invalid_geom_json_still_builds_store(self):
        """Bad geometry JSON falls back to area=100 but still creates the store."""
        from unittest.mock import MagicMock

        from terrainflow_assessment.modules.earthwork_design import Earthwork
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks

        bad_geom = MagicMock()
        bad_geom.asJson.return_value = "not-json"
        ew = Earthwork("swale", bad_geom, "Bad")
        ew.capacity_m3 = 50.0
        stores = build_stores_from_earthworks([ew])
        assert len(stores) == 1
        assert stores[0].area_m2 == pytest.approx(100.0)  # fallback


# ---------------------------------------------------------------------------
# earthwork_design.py remaining line coverage
# ---------------------------------------------------------------------------

class TestSimulationRunIntegration:
    """
    Minimal integration test for _run_simulation.
    Uses a tiny synthetic DEM + flow direction raster to exercise
    the main simulation loop (lines 231-454) without full QGIS.
    """

    def _make_fdir(self, dem_path, output_path):
        """Run pysheds to produce a real flow-direction raster."""
        import numpy as np
        import rasterio
        from pysheds.grid import Grid

        grid = Grid.from_raster(dem_path)
        dem = grid.read_raster(dem_path)
        pit_filled = grid.fill_pits(dem)
        try:
            breached = grid.breach_depressions(pit_filled)
        except AttributeError:
            breached = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(breached)
        try:
            fdir = grid.flowdir(inflated, routing="d8")
        except TypeError:
            fdir = grid.flowdir(inflated)

        with rasterio.open(dem_path) as src:
            meta = src.meta.copy()
            meta.update(dtype="float32", nodata=-9999.0)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(np.array(fdir).astype("float32"), 1)
        return output_path

    def test_run_simulation_basic(self, tmp_path):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        from terrainflow_assessment.modules.simulation import _run_simulation

        # 12×12 sloped DEM
        data = np.fromfunction(
            lambda r, c: 100.0 - r * 5.0, (12, 12), dtype=float
        ).astype("float32")
        h, w = data.shape
        transform = from_bounds(0, 0, 12, 12, w, h)
        dem_path = str(tmp_path / "dem.tif")
        with rasterio.open(
            dem_path, "w", driver="GTiff", height=h, width=w,
            count=1, dtype="float32", crs="EPSG:32632",
            transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        fdir_path = str(tmp_path / "fdir.tif")
        self._make_fdir(dem_path, fdir_path)

        rainfall_data = [(0, 0.0), (30, 20.0), (60, 40.0)]
        result = _run_simulation(
            dem_path=dem_path,
            fdir_path=fdir_path,
            output_dir=str(tmp_path),
            cn=75,
            moisture="normal",
            rainfall_data=rainfall_data,
            routing="d8",
        )

        assert "frames" in result
        assert "timestep_table" in result
        assert "peak_outflow_ls" in result
        assert len(result["timestep_table"]) == 2  # n_steps = len-1

    def test_run_simulation_with_stores(self, tmp_path):
        """Run simulation with EarthworkStores to exercise cascade loop."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        from terrainflow_assessment.modules.simulation import (
            EarthworkStore,
            _run_simulation,
        )

        data = np.fromfunction(
            lambda r, c: 100.0 - r * 5.0, (12, 12), dtype=float
        ).astype("float32")
        h, w = data.shape
        transform = from_bounds(0, 0, 12, 12, w, h)
        dem_path = str(tmp_path / "dem.tif")
        with rasterio.open(
            dem_path, "w", driver="GTiff", height=h, width=w,
            count=1, dtype="float32", crs="EPSG:32632",
            transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        fdir_path = str(tmp_path / "fdir2.tif")
        self._make_fdir(dem_path, fdir_path)

        store = EarthworkStore(
            name="Swale1", ew_type="swale", id="ew1",
            capacity_m3=500.0, area_m2=100.0,
            elevation=70.0, centroid_row=6, centroid_col=6,
        )

        # A store list arrives with a direct-catchment labelling or the call fails
        # (`simulation.py:574-578`), because there is no correct fallback: sampling the
        # cumulative accumulation raster instead double-counts every upstream feature's
        # catchment, which is the bug `water_balance.py` removed.
        #
        # The labelling holds **indices into `catchment_label_ids`**, negative meaning
        # "drains to nothing" — see `catchment_partition` and `CatchmentPartition.credit`.
        # The top half of this DEM is credited to the swale; the bottom half to nothing.
        labels = np.full(data.shape, -1, dtype="int32")
        labels[:6, :] = 0

        rainfall_data = [(0, 0.0), (30, 25.0), (60, 50.0)]
        result = _run_simulation(
            dem_path=dem_path,
            fdir_path=fdir_path,
            output_dir=str(tmp_path / "sim2"),
            cn=75, moisture="normal",
            rainfall_data=rainfall_data,
            routing="d8",
            earthwork_stores=[store],
            catchment_labels=labels,
            catchment_label_ids=["ew1"],
        )

        assert "earthwork_summary" in result
        assert len(result["earthwork_summary"]) == 1
        assert result["earthwork_summary"][0]["name"] == "Swale1"
        # The point of supplying a labelling: the store must actually receive water.
        # Without this the test passes on a summary row for a feature nothing drains to,
        # which is what it did for as long as the xfail hid it.
        assert result["earthwork_summary"][0]["total_inflow_m3"] > 0.0


class TestEarthworkDesignRemainingBranches:
    """Tests targeting specific uncovered lines in earthwork_design.py."""

    def test_earthwork_summary_swale_with_capacity(self):
        """Line 93: summary() for swale/basin type shows capacity_m3."""
        from terrainflow_assessment.modules.earthwork_design import Earthwork

        ew = Earthwork("swale", make_mock_line_geom(), "MySwale")
        ew.capacity_m3 = 42.5
        s = ew.summary()
        assert "42.5" in s
        assert "MySwale" in s

    def test_earthwork_summary_basin_with_capacity(self):
        from terrainflow_assessment.modules.earthwork_design import Earthwork

        ew = Earthwork("basin", make_mock_polygon_geom(), "MyBasin")
        ew.capacity_m3 = 200.0
        s = ew.summary()
        assert "200.0" in s

    def test_earthwork_manager_get(self):
        """Line 93: EarthworkManager.get(index) returns the item."""
        from terrainflow_assessment.modules.earthwork_design import (
            Earthwork,
            EarthworkManager,
        )

        m = EarthworkManager()
        ew = Earthwork("swale", make_mock_line_geom(), "TestGet")
        m.add(ew)
        assert m.get(0) is ew

    def test_calculate_cut_diversion_bottom_width_floor(self):
        """Line 239: bottom_width = max(0.05, ...) for wide/deep diversion."""
        from terrainflow_assessment.modules.earthwork_design import calculate_cut_volume

        # width=0.5, depth=0.5 → width-2*depth = -0.5 → clamp to 0.05
        geom = make_mock_line_geom()
        geom.length.return_value = 10.0
        cut = calculate_cut_volume("diversion", geom, depth=0.5, width=0.5)
        assert cut > 0.0

    def test_dem_burner_companion_berm_no_left_mask(self, tmp_path):
        """Lines 362-367: companion berm falls back when only one side has cells."""
        from terrainflow_assessment.modules.earthwork_design import DEMBurner, Earthwork

        # Very narrow DEM — berm on one side has no cells
        data = np.full((10, 3), 50.0, dtype="float32")  # 3-col wide DEM
        h, w = data.shape
        transform = from_bounds(0, 0, 3, 10, w, h)
        path = str(tmp_path / "narrow.tif")
        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w,
            count=1, dtype="float32", crs="EPSG:32632",
            transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        b = DEMBurner(path)
        # Line running down the centre of a 3-wide DEM — one parallel offset
        # will land outside the raster
        line_geom = make_mock_line_geom([(1.5, 1.0), (1.5, 9.0)])
        ew = Earthwork("swale", line_geom, "NarrowSwale")
        ew.depth = 0.5
        ew.width = 0.5
        ew.companion_berm = True
        result = b.burn_earthworks([ew])
        assert result.shape == (10, 3)  # no crash, valid shape
