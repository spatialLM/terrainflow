"""
Tests for terrainflow_assessment/modules/catchment.py

Covers the SCSRunoff class (duplicate of plugin's but in assessment package),
clip_dem_to_polygon, and fast_contributing_area.
"""
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from terrainflow_assessment.modules.catchment import (
    SCSRunoff,
    clip_dem_to_polygon,
    fast_contributing_area,
)


# ---------------------------------------------------------------------------
# SCSRunoff — assessment package version
# ---------------------------------------------------------------------------

class TestSCSRunoffAssessment:
    """Mirror tests from test_scs_runoff.py to hit catchment.py coverage."""

    def setup_method(self):
        self.scs = SCSRunoff()

    # adjust_cn
    def test_normal_unchanged(self):
        assert self.scs.adjust_cn(70, "normal") == pytest.approx(70.0)

    def test_dry_lower(self):
        assert self.scs.adjust_cn(70, "dry") < 70.0

    def test_wet_higher(self):
        assert self.scs.adjust_cn(70, "wet") > 70.0

    def test_clamp_high(self):
        assert self.scs.adjust_cn(99, "wet") <= 100.0

    def test_clamp_low(self):
        assert self.scs.adjust_cn(2, "dry") >= 1.0

    # runoff_depth
    def test_zero_cn(self):
        assert self.scs.runoff_depth(100, 0) == 0.0

    def test_below_ia(self):
        assert self.scs.runoff_depth(5, 70) == 0.0

    def test_above_ia(self):
        assert self.scs.runoff_depth(100, 70) > 0.0

    def test_cn_100_equals_rainfall(self):
        assert self.scs.runoff_depth(50, 100) == pytest.approx(50.0, rel=1e-3)

    def test_never_negative(self):
        for cn in [10, 50, 75, 98]:
            for rain in [0, 5, 50, 200]:
                assert self.scs.runoff_depth(rain, cn) >= 0.0

    # runoff_ratio
    def test_ratio_zero_rain(self):
        assert self.scs.runoff_ratio(0, 75) == 0.0

    def test_ratio_between_0_and_1(self):
        r = self.scs.runoff_ratio(100, 75)
        assert 0.0 <= r <= 1.0

    # catchment_volume
    def test_10mm_1ha(self):
        assert self.scs.catchment_volume(10.0, 10_000) == pytest.approx(100.0)

    def test_zero_runoff(self):
        assert self.scs.catchment_volume(0.0, 10_000) == 0.0

    # build_runoff_raster
    def test_raster_shape(self):
        cn = np.full((5, 5), 75, dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        assert result.shape == (5, 5)

    def test_raster_zero_cn(self):
        cn = np.zeros((3, 3), dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        assert np.all(result == 0.0)

    def test_raster_dtype(self):
        cn = np.full((3, 3), 70, dtype="float32")
        assert self.scs.build_runoff_raster(cn, 50.0).dtype == np.float32

    # build_cn_raster
    def test_cn_raster_default(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "normal")
        assert np.allclose(result, 70.0)

    def test_cn_raster_dry(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "dry")
        assert result.mean() < 70.0

    def test_cn_raster_wet(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "wet")
        assert result.mean() > 70.0

    def test_cn_raster_zone_override(self):
        shape = (10, 10)
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        zone = box(0, 5, 5, 10)
        result = self.scs.build_cn_raster(shape, transform, [(zone, 90)], 70, "normal")
        assert result.max() > 70.0

    def test_cn_raster_none_geom_skipped(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [(None, 90)], 70, "normal")
        assert np.allclose(result, 70.0)

    def test_cn_raster_empty_geom_skipped(self):
        from shapely.geometry import Polygon
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [(Polygon(), 90)], 70, "normal")
        assert np.allclose(result, 70.0)

    def test_cn_raster_float32(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "normal")
        assert result.dtype == np.float32

    # parse_hyetograph_csv
    def test_parse_cumulative(self, tmp_path):
        import csv as _csv
        p = str(tmp_path / "rain.csv")
        with open(p, "w", newline="") as f:
            _csv.writer(f).writerows([
                ["time_min", "rainfall_mm"], [10, 5], [20, 15], [30, 30]
            ])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[0] == (0, 0.0)
        assert any(t == 10 for t, _ in result)

    def test_parse_missing_column_raises(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("a,b\n1,2\n")
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_parse_empty_raises(self, tmp_path):
        p = str(tmp_path / "empty.csv")
        open(p, "w").close()
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_parse_per_interval(self, tmp_path):
        import csv as _csv
        p = str(tmp_path / "rain.csv")
        with open(p, "w", newline="") as f:
            _csv.writer(f).writerows([
                ["time_min", "rainfall_mm"], [10, 5], [20, 10], [30, 3]
            ])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[-1][1] == pytest.approx(18.0, abs=0.01)

    # constants
    def test_soil_reference(self):
        assert "Clay" in SCSRunoff.SOIL_REFERENCE

    def test_storm_presets(self):
        assert "Custom" in SCSRunoff.STORM_PRESETS


# ---------------------------------------------------------------------------
# clip_dem_to_polygon (assessment version)
# ---------------------------------------------------------------------------

class TestClipDEMToPolygonCatchment:
    def test_creates_file(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_dem_to_polygon(tmp_dem, box(3, 3, 15, 15), out)
        assert os.path.exists(out)

    def test_returns_path(self, tmp_dem, tmp_path):
        out = str(tmp_path / "c.tif")
        assert clip_dem_to_polygon(tmp_dem, box(3, 3, 15, 15), out) == out


# ---------------------------------------------------------------------------
# SCSRunoff CSV gap coverage (lines 357-358, 361)
# ---------------------------------------------------------------------------

class TestSCSRunoffCSVGaps:
    def test_invalid_rows_skipped(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm\n10,5\nbad,row\n20,15\n")
        result = SCSRunoff.parse_hyetograph_csv(p)
        times = [t for t, _ in result]
        assert 10 in times and 20 in times

    def test_no_valid_rows_raises(self, tmp_path):
        p = str(tmp_path / "empty_rows.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm\nx,y\n")
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_build_runoff_raster_below_ia_returns_zero(self):
        """Line 326: rainfall <= Ia → _q returns 0.0."""
        scs = SCSRunoff()
        cn = np.full((3, 3), 70, dtype="float32")
        # CN=70: S≈108.9, Ia≈21.8 — rainfall=5 is below Ia
        result = scs.build_runoff_raster(cn, 5.0)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# fast_contributing_area (assessment version)
# ---------------------------------------------------------------------------

def _write_sloped_dem(path, shape=(25, 25), cell_size=1.0, crs="EPSG:32632"):
    data = np.fromfunction(
        lambda r, c: 100.0 - r * 2.0 - c * 0.5, shape
    ).astype("float32")
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


class TestFastContributingAreaAssessment:
    def _setup(self, tmp_path):
        dem = _write_sloped_dem(str(tmp_path / "dem.tif"))
        gdf = gpd.GeoDataFrame(geometry=[box(5, 5, 15, 15)], crs="EPSG:32632")
        bp = str(tmp_path / "boundary.gpkg")
        gdf.to_file(bp, driver="GPKG")
        return dem, bp

    def test_returns_required_keys(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        result = fast_contributing_area(dem, bp)
        for key in ("catchment_polygon", "area_ha", "dem_area_ha",
                    "coverage_pct", "scale"):
            assert key in result

    def test_area_positive(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        result = fast_contributing_area(dem, bp)
        assert result["area_ha"] >= 0.0
        assert not result["catchment_polygon"].is_empty

    def test_scale_1_small_dem(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        result = fast_contributing_area(dem, bp)
        assert result["scale"] == pytest.approx(1.0)

    def test_progress_callback(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        pcts = []
        fast_contributing_area(dem, bp, progress_callback=lambda p, m: pcts.append(p))
        assert len(pcts) > 0
        assert pcts[-1] == 100

    def test_boundary_outside_raises(self, tmp_path):
        dem = _write_sloped_dem(str(tmp_path / "dem2.tif"))
        gdf = gpd.GeoDataFrame(geometry=[box(200, 200, 300, 300)], crs="EPSG:32632")
        bp = str(tmp_path / "far.gpkg")
        gdf.to_file(bp, driver="GPKG")
        with pytest.raises(RuntimeError):
            fast_contributing_area(dem, bp)

    def test_nodata_cells_handled(self, tmp_path):
        data = np.fromfunction(
            lambda r, c: 100.0 - r * 2.0, (25, 25)
        ).astype("float32")
        data[0:3, 0:3] = -9999.0
        dem = str(tmp_path / "nd_dem.tif")
        h, w = data.shape
        transform = from_bounds(0, 0, 25, 25, w, h)
        with rasterio.open(dem, "w", driver="GTiff", height=h, width=w,
                           count=1, dtype="float32", crs="EPSG:32632",
                           transform=transform, nodata=-9999.0) as dst:
            dst.write(data, 1)
        gdf = gpd.GeoDataFrame(geometry=[box(8, 8, 18, 18)], crs="EPSG:32632")
        bp = str(tmp_path / "b2.gpkg")
        gdf.to_file(bp, driver="GPKG")
        result = fast_contributing_area(dem, bp)
        assert not result["catchment_polygon"].is_empty
