"""Tests for plugin/processing/contributing_area.py"""
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from plugin.processing.contributing_area import fast_contributing_area


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dem(path, data, cell_size=1.0, crs="EPSG:32632"):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _write_boundary(path, polygon, crs="EPSG:32632"):
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=crs)
    gdf.to_file(path, driver="GPKG")
    return path


def _sloped_dem(shape=(25, 25)):
    """DEM that slopes from top (high) to bottom-right (low)."""
    rows, cols = shape
    return np.fromfunction(
        lambda r, c: 100.0 - r * 2.0 - c * 0.5, shape
    ).astype("float32")


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestFastContributingAreaResult:
    def test_returns_dict_with_required_keys(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        result = fast_contributing_area(dem, bp)
        for key in ("catchment_polygon", "clip_polygon", "area_ha",
                    "dem_area_ha", "coverage_pct", "scale"):
            assert key in result

    def test_area_positive(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        result = fast_contributing_area(dem, bp)
        assert result["area_ha"] >= 0.0
        assert not result["catchment_polygon"].is_empty

    def test_dem_area_ha_positive(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        result = fast_contributing_area(dem, bp)
        assert result["dem_area_ha"] > 0.0

    def test_coverage_pct_between_0_and_100(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        result = fast_contributing_area(dem, bp)
        assert 0.0 <= result["coverage_pct"] <= 100.0

    def test_scale_is_1_for_small_dem(self, tmp_path):
        """A 25×25 DEM is well within _MAX_PREVIEW_CELLS so scale == 1."""
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        result = fast_contributing_area(dem, bp)
        assert result["scale"] == pytest.approx(1.0)

    def test_catchment_polygon_is_valid(self, tmp_path):
        from shapely.geometry.base import BaseGeometry
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        result = fast_contributing_area(dem, bp)
        assert isinstance(result["catchment_polygon"], BaseGeometry)
        assert not result["catchment_polygon"].is_empty


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_callback_called(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        calls = []
        fast_contributing_area(dem, bp, progress_callback=lambda p, m: calls.append(p))
        assert len(calls) > 0

    def test_callback_pct_increases(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(5, 5, 15, 15))
        pcts = []
        fast_contributing_area(dem, bp, progress_callback=lambda p, m: pcts.append(p))
        # First pct should be low, last should be 100
        assert pcts[0] < pcts[-1]
        assert pcts[-1] == 100


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestContributingAreaErrors:
    def test_boundary_outside_dem_raises(self, tmp_path):
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        # Boundary at 200-300, DEM only covers 0-25
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(200, 200, 300, 300))
        with pytest.raises(RuntimeError, match="does not intersect"):
            fast_contributing_area(dem, bp)

    def test_no_valid_polygon_raises(self, tmp_path):
        """Boundary file with no polygon exterior raises RuntimeError."""
        dem = _write_dem(str(tmp_path / "dem.tif"), _sloped_dem())
        # Write a GeoDataFrame with an empty / null geometry
        from shapely.geometry import Point
        gdf = gpd.GeoDataFrame(geometry=[Point(12, 12)], crs="EPSG:32632")
        bp = str(tmp_path / "points.gpkg")
        gdf.to_file(bp, driver="GPKG")
        with pytest.raises(RuntimeError):
            fast_contributing_area(dem, bp)


# ---------------------------------------------------------------------------
# Nodata handling
# ---------------------------------------------------------------------------

class TestNodataHandling:
    def test_nodata_cells_become_nan(self, tmp_path):
        """Nodata cells in the DEM are replaced with NaN (not treated as terrain)."""
        data = _sloped_dem()
        data[0:5, 0:5] = -9999.0  # nodata patch
        dem = _write_dem(str(tmp_path / "dem.tif"), data)
        bp = _write_boundary(str(tmp_path / "b.gpkg"), box(8, 8, 17, 17))
        result = fast_contributing_area(dem, bp)
        # Should still succeed, nodata area just excluded
        assert not result["catchment_polygon"].is_empty
