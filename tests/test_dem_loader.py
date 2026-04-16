"""Tests for terrainflow_assessment/modules/dem_loader.py"""
import os

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from terrainflow_assessment.modules.dem_loader import (
    DEMInfo,
    clip_dem_to_polygon,
    compute_slope_raster,
    load_dem,
)


# ---------------------------------------------------------------------------
# load_dem
# ---------------------------------------------------------------------------

class TestLoadDEM:
    def test_returns_dem_info(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert isinstance(info, DEMInfo)

    def test_path_preserved(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.path == tmp_dem

    def test_dimensions_correct(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.width == 20
        assert info.height == 20

    def test_cell_size(self, tmp_dem):
        # 20m extent / 20 cells = 1m cell size
        info = load_dem(tmp_dem)
        assert info.cell_size_m == pytest.approx(1.0, rel=1e-4)

    def test_cell_area(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.cell_area_m2 == pytest.approx(1.0, rel=1e-4)

    def test_area_ha(self, tmp_dem):
        # 20×20 m = 400 m² = 0.04 ha
        info = load_dem(tmp_dem)
        assert info.area_ha == pytest.approx(0.04, rel=1e-4)

    def test_crs_is_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.crs is not None

    def test_crs_wkt_is_string(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert isinstance(info.crs_wkt, str)
        assert len(info.crs_wkt) > 10

    def test_transform_is_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.transform is not None

    def test_bounds_is_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.bounds is not None
        left, bottom, right, top = info.bounds
        assert right > left
        assert top > bottom

    def test_nodata_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.nodata == pytest.approx(-9999.0)

    def test_invalid_path_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Cannot load DEM"):
            load_dem("/no/such/file.tif")

    def test_repr_contains_dimensions(self, tmp_dem):
        info = load_dem(tmp_dem)
        r = repr(info)
        assert "20" in r  # width or height


# ---------------------------------------------------------------------------
# DEMInfo repr
# ---------------------------------------------------------------------------

class TestDEMInfoRepr:
    def test_repr_contains_cell_size(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert "1.00 m" in repr(info)

    def test_repr_contains_area(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert "ha" in repr(info)


# ---------------------------------------------------------------------------
# clip_dem_to_polygon
# ---------------------------------------------------------------------------

class TestClipDEMToPolygon:
    def test_creates_output_file(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_polygon = box(2, 2, 15, 15)  # within the 20×20 DEM
        clip_dem_to_polygon(tmp_dem, clip_polygon, out)
        assert os.path.exists(out)

    def test_clipped_smaller_than_original(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_polygon = box(5, 5, 15, 15)
        clip_dem_to_polygon(tmp_dem, clip_polygon, out)
        with rasterio.open(out) as src:
            w, h = src.width, src.height
        assert w < 20 or h < 20

    def test_returns_output_path(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        result = clip_dem_to_polygon(tmp_dem, box(2, 2, 15, 15), out)
        assert result == out

    def test_crs_preserved(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_dem_to_polygon(tmp_dem, box(2, 2, 15, 15), out)
        with rasterio.open(out) as src:
            assert src.crs is not None


# ---------------------------------------------------------------------------
# compute_slope_raster
# ---------------------------------------------------------------------------

class TestComputeSlopeRaster:
    def test_creates_output_file(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        compute_slope_raster(tmp_dem, out)
        assert os.path.exists(out)

    def test_returns_output_path(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        result = compute_slope_raster(tmp_dem, out)
        assert result == out

    def test_slope_non_negative(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        compute_slope_raster(tmp_dem, out)
        with rasterio.open(out) as src:
            data = src.read(1)
        # All slope values ≥ 0 (nodata masked)
        nodata = -9999.0
        valid = data[data != nodata]
        assert np.all(valid >= 0.0)

    def test_flat_dem_low_slope(self, tmp_dem_flat, tmp_path):
        out = str(tmp_path / "slope_flat.tif")
        compute_slope_raster(tmp_dem_flat, out)
        with rasterio.open(out) as src:
            data = src.read(1)
        nodata = -9999.0
        valid = data[data != nodata]
        # Flat DEM interior should have near-zero slope
        assert np.percentile(valid, 50) < 5.0  # median slope < 5°

    def test_same_shape_as_input(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        compute_slope_raster(tmp_dem, out)
        with rasterio.open(tmp_dem) as src_in:
            in_shape = (src_in.height, src_in.width)
        with rasterio.open(out) as src_out:
            out_shape = (src_out.height, src_out.width)
        assert in_shape == out_shape

    def test_dem_with_nodata(self, tmp_dem_flat, tmp_path):
        """Should not crash when DEM has nodata cells."""
        # Patch one cell to nodata
        import shutil
        patched = str(tmp_path / "nodata_dem.tif")
        shutil.copy(tmp_dem_flat, patched)
        with rasterio.open(patched, "r+") as dst:
            arr = dst.read(1)
            arr[0, 0] = -9999.0
            dst.write(arr, 1)

        out = str(tmp_path / "slope_nodata.tif")
        compute_slope_raster(patched, out)  # should not raise
        assert os.path.exists(out)
