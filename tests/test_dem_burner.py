"""Tests for plugin/processing/dem_burner.py"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box, mapping

from plugin.processing.dem_burner import DEMBurner
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


def _mock_ew(ew_type, geom, **kwargs):
    """Build a minimal mock earthwork."""
    from unittest.mock import MagicMock

    ew = MagicMock()
    ew.type = ew_type
    ew.geometry = geom
    ew.enabled = True
    ew.depth = kwargs.get("depth", 0.5)
    ew.width = kwargs.get("width", 2.0)
    ew.companion_berm = kwargs.get("companion_berm", False)
    ew.crest_elevation = kwargs.get("crest_elevation", None)
    ew.gradient_pct = kwargs.get("gradient_pct", 1.0)
    return ew


# ---------------------------------------------------------------------------
# DEMBurner.__init__
# ---------------------------------------------------------------------------

class TestDEMBurnerInit:
    def test_shape_read(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((15, 20), 50.0))
        b = DEMBurner(path)
        assert b.shape == (15, 20)

    def test_cell_size_read(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        assert b.cell_size == pytest.approx(1.0)

    def test_original_is_float32(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        assert b.original.dtype == np.float32


# ---------------------------------------------------------------------------
# burn_earthworks — no earthworks
# ---------------------------------------------------------------------------

class TestBurnEarthworksEmpty:
    def test_empty_list_returns_original(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.burn_earthworks([])
        assert np.allclose(result, 50.0)

    def test_invalid_geometry_skipped(self, tmp_path):
        """Earthwork with bad JSON is silently skipped (line 47 continue)."""
        from unittest.mock import MagicMock
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        ew = MagicMock()
        ew.enabled = True
        ew.type = "swale"
        ew.geometry = MagicMock()
        ew.geometry.asJson.return_value = "not-valid-json"
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)


# ---------------------------------------------------------------------------
# _to_shapely
# ---------------------------------------------------------------------------

class TestToShapely:
    def test_valid_geojson(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        geom = make_mock_line_geom()
        result = b._to_shapely(geom)
        assert result is not None

    def test_invalid_json_returns_none(self, tmp_path):
        from unittest.mock import MagicMock
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        bad = MagicMock()
        bad.asJson.return_value = "not-valid-json"
        assert b._to_shapely(bad) is None


# ---------------------------------------------------------------------------
# _burn_swale
# ---------------------------------------------------------------------------

class TestBurnSwale:
    def test_swale_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_swale_depth_1m(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() == pytest.approx(49.0, abs=0.1)

    def test_swale_with_companion_berm(self, tmp_path):
        # Sloped DEM so downhill side is detectable
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.5, (20, 20)).astype("float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        # Some cells should be raised above their original value (berm built on downhill side)
        assert np.any(result > b.original)


# ---------------------------------------------------------------------------
# _burn_berm
# ---------------------------------------------------------------------------

class TestBurnBerm:
    def test_berm_raises_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=2.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0

    def test_berm_depth_2m(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=2.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.max() == pytest.approx(52.0, abs=0.1)


# ---------------------------------------------------------------------------
# _burn_basin
# ---------------------------------------------------------------------------

class TestBurnBasin:
    def test_basin_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_polygon_geom((3.0, 3.0, 12.0, 12.0))
        ew = _mock_ew("basin", geom, depth=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_basin_depth_2m(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_polygon_geom((3.0, 3.0, 12.0, 12.0))
        ew = _mock_ew("basin", geom, depth=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() == pytest.approx(48.0, abs=0.1)


# ---------------------------------------------------------------------------
# _burn_dam
# ---------------------------------------------------------------------------

class TestBurnDam:
    def test_dam_with_crest_elevation(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        result = b.burn_earthworks([ew])
        assert result.max() >= 60.0

    def test_dam_no_crest_acts_like_berm(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=2.0, width=2.0, crest_elevation=None)
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0

    def test_dam_lower_than_crest_raised(self, tmp_path):
        """DEM cells below crest elevation should be raised to crest."""
        data = np.full((20, 20), 40.0)  # all cells below crest=55
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=55.0)
        result = b.burn_earthworks([ew])
        # Dam footprint cells should be at least 55
        assert result.max() >= 55.0

    def test_dam_above_crest_unchanged(self, tmp_path):
        """DEM cells already above crest should not be changed."""
        data = np.full((20, 20), 70.0)  # all cells above crest=55
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=55.0)
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 70.0)  # unchanged


# ---------------------------------------------------------------------------
# _burn_diversion
# ---------------------------------------------------------------------------

class TestBurnDiversion:
    def test_diversion_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=1.5, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_diversion_single_segment_zero_length(self, tmp_path):
        """Zero-length segment should be skipped (continue branch)."""
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        # Two identical consecutive points create a zero-length segment
        geom = make_mock_line_geom([(10.0, 10.0), (10.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=2.0, gradient_pct=1.0)
        result = b.burn_earthworks([ew])  # should not crash
        assert result.shape == (20, 20)

    def test_diversion_zero_total_length(self, tmp_path):
        """If total_length == 0, returns unchanged DEM."""
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(10.0, 10.0), (10.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=2.0, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        # Should be the same as original (zero-length line)
        assert result.shape == (20, 20)


# ---------------------------------------------------------------------------
# Disabled earthwork
# ---------------------------------------------------------------------------

class TestDisabledEarthwork:
    def test_disabled_skipped(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=5.0)
        ew.enabled = False
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_writes_geotiff(self, tmp_path):
        data = np.full((10, 10), 42.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        out = str(tmp_path / "out.tif")
        b.save(b.original, out)
        with rasterio.open(out) as src:
            assert np.allclose(src.read(1), 42.0)

    def test_save_preserves_crs(self, tmp_path):
        data = np.full((10, 10), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        out = str(tmp_path / "out.tif")
        b.save(b.original, out)
        with rasterio.open(out) as src:
            assert src.crs is not None


# ---------------------------------------------------------------------------
# get_ponding_layer
# ---------------------------------------------------------------------------

class TestGetPondingLayer:
    def test_returns_same_shape(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (20, 20)

    def test_non_negative(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert np.all(result >= 0.0)

    def test_float32_output(self, tmp_path):
        data = np.full((10, 10), 30.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.dtype == np.float32

    def test_bowl_dem_ponding(self, tmp_path):
        """A depression (manually lowered centre) should show ponding."""
        r = np.arange(20)
        c = np.arange(20)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        data = (50.0 + ((rr - 9.5) ** 2 + (cc - 9.5) ** 2) * 0.3).astype("float32")
        data[8:12, 8:12] -= 10.0  # create a depression
        path = _write_dem(str(tmp_path / "bowl.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (20, 20)
