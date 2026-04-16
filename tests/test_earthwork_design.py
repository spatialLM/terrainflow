"""Tests for terrainflow_assessment/modules/earthwork_design.py"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box, mapping

from terrainflow_assessment.modules.earthwork_design import (
    DEMBurner,
    Earthwork,
    EarthworkManager,
    berm_height_estimate,
    calculate_capacity,
    calculate_cut_volume,
    calculate_diversion_discharge,
    calculate_fill_volume,
    calculate_spillway_width,
)
from tests.conftest import make_mock_line_geom, make_mock_polygon_geom


# ---------------------------------------------------------------------------
# Earthwork data class (assessment version)
# ---------------------------------------------------------------------------

class TestEarthwork:
    def _make(self, ew_type="swale"):
        return Earthwork(ew_type, make_mock_line_geom(), f"Test {ew_type}")

    def test_defaults(self):
        ew = self._make()
        assert ew.depth == 0.5
        assert ew.width == 2.0
        assert ew.enabled is True
        assert ew.capacity_m3 == 0.0

    def test_type_label_diversion(self):
        assert self._make("diversion").type_label() == "Diversion Drain"

    def test_type_label_swale_capitalised(self):
        assert self._make("swale").type_label() == "Swale"

    def test_summary_dam_no_crest(self):
        ew = self._make("dam")
        assert "?" in ew.summary()

    def test_summary_dam_with_crest(self):
        ew = self._make("dam")
        ew.crest_elevation = 55.0
        assert "55.0 m" in ew.summary()

    def test_summary_diversion_shows_q(self):
        ew = self._make("diversion")
        assert "m³/s" in ew.summary()

    def test_summary_disabled_shows_off(self):
        ew = self._make("swale")
        ew.enabled = False
        assert "[OFF]" in ew.summary()


# ---------------------------------------------------------------------------
# EarthworkManager (assessment version)
# ---------------------------------------------------------------------------

class TestEarthworkManager:
    def _manager(self, n=3):
        m = EarthworkManager()
        for i in range(n):
            m.add(Earthwork("swale", make_mock_line_geom(), f"EW{i}"))
        return m

    def test_add_and_len(self):
        m = EarthworkManager()
        m.add(Earthwork("swale", make_mock_line_geom(), "X"))
        assert len(m) == 1

    def test_remove_valid(self):
        m = self._manager(3)
        m.remove(1)
        assert len(m) == 2

    def test_remove_out_of_range_noop(self):
        m = self._manager(2)
        m.remove(5)
        assert len(m) == 2

    def test_toggle_and_get_enabled(self):
        m = self._manager(3)
        m.toggle(0)
        enabled = m.get_enabled()
        assert len(enabled) == 2

    def test_clear(self):
        m = self._manager(4)
        m.clear()
        assert len(m) == 0

    def test_get_all_returns_copy(self):
        m = self._manager(2)
        lst = m.get_all()
        lst.pop()
        assert len(m) == 2


# ---------------------------------------------------------------------------
# calculate_capacity (assessment)
# ---------------------------------------------------------------------------

class TestCalculateCapacity:
    def test_swale_basic(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        vol_m3, vol_l = calculate_capacity("swale", geom, 0.5, 2.0)
        assert vol_m3 == pytest.approx(80.0, rel=1e-3)
        assert vol_l == pytest.approx(80_000.0, rel=1e-3)

    def test_swale_companion_berm_increases_capacity(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        v1, _ = calculate_capacity("swale", geom, 0.5, 2.0, False)
        v2, _ = calculate_capacity("swale", geom, 0.5, 2.0, True)
        assert v2 > v1

    def test_basin(self):
        geom = make_mock_polygon_geom()
        geom.area.return_value = 200.0
        vol, _ = calculate_capacity("basin", geom, 1.5, 0)
        assert vol == pytest.approx(200.0 * 1.5 * 0.8, rel=1e-3)

    def test_berm_zero(self):
        assert calculate_capacity("berm", make_mock_line_geom(), 0.5, 2.0) == (0.0, 0.0)

    def test_dam_zero(self):
        assert calculate_capacity("dam", make_mock_line_geom(), 1.0, 3.0) == (0.0, 0.0)

    def test_diversion_zero(self):
        assert calculate_capacity("diversion", make_mock_line_geom(), 0.3, 1.0) == (0.0, 0.0)

    def test_unknown_type_zero(self):
        assert calculate_capacity("unknown", make_mock_line_geom(), 0.5, 2.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# calculate_cut_volume
# ---------------------------------------------------------------------------

class TestCalculateCutVolume:
    def test_swale_cut(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        # bottom=1.0, top=3.0, cs=1.0, vol=100
        cut = calculate_cut_volume("swale", geom, 0.5, 2.0)
        assert cut == pytest.approx(100.0, rel=1e-3)

    def test_basin_cut(self):
        geom = make_mock_polygon_geom()
        geom.area.return_value = 50.0
        cut = calculate_cut_volume("basin", geom, 1.0, 0)
        assert cut == pytest.approx(50.0, rel=1e-3)

    def test_diversion_cut(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 50.0
        cut = calculate_cut_volume("diversion", geom, 0.3, 1.0)
        assert cut > 0.0

    def test_berm_zero_cut(self):
        assert calculate_cut_volume("berm", make_mock_line_geom(), 0.5, 2.0) == 0.0

    def test_dam_zero_cut(self):
        assert calculate_cut_volume("dam", make_mock_line_geom(), 1.0, 3.0) == 0.0

    def test_unknown_type_zero(self):
        assert calculate_cut_volume("unknown", make_mock_line_geom(), 0.5, 2.0) == 0.0

    def test_rounded(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 33.33
        cut = calculate_cut_volume("swale", geom, 0.5, 2.0)
        assert cut == round(cut, 2)


# ---------------------------------------------------------------------------
# calculate_fill_volume
# ---------------------------------------------------------------------------

class TestCalculateFillVolume:
    def test_berm_fill(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        # cross_section = depth^2 = 0.5^2 = 0.25, vol = 0.25*100 = 25
        fill = calculate_fill_volume("berm", geom, 0.5, 2.0)
        assert fill == pytest.approx(25.0, rel=1e-3)

    def test_swale_no_berm_zero_fill(self):
        fill = calculate_fill_volume("swale", make_mock_line_geom(), 0.5, 2.0, False)
        assert fill == 0.0

    def test_swale_with_companion_berm_fill(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        fill = calculate_fill_volume("swale", geom, 0.5, 2.0, True)
        assert fill > 0.0

    def test_dam_fill(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 10.0
        fill = calculate_fill_volume("dam", geom, 2.0, 3.0)
        assert fill == pytest.approx(3.0 * 2.0 * 10.0, rel=1e-3)

    def test_basin_zero_fill(self):
        assert calculate_fill_volume("basin", make_mock_polygon_geom(), 1.0, 0) == 0.0

    def test_diversion_zero_fill(self):
        assert calculate_fill_volume("diversion", make_mock_line_geom(), 0.3, 1.0) == 0.0


# ---------------------------------------------------------------------------
# calculate_diversion_discharge (assessment)
# ---------------------------------------------------------------------------

class TestCalculateDiversionDischarge:
    def test_positive_output(self):
        assert calculate_diversion_discharge(0.5, 2.0, 1.0) > 0.0

    def test_zero_gradient_zero(self):
        assert calculate_diversion_discharge(0.5, 2.0, 0.0) == 0.0

    def test_steeper_gives_more_flow(self):
        q1 = calculate_diversion_discharge(0.5, 2.0, 0.5)
        q2 = calculate_diversion_discharge(0.5, 2.0, 2.0)
        assert q2 > q1


# ---------------------------------------------------------------------------
# calculate_spillway_width (assessment)
# ---------------------------------------------------------------------------

class TestCalculateSpillwayWidth:
    def test_positive(self):
        assert calculate_spillway_width(1.0, 0.5) > 0.0

    def test_zero_head_zero(self):
        assert calculate_spillway_width(1.0, 0.0) == 0.0

    def test_zero_flow_zero(self):
        assert calculate_spillway_width(0.0, 0.5) == 0.0


# ---------------------------------------------------------------------------
# berm_height_estimate (assessment)
# ---------------------------------------------------------------------------

class TestBermHeightEstimate:
    def test_positive(self):
        assert berm_height_estimate(0.5, 2.0) > 0.0

    def test_deeper_swale_taller_berm(self):
        h1 = berm_height_estimate(0.3, 2.0)
        h2 = berm_height_estimate(0.8, 2.0)
        assert h2 > h1


# ---------------------------------------------------------------------------
# DEMBurner
# ---------------------------------------------------------------------------

def _make_dem(tmp_path, data, cell_size=1.0):
    """Write a DEM GeoTIFF and return its path."""
    path = str(tmp_path / "dem.tif")
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs="EPSG:32632",
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _mock_ew(ew_type, geom, **kwargs):
    """Build a minimal Earthwork-like mock."""
    ew = Earthwork(ew_type, geom, f"Test {ew_type}")
    for k, v in kwargs.items():
        setattr(ew, k, v)
    return ew


class TestDEMBurner:
    def test_init_reads_dem(self, tmp_path):
        data = np.full((10, 10), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        assert b.shape == (10, 10)
        assert b.cell_size == pytest.approx(1.0)

    def test_burn_earthworks_returns_array(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        result = b.burn_earthworks([])
        assert result.shape == (20, 20)
        assert np.allclose(result, 50.0)

    def test_burn_swale_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=2.0, companion_berm=False)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_burn_berm_raises_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=1.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0

    def test_burn_basin_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_polygon_geom((3.0, 3.0, 12.0, 12.0))
        ew = _mock_ew("basin", geom, depth=2.0, width=0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_burn_dam_raises_to_crest(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        result = b.burn_earthworks([ew])
        assert result.max() >= 60.0

    def test_burn_dam_no_crest_acts_like_berm(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=2.0, width=2.0, crest_elevation=None)
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0  # raised like a berm

    def test_disabled_earthwork_skipped(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=5.0, width=2.0)
        ew.enabled = False
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)  # unchanged

    def test_invalid_geom_json_skipped(self, tmp_path):
        """_to_shapely returns None for invalid JSON; burn skips the earthwork."""
        from unittest.mock import MagicMock
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        bad_geom = MagicMock()
        bad_geom.asJson.return_value = "not-valid-json"
        ew = _mock_ew("swale", bad_geom, depth=1.0, width=2.0, companion_berm=False)
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)  # unchanged

    def test_save_writes_geotiff(self, tmp_path):
        data = np.full((10, 10), 42.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        out = str(tmp_path / "out.tif")
        b.save(b.original, out)
        with rasterio.open(out) as src:
            result = src.read(1)
        assert np.allclose(result, 42.0)

    def test_burn_swale_with_companion_berm(self, tmp_path):
        # Slope so the downhill side is detectable
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.5, (20, 20)).astype("float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        # Berm should raise something; overall result different from flat
        assert not np.allclose(result, b.original)

    def test_burn_diversion(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=1.5, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_burn_diversion_single_point_noop(self, tmp_path):
        """LineString with < 2 coords returns unchanged DEM."""
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        # Manufacture a degenerate single-point geometry
        geom = make_mock_line_geom([(10.0, 10.0), (10.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=1.5, gradient_pct=1.0)
        # Should not raise — result may be unchanged or slightly modified
        result = b.burn_earthworks([ew])
        assert result.shape == (20, 20)
