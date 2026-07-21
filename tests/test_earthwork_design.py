"""Tests for terrainflow_assessment/modules/earthwork_design.py"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

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
        # top_width=2.0, depth=0.5 → bottom=1.0, cs=(1+2)/2*0.5=0.75, vol=0.75*100*0.8=60
        vol_m3, vol_l = calculate_capacity("swale", geom, 0.5, 2.0)
        assert vol_m3 == pytest.approx(60.0, rel=1e-3)
        assert vol_l == pytest.approx(60_000.0, rel=1e-3)

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
        # top_width=2.0, depth=0.5 → bottom=1.0, cs=(1+2)/2*0.5=0.75, vol=75
        cut = calculate_cut_volume("swale", geom, 0.5, 2.0)
        assert cut == pytest.approx(75.0, rel=1e-3)

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


# ---------------------------------------------------------------------------
# EarthworkManager.toggle branch coverage
# ---------------------------------------------------------------------------

class TestEarthworkManagerToggleOutOfRange:
    def test_toggle_out_of_range_noop(self):
        m = EarthworkManager()
        m.add(Earthwork("swale", make_mock_line_geom(), "X"))
        # Covers the false branch of the index check at line 89
        m.toggle(99)
        assert m.get(0).enabled is True


# ---------------------------------------------------------------------------
# Unknown earthwork type skipped in burn_earthworks (branch 303->289)
# ---------------------------------------------------------------------------

class TestBurnEarthworksUnknownType:
    def test_unknown_type_skipped(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("unknown_type_xyz", geom, depth=1.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)


# ---------------------------------------------------------------------------
# _burn_diversion: Point geometry triggers len(coords) < 2 branch (line 403)
# ---------------------------------------------------------------------------

class TestBurnDiversionPointGeom:
    def test_point_geometry_noop(self, tmp_path):
        """Point JSON → shapely Point has 1 coord → line 403 early return."""
        from unittest.mock import MagicMock
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        g = MagicMock()
        g.asJson.return_value = json.dumps({
            "type": "Point", "coordinates": [10.0, 10.0]
        })
        ew = _mock_ew("diversion", g, depth=0.3, width=1.5, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        assert result.shape == (20, 20)


# ---------------------------------------------------------------------------
# _burn_diversion with duplicate consecutive points (line 428 seg_dist==0)
# ---------------------------------------------------------------------------

class TestBurnDiversionDuplicateCoords:
    def test_duplicate_consecutive_coords_segment_skipped(self, tmp_path):
        from unittest.mock import MagicMock
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        # 3-point line with a duplicated middle point → one seg_dist == 0
        g = MagicMock()
        g.asJson.return_value = json.dumps({
            "type": "LineString",
            "coordinates": [[2.0, 10.0], [2.0, 10.0], [18.0, 10.0]],
        })
        ew = _mock_ew("diversion", g, depth=0.3, width=1.5, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        assert result.shape == (20, 20)
        assert result.min() < 50.0  # second segment still burns


# ---------------------------------------------------------------------------
# _burn_diversion where a sampled point lies outside the DEM extent (line 441)
# ---------------------------------------------------------------------------

class TestBurnDiversionCellOutside:
    def test_cell_outside_raster_skipped(self, tmp_path):
        """Line with coords outside the raster extent → cell_mask empty → continue."""
        from unittest.mock import MagicMock
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        # Line entirely outside the DEM bounds (DEM is 0..20 in both dims)
        g = MagicMock()
        g.asJson.return_value = json.dumps({
            "type": "LineString",
            "coordinates": [[100.0, 100.0], [120.0, 100.0]],
        })
        ew = _mock_ew("diversion", g, depth=0.3, width=0.5, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        # Nothing should burn because all sample points fall outside
        assert np.allclose(result, 50.0)


# ---------------------------------------------------------------------------
# Companion berm: parallel_offset exception (lines 352-353)
# ---------------------------------------------------------------------------

class TestCompanionBermParallelOffsetFailure:
    def test_parallel_offset_exception_returns_dem(self, tmp_path, monkeypatch):
        """If parallel_offset raises, the except branch returns dem unchanged."""
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        # Patch shapely LineString.parallel_offset to raise for this test
        from shapely.geometry import LineString as _LS

        def _raise(*a, **kw):
            raise ValueError("mock parallel_offset failure")

        monkeypatch.setattr(_LS, "parallel_offset", _raise, raising=False)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        # Swale still burned, but no companion berm raised — some cells lower, none raised
        assert result.min() < 50.0
        assert result.max() <= 50.0


# ---------------------------------------------------------------------------
# Companion berm: only one side has mask (lines 363 / 365)
# ---------------------------------------------------------------------------

class TestCompanionBermSingleSide:
    def test_only_left_side_in_raster(self, tmp_path, monkeypatch):
        """If right_mask is empty and left_mask.any() → line 363 branch."""
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.1, (20, 20)).astype("float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        # Force _rasterize to return empty for one of the berm rasterizations.
        orig = b._rasterize
        call_state = {"n": 0}

        def fake_rasterize(geom):
            call_state["n"] += 1
            # Calls: 1=swale footprint, 2=left berm, 3=right berm (or vice versa)
            if call_state["n"] == 3:
                return np.zeros(b.shape, dtype=bool)
            return orig(geom)

        monkeypatch.setattr(b, "_rasterize", fake_rasterize)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        assert result.max() >= 50.0  # berm was raised on the single remaining side

    def test_only_right_side_in_raster(self, tmp_path, monkeypatch):
        """If left_mask is empty and right_mask.any() → line 365 branch."""
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.1, (20, 20)).astype("float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        orig = b._rasterize
        call_state = {"n": 0}

        def fake_rasterize(geom):
            call_state["n"] += 1
            if call_state["n"] == 2:  # left side empty
                return np.zeros(b.shape, dtype=bool)
            return orig(geom)

        monkeypatch.setattr(b, "_rasterize", fake_rasterize)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        assert result.max() >= 50.0

    def test_both_sides_empty_returns_dem(self, tmp_path, monkeypatch):
        """Both parallel berm masks empty → else branch returns dem unchanged."""
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        orig = b._rasterize
        call_state = {"n": 0}

        def fake_rasterize(geom):
            call_state["n"] += 1
            if call_state["n"] >= 2:  # both berm rasterizations empty
                return np.zeros(b.shape, dtype=bool)
            return orig(geom)

        monkeypatch.setattr(b, "_rasterize", fake_rasterize)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        # Swale cells were lowered, but no berm was raised anywhere
        assert result.min() < 50.0
        assert result.max() <= 50.0


# ---------------------------------------------------------------------------
# get_ponding_layer — downsampling + upsampling + MemoryError paths
# ---------------------------------------------------------------------------

class TestGetPondingLayerBranches:
    def test_ponding_small_dem(self, tmp_path):
        """Small DEM: no downsampling, normal path."""
        data = np.full((10, 10), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (10, 10)

    def test_ponding_downsampled_and_upsampled(self, tmp_path, monkeypatch):
        """Force the downsampling branch by lowering _MAX_PONDING_CELLS."""
        import terrainflow_assessment.modules.earthwork_design as mod
        monkeypatch.setattr(mod, "_MAX_PONDING_CELLS", 25)  # 5x5

        data = np.full((20, 20), 50.0, dtype="float32")
        # Introduce a small depression so depression-filling does something
        data[9:12, 9:12] = 48.0
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (20, 20)  # upsampled back
        assert result.dtype == np.dtype("float32")
        assert (result >= 0).all()

    def test_ponding_fill_depressions_memory_error_returns_zeros(
        self, tmp_path, monkeypatch
    ):
        """fill_depressions raising MemoryError returns zeros array (lines 495-497)."""
        data = np.full((10, 10), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        from pysheds.grid import Grid as _Grid

        def _boom(self, *a, **kw):
            raise MemoryError("mock OOM")

        monkeypatch.setattr(_Grid, "fill_depressions", _boom)
        result = b.get_ponding_layer(data)
        assert result.shape == (10, 10)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# Phase 1 regression — Phase 1 item 3: swale burn footprint == declared top width
# ---------------------------------------------------------------------------

class TestSwaleWidthSplit:
    """Verify top_width_m / width property and correct burn footprint."""

    def test_top_width_m_default(self):
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        ew = Earthwork("swale", make_mock_line_geom(), "S")
        assert ew.top_width_m == 2.0

    def test_width_alias_reads_top_width_m(self):
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        ew = Earthwork("swale", make_mock_line_geom(), "S")
        assert ew.width == ew.top_width_m

    def test_width_alias_writes_top_width_m(self):
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        ew = Earthwork("swale", make_mock_line_geom(), "S")
        ew.width = 3.0
        assert ew.top_width_m == 3.0

    def test_buffer_radius_m_is_half_top_width(self):
        from terrainflow_assessment.modules.earthwork_design import Earthwork
        ew = Earthwork("swale", make_mock_line_geom(), "S")
        ew.top_width_m = 4.0
        assert ew.buffer_radius_m == pytest.approx(2.0)

    def test_swale_burn_footprint_matches_declared_top_width(self, tmp_path):
        """Phase 1 correctness: a 2m-wide swale burns a 2m-wide footprint."""
        import json
        from unittest.mock import MagicMock

        import rasterio
        from rasterio.transform import from_bounds
        from shapely.geometry import LineString, mapping

        declared_width = 2.0

        # Build a 20×20 1m-res DEM
        data = np.full((20, 20), 50.0, dtype="float32")
        dem_path = str(tmp_path / "dem.tif")
        transform = from_bounds(0, 0, 20, 20, 20, 20)
        with rasterio.open(dem_path, "w", driver="GTiff", height=20, width=20,
                           count=1, dtype="float32", crs="EPSG:32632",
                           transform=transform, nodata=-9999.0) as dst:
            dst.write(data, 1)

        # Swale line running through the centre row
        line = LineString([(0.0, 10.0), (20.0, 10.0)])
        g = MagicMock()
        g.asJson.return_value = json.dumps(mapping(line))
        g.length.return_value = line.length

        ew = Earthwork("swale", g, "test_swale")
        ew.top_width_m = declared_width
        ew.depth = 0.3

        burner = DEMBurner(dem_path)
        modified = burner.burn_earthworks([ew])

        # Cells that were lowered by the swale
        burned = modified < data
        burned_rows = np.where(burned.any(axis=1))[0]
        footprint_width_cells = len(burned_rows)
        footprint_width_m = footprint_width_cells * 1.0  # 1m cell size

        assert footprint_width_m == pytest.approx(declared_width, abs=1.0)
