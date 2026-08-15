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

    def test_registry_seeds_basin_defaults(self):
        # Sizing defaults come from the registry (basin: depth 1.5 m).
        ew = self._make("basin")
        assert ew.depth == pytest.approx(1.5)

    def test_registry_seeds_dam_defaults(self):
        ew = self._make("dam")
        assert ew.depth == pytest.approx(2.0)

    def test_unknown_type_seeds_historical_defaults(self):
        ew = self._make("moat")
        assert ew.depth == 0.5
        assert ew.top_width_m == 2.0

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

    def test_summary_dam_capacity_labels_as_drawn(self):
        ew = self._make("dam")
        ew.crest_elevation = 55.0
        ew.capacity_m3 = 1240.0
        assert "as-drawn" in ew.summary()

    def test_summary_dam_capacity_labels_keyed(self):
        ew = self._make("dam")
        ew.crest_elevation = 55.0
        ew.capacity_m3 = 1240.0
        ew.key_into_banks = True
        assert "keyed" in ew.summary()

    def test_summary_diversion_shows_q(self):
        ew = self._make("diversion")
        assert "m³/s" in ew.summary()

    def test_summary_disabled_shows_off(self):
        ew = self._make("swale")
        ew.enabled = False
        assert "[OFF]" in ew.summary()


# ---------------------------------------------------------------------------
# Earthwork data-model SHAPE (side slope, wall batter, length, id, overflow link)
# ---------------------------------------------------------------------------

class TestEarthworkShape:
    def _make(self, ew_type="swale"):
        return Earthwork(ew_type, make_mock_line_geom(), f"Test {ew_type}")

    # -- id --------------------------------------------------------------
    def test_id_auto_assigned_and_unique(self):
        a, b = self._make(), self._make()
        assert isinstance(a.id, str) and len(a.id) == 32
        assert a.id != b.id

    # -- bottom_width_m is now a stored field ---------------------------
    def test_bottom_width_stored_default_channel(self):
        # swale: top=2.0, depth=0.5, default_side_slope=1.0 → bottom = 2 − 2·1·0.5 = 1.0
        assert self._make("swale").bottom_width_m == pytest.approx(1.0)

    def test_bottom_width_is_writable_field(self):
        ew = self._make("swale")
        ew.bottom_width_m = 0.4
        assert ew.bottom_width_m == 0.4

    def test_unknown_type_falls_back_to_1to1_default(self):
        # KeyError branch in __init__ → default_side_slope 1.0
        ew = Earthwork("not_a_real_type", make_mock_line_geom(), "x")
        assert ew.bottom_width_m == pytest.approx(1.0)

    # -- side_slope derived property + setter ---------------------------
    def test_side_slope_derived_from_widths(self):
        ew = self._make("swale")  # top 2.0, bottom 1.0, depth 0.5
        assert ew.side_slope == pytest.approx(1.0)

    def test_side_slope_setter_back_solves_bottom_width(self):
        ew = self._make("swale")
        ew.side_slope = 0.5  # bottom = 2.0 − 2·0.5·0.5 = 1.5
        assert ew.bottom_width_m == pytest.approx(1.5)
        assert ew.side_slope == pytest.approx(0.5)  # round-trips

    def test_side_slope_zero_depth_is_safe(self):
        ew = self._make("swale")
        ew.depth = 0.0
        assert ew.side_slope == 0.0

    # -- wall_slope (basin) derived property + setter -------------------
    def test_wall_slope_default_vertical(self):
        ew = self._make("basin")
        assert ew.batter_run_m == 0.0
        assert ew.wall_slope == 0.0

    def test_wall_slope_setter_back_solves_batter_run(self):
        ew = self._make("basin")
        ew.depth = 0.5
        ew.wall_slope = 1.5  # run = 1.5 · 0.5 = 0.75
        assert ew.batter_run_m == pytest.approx(0.75)
        assert ew.wall_slope == pytest.approx(1.5)

    def test_wall_slope_zero_depth_is_safe(self):
        ew = self._make("basin")
        ew.depth = 0.0
        assert ew.wall_slope == 0.0

    # -- length_m derived from geometry ---------------------------------
    def test_length_m_reads_from_geometry(self):
        ew = self._make("swale")
        ew.geometry.length.return_value = 137.5
        assert ew.length_m == pytest.approx(137.5)

    # -- overflow linkage ----------------------------------------------
    def test_overflow_target_defaults_none(self):
        assert self._make("swale").overflow_target_id is None

    def test_overflow_target_settable(self):
        a, b = self._make(), self._make()
        a.overflow_target_id = b.id
        assert a.overflow_target_id == b.id


# ---------------------------------------------------------------------------
# Calc functions honour an explicit bottom_width (stored side slope)
# ---------------------------------------------------------------------------

class TestExplicitBottomWidth:
    def _line(self, length=100.0):
        geom = make_mock_line_geom()
        geom.length.return_value = length
        return geom

    def test_capacity_explicit_bottom_differs_from_default(self):
        geom = self._line()
        default_v, _ = calculate_capacity("swale", geom, 0.5, 2.0)
        narrow_v, _ = calculate_capacity("swale", geom, 0.5, 2.0, bottom_width=0.5)
        # bottom 0.5 < derived 1.0 → smaller trapezoid
        assert narrow_v < default_v
        assert narrow_v == pytest.approx(((0.5 + 2.0) / 2) * 0.5 * 100.0 * 0.8, rel=1e-3)

    def test_cut_volume_explicit_bottom(self):
        geom = self._line()
        cut = calculate_cut_volume("swale", geom, 0.5, 2.0, bottom_width=0.5)
        assert cut == pytest.approx(((0.5 + 2.0) / 2) * 0.5 * 100.0, rel=1e-3)

    def test_fill_volume_companion_explicit_bottom(self):
        geom = self._line()
        v = calculate_fill_volume("swale", geom, 0.5, 2.0, companion_berm=True,
                                  bottom_width=0.5)
        assert v > 0.0

    def test_berm_height_explicit_bottom(self):
        h = berm_height_estimate(0.5, 2.0, bottom_width=0.5)
        assert h > 0.0

    def test_diversion_discharge_stored_geometry_path(self):
        # explicit bottom_width switches to the top-width convention; still positive
        q = calculate_diversion_discharge(0.3, 1.0, 1.0, bottom_width=0.4)
        assert q > 0.0

    def test_diversion_discharge_default_unchanged(self):
        # None path must stay byte-identical to legacy behaviour
        q = calculate_diversion_discharge(0.3, 1.0, 1.0)
        assert q > 0.0


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

    def test_basin_batter_none_matches_zero(self):
        # batter_run=None and 0 both mean vertical walls — historical numbers.
        geom = make_mock_polygon_geom()  # 10×10 → A=100, P=40
        v_none, _ = calculate_capacity("basin", geom, 1.5, 0, batter_run=None)
        v_zero, _ = calculate_capacity("basin", geom, 1.5, 0, batter_run=0.0)
        assert v_none == v_zero == pytest.approx(100.0 * 1.5 * 0.8, rel=1e-3)

    def test_basin_batter_reduces_capacity(self):
        geom = make_mock_polygon_geom()  # A=100, P=40
        vertical, _ = calculate_capacity("basin", geom, 1.0, 0)
        battered, _ = calculate_capacity("basin", geom, 1.0, 0, batter_run=1.0)
        # z=1: V = (100·1 − 40·1·1²/2) × 0.8 = 64.0
        assert battered == pytest.approx(64.0, rel=1e-3)
        assert battered < vertical

    def test_basin_converging_batter_clamps(self):
        # Tiny basin, huge batter: walls meet before design depth — volume clamps,
        # never negative.
        geom = make_mock_polygon_geom((0.0, 0.0, 1.0, 1.0))  # A=1, P=4
        vol, _ = calculate_capacity("basin", geom, 2.0, 0, batter_run=4.0)
        # z=2: t*=1/8, V = 1²/(2·8) × 0.8 = 0.05
        assert vol == pytest.approx(0.05, rel=1e-3)
        assert vol > 0.0

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

    def test_burn_dam_wall_is_the_drawn_width(self, tmp_path):
        """A 2 m wall burns 2 m thick, not the 4 m ``all_touched`` used to claim."""
        data = np.full((20, 20), 50.0)
        path = _make_dem(tmp_path, data)          # 1 m cells
        b = DEMBurner(path)

        # On a cell boundary, so the 2 m band's own edges do not land on cell centres.
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        burned = b.burn_earthworks([ew])

        # Cross-section through the middle of the wall: 2 m of wall on 1 m cells is two
        # cells. Rasterising by brush rather than by cell centre gave four.
        assert int((burned[:, 10] > 50.0).sum()) == 2

    def test_burn_dam_diagonal_wall_is_not_widened(self, tmp_path):
        """The diagonal case, where brushed rasterising over-claimed most."""
        data = np.full((30, 30), 50.0)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 5.0), (20.0, 20.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        burned = b.burn_earthworks([ew])

        length = (15.0 ** 2 + 15.0 ** 2) ** 0.5
        effective_width = int((burned > 50.0).sum()) / length   # cells are 1 m²
        assert effective_width < 2.6      # drawn 2.0 m, plus the buffer's end caps

    def test_burn_dam_keys_into_the_banks_when_asked(self, tmp_path):
        """``key_into_banks`` is ground, not just a capacity estimate.

        The wall used to be keyed only inside ``_keyed_dam_dem``, so the capacity was
        measured against a wall reaching its abutments while the burn every raster is
        built from kept the short one as drawn — and the pond ran round its ends.
        """
        # Flat channel between cols 7 and 13, banks rising 1 m per cell outside it.
        cols = np.arange(20) + 0.5
        profile = 50.0 + np.maximum(0.0, np.abs(cols - 10.0) - 3.0)
        data = np.tile(profile, (20, 1)).astype("float32")
        path = _make_dem(tmp_path, data)

        geom = make_mock_line_geom([(7.5, 10.5), (12.5, 10.5)])
        # Ground at col 14 is 51.5 — below the 52.0 crest, so a keyed wall has to run
        # out through it, and an as-drawn wall must leave it alone.
        plain = DEMBurner(path).burn_earthworks(
            [_mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=52.0,
                      key_into_banks=False)]
        )
        assert plain[9, 14] == pytest.approx(51.5)

        keyed_burner = DEMBurner(path)
        keyed = keyed_burner.burn_earthworks(
            [_mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=52.0,
                      key_into_banks=True)]
        )
        assert keyed[9, 14] == pytest.approx(52.0)
        assert keyed[9, 5] == pytest.approx(52.0)      # and the other abutment
        assert any("keyed" in w for w in keyed_burner.warnings)

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

        def fake_rasterize(geom, **kwargs):
            call_state["n"] += 1
            # Calls: 1=swale footprint, 2=left berm, 3=right berm (or vice versa)
            if call_state["n"] == 3:
                return np.zeros(b.shape, dtype=bool)
            return orig(geom, **kwargs)

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

        def fake_rasterize(geom, **kwargs):
            call_state["n"] += 1
            if call_state["n"] == 2:  # left side empty
                return np.zeros(b.shape, dtype=bool)
            return orig(geom, **kwargs)

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

        def fake_rasterize(geom, **kwargs):
            call_state["n"] += 1
            if call_state["n"] >= 2:  # both berm rasterizations empty
                return np.zeros(b.shape, dtype=bool)
            return orig(geom, **kwargs)

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


# ---------------------------------------------------------------------------
# Step 5 — Strategy-C burn stage
# ---------------------------------------------------------------------------

class TestStrategyCConveyances:
    """Sub-cell snap (empty-mask no-op fix) + monotonic breach + warnings."""

    def test_sub_cell_swale_modifies_at_least_one_cell(self, tmp_path):
        # Regression: a sub-metre swale on a 1 m DEM used to burn nothing (empty mask).
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, companion_berm=False)
        ew.width = 0.4          # top width < cell → buffer rasterises empty
        ew.bottom_width_m = 0.3  # narrowest dim < cell → sub-cell warning
        result = b.burn_earthworks([ew])

        assert (result < 50.0).sum() >= 1
        assert any("1-cell width" in w for w in b.warnings)

    def test_resolvable_swale_keeps_true_footprint(self, tmp_path):
        # A 2 m swale on a 1 m DEM is resolvable → no sub-cell warning.
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=False)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0
        assert b.warnings == []  # bottom width 1.0 == cell size, not sub-cell

    def test_before_after_integrity_changes_localized(self, tmp_path):
        """Baseline vs with-earthwork must be identical away from the feature.

        The corridor is the drawn 2 m, not a cell wider. The volumetric burns rasterise
        on cell **centres**: ``all_touched`` claims every cell the geometry brushes,
        which is the right answer to "did we lose the feature?" and the wrong one to
        "how much earth came out?". Left on, a drawn 3.0 m swale was cut 4.19 m wide on
        the Quail Island design — 628 cells over 149.8 m — so both the excavation and the
        storage credited to the terrain model described a trench half again as wide as
        the one specified. The sub-cell fallbacks (nearest-cell path, centroid cell)
        still guarantee a feature too narrow to claim a centre is not lost.
        """
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=2.0, companion_berm=False)
        result = b.burn_earthworks([ew])

        cut_rows = np.where((result < data).any(axis=1))[0]
        assert cut_rows.min() >= 9 and cut_rows.max() <= 10, (
            f"a 2 m swale cut rows {cut_rows} — wider than it was drawn"
        )
        assert np.array_equal(result[:9, :], data[:9, :])
        assert np.array_equal(result[11:, :], data[11:, :])

    def test_a_drawn_swale_is_cut_to_the_width_it_was_drawn(self, tmp_path):
        """Cell count over length tracks the drawn top width, not a cell more.

        This is the whole of the At-grid over-count. ``n_cells / length`` on Swale 22
        read 4.19 m against a drawn 3.00 m, and ``rasterisable_capacity`` — correctly —
        reported the trench that produced it, so At grid sat 2.10x the drawn section.

        Drawn on a **diagonal**, because that is where the two rasterisations differ and
        because a contour swale is never axis-aligned. Along a grid axis ``all_touched``
        costs almost nothing (4.07 m against 4.04 m here); off-axis it claims a whole
        extra cell of width, and the Quail Island swales are all off-axis.
        """
        import math

        data = np.full((220, 220), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)

        angle = math.radians(17.0)
        length = 150.0
        line = [(25.0, 25.0),
                (25.0 + length * math.cos(angle), 25.0 + length * math.sin(angle))]
        ew = _mock_ew("swale", make_mock_line_geom(line), depth=1.0, width=3.0,
                      bottom_width_m=1.0, companion_berm=False)
        b = DEMBurner(path)
        b.burn_earthworks([ew])
        drawn_width = next(iter(b.burned_masks.values())).sum() / length
        assert 2.8 <= drawn_width <= 3.4, (
            f"drawn 3.0 m, burned {drawn_width:.2f} m wide"
        )

        # The comparison that names what changed.
        touched = b._rasterize(
            b._to_shapely(make_mock_line_geom(line)).buffer(1.5),
            all_touched=True).sum() / length
        assert touched > drawn_width + 0.8, (
            f"all_touched {touched:.2f} m vs centres {drawn_width:.2f} m — expected the "
            f"all_touched corridor to be visibly wider, so this test is not proving "
            f"anything about which one the burn uses"
        )

    def test_a_battered_swale_holds_its_drawn_section(self, tmp_path):
        """The tapered cut conserves the trapezoid, which is the point of cutting it.

        Burned as a rectangle at full depth, a 10 / 6 / 1 m section held 10 m²/m for an
        8 m²/m design — and that is before the footprint's own over-claim. The
        distance-transform taper puts it back: what the burner reports as storage is what
        the drawn section holds, which is what lets At grid be read against Geometric.

        Deliberately a wide feature. A 3 m swale on a 1 m grid is 3 or 4 cells depending
        on where its edges fall between cell centres, so its section is 2.0–3.0 m²/m
        whatever the burn does — a real limit of the grid, and not what this is testing.
        """
        data = np.full((60, 300), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(10.0, 30.0), (290.0, 30.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=10.0, bottom_width_m=6.0,
                      companion_berm=False)
        b.burn_earthworks([ew])
        held = next(iter(b.burned_cut.values()))
        drawn = ((10.0 + 6.0) / 2.0) * 1.0 * 280.0    # trapezoid section x length
        assert held == pytest.approx(drawn, rel=0.10), (
            f"burned trench holds {held:,.0f} m³ against a drawn {drawn:,.0f} m³"
        )

    def test_a_vertical_walled_channel_is_still_cut_square(self, tmp_path):
        """No batter drawn, no batter cut — the taper must not invent one.

        The converse of the trapezoid case, and the one that keeps
        ``rasterisable_capacity``'s rectangular branch honest: a channel specified with
        equal top and bottom widths is a rectangle, and the burn owes it full depth
        across its whole floor.
        """
        data = np.full((60, 300), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(10.0, 30.0), (290.0, 30.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=10.0, bottom_width_m=10.0,
                      companion_berm=False)
        b.burn_earthworks([ew])
        result = b.burn_earthworks([ew])
        mask = next(iter(b.burned_masks.values()))
        cut = b.original[mask] - result[mask]
        assert cut.min() == pytest.approx(1.0, abs=1e-3)
        assert cut.max() == pytest.approx(1.0, abs=1e-3)

    def test_burn_order_does_not_change_the_result(self, tmp_path):
        """Two features whose rims touch must burn the same either way round.

        The invert datum is a single minimum over a one-cell-wide rim. Read off the
        accumulating DEM, one neighbouring cell that an earlier feature has already cut
        drags the whole floor down with it — so the answer depended on the order of the
        feature list. On the Quail Island design, Swale 27 shares **exactly one** rim
        cell with Swale 25; that cell had been cut to 71.78 m against a natural 72.44 m,
        and Swale 27 was floored 0.66 m too deep, ponding 1.94 m for a 1.00 m design and
        reporting Δ +94%.
        """
        data = np.full((20, 30), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)

        # Adjacent corridors, close enough that each lies in the other's rim.
        a = _mock_ew("swale", make_mock_line_geom([(4.0, 9.0), (26.0, 9.0)]),
                     depth=1.0, width=2.0, companion_berm=False)
        b_ew = _mock_ew("swale", make_mock_line_geom([(4.0, 12.0), (26.0, 12.0)]),
                        depth=1.0, width=2.0, companion_berm=False)
        a.name, b_ew.name = "A", "B"

        forward = DEMBurner(path).burn_earthworks([a, b_ew])
        reverse = DEMBurner(path).burn_earthworks([b_ew, a])
        assert np.allclose(forward, reverse), (
            "burning the same design in a different order produced a different DEM"
        )
        # And each is cut to its own design depth, not to its neighbour's floor.
        assert forward.min() == pytest.approx(49.0)

    def test_over_excavation_is_warned_even_on_flat_ground(self, tmp_path):
        """Relief across the footprint is the wrong question; the cut is the right one.

        A footprint can be internally flat and still be cut far below its design depth,
        because the datum is the lowest cell of the *rim*: a dip just outside the
        footprint takes the whole floor down with it. Swale 27 was cut 2.23 m mean for a
        1.00 m design — 517 m³ moved for 232 m³ of storage — with 0.67 m of relief, and
        the old ``relief > depth`` gate said nothing.
        """
        data = np.full((20, 30), 50.0, dtype="float32")
        # Row 8 is the rim above the trench: put one cell of it 2.5 m down, and the
        # whole otherwise-flat strip is floored to that rather than to its own ground.
        data[8, 14] = 47.5
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        ew = _mock_ew("swale", make_mock_line_geom([(4.0, 10.0), (26.0, 10.0)]),
                      depth=1.0, width=2.0, companion_berm=False)
        ew.name = "Swale 27"
        b.burn_earthworks([ew])
        assert any("excavation" in w for w in b.warnings), (
            f"over-excavation went unreported: {b.warnings}"
        )

    def _bermed_swale(self, tmp_path, keyed, slope_along=0.0):
        """A contour swale with a companion berm on 5% ground. Returns (burner, dem, ew)."""
        rows, cols = 120, 300
        data = np.fromfunction(
            lambda r, c: 100.0 - r * 0.05 - c * slope_along, (rows, cols)
        ).astype("float32")
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        ew = _mock_ew("swale", make_mock_line_geom([(20.0, 60.0), (280.0, 60.0)]),
                      depth=1.0, width=3.0, bottom_width_m=1.0, companion_berm=True)
        ew.key_into_banks = keyed
        return b, b.burn_earthworks([ew]), ew

    def test_the_companion_berm_is_built_to_a_level_crest(self, tmp_path):
        """Level, not a constant raise — otherwise the crest follows the slope.

        ``dem[mask] += height`` made the bank highest where the water was shallowest,
        so the pool ran out of its low end and the berm impounded nothing it was
        credited with. Measured on the Quail Island design the raised bank ran
        1.44–3.53 m tall against a declared 1.22 m and still held almost nothing.
        """
        b, out, ew = self._bermed_swale(tmp_path, keyed=False)
        raised = out > b.original
        assert raised.any(), "no berm was built"
        crest = out[raised]
        assert crest.max() - crest.min() < 1e-3, (
            f"berm crest spans {crest.max() - crest.min():.3f} m — it is following the "
            f"ground rather than being levelled"
        )
        assert ew.berm_crest_elevation == pytest.approx(float(crest.max()), abs=1e-3)

    def test_the_dialog_height_is_the_bank_the_burn_builds(self, tmp_path):
        """What the user is told while drawing must be what the burner lays down.

        The dialog used to quote ``√(0.75 × section)`` — a 1:1 triangular ridge — while
        the burn spread the same spoil across a band as wide as the swale to a level
        crest. Same earth, different shape: 1.22 m against 0.50 m for a 3 / 1 / 1 m
        swale, so the readout was 2.4× the bank that actually appeared.

        The centreline sits at a half-cell offset deliberately. Placed so its buffer
        edges fall exactly on cell centres, a 3 m swale claims four cells rather than
        three, cuts half again the spoil, and builds a 0.75 m bank — a real ±half-cell
        grid effect that no drawn-dimension estimate can predict, and one the At-grid
        column exists to expose.
        """
        rows, cols = 140, 320
        data = np.full((rows, cols), 100.0, dtype="float32")
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        ew = _mock_ew("swale", make_mock_line_geom([(20.0, 70.5), (300.0, 70.5)]),
                      depth=1.0, width=3.0, bottom_width_m=1.0, companion_berm=True)
        ew.key_into_banks = False
        b.burn_earthworks([ew])

        predicted = berm_height_estimate(1.0, 3.0, 1.0)
        low, mean, high = ew.berm_height_m
        assert mean == pytest.approx(predicted, abs=0.02), (
            f"dialog predicts {predicted:.2f} m, burn builds {mean:.2f} m"
        )
        # Flat ground, so there is nothing for the crest to vary against.
        assert high - low < 0.02

    def test_a_level_crest_over_uneven_ground_is_reported_as_a_range(self, tmp_path):
        """One elevation over ground that is not one elevation — so height varies.

        A single "0.98 m" hides it: Swale 29 on the Quail Island design stands 0.60 m at
        one end and 1.73 m at the other. The bank is reported as (min, mean, max) so the
        panel can say so, and past a fall greater than the bank's own mean height the
        burner warns that the run wants segmenting.
        """
        rows, cols = 140, 320
        # Falls along the swale as well as across it, so the ground under the bank runs
        # downhill while its crest does not.
        data = np.fromfunction(
            lambda r, c: 100.0 - r * 0.02 - c * 0.03, (rows, cols)).astype("float32")
        tmp_path.mkdir(parents=True, exist_ok=True)
        b = DEMBurner(_make_dem(tmp_path, data))

        ew = _mock_ew("swale", make_mock_line_geom([(20.0, 70.5), (300.0, 70.5)]),
                      depth=1.0, width=3.0, bottom_width_m=1.0, companion_berm=True)
        ew.name = "Swale 29"
        ew.key_into_banks = False
        b.burn_earthworks([ew])

        low, mean, high = ew.berm_height_m
        assert low < mean < high, (low, mean, high)
        assert high - low > mean, (
            f"fixture is not uneven enough to exercise the warning: "
            f"{low:.2f}–{high:.2f} m against a {mean:.2f} m mean")
        assert any("companion berm falls" in w for w in b.warnings), b.warnings

    def test_an_even_berm_is_not_warned_about(self, tmp_path):
        """Level crests are always somewhat uneven; only a fall past the mean matters."""
        rows, cols = 140, 320
        data = np.full((rows, cols), 100.0, dtype="float32")
        tmp_path.mkdir(parents=True, exist_ok=True)
        b = DEMBurner(_make_dem(tmp_path, data))
        ew = _mock_ew("swale", make_mock_line_geom([(20.0, 70.5), (300.0, 70.5)]),
                      depth=1.0, width=3.0, bottom_width_m=1.0, companion_berm=True)
        ew.key_into_banks = False
        b.burn_earthworks([ew])
        assert not any("companion berm falls" in w for w in b.warnings), b.warnings

    def test_the_berm_is_built_from_the_spoil_the_trench_produced(self, tmp_path):
        b, out, _ = self._bermed_swale(tmp_path, keyed=False)
        cut = float(np.clip(b.original - out, 0.0, None).sum())
        fill = float(np.clip(out - b.original, 0.0, None).sum())
        assert fill == pytest.approx(cut * 0.75, rel=0.02), (
            f"{fill:,.0f} m³ of bank from {cut:,.0f} m³ of cut — the 0.75 compaction "
            f"allowance is not being conserved"
        )

    def test_keying_the_berm_into_the_banks_makes_it_hold_water(self, tmp_path):
        """A bank open at its ends impounds nothing; closing them is what makes it real.

        Note this is *not* ``extend_to_abutments``, the dam version — walking outward
        along the alignment's own bearing. A swale is laid along a contour, so the
        ground off each end is at the same elevation as the ground under it and that
        walk keys into nothing: measured on this fixture it changed the ponded volume
        by 0 m³. A return at each end does the job.
        """
        open_b, open_dem, _ = self._bermed_swale(tmp_path / "open", keyed=False)
        keyed_b, keyed_dem, _ = self._bermed_swale(tmp_path / "keyed", keyed=True)

        open_pond = open_b.get_ponding_layer(open_dem).sum()
        keyed_pond = keyed_b.get_ponding_layer(keyed_dem).sum()
        assert keyed_pond > open_pond * 1.15, (
            f"keyed {keyed_pond:,.0f} m³ vs open {open_pond:,.0f} m³ — closing the ends "
            f"made no material difference, so the berm is still leaking round them"
        )

    def test_a_keyed_berm_uses_no_more_earth_than_an_open_one(self, tmp_path):
        """The spoil is fixed by the cut, so keying in spreads it, never invents it."""
        open_b, open_dem, _ = self._bermed_swale(tmp_path / "open2", keyed=False)
        keyed_b, keyed_dem, _ = self._bermed_swale(tmp_path / "keyed2", keyed=True)
        open_fill = float(np.clip(open_dem - open_b.original, 0.0, None).sum())
        keyed_fill = float(np.clip(keyed_dem - keyed_b.original, 0.0, None).sum())
        assert keyed_fill == pytest.approx(open_fill, rel=0.02)

    def test_sub_cell_diversion_snaps_and_warns(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=0.4, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0
        assert any("1-cell width" in w for w in b.warnings)


class TestStrategyCBarriers:
    """Barriers raise a ridge (never a cut); dam uses the downstream inner-wall offset."""

    def test_sub_cell_berm_raises_via_path(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=1.0, width=0.4)  # sub-cell width
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0            # raised, not cut
        assert result.min() >= 50.0           # never lowers
        assert any("1-cell width" in w for w in b.warnings)

    def test_dam_downstream_offset_raises_to_crest(self, tmp_path):
        # Sloped DEM so a downstream (lower) side genuinely exists.
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.5, (20, 20)).astype("float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        result = b.burn_earthworks([ew])
        assert result.max() >= 60.0

    def test_dam_sub_cell_raises_via_path(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=0.4, crest_elevation=60.0)
        result = b.burn_earthworks([ew])
        assert result.max() >= 60.0
        assert any("1-cell width" in w for w in b.warnings)

    def test_dam_parallel_offset_failure_falls_back_to_centred(self, tmp_path, monkeypatch):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        from shapely.geometry import LineString as _LS

        def _raise(*a, **kw):
            raise ValueError("mock parallel_offset failure")

        monkeypatch.setattr(_LS, "parallel_offset", _raise, raising=False)

        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        result = b.burn_earthworks([ew])
        assert result.max() >= 60.0  # centred-buffer fallback still raised the wall


class TestStrategyCWarnings:
    def test_warnings_reset_between_burns(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)

        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, companion_berm=False)
        ew.width = 0.4
        ew.bottom_width_m = 0.3
        b.burn_earthworks([ew])
        assert b.warnings  # sub-cell warning present

        b.burn_earthworks([])  # fresh burn clears prior warnings
        assert b.warnings == []

    def test_ponding_cap_warning_recorded(self, tmp_path, monkeypatch):
        import terrainflow_assessment.modules.earthwork_design as mod
        monkeypatch.setattr(mod, "_MAX_PONDING_CELLS", 25)  # force over-cap on 20×20

        data = np.full((20, 20), 50.0, dtype="float32")
        data[9:12, 9:12] = 48.0
        path = _make_dem(tmp_path, data)
        b = DEMBurner(path)
        b.get_ponding_layer(data)
        assert any("reduced resolution" in w for w in b.warnings)


# ---------------------------------------------------------------------------
# Dam analytical capacity — DEMBurner.dam_stage_storage (DEM flood behind crest)
# ---------------------------------------------------------------------------

class TestDamStageStorage:
    def _valley_dem(self, tmp_path):
        # A valley running N–S: floor at column 10, sides rise gently, slopes down
        # southward. A dam wall across it (E–W) pools water upstream (to the north).
        data = np.fromfunction(
            lambda r, c: 50.0 - r * 0.5 + np.abs(c - 10) * 1.0, (20, 20)
        ).astype("float32")
        return _make_dem(tmp_path, data)

    def _dam(self, crest):
        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        return _mock_ew("dam", geom, width=2.0, crest_elevation=crest)

    def test_dam_impounds_positive_volume(self, tmp_path):
        b = DEMBurner(self._valley_dem(tmp_path))
        assert b.dam_stage_storage(self._dam(48.0)) > 0.0

    def test_higher_crest_impounds_more(self, tmp_path):
        b = DEMBurner(self._valley_dem(tmp_path))
        assert b.dam_stage_storage(self._dam(49.0)) > b.dam_stage_storage(self._dam(46.0))

    def test_no_crest_returns_zero(self, tmp_path):
        b = DEMBurner(self._valley_dem(tmp_path))
        assert b.dam_stage_storage(self._dam(None)) == 0.0

    def test_reuses_supplied_baseline_ponding(self, tmp_path):
        # Passing a baseline ponding array must not raise and yields a finite volume.
        b = DEMBurner(self._valley_dem(tmp_path))
        baseline = np.zeros(b.shape, dtype="float32")
        assert b.dam_stage_storage(self._dam(48.0), baseline_ponding=baseline) >= 0.0

    # -- Keyed (opt-in) vs as-drawn on an ENCLOSED valley --------------------
    # High side walls (cols <6 / >14 = 60 m), a channel that rises northward
    # (upstream), and a low outlet to the south. A short dam across only the
    # channel middle leaks around its ends (water escapes south) unless keyed
    # into the walls, so it cleanly separates the two modes.
    def _enclosed_valley_dem(self, tmp_path):
        def elev(r, c):
            north_floor = 48.0 + np.clip(12 - r, 0, None) * 0.5
            channel = np.where(r >= 13, 40.0, north_floor)  # low southern outlet
            return np.where((c < 6) | (c > 14), 60.0, channel)  # side walls
        return _make_dem(tmp_path, np.fromfunction(elev, (20, 20)).astype("float32"))

    def _channel_dam(self, crest):
        # Spans only the channel middle (cols 8–12 at y=8 → row 12), not the walls.
        return _mock_ew(
            "dam", make_mock_line_geom([(8.0, 8.0), (12.0, 8.0)]),
            width=2.0, crest_elevation=crest,
        )

    def test_as_drawn_short_dam_leaks_no_storage(self, tmp_path):
        # Default (as drawn): a short wall leaks around its ends — no reliable storage,
        # and raising the crest doesn't inflate the number.
        b = DEMBurner(self._enclosed_valley_dem(tmp_path))
        assert b.dam_stage_storage(self._channel_dam(50.0)) == pytest.approx(0.0)
        assert b.dam_stage_storage(self._channel_dam(54.0)) == pytest.approx(0.0)

    def test_keyed_short_dam_keeps_growing_with_crest(self, tmp_path):
        # Opt-in keyed estimate: the wall is virtually extended into the walls, so it
        # fills to the crest and storage keeps rising with the crest.
        b = DEMBurner(self._enclosed_valley_dem(tmp_path))
        mid = b.dam_stage_storage(self._channel_dam(50.0), key_into_banks=True)
        high = b.dam_stage_storage(self._channel_dam(54.0), key_into_banks=True)
        assert high > mid > 0.0

    def test_keyed_beats_as_drawn_for_short_dam(self, tmp_path):
        # For the same short dam, keying holds far more than the honest as-drawn number.
        b = DEMBurner(self._enclosed_valley_dem(tmp_path))
        as_drawn = b.dam_stage_storage(self._channel_dam(52.0))
        keyed = b.dam_stage_storage(self._channel_dam(52.0), key_into_banks=True)
        assert keyed > as_drawn

    def test_keying_emits_bank_warning(self, tmp_path):
        # A short dam whose crest tops its banks must key in and warn (keyed mode only).
        b = DEMBurner(self._enclosed_valley_dem(tmp_path))
        b.dam_stage_storage(self._channel_dam(52.0), key_into_banks=True)
        assert any("bank" in w.lower() for w in b.warnings)

    def test_as_drawn_short_dam_no_bank_warning(self, tmp_path):
        # Default mode never keys in, so it never emits the bank advisory.
        b = DEMBurner(self._enclosed_valley_dem(tmp_path))
        b.dam_stage_storage(self._channel_dam(52.0))
        assert not any("bank" in w.lower() for w in b.warnings)


class TestFeatureStorage:
    """What a feature impounds is measured by flooding it, not by filling its footprint.

    The distinction is invisible on flat ground and is the whole answer on a slope. A
    swale with a companion berm keyed into its banks holds water **above natural
    ground**, which stands deeper than the trench and reaches further up the hill than
    the trench does. A footprint integrated to its own one-cell rim can see neither: the
    rim's minimum is the *uphill* lip, and water crossing that lip runs into rising
    ground and cannot leave, so the lip is not an outlet and the level it implies is too
    low. On Swale 5 of the Quail Island design that cost 439 m³ against a pond of 1,095.
    """

    def _hillside(self, tmp_path, slope=0.05, rows=120, cols=160):
        """Planar ground falling south — the case a footprint integral gets wrong."""
        data = np.fromfunction(
            lambda r, c: 60.0 - r * slope, (rows, cols)).astype("float32")
        return _make_dem(tmp_path, data)

    def _contour_swale(self, y=60.0):
        return _mock_ew(
            "swale", make_mock_line_geom([(30.0, y), (130.0, y)]),
            depth=1.0, width=3.0, bottom_width_m=1.0,
            companion_berm=True, key_into_banks=True,
        )

    def test_the_pond_is_larger_than_the_trench_it_is_held_in(self, tmp_path):
        b = DEMBurner(self._hillside(tmp_path))
        ew = self._contour_swale()
        b.burn_earthworks([ew])
        trench = next(iter(b.burned_cut.values()))
        pond = b.feature_storage(ew).volume_m3

        assert trench > 0.0
        assert pond > trench * 1.2, (
            f"a keyed berm on falling ground impounds beyond its trench, but the pond "
            f"({pond:,.0f} m³) is barely past the cut ({trench:,.0f} m³)"
        )

    def test_a_bare_trench_on_flat_ground_holds_only_its_trench(self, tmp_path):
        """The converse — without a bank to retain anything the two must agree.

        This is what stops the flood being read as a licence to inflate. Where there is
        no structure holding water above ground, the pond *is* the hole.
        """
        data = np.full((80, 160), 50.0, dtype="float32")
        b = DEMBurner(_make_dem(tmp_path, data))
        ew = _mock_ew("swale", make_mock_line_geom([(30.0, 40.0), (130.0, 40.0)]),
                      depth=1.0, width=6.0, bottom_width_m=2.0, companion_berm=False)
        b.burn_earthworks([ew])
        trench = next(iter(b.burned_cut.values()))
        pond = b.feature_storage(ew).volume_m3
        assert pond == pytest.approx(trench, rel=0.05)

    def test_the_pond_is_attributed_by_region_not_by_window(self, tmp_path):
        """A natural hollow that shares the crop is not this feature's water.

        Summing the window would credit the swale with the pothole beside it. The new
        ponding is labelled and only the regions touching the feature's own burn mask
        are counted, so the hollow — which ponds identically before and after — is both
        subtracted by the baseline *and* excluded by attribution.
        """
        data = np.fromfunction(
            lambda r, c: 60.0 - r * 0.05, (120, 160)).astype("float32")
        data[20:30, 120:140] -= 5.0          # a deep pothole, well off the alignment
        b = DEMBurner(_make_dem(tmp_path, data))
        ew = self._contour_swale()
        b.burn_earthworks([ew])
        storage = b.feature_storage(ew)

        hollow = np.zeros(b.shape, dtype=bool)
        hollow[20:30, 120:140] = True
        r_lo, r_hi, c_lo, c_hi = b._feature_cell_bounds(ew)
        assert hollow[max(0, r_lo - 64):r_hi + 65, max(0, c_lo - 64):c_hi + 65].any(), (
            "fixture is wrong: the hollow must fall inside the flood window to be a test"
        )
        assert storage.volume_m3 > 0.0
        assert storage.volume_m3 < 5.0 * 10.0 * 20.0, (
            "the pothole's volume has been credited to the swale"
        )

    def test_it_reports_what_stands_above_natural_ground(self, tmp_path):
        """The two figures the retaining-structure warning is gated on."""
        b = DEMBurner(self._hillside(tmp_path))
        ew = self._contour_swale()
        b.burn_earthworks([ew])
        s = b.feature_storage(ew)
        assert 0.0 < s.above_ground_m3 < s.volume_m3
        assert s.retained_depth_m > 0.0
        assert s.level_m is not None

    def test_it_is_order_independent(self, tmp_path):
        """Two neighbouring swales measure the same whichever was burned first.

        The point of flooding the feature *alone*: read off the running array instead,
        one rim cell a neighbour has already trenched takes the whole datum with it.
        """
        b = DEMBurner(self._hillside(tmp_path))
        a = self._contour_swale(y=60.0)
        c = self._contour_swale(y=64.0)
        b.burn_earthworks([a, c])
        first = (b.feature_storage(a).volume_m3, b.feature_storage(c).volume_m3)
        b.burn_earthworks([c, a])
        second = (b.feature_storage(a).volume_m3, b.feature_storage(c).volume_m3)
        assert first == pytest.approx(second)

    def test_a_feature_that_ponds_nothing_reports_zero(self, tmp_path):
        b = DEMBurner(self._hillside(tmp_path))
        ew = _mock_ew("swale", make_mock_line_geom([(500.0, 500.0), (600.0, 500.0)]),
                      depth=1.0, width=3.0, bottom_width_m=1.0)
        s = b.feature_storage(ew)
        assert s.volume_m3 == 0.0
        assert s.level_m is None


class TestDamWindowedFlood:
    """The dam flood runs on a crop around the dam, growing until the pond fits."""

    def _long_valley_dem(self, tmp_path):
        # 300-row valley descending gently southward (0.02 m/row), centre col 20.
        # A dam near the south end ponds ~100 rows upstream — larger than the
        # initial 64-cell window pad, so the crop must grow to fit the pond.
        data = np.fromfunction(
            lambda r, c: 100.0 - r * 0.02 + np.abs(c - 20) * 1.0, (300, 40)
        ).astype("float32")
        return _make_dem(tmp_path, data)

    def _dam(self, crest):
        # Across the valley at y=50 → row 250 (floor there = 95.0).
        return _mock_ew(
            "dam", make_mock_line_geom([(10.0, 50.0), (30.0, 50.0)]),
            width=2.0, crest_elevation=crest,
        )

    def test_windowed_matches_full_dem_flood(self, tmp_path):
        from terrainflow_assessment.modules.reporting import impounded_volume
        b = DEMBurner(self._long_valley_dem(tmp_path))
        dam = self._dam(97.0)  # pond reaches ~row 150 → beyond the initial pad
        windowed = b.dam_stage_storage(dam)

        dammed = b.burn_earthworks([dam])
        full = impounded_volume(
            b.get_ponding_layer(b.original), b.get_ponding_layer(dammed), 1.0
        )
        assert windowed > 0.0
        assert windowed == pytest.approx(full, rel=1e-3)

    def test_small_pond_stays_positive(self, tmp_path):
        # A modest crest ponds well inside the first window — no growth needed.
        b = DEMBurner(self._long_valley_dem(tmp_path))
        assert b.dam_stage_storage(self._dam(95.5)) > 0.0

    def test_pond_touches_edge_helper(self):
        # The check reads the ring ONE cell in from the boundary — depression-fill
        # drains the boundary itself, so a clipped pond appears there instead.
        from terrainflow_assessment.modules.earthwork_design import _pond_touches_edge
        z = np.zeros((6, 6))
        assert not _pond_touches_edge(z)
        for r, c in ((1, 3), (4, 3), (3, 1), (3, 4)):
            arr = np.zeros((6, 6))
            arr[r, c] = 0.5
            assert _pond_touches_edge(arr)
        centre = np.zeros((7, 7))
        centre[3, 3] = 0.5
        assert not _pond_touches_edge(centre)
        assert _pond_touches_edge(np.zeros((3, 3)))  # tiny window always grows

    def test_feature_cell_bounds_clamped(self, tmp_path):
        b = DEMBurner(self._long_valley_dem(tmp_path))
        r_lo, r_hi, c_lo, c_hi = b._feature_cell_bounds(self._dam(97.0))
        assert 0 <= r_lo <= r_hi < 300
        assert 0 <= c_lo <= c_hi < 40

    def test_feature_cell_bounds_bad_geometry_full_dem(self, tmp_path):
        from unittest.mock import MagicMock
        b = DEMBurner(self._long_valley_dem(tmp_path))
        bad = MagicMock()
        bad.asJson.side_effect = RuntimeError("no geometry")
        dam = _mock_ew("dam", bad, width=2.0, crest_elevation=97.0)
        assert b._feature_cell_bounds(dam) == (0, 299, 0, 39)

    def test_feature_cell_bounds_takes_a_polygon(self, tmp_path):
        """Generalised from the dam-only version, and a basin is why.

        Every feature's pond is now flooded in its own window, not just a dam's, so the
        bounds helper has to accept the polygon types too. ``.bounds`` is a shapely
        property common to both, so this only ever needed the name to stop lying.
        """
        b = DEMBurner(self._long_valley_dem(tmp_path))
        basin = _mock_ew("basin", make_mock_polygon_geom((5.0, 100.0, 25.0, 140.0)))
        r_lo, r_hi, c_lo, c_hi = b._feature_cell_bounds(basin)
        assert 0 <= r_lo < r_hi < 300
        assert 0 <= c_lo < c_hi < 40
