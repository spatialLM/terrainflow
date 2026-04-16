"""Tests for plugin/processing/earthwork.py"""
import pytest

from plugin.processing.earthwork import (
    Earthwork,
    EarthworkManager,
    berm_height_estimate,
    calculate_capacity,
    calculate_diversion_discharge,
    calculate_spillway_width,
)
from tests.conftest import make_mock_line_geom, make_mock_polygon_geom


# ---------------------------------------------------------------------------
# Earthwork data class
# ---------------------------------------------------------------------------

class TestEarthwork:
    def _make(self, ew_type="swale"):
        geom = make_mock_line_geom()
        return Earthwork(ew_type, geom, f"Test {ew_type}")

    def test_defaults(self):
        ew = self._make("swale")
        assert ew.depth == 0.5
        assert ew.width == 2.0
        assert ew.enabled is True
        assert ew.companion_berm is False
        assert ew.crest_elevation is None
        assert ew.capacity_m3 == 0.0

    def test_type_label_swale(self):
        assert self._make("swale").type_label() == "Swale"

    def test_type_label_berm(self):
        assert self._make("berm").type_label() == "Berm"

    def test_type_label_basin(self):
        assert self._make("basin").type_label() == "Basin"

    def test_type_label_diversion(self):
        assert self._make("diversion").type_label() == "Diversion Drain"

    def test_summary_dam_no_crest(self):
        ew = self._make("dam")
        s = ew.summary()
        assert "Dam" in s
        assert "?" in s  # no crest elevation set

    def test_summary_dam_with_crest(self):
        ew = self._make("dam")
        ew.crest_elevation = 42.5
        s = ew.summary()
        assert "42.5 m" in s

    def test_summary_diversion(self):
        ew = self._make("diversion")
        s = ew.summary()
        assert "Diversion" in s
        assert "m³/s" in s

    def test_summary_swale_enabled(self):
        ew = self._make("swale")
        ew.capacity_m3 = 10.0
        ew.capacity_l = 10_000.0
        s = ew.summary()
        assert "[OFF]" not in s

    def test_summary_swale_disabled(self):
        ew = self._make("swale")
        ew.enabled = False
        s = ew.summary()
        assert "[OFF]" in s


# ---------------------------------------------------------------------------
# EarthworkManager
# ---------------------------------------------------------------------------

class TestEarthworkManager:
    def _manager_with(self, n):
        m = EarthworkManager()
        for i in range(n):
            m.add(Earthwork("swale", make_mock_line_geom(), f"S{i}"))
        return m

    def test_initial_empty(self):
        assert len(EarthworkManager()) == 0

    def test_add_increments_length(self):
        m = EarthworkManager()
        m.add(Earthwork("swale", make_mock_line_geom(), "S1"))
        assert len(m) == 1

    def test_get_returns_correct_item(self):
        m = self._manager_with(3)
        assert m.get(1).name == "S1"

    def test_get_all_returns_copy(self):
        m = self._manager_with(2)
        lst = m.get_all()
        lst.clear()
        assert len(m) == 2  # original unaffected

    def test_remove_valid_index(self):
        m = self._manager_with(3)
        m.remove(1)
        assert len(m) == 2
        assert m.get(0).name == "S0"
        assert m.get(1).name == "S2"

    def test_remove_invalid_index_noop(self):
        m = self._manager_with(2)
        m.remove(99)
        assert len(m) == 2

    def test_remove_negative_index_noop(self):
        m = self._manager_with(2)
        m.remove(-1)
        assert len(m) == 2

    def test_toggle_disables(self):
        m = self._manager_with(1)
        assert m.get(0).enabled is True
        m.toggle(0)
        assert m.get(0).enabled is False

    def test_toggle_re_enables(self):
        m = self._manager_with(1)
        m.toggle(0)
        m.toggle(0)
        assert m.get(0).enabled is True

    def test_toggle_invalid_index_noop(self):
        m = self._manager_with(1)
        m.toggle(99)  # should not raise
        assert m.get(0).enabled is True

    def test_get_enabled_filters_disabled(self):
        m = self._manager_with(3)
        m.toggle(1)  # disable index 1
        enabled = m.get_enabled()
        assert len(enabled) == 2
        assert all(e.enabled for e in enabled)

    def test_clear_empties_list(self):
        m = self._manager_with(5)
        m.clear()
        assert len(m) == 0

    def test_get_returns_correct_earthwork(self):
        m = self._manager_with(3)
        assert m.get(0).name == "S0"
        assert m.get(2).name == "S2"

    def test_get_enabled_empty_when_all_off(self):
        m = self._manager_with(2)
        m.toggle(0)
        m.toggle(1)
        assert m.get_enabled() == []


# ---------------------------------------------------------------------------
# calculate_capacity
# ---------------------------------------------------------------------------

class TestCalculateCapacity:
    def test_swale_basic(self):
        geom = make_mock_line_geom()  # length=10m
        geom.length.return_value = 10.0
        vol_m3, vol_l = calculate_capacity("swale", geom, depth=0.5, width=2.0)
        assert vol_m3 > 0
        assert vol_l == pytest.approx(vol_m3 * 1000, rel=1e-6)

    def test_swale_known_value(self):
        # depth=0.5, width=2.0, length=100
        # bottom_width = max(0.1, 2.0 - 2*0.5) = 1.0
        # top_width = 2.0 + 2*0.5 = 3.0
        # cross_section = (1.0+3.0)/2 * 0.5 = 1.0
        # volume = 1.0 * 100 * 0.8 = 80.0 m³
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        vol, _ = calculate_capacity("swale", geom, depth=0.5, width=2.0)
        assert vol == pytest.approx(80.0, rel=1e-3)

    def test_swale_with_companion_berm_larger(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 100.0
        vol_no_berm, _ = calculate_capacity("swale", geom, 0.5, 2.0, False)
        vol_with_berm, _ = calculate_capacity("swale", geom, 0.5, 2.0, True)
        assert vol_with_berm > vol_no_berm

    def test_basin_known_value(self):
        # area=100 m², depth=1.0 → vol = 100*1.0*0.8 = 80.0 m³
        geom = make_mock_polygon_geom()
        geom.area.return_value = 100.0
        vol, _ = calculate_capacity("basin", geom, depth=1.0, width=0)
        assert vol == pytest.approx(80.0, rel=1e-3)

    def test_berm_returns_zero(self):
        geom = make_mock_line_geom()
        assert calculate_capacity("berm", geom, 0.5, 2.0) == (0.0, 0.0)

    def test_dam_returns_zero(self):
        geom = make_mock_line_geom()
        assert calculate_capacity("dam", geom, 1.0, 3.0) == (0.0, 0.0)

    def test_diversion_returns_zero(self):
        geom = make_mock_line_geom()
        assert calculate_capacity("diversion", geom, 0.3, 1.0) == (0.0, 0.0)

    def test_volumes_rounded_to_2dp(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 33.333
        vol, _ = calculate_capacity("swale", geom, 0.5, 2.0)
        assert vol == round(vol, 2)

    def test_volume_l_is_1000x_m3(self):
        geom = make_mock_line_geom()
        geom.length.return_value = 50.0
        vol_m3, vol_l = calculate_capacity("swale", geom, 0.5, 2.0)
        assert vol_l == pytest.approx(vol_m3 * 1000, rel=1e-4)

    def test_bottom_width_floor_at_point1(self):
        # When width < 2*depth bottom_width = max(0.1, ...)
        geom = make_mock_line_geom()
        geom.length.return_value = 10.0
        vol, _ = calculate_capacity("swale", geom, depth=2.0, width=1.0)
        assert vol > 0


# ---------------------------------------------------------------------------
# calculate_diversion_discharge
# ---------------------------------------------------------------------------

class TestCalculateDiversionDischarge:
    def test_basic_positive_result(self):
        q = calculate_diversion_discharge(depth=0.5, width=2.0, gradient_pct=1.0)
        assert q > 0.0

    def test_zero_gradient_returns_zero(self):
        assert calculate_diversion_discharge(0.5, 2.0, 0.0) == 0.0

    def test_negative_gradient_returns_zero(self):
        assert calculate_diversion_discharge(0.5, 2.0, -1.0) == 0.0

    def test_zero_depth_returns_zero(self):
        assert calculate_diversion_discharge(0.0, 2.0, 1.0) == 0.0

    def test_zero_width_returns_zero(self):
        assert calculate_diversion_discharge(0.5, 0.0, 1.0) == 0.0

    def test_steeper_gradient_higher_flow(self):
        q1 = calculate_diversion_discharge(0.5, 2.0, 0.5)
        q2 = calculate_diversion_discharge(0.5, 2.0, 2.0)
        assert q2 > q1

    def test_deeper_channel_higher_flow(self):
        q1 = calculate_diversion_discharge(0.3, 2.0, 1.0)
        q2 = calculate_diversion_discharge(0.6, 2.0, 1.0)
        assert q2 > q1

    def test_returns_rounded_to_4dp(self):
        q = calculate_diversion_discharge(0.5, 2.0, 1.0)
        assert q == round(q, 4)

    def test_known_mannings(self):
        # For a known input, verify the Manning's equation result
        # n=0.025, depth=0.5, width=2.0, grade=1%
        # bottom_width = max(0.05, 2.0 - 2*0.5) = 1.0
        # top_width = 3.0
        # A = (1.0+3.0)/2 * 0.5 = 1.0
        # P = 1.0 + 2*sqrt(2)*0.5 = 1.0 + 1.414 = 2.414
        # R = 1.0/2.414 ≈ 0.4143
        # Q = (1/0.025) * 1.0 * 0.4143^(2/3) * 0.1^0.5
        import math
        n = 0.025
        d, w, g = 0.5, 2.0, 1.0
        s = g / 100.0
        bw = max(0.05, w - 2 * d)
        tw = w + 2 * d
        A = ((bw + tw) / 2) * d
        P = bw + 2 * math.sqrt(2) * d
        R = A / P
        expected = round((1 / n) * A * R ** (2/3) * s ** 0.5, 4)
        assert calculate_diversion_discharge(d, w, g) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# calculate_spillway_width
# ---------------------------------------------------------------------------

class TestCalculateSpillwayWidth:
    def test_basic_result(self):
        w = calculate_spillway_width(peak_flow_m3s=1.0, head_m=0.5)
        assert w > 0.0

    def test_zero_head_returns_zero(self):
        assert calculate_spillway_width(1.0, 0.0) == 0.0

    def test_negative_head_returns_zero(self):
        assert calculate_spillway_width(1.0, -0.5) == 0.0

    def test_zero_flow_returns_zero(self):
        assert calculate_spillway_width(0.0, 0.5) == 0.0

    def test_negative_flow_returns_zero(self):
        assert calculate_spillway_width(-1.0, 0.5) == 0.0

    def test_higher_flow_requires_wider_spillway(self):
        w1 = calculate_spillway_width(0.5, 0.5)
        w2 = calculate_spillway_width(1.5, 0.5)
        assert w2 > w1

    def test_known_value(self):
        # Q = C * L * H^1.5  →  L = Q / (C * H^1.5)
        # Q=1.0, C=1.7, H=0.5
        expected = round(1.0 / (1.7 * 0.5 ** 1.5), 2)
        assert calculate_spillway_width(1.0, 0.5) == pytest.approx(expected, rel=1e-4)

    def test_custom_weir_coefficient(self):
        w_default = calculate_spillway_width(1.0, 0.5, weir_coeff=1.7)
        w_custom = calculate_spillway_width(1.0, 0.5, weir_coeff=2.1)
        assert w_custom < w_default  # higher coeff → narrower spillway

    def test_rounded_to_2dp(self):
        w = calculate_spillway_width(1.23, 0.47)
        assert w == round(w, 2)


# ---------------------------------------------------------------------------
# berm_height_estimate
# ---------------------------------------------------------------------------

class TestBermHeightEstimate:
    def test_returns_positive(self):
        assert berm_height_estimate(0.5, 2.0) > 0.0

    def test_deeper_swale_taller_berm(self):
        h1 = berm_height_estimate(0.3, 2.0)
        h2 = berm_height_estimate(0.8, 2.0)
        assert h2 > h1

    def test_rounded_to_2dp(self):
        h = berm_height_estimate(0.5, 2.0)
        assert h == round(h, 2)

    def test_consistent_with_capacity_formula(self):
        # berm_height = sqrt(cross_section * 0.75)
        depth, width = 0.5, 2.0
        bottom_width = max(0.1, width - 2 * depth)
        top_width = width + 2 * depth
        cs = ((bottom_width + top_width) / 2) * depth
        expected = round((cs * 0.75) ** 0.5, 2)
        assert berm_height_estimate(depth, width) == pytest.approx(expected)
