"""Tests for plugin/processing/scs_runoff.py"""
import csv
import os

import numpy as np
import pytest
from rasterio.transform import from_bounds

from plugin.processing.scs_runoff import SCSRunoff


# ---------------------------------------------------------------------------
# adjust_cn
# ---------------------------------------------------------------------------

class TestAdjustCN:
    def setup_method(self):
        self.scs = SCSRunoff()

    def test_normal_returns_unchanged(self):
        assert self.scs.adjust_cn(70, "normal") == pytest.approx(70.0)

    def test_dry_lower_than_normal(self):
        assert self.scs.adjust_cn(70, "dry") < 70.0

    def test_wet_higher_than_normal(self):
        assert self.scs.adjust_cn(70, "wet") > 70.0

    def test_clamped_to_100(self):
        assert self.scs.adjust_cn(99, "wet") <= 100.0

    def test_clamped_to_1(self):
        assert self.scs.adjust_cn(2, "dry") >= 1.0

    def test_dry_known_value(self):
        # 4.2*70 / (10 - 0.058*70)
        expected = 4.2 * 70 / (10 - 0.058 * 70)
        assert self.scs.adjust_cn(70, "dry") == pytest.approx(expected, rel=1e-4)

    def test_wet_known_value(self):
        # 23*70 / (10 + 0.13*70)
        expected = 23 * 70 / (10 + 0.13 * 70)
        assert self.scs.adjust_cn(70, "wet") == pytest.approx(expected, rel=1e-4)

    def test_all_moisture_conditions_accepted(self):
        for cond in ("dry", "normal", "wet"):
            result = self.scs.adjust_cn(75, cond)
            assert 1.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# runoff_depth
# ---------------------------------------------------------------------------

class TestRunoffDepth:
    def setup_method(self):
        self.scs = SCSRunoff()

    def test_zero_cn_returns_zero(self):
        assert self.scs.runoff_depth(100, 0) == 0.0

    def test_negative_cn_returns_zero(self):
        assert self.scs.runoff_depth(100, -5) == 0.0

    def test_below_initial_abstraction_returns_zero(self):
        # CN=70 → S≈108.9 mm, Ia≈21.8 mm  — 10 mm rainfall < Ia
        assert self.scs.runoff_depth(10, 70) == 0.0

    def test_at_initial_abstraction_boundary_returns_zero(self):
        # rainfall exactly equal to Ia → still 0
        s = (25400 / 70) - 254
        ia = 0.2 * s
        assert self.scs.runoff_depth(ia, 70) == 0.0

    def test_above_initial_abstraction_positive(self):
        assert self.scs.runoff_depth(100, 70) > 0.0

    def test_increases_with_rainfall(self):
        q1 = self.scs.runoff_depth(50, 75)
        q2 = self.scs.runoff_depth(100, 75)
        assert q2 > q1

    def test_increases_with_cn(self):
        q_low = self.scs.runoff_depth(50, 60)
        q_high = self.scs.runoff_depth(50, 90)
        assert q_high > q_low

    def test_cn_100_equals_rainfall(self):
        # CN=100 → S=0, Ia=0, Q=P
        q = self.scs.runoff_depth(50, 100)
        assert q == pytest.approx(50.0, rel=1e-4)

    def test_known_value_cn75_p100(self):
        # CN=75: S=84.67, Ia=16.93
        # Q = (100-16.93)^2 / (100-16.93+84.67) = 83.07^2/167.74 ≈ 41.16
        q = self.scs.runoff_depth(100, 75)
        assert q == pytest.approx(41.16, abs=0.1)

    def test_never_negative(self):
        for cn in [10, 50, 75, 98]:
            for rain in [0, 5, 50, 200]:
                assert self.scs.runoff_depth(rain, cn) >= 0.0

    def test_runoff_less_than_or_equal_to_rainfall(self):
        q = self.scs.runoff_depth(80, 85)
        assert q <= 80.0


# ---------------------------------------------------------------------------
# runoff_ratio
# ---------------------------------------------------------------------------

class TestRunoffRatio:
    def setup_method(self):
        self.scs = SCSRunoff()

    def test_zero_rainfall_returns_zero(self):
        assert self.scs.runoff_ratio(0, 70) == 0.0

    def test_negative_rainfall_returns_zero(self):
        assert self.scs.runoff_ratio(-10, 70) == 0.0

    def test_below_ia_returns_zero(self):
        assert self.scs.runoff_ratio(5, 70) == 0.0

    def test_ratio_between_0_and_1(self):
        r = self.scs.runoff_ratio(100, 75)
        assert 0.0 <= r <= 1.0

    def test_higher_cn_gives_higher_ratio(self):
        r1 = self.scs.runoff_ratio(50, 65)
        r2 = self.scs.runoff_ratio(50, 85)
        assert r2 > r1

    def test_ratio_consistent_with_runoff_depth(self):
        rain = 80.0
        cn = 78
        depth = self.scs.runoff_depth(rain, cn)
        ratio = self.scs.runoff_ratio(rain, cn)
        assert ratio == pytest.approx(depth / rain, rel=1e-6)


# ---------------------------------------------------------------------------
# catchment_volume
# ---------------------------------------------------------------------------

class TestCatchmentVolume:
    def setup_method(self):
        self.scs = SCSRunoff()

    def test_10mm_over_1ha(self):
        # 10mm over 10,000 m² = 100 m³
        assert self.scs.catchment_volume(10.0, 10_000) == pytest.approx(100.0, rel=1e-6)

    def test_zero_runoff(self):
        assert self.scs.catchment_volume(0.0, 10_000) == 0.0

    def test_zero_area(self):
        assert self.scs.catchment_volume(50.0, 0.0) == 0.0

    def test_unit_conversion(self):
        # 1mm over 1 m² = 0.001 m³
        assert self.scs.catchment_volume(1.0, 1.0) == pytest.approx(0.001, rel=1e-9)


# ---------------------------------------------------------------------------
# build_runoff_raster
# ---------------------------------------------------------------------------

class TestBuildRunoffRaster:
    def setup_method(self):
        self.scs = SCSRunoff()

    def test_shape_preserved(self):
        cn = np.full((5, 5), 75, dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        assert result.shape == (5, 5)

    def test_dtype_float32(self):
        cn = np.full((3, 3), 70, dtype="float32")
        result = self.scs.build_runoff_raster(cn, 50.0)
        assert result.dtype == np.float32

    def test_zero_cn_produces_zero(self):
        cn = np.zeros((4, 4), dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        assert np.all(result == 0.0)

    def test_low_rainfall_produces_zero(self):
        cn = np.full((3, 3), 70, dtype="float32")
        # CN=70 Ia ≈ 21.8 mm — rainfall 5mm should be zero
        result = self.scs.build_runoff_raster(cn, 5.0)
        assert np.all(result == 0.0)

    def test_values_match_scalar_method(self):
        cn = np.array([[70, 80], [75, 90]], dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        for r in range(2):
            for c in range(2):
                expected = self.scs.runoff_depth(100.0, float(cn[r, c]))
                assert float(result[r, c]) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# build_cn_raster
# ---------------------------------------------------------------------------

class TestBuildCNRaster:
    def setup_method(self):
        self.scs = SCSRunoff()
        self.shape = (10, 10)
        self.transform = from_bounds(0, 0, 10, 10, 10, 10)

    def test_no_zones_returns_default(self):
        result = self.scs.build_cn_raster(self.shape, self.transform, [], 70, "normal")
        assert result.shape == self.shape
        assert np.allclose(result, 70.0)

    def test_zone_overrides_default(self):
        from shapely.geometry import box
        zone = box(0, 5, 5, 10)  # top-left quadrant
        result = self.scs.build_cn_raster(
            self.shape, self.transform, [(zone, 90)], 70, "normal"
        )
        assert result.max() > 70.0

    def test_none_geometry_skipped(self):
        result = self.scs.build_cn_raster(
            self.shape, self.transform, [(None, 90)], 70, "normal"
        )
        assert np.allclose(result, 70.0)

    def test_empty_geometry_skipped(self):
        from shapely.geometry import Polygon
        result = self.scs.build_cn_raster(
            self.shape, self.transform, [(Polygon(), 90)], 70, "normal"
        )
        assert np.allclose(result, 70.0)

    def test_moisture_dry_lowers_cn(self):
        result = self.scs.build_cn_raster(self.shape, self.transform, [], 70, "dry")
        assert result.mean() < 70.0

    def test_moisture_wet_raises_cn(self):
        result = self.scs.build_cn_raster(self.shape, self.transform, [], 70, "wet")
        assert result.mean() > 70.0

    def test_output_dtype_float32(self):
        result = self.scs.build_cn_raster(self.shape, self.transform, [], 70, "normal")
        assert result.dtype == np.float32

    def test_multiple_zones_later_overrides_earlier(self):
        from shapely.geometry import box
        full = box(0, 0, 10, 10)
        result = self.scs.build_cn_raster(
            self.shape, self.transform,
            [(full, 80), (full, 95)],  # second zone should win
            70, "normal"
        )
        assert np.allclose(result, 95.0, atol=1.0)


# ---------------------------------------------------------------------------
# parse_hyetograph_csv
# ---------------------------------------------------------------------------

class TestParseHyetographCSV:

    def _write(self, path, header, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header.split(","))
            writer.writerows(rows)

    def test_cumulative_data_preserved(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        self._write(p, "time_min,rainfall_mm", [(10, 5), (20, 15), (30, 30)])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[0] == (0, 0.0)
        times = [t for t, _ in result]
        assert 10 in times and 20 in times and 30 in times

    def test_per_interval_is_cumulated(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        # non-monotonic → treated as per-interval
        self._write(p, "time_min,rainfall_mm", [(10, 5), (20, 10), (30, 8)])
        result = SCSRunoff.parse_hyetograph_csv(p)
        final_rain = result[-1][1]
        assert final_rain == pytest.approx(23.0, abs=0.01)

    def test_missing_columns_raises_valueerror(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        self._write(p, "time_hours,millimetres", [(10, 5)])
        with pytest.raises(ValueError, match="time_min"):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_empty_file_raises_valueerror(self, tmp_path):
        p = str(tmp_path / "empty.csv")
        open(p, "w").close()
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_nonexistent_path_raises_valueerror(self):
        with pytest.raises(ValueError, match="Cannot open"):
            SCSRunoff.parse_hyetograph_csv("/no/such/file.csv")

    def test_prepends_t0_if_missing(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        self._write(p, "time_min,rainfall_mm", [(10, 5), (20, 15)])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[0][0] == 0

    def test_no_duplicate_zero_if_data_starts_at_t0(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        self._write(p, "time_min,rainfall_mm", [(0, 0), (10, 5), (20, 15)])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[0] == (0, 0.0)
        assert result[1][0] == 10  # no double zero

    def test_result_sorted_ascending(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        # Out-of-order input
        self._write(p, "time_min,rainfall_mm", [(30, 15), (10, 5), (20, 10)])
        result = SCSRunoff.parse_hyetograph_csv(p)
        times = [t for t, _ in result]
        assert times == sorted(times)

    def test_invalid_rows_skipped(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm\n")
            f.write("10,5\n")
            f.write("bad,row\n")  # will be skipped
            f.write("20,15\n")
        result = SCSRunoff.parse_hyetograph_csv(p)
        times = [t for t, _ in result]
        assert 10 in times and 20 in times

    def test_no_valid_rows_raises_valueerror(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm\n")
            f.write("x,y\n")
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_extra_columns_ignored(self, tmp_path):
        """Extra columns beyond time_min and rainfall_mm are silently ignored."""
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm,notes\n10,5,light\n20,15,moderate\n")
        result = SCSRunoff.parse_hyetograph_csv(p)
        times = [t for t, _ in result]
        assert 10 in times and 20 in times


# ---------------------------------------------------------------------------
# Class-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_soil_reference_has_5_entries(self):
        assert len(SCSRunoff.SOIL_REFERENCE) == 5

    def test_clay_higher_cn_than_sand(self):
        ref = SCSRunoff.SOIL_REFERENCE
        assert ref["Clay"] > ref["Sand"]

    def test_storm_presets_include_custom(self):
        assert "Custom" in SCSRunoff.STORM_PRESETS

    def test_storm_presets_has_at_least_4(self):
        assert len(SCSRunoff.STORM_PRESETS) >= 4
