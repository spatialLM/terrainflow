"""Tests for terrainflow_assessment/modules/swale_design.py"""
import pytest

from terrainflow_assessment.modules.swale_design import (
    INFILTRATION_RATE_MM_HR,
    SOIL_REFERENCE,
    contour_to_swale_geometry,
    get_infiltration_rate,
    recommend_swale_length,
    snap_point_to_contour_elevation,
)
from plugin.processing.scs_runoff import SCSRunoff


# ---------------------------------------------------------------------------
# SOIL_REFERENCE
# ---------------------------------------------------------------------------

class TestSoilReference:
    def test_matches_scs_soil_reference(self):
        assert SOIL_REFERENCE == SCSRunoff.SOIL_REFERENCE

    def test_has_5_soils(self):
        assert len(SOIL_REFERENCE) == 5

    def test_contains_expected_soils(self):
        for soil in ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]:
            assert soil in SOIL_REFERENCE

    def test_cn_values_in_range(self):
        for cn in SOIL_REFERENCE.values():
            assert 1 <= cn <= 100


# ---------------------------------------------------------------------------
# INFILTRATION_RATE_MM_HR
# ---------------------------------------------------------------------------

class TestInfiltrationRates:
    def test_has_5_soils(self):
        assert len(INFILTRATION_RATE_MM_HR) == 5

    def test_sand_highest_rate(self):
        assert INFILTRATION_RATE_MM_HR["Sand"] == max(INFILTRATION_RATE_MM_HR.values())

    def test_clay_lowest_rate(self):
        assert INFILTRATION_RATE_MM_HR["Clay"] == min(INFILTRATION_RATE_MM_HR.values())

    def test_rates_positive(self):
        assert all(v > 0 for v in INFILTRATION_RATE_MM_HR.values())

    def test_descending_order(self):
        order = ["Sand", "Sandy loam", "Loam", "Clay loam", "Clay"]
        rates = [INFILTRATION_RATE_MM_HR[s] for s in order]
        assert rates == sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# get_infiltration_rate
# ---------------------------------------------------------------------------

class TestGetInfiltrationRate:
    def test_known_soil_sand(self):
        assert get_infiltration_rate("Sand") == INFILTRATION_RATE_MM_HR["Sand"]

    def test_known_soil_clay(self):
        assert get_infiltration_rate("Clay") == INFILTRATION_RATE_MM_HR["Clay"]

    def test_known_soil_loam(self):
        assert get_infiltration_rate("Loam") == INFILTRATION_RATE_MM_HR["Loam"]

    def test_unknown_soil_returns_loam(self):
        assert get_infiltration_rate("Gravel") == INFILTRATION_RATE_MM_HR["Loam"]

    def test_empty_string_returns_loam(self):
        assert get_infiltration_rate("") == INFILTRATION_RATE_MM_HR["Loam"]

    def test_all_known_soils_return_correct_rate(self):
        for soil, rate in INFILTRATION_RATE_MM_HR.items():
            assert get_infiltration_rate(soil) == rate


# ---------------------------------------------------------------------------
# recommend_swale_length
# ---------------------------------------------------------------------------

class TestRecommendSwaleLength:
    def test_basic_calculation(self):
        # V=100, depth=0.5, width=2.0, cs=1.0, length=100/0.8=125
        length = recommend_swale_length(100.0, 0.5, 2.0)
        assert length == pytest.approx(125.0, rel=1e-3)

    def test_zero_depth_returns_zero(self):
        assert recommend_swale_length(100.0, 0.0, 2.0) == 0.0

    def test_zero_width_returns_zero(self):
        assert recommend_swale_length(100.0, 0.5, 0.0) == 0.0

    def test_zero_inflow_returns_zero(self):
        assert recommend_swale_length(0.0, 0.5, 2.0) == 0.0

    def test_negative_depth_returns_zero(self):
        assert recommend_swale_length(100.0, -1.0, 2.0) == 0.0

    def test_larger_volume_longer_swale(self):
        l1 = recommend_swale_length(100.0, 0.5, 2.0)
        l2 = recommend_swale_length(200.0, 0.5, 2.0)
        assert l2 > l1

    def test_deeper_swale_shorter_length(self):
        l1 = recommend_swale_length(100.0, 0.5, 2.0)
        l2 = recommend_swale_length(100.0, 1.0, 2.0)
        assert l2 < l1

    def test_wider_swale_shorter_length(self):
        l1 = recommend_swale_length(100.0, 0.5, 1.0)
        l2 = recommend_swale_length(100.0, 0.5, 3.0)
        assert l2 < l1

    def test_rounded_to_1dp(self):
        length = recommend_swale_length(100.0, 0.5, 2.0)
        assert length == round(length, 1)

    def test_formula_consistency(self):
        """length × cross_section × 0.8 == inflow volume."""
        depth, width, volume = 0.5, 2.0, 150.0
        length = recommend_swale_length(volume, depth, width)
        cs = depth * width
        assert length * cs * 0.8 == pytest.approx(volume, rel=1e-3)


# ---------------------------------------------------------------------------
# contour_to_swale_geometry
# ---------------------------------------------------------------------------

class TestContourToSwaleGeometry:
    def test_returns_same_object(self):
        from unittest.mock import MagicMock
        geom = MagicMock()
        assert contour_to_swale_geometry(geom) is geom


# ---------------------------------------------------------------------------
# snap_point_to_contour_elevation
# ---------------------------------------------------------------------------

class TestSnapPointToContourElevation:
    def test_returns_elevation_for_valid_point(self, tmp_dem):
        # tmp_dem is 20×20 with 1m cells, elevation 100 - 2*row
        # Point (0.5, 19.5) → row=0, col=0 → elevation=100
        elev = snap_point_to_contour_elevation((0.5, 19.5), tmp_dem)
        assert elev is not None
        assert isinstance(elev, float)

    def test_returns_none_for_outside_bounds(self, tmp_dem):
        elev = snap_point_to_contour_elevation((999.0, 999.0), tmp_dem)
        assert elev is None

    def test_returns_none_for_bad_path(self):
        elev = snap_point_to_contour_elevation((5.0, 5.0), "/no/such/dem.tif")
        assert elev is None

    def test_nodata_cell_returns_none(self, tmp_path):
        """A point landing on a nodata cell should return None."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        data = np.full((10, 10), -9999.0, dtype="float32")
        path = str(tmp_path / "nodata.tif")
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10,
            count=1, dtype="float32", crs="EPSG:32632",
            transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)

        elev = snap_point_to_contour_elevation((5.0, 5.0), path)
        assert elev is None
