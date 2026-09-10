"""Tests for terrainflow_assessment/modules/swale_design.py"""
import pytest

from terrainflow_assessment.modules.catchment import SCSRunoff
from terrainflow_assessment.modules.swale_design import (
    INFILTRATION_RATE_MM_HR,
    SOIL_REFERENCE,
    contour_section,
    contour_to_swale_geometry,
    get_infiltration_rate,
    inflow_profile,
    overtopping_station,
    recommend_swale_length,
    required_storage_at_length,
    snap_point_to_contour_elevation,
)

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
    def test_basic_calculation_storage_only(self):
        # Trapezoidal area for T=2.0, d=0.5, side_slope=1.0:
        #   b = 2 - 2·1·0.5 = 1.0;  A = (2+1)/2 · 0.5 = 0.75 m²
        #   capacity/m = 0.75 m³/m, brim-full;  L = 100 / 0.75 = 133.3 m
        length = recommend_swale_length(100.0, 0.5, 2.0)
        assert length == pytest.approx(133.3, rel=1e-3)

    def test_infiltration_shortens_length(self):
        # Adding infiltration over the event increases capacity per metre,
        # so the required length falls below the storage-only figure.
        storage_only = recommend_swale_length(100.0, 0.5, 2.0)
        with_infil = recommend_swale_length(
            100.0, 0.5, 2.0, infiltration_mm_hr=4.0, duration_hr=6.0)
        assert with_infil < storage_only

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

    def test_storage_only_formula_consistency(self):
        """With no infiltration, length × trapezoidal_area == inflow volume.

        No allowance in the middle of that identity: the sizing used to divide by
        ``area × 0.8``, which made every recommended length 25% longer than the
        storage it was sized from. Freeboard is the spillway's job now.
        """
        depth, width, volume = 0.5, 2.0, 150.0
        length = recommend_swale_length(volume, depth, width)
        bottom = max(0.0, width - 2.0 * 1.0 * depth)
        area = (width + bottom) / 2.0 * depth
        assert length * area == pytest.approx(volume, rel=1e-3)


# ---------------------------------------------------------------------------
# contour_to_swale_geometry
# ---------------------------------------------------------------------------

class TestContourToSwaleGeometry:
    def test_returns_same_object(self):
        from unittest.mock import MagicMock
        geom = MagicMock()
        assert contour_to_swale_geometry(geom) is geom


# ---------------------------------------------------------------------------
# contour_section
# ---------------------------------------------------------------------------

class TestContourSection:
    # A simple L-shaped contour: 10 m east then 10 m north.
    CONTOUR = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

    def test_section_between_two_points(self):
        section = contour_section(self.CONTOUR, (2.0, 0.0), (8.0, 0.0))
        assert section is not None
        assert section[0] == (2.0, 0.0)
        assert section[-1] == (8.0, 0.0)

    def test_follows_contour_around_corner(self):
        # Section spanning the corner must include the corner vertex — i.e. it
        # follows the contour rather than cutting straight across.
        section = contour_section(self.CONTOUR, (5.0, 0.0), (10.0, 5.0))
        assert (10.0, 0.0) in section

    def test_endpoint_order_is_irrelevant(self):
        a = contour_section(self.CONTOUR, (2.0, 0.0), (8.0, 0.0))
        b = contour_section(self.CONTOUR, (8.0, 0.0), (2.0, 0.0))
        assert a == b

    def test_off_contour_points_project_onto_it(self):
        # Points near (not on) the contour snap to their nearest point on it.
        section = contour_section(self.CONTOUR, (2.0, 1.5), (8.0, -1.5))
        assert section[0] == (2.0, 0.0)
        assert section[-1] == (8.0, 0.0)

    def test_zero_length_section_none(self):
        assert contour_section(self.CONTOUR, (5.0, 0.0), (5.0, 0.0)) is None

    def test_empty_contour_none(self):
        assert contour_section([], (0.0, 0.0), (1.0, 0.0)) is None

    def test_single_point_contour_none(self):
        assert contour_section([(0.0, 0.0)], (0.0, 0.0), (1.0, 0.0)) is None

    def test_invalid_coords_none(self):
        assert contour_section([("a", "b"), ("c", "d")], (0.0, 0.0), (1.0, 0.0)) is None


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


# ---------------------------------------------------------------------------
# required_storage_at_length — the deficit-at-drawn-length readout
# ---------------------------------------------------------------------------

class TestRequiredStorageAtLength:
    # Default swale: top 2.0 m, depth 0.5 m, 1:1 batter → bottom 1.0 m,
    # trapezoid section 0.75 m², brim-full = 0.75 m³/m.
    DIMS = dict(depth=0.5, width=2.0, side_slope=1.0)

    def test_available_storage_uses_the_trapezoid_not_a_rectangle(self):
        r = required_storage_at_length(0.0, 100.0, **self.DIMS)
        assert r.storage_m3 == pytest.approx(75.0)     # 0.75 × 100
        # The old dialog assumed depth×width = 1.0 m³/m → 100 m³, 67% optimistic.
        assert r.storage_m3 < 100.0

    def test_holds_when_inflow_fits(self):
        r = required_storage_at_length(50.0, 100.0, **self.DIMS)
        assert r.holds is True
        assert r.deficit_m3 == 0.0

    def test_deficit_reported_when_short(self):
        r = required_storage_at_length(100.0, 100.0, **self.DIMS)
        assert r.holds is False
        assert r.deficit_m3 == pytest.approx(25.0)

    def test_required_depth_closes_the_deficit_exactly(self):
        """The headline promise: deepen to this and the deficit goes to zero."""
        inflow, length = 90.0, 100.0          # 90 m³ against 75 m³ available
        r = required_storage_at_length(inflow, length, **self.DIMS)
        assert not r.holds
        assert r.depth_reachable

        deeper = dict(self.DIMS, depth=r.required_depth_m)
        r2 = required_storage_at_length(inflow, length, **deeper)
        assert r2.deficit_m3 == pytest.approx(0.0, abs=1e-6)
        assert r2.holds

    def test_required_depth_with_vertical_walls(self):
        r = required_storage_at_length(80.0, 100.0, depth=0.5, width=2.0,
                                       side_slope=0.0)
        # Needs 0.8 m³/m over a 2.0 m rectangle → 0.4 m.
        assert r.required_depth_m == pytest.approx(0.4)

    def test_unreachable_section_is_flagged_not_faked(self):
        """Battered walls close in before the section is reached.

        At a 2 m top width with 1:1 batters the deepest possible section is 1.0 m²
        (walls meeting at 1.0 m depth), so anything beyond that is unreachable —
        the caller must say "widen it", not "deepen to 0.50 m" (the current depth).
        """
        r = required_storage_at_length(1e6, 10.0, depth=0.5, width=2.0,
                                       side_slope=1.0)
        assert r.depth_reachable is False
        assert r.required_depth_m == 0.5      # unchanged → no bogus advice
        assert not r.holds

    def test_reachable_deficit_is_not_flagged_unreachable(self):
        r = required_storage_at_length(90.0, 100.0, **self.DIMS)
        assert r.depth_reachable is True

    def test_infiltration_counts_toward_holding_the_event(self):
        dry = required_storage_at_length(100.0, 100.0, **self.DIMS)
        wet = required_storage_at_length(100.0, 100.0, infiltration_mm_hr=10.0,
                                         duration_hr=24.0, **self.DIMS)
        assert wet.infiltration_m3 == pytest.approx(48.0)   # 0.01 × 24 × 2 × 100
        assert wet.available_m3 > dry.available_m3
        assert wet.holds and not dry.holds

    def test_deficit_is_monotone_in_depth_and_width(self):
        base = required_storage_at_length(200.0, 100.0, **self.DIMS)
        deeper = required_storage_at_length(200.0, 100.0,
                                            **dict(self.DIMS, depth=0.8))
        wider = required_storage_at_length(200.0, 100.0,
                                           **dict(self.DIMS, width=3.0))
        assert deeper.deficit_m3 < base.deficit_m3
        assert wider.deficit_m3 < base.deficit_m3

    def test_recommended_length_is_carried_as_a_secondary_figure(self):
        r = required_storage_at_length(120.0, 100.0, **self.DIMS)
        assert r.recommended_length_m == pytest.approx(
            recommend_swale_length(120.0, 0.5, 2.0, side_slope=1.0)
        )

    def test_invalid_inputs_give_a_safe_empty_result(self):
        for kwargs in (dict(depth=0.0, width=2.0), dict(depth=0.5, width=0.0)):
            r = required_storage_at_length(100.0, 100.0, **kwargs)
            assert r.available_m3 == 0.0 and r.holds is False
        assert required_storage_at_length(100.0, 0.0, depth=0.5, width=2.0).holds is False


# ---------------------------------------------------------------------------
# inflow_profile / overtopping_station — inflow is not uniform along a swale
# ---------------------------------------------------------------------------

class TestInflowProfile:
    def test_evenly_spread_inflow_reads_as_uniform(self):
        d = [i * 5.0 for i in range(20)]        # one cell every 5 m over 100 m
        p = inflow_profile(d, [10.0] * 20, 100.0, n_stations=20)
        assert p["uniformity"] == pytest.approx(1.0)
        assert sum(p["inflow_m3"]) == pytest.approx(200.0)
        assert p["cumulative_m3"][-1] == pytest.approx(200.0)

    def test_concentrated_inflow_is_flagged_and_located(self):
        p = inflow_profile([40.0] * 10, [20.0] * 10, 100.0, n_stations=20)
        assert p["uniformity"] < 0.2            # it all arrives in one reach
        assert p["peak_station"] == pytest.approx(42.5)

    def test_stations_span_the_alignment(self):
        p = inflow_profile([1.0], [1.0], 100.0, n_stations=4)
        assert p["stations"] == [12.5, 37.5, 62.5, 87.5]
        assert p["station_length_m"] == pytest.approx(25.0)

    def test_a_cell_past_the_end_is_clamped_into_the_last_station(self):
        p = inflow_profile([150.0], [5.0], 100.0, n_stations=4)
        assert p["inflow_m3"][-1] == pytest.approx(5.0)

    def test_empty_or_degenerate_input_is_safe(self):
        assert inflow_profile([], [], 100.0)["uniformity"] == 1.0
        assert inflow_profile([1.0], [1.0], 0.0)["inflow_m3"] == [0.0] * 24
        assert inflow_profile([1.0, 2.0], [1.0], 100.0)["peak_station"] == 0.0
        assert inflow_profile([1.0], [0.0], 100.0)["uniformity"] == 1.0


class TestOvertoppingStation:
    def test_uniform_inflow_within_capacity_never_overtops(self):
        d = [i * 5.0 for i in range(20)]
        p = inflow_profile(d, [10.0] * 20, 100.0, n_stations=20)
        station, surplus = overtopping_station(p, capacity_per_m=5.0)
        assert station is None and surplus == 0.0

    def test_concentrated_inflow_overtops_even_when_the_total_fits(self):
        """Total capacity 100 m³ vs total inflow 60 m³ — yet it fails locally."""
        p = inflow_profile([10.0] * 6, [10.0] * 6, 100.0, n_stations=20)
        station, surplus = overtopping_station(p, capacity_per_m=1.0)
        assert station is not None
        assert station < 30.0            # fails near where the water arrives
        assert surplus > 0

    def test_degenerate_inputs_return_no_station(self):
        p = inflow_profile([10.0], [10.0], 100.0)
        assert overtopping_station(p, capacity_per_m=0.0) == (None, 0.0)
        assert overtopping_station(None, capacity_per_m=5.0) == (None, 0.0)
        assert overtopping_station({}, capacity_per_m=5.0) == (None, 0.0)


# ---------------------------------------------------------------------------
# SWL-03 — converged battered walls
# ---------------------------------------------------------------------------

class TestConvergedWallSection:
    """When 2·z·d exceeds the top width the walls meet before the drawn depth.

    Clamping only the bottom width to zero while keeping the full depth built a
    triangle taller than the batter permits: area T·d/2 instead of the true maximum
    T²/(4z), over-stating storage in the narrowest, deepest corner of the range.
    """

    def test_section_is_capped_at_the_batters_own_maximum(self):
        from terrainflow_assessment.modules.swale_design import _channel_section
        # T = 1.0, z = 1.0, d = 2.0 → walls meet at 0.5 m, well short of 2.0 m.
        sec = _channel_section(top_width=1.0, depth=2.0, side_slope=1.0)
        assert sec.depth == pytest.approx(0.5)
        assert sec.bottom_width == pytest.approx(0.0)
        assert sec.area == pytest.approx(1.0 ** 2 / (4 * 1.0))    # T²/(4z) = 0.25
        assert sec.area < 1.0 * 2.0 / 2                            # the old T·d/2

    def test_ordinary_section_is_untouched(self):
        from terrainflow_assessment.modules.swale_design import _channel_section
        sec = _channel_section(top_width=2.0, depth=0.5, side_slope=1.0)
        assert sec.depth == pytest.approx(0.5)
        assert sec.bottom_width == pytest.approx(1.0)

    def test_sizing_does_not_credit_the_impossible_depth(self):
        """A swale drawn past its convergence point must not read as longer-lasting."""
        converged = recommend_swale_length(100.0, depth=2.0, width=1.0, side_slope=1.0)
        at_convergence = recommend_swale_length(
            100.0, depth=0.5, width=1.0, side_slope=1.0)
        assert converged == pytest.approx(at_convergence)


class TestCapacityPerMetre:
    """The extracted capacity term, and the round trip that keeps it honest."""

    def test_recommend_swale_length_is_inflow_over_this(self):
        """One model, two questions. If these diverge the plugin answers 'does the
        swale hold its storm' two different ways depending on which button was pressed.
        """
        from terrainflow_assessment.modules.swale_design import (
            capacity_per_metre,
            recommend_swale_length,
        )

        kwargs = dict(side_slope=1.0, infiltration_mm_hr=12.0, duration_hr=24.0)
        cap = capacity_per_metre(0.5, 2.0, **kwargs)
        length = recommend_swale_length(120.0, 0.5, 2.0, **kwargs)
        assert length == pytest.approx(round(120.0 / cap, 1))

    def test_infiltration_adds_to_storage_over_the_event(self):
        from terrainflow_assessment.modules.swale_design import capacity_per_metre

        dry = capacity_per_metre(0.5, 2.0)
        wet = capacity_per_metre(0.5, 2.0, infiltration_mm_hr=20.0, duration_hr=24.0)
        assert wet > dry

    def test_degenerate_section_has_no_capacity(self):
        from terrainflow_assessment.modules.swale_design import capacity_per_metre

        assert capacity_per_metre(0.0, 2.0) == 0.0
        assert capacity_per_metre(0.5, 0.0) == 0.0

    def test_it_feeds_the_spacing_advisory(self):
        """The transposed question: how wide a strip does one metre of this hold?"""
        from terrainflow_assessment.core.sizing import spacing_advisory
        from terrainflow_assessment.modules.swale_design import capacity_per_metre

        cap = capacity_per_metre(0.5, 2.0)
        r = spacing_advisory(2.0, "Loam", runoff_mm=25.0, capacity_m3_per_m=cap)
        assert r["capture_spacing_m"] == pytest.approx(cap / 0.025)
