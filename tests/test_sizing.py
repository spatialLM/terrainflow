"""Tests for terrainflow_assessment/core/sizing — primitives + advisories.

Inline-asserted arithmetic (the spec formulas worked by hand), plus a regression
case per §1 fix: drawdown units, slope→0 guards, frustum negative-area guard, and
the advisory CN cross-check.
"""
import math

import pytest

from terrainflow_assessment.core.sizing import (
    basin_volume_battered,
    contour_spacing,
    drawdown_time,
    level_crest_from_spoil,
    manning_flow,
    pond_volume_frustum,
    prismatic_volume,
    trapezoid_section,
)
from terrainflow_assessment.core.sizing.advisories import (
    SOIL_MAX_GRADE_PCT,
    SOIL_MIN_BATTER,
    batter_advisory,
    cn_slope_crosscheck,
    grade_advisory,
    soil_from_cn,
)

# ---------------------------------------------------------------------------
# trapezoid_section — A, P, R, side slope, min_dimension, depth→0 guard
# ---------------------------------------------------------------------------

class TestTrapezoidSection:
    def test_basic_geometry(self):
        # top=3, bottom=1, depth=1 → z=(3-1)/2=1, A=(3+1)/2*1=2,
        # P=1+2*1*sqrt(2)=1+2.828..., R=A/P
        sec = trapezoid_section(3.0, 1.0, 1.0)
        assert sec.side_slope == pytest.approx(1.0)
        assert sec.area == pytest.approx(2.0)
        assert sec.wetted_perimeter == pytest.approx(1.0 + 2.0 * math.sqrt(2))
        assert sec.hydraulic_radius == pytest.approx(2.0 / (1.0 + 2.0 * math.sqrt(2)))

    def test_min_dimension_is_narrowest_width(self):
        sec = trapezoid_section(3.0, 1.0, 1.0)
        assert sec.min_dimension == pytest.approx(1.0)

    def test_rectangular_when_top_equals_bottom(self):
        # z=0, A=b*d, P=b+2d, R=A/P
        sec = trapezoid_section(2.0, 2.0, 0.5)
        assert sec.side_slope == 0.0
        assert sec.area == pytest.approx(1.0)
        assert sec.wetted_perimeter == pytest.approx(2.0 + 2.0 * 0.5)
        assert sec.hydraulic_radius == pytest.approx(1.0 / 3.0)

    def test_zero_depth_guard(self):
        # §1.3: depth→0 must not divide by d
        sec = trapezoid_section(3.0, 1.0, 0.0)
        assert sec.side_slope == 0.0
        assert sec.area == 0.0
        assert sec.wetted_perimeter == pytest.approx(1.0)
        assert sec.hydraulic_radius == 0.0

    def test_negative_depth_guard(self):
        sec = trapezoid_section(3.0, 1.0, -2.0)
        assert sec.area == 0.0
        assert sec.hydraulic_radius == 0.0

    def test_zero_wetted_perimeter_gives_zero_radius(self):
        # Degenerate: top=bottom=-2, depth=1 → P = -2 + 2*1*1 = 0 → R = 0 (else branch)
        sec = trapezoid_section(-2.0, -2.0, 1.0)
        assert sec.wetted_perimeter == pytest.approx(0.0)
        assert sec.hydraulic_radius == 0.0


# ---------------------------------------------------------------------------
# prismatic_volume
# ---------------------------------------------------------------------------

class TestPrismaticVolume:
    def test_volume_is_area_times_length(self):
        v = prismatic_volume(2.0, 50.0)
        assert v.volume == pytest.approx(100.0)
        assert v.section_area == 2.0
        assert v.length == 50.0

    def test_min_dimension_passthrough(self):
        v = prismatic_volume(2.0, 50.0, min_dimension=0.75)
        assert v.min_dimension == pytest.approx(0.75)

    def test_min_dimension_defaults_none(self):
        assert prismatic_volume(2.0, 50.0).min_dimension is None

    def test_negative_inputs_clamped_to_zero(self):
        assert prismatic_volume(-2.0, 50.0).volume == 0.0
        assert prismatic_volume(2.0, -50.0).volume == 0.0


# ---------------------------------------------------------------------------
# contour_spacing — terrace HI, slope→0 guard (§1.3)
# ---------------------------------------------------------------------------

class TestContourSpacing:
    def test_horizontal_interval(self):
        # VI=0.5, slope=0.05 (5%) → HI = 0.5 / 0.05 = 10 m
        r = contour_spacing(0.5, 0.05)
        assert r.spacing == pytest.approx(10.0)

    def test_flat_ground_infinite_spacing(self):
        # §1.3: slope→0 must not divide by zero
        assert contour_spacing(0.5, 0.0).spacing == math.inf

    def test_negative_slope_infinite_spacing(self):
        assert contour_spacing(0.5, -0.01).spacing == math.inf

    def test_min_dimension_not_applicable(self):
        assert contour_spacing(0.5, 0.05).min_dimension is None


# ---------------------------------------------------------------------------
# manning_flow — bankfull ceiling + velocity, slope→0 / empty-section guard
# ---------------------------------------------------------------------------

class TestManningFlow:
    def test_discharge_and_velocity(self):
        # A=2, R=0.5, s=0.01, n=0.025 → Q=(1/0.025)*2*0.5^(2/3)*0.1
        area, r, s, n = 2.0, 0.5, 0.01, 0.025
        expected_q = (1.0 / n) * area * (r ** (2.0 / 3.0)) * (s ** 0.5)
        res = manning_flow(area, r, s, n)
        assert res.discharge == pytest.approx(expected_q)
        assert res.velocity == pytest.approx(expected_q / area)

    def test_zero_slope_zero_flow(self):
        res = manning_flow(2.0, 0.5, 0.0, 0.025)
        assert res.discharge == 0.0
        assert res.velocity == 0.0

    def test_empty_section_zero_flow(self):
        assert manning_flow(0.0, 0.5, 0.01, 0.025).discharge == 0.0
        assert manning_flow(2.0, 0.0, 0.01, 0.025).discharge == 0.0

    def test_zero_n_zero_flow(self):
        assert manning_flow(2.0, 0.5, 0.01, 0.0).discharge == 0.0

    def test_min_dimension_passthrough(self):
        assert manning_flow(2.0, 0.5, 0.01, 0.025, min_dimension=0.4).min_dimension == 0.4


# ---------------------------------------------------------------------------
# pond_volume_frustum — prismoidal rule + 2zd<min(L,W) guard (§1.3)
# ---------------------------------------------------------------------------

class TestPondVolumeFrustum:
    def test_vertical_walls_is_box(self):
        # z=0 → all three areas equal → V = d * (L*W)
        r = pond_volume_frustum(10.0, 8.0, 2.0, 0.0)
        assert r.top_area == pytest.approx(80.0)
        assert r.mid_area == pytest.approx(80.0)
        assert r.bottom_area == pytest.approx(80.0)
        assert r.volume == pytest.approx(2.0 * 80.0)
        assert r.min_dimension == pytest.approx(8.0)

    def test_battered_walls_prismoidal(self):
        # L=10, W=8, d=2, z=1 → inset each side per depth = z*d = 2
        # mid: (10-2)*(8-2)=8*6=48 ; bottom: (10-4)*(8-4)=6*4=24 ; top=80
        # V = (2/6)*(80 + 4*48 + 24) = (1/3)*(296) = 98.666...
        r = pond_volume_frustum(10.0, 8.0, 2.0, 1.0)
        assert r.top_area == pytest.approx(80.0)
        assert r.mid_area == pytest.approx(48.0)
        assert r.bottom_area == pytest.approx(24.0)
        assert r.volume == pytest.approx((2.0 / 6.0) * (80.0 + 4.0 * 48.0 + 24.0))
        assert r.bottom_length == pytest.approx(6.0)
        assert r.bottom_width == pytest.approx(4.0)
        assert r.min_dimension == pytest.approx(4.0)

    def test_converging_walls_raise_before_area_computed(self):
        # §1.3 regression: 2zd >= min(L,W) → walls converge before depth → ValueError
        # L=4, W=4, d=2, z=1 → 2*1*2 = 4 >= 4
        with pytest.raises(ValueError, match="converge"):
            pond_volume_frustum(4.0, 4.0, 2.0, 1.0)

    def test_zero_depth_raises(self):
        with pytest.raises(ValueError, match="depth"):
            pond_volume_frustum(10.0, 8.0, 0.0, 1.0)

    def test_negative_side_slope_raises(self):
        with pytest.raises(ValueError, match="side slope"):
            pond_volume_frustum(10.0, 8.0, 2.0, -0.5)


# ---------------------------------------------------------------------------
# drawdown_time — §1.1 dimensional fix
# ---------------------------------------------------------------------------

class TestDrawdownTime:
    def test_storage_over_rate_times_area(self):
        # §1.1: t = storage / (rate * area). storage=100 m³, rate=0.004 m/hr, area=50 m²
        # t = 100 / (0.004*50) = 100 / 0.2 = 500 hr
        r = drawdown_time(100.0, 0.004, 50.0)
        assert r.time_hr == pytest.approx(500.0)

    def test_equivalent_to_depth_over_rate(self):
        # storage/area = depth, so t must equal depth/rate (the §1.1 identity)
        storage, rate, area = 100.0, 0.004, 50.0
        depth = storage / area  # 2 m
        r = drawdown_time(storage, rate, area)
        assert r.time_hr == pytest.approx(depth / rate)

    def test_zero_rate_infinite(self):
        assert drawdown_time(100.0, 0.0, 50.0).time_hr == math.inf

    def test_zero_area_infinite(self):
        assert drawdown_time(100.0, 0.004, 0.0).time_hr == math.inf

    def test_min_dimension_not_applicable(self):
        assert drawdown_time(100.0, 0.004, 50.0).min_dimension is None


# ---------------------------------------------------------------------------
# advisories — soil batter/grade + CN cross-check (§1.2, advisory not clamp)
# ---------------------------------------------------------------------------

class TestBatterAdvisory:
    def test_within_envelope_flatter_batter(self):
        # Clay min batter 1.0 → a 1.5:1 batter is flatter → within
        within, text = batter_advisory("Clay", 1.5)
        assert within is True
        assert "stable" in text

    def test_out_of_envelope_steeper_batter(self):
        # Sand min batter 2.0 → a 1.0:1 batter is steeper → flagged, not clamped
        within, text = batter_advisory("Sand", 1.0)
        assert within is False
        assert "steeper" in text
        assert "Advisory" in text  # never a silent clamp

    def test_unknown_soil_defaults_to_loam(self):
        within_default, _ = batter_advisory(None, SOIL_MIN_BATTER["Loam"])
        assert within_default is True

    def test_exactly_at_minimum_is_within(self):
        within, _ = batter_advisory("Loam", SOIL_MIN_BATTER["Loam"])
        assert within is True


class TestGradeAdvisory:
    def test_within_envelope(self):
        within, text = grade_advisory("Loam", 1.0)
        assert within is True
        assert "low scour" in text

    def test_out_of_envelope(self):
        # Sand max grade 0.5% → 2% exceeds → flagged
        within, text = grade_advisory("Sand", 2.0)
        assert within is False
        assert "exceeds" in text
        assert "Advisory" in text

    def test_exactly_at_maximum_is_within(self):
        within, _ = grade_advisory("Clay", SOIL_MAX_GRADE_PCT["Clay"])
        assert within is True


class TestSoilFromCn:
    def test_low_cn_maps_to_sand(self):
        assert soil_from_cn(39) == "Sand"

    def test_high_cn_maps_to_clay(self):
        assert soil_from_cn(80) == "Clay"

    def test_intermediate_cn_maps_to_nearest(self):
        # 60 is nearest to Loam (61)
        assert soil_from_cn(60) == "Loam"


class TestCnSlopeCrosscheck:
    def test_crosscheck_flags_steep_batter_for_sandy_cn(self):
        # CN 39 → Sand (min batter 2.0); a 1.0:1 batter is too steep → flagged
        within, text = cn_slope_crosscheck(39, 1.0)
        assert within is False
        assert "Sand" in text
        assert "CN 39" in text

    def test_crosscheck_passes_flat_batter_for_clay_cn(self):
        # CN 80 → Clay (min batter 1.0); a 1.5:1 batter is fine → within
        within, text = cn_slope_crosscheck(80, 1.5)
        assert within is True
        assert "Clay" in text


# ---------------------------------------------------------------------------
# basin_volume_battered — inset-prism model for arbitrary footprints
# ---------------------------------------------------------------------------

class TestBasinVolumeBattered:
    def test_vertical_walls_exact_prism(self):
        r = basin_volume_battered(100.0, 40.0, 2.0, 0.0)
        assert r.volume == pytest.approx(200.0)
        assert r.bottom_area == pytest.approx(100.0)
        assert r.effective_depth == pytest.approx(2.0)

    def test_battered_square_conservative(self):
        # Square 10×10, z=1, d=1: inset-prism gives 80.0 (exact frustum 81.33).
        r = basin_volume_battered(100.0, 40.0, 1.0, 1.0)
        assert r.volume == pytest.approx(80.0)
        exact = pond_volume_frustum(10.0, 10.0, 1.0, 1.0).volume
        assert r.volume < exact  # conservative underestimate

    def test_convergence_clamps(self):
        # A=4, P=8, z=1: walls meet at t* = 4/8 = 0.5 m; V = 16/16 = 1.0.
        r = basin_volume_battered(4.0, 8.0, 5.0, 1.0)
        assert r.volume == pytest.approx(1.0)
        assert r.effective_depth == pytest.approx(0.5)
        assert r.bottom_area == 0.0

    def test_zero_depth_degenerate(self):
        r = basin_volume_battered(100.0, 40.0, 0.0, 1.0)
        assert r.volume == 0.0
        assert r.effective_depth == 0.0

    def test_zero_area_degenerate(self):
        assert basin_volume_battered(0.0, 0.0, 1.0, 1.0).volume == 0.0

    def test_negative_slope_raises(self):
        with pytest.raises(ValueError):
            basin_volume_battered(100.0, 40.0, 1.0, -0.5)

    def test_volume_decreases_with_batter(self):
        vols = [basin_volume_battered(100.0, 40.0, 1.5, z).volume
                for z in (0.0, 0.5, 1.0, 2.0)]
        assert vols == sorted(vols, reverse=True)
        assert vols[0] > vols[-1]

    def test_effective_depth_full_when_not_converging(self):
        r = basin_volume_battered(1000.0, 130.0, 2.0, 1.0)
        assert r.effective_depth == pytest.approx(2.0)
        assert r.min_dimension is None


class TestLevelCrestFromSpoil:
    """A berm has to be LEVEL to impound, and it is built from the trench's own spoil.

    Raising every cell by one height puts the crest on the slope of the ground under it,
    so the water leaves at the low end — which is what the burn did before, and why a
    companion berm could be credited with storage it could not hold.
    """

    def test_flat_ground_spreads_evenly(self):
        # 10 cells of 1 m² at 50.0, 20 m³ of spoil → 2 m over the lot.
        assert level_crest_from_spoil([50.0] * 10, 1.0, 20.0) == pytest.approx(52.0)

    def test_sloping_ground_fills_the_low_end_first(self):
        # Ground 0..9, and exactly enough spoil to bring the lowest five up to 5.0:
        # (5-0)+(5-1)+(5-2)+(5-3)+(5-4) = 15 m³.
        assert level_crest_from_spoil(range(10), 1.0, 15.0) == pytest.approx(5.0)

    def test_more_spoil_than_the_band_holds_keeps_going_above_it(self):
        # Filling 0..9 to level 9 takes 45 m³; another 10 m³ over 10 cells adds 1 m.
        assert level_crest_from_spoil(range(10), 1.0, 55.0) == pytest.approx(10.0)

    def test_cell_area_scales_the_answer(self):
        assert level_crest_from_spoil([50.0], 2.0, 10.0) == pytest.approx(55.0)

    def test_the_crest_reproduces_the_spoil_volume(self):
        """The property that matters: fill under the crest is the earth that was dug."""
        import random
        rng = random.Random(3)
        ground = [rng.uniform(40.0, 45.0) for _ in range(500)]
        for spoil in (5.0, 100.0, 900.0):
            crest = level_crest_from_spoil(ground, 1.0, spoil)
            rebuilt = sum(max(0.0, crest - g) for g in ground)
            assert rebuilt == pytest.approx(spoil, abs=1e-6)

    def test_no_spoil_builds_nothing(self):
        assert level_crest_from_spoil([50.0] * 5, 1.0, 0.0) is None

    def test_nowhere_to_build_returns_none(self):
        assert level_crest_from_spoil([], 1.0, 10.0) is None

    def test_nan_ground_is_ignored_rather_than_poisoning_the_sort(self):
        import math
        assert level_crest_from_spoil(
            [50.0, math.nan, 50.0], 1.0, 4.0) == pytest.approx(52.0)


# ---------------------------------------------------------------------------
# Spacing advisory — the terrace rule, the capture rule, and which governs
# ---------------------------------------------------------------------------

class TestTerraceVerticalInterval:
    def test_matches_the_published_form_by_hand(self):
        """VI(ft) = X*S% + Y, then feet to metres. Nothing else."""
        from terrainflow_assessment.core.sizing import terrace_vertical_interval

        # X=0.6, Y=2.0, S=10% -> 0.6*10 + 2.0 = 8.0 ft -> 2.4384 m
        got = terrace_vertical_interval(10.0, x=0.6, y=2.0)
        assert got == pytest.approx(8.0 * 0.3048)

    def test_rises_with_slope(self):
        from terrainflow_assessment.core.sizing import terrace_vertical_interval

        assert (terrace_vertical_interval(20.0, "Loam")
                > terrace_vertical_interval(5.0, "Loam"))

    def test_erodible_soils_get_the_closer_spacing(self):
        """Sand carries the lowest Y, so its interval is the tightest."""
        from terrainflow_assessment.core.sizing import terrace_vertical_interval

        assert (terrace_vertical_interval(10.0, "Sand")
                < terrace_vertical_interval(10.0, "Clay"))

    def test_level_ground_returns_the_constant_not_zero(self):
        from terrainflow_assessment.core.sizing import terrace_vertical_interval

        assert terrace_vertical_interval(0.0, x=0.6, y=2.0) == pytest.approx(
            2.0 * 0.3048)


class TestCaptureSpacing:
    def test_is_capacity_over_runoff_depth(self):
        from terrainflow_assessment.core.sizing import capture_spacing

        # 0.5 m3 per metre of swale, 25 mm of runoff -> 0.5 / 0.025 = 20 m strip
        assert capture_spacing(25.0, 0.5) == pytest.approx(20.0)

    def test_no_runoff_means_no_limit(self):
        from terrainflow_assessment.core.sizing import capture_spacing

        assert capture_spacing(0.0, 0.5) == math.inf


class TestSpacingAdvisory:
    def test_the_smaller_rule_governs(self):
        """Two different failure modes, and neither excuses the other."""
        from terrainflow_assessment.core.sizing import spacing_advisory

        # Steep ground: the terrace rule bites first.
        steep = spacing_advisory(25.0, "Loam", runoff_mm=10.0,
                                 capacity_m3_per_m=5.0)
        assert steep["governing"] == "erosion"
        assert steep["recommended_spacing_m"] == pytest.approx(
            steep["erosion_spacing_m"])

        # Gentle ground with a small section: capture bites first.
        gentle = spacing_advisory(1.0, "Loam", runoff_mm=40.0,
                                  capacity_m3_per_m=0.2)
        assert gentle["governing"] == "capture"
        assert gentle["recommended_spacing_m"] == pytest.approx(
            gentle["capture_spacing_m"])

    def test_the_recommendation_never_exceeds_either_rule(self):
        from terrainflow_assessment.core.sizing import spacing_advisory

        for slope in (1.0, 5.0, 12.0, 30.0):
            r = spacing_advisory(slope, "Loam", runoff_mm=25.0,
                                 capacity_m3_per_m=0.6)
            assert r["recommended_spacing_m"] <= r["erosion_spacing_m"] + 1e-9
            assert r["recommended_spacing_m"] <= r["capture_spacing_m"] + 1e-9

    def test_flat_ground_reports_rather_than_returning_infinity(self):
        """`inf` in a spin box is how a recommendation becomes a bug report."""
        from terrainflow_assessment.core.sizing import spacing_advisory

        r = spacing_advisory(0.0, "Loam")
        assert r["erosion_spacing_m"] == math.inf
        assert r["recommended_spacing_m"] is None
        assert r["governing"] == "none"
        assert "level" in r["text"]

    def test_without_a_storm_only_the_erosion_rule_answers(self):
        from terrainflow_assessment.core.sizing import spacing_advisory

        r = spacing_advisory(10.0, "Loam")
        assert r["capture_spacing_m"] is None
        assert r["governing"] == "erosion"

    def test_the_text_names_its_own_basis(self):
        """An advisory that does not say where its number came from is a rumour."""
        from terrainflow_assessment.core.sizing import spacing_advisory

        r = spacing_advisory(10.0, "Clay loam", runoff_mm=25.0,
                             capacity_m3_per_m=0.6)
        assert "Clay loam" in r["text"]
        assert r["governing"] in r["text"] or "governs" in r["text"]

    def test_it_goes_through_contour_spacing(self):
        """The primitive had no production caller until this advisory."""
        from terrainflow_assessment.core.sizing import (
            contour_spacing,
            spacing_advisory,
        )

        r = spacing_advisory(8.0, "Loam")
        expected = contour_spacing(r["vertical_interval_m"], 0.08).spacing
        assert r["erosion_spacing_m"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# bench_geometry — FAO Conservation Guide 13/3 §6.1, Table 1 as the oracle
# ---------------------------------------------------------------------------

class TestBenchGeometry:
    """FAO's Table 1 (hand-made benches, riser 0.75:1, reverse 5 %), reproduced to the
    table's own two decimals by rounding every step half-up before the next."""

    def test_fao_table_1_the_3m_bench_at_30_pct(self):
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        b = bench_geometry(3.00, 30.0, riser_slope=0.75, mode="reverse")
        assert b.vertical_interval == 1.16
        assert b.edge_rise == 0.15
        assert b.riser_height == 1.31
        # The table's 0.53. Float arithmetic gives 0.9 + 0.15 = 1.0499… and 0.52.
        assert b.depth_of_cut == 0.53
        assert b.riser_width == 0.98
        assert b.terrace_width == 3.98
        assert b.length_per_ha == 2513
        assert b.cut_section == 0.49
        assert b.volume_per_ha == 1231

    def test_fao_table_1_the_4m_bench_at_24_pct(self):
        """Every figure but one. The table prints W_t 5.03 and L 1989, and 10 000 / 5.03
        is 1988.07: FAO carried an unrounded riser width into that one division and a
        rounded one into the 3 m row's. No single convention gives both, so the
        convention is the stated one and the table's slip is recorded here rather than
        reproduced."""
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        b = bench_geometry(4.00, 24.0, riser_slope=0.75, mode="reverse")
        assert b.vertical_interval == 1.17
        assert b.edge_rise == 0.20
        assert b.riser_height == 1.37
        assert b.depth_of_cut == 0.58
        # 1.0275 on paper rounds up. As a float it is 1.02749999… and would round down.
        assert b.riser_width == 1.03
        assert b.terrace_width == 5.03
        assert b.length_per_ha == 1988          # FAO prints 1989
        assert b.cut_section == 0.69
        assert b.volume_per_ha == 1372

    def test_step_rounding_is_what_makes_the_table_reconcile(self):
        """Carry the chain unrounded and the 3 m row gives 2510 m/ha, not 2513."""
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        vi = 30.0 * 3.00 / (100.0 - 30.0 * 0.75)
        w_t = 3.00 + (vi + 0.15) * 0.75
        assert round(10000.0 / w_t) == 2510
        assert bench_geometry(3.00, 30.0, riser_slope=0.75).length_per_ha == 2513

    def test_the_level_bench_with_a_dyke_from_the_spec_worked_example(self):
        """25 % ground, 3.0 m bench, hand-made riser, 0.20 m dyke."""
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        b = bench_geometry(3.0, 25.0, riser_slope=0.75, mode="level", dyke_height=0.20)
        assert b.vertical_interval == 0.92
        assert b.edge_rise == 0.0
        assert b.riser_height == 1.12
        assert b.riser_width == 0.84
        assert b.terrace_width == 3.84
        assert b.length_per_ha == 2604
        # The dyke is fill on the finished platform: it raises the riser and takes no
        # part in the cut balance, so the cut at the inner edge is half the fall alone.
        assert b.depth_of_cut == 0.38            # (0.75 + 0) / 2 = 0.375, half-up

    def test_the_machine_built_level_bench_the_registry_uses(self):
        """Hand-worked: 4.0 m bench, 25 %, 1:1 riser, 0.20 m dyke. VI = 100/75 = 1.33,
        H_r = 1.53, W_r = 1.53, W_t = 5.53, L = 1808, C = 0.765 → 0.77, V = 1392,
        D_c = 1.00 / 2 = 0.50 (the dyke is not cut)."""
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        b = bench_geometry(4.0, 25.0, riser_slope=1.0, mode="level", dyke_height=0.20)
        assert b.vertical_interval == 1.33
        assert b.riser_height == 1.53
        assert b.riser_width == 1.53
        assert b.terrace_width == 5.53
        assert b.length_per_ha == 1808
        assert b.cut_section == 0.77
        assert b.volume_per_ha == 1392
        assert b.depth_of_cut == 0.50

    def test_an_outward_bench_drops_over_its_riser(self):
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        b = bench_geometry(4.00, 24.0, riser_slope=0.75, mode="outward")
        assert b.edge_rise == -0.12
        assert b.riser_height == 1.05            # 1.17 − 0.12
        assert b.depth_of_cut == 0.42            # (0.96 − 0.12) / 2

    def test_a_dyke_raises_the_riser_and_not_the_cut(self):
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        bare = bench_geometry(4.0, 20.0, mode="level")
        dyked = bench_geometry(4.0, 20.0, mode="level", dyke_height=0.25)
        assert dyked.riser_height == pytest.approx(bare.riser_height + 0.25)
        assert dyked.depth_of_cut == bare.depth_of_cut

    def test_flat_ground_has_no_vertical_interval(self):
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        b = bench_geometry(4.0, 0.0, mode="reverse")
        assert b.vertical_interval == 0.0
        assert b.riser_height == 0.20            # the 5 % rise across the bench alone
        assert b.terrace_width == 4.20

    def test_the_bench_width_is_the_min_dimension(self):
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        assert bench_geometry(2.5, 15.0).min_dimension == 2.5

    def test_ground_as_steep_as_the_riser_is_refused(self):
        """S·U ≥ 100: the next bench's riser toe lands above this bench's top."""
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        with pytest.raises(ValueError, match="riser"):
            bench_geometry(4.0, 100.0, riser_slope=1.0)
        with pytest.raises(ValueError, match="riser"):
            bench_geometry(4.0, 140.0, riser_slope=0.75)
        assert bench_geometry(4.0, 99.0, riser_slope=1.0).vertical_interval == 396.0

    def test_bad_inputs_are_refused_by_name(self):
        from terrainflow_assessment.core.sizing.bench import bench_geometry

        with pytest.raises(ValueError, match="width"):
            bench_geometry(0.0, 20.0)
        with pytest.raises(ValueError, match="mode"):
            bench_geometry(4.0, 20.0, mode="sideways")
        with pytest.raises(ValueError, match="dyke"):
            bench_geometry(4.0, 20.0, mode="level", dyke_height=-0.1)

    def test_the_package_re_exports_it(self):
        from terrainflow_assessment.core.sizing import BenchResult, bench_geometry

        assert isinstance(bench_geometry(3.0, 10.0), BenchResult)


# ---------------------------------------------------------------------------
# bench_spacing_advisory — the layout a bench section supports
# ---------------------------------------------------------------------------

class TestBenchSpacingAdvisory:
    """The spec's worked example: 25 % ground, a 3.0 m bench with a 0.20 m dyke on a
    hand-made riser, a section holding 0.405 m³/m against 58.6 mm of runoff — a 6.91 m
    strip. One bench in every two terrace widths (7.68 m) is supported; one in every
    three (needs 7.68 m of strip) is not."""

    def _example(self, **kw):
        from terrainflow_assessment.core.sizing import bench_spacing_advisory

        args = dict(riser_slope=0.75, mode="level", dyke_height=0.20,
                    runoff_mm=58.6, capacity_m3_per_m=0.405)
        args.update(kw)
        return bench_spacing_advisory(25.0, 3.0, **args)

    def test_the_geometry_comes_from_the_fao_chain(self):
        r = self._example()
        assert r["vertical_interval_m"] == 0.92
        assert r["terrace_width_m"] == 3.84

    def test_the_layout_is_the_widest_the_strip_supports(self):
        r = self._example()
        assert r["capture_spacing_m"] == pytest.approx(6.91, abs=0.01)
        assert r["layout_every"] == 2
        assert r["recommended_spacing_m"] == pytest.approx(7.68)
        assert r["governing"] == "capture"

    def test_a_section_holding_more_supports_a_sparser_layout(self):
        r = self._example(capacity_m3_per_m=1.00)      # a 17.1 m strip
        assert r["layout_every"] == 5                  # 17.1 / 3.84 = 4.4 → 4, +1
        assert r["recommended_spacing_m"] == pytest.approx(5 * 3.84)

    def test_without_a_storm_it_assumes_continuous_benching(self):
        r = self._example(runoff_mm=None, capacity_m3_per_m=None)
        assert r["capture_spacing_m"] is None
        assert r["layout_every"] == 1
        assert r["recommended_spacing_m"] == pytest.approx(3.84)
        assert r["governing"] == "none"

    def test_no_runoff_at_all_is_not_a_division_by_zero(self):
        r = self._example(runoff_mm=0.0)
        assert r["layout_every"] == 1
        assert r["governing"] == "none"

    def test_ground_the_bench_cannot_be_cut_into_is_said_not_raised(self):
        from terrainflow_assessment.core.sizing import bench_spacing_advisory

        r = bench_spacing_advisory(120.0, 4.0, riser_slope=1.0)
        assert r["vertical_interval_m"] is None
        assert r["terrace_width_m"] is None
        assert r["governing"] == "none"
        assert "riser" in r["text"]

    def test_the_text_names_its_own_basis(self):
        r = self._example()
        assert "FAO" in r["text"]
        assert "one bench in every 2" in r["text"]
        assert "3.84" in r["text"]

    def test_there_is_no_erosion_arm(self):
        """A bench system replaces the natural slope; the NRCS strip rule does not
        apply between benches and must not be reported as if it did."""
        r = self._example()
        assert "erosion_spacing_m" not in r
