"""TR-55 segmental travel time.

Time of concentration decides which duration to read an IDF curve at, and therefore
the design intensity, and therefore every spillway width. The formulae are published
in English units, so the highest-value tests here check the metric conversions against
the original coefficients rather than against a restatement of my own arithmetic.

Reference: USDA-NRCS, TR-55 *Urban Hydrology for Small Watersheds*, 2nd ed. 1986, ch. 3.
"""

import pytest

from terrainflow_assessment.modules.time_of_concentration import (
    CHANNEL_RADII,
    DEFAULT_CHANNEL_RADIUS_M,
    DEFAULT_CHANNEL_ROUGHNESS,
    DEFAULT_SHEET_ROUGHNESS,
    EARTHWORK_CHANNEL_ROUGHNESS,
    MAX_SHEET_FLOW_M,
    MIN_VALID_TC_HR,
    SHEET_ROUGHNESS,
    channel_flow_hours,
    channel_length_from_area,
    mixed_channel_hours,
    profile_leg_slopes,
    section_hydraulic_radius,
    shallow_concentrated_hours,
    sheet_flow_hours,
    split_flow_path,
    time_of_concentration,
)


class TestAgainstThePublishedEnglishForms:
    """The conversions, checked against TR-55's own coefficients."""

    @pytest.mark.parametrize("n,L_m,P2_mm,s", [
        (0.24, 30.0, 75.0, 0.05),
        (0.011, 15.0, 120.0, 0.02),
        (0.80, 30.48, 180.0, 0.15),
        (0.15, 5.0, 40.0, 0.005),
    ])
    def test_sheet_flow_matches_tr55_equation_3_3(self, n, L_m, P2_mm, s):
        """Tt = 0.007 (nL)^0.8 / (P2^0.5 s^0.4), L in ft, P2 in inches, Tt in hours."""
        L_ft, P2_in = L_m / 0.3048, P2_mm / 25.4
        english = 0.007 * (n * L_ft) ** 0.8 / (P2_in ** 0.5 * s ** 0.4)
        assert sheet_flow_hours(L_m, s, n, P2_mm) == pytest.approx(english, rel=1e-12)

    @pytest.mark.parametrize("s", [0.005, 0.02, 0.08, 0.25])
    def test_shallow_unpaved_matches_tr55_velocity(self, s):
        """V = 16.1345 √s ft/s unpaved."""
        v_ms = 16.1345 * (s ** 0.5) * 0.3048
        assert shallow_concentrated_hours(100.0, s) == pytest.approx(
            100.0 / (v_ms * 3600.0), rel=1e-12)

    @pytest.mark.parametrize("s", [0.005, 0.02, 0.08])
    def test_shallow_paved_matches_tr55_velocity(self, s):
        """V = 20.3282 √s ft/s paved."""
        v_ms = 20.3282 * (s ** 0.5) * 0.3048
        assert shallow_concentrated_hours(100.0, s, paved=True) == pytest.approx(
            100.0 / (v_ms * 3600.0), rel=1e-12)

    def test_paved_is_faster_than_unpaved(self):
        assert (shallow_concentrated_hours(100.0, 0.05, paved=True)
                < shallow_concentrated_hours(100.0, 0.05, paved=False))

    def test_channel_flow_is_mannings_equation(self):
        v = ((1.0 / DEFAULT_CHANNEL_ROUGHNESS)
             * (DEFAULT_CHANNEL_RADIUS_M ** (2.0 / 3.0)) * (0.02 ** 0.5))
        assert channel_flow_hours(200.0, 0.02) == pytest.approx(
            200.0 / (v * 3600.0), rel=1e-12)


class TestBehaviour:
    def test_steeper_ground_is_faster(self):
        assert (sheet_flow_hours(30.0, 0.10, 0.24, 75.0)
                < sheet_flow_hours(30.0, 0.01, 0.24, 75.0))

    def test_rougher_cover_is_slower(self):
        smooth = sheet_flow_hours(30.0, 0.05, SHEET_ROUGHNESS["Smooth surfaces "
                                                              "(concrete, asphalt, bare soil)"], 75.0)
        woods = sheet_flow_hours(30.0, 0.05, SHEET_ROUGHNESS["Woods, dense underbrush"], 75.0)
        assert woods > smooth

    def test_heavier_rain_moves_sheet_flow_faster(self):
        """Sheet-flow depth rises with rainfall, and deeper sheet flow is quicker —
        which is why P2 appears in the equation at all."""
        assert (sheet_flow_hours(30.0, 0.05, 0.24, 150.0)
                < sheet_flow_hours(30.0, 0.05, 0.24, 75.0))

    def test_sheet_flow_is_capped_at_the_tr55_limit(self):
        """Past 30.5 m runoff has concentrated into rills and the equation no longer
        describes it, so extra length must not keep accruing sheet-flow time."""
        assert (sheet_flow_hours(500.0, 0.05, 0.24, 75.0)
                == pytest.approx(sheet_flow_hours(MAX_SHEET_FLOW_M, 0.05, 0.24, 75.0)))

    def test_flat_ground_does_not_divide_by_zero(self):
        assert sheet_flow_hours(30.0, 0.0, 0.24, 75.0) > 0
        assert shallow_concentrated_hours(100.0, 0.0) > 0
        assert channel_flow_hours(100.0, 0.0) > 0

    @pytest.mark.parametrize("fn", [
        lambda: sheet_flow_hours(0.0, 0.05, 0.24, 75.0),
        lambda: shallow_concentrated_hours(0.0, 0.05),
        lambda: channel_flow_hours(0.0, 0.05),
    ])
    def test_a_zero_length_leg_takes_no_time(self, fn):
        assert fn() == 0.0

    def test_sheet_flow_without_p2_is_skipped_not_guessed(self):
        assert sheet_flow_hours(30.0, 0.05, 0.24, None) == 0.0


class TestSplitFlowPath:
    def test_sheet_takes_the_first_thirty_metres(self):
        sheet, shallow, channel = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        assert sheet[0] == pytest.approx(MAX_SHEET_FLOW_M)
        assert shallow[0] == pytest.approx(350.0 - 120.0 - MAX_SHEET_FLOW_M)
        assert channel[0] == pytest.approx(120.0)

    def test_the_legs_sum_to_the_whole_path(self):
        sheet, shallow, channel = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        assert sheet[0] + shallow[0] + channel[0] == pytest.approx(350.0)

    def test_a_short_path_is_all_sheet_flow(self):
        sheet, shallow, channel = split_flow_path(20.0, 0.05)
        assert sheet[0] == pytest.approx(20.0)
        assert shallow[0] == 0.0
        assert channel[0] == 0.0

    def test_a_channel_longer_than_the_path_is_clamped(self):
        sheet, shallow, channel = split_flow_path(50.0, 0.05, channel_length_m=900.0)
        assert channel[0] == pytest.approx(50.0)
        assert sheet[0] == 0.0 and shallow[0] == 0.0

    def test_a_zero_length_path_produces_zero_legs(self):
        assert all(leg[0] == 0.0 for leg in split_flow_path(0.0, 0.05))


def _concave_profile(length=350.0, drop=28.0, n=400):
    """A realistic hillslope: steep off the ridge, flattening into the valley."""
    distances = [length * i / (n - 1) for i in range(n)]
    elevations = [drop * (1.0 - (d / length) ** 0.5) for d in distances]
    return distances, elevations


class TestPerLegSlopes:
    """One average slope over a concave hillslope is not a harmless simplification:
    it is far too gentle where sheet flow happens, which lengthens Tc, lowers the
    design intensity, and undersizes the overflow."""

    def test_a_concave_slope_is_steepest_at_the_top(self):
        sheet, shallow, channel = profile_leg_slopes(
            *_concave_profile(), channel_length_m=120.0)
        assert sheet > shallow > channel

    def test_the_sheet_leg_is_much_steeper_than_the_average(self):
        distances, elevations = _concave_profile()
        average = (elevations[0] - elevations[-1]) / distances[-1]
        sheet, _sh, _ch = profile_leg_slopes(distances, elevations, channel_length_m=120.0)
        assert sheet > average * 2

    def test_per_leg_slopes_shorten_tc_on_a_concave_profile(self):
        """The whole reason for tracing the path — and the error is in the unsafe
        direction, so it cannot be dismissed as conservative."""
        distances, elevations = _concave_profile()
        average = (elevations[0] - elevations[-1]) / distances[-1]
        one = time_of_concentration(
            *split_flow_path(350.0, average, channel_length_m=120.0), p2_mm=75.0)
        per = time_of_concentration(
            *split_flow_path(350.0,
                             profile_leg_slopes(distances, elevations, 120.0),
                             channel_length_m=120.0), p2_mm=75.0)
        assert per.total_hr < one.total_hr
        assert per.total_hr / one.total_hr < 0.85      # ~25% shorter

    def test_a_uniform_slope_gives_the_same_answer_either_way(self):
        """A planar hillside has nothing for per-leg slopes to discover, and must not
        drift from the single-average result."""
        distances = [0.0, 100.0, 200.0, 350.0]
        elevations = [28.0, 20.0, 12.0, 0.0]           # constant 8%
        legs = profile_leg_slopes(distances, elevations, channel_length_m=120.0)
        assert all(s == pytest.approx(0.08, abs=1e-9) for s in legs)

    def test_a_flat_profile_has_no_slope(self):
        assert profile_leg_slopes([0.0, 100.0], [10.0, 10.0]) == (0.0, 0.0, 0.0)

    def test_split_flow_path_accepts_a_slope_triple(self):
        sheet, shallow, channel = split_flow_path(
            350.0, (0.27, 0.07, 0.04), channel_length_m=120.0)
        assert sheet[1] == pytest.approx(0.27)
        assert shallow[1] == pytest.approx(0.07)
        assert channel[1] == pytest.approx(0.04)

    def test_split_flow_path_still_accepts_a_scalar(self):
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        assert all(leg[1] == pytest.approx(0.08) for leg in legs)

    @pytest.mark.parametrize("d,z", [
        (None, None), ([], []), ([0.0], [1.0]), ([0.0, 1.0], [1.0]),
    ])
    def test_degenerate_profiles_give_no_slope(self, d, z):
        assert profile_leg_slopes(d, z) == (0.0, 0.0, 0.0)

    def test_a_zero_length_profile_gives_no_slope(self):
        assert profile_leg_slopes([0.0, 0.0], [5.0, 5.0]) == (0.0, 0.0, 0.0)


class TestChannelDetection:
    """Channel length is measured, not assumed — and zero is not the safe default,
    since channel flow is about twice the speed of shallow concentrated flow."""

    def test_the_channel_starts_where_area_crosses_the_threshold(self):
        distances = [0.0, 50.0, 100.0, 150.0, 200.0]
        areas = [500.0, 3_000.0, 8_000.0, 12_000.0, 20_000.0]     # m²
        assert channel_length_from_area(distances, areas, threshold_ha=1.0) == \
            pytest.approx(50.0)                                   # crosses at 150 m

    def test_a_small_catchment_that_never_concentrates_has_no_channel(self):
        distances = [0.0, 50.0, 100.0]
        areas = [200.0, 1_500.0, 4_000.0]
        assert channel_length_from_area(distances, areas, threshold_ha=1.0) == 0.0

    def test_a_lower_threshold_finds_a_longer_channel(self):
        distances = [0.0, 50.0, 100.0, 150.0, 200.0]
        areas = [500.0, 3_000.0, 8_000.0, 12_000.0, 20_000.0]
        loose = channel_length_from_area(distances, areas, threshold_ha=0.5)
        tight = channel_length_from_area(distances, areas, threshold_ha=1.0)
        assert loose > tight

    def test_a_channel_shortens_tc(self):
        """Which raises the design intensity — so measuring it rather than leaving it
        at zero moves the spillway in the safe direction."""
        with_channel = time_of_concentration(
            *split_flow_path(350.0, 0.08, channel_length_m=120.0), p2_mm=75.0)
        without = time_of_concentration(
            *split_flow_path(350.0, 0.08, channel_length_m=0.0), p2_mm=75.0)
        assert with_channel.total_hr < without.total_hr

    @pytest.mark.parametrize("d,a", [
        (None, None), ([], []), ([0.0, 1.0], [1.0]),
    ])
    def test_missing_data_means_no_channel(self, d, a):
        assert channel_length_from_area(d, a) == 0.0


class TestSectionHydraulicRadius:
    """Where the channel is one of the user's own drains, nothing needs estimating —
    the trapezoid was entered exactly."""

    def test_matches_the_shared_trapezoid_primitive(self):
        from terrainflow_assessment.core.sizing import trapezoid_section
        assert section_hydraulic_radius(2.0, 1.0, 0.5) == pytest.approx(
            trapezoid_section(2.0, 1.0, 0.5).hydraulic_radius)

    def test_a_large_swale_is_far_above_the_default_preset(self):
        """4 m x 1 m gives R = 0.62 against the 0.211 preset — nearly 3x, so guessing
        instead of reading the entered dimensions is a real error."""
        assert section_hydraulic_radius(4.0, 2.0, 1.0) == pytest.approx(0.621, abs=0.001)

    def test_a_shallow_swale_is_below_the_default_preset(self):
        assert section_hydraulic_radius(0.8, 0.4, 0.2) < DEFAULT_CHANNEL_RADIUS_M

    def test_a_deeper_channel_conveys_more_efficiently(self):
        assert (section_hydraulic_radius(2.0, 1.0, 1.0)
                > section_hydraulic_radius(2.0, 1.0, 0.3))

    def test_it_is_the_bankfull_radius(self):
        """Documented assumption: the section running full. Real flow is shallower,
        so this overstates velocity, shortens Tc, and oversizes the overflow — the
        safe direction, and consistent with how diversion capacity is computed."""
        from terrainflow_assessment.core.sizing import trapezoid_section
        full = section_hydraulic_radius(2.0, 1.0, 0.5)
        half_depth = trapezoid_section(1.5, 1.0, 0.25).hydraulic_radius
        assert full > half_depth

    @pytest.mark.parametrize("args", [
        (None, 1.0, 0.5), (2.0, None, 0.5), (2.0, 1.0, None), ("x", 1.0, 0.5),
    ])
    def test_missing_or_bad_dimensions_give_nothing(self, args):
        assert section_hydraulic_radius(*args) is None

    def test_zero_depth_gives_nothing(self):
        assert section_hydraulic_radius(2.0, 1.0, 0.0) is None


class TestMixedChannelPaths:
    """A flow path is rarely one channel: natural gully, then the drain you dug, then
    a swale. Averaging that away discards dimensions that were entered exactly."""

    def test_a_uniform_channel_matches_the_single_leg_calculation(self):
        d = [0.0, 50.0, 100.0, 150.0]
        radii = [0.211] * 4
        hours, runs = mixed_channel_hours(d, radii, slope=0.05)
        assert len(runs) == 1
        assert hours == pytest.approx(channel_flow_hours(150.0, 0.05, 0.211,
                                                         DEFAULT_CHANNEL_ROUGHNESS))

    def test_a_change_of_section_splits_the_leg(self):
        """A section belongs to the segment between two points, taken from the
        downstream one — you are in the drain once you reach it. So the transition
        segment (50-100 m) counts as drain, not as natural ground."""
        d = [0.0, 50.0, 100.0, 150.0]
        radii = [0.211, 0.211, 0.621, 0.621]
        _hours, runs = mixed_channel_hours(d, radii, slope=0.05)
        assert len(runs) == 2
        assert runs[0][0] == pytest.approx(50.0)       # natural ground
        assert runs[0][2] == pytest.approx(0.211)
        assert runs[1][0] == pytest.approx(100.0)      # the user's swale
        assert runs[1][2] == pytest.approx(0.621)

    def test_a_bigger_section_downstream_is_quicker_than_all_natural(self):
        d = [0.0, 50.0, 100.0, 150.0]
        natural = mixed_channel_hours(d, [0.211] * 4, slope=0.05)[0]
        with_drain = mixed_channel_hours(d, [0.211, 0.211, 0.621, 0.621],
                                         slope=0.05)[0]
        assert with_drain < natural

    def test_roughness_changes_also_split_the_leg(self):
        d = [0.0, 50.0, 100.0]
        _h, runs = mixed_channel_hours(d, [0.211] * 3, slope=0.05,
                                       roughnesses=[0.05, 0.05, 0.025])
        assert len(runs) == 2

    def test_a_smoother_channel_is_faster(self):
        d = [0.0, 100.0]
        rough = mixed_channel_hours(d, [0.211] * 2, slope=0.05, roughnesses=[0.05] * 2)[0]
        smooth = mixed_channel_hours(d, [0.211] * 2, slope=0.05,
                                     roughnesses=[EARTHWORK_CHANNEL_ROUGHNESS] * 2)[0]
        assert smooth < rough

    def test_slopes_come_from_the_profile_when_one_is_given(self):
        d = [0.0, 100.0, 200.0]
        z = [10.0, 5.0, 4.0]                            # steep then flat
        _h, runs = mixed_channel_hours(d, [0.211, 0.211, 0.621], elevations_m=z)
        assert runs[0][1] == pytest.approx(0.05)        # 5 m over 100 m
        assert runs[1][1] == pytest.approx(0.01)        # 1 m over 100 m

    def test_the_runs_sum_to_the_whole_channel_leg(self):
        d = [0.0, 40.0, 90.0, 150.0]
        _h, runs = mixed_channel_hours(d, [0.211, 0.621, 0.621, 0.125], slope=0.04)
        assert sum(r[0] for r in runs) == pytest.approx(150.0)

    def test_the_total_is_the_sum_of_its_runs(self):
        d = [0.0, 40.0, 90.0, 150.0]
        hours, runs = mixed_channel_hours(d, [0.211, 0.621, 0.621, 0.125], slope=0.04)
        assert hours == pytest.approx(
            sum(channel_flow_hours(L, s, r, n) for L, s, r, n in runs))

    @pytest.mark.parametrize("d,r", [
        (None, None), ([], []), ([0.0, 1.0], [0.2]),
    ])
    def test_degenerate_input_gives_nothing(self, d, r):
        assert mixed_channel_hours(d, r) == (0.0, [])

    def test_a_zero_length_leg_produces_no_runs(self):
        assert mixed_channel_hours([0.0, 0.0], [0.211, 0.211], slope=0.05) == (0.0, [])

    def test_time_of_concentration_accepts_a_precomputed_channel(self):
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        tc = time_of_concentration(*legs, p2_mm=75.0, channel_hours=0.25)
        assert tc.channel_hr == pytest.approx(0.25)

    def test_a_precomputed_channel_overrides_the_leg_calculation(self):
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        plain = time_of_concentration(*legs, p2_mm=75.0)
        overridden = time_of_concentration(*legs, p2_mm=75.0, channel_hours=0.0)
        assert overridden.total_hr < plain.total_hr


class TestChannelRadii:
    def test_the_published_radii_are_real_trapezoid_sections(self):
        """The presets are computed from actual channel shapes, not asserted — this
        pins them to the same geometry the rest of the plugin uses."""
        from terrainflow_assessment.core.sizing import trapezoid_section
        for label, expected in (
            ("Shallow grassed swale (1.5 m x 0.15 m)", (1.5, 1.2, 0.15)),
            ("Small farm drain (1.6 m x 0.30 m)", (1.6, 1.0, 0.30)),
            ("Gully / farm stream (3 m x 0.5 m)", (3.0, 2.0, 0.50)),
            ("Incised watercourse (5 m x 1.2 m)", (5.0, 2.6, 1.20)),
        ):
            section = trapezoid_section(*expected)
            assert CHANNEL_RADII[label] == pytest.approx(
                section.hydraulic_radius, abs=0.001)

    def test_the_default_is_the_farm_drain(self):
        assert DEFAULT_CHANNEL_RADIUS_M == pytest.approx(
            CHANNEL_RADII["Small farm drain (1.6 m x 0.30 m)"])

    def test_radius_barely_moves_tc(self):
        """It is the weakest assumption in the chain and the least consequential: the
        channel leg is the fastest part of the path, so it contributes least."""
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        sheet, shallow, channel = legs
        results = []
        for radius in (0.1, 1.0):                      # a tenfold change
            tc = time_of_concentration(
                sheet, shallow, (channel[0], channel[1],
                                 {"hydraulic_radius_m": radius}), p2_mm=75.0)
            results.append(tc.design_min)
        assert abs(results[0] - results[1]) < 2.0      # under two minutes


class TestTimeOfConcentration:
    def test_sums_the_three_legs(self):
        tt = time_of_concentration((30.0, 0.05), (200.0, 0.05), (120.0, 0.05), p2_mm=75.0)
        assert tt.total_hr == pytest.approx(
            sheet_flow_hours(30.0, 0.05, DEFAULT_SHEET_ROUGHNESS, 75.0)
            + shallow_concentrated_hours(200.0, 0.05)
            + channel_flow_hours(120.0, 0.05))

    def test_a_realistic_pasture_catchment_responds_in_minutes_not_hours(self):
        """The point of the whole exercise: a 2.8 ha catchment responds in about a
        quarter of an hour, so reading intensity off a 24-hour storm is meaningless."""
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        tt = time_of_concentration(*legs, p2_mm=75.0)
        assert 5.0 < tt.design_min < 40.0

    def test_missing_p2_skips_sheet_flow_and_says_so(self):
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        tt = time_of_concentration(*legs, p2_mm=None)
        assert tt.sheet_hr == 0.0
        assert any("sheet-flow leg was skipped" in w for w in tt.warnings)

    def test_skipping_sheet_flow_understates_tc(self):
        """Which overstates the intensity and so oversizes — the safe direction, but
        the user is told rather than left to infer it."""
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        with_p2 = time_of_concentration(*legs, p2_mm=75.0)
        without = time_of_concentration(*legs, p2_mm=None)
        assert without.total_hr < with_p2.total_hr

    def test_below_the_validity_floor_is_held_and_flagged(self):
        tt = time_of_concentration((5.0, 0.30), (10.0, 0.30), None, p2_mm=75.0)
        assert tt.clamped
        assert tt.design_hr == pytest.approx(MIN_VALID_TC_HR)
        assert any("validity floor" in w for w in tt.warnings)

    def test_a_long_path_is_not_clamped(self):
        legs = split_flow_path(1200.0, 0.03, channel_length_m=600.0)
        assert not time_of_concentration(*legs, p2_mm=75.0).clamped

    def test_no_legs_at_all_is_zero_not_an_error(self):
        tt = time_of_concentration(None, None, None, p2_mm=75.0)
        assert tt.total_hr == 0.0
        assert tt.warnings == []

    def test_roughness_overrides_are_honoured(self):
        rough = time_of_concentration(
            (30.0, 0.05, {"roughness": 0.80}), None, None, p2_mm=75.0)
        smooth = time_of_concentration(
            (30.0, 0.05, {"roughness": 0.011}), None, None, p2_mm=75.0)
        assert rough.total_hr > smooth.total_hr

    def test_channel_overrides_are_honoured(self):
        wide = time_of_concentration(
            None, None, (500.0, 0.02, {"hydraulic_radius_m": 1.0}), p2_mm=75.0)
        narrow = time_of_concentration(
            None, None, (500.0, 0.02, {"hydraulic_radius_m": 0.1}), p2_mm=75.0)
        assert narrow.total_hr > wide.total_hr

    def test_paved_override_is_honoured(self):
        paved = time_of_concentration(None, (300.0, 0.05, {"paved": True}), None)
        unpaved = time_of_concentration(None, (300.0, 0.05, {"paved": False}), None)
        assert paved.total_hr < unpaved.total_hr

    def test_an_over_length_sheet_leg_reports_the_cap(self):
        tt = time_of_concentration((200.0, 0.05), None, None, p2_mm=75.0)
        assert any("capped at" in w for w in tt.warnings)

    def test_summary_names_all_three_legs(self):
        legs = split_flow_path(350.0, 0.08, channel_length_m=120.0)
        text = time_of_concentration(*legs, p2_mm=75.0).summary()
        for word in ("sheet", "shallow", "channel", "min"):
            assert word in text

    def test_defaults_are_the_documented_ones(self):
        assert DEFAULT_SHEET_ROUGHNESS == SHEET_ROUGHNESS["Dense grasses / pasture"]
        assert DEFAULT_CHANNEL_ROUGHNESS == 0.05


class TestValidityCeiling:
    """TOC-09: TR-55 App. F states Tc validity as "minimum, 0.1; maximum, 10.0" hr.

    Only the floor was enforced. A Tc past the ceiling reads a low intensity off the
    far tail of the IDF curve — the under-sizing direction.
    """

    def test_design_hr_is_held_at_the_ceiling(self):
        from terrainflow_assessment.modules.time_of_concentration import (
            MAX_VALID_TC_HR,
            TravelTime,
        )
        tt = TravelTime(sheet_hr=0.0, shallow_hr=0.0, channel_hr=25.0)
        assert tt.total_hr == pytest.approx(25.0)     # the raw legs stay honest
        assert tt.design_hr == pytest.approx(MAX_VALID_TC_HR)
        assert tt.clamped is True

    def test_ordinary_tc_is_not_clamped(self):
        from terrainflow_assessment.modules.time_of_concentration import TravelTime
        tt = TravelTime(sheet_hr=0.2, shallow_hr=0.3, channel_hr=0.5)
        assert tt.clamped is False
        assert tt.design_hr == pytest.approx(1.0)

    def test_a_path_over_the_ceiling_is_warned_about(self):
        from terrainflow_assessment.modules.time_of_concentration import (
            time_of_concentration,
        )
        tt = time_of_concentration(sheet=None, shallow=None, channel_hours=25.0)
        assert any("ceiling" in w for w in tt.warnings)
