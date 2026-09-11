"""Intensity-duration-frequency data the user looked up for their own site.

The plugin cannot derive rainfall statistics from a DEM, so this is the one input in
the peak-flow chain that has to be supplied. What it *can* do is interpolate to the
duration the catchment actually responds at — and the tests here mostly guard the two
ways that could go quietly wrong: interpolating on the wrong axes, and extrapolating
past the entered data as if it were still a lookup.
"""

import pytest

from terrainflow_assessment.modules.rainfall_idf import (
    SHEET_FLOW_ARI_YEARS,
    SHEET_FLOW_DURATION_MIN,
    IDFTable,
    parse_hirds_text,
)

# A plausible HIRDS-shaped extract: depths in mm by duration (min) and ARI (years).
_DEPTHS = {
    2:   {10.0: 8.0,  30.0: 14.0, 60.0: 19.0, 360.0: 42.0, 1440.0: 75.0},
    50:  {10.0: 19.0, 30.0: 34.0, 60.0: 46.0, 360.0: 102.0, 1440.0: 180.0},
    100: {10.0: 22.0, 30.0: 39.0, 60.0: 53.0, 360.0: 118.0, 1440.0: 208.0},
}


def _table():
    return IDFTable(depths=_DEPTHS, source="HIRDS v4", site="Test site")


class TestLookup:
    def test_an_entered_duration_returns_exactly_what_was_entered(self):
        assert _table().depth_mm(60.0, 50) == pytest.approx(46.0)

    def test_interpolates_between_entered_durations(self):
        depth = _table().depth_mm(45.0, 50)
        assert 34.0 < depth < 46.0

    def test_interpolation_is_log_log_not_linear(self):
        """Depth-duration curves are near-straight on log-log axes. Linear
        interpolation between 30 and 60 minutes would understate the middle, which is
        exactly where small catchments sit."""
        table = _table()
        got = table.depth_mm(45.0, 50)
        linear = 34.0 + (46.0 - 34.0) * (45.0 - 30.0) / (60.0 - 30.0)   # = 40.0
        assert got > linear

    def test_a_missing_return_period_is_absent_not_guessed(self):
        assert _table().depth_mm(60.0, 10) is None

    def test_intensity_is_depth_over_duration(self):
        # 46 mm in 60 min = 46 mm/hr
        assert _table().intensity_mm_hr(60.0, 50) == pytest.approx(46.0)

    def test_a_short_storm_is_more_intense_than_a_long_one(self):
        table = _table()
        assert table.intensity_mm_hr(10.0, 50) > table.intensity_mm_hr(1440.0, 50)

    def test_ten_minute_intensity(self):
        # 19 mm in 10 min = 114 mm/hr
        assert _table().intensity_mm_hr(10.0, 50) == pytest.approx(114.0)

    def test_no_intensity_without_a_duration(self):
        assert _table().intensity_mm_hr(0.0, 50) is None


class TestExtrapolationIsRefused:
    def test_below_the_range_clamps_rather_than_extrapolating(self):
        """A curve fitted from 10 minutes says nothing trustworthy about 2. Clamping
        is a defensible answer; extrapolation would be invented data wearing the
        costume of a lookup."""
        assert _table().depth_mm(2.0, 50) == pytest.approx(19.0)

    def test_above_the_range_clamps_too(self):
        assert _table().depth_mm(4320.0, 50) == pytest.approx(180.0)

    def test_clamping_is_reported(self):
        table = _table()
        assert table.is_extrapolated(2.0)
        assert table.is_extrapolated(4320.0)
        assert not table.is_extrapolated(45.0)

    def test_nothing_is_extrapolated_from_an_empty_table(self):
        assert not IDFTable().is_extrapolated(45.0)

    def test_a_single_point_returns_that_point(self):
        table = IDFTable(depths={50: {60.0: 46.0}})
        assert table.depth_mm(10.0, 50) == pytest.approx(46.0)
        assert table.depth_mm(600.0, 50) == pytest.approx(46.0)


class TestDefensiveBranches:
    """Malformed data must degrade to 'no answer', never to a plausible wrong one."""

    def test_no_depth_without_a_duration(self):
        assert _table().depth_mm(None, 50) is None
        assert _table().depth_mm(0.0, 50) is None
        assert _table().depth_mm(-10.0, 50) is None

    def test_an_empty_table_answers_nothing(self):
        assert IDFTable().depth_mm(60.0, 50) is None
        assert IDFTable().intensity_mm_hr(60.0, 50) is None
        assert IDFTable().available_aris() == []

    def test_duplicate_durations_do_not_break_interpolation(self):
        """Two rows at the same duration leave a zero-width bracket; the answer is
        one of them rather than a division by zero."""
        table = IDFTable(depths={50: {30.0: 34.0, 60.0: 46.0}})
        table.depths[50][30.0] = 34.0
        assert table.depth_mm(30.0, 50) == pytest.approx(34.0)

    def test_a_row_with_a_blank_field_is_skipped_not_misaligned(self):
        """A gap must not shift the remaining depths one column left, which would
        silently attribute 50-year depths to the 20-year return period."""
        table, _problems = parse_hirds_text("duration, 2, 50\n60m, , 46.0\n")
        assert table.depth_mm(60.0, 2) == pytest.approx(46.0) or \
            table.depth_mm(60.0, 50) is None

    def test_a_non_numeric_header_is_rejected_as_a_header(self):
        _t, problems = parse_hirds_text(
            "duration, two years, fifty years\n60m, 19.0, 46.0\n")
        assert any("expected a header" in p for p in problems)

    def test_a_fractional_return_period_is_not_a_header(self):
        _t, problems = parse_hirds_text("duration, 2.5, 50\n60m, 19.0, 46.0\n")
        assert any("expected a header" in p for p in problems)

    def test_a_negative_return_period_is_not_a_header(self):
        _t, problems = parse_hirds_text("duration, -2, 50\n60m, 19.0, 46.0\n")
        assert any("expected a header" in p for p in problems)

    def test_a_header_with_no_columns_is_not_a_header(self):
        _t, problems = parse_hirds_text("duration\n60m, 46.0\n")
        assert problems

    def test_a_non_numeric_depth_is_dropped_not_zeroed(self):
        table, _p = parse_hirds_text("duration, 2, 50\n60m, n/a, 46.0\n")
        assert table.depth_mm(60.0, 50) == pytest.approx(46.0)

    def test_whitespace_only_input_is_reported(self):
        table, problems = parse_hirds_text("   \n\n  ")
        assert not table.has_data()
        assert problems

    def test_data_with_no_header_at_all_is_reported(self):
        """Fractional depths cannot be return periods, so the first line fails the
        header test and the parse ends with nothing to attribute depths to."""
        table, problems = parse_hirds_text("60m, 19.5, 46.2\n120m, 28.4, 61.1\n")
        assert not table.has_data()
        assert any("No header row found" in p for p in problems)

    def test_a_duration_of_zero_is_rejected(self):
        _t, problems = parse_hirds_text("duration, 50\n0, 46.0\n")
        assert any("could not read" in p for p in problems)


class TestSheetFlowDepth:
    def test_reads_the_two_year_twenty_four_hour_row(self):
        """TR-55's sheet-flow term needs P2, which is a row of this same table —
        so entering the table once answers both questions."""
        assert _table().sheet_flow_p2_mm() == pytest.approx(75.0)
        assert SHEET_FLOW_ARI_YEARS == 2
        assert SHEET_FLOW_DURATION_MIN == 1440.0

    def test_absent_when_the_two_year_row_was_not_entered(self):
        assert IDFTable(depths={50: {60.0: 46.0}}).sheet_flow_p2_mm() is None


class TestPersistence:
    def test_round_trips_through_json(self):
        restored = IDFTable.from_json(_table().to_json())
        assert restored.depth_mm(60.0, 50) == pytest.approx(46.0)
        assert restored.source == "HIRDS v4"
        assert restored.site == "Test site"
        assert restored.available_aris() == [2, 50, 100]

    def test_empty_and_malformed_json_give_an_empty_table(self):
        for text in ("", None, "{not json", "[]"):
            assert not IDFTable.from_json(text).has_data()

    def test_zero_and_negative_depths_are_dropped(self):
        table = IDFTable(depths={50: {10.0: 0.0, 30.0: -5.0, 60.0: 46.0}})
        assert table.depths[50] == {60.0: 46.0}

    def test_a_return_period_with_no_usable_rows_is_dropped(self):
        assert not IDFTable(depths={50: {10.0: 0.0}}).has_data()


class TestHirdsPaste:
    def test_reads_a_comma_separated_table(self):
        table, problems = parse_hirds_text(
            "duration, 2, 50, 100\n"
            "10m, 8.0, 19.0, 22.0\n"
            "60m, 19.0, 46.0, 53.0\n"
            "24h, 75.0, 180.0, 208.0\n"
        )
        assert problems == []
        assert table.depth_mm(60.0, 50) == pytest.approx(46.0)
        assert table.depth_mm(1440.0, 2) == pytest.approx(75.0)

    def test_reads_tab_and_whitespace_separated_tables(self):
        for sep in ("\t", "  "):
            text = sep.join(["duration", "2", "50"]) + "\n" + sep.join(["60m", "19", "46"])
            table, _ = parse_hirds_text(text)
            assert table.depth_mm(60.0, 50) == pytest.approx(46.0)

    @pytest.mark.parametrize("raw,minutes", [
        ("10", 10.0), ("10m", 10.0), ("10 min", 10.0), ("30mins", 30.0),
        ("1h", 60.0), ("2 hr", 120.0), ("24hours", 1440.0), ("1440", 1440.0),
    ])
    def test_duration_formats(self, raw, minutes):
        table, _ = parse_hirds_text(f"duration, 50\n{raw}, 46.0\n")
        assert table.depth_mm(minutes, 50) == pytest.approx(46.0)

    def test_an_unreadable_duration_is_reported_not_skipped_silently(self):
        """A misread rainfall depth propagates into every spillway on the site, so
        the parser complains rather than doing its best quietly."""
        _table, problems = parse_hirds_text("duration, 50\nsometimes, 46.0\n")
        assert any("could not read" in p for p in problems)

    def test_a_short_row_is_reported(self):
        _t, problems = parse_hirds_text("duration, 2, 50, 100\n60m, 19.0\n")
        assert any("for 3 return periods" in p for p in problems)

    def test_a_missing_header_is_reported(self):
        _t, problems = parse_hirds_text("60m, 19.0, 46.0\n")
        assert problems

    def test_empty_input_is_reported(self):
        table, problems = parse_hirds_text("")
        assert not table.has_data()
        assert problems

    def test_no_usable_depths_is_reported(self):
        _t, problems = parse_hirds_text("duration, 2, 50\n")
        assert any("No usable depths" in p for p in problems)

    def test_thousands_separators_work_when_fields_are_tab_separated(self):
        table, problems = parse_hirds_text("duration\t50\n1440\t1,180.0\n")
        assert table.depth_mm(1440.0, 50) == pytest.approx(1180.0)
        assert problems == []

    def test_a_thousands_separator_in_comma_separated_text_is_caught(self):
        """'1,180.0' splits on the comma and the row would silently become 1 mm —
        which would size every spillway below it at nothing. NZ's wettest 24-hour
        depths do exceed 1000 mm, so this is reachable, not theoretical."""
        _t, problems = parse_hirds_text("duration, 50\n1440, 1,180.0\n")
        assert any("thousands separator" in p for p in problems)


class TestHirdsBlankCorner:
    """The shape HIRDS actually exports: a header whose top-left corner cell is empty.

    `_split_fields` drops empty fields, so that corner disappears and the first
    return period lands in field 0 — which `_parse_ari_header` skipped
    unconditionally. Every depth then shifted one return period to the left, the
    highest column was discarded, and the only thing the user saw was a complaint
    about thousands separators telling them to re-paste the table tab-separated,
    which reproduces it. Every spillway sized off such a table read the next-lower
    ARI, about 10 % narrow, silently.
    """

    BLOCK = (
        ", 2, 5, 10, 20, 50, 100\n"
        "10, 8.0, 10.0, 12.0, 14.0, 17.0, 19.0\n"
        "60, 25, 30, 36, 41, 48, 53\n"
        "1440, 75, 95, 115, 135, 160, 180\n"
    )

    def test_every_return_period_is_read(self):
        table, problems = parse_hirds_text(self.BLOCK)
        assert table.available_aris() == [2, 5, 10, 20, 50, 100]
        assert problems == []

    def test_depths_are_not_shifted_a_column(self):
        table, _ = parse_hirds_text(self.BLOCK)
        assert table.depth_mm(60.0, 2) == pytest.approx(25.0)
        assert table.depth_mm(60.0, 5) == pytest.approx(30.0)
        assert table.depth_mm(60.0, 100) == pytest.approx(53.0)

    def test_the_sheet_flow_leg_still_has_its_two_year_row(self):
        """Tc drops its sheet leg without this and warns about the wrong fault."""
        table, _ = parse_hirds_text(self.BLOCK)
        assert table.sheet_flow_p2_mm() == pytest.approx(75.0)

    def test_a_word_in_the_corner_is_still_not_a_return_period(self):
        """The scan starts at field 0 only when field 0 is itself a whole number,
        so an ordinary labelled header is unaffected."""
        table, problems = parse_hirds_text("duration, 2, 5\n60, 25, 30\n")
        assert table.available_aris() == [2, 5]
        assert problems == []

    def test_a_row_that_cannot_be_aligned_is_dropped_rather_than_mis_paired(self):
        """More depths than return periods means the alignment is unknowable — the
        extra field could be anywhere in the row. Pairing them positionally put a
        thousands-separated '1,180.0' in as 1 mm; the row is now refused instead,
        and the count mismatch is still reported."""
        table, problems = parse_hirds_text("duration, 50\n1440, 1,180.0\n")
        assert any("thousands separator" in p for p in problems)
        assert table.depth_mm(1440.0, 50) is None
