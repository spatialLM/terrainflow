"""Tests for terrainflow_assessment/modules/reporting.py"""
import os

import numpy as np
import pytest

from terrainflow_assessment.modules.burn_strategy import overtopping_warning
from terrainflow_assessment.modules.earthwork_design import capacity_breakdown
from terrainflow_assessment.modules.reporting import (
    BaselineReport,
    ComparisonResult,
    PostInterventionReport,
    VerificationResult,
    _build_fill_timeline_chart,
    _build_hydrograph_chart,
    _fig_to_base64,
    _mini_bar,
    attribute_ponding_volume,
    build_verification,
    compare,
    event_pond_depth,
    format_live_assessment,
    level_for_volume,
    overtopping_spill,
    raster_ponding_volume,
)
from terrainflow_assessment.modules.water_balance import BalanceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline(**kwargs):
    defaults = dict(
        site_name="Test Site",
        catchment_area_ha=50.0,
        rainfall_mm=80.0,
        duration_hr=1.0,
        cn=75.0,
        total_runoff_m3=4000.0,
        exit_volume_m3=3500.0,
        peak_outflow_ls=120.0,
        peak_outflow_time_hr=0.5,
        exit_points=[{"label": "Exit 1", "volume_m3": 3500}],
        timestep_table=[
            {"time_hr": 0.25, "outflow_ls": 60.0},
            {"time_hr": 0.5, "outflow_ls": 120.0},
            {"time_hr": 0.75, "outflow_ls": 90.0},
        ],
    )
    defaults.update(kwargs)
    return BaselineReport(**defaults)


def _post(**kwargs):
    defaults = dict(
        exit_volume_m3=1000.0,
        peak_outflow_ls=40.0,
        peak_outflow_time_hr=0.8,
        total_infiltrated_m3=500.0,
        earthwork_summary=[
            {
                "name": "Swale 1",
                "type": "swale",
                "capacity_m3": 200.0,
                "stored_m3": 150.0,
                "peak_fill_pct": 75.0,
                "final_fill_pct": 75.0,
                "overflowed": False,
                "first_overflow_hr": None,
                "total_overflow_m3": 0.0,
                "total_inflow_m3": 180.0,
                "total_infiltration_m3": 30.0,
                "cut_vol_m3": 120.0,
                "fill_vol_m3": 20.0,
            }
        ],
        timestep_table=[
            {"time_hr": 0.25, "outflow_ls": 20.0, "Swale 1_fill_pct": 20.0, "Swale 1_overflow": False},
            {"time_hr": 0.5, "outflow_ls": 40.0, "Swale 1_fill_pct": 60.0, "Swale 1_overflow": False},
            {"time_hr": 0.75, "outflow_ls": 30.0, "Swale 1_fill_pct": 75.0, "Swale 1_overflow": False},
        ],
        exit_points=[{"label": "Exit 1", "volume_m3": 1000}],
    )
    defaults.update(kwargs)
    return PostInterventionReport(**defaults)


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:
    def test_returns_comparison_result(self):
        result = compare(_baseline(), _post())
        assert isinstance(result, ComparisonResult)

    def test_exit_reduction_pct(self):
        b = _baseline(exit_volume_m3=3500.0)
        p = _post(exit_volume_m3=1000.0)
        result = compare(b, p)
        expected = (3500 - 1000) / 3500 * 100
        assert result.exit_reduction_pct == pytest.approx(expected, rel=1e-4)

    def test_peak_reduction_pct(self):
        b = _baseline(peak_outflow_ls=120.0)
        p = _post(peak_outflow_ls=40.0)
        result = compare(b, p)
        expected = (120 - 40) / 120 * 100
        assert result.peak_reduction_pct == pytest.approx(expected, rel=1e-4)

    def test_peak_delay(self):
        b = _baseline(peak_outflow_time_hr=0.5)
        p = _post(peak_outflow_time_hr=0.9)
        result = compare(b, p)
        assert result.peak_delay_hr == pytest.approx(0.4, rel=1e-4)

    def test_peak_delay_zero_when_not_delayed(self):
        b = _baseline(peak_outflow_time_hr=0.8)
        p = _post(peak_outflow_time_hr=0.5)  # earlier — no delay
        result = compare(b, p)
        assert result.peak_delay_hr == 0.0

    def test_captured_pct(self):
        # total_runoff=4000, post_exit=1000 → captured=3000 → 75%
        b = _baseline(total_runoff_m3=4000.0)
        p = _post(exit_volume_m3=1000.0)
        result = compare(b, p)
        assert result.captured_pct == pytest.approx(75.0, rel=1e-4)

    def test_captured_pct_clamped_to_100(self):
        b = _baseline(total_runoff_m3=100.0)
        p = _post(exit_volume_m3=-500.0)  # impossible but check clamp
        result = compare(b, p)
        assert result.captured_pct <= 100.0

    def test_captured_pct_clamped_to_0(self):
        b = _baseline(total_runoff_m3=100.0)
        p = _post(exit_volume_m3=200.0)  # more exits than total runoff
        result = compare(b, p)
        assert result.captured_pct >= 0.0

    def test_zero_total_runoff_no_error(self):
        b = _baseline(total_runoff_m3=0.0)
        result = compare(b, _post())
        assert result.captured_pct == 0.0

    def test_zero_baseline_exit_no_error(self):
        b = _baseline(exit_volume_m3=0.0)
        result = compare(b, _post())
        assert result.exit_reduction_pct == 0.0

    def test_zero_baseline_peak_no_error(self):
        b = _baseline(peak_outflow_ls=0.0)
        result = compare(b, _post())
        assert result.peak_reduction_pct == 0.0

    def test_net_cut_and_fill(self):
        p = _post()  # earthwork_summary has cut=120, fill=20
        result = compare(_baseline(), p)
        assert result.net_cut_m3 == pytest.approx(120.0)
        assert result.net_fill_m3 == pytest.approx(20.0)
        assert result.net_cut_fill_m3 == pytest.approx(100.0)

    def test_multiple_earthworks_cut_fill_summed(self):
        p = _post(earthwork_summary=[
            {"cut_vol_m3": 50.0, "fill_vol_m3": 10.0},
            {"cut_vol_m3": 30.0, "fill_vol_m3": 5.0},
        ])
        result = compare(_baseline(), p)
        assert result.net_cut_m3 == pytest.approx(80.0)
        assert result.net_fill_m3 == pytest.approx(15.0)

    def test_references_preserved(self):
        b, p = _baseline(), _post()
        result = compare(b, p)
        assert result.baseline is b
        assert result.post is p

    def test_exit_reduction_clamped_non_negative(self):
        b = _baseline(exit_volume_m3=100.0)
        p = _post(exit_volume_m3=200.0)  # worse than baseline
        result = compare(b, p)
        assert result.exit_reduction_pct >= 0.0

    def test_peak_reduction_clamped_non_negative(self):
        b = _baseline(peak_outflow_ls=50.0)
        p = _post(peak_outflow_ls=100.0)  # worse than baseline
        result = compare(b, p)
        assert result.peak_reduction_pct >= 0.0


# ---------------------------------------------------------------------------
# ComparisonResult dataclass defaults
# ---------------------------------------------------------------------------

class TestComparisonResultDefaults:
    def test_all_zero_defaults(self):
        cr = ComparisonResult()
        assert cr.captured_pct == 0.0
        assert cr.exit_reduction_pct == 0.0
        assert cr.peak_reduction_pct == 0.0
        assert cr.peak_delay_hr == 0.0
        assert cr.net_cut_m3 == 0.0
        assert cr.net_fill_m3 == 0.0

    def test_baseline_none_by_default(self):
        assert ComparisonResult().baseline is None

    def test_post_none_by_default(self):
        assert ComparisonResult().post is None


# ---------------------------------------------------------------------------
# BaselineReport and PostInterventionReport
# ---------------------------------------------------------------------------

class TestReportDataclasses:
    def test_baseline_defaults(self):
        b = BaselineReport()
        assert b.site_name == "Unnamed Site"
        assert b.catchment_area_ha == 0.0
        assert b.exit_points == []
        assert b.timestep_table == []

    def test_post_defaults(self):
        p = PostInterventionReport()
        assert p.exit_volume_m3 == 0.0
        assert p.earthwork_summary == []
        assert p.timestep_table == []


# ---------------------------------------------------------------------------
# _fig_to_base64
# ---------------------------------------------------------------------------

class TestFigToBase64:
    def test_returns_non_empty_string(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        result = _fig_to_base64(fig)
        plt.close(fig)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_result_is_valid_base64(self):
        import base64

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        result = _fig_to_base64(fig)
        plt.close(fig)
        # Should not raise
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# _build_hydrograph_chart
# ---------------------------------------------------------------------------

class TestBuildHydrographChart:
    def test_returns_base64_with_data(self):
        result = _build_hydrograph_chart(_baseline(), _post())
        assert result is not None
        assert isinstance(result, str)

    def test_returns_none_without_timestep_data(self):
        b = _baseline(timestep_table=[])
        p = _post(timestep_table=[])
        # Should still return something (empty axes) or None-ish
        result = _build_hydrograph_chart(b, p)
        assert result is None or isinstance(result, str)

    def test_handles_empty_post_timestep(self):
        b = _baseline()
        p = _post(timestep_table=[])
        result = _build_hydrograph_chart(b, p)
        assert result is not None


# ---------------------------------------------------------------------------
# _build_fill_timeline_chart
# ---------------------------------------------------------------------------

class TestBuildFillTimelineChart:
    def test_returns_base64_with_data(self):
        p = _post()
        result = _build_fill_timeline_chart(p)
        assert result is not None
        assert isinstance(result, str)

    def test_returns_none_with_no_timestep(self):
        p = _post(timestep_table=[])
        result = _build_fill_timeline_chart(p)
        assert result is None

    def test_returns_none_with_no_earthworks(self):
        p = _post(earthwork_summary=[])
        result = _build_fill_timeline_chart(p)
        assert result is None

    def test_marks_overflow_event(self):
        """Overflow earthworks should add a vertical line — no crash."""
        ew = _post().earthwork_summary[0].copy()
        ew["overflowed"] = True
        ew["first_overflow_hr"] = 0.5
        p = _post(earthwork_summary=[ew])
        result = _build_fill_timeline_chart(p)
        assert result is not None


# ---------------------------------------------------------------------------
# export_html
# ---------------------------------------------------------------------------

class TestHtmlOutputKeepsWhatTheLegacyReportHad:
    """The old export_html built its own document from a ComparisonResult.

    It is gone: both formats now render the shared model. These assert that the
    content the legacy tests protected still reaches the HTML by that route, so
    the convergence lost nothing.
    """

    def _html(self, comparison=None, **kw):
        from terrainflow_assessment.modules.report_html import render_html
        from terrainflow_assessment.modules.report_model import (
            ReportData,
            build_report,
        )
        comparison = comparison or ComparisonResult(
            baseline=_baseline(), post=_post())
        data = ReportData(site_name=comparison.baseline.site_name
                          if comparison.baseline else "Unnamed Site",
                          baseline=comparison.baseline,
                          comparison=comparison, **kw)
        return render_html(build_report(data))

    def test_writes_a_file(self, tmp_path):
        from terrainflow_assessment.modules.report_html import write_html
        from terrainflow_assessment.modules.report_model import (
            ReportData,
            build_report,
        )
        out = str(tmp_path / "report.html")
        result = write_html(out, build_report(ReportData(baseline=_baseline())))
        assert os.path.exists(out) and result == out

    def test_site_name_reaches_the_page(self):
        html = self._html(ComparisonResult(
            baseline=_baseline(site_name="MyFarm"), post=_post()))
        assert "MyFarm" in html

    def test_has_doctype(self):
        assert "<!DOCTYPE html>" in self._html()

    def test_has_the_headline_sections(self):
        html = self._html()
        for section in ("Your scheme in one page", "Where your water goes today",
                        "How the storm plays out over time",
                        "Inputs, method and limits"):
            assert section in html, section

    def test_before_and_after_figures_survive(self):
        html = self._html()
        assert "Before and after the earthworks" in html
        assert "Fastest flow at the boundary" in html

    def test_exit_points_survive(self):
        assert "Where water leaves the boundary" in self._html()

    def test_none_baseline_no_crash(self):
        assert self._html(ComparisonResult(baseline=None, post=None))


# ---------------------------------------------------------------------------
# Non-circular verification (§4) — pure reducers
# ---------------------------------------------------------------------------

class TestRasterPondingVolume:
    def test_sum_times_cell_area(self):
        pond = np.array([[0.0, 0.5], [1.0, 0.0]], dtype="float32")
        # (0.5 + 1.0) * cell_area 4 = 6.0
        assert raster_ponding_volume(pond, 4.0) == pytest.approx(6.0)

    def test_min_depth_excludes_shallow(self):
        pond = np.array([[0.0005, 0.0005]], dtype="float32")  # below 0.001
        assert raster_ponding_volume(pond, 1.0) == pytest.approx(0.0)


class TestImpoundedVolume:
    def test_new_ponding_over_baseline(self):
        from terrainflow_assessment.modules.reporting import impounded_volume
        baseline = np.array([[0.0, 1.0], [0.0, 0.0]], dtype="float32")
        dammed = np.array([[0.0, 2.0], [1.0, 0.0]], dtype="float32")
        # positive diffs: (2−1)=1 and (1−0)=1 → 2 × cell_area 3 = 6
        assert impounded_volume(baseline, dammed, 3.0) == pytest.approx(6.0)

    def test_negative_diffs_clipped(self):
        from terrainflow_assessment.modules.reporting import impounded_volume
        baseline = np.array([[5.0]], dtype="float32")
        dammed = np.array([[2.0]], dtype="float32")  # dam removed ponding here → clip to 0
        assert impounded_volume(baseline, dammed, 1.0) == pytest.approx(0.0)

    def test_zero_baseline_is_total_ponding(self):
        from terrainflow_assessment.modules.reporting import impounded_volume
        baseline = np.zeros((2, 2), dtype="float32")
        dammed = np.array([[0.0, 1.5], [0.5, 0.0]], dtype="float32")
        assert impounded_volume(baseline, dammed, 2.0) == pytest.approx(4.0)


class TestOvertoppingSpill:
    """Does a pool leave over the wall that made it, and along how much of it."""

    def _valley(self):
        """A channel at 10 m between banks at 20 m, dammed across at 12 m.

        row 2 is the channel; the dam occupies col 3 and stands at 12.0, so the pool
        upstream fills to 12.0 and its only way out is over the dam.
        """
        ground = np.full((5, 7), 20.0)
        ground[2, :] = 10.0
        ground[2, 3] = 12.0            # the wall, across the channel
        pond = np.zeros((5, 7))
        pond[2, 0:3] = 2.0             # water to 12.0 upstream of it
        crest = np.zeros((5, 7), dtype=bool)
        crest[2, 3] = True
        return ground, pond, crest

    def test_pool_that_leaves_over_its_own_wall_is_reported(self):
        ground, pond, crest = self._valley()
        got = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 10.0)])
        assert len(got) == 1, got
        assert got[0].name == "Dam A"
        assert got[0].pour_level_m == pytest.approx(12.0)
        assert got[0].length_m == pytest.approx(1.0)      # one 1 m cell of crest
        assert got[0].pool_volume_m3 == pytest.approx(6.0)

    def test_a_lower_way_out_means_no_overtopping(self):
        """A notch in the bank below the crest takes the water instead."""
        ground, pond, crest = self._valley()
        ground[1, 1] = 11.0            # saddle in the bank, below the 12.0 wall
        pond[2, 0:3] = 1.0             # so the pool only fills to 11.0
        got = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 10.0)])
        assert got == []

    def test_length_is_the_whole_crest_at_the_pour_level(self):
        """A level crest spills along all of itself, not at the one cell D8 picks."""
        ground = np.full((5, 7), 20.0)
        ground[1:4, :] = 10.0
        ground[1:4, 3] = 12.0          # a three-cell-wide wall, all at one level
        pond = np.zeros((5, 7))
        pond[1:4, 0:3] = 2.0
        crest = np.zeros((5, 7), dtype=bool)
        crest[1:4, 3] = True
        got = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 10.0)])
        assert got[0].length_m == pytest.approx(3.0)

    def test_length_cannot_exceed_the_drawn_wall(self):
        """Diagonal cell runs over-count; the wall's own length is the ceiling."""
        ground, pond, crest = self._valley()
        got = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 0.4)])
        assert got[0].length_m == pytest.approx(0.4)

    def test_alt_saddle_reports_the_next_way_out(self):
        ground, pond, crest = self._valley()
        ground[1, 1] = 13.0            # higher than the wall, so still overtops
        got = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 10.0)])
        assert got[0].alt_saddle_m == pytest.approx(13.0)

    def test_built_ground_is_not_a_natural_saddle(self):
        """A dam's keyed returns are its own wall, not the next way out.

        They sit outside the recorded mask, so without ``built`` they come back as the
        alternative at the dam's own level and the advice reads "raise the crest 0.00 m"
        — which is what the real Quail Island design produced.
        """
        ground = np.full((5, 7), 20.0)
        ground[1:4, 0:3] = 10.0        # the basin
        ground[1:4, 3] = 12.0          # the wall, three cells of it
        ground[0, 1] = 15.0            # the genuine natural saddle
        pond = np.zeros((5, 7))
        pond[1:4, 0:3] = 2.0           # filled to 12.0
        crest = np.zeros((5, 7), dtype=bool)
        crest[2, 3] = True             # only the drawn line's own contact cell
        built = np.zeros((5, 7), dtype=bool)
        built[1:4, 3] = True           # the wall AND its keyed returns

        loose = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 10.0)])
        assert loose[0].alt_saddle_m == pytest.approx(12.0)   # its own wall

        tight = overtopping_spill(pond, ground, 1.0, [("Dam A", crest, 10.0)],
                                  built=built)
        assert tight[0].alt_saddle_m == pytest.approx(15.0)   # real ground

    def test_a_cut_is_not_a_barrier(self):
        """An empty crest mask contributes nothing — a hole cannot be overtopped."""
        ground, pond, _ = self._valley()
        empty = np.zeros((5, 7), dtype=bool)
        assert overtopping_spill(pond, ground, 1.0, [("Swale A", empty, 10.0)]) == []


class TestOvertoppingWarning:
    def test_names_the_length_and_asks_for_a_spillway(self):
        msg = overtopping_warning("Dam 15", 34.0, 56.12, alt_saddle_m=57.9)
        assert "Dam 15" in msg and "56.12" in msg and "34 m" in msg
        assert "spillway" in msg
        # The length exists to contradict the single channel the stream layer draws. It
        # used to call that an "artefact of the flow routing", which Round 14 disproved:
        # flow leaves a pool at many cells, the flat is resolved by distance so one of them
        # takes most of the flux, and the Streams layer only draws what clears its
        # contributing-area threshold. The claim the message must still make is that the
        # water does not all go where the map shows a channel.
        assert "at once" in msg
        assert "stream layer" in msg
        assert "artefact" not in msg
        assert "1.78 m" in msg     # how much crest would have to rise to miss it

    def test_a_designed_spillway_makes_it_a_model_caveat_not_a_fault(self):
        msg = overtopping_warning("Dam 15", 34.0, 56.12, has_spillway=True)
        assert "not cut into the terrain" in msg
        assert "Nothing is designed to take it" not in msg

    def test_no_length_no_warning(self):
        assert overtopping_warning("Dam 15", 0.0, 56.12) is None

    def test_an_alternative_at_the_same_level_reports_no_freeboard(self):
        """"Raising the crest 0.00 m" is worse than saying nothing.

        A crest seeded from the highest ground the line touches lands exactly on the
        bank — both Quail Island dams do — so the honest statement is that the abutment
        has no freeboard, not a rise of zero.
        """
        for alt in (56.12, 56.124):
            msg = overtopping_warning("Dam 15", 24.0, 56.12, alt_saddle_m=alt)
            assert "Raising the crest" not in msg
            assert "no freeboard at the abutment" in msg
            assert "round the end of the wall" in msg


class TestLevelForVolume:
    def test_flat_bed_is_volume_over_area(self):
        # Four cells at 10.0 m, 1 m² each, 2 m³ → 0.5 m of water at 10.5 m.
        assert level_for_volume([10.0] * 4, 1.0, 2.0) == pytest.approx(10.5)

    def test_only_the_wetted_cells_count(self):
        """A stepped bed fills from the bottom, not across the whole region.

        Cells at 10, 11, 12, 13. 0.5 m³ over 1 m² cells cannot reach 11.0 (that would
        take 1.0 m³), so it stands 0.5 m deep over the one lowest cell. Dividing the
        volume by the whole region's area instead would put it at 10.125 m — under
        water that is not there, and a pond drawn four times too wide.
        """
        assert level_for_volume([10.0, 11.0, 12.0, 13.0], 1.0, 0.5) == pytest.approx(10.5)

    def test_level_rises_across_the_step_once_the_first_cell_is_full(self):
        # 1.0 m³ exactly fills to 11.0; a further 1.0 m³ spreads over two cells.
        assert level_for_volume([10.0, 11.0, 12.0], 1.0, 2.0) == pytest.approx(11.5)

    def test_ceiling_caps_at_the_spill_level(self):
        # More water than the pool can hold is overflow, not a taller pond.
        assert level_for_volume([10.0] * 4, 1.0, 999.0, ceiling=10.4) == pytest.approx(10.4)

    def test_empty_region_returns_none(self):
        assert level_for_volume([], 1.0, 5.0) is None


class TestEventPondDepth:
    """The part-full pond: smaller and shallower, not the full pond faded."""

    def _bowl(self):
        # A 1-D valley in a 5x5 grid: one row of bed at 10, 11, 12, brim-full to 12.
        ground = np.full((5, 5), 20.0)
        ground[2, 1], ground[2, 2], ground[2, 3] = 11.0, 10.0, 12.0
        full = np.zeros((5, 5))
        full[2, 1], full[2, 2], full[2, 3] = 1.0, 2.0, 0.0   # water surface at 12.0
        return ground, full

    def _mask_at(self, cells):
        m = np.zeros((5, 5), dtype=bool)
        for r, c in cells:
            m[r, c] = True
        return m

    def test_part_full_pond_stands_in_the_bottom(self):
        ground, full = self._bowl()
        # 1 m³ of the pond's 3 m³: it stands 1 m deep over the single lowest cell and
        # does not reach the 11.0 m cell beside it.
        got = event_pond_depth(full, ground, 1.0,
                               [("A", self._mask_at([(2, 2)]))], {"A": 1.0})
        assert got[2, 2] == pytest.approx(1.0)
        assert got[2, 1] == pytest.approx(0.0)
        assert got.sum() == pytest.approx(1.0)

    def test_a_full_feature_reproduces_the_capacity_pond(self):
        ground, full = self._bowl()
        got = event_pond_depth(full, ground, 1.0,
                               [("A", self._mask_at([(2, 2)]))], {"A": 3.0})
        assert got == pytest.approx(full)

    def test_more_water_than_capacity_does_not_overtop(self):
        ground, full = self._bowl()
        got = event_pond_depth(full, ground, 1.0,
                               [("A", self._mask_at([(2, 2)]))], {"A": 500.0})
        assert got.sum() == pytest.approx(full.sum())

    def test_natural_water_already_there_is_added_back(self):
        """``stored_m3`` is what the feature adds; the hollow's own water is not gone."""
        ground, full = self._bowl()
        existing = np.zeros((5, 5))
        existing[2, 2] = 1.0                      # 1 m³ ponds here with no earthwork
        got = event_pond_depth(full, ground, 1.0,
                               [("A", self._mask_at([(2, 2)]))], {"A": 1.0},
                               existing=existing)
        assert got.sum() == pytest.approx(2.0)

    def test_a_feature_with_no_water_draws_nothing(self):
        ground, full = self._bowl()
        got = event_pond_depth(full, ground, 1.0,
                               [("A", self._mask_at([(2, 2)]))], {"A": 0.0})
        assert not got.any()

    def test_joined_features_fill_their_shared_pool_together(self):
        """One sheet of water, one level — the pair's volume, not each one's own."""
        ground, full = self._bowl()
        got = event_pond_depth(
            full, ground, 1.0,
            [("A", self._mask_at([(2, 2)])), ("B", self._mask_at([(2, 1)]))],
            {"A": 1.0, "B": 2.0},
        )
        assert got.sum() == pytest.approx(3.0)


class TestAttributePondingVolume:
    def _pond_band(self):
        # A single connected pond: row 2, cols 1-3, depth 1.0 → 3 cells
        arr = np.zeros((5, 5), dtype="float64")
        arr[2, 1:4] = 1.0
        return arr

    def _mask_at(self, cells):
        m = np.zeros((5, 5), dtype=bool)
        for r, c in cells:
            m[r, c] = True
        return m

    def test_whole_region_attributed_even_beyond_footprint(self):
        # Footprint touches one cell of a 3-cell pond → the *whole* connected region
        # (incl. cells outside the footprint, i.e. a dam pool) attributes to it.
        pond = self._pond_band()
        got = attribute_ponding_volume(
            pond, 1.0, [("A", self._mask_at([(2, 1)]))]
        )
        assert got.per_name["A"] == pytest.approx(3.0)
        assert got.unattributed_m3 == pytest.approx(0.0)
        assert got.groups == []

    def test_region_touching_no_footprint_is_unattributed(self):
        pond = self._pond_band()
        got = attribute_ponding_volume(
            pond, 1.0, [("A", self._mask_at([(0, 0)]))]
        )
        assert got.per_name["A"] == pytest.approx(0.0)
        assert got.unattributed_m3 == pytest.approx(3.0)

    def test_shared_region_belongs_to_neither_feature(self):
        """A pool two features both reach is theirs jointly, and no one's severally.

        This replaces a largest-overlap rule. On the Quail Island design that rule put
        a basin and the dam on its downhill lip — one structure, one sheet of water —
        at Δ +211% and Δ −100% respectively, for a 3,593 m³ pool neither of them holds
        alone. The volume was never wrong; awarding it was.
        """
        pond = self._pond_band()
        footprints = [
            ("A", self._mask_at([(2, 1)])),           # overlap 1
            ("B", self._mask_at([(2, 2), (2, 3)])),   # overlap 2
        ]
        got = attribute_ponding_volume(pond, 1.0, footprints)
        assert got.per_name["A"] == pytest.approx(0.0)
        assert got.per_name["B"] == pytest.approx(0.0)
        assert len(got.groups) == 1
        assert got.groups[0]["names"] == ("A", "B")
        assert got.groups[0]["volume_m3"] == pytest.approx(3.0)
        assert got.groups[0]["overlaps"] == {"A": 1, "B": 2}

    def test_three_features_on_one_pool_form_one_group(self):
        pond = self._pond_band()
        footprints = [
            ("A", self._mask_at([(2, 1)])),
            ("B", self._mask_at([(2, 2)])),
            ("C", self._mask_at([(2, 3)])),
        ]
        got = attribute_ponding_volume(pond, 1.0, footprints)
        assert len(got.groups) == 1
        assert got.groups[0]["names"] == ("A", "B", "C")
        assert got.groups[0]["volume_m3"] == pytest.approx(3.0)

    def test_joining_is_transitive(self):
        # A shares a pool with B; B shares a different pool with C. No cut separates
        # A's water from C's, so all three are one set.
        arr = np.zeros((5, 8), dtype="float64")
        arr[1, 1:3] = 1.0          # A + B
        arr[3, 4:6] = 1.0          # B + C
        a = np.zeros((5, 8), dtype=bool)
        b = np.zeros((5, 8), dtype=bool)
        c = np.zeros((5, 8), dtype=bool)
        a[1, 1] = True
        b[1, 2] = b[3, 4] = True
        c[3, 5] = True
        got = attribute_ponding_volume(arr, 1.0, [("A", a), ("B", b), ("C", c)])
        assert len(got.groups) == 1
        assert got.groups[0]["names"] == ("A", "B", "C")
        assert got.groups[0]["volume_m3"] == pytest.approx(4.0)

    def test_a_joined_feature_brings_its_solo_pool_into_the_set(self):
        """Otherwise the set is measured on part of its water and all of its capacity.

        Dam 3 holds 158 m³ alone and 321 m³ jointly with Swale 6. Scoring only the
        joint pool against both capacities reported Δ −33%; counting the dam's own
        pool into the set puts the pair within 1% of what the grid holds.
        """
        arr = np.zeros((5, 8), dtype="float64")
        arr[1, 1:3] = 1.0          # shared by A and B
        arr[3, 1:3] = 1.0          # A alone
        a = np.zeros((5, 8), dtype=bool)
        b = np.zeros((5, 8), dtype=bool)
        a[1, 1] = a[3, 1] = True
        b[1, 2] = True
        got = attribute_ponding_volume(arr, 1.0, [("A", a), ("B", b)])
        assert got.per_name["A"] == pytest.approx(0.0)
        assert len(got.groups) == 1
        assert got.groups[0]["volume_m3"] == pytest.approx(4.0)

    def test_separate_pools_with_the_same_members_merge_into_one_group(self):
        # Two disconnected pools, both reached by A and B: one group, both volumes.
        arr = np.zeros((5, 7), dtype="float64")
        arr[1, 1:3] = 1.0
        arr[3, 1:3] = 1.0
        masks = {"A": np.zeros((5, 7), dtype=bool), "B": np.zeros((5, 7), dtype=bool)}
        masks["A"][1, 1] = masks["A"][3, 1] = True
        masks["B"][1, 2] = masks["B"][3, 2] = True
        got = attribute_ponding_volume(arr, 1.0, list(masks.items()))
        assert len(got.groups) == 1
        assert got.groups[0]["volume_m3"] == pytest.approx(4.0)

    def test_volume_is_conserved_across_the_three_buckets(self):
        """Grouping moves water between buckets; it must never create or lose any."""
        arr = np.zeros((6, 8), dtype="float64")
        arr[1, 1:4] = 1.0          # shared by A and B
        arr[3, 1:3] = 2.0          # A alone
        arr[5, 5:7] = 0.5          # nobody
        a = np.zeros((6, 8), dtype=bool)
        b = np.zeros((6, 8), dtype=bool)
        a[1, 1] = a[3, 1] = True
        b[1, 3] = True
        got = attribute_ponding_volume(arr, 1.0, [("A", a), ("B", b)])
        total = (sum(got.per_name.values())
                 + sum(g["volume_m3"] for g in got.groups)
                 + got.unattributed_m3)
        assert total == pytest.approx(float(arr.sum()))

    def test_no_footprints_all_unattributed(self):
        pond = self._pond_band()
        got = attribute_ponding_volume(pond, 1.0, [])
        assert got.per_name == {}
        assert got.unattributed_m3 == pytest.approx(3.0)


class TestBuildVerification:
    def test_site_and_per_feature_math(self):
        v = build_verification(
            analytic_by_name={"S1": 200.0, "S2": 100.0},
            terrain_by_name={"S1": 180.0, "S2": 90.0},
            baseline_total_m3=10.0,
            earthworks_total_m3=290.0,
            min_dims={"S1": 1.0, "S2": 0.3},
            cell_size=1.0,
        )
        assert v.analytic_total_m3 == pytest.approx(300.0)
        assert v.terrain_total_m3 == pytest.approx(280.0)   # 290 − 10
        assert v.delta_m3 == pytest.approx(-20.0)
        assert v.delta_pct == pytest.approx(-20.0 / 300.0 * 100.0)

        by_name = {f["name"]: f for f in v.per_feature}
        # S1 resolvable (bottom 1.0 == cell) → independent terrain + delta
        assert by_name["S1"]["routing_only"] is False
        assert by_name["S1"]["terrain_m3"] == pytest.approx(180.0)
        assert by_name["S1"]["delta_pct"] == pytest.approx(-10.0)
        # S2 sub-cell (0.3 < 1.0) → routing-only, no volume claim
        assert by_name["S2"]["routing_only"] is True
        assert by_name["S2"]["terrain_m3"] is None
        assert by_name["S2"]["delta_pct"] is None

    def test_existing_ponding_is_carried_and_totals_add_up(self):
        """Dam 5 from the field design: 555 m³ already there, 726 m³ added, 1,281 total.

        Every other figure in the table is marginal, which under-describes the pool
        someone standing at the dam would see. total = existing + terrain by
        construction so the three numbers cannot drift apart.
        """
        v = build_verification(
            analytic_by_name={"Dam 5": 726.0, "Swale 1": 261.0},
            terrain_by_name={"Dam 5": 726.0, "Swale 1": 416.0},
            baseline_total_m3=1212.0,
            earthworks_total_m3=7233.0,
            min_dims={"Dam 5": 2.0, "Swale 1": 1.0},
            cell_size=1.0,
            existing_by_name={"Dam 5": 555.0},
        )
        by_name = {f["name"]: f for f in v.per_feature}
        assert by_name["Dam 5"]["existing_m3"] == pytest.approx(555.0)
        assert by_name["Dam 5"]["total_m3"] == pytest.approx(1281.0)
        assert by_name["Dam 5"]["terrain_m3"] == pytest.approx(726.0)   # still marginal

        # A swale cut into a slope ponds nothing beforehand: total collapses onto added.
        assert by_name["Swale 1"]["existing_m3"] == pytest.approx(0.0)
        assert by_name["Swale 1"]["total_m3"] == pytest.approx(416.0)

    def test_existing_ponding_defaults_to_zero_when_not_supplied(self):
        v = build_verification(
            analytic_by_name={"S1": 200.0},
            terrain_by_name={"S1": 180.0},
            baseline_total_m3=0.0,
            earthworks_total_m3=180.0,
            min_dims={"S1": 1.0},
            cell_size=1.0,
        )
        feat = v.per_feature[0]
        assert feat["existing_m3"] == pytest.approx(0.0)
        assert feat["total_m3"] == pytest.approx(180.0)

    def test_sub_cell_feature_has_no_total_to_claim(self):
        """No measured volume means no total either — inventing one would be precision
        the grid does not have."""
        v = build_verification(
            analytic_by_name={"S2": 100.0},
            terrain_by_name={"S2": 90.0},
            baseline_total_m3=0.0,
            earthworks_total_m3=90.0,
            min_dims={"S2": 0.3},
            cell_size=1.0,
            existing_by_name={"S2": 40.0},
        )
        feat = v.per_feature[0]
        assert feat["terrain_m3"] is None
        assert feat["total_m3"] is None

    def test_zero_analytic_total_no_divide(self):
        v = build_verification(
            analytic_by_name={"S1": 0.0},
            terrain_by_name={"S1": 5.0},
            baseline_total_m3=0.0,
            earthworks_total_m3=5.0,
            min_dims={},
            cell_size=1.0,
        )
        assert v.delta_pct == 0.0
        assert v.per_feature[0]["delta_pct"] is None  # analytic 0 → no % for the feature

    def test_terrain_total_floored_at_zero(self):
        v = build_verification(
            analytic_by_name={"S1": 100.0},
            terrain_by_name={"S1": 0.0},
            baseline_total_m3=50.0,
            earthworks_total_m3=40.0,   # earthworks < baseline → floored
            min_dims={"S1": 2.0},
            cell_size=1.0,
        )
        assert v.terrain_total_m3 == 0.0


class TestHtmlVerificationSurvivesConvergence:
    """The legacy report had its own verification block; the shared model has
    the volume ladder. Assert the same facts reach the HTML."""

    def _html(self, verification):
        from terrainflow_assessment.modules.report_html import render_html
        from terrainflow_assessment.modules.report_model import (
            ReportData,
            build_report,
        )
        return render_html(build_report(ReportData(
            baseline=_baseline(), verification=verification)))

    def test_ladder_and_caveats_present(self):
        v = VerificationResult(
            analytic_total_m3=1000.0, terrain_total_m3=950.0,
            unattributed_m3=12.0, caveats=["A caveat about attribution."],
            per_feature=[
                {"name": "Swale 1", "analytic_m3": 1000.0,
                 "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                 "terrain_m3": 950.0, "delta_pct": -13.6},
                {"name": "Swale 8", "analytic_m3": 80.0,
                 "geometric_m3": 100.0, "routing_only": True},
            ])
        html = self._html(v)
        assert "Checked against the ground" in html
        assert "Design storage" in html and "At this grid" in html
        assert "Δ vs grid" in html
        assert "n/a — sub-cell" in html          # sub-cell feature flagged
        assert "A caveat about attribution." in html

    def test_no_verification_says_so(self):
        html = self._html(None)
        assert "Checked against the ground" in html
        assert "nothing has been measured against the ground" in html


# ---------------------------------------------------------------------------
# format_live_assessment (Live Assessment panel readout)
# ---------------------------------------------------------------------------

def _balance(**kwargs):
    defaults = dict(
        capture_pct=85.0,
        total_inflow_m3=1000.0,
        total_captured_m3=850.0,
        site_exit_m3=150.0,
        total_capacity_m3=900.0,
        total_infiltration_m3=200.0,
        total_cut_m3=320.0,
        total_fill_m3=180.0,
        per_feature=[
            {"name": "Swale 1", "total_inflow_m3": 500.0, "stored_m3": 400.0,
             "capacity_m3": 450.0, "fill_pct": 88.9, "overflowed": False},
        ],
    )
    defaults.update(kwargs)
    return BalanceResult(**defaults)


class TestMiniBar:
    def test_mid_has_both_cells(self):
        html = _mini_bar(60.0, "#1e8449")
        assert "width='60%'" in html and "width='40%'" in html

    def test_zero_only_background(self):
        html = _mini_bar(0.0, "#1e8449")
        assert "#1e8449" not in html
        assert "width='100%'" in html

    def test_full_only_fill(self):
        html = _mini_bar(100.0, "#1e8449")
        assert "#1e8449" in html
        assert "d6dbdf" not in html

    def test_clamps_out_of_range(self):
        assert "width='100%'" in _mini_bar(140.0, "#1e8449")
        assert "#1e8449" not in _mini_bar(-5.0, "#1e8449")


class TestFormatLiveAssessment:
    def test_high_capture_green_headline(self):
        html = format_live_assessment(_balance(capture_pct=85.0), have_flow=True)
        assert "85%" in html
        assert "#1e8449" in html          # green traffic light
        assert "of storm runoff captured" in html

    def test_mid_capture_amber(self):
        html = format_live_assessment(_balance(capture_pct=55.0), have_flow=True)
        assert "#b9770e" in html

    def test_low_capture_red(self):
        html = format_live_assessment(_balance(capture_pct=20.0), have_flow=True)
        assert "#c0392b" in html

    def test_held_split_line(self):
        html = format_live_assessment(_balance(), have_flow=True)
        # 850 held = 650 stored + 200 soaked in; 150 leaves site
        assert "850 m³ held" in html
        assert "650 stored" in html
        assert "200 soaked in" in html
        assert "150 m³ leaves site" in html

    def test_per_feature_row_inflow_to_stored(self):
        html = format_live_assessment(_balance(), have_flow=True)
        assert "Swale 1" in html
        assert "500 → 400 m³" in html
        assert "89%" in html

    def test_overflowed_feature_flagged_full(self):
        r = _balance(per_feature=[
            {"name": "Basin 1", "total_inflow_m3": 900.0, "stored_m3": 300.0,
             "capacity_m3": 300.0, "fill_pct": 100.0, "overflowed": True},
        ])
        html = format_live_assessment(r, have_flow=True)
        assert "⚠ full" in html

    def test_fill_pct_100_flagged_even_without_overflow_flag(self):
        r = _balance(per_feature=[
            {"name": "Basin 1", "total_inflow_m3": 300.0, "stored_m3": 300.0,
             "capacity_m3": 300.0, "fill_pct": 100.0, "overflowed": False},
        ])
        html = format_live_assessment(r, have_flow=True)
        assert "⚠ full" in html

    def test_no_flow_shows_hint_and_capacities(self):
        html = format_live_assessment(_balance(), have_flow=False)
        assert "Run baseline analysis" in html
        assert "450 m³" in html            # per-feature capacity shown instead
        assert "→" not in html             # no inflow routing without flow data

    def test_no_flow_missing_capacity_key_defaults_zero(self):
        r = _balance(per_feature=[
            {"name": "Swale 1", "total_inflow_m3": 0.0, "stored_m3": 0.0,
             "fill_pct": 0.0, "overflowed": False},
        ])
        html = format_live_assessment(r, have_flow=False)
        assert "0 m³" in html

    def test_empty_features_skips_table(self):
        html = format_live_assessment(_balance(per_feature=[]), have_flow=True)
        assert "<table width='100%' cellspacing='0' cellpadding='1'" not in html

    def test_footer_totals_and_disclaimer(self):
        html = format_live_assessment(_balance(), have_flow=True)
        assert "Capacity 900 m³" in html
        assert "Cut 320" in html and "Fill 180" in html
        assert "Analytical estimate" in html


class TestVerificationSeparatesTheThreeGaps:
    """Δ must isolate burn error from resolution and freeboard.

    The old single delta summed all three, which is why "Verified · Δ −38%" told the
    user nothing they could act on.
    """

    BREAKDOWN = {
        "Swale 2": {
            "design": 112.0, "geometric": 140.0, "rasterisable": 187.0,
            "freeboard_m3": 28.0, "resolution_penalty_m3": 47.0,
        },
    }

    def _build(self, terrain, breakdowns=None):
        from terrainflow_assessment.modules.reporting import build_verification
        return build_verification(
            analytic_by_name={"Swale 2": 112.0},
            terrain_by_name={"Swale 2": terrain},
            baseline_total_m3=0.0,
            earthworks_total_m3=terrain,
            min_dims={"Swale 2": 1.0},
            cell_size=1.0,
            breakdowns=breakdowns,
        )

    def test_delta_measures_the_burn_against_what_the_grid_can_hold(self):
        v = self._build(183.0, self.BREAKDOWN)
        # 183 measured vs 187 representable → a small, honest burn delta.
        assert v.delta_pct == pytest.approx(-2.1, abs=0.2)

    def test_the_same_burn_looked_terrible_against_the_design_figure(self):
        """Without the breakdown the delta reverts to design capacity — +63%."""
        v = self._build(183.0, breakdowns=None)
        assert v.delta_pct > 50

    def test_per_feature_row_carries_all_four_numbers(self):
        row = self._build(183.0, self.BREAKDOWN).per_feature[0]
        assert row["analytic_m3"] == pytest.approx(112.0)
        assert row["geometric_m3"] == pytest.approx(140.0)
        assert row["rasterisable_m3"] == pytest.approx(187.0)
        assert row["terrain_m3"] == pytest.approx(183.0)
        assert row["freeboard_m3"] == pytest.approx(28.0)
        assert row["resolution_penalty_m3"] == pytest.approx(47.0)

    def test_a_large_resolution_penalty_is_named_in_the_caveats(self):
        v = self._build(183.0, self.BREAKDOWN)
        assert any("Swale 2" in c and "cannot hold its drawn section" in c
                   for c in v.caveats)

    def test_the_caveat_points_at_the_cell_size_and_names_the_capacity(self):
        """The burn now cuts the drawn section, so what is left is the grid.

        Before ``tapered_invert``, ``level_invert`` squared every drawn channel off to a
        full-depth rectangle whatever its cross-section, and the copy had to say so —
        a finer DEM would not have closed that gap. Now the burn tapers to the two
        widths the feature carries, so a material gap means the cells genuinely cannot
        hold the section, and the copy has to say *that* instead.
        """
        v = self._build(183.0, self.BREAKDOWN)
        caveat = [c for c in v.caveats if "Swale 2" in c][0]
        assert "cannot hold its drawn section" in caveat
        assert "Read Geometric" in caveat

    def test_a_grid_that_cannot_hold_the_section_is_described_the_other_way(self):
        under = {"Swale 2": {"design": 112.0, "geometric": 140.0,
                             "rasterisable": 100.0, "freeboard_m3": 28.0,
                             "resolution_penalty_m3": -40.0}}
        caveat = [c for c in self._build(99.0, under).caveats if "Swale 2" in c][0]
        assert "cannot hold its drawn section" in caveat
        assert "-29%" in caveat

    def test_a_small_resolution_penalty_is_not_flagged(self):
        small = {"Swale 2": {"design": 112.0, "geometric": 140.0,
                             "rasterisable": 143.0, "freeboard_m3": 28.0,
                             "resolution_penalty_m3": 3.0}}
        v = self._build(142.0, small)
        assert not any("cannot hold its drawn section" in c for c in v.caveats)

    def test_the_delta_caveat_states_what_is_being_compared(self):
        v = self._build(183.0, self.BREAKDOWN)
        assert any("burn issue" in c for c in v.caveats)

    def test_sub_cell_feature_still_makes_no_volume_claim(self):
        from terrainflow_assessment.modules.reporting import build_verification
        v = build_verification(
            analytic_by_name={"Swale 9": 50.0},
            terrain_by_name={"Swale 9": 40.0},
            baseline_total_m3=0.0, earthworks_total_m3=40.0,
            min_dims={"Swale 9": 0.3}, cell_size=1.0,
        )
        assert v.per_feature[0]["routing_only"] is True
        assert v.per_feature[0]["terrain_m3"] is None


class TestTheRowSaysWhenAtGridIsNotACapacity:
    """A near-zero Δ read as reassurance about a volume it never tested.

    Δ compares measured against *at-grid*, so a swale burned as a flat-floored trench
    24% above its drawn section reported "+0%" in green while the real gap — Geometric
    against At-grid — sat in a different pair of columns entirely. The row now carries
    the flag both renderers style from.

    The trigger is the measured gap rather than a width-vs-cell-size proxy: with no
    batter run the burn lays a rectangle however wide the footprint, so width predicts
    nothing about whether the section survived.
    """

    def _row(self, breakdown, terrain=183.0, min_dim=1.0, name="Swale 2"):
        from terrainflow_assessment.modules.reporting import build_verification
        return build_verification(
            analytic_by_name={name: 112.0},
            terrain_by_name={name: terrain},
            baseline_total_m3=0.0, earthworks_total_m3=terrain,
            min_dims={name: min_dim}, cell_size=1.0,
            breakdowns={name: breakdown} if breakdown else None,
        ).per_feature[0]

    OVERSTATED = {"design": 112.0, "geometric": 140.0, "rasterisable": 187.0,
                  "freeboard_m3": 28.0, "resolution_penalty_m3": 47.0}
    CLEAN = {"design": 112.0, "geometric": 140.0, "rasterisable": 143.0,
             "freeboard_m3": 28.0, "resolution_penalty_m3": 3.0}

    def test_a_material_gap_flags_the_row(self):
        row = self._row(self.OVERSTATED)
        assert row["section_overstated"] is True
        assert row["section_gap_pct"] == pytest.approx(47.0 / 140.0 * 100.0)

    def test_a_gap_the_other_way_flags_too(self):
        """The grid holding *less* than the drawn section is equally not a capacity."""
        under = dict(self.OVERSTATED, rasterisable=100.0,
                     resolution_penalty_m3=-40.0)
        row = self._row(under, terrain=99.0)
        assert row["section_overstated"] is True
        assert row["section_gap_pct"] < 0

    def test_a_small_gap_leaves_the_row_clean(self):
        row = self._row(self.CLEAN, terrain=142.0)
        assert row["section_overstated"] is False
        assert row["section_gap_pct"] == pytest.approx(3.0 / 140.0 * 100.0)

    def test_the_flag_and_the_caveats_never_disagree(self):
        from terrainflow_assessment.modules.reporting import build_verification
        v = build_verification(
            analytic_by_name={"Swale 2": 112.0, "Swale 3": 112.0},
            terrain_by_name={"Swale 2": 183.0, "Swale 3": 142.0},
            baseline_total_m3=0.0, earthworks_total_m3=325.0,
            min_dims={"Swale 2": 1.0, "Swale 3": 1.0}, cell_size=1.0,
            breakdowns={"Swale 2": self.OVERSTATED, "Swale 3": self.CLEAN},
        )
        flagged = {f["name"] for f in v.per_feature if f["section_overstated"]}
        named = {n for n in ("Swale 2", "Swale 3")
                 if any(n in c and "cannot hold its drawn section" in c
                        for c in v.caveats)}
        assert flagged == named == {"Swale 2"}

    def test_a_sub_cell_feature_is_never_flagged(self):
        """It claims no measured volume at all, so there is nothing to overstate."""
        row = self._row(self.OVERSTATED, min_dim=0.3)
        assert row["routing_only"] is True
        assert row["section_overstated"] is False

    def test_a_dam_is_never_flagged(self):
        """No drawn cross-section to be overstated against — see barrier_impounded."""
        row = self._row({"design": 2148.0, "geometric": 2148.0,
                         "rasterisable": 2148.0, "freeboard_m3": 0.0,
                         "resolution_penalty_m3": 0.0,
                         "barrier_impounded": True}, terrain=2148.0)
        assert row["barrier_impounded"] is True
        assert row["section_overstated"] is False

    def test_no_breakdown_means_nothing_to_compare(self):
        row = self._row(None)
        assert row["section_overstated"] is False
        assert row["section_gap_pct"] == pytest.approx(0.0)


class TestBarrierImpoundedVerification:
    """A dam holds water against the terrain, not inside a drawn cross-section.

    Field report: a dam whose burn matched its design exactly (2,148 m³ against
    2,148 m³) reported Δ +1712%, because it was pushed through the trapezoid path —
    geometric came back 0, and the delta was measured against whatever a channel of
    that width would rasterise to.
    """

    class _Dam:
        type, depth, width = "dam", 3.0, 4.0
        bottom_width_m, batter_run_m, companion_berm = 4.0, 0.0, False
        geometry = None
        capacity_m3 = 2148.0

    def test_a_dam_is_flagged_as_barrier_impounded(self):
        b = capacity_breakdown(self._Dam(), cell_size=1.0, n_cells=900)
        assert b["barrier_impounded"] is True

    def test_its_three_design_columns_collapse_to_one_number(self):
        """There is no drawn section to rasterise, so claiming a resolution penalty
        or a freeboard split would invent a comparison that was never made."""
        b = capacity_breakdown(self._Dam(), cell_size=1.0, n_cells=900)
        assert b["geometric"] == b["rasterisable"] == b["design"] == 2148.0
        assert b["freeboard_m3"] == 0.0
        assert b["resolution_penalty_m3"] == 0.0

    def test_a_dam_that_burns_to_its_design_reads_zero_delta(self):
        b = capacity_breakdown(self._Dam(), cell_size=1.0, n_cells=900)
        v = build_verification({"Dam 2": 2148.0}, {"Dam 2": 2148.0}, 0.0, 2148.0,
                               min_dims={"Dam 2": 20.0}, cell_size=1.0,
                               breakdowns={"Dam 2": b})
        assert v.per_feature[0]["delta_pct"] == pytest.approx(0.0, abs=0.5)

    def test_a_dam_that_misses_still_shows_it(self):
        """The flag must not become a way of always reporting success."""
        b = capacity_breakdown(self._Dam(), cell_size=1.0, n_cells=900)
        v = build_verification({"Dam 2": 2148.0}, {"Dam 2": 1074.0}, 0.0, 1074.0,
                               min_dims={"Dam 2": 20.0}, cell_size=1.0,
                               breakdowns={"Dam 2": b})
        assert v.per_feature[0]["delta_pct"] == pytest.approx(-50.0, abs=0.5)

    def test_a_swale_is_unaffected(self):
        """The flag keys off "no analytic section", so a type that has one must keep
        its freeboard and resolution split intact."""
        class _Line:
            length = 100.0

        class _Swale:
            type, depth, width = "swale", 0.5, 2.0
            bottom_width_m, batter_run_m, companion_berm = 1.0, 0.0, False
            geometry = _Line()
            capacity_m3 = 100.0

        b = capacity_breakdown(_Swale(), cell_size=1.0, n_cells=200)
        assert b["barrier_impounded"] is False
        assert b["geometric"] > 0
        assert b["freeboard_m3"] != 0.0


class TestLiveAssessmentAgainstRealBalanceOutput:
    """RPT-07: the reader subscripted `inflow_m3`; water_balance never emits it.

    Every render with flow data raised KeyError. The fixtures in this file had been
    written to match the reader rather than the producer, so the two agreed with each
    other and with nothing that ships. This test drives the real producer.
    """

    def test_renders_a_real_water_balance_result(self):
        from terrainflow_assessment.modules.simulation import EarthworkStore
        from terrainflow_assessment.modules.water_balance import run_water_balance

        store = EarthworkStore(
            name="Swale 1", ew_type="swale", capacity_m3=450.0, area_m2=120.0,
            infiltration_rate_mm_hr=0.0, id="ew-1", elevation=100.0,
        )
        store.inflow_m3 = 400.0

        result = run_water_balance([store], duration_hr=1.0, total_runoff_m3=1000.0)
        row = result.per_feature[0]
        assert "total_inflow_m3" in row
        assert "inflow_m3" not in row       # the key the reader used to subscript

        html = format_live_assessment(result, have_flow=True)
        assert "Swale 1" in html
        assert "400 → 400 m³" in html


class TestOneCaptureBand:
    """The panel scorecard, the live readout and the printed page grade the same
    percentage. Green on screen and amber on paper is a contradiction the reader
    cannot resolve, so the thresholds and colours have one definition."""

    def test_the_band_is_where_the_wording_is(self):
        from terrainflow_assessment.modules import reporting as R

        assert (R.CAPTURE_GOOD_PCT, R.CAPTURE_FAIR_PCT) == (80.0, 40.0)
        assert set(R.CAPTURE_COLOURS) == {"good", "warn", "bad"}

    def test_tone_and_colour_agree_across_the_band(self):
        from terrainflow_assessment.modules import reporting as R

        for pct, tone in ((100, "good"), (80, "good"), (79.9, "warn"),
                          (40, "warn"), (39.9, "bad"), (0, "bad")):
            assert R.capture_tone(pct) == tone, pct
            assert R.capture_colour(pct) == R.CAPTURE_COLOURS[tone], pct

    def test_the_report_grades_by_the_shared_band(self):
        from terrainflow_assessment.modules import reporting as R
        from terrainflow_assessment.modules.report_model import _capture_tone

        for pct in (0, 39.9, 40, 79.9, 80, 100):
            assert _capture_tone(pct) == R.capture_tone(pct), pct

    def test_moving_the_threshold_moves_every_readout(self):
        """The point of the exercise: one edit, not four."""
        from terrainflow_assessment.modules import reporting as R
        from terrainflow_assessment.modules.report_model import _capture_tone

        original = R.CAPTURE_GOOD_PCT
        try:
            R.CAPTURE_GOOD_PCT = 90.0
            assert R.capture_tone(85) == "warn"
            assert _capture_tone(85) == "warn"
        finally:
            R.CAPTURE_GOOD_PCT = original


class TestPoolsAreGroupedOnce:
    """Both views of the water — the total and the drawing — must agree about what
    a pool is. They agreed by hand-copied union-find; now there is one of each."""

    def _two_pools_one_shared(self):
        import numpy as np

        # Two ponded regions. Region 1 is touched by A and B; region 2 by B alone.
        depth = np.zeros((3, 7))
        depth[1, 0:2] = 0.5
        depth[1, 4:6] = 0.5
        a = np.zeros((3, 7), dtype=bool)
        a[1, 0] = True
        b = np.zeros((3, 7), dtype=bool)
        b[1, 1] = True
        b[1, 4] = True
        return depth, [("A", a), ("B", b)]

    def test_sharing_one_region_joins_the_features(self):
        from terrainflow_assessment.modules.reporting import group_pools

        depth, footprints = self._two_pools_one_shared()
        pools = group_pools(depth, footprints)
        assert pools.n_regions == 2
        roots = {pools.root_of_region[1], pools.root_of_region[2]}
        assert len(roots) == 1, "B's solo pool did not follow it into the joined set"
        (root,) = roots
        assert pools.members[root] == ("A", "B")

    def test_joining_is_transitive(self):
        import numpy as np

        from terrainflow_assessment.modules.reporting import group_pools

        depth = np.zeros((3, 9))
        depth[1, 0:2] = 0.5      # A + B
        depth[1, 4:6] = 0.5      # B + C
        a = np.zeros((3, 9), dtype=bool)
        a[1, 0] = True
        b = np.zeros((3, 9), dtype=bool)
        b[1, 1] = True
        b[1, 4] = True
        c = np.zeros((3, 9), dtype=bool)
        c[1, 5] = True
        pools = group_pools(depth, [("A", a), ("B", b), ("C", c)])
        assert pools.members[pools.root_of_region[1]] == ("A", "B", "C"), (
            "no cut separates A's water from C's, so they are one set")

    def test_a_region_touching_nothing_has_no_root(self):
        import numpy as np

        from terrainflow_assessment.modules.reporting import group_pools

        depth = np.zeros((3, 5))
        depth[1, 2] = 0.5
        pools = group_pools(depth, [("A", np.zeros((3, 5), dtype=bool))])
        assert pools.root_of_region[1] is None

    def test_both_attributions_see_the_same_sets(self):
        """The property the two docstrings promise each other."""
        from terrainflow_assessment.modules.reporting import (
            attribute_ponding_volume,
            group_pools,
        )

        depth, footprints = self._two_pools_one_shared()
        got = attribute_ponding_volume(depth, 1.0, footprints)
        pools = group_pools(depth, footprints)

        assert [g["names"] for g in got.groups] == [("A", "B")]
        assert set(pools.members.values()) == {("A", "B")}
        assert got.per_name == {"A": 0.0, "B": 0.0}, (
            "nothing is held alone once the two are joined")

    def test_a_prepared_grouping_gives_the_same_depths(self):
        """The playback passes one in per session rather than relabelling per frame."""
        import numpy as np

        from terrainflow_assessment.modules.reporting import (
            event_pond_depth,
            group_pools,
        )

        depth = np.zeros((3, 5))
        depth[1, 1:4] = 1.0
        ground = np.full((3, 5), 10.0)
        ground[1, 1:4] = 9.0
        mask = np.zeros((3, 5), dtype=bool)
        mask[1, 2] = True
        footprints = [("A", mask)]

        fresh = event_pond_depth(depth, ground, 1.0, footprints, {"A": 1.5})
        prepared = event_pond_depth(depth, ground, 1.0, footprints, {"A": 1.5},
                                    pools=group_pools(depth, footprints))
        assert np.allclose(fresh, prepared)


class TestUnattributedIsAParameter:
    """It was hard-coded to zero and monkey-patched by the one caller that knew
    better, so every other caller — the tests included — got a silent zero."""

    def test_it_reaches_the_result(self):
        from terrainflow_assessment.modules.reporting import build_verification

        result = build_verification({}, {}, 0.0, 40.0, {}, 1.0,
                                    unattributed_m3=12.5)
        assert result.unattributed_m3 == 12.5

    def test_it_still_defaults_to_zero(self):
        from terrainflow_assessment.modules.reporting import build_verification

        assert build_verification({}, {}, 0.0, 0.0, {}, 1.0).unattributed_m3 == 0.0


class TestNodataInsideAFootprint:
    """A hole in the DEM must not decide what the water around it does.

    A clipped DEM has NaN outside the clip, and an interior hole — a building
    removed, a lake masked out, a tile with no return — puts NaN *inside* a
    footprint. The three functions that measure ponding all have to answer the
    same way there, because the report prints their answers side by side: a
    volume that silently drops the cell, next to a depth raster with a nodata
    pixel in the middle of a pond, reads as two different measurements of the
    same water.

    The policy is **mask, don't propagate**: a cell with no ground under it
    holds no water, contributes nothing, and does not poison its neighbours.
    """

    def _pond(self):
        """A 4x4 pond 0.5 m deep in a 6x6 grid, one footprint over all of it."""
        ponding = np.zeros((6, 6))
        ponding[1:5, 1:5] = 0.5
        ground = np.full((6, 6), 100.0)
        ground[1:5, 1:5] = 99.5
        foot = np.zeros((6, 6), dtype=bool)
        foot[1:5, 1:5] = True
        return ponding, ground, [("Pond 1", foot)]

    def test_raster_volume_skips_a_nodata_depth(self):
        ponding, _, _ = self._pond()
        assert raster_ponding_volume(ponding, 1.0) == pytest.approx(8.0)
        ponding[2, 2] = np.nan
        # The one cell drops out; the rest is unchanged and the total is finite.
        assert raster_ponding_volume(ponding, 1.0) == pytest.approx(7.5)

    def test_attribution_skips_a_nodata_depth(self):
        ponding, _, foots = self._pond()
        ponding[2, 2] = np.nan
        result = attribute_ponding_volume(ponding, 1.0, foots)
        assert result.per_name["Pond 1"] == pytest.approx(7.5)
        assert result.unattributed_m3 == pytest.approx(0.0)

    def test_event_depth_leaves_no_nodata_in_the_raster(self):
        """The regression: one NaN bed cell used to come back out as a NaN pixel.

        ``spill`` is the maximum of bed+depth over the pool, so a single NaN made
        the ceiling NaN for every cell in it, and ``level - g`` wrote NaN back at
        the hole. The raster goes to a map and to ``.sum()``, and neither can say
        what a nodata pixel in the middle of a pond means.
        """
        ponding, ground, foots = self._pond()
        ground[2, 2] = np.nan
        got = event_pond_depth(ponding, ground, 1.0, foots, {"Pond 1": 4.0})
        assert not np.isnan(got).any(), "a nodata cell reached the depth raster"
        assert got[2, 2] == pytest.approx(0.0), "the hole itself holds nothing"

    def test_event_depth_still_places_the_water_it_was_given(self):
        """Masking the hole must not lose the volume — it is solved over the rest."""
        ponding, ground, foots = self._pond()
        clean = event_pond_depth(ponding, ground, 1.0, foots, {"Pond 1": 4.0})
        ground[2, 2] = np.nan
        holed = event_pond_depth(ponding, ground, 1.0, foots, {"Pond 1": 4.0})
        assert holed.sum() == pytest.approx(clean.sum(), rel=1e-6)

    def test_a_pool_that_is_all_nodata_is_skipped(self):
        ponding, ground, foots = self._pond()
        ground[1:5, 1:5] = np.nan
        got = event_pond_depth(ponding, ground, 1.0, foots, {"Pond 1": 4.0})
        assert not np.isnan(got).any()
        assert got.sum() == pytest.approx(0.0)
