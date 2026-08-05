"""Tests for terrainflow_assessment/modules/reporting.py"""
import os

import numpy as np
import pytest

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
    export_html,
    format_live_assessment,
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

class TestExportHTML:
    def test_creates_file(self, tmp_path):
        out = str(tmp_path / "report.html")
        result = export_html(
            ComparisonResult(baseline=_baseline(), post=_post()),
            out,
        )
        assert os.path.exists(out)
        assert result == out

    def test_file_contains_site_name(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(
            ComparisonResult(baseline=_baseline(site_name="MyFarm"), post=_post()),
            out,
        )
        content = open(out, encoding="utf-8").read()
        assert "MyFarm" in content

    def test_html_has_doctype(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=_post()), out)
        content = open(out, encoding="utf-8").read()
        assert "<!DOCTYPE html>" in content

    def test_html_has_sections(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=_post()), out)
        content = open(out, encoding="utf-8").read()
        for section in ["Site Summary", "Before / After", "Earthwork Summary", "Methodology"]:
            assert section in content

    def test_overflow_yes_shown(self, tmp_path):
        ew = _post().earthwork_summary[0].copy()
        ew["overflowed"] = True
        ew["first_overflow_hr"] = 0.6
        p = _post(earthwork_summary=[ew])
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=p), out)
        content = open(out, encoding="utf-8").read()
        assert "Yes" in content

    def test_no_overflow_shown(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=_post()), out)
        content = open(out, encoding="utf-8").read()
        assert "No" in content

    def test_methodology_text_included(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(
            ComparisonResult(baseline=_baseline(), post=_post()),
            out,
            methodology_text="Custom methodology note.",
        )
        content = open(out, encoding="utf-8").read()
        assert "Custom methodology note." in content

    def test_none_baseline_no_crash(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=None, post=None), out)
        assert os.path.exists(out)

    def test_exit_points_in_html(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=_post()), out)
        content = open(out, encoding="utf-8").read()
        assert "Exit Point" in content

    def test_cut_fill_summary_in_html(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=_post()), out)
        content = open(out, encoding="utf-8").read()
        assert "Net cut" in content


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
        per_name, unattr = attribute_ponding_volume(
            pond, 1.0, [("A", self._mask_at([(2, 1)]))]
        )
        assert per_name["A"] == pytest.approx(3.0)
        assert unattr == pytest.approx(0.0)

    def test_region_touching_no_footprint_is_unattributed(self):
        pond = self._pond_band()
        per_name, unattr = attribute_ponding_volume(
            pond, 1.0, [("A", self._mask_at([(0, 0)]))]
        )
        assert per_name["A"] == pytest.approx(0.0)
        assert unattr == pytest.approx(3.0)

    def test_region_goes_to_largest_overlap(self):
        pond = self._pond_band()
        footprints = [
            ("A", self._mask_at([(2, 1)])),           # overlap 1
            ("B", self._mask_at([(2, 2), (2, 3)])),   # overlap 2 → wins
        ]
        per_name, unattr = attribute_ponding_volume(pond, 1.0, footprints)
        assert per_name["B"] == pytest.approx(3.0)
        assert per_name["A"] == pytest.approx(0.0)

    def test_no_footprints_all_unattributed(self):
        pond = self._pond_band()
        per_name, unattr = attribute_ponding_volume(pond, 1.0, [])
        assert per_name == {}
        assert unattr == pytest.approx(3.0)


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


class TestExportHtmlVerification:
    def _summary_with_terrain(self):
        base = _post().earthwork_summary[0].copy()
        base.update(terrain_ponding_m3=180.0, capacity_delta_pct=-10.0, routing_only=False)
        sub = {
            "name": "Swale 2", "type": "swale", "capacity_m3": 50.0,
            "peak_fill_pct": 40.0, "overflowed": False, "first_overflow_hr": None,
            "terrain_ponding_m3": None, "capacity_delta_pct": None, "routing_only": True,
        }
        return [base, sub]

    def _comparison(self):
        v = VerificationResult(
            analytic_total_m3=250.0, terrain_total_m3=230.0, delta_m3=-20.0,
            delta_pct=-8.0, unattributed_m3=5.0,
            per_feature=[], caveats=["Site total is robust; per-feature is indicative."],
        )
        c = ComparisonResult(
            baseline=_baseline(),
            post=_post(earthwork_summary=self._summary_with_terrain()),
            verification=v,
        )
        return c

    def test_verification_section_and_columns(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(self._comparison(), out)
        content = open(out, encoding="utf-8").read()
        assert "Non-circular Verification" in content
        assert "Terrain Ponding (m³)" in content
        assert "Terrain-derived storage" in content
        assert "routing-only" in content          # sub-cell feature flagged
        assert "180.0" in content                  # resolvable feature terrain volume
        assert "Unattributed terrain ponding" in content

    def test_no_verification_still_renders(self, tmp_path):
        out = str(tmp_path / "report.html")
        export_html(ComparisonResult(baseline=_baseline(), post=_post()), out)
        content = open(out, encoding="utf-8").read()
        # No verification attached → section omitted, table still present
        assert "Non-circular Verification" not in content
        assert "Earthwork Summary" in content


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
        assert any("Swale 2" in c and "grid represents it" in c for c in v.caveats)

    def test_a_small_resolution_penalty_is_not_flagged(self):
        small = {"Swale 2": {"design": 112.0, "geometric": 140.0,
                             "rasterisable": 143.0, "freeboard_m3": 28.0,
                             "resolution_penalty_m3": 3.0}}
        v = self._build(142.0, small)
        assert not any("grid represents it" in c for c in v.caveats)

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
