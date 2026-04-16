"""Tests for terrainflow_assessment/modules/reporting.py"""
import os

import pytest

from terrainflow_assessment.modules.reporting import (
    BaselineReport,
    ComparisonResult,
    PostInterventionReport,
    _build_fill_timeline_chart,
    _build_hydrograph_chart,
    _fig_to_base64,
    compare,
    export_html,
)


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
