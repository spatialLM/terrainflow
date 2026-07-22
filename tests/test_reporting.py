"""Tests for terrainflow_assessment/modules/reporting.py"""
import os

import numpy as np
import pytest

from terrainflow_assessment.modules.reporting import (
    BaselineReport,
    ComparisonResult,
    PostInterventionReport,
    VerificationResult,
    _build_fill_timeline_chart,
    _build_hydrograph_chart,
    _fig_to_base64,
    attribute_ponding_volume,
    build_verification,
    compare,
    export_html,
    raster_ponding_volume,
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
