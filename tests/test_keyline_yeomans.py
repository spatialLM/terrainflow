"""
Phase 1 regression tests for YeomansKeylineAnalysis and the
DrainageLineAnalysis / KeylineAnalysis rename.

Phase 1 item 4:
  4a. KeylineAnalysis → DrainageLineAnalysis (with deprecated alias)
  4b. YeomansKeylineAnalysis: thalweg extraction, keypoint detection,
      cultivation runs with constant cross-grade.
"""
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from terrainflow_assessment.modules.keypoint_analysis import (
    DrainageLineAnalysis,
    KeylineAnalysis,
    YeomansKeylineAnalysis,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dem(path, data, cell_size=1.0, crs="EPSG:32632", nodata=-9999.0):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _make_acc(path, shape, cell_size=1.0, crs="EPSG:32632"):
    """Minimal accumulation raster (all ones) for DrainageLineAnalysis init."""
    data = np.ones(shape, dtype="float32")
    h, w = shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs, transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


# ---------------------------------------------------------------------------
# 4a — rename + deprecation alias
# ---------------------------------------------------------------------------

class TestDrainageLineAnalysisRename:
    def test_drainage_line_analysis_is_class(self):
        assert isinstance(DrainageLineAnalysis, type)

    def test_keyline_alias_emits_deprecation_warning(self, tmp_path):
        """Instantiating KeylineAnalysis must emit DeprecationWarning."""
        dem_path = _write_dem(
            str(tmp_path / "dem.tif"),
            np.fromfunction(lambda r, c: 100.0 - r * 2.0, (20, 20)),
        )
        acc_path = _make_acc(str(tmp_path / "acc.tif"), (20, 20))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            KeylineAnalysis(dem_path, acc_path)
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1
        assert "DrainageLineAnalysis" in str(dep_warnings[0].message)

    def test_keyline_alias_is_subclass_of_drainage_line(self):
        assert issubclass(KeylineAnalysis, DrainageLineAnalysis)


# ---------------------------------------------------------------------------
# 4b — YeomansKeylineAnalysis: keypoint detection
# ---------------------------------------------------------------------------

def _make_valley_dem(tmp_path, steep_rows=22, gentle_rows=18):
    """
    40×40 valley DEM centred on col 20.

    Row 0 → steep_rows:        slope = 8 m per row
    Row steep_rows → end:      slope = 0.5 m per row
    Lateral: parabolic valley (deepest at col 20)

    Known inflection: row ``steep_rows`` (index 22 for the default).
    """
    total_rows = steep_rows + gentle_rows
    data = np.zeros((total_rows, 40), dtype="float32")
    for r in range(total_rows):
        lateral = ((np.arange(40) - 19.5) ** 2) * 0.05
        if r < steep_rows:
            row_base = 300.0 - r * 8.0
        else:
            row_base = 300.0 - steep_rows * 8.0 - (r - steep_rows) * 0.5
        data[r, :] = row_base + lateral
    path = _write_dem(str(tmp_path / "valley.tif"), data)
    return path, steep_rows


class TestYeomansKeylineAnalysis:

    def test_returns_keypoint_dict(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        assert kp is not None
        for key in ("x", "y", "elevation", "row", "col", "arc_length_m"):
            assert key in kp

    def test_keypoint_on_synthetic_valley(self, tmp_path):
        """
        Phase 1 item 4b: keypoint must be within 1 cell of the known
        inflection row (row 22) on the synthetic valley DEM.
        """
        dem_path, known_inflection_row = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        assert kp is not None

        detected_row = kp["row"]
        assert abs(detected_row - known_inflection_row) <= 1, (
            f"Keypoint detected at row {detected_row}, "
            f"expected near row {known_inflection_row} (±1 cell)"
        )

    def test_cultivation_runs_have_correct_count(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        runs = ya.get_cultivation_runs(kp, n_runs=2)
        # 2 above + keyline + 2 below = 5 total
        assert len(runs) == 5

    def test_every_run_reports_the_drift_it_actually_achieves(self, tmp_path):
        """The measurement that replaced the control which never did anything.

        `cross_grade` was echoed onto every run and read by nothing, so a user could
        set 1:50 or 1:5000 and get byte-identical lines — while the map carried a
        column asserting the grade the geometry did not have. Now the geometry is
        measured and the number is an output.
        """
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()

        runs = ya.get_cultivation_runs(kp, n_runs=1)
        assert runs
        for run in runs:
            assert "cross_grade" not in run, "the inert field must be gone, not renamed"
            assert "drift_1_in_n" in run and "drift_fall_m" in run
            assert "over_limit" in run

    def test_the_grade_input_is_a_limit_that_flags_rather_than_a_generator(self, tmp_path):
        """Two limits, same geometry, different flags — which is what a limit means."""
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()

        loose = ya.get_cultivation_runs(kp, n_runs=1, max_grade_n=1)
        strict = ya.get_cultivation_runs(kp, n_runs=1, max_grade_n=100_000)

        loose_xy = [[(x, y) for x, y, *_ in r["geometry"].coords] for r in loose]
        strict_xy = [[(x, y) for x, y, *_ in r["geometry"].coords] for r in strict]
        assert loose_xy == strict_xy, "a limit must not move the geometry"

        drifting = [r for r in strict if r["drift_1_in_n"] is not None]
        if drifting:
            assert any(r["over_limit"] for r in strict), (
                "an impossibly strict limit should flag something")
        assert not any(r["over_limit"] for r in loose)

    def test_the_old_cross_grade_argument_warns(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        with pytest.warns(DeprecationWarning):
            ya.get_cultivation_runs(kp, n_runs=1, cross_grade=1 / 500)

    def test_cultivation_run_geometry_has_z_coords(self, tmp_path):
        """Cultivation run LineStrings must be 3D (Z = elevation with grade)."""
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        runs = ya.get_cultivation_runs(kp, n_runs=1)
        keyline = next(r for r in runs if r["line_type"] == "keyline")
        geom = keyline["geometry"]
        # 3D geometry: coordinates have length 3
        coords = list(geom.coords)
        assert all(len(pt) == 3 for pt in coords), "Cultivation run must be 3D"

    def test_keyline_is_in_runs(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        runs = ya.get_cultivation_runs(kp, n_runs=1)
        types = [r["line_type"] for r in runs]
        assert "keyline" in types
        # Yeomans names two patterns and stresses that most of a landscape is the
        # second; the guides are now labelled by which one they belong to, and by
        # MEASURED elevation rather than by the sign of the offset.
        assert "ridge_guide" in types
        assert "valley_guide" in types

    def test_a_pattern_can_be_asked_for_on_its_own(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()

        valley = ya.get_cultivation_runs(kp, n_runs=2, pattern="valley")
        assert {r["line_type"] for r in valley} <= {"keyline", "valley_guide"}

        ridge = ya.get_cultivation_runs(kp, n_runs=2, pattern="ridge")
        assert {r["line_type"] for r in ridge} <= {"keyline", "ridge_guide"}

    def test_guides_are_labelled_by_measured_elevation(self, tmp_path):
        """`offset_curve`'s sign means left-of-travel, not uphill.

        The traced contour's winding comes from find_contours and is never normalised,
        so labelling by the sign of the offset assigned upper and lower arbitrarily.
        """
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        runs = ya.get_cultivation_runs(kp, n_runs=2)

        keyline = next(r for r in runs if r["line_type"] == "keyline")
        for run in runs:
            if run["line_type"] == "ridge_guide":
                assert run["elevation"] >= keyline["elevation"]
            elif run["line_type"] == "valley_guide":
                assert run["elevation"] <= keyline["elevation"]

    def test_keyline_traces_curved_contour(self, tmp_path):
        """The keyline should follow the valley contour (many vertices), not a
        straight 2-point line, on a curved (parabolic) valley."""
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        runs = ya.get_cultivation_runs(kp, n_runs=1)
        keyline = next(r for r in runs if r["line_type"] == "keyline")
        assert len(list(keyline["geometry"].coords)) > 2

    def test_guides_are_offset_from_keyline(self, tmp_path):
        """Cultivation guides must be geometrically distinct from the keyline
        (parallel offsets), not identical lines."""
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        runs = ya.get_cultivation_runs(kp, n_runs=1, spacing_m=5.0)
        keyline = next(r for r in runs if r["line_type"] == "keyline")
        upper = next(r for r in runs if r["line_type"] != "keyline")
        # 2-D footprints should differ once offset.
        kl_xy = [(x, y) for x, y, *_ in keyline["geometry"].coords]
        up_xy = [(x, y) for x, y, *_ in upper["geometry"].coords]
        assert kl_xy != up_xy

    def test_find_keypoint_too_small_returns_none(self, tmp_path):
        """A 3×3 DEM is too small to produce a meaningful thalweg."""
        data = np.array([[10, 8, 6], [7, 5, 3], [4, 2, 1]], dtype="float32")
        dem_path = _write_dem(str(tmp_path / "tiny.tif"), data)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        # Either None or a valid dict — must not crash
        assert kp is None or isinstance(kp, dict)


# ---------------------------------------------------------------------------
# Keyline geometry helpers (parallel-offset construction)
# ---------------------------------------------------------------------------

class TestKeylineHelpers:
    def test_fallback_keyline_returns_segment(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        seg = ya._fallback_keyline(20, 20)
        assert len(seg) == 2
        assert all(len(p) == 2 for p in seg)

    def test_trace_keyline_absent_level_returns_none(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        # An elevation far outside the DEM range has no contour.
        assert ya._trace_keyline(1.0e6, 20, 20) is None

    def test_offset_line_happy_path(self, tmp_path):
        from shapely.geometry import LineString
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        line = LineString([(0, 0), (10, 0), (20, 0)])
        off = ya._offset_line(line, 5.0, 20, 20)
        assert off is not None and off.length > 0

    def test_offset_line_fallback_translate(self, tmp_path):
        from shapely.geometry import LineString
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        # Zero-length line → offset_curve empty → translate fallback returns a line.
        degenerate = LineString([(5, 5), (5, 5)])
        off = ya._offset_line(degenerate, 5.0, 20, 20)
        assert off is not None

    def test_sample_dem_out_of_bounds_returns_default(self, tmp_path):
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        assert ya._sample_dem(1.0e9, 1.0e9, 42.0) == 42.0

    def test_get_cultivation_runs_uses_fallback_keyline(self, tmp_path):
        """When contour tracing yields nothing (elevation off the DEM), the run
        builder still returns guides via the straight fallback keyline."""
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        kp = dict(kp)
        kp["elevation"] = 1.0e6  # no contour at this level → fallback path
        runs = ya.get_cultivation_runs(kp, n_runs=1)
        assert any(r["line_type"] == "keyline" for r in runs)


# ---------------------------------------------------------------------------
# Maths-audit regressions
# ---------------------------------------------------------------------------

class TestMathsAuditRegressions:
    def test_fallback_keyline_runs_along_the_contour_not_down_the_fall_line(
            self, tmp_path):
        """KPA-33: the fallback direction was −∇z in map space — the fall line.

        On a south-facing slope (elevation falling as row increases) the contour
        runs east-west. The old (−dz_dc, dz_dr) drew it north-south, 90° out.
        """
        dem = np.fromfunction(lambda r, c: 100.0 - r * 2.0, (20, 20))
        dem_path = _write_dem(str(tmp_path / "south.tif"), dem)
        ya = YeomansKeylineAnalysis(dem_path)
        (x0, y0), (x1, y1) = ya._fallback_keyline(10, 10)
        assert abs(x1 - x0) > 0
        assert abs(y1 - y0) == pytest.approx(0.0, abs=1e-6)   # runs east-west

    def test_valley_width_is_measured_near_the_western_edge(self, tmp_path):
        """KPA-21: `break` on nc < 0 stopped the scan on its FIRST iteration.

        The scan starts 200 columns left of centre, so any candidate within 200
        columns of the west edge reported width 0 m — which maximises the
        acc/(width+1) dam score and dragged pond sites to the raster's left edge.
        """
        dem = np.full((30, 30), 50.0, dtype="float32")
        dem_path = _write_dem(str(tmp_path / "flat.tif"), dem)
        acc_path = _make_acc(str(tmp_path / "acc.tif"), (30, 30))
        dla = DrainageLineAnalysis(dem_path, acc_path)
        near_edge = dla._valley_cross_width(15, 2, 60.0)
        mid = dla._valley_cross_width(15, 15, 60.0)
        assert near_edge > 0
        assert near_edge == pytest.approx(mid)   # uniform ground → same width

    def test_ridges_narrower_than_six_cells_survive_thinning(self, tmp_path):
        """KPA-14: three erosion passes delete anything ≤6 cells across."""
        from terrainflow_assessment.modules.keypoint_analysis import (
            _thin_to_centreline,
        )
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 2:18] = True              # a one-cell-wide, 16-cell-long ridge
        thinned = _thin_to_centreline(mask)
        assert thinned.any()
        assert thinned.sum() >= 12         # length preserved, not annihilated

    def test_thalweg_profile_ignores_nodata_instead_of_calling_it_sea_level(
            self, tmp_path):
        """NEW-W8-01: substituting 0.0 m for NaN puts a full-terrain-height cliff
        in the profile, and the keypoint is the argmax of its 2nd derivative."""
        dem = np.fromfunction(lambda r, c: 100.0 - r * 2.0, (30, 30))
        clean = YeomansKeylineAnalysis(
            _write_dem(str(tmp_path / "clean.tif"), dem)).find_keypoint()
        holed = dem.copy()
        holed[15, :] = np.nan
        pitted = YeomansKeylineAnalysis(
            _write_dem(str(tmp_path / "holed.tif"), holed)).find_keypoint()
        if clean is not None and pitted is not None:
            # A single nodata row must not relocate the keypoint to it.
            assert pitted["elevation"] > 0.0


# ---------------------------------------------------------------------------
# Per-primary-valley keypoints — the scope fix
# ---------------------------------------------------------------------------

def _two_valley_dem(path, n=120):
    """Two parallel valleys draining south, each with a real break in its floor.

    The floor of each valley falls steeply for the top half and gently for the bottom
    half, so there is a genuine steep-above / gentler-below break — Yeomans' keypoint —
    at the join. Two of them, so a per-valley pass must find two.
    """
    r, c = np.mgrid[0:n, 0:n].astype("float64")
    # Floor profile: steep to row 60, then gentle.
    fall = np.where(r < 60, 0.30 * r, 0.30 * 60 + 0.05 * (r - 60))
    # Two V-shaped valleys centred on columns 30 and 90.
    across = np.minimum(np.abs(c - 30), np.abs(c - 90))
    return _write_dem(path, 200.0 - fall + 0.5 * across, cell_size=1.0)


def _uniform_valley_dem(path, n=120):
    """One valley whose floor falls at a constant grade — so it has NO keypoint."""
    r, c = np.mgrid[0:n, 0:n].astype("float64")
    across = np.abs(c - n // 2)
    return _write_dem(path, 200.0 - 0.20 * r + 0.5 * across, cell_size=1.0)


class TestKeypointsPerPrimaryValley:
    def test_two_valleys_give_two_keypoints(self, tmp_path):
        """The scope fix. `find_keypoint` walks the single largest stream, so it
        answers the right question about the wrong feature — once."""
        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)
        keypoints, _skipped = ya.find_keypoints(max_valleys=8)
        assert len(keypoints) >= 2, (
            f"expected a keypoint per primary valley, got {len(keypoints)}")

        # They should be in different valleys, not two picks on one.
        cols = sorted(kp["col"] for kp in keypoints[:2])
        assert cols[1] - cols[0] > 20, f"both keypoints landed in one valley: {cols}"

    def test_a_uniform_valley_has_no_keypoint_and_says_so(self, tmp_path):
        """`argmax` always returns something; a constant-gradient floor has no break.

        Harmless while one keypoint was found on one stem. Run per valley it would
        fabricate them at scale, which is why the prominence bar exists.
        """
        dem_path = _uniform_valley_dem(str(tmp_path / "uniform.tif"))
        ya = YeomansKeylineAnalysis(dem_path)
        keypoints, skipped = ya.find_keypoints(max_valleys=8)
        assert keypoints == [], (
            "a uniform-gradient valley has no steep-to-gentle break, so a keypoint "
            "here is invented")
        assert skipped, "a refusal must be reported, not returned as a shorter list"
        assert "grade" in skipped[0]

    def test_keypoints_carry_their_catchment_and_a_label(self, tmp_path):
        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)
        keypoints, _ = ya.find_keypoints(max_valleys=4)
        assert keypoints
        for kp in keypoints:
            assert kp["catchment_ha"] > 0
            assert kp["label"]
            assert kp["_row"] == kp["row"] and kp["_col"] == kp["col"]

    def test_the_cap_is_honoured(self, tmp_path):
        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)
        keypoints, _ = ya.find_keypoints(max_valleys=1)
        assert len(keypoints) <= 1

    def test_keypoint_on_path_keeps_the_verified_criterion(self, tmp_path):
        """The maths is unchanged: the strongest easing of the valley floor.

        Only its scope moved. A profile that steepens throughout has its keypoint at
        the *least* steepening, and one that eases has it at the break.
        """
        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)
        path = [(r, 30) for r in range(5, 115)]
        kp = ya.keypoint_on_path(path)
        assert kp is not None
        # The break is at row 60; allow for the smoothing window.
        assert 40 <= kp["row"] <= 80, kp
        assert kp["slope_ease"] > 0


class TestOffsetParts:
    def test_a_fold_is_refused_rather_than_returned(self, tmp_path):
        """`offset_curve` self-intersects wherever the offset exceeds the local radius
        of curvature — which is every tight valley head, which is where a keyline is."""
        from shapely.geometry import LineString

        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)

        # A hairpin with a 2 m radius, offset by 20 m on the inside.
        hairpin = LineString([(0, 0), (40, 0), (42, 2), (40, 4), (0, 4)])
        parts = ya.offset_parts(hairpin, -20.0)
        for part in parts:
            n = 12
            dists = [part.interpolate(part.length * i / n).distance(hairpin)
                     for i in range(n + 1)]
            assert max(dists) - min(dists) < 20.0, (
                "a folded lobe came back as a guide")

    def test_both_limbs_of_a_split_guide_survive(self, tmp_path):
        """A guide that splits around a spur is two real plough runs, not one."""
        from shapely.geometry import LineString

        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)

        # A deep notch: offsetting outward splits the result in two.
        notched = LineString([(0, 0), (20, 0), (25, 30), (30, 0), (50, 0)])
        parts = ya.offset_parts(notched, -6.0)
        assert len(parts) >= 1
        assert all(p.length > 0 for p in parts)

    def test_a_straight_line_offsets_to_one_clean_part(self, tmp_path):
        from shapely.geometry import LineString

        dem_path = _two_valley_dem(str(tmp_path / "two.tif"))
        ya = YeomansKeylineAnalysis(dem_path)
        parts = ya.offset_parts(LineString([(0, 0), (100, 0)]), 5.0)
        assert len(parts) == 1
        assert parts[0].length == pytest.approx(100.0, rel=0.05)
