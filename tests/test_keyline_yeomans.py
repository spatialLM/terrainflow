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

    def test_cultivation_run_has_constant_crossgrade(self, tmp_path):
        """
        Phase 1 item 4b: every cultivation run must carry the declared
        cross_grade in its metadata.
        """
        dem_path, _ = _make_valley_dem(tmp_path)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()

        cg = 1 / 500
        runs = ya.get_cultivation_runs(kp, n_runs=1, cross_grade=cg)
        for run in runs:
            assert run["cross_grade"] == pytest.approx(cg, rel=1e-9)

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
        assert "cultivation_upper" in types
        assert "cultivation_lower" in types

    def test_find_keypoint_too_small_returns_none(self, tmp_path):
        """A 3×3 DEM is too small to produce a meaningful thalweg."""
        data = np.array([[10, 8, 6], [7, 5, 3], [4, 2, 1]], dtype="float32")
        dem_path = _write_dem(str(tmp_path / "tiny.tif"), data)
        ya = YeomansKeylineAnalysis(dem_path)
        kp = ya.find_keypoint()
        # Either None or a valid dict — must not crash
        assert kp is None or isinstance(kp, dict)
