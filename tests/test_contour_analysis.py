"""Tests for terrainflow_assessment/modules/contour_analysis.py"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box

from terrainflow_assessment.modules.contour_analysis import (
    ContourFeature,
    _extract_contours_scipy,
    filter_by_slope,
    rank_by_flow_crossing,
    clip_to_usable_area,
    analyse_contours,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_raster(path, data, cell_size=1.0, crs="EPSG:32632", nodata=None):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    kwargs = dict(
        driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs, transform=transform,
    )
    if nodata is not None:
        kwargs["nodata"] = nodata
    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _make_dem(tmp_path, slope=True):
    """20×20 DEM, optionally sloped."""
    if slope:
        data = np.fromfunction(lambda r, c: 100.0 - r * 2.0, (20, 20)).astype("float32")
    else:
        data = np.full((20, 20), 50.0, dtype="float32")
    return _write_raster(str(tmp_path / "dem.tif"), data)


def _make_acc(tmp_path):
    """20×20 accumulation raster with a high column in the centre."""
    data = np.ones((20, 20), dtype="float32") * 10.0
    data[:, 10] = 5000.0
    return _write_raster(str(tmp_path / "acc.tif"), data)


def _make_feature(elevation=50.0, length_m=100.0, peak_acc=0.0, slope=5.0):
    geom = LineString([(0, elevation), (length_m, elevation)])
    return ContourFeature(
        geometry=geom,
        elevation=elevation,
        peak_acc=peak_acc,
        mean_slope_deg=slope,
        length_m=length_m,
    )


# ---------------------------------------------------------------------------
# ContourFeature
# ---------------------------------------------------------------------------

class TestContourFeature:
    def test_defaults(self):
        geom = LineString([(0, 0), (10, 0)])
        cf = ContourFeature(geometry=geom, elevation=50.0)
        assert cf.rank is None
        assert cf.peak_acc == 0.0
        assert cf.mean_slope_deg == 0.0
        assert cf.selected is True

    def test_label_without_cell_area(self):
        cf = _make_feature(elevation=42.5, peak_acc=1234.0)
        label = cf.label
        assert "42.5" in label
        assert "1,234" in label or "1234" in label  # locale may add comma

    def test_label_with_cell_area(self):
        geom = LineString([(0, 0), (100, 0)])
        cf = ContourFeature(
            geometry=geom, elevation=55.0,
            peak_acc=10000.0, cell_area_m2=1.0, runoff_mm=None
        )
        label = cf.label
        assert "55.0" in label
        assert "ha" in label

    def test_label_with_cell_area_and_runoff(self):
        geom = LineString([(0, 0), (100, 0)])
        cf = ContourFeature(
            geometry=geom, elevation=60.0,
            peak_acc=5000.0, cell_area_m2=4.0, runoff_mm=20.0
        )
        label = cf.label
        assert "m³" in label

    def test_label_with_rank(self):
        cf = _make_feature()
        cf.rank = 3
        assert "#3" in cf.label

    def test_label_no_rank_no_prefix(self):
        cf = _make_feature()
        cf.rank = None
        assert "#" not in cf.label


# ---------------------------------------------------------------------------
# _extract_contours_scipy (fallback)
# ---------------------------------------------------------------------------

class TestExtractContoursScipy:
    def test_returns_list(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        result = _extract_contours_scipy(dem_path, interval_m=5.0)
        assert isinstance(result, list)

    def test_features_have_geometry(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        result = _extract_contours_scipy(dem_path, interval_m=5.0)
        for feat in result:
            assert feat.geometry is not None
            assert isinstance(feat.geometry, LineString)

    def test_elevations_are_multiples_of_interval(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        result = _extract_contours_scipy(dem_path, interval_m=4.0)
        for feat in result:
            assert abs(feat.elevation % 4.0) < 0.5  # within half an interval

    def test_flat_dem_no_contours(self, tmp_path):
        dem_path = _make_dem(tmp_path, slope=False)
        result = _extract_contours_scipy(dem_path, interval_m=1.0)
        assert isinstance(result, list)  # may be empty

    def test_nodata_dem_returns_empty_or_list(self, tmp_path):
        data = np.full((10, 10), -9999.0, dtype="float32")
        path = _write_raster(str(tmp_path / "nodata.tif"), data, nodata=-9999.0)
        result = _extract_contours_scipy(path, interval_m=1.0)
        assert isinstance(result, list)

    def test_with_nodata_cells_handled(self, tmp_path):
        data = np.fromfunction(lambda r, c: 50.0 - r, (20, 20)).astype("float32")
        data[0:3, 0:3] = -9999.0
        path = _write_raster(str(tmp_path / "partial_nodata.tif"), data, nodata=-9999.0)
        result = _extract_contours_scipy(path, interval_m=2.0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# extract_contours (with gdal_contour fallback to scipy)
# ---------------------------------------------------------------------------

class TestExtractContours:
    def test_returns_features(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import extract_contours
        dem_path = _make_dem(tmp_path)
        result = extract_contours(dem_path, interval_m=5.0)
        assert isinstance(result, list)

    def test_features_have_elevation(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import extract_contours
        dem_path = _make_dem(tmp_path)
        result = extract_contours(dem_path, interval_m=5.0)
        for feat in result:
            assert isinstance(feat.elevation, float)

    def test_with_output_path(self, tmp_path):
        """Pass output_path; should not raise even if gdal_contour is absent."""
        from terrainflow_assessment.modules.contour_analysis import extract_contours
        dem_path = _make_dem(tmp_path)
        out = str(tmp_path / "contours.gpkg")
        result = extract_contours(dem_path, interval_m=5.0, output_path=out)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# filter_by_slope
# ---------------------------------------------------------------------------

class TestFilterBySlope:
    def test_returns_subset(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        features = [_make_feature(elevation=e, slope=0.0) for e in [62, 64, 66, 68]]
        result = filter_by_slope(features, dem_path, max_slope_deg=90.0)
        assert len(result) <= len(features)

    def test_steep_feature_removed(self, tmp_path):
        """A contour with mean slope > max_slope_deg should be removed."""
        dem_path = _make_dem(tmp_path)
        geom = LineString([(0.5, 9.5), (19.5, 9.5)])
        steep = ContourFeature(geometry=geom, elevation=80.0, length_m=19.0)
        result = filter_by_slope([steep], dem_path, max_slope_deg=0.001)
        # Any non-flat DEM will have slope > 0.001° — feature should be removed
        assert result == []

    def test_zero_length_geom_not_crash(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        geom = LineString([(5, 5), (5, 5)])  # zero length
        feat = ContourFeature(geometry=geom, elevation=50.0, length_m=0.0)
        result = filter_by_slope([feat], dem_path, max_slope_deg=45.0)
        assert isinstance(result, list)

    def test_empty_list(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        assert filter_by_slope([], dem_path) == []

    def test_mean_slope_populated(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        geom = LineString([(0.5, 9.5), (9.5, 9.5)])
        feat = ContourFeature(geometry=geom, elevation=80.0, length_m=9.0)
        filter_by_slope([feat], dem_path, max_slope_deg=90.0)
        assert feat.mean_slope_deg >= 0.0

    def test_nodata_in_dem(self, tmp_path):
        data = np.fromfunction(lambda r, c: 100.0 - r * 2.0, (20, 20)).astype("float32")
        data[5:10, :] = -9999.0
        dem_path = _write_raster(str(tmp_path / "nd.tif"), data, nodata=-9999.0)
        geom = LineString([(1, 9.5), (19, 9.5)])
        feat = ContourFeature(geometry=geom, elevation=80.0, length_m=18.0)
        result = filter_by_slope([feat], dem_path, max_slope_deg=90.0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# rank_by_flow_crossing
# ---------------------------------------------------------------------------

class TestRankByFlowCrossing:
    def test_assigns_rank(self, tmp_path):
        acc_path = _make_acc(tmp_path)
        features = [_make_feature(elevation=e) for e in [50, 60, 70]]
        # Give features geometries that cross the high-acc column
        for i, f in enumerate(features):
            f.geometry = LineString([(5.0, 10.0 + i), (15.0, 10.0 + i)])
        result = rank_by_flow_crossing(features, acc_path)
        ranks = [f.rank for f in result]
        assert sorted(ranks) == [1, 2, 3]

    def test_sorted_descending_by_acc(self, tmp_path):
        acc_path = _make_acc(tmp_path)
        features = [
            ContourFeature(LineString([(5.0, 5.0), (15.0, 5.0)]), 60.0, length_m=10.0),
            ContourFeature(LineString([(9.5, 5.0), (10.5, 5.0)]), 60.0, length_m=1.0),
        ]
        result = rank_by_flow_crossing(features, acc_path, n_samples=20)
        # First in result should have higher peak_acc
        assert result[0].peak_acc >= result[-1].peak_acc

    def test_empty_list(self, tmp_path):
        acc_path = _make_acc(tmp_path)
        assert rank_by_flow_crossing([], acc_path) == []

    def test_zero_length_geom(self, tmp_path):
        acc_path = _make_acc(tmp_path)
        geom = LineString([(5, 5), (5, 5)])
        feat = ContourFeature(geometry=geom, elevation=50.0, length_m=0.0)
        result = rank_by_flow_crossing([feat], acc_path)
        assert result[0].peak_acc == 0.0

    def test_nodata_acc_treated_as_zero(self, tmp_path):
        data = np.full((20, 20), -9999.0, dtype="float32")
        acc_path = _write_raster(str(tmp_path / "nd_acc.tif"), data, nodata=-9999.0)
        feat = _make_feature()
        feat.geometry = LineString([(5, 10), (15, 10)])
        result = rank_by_flow_crossing([feat], acc_path)
        assert result[0].peak_acc == 0.0


# ---------------------------------------------------------------------------
# clip_to_usable_area
# ---------------------------------------------------------------------------

class TestClipToUsableArea:
    def test_feature_inside_not_clipped(self):
        polygon = box(0, 0, 200, 200)
        feat = _make_feature(elevation=50.0, length_m=100.0)
        feat.geometry = LineString([(10, 10), (90, 10)])
        result = clip_to_usable_area([feat], polygon)
        assert len(result) == 1

    def test_feature_outside_dropped(self):
        polygon = box(0, 0, 5, 5)
        feat = _make_feature()
        feat.geometry = LineString([(50, 10), (100, 10)])
        result = clip_to_usable_area([feat], polygon)
        assert len(result) == 0

    def test_empty_list(self):
        polygon = box(0, 0, 100, 100)
        assert clip_to_usable_area([], polygon) == []

    def test_clipped_feature_keeps_elevation(self):
        polygon = box(0, 0, 50, 50)
        feat = _make_feature(elevation=42.0)
        feat.geometry = LineString([(10, 10), (80, 10)])  # extends beyond polygon
        result = clip_to_usable_area([feat], polygon)
        if result:
            assert result[0].elevation == pytest.approx(42.0)

    def test_sliver_dropped(self):
        """Clipped segments < 0.5 m are dropped."""
        polygon = box(0, 0, 0.3, 100)  # very narrow — intersection < 0.5 m
        feat = _make_feature()
        feat.geometry = LineString([(0.1, 10), (100, 10)])
        result = clip_to_usable_area([feat], polygon)
        # Either dropped or a very short segment is returned
        for r in result:
            assert r.geometry.length >= 0.5


# ---------------------------------------------------------------------------
# analyse_contours (full pipeline)
# ---------------------------------------------------------------------------

class TestAnalyseContours:
    def test_returns_list(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        acc_path = _make_acc(tmp_path)
        result = analyse_contours(dem_path, acc_path, interval_m=5.0)
        assert isinstance(result, list)

    def test_with_progress_callback(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        acc_path = _make_acc(tmp_path)
        calls = []
        result = analyse_contours(
            dem_path, acc_path,
            interval_m=5.0,
            progress_callback=lambda pct, msg: calls.append(pct),
        )
        assert isinstance(result, list)

    def test_with_usable_polygon(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        acc_path = _make_acc(tmp_path)
        polygon = box(2, 2, 18, 18)
        result = analyse_contours(dem_path, acc_path, interval_m=5.0,
                                  usable_polygon=polygon)
        assert isinstance(result, list)

    def test_with_cell_area_and_runoff(self, tmp_path):
        dem_path = _make_dem(tmp_path)
        acc_path = _make_acc(tmp_path)
        result = analyse_contours(
            dem_path, acc_path,
            interval_m=5.0, cell_area_m2=1.0, runoff_mm=25.0
        )
        assert isinstance(result, list)

    def test_min_length_filter(self, tmp_path):
        """Features shorter than min_length_m should be dropped."""
        dem_path = _make_dem(tmp_path)
        acc_path = _make_acc(tmp_path)
        result_short = analyse_contours(dem_path, acc_path, interval_m=5.0,
                                        min_length_m=0.0)
        result_long = analyse_contours(dem_path, acc_path, interval_m=5.0,
                                       min_length_m=1000.0)
        assert len(result_long) <= len(result_short)
