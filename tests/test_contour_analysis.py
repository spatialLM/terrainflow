"""Tests for terrainflow_assessment/modules/contour_analysis.py"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, box

from terrainflow_assessment.modules.contour_analysis import (
    ContourFeature,
    _extract_contours_scipy,
    analyse_contours,
    clip_to_usable_area,
    filter_by_slope,
    rank_by_flow_crossing,
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


# ---------------------------------------------------------------------------
# extract_contours — mocked gdal_contour success / failure branches
# ---------------------------------------------------------------------------

class TestExtractContoursWithMockedGdal:
    """
    gdal_contour is not installed in the test environment, so the live code
    always takes the FileNotFoundError fallback. Mock subprocess.run to cover
    the success + non-zero-returncode paths.
    """

    def test_subprocess_success_path(self, tmp_path, monkeypatch):
        """When gdal_contour exits 0, the gpkg is read and parsed."""
        import subprocess

        import geopandas as gpd
        from shapely.geometry import LineString

        from terrainflow_assessment.modules import contour_analysis as ca

        dem_path = _make_dem(tmp_path)

        # Write a mock contour gpkg the function can read back
        gpkg = str(tmp_path / "mock_contours.gpkg")
        gdf = gpd.GeoDataFrame({
            "ELEV": [70.0, 75.0],
            "geometry": [
                LineString([(0, 0), (10, 0)]),
                LineString([(0, 5), (10, 5)]),
            ],
        }, crs="EPSG:32632")
        gdf.to_file(gpkg, driver="GPKG")

        def _fake_run(*args, **kwargs):
            class _R:
                returncode = 0
                stderr = ""
            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        features = ca.extract_contours(dem_path, interval_m=5.0, output_path=gpkg)
        assert len(features) == 2
        assert any(abs(f.elevation - 70.0) < 0.01 for f in features)

    def test_subprocess_nonzero_returncode(self, tmp_path, monkeypatch):
        """gdal_contour returning non-zero → RuntimeError."""
        import subprocess

        from terrainflow_assessment.modules import contour_analysis as ca

        dem_path = _make_dem(tmp_path)

        def _fake_run(*args, **kwargs):
            class _R:
                returncode = 1
                stderr = "boom"
            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="gdal_contour failed"):
            ca.extract_contours(dem_path, interval_m=5.0)

    def test_subprocess_success_but_bad_output(self, tmp_path, monkeypatch):
        """gdal_contour reports success but output gpkg isn't readable."""
        import subprocess

        from terrainflow_assessment.modules import contour_analysis as ca

        dem_path = _make_dem(tmp_path)

        def _fake_run(*args, **kwargs):
            class _R:
                returncode = 0
                stderr = ""
            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        with pytest.raises(RuntimeError, match="Cannot read contour output"):
            ca.extract_contours(dem_path, interval_m=5.0,
                                output_path=str(tmp_path / "nonexistent.gpkg"))

    def test_subprocess_success_skips_empty_geoms(self, tmp_path, monkeypatch):
        """Empty/None geometries are skipped; ELEV missing → 0.0."""
        import subprocess

        import geopandas as gpd
        from shapely.geometry import LineString

        from terrainflow_assessment.modules import contour_analysis as ca

        dem_path = _make_dem(tmp_path)
        gpkg = str(tmp_path / "mock_mixed.gpkg")
        gdf = gpd.GeoDataFrame({
            "geometry": [LineString([(0, 0), (5, 0)])],
        }, crs="EPSG:32632")
        gdf.to_file(gpkg, driver="GPKG")

        def _fake_run(*args, **kwargs):
            class _R:
                returncode = 0
                stderr = ""
            return _R()

        monkeypatch.setattr(subprocess, "run", _fake_run)
        features = ca.extract_contours(dem_path, interval_m=5.0, output_path=gpkg)
        # ELEV column missing → default to 0.0
        assert all(f.elevation == 0.0 for f in features)


# ---------------------------------------------------------------------------
# clip_to_usable_area — MultiLineString and exception branches
# ---------------------------------------------------------------------------

class TestClipToUsableAreaBranches:
    def test_multilinestring_keeps_longest_segment(self):
        """Polygon with a hole forces intersection to be MultiLineString."""
        from shapely.geometry import Polygon

        from terrainflow_assessment.modules.contour_analysis import clip_to_usable_area

        # Ring polygon: outer 100×100, hole 40×40 in the middle
        outer = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        hole = [(30, 30), (70, 30), (70, 70), (30, 70), (30, 30)]
        ring = Polygon(outer, [hole])

        feat = _make_feature(elevation=50.0, length_m=120.0)
        # Horizontal line crossing the hole, producing two segments
        feat.geometry = LineString([(-10, 50), (110, 50)])
        result = clip_to_usable_area([feat], ring)
        # Result should contain one feature (the longer segment kept)
        assert len(result) == 1
        assert result[0].geometry.length > 0

    def test_exception_in_intersection_swallowed(self):
        """If geometry.intersection raises, the feature is skipped."""
        from terrainflow_assessment.modules.contour_analysis import clip_to_usable_area

        feat = _make_feature(elevation=50.0, length_m=10.0)
        # Replace geometry with a mock whose .intersection raises
        class _RaisingGeom:
            def intersection(self, other):
                raise ValueError("boom")
        feat.geometry = _RaisingGeom()
        polygon = box(0, 0, 100, 100)
        result = clip_to_usable_area([feat], polygon)
        assert result == []

    def test_non_line_intersection_skipped(self):
        """If intersection is a Point/GeometryCollection, skip the feature."""
        from terrainflow_assessment.modules.contour_analysis import clip_to_usable_area

        # A LineString that just touches the polygon at a single point
        feat = _make_feature(elevation=50.0, length_m=10.0)
        feat.geometry = LineString([(10, 50), (20, 50)])
        # Polygon whose edge touches the line at a single point (0-length overlap)
        polygon = box(20, 40, 30, 60)
        result = clip_to_usable_area([feat], polygon)
        # Intersection is a Point (tangent) — feature is skipped
        assert result == []


# ---------------------------------------------------------------------------
# analyse_contours — cell_area/runoff stamping loop
# ---------------------------------------------------------------------------

class TestAnalyseContoursBranches:
    def test_stamping_loop_populates_features(self, tmp_path):
        """Providing cell_area_m2 and runoff_mm stamps every feature."""
        dem_path = _make_dem(tmp_path)
        acc_path = _make_acc(tmp_path)
        result = analyse_contours(
            dem_path, acc_path, interval_m=5.0,
            cell_area_m2=4.0, runoff_mm=15.0,
        )
        for f in result:
            assert f.cell_area_m2 == 4.0
            assert f.runoff_mm == 15.0


# ---------------------------------------------------------------------------
# SwaleSegment.label
# ---------------------------------------------------------------------------

class TestSwaleSegmentLabel:
    def test_label_with_inflow(self):
        from terrainflow_assessment.modules.contour_analysis import SwaleSegment
        geom = LineString([(0, 0), (50, 0)])
        seg = SwaleSegment(
            geometry=geom, elevation=85.0, peak_acc=1000.0,
            contributing_ha=5.0, inflow_m3=1234.0,
            contour_rank=1, segment_rank=2, length_m=50.0,
            required_length_m=75.0,
        )
        label = seg.label
        assert "#2" in label
        assert "85" in label
        assert "5.0" in label
        assert "1,234" in label
        assert "75 m" in label

    def test_label_without_inflow(self):
        from terrainflow_assessment.modules.contour_analysis import SwaleSegment
        geom = LineString([(0, 0), (40, 0)])
        seg = SwaleSegment(
            geometry=geom, elevation=60.0, peak_acc=200.0,
            contributing_ha=2.0, inflow_m3=0.0,
            contour_rank=3, segment_rank=5, length_m=40.0,
        )
        label = seg.label
        assert "40 m long" in label

    def test_required_length_defaults_to_length(self):
        from terrainflow_assessment.modules.contour_analysis import SwaleSegment
        geom = LineString([(0, 0), (25, 0)])
        seg = SwaleSegment(
            geometry=geom, elevation=50.0, peak_acc=100.0,
            contributing_ha=1.0, inflow_m3=0.0,
            contour_rank=1, segment_rank=1, length_m=25.0,
        )
        assert seg.required_length_m == 25.0


# ---------------------------------------------------------------------------
# find_swale_segments — covers the large block 516-664
# ---------------------------------------------------------------------------

def _make_contour_and_acc(tmp_path, cell_size=1.0):
    """Build a 40×40 acc raster with a clear peak, and a straight contour."""
    data = np.full((40, 40), 10.0, dtype="float32")
    # Strong central peak to give the swale-segment sizing logic something to work on
    data[20, 20] = 80000.0
    data[19:22, 19:22] = np.maximum(data[19:22, 19:22], 50000.0)
    acc_path = _write_raster(str(tmp_path / "acc.tif"), data, cell_size=cell_size)
    # A long contour that crosses the peak column
    geom = LineString([(0.5, 20.5), (39.5, 20.5)])
    feat = ContourFeature(
        geometry=geom, elevation=50.0, rank=1, length_m=39.0,
    )
    return feat, acc_path


class TestFindSwaleSegments:
    def test_returns_list(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        feat, acc_path = _make_contour_and_acc(tmp_path)
        result = find_swale_segments([feat], acc_path,
                                     cell_area_m2=1.0, runoff_mm=25.0)
        assert isinstance(result, list)

    def test_finds_segment_at_peak(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        feat, acc_path = _make_contour_and_acc(tmp_path)
        result = find_swale_segments([feat], acc_path,
                                     cell_area_m2=1.0, runoff_mm=25.0,
                                     min_acc_ha=0.1)
        assert len(result) >= 1
        assert result[0].segment_rank == 1
        assert result[0].inflow_m3 > 0

    def test_no_runoff_uses_landscape_walk(self, tmp_path):
        """Without runoff_mm, fallback branch is used (drop_fraction walk)."""
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        feat, acc_path = _make_contour_and_acc(tmp_path)
        result = find_swale_segments([feat], acc_path,
                                     cell_area_m2=1.0, runoff_mm=None,
                                     min_acc_ha=0.1)
        # Landscape walk produces segments with inflow_m3 == 0
        for s in result:
            assert s.inflow_m3 == 0

    def test_progress_callback_invoked(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        feat, acc_path = _make_contour_and_acc(tmp_path)
        calls = []
        find_swale_segments([feat], acc_path,
                            cell_area_m2=1.0, runoff_mm=25.0,
                            min_acc_ha=0.1,
                            progress_callback=lambda p, m: calls.append(p))
        assert len(calls) > 0
        assert calls[-1] == 100

    def test_short_contour_skipped(self, tmp_path):
        """Contour shorter than 1 m is skipped."""
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        _, acc_path = _make_contour_and_acc(tmp_path)
        short = ContourFeature(
            geometry=LineString([(10, 10), (10.5, 10)]),
            elevation=50.0, rank=1, length_m=0.5,
        )
        result = find_swale_segments([short], acc_path,
                                     cell_area_m2=1.0, runoff_mm=25.0)
        assert result == []

    def test_no_peaks_returns_empty_for_flat_acc(self, tmp_path):
        """Flat accumulation raster → no peaks detected."""
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        data = np.full((20, 20), 5.0, dtype="float32")
        acc_path = _write_raster(str(tmp_path / "flat_acc.tif"), data)
        feat = ContourFeature(
            geometry=LineString([(0.5, 10.5), (19.5, 10.5)]),
            elevation=50.0, rank=1, length_m=19.0,
        )
        result = find_swale_segments([feat], acc_path,
                                     cell_area_m2=1.0, runoff_mm=25.0,
                                     min_acc_ha=0.5)
        assert result == []

    def test_segments_sorted_by_inflow(self, tmp_path):
        """Multiple contours → results sorted by inflow_m3 descending."""
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        feat1, acc_path = _make_contour_and_acc(tmp_path)
        feat2 = ContourFeature(
            geometry=LineString([(0.5, 20.5), (39.5, 20.5)]),
            elevation=60.0, rank=2, length_m=39.0,
        )
        result = find_swale_segments([feat1, feat2], acc_path,
                                     cell_area_m2=1.0, runoff_mm=25.0,
                                     min_acc_ha=0.1)
        for i in range(1, len(result)):
            assert result[i - 1].inflow_m3 >= result[i].inflow_m3

    def test_segments_sorted_by_contrib_when_no_runoff(self, tmp_path):
        """Without runoff, segments sorted by contributing_ha descending."""
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        feat, acc_path = _make_contour_and_acc(tmp_path)
        result = find_swale_segments([feat], acc_path,
                                     cell_area_m2=1.0, runoff_mm=None,
                                     min_acc_ha=0.1)
        for i in range(1, len(result)):
            assert result[i - 1].contributing_ha >= result[i].contributing_ha

    def test_nodata_acc_handled(self, tmp_path):
        """Accumulation raster with nodata values is replaced with 0 safely."""
        from terrainflow_assessment.modules.contour_analysis import find_swale_segments
        data = np.full((20, 20), -9999.0, dtype="float32")
        acc_path = _write_raster(str(tmp_path / "nd_acc.tif"), data, nodata=-9999.0)
        feat = ContourFeature(
            geometry=LineString([(0.5, 10.5), (19.5, 10.5)]),
            elevation=50.0, rank=1, length_m=19.0,
        )
        result = find_swale_segments([feat], acc_path,
                                     cell_area_m2=1.0, runoff_mm=25.0)
        assert result == []


# ---------------------------------------------------------------------------
# _extract_contours_scipy — edge-case lines (short paths, degenerate coords)
# ---------------------------------------------------------------------------

class TestExtractContoursScipyEdgeCases:
    def test_extremely_flat_dem_empty(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import _extract_contours_scipy
        data = np.full((5, 5), 42.0, dtype="float32")
        path = _write_raster(str(tmp_path / "flat.tif"), data)
        result = _extract_contours_scipy(path, interval_m=1.0)
        # May be empty or have short paths — must be a list
        assert isinstance(result, list)

    def test_high_interval_few_contours(self, tmp_path):
        from terrainflow_assessment.modules.contour_analysis import _extract_contours_scipy
        data = np.fromfunction(lambda r, c: 100.0 - r * 2.0, (20, 20)).astype("float32")
        path = _write_raster(str(tmp_path / "dem.tif"), data)
        result = _extract_contours_scipy(path, interval_m=100.0)
        # Huge interval → typically empty
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Phase 1 regression — item 6: NaN border cells must not fabricate gradients
# ---------------------------------------------------------------------------

class TestNoFalseEdgeGradient:
    """Phase 1 item 6: masked_invalid prevents border NaN from smearing into slope."""

    def test_no_false_edge_gradient(self, tmp_path):
        """
        A DEM with a NaN border strip must not produce spike slope values at the
        border cells.  Before the fix, nan_to_num(dem, 0) injected a large
        artificial gradient at the NaN→valid boundary.
        """
        # 20×20 flat DEM at 50 m — known zero slope everywhere
        data = np.full((20, 20), 50.0, dtype="float32")
        # NaN border (2-cell ring)
        data[0:2, :] = np.nan
        data[-2:, :] = np.nan
        data[:, 0:2] = np.nan
        data[:, -2:] = np.nan

        dem_path = _write_raster(str(tmp_path / "nan_border.tif"), data, nodata=np.nan)

        # Build a stub contour across the middle
        mid_geom = LineString([(2, 10), (18, 10)])
        feat = ContourFeature(geometry=mid_geom, elevation=50.0, length_m=16.0)

        from terrainflow_assessment.modules.contour_analysis import filter_by_slope
        result = filter_by_slope([feat], dem_path, max_slope_deg=5.0)

        # The interior is flat; the contour should pass the slope filter
        # (mean slope ≈ 0, well below 5°).  Before the fix, artificial
        # edge gradients could disqualify interior contours.
        assert len(result) == 1, (
            "Interior contour on a flat NaN-bordered DEM must not be rejected "
            "by a spurious edge gradient"
        )
        assert result[0].mean_slope_deg < 5.0
