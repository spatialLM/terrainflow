"""
Tests for terrainflow_assessment/modules/catchment.py

Covers the SCSRunoff class (duplicate of plugin's but in assessment package),
clip_dem_to_polygon, and fast_contributing_area.
"""
import os

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

from terrainflow_assessment.modules.catchment import (
    SCSRunoff,
    clip_dem_to_polygon,
    fast_contributing_area,
)


# ---------------------------------------------------------------------------
# SCSRunoff — assessment package version
# ---------------------------------------------------------------------------

class TestSCSRunoffAssessment:
    """Mirror tests from test_scs_runoff.py to hit catchment.py coverage."""

    def setup_method(self):
        self.scs = SCSRunoff()

    # adjust_cn
    def test_normal_unchanged(self):
        assert self.scs.adjust_cn(70, "normal") == pytest.approx(70.0)

    def test_dry_lower(self):
        assert self.scs.adjust_cn(70, "dry") < 70.0

    def test_wet_higher(self):
        assert self.scs.adjust_cn(70, "wet") > 70.0

    def test_clamp_high(self):
        assert self.scs.adjust_cn(99, "wet") <= 100.0

    def test_clamp_low(self):
        assert self.scs.adjust_cn(2, "dry") >= 1.0

    # runoff_depth
    def test_zero_cn(self):
        assert self.scs.runoff_depth(100, 0) == 0.0

    def test_below_ia(self):
        assert self.scs.runoff_depth(5, 70) == 0.0

    def test_above_ia(self):
        assert self.scs.runoff_depth(100, 70) > 0.0

    def test_cn_100_equals_rainfall(self):
        assert self.scs.runoff_depth(50, 100) == pytest.approx(50.0, rel=1e-3)

    def test_never_negative(self):
        for cn in [10, 50, 75, 98]:
            for rain in [0, 5, 50, 200]:
                assert self.scs.runoff_depth(rain, cn) >= 0.0

    # runoff_ratio
    def test_ratio_zero_rain(self):
        assert self.scs.runoff_ratio(0, 75) == 0.0

    def test_ratio_between_0_and_1(self):
        r = self.scs.runoff_ratio(100, 75)
        assert 0.0 <= r <= 1.0

    # catchment_volume
    def test_10mm_1ha(self):
        assert self.scs.catchment_volume(10.0, 10_000) == pytest.approx(100.0)

    def test_zero_runoff(self):
        assert self.scs.catchment_volume(0.0, 10_000) == 0.0

    # build_runoff_raster
    def test_raster_shape(self):
        cn = np.full((5, 5), 75, dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        assert result.shape == (5, 5)

    def test_raster_zero_cn(self):
        cn = np.zeros((3, 3), dtype="float32")
        result = self.scs.build_runoff_raster(cn, 100.0)
        assert np.all(result == 0.0)

    def test_raster_dtype(self):
        cn = np.full((3, 3), 70, dtype="float32")
        assert self.scs.build_runoff_raster(cn, 50.0).dtype == np.float32

    # build_cn_raster
    def test_cn_raster_default(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "normal")
        assert np.allclose(result, 70.0)

    def test_cn_raster_dry(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "dry")
        assert result.mean() < 70.0

    def test_cn_raster_wet(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "wet")
        assert result.mean() > 70.0

    def test_cn_raster_zone_override(self):
        shape = (10, 10)
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        zone = box(0, 5, 5, 10)
        result = self.scs.build_cn_raster(shape, transform, [(zone, 90)], 70, "normal")
        assert result.max() > 70.0

    def test_cn_raster_none_geom_skipped(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [(None, 90)], 70, "normal")
        assert np.allclose(result, 70.0)

    def test_cn_raster_empty_geom_skipped(self):
        from shapely.geometry import Polygon
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [(Polygon(), 90)], 70, "normal")
        assert np.allclose(result, 70.0)

    def test_cn_raster_float32(self):
        shape = (5, 5)
        transform = from_bounds(0, 0, 5, 5, 5, 5)
        result = self.scs.build_cn_raster(shape, transform, [], 70, "normal")
        assert result.dtype == np.float32

    # parse_hyetograph_csv
    def test_parse_cumulative(self, tmp_path):
        import csv as _csv
        p = str(tmp_path / "rain.csv")
        with open(p, "w", newline="") as f:
            _csv.writer(f).writerows([
                ["time_min", "rainfall_mm"], [10, 5], [20, 15], [30, 30]
            ])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[0] == (0, 0.0)
        assert any(t == 10 for t, _ in result)

    def test_parse_missing_column_raises(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("a,b\n1,2\n")
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_parse_empty_raises(self, tmp_path):
        p = str(tmp_path / "empty.csv")
        open(p, "w").close()
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_parse_per_interval(self, tmp_path):
        import csv as _csv
        p = str(tmp_path / "rain.csv")
        with open(p, "w", newline="") as f:
            _csv.writer(f).writerows([
                ["time_min", "rainfall_mm"], [10, 5], [20, 10], [30, 3]
            ])
        result = SCSRunoff.parse_hyetograph_csv(p)
        assert result[-1][1] == pytest.approx(18.0, abs=0.01)

    # constants
    def test_soil_reference(self):
        assert "Clay" in SCSRunoff.SOIL_REFERENCE

    def test_storm_presets(self):
        assert "Custom" in SCSRunoff.STORM_PRESETS


# ---------------------------------------------------------------------------
# clip_dem_to_polygon (assessment version)
# ---------------------------------------------------------------------------

class TestClipDEMToPolygonCatchment:
    def test_creates_file(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_dem_to_polygon(tmp_dem, box(3, 3, 15, 15), out)
        assert os.path.exists(out)

    def test_returns_path(self, tmp_dem, tmp_path):
        out = str(tmp_path / "c.tif")
        assert clip_dem_to_polygon(tmp_dem, box(3, 3, 15, 15), out) == out


# ---------------------------------------------------------------------------
# SCSRunoff CSV gap coverage (lines 357-358, 361)
# ---------------------------------------------------------------------------

class TestSCSRunoffCSVGaps:
    def test_invalid_rows_skipped(self, tmp_path):
        p = str(tmp_path / "rain.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm\n10,5\nbad,row\n20,15\n")
        result = SCSRunoff.parse_hyetograph_csv(p)
        times = [t for t, _ in result]
        assert 10 in times and 20 in times

    def test_no_valid_rows_raises(self, tmp_path):
        p = str(tmp_path / "empty_rows.csv")
        with open(p, "w") as f:
            f.write("time_min,rainfall_mm\nx,y\n")
        with pytest.raises(ValueError):
            SCSRunoff.parse_hyetograph_csv(p)

    def test_build_runoff_raster_below_ia_returns_zero(self):
        """Line 326: rainfall <= Ia → _q returns 0.0."""
        scs = SCSRunoff()
        cn = np.full((3, 3), 70, dtype="float32")
        # CN=70: S≈108.9, Ia≈21.8 — rainfall=5 is below Ia
        result = scs.build_runoff_raster(cn, 5.0)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# fast_contributing_area (assessment version)
# ---------------------------------------------------------------------------

def _write_sloped_dem(path, shape=(25, 25), cell_size=1.0, crs="EPSG:32632"):
    data = np.fromfunction(
        lambda r, c: 100.0 - r * 2.0 - c * 0.5, shape
    ).astype("float32")
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path


class TestFastContributingAreaAssessment:
    def _setup(self, tmp_path):
        dem = _write_sloped_dem(str(tmp_path / "dem.tif"))
        gdf = gpd.GeoDataFrame(geometry=[box(5, 5, 15, 15)], crs="EPSG:32632")
        bp = str(tmp_path / "boundary.gpkg")
        gdf.to_file(bp, driver="GPKG")
        return dem, bp

    def test_returns_required_keys(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        result = fast_contributing_area(dem, bp)
        for key in ("catchment_polygon", "area_ha", "dem_area_ha",
                    "coverage_pct", "scale"):
            assert key in result

    def test_area_positive(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        result = fast_contributing_area(dem, bp)
        assert result["area_ha"] >= 0.0
        assert not result["catchment_polygon"].is_empty

    def test_scale_1_small_dem(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        result = fast_contributing_area(dem, bp)
        assert result["scale"] == pytest.approx(1.0)

    def test_progress_callback(self, tmp_path):
        dem, bp = self._setup(tmp_path)
        pcts = []
        fast_contributing_area(dem, bp, progress_callback=lambda p, m: pcts.append(p))
        assert len(pcts) > 0
        assert pcts[-1] == 100

    def test_boundary_outside_raises(self, tmp_path):
        dem = _write_sloped_dem(str(tmp_path / "dem2.tif"))
        gdf = gpd.GeoDataFrame(geometry=[box(200, 200, 300, 300)], crs="EPSG:32632")
        bp = str(tmp_path / "far.gpkg")
        gdf.to_file(bp, driver="GPKG")
        with pytest.raises(RuntimeError):
            fast_contributing_area(dem, bp)

    def test_downsample_branch(self, tmp_path, monkeypatch):
        """Lines 80-82: force downsampling by lowering _MAX_PREVIEW_CELLS."""
        import terrainflow_assessment.modules.catchment as cat
        monkeypatch.setattr(cat, "_MAX_PREVIEW_CELLS", 100)  # 10×10 threshold
        dem, bp = self._setup(tmp_path)  # 25×25 DEM → 625 cells > 100
        result = fast_contributing_area(dem, bp)
        assert result["scale"] < 1.0  # downsampled

    def test_boundary_without_exterior_raises(self, tmp_path):
        """Line 116: boundary lines empty (non-polygon geom) → RuntimeError."""
        from shapely.geometry import LineString
        dem = _write_sloped_dem(str(tmp_path / "nopoly_dem.tif"))
        # Force rasterize to populate inside_mask (via all_touched) then fail on exterior
        gdf = gpd.GeoDataFrame(
            geometry=[LineString([(5, 5), (15, 15)])], crs="EPSG:32632"
        )
        bp = str(tmp_path / "line_bnd.gpkg")
        gdf.to_file(bp, driver="GPKG")
        with pytest.raises(RuntimeError):
            fast_contributing_area(dem, bp)

    def test_nodata_cells_handled(self, tmp_path):
        data = np.fromfunction(
            lambda r, c: 100.0 - r * 2.0, (25, 25)
        ).astype("float32")
        data[0:3, 0:3] = -9999.0
        dem = str(tmp_path / "nd_dem.tif")
        h, w = data.shape
        transform = from_bounds(0, 0, 25, 25, w, h)
        with rasterio.open(dem, "w", driver="GTiff", height=h, width=w,
                           count=1, dtype="float32", crs="EPSG:32632",
                           transform=transform, nodata=-9999.0) as dst:
            dst.write(data, 1)
        gdf = gpd.GeoDataFrame(geometry=[box(8, 8, 18, 18)], crs="EPSG:32632")
        bp = str(tmp_path / "b2.gpkg")
        gdf.to_file(bp, driver="GPKG")
        result = fast_contributing_area(dem, bp)
        assert not result["catchment_polygon"].is_empty


# ---------------------------------------------------------------------------
# Phase 1 regression — item 1: D8 reverse traversal correctness
# ---------------------------------------------------------------------------

def _write_dem_array(path, data, cell_size=1.0, crs="EPSG:32632", nodata=-9999.0):
    """Write a numpy array as a GeoTIFF DEM."""
    h, w = data.shape
    from rasterio.transform import from_bounds
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w,
                       count=1, dtype="float32", crs=crs,
                       transform=transform, nodata=nodata) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _write_boundary(path, bounds_xy, crs="EPSG:32632"):
    """Write a bounding-box polygon to a GeoPackage."""
    gdf = gpd.GeoDataFrame(geometry=[box(*bounds_xy)], crs=crs)
    gdf.to_file(path, driver="GPKG")
    return path


class TestD8CatchmentCorrectness:
    """Phase 1 item 1: D8 reverse traversal replaces the broken BFS."""

    def test_catchment_matches_pysheds_on_bowl(self, tmp_path):
        """
        Bowl DEM: all cells drain to the centre.  The D8 catchment of a
        site on the bowl floor should cover essentially the entire DEM
        (within 10 % — the bowl's contributing area = whole DEM).
        """
        r = np.arange(20)
        c = np.arange(20)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        bowl = (50.0 + ((rr - 9.5) ** 2 + (cc - 9.5) ** 2) * 0.3).astype("float32")

        dem_path = _write_dem_array(str(tmp_path / "bowl.tif"), bowl)
        # Site boundary around the bowl centre (3×3 m)
        bp = _write_boundary(str(tmp_path / "site.gpkg"), (8, 8, 12, 12))

        result = fast_contributing_area(dem_path, bp)

        # All bowl cells drain to the centre → catchment ≈ full DEM area
        dem_area = result["dem_area_ha"]
        catchment_area = result["area_ha"]
        assert catchment_area >= dem_area * 0.90, (
            f"D8 catchment ({catchment_area:.2f} ha) should cover ≥90 % "
            f"of bowl DEM ({dem_area:.2f} ha)"
        )

    def test_excludes_knoll_behind_ridge(self, tmp_path):
        """
        Synthetic 30×20 DEM (1 m cells, EPSG:32632).

        Layout (row 0 = north/top, row 29 = south/bottom, y increases southward):
          Rows  0-9  (y=20-30): knoll — elevation increases toward row 9
                                (min at row 0 → drains NORTH off DEM)
          Row  10    (y=19-20): sharp ridge (elev=120, blocks southward drainage)
          Rows 11-29 (y=0-19):  south slope (elev decreases toward row 29)
                                → all water flows SOUTH toward site

        Site boundary: bottom-centre of DEM (y=0-5, x=5-15 → rows 25-29).

        Assertion: catchment must not extend north of the ridge top (y > 20.5).
        Knoll cells at y=20-30 drain off the north DEM edge — they must be
        excluded from the contributing area of the southern site.
        """
        data = np.zeros((30, 20), dtype="float32")
        for r in range(30):
            lateral = ((np.arange(20) - 9.5) ** 2) * 0.1
            if r < 10:
                # Knoll: elevation INCREASES toward ridge (row 9 is highest).
                # Row 0 is the local minimum → drains north off DEM.
                data[r, :] = (60.0 + r * 5.0 + lateral).astype("float32")
            elif r == 10:
                # Sharp ridge — elevation much higher than anything adjacent.
                data[r, :] = (120.0 + lateral).astype("float32")
            else:
                # South slope: elevation decreases going south (rows 11→29).
                data[r, :] = (100.0 - (r - 11) * 3.5 + lateral).astype("float32")

        dem_path = _write_dem_array(str(tmp_path / "ridge.tif"), data)
        # Site in the southern third of the DEM, clear of the ridge.
        bp = _write_boundary(str(tmp_path / "site.gpkg"), (5, 0, 15, 5))

        result = fast_contributing_area(dem_path, bp)
        poly = result["catchment_polygon"]

        # Ridge centre y ≈ 19.5 (row 10, from_bounds gives y=19-20).
        # Knoll rows 0-9 → y=20-30.  None of them should appear in the catchment.
        _, _, _, maxy = poly.bounds
        assert maxy <= 20.5, (
            f"Catchment extends to y={maxy:.1f} m — knoll cells at y>20 "
            "(north of ridge) should be excluded from contributing area"
        )
