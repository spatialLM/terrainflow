"""Tests for terrainflow_assessment/modules/dem_loader.py"""
import os

import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from terrainflow_assessment.modules.dem_loader import (
    DEMInfo,
    DEMValidationError,
    align_to_grid,
    clip_dem_to_polygon,
    compute_slope_raster,
    load_dem,
    slope_degrees,
)

# ---------------------------------------------------------------------------
# align_to_grid
# ---------------------------------------------------------------------------

def _tf(left, top, cell=1.0):
    from rasterio.transform import Affine
    return Affine(cell, 0.0, left, 0.0, -cell, top)


class TestAlignToGrid:
    def test_sub_window_lands_at_its_offset(self):
        """The Quail Island case: a clip 66 rows down and 287 columns in.

        Baseline ran on the design file's embedded clip (1027x858) while the burn ran on
        the parent tile (2157x1319) — same 1 m cell, same CRS, an exact integer offset.
        Lining the two up recovers the subtraction that isolates earthwork storage; the
        old shape-equality test threw it away and reported every measured volume with the
        site's natural ponding still in it.
        """
        src = np.arange(6, dtype="float64").reshape(2, 3)
        dst = align_to_grid(src, _tf(1574398.0, 5169966.0), _tf(1574111.0, 5170032.0),
                            (100, 400))
        assert dst is not None
        assert dst[66:68, 287:290] == pytest.approx(src)
        assert dst.sum() == pytest.approx(src.sum())

    def test_identical_grids_round_trip(self):
        src = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = align_to_grid(src, _tf(0.0, 10.0), _tf(0.0, 10.0), (2, 2))
        assert out == pytest.approx(src)

    def test_larger_source_is_cropped_to_the_destination(self):
        src = np.arange(16, dtype="float64").reshape(4, 4)
        out = align_to_grid(src, _tf(0.0, 10.0), _tf(1.0, 9.0), (2, 2))
        assert out == pytest.approx(src[1:3, 1:3])

    def test_different_cell_size_is_refused(self):
        src = np.ones((2, 2))
        assert align_to_grid(src, _tf(0.0, 10.0, cell=0.5),
                             _tf(0.0, 10.0, cell=1.0), (4, 4)) is None

    def test_fractional_offset_is_refused(self):
        """Half a cell out is not a window — and must never be nudged into one."""
        src = np.ones((2, 2))
        assert align_to_grid(src, _tf(0.5, 10.0), _tf(0.0, 10.0), (4, 4)) is None

    def test_disjoint_but_commensurate_grids_give_zeros_not_none(self):
        # Same grid, no overlap: the correction is legitimately zero everywhere, which
        # is a different statement from "these rasters cannot be compared".
        src = np.ones((2, 2))
        out = align_to_grid(src, _tf(100.0, 10.0), _tf(0.0, 10.0), (2, 2))
        assert out is not None
        assert out.sum() == pytest.approx(0.0)

    def test_volume_is_preserved_when_fully_contained(self):
        rng = np.random.default_rng(0)
        src = rng.random((7, 5))
        out = align_to_grid(src, _tf(3.0, 17.0), _tf(0.0, 20.0), (20, 20))
        assert out.sum() == pytest.approx(src.sum())

# ---------------------------------------------------------------------------
# load_dem
# ---------------------------------------------------------------------------

class TestLoadDEM:
    def test_returns_dem_info(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert isinstance(info, DEMInfo)

    def test_path_preserved(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.path == tmp_dem

    def test_dimensions_correct(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.width == 20
        assert info.height == 20

    def test_cell_size(self, tmp_dem):
        # 20m extent / 20 cells = 1m cell size
        info = load_dem(tmp_dem)
        assert info.cell_size_m == pytest.approx(1.0, rel=1e-4)

    def test_cell_area(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.cell_area_m2 == pytest.approx(1.0, rel=1e-4)

    def test_area_ha(self, tmp_dem):
        # 20×20 m = 400 m² = 0.04 ha
        info = load_dem(tmp_dem)
        assert info.area_ha == pytest.approx(0.04, rel=1e-4)

    def test_crs_is_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.crs is not None

    def test_crs_wkt_is_string(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert isinstance(info.crs_wkt, str)
        assert len(info.crs_wkt) > 10

    def test_transform_is_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.transform is not None

    def test_bounds_is_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.bounds is not None
        left, bottom, right, top = info.bounds
        assert right > left
        assert top > bottom

    def test_nodata_set(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert info.nodata == pytest.approx(-9999.0)

    def test_invalid_path_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="Cannot load DEM"):
            load_dem("/no/such/file.tif")

    def test_repr_contains_dimensions(self, tmp_dem):
        info = load_dem(tmp_dem)
        r = repr(info)
        assert "20" in r  # width or height


# ---------------------------------------------------------------------------
# DEMInfo repr
# ---------------------------------------------------------------------------

class TestDEMInfoRepr:
    def test_repr_contains_cell_size(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert "1.00 m" in repr(info)

    def test_repr_contains_area(self, tmp_dem):
        info = load_dem(tmp_dem)
        assert "ha" in repr(info)


# ---------------------------------------------------------------------------
# clip_dem_to_polygon
# ---------------------------------------------------------------------------

class TestClipDEMToPolygon:
    def test_creates_output_file(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_polygon = box(2, 2, 15, 15)  # within the 20×20 DEM
        clip_dem_to_polygon(tmp_dem, clip_polygon, out)
        assert os.path.exists(out)

    def test_clipped_smaller_than_original(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_polygon = box(5, 5, 15, 15)
        clip_dem_to_polygon(tmp_dem, clip_polygon, out)
        with rasterio.open(out) as src:
            w, h = src.width, src.height
        assert w < 20 or h < 20

    def test_returns_output_path(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        result = clip_dem_to_polygon(tmp_dem, box(2, 2, 15, 15), out)
        assert result == out

    def test_crs_preserved(self, tmp_dem, tmp_path):
        out = str(tmp_path / "clipped.tif")
        clip_dem_to_polygon(tmp_dem, box(2, 2, 15, 15), out)
        with rasterio.open(out) as src:
            assert src.crs is not None


# ---------------------------------------------------------------------------
# compute_slope_raster
# ---------------------------------------------------------------------------

class TestComputeSlopeRaster:
    def test_creates_output_file(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        compute_slope_raster(tmp_dem, out)
        assert os.path.exists(out)

    def test_returns_output_path(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        result = compute_slope_raster(tmp_dem, out)
        assert result == out

    def test_slope_non_negative(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        compute_slope_raster(tmp_dem, out)
        with rasterio.open(out) as src:
            data = src.read(1)
        # All slope values ≥ 0 (nodata masked)
        nodata = -9999.0
        valid = data[data != nodata]
        assert np.all(valid >= 0.0)

    def test_flat_dem_low_slope(self, tmp_dem_flat, tmp_path):
        out = str(tmp_path / "slope_flat.tif")
        compute_slope_raster(tmp_dem_flat, out)
        with rasterio.open(out) as src:
            data = src.read(1)
        nodata = -9999.0
        valid = data[data != nodata]
        # Flat DEM interior should have near-zero slope
        assert np.percentile(valid, 50) < 5.0  # median slope < 5°

    def test_same_shape_as_input(self, tmp_dem, tmp_path):
        out = str(tmp_path / "slope.tif")
        compute_slope_raster(tmp_dem, out)
        with rasterio.open(tmp_dem) as src_in:
            in_shape = (src_in.height, src_in.width)
        with rasterio.open(out) as src_out:
            out_shape = (src_out.height, src_out.width)
        assert in_shape == out_shape

    def test_dem_with_nodata(self, tmp_dem_flat, tmp_path):
        """Should not crash when DEM has nodata cells."""
        # Patch one cell to nodata
        import shutil
        patched = str(tmp_path / "nodata_dem.tif")
        shutil.copy(tmp_dem_flat, patched)
        with rasterio.open(patched, "r+") as dst:
            arr = dst.read(1)
            arr[0, 0] = -9999.0
            dst.write(arr, 1)

        out = str(tmp_path / "slope_nodata.tif")
        compute_slope_raster(patched, out)  # should not raise
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# slope_degrees (shared Horn's-method helper)
# ---------------------------------------------------------------------------

class TestSlopeDegrees:
    def test_flat_is_zero(self):
        dem = np.full((10, 10), 50.0, dtype="float32")
        slope = slope_degrees(dem, 1.0, 1.0)
        assert np.allclose(slope, 0.0)

    def test_shape_preserved(self):
        dem = np.zeros((7, 9), dtype="float32")
        assert slope_degrees(dem, 1.0, 1.0).shape == (7, 9)

    def test_non_negative(self):
        rng = np.arange(100, dtype="float32").reshape(10, 10)
        assert np.all(slope_degrees(rng, 1.0, 1.0) >= 0.0)

    def test_known_45_degree_ramp(self):
        # Elevation rising 1 m per 1 m cell in x → 45° slope in the interior.
        dem = np.tile(np.arange(10, dtype="float32"), (10, 1))  # z = x
        slope = slope_degrees(dem, 1.0, 1.0)
        assert slope[5, 5] == pytest.approx(45.0, abs=1e-3)

    def test_nodata_cell_is_nan_not_a_number(self):
        dem = np.full((6, 6), 50.0, dtype="float32")
        dem[0, 0] = np.nan
        slope = slope_degrees(dem, 1.0, 1.0)
        assert slope.shape == (6, 6)
        assert np.isnan(slope[0, 0])

    def test_nodata_does_not_fabricate_a_cliff_beside_it(self):
        """A hole in flat ground must not ring itself in near-vertical slope.

        Filling NaN with 0.0 m before differencing made every neighbour of a nodata
        cell read a ~50 m drop over one cell — indistinguishable from real terrain,
        and a clipped DEM is nothing but nodata boundary.
        """
        dem = np.full((7, 7), 50.0, dtype="float32")
        dem[3, 3] = np.nan
        slope = slope_degrees(dem, 1.0, 1.0)
        neighbours = slope[2:5, 2:5][~np.isnan(slope[2:5, 2:5])]
        assert np.all(neighbours < 1.0)

    def test_border_slope_matches_interior_on_a_uniform_ramp(self):
        """Edge replication halves the sampled separation, so the divisor halves too.

        Without that correction every border cell of every slope raster read half the
        true grade — a 45° scarp at the DEM edge reported as ~27°.
        """
        dem = np.tile(np.arange(10, dtype="float32"), (10, 1))  # z = x, 45° everywhere
        slope = slope_degrees(dem, 1.0, 1.0)
        assert slope[0, 5] == pytest.approx(45.0, abs=1e-3)   # top border row
        assert slope[5, 0] == pytest.approx(45.0, abs=1e-3)   # left border column


# ---------------------------------------------------------------------------
# Phase 1 regression — item 2: projected CRS guard
# ---------------------------------------------------------------------------

class TestProjectedCRSGuard:
    def _write_dem(self, path, crs):
        from rasterio.transform import from_bounds
        data = np.full((10, 10), 50.0, dtype="float32")
        transform = from_bounds(0, 0, 10, 10, 10, 10)
        with rasterio.open(path, "w", driver="GTiff", height=10, width=10,
                           count=1, dtype="float32", crs=crs,
                           transform=transform, nodata=-9999.0) as dst:
            dst.write(data, 1)
        return path

    def test_rejects_geographic_crs(self, tmp_path):
        """Phase 1 item 2: geographic CRS raises DEMValidationError."""
        path = self._write_dem(str(tmp_path / "geo.tif"), "EPSG:4326")
        with pytest.raises(DEMValidationError, match="geographic"):
            load_dem(path)

    def test_accepts_projected_crs(self, tmp_path):
        """NZTM2000 (projected) must load without error."""
        path = self._write_dem(str(tmp_path / "proj.tif"), "EPSG:2193")
        info = load_dem(path)
        assert info.crs is not None

    def test_dem_validation_error_is_value_error(self):
        """DEMValidationError must be a ValueError subclass."""
        assert issubclass(DEMValidationError, ValueError)


# ---------------------------------------------------------------------------
# DEM identity — what lets a design file verify it reopened against the same DEM
# ---------------------------------------------------------------------------

class TestDemFingerprint:
    def _write_dem(self, path, fill=50.0, size=10):
        from rasterio.transform import from_bounds
        data = np.full((size, size), fill, dtype="float32")
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size,
                           count=1, dtype="float32", crs="EPSG:2193",
                           transform=from_bounds(0, 0, size, size, size, size),
                           nodata=-9999.0) as dst:
            dst.write(data, 1)
        return path

    def test_digest_is_stable_for_the_same_file(self, tmp_path):
        from terrainflow_assessment.modules.dem_loader import dem_content_digest
        path = self._write_dem(str(tmp_path / "a.tif"))
        assert dem_content_digest(path) == dem_content_digest(path)

    def test_digest_differs_for_different_content_on_an_identical_grid(self, tmp_path):
        """The whole point: same cell size, CRS and extent, different elevations."""
        from terrainflow_assessment.modules.dem_loader import (
            dem_content_digest,
            fingerprint_dem,
        )
        a = self._write_dem(str(tmp_path / "a.tif"), fill=50.0)
        b = self._write_dem(str(tmp_path / "b.tif"), fill=75.0)

        assert dem_content_digest(a) != dem_content_digest(b)
        fa, fb = fingerprint_dem(a), fingerprint_dem(b)
        assert (fa["cell_size_m"], fa["extent"], fa["width"]) == \
               (fb["cell_size_m"], fb["extent"], fb["width"])
        assert fa["fingerprint"] != fb["fingerprint"]

    def test_digest_records_its_algorithm(self, tmp_path):
        from terrainflow_assessment.modules.dem_loader import dem_content_digest
        assert dem_content_digest(
            self._write_dem(str(tmp_path / "a.tif"))).startswith("sha256:")

    def test_large_files_switch_to_a_distinguishable_sampled_digest(self, tmp_path, monkeypatch):
        """A sampled digest must never compare equal to a full one for the same bytes."""
        from terrainflow_assessment.modules import dem_loader

        path = self._write_dem(str(tmp_path / "a.tif"), size=64)
        full = dem_loader.dem_content_digest(path)

        monkeypatch.setattr(dem_loader, "_FULL_HASH_MAX_BYTES", 1)
        monkeypatch.setattr(dem_loader, "_HASH_CHUNK_BYTES", 64)
        sampled = dem_loader.dem_content_digest(path)

        assert sampled.startswith("sha256-sampled:")
        assert sampled != full
        assert sampled == dem_loader.dem_content_digest(path)

    def test_fingerprint_carries_every_field_the_design_file_needs(self, tmp_path):
        from terrainflow_assessment.modules.dem_loader import fingerprint_dem
        path = self._write_dem(str(tmp_path / "a.tif"))
        fp = fingerprint_dem(path)

        assert fp["fingerprint"]
        assert fp["original_path"] == path
        assert fp["cell_size_m"] == pytest.approx(1.0)
        assert "2193" in fp["crs"]
        assert len(fp["extent"]) == 4
        assert (fp["width"], fp["height"]) == (10, 10)

    def test_fingerprint_accepts_an_already_loaded_info(self, tmp_path):
        from terrainflow_assessment.modules.dem_loader import fingerprint_dem
        path = self._write_dem(str(tmp_path / "a.tif"))
        assert fingerprint_dem(path, info=load_dem(path)) == fingerprint_dem(path)

    def test_fingerprint_feeds_the_design_file_reference(self, tmp_path):
        """End-to-end: dem_loader produces identity, project_io compares it."""
        from terrainflow_assessment.modules.dem_loader import fingerprint_dem
        from terrainflow_assessment.modules.project_io import DemReference

        a = self._write_dem(str(tmp_path / "a.tif"), fill=50.0)
        b = self._write_dem(str(tmp_path / "b.tif"), fill=75.0)

        saved = DemReference.from_dict(fingerprint_dem(a))
        assert saved.matches(DemReference.from_dict(fingerprint_dem(a)))
        assert not saved.matches(DemReference.from_dict(fingerprint_dem(b)))

    def test_clipping_produces_a_different_identity(self, tmp_path):
        """An embedded clip is a different raster and must not pass as the original."""
        from shapely.geometry import box

        from terrainflow_assessment.modules.dem_loader import (
            clip_dem_to_polygon,
            fingerprint_dem,
        )
        from terrainflow_assessment.modules.project_io import DemReference

        source = self._write_dem(str(tmp_path / "a.tif"))
        clipped = clip_dem_to_polygon(
            source, box(2, 2, 8, 8), str(tmp_path / "clip.tif"))

        original = DemReference.from_dict(fingerprint_dem(source))
        assert not original.matches(DemReference.from_dict(fingerprint_dem(clipped)))
