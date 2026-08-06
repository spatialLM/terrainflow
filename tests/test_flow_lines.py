"""Tests for modules/flow_lines — the sampled slope field and its hachures."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from terrainflow_assessment.modules.flow_lines import hachure_segments, slope_vectors


def _write_dem(path, data, cell=1.0, nodata=None):
    data = np.asarray(data, dtype="float32")
    transform = from_origin(0, data.shape[0] * cell, cell, cell)
    profile = {
        "driver": "GTiff", "height": data.shape[0], "width": data.shape[1],
        "count": 1, "dtype": "float32", "crs": "EPSG:32632", "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return str(path)


def _planar_slope(n=60):
    # Elevation decreases with row → consistent downhill (toward +y/south edge).
    return np.fromfunction(lambda r, c: 100.0 - r * 1.0, (n, n)).astype("float32")


class TestSlopeVectors:
    def test_returns_vectors_on_slope(self, tmp_path):
        path = _write_dem(tmp_path / "sv_slope.tif", _planar_slope())
        vecs = slope_vectors(path, spacing_m=10.0)
        assert len(vecs) >= 1
        for v in vecs:
            assert {"x", "y", "angle_deg", "slope_deg"} <= set(v)
            assert 0 <= v["angle_deg"] < 360
            assert v["slope_deg"] >= 0.5

    def test_flat_dem_no_vectors(self, tmp_path):
        path = _write_dem(tmp_path / "sv_flat.tif", np.full((30, 30), 12.0))
        assert slope_vectors(path, spacing_m=10.0) == []

    def test_nodata_cells_skipped(self, tmp_path):
        data = _planar_slope()
        data[:12, :] = -9999.0
        path = _write_dem(tmp_path / "sv_nd.tif", data, nodata=-9999.0)
        vecs = slope_vectors(path, spacing_m=8.0)
        assert isinstance(vecs, list)
        assert all(not np.isnan(v["slope_deg"]) for v in vecs)

    def test_all_nodata_returns_empty(self, tmp_path):
        data = np.full((30, 30), -9999.0, dtype="float32")
        path = _write_dem(tmp_path / "sv_allnd.tif", data, nodata=-9999.0)
        assert slope_vectors(path, spacing_m=10.0) == []

    def test_bearing_points_downhill_south(self, tmp_path):
        """Elevation falls toward +row (south) → downslope bearing ≈ 180°."""
        path = _write_dem(tmp_path / "sv_slope2.tif", _planar_slope())
        vecs = slope_vectors(path, spacing_m=15.0)
        assert vecs
        # from_origin puts row 0 at the top (north); elevation decreases with row,
        # so water heads south → bearing near 180°.
        assert all(150 <= v["angle_deg"] <= 210 for v in vecs)


class TestHachureSegments:
    def test_segments_are_two_point_lines(self, tmp_path):
        path = _write_dem(tmp_path / "h_slope.tif", _planar_slope())
        segs = hachure_segments(path, spacing_m=10.0)
        assert segs
        for s in segs:
            assert isinstance(s["geometry"], LineString)
            assert len(s["geometry"].coords) == 2
            assert s["slope_deg"] >= 0.5

    def test_length_follows_spacing_fraction(self, tmp_path):
        """Segment length is a fixed fraction of the ground spacing, so hachures
        never touch their neighbours regardless of what spacing is asked for."""
        path = _write_dem(tmp_path / "h_len.tif", _planar_slope())
        segs = hachure_segments(path, spacing_m=10.0, length_fraction=0.7)
        assert segs
        assert all(abs(s["geometry"].length - 7.0) < 1e-6 for s in segs)

    def test_spacing_controls_density(self, tmp_path):
        path = _write_dem(tmp_path / "h_dens.tif", _planar_slope())
        coarse = hachure_segments(path, spacing_m=20.0)
        fine = hachure_segments(path, spacing_m=5.0)
        assert len(fine) > len(coarse)

    def test_segments_run_downhill(self, tmp_path):
        """The stroke must start uphill and end downhill — the taper depends on it."""
        path = _write_dem(tmp_path / "h_dir.tif", _planar_slope())
        segs = hachure_segments(path, spacing_m=15.0)
        assert segs
        for s in segs:
            (_, y0), (_, y1) = s["geometry"].coords
            assert y1 < y0  # elevation falls toward smaller y on this DEM

    def test_flat_dem_no_hachures(self, tmp_path):
        path = _write_dem(tmp_path / "h_flat.tif", np.full((30, 30), 7.0))
        assert hachure_segments(path, spacing_m=10.0) == []
