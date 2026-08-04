"""Tests for modules/flow_lines.trace_flow_lines."""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from terrainflow_assessment.modules.flow_lines import slope_vectors, trace_flow_lines


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


class TestTraceFlowLines:
    def test_sloped_dem_produces_lines(self, tmp_path):
        path = _write_dem(tmp_path / "slope.tif", _planar_slope())
        lines = trace_flow_lines(path, seed_spacing_m=10.0)
        assert isinstance(lines, list)
        assert len(lines) >= 1
        assert all(isinstance(g, LineString) for g in lines)
        assert all(len(g.coords) >= 2 for g in lines)

    def test_flat_dem_no_lines(self, tmp_path):
        path = _write_dem(tmp_path / "flat.tif", np.full((40, 40), 50.0))
        assert trace_flow_lines(path, seed_spacing_m=10.0) == []

    def test_min_length_filter(self, tmp_path):
        path = _write_dem(tmp_path / "slope2.tif", _planar_slope())
        long_only = trace_flow_lines(path, seed_spacing_m=10.0, min_length_m=1e6)
        assert long_only == []

    def test_nodata_seeds_skipped(self, tmp_path):
        data = _planar_slope()
        data[:10, :] = -9999.0
        path = _write_dem(tmp_path / "nd.tif", data, nodata=-9999.0)
        lines = trace_flow_lines(path, seed_spacing_m=10.0)
        # Still traces from valid seeds without error.
        assert isinstance(lines, list)

    def test_max_steps_caps_length(self, tmp_path):
        path = _write_dem(tmp_path / "slope4.tif", _planar_slope())
        capped = trace_flow_lines(path, seed_spacing_m=10.0, max_steps=3,
                                  min_length_m=0.0)
        # Each line traverses at most max_steps cells → few vertices.
        assert all(len(g.coords) <= 4 for g in capped)

    def test_all_nodata_returns_empty(self, tmp_path):
        data = np.full((30, 30), -9999.0, dtype="float32")
        path = _write_dem(tmp_path / "allnd.tif", data, nodata=-9999.0)
        assert trace_flow_lines(path, seed_spacing_m=10.0) == []

    def test_lines_head_downhill(self, tmp_path):
        """A traced line should end at lower elevation than it starts."""
        path = _write_dem(tmp_path / "slope3.tif", _planar_slope())
        lines = trace_flow_lines(path, seed_spacing_m=15.0)
        assert lines
        g = max(lines, key=lambda ln: ln.length)
        # Elevation decreases with row → y decreases downhill (from_origin: y at
        # top is largest). Descent moves toward smaller y.
        assert g.coords[-1][1] < g.coords[0][1]

    def test_return_slope_carries_mean_slope(self, tmp_path):
        path = _write_dem(tmp_path / "slope5.tif", _planar_slope())
        recs = trace_flow_lines(path, seed_spacing_m=15.0, return_slope=True)
        assert recs
        for r in recs:
            assert isinstance(r["geometry"], LineString)
            # 1 m drop over 1 m cell ≈ 45°.
            assert 30.0 <= r["mean_slope_deg"] <= 50.0


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

    def test_bearing_points_downhill_south(self, tmp_path):
        """Elevation falls toward +row (south) → downslope bearing ≈ 180°."""
        path = _write_dem(tmp_path / "sv_slope2.tif", _planar_slope())
        vecs = slope_vectors(path, spacing_m=15.0)
        assert vecs
        # from_origin puts row 0 at the top (north); elevation decreases with row,
        # so water heads south → bearing near 180°.
        assert all(150 <= v["angle_deg"] <= 210 for v in vecs)


class TestTraceIntoNodata:
    def test_line_stops_at_interior_nodata(self, tmp_path):
        # Slope downhill toward a nodata band in the lower half → lines stop there.
        data = _planar_slope()
        data[40:, :] = -9999.0
        path = _write_dem(tmp_path / "interior_nd.tif", data, nodata=-9999.0)
        lines = trace_flow_lines(path, seed_spacing_m=10.0, min_length_m=0.0)
        assert isinstance(lines, list)
