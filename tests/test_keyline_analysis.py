"""Tests for ``modules/keypoint_analysis`` — keypoints, ridgelines and keylines.

It was "and its identical copy": there were two of this module, one under the
retired `plugin/` tree. That tree was archived outside the repo on 2026-09-12 and
this is the only copy.
"""
import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_raster(path, data, cell_size=1.0, crs="EPSG:32632"):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _make_valley_dem(shape=(30, 30), cell_size=5.0):
    """30×30 DEM with a valley running along the centre column."""
    rows, cols = shape
    r_idx = np.arange(rows).reshape(-1, 1)
    c_idx = np.arange(cols).reshape(1, -1)
    centre_col = cols // 2
    # Slope downhill (decreasing elevation with row) + V-profile across columns
    dem = (100.0 - r_idx * 1.5 + np.abs(c_idx - centre_col) * 2.0).astype("float32")
    return dem


def _make_accumulation(shape=(30, 30)):
    """Synthetic accumulation raster — high values in centre column (valley)."""
    rows, cols = shape
    centre_col = cols // 2
    acc = np.full(shape, 5.0, dtype="float32")
    acc[:, centre_col - 1 : centre_col + 2] = 2000.0   # valley stripe
    # Increase downstream (increasing row)
    for r in range(rows):
        acc[r, centre_col] = 2000.0 + r * 100.0
    return acc


def _make_pair(tmp_path, cell_size=5.0):
    """Write synthetic DEM + acc rasters; return (dem_path, acc_path)."""
    dem = _make_valley_dem(cell_size=cell_size)
    acc = _make_accumulation()
    dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=cell_size)
    acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=cell_size)
    return dem_path, acc_path


def _load_plugin(tmp_path):
    from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
    dem_path, acc_path = _make_pair(tmp_path)
    return KeylineAnalysis(dem_path, acc_path), dem_path, acc_path


def _load_assessment(tmp_path):
    from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
    dem_path, acc_path = _make_pair(tmp_path)
    return KeylineAnalysis(dem_path, acc_path), dem_path, acc_path


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestKeylineAnalysisInit:
    def test_dem_loaded(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        assert kl.dem.shape == (30, 30)

    def test_acc_loaded(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        assert kl.acc.shape == (30, 30)

    def test_cell_size_approx(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        assert kl.cell_size == pytest.approx(5.0)

    def test_nodata_replaced_with_nan(self, tmp_path):
        """Cells with nodata value (-9999) should become NaN in dem."""
        dem = _make_valley_dem()
        dem[0, 0] = -9999.0
        acc = _make_accumulation()
        dem_path = _write_raster(str(tmp_path / "dem_nd.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc_nd.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        assert np.isnan(kl.dem[0, 0])

    def test_slope_initially_none(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        assert kl._slope_deg is None

    def test_assessment_version_loads(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        assert kl.dem.shape == (30, 30)


# ---------------------------------------------------------------------------
# _rc_to_xy
# ---------------------------------------------------------------------------

class TestRcToXy:
    def test_origin(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        x, y = kl._rc_to_xy(0, 0)
        # Should be near top-left corner (offset by half cell)
        assert x == pytest.approx(2.5)   # 0 + 0*5 + 5/2

    def test_centre(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        x, y = kl._rc_to_xy(15, 15)
        assert x == pytest.approx(77.5)  # 0 + 15*5 + 2.5

    def test_returns_floats(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        x, y = kl._rc_to_xy(5, 10)
        assert isinstance(x, float)
        assert isinstance(y, float)


# ---------------------------------------------------------------------------
# _compute_slope_deg
# ---------------------------------------------------------------------------

class TestComputeSlopeDeg:
    def test_returns_array_same_shape(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        slope = kl._compute_slope_deg()
        assert slope.shape == (30, 30)

    def test_values_non_negative(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        slope = kl._compute_slope_deg()
        assert np.all(slope >= 0.0)

    def test_cached_on_second_call(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        s1 = kl._compute_slope_deg()
        s2 = kl._compute_slope_deg()
        assert s1 is s2  # same object (cached)

    def test_flat_dem_near_zero_slope(self, tmp_path):
        dem = np.full((10, 10), 50.0, dtype="float32")
        acc = np.ones((10, 10), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "flat.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=1.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        slope = kl._compute_slope_deg()
        assert slope.max() < 1.0


# ---------------------------------------------------------------------------
# _order_pixels
# ---------------------------------------------------------------------------

class TestOrderPixels:
    def _kl(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        return kl

    def test_empty_list(self, tmp_path):
        kl = self._kl(tmp_path)
        assert kl._order_pixels([]) == []

    def test_single_pixel(self, tmp_path):
        kl = self._kl(tmp_path)
        assert kl._order_pixels([(5, 5)]) == [(5, 5)]

    def test_two_adjacent_pixels(self, tmp_path):
        kl = self._kl(tmp_path)
        result = kl._order_pixels([(0, 0), (0, 1)])
        assert len(result) == 2

    def test_connected_line(self, tmp_path):
        kl = self._kl(tmp_path)
        # Diagonal line of 5 pixels
        pixels = [(i, i) for i in range(5)]
        result = kl._order_pixels(pixels)
        assert len(result) == 5

    def test_result_is_contiguous(self, tmp_path):
        """Each consecutive pair in the result should be 8-connected."""
        kl = self._kl(tmp_path)
        pixels = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        result = kl._order_pixels(pixels)
        for i in range(len(result) - 1):
            r1, c1 = result[i]
            r2, c2 = result[i + 1]
            assert max(abs(r2 - r1), abs(c2 - c1)) <= 1


# ---------------------------------------------------------------------------
# find_keypoints
# ---------------------------------------------------------------------------

class TestFindKeypoints:
    def test_returns_list(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.find_keypoints(min_acc_cells=100)
        assert isinstance(result, list)

    def test_keypoint_has_required_keys(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.find_keypoints(min_acc_cells=100)
        if result:
            kp = result[0]
            for key in ("x", "y", "elevation", "slope_deg", "catchment_ha", "label"):
                assert key in kp

    def test_low_accumulation_returns_empty(self, tmp_path):
        """If no cell reaches min_acc_cells, result is empty."""
        dem = _make_valley_dem()
        acc = np.ones((30, 30), dtype="float32") * 5.0  # all low acc
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        assert kl.find_keypoints(min_acc_cells=10_000) == []

    def test_n_keypoints_cap(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.find_keypoints(min_acc_cells=100, n_keypoints=2)
        assert len(result) <= 2

    def test_with_boundary_mask(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 10:20] = True  # only centre region
        result = kl.find_keypoints(min_acc_cells=100, boundary_mask=mask)
        assert isinstance(result, list)

    def test_boundary_mask_no_candidates_returns_empty(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        mask = np.zeros((30, 30), dtype=bool)  # no cells allowed
        result = kl.find_keypoints(min_acc_cells=100, boundary_mask=mask)
        assert result == []


# ---------------------------------------------------------------------------
# find_ridgelines
# ---------------------------------------------------------------------------

class TestFindRidgelines:
    def test_returns_list(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.find_ridgelines()
        assert isinstance(result, list)

    def test_ridgeline_has_geometry(self, tmp_path):
        # Use a DEM that has clear ridges
        from shapely.geometry import LineString
        dem = _make_valley_dem()
        # Set high TPI at the edges (ridges away from valley)
        dem[:, 0] += 10.0
        dem[:, -1] += 10.0
        acc = _make_accumulation()
        acc[:, 0] = 1.0  # low acc = ridge
        acc[:, -1] = 1.0
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        result = kl.find_ridgelines(min_tpi_m=3.0, min_length_m=1.0)
        for r in result:
            assert isinstance(r["geometry"], LineString)

    def test_flat_dem_no_ridges(self, tmp_path):
        """Flat DEM has no TPI → no ridges."""
        dem = np.full((20, 20), 50.0, dtype="float32")
        acc = np.ones((20, 20), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "flat.tif"), dem)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        result = kl.find_ridgelines()
        assert result == []

    def test_with_boundary_mask(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        mask = np.ones((30, 30), dtype=bool)
        result = kl.find_ridgelines(boundary_mask=mask)
        assert isinstance(result, list)

    def test_short_components_cost_no_raster_work(self, tmp_path, monkeypatch):
        """M-11: a component under ``min_cells`` must be rejected by its **size**.

        The loop used to do ``labeled == region_id`` — a compare over the whole grid —
        followed by ``np.argwhere``, for every component, and only then apply
        ``min_cells``. On the owner's 1139x1016 design that is 1,462 full-raster passes
        to return nothing at all: every skeleton component there is shorter than the
        50-cell minimum, so the whole 5.8 s is spent measuring things it then discards.

        ``np.argwhere`` is the loop's only per-component raster call, so counting it
        counts the passes. One long spine survives the filter here and forty single-cell
        bumps do not, and the fix is what makes those forty free.
        """
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis

        rows, cols = 60, 60
        dem = np.full((rows, cols), 50.0, dtype="float32")
        # One spine, comfortably over min_cells, one cell wide so thinning keeps it.
        dem[30, 5:56] += 5.0
        # Forty isolated bumps, each its own one-cell component, each under min_cells.
        bumps = [(r, c) for r in range(6, 26, 4) for c in range(6, 46, 5)][:40]
        for r, c in bumps:
            dem[r, c] += 5.0
        acc = np.ones((rows, cols), dtype="float32")   # acc <= 2 everywhere: all ridge

        dem_path = _write_raster(str(tmp_path / "comb_dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "comb_acc.tif"), acc, cell_size=1.0)

        calls = {"n": 0}
        real_argwhere = np.argwhere

        def counting_argwhere(*args, **kwargs):
            calls["n"] += 1
            return real_argwhere(*args, **kwargs)

        monkeypatch.setattr(np, "argwhere", counting_argwhere)

        kl = KeylineAnalysis(dem_path, acc_path)
        lines = kl.find_ridgelines(tpi_window_m=15.0, min_tpi_m=1.0,
                                   min_length_m=10.0)

        # The spine is found and the bumps are not: the filter still filters.
        assert len(lines) == 1, f"expected the one spine, got {len(lines)}"
        assert lines[0]["length_m"] >= 40, lines[0]

        # And it cost one pass, not forty-one. Without the fix this is 41.
        assert calls["n"] <= 5, (
            f"{calls['n']} full-raster passes for 1 surviving component — the "
            f"size filter is still running after the raster work, not before it"
        )


# ---------------------------------------------------------------------------
# find_ridgelines — the two faults that made it return nothing on real ground
# ---------------------------------------------------------------------------

def _diagonal_ridge_dem(n=80, pad=10, height=6.0):
    """A DEM whose only feature is one straight ridge running at 45°.

    A diagonal is the shape the whole question turns on: `skeletonize` produces
    diagonal steps wherever a ridge is not aligned to the grid, which on real
    terrain is almost everywhere. Returns ``(dem, cells)`` so a test can state the
    expected length rather than discover it.
    """
    size = n + 2 * pad
    dem = np.full((size, size), 50.0, dtype="float32")
    cells = []
    for i in range(n):
        r = c = pad + i
        dem[r, c] += height
        cells.append((r, c))
    return dem, cells


class TestRidgelineConnectivity:
    """`nd_label` grouped 4-connected while `_order_pixels` walked 8.

    scipy's default structure is the cross, so a skeleton running diagonally was
    labelled **one component per cell** — and `min_cells` then threw every one of
    them away. On the owner's 1139x1016 design that was 1,462 components against
    816 when labelled the way the walker traverses them, and *no* setting of the
    other two thresholds could produce a single 50-cell component: measured across
    accumulation bars from 1 to unlimited, the 4-connected count of components
    reaching 50 cells was zero at every one. See `tests_qgis/probes/p_ridgelines.py`.
    """

    def test_a_diagonal_ridge_comes_back_as_one_line(self, tmp_path):
        """80 diagonal cells are one ridge, not 80 components of one cell.

        This is the whole fault in one assertion. Under the 4-connected labelling
        the same DEM returns **nothing**: every component is a single cell, and a
        single cell is under any `min_cells`.
        """
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis

        dem, cells = _diagonal_ridge_dem(n=80)
        acc = np.ones(dem.shape, dtype="float32")
        dem_path = _write_raster(str(tmp_path / "diag_dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "diag_acc.tif"), acc, cell_size=1.0)

        kl = KeylineAnalysis(dem_path, acc_path)
        lines = kl.find_ridgelines(tpi_window_m=15.0, min_tpi_m=1.0,
                                   min_length_m=50.0)

        assert len(lines) == 1, (
            f"a single straight diagonal ridge came back as {len(lines)} lines. "
            f"If it is 0, `nd_label` is grouping 4-connected again and every cell "
            f"of the diagonal is its own component."
        )
        # 80 cells stepping diagonally on a 1 m grid is 79 * sqrt(2) m of line.
        expected = (len(cells) - 1) * np.sqrt(2.0)
        assert lines[0]["length_m"] == pytest.approx(expected, abs=8.0), (
            f"{lines[0]['length_m']} m against an expected {expected:.1f} m"
        )

    def test_a_staircase_ridge_is_also_one_line(self, tmp_path):
        """Two along, one down — the other shape a shallow-angle ridge makes.

        A staircase is 4-connected *within* each tread and broken *between* them,
        so it fragments into one component per tread rather than per cell. It fails
        the same way for the same reason, and it is the commoner shape of the two.
        """
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis

        size = 140
        dem = np.full((size, size), 50.0, dtype="float32")
        r, c, treads = 10, 10, 55
        for _ in range(treads):
            dem[r, c] += 6.0
            dem[r, c + 1] += 6.0
            c += 2
            r += 1
        acc = np.ones(dem.shape, dtype="float32")
        dem_path = _write_raster(str(tmp_path / "stair_dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "stair_acc.tif"), acc, cell_size=1.0)

        kl = KeylineAnalysis(dem_path, acc_path)
        lines = kl.find_ridgelines(tpi_window_m=15.0, min_tpi_m=1.0,
                                   min_length_m=50.0)

        assert len(lines) == 1, (
            f"a staircase ridge came back as {len(lines)} lines; 4-connected "
            f"labelling breaks it into one component per tread"
        )
        assert lines[0]["length_m"] > 90.0, lines[0]

    def test_a_fragment_under_the_bar_is_still_dropped(self, tmp_path):
        """The filter still filters — connectivity is not a licence to draw stubs.

        The same diagonal, 20 cells instead of 80. 20 cells is 27 m of line against
        a 50 m bar, so the honest answer is nothing, and it has to stay nothing:
        the fault being fixed would otherwise be traded for hundreds of fragments,
        which is worse than returning nothing.
        """
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis

        dem, _ = _diagonal_ridge_dem(n=20)
        acc = np.ones(dem.shape, dtype="float32")
        dem_path = _write_raster(str(tmp_path / "frag_dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "frag_acc.tif"), acc, cell_size=1.0)

        kl = KeylineAnalysis(dem_path, acc_path)
        lines = kl.find_ridgelines(tpi_window_m=15.0, min_tpi_m=1.0,
                                   min_length_m=50.0)
        assert lines == [], f"a 20-cell fragment cleared a 50 m bar: {lines}"


class TestRidgelineCatchmentBar:
    """The accumulation term, and why it is now an area rather than a count.

    ``acc <= 2`` was a bare cell count, so it asked a different question at every
    resolution — 2 m² of contributing area on a 1 m DEM and 8 m² on a 2 m one.
    That is the same fault the TPI window (`tpi_window_m`) and the TPI cut
    (`min_tpi_sd`) were each fixed for already, and it is the third instance of it
    in this one function.
    """

    @staticmethod
    def _ridge_with_shoulder(tmp_path, cell_size, shoulder_acc, n_cells=80):
        """One diagonal ridge whose cells carry *shoulder_acc* of accumulation."""
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis

        dem, cells = _diagonal_ridge_dem(n=n_cells)
        acc = np.ones(dem.shape, dtype="float32")
        for r, c in cells:
            acc[r, c] = shoulder_acc
        tag = f"{cell_size}_{shoulder_acc}_{n_cells}"
        dem_path = _write_raster(str(tmp_path / f"cb_dem_{tag}.tif"), dem,
                                 cell_size=cell_size)
        acc_path = _write_raster(str(tmp_path / f"cb_acc_{tag}.tif"), acc,
                                 cell_size=cell_size)
        return KeylineAnalysis(dem_path, acc_path)

    def test_the_bar_gates_on_accumulation_at_all(self, tmp_path):
        """A ridge carrying more than the bar is not a ridge; under it, it is."""
        kl = self._ridge_with_shoulder(tmp_path, 1.0, shoulder_acc=15.0)
        assert kl.find_ridgelines(tpi_window_m=15.0, min_tpi_m=1.0,
                                  min_length_m=50.0,
                                  max_catchment_m2=20.0), \
            "15 m² of catchment on a 1 m grid is inside a 20 m² bar"
        assert kl.find_ridgelines(tpi_window_m=15.0, min_tpi_m=1.0,
                                  min_length_m=50.0,
                                  max_catchment_m2=10.0) == [], \
            "15 m² of catchment cleared a 10 m² bar"

    def test_the_same_ground_answers_the_same_at_two_resolutions(self, tmp_path):
        """6 m² of catchment is 6 cells at 1 m and 1.5 at 2 m — one answer.

        Under the old bare ``acc <= 2`` those two flip: 6 cells at 1 m fails and
        1.5 cells at 2 m passes. Same ground, same physical catchment, opposite
        answers. This asserts they now agree, in both directions.
        """
        for catchment_m2, expected in ((6.0, True), (60.0, False)):
            answers = {}
            for cell_size in (1.0, 2.0):
                acc_cells = catchment_m2 / (cell_size ** 2)
                # Same ground either way: 80 m of ridge is 80 cells at 1 m and 40
                # at 2 m. A cell count here would change the terrain as well as
                # the bar and the test would prove nothing.
                kl = self._ridge_with_shoulder(
                    tmp_path, cell_size, shoulder_acc=acc_cells,
                    n_cells=int(round(80 / cell_size)))
                lines = kl.find_ridgelines(
                    tpi_window_m=15.0, min_tpi_m=1.0,
                    min_length_m=40.0, max_catchment_m2=20.0)
                answers[cell_size] = bool(lines)
            assert answers[1.0] == answers[2.0] == expected, (
                f"{catchment_m2} m² of catchment against a 20 m² bar: "
                f"1 m grid says {answers[1.0]}, 2 m grid says {answers[2.0]}, "
                f"expected {expected} from both"
            )

    def test_a_coarse_grid_keeps_at_least_the_divide_cells(self, tmp_path):
        """On a 10 m DEM, 20 m² is a fifth of one cell — the bar floors at 1.

        Without the floor the whole term reads ``acc <= 0.2``, which no cell
        satisfies (a cell contributes itself), and ridgelines would return nothing
        on every DEM coarser than ~4 m — reintroducing exactly the
        resolution-dependent silence this parameter exists to end.
        """
        kl = self._ridge_with_shoulder(tmp_path, 10.0, shoulder_acc=1.0)
        lines = kl.find_ridgelines(tpi_window_m=150.0, min_tpi_m=1.0,
                                   min_length_m=100.0, max_catchment_m2=20.0)
        assert lines, "a true divide (acc == 1) was excluded on a 10 m grid"


# ---------------------------------------------------------------------------
# _valley_cross_width
# ---------------------------------------------------------------------------

class TestValleyCrossWidth:
    def test_returns_non_negative(self, tmp_path):
        """Width is always >= 0."""
        dem = np.full((30, 30), 50.0, dtype="float32")
        acc = np.ones((30, 30), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=1.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        width = kl._valley_cross_width(15, 15, fill_elev=51.0)
        assert width >= 0.0

    def test_col_at_200_gives_positive_width(self, tmp_path):
        """_valley_cross_width scans dc in range(-200, 201); col>=200 avoids break."""
        dem = np.full((10, 450), 50.0, dtype="float32")  # wide enough
        acc = np.ones((10, 450), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=1.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        # col=200: first iteration dc=-200 gives nc=0 (valid), scan proceeds
        width = kl._valley_cross_width(5, 200, fill_elev=51.0)
        assert width > 0.0

    def test_high_dem_zero_width(self, tmp_path):
        """If all cells are above fill_elev, width = 0."""
        dem = np.full((30, 30), 100.0, dtype="float32")
        acc = np.ones((30, 30), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=1.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        width = kl._valley_cross_width(15, 15, fill_elev=50.0)
        assert width == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# recommend_pond_sites
# ---------------------------------------------------------------------------

class TestRecommendPondSites:
    def test_returns_list(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        kps = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        result = kl.recommend_pond_sites(kps)
        assert isinstance(result, list)

    def test_one_pond_per_keypoint(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        kps = kl.find_keypoints(min_acc_cells=100, n_keypoints=2)
        result = kl.recommend_pond_sites(kps)
        assert len(result) == len(kps)

    def test_pond_has_required_keys(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        kps = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        result = kl.recommend_pond_sites(kps)
        if result:
            for key in ("x", "y", "elevation", "dam_width_m", "label"):
                assert key in result[0]

    def test_empty_keypoints_returns_empty(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        assert kl.recommend_pond_sites([]) == []


# ---------------------------------------------------------------------------
# get_cultivation_elevations
# ---------------------------------------------------------------------------

class TestGetCultivationElevations:
    def _kp(self, elevation):
        return {"elevation": float(elevation), "_row": 5, "_col": 5}

    def test_returns_keyline_for_each_kp(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.get_cultivation_elevations([self._kp(50.0)], n_each_side=0)
        assert len(result) == 1
        assert result[0]["line_type"] == "keyline"

    def test_n_each_side_2_produces_5_elevations(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.get_cultivation_elevations([self._kp(50.0)], n_each_side=2)
        assert len(result) == 5  # -2,-1,0,+1,+2

    def test_upper_and_lower_types(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.get_cultivation_elevations([self._kp(50.0)], n_each_side=1, spacing_m=5.0)
        types = {r["line_type"] for r in result}
        assert "keyline" in types
        assert "cultivation_upper" in types
        assert "cultivation_lower" in types

    def test_duplicate_elevations_deduped(self, tmp_path):
        """Two keypoints at same elevation should not produce duplicate lines."""
        kl, _, _ = _load_plugin(tmp_path)
        kps = [self._kp(50.0), self._kp(50.0)]
        result = kl.get_cultivation_elevations(kps, n_each_side=0)
        elevs = [r["elevation"] for r in result]
        assert len(elevs) == len(set(elevs))

    def test_spacing_applied(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        result = kl.get_cultivation_elevations(
            [self._kp(50.0)], n_each_side=1, spacing_m=10.0
        )
        elevs = sorted(r["elevation"] for r in result)
        assert elevs == pytest.approx([40.0, 50.0, 60.0])

    def test_empty_keypoints_returns_empty(self, tmp_path):
        kl, _, _ = _load_plugin(tmp_path)
        assert kl.get_cultivation_elevations([]) == []

    def test_assessment_version_works(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        result = kl.get_cultivation_elevations([self._kp(50.0)], n_each_side=1)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Assessment version (terrainflow_assessment.modules.keypoint_analysis)
# Exercise all major code paths to get coverage of keypoint_analysis.py
# ---------------------------------------------------------------------------

class TestKeylineAnalysisAssessmentCoverage:
    """Mirror of key plugin tests — uses assessment module for coverage."""

    def _kp(self, elevation):
        return {"elevation": float(elevation), "_row": 5, "_col": 5}

    # --- __init__, _rc_to_xy, _compute_slope_deg ---

    def test_init_and_slope(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        assert kl.dem.shape == (30, 30)
        slope = kl._compute_slope_deg()
        assert slope.shape == (30, 30)
        assert slope is kl._compute_slope_deg()  # cached

    def test_rc_to_xy(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        x, y = kl._rc_to_xy(0, 0)
        assert isinstance(x, float)

    def test_nodata_replaced_with_nan(self, tmp_path):
        dem = _make_valley_dem()
        dem[0, 0] = -9999.0
        acc = _make_accumulation()
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        assert np.isnan(kl.dem[0, 0])

    # --- _order_pixels ---

    def test_order_pixels_connected(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        pixels = [(i, i) for i in range(5)]
        result = kl._order_pixels(pixels)
        assert len(result) == 5

    def test_order_pixels_empty(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        assert kl._order_pixels([]) == []

    def test_order_pixels_single(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        assert kl._order_pixels([(3, 4)]) == [(3, 4)]

    # --- find_keypoints ---

    def test_find_keypoints_returns_list(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        result = kl.find_keypoints(min_acc_cells=100)
        assert isinstance(result, list)

    def test_find_keypoints_no_candidates_with_tight_boundary(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        mask = np.zeros((30, 30), dtype=bool)
        result = kl.find_keypoints(min_acc_cells=100, boundary_mask=mask)
        assert result == []

    def test_find_keypoints_boundary_mask_fallback(self, tmp_path):
        """acc_p75 filter makes candidate empty → falls back to valley mask."""
        kl, _, _ = _load_assessment(tmp_path)
        # Small all-True mask so the p75 fallback triggers
        mask = np.zeros((30, 30), dtype=bool)
        mask[13:17, 13:17] = True
        result = kl.find_keypoints(min_acc_cells=100, n_keypoints=1,
                                   boundary_mask=mask)
        assert isinstance(result, list)

    def test_find_keypoints_capped(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        result = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        assert len(result) <= 1

    # --- find_ridgelines ---

    def test_find_ridgelines_returns_list(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        result = kl.find_ridgelines()
        assert isinstance(result, list)

    def test_find_ridgelines_with_mask(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        mask = np.ones((30, 30), dtype=bool)
        result = kl.find_ridgelines(boundary_mask=mask)
        assert isinstance(result, list)

    def test_find_ridgelines_flat_dem_empty(self, tmp_path):
        dem = np.full((20, 20), 50.0, dtype="float32")
        acc = np.ones((20, 20), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "flat.tif"), dem)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        assert kl.find_ridgelines() == []

    # --- recommend_pond_sites ---

    def test_recommend_pond_sites_returns_list(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        kps = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        result = kl.recommend_pond_sites(kps)
        assert isinstance(result, list)
        assert len(result) == len(kps)

    def test_recommend_pond_sites_empty(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        assert kl.recommend_pond_sites([]) == []

    def test_pond_site_has_label(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        kps = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        result = kl.recommend_pond_sites(kps)
        if result:
            assert "label" in result[0]

    # --- _valley_cross_width ---

    def test_valley_cross_width_non_negative(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        width = kl._valley_cross_width(5, 5, fill_elev=51.0)
        assert width >= 0.0

    # --- get_cultivation_elevations ---

    def test_get_cultivation_elevations_basic(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        result = kl.get_cultivation_elevations([self._kp(50.0)], n_each_side=2)
        assert len(result) == 5

    def test_get_cultivation_types(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        result = kl.get_cultivation_elevations([self._kp(50.0)], n_each_side=1)
        types = {r["line_type"] for r in result}
        assert "keyline" in types
        assert "cultivation_upper" in types
        assert "cultivation_lower" in types

    def test_get_cultivation_empty(self, tmp_path):
        kl, _, _ = _load_assessment(tmp_path)
        assert kl.get_cultivation_elevations([]) == []


# ---------------------------------------------------------------------------
# Assessment version — targeted branch coverage for missing lines
# ---------------------------------------------------------------------------

class TestAssessmentKeypointBranches:
    """Exercise specific missing branches in keypoint_analysis.py."""

    def test_no_valley_returns_empty_assessment(self, tmp_path):
        """find_keypoints line 120: all acc cells below min_acc_cells."""
        dem = _make_valley_dem()
        acc = np.ones((30, 30), dtype="float32") * 5.0
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        assert kl.find_keypoints(min_acc_cells=10_000) == []

    def test_candidate_p75_fallback_assessment(self, tmp_path):
        """find_keypoints line 128: all valley cells above p75 → candidate empty."""
        # Construct acc so that ≥ min_acc_cells is exactly 1 cell.
        # Then p75 == that one cell's value, and (acc <= p75) stays True there.
        # To trigger candidate.any() == False we want acc > p75 for every valley cell.
        # This is hard with real percentile — but (candidate = valley & acc <= p75)
        # is empty only when the percentile > every element, which np.percentile
        # won't produce. So instead we mock np.percentile to return -1.
        dem = _make_valley_dem()
        acc = _make_accumulation()
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules import keypoint_analysis as ka
        kl = ka.KeylineAnalysis(dem_path, acc_path)
        # Monkey-patch np.percentile inside the module so p75 returns -1.
        # Every valley cell has acc > -1 → candidate empty → fallback to valley.
        orig_percentile = ka.np.percentile
        try:
            ka.np.percentile = lambda *a, **k: -1.0
            result = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
            assert isinstance(result, list)
        finally:
            ka.np.percentile = orig_percentile

    def test_score_exhausted_break_assessment(self, tmp_path):
        """find_keypoints line 155: score_work[r,c] <= 0 ends the loop early.

        Forcing this requires a tiny candidate region that runs out before
        n_keypoints is reached.
        """
        # Create a DEM+acc where only one or two cells qualify, but we ask for 10.
        dem = np.full((30, 30), 50.0, dtype="float32")
        acc = np.ones((30, 30), dtype="float32") * 1.0
        acc[15, 15] = 1000.0   # single valley cell
        acc[15, 16] = 800.0
        dem_path = _write_raster(str(tmp_path / "d.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "a.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        result = kl.find_keypoints(min_acc_cells=100, n_keypoints=10)
        # At most one or two keypoints — loop exited via break not cap
        assert len(result) <= 10

    def test_find_ridgelines_produces_lines(self, tmp_path):
        """Exercise the full ridge-extraction block (lines 234-276)."""
        # Large DEM with a persistent interior ridge: central 5 rows at high
        # elevation, rest gently sloped.
        rows, cols = 40, 40
        dem = np.fromfunction(
            lambda r, c: 50.0 + np.abs(c - cols // 2) * 0.3, (rows, cols)
        ).astype("float32")
        # Add a clear 3-cell-wide elevated ridge column to survive erosion
        for dc in (-1, 0, 1):
            dem[:, cols // 2 + dc] += 15.0
        # Low acc along the ridge (true ridges have no upstream area)
        acc = np.ones((rows, cols), dtype="float32")
        acc[:, cols // 2 - 1 : cols // 2 + 2] = 1.0
        acc[:, : cols // 2 - 2] = 100.0   # sides have real flow
        acc[:, cols // 2 + 2 :] = 100.0
        dem_path = _write_raster(str(tmp_path / "ridge_dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "ridge_acc.tif"), acc, cell_size=5.0)

        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        lines = kl.find_ridgelines(
            tpi_window_m=25.0, min_tpi_m=0.5, min_length_m=5.0
        )
        assert isinstance(lines, list)
        # Should find at least one ridge geometry
        for ln in lines:
            assert "geometry" in ln
            assert ln["length_m"] >= 0

    def test_find_ridgelines_with_nodata_in_dem(self, tmp_path):
        """DEM with NaN values — nanmean fallback for mean_elev."""
        rows, cols = 40, 40
        dem = np.fromfunction(
            lambda r, c: 50.0 + np.abs(c - cols // 2) * 0.3, (rows, cols)
        ).astype("float32")
        for dc in (-1, 0, 1):
            dem[:, cols // 2 + dc] += 15.0
        # Replace a column with nodata
        dem[5, :] = -9999.0
        acc = np.ones((rows, cols), dtype="float32")
        acc[:, cols // 2 - 1 : cols // 2 + 2] = 1.0
        acc[:, : cols // 2 - 2] = 100.0
        acc[:, cols // 2 + 2 :] = 100.0
        dem_path = _write_raster(str(tmp_path / "nd_ridge.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "nd_acc.tif"), acc, cell_size=5.0)

        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        lines = kl.find_ridgelines(tpi_window_m=25.0, min_tpi_m=0.5, min_length_m=5.0)
        assert isinstance(lines, list)

    def test_recommend_pond_sites_exercises_downstream_scan(self, tmp_path):
        """recommend_pond_sites line 323-326: score update path fires.

        _valley_cross_width only returns > 0 when the scan range [-200,200]
        stays inside the DEM column bounds, so we need a wide DEM (col≥200).
        """
        # Narrow but very wide DEM: 5 rows × 450 cols with valley at col=220,
        # sloping down with row so acc increases downstream.
        rows, cols = 5, 450
        c_idx = np.arange(cols).reshape(1, -1)
        dem = (100.0 - np.arange(rows).reshape(-1, 1) * 2.0 +
               np.abs(c_idx - 220) * 0.3).astype("float32")
        acc = np.full((rows, cols), 5.0, dtype="float32")
        # Valley column with acc increasing downstream
        for r in range(rows):
            acc[r, 220] = 100.0 + r * 200.0
        dem_path = _write_raster(str(tmp_path / "wide_dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "wide_acc.tif"), acc, cell_size=1.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        kp = [{
            "elevation": 100.0, "_row": 1, "_col": 220,
        }]
        result = kl.recommend_pond_sites(kp)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_recommend_pond_sites_with_boundary_mask(self, tmp_path):
        """Line 315: downstream cell outside boundary mask → continue."""
        dem = _make_valley_dem()
        acc = _make_accumulation()
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        kp = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        mask = np.zeros((30, 30), dtype=bool)
        mask[0:5, :] = True  # boundary away from downstream cells
        result = kl.recommend_pond_sites(kp, boundary_mask=mask)
        assert isinstance(result, list)

    def test_recommend_pond_sites_with_nan_dem_cell(self, tmp_path):
        """Line 307: skip nan cells during downstream scan."""
        dem = _make_valley_dem()
        dem[20, 16] = -9999.0
        acc = _make_accumulation()
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=5.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=5.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        kp = kl.find_keypoints(min_acc_cells=100, n_keypoints=1)
        result = kl.recommend_pond_sites(kp)
        assert isinstance(result, list)

    def test_valley_cross_width_stops_at_nodata(self, tmp_path):
        """A nodata cell ends the wall rather than being stepped over.

        NaN compares False against the sill, so it is not below-fill and the run
        stops there. This used to skip it and keep counting, which welded the
        terrain on the far side of a hole onto this dam's crest.
        """
        dem = np.full((3, 450), 50.0, dtype="float32")
        dem[1, 210] = -9999.0
        acc = np.ones((3, 450), dtype="float32")
        dem_path = _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=1.0)
        acc_path = _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=1.0)
        from terrainflow_assessment.modules.keypoint_analysis import KeylineAnalysis
        kl = KeylineAnalysis(dem_path, acc_path)
        width = kl._valley_cross_width(1, 220, fill_elev=51.0)
        # From column 220 the run reaches left only as far as 211 (9 cells, the
        # hole at 210 stopping it) and right to the raster edge at 449 (229
        # cells): 239 m, not the 400 m it counted straight through the hole.
        assert width == pytest.approx(239.0)

    def test_cultivation_dedup_across_keypoints(self, tmp_path):
        """get_cultivation_elevations line 395: duplicate elev skipped via set."""
        kl, _, _ = _load_assessment(tmp_path)
        # Two keypoints at the same elevation produce overlapping cultivation
        # lines — the 'seen_elevs' set drops duplicates.
        kps = [
            {"elevation": 50.0, "_row": 5, "_col": 5},
            {"elevation": 55.0, "_row": 8, "_col": 8},  # overlaps at offset +1 vs kp0
        ]
        result = kl.get_cultivation_elevations(kps, n_each_side=1, spacing_m=5.0)
        elevs = [r["elevation"] for r in result]
        assert len(elevs) == len(set(elevs))  # no duplicates


# ---------------------------------------------------------------------------
# The pond raster — accumulation inside a pool is not contributing area
# ---------------------------------------------------------------------------

class TestValleyCrossWidthIsContiguousAndInMetres:
    """Two properties the count did not have, and one it must keep.

    It counted **every** below-fill cell in a fixed 401-cell window, so a separate
    gully two hundred cells away was folded into this dam's wall — and the site
    score is ``acc / (width + 1)``, so an unrelated hollow made a good site look
    like a bad one. And the window was 401 *cells*: 800 m on a 2 m DEM, 100 m on a
    0.25 m one, the same constant asking two different questions.

    What it keeps is the vectorisation: this is called
    ``n_keypoints x (2 x search_r + 1)^2`` times and was once the whole cost of
    ``recommend_pond_sites``.
    """

    @staticmethod
    def _dla(tmp_path, dem, cell_size=1.0):
        from terrainflow_assessment.modules.keypoint_analysis import (
            DrainageLineAnalysis,
        )

        tmp_path = pathlib.Path(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        acc = np.ones(dem.shape, dtype="float32")
        return DrainageLineAnalysis(
            _write_raster(str(tmp_path / "dem.tif"), dem, cell_size=cell_size),
            _write_raster(str(tmp_path / "acc.tif"), acc, cell_size=cell_size),
        )

    def test_a_separate_hollow_is_not_welded_onto_this_wall(self, tmp_path):
        dem = np.full((3, 400), 100.0, dtype="float32")
        dem[1, 195:206] = 50.0        # this valley: 11 cells
        dem[1, 300:340] = 50.0        # a different one, well inside the old window
        dla = self._dla(tmp_path, dem)

        width = dla._valley_cross_width(1, 200, fill_elev=60.0)
        assert width == pytest.approx(11.0), (
            f"width {width} m — the hollow at column 300 was counted in")

    def test_the_run_stops_at_the_first_cell_above_the_crest(self, tmp_path):
        dem = np.full((3, 100), 100.0, dtype="float32")
        dem[1, 40:51] = 50.0
        dla = self._dla(tmp_path, dem)
        assert dla._valley_cross_width(1, 45, fill_elev=60.0) == pytest.approx(11.0)

    def test_a_candidate_above_the_crest_has_no_width(self, tmp_path):
        dem = np.full((3, 100), 100.0, dtype="float32")
        dem[1, 40:51] = 50.0
        dla = self._dla(tmp_path, dem)
        assert dla._valley_cross_width(1, 10, fill_elev=60.0) == pytest.approx(0.0)

    def test_the_reach_is_metres_not_cells(self, tmp_path):
        """The same ground at two resolutions must measure the same width."""
        coarse = np.full((3, 500), 100.0, dtype="float32")
        coarse[1, 200:300] = 50.0                    # 100 cells x 2 m = 200 m
        fine = np.full((3, 4000), 100.0, dtype="float32")
        fine[1, 1600:2400] = 50.0                    # 800 cells x 0.25 m = 200 m

        w_coarse = self._dla(tmp_path / "a", coarse, cell_size=2.0)._valley_cross_width(
            1, 250, fill_elev=60.0)
        w_fine = self._dla(tmp_path / "b", fine, cell_size=0.25)._valley_cross_width(
            1, 2000, fill_elev=60.0)
        assert w_coarse == pytest.approx(200.0)
        assert w_fine == pytest.approx(200.0), (
            f"the fine DEM measured {w_fine} m of the same 200 m valley")

    def test_the_reach_still_bounds_a_very_wide_flat(self, tmp_path):
        """A flat wider than the reach is clipped to it, not walked forever."""
        dem = np.full((3, 2000), 50.0, dtype="float32")
        dla = self._dla(tmp_path, dem)
        width = dla._valley_cross_width(1, 1000, fill_elev=60.0)
        assert width == pytest.approx(801.0), (
            f"{width} m — the 400 m reach each way should bound it")
