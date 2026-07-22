"""Tests for terrainflow_assessment/modules/burn_strategy — pure Strategy-C helpers."""
import numpy as np
import pytest

from terrainflow_assessment.modules.burn_strategy import (
    bresenham,
    enforce_monotonic_path,
    line_cells,
    ponding_resolution_warning,
    sub_cell_warning,
)


class _Transform:
    """Minimal north-up affine stub (matches rasterio Affine .a/.e/.c/.f access)."""
    def __init__(self, a=1.0, e=-1.0, c=0.0, f=20.0):
        self.a, self.e, self.c, self.f = a, e, c, f


# ---------------------------------------------------------------------------
# bresenham
# ---------------------------------------------------------------------------

class TestBresenham:
    def test_single_point(self):
        assert bresenham(3, 3, 3, 3) == [(3, 3)]

    def test_horizontal(self):
        assert bresenham(2, 0, 2, 3) == [(2, 0), (2, 1), (2, 2), (2, 3)]

    def test_vertical(self):
        assert bresenham(0, 5, 3, 5) == [(0, 5), (1, 5), (2, 5), (3, 5)]

    def test_diagonal_is_connected_and_endpoint_inclusive(self):
        cells = bresenham(0, 0, 3, 3)
        assert cells[0] == (0, 0)
        assert cells[-1] == (3, 3)
        # each step moves at most one cell in each axis (8-connected)
        for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
            assert abs(r1 - r0) <= 1 and abs(c1 - c0) <= 1

    def test_negative_direction(self):
        assert bresenham(3, 3, 0, 0)[0] == (3, 3)
        assert bresenham(3, 3, 0, 0)[-1] == (0, 0)


# ---------------------------------------------------------------------------
# line_cells
# ---------------------------------------------------------------------------

class TestLineCells:
    def test_horizontal_line_maps_to_row(self):
        # north-up: row = f - y = 20 - 10 = 10; col = x
        cells = line_cells([(2.0, 10.0), (5.0, 10.0)], _Transform(), (20, 20))
        assert cells == [(10, 2), (10, 3), (10, 4), (10, 5)]

    def test_single_vertex_snaps_to_one_cell(self):
        assert line_cells([(5.0, 10.0)], _Transform(), (20, 20)) == [(10, 5)]

    def test_out_of_bounds_vertices_dropped(self):
        # both vertices outside the 20×20 extent → empty path (no edge-cell burn)
        assert line_cells([(100.0, 100.0), (120.0, 100.0)], _Transform(), (20, 20)) == []

    def test_partial_out_of_bounds_keeps_in_bounds_cells(self):
        cells = line_cells([(2.0, 10.0), (100.0, 10.0)], _Transform(), (20, 20))
        assert (10, 2) in cells
        assert all(0 <= c < 20 for _, c in cells)

    def test_dedupes_repeated_cells(self):
        # a tiny zig within one cell shouldn't emit duplicates
        cells = line_cells([(5.0, 10.0), (5.4, 10.0), (5.0, 10.0)], _Transform(), (20, 20))
        assert len(cells) == len(set(cells))


# ---------------------------------------------------------------------------
# enforce_monotonic_path
# ---------------------------------------------------------------------------

class TestEnforceMonotonicPath:
    def test_short_path_unchanged(self):
        dem = np.full((5, 5), 10.0, dtype="float32")
        out = enforce_monotonic_path(dem, [(0, 0)])
        assert np.array_equal(out, dem)

    def test_flat_path_becomes_strictly_descending(self):
        dem = np.full((1, 5), 10.0, dtype="float32")
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        out = enforce_monotonic_path(dem, path, min_drop=0.5)
        invert = [float(out[rc]) for rc in path]
        assert invert == [10.0, 9.5, 9.0, 8.5, 8.0]

    def test_hump_is_breached_but_valleys_kept(self):
        # start high, a hump in the middle must be cut; the natural dip below the
        # running invert is preserved (not raised)
        dem = np.array([[10.0, 20.0, 3.0, 2.0]], dtype="float32")
        path = [(0, 0), (0, 1), (0, 2), (0, 3)]
        out = enforce_monotonic_path(dem, path, min_drop=1.0)
        invert = [float(out[rc]) for rc in path]
        # strictly descending
        assert all(b < a for a, b in zip(invert, invert[1:]))
        assert invert[1] == pytest.approx(9.0)  # hump 20 cut to 10-1
        assert invert[2] == pytest.approx(3.0)  # natural dip kept

    def test_reversed_when_low_end_first(self):
        # path listed low→high; function orients from the higher end
        dem = np.array([[2.0, 3.0, 10.0]], dtype="float32")
        path = [(0, 0), (0, 1), (0, 2)]
        out = enforce_monotonic_path(dem, path, min_drop=1.0)
        # highest end (col 2) stays the source; col 0 remains the low outlet
        assert float(out[(0, 2)]) == pytest.approx(10.0)
        assert float(out[(0, 0)]) <= float(out[(0, 1)])

    def test_does_not_mutate_input(self):
        dem = np.full((1, 3), 5.0, dtype="float32")
        enforce_monotonic_path(dem, [(0, 0), (0, 1), (0, 2)])
        assert np.all(dem == 5.0)


# ---------------------------------------------------------------------------
# sub_cell_warning
# ---------------------------------------------------------------------------

class TestSubCellWarning:
    def test_below_cell_size_warns(self):
        w = sub_cell_warning("Swale 1", 0.4, 1.0)
        assert w is not None
        assert "1-cell width" in w
        assert "Swale 1" in w

    def test_at_or_above_cell_size_no_warning(self):
        assert sub_cell_warning("S", 1.0, 1.0) is None
        assert sub_cell_warning("S", 2.0, 1.0) is None

    def test_none_min_dimension_no_warning(self):
        assert sub_cell_warning("S", None, 1.0) is None

    def test_zero_cell_size_no_warning(self):
        assert sub_cell_warning("S", 0.4, 0.0) is None


# ---------------------------------------------------------------------------
# ponding_resolution_warning
# ---------------------------------------------------------------------------

class TestPondingResolutionWarning:
    def test_over_cap_warns(self):
        w = ponding_resolution_warning(5_000_000, 4_000_000)
        assert w is not None
        assert "reduced resolution" in w

    def test_within_cap_no_warning(self):
        assert ponding_resolution_warning(1_000, 4_000_000) is None

    def test_exactly_at_cap_no_warning(self):
        assert ponding_resolution_warning(4_000_000, 4_000_000) is None
