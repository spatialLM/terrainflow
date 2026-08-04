"""Tests for modules/footprint — footprint rasterisation and terrain datums."""

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from terrainflow_assessment.modules.footprint import (
    domain_mask,
    internal_relief,
    min_dimension,
    outer_ring,
    outlet_cell,
    pour_point,
    rasterize_footprint,
)

CELL = 1.0
ROWS = COLS = 12
# Row 0 sits at the top (max y), so increasing row = decreasing y.
TRANSFORM = from_origin(0.0, ROWS * CELL, CELL, CELL)


def _tilted():
    """Plane falling south: z = 100 − row."""
    return np.fromfunction(lambda r, c: 100.0 - r, (ROWS, COLS)).astype("float64")


def _block_mask(r0, r1, c0, c1):
    m = np.zeros((ROWS, COLS), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


class TestRasterizeFootprint:
    def test_polygon_covers_its_cells(self):
        # World y for rows 4..6 spans 12-7=5 .. 12-4=8.
        poly = box(4.0, 5.0, 7.0, 8.0)
        mask = rasterize_footprint(poly, (ROWS, COLS), TRANSFORM)
        assert mask.any()
        assert mask[4:7, 4:7].all()

    def test_all_touched_widens_a_diagonal_line(self):
        line = LineString([(1.0, 1.0), (10.0, 10.0)])
        touched = rasterize_footprint(line, (ROWS, COLS), TRANSFORM, all_touched=True)
        centres = rasterize_footprint(line, (ROWS, COLS), TRANSFORM, all_touched=False)
        assert touched.sum() > centres.sum()

    def test_sub_cell_polygon_is_not_lost(self):
        """A 0.4 m square on a 1 m grid still claims the cell it sits in."""
        tiny = box(3.3, 3.3, 3.7, 3.7)
        assert rasterize_footprint(tiny, (ROWS, COLS), TRANSFORM).sum() >= 1

    def test_empty_geometry_gives_an_empty_mask(self):
        mask = rasterize_footprint(Polygon(), (ROWS, COLS), TRANSFORM)
        assert mask.shape == (ROWS, COLS)
        assert not mask.any()

    def test_none_geometry_gives_an_empty_mask(self):
        assert not rasterize_footprint(None, (ROWS, COLS), TRANSFORM).any()


class TestOuterRing:
    def test_ring_surrounds_the_block_without_overlapping_it(self):
        mask = _block_mask(4, 7, 4, 7)
        ring = outer_ring(mask)
        assert not (ring & mask).any()
        assert ring[3, 3] and ring[7, 7] and ring[3, 5]
        assert ring.sum() == 5 * 5 - 3 * 3

    def test_empty_mask_has_no_ring(self):
        assert not outer_ring(np.zeros((ROWS, COLS), dtype=bool)).any()

    def test_full_grid_has_no_ring(self):
        assert not outer_ring(np.ones((ROWS, COLS), dtype=bool)).any()


class TestPourPoint:
    def test_lowest_rim_cell_on_a_tilted_plane(self):
        dem = _tilted()
        mask = _block_mask(4, 7, 4, 7)
        elev, rc = pour_point(dem, mask)
        # The rim runs rows 3..7; the lowest is row 7 at z = 100 − 7.
        assert elev == 93.0
        assert rc[0] == 7

    def test_a_rimmed_hollow_spills_at_its_low_lip(self):
        dem = np.full((ROWS, COLS), 50.0)
        dem[4:7, 4:7] = 45.0     # the hollow
        dem[7, 5] = 48.0         # one lip lower than the rest of the rim
        elev, rc = pour_point(dem, _block_mask(4, 7, 4, 7))
        assert elev == 48.0
        assert rc == (7, 5)

    def test_empty_mask_returns_nothing(self):
        assert pour_point(_tilted(), np.zeros((ROWS, COLS), dtype=bool)) == (None, None)

    def test_falls_back_to_the_highest_inside_cell_when_there_is_no_rim(self):
        dem = _tilted()
        elev, _ = pour_point(dem, np.ones((ROWS, COLS), dtype=bool))
        assert elev == 100.0     # conservative: the most it could hold before spilling

    def test_nodata_rim_cells_are_ignored(self):
        dem = _tilted()
        dem[7, :] = -9999.0
        elev, rc = pour_point(dem, _block_mask(4, 7, 4, 7), nodata=-9999.0)
        # Row 7 is out, so the lowest usable rim cell is a row-6 side column.
        assert elev == 94.0
        assert rc[0] == 6 and rc[1] in (3, 7)


class TestOutletCell:
    def test_lowest_cell_inside_the_mask(self):
        dem = _tilted()
        assert outlet_cell(dem, _block_mask(4, 7, 4, 7))[0] == 6

    def test_empty_mask_returns_none(self):
        assert outlet_cell(_tilted(), np.zeros((ROWS, COLS), dtype=bool)) is None

    def test_all_nodata_returns_none(self):
        dem = np.full((ROWS, COLS), -9999.0)
        assert outlet_cell(dem, _block_mask(4, 7, 4, 7), nodata=-9999.0) is None


class TestInternalRelief:
    def test_relief_matches_the_slope_across_the_footprint(self):
        # Rows 4..6 on a 1 m/row plane span 2 m of fall.
        assert internal_relief(_tilted(), _block_mask(4, 7, 4, 7)) == 2.0

    def test_flat_ground_has_no_relief(self):
        assert internal_relief(np.full((ROWS, COLS), 30.0), _block_mask(2, 5, 2, 5)) == 0.0

    def test_empty_mask_is_zero(self):
        assert internal_relief(_tilted(), np.zeros((ROWS, COLS), dtype=bool)) == 0.0


class TestDomainMask:
    def test_most_specific_area_wins(self):
        analysis = [box(2.0, 2.0, 6.0, 6.0)]
        boundary = [box(0.0, 0.0, 12.0, 12.0)]
        mask = domain_mask((ROWS, COLS), TRANSFORM, polygons=[analysis, boundary])
        assert mask.sum() < ROWS * COLS      # the small area, not the whole site
        assert mask.any()

    def test_falls_back_to_the_boundary_when_no_analysis_area(self):
        boundary = [box(0.0, 0.0, 12.0, 12.0)]
        mask = domain_mask((ROWS, COLS), TRANSFORM, polygons=[[], boundary])
        assert mask.all()

    def test_falls_back_to_valid_cells_when_no_polygons(self):
        valid = np.ones((ROWS, COLS), dtype=bool)
        valid[0, :] = False                  # a nodata strip
        mask = domain_mask((ROWS, COLS), TRANSFORM, polygons=[], valid=valid)
        assert mask.sum() == (ROWS - 1) * COLS

    def test_nodata_is_excluded_from_a_polygon_area(self):
        valid = np.ones((ROWS, COLS), dtype=bool)
        valid[:, 0] = False
        mask = domain_mask((ROWS, COLS), TRANSFORM,
                           polygons=[[box(0.0, 0.0, 12.0, 12.0)]], valid=valid)
        assert not mask[:, 0].any()

    def test_empty_everything_falls_back_to_the_whole_grid(self):
        assert domain_mask((ROWS, COLS), TRANSFORM).all()

    def test_a_polygon_that_misses_the_grid_is_skipped(self):
        far_away = [box(500.0, 500.0, 510.0, 510.0)]
        boundary = [box(0.0, 0.0, 12.0, 12.0)]
        mask = domain_mask((ROWS, COLS), TRANSFORM, polygons=[far_away, boundary])
        assert mask.all()                    # falls through to the boundary


class TestMinDimension:
    def test_line_uses_its_declared_width(self):
        line = LineString([(0, 0), (100, 0)])
        assert min_dimension(line, bottom_width=0.8) == 0.8

    def test_long_thin_polygon_tends_to_its_width(self):
        strip = box(0.0, 0.0, 1000.0, 1.0)
        assert abs(min_dimension(strip) - 1.0) < 0.01

    def test_square_gives_its_inscribed_width(self):
        # 2·area/perimeter for a 10×10 square = 200/40 = 5 (the inradius).
        assert min_dimension(box(0.0, 0.0, 10.0, 10.0)) == 5.0

    def test_degenerate_geometry_returns_none(self):
        assert min_dimension(LineString([(0, 0), (1, 1)])) is None
        assert min_dimension(None) is None
        assert min_dimension(Polygon()) is None


class TestGuards:
    def test_a_group_of_empty_geometries_is_skipped(self):
        boundary = [box(0.0, 0.0, 12.0, 12.0)]
        mask = domain_mask((ROWS, COLS), TRANSFORM,
                           polygons=[[None, Polygon()], boundary])
        assert mask.all()

    def test_pour_point_with_no_usable_elevation_anywhere(self):
        dem = np.full((ROWS, COLS), -9999.0)
        assert pour_point(dem, _block_mask(4, 7, 4, 7), nodata=-9999.0) == (None, None)
