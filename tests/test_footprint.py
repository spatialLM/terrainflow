"""Tests for modules/footprint — footprint rasterisation and terrain datums."""

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from terrainflow_assessment.modules.footprint import (
    DOMAIN_FROM_POLYGON,
    DOMAIN_FROM_VALID_DEM,
    DOMAIN_FROM_WHOLE_GRID,
    clip_raster_to_polygons,
    domain_fallback_warning,
    domain_mask,
    internal_relief,
    line_points,
    min_dimension,
    outer_ring,
    outlet_cell,
    pour_point,
    rasterize_footprint,
    sample_along_line,
    xy_to_rc,
    xy_to_rc_array,
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


class TestXyToRc:
    """Map coordinate → cell, by floor rather than truncation.

    CELL is 1.0 and the grid's top-left corner is (0.0, 12.0), so a point at
    y = 12.5 or x = -0.5 is half a cell outside the grid to the north or west.
    """

    def test_a_point_inside_maps_to_its_own_cell(self):
        assert xy_to_rc(TRANSFORM, 0.5, 11.5) == (0, 0)
        assert xy_to_rc(TRANSFORM, 3.2, 8.7) == (3, 3)
        assert xy_to_rc(TRANSFORM, 11.9, 0.1) == (11, 11)

    def test_a_cell_centre_maps_to_that_cell(self):
        for r in range(ROWS):
            for c in range(COLS):
                x = TRANSFORM.c + (c + 0.5) * TRANSFORM.a
                y = TRANSFORM.f + (r + 0.5) * TRANSFORM.e
                assert xy_to_rc(TRANSFORM, x, y) == (r, c)

    def test_just_north_of_the_grid_is_out_of_range_not_row_zero(self):
        row, _ = xy_to_rc(TRANSFORM, 5.0, 12.5)
        assert row == -1, (
            "int() truncates -0.5 to 0, so the point passes a `0 <= row` bounds "
            "check and the feature is burned into the top row of the DEM")

    def test_just_west_of_the_grid_is_out_of_range_not_column_zero(self):
        _, col = xy_to_rc(TRANSFORM, -0.5, 5.0)
        assert col == -1

    def test_the_south_and_east_edges_were_never_affected(self):
        """Only the north/west band truncates the wrong way — pinned as the contrast."""
        row, _ = xy_to_rc(TRANSFORM, 5.0, -0.5)
        _, col = xy_to_rc(TRANSFORM, 12.5, 5.0)
        assert row == 12 and col == 12          # past the far edge either way

    def test_bounds_checking_is_left_to_the_caller(self):
        """Out of range comes back as out of range, not clamped and not raised."""
        assert xy_to_rc(TRANSFORM, -40.0, 60.0) == (-48, -40)


class TestXyToRcArray:
    """The array form must be the scalar form, exactly — including the sharp edges.

    It replaces ``xy_to_rc`` inside the vectorised samplers, so any divergence would
    move every sample they take rather than failing loudly.
    """

    def test_matches_the_scalar_form_cell_for_cell(self):
        rng = np.random.default_rng(20260910)
        # Deliberately over-range on both axes: the north/west band is where the
        # floor-vs-truncate distinction bites, so it has to be in the sample.
        xs = rng.uniform(-5.0, 17.0, 500)
        ys = rng.uniform(-5.0, 17.0, 500)

        rows, cols = xy_to_rc_array(TRANSFORM, xs, ys)
        for i, (x, y) in enumerate(zip(xs, ys)):
            assert (int(rows[i]), int(cols[i])) == xy_to_rc(TRANSFORM, x, y)

    def test_out_of_range_is_returned_not_clamped(self):
        rows, cols = xy_to_rc_array(TRANSFORM, [-0.5, 5.0], [5.0, 12.5])
        assert cols[0] == -1
        assert rows[1] == -1

    def test_shape_is_preserved(self):
        rows, cols = xy_to_rc_array(TRANSFORM, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert rows.shape == (3,) and cols.shape == (3,)


class TestLinePoints:
    """``line_points`` must agree with ``geom.interpolate`` — that is its whole claim."""

    def test_matches_shapely_interpolate_on_a_polyline(self):
        line = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        dists = np.linspace(0.0, line.length, 37)

        xs, ys = line_points(list(line.coords), dists)
        for i, d in enumerate(dists):
            pt = line.interpolate(float(d))
            assert xs[i] == pytest.approx(pt.x, abs=1e-9)
            assert ys[i] == pytest.approx(pt.y, abs=1e-9)

    def test_duplicate_vertices_do_not_break_the_chainage(self):
        """A repeated vertex is a zero-length segment, and np.interp needs increasing xp."""
        coords = [(0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        line = LineString(coords)
        dists = np.linspace(0.0, line.length, 11)

        xs, ys = line_points(coords, dists)
        for i, d in enumerate(dists):
            pt = line.interpolate(float(d))
            assert xs[i] == pytest.approx(pt.x, abs=1e-9)
            assert ys[i] == pytest.approx(pt.y, abs=1e-9)

    def test_distances_past_the_ends_clamp_as_interpolate_does(self):
        coords = [(0.0, 0.0), (10.0, 0.0)]
        xs, ys = line_points(coords, [-5.0, 15.0])
        assert (xs[0], ys[0]) == (0.0, 0.0)
        assert (xs[1], ys[1]) == (10.0, 0.0)

    def test_a_degenerate_line_returns_its_only_point(self):
        xs, ys = line_points([(3.0, 4.0), (3.0, 4.0)], [0.0, 1.0])
        assert list(xs) == [3.0, 3.0]
        assert list(ys) == [4.0, 4.0]


class TestSampleAlongLine:
    """The vectorised sampler against the point-at-a-time loop it replaces."""

    @staticmethod
    def _scalar_loop(line, array, dists, fill, nodata=None):
        out = []
        rows, cols = array.shape
        for d in dists:
            pt = line.interpolate(float(d))
            r, c = xy_to_rc(TRANSFORM, pt.x, pt.y)
            if 0 <= r < rows and 0 <= c < cols:
                v = float(array[r, c])
                if nodata is not None and v == nodata:
                    v = float("nan")
                out.append(v)
            else:
                out.append(fill)
        return np.asarray(out, dtype="float64")

    def test_equals_the_scalar_loop_it_replaces(self):
        array = _tilted()
        line = LineString([(0.5, 11.5), (11.5, 11.5), (11.5, 0.5)])
        dists = np.linspace(0.0, line.length, 61)

        got = sample_along_line(list(line.coords), TRANSFORM, array, dists, fill=0.0)
        want = self._scalar_loop(line, array, dists, fill=0.0)
        np.testing.assert_allclose(got, want)

    def test_off_grid_points_take_the_fill_and_keep_their_slot(self):
        """A profile indexed by position cannot drop samples, so length is fixed."""
        array = _tilted()
        # Runs off the west edge and back on.
        line = LineString([(-6.0, 6.0), (6.0, 6.0)])
        dists = np.linspace(0.0, line.length, 13)

        got = sample_along_line(list(line.coords), TRANSFORM, array, dists, fill=0.0)
        assert got.shape == dists.shape
        assert got[0] == 0.0                       # off-grid → fill
        assert got[-1] == array[6, 5]              # on-grid → the real value

    def test_declared_nodata_becomes_nan_rather_than_a_number(self):
        """The defect this exists for: -9999 averaged as a slope, not excluded."""
        array = np.full((ROWS, COLS), 12.0)
        array[6, :] = -9999.0
        line = LineString([(0.5, 5.5), (11.5, 5.5)])   # runs along the nodata row
        dists = np.linspace(0.0, line.length, 12)

        got = sample_along_line(list(line.coords), TRANSFORM, array, dists,
                                nodata=-9999.0, fill=np.nan)
        assert np.isnan(got).all()

        raw = sample_along_line(list(line.coords), TRANSFORM, array, dists, fill=np.nan)
        assert raw.mean() == pytest.approx(-9999.0), (
            "without the nodata argument the sentinel is averaged as a real value — "
            "which is how a hole became the flattest ground on the site")

    def test_a_whole_line_off_the_grid_is_all_fill(self):
        array = _tilted()
        line = LineString([(-50.0, -50.0), (-40.0, -50.0)])
        got = sample_along_line(list(line.coords), TRANSFORM, array,
                                np.linspace(0.0, 10.0, 5), fill=np.nan)
        assert np.isnan(got).all()


class TestDomainSource:
    """Which branch answered — because the mask alone cannot say, and it matters.

    On the Quail Island tile the DEM declares a nodata sentinel and contains no nodata
    cells, so "every usable DEM cell" was all 2.85 km², 70% of it harbour. The mask looked
    like a site; it was a guess.
    """

    def test_a_drawn_area_reports_polygon(self):
        mask, source = domain_mask((ROWS, COLS), TRANSFORM,
                                   polygons=[[box(2.0, 2.0, 6.0, 6.0)]],
                                   with_source=True)
        assert source == DOMAIN_FROM_POLYGON
        assert mask.any()

    def test_falling_through_to_the_dem_reports_valid_dem(self):
        valid = np.ones((ROWS, COLS), dtype=bool)
        valid[0, :] = False
        mask, source = domain_mask((ROWS, COLS), TRANSFORM, polygons=[],
                                   valid=valid, with_source=True)
        assert source == DOMAIN_FROM_VALID_DEM
        assert mask.sum() == (ROWS - 1) * COLS

    def test_falling_through_to_nothing_reports_whole_grid(self):
        mask, source = domain_mask((ROWS, COLS), TRANSFORM, with_source=True)
        assert source == DOMAIN_FROM_WHOLE_GRID
        assert mask.all()

    def test_the_mask_is_unchanged_without_the_flag(self):
        """Existing callers keep getting an array, not a tuple."""
        assert domain_mask((ROWS, COLS), TRANSFORM).all()


class TestDomainFallbackWarning:
    def test_a_drawn_site_says_nothing(self):
        assert domain_fallback_warning(DOMAIN_FROM_POLYGON, 1000, 1.0) is None

    def test_a_guessed_site_gives_its_area_in_hectares(self):
        msg = domain_fallback_warning(DOMAIN_FROM_VALID_DEM, 2_845_083, 1.0)
        assert "285 ha" in msg
        assert "Site Boundary" in msg

    def test_the_whole_grid_says_so(self):
        msg = domain_fallback_warning(DOMAIN_FROM_WHOLE_GRID, 10_000, 1.0)
        assert "the whole DEM" in msg

    def test_no_figure_is_quoted_as_a_percentage(self):
        """The denominator is what is unset; a share measured over it would be circular.

        "capture %" as the *name* of a quantity is fine — a number in front of one is not.
        """
        import re

        msg = domain_fallback_warning(DOMAIN_FROM_WHOLE_GRID, 10_000, 1.0)
        assert re.search(r"[\d.]\s*%", msg) is None


# ---------------------------------------------------------------------------
# clip_raster_to_polygons — the runoff wash stops at the boundary
# ---------------------------------------------------------------------------

class TestClipRasterToPolygons:
    """The Surface Runoff raster covers the whole tile, and on a coastal tile
    most of that tile is not the block. Masking it to the drawn boundary leaves
    the same ramp saying the same thing about the same ground."""

    def _raster(self, tmp_path, nodata=-9999.0, dtype="float32"):
        import rasterio

        path = str(tmp_path / "src.tif")
        data = np.arange(ROWS * COLS, dtype=dtype).reshape(ROWS, COLS)
        profile = dict(driver="GTiff", height=ROWS, width=COLS, count=1,
                       dtype=dtype, transform=TRANSFORM, crs="EPSG:2193")
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
        return path, data

    def _read(self, path):
        import rasterio

        with rasterio.open(path) as src:
            return src.read(1), src.nodata

    def test_inside_is_untouched_and_outside_is_nodata(self, tmp_path):
        src, data = self._raster(tmp_path)
        out = str(tmp_path / "clipped.tif")
        assert clip_raster_to_polygons(src, [box(2.0, 2.0, 6.0, 6.0)], out) == out
        got, nodata = self._read(out)
        keep = got != nodata
        assert keep.any() and not keep.all()
        assert np.array_equal(got[keep], data[keep])

    def test_the_kept_cells_are_the_ones_the_polygon_covers(self, tmp_path):
        src, _ = self._raster(tmp_path)
        out = str(tmp_path / "clipped.tif")
        poly = box(2.0, 2.0, 6.0, 6.0)
        clip_raster_to_polygons(src, [poly], out)
        got, nodata = self._read(out)
        assert np.array_equal(got != nodata,
                              rasterize_footprint(poly, (ROWS, COLS), TRANSFORM))

    def test_nothing_to_clip_to_returns_none(self, tmp_path):
        """An empty list means "print the unclipped layer", not "print an empty
        map" — the caller falls back on None."""
        src, _ = self._raster(tmp_path)
        assert clip_raster_to_polygons(src, [], str(tmp_path / "o.tif")) is None
        assert clip_raster_to_polygons(src, None, str(tmp_path / "o.tif")) is None

    def test_a_polygon_that_misses_the_raster_returns_none(self, tmp_path):
        src, _ = self._raster(tmp_path)
        out = clip_raster_to_polygons(src, [box(500.0, 500.0, 600.0, 600.0)],
                                      str(tmp_path / "o.tif"))
        assert out is None

    def test_a_float_band_with_no_sentinel_gets_nan(self, tmp_path):
        src, _ = self._raster(tmp_path, nodata=None)
        out = str(tmp_path / "clipped.tif")
        clip_raster_to_polygons(src, [box(2.0, 2.0, 6.0, 6.0)], out)
        got, nodata = self._read(out)
        assert np.isnan(nodata)
        assert np.isnan(got[0, 0])

    def test_an_integer_band_with_no_sentinel_keeps_its_zeros(self, tmp_path):
        """NaN is not a value an integer band can hold, so the band minimum
        stands in — and a real zero inside the boundary survives it."""
        src, _ = self._raster(tmp_path, nodata=None, dtype="int32")
        out = str(tmp_path / "clipped.tif")
        clip_raster_to_polygons(src, [box(0.0, 0.0, 12.0, 12.0)], out)
        got, nodata = self._read(out)
        assert nodata == 0
        assert got.max() == ROWS * COLS - 1

    def test_the_source_is_left_alone(self, tmp_path):
        """The canvas layer is the analysis output, and the numbers behind it
        are measured over the whole grid."""
        src, data = self._raster(tmp_path)
        clip_raster_to_polygons(src, [box(2.0, 2.0, 6.0, 6.0)],
                                str(tmp_path / "clipped.tif"))
        again, _ = self._read(src)
        assert np.array_equal(again, data)
