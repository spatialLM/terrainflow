"""Tests for modules/terrain_indices.

Analytic surfaces do the work here. Every index below has a closed form on a plane, a
cone or a paraboloid, so the tests assert against arithmetic rather than against a
previous run — which is the only way to catch a sign convention that is self-consistent
and backwards.
"""

import numpy as np
import pytest

from terrainflow_assessment.modules.dem_loader import aspect_degrees, slope_degrees
from terrainflow_assessment.modules.terrain_indices import (
    curvature,
    landform_classes,
    landform_tpi,
    sediment_transport_index,
    slope_statistics,
    specific_catchment_area,
    stream_power_index,
    topographic_wetness_index,
)

N = 61
CENTRE = N // 2


def _grid():
    r, c = np.mgrid[0:N, 0:N].astype("float64")
    # Map coordinates: x east, y north. Row index increases southward.
    return (c - CENTRE), (CENTRE - r), r, c


def _cone():
    """A hill. Contours are circles, so flow diverges radially."""
    x, y, _r, _c = _grid()
    return 100.0 - np.hypot(x, y)


def _bowl():
    """A pit. Flow converges radially."""
    x, y, _r, _c = _grid()
    return 100.0 + np.hypot(x, y)


class TestSpecificCatchmentArea:
    def test_a_ridge_cell_contributes_its_own_footprint(self):
        """The +1, and why it is not optional.

        pysheds' accumulation counts the cells *upslope* and excludes the cell itself —
        measured on a uniform plane, the ridge row reads 0. Without the +1 a ridge cell
        has zero catchment and ln(a) is -inf along the top of every hill in the DEM.
        """
        a = specific_catchment_area(np.zeros((3, 3)), 2.0, 2.0)
        # One cell of 2x2 m, spread over a 2 m contour width.
        assert np.allclose(a, 4.0 / 2.0)

    def test_scales_with_upslope_cell_count(self):
        acc = np.array([[0.0, 1.0, 9.0]])
        a = specific_catchment_area(acc, 1.0, 1.0)
        assert np.allclose(a, [1.0, 2.0, 10.0])


class TestTopographicWetnessIndex:
    def test_matches_the_closed_form_on_a_uniform_slope(self):
        acc = np.full((5, 5), 9.0)
        slope = np.full((5, 5), 45.0)            # tan 45 = 1
        twi, floored = topographic_wetness_index(acc, slope, 1.0, 1.0)
        assert floored == 0
        assert twi == pytest.approx(np.log(10.0 / 1.0), abs=1e-5)

    def test_rises_downslope_at_constant_gradient(self):
        acc = np.tile(np.arange(N, dtype="float64")[:, None], (1, N))
        slope = np.full((N, N), 30.0)
        twi, _ = topographic_wetness_index(acc, slope, 1.0, 1.0)
        assert twi[50, 30] > twi[5, 30]

    def test_falls_as_ground_steepens_at_constant_catchment(self):
        acc = np.full((5, 5), 100.0)
        gentle, _ = topographic_wetness_index(acc, np.full((5, 5), 2.0), 1.0, 1.0)
        steep, _ = topographic_wetness_index(acc, np.full((5, 5), 30.0), 1.0, 1.0)
        assert gentle[2, 2] > steep[2, 2]

    def test_flat_ground_is_floored_and_the_count_is_returned(self):
        """A flat cell has no defensible wetness index, and the map must say how many.

        On a tile that is mostly harbour plane or terrace this is most of the picture,
        and an index computed over invented slope should be reported as such rather
        than clamped in silence.
        """
        acc = np.full((4, 4), 5.0)
        twi, floored = topographic_wetness_index(acc, np.zeros((4, 4)), 1.0, 1.0)
        assert floored == 16
        assert np.isfinite(twi).all(), "the floor exists so this cannot be infinite"

    def test_nodata_slope_propagates_as_nan_and_is_not_floored(self):
        acc = np.full((3, 3), 5.0)
        slope = np.full((3, 3), np.nan)
        twi, floored = topographic_wetness_index(acc, slope, 1.0, 1.0)
        assert np.isnan(twi).all()
        assert floored == 0, "unknown ground is not flat ground"

    def test_is_invariant_to_the_storm(self):
        """The index takes a cell COUNT, so no rainfall depth can move it.

        This is the structural guard on CLAUDE.md's two-accumulation-fields rule: had
        the volume field been wired in here by mistake, the index would move with the
        design storm and this test would fail.
        """
        acc = np.full((5, 5), 12.0)
        slope = np.full((5, 5), 10.0)
        a, _ = topographic_wetness_index(acc, slope, 1.0, 1.0)
        b, _ = topographic_wetness_index(acc, slope, 1.0, 1.0)
        assert np.array_equal(a, b)


class TestStreamPowerIndex:
    def test_zero_on_flat_ground_without_needing_a_floor(self):
        spi = stream_power_index(np.full((3, 3), 50.0), np.zeros((3, 3)), 1.0, 1.0)
        assert np.allclose(spi, 0.0)

    def test_rises_with_both_catchment_and_slope(self):
        small = stream_power_index(np.full((3, 3), 1.0), np.full((3, 3), 10.0), 1.0, 1.0)
        big = stream_power_index(np.full((3, 3), 100.0), np.full((3, 3), 10.0), 1.0, 1.0)
        steep = stream_power_index(np.full((3, 3), 1.0), np.full((3, 3), 40.0), 1.0, 1.0)
        assert big[1, 1] > small[1, 1]
        assert steep[1, 1] > small[1, 1]

    def test_raw_form_matches_a_times_tan_beta(self):
        spi = stream_power_index(np.full((2, 2), 9.0), np.full((2, 2), 45.0),
                                 1.0, 1.0, log=False)
        assert spi == pytest.approx(10.0, abs=1e-4)


class TestSedimentTransportIndex:
    def test_zero_on_the_flat(self):
        sti = sediment_transport_index(np.full((3, 3), 20.0), np.zeros((3, 3)), 1.0, 1.0)
        assert np.allclose(sti, 0.0)

    def test_rises_with_slope(self):
        gentle = sediment_transport_index(np.full((3, 3), 20.0),
                                          np.full((3, 3), 5.0), 1.0, 1.0)
        steep = sediment_transport_index(np.full((3, 3), 20.0),
                                         np.full((3, 3), 25.0), 1.0, 1.0)
        assert steep[1, 1] > gentle[1, 1]


class TestCurvature:
    """The sign convention, pinned. Every published source states a different one.

    This module's convention is: **plan positive = convex = diverging (a nose or
    ridge)**, **profile positive = convex = steepening downhill**. It is the opposite of
    Zevenbergen & Thorne's printed plan term, deliberately, so that plan curvature and
    TPI agree about which way is "ridge".
    """

    def test_a_cone_diverges_so_plan_is_positive(self):
        plan, _profile = curvature(_cone(), 1.0, 1.0)
        assert plan[CENTRE, CENTRE + 10] > 0

    def test_a_bowl_converges_so_plan_is_negative(self):
        plan, _profile = curvature(_bowl(), 1.0, 1.0)
        assert plan[CENTRE, CENTRE + 10] < 0

    def test_a_hollow_is_negative_and_a_spur_positive(self):
        """The distinction the two Yeomans cultivation patterns turn on."""
        x, _y, r, _c = _grid()
        hollow = 100.0 - 0.1 * (CENTRE - r) + 0.01 * x ** 2
        spur = 100.0 - 0.1 * (CENTRE - r) - 0.01 * x ** 2
        assert curvature(hollow, 1.0, 1.0)[0][CENTRE + 5, CENTRE] < 0
        assert curvature(spur, 1.0, 1.0)[0][CENTRE + 5, CENTRE] > 0

    def test_profile_positive_where_the_slope_steepens_downhill(self):
        _x, _y, r, _c = _grid()
        steepening = 100.0 - 0.01 * (CENTRE - r) ** 2
        _plan, profile = curvature(steepening, 1.0, 1.0)
        assert profile[CENTRE + 6, CENTRE] > 0

    def test_profile_negative_where_the_slope_eases_downhill(self):
        """This is the Yeomans keypoint's own signature on a valley floor."""
        _x, _y, r, _c = _grid()
        easing = 100.0 - 0.5 * r + 0.01 * r ** 2
        _plan, profile = curvature(easing, 1.0, 1.0)
        assert profile[CENTRE - 15, CENTRE] < 0

    def test_a_plane_has_no_curvature_either_way(self):
        _x, _y, r, c = _grid()
        plane = 100.0 - 0.3 * r - 0.2 * c
        plan, profile = curvature(plane, 1.0, 1.0)
        assert plan[CENTRE, CENTRE] == pytest.approx(0.0, abs=1e-9)
        assert profile[CENTRE, CENTRE] == pytest.approx(0.0, abs=1e-9)

    def test_flat_ground_is_zero_not_nan(self):
        plan, profile = curvature(np.full((9, 9), 7.0), 1.0, 1.0)
        assert plan[4, 4] == 0.0 and profile[4, 4] == 0.0

    def test_nodata_propagates(self):
        dem = _cone()
        dem[CENTRE, CENTRE] = np.nan
        plan, profile = curvature(dem, 1.0, 1.0)
        assert np.isnan(plan[CENTRE, CENTRE])
        assert np.isnan(profile[CENTRE, CENTRE])

    def test_anisotropic_cells_are_honoured(self):
        """The most common curvature bug: one cell size used for both axes."""
        _x, _y, r, c = _grid()
        surf = 100.0 - 0.01 * (c - CENTRE) ** 2
        square = curvature(surf, 1.0, 1.0)[1]
        wide = curvature(surf, 2.0, 1.0)[1]
        assert not np.allclose(square[CENTRE, CENTRE + 8], wide[CENTRE, CENTRE + 8])


class TestAspectAndSlopeShareOneStencil:
    """``aspect_degrees`` was added beside ``slope_degrees`` so they cannot disagree."""

    @pytest.mark.parametrize("surface,expected", [
        ("east", 90.0), ("south", 180.0), ("west", 270.0), ("north", 0.0),
    ])
    def test_cardinal_planes(self, surface, expected):
        _x, _y, r, c = _grid()
        dem = {
            "east": 100.0 - c,     # falls east
            "south": 100.0 - r,    # falls south (row increases southward)
            "west": 100.0 + c,
            "north": 100.0 + r,
        }[surface]
        assert aspect_degrees(dem, 1.0, 1.0)[CENTRE, CENTRE] == pytest.approx(expected)

    def test_flat_has_no_aspect_rather_than_facing_north(self):
        """A rainbow across a plateau would be a seam the terrain does not have."""
        assert aspect_degrees(np.full((9, 9), 4.0), 1.0, 1.0)[4, 4] == -1.0

    def test_slope_is_unchanged_by_the_stencil_extraction(self):
        """`horn_gradient` was lifted out of `slope_degrees`; the numbers must not move."""
        dem = _cone()
        got = slope_degrees(dem, 1.0, 1.0)
        # A unit cone falls 1 m per 1 m horizontally away from the apex → 45°.
        assert got[CENTRE, CENTRE + 12] == pytest.approx(45.0, abs=0.5)


def _ridge_and_hollow():
    """Alternating crests and troughs across a gently falling slope.

    A *planar* flank has zero TPI everywhere by definition — the neighbourhood mean
    equals the centre — so a surface with a crest and straight sides has no hollow to
    find. The corrugation gives both landforms at known positions: crest at x = 0,
    trough at x = ±15.
    """
    x, _y, r, _c = _grid()
    return 100.0 - 0.05 * r + 3.0 * np.cos(np.pi * x / 15.0)


class TestLandformTpi:
    def test_positive_on_a_ridge_and_negative_in_a_hollow(self):
        tpi = landform_tpi(_ridge_and_hollow(), 1.0, 1.0, window_m=9.0)
        assert tpi[CENTRE, CENTRE] > 0, "the crest should read as a ridge"
        assert tpi[CENTRE, CENTRE + 15] < 0, "the trough should read as a hollow"

    def test_the_window_is_metres_so_the_answer_survives_resampling(self):
        """A cell-count window made the landform scale silently resolution-dependent."""
        surf = _ridge_and_hollow()
        fine = landform_tpi(surf, 1.0, 1.0, window_m=9.0)
        # Same ground, 3 m cells: the window is still 9 m, so it spans 3 cells not 9.
        coarse = landform_tpi(surf[::3, ::3], 3.0, 3.0, window_m=9.0)
        assert np.sign(fine[CENTRE, CENTRE]) == np.sign(coarse[CENTRE // 3, CENTRE // 3])

    def test_nodata_does_not_fabricate_a_ridge_around_the_hole(self):
        dem = np.full((21, 21), 50.0)
        dem[8:13, 8:13] = np.nan
        tpi = landform_tpi(dem, 1.0, 1.0, window_m=5.0)
        edge = tpi[7, 7]
        assert np.isfinite(edge)
        assert abs(edge) < 1e-6, (
            "filling nodata with a global mean drags the neighbourhood mean and rings "
            "the data boundary in fabricated relief")


class TestLandformClasses:
    def test_thresholds_in_standard_deviations_not_metres(self):
        tpi = landform_tpi(_ridge_and_hollow(), 1.0, 1.0, window_m=9.0)
        classes = landform_classes(tpi, None)
        assert classes[CENTRE, CENTRE] == 1            # the crest
        assert classes[CENTRE, CENTRE + 15] == -1      # the trough
        assert set(np.unique(classes)) <= {-1, 0, 1}

    def test_a_scaled_surface_classifies_the_same_way(self):
        """SD normalisation is what makes one threshold mean the same on any relief."""
        x, _y, r, _c = _grid()
        gentle = 100.0 - 0.05 * r + 3.0 * np.cos(np.pi * x / 15.0)
        steep = 100.0 - 0.5 * r + 30.0 * np.cos(np.pi * x / 15.0)
        a = landform_classes(landform_tpi(gentle, 1.0, 1.0, 9.0), None)
        b = landform_classes(landform_tpi(steep, 1.0, 1.0, 9.0), None)
        assert np.array_equal(a, b)

    def test_unknown_ground_is_midslope_not_a_landform(self):
        tpi = np.full((5, 5), np.nan)
        assert (landform_classes(tpi, None) == 0).all()


class TestSlopeStatistics:
    def test_reports_the_distribution_not_a_single_number(self):
        slope = np.concatenate([np.full(50, 2.0), np.full(50, 20.0)]).reshape(10, 10)
        stats = slope_statistics(slope)
        assert stats["n"] == 100
        assert stats["p25"] == pytest.approx(2.0)
        assert stats["p75"] == pytest.approx(20.0)

    def test_grade_is_reported_beside_degrees(self):
        stats = slope_statistics(np.full((4, 4), 45.0))
        assert stats["p50_grade"] == pytest.approx(1.0)

    def test_the_mask_restricts_the_sample(self):
        slope = np.full((10, 10), 30.0)
        slope[:5] = 2.0
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5] = True
        assert slope_statistics(slope, mask)["p50"] == pytest.approx(2.0)

    def test_nothing_measurable_returns_none_not_a_dict_of_nan(self):
        assert slope_statistics(np.full((3, 3), np.nan)) is None
