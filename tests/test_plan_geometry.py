"""Tests for modules/plan_geometry.py — sill placement across an alignment."""

import math

import pytest

from terrainflow_assessment.modules.plan_geometry import (
    MIN_SILL_M,
    bearing_at,
    perpendicular_sill,
)


class TestBearingAt:
    def test_due_east(self):
        assert bearing_at([(0, 0), (10, 0)], 0) == pytest.approx(0.0)

    def test_due_north(self):
        assert bearing_at([(0, 0), (0, 10)], 0) == pytest.approx(math.pi / 2)

    def test_diagonal(self):
        assert bearing_at([(0, 0), (10, 10)], 0) == pytest.approx(math.pi / 4)

    def test_uses_the_named_segment_not_the_first(self):
        pts = [(0, 0), (10, 0), (10, 10)]
        assert bearing_at(pts, 1) == pytest.approx(math.pi / 2)

    def test_segment_index_is_clamped_into_range(self):
        pts = [(0, 0), (10, 0)]
        assert bearing_at(pts, 99) == pytest.approx(0.0)
        assert bearing_at(pts, -5) == pytest.approx(0.0)

    def test_none_when_there_is_no_segment(self):
        assert bearing_at([], 0) is None
        assert bearing_at([(1, 1)], 0) is None

    def test_duplicate_vertices_fall_through_to_a_real_segment(self):
        # A zero-length segment has no direction; returning 0.0 would silently
        # mean "due east" and put the sill at a plausible but wrong angle.
        pts = [(0, 0), (0, 0), (0, 10)]
        assert bearing_at(pts, 0) == pytest.approx(math.pi / 2)

    def test_none_when_every_vertex_is_identical(self):
        assert bearing_at([(5, 5), (5, 5), (5, 5)], 0) is None

    def test_accepts_point_like_objects(self):
        class P:
            def __init__(self, x, y):
                self._x, self._y = x, y

            def x(self):
                return self._x

            def y(self):
                return self._y

        assert bearing_at([P(0, 0), P(10, 0)], 0) == pytest.approx(0.0)


class TestPerpendicularSill:
    def test_sill_is_square_across_an_east_west_line(self):
        (x1, y1), (x2, y2), bearing = perpendicular_sill(
            [(0, 0), (10, 0)], 0, (5, 0), 4.0)
        assert bearing == pytest.approx(0.0)
        assert x1 == pytest.approx(5.0)
        assert x2 == pytest.approx(5.0)
        assert {round(y1, 6), round(y2, 6)} == {-2.0, 2.0}

    def test_sill_length_equals_the_built_width(self):
        (x1, y1), (x2, y2), _ = perpendicular_sill(
            [(0, 0), (10, 0)], 0, (5, 0), 6.0)
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(6.0)

    def test_sill_is_perpendicular_to_a_diagonal(self):
        pts = [(0, 0), (10, 10)]
        (x1, y1), (x2, y2), bearing = perpendicular_sill(pts, 0, (5, 5), 2.0)
        sill = math.atan2(y2 - y1, x2 - x1)
        # Perpendicular means a quarter turn from the alignment.
        assert abs((sill - bearing) % math.pi - math.pi / 2) == pytest.approx(0.0)

    def test_centred_on_the_snapped_point(self):
        (x1, y1), (x2, y2), _ = perpendicular_sill(
            [(0, 0), (10, 0)], 0, (7, 0), 3.0)
        assert (x1 + x2) / 2 == pytest.approx(7.0)
        assert (y1 + y2) / 2 == pytest.approx(0.0)

    def test_narrow_weirs_are_floored_so_they_stay_findable(self):
        (x1, y1), (x2, y2), _ = perpendicular_sill(
            [(0, 0), (10, 0)], 0, (5, 0), 0.05)
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(MIN_SILL_M)

    @pytest.mark.parametrize("width", [0.0, None])
    def test_missing_width_still_draws_something(self, width):
        result = perpendicular_sill([(0, 0), (10, 0)], 0, (5, 0), width)
        (x1, y1), (x2, y2), _ = result
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(MIN_SILL_M)

    def test_none_when_the_alignment_has_no_direction(self):
        assert perpendicular_sill([(1, 1)], 0, (1, 1), 2.0) is None
        assert perpendicular_sill([], 0, (0, 0), 2.0) is None

    def test_uses_the_local_tangent_on_a_bent_alignment(self):
        # Corner at (10, 0): the second segment runs north, so a sill on it must
        # run east-west, not north-south.
        pts = [(0, 0), (10, 0), (10, 10)]
        (x1, y1), (x2, y2), _ = perpendicular_sill(pts, 1, (10, 5), 4.0)
        assert y1 == pytest.approx(5.0)
        assert y2 == pytest.approx(5.0)
        assert {round(x1, 6), round(x2, 6)} == {8.0, 12.0}
