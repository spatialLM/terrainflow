"""Dam wall extension — keying the wall into the banks so it actually impounds.

A wall that stops short of ground at crest height does not hold water: the pond flows
around the end regardless of how tall the wall is in the middle. This is why a dam
could report "69% full" analytically while the burned map showed a stream running over
it — the wall as drawn spanned only part of the valley.
"""

import pytest

from terrainflow_assessment.modules.earthwork_design import (
    abutment_warning,
    extend_to_abutments,
)


def _valley(floor=100.0, rise=0.5):
    """V-shaped valley along y=0: ground rises with |x| away from the centreline."""
    def _at(x, y):
        return floor + abs(x) * rise
    return _at


def _capped_valley(floor=100.0, rise=0.5, plateau_at=20.0):
    """Valley whose banks stop rising past *plateau_at* — no high abutment beyond."""
    def _at(x, y):
        return floor + min(abs(x), plateau_at) * rise
    return _at


class TestExtendToAbutments:
    def test_wall_reaches_ground_at_crest_height(self):
        # Crest 105 m; ground reaches that at |x| = 10 on a 0.5 m/m bank.
        coords, info = extend_to_abutments(
            [(-4.0, 0.0), (4.0, 0.0)], _valley(), crest_elev=105.0, step_m=1.0)
        assert info["start_keyed"] and info["end_keyed"]
        assert coords[0][0] == pytest.approx(-10.0, abs=1.0)
        assert coords[-1][0] == pytest.approx(10.0, abs=1.0)

    def test_the_wall_gets_longer_than_drawn(self):
        drawn = [(-4.0, 0.0), (4.0, 0.0)]
        coords, info = extend_to_abutments(drawn, _valley(), 105.0, step_m=1.0)
        assert len(coords) > len(drawn)
        assert info["start_m"] > 0 and info["end_m"] > 0

    def test_an_end_already_in_high_ground_is_left_alone(self):
        # The left end starts at x=-30 → ground 115 m, already above a 105 m crest.
        coords, info = extend_to_abutments(
            [(-30.0, 0.0), (4.0, 0.0)], _valley(), 105.0, step_m=1.0)
        assert info["start_m"] == 0.0
        assert info["start_keyed"] is True
        assert coords[0] == (-30.0, 0.0)
        assert info["end_m"] > 0            # the low end still extends

    def test_extension_follows_the_terminal_bearing(self):
        """A wall drawn at an angle extends along its own line, not along an axis."""
        def ground(x, y):
            return 100.0 + (x + y) * 0.5    # rises to the north-east

        coords, _ = extend_to_abutments(
            [(0.0, 0.0), (2.0, 2.0)], ground, crest_elev=110.0, step_m=1.0)
        ex, ey = coords[-1]
        assert ex == pytest.approx(ey, abs=1e-6)   # stayed on the 45° bearing
        assert ex > 2.0

    def test_no_high_ground_stops_at_the_cap_and_reports_not_keyed(self):
        coords, info = extend_to_abutments(
            [(-4.0, 0.0), (4.0, 0.0)], _capped_valley(),
            crest_elev=130.0, max_extend_m=40.0, step_m=1.0)
        assert info["start_keyed"] is False and info["end_keyed"] is False
        assert info["end_m"] == pytest.approx(40.0)

    def test_running_off_the_dem_ends_that_walk(self):
        def edge(x, y):
            return None if abs(x) > 8.0 else 100.0    # flat, then off-grid

        coords, info = extend_to_abutments(
            [(-2.0, 0.0), (2.0, 0.0)], edge, crest_elev=200.0,
            max_extend_m=100.0, step_m=1.0)
        assert info["end_keyed"] is False
        assert coords[-1][0] <= 8.0          # stopped at the last on-grid point

    def test_degenerate_inputs_return_the_line_unchanged(self):
        flat = _valley()
        for coords in ([], [(0.0, 0.0)]):
            out, info = extend_to_abutments(coords, flat, 105.0)
            assert out == coords
            assert info["start_m"] == info["end_m"] == 0.0

        out, info = extend_to_abutments([(0.0, 0.0), (1.0, 0.0)], flat, None)
        assert len(out) == 2
        assert info["start_keyed"] is False

    def test_a_zero_length_terminal_segment_is_skipped(self):
        out, info = extend_to_abutments(
            [(0.0, 0.0), (0.0, 0.0)], _valley(), 105.0, step_m=1.0)
        assert info["start_m"] == 0.0 and info["end_m"] == 0.0
        assert len(out) == 2

    def test_bad_coordinates_are_tolerated(self):
        out, _ = extend_to_abutments(["nonsense"], _valley(), 105.0)
        assert out == ["nonsense"]

    def test_a_lower_crest_needs_less_extension(self):
        _, low = extend_to_abutments(
            [(-4.0, 0.0), (4.0, 0.0)], _valley(), 103.0, step_m=1.0)
        _, high = extend_to_abutments(
            [(-4.0, 0.0), (4.0, 0.0)], _valley(), 108.0, step_m=1.0)
        assert low["end_m"] < high["end_m"]


class TestAbutmentWarning:
    def test_silent_when_both_ends_key_in(self):
        info = {"start_keyed": True, "end_keyed": True}
        assert abutment_warning("Dam 4", info, 105.0, 250.0) is None

    def test_names_the_open_end_and_the_crest(self):
        info = {"start_keyed": True, "end_keyed": False}
        msg = abutment_warning("Dam 4", info, 105.0, 250.0)
        assert "Dam 4" in msg and "105.00 m" in msg and "east/end" in msg
        assert "west/start" not in msg

    def test_reports_both_ends_when_neither_keys_in(self):
        info = {"start_keyed": False, "end_keyed": False}
        msg = abutment_warning("Dam 4", info, 105.0, 250.0)
        assert "west/start" in msg and "east/end" in msg
