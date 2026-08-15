"""The shared raster ramps.

These stops used to live in the controllers, copied between baseline and
simulation and hand-mirrored into the panel's inline key. The point of the
module is that there is now one of each; the point of these tests is the two
rules the report leans on — darker means more water, and alpha is for absence
rather than for magnitude.
"""

import pytest

from terrainflow_assessment.core.registry.map_palette import (
    DEFAULT_SURFACE_RUNOFF_SCALE,
    STREAMS,
    SURFACE_RUNOFF_COLOURS,
    SURFACE_RUNOFF_SCALES,
    WATER_CAPTURED,
    hex_of,
    surface_runoff_ramp,
    visible_stops,
)

RAMPS = {"streams": STREAMS, "water captured": WATER_CAPTURED,
         "surface runoff": surface_runoff_ramp()}


def _luma(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


class TestRampShape:
    @pytest.mark.parametrize("name", sorted(RAMPS))
    def test_fractions_ascend(self, name):
        fractions = [f for f, _c, _l in RAMPS[name]]
        assert fractions == sorted(fractions)

    @pytest.mark.parametrize("name", sorted(RAMPS))
    def test_starts_at_zero_and_ends_at_the_maximum(self, name):
        ramp = RAMPS[name]
        assert ramp[0][0] == 0.0
        assert ramp[-1][0] == 1.0

    @pytest.mark.parametrize("name", sorted(RAMPS))
    def test_every_stop_is_labelled(self, name):
        assert all(label for _f, _c, label in RAMPS[name])

    @pytest.mark.parametrize("name", sorted(RAMPS))
    def test_rgba_is_four_channels_in_range(self, name):
        for _f, rgba, _l in RAMPS[name]:
            assert len(rgba) == 4
            assert all(0 <= channel <= 255 for channel in rgba)


class TestTheTwoRules:
    @pytest.mark.parametrize("name", sorted(RAMPS))
    def test_darker_means_more_water(self, name):
        """One learnt map should read the next. Every ramp runs light to dark
        with volume, so the eye does not have to be re-taught per layer."""
        visible = [rgba for _f, rgba, _l in RAMPS[name] if rgba[3]]
        lumas = [_luma(rgba) for rgba in visible]
        assert lumas == sorted(lumas, reverse=True), lumas

    def test_only_absence_is_transparent(self):
        """Water captured used to climb 0 -> 160 -> 220 in alpha, so a shallow
        pond washed out into the basemap and the layer read as a stain."""
        for fraction, rgba, _l in WATER_CAPTURED:
            assert (rgba[3] == 0) == (fraction == 0.0), (fraction, rgba)

    def test_streams_jump_straight_to_a_solid_colour(self):
        """Every non-zero cell already passed the threshold, so there is no
        low end to fade in — and a one-cell raster line has no casing, so its
        colour is the whole of its contrast."""
        assert STREAMS[0][1][3] == 0
        assert all(rgba[3] == 255 for _f, rgba, _l in STREAMS[1:])

    def test_streams_are_dark_enough_to_read_over_pasture(self):
        """They opened at a light blue that vanished against grass, bush
        shadow and bare ground on an aerial."""
        for _f, rgba, _l in STREAMS[1:]:
            assert _luma(rgba) < 110, rgba


class TestSurfaceRunoff:
    def test_every_scale_mode_has_a_stop_per_colour(self):
        for mode, fractions in SURFACE_RUNOFF_SCALES.items():
            assert len(fractions) == len(SURFACE_RUNOFF_COLOURS), mode

    @pytest.mark.parametrize("mode", sorted(SURFACE_RUNOFF_SCALES))
    def test_each_mode_builds_a_well_formed_ramp(self, mode):
        ramp = surface_runoff_ramp(mode)
        assert [f for f, _c, _l in ramp] == list(SURFACE_RUNOFF_SCALES[mode])

    def test_an_unknown_mode_falls_back_rather_than_raising(self):
        assert surface_runoff_ramp("nonsense") == surface_runoff_ramp("log")

    def test_the_default_mode_exists(self):
        assert DEFAULT_SURFACE_RUNOFF_SCALE in SURFACE_RUNOFF_SCALES

    def test_the_low_end_stays_faint(self):
        """Diffuse sheet flow covers nearly the whole site; at any real weight
        it hides the map it is supposed to be describing."""
        _f, rgba, label = surface_runoff_ramp()[1]
        assert label == "diffuse"
        assert 0 < rgba[3] < 80


class TestKeyHelpers:
    def test_hex_drops_alpha(self):
        """A key shows the hue a stop stands for, not the blend it lands as —
        a swatch drawn at 16% alpha over a white panel is white."""
        assert hex_of((226, 240, 250, 40)) == "#E2F0FA"
        assert hex_of((8, 36, 110, 245)) == "#08246E"

    def test_visible_stops_drop_the_transparent_one(self):
        stops = visible_stops(WATER_CAPTURED)
        assert len(stops) == len(WATER_CAPTURED) - 1
        assert all(colour.startswith("#") for colour, _l in stops)

    def test_visible_stops_stay_in_ramp_order(self):
        assert [label for _c, label in visible_stops(STREAMS)] == [
            label for _f, rgba, label in STREAMS if rgba[3]]
