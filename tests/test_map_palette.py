"""The shared raster ramps.

These stops used to live in the controllers, copied between baseline and
simulation and hand-mirrored into the panel's inline key. The point of the
module is that there is now one of each; the point of these tests is the two
rules the report leans on — darker means more water, and alpha is for absence
rather than for magnitude.
"""

import pytest

from terrainflow_assessment.core.registry.map_palette import (
    ASPECT_CLASSES,
    CURVATURE,
    DEFAULT_SURFACE_RUNOFF_SCALE,
    EROSIVE_POWER,
    STREAMS,
    SURFACE_RUNOFF_COLOURS,
    SURFACE_RUNOFF_FADE_TOP_M3,
    SURFACE_RUNOFF_SCALES,
    WATER_CAPTURED,
    WETNESS_INDEX,
    hex_of,
    surface_runoff_alpha,
    surface_runoff_ramp,
    visible_stops,
)

RAMPS = {"streams": STREAMS, "water captured": WATER_CAPTURED,
         "surface runoff": surface_runoff_ramp()}

#: Every ramp in the file, including the four `RAMPS` leaves out.
#:
#: `RAMPS` is the set whose *shape* is asserted — ascending fractions from 0.0 to 1.0,
#: monotone darkening. Four ramps cannot join it: `CURVATURE` diverges about a midpoint
#: so its darkening is not monotone, and `ASPECT_CLASSES` is categorical and measured in
#: compass degrees, not a 0-1 fraction.
#:
#: The **alpha** rule has no such excuse — it is about absence versus magnitude and
#: applies to a diverging and a categorical ramp exactly as it does to a sequential one.
#: Leaving them out is how `CURVATURE` came to encode |curvature| as opacity while the
#: module header said, in bold, that nothing does (H-10).
ALL_RAMPS = {
    "streams": STREAMS,
    "water captured": WATER_CAPTURED,
    "surface runoff": surface_runoff_ramp(),
    "wetness index": WETNESS_INDEX,
    "erosive power": EROSIVE_POWER,
    "curvature": CURVATURE,
    "aspect classes": ASPECT_CLASSES,
}

#: Ramps that diverge about a midpoint, where the absence stop is that midpoint
#: rather than the first stop.
_DIVERGING = {"curvature"}


class TestAlphaIsForAbsence:
    """H-10. The rule the module header states in bold, asserted over every ramp.

    Stated so it can be tested rather than remembered: a ramp has **at most one**
    transparent stop, it sits at the **absence** end, and every other stop is fully
    opaque. Surface runoff's documented exception is a fade *below* the ramp's
    bottom stop, so it does not appear in the tuples these read.

    `CURVATURE` failed this the day it was written — `gathering` and `shedding` sat
    at alpha 200, so opacity climbed 0 -> 200 -> 255 with |curvature|. That is the
    magnitude encoding the header rules out, and the reason it survived is that the
    test set stopped at three ramps.
    """

    @pytest.mark.parametrize("name", sorted(ALL_RAMPS))
    def test_at_most_one_stop_is_transparent(self, name):
        transparent = [label for _f, rgba, label in ALL_RAMPS[name] if rgba[3] == 0]
        assert len(transparent) <= 1, (
            f"{name} has {len(transparent)} transparent stops ({transparent}); "
            f"absence is one place on a ramp, not a range"
        )

    @pytest.mark.parametrize("name", sorted(ALL_RAMPS))
    def test_every_other_stop_is_fully_opaque(self, name):
        partial = [(f, rgba[3], label) for f, rgba, label in ALL_RAMPS[name]
                   if rgba[3] not in (0, 255)]
        assert not partial, (
            f"{name} carries partial alpha at {partial} — opacity between 0 and 255 "
            f"encodes magnitude, which is what this rule forbids"
        )

    @pytest.mark.parametrize("name", sorted(ALL_RAMPS))
    def test_the_transparent_stop_is_the_absence_end(self, name):
        ramp = ALL_RAMPS[name]
        transparent = [i for i, (_f, rgba, _l) in enumerate(ramp) if rgba[3] == 0]
        if not transparent:
            return                      # no absence stop is allowed (ASPECT_CLASSES)
        index = transparent[0]
        if name in _DIVERGING:
            assert ramp[index][0] == 0.0, (
                f"{name} diverges, so its transparent stop must be the 0.0 midpoint, "
                f"not the stop at {ramp[index][0]}"
            )
        else:
            assert index == 0, (
                f"{name}'s transparent stop is {ramp[index][2]!r} at position "
                f"{index}, not the first stop — alpha would then rise and fall"
            )


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

    @pytest.mark.parametrize("name", ["streams", "water captured"])
    def test_only_absence_is_transparent(self, name):
        """Water captured used to climb 0 -> 160 -> 220 in alpha, so a shallow pond
        washed out into the basemap and the layer read as a stain.

        Surface runoff is out of this list, and only it: its stops are all opaque and
        the fade below them is one colour at varying alpha, so it cannot re-order two
        stops against each other — which is the defect the rule exists to prevent. See
        ``TestTheFadeIn``, which pins the exception rather than leaving it unstated.
        """
        for fraction, rgba, _l in RAMPS[name]:
            assert (rgba[3] == 0) == (fraction == 0.0), (name, fraction, rgba)

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

    def test_the_low_end_is_light_cyan_and_opaque(self):
        """Diffuse sheet flow recedes by being light, not by being see-through.

        It used to be drawn at alpha 40. The ramp is anchored on the band maximum and
        ``log`` opens at 1e-3 of it, so on a real site that stop lands a good way in —
        and everything below it was rendered under a fifth opacity. Runoff then
        appeared to *stop* below a pond that retains its catchment, when it was drawn
        all along and simply could not be seen. Weight lives in the value now, and the
        bottom of the ramp is a colour rather than white — the fade beneath it needs
        something to fade, and white fading out over an aerial is white fading out
        over paper.
        """
        _f, rgba, label = surface_runoff_ramp()[0]
        assert label == "diffuse"
        assert rgba[3] == 255, "the bottom of the colour ramp is translucent"
        assert rgba == WATER_CAPTURED[1][1], (
            "the faintest flow and the shallowest water are drawn in different "
            f"cyans: {rgba} vs {WATER_CAPTURED[1][1]}")

    def test_the_layer_is_not_washed_out_wholesale(self):
        """Layer opacity is gone with per-stop alpha, and for the same reason.

        It was the better instrument — one number over the whole raster dilutes evenly
        instead of re-ordering the stops — but a 55% wash still hands the basemap's own
        lightness range to a ramp whose low end is light. What thins this layer now is
        the bottom couple of cubic metres fading, and nothing above them.
        """
        import terrainflow_assessment.core.registry.map_palette as palette

        assert not hasattr(palette, "SURFACE_RUNOFF_OPACITY")


class TestTheFadeIn:
    """The one place alpha carries a quantity, and the bound that makes it safe.

    Every cell on the site has runoff — the rain that landed on it has to go
    somewhere — so drawing them all solid painted the map with "it rained here".
    A hard floor answers that but draws an edge, and an edge on this layer reads
    as water *stopping* there, which is a complaint it has already collected
    twice. So the bottom couple of cubic metres fade instead.

    The rule the fade must not break: it is **one colour** at varying alpha, so
    no two stops can swap places against a light or dark background. Above the
    fade top every stop is opaque.
    """

    def test_the_stated_anchors(self):
        assert surface_runoff_alpha(0.0) == 0
        assert surface_runoff_alpha(SURFACE_RUNOFF_FADE_TOP_M3 / 2.0) == 128
        assert surface_runoff_alpha(SURFACE_RUNOFF_FADE_TOP_M3) == 255

    def test_it_is_a_straight_line_between_them(self):
        quarter = surface_runoff_alpha(SURFACE_RUNOFF_FADE_TOP_M3 / 4.0)
        assert quarter == pytest.approx(255 / 4.0, abs=1)

    def test_it_clamps_rather_than_running_past_either_end(self):
        """A negative cell is a nodata artefact, not a hole in the map, and a
        channel carrying a thousand cubic metres is not 128,000 opaque."""
        assert surface_runoff_alpha(-5.0) == 0
        assert surface_runoff_alpha(1000.0) == 255

    def test_a_cell_carrying_only_its_own_rain_is_barely_drawn(self):
        """The case the fade exists for: a 2 m grid cell in a 65 mm storm with
        nothing draining into it — 0.26 m³, and it must not read as flow."""
        own_rain_m3 = (65.0 / 1000.0) * 4.0
        assert surface_runoff_alpha(own_rain_m3) < 40

    def test_every_colour_stop_stays_opaque(self):
        """The fade is beneath the ramp, not inside it. A translucent stop above
        the fade top would be the magnitude-as-alpha defect coming back."""
        assert all(rgba[3] == 255 for _f, rgba, _l in surface_runoff_ramp())

    def test_the_fade_covers_one_colour_only(self):
        """What makes the exception safe: there is nothing below the bottom stop
        for the alpha to re-order it against."""
        assert surface_runoff_ramp()[0][0] == 0.0


class TestOvertoppingFills:
    """One fault, one colour, two reference states.

    The band is measured on the full pond, so it says "filled, this leaves over its
    own crest" whatever the storm does. Where the modelled event also reaches that
    level it says both — and the difference has to be visible without reading as a
    different kind of problem, which a second hue would.
    """

    def test_the_two_states_differ_only_in_alpha(self):
        from terrainflow_assessment.core.registry.map_palette import (
            OVERTOPPING_CAPACITY_FILL,
            OVERTOPPING_FILL,
        )

        assert OVERTOPPING_FILL[:3] == OVERTOPPING_CAPACITY_FILL[:3]
        assert OVERTOPPING_FILL[3] != OVERTOPPING_CAPACITY_FILL[3]

    def test_the_hatched_state_carries_more_alpha(self):
        """It is drawn as a diagonal hatch rather than a solid fill, and a hatch at
        the solid fill's alpha is barely on the map. The two have to land at
        comparable weight or the qualified one reads as the lesser problem."""
        from terrainflow_assessment.core.registry.map_palette import (
            OVERTOPPING_CAPACITY_FILL,
            OVERTOPPING_FILL,
        )

        assert OVERTOPPING_CAPACITY_FILL[3] > OVERTOPPING_FILL[3]


class TestKeyHelpers:
    def test_hex_drops_alpha(self):
        """A key shows the hue a stop stands for, not the blend it lands as —
        a swatch drawn at 16% alpha over a white panel is white.

        This used to paper over a real divergence: the surface-runoff stops climbed in
        alpha, so the key drew "diffuse" solid while the map drew it at 16% and the two
        disagreed about the same layer. Every ramp is opaque above zero now, so the
        behaviour is a safeguard rather than a compensation — the inputs below are kept
        translucent on purpose, to keep testing it.
        """
        assert hex_of((226, 240, 250, 40)) == "#E2F0FA"
        assert hex_of((8, 36, 110, 245)) == "#08246E"

    def test_visible_stops_drop_the_transparent_one(self):
        stops = visible_stops(WATER_CAPTURED)
        assert len(stops) == len(WATER_CAPTURED) - 1
        assert all(colour.startswith("#") for colour, _l in stops)

    def test_visible_stops_stay_in_ramp_order(self):
        assert [label for _c, label in visible_stops(STREAMS)] == [
            label for _f, rgba, label in STREAMS if rgba[3]]


class TestTheMapKeyNamesTheColoursTheMapUses:
    """The comment above these constants used to claim a test asserted them.

    It did not, and "Site boundary" printed red in the key while the map drew it
    bright blue. A key that names a colour the map does not use is worse than no
    key — the reader trusts it and looks for the wrong thing.
    """

    def test_the_boundary_swatch_is_the_colour_the_boundary_is_drawn_in(self):
        from terrainflow_assessment.core.registry.map_palette import (
            AREA_OUTLINES,
            hex_of,
        )
        from terrainflow_assessment.modules.report_model import _BOUNDARY_COLOUR

        assert _BOUNDARY_COLOUR == hex_of(AREA_OUTLINES["boundary"])

    def test_the_watercourse_swatch_is_a_stop_of_the_stream_ramp(self):
        from terrainflow_assessment.core.registry.map_palette import (
            STREAMS,
            stop_colour,
        )
        from terrainflow_assessment.modules.report_model import _map_legend

        colour = stop_colour(STREAMS, "channel")
        entries = _map_legend(_DataStub(), "flow")
        watercourse = [e for e in entries if e.label == "Watercourse"]
        assert watercourse, "the flow map's key lost its watercourse row"
        assert watercourse[0].colour == colour

    def test_the_exit_swatch_matches_the_exit_marker(self):
        """`baseline.py` styles exit points 220,0,0."""
        from terrainflow_assessment.modules.report_model import _EXIT_COLOUR

        assert _EXIT_COLOUR.upper() == "#DC0000"

    def test_the_spillway_and_link_swatches_match_their_symbols(self):
        """`_symbols.spillway_symbol` uses #1273b5; `connection_symbol` 58,96,140."""
        from terrainflow_assessment.modules.report_model import (
            _CONNECTION_COLOUR,
            _SPILLWAY_COLOUR,
        )

        assert _SPILLWAY_COLOUR.upper() == "#1273B5"
        assert _CONNECTION_COLOUR.upper() == "#3A608C"

    def test_a_stop_that_is_not_there_is_an_error_not_a_default(self):
        import pytest

        from terrainflow_assessment.core.registry.map_palette import (
            STREAMS,
            stop_colour,
        )

        with pytest.raises(KeyError):
            stop_colour(STREAMS, "estuary")


class _DataStub:
    """The little `_map_legend` needs: it reads `earthworks` and nothing else here."""
    earthworks = ()
    balance = None
    spillway_rows = ()
    maps = {}
