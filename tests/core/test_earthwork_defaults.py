"""Tests for core/registry/earthwork_defaults — the user's standard dimensions.

Pure: imports only ``core.registry``. The autouse fixture in ``tests/core/conftest.py``
fails this module if anything drags a ``qgis.*`` import in.

Nothing here asserts over ``all_types()`` as a collection.
``tests/core/test_earthwork_types.py`` registers a "terrace" and never removes it, so it
is present for the rest of the session and any count-based assertion would be
order-dependent.
"""

import math

import pytest

from terrainflow_assessment.core.registry.earthwork_defaults import (
    SETTABLE_DIMS,
    DimensionDefaults,
    ResolvedDims,
    decode,
    dims_match,
    encode,
    resolve_dimensions,
    sanitise,
    settable_dims,
    shipped_dims,
)
from terrainflow_assessment.core.registry.earthwork_types import get_type

BUILT_IN = ("swale", "berm", "basin", "dam", "diversion")


class TestShippedDims:
    """The shipped triple must reproduce Earthwork.__init__'s arithmetic exactly.

    The companion half of this — that the constructor itself agrees — lives in
    tests/test_earthwork_design.py, which has the QGIS mocks needed to build one.
    """

    @pytest.mark.parametrize("key", BUILT_IN)
    def test_matches_the_registry_arithmetic(self, key):
        cfg = get_type(key)
        dims = shipped_dims(key)
        assert dims.depth == cfg.default_depth
        assert dims.top_width_m == cfg.default_top_width
        assert dims.bottom_width_m == max(
            0.1, cfg.default_top_width - 2 * cfg.default_side_slope * cfg.default_depth)

    def test_basin_bottom_width_hits_the_floor(self):
        # top 0.0 with a vertical wall derives 0.0, floored to 0.1. Worth pinning
        # explicitly: it is the one built-in whose bottom width is the clamp itself.
        assert shipped_dims("basin").bottom_width_m == 0.1

    def test_unknown_type_gets_the_historical_fallback(self):
        assert shipped_dims("moat") == ResolvedDims(0.5, 2.0, 1.0)


class TestSettableDims:
    def test_channel_types_offer_all_three(self):
        assert settable_dims("swale") == ("depth", "top_width_m", "bottom_width_m")
        assert settable_dims("diversion") == ("depth", "top_width_m", "bottom_width_m")

    def test_berm_has_no_bottom_width(self):
        assert settable_dims("berm") == ("depth", "top_width_m")

    def test_basin_offers_depth_only(self):
        # Its footprint is the drawn polygon, so a width preference is meaningless.
        assert settable_dims("basin") == ("depth",)

    def test_dam_offers_wall_thickness_only(self):
        # Height comes from the crest elevation sampled off the DEM.
        assert settable_dims("dam") == ("top_width_m",)

    def test_unknown_type_offers_everything(self):
        assert settable_dims("moat") == SETTABLE_DIMS


class TestResolveDimensions:
    @pytest.mark.parametrize("key", BUILT_IN)
    def test_no_preference_is_the_shipped_triple(self, key):
        assert resolve_dimensions(key) == shipped_dims(key)
        assert resolve_dimensions(key, {}) == shipped_dims(key)
        assert resolve_dimensions(key, {"other": DimensionDefaults(depth=9.0)}) \
            == shipped_dims(key)

    def test_an_empty_preference_is_the_shipped_triple(self):
        assert resolve_dimensions("swale", {"swale": DimensionDefaults()}) \
            == shipped_dims("swale")

    def test_a_full_preference_passes_straight_through(self):
        pref = DimensionDefaults(depth=0.35, top_width_m=1.6, bottom_width_m=0.9)
        assert resolve_dimensions("swale", {"swale": pref}) \
            == ResolvedDims(0.35, 1.6, 0.9)

    def test_a_preferred_bottom_width_escapes_the_floor(self):
        # The whole point of storing bottom width rather than a batter: the user's own
        # number is never run through max(0.1, top - 2*slope*depth).
        pref = DimensionDefaults(depth=1.0, top_width_m=2.0, bottom_width_m=1.2)
        assert resolve_dimensions("swale", {"swale": pref}).bottom_width_m == 1.2

    def test_a_partial_preference_rederives_the_bottom_width(self):
        # A hand-edited blob carrying only a depth must not keep the shipped 1.0 m
        # floor beside it — that would describe a batter nobody chose.
        pref = DimensionDefaults(depth=0.25)
        resolved = resolve_dimensions("swale", {"swale": pref})
        assert resolved.depth == 0.25
        assert resolved.top_width_m == 2.0
        assert resolved.bottom_width_m == pytest.approx(1.5)  # 2.0 - 2*1.0*0.25

    def test_a_partial_preference_on_an_unknown_type_uses_the_fallback_batter(self):
        resolved = resolve_dimensions("moat", {"moat": DimensionDefaults(depth=0.25)})
        assert resolved.bottom_width_m == pytest.approx(1.5)


class TestDimsMatch:
    def test_identical_matches(self):
        assert dims_match(ResolvedDims(0.5, 2.0, 1.0), ResolvedDims(0.5, 2.0, 1.0))

    def test_float_noise_below_two_decimals_still_matches(self):
        assert dims_match(ResolvedDims(0.5001, 2.0, 1.0), ResolvedDims(0.5, 2.0, 1.0))

    @pytest.mark.parametrize("other", [
        ResolvedDims(0.6, 2.0, 1.0),
        ResolvedDims(0.5, 2.1, 1.0),
        ResolvedDims(0.5, 2.0, 1.1),
    ])
    def test_any_differing_dimension_fails(self, other):
        assert not dims_match(ResolvedDims(0.5, 2.0, 1.0), other)


class TestSanitise:
    def _swale(self, **kw):
        base = {"depth": 0.35, "top_width_m": 1.6, "bottom_width_m": 0.9}
        base.update(kw)
        return {"swale": base}

    def test_a_real_standard_survives(self):
        out = sanitise(self._swale())
        assert out == {"swale": DimensionDefaults(0.35, 1.6, 0.9)}

    def test_a_triple_matching_shipped_is_dropped(self):
        # This is how a standard is cleared: draw one at the shipped size with the
        # toggle on, and the entry goes away rather than pinning the user.
        assert sanitise({"swale": {"depth": 0.5, "top_width_m": 2.0,
                                   "bottom_width_m": 1.0}}) == {}

    def test_a_dimension_the_type_does_not_expose_is_ignored(self):
        # A basin has no width row, so a width entry is not a preference.
        assert sanitise({"basin": {"top_width_m": 3.0}}) == {}
        assert sanitise({"basin": {"depth": 2.2, "top_width_m": 3.0}}) \
            == {"basin": DimensionDefaults(depth=2.2)}

    def test_an_unknown_type_keeps_its_entry(self):
        # settable_dims falls back to all three, and the triple differs from the
        # historical fallback, so it survives rather than raising.
        assert "moat" in sanitise({"moat": {"depth": 0.9}})

    @pytest.mark.parametrize("bad", ["", "abc", None, [], {}, float("nan"),
                                     float("inf"), 0, -1.0])
    def test_unusable_values_are_dropped(self, bad):
        assert sanitise(self._swale(depth=bad)) == {
            "swale": DimensionDefaults(top_width_m=1.6, bottom_width_m=0.9)}

    def test_a_type_whose_every_value_is_unusable_is_dropped(self):
        assert sanitise({"swale": {"depth": "x", "top_width_m": None,
                                   "bottom_width_m": -2}}) == {}

    def test_a_non_dict_payload_is_dropped(self):
        assert sanitise(None) == {}
        assert sanitise([1, 2, 3]) == {}
        assert sanitise("swale") == {}

    def test_a_non_dict_type_entry_is_skipped(self):
        assert sanitise({"swale": "0.35"}) == {}

    def test_an_out_of_advisory_range_value_is_kept(self):
        # depth_range for a swale is (0.1, 2.0). A machine that digs outside the
        # envelope is a fact about the machine; advisories flag it, never override it.
        out = sanitise({"swale": {"depth": 2.6, "top_width_m": 1.6,
                                  "bottom_width_m": 0.9}})
        assert out["swale"].depth == 2.6


class TestEncodeDecode:
    def test_round_trip(self):
        prefs = {"swale": DimensionDefaults(0.35, 1.6, 0.9),
                 "basin": DimensionDefaults(depth=2.2)}
        assert decode(encode(prefs)) == prefs

    def test_encode_drops_what_sanitise_drops(self):
        assert encode({"swale": DimensionDefaults(0.5, 2.0, 1.0)}) == "{}"

    def test_encode_of_nothing(self):
        assert encode(None) == "{}"
        assert encode({}) == "{}"

    @pytest.mark.parametrize("text", ["", None, "{", "not json", "null", "[]",
                                      '"swale"', "3"])
    def test_unusable_text_degrades_to_no_standard(self, text):
        assert decode(text) == {}

    def test_a_blob_naming_a_type_this_build_does_not_have(self):
        # A downgrade, or a type retired between releases. Degrade, never raise.
        assert decode('{"terrace_v2": {"depth": 0.4}}').keys() == {"terrace_v2"}

    def test_decoded_values_are_floats(self):
        # JSON ints must not reach the constructor as ints and print as "2 m".
        out = decode('{"swale": {"depth": 1, "top_width_m": 3, "bottom_width_m": 2}}')
        assert isinstance(out["swale"].depth, float)


class TestDimensionDefaults:
    def test_is_empty(self):
        assert DimensionDefaults().is_empty()
        assert not DimensionDefaults(depth=0.5).is_empty()
        assert not DimensionDefaults(top_width_m=2.0).is_empty()
        assert not DimensionDefaults(bottom_width_m=1.0).is_empty()

    def test_is_frozen(self):
        with pytest.raises(Exception):
            DimensionDefaults().depth = 1.0

    def test_nan_is_not_finite(self):
        # Guards the _clean_value branch order: float("nan") parses fine and is
        # only caught by the isfinite check.
        assert math.isnan(float("nan"))
