"""Spillway model, crest binding and validity.

A spillway decides *where* a feature overflows: whether that happens at a place the
user chose and armoured, or at whichever point of the rim happens to be lowest. It
does **not** decide which feature spills into which — routing comes from the user's
overflow link and the D8 downslope walk (``simulation.resolve_targets``), and nothing
in the burn, the water balance or the routing reads ``Spillway.crest_elevation``. The
crest is a setting-out level and the anchor for the freeboard budget.

The arithmetic behind the crest is small, but it is bound two ways through the dialog
and clamped against what the ground can offer, and that combination is exactly where
hand-written bindings drift apart. These tests pin the round trip.
"""

import pytest

from terrainflow_assessment.modules.earthwork_design import (
    SPILLWAY_MIN_FREEBOARD_M,
    Earthwork,
    EarthworkManager,
    Spillway,
    bind_crest,
    calculate_spillway_width,
    effective_freeboard_m,
    effective_head_m,
    head_for_width,
    spillway_datum,
    spillway_policy,
    spillway_validity,
)


class _WktGeom:
    """Stand-in geometry: records its WKT and rebuilds from it."""

    def __init__(self, wkt):
        self._wkt = wkt

    def asWkt(self):
        return self._wkt


def _factory(wkt):
    return _WktGeom(wkt) if wkt else None


# ---------------------------------------------------------------------------
# spillway_datum
# ---------------------------------------------------------------------------

class TestSpillwayDatum:
    def test_ceiling_leaves_room_for_head_and_freeboard(self):
        """The crest ceiling is not the rim — a full nappe still has to clear it."""
        _lo, hi = spillway_datum(100.0, 98.0, head_m=0.30)
        assert hi == pytest.approx(100.0 - 0.30 - SPILLWAY_MIN_FREEBOARD_M)

    def test_floor_is_the_invert(self):
        lo, _hi = spillway_datum(100.0, 98.0, head_m=0.30)
        assert lo == pytest.approx(98.0)

    def test_more_head_lowers_the_ceiling(self):
        """The head/crest trade-off the dialog has to show: they compete."""
        _lo, shallow = spillway_datum(100.0, 98.0, head_m=0.20)
        _lo, deep = spillway_datum(100.0, 98.0, head_m=0.60)
        assert deep < shallow
        assert shallow - deep == pytest.approx(0.40)

    def test_unknown_rim_gives_no_band(self):
        assert spillway_datum(None, 98.0) == (None, None)

    def test_a_feature_too_shallow_for_the_head_inverts_the_band(self):
        """0.2 m deep cannot pass 0.5 m of head. An inverted band says so rather
        than quietly offering a crest that does not exist."""
        lo, hi = spillway_datum(100.0, 99.8, head_m=0.50)
        assert hi < lo

    def test_missing_invert_collapses_the_band_to_the_ceiling(self):
        lo, hi = spillway_datum(100.0, None, head_m=0.30)
        assert lo == hi


# ---------------------------------------------------------------------------
# bind_crest
# ---------------------------------------------------------------------------

class TestBindCrest:
    def test_crest_gives_drop(self):
        crest, drop, _h = bind_crest(100.0, crest=99.4)
        assert (crest, drop) == pytest.approx((99.4, 0.6))

    def test_drop_gives_crest(self):
        crest, drop, _h = bind_crest(100.0, drop=0.6)
        assert (crest, drop) == pytest.approx((99.4, 0.6))

    @pytest.mark.parametrize("value", [0.05, 0.3, 0.62, 1.4, 3.0])
    def test_round_trip_is_exact(self, value):
        """Crest → drop → crest must not drift; the dialog cycles this on every
        keystroke, so any loss compounds."""
        crest, drop, _h = bind_crest(100.0, drop=value)
        back_crest, back_drop, _h = bind_crest(100.0, crest=crest)
        assert back_crest == pytest.approx(crest)
        assert back_drop == pytest.approx(value)

    def test_absolute_crest_wins_when_both_are_given(self):
        crest, drop, _h = bind_crest(100.0, crest=99.0, drop=5.0)
        assert (crest, drop) == pytest.approx((99.0, 1.0))

    def test_clamps_into_the_band_and_the_partner_follows(self):
        """The clamp must happen before the partner is derived, or the two
        controls end up describing different crests."""
        # ceiling = rim − head − freeboard = 100.0 − 0.30 − 0.30 = 99.40
        band = spillway_datum(100.0, 98.0, head_m=0.30)
        crest, drop, _h = bind_crest(100.0, crest=99.9, band=band)
        assert crest == pytest.approx(99.40)
        assert drop == pytest.approx(0.60)
        assert crest + drop == pytest.approx(100.0)

    def test_clamps_up_to_the_floor(self):
        band = spillway_datum(100.0, 98.0, head_m=0.30)
        crest, _drop, _h = bind_crest(100.0, crest=90.0, band=band)
        assert crest == pytest.approx(98.0)

    def test_an_inverted_band_does_not_clamp(self):
        """No value satisfies an inverted band; clamping to either end would
        invent one and hide the real problem."""
        band = spillway_datum(100.0, 99.8, head_m=0.50)
        crest, _drop, _h = bind_crest(100.0, crest=99.9, band=band)
        assert crest == pytest.approx(99.9)

    def test_no_rim_passes_values_through_untouched(self):
        assert bind_crest(None, crest=99.0, drop=None) == (99.0, None, None)

    def test_nothing_given_returns_nothing(self):
        assert bind_crest(100.0) == (None, None, None)


# ---------------------------------------------------------------------------
# spillway_validity
# ---------------------------------------------------------------------------

class TestSpillwayValidity:
    def test_a_good_spillway_has_nothing_to_say(self):
        assert spillway_validity(99.4, 100.0, invert_elevation=98.0, head_m=0.30) == []

    def test_crest_above_the_rim_is_flagged(self):
        problems = spillway_validity(100.5, 100.0, head_m=0.30)
        assert any("above the lowest containing ground" in p for p in problems)

    def test_insufficient_freeboard_is_flagged_with_the_numbers(self):
        """0.2 m of drop cannot carry 0.3 m of head plus freeboard."""
        problems = spillway_validity(99.8, 100.0, head_m=0.30)
        assert problems
        assert "0.20 m between the crest and the containing" in problems[0]
        assert "0.60 m" in problems[0]      # head + NRCS-378's 0.30 m freeboard

    def test_crest_at_the_floor_stores_nothing(self):
        problems = spillway_validity(98.0, 100.0, invert_elevation=98.0, head_m=0.30)
        assert any("hold nothing" in p for p in problems)

    @pytest.mark.parametrize("head,hint", [(0.10, "wide weir"), (0.90, "freeboard")])
    def test_unusual_head_advises_in_the_right_direction(self, head, hint):
        problems = spillway_validity(97.0, 100.0, invert_elevation=95.0, head_m=head)
        assert any(hint in p for p in problems)

    def test_built_width_under_the_required_width_is_flagged(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.30,
            width_m=1.0, required_width_m=2.5)
        assert any("under the 2.5 m" in p for p in problems)

    def test_adequate_width_is_not_flagged(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.30,
            width_m=3.0, required_width_m=2.5)
        assert problems == []

    @pytest.mark.parametrize("head", [0.10, 0.20, 0.25, 0.30, 0.35, 0.50, 0.75])
    @pytest.mark.parametrize("rim", [148.30, 100.0, 12.7, 1013.45])
    def test_the_highest_crest_on_offer_is_never_reported_as_too_low(self, rim, head):
        """Regression: seeding takes the ceiling straight from spillway_datum, so a
        crest exactly at the ceiling must validate. It did not — ``rim − head −
        freeboard`` does not reconstruct ``head + freeboard`` in binary floating
        point, so a freshly seeded spillway accused itself of insufficient
        freeboard at some rim/head combinations and not others."""
        band = spillway_datum(rim, rim - 2.0, head_m=head)
        crest, _drop, _h = bind_crest(rim, crest=rim, band=band)   # clamps to the ceiling
        problems = spillway_validity(
            crest, rim, invert_elevation=rim - 2.0, head_m=head)
        assert not any("between the crest and the containing" in p for p in problems)

    def test_a_crest_genuinely_short_of_freeboard_is_still_caught(self):
        """The tolerance is a millimetre, not a licence — 10 cm short still fails."""
        problems = spillway_validity(99.65, 100.0, invert_elevation=98.0, head_m=0.30)
        assert any("between the crest and the containing" in p for p in problems)

    def test_nothing_to_check_without_a_crest_or_rim(self):
        assert spillway_validity(None, 100.0) == []
        assert spillway_validity(99.0, None) == []


# ---------------------------------------------------------------------------
# Weir sizing, tied to the crest band
# ---------------------------------------------------------------------------

class TestWeirSizing:
    def test_lower_head_needs_a_wider_weir(self):
        wide = calculate_spillway_width(0.5, 0.20)
        narrow = calculate_spillway_width(0.5, 0.50)
        assert wide > narrow

    def test_matches_the_broad_crested_formula(self):
        # L = Q / (C * H^1.5), C = Brater & King's SI wide-crest coefficient
        from terrainflow_assessment.modules.earthwork_design import (
            BROAD_CRESTED_WEIR_C,
        )
        assert calculate_spillway_width(1.0, 0.30) == pytest.approx(
            1.0 / (BROAD_CRESTED_WEIR_C * 0.30 ** 1.5), abs=0.01)

    def test_raising_head_buys_width_but_costs_crest(self):
        """The design tension in one assertion: the head that narrows the weir is
        the same head that forces the crest down."""
        _lo, hi_low = spillway_datum(100.0, 98.0, head_m=0.20)
        _lo, hi_high = spillway_datum(100.0, 98.0, head_m=0.50)
        assert calculate_spillway_width(0.5, 0.50) < calculate_spillway_width(0.5, 0.20)
        assert hi_high < hi_low


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestSpillwayPersistence:
    def _sited_swale(self):
        ew = Earthwork("swale", _WktGeom("LINESTRING (0 0, 100 0)"), "Swale 1")
        ew.spillway = Spillway(
            crest_elevation=99.55, drop_below_rim_m=0.45, head_m=0.30,
            width_m=2.4, point_wkt="POINT (50 0)", auto=False,
        )
        return ew

    def test_spillway_survives_a_round_trip(self):
        mgr = EarthworkManager()
        mgr.add(self._sited_swale())

        restored = EarthworkManager()
        restored.from_json(mgr.to_json(), geometry_factory=_factory)
        sp = restored.get(0).spillway

        assert sp is not None
        assert sp.crest_elevation == pytest.approx(99.55)
        assert sp.drop_below_rim_m == pytest.approx(0.45)
        assert sp.head_m == pytest.approx(0.30)
        assert sp.width_m == pytest.approx(2.4)
        assert sp.point_wkt == "POINT (50 0)"
        assert sp.auto is False

    def test_a_feature_without_a_spillway_restores_as_none(self):
        mgr = EarthworkManager()
        mgr.add(Earthwork("basin", _WktGeom("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), "B"))

        restored = EarthworkManager()
        restored.from_json(mgr.to_json(), geometry_factory=_factory)
        assert restored.get(0).spillway is None

    def test_auto_false_is_not_lost_to_a_truthiness_test(self):
        """``auto`` defaults True, so a restore that skips falsy values would
        silently re-enable crest tracking on a hand-set crest."""
        ew = self._sited_swale()
        data = ew.to_dict()
        assert data["spillway"]["auto"] is False
        assert Spillway.from_dict(data["spillway"]).auto is False

    def test_built_width_and_its_auto_flag_survive(self):
        """A width the user committed to must come back as *theirs*, not silently
        revert to tracking the computed requirement — that would erase the only
        state that makes the 'too narrow' warning possible."""
        ew = Earthwork("basin", _WktGeom("POLYGON ((0 0, 9 0, 9 9, 0 9, 0 0))"), "B")
        ew.spillway = Spillway(crest_elevation=99.0, width_m=1.75, width_auto=False)

        mgr = EarthworkManager()
        mgr.add(ew)
        restored = EarthworkManager()
        restored.from_json(mgr.to_json(), geometry_factory=_factory)
        sp = restored.get(0).spillway

        assert sp.width_m == pytest.approx(1.75)
        assert sp.width_auto is False

    def test_width_auto_defaults_to_tracking(self):
        assert Spillway().width_auto is True

    def test_from_dict_rejects_a_non_mapping(self):
        assert Spillway.from_dict(None) is None
        assert Spillway.from_dict("nonsense") is None

    def test_summary_reports_siting(self):
        assert "not sited" in Spillway(crest_elevation=99.0).summary()
        assert "not sited" not in Spillway(
            crest_elevation=99.0, point_wkt="POINT (0 0)").summary()
        assert Spillway().summary() == "no crest set"


class TestInflowAndOutflowSpillways:
    """Two structures, two jobs. An outflow is a weir sized to pass a peak; an inlet
    is a protected entry that stops the incoming jet cutting the bank. Keeping them
    apart also lets a routed connection be drawn between the two real points instead
    of centroid to centroid."""

    def _pair(self):
        ew = Earthwork("basin", _WktGeom("POLYGON ((0 0, 9 0, 9 9, 0 9, 0 0))"), "B1")
        ew.spillway = Spillway(crest_elevation=99.5, point_wkt="POINT (9 4)")
        ew.inflow_spillway = Spillway(crest_elevation=100.2, point_wkt="POINT (0 4)")
        return ew

    def test_both_survive_a_round_trip(self):
        mgr = EarthworkManager()
        mgr.add(self._pair())
        restored = EarthworkManager()
        restored.from_json(mgr.to_json(), geometry_factory=_factory)
        ew = restored.get(0)

        assert ew.spillway.point_wkt == "POINT (9 4)"
        assert ew.inflow_spillway.point_wkt == "POINT (0 4)"
        assert ew.inflow_spillway.crest_elevation == pytest.approx(100.2)

    def test_outflow_is_an_alias_for_the_original_field(self):
        """Projects saved before inlets existed store their outflow under
        ``spillway``, so a rename would have silently dropped it on load."""
        ew = self._pair()
        assert ew.outflow_spillway is ew.spillway
        replacement = Spillway(crest_elevation=98.0)
        ew.outflow_spillway = replacement
        assert ew.spillway is replacement

    def test_an_old_project_without_an_inlet_still_loads(self):
        ew = Earthwork("swale", _WktGeom("LINESTRING (0 0, 10 0)"), "S1")
        ew.spillway = Spillway(crest_elevation=99.0, point_wkt="POINT (5 0)")
        data = ew.to_dict()
        del data["inflow_spillway"]                 # as written before inlets existed

        restored = Earthwork.from_dict(data, geometry_factory=_factory)
        assert restored.spillway.point_wkt == "POINT (5 0)"
        assert restored.inflow_spillway is None

    def test_a_feature_may_have_an_inlet_but_no_outlet(self):
        ew = Earthwork("dam", _WktGeom("LINESTRING (0 0, 10 0)"), "D1")
        ew.inflow_spillway = Spillway(point_wkt="POINT (2 0)")
        restored = Earthwork.from_dict(ew.to_dict(), geometry_factory=_factory)
        assert restored.spillway is None
        assert restored.inflow_spillway.point_wkt == "POINT (2 0)"


# ---------------------------------------------------------------------------
# Per-type policy
# ---------------------------------------------------------------------------

class TestSpillwayPolicy:
    """Freeboard and head are per type, because 0.30 m is an embankment figure."""

    def test_a_dam_keeps_the_embankment_standard(self):
        freeboard, head, _band = spillway_policy("dam")
        assert freeboard == pytest.approx(SPILLWAY_MIN_FREEBOARD_M)
        assert head == pytest.approx(0.30)

    def test_a_basin_keeps_the_embankment_standard(self):
        assert spillway_policy("basin")[0] == pytest.approx(SPILLWAY_MIN_FREEBOARD_M)

    def test_a_swale_asks_for_less(self):
        """A cut channel has no embankment to breach, and 0.30 m would spend more
        than the whole depth of the registry's default swale."""
        freeboard, head, _band = spillway_policy("swale")
        assert freeboard < SPILLWAY_MIN_FREEBOARD_M
        assert head < 0.30

    def test_an_unknown_type_falls_back_to_the_embankment_figures(self):
        """The conservative direction: over-wide freeboard costs excavation, under-wide
        costs the structure."""
        assert spillway_policy("no-such-type") == (
            SPILLWAY_MIN_FREEBOARD_M, 0.30, (0.20, 0.50))

    def test_the_head_band_moves_with_the_type(self):
        """Or a swale designed at its own default head would warn permanently, and an
        advisory that always fires is one the user learns to skip."""
        _fb, swale_head, swale_band = spillway_policy("swale")
        assert swale_band[0] <= swale_head <= swale_band[1]
        assert swale_band != spillway_policy("dam")[2]


class TestEffectiveFreeboard:
    def test_none_inherits_the_type_policy(self):
        sp = Spillway()
        assert sp.freeboard_m is None
        assert effective_freeboard_m(sp, "swale") == pytest.approx(
            spillway_policy("swale")[0])
        assert effective_freeboard_m(sp, "dam") == pytest.approx(
            spillway_policy("dam")[0])

    def test_a_stored_value_overrides_the_type(self):
        assert effective_freeboard_m(Spillway(freeboard_m=0.05), "dam") == pytest.approx(0.05)

    def test_zero_is_an_override_not_an_absence(self):
        """0.0 is falsy; read as "unset" it would silently restore the type default and
        quietly overrule a deliberate choice."""
        assert effective_freeboard_m(Spillway(freeboard_m=0.0), "dam") == pytest.approx(0.0)

    def test_no_spillway_still_answers_with_the_type_policy(self):
        assert effective_freeboard_m(None, "swale") == pytest.approx(
            spillway_policy("swale")[0])

    def test_the_override_survives_a_round_trip(self):
        mgr = EarthworkManager()
        ew = Earthwork("swale", _WktGeom("LINESTRING (0 0, 100 0)"), "S1")
        ew.spillway = Spillway(crest_elevation=99.0, freeboard_m=0.05)
        mgr.add(ew)

        restored = EarthworkManager()
        restored.from_json(mgr.to_json(), geometry_factory=_factory)
        assert restored.get(0).spillway.freeboard_m == pytest.approx(0.05)

    def test_an_older_design_restores_as_inheriting(self):
        """Designs saved before the field existed carry no key, and must pick up the
        type policy rather than freezing whatever the constant was that day."""
        ew = Earthwork("swale", _WktGeom("LINESTRING (0 0, 10 0)"), "S1")
        ew.spillway = Spillway(crest_elevation=99.0)
        data = ew.to_dict()
        del data["spillway"]["freeboard_m"]

        restored = Earthwork.from_dict(data, geometry_factory=_factory)
        assert restored.spillway.freeboard_m is None
        assert effective_freeboard_m(restored.spillway, "swale") == pytest.approx(
            spillway_policy("swale")[0])


class TestTheDefaultSwaleIsBuildable:
    """The contradiction the per-type policy exists to remove.

    At the embankment figures a swale at the registry's own ``default_depth`` needs
    0.30 m of head plus 0.30 m of freeboard — 0.60 m of a 0.50 m dig. ``spillway_datum``
    returned an inverted band, ``bind_crest`` correctly refused to invent a crest inside
    it, and the most ordinary feature in the plugin reported that it would hold nothing.
    """

    def _default_swale(self):
        from terrainflow_assessment.core.registry.earthwork_types import get_type
        cfg = get_type("swale")
        freeboard, head, _band = spillway_policy("swale")
        return cfg.default_depth, head, freeboard

    def test_the_band_is_not_inverted(self):
        depth, head, freeboard = self._default_swale()
        rim = 100.0
        lo, hi = spillway_datum(rim, rim - depth, head_m=head, min_freeboard_m=freeboard)
        assert hi > lo, (
            f"a {depth:.2f} m swale cannot carry {head:.2f} m head + "
            f"{freeboard:.2f} m freeboard"
        )

    def test_the_seeded_crest_raises_nothing(self):
        depth, head, freeboard = self._default_swale()
        rim = 100.0
        band = spillway_datum(rim, rim - depth, head_m=head, min_freeboard_m=freeboard)
        crest, _drop, _h = bind_crest(rim, crest=band[1], band=band)
        problems = spillway_validity(
            crest, rim, invert_elevation=rim - depth, head_m=head,
            min_freeboard_m=freeboard,
            standard_freeboard_m=freeboard,
            typical_head_m=spillway_policy("swale")[2],
        )
        assert problems == [], problems

    def test_the_embankment_figures_are_what_used_to_break_it(self):
        """Pins the cause, so a future change that reverts the policy fails loudly here
        rather than silently reintroducing the error."""
        depth, _head, _freeboard = self._default_swale()
        rim = 100.0
        lo, hi = spillway_datum(rim, rim - depth, head_m=0.30, min_freeboard_m=0.30)
        assert hi < lo


# ---------------------------------------------------------------------------
# The weir equation, inverted
# ---------------------------------------------------------------------------

class TestHeadForWidth:
    def test_matches_the_broad_crested_formula(self):
        from terrainflow_assessment.modules.earthwork_design import (
            BROAD_CRESTED_WEIR_C,
        )
        # Hand-computed, NOT round-tripped through calculate_spillway_width — that
        # rounds to a centimetre, so a round trip would prove the rounding, not the
        # algebra. See test_a_rounded_width_does_not_invert_exactly.
        expected = (0.5 / (BROAD_CRESTED_WEIR_C * 2.0)) ** (2.0 / 3.0)
        assert head_for_width(0.5, 2.0) == pytest.approx(expected)

    def test_a_narrower_weir_stands_the_water_deeper(self):
        assert head_for_width(0.5, 1.0) > head_for_width(0.5, 4.0)

    def test_more_flow_over_the_same_weir_stands_deeper(self):
        assert head_for_width(1.0, 2.0) > head_for_width(0.5, 2.0)

    def test_head_grows_more_slowly_than_flow(self):
        """H scales as Q^(2/3): doubling the flow raises the head by ~1.59x, not 2x.
        This is why committing to a width is often safer than it looks — provided the
        freeboard is there to absorb it."""
        assert head_for_width(1.0, 2.0) / head_for_width(0.5, 2.0) == pytest.approx(
            2 ** (2.0 / 3.0))

    @pytest.mark.parametrize("flow,width", [(0.0, 2.0), (-1.0, 2.0), (0.5, 0.0),
                                            (0.5, -2.0), (None, 2.0), (0.5, None)])
    def test_invalid_input_is_none_not_zero(self, flow, width):
        """Deliberately not its sibling's 0.0: a head of zero is a real reading (nothing
        is flowing) and must not be confused with "these inputs say nothing"."""
        assert head_for_width(flow, width) is None

    def test_a_rounded_width_does_not_invert_exactly(self):
        """The trap this function must never be fed back into.

        ``calculate_spillway_width`` rounds to a centimetre, so inverting its answer
        returns a head a few millimetres from the one that produced it. That is far
        larger than the millimetre tolerance the elevation checks use, which is why the
        auto path must never compute an inverse at all.
        """
        width = calculate_spillway_width(0.12, 0.30)
        recovered = head_for_width(0.12, width)
        assert recovered != pytest.approx(0.30)
        assert abs(recovered - 0.30) > 0.001


class TestEffectiveHead:
    """Which weir framing applies, decided once so no caller decides it twice."""

    def test_an_auto_width_is_the_design_head_by_construction(self):
        assert effective_head_m(0.30, peak_flow_m3s=99.0, width_m=0.01,
                                width_auto=True) == pytest.approx(0.30)

    def test_an_adequate_built_width_keeps_the_design_head(self):
        """A wider weir than required passes the flow at *less* than the design head,
        so the freeboard budget is already met — and reporting a head below target as
        if it were the operating point would be its own confusion."""
        width = calculate_spillway_width(0.48, 0.30)
        assert effective_head_m(0.30, peak_flow_m3s=0.48, width_m=width * 2,
                                width_auto=False) == pytest.approx(0.30)

    def test_a_matched_built_width_keeps_the_design_head(self):
        """The rounding case: exactly-adequate must not invert."""
        width = calculate_spillway_width(0.12, 0.30)
        assert effective_head_m(0.30, peak_flow_m3s=0.12, width_m=width,
                                width_auto=False) == pytest.approx(0.30)

    def test_a_short_built_width_stands_the_water_deeper(self):
        """The case that matters. 0.48 m³/s over 2.0 m instead of the ~4.0 m it needs."""
        actual = effective_head_m(0.30, peak_flow_m3s=0.96, width_m=2.0,
                                  width_auto=False)
        assert actual > 0.30
        assert actual == pytest.approx(head_for_width(0.96, 2.0))

    @pytest.mark.parametrize("kwargs", [
        {"peak_flow_m3s": None, "width_m": 2.0},
        {"peak_flow_m3s": 0.5, "width_m": None},
        {"peak_flow_m3s": 0.0, "width_m": 2.0},
        {"peak_flow_m3s": 0.5, "width_m": 0.0},
    ])
    def test_missing_information_falls_back_to_the_design_head(self, kwargs):
        assert effective_head_m(0.30, width_auto=False, **kwargs) == pytest.approx(0.30)

    def test_no_head_stays_none(self):
        assert effective_head_m(None, peak_flow_m3s=0.5, width_m=2.0,
                                width_auto=False) is None


class TestTheSeededCrestNeverAccusesItself:
    """Regression guard for the rounding trap, in both design modes.

    ``_seed_spillway`` takes the crest straight from ``spillway_datum``'s ceiling, so
    ``rim - crest`` equals ``head + freeboard`` exactly. Feeding a head recovered from a
    rounded width into that comparison pushes it over by millimetres and fires a
    freeboard warning whose own quoted numbers satisfy it.
    """

    @pytest.mark.parametrize("head", [0.10, 0.15, 0.20, 0.30, 0.50])
    @pytest.mark.parametrize("rim", [148.30, 100.0, 12.7, 1013.45])
    @pytest.mark.parametrize("freeboard", [0.0, 0.15, 0.30])
    def test_auto_width_mode_is_silent_at_the_ceiling(self, rim, head, freeboard):
        """The auto path passes the design head itself, never an inverted one."""
        band = spillway_datum(rim, rim - 2.0, head_m=head, min_freeboard_m=freeboard)
        crest, _drop, _h = bind_crest(rim, crest=rim, band=band)
        problems = spillway_validity(
            crest, rim, invert_elevation=rim - 2.0, head_m=head,
            min_freeboard_m=freeboard, standard_freeboard_m=freeboard,
            typical_head_m=(0.05, 1.0),
        )
        assert not any("between the crest and the containing" in p for p in problems)

    @pytest.mark.parametrize("flow", [0.05, 0.12, 0.48, 1.5])
    @pytest.mark.parametrize("head", [0.15, 0.30, 0.50])
    def test_a_matched_built_width_does_not_manufacture_a_shortfall(self, flow, head):
        """Committed-width mode, with the width set to exactly what the design asks.

        Fed the raw inverse this fires: a rounded width inverts to a head a few
        millimetres above the design head, which pushes ``rim - crest`` past
        ``head + freeboard`` and accuses a crest seeded at the ceiling. Going through
        ``effective_head_m`` is what makes it correct, which is why production reads the
        head from there and never calls the inverse directly.
        """
        rim, freeboard = 100.0, 0.15
        width = calculate_spillway_width(flow, head)
        actual = effective_head_m(head, peak_flow_m3s=flow, width_m=width,
                                  width_auto=False)
        band = spillway_datum(rim, rim - 2.0, head_m=head, min_freeboard_m=freeboard)
        crest, _drop, _h = bind_crest(rim, crest=rim, band=band)

        problems = spillway_validity(
            crest, rim, invert_elevation=rim - 2.0, head_m=actual,
            min_freeboard_m=freeboard, standard_freeboard_m=freeboard,
            typical_head_m=(0.05, 1.0), width_m=width, required_width_m=width,
        )
        assert not any("between the crest and the containing" in p for p in problems), (
            f"design head {head}, recovered {actual:.5f}: {problems}"
        )


# ---------------------------------------------------------------------------
# The new validity branches
# ---------------------------------------------------------------------------

class TestFreeboardBelowStandard:
    def test_a_reduced_freeboard_is_named_as_reduced(self):
        problems = spillway_validity(
            99.6, 100.0, invert_elevation=98.0, head_m=0.15,
            min_freeboard_m=0.05, standard_freeboard_m=0.15)
        assert any("under the 0.15 m this type" in p for p in problems)

    def test_zero_freeboard_says_what_is_given_up(self):
        problems = spillway_validity(
            99.8, 100.0, invert_elevation=98.0, head_m=0.15,
            min_freeboard_m=0.0, standard_freeboard_m=0.15)
        assert any("Freeboard is zero" in p for p in problems)
        assert any("stops being the control" in p for p in problems)

    def test_meeting_the_standard_says_nothing(self):
        problems = spillway_validity(
            99.7, 100.0, invert_elevation=98.0, head_m=0.15,
            min_freeboard_m=0.15, standard_freeboard_m=0.15,
            typical_head_m=(0.10, 0.30))
        assert problems == []

    def test_exceeding_the_standard_is_not_flagged(self):
        problems = spillway_validity(
            99.0, 100.0, invert_elevation=98.0, head_m=0.15,
            min_freeboard_m=0.50, standard_freeboard_m=0.15,
            typical_head_m=(0.10, 0.30))
        assert problems == []

    def test_omitting_the_standard_raises_nothing(self):
        """The keyword is additive: existing callers must see no new objection."""
        problems = spillway_validity(
            99.6, 100.0, invert_elevation=98.0, head_m=0.15, min_freeboard_m=0.05)
        assert not any("this type is designed for" in p.lower() for p in problems)


class TestWeirFitsTheFeature:
    def test_a_weir_wider_than_the_feature_is_flagged(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.30,
            width_m=14.0, required_width_m=14.0, feature_length_m=30.0)
        assert problems == []

    def test_a_weir_longer_than_the_whole_feature_cannot_work(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.30,
            width_m=40.0, required_width_m=40.0, feature_length_m=30.0)
        assert any("cannot pass its own" in p for p in problems)
        assert any("40.0 m weir" in p for p in problems)

    def test_the_check_is_skipped_without_a_length(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.30,
            width_m=40.0, required_width_m=40.0)
        assert not any("cannot pass its own" in p for p in problems)

    def test_the_check_needs_a_real_required_width(self):
        """calculate_spillway_width returns 0.0 for invalid input; that must not read
        as a weir that trivially fits."""
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.30,
            required_width_m=0.0, feature_length_m=30.0)
        assert not any("cannot pass its own" in p for p in problems)


class TestPerTypeHeadBand:
    def test_a_swale_head_is_ordinary_in_its_own_band(self):
        problems = spillway_validity(
            99.7, 100.0, invert_elevation=98.0, head_m=0.15,
            min_freeboard_m=0.15, typical_head_m=(0.10, 0.30))
        assert not any("outside the usual" in p for p in problems)

    def test_the_same_head_is_low_for_an_embankment(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.15,
            typical_head_m=(0.20, 0.50))
        assert any("A low head needs a wide weir" in p for p in problems)

    def test_omitting_the_band_keeps_the_embankment_default(self):
        problems = spillway_validity(
            99.4, 100.0, invert_elevation=98.0, head_m=0.15)
        assert any("0.20–0.50 m range" in p for p in problems)
