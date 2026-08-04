"""Spillway model, crest binding and validity.

A spillway decides *where* a feature overflows, and therefore which feature spills
into which — it sets the fill order of the whole system. The arithmetic behind the
crest is small, but it is bound two ways through the dialog and clamped against
what the ground can offer, and that combination is exactly where hand-written
bindings drift apart. These tests pin the round trip.
"""

import pytest

from terrainflow_assessment.modules.earthwork_design import (
    SPILLWAY_MIN_FREEBOARD_M,
    Earthwork,
    EarthworkManager,
    Spillway,
    bind_crest,
    calculate_spillway_width,
    spillway_datum,
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
        crest, drop = bind_crest(100.0, crest=99.4)
        assert (crest, drop) == pytest.approx((99.4, 0.6))

    def test_drop_gives_crest(self):
        crest, drop = bind_crest(100.0, drop=0.6)
        assert (crest, drop) == pytest.approx((99.4, 0.6))

    @pytest.mark.parametrize("value", [0.05, 0.3, 0.62, 1.4, 3.0])
    def test_round_trip_is_exact(self, value):
        """Crest → drop → crest must not drift; the dialog cycles this on every
        keystroke, so any loss compounds."""
        crest, drop = bind_crest(100.0, drop=value)
        back_crest, back_drop = bind_crest(100.0, crest=crest)
        assert back_crest == pytest.approx(crest)
        assert back_drop == pytest.approx(value)

    def test_absolute_crest_wins_when_both_are_given(self):
        crest, drop = bind_crest(100.0, crest=99.0, drop=5.0)
        assert (crest, drop) == pytest.approx((99.0, 1.0))

    def test_clamps_into_the_band_and_the_partner_follows(self):
        """The clamp must happen before the partner is derived, or the two
        controls end up describing different crests."""
        # ceiling = rim − head − freeboard = 100.0 − 0.30 − 0.30 = 99.40
        band = spillway_datum(100.0, 98.0, head_m=0.30)
        crest, drop = bind_crest(100.0, crest=99.9, band=band)
        assert crest == pytest.approx(99.40)
        assert drop == pytest.approx(0.60)
        assert crest + drop == pytest.approx(100.0)

    def test_clamps_up_to_the_floor(self):
        band = spillway_datum(100.0, 98.0, head_m=0.30)
        crest, _drop = bind_crest(100.0, crest=90.0, band=band)
        assert crest == pytest.approx(98.0)

    def test_an_inverted_band_does_not_clamp(self):
        """No value satisfies an inverted band; clamping to either end would
        invent one and hide the real problem."""
        band = spillway_datum(100.0, 99.8, head_m=0.50)
        crest, _drop = bind_crest(100.0, crest=99.9, band=band)
        assert crest == pytest.approx(99.9)

    def test_no_rim_passes_values_through_untouched(self):
        assert bind_crest(None, crest=99.0, drop=None) == (99.0, None)

    def test_nothing_given_returns_nothing(self):
        assert bind_crest(100.0) == (None, None)


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
        assert "0.20 m between the crest and the rim" in problems[0]
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
        crest, _drop = bind_crest(rim, crest=rim, band=band)   # clamps to the ceiling
        problems = spillway_validity(
            crest, rim, invert_elevation=rim - 2.0, head_m=head)
        assert not any("between the crest and the rim" in p for p in problems)

    def test_a_crest_genuinely_short_of_freeboard_is_still_caught(self):
        """The tolerance is a millimetre, not a licence — 10 cm short still fails."""
        problems = spillway_validity(99.65, 100.0, invert_elevation=98.0, head_m=0.30)
        assert any("between the crest and the rim" in p for p in problems)

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
