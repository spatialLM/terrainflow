"""
test_spillway_link.py — the drain that starts where another feature spills.

Stage C of ``CLudeDocs/SPILLWAY_NOTCH_PLAN.md``. The burn change itself is one line;
the risk is entirely in the link's lifecycle, so that is what this file is about:
parsing, resolution when the far end of the link has moved or gone, the cycle refusal,
and the burn order the link imposes.

The three cases worth stating up front, because each is a fault that looks like nothing
from the outside:

* **A dangling link is not an error.** The drain falls back to the ground sample it used
  before it was linked, and something has to say so — otherwise a design quietly stops
  meaning what it said.
* **Grading from the wrong end** produces a drain running uphill from an entirely
  plausible-looking level. There is no exception and no odd number to notice.
* **A cycle** is refused rather than assumed impossible: nothing on the model stops a
  diversion carrying a spillway of its own.
"""

from terrainflow_assessment.modules.earthwork_design import (
    Earthwork,
    Spillway,
    burn_order,
    format_spillway_link,
    parse_spillway_link,
    resolve_spillway_links,
    spillway_link_cycle,
)


class _Geom:
    """The narrowest geometry the model needs to build an Earthwork."""

    def __init__(self, wkt="LINESTRING (0 0, 10 0)"):
        self._wkt = wkt

    def asWkt(self):
        return self._wkt


def _feature(ew_type, name, ew_id=None, crest=None, enabled=True, inlet_crest=None):
    ew = Earthwork(ew_type, _Geom(), name)
    if ew_id is not None:
        ew.id = ew_id
    ew.enabled = enabled
    if crest is not None:
        ew.spillway = Spillway(crest_elevation=crest, point_wkt="POINT (5 0)")
    if inlet_crest is not None:
        ew.inflow_spillway = Spillway(crest_elevation=inlet_crest,
                                      point_wkt="POINT (5 0)")
    return ew


def _drain(name, ew_id, link=None):
    ew = _feature("diversion", name, ew_id=ew_id)
    ew.spillway_link_id = link
    return ew


class TestTheStoredForm:
    def test_a_link_round_trips_through_its_string(self):
        text = format_spillway_link("abc123", "outflow", "end")
        assert text == "abc123:outflow:end"
        assert parse_spillway_link(text) == ("abc123", "outflow", "end")

    def test_the_defaults_are_the_common_case(self):
        assert format_spillway_link("abc") == "abc:outflow:start"

    def test_nothing_to_link_to_stores_nothing(self):
        assert format_spillway_link(None) is None
        assert format_spillway_link("") is None

    def test_a_nonsense_kind_or_end_is_normalised_rather_than_stored(self):
        """Never write a token the parser would then refuse to read back."""
        assert format_spillway_link("abc", "sideways", "middle") == "abc:outflow:start"

    def test_a_two_token_link_reads_as_the_first_vertex(self):
        """The end ``_burn_diversion`` has always graded from.

        A hand-edited file is the realistic source of one of these, and degrading to
        today's behaviour is the only fallback that cannot silently reverse a drain.
        """
        assert parse_spillway_link("abc:outflow") == ("abc", "outflow", "start")
        assert parse_spillway_link("abc") == ("abc", "outflow", "start")

    def test_an_unrecognised_end_falls_back_the_same_way(self):
        assert parse_spillway_link("abc:outflow:middle") == ("abc", "outflow", "start")

    def test_an_unrecognised_kind_is_not_a_link_at_all(self):
        """Unlike the end, there is no safe default here — a kind names a structure."""
        assert parse_spillway_link("abc:sideways:start") is None

    def test_nothing_at_all_parses_to_nothing(self):
        for value in (None, "", "   ", 12, [], ":outflow:start"):
            assert parse_spillway_link(value) is None


class TestResolution:
    def test_a_live_link_resolves_to_the_crest(self):
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.52)
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        inverts, dangling = resolve_spillway_links([source, drain])
        assert inverts == {"d1": 55.52}
        assert dangling == []

    def test_an_unlinked_drain_is_neither_resolved_nor_reported(self):
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.52)
        inverts, dangling = resolve_spillway_links([source, _drain("Drain 1", "d1")])
        assert inverts == {}
        assert dangling == []

    def test_a_deleted_source_dangles_and_names_the_drain(self):
        drain = _drain("Drain 1", "d1", "gone:outflow:start")
        inverts, dangling = resolve_spillway_links([drain])
        assert inverts == {}
        assert [n for n, _why in dangling] == ["Drain 1"]
        assert "no longer in the design" in dangling[0][1]

    def test_a_switched_off_source_dangles_with_its_own_reason(self):
        """Four rejections, four different things for the user to do."""
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.52, enabled=False)
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        _inverts, dangling = resolve_spillway_links([source, drain])
        assert "switched off" in dangling[0][1]

    def test_a_source_whose_spillway_was_cleared_dangles(self):
        source = _feature("dam", "Dam 15", ew_id="s1")
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        _inverts, dangling = resolve_spillway_links([source, drain])
        assert "no longer has a spillway crest" in dangling[0][1]

    def test_a_link_to_an_inlet_is_refused_rather_than_graded_backwards(self):
        """An inlet is where water arrives, so the start level would be at the far end."""
        source = _feature("basin", "Basin 3", ew_id="s1", inlet_crest=60.0)
        drain = _drain("Drain 1", "d1", "s1:inflow:start")
        inverts, dangling = resolve_spillway_links([source, drain])
        assert inverts == {}
        assert "inlet" in dangling[0][1]

    def test_a_self_link_resolves_to_nothing(self):
        drain = _drain("Drain 1", "d1", "d1:outflow:start")
        inverts, dangling = resolve_spillway_links([drain])
        assert inverts == {}
        assert "linked to itself" in dangling[0][1]

    def test_one_spillway_can_feed_several_drains(self):
        source = _feature("swale", "Swale 4", ew_id="s1", crest=69.19)
        drains = [_drain("Drain 1", "d1", "s1:outflow:start"),
                  _drain("Drain 2", "d2", "s1:outflow:end")]
        inverts, dangling = resolve_spillway_links([source, *drains])
        assert inverts == {"d1": 69.19, "d2": 69.19}
        assert dangling == []

    def test_a_crest_that_is_not_a_number_dangles_rather_than_raising(self):
        """A design file can carry anything; opening one must not depend on it not."""
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.52)
        source.spillway.crest_elevation = "fifty-five and a bit"
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        inverts, dangling = resolve_spillway_links([source, drain])
        assert inverts == {}
        assert "not a level" in dangling[0][1]

    def test_the_datum_follows_the_crest_rather_than_being_frozen(self):
        """Which is the whole reason it is derived and never serialised."""
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.52)
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        assert resolve_spillway_links([source, drain])[0]["d1"] == 55.52
        source.spillway.crest_elevation = 54.90
        assert resolve_spillway_links([source, drain])[0]["d1"] == 54.90


class TestCycleRefusal:
    def test_an_ordinary_link_is_not_a_loop(self):
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.5)
        drain = _drain("Drain 1", "d1")
        assert spillway_link_cycle([source, drain], extra=("d1", "s1")) == []

    def test_a_proposed_link_that_closes_a_loop_is_named_before_it_is_made(self):
        """Nothing on the model stops a diversion from carrying a spillway of its own."""
        a = _drain("Drain A", "a", "b:outflow:start")
        b = _drain("Drain B", "b")
        broken = spillway_link_cycle([a, b], extra=("b", "a"))
        assert set(broken) == {"a", "b"}

    def test_an_existing_loop_is_reported_without_a_proposal(self):
        a = _drain("Drain A", "a", "b:outflow:start")
        b = _drain("Drain B", "b", "a:outflow:start")
        assert set(spillway_link_cycle([a, b])) == {"a", "b"}

    def test_a_three_drain_ring_is_caught(self):
        a = _drain("A", "a", "b:outflow:start")
        b = _drain("B", "b", "c:outflow:start")
        c = _drain("C", "c")
        assert set(spillway_link_cycle([a, b, c], extra=("c", "a"))) == {"a", "b", "c"}

    def test_no_links_is_not_a_loop(self):
        assert spillway_link_cycle([_drain("Drain 1", "d1")]) == []
        assert spillway_link_cycle([]) == []

    def test_a_feature_with_no_id_is_skipped_rather_than_keyed_on_none(self):
        """One `None` key would collapse every such feature onto one graph node."""
        nameless = _drain("Nameless", "x", "y:outflow:start")
        nameless.id = None
        assert spillway_link_cycle([nameless]) == []


class TestBurnOrder:
    def test_a_design_with_no_links_keeps_the_order_it_was_given(self):
        """The property that keeps this from moving any existing number."""
        items = [_feature("swale", f"S{i}", ew_id=str(i)) for i in range(5)]
        assert burn_order(items) == items

    def test_a_source_that_is_already_first_is_not_moved(self):
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.5)
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        others = [_feature("swale", "Swale 4", ew_id="w1")]
        given = [source, *others, drain]
        assert burn_order(given) == given

    def test_a_source_drawn_after_its_drain_is_moved_ahead_of_it(self):
        drain = _drain("Drain 1", "d1", "s1:outflow:start")
        source = _feature("dam", "Dam 15", ew_id="s1", crest=55.5)
        order = burn_order([drain, source])
        assert [ew.name for ew in order] == ["Dam 15", "Drain 1"]

    def test_only_what_has_to_move_moves(self):
        """Stable and minimal, and it is the **drain** that gives way.

        Given A, Drain, B, Source, C, there are two orders that satisfy the constraint:
        pull Source up to just before Drain, which displaces both Source and B, or push
        Drain down to just after Source, which displaces only Drain. Kahn's with a
        smallest-index-first tie-break takes the second, so an unconstrained feature
        never moves to make room for one that is.
        """
        a = _feature("swale", "A", ew_id="a")
        drain = _drain("Drain", "d", "s:outflow:start")
        b = _feature("swale", "B", ew_id="b")
        source = _feature("dam", "Source", ew_id="s", crest=50.0)
        c = _feature("swale", "C", ew_id="c")
        assert [ew.name for ew in burn_order([a, drain, b, source, c])] == [
            "A", "B", "Source", "Drain", "C"]

    def test_a_chain_of_links_orders_end_to_end(self):
        third = _drain("Third", "c", "b:outflow:start")
        second = _drain("Second", "b", "a:outflow:start")
        second.spillway = Spillway(crest_elevation=40.0, point_wkt="POINT (1 1)")
        first = _feature("dam", "First", ew_id="a", crest=50.0)
        assert [ew.name for ew in burn_order([third, second, first])] == [
            "First", "Second", "Third"]

    def test_a_dangling_link_imposes_no_order(self):
        drain = _drain("Drain 1", "d1", "gone:outflow:start")
        other = _feature("swale", "Swale 4", ew_id="w1")
        given = [drain, other]
        assert burn_order(given) == given

    def test_a_cycle_is_left_in_the_order_it_was_given_rather_than_raising(self):
        """Refused when a link is made, but a hand-edited file can still carry one."""
        a = _drain("A", "a", "b:outflow:start")
        b = _drain("B", "b", "a:outflow:start")
        assert [ew.name for ew in burn_order([a, b])] == ["A", "B"]

    def test_a_cycle_does_not_lose_the_features_around_it(self):
        a = _drain("A", "a", "b:outflow:start")
        b = _drain("B", "b", "a:outflow:start")
        c = _feature("swale", "C", ew_id="c")
        assert sorted(ew.name for ew in burn_order([a, b, c])) == ["A", "B", "C"]

    def test_a_self_link_imposes_no_order(self):
        drain = _drain("Drain 1", "d1", "d1:outflow:start")
        assert burn_order([drain]) == [drain]

    def test_nothing_to_order(self):
        assert burn_order([]) == []
        assert burn_order(None) == []
