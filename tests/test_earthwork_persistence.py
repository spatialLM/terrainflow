"""Round-trip tests for the earthwork design → QGIS project persistence.

Earthworks previously existed only in memory, so closing QGIS silently discarded an
entire design. The map-layer mirror carried 5 of ~16 fields, so even a saved project
could not reconstruct them.
"""

import pytest

from terrainflow_assessment.modules.earthwork_design import (
    Earthwork,
    EarthworkManager,
)


class _WktGeom:
    """Stand-in geometry: records its WKT and rebuilds from it."""

    def __init__(self, wkt):
        self._wkt = wkt

    def asWkt(self):
        return self._wkt


def _factory(wkt):
    return _WktGeom(wkt) if wkt else None


def _swale():
    ew = Earthwork("swale", _WktGeom("LINESTRING (0 0, 100 0)"), "Swale 1")
    ew.depth = 0.6
    ew.top_width_m = 2.5
    ew.bottom_width_m = 1.3
    ew.companion_berm = True
    ew.overflow_target_id = "target-abc"
    ew.source_contour_coords = [(0.0, 0.0), (50.0, 1.0)]
    ew.capacity_m3 = 91.5
    ew.capacity_l = 91500.0
    return ew


def _basin():
    ew = Earthwork("basin", _WktGeom("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), "Basin 2")
    ew.depth = 1.8
    ew.batter_run_m = 2.4
    ew.enabled = False
    return ew


def _dam():
    ew = Earthwork("dam", _WktGeom("LINESTRING (0 0, 40 0)"), "Dam 3")
    ew.crest_elevation = 137.25
    ew.key_into_banks = True
    return ew


def _diversion():
    ew = Earthwork("diversion", _WktGeom("LINESTRING (0 0, 60 20)"), "Diversion 4")
    ew.gradient_pct = 1.4
    return ew


class TestEarthworkRoundTrip:
    @pytest.mark.parametrize("factory", [_swale, _basin, _dam, _diversion])
    def test_every_field_survives(self, factory):
        original = factory()
        restored = Earthwork.from_dict(original.to_dict(), geometry_factory=_factory)

        for field in Earthwork._SERIAL_FIELDS:
            assert getattr(restored, field) == getattr(original, field), field
        assert restored.geometry.asWkt() == original.geometry.asWkt()

    def test_id_is_preserved_so_overflow_links_survive(self):
        original = _swale()
        restored = Earthwork.from_dict(original.to_dict(), geometry_factory=_factory)
        assert restored.id == original.id
        assert restored.overflow_target_id == "target-abc"

    def test_disabled_state_survives(self):
        restored = Earthwork.from_dict(_basin().to_dict(), geometry_factory=_factory)
        assert restored.enabled is False

    def test_missing_geometry_yields_nothing(self):
        data = _swale().to_dict()
        data["geometry_wkt"] = None
        assert Earthwork.from_dict(data, geometry_factory=_factory) is None

    def test_unrebuildable_geometry_yields_nothing(self):
        data = _swale().to_dict()
        assert Earthwork.from_dict(data, geometry_factory=lambda _w: None) is None

    def test_partial_data_falls_back_to_defaults(self):
        restored = Earthwork.from_dict(
            {"type": "swale", "name": "Bare", "geometry_wkt": "LINESTRING (0 0, 1 1)"},
            geometry_factory=_factory,
        )
        assert restored.name == "Bare"
        assert restored.depth > 0            # registry default, not None
        assert restored.enabled is True


class TestManagerRoundTrip:
    def _manager(self):
        m = EarthworkManager()
        for f in (_swale, _basin, _dam, _diversion):
            m.add(f())
        return m

    def test_whole_design_survives(self):
        original = self._manager()
        restored = EarthworkManager()
        n = restored.from_json(original.to_json(), geometry_factory=_factory)

        assert n == 4
        assert len(restored) == 4
        assert [e.name for e in restored.get_all()] == \
               [e.name for e in original.get_all()]
        assert [e.id for e in restored.get_all()] == \
               [e.id for e in original.get_all()]

    def test_order_is_preserved(self):
        original = self._manager()
        restored = EarthworkManager()
        restored.from_json(original.to_json(), geometry_factory=_factory)
        assert [e.type for e in restored.get_all()] == \
               ["swale", "basin", "dam", "diversion"]

    def test_loading_replaces_rather_than_appends(self):
        m = self._manager()
        m.from_json(m.to_json(), geometry_factory=_factory)
        assert len(m) == 4

    def test_one_bad_feature_does_not_lose_the_rest(self):
        """Losing an unreadable feature beats discarding the whole design."""
        import json
        original = self._manager()
        payload = json.loads(original.to_json())
        payload["earthworks"][1]["geometry_wkt"] = None
        restored = EarthworkManager()
        assert restored.from_json(json.dumps(payload), geometry_factory=_factory) == 3

    def test_empty_and_malformed_input_is_safe(self):
        m = self._manager()
        assert m.from_json("") == 0
        assert len(m) == 0
        assert EarthworkManager().from_json("not json at all") == 0
        assert EarthworkManager().from_json(None) == 0
        assert EarthworkManager().from_json("{}") == 0

    def test_round_trip_is_stable_across_two_saves(self):
        original = self._manager()
        once = EarthworkManager()
        once.from_json(original.to_json(), geometry_factory=_factory)
        twice = EarthworkManager()
        twice.from_json(once.to_json(), geometry_factory=_factory)
        assert once.to_json() == twice.to_json()


class TestAPayloadWithNoBottomWidthReloadsAtTheBatterItWasDrawnAt:
    """STRETCH_GOALS §12. `from_dict` restored `bottom_width_m` independently of
    `depth` and `top_width_m` and nothing re-derived it, so a payload written
    before the key existed kept the bottom the *constructor* seeded from the
    shipped defaults while the other two came off the file. The three then
    described a section nobody drew, and `side_slope` — a derived property —
    reported the wrong batter to capacity, the burn footprint and
    `channel_batter_run` from then on.

    The payload is the shape §12 documents rather than a real `.tfd`: none is
    committed, and what decides this is which keys are present, not which build
    wrote them.
    """

    # §12's own worked example: drawn 1.2 m deep and 3.0 m across at 1:1, from
    # before `bottom_width_m` was serialised at all.
    V1 = {
        "type": "swale",
        "name": "Swale 1",
        "geometry_wkt": "LINESTRING (0 0, 100 0)",
        "depth": 1.2,
        "top_width_m": 3.0,
    }

    def test_the_batter_is_the_types_shipped_one_not_the_seeded_bottom(self):
        restored = Earthwork.from_dict(dict(self.V1), geometry_factory=_factory)
        # 3.0 - 2 * 1.0 * 1.2 = 0.6, so 1:1 — what it was drawn at.
        assert restored.bottom_width_m == pytest.approx(0.6)
        assert restored.side_slope == pytest.approx(1.0), (
            f"reloaded at {restored.side_slope:.2f}:1; the seeded 1.0 m bottom "
            f"against a 1.2 m depth is 0.83:1, a section nobody drew")

    def test_a_null_bottom_width_counts_as_absent(self):
        """The restore loop skips a `None`, so the seeded value survived it in
        exactly the same way — and it means the same thing: nobody chose one."""
        payload = dict(self.V1, bottom_width_m=None)
        restored = Earthwork.from_dict(payload, geometry_factory=_factory)
        assert restored.bottom_width_m == pytest.approx(0.6)

    def test_a_stored_bottom_width_is_never_recomputed(self):
        """The promise the standard-dimensions work is built on: a preference can
        never retroactively resize a stored design. A present value is a decision,
        and recomputing it would be the silent rewrite `advisories.py` forbids —
        even where it disagrees with the type's shipped batter, as this one does.
        """
        payload = dict(self.V1, bottom_width_m=2.2)      # 0.33:1, deliberately odd
        restored = Earthwork.from_dict(payload, geometry_factory=_factory)
        assert restored.bottom_width_m == pytest.approx(2.2)

    def test_a_payload_carrying_neither_dimension_is_left_alone(self):
        """Nothing came off the file to disagree with the seed, so there is
        nothing to re-derive against."""
        payload = {"type": "swale", "name": "Bare",
                   "geometry_wkt": "LINESTRING (0 0, 1 1)"}
        restored = Earthwork.from_dict(payload, geometry_factory=_factory)
        fresh = Earthwork("swale", _WktGeom("LINESTRING (0 0, 1 1)"), "Bare")
        assert restored.bottom_width_m == pytest.approx(fresh.bottom_width_m)

    def test_a_round_trip_of_a_current_design_is_unchanged(self):
        """The guard that says this only touches old payloads: anything this
        build wrote carries all three, so nothing here fires."""
        original = _swale()
        restored = Earthwork.from_dict(original.to_dict(), geometry_factory=_factory)
        assert restored.bottom_width_m == pytest.approx(original.bottom_width_m)
        assert restored.side_slope == pytest.approx(original.side_slope)
