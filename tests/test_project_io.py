"""Round-trip and degradation tests for the portable design document.

A design file is the only thing standing between a user and losing a session, so the
bias throughout is that a partial or unfamiliar file must still open. These tests pin
that: unknown keys, missing keys, wrong types, a newer schema version and a bogus enum
all resolve to something usable rather than an exception.
"""

import json

import pytest

from terrainflow_assessment.modules.earthwork_design import (
    Earthwork,
    EarthworkManager,
)
from terrainflow_assessment.modules.project_io import (
    AREA_KEYS,
    DEM_MODES,
    INPUT_FIELDS,
    SCHEMA_VERSION,
    DemReference,
    DesignDocument,
    default_inputs,
    normalise_inputs,
)


class _WktGeom:
    """Stand-in geometry, matching tests/test_earthwork_persistence.py."""

    def __init__(self, wkt):
        self._wkt = wkt

    def asWkt(self):
        return self._wkt


def _factory(wkt):
    return _WktGeom(wkt) if wkt else None


def _filled_inputs():
    """Every input set to something distinguishable from its default."""
    return {
        "site_name": "BX24",
        "rainfall_mm": 145.0,
        "duration_hr": 6.0,
        "soil_name": "Silt loam",
        "ground_condition": "poor",
        "cn": 78,
        "moisture": "Wet",
        # "d8" because the default is now "dinf": this fixture's whole job is to
        # differ from every default, and ROUTING_VALUES holds only the two.
        "routing": "d8",
        "stream_threshold_ha": 2.5,
        "exit_flow_ls": 12.0,
        "sizing_basis": "runoff",
        "runoff_coefficient": 0.35,
        "earthwork_soil_name": "Clay",
        "peak_intensity_mm_hr": 44.0,
        "count_infiltration": True,
        "contour_interval_m": 0.5,
        "simple_contour_interval_m": 10.0,
        "max_slope_deg": 22.0,
        "min_contour_length_m": 80.0,
        "min_catchment_ha": 1.25,
        "keypoint_count": 7,
        "keyline_runs": 5,
        "keyline_spacing_m": 8.0,
        "keyline_max_grade_n": 250,
        "keyline_max_valleys": 12,
        "swale_depth_m": 0.9,
        "swale_width_m": 3.0,
        "swale_bottom_width_m": 0.7,
    }


def _dem():
    return DemReference(
        mode="embedded",
        filename="dem.tif",
        original_path="C:/sites/bx24.tif",
        fingerprint="abc123",
        cell_size_m=1.0,
        crs="EPSG:2193",
        extent=[1000.0, 5000.0, 1500.0, 5500.0],
        width=500,
        height=500,
        is_clip=True,
        clip_buffer_m=25.0,
    )


def _manager():
    m = EarthworkManager()
    swale = Earthwork("swale", _WktGeom("LINESTRING (0 0, 100 0)"), "Swale 1")
    swale.depth = 0.6
    swale.capacity_m3 = 91.5
    m.add(swale)
    basin = Earthwork("basin", _WktGeom("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), "Basin 2")
    basin.depth = 1.8
    m.add(basin)
    return m


class TestInputCoercion:
    def test_defaults_cover_every_declared_field(self):
        assert set(default_inputs()) == set(INPUT_FIELDS)

    def test_every_field_survives_a_round_trip(self):
        original = _filled_inputs()
        restored = DesignDocument.from_json(
            DesignDocument.build(original).to_json()).inputs
        for name in INPUT_FIELDS:
            assert restored[name] == original[name], name

    def test_filled_inputs_differ_from_defaults(self):
        """Guards the round-trip test above from passing on defaults alone."""
        defaults = default_inputs()
        assert all(_filled_inputs()[n] != defaults[n] for n in INPUT_FIELDS)

    def test_the_swale_section_defaults_match_the_panel_spin_boxes(self):
        """A default here is not a suggestion — it is what an older file gets loaded as.

        ``normalise_inputs`` fills every absent key, so a schema default that disagrees
        with the widget it restores silently rewrites the user's criteria on open. This
        pair disagreed: 0.6/2.0 here against 0.3/0.6 in the criteria box, so a file
        saved before the keys existed came back with a different cross-section from the
        one the session was using, and the next run answered a different question.

        The panel cannot be imported here — ``tests/`` stubs QGIS out — so the values
        are restated rather than read. That is the point: if either side moves, this
        fails and someone has to look at both.
        """
        assert INPUT_FIELDS["swale_width_m"][1] == 2.0
        assert INPUT_FIELDS["swale_depth_m"][1] == 0.5
        assert INPUT_FIELDS["swale_bottom_width_m"][1] == 1.0

    def test_the_storm_and_routing_defaults_match_the_panel(self):
        """The same rule as the swale trio, on the two widest-reaching inputs.

        Both disagreed: 120.0 mm here against a spin box that has read 65 since the
        initial commit, and "d8" against a combo that opens on "D-infinity
        (recommended)". A file saved before either key existed therefore reopened on a
        storm that nearly doubles every volume in the report, and on a flow algorithm
        whose pysheds path calls ``np.in1d`` — removed in NumPy 2, so it can raise
        rather than merely answer differently.

        The QGIS parity check in ``tests_qgis/checks_design_file`` walks the whole
        table against the live widgets, but it needs a QGIS runtime and it is also the
        check that carried these two on an allowlist. Restating the two values here
        costs nothing and holds them in the suite that always runs.
        """
        assert INPUT_FIELDS["rainfall_mm"][1] == 65.0
        assert INPUT_FIELDS["routing"][1] == "dinf"

    def test_the_batter_is_derived_and_therefore_not_stored(self):
        """Three dimensions are entered; the side slope is what they come out as.

        Saving a derived value beside the three it derives from is how a reloaded file
        comes back describing a section that never existed — the batter says one thing,
        the widths say another, and whichever the sizing reads wins silently.
        """
        assert "swale_side_slope" not in INPUT_FIELDS

    def test_the_default_swale_has_a_floor(self):
        """A swale is dug with a flat bottom. The defaults have to describe one.

        They did not: 0.6 m top over 0.3 m deep at a 1:1 batter puts the walls together
        exactly at the drawn depth, so the floor was 0.00 m and the section a V-drain of
        0.09 m². Not merely inelegant — at 0.072 m³ per metre the segment sizing asked
        for 33.6 km of swale on a 7.4 ha catchment, so every recommendation came back
        capped as "contour too short" and the ranking carried no information at all.

        Asserted on the schema rather than on the widgets because this is the copy that
        survives a save/reload, and a floor that only exists until the file is reopened
        is not a floor.
        """
        top = INPUT_FIELDS["swale_width_m"][1]
        depth = INPUT_FIELDS["swale_depth_m"][1]
        bottom = INPUT_FIELDS["swale_bottom_width_m"][1]

        assert 0 < bottom <= top, (
            f"a {bottom} m floor under a {top} m top is not a section")
        # Wide enough to be worth calling a floor, and to survive a DEM cell. A 5 cm
        # bottom is a V with a rounding error on it.
        assert bottom >= 0.3, f"a {bottom:.2f} m floor is not a buildable trench"

        z = (top - bottom) / (2.0 * depth)
        assert z > 0, "vertical walls are not a swale batter"

    def test_the_default_section_is_the_registrys_swale(self):
        """The criteria box and a swale you draw must describe the same swale.

        They did not, and the gap was three-quarters of a square metre of section. That
        divergence is also what made wiring the panel to seed drawn features look
        sensible — it is the right instinct against the wrong pair of numbers.
        """
        from terrainflow_assessment.core.registry.earthwork_types import get_type
        from terrainflow_assessment.core.sizing import trapezoid_section

        cfg = get_type("swale")
        assert INPUT_FIELDS["swale_width_m"][1] == cfg.default_top_width
        assert INPUT_FIELDS["swale_depth_m"][1] == cfg.default_depth

        sec = trapezoid_section(INPUT_FIELDS["swale_width_m"][1],
                                INPUT_FIELDS["swale_bottom_width_m"][1],
                                INPUT_FIELDS["swale_depth_m"][1])
        assert sec.area == pytest.approx(0.75)
        assert sec.side_slope == pytest.approx(cfg.default_side_slope)

    def test_missing_keys_fall_back_to_defaults(self):
        result = normalise_inputs({"site_name": "Partial"})
        assert result["site_name"] == "Partial"
        assert result["rainfall_mm"] == INPUT_FIELDS["rainfall_mm"][1]
        assert set(result) == set(INPUT_FIELDS)

    def test_unknown_keys_are_dropped(self):
        result = normalise_inputs({"site_name": "X", "invented_field": 9})
        assert "invented_field" not in result

    def test_none_and_empty_input_yields_defaults(self):
        assert normalise_inputs(None) == default_inputs()
        assert normalise_inputs({}) == default_inputs()

    @pytest.mark.parametrize("raw,expected", [("145.5", 145.5), (145, 145.0)])
    def test_numeric_strings_coerce(self, raw, expected):
        assert normalise_inputs({"rainfall_mm": raw})["rainfall_mm"] == expected

    def test_float_to_int_field_rounds(self):
        assert normalise_inputs({"cn": 78.6})["cn"] == 79

    def test_unconvertible_value_falls_back(self):
        assert normalise_inputs({"rainfall_mm": "not a number"})["rainfall_mm"] == \
            INPUT_FIELDS["rainfall_mm"][1]
        assert normalise_inputs({"cn": [1, 2]})["cn"] == INPUT_FIELDS["cn"][1]

    def test_explicit_none_falls_back(self):
        assert normalise_inputs({"duration_hr": None})["duration_hr"] == \
            INPUT_FIELDS["duration_hr"][1]

    def test_non_string_coerces_to_string_field(self):
        assert normalise_inputs({"site_name": 42})["site_name"] == "42"

    @pytest.mark.parametrize("field,good,bad", [
        ("routing", "dinf", "quadtree"),
        ("sizing_basis", "rainfall", "vibes"),
        ("ground_condition", "poor", "lush"),
    ])
    def test_enum_fields_reject_unknown_values(self, field, good, bad):
        """A bogus enum must not reach an analysis run — it would fail far less visibly."""
        assert normalise_inputs({field: good})[field] == good
        assert normalise_inputs({field: bad})[field] == INPUT_FIELDS[field][1]

    def test_a_design_predating_ground_condition_loads_as_good(self):
        """Good reproduces the curve numbers those designs were sized against, so an
        older file must not silently reopen against a wetter storm than it was built
        for — nor a drier one."""
        older = _filled_inputs()
        del older["ground_condition"]
        assert normalise_inputs(older)["ground_condition"] == "good"

    @pytest.mark.parametrize("raw,expected", [
        (True, True), (False, False), (1, True), (0, False),
        ("true", True), ("True", True), ("yes", True), ("1", True),
        ("false", False), ("False", False), ("no", False), ("0", False), ("", False),
    ])
    def test_boolean_field_reads_json_and_hand_edited_forms(self, raw, expected):
        """`bool("false")` is True, so string forms need explicit handling."""
        assert normalise_inputs({"count_infiltration": raw})["count_infiltration"] is expected

    def test_unrecognised_boolean_string_falls_back(self):
        assert normalise_inputs({"count_infiltration": "maybe"})["count_infiltration"] is False

    def test_boolean_survives_a_json_round_trip(self):
        for value in (True, False):
            doc = DesignDocument.build({"count_infiltration": value})
            assert DesignDocument.from_json(
                doc.to_json()).inputs["count_infiltration"] is value


class TestDemReference:
    def test_round_trip_preserves_every_field(self):
        original = _dem()
        restored = DemReference.from_dict(original.to_dict())
        assert restored == original

    @pytest.mark.parametrize("mode", DEM_MODES)
    def test_both_modes_parse(self, mode):
        assert DemReference.from_dict({"mode": mode}).mode == mode

    def test_unknown_mode_reads_as_reference(self):
        """Reference mode forces resolve-and-verify rather than trusting a missing member."""
        assert DemReference.from_dict({"mode": "magic"}).mode == "reference"
        assert DemReference.from_dict({}).mode == "reference"
        assert DemReference.from_dict(None).mode == "reference"

    def test_malformed_extent_is_discarded(self):
        assert DemReference.from_dict({"extent": [1, 2]}).extent is None
        assert DemReference.from_dict({"extent": ["a", "b", "c", "d"]}).extent is None
        assert DemReference.from_dict({"extent": []}).extent is None

    def test_valid_extent_coerces_to_floats(self):
        assert DemReference.from_dict({"extent": [1, 2, 3, 4]}).extent == \
            [1.0, 2.0, 3.0, 4.0]

    def test_malformed_numbers_become_none(self):
        ref = DemReference.from_dict({"cell_size_m": "wide", "width": "lots"})
        assert ref.cell_size_m is None
        assert ref.width is None

    def test_missing_clip_buffer_defaults_to_zero(self):
        assert DemReference.from_dict({"clip_buffer_m": None}).clip_buffer_m == 0.0
        assert DemReference.from_dict({"clip_buffer_m": "x"}).clip_buffer_m == 0.0

    def test_matches_on_fingerprint(self):
        a = DemReference(fingerprint="same")
        assert a.matches(DemReference(fingerprint="same"))
        assert not a.matches(DemReference(fingerprint="different"))

    def test_no_fingerprint_never_matches(self):
        """Grid properties alone are not identity — two rasters over one site share them."""
        assert not DemReference(cell_size_m=1.0).matches(DemReference(cell_size_m=1.0))
        assert not DemReference(fingerprint="x").matches(DemReference())

    def test_matches_rejects_foreign_types(self):
        assert not DemReference(fingerprint="x").matches("x")
        assert not DemReference(fingerprint="x").matches(None)


class TestAreas:
    def test_recognised_areas_survive(self):
        areas = {k: {"wkt": f"POINT (0 {i})", "crs": "EPSG:2193"}
                 for i, k in enumerate(AREA_KEYS)}
        restored = DesignDocument.from_json(
            DesignDocument.build({}, areas=areas).to_json()).areas
        assert restored == areas

    def test_unknown_area_keys_are_dropped(self):
        doc = DesignDocument.build({}, areas={"paddock": {"wkt": "POINT (0 0)"}})
        assert doc.areas == {}

    def test_area_without_geometry_is_dropped(self):
        doc = DesignDocument.build({}, areas={
            "boundary": {"wkt": "", "crs": "EPSG:2193"},
            "analysis": {"crs": "EPSG:2193"},
            "earthworks": {"wkt": "POINT (0 0)"},
        })
        assert set(doc.areas) == {"earthworks"}

    def test_crs_may_be_absent(self):
        doc = DesignDocument.build({}, areas={"boundary": {"wkt": "POINT (0 0)"}})
        assert doc.areas["boundary"]["crs"] is None

    def test_malformed_areas_container_is_ignored(self):
        assert DesignDocument.build({}, areas="nope").areas == {}
        assert DesignDocument.build({}, areas={"boundary": "nope"}).areas == {}
        assert DesignDocument.build({}, areas=None).areas == {}


class TestEarthworkDelegation:
    def test_design_carries_the_manager_payload(self):
        original = _manager()
        doc = DesignDocument.build({}, earthworks_json=original.to_json())

        restored = EarthworkManager()
        n = restored.from_json(doc.earthworks_json(), geometry_factory=_factory)

        assert n == 2
        assert [e.name for e in restored.get_all()] == \
               [e.name for e in original.get_all()]
        assert [e.id for e in restored.get_all()] == \
               [e.id for e in original.get_all()]

    def test_payload_survives_the_document_round_trip(self):
        doc = DesignDocument.build({}, earthworks_json=_manager().to_json())
        reopened = DesignDocument.from_json(doc.to_json())

        restored = EarthworkManager()
        assert restored.from_json(
            reopened.earthworks_json(), geometry_factory=_factory) == 2

    def test_earthworks_stored_unwrapped_not_as_nested_json_string(self):
        """A JSON string inside JSON would double-escape and defeat fixture diffs."""
        payload = json.loads(DesignDocument.build(
            {}, earthworks_json=_manager().to_json()).to_json())
        assert isinstance(payload["earthworks"], dict)
        assert isinstance(payload["earthworks"]["earthworks"], list)

    def test_accepts_an_already_parsed_payload(self):
        parsed = json.loads(_manager().to_json())
        assert DesignDocument.build({}, earthworks_json=parsed).earthwork_count() == 2

    def test_count_and_payload_for_a_design_with_no_earthworks(self):
        doc = DesignDocument.build({})
        assert doc.earthwork_count() == 0
        assert EarthworkManager().from_json(
            doc.earthworks_json(), geometry_factory=_factory) == 0

    def test_a_spillway_link_survives_the_document_round_trip(self):
        """The link is a *decision*, so unlike the rest of the derived family it is
        stored — and it is what took ``SCHEMA_VERSION`` to 3. A height above floor can
        be re-derived from the crest; which spillway a drain was linked to is
        recoverable from nothing.
        """
        manager = _manager()
        drain = Earthwork("diversion", _WktGeom("LINESTRING (0 0, 50 0)"), "Drain 3")
        source_id = manager.get_all()[0].id
        drain.spillway_link_id = f"{source_id}:outflow:end"
        manager.add(drain)

        reopened = DesignDocument.from_json(
            DesignDocument.build({}, earthworks_json=manager.to_json()).to_json())
        restored = EarthworkManager()
        restored.from_json(reopened.earthworks_json(), geometry_factory=_factory)

        back = [e for e in restored.get_all() if e.name == "Drain 3"][0]
        assert back.spillway_link_id == f"{source_id}:outflow:end"

    def test_the_level_a_link_resolves_to_is_not_stored(self):
        """``invert_start_m`` is another feature's crest. A level cached in a project
        file outlives the design that produced it, so it is re-derived on restore — the
        same rule ``terrain_capacity_m3`` follows.
        """
        manager = _manager()
        drain = Earthwork("diversion", _WktGeom("LINESTRING (0 0, 50 0)"), "Drain 3")
        drain.spillway_link_id = f"{manager.get_all()[0].id}:outflow:start"
        drain.invert_start_m = 55.52
        manager.add(drain)

        payload = json.loads(manager.to_json())
        stored = [e for e in payload["earthworks"] if e["name"] == "Drain 3"][0]
        assert "invert_start_m" not in stored

        restored = EarthworkManager()
        restored.from_json(manager.to_json(), geometry_factory=_factory)
        back = [e for e in restored.get_all() if e.name == "Drain 3"][0]
        assert back.invert_start_m is None

    def test_a_link_to_a_feature_that_is_gone_still_opens_the_design(self):
        """Dangling is tolerated at read time, not repaired at write time. Clearing the
        link when the source is deleted would be a second and silently different
        failure: the design would stop meaning what it said with nothing left to report.
        """
        manager = _manager()
        drain = Earthwork("diversion", _WktGeom("LINESTRING (0 0, 50 0)"), "Drain 3")
        drain.spillway_link_id = "a-feature-that-was-deleted:outflow:start"
        manager.add(drain)

        restored = EarthworkManager()
        assert restored.from_json(manager.to_json(), geometry_factory=_factory) == 3
        back = [e for e in restored.get_all() if e.name == "Drain 3"][0]
        assert back.spillway_link_id == "a-feature-that-was-deleted:outflow:start"

        from terrainflow_assessment.modules.earthwork_design import (
            resolve_spillway_links,
        )
        inverts, dangling = resolve_spillway_links(restored.get_all())
        assert inverts == {}
        assert [name for name, _why in dangling] == ["Drain 3"]

    def test_an_unlinked_drain_writes_no_link(self):
        manager = EarthworkManager()
        manager.add(Earthwork("diversion", _WktGeom("LINESTRING (0 0, 50 0)"), "Drain"))
        stored = json.loads(manager.to_json())["earthworks"][0]
        assert stored["spillway_link_id"] is None

    @pytest.mark.parametrize("bad", ["not json", "[1, 2, 3]", ""])
    def test_malformed_payload_degrades_to_empty(self, bad):
        assert DesignDocument.build({}, earthworks_json=bad).earthworks == {}

    def test_count_survives_a_non_list_earthworks_key(self):
        assert DesignDocument.from_dict(
            {"earthworks": {"earthworks": "oops"}}).earthwork_count() == 0


class TestDocumentRoundTrip:
    def _full(self):
        return DesignDocument.build(
            _filled_inputs(),
            areas={"boundary": {"wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))", "crs": "EPSG:2193"}},
            dem=_dem(),
            earthworks_json=_manager().to_json(),
            idf='{"table": 1}',
        )

    def test_everything_survives(self):
        original = self._full()
        restored = DesignDocument.from_json(original.to_json())
        assert restored.inputs == original.inputs
        assert restored.areas == original.areas
        assert restored.dem == original.dem
        assert restored.earthworks == original.earthworks
        assert restored.idf == original.idf

    def test_round_trip_is_stable_across_two_saves(self):
        once = DesignDocument.from_json(self._full().to_json())
        twice = DesignDocument.from_json(once.to_json())
        assert once.to_json() == twice.to_json()

    def test_document_is_stamped_for_identification(self):
        payload = json.loads(self._full().to_json())
        assert payload["format"] == "terrainflow-design"
        assert payload["version"] == SCHEMA_VERSION

    def test_json_is_stable_ordered_for_fixture_diffs(self):
        text = self._full().to_json()
        assert json.dumps(json.loads(text), indent=2, sort_keys=True) == text

    def test_dem_may_be_supplied_as_a_dict(self):
        doc = DesignDocument.build({}, dem={"mode": "embedded", "filename": "dem.tif"})
        assert doc.dem.mode == "embedded"
        assert doc.dem.filename == "dem.tif"

    def test_absent_idf_stays_none(self):
        assert DesignDocument.build({}).idf is None
        assert DesignDocument.build({}, idf="").idf is None

    def test_empty_document_is_usable(self):
        doc = DesignDocument()
        assert doc.inputs == default_inputs()
        assert DesignDocument.from_json(doc.to_json()).inputs == default_inputs()


class TestUnopenableAndUnfamiliarFiles:
    @pytest.mark.parametrize("text", ["", None, "not json at all", "[1, 2]", '"a string"'])
    def test_unreadable_text_yields_none(self, text):
        """The one path that reports failure — an empty design shown as success is worse."""
        assert DesignDocument.from_json(text) is None

    def test_a_document_from_a_newer_build_loads_and_flags_itself(self):
        payload = json.loads(DesignDocument.build(_filled_inputs()).to_json())
        payload["version"] = SCHEMA_VERSION + 5
        payload["future_section"] = {"unknown": True}

        doc = DesignDocument.from_json(json.dumps(payload))
        assert doc is not None
        assert doc.is_from_newer_build()
        assert doc.source_version == SCHEMA_VERSION + 5
        assert doc.inputs["site_name"] == "BX24"

    def test_current_version_is_not_flagged(self):
        assert not DesignDocument.build({}).is_from_newer_build()
        assert not DesignDocument.from_json(
            DesignDocument.build({}).to_json()).is_from_newer_build()

    def test_malformed_version_is_read_as_current(self):
        doc = DesignDocument.from_dict({"version": "tomorrow"})
        assert doc.source_version == SCHEMA_VERSION
        assert not doc.is_from_newer_build()

    def test_from_dict_tolerates_a_non_dict(self):
        assert DesignDocument.from_dict(None).inputs == default_inputs()
        assert DesignDocument.from_dict("nope").inputs == default_inputs()

    def test_a_document_missing_every_section_still_opens(self):
        doc = DesignDocument.from_json('{"version": 1}')
        assert doc is not None
        assert doc.inputs == default_inputs()
        assert doc.areas == {}
        assert doc.dem.mode == "reference"
        assert doc.earthwork_count() == 0
        assert doc.idf is None
