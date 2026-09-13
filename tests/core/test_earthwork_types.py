"""Tests for the earthwork type registry (core/registry/earthwork_types.py)."""
import pytest

from terrainflow_assessment.core.registry.earthwork_types import (
    EarthworkTypeConfig,
    all_types,
    get_type,
    is_crest_type,
    is_linear_store,
    name_stem,
    offers_spillway,
    register_type,
)

# `registry_restored`, which the registration tests below take, lives in
# `tests/conftest.py` so that `tests/test_registry_completeness.py` can take it too.


class TestGetType:
    def test_swale_has_storage(self):
        cfg = get_type("swale")
        assert cfg.has_storage is True
        assert cfg.has_capacity is True

    def test_swale_has_cut_and_fill(self):
        cfg = get_type("swale")
        assert cfg.has_cut is True
        assert cfg.has_fill is True

    def test_berm_no_capacity(self):
        cfg = get_type("berm")
        assert cfg.has_capacity is False
        assert cfg.has_cut is False
        assert cfg.has_fill is True

    def test_dam_no_capacity(self):
        cfg = get_type("dam")
        assert cfg.has_capacity is False
        assert cfg.has_cut is False

    def test_basin_has_capacity_no_fill(self):
        cfg = get_type("basin")
        assert cfg.has_capacity is True
        assert cfg.has_fill is False

    def test_diversion_no_capacity_has_cut(self):
        cfg = get_type("diversion")
        assert cfg.has_capacity is False
        assert cfg.has_cut is True

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError):
            get_type("unknown_type_xyz")

    def test_swale_label_capitalised(self):
        assert get_type("swale").label == "Swale"

    def test_diversion_label(self):
        assert get_type("diversion").label == "Diversion Drain"


class TestRegisterType:
    def test_register_terrace_addable_in_one_file(self, registry_restored):
        """New earthwork types can be added via register_type — single-file extension."""
        terrace = EarthworkTypeConfig(
            key="terrace",
            label="Terrace",
            geom_type="LineString",
            has_storage=True,
            has_capacity=True,
            has_cut=True,
            has_fill=True,
            burn_method="berm",   # reuses berm burn logic
            style=("line", "#9C27B0", "2.0"),
        )
        register_type(terrace)
        assert get_type("terrace").label == "Terrace"
        assert get_type("terrace").has_storage is True

    def test_registering_a_type_does_not_outlive_the_test(self, registry_restored):
        """The fixture's own contract — without it the test above leaks into the suite."""
        assert "terrace" not in all_types(), (
            "`terrace` is registered before this test runs, so something registered it "
            "and did not put the registry back"
        )

    def test_all_types_returns_dict(self):
        types = all_types()
        assert isinstance(types, dict)
        assert "swale" in types
        assert "berm" in types
        assert "basin" in types
        assert "dam" in types
        assert "diversion" in types


class TestUiGrouping:
    def test_every_type_has_label_and_valid_category(self):
        for key, cfg in all_types().items():
            assert cfg.label, f"{key} has no label"
            assert cfg.category in ("storage", "control"), f"{key}: {cfg.category!r}"

    def test_storage_group_includes_dam(self):
        # A dam's has_storage=False is a capacity-path flag; to users it's storage.
        assert get_type("dam").category == "storage"
        assert get_type("swale").category == "storage"
        assert get_type("basin").category == "storage"

    def test_control_group(self):
        assert get_type("berm").category == "control"
        assert get_type("diversion").category == "control"

    def test_builtin_tooltips_populated(self):
        for key in ("swale", "berm", "basin", "dam", "diversion"):
            assert get_type(key).tooltip.strip(), f"{key} tooltip empty"


class TestPredicates:
    """The questions the UI asks about a type, answered here and not by comparing keys.

    Pinned per shipped key rather than as a list of the registry, so registering a new
    type changes none of these — a bund that is a crest type is a new assertion, not a
    broken one.
    """

    def test_the_dam_is_the_shipped_crest_type(self):
        assert is_crest_type("dam")
        for key in ("swale", "berm", "basin", "diversion"):
            assert not is_crest_type(key), key

    def test_a_spillway_is_offered_to_what_holds_water_or_stands_a_wall(self):
        for key in ("swale", "basin", "dam"):
            assert offers_spillway(key), key
        for key in ("berm", "diversion"):
            assert not offers_spillway(key), key

    def test_the_swale_is_the_shipped_linear_store(self):
        assert is_linear_store("swale")
        for key in ("berm", "basin", "dam", "diversion"):
            assert not is_linear_store(key), key

    def test_an_unknown_key_answers_no_rather_than_raising(self):
        assert not is_crest_type("nope")
        assert not offers_spillway("nope")
        assert not is_linear_store("nope")

    def test_name_stem_keeps_a_diversion_short(self):
        assert name_stem("diversion") == "Diversion"
        assert name_stem("swale") == "Swale"
        assert name_stem("nope") == "Nope"

    def test_dialog_row_labels(self):
        assert get_type("dam").width_label == "Wall thickness:"
        for key in ("swale", "berm", "basin", "diversion"):
            assert get_type(key).depth_label == "Depth:", key
            assert get_type(key).width_label == "Width:", key
