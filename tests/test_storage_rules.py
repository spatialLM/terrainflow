"""The sizing rules a detainment bund and a WASCOB are judged by.

Both are crest types — a wall to an absolute crest, storage measured by flooding the DEM —
so neither has a drawn section to check. What they have instead is a published rule each:

* detainment bund: at least 120 m³ of pond per hectare of contributing catchment, drained
  within three days (Clarke 2013, University of Waikato, from the Lake Rotorua trials);
* WASCOB: no more than 30 acres (12.1 ha) of uncontrolled drainage area per basin
  (NRCS CPS 638).

And one boundary that belongs to neither type but that both can cross: a dam 4 m or higher
**and** holding 20,000 m³ or more is classifiable under NZ's Building (Dam Safety)
Regulations 2022, which is an engineer's signature and not a plugin output.
"""

import math

import pytest

from terrainflow_assessment.core.registry.earthwork_types import (
    get_type,
    is_crest_type,
    offers_spillway,
)


def _advise(**kw):
    from terrainflow_assessment.core.sizing import storage_rule_advisory

    return storage_rule_advisory(**kw)


class TestRegistry:
    def test_both_are_crest_types_on_the_dam_path(self):
        for key in ("detainment_bund", "wascob"):
            cfg = get_type(key)
            assert is_crest_type(key), key
            assert offers_spillway(key), key
            assert cfg.burn_method == "dam", key
            assert not cfg.has_storage and not cfg.has_capacity, key
            assert cfg.has_fill and not cfg.has_cut, key

    def test_the_bund_carries_its_rule(self):
        cfg = get_type("detainment_bund")
        assert cfg.min_storage_m3_per_ha == 120.0
        assert cfg.max_drawdown_hr == 72.0
        assert cfg.max_catchment_ha is None

    def test_the_wascob_carries_its_rule(self):
        cfg = get_type("wascob")
        assert cfg.max_catchment_ha == pytest.approx(12.14, abs=0.01)
        assert cfg.min_storage_m3_per_ha is None

    def test_the_dam_carries_no_rule(self):
        cfg = get_type("dam")
        assert cfg.min_storage_m3_per_ha is None
        assert cfg.max_drawdown_hr is None
        assert cfg.max_catchment_ha is None
        assert cfg.max_storage_m3 is None


class TestDetainmentBundRule:
    def test_the_requirement_is_120_per_hectare(self):
        r = _advise(catchment_m2=30_000.0, storage_m3=400.0, min_storage_m3_per_ha=120.0)
        assert r["required_m3"] == pytest.approx(360.0)
        assert r["holds"] is True
        assert "360" in r["text"]

    def test_short_storage_is_flagged_with_both_figures(self):
        r = _advise(catchment_m2=30_000.0, storage_m3=300.0, min_storage_m3_per_ha=120.0)
        assert r["holds"] is False
        assert any("300" in f and "360" in f for f in r["flags"]), r["flags"]

    def test_unmeasured_storage_claims_nothing(self):
        r = _advise(catchment_m2=30_000.0, storage_m3=None, min_storage_m3_per_ha=120.0)
        assert r["required_m3"] == pytest.approx(360.0)
        assert r["holds"] is None
        assert "not measured" in r["text"]

    def test_no_catchment_means_no_requirement(self):
        r = _advise(catchment_m2=None, storage_m3=300.0, min_storage_m3_per_ha=120.0)
        assert r["required_m3"] is None
        assert r["holds"] is None

    def test_drawdown_is_an_infiltration_only_bound(self):
        """300 m³ over 400 m² at 10 mm/h is 75 h — past three days, and said to be the
        slow case because a decant would shorten it."""
        r = _advise(catchment_m2=30_000.0, storage_m3=300.0, min_storage_m3_per_ha=120.0,
                    max_drawdown_hr=72.0, infiltration_mm_hr=10.0, pond_area_m2=400.0)
        assert r["drawdown_hr"] == pytest.approx(75.0)
        flag = next(f for f in r["flags"] if "75" in f)
        assert "infiltration" in flag and "decant" in flag

    def test_a_pond_that_soaks_away_in_time_is_not_flagged(self):
        r = _advise(catchment_m2=30_000.0, storage_m3=300.0, min_storage_m3_per_ha=120.0,
                    max_drawdown_hr=72.0, infiltration_mm_hr=20.0, pond_area_m2=400.0)
        assert r["drawdown_hr"] == pytest.approx(37.5)
        assert not any("drain" in f for f in r["flags"])

    def test_no_infiltration_is_an_unbounded_drawdown_not_a_crash(self):
        r = _advise(storage_m3=300.0, max_drawdown_hr=72.0, infiltration_mm_hr=0.0,
                    pond_area_m2=400.0)
        assert math.isinf(r["drawdown_hr"])

    def test_an_unverified_ceiling_says_so(self):
        r = _advise(storage_m3=12_000.0, max_storage_m3=10_000.0,
                    max_storage_verified=False)
        flag = next(f for f in r["flags"] if "10,000" in f)
        assert "unverified" in flag


class TestWascobRule:
    def test_a_catchment_over_30_acres_is_flagged(self):
        r = _advise(catchment_m2=150_000.0, max_catchment_ha=12.14)
        assert any("15.0 ha" in f and "CPS 638" in f for f in r["flags"]), r["flags"]

    def test_a_catchment_inside_it_is_not(self):
        r = _advise(catchment_m2=100_000.0, max_catchment_ha=12.14)
        assert r["flags"] == []


class TestClassifiableDam:
    def test_both_thresholds_together_make_it_classifiable(self):
        r = _advise(storage_m3=25_000.0, max_height_m=4.5)
        assert r["classifiable_dam"] is True
        assert any("Dam Safety" in f for f in r["flags"])

    def test_either_alone_does_not(self):
        assert _advise(storage_m3=25_000.0, max_height_m=3.0)["classifiable_dam"] is False
        assert _advise(storage_m3=15_000.0, max_height_m=6.0)["classifiable_dam"] is False

    def test_the_boundary_is_inclusive(self):
        assert _advise(storage_m3=20_000.0, max_height_m=4.0)["classifiable_dam"] is True

    def test_unknown_height_or_volume_is_not_a_verdict(self):
        assert _advise(storage_m3=25_000.0, max_height_m=None)["classifiable_dam"] is None
