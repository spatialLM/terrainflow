"""Tests for terrainflow_assessment/modules/water_balance.py — analytical routed balance.

The balance is now driven by **mutually exclusive direct catchments**: each store's
``inflow_m3`` is the runoff from cells it is the first to intercept, and whatever
reaches no feature is passed in as ``uncaptured_m3``. So the site total is always
``Σ direct inflows + uncaptured``, and every test states all three consistently —
that consistency is precisely what the old model lacked when it reported 100%.
"""
import pytest

from terrainflow_assessment.modules.simulation import EarthworkStore, resolve_targets
from terrainflow_assessment.modules.water_balance import area_subtotals, run_water_balance


def _store(name, capacity, elevation, inflow, area=100.0, infil=0.0, target=None,
           ew_type="swale"):
    s = EarthworkStore(
        name=name, ew_type=ew_type, capacity_m3=capacity, area_m2=area,
        infiltration_rate_mm_hr=infil, elevation=elevation,
        id=name, overflow_target_id=target,
    )
    s.inflow_m3 = inflow
    return s


class TestRunWaterBalance:
    def test_single_feature_capture_pct(self):
        # 50 m³ lands in the store, 50 m³ of the site drains nowhere near it → 50%.
        r = run_water_balance([_store("S1", 100.0, 10.0, 50.0)], 1.0,
                              total_runoff_m3=100.0, uncaptured_m3=50.0)
        assert r.total_captured_m3 == pytest.approx(50.0)
        assert r.capture_pct == pytest.approx(50.0)
        assert r.site_exit_m3 == pytest.approx(50.0)
        assert r.uncaptured_m3 == pytest.approx(50.0)
        assert r.routed_exit_m3 == pytest.approx(0.0)
        assert r.total_capacity_m3 == pytest.approx(100.0)
        assert r.mass_balance_ok

    def test_infiltration_scales_with_duration(self):
        # rate 10 mm/hr × 100 m² × 2 hr = 2 m³ infiltrated
        r = run_water_balance(
            [_store("S1", 1000.0, 10.0, 500.0, area=100.0, infil=10.0)],
            2.0, total_runoff_m3=500.0,
        )
        assert r.total_infiltration_m3 == pytest.approx(2.0)
        assert r.total_captured_m3 == pytest.approx(500.0)  # 498 stored + 2 infiltrated

    def test_stacked_features_do_not_double_count_the_hillside(self):
        """Regression for the deleted ``_apply_interception``.

        Direct catchments are already disjoint, so the downstream feature is handed
        only its own band — no correction is applied, and none is needed. The upper
        feature's overflow arrives on top of that band, once.
        """
        s1 = _store("S1", 30.0, 20.0, 50.0)   # upslope band
        s2 = _store("S2", 100.0, 10.0, 30.0)  # the band between them only
        routing = resolve_targets([s1, s2], walker=lambda s: "S2" if s.id == "S1" else None)
        r = run_water_balance([s1, s2], 1.0, total_runoff_m3=80.0, routing=routing)

        by_id = {f["id"]: f for f in r.per_feature}
        assert by_id["S2"]["direct_inflow_m3"] == pytest.approx(30.0)
        assert by_id["S1"]["overflowed"] is True            # 50 into a 30 store
        assert by_id["S1"]["overflow_m3"] == pytest.approx(20.0)
        # S2 receives its own 30 plus S1's 20 overflow — counted once, not twice.
        assert by_id["S2"]["upstream_inflow_m3"] == pytest.approx(20.0)
        assert by_id["S2"]["stored_m3"] == pytest.approx(50.0)
        assert r.total_captured_m3 == pytest.approx(80.0)
        assert r.capture_pct == pytest.approx(100.0)
        assert r.mass_balance_ok

    def test_upstream_capture_frees_downstream_capacity(self):
        """Adding upslope storage must visibly reduce what reaches the feature below."""
        alone = _store("S2", 100.0, 10.0, 80.0)
        r_alone = run_water_balance([alone], 1.0, total_runoff_m3=80.0)

        s1 = _store("S1", 50.0, 20.0, 50.0)
        s2 = _store("S2", 100.0, 10.0, 30.0)
        routing = resolve_targets([s1, s2], walker=lambda s: "S2" if s.id == "S1" else None)
        r_pair = run_water_balance([s1, s2], 1.0, total_runoff_m3=80.0, routing=routing)

        by_id = {f["id"]: f for f in r_pair.per_feature}
        assert by_id["S2"]["total_inflow_m3"] < r_alone.per_feature[0]["total_inflow_m3"]

    def test_mass_balance_closes_across_a_five_feature_network(self):
        stores = [_store(f"S{i}", 40.0, 50.0 - i * 5, 30.0) for i in range(5)]
        links = {f"S{i}": f"S{i + 1}" for i in range(4)}
        routing = resolve_targets(stores, walker=lambda s: links.get(s.id))
        r = run_water_balance(stores, 1.0, total_runoff_m3=200.0,
                              uncaptured_m3=50.0, routing=routing)
        assert r.total_captured_m3 + r.site_exit_m3 == pytest.approx(200.0, abs=1e-6)
        assert r.mass_balance_ok
        assert 0.0 <= r.capture_pct <= 100.0

    def test_terminal_deficit_names_the_storage_shortfall(self):
        """Overflow past the last feature is what "more storage needed upslope" means."""
        s1 = _store("S1", 10.0, 20.0, 40.0)
        s2 = _store("S2", 10.0, 10.0, 0.0)
        routing = resolve_targets([s1, s2], walker=lambda s: "S2" if s.id == "S1" else None)
        r = run_water_balance([s1, s2], 1.0, total_runoff_m3=40.0, routing=routing)

        by_id = {f["id"]: f for f in r.per_feature}
        assert by_id["S2"]["is_terminal"] is True
        assert by_id["S1"]["is_terminal"] is False
        # S1 holds 10 and spills 30; S2 holds 10 and spills 20 off site.
        assert r.terminal_deficit_m3 == pytest.approx(20.0)
        assert r.routed_exit_m3 == pytest.approx(20.0)

    def test_zero_capacity_feature_redirects_everything_it_intercepts(self):
        """A berm stores nothing but must hand its catchment to its target."""
        berm = _store("B1", 0.0, 20.0, 60.0, ew_type="berm")
        basin = _store("P1", 100.0, 10.0, 0.0, ew_type="basin")
        routing = resolve_targets([berm, basin],
                                  walker=lambda s: "P1" if s.id == "B1" else None)
        r = run_water_balance([berm, basin], 1.0, total_runoff_m3=60.0, routing=routing)

        by_id = {f["id"]: f for f in r.per_feature}
        assert by_id["B1"]["stored_m3"] == pytest.approx(0.0)
        assert by_id["B1"]["overflow_m3"] == pytest.approx(60.0)
        assert by_id["B1"]["target_id"] == "P1"
        assert by_id["P1"]["stored_m3"] == pytest.approx(60.0)
        assert r.capture_pct == pytest.approx(100.0)

    def test_no_runoff_gives_zero_capture(self):
        # geometry-only (no baseline): capture % is 0 but capacity still reported
        r = run_water_balance([_store("S1", 100.0, 10.0, 0.0)], 1.0, total_runoff_m3=0.0)
        assert r.capture_pct == 0.0
        assert r.site_exit_m3 == 0.0
        assert r.total_capacity_m3 == pytest.approx(100.0)
        assert r.mass_balance_ok

    def test_empty_stores_capture_nothing(self):
        r = run_water_balance([], 1.0, total_runoff_m3=100.0, uncaptured_m3=100.0)
        assert r.capture_pct == 0.0
        assert r.site_exit_m3 == pytest.approx(100.0)
        assert r.total_capacity_m3 == 0.0
        assert r.per_feature == []
        assert r.mass_balance_ok

    def test_capture_pct_is_not_clamped_and_flags_a_broken_balance(self):
        """An impossible score must be visible, not hidden behind ``min(100, …)``.

        Capturing 50 m³ out of a stated 10 m³ storm means the inflow bookkeeping and
        the denominator disagree; the old code clamped to a serene 100%.
        """
        r = run_water_balance([_store("S1", 100.0, 10.0, 50.0)], 1.0, total_runoff_m3=10.0)
        assert r.capture_pct == 100.0        # (10 − 0) / 10
        assert r.mass_balance_ok is False    # but 50 captured out of 10 is impossible

    def test_zero_capacity_feature_fill_pct_safe(self):
        # capacity 0 must not divide by zero in the per-feature fill %
        r = run_water_balance([_store("S0", 0.0, 10.0, 5.0)], 1.0, total_runoff_m3=5.0)
        assert r.per_feature[0]["fill_pct"] == 0.0

    def test_routing_warnings_are_surfaced(self):
        a = _store("A", 10.0, 20.0, 5.0, target="B")
        b = _store("B", 10.0, 10.0, 5.0, target="A")
        r = run_water_balance([a, b], 1.0, total_runoff_m3=10.0)
        assert r.routing_warnings
        assert "loop" in r.routing_warnings[0].lower()


class TestInfiltrationToggle:
    """Sizing on held volume alone, with soakage reported as a buffer.

    The infiltration model applies one steady-state rate per soil texture for the
    whole event and has no saturation limit, so it will keep absorbing as long as
    water is present. Treating that as spare capacity rather than capture is the
    defensible way to size.
    """

    def _basin(self):
        # 3,459 m² floor, Loam 4 mm/hr → 332 m³ of soakage over 24 h.
        return _store("B1", capacity=3974.0, elevation=10.0, inflow=230.0,
                      area=3459.0, infil=4.0, ew_type="basin")

    def test_soakage_credited_when_on(self):
        r = run_water_balance([self._basin()], 24.0, total_runoff_m3=230.0,
                              count_infiltration=True)
        assert r.total_infiltration_m3 == pytest.approx(230.0)
        assert r.per_feature[0]["stored_m3"] == pytest.approx(0.0)
        assert r.capture_pct == pytest.approx(100.0)
        assert r.counts_infiltration is True

    def test_water_must_be_held_when_off(self):
        r = run_water_balance([self._basin()], 24.0, total_runoff_m3=230.0,
                              count_infiltration=False)
        assert r.total_infiltration_m3 == pytest.approx(0.0)
        assert r.per_feature[0]["stored_m3"] == pytest.approx(230.0)
        assert r.capture_pct == pytest.approx(100.0)
        assert r.counts_infiltration is False

    def test_the_buffer_is_still_reported_when_off(self):
        r = run_water_balance([self._basin()], 24.0, total_runoff_m3=230.0,
                              count_infiltration=False)
        # Capped by the water present (230), not the 332 the soil could take.
        assert r.infiltration_buffer_m3 == pytest.approx(230.0)
        assert r.per_feature[0]["infiltration_buffer_m3"] == pytest.approx(230.0)

    def test_turning_it_off_can_overflow_a_feature_that_previously_coped(self):
        """The point of the toggle: undersized features stop hiding behind soakage."""
        small = _store("S1", capacity=50.0, elevation=10.0, inflow=230.0,
                       area=3459.0, infil=4.0)
        on = run_water_balance([small], 24.0, total_runoff_m3=230.0,
                               count_infiltration=True)
        small = _store("S1", capacity=50.0, elevation=10.0, inflow=230.0,
                       area=3459.0, infil=4.0)
        off = run_water_balance([small], 24.0, total_runoff_m3=230.0,
                                count_infiltration=False)
        assert on.per_feature[0]["overflowed"] is False
        assert off.per_feature[0]["overflowed"] is True
        assert off.capture_pct < on.capture_pct

    def test_default_is_to_credit_soakage(self):
        r = run_water_balance([self._basin()], 24.0, total_runoff_m3=230.0)
        assert r.counts_infiltration is True


class TestPerFeatureSoil:
    def test_a_feature_can_override_the_site_soil(self):
        from terrainflow_assessment.modules.swale_design import get_infiltration_rate
        assert get_infiltration_rate("Clay") < get_infiltration_rate("Sand")

    def test_clay_under_a_basin_stores_what_loam_would_soak_away(self):
        clay = _store("B1", capacity=3974.0, elevation=10.0, inflow=230.0,
                      area=3459.0, infil=1.5, ew_type="basin")     # Clay
        loam = _store("B2", capacity=3974.0, elevation=10.0, inflow=230.0,
                      area=3459.0, infil=4.0, ew_type="basin")     # Loam
        r_clay = run_water_balance([clay], 24.0, total_runoff_m3=230.0,
                                   count_infiltration=True)
        r_loam = run_water_balance([loam], 24.0, total_runoff_m3=230.0,
                                   count_infiltration=True)
        # Clay soaks 124 m³ over the event, so 106 m³ has to pond.
        assert r_clay.per_feature[0]["stored_m3"] > 0
        assert r_loam.per_feature[0]["stored_m3"] == pytest.approx(0.0)


class TestAreaSubtotals:
    """One site-wide capture figure hides where the problem is. These rows partition
    the same balance rather than computing a second, disagreeing one."""

    @staticmethod
    def _grid():
        import numpy as np

        from terrainflow_assessment.modules.flow_graph import LABEL_EXIT, LABEL_SINK
        # 4x4: top half drains to feature 0, bottom half leaves, one interior sink.
        labels = np.full((4, 4), LABEL_EXIT, dtype=np.int32)
        labels[0:2, :] = 0
        labels[3, 3] = LABEL_SINK
        domain = np.ones((4, 4), dtype=bool)
        return labels, domain

    def _masks(self):
        import numpy as np
        top = np.zeros((4, 4), dtype=bool)
        top[0:2, :] = True
        bottom = np.zeros((4, 4), dtype=bool)
        bottom[2:4, :] = True
        return {"Top paddock": top, "Bottom paddock": bottom}

    def test_each_area_reports_its_own_capture(self):
        labels, domain = self._grid()
        rows = area_subtotals(labels, domain, self._masks(), 1.0, 100.0)
        by_name = {r["name"]: r for r in rows}
        assert by_name["Top paddock"]["capture_pct"] == pytest.approx(100.0)
        assert by_name["Bottom paddock"]["capture_pct"] == pytest.approx(0.0)

    def test_a_site_wide_figure_would_have_hidden_that(self):
        """50% overall, but it is 100% and 0% — different work in each half."""
        labels, domain = self._grid()
        rows = area_subtotals(labels, domain, self._masks(), 1.0, 100.0)
        overall = (sum(r["intercepted_m3"] for r in rows)
                   / sum(r["runoff_m3"] for r in rows) * 100.0)
        assert overall == pytest.approx(50.0)
        assert {round(r["capture_pct"]) for r in rows} == {0, 100}

    def test_the_rows_partition_the_site(self):
        labels, domain = self._grid()
        rows = area_subtotals(labels, domain, self._masks(), 1.0, 100.0)
        assert sum(r["cells"] for r in rows) == int(domain.sum())
        for r in rows:
            assert (r["intercepted_m3"] + r["exit_m3"] + r["sink_m3"]
                    == pytest.approx(r["runoff_m3"]))

    def test_interior_sinks_are_reported_separately(self):
        labels, domain = self._grid()
        rows = area_subtotals(labels, domain, self._masks(), 1.0, 100.0)
        bottom = next(r for r in rows if r["name"] == "Bottom paddock")
        assert bottom["sink_m3"] > 0

    def test_runoff_scales_with_depth_and_cell_area(self):
        labels, domain = self._grid()
        a = area_subtotals(labels, domain, self._masks(), 1.0, 100.0)
        b = area_subtotals(labels, domain, self._masks(), 4.0, 200.0)
        assert b[0]["runoff_m3"] == pytest.approx(a[0]["runoff_m3"] * 8)

    def test_cells_outside_the_domain_are_not_counted(self):
        labels, domain = self._grid()
        domain = domain.copy()
        domain[0, :] = False
        rows = area_subtotals(labels, domain, self._masks(), 1.0, 100.0)
        top = next(r for r in rows if r["name"] == "Top paddock")
        assert top["cells"] == 4

    def test_an_empty_area_is_omitted(self):
        import numpy as np
        labels, domain = self._grid()
        masks = {"Nowhere": np.zeros((4, 4), dtype=bool)}
        assert area_subtotals(labels, domain, masks, 1.0, 100.0) == []

    def test_a_mismatched_mask_is_skipped_not_crashed(self):
        import numpy as np
        labels, domain = self._grid()
        masks = {"Wrong shape": np.ones((3, 3), dtype=bool)}
        assert area_subtotals(labels, domain, masks, 1.0, 100.0) == []

    def test_no_areas_gives_no_rows(self):
        labels, domain = self._grid()
        assert area_subtotals(labels, domain, None, 1.0, 100.0) == []
