"""Design flow rates: the runoff fraction, the rational method, and the cascade.

Spillway sizing is where being wrong is least forgiving — too narrow does not degrade
gracefully, it overtops and cuts the embankment. Two errors are pinned here because
both were live in the code before this module existed: using the storm-*average*
intensity (5 mm/hr on a 120 mm / 24 h storm, roughly an order of magnitude low), and
ignoring the overflow arriving from upslope features.
"""

import pytest

from terrainflow_assessment.modules.catchment import (
    SCSRunoff,
    scs_marginal_runoff_fraction,
)
from terrainflow_assessment.modules.flow_graph import topological_order
from terrainflow_assessment.modules.peak_flow import (
    BASIS_COEFFICIENT,
    BASIS_RAINFALL,
    BASIS_SCS,
    cascade_peak_flows,
    coefficient_is_harvesting_grade,
    peak_runoff_fraction,
    rational_peak_flow,
    upstream_contributions,
)


def _numeric_dQdP(rainfall_mm, cn, h=1e-6):
    """dQ/dP by central difference, straight off the model's own runoff_depth."""
    s = SCSRunoff()
    return (s.runoff_depth(rainfall_mm + h, cn)
            - s.runoff_depth(rainfall_mm - h, cn)) / (2 * h)


class TestMarginalRunoffFraction:
    @pytest.mark.parametrize("cn", [40.0, 61.0, 78.2, 80.0, 90.2, 95.0])
    @pytest.mark.parametrize("rain", [20.0, 60.0, 120.0, 300.0])
    def test_matches_a_numerical_derivative_of_the_scs_curve(self, cn, rain):
        """The closed form is only worth having if it is the actual derivative of
        the runoff model in use, so check it against that model rather than against
        a restatement of the same algebra."""
        closed = scs_marginal_runoff_fraction(rain, cn)
        if closed == 0.0:
            return                       # below Ia; the difference straddles the kink
        assert closed == pytest.approx(_numeric_dQdP(rain, cn), abs=1e-6)

    def test_zero_at_the_initial_abstraction(self):
        s = 25400.0 / 61.0 - 254.0
        assert scs_marginal_runoff_fraction(0.2 * s, 61.0) == 0.0

    def test_nothing_runs_off_below_the_initial_abstraction(self):
        assert scs_marginal_runoff_fraction(10.0, 61.0) == 0.0

    def test_approaches_one_for_very_large_storms(self):
        assert scs_marginal_runoff_fraction(10_000.0, 61.0) == pytest.approx(1.0, abs=1e-3)

    def test_monotonically_increasing(self):
        vals = [scs_marginal_runoff_fraction(p, 61.0) for p in range(40, 600, 10)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    @pytest.mark.parametrize("cn", [30.0, 61.0, 95.0])
    @pytest.mark.parametrize("rain", [50.0, 120.0, 400.0])
    def test_always_a_fraction(self, cn, rain):
        assert 0.0 <= scs_marginal_runoff_fraction(rain, cn) <= 1.0

    def test_always_exceeds_the_event_average(self):
        """The whole reason this function exists: Q/P understates the peak."""
        s = SCSRunoff()
        for rain in (60.0, 120.0, 250.0):
            avg = s.runoff_depth(rain, 61.0) / rain
            assert scs_marginal_runoff_fraction(rain, 61.0) > avg

    def test_the_understatement_is_large_at_the_design_storm(self):
        """CN 61, 120 mm: 0.578 marginal against 0.255 average. Sizing on the
        average would more than halve the spillway."""
        assert scs_marginal_runoff_fraction(120.0, 61.0) == pytest.approx(0.578, abs=0.001)
        assert SCSRunoff().runoff_depth(120.0, 61.0) / 120.0 == pytest.approx(0.255, abs=0.001)

    def test_wetter_ground_sheds_more_of_the_next_millimetre(self):
        s = SCSRunoff()
        normal = scs_marginal_runoff_fraction(120.0, 61.0)
        wet = scs_marginal_runoff_fraction(120.0, s.adjust_cn(61.0, "wet"))
        assert wet > normal

    @pytest.mark.parametrize("cn", [0.0, -5.0, None])
    def test_degenerate_curve_numbers_yield_nothing(self, cn):
        assert scs_marginal_runoff_fraction(120.0, cn) == 0.0

    def test_missing_rainfall_yields_nothing(self):
        assert scs_marginal_runoff_fraction(None, 61.0) == 0.0


class TestPeakRunoffFraction:
    def test_rainfall_basis_is_everything(self):
        assert peak_runoff_fraction(BASIS_RAINFALL) == 1.0

    def test_coefficient_basis_uses_the_coefficient_directly(self):
        """A rational-method C is already an instantaneous fraction — no translation."""
        assert peak_runoff_fraction(BASIS_COEFFICIENT, coefficient=0.5) == 0.5

    def test_coefficient_is_clamped(self):
        assert peak_runoff_fraction(BASIS_COEFFICIENT, coefficient=1.8) == 1.0
        assert peak_runoff_fraction(BASIS_COEFFICIENT, coefficient=-0.2) == 0.0

    def test_scs_basis_uses_the_marginal_not_the_average(self):
        got = peak_runoff_fraction(BASIS_SCS, rainfall_mm=120.0, cn=61.0)
        assert got == pytest.approx(scs_marginal_runoff_fraction(120.0, 61.0))
        assert got > SCSRunoff().runoff_depth(120.0, 61.0) / 120.0

    def test_unknown_basis_falls_back_conservatively(self):
        assert peak_runoff_fraction("mystery") == 1.0
        assert peak_runoff_fraction("mystery", coefficient=0.4) == 0.4

    def test_coefficient_basis_without_a_coefficient_does_not_silently_zero(self):
        """Returning 0 would size a spillway at nothing; 1.0 errs the safe way."""
        assert peak_runoff_fraction(BASIS_COEFFICIENT) == 1.0


class TestRationalPeakFlow:
    def test_area_is_the_whole_catchment_not_a_unit_area(self):
        """2.8 ha at 40 mm/hr and C=0.5 is 156 L/s arriving at the one feature."""
        q = rational_peak_flow(0.5, 40.0, 28_000.0)
        assert q * 1000 == pytest.approx(155.6, abs=0.1)

    def test_scales_linearly_with_area(self):
        assert (rational_peak_flow(0.5, 40.0, 20_000.0)
                == pytest.approx(2 * rational_peak_flow(0.5, 40.0, 10_000.0)))

    def test_scales_linearly_with_intensity(self):
        assert (rational_peak_flow(0.5, 80.0, 10_000.0)
                == pytest.approx(2 * rational_peak_flow(0.5, 40.0, 10_000.0)))

    def test_the_storm_average_is_the_bug_this_replaces(self):
        """120 mm over 24 h averages 5 mm/hr. A real design intensity is many times
        that, and the flow — hence the spillway width — follows proportionally."""
        avg = rational_peak_flow(0.5, 120.0 / 24.0, 28_000.0)
        design = rational_peak_flow(0.5, 40.0, 28_000.0)
        assert design / avg == pytest.approx(8.0, abs=0.01)

    @pytest.mark.parametrize("i,a", [(0, 100.0), (40.0, 0), (None, 100.0), (40.0, None)])
    def test_missing_inputs_give_no_flow(self, i, a):
        assert rational_peak_flow(0.5, i, a) == 0.0


class TestHarvestingCoefficientGuard:
    def test_lancasters_grass_value_is_flagged(self):
        """0.18 is a harvesting figure. Sizing an overflow structure on it gives
        roughly a fifth of what saturated ground on SCS-CN demands."""
        assert coefficient_is_harvesting_grade(BASIS_COEFFICIENT, 0.18)

    def test_the_design_default_is_not_flagged(self):
        assert not coefficient_is_harvesting_grade(BASIS_COEFFICIENT, 0.50)

    def test_only_applies_to_the_coefficient_basis(self):
        assert not coefficient_is_harvesting_grade(BASIS_SCS, 0.18)
        assert not coefficient_is_harvesting_grade(BASIS_COEFFICIENT, None)


class TestCascade:
    def test_a_lone_feature_passes_only_its_own(self):
        assert cascade_peak_flows({"a": 0.1}, {"a": None}) == {"a": pytest.approx(0.1)}

    def test_a_downstream_feature_carries_the_one_above_it(self):
        edges = {"basin": "swale", "swale": None}
        order, _ = topological_order(edges)
        totals = cascade_peak_flows({"basin": 0.0667, "swale": 0.1556}, edges, order)
        assert totals["basin"] == pytest.approx(0.0667)
        assert totals["swale"] == pytest.approx(0.2223)

    def test_ignoring_upstream_undersizes_the_receiver(self):
        """The 43% the worked example found."""
        edges = {"basin": "swale", "swale": None}
        order, _ = topological_order(edges)
        totals = cascade_peak_flows({"basin": 0.0667, "swale": 0.1556}, edges, order)
        assert totals["swale"] / 0.1556 == pytest.approx(1.43, abs=0.01)

    def test_accumulates_along_a_chain(self):
        edges = {"a": "b", "b": "c", "c": None}
        order, _ = topological_order(edges)
        totals = cascade_peak_flows({"a": 1.0, "b": 2.0, "c": 3.0}, edges, order)
        assert totals == {"a": pytest.approx(1.0), "b": pytest.approx(3.0),
                          "c": pytest.approx(6.0)}

    def test_two_features_spilling_into_one_both_count(self):
        edges = {"a": "c", "b": "c", "c": None}
        order, _ = topological_order(edges)
        totals = cascade_peak_flows({"a": 1.0, "b": 2.0, "c": 0.5}, edges, order)
        assert totals["c"] == pytest.approx(3.5)

    def test_a_target_outside_the_set_is_water_leaving_the_site(self):
        totals = cascade_peak_flows({"a": 1.0}, {"a": "gone"}, ["a"])
        assert totals == {"a": pytest.approx(1.0)}

    def test_a_self_link_does_not_double_count(self):
        totals = cascade_peak_flows({"a": 1.0}, {"a": "a"}, ["a"])
        assert totals["a"] == pytest.approx(1.0)

    def test_a_cycle_terminates_rather_than_deadlocking(self):
        """Cycles are refused at the UI, but the cascade must not hang if one
        reaches it — every node is read exactly once, in the order given."""
        edges = {"a": "b", "b": "a"}
        order, broken = topological_order(edges)
        assert broken                      # it is genuinely a ring
        totals = cascade_peak_flows({"a": 1.0, "b": 1.0}, edges, order)
        assert all(v > 0 for v in totals.values())

    def test_a_routed_feature_with_no_catchment_is_skipped(self):
        """The order comes from the routing graph, which can name features the
        catchment labelling never gave a cell to — a berm sitting on a ridge, or one
        whose footprint missed the domain. It contributes nothing rather than
        raising, and must not take the features around it down with it."""
        edges = {"a": "ghost", "ghost": "b", "b": None}
        order, _ = topological_order(edges)
        totals = cascade_peak_flows({"a": 1.0, "b": 2.0}, edges, order)
        assert set(totals) == {"a", "b"}
        assert totals["a"] == pytest.approx(1.0)
        assert totals["b"] == pytest.approx(2.0)   # the gap breaks the chain, safely

    def test_missing_order_still_works(self):
        totals = cascade_peak_flows({"a": 1.0, "b": 2.0}, {"a": "b", "b": None})
        assert set(totals) == {"a", "b"}

    def test_none_flows_are_treated_as_zero(self):
        totals = cascade_peak_flows({"a": None, "b": 1.0}, {"a": "b", "b": None}, ["a", "b"])
        assert totals["b"] == pytest.approx(1.0)


class TestUpstreamContributions:
    def test_splits_own_from_received(self):
        edges = {"basin": "swale", "swale": None}
        order, _ = topological_order(edges)
        parts = upstream_contributions({"basin": 0.0667, "swale": 0.1556}, edges, order)
        own, upstream = parts["swale"]
        assert own == pytest.approx(0.1556)
        assert upstream == pytest.approx(0.0667)

    def test_a_source_feature_receives_nothing(self):
        edges = {"basin": "swale", "swale": None}
        order, _ = topological_order(edges)
        parts = upstream_contributions({"basin": 0.0667, "swale": 0.1556}, edges, order)
        assert parts["basin"][1] == pytest.approx(0.0)

    def test_the_parts_sum_to_the_cascade_total(self):
        edges = {"a": "b", "b": "c", "c": None}
        order, _ = topological_order(edges)
        direct = {"a": 1.0, "b": 2.0, "c": 3.0}
        totals = cascade_peak_flows(direct, edges, order)
        for k, (own, up) in upstream_contributions(direct, edges, order).items():
            assert own + up == pytest.approx(totals[k])
