"""
peak_flow.py — design flow *rates*, and how they accumulate down a network.

Storage sizing asks how many cubic metres arrive over an event. Spillway sizing asks
how many litres per second arrive at the worst moment. They are different questions
with different answers, and conflating them is how a spillway ends up an order of
magnitude too narrow: dividing event volume by event duration yields the storm-average
intensity, which for a 120 mm / 24 h design storm is 5 mm/hr — a daily mean, not a peak.

Three things live here.

``peak_runoff_fraction``
    The instantaneous fraction of rainfall becoming flow, derived from whichever
    sizing basis the Baseline tab is set to, so the spillway cannot contradict the
    storage it protects.

``rational_peak_flow``
    ``Q = fraction × i × A``. The area comes from the flow graph's measured direct
    catchment, so it is a real number rather than an assumption.

``cascade_peak_flows``
    Once a feature is full, what arrives leaves. A feature therefore has to pass its
    own catchment's peak *plus* everything spilling into it from upslope — ignoring
    that undersized a downstream swale by 43% in the worked example that prompted
    this module.

All pure: no QGIS, no rasters, no I/O.
"""

from terrainflow_assessment.modules.catchment import scs_marginal_runoff_fraction

# Basis keys as the panel emits them (``panel.sizing_basis``).
BASIS_RAINFALL = "rainfall"
BASIS_COEFFICIENT = "coefficient"
BASIS_SCS = "runoff"

# Below this, a runoff coefficient is a water-*harvesting* figure rather than a
# design-storm one. Lancaster's table describes how much can be captured from
# ordinary rain; a spillway answers what happens in the extreme event on saturated
# ground. His "Grass / lawn" 0.18 sizes a 2.8 ha spillway at 0.20 m where the same
# ground wet on SCS-CN needs 0.94 m — 4.7x wider — so the gap is worth naming.
HARVESTING_COEFFICIENT_CEILING = 0.35

# Default peak design intensity (mm/hr). Sits between the "Moderate" and "Heavy"
# storm presets already in SCSRunoff. There is no defensible way to derive this from
# a depth and a duration, so it is a starting point to be replaced with a local
# figure, not a computed answer — the Compare dialog exists to make that visible.
DEFAULT_PEAK_INTENSITY_MM_HR = 40.0


def peak_runoff_fraction(basis, rainfall_mm=None, coefficient=None, cn=None):
    """Instantaneous runoff fraction for *basis* — the ``C`` of the rational method.

    - ``rainfall``    → 1.0. Every millimetre runs off, by definition of the basis.
    - ``coefficient`` → the coefficient itself, which is already instantaneous: that
      is what a rational-method ``C`` means.
    - ``runoff``      → the SCS **marginal** fraction ``dQ/dP``, not the event
      average ``Q/P``. See :func:`~terrainflow_assessment.modules.catchment.
      scs_marginal_runoff_fraction` for why, and for what evaluating it at the full
      storm depth assumes.

    Unknown bases fall back to the coefficient when one is supplied, else 1.0 — the
    conservative direction, because a spillway that is too wide costs excavation and
    one that is too narrow costs the embankment.
    """
    if basis == BASIS_RAINFALL:
        return 1.0
    if basis == BASIS_COEFFICIENT:
        if coefficient is None:
            return 1.0
        return max(0.0, min(1.0, float(coefficient)))
    if basis == BASIS_SCS:
        return scs_marginal_runoff_fraction(rainfall_mm, cn)
    if coefficient is not None:
        return max(0.0, min(1.0, float(coefficient)))
    return 1.0


def rational_peak_flow(fraction, intensity_mm_hr, area_m2):
    """Peak flow in m³/s — ``Q = C × i × A``, the rational method.

    *intensity_mm_hr* is the design intensity read off an IDF curve **at a duration
    equal to the time of concentration** — standard rational-method practice, and what
    :meth:`RainfallIDF.intensity_mm_hr` supplies. What it must not be is a whole storm
    averaged over its own length (120 mm spread across 24 hours), which is a far lower
    number and the reason for the warning this sentence used to carry.

    *area_m2* is the feature's measured direct catchment: the intensity is per unit
    area, the area is the whole contributing catchment, and the product is everything
    arriving at that one feature.
    """
    if not intensity_mm_hr or not area_m2 or intensity_mm_hr <= 0 or area_m2 <= 0:
        return 0.0
    return max(0.0, float(fraction)) * (float(intensity_mm_hr) / 3_600_000.0) * float(area_m2)


def coefficient_is_harvesting_grade(basis, coefficient):
    """True when a runoff coefficient is too low to size an overflow structure on.

    Not a correctness test — a judgement that the value was calibrated for a
    different question. Only meaningful on the coefficient basis.
    """
    return (basis == BASIS_COEFFICIENT
            and coefficient is not None
            and float(coefficient) < HARVESTING_COEFFICIENT_CEILING)


def cascade_peak_flows(direct_flows, edges, order=None):
    """Accumulate peak flow downstream — ``{id: total_m3s}``.

    *direct_flows* maps feature id to the peak from its **own** catchment; *edges*
    maps feature id to the id it overflows into (``None`` = leaves the site); *order*
    is a topological ordering from :func:`~terrainflow_assessment.modules.flow_graph.
    topological_order`, so each feature is processed after everything feeding it.

    The routing assumption is that a full feature passes its whole peak on. That is
    correct at the moment a spillway matters — steady state, storage already
    committed — and deliberately ignores attenuation, which would need a routed
    hydrograph and would only ever reduce the answer.

    Cycles cannot deadlock this: nodes are visited in the order given and each is
    read once, so a ring simply routes without converging. Cycle *detection* belongs
    upstream, where the user's link can be refused with an explanation.
    """
    totals = {k: float(v or 0.0) for k, v in direct_flows.items()}
    if order is None:
        order = list(direct_flows.keys())

    received = {k: 0.0 for k in totals}
    for node in order:
        if node not in totals:
            continue
        total = totals[node] + received.get(node, 0.0)
        totals[node] = total
        target = edges.get(node)
        if target is not None and target in received and target != node:
            received[target] += total
    return totals


def upstream_contributions(direct_flows, edges, order=None):
    """``{id: (own_m3s, from_upstream_m3s)}`` — the cascade, itemised.

    Same arithmetic as :func:`cascade_peak_flows`, kept separate because the UI needs
    to say *why* a feature's design flow grew, not merely that it did.
    """
    totals = cascade_peak_flows(direct_flows, edges, order=order)
    return {
        k: (float(direct_flows.get(k, 0.0) or 0.0),
            totals[k] - float(direct_flows.get(k, 0.0) or 0.0))
        for k in totals
    }
