"""
water_balance.py — design-tier analytical earthwork assessment.

The fast, no-burn half of the two-tier method (*"analytical is where you design;
burning is where you verify"*). Given the current earthwork network it computes a
lumped, event-total **routed water balance** — an *estimate*, not hydrodynamics —
suitable to recompute live as features are added/moved/edited.

Model
-----
1. **Direct catchment.** Each feature's inflow is the runoff from the cells it is the
   *first* to intercept, measured by ``flow_graph.label_direct_catchments`` and
   preloaded onto ``EarthworkStore.inflow_m3`` by the caller. Those catchments are
   mutually exclusive and exhaustive by construction, so a shared hillside is never
   counted twice and no interception correction is needed.
2. **Routing + capture**: one pass of ``simulation.cascade_overflow`` applies
   infiltration, fills each feature to capacity, and cascades overflow along the
   resolved flow paths.
3. **Site exit** = runoff that reached no feature at all (*uncaptured*, measured from
   the label raster) **plus** overflow that cascaded off the site (*routed*, returned
   by the cascade). Both are measured; neither is a residual.
4. **Score**: capture % = (total runoff − site exit) ÷ total runoff.

Why this file was rewritten
---------------------------
The previous model reported "100% of the storm held on site, 0 m³ leaves" for designs
that plainly did not capture everything. Three compounding causes, all fixed here:

* ``site_exit`` was a **residual**, ``max(0, total_runoff − captured)``, and the score
  was clamped with ``min(100, …)`` — so any over-estimate of capture silently pinned
  the readout at 100% instead of showing an impossible number.
* The denominator was the area draining to the single busiest accumulation cell, not
  the site area (fixed upstream, in the analysis worker).
* Per-feature inflow summed *cumulative* flow accumulation over every cell a line
  touched, and adjacent contour cells each carry nearly the whole hillside above them,
  so it over-counted by roughly (line length ÷ cell size).

``_apply_interception`` is gone with them. It existed to unpick that double-count by
subtracting upstream capture from downstream inflow, keyed on elevation alone — but
``cascade_overflow`` then *added* the upstream overflow back on top, so pass-through
water was counted twice regardless. With mutually exclusive direct catchments the
correction is not merely unnecessary, it would be a second error.

Pure and headless: operates on ``EarthworkStore`` objects (no QGIS), so it is unit-
testable under the ``tests/conftest.py`` QGIS mock.
"""

from dataclasses import dataclass, field

from .catchment import drain_down_hours
from .simulation import cascade_overflow, resolve_targets


def _round_or_none(value, places=1):
    return None if value is None else round(value, places)


@dataclass
class BalanceResult:
    """Result of a single-shot analytical water balance over the earthwork network."""
    capture_pct: float = 0.0            # (total runoff − site exit) / total runoff, %
    total_inflow_m3: float = 0.0        # total design-storm runoff (the score denominator)
    total_captured_m3: float = 0.0      # stored + infiltrated on-site
    site_exit_m3: float = 0.0           # runoff that leaves the site (measured, not residual)
    uncaptured_m3: float = 0.0          # of which: never reached any feature
    routed_exit_m3: float = 0.0         # of which: overflowed past the last feature
    total_capacity_m3: float = 0.0      # Σ storage capacity
    total_infiltration_m3: float = 0.0  # Σ infiltration credited as capture
    infiltration_buffer_m3: float = 0.0  # Σ soakage available but not relied upon
    counts_infiltration: bool = True     # was soakage credited to the score?
    total_cut_m3: float = 0.0
    total_fill_m3: float = 0.0
    terminal_deficit_m3: float = 0.0    # overflow leaving features with no downstream
    mass_balance_ok: bool = True        # captured + exit == runoff (within tolerance)
    routing_warnings: list = field(default_factory=list)
    per_feature: list = field(default_factory=list)


def run_water_balance(stores, duration_hr, total_runoff_m3=0.0,
                      uncaptured_m3=0.0, routing=None, count_infiltration=True):
    """Run the analytical balance and return a :class:`BalanceResult`.

    Parameters
    ----------
    stores : list of ``EarthworkStore`` with ``inflow_m3`` preloaded to each feature's
        **direct** catchment runoff (0 when no flow data yet).
    duration_hr : storm duration — drives the single-shot infiltration volume.
    total_runoff_m3 : total design-storm runoff over the whole site (the capture-%
        denominator). Pass 0 when no baseline exists yet, in which case the geometry
        metrics (capacity / cut / fill) are still valid and capture % is 0.
    uncaptured_m3 : runoff from cells that reach no earthwork — measured from the
        catchment label raster, including interior sinks.
    routing : ``RoutingResult`` from :func:`~terrainflow_assessment.modules.simulation.resolve_targets`;
        resolved here from the stores if omitted.
    count_infiltration : when False, features must **hold** their water — soakage is
        measured but not credited as capture. Sizing then rests on impounded volume
        alone and whatever the ground takes is spare capacity, which is the safer way
        to size against a steady-state infiltration rate that real soil rarely honours
        (and which this model applies with no saturation limit). The soakage is still
        reported per feature as ``infiltration_buffer_m3``.

    The capture percentage is deliberately **not clamped**. If it lands outside 0–100
    that is a mass-balance bug, and ``mass_balance_ok`` goes False so it is visible
    rather than hidden behind a ``min(100, …)``.
    """
    if routing is None:
        routing = resolve_targets(stores)

    direct_inflow = {s.id: s.inflow_m3 for s in stores}
    routed_exit = cascade_overflow(stores, time_hr=duration_hr, dt_hr=duration_hr,
                                   routing=routing,
                                   count_infiltration=count_infiltration)

    total_stored = sum(s.stored_m3 for s in stores)
    total_infiltration = sum(s.total_infiltration_m3 for s in stores)
    captured = total_stored + total_infiltration
    site_exit = uncaptured_m3 + routed_exit

    if total_runoff_m3 > 0:
        capture_pct = (total_runoff_m3 - site_exit) / total_runoff_m3 * 100.0
    else:
        capture_pct = 0.0

    # Water is conserved: everything that fell either stayed or left. A gap means the
    # inflow bookkeeping and the denominator disagree — surface it, never paper over it.
    tolerance = max(1.0, total_runoff_m3 * 0.01)
    mass_balance_ok = (
        total_runoff_m3 <= 0
        or abs((captured + site_exit) - total_runoff_m3) <= tolerance
    )

    per_feature = []
    terminal_deficit = 0.0
    for s in stores:
        fill_pct = s.fill_pct()
        direct = direct_inflow.get(s.id, 0.0)
        total_in = s.total_inflow_m3
        target_id = routing.edges.get(s.id)
        is_terminal = target_id is None
        if is_terminal:
            terminal_deficit += s.total_overflow_m3
        per_feature.append({
            "id": s.id,
            "name": s.name,
            "ew_type": s.ew_type,
            "direct_catchment_m2": round(s.direct_catchment_m2, 1),
            "direct_inflow_m3": round(direct, 1),
            "upstream_inflow_m3": round(max(0.0, total_in - direct), 1),
            "total_inflow_m3": round(total_in, 1),
            "stored_m3": round(s.stored_m3, 1),
            "infiltration_m3": round(s.total_infiltration_m3, 1),
            "infiltration_buffer_m3": round(s.total_infiltration_potential_m3, 1),
            "drain_hours": _round_or_none(
                drain_down_hours(s.stored_m3, s.area_m2, s.infiltration_rate_mm_hr)),
            "overflow_m3": round(s.total_overflow_m3, 1),
            "capacity_m3": round(s.capacity_m3, 1),
            "fill_pct": round(min(fill_pct, 100.0), 1),
            "overflowed": s.overflowed,
            "target_id": target_id,
            "is_user_link": bool(routing.is_user.get(s.id, False)),
            "is_terminal": is_terminal,
            # Centroid elevation, carried through for the report's cascade
            # diagram — which draws the chain down the page by height, so a
            # stand-in for an unknown elevation would put a feature on the wrong
            # contour rather than merely mis-ordering it. ``elevation_known``
            # travels with it for exactly that reason.
            "elevation": float(getattr(s, "elevation", 0.0) or 0.0),
            "elevation_known": bool(getattr(s, "elevation_known", False)),
        })

    return BalanceResult(
        capture_pct=round(capture_pct, 1),
        total_inflow_m3=round(total_runoff_m3, 1),
        total_captured_m3=round(captured, 1),
        site_exit_m3=round(site_exit, 1),
        uncaptured_m3=round(uncaptured_m3, 1),
        routed_exit_m3=round(routed_exit, 1),
        total_capacity_m3=round(sum(s.capacity_m3 for s in stores), 1),
        total_infiltration_m3=round(total_infiltration, 1),
        infiltration_buffer_m3=round(
            sum(s.total_infiltration_potential_m3 for s in stores), 1),
        counts_infiltration=count_infiltration,
        total_cut_m3=round(sum(s.cut_vol_m3 for s in stores), 1),
        total_fill_m3=round(sum(s.fill_vol_m3 for s in stores), 1),
        terminal_deficit_m3=round(terminal_deficit, 1),
        mass_balance_ok=mass_balance_ok,
        routing_warnings=list(routing.warnings),
        per_feature=per_feature,
    )


def area_subtotals(labels, domain_mask, area_masks, cell_area_m2, runoff_mm):
    """Break the site water balance down by area — one row per named area.

    A single site-wide capture figure hides where the problem is: 19% overall might be
    80% on the developed half and nothing on the rest, and those call for different
    work. This partitions the same cell-exact balance rather than computing a second,
    disagreeing one.

    Each row reports runoff **generated** in the area and where it ended up. A cell in
    one area may well drain to a feature in another, so "intercepted" means reaching
    some earthwork anywhere, not one inside these bounds. That keeps the rows a true
    partition — they sum to the site totals whenever the areas do — where attributing
    capture to the receiving area instead would double-count nothing but explain less.

    *labels* is the raster from :func:`~terrainflow_assessment.modules.flow_graph.
    label_direct_catchments`; *area_masks* is ``{name: bool array}``.

    Returns a list of dicts ordered as *area_masks* was given, each with ``name``,
    ``cells``, ``area_m2``, ``runoff_m3``, ``intercepted_m3``, ``exit_m3``,
    ``sink_m3``, ``unresolved_m3`` and ``capture_pct``.

    The four volume buckets are a **partition**: intercepted + exit + sink +
    unresolved is the whole of ``runoff_m3``, every time. ``unresolved`` is
    ``LABEL_UNRESOLVED`` — a cell trapped in a routing cycle, which a conditioned
    DEM should never produce and which used to fall into no bucket at all, so the
    rows quietly failed to add up.
    """
    import numpy as np

    from terrainflow_assessment.modules.flow_graph import LABEL_EXIT, LABEL_SINK

    labels = np.asarray(labels)
    domain = np.asarray(domain_mask, dtype=bool)
    depth_m = float(runoff_mm or 0.0) / 1000.0

    rows = []
    for name, mask in (area_masks or {}).items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != labels.shape:
            continue
        inside = mask & domain
        n = int(inside.sum())
        if n == 0:
            continue
        values = labels[inside]
        runoff = n * cell_area_m2 * depth_m
        exit_cells = int((values == LABEL_EXIT).sum())
        sink_cells = int((values == LABEL_SINK).sum())
        intercepted_cells = int((values >= 0).sum())
        # The fourth bucket, as the remainder rather than as `== LABEL_UNRESOLVED`:
        # taken that way the four are a partition by construction, and it catches
        # LABEL_NONE too — a cell inside the area mask but outside the labelling's
        # domain, which is the other way the rows failed to add up. Without it the
        # three above sum to less than `cells` and a reader totalling them finds a
        # gap with no name on it. (Not, as was once claimed, an overstated capture
        # figure: an unresolved cell is in the denominator and out of the
        # numerator, so capture reads low, not high.)
        unresolved_cells = n - exit_cells - sink_cells - intercepted_cells
        rows.append({
            "name": name,
            "cells": n,
            "area_m2": n * cell_area_m2,
            "runoff_m3": runoff,
            "intercepted_m3": intercepted_cells * cell_area_m2 * depth_m,
            "exit_m3": exit_cells * cell_area_m2 * depth_m,
            "sink_m3": sink_cells * cell_area_m2 * depth_m,
            "unresolved_m3": unresolved_cells * cell_area_m2 * depth_m,
            "capture_pct": (intercepted_cells / n * 100.0) if n else 0.0,
        })
    return rows
