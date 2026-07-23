"""
water_balance.py — design-tier analytical earthwork assessment.

The fast, no-burn half of the two-tier method (*"analytical is where you design;
burning is where you verify"*). Given the current earthwork network it computes a
lumped, event-total **routed water balance** — an *estimate*, not hydrodynamics —
suitable to recompute live as features are added/moved/edited.

Model
-----
1. **Inflow** per feature is its catchment runoff (accumulation × runoff depth),
   preloaded onto each ``EarthworkStore.inflow_m3`` by the caller.
2. **Interception** (``_apply_interception``): stacked features share upstream
   catchment, so each feature's inflow is reduced by what upstream features capture —
   preventing double-counting and making upslope interception visible (add an upslope
   swale and the downslope one's inflow drops). Mirrors the correction in
   ``simulation._run_simulation``.
3. **Routing + capture**: one pass of ``simulation.cascade_overflow`` (reused, single
   shot with ``dt_hr = duration``) applies infiltration, fills each feature to
   capacity, and cascades overflow to the next downslope / linked feature.
4. **Score**: system **capture %** = (stored + infiltrated) ÷ total design-storm
   runoff — "how much of the storm does this design hold?".

Pure and headless: operates on ``EarthworkStore`` objects (no QGIS), so it is unit-
testable under the ``tests/conftest.py`` QGIS mock.
"""

from dataclasses import dataclass, field

from .simulation import cascade_overflow


@dataclass
class BalanceResult:
    """Result of a single-shot analytical water balance over the earthwork network."""
    capture_pct: float = 0.0            # (stored + infiltrated) / total runoff, %
    total_inflow_m3: float = 0.0        # total design-storm runoff (the score denominator)
    total_captured_m3: float = 0.0      # stored + infiltrated on-site
    site_exit_m3: float = 0.0           # runoff that leaves the site
    total_capacity_m3: float = 0.0      # Σ storage capacity
    total_infiltration_m3: float = 0.0  # Σ infiltration over the event
    total_cut_m3: float = 0.0
    total_fill_m3: float = 0.0
    per_feature: list = field(default_factory=list)  # {name, inflow_m3, stored_m3, fill_pct, overflowed}


def _apply_interception(stores):
    """Reduce each store's (cumulative-catchment) inflow by what upstream stores capture.

    Processes highest-elevation first, tracking each store's capture; a downstream
    store's inflow is cut by the sum of upstream captures so the shared upslope
    catchment is not counted twice. Mirrors ``simulation._run_simulation`` (the
    stacked-feature interception fix). Mutates ``store.inflow_m3`` in place.
    """
    ordered = sorted(stores, key=lambda s: s.elevation, reverse=True)
    captured = {}
    for store in ordered:
        upstream_captured = sum(
            captured.get(s.name, 0.0)
            for s in stores
            if s is not store and s.elevation > store.elevation
        )
        actual = max(0.0, store.inflow_m3 - upstream_captured)
        store.inflow_m3 = actual
        available = max(0.0, store.capacity_m3 - store.stored_m3)
        captured[store.name] = min(actual, available)


def run_water_balance(stores, duration_hr, total_runoff_m3=0.0):
    """Run the analytical balance and return a :class:`BalanceResult`.

    ``stores`` — ``EarthworkStore`` list with ``inflow_m3`` preloaded to each feature's
    catchment runoff (0 when no flow data yet). ``duration_hr`` — storm duration (drives
    the single-shot infiltration volume). ``total_runoff_m3`` — total design-storm runoff
    for the site (the capture-% denominator); pass 0 when no baseline exists yet, in which
    case the geometry metrics (capacity / cut / fill) are still valid and capture % is 0.
    """
    _apply_interception(stores)
    # The interception-corrected inflow is what actually reaches each feature.
    inflow_by_name = {s.name: s.inflow_m3 for s in stores}

    cascade_overflow(stores, time_hr=duration_hr, dt_hr=duration_hr)

    total_stored = sum(s.stored_m3 for s in stores)
    total_infiltration = sum(s.total_infiltration_m3 for s in stores)
    captured = total_stored + total_infiltration

    if total_runoff_m3 > 0:
        capture_pct = min(100.0, captured / total_runoff_m3 * 100.0)
        site_exit = max(0.0, total_runoff_m3 - captured)
    else:
        capture_pct = 0.0
        site_exit = 0.0

    per_feature = []
    for s in stores:
        fill_pct = (s.stored_m3 / s.capacity_m3 * 100.0) if s.capacity_m3 > 0 else 0.0
        per_feature.append({
            "name": s.name,
            "inflow_m3": round(inflow_by_name.get(s.name, 0.0), 1),
            "stored_m3": round(s.stored_m3, 1),
            "capacity_m3": round(s.capacity_m3, 1),
            "fill_pct": round(min(fill_pct, 100.0), 1),
            "overflowed": s.overflowed,
        })

    return BalanceResult(
        capture_pct=round(capture_pct, 1),
        total_inflow_m3=round(total_runoff_m3, 1),
        total_captured_m3=round(captured, 1),
        site_exit_m3=round(site_exit, 1),
        total_capacity_m3=round(sum(s.capacity_m3 for s in stores), 1),
        total_infiltration_m3=round(total_infiltration, 1),
        total_cut_m3=round(sum(s.cut_vol_m3 for s in stores), 1),
        total_fill_m3=round(sum(s.fill_vol_m3 for s in stores), 1),
        per_feature=per_feature,
    )
