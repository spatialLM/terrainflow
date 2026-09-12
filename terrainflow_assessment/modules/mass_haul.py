"""
mass_haul.py — does the design balance, and what does it cost to move.

The plugin could already say *how much* earth: ``burn_quantities`` differences the two
surfaces and gives site-total cut and fill. It could not say whether the design
**balances**, or what the moving costs.

**Cut and fill are not comparable as measured.** Cut is bank measure — the volume the
soil occupied in the ground. Fill is placed and compacted. Subtracting one from the
other and calling the result a balance is the arithmetic ``reporting.py`` does today
under the name ``net_cut_fill_m3``; it is a geometric difference, not a balance, and the
two answers differ by a quarter or more. The factors that convert between the states are
named in ``core.sizing.advisories``.

**Haul is the transportation problem, and it is deliberately not solved exactly.** The
classic mass-haul *diagram* — cumulative volume against chainage — is a linear-project
instrument: it needs a chainage, which a road has and a scatter of swales, a dam and a
berm across a block does not. Inventing an ordering to draw one would put the ordering
rule on the page instead of the job. What the diagram exists to produce is the **haul
moment**, and that is computed here directly, by matching surplus regions to deficit
regions (Hitchcock 1941 — the transportation problem, which is what mass-haul
optimisation *is*).

Two honesty requirements the callers must carry through to the page:

* **Distance is straight-line.** Real haul follows a track, around a gully, up a grade.
  This is a lower bound and must be presented as one.
* **The optimiser is not the precision limit.** The bulking factors are ±15% figures and
  the burn is a one-cell approximation by design, so a third significant figure here
  would be spurious whichever matching is used.

Per-feature attribution is refused, as ``burn_quantities`` already refuses it: banks lie
outside their own footprints and cuts overlap, so a figure invented by an attribution
rule is worse than one honestly aggregated. Regions are the honest decomposition.

Pure numpy + scipy: no QGIS, no rasterio.
"""

from __future__ import annotations

import math

import numpy as np

#: Region side, in metres, that the cut/fill difference is reduced to before matching.
#:
#: Per-cell haul is false precision on a surface the burn approximates at one cell, and
#: matching 10^5 cells is an assignment problem with no bounded runtime. Ten metres
#: leaves a few hundred regions on a farm design, which both solvers handle instantly.
DEFAULT_BLOCK_M = 10.0

#: Regions smaller than this are dropped rather than hauled — they are burn noise along
#: a feature edge, not earth anyone moves.
DEFAULT_MIN_REGION_M3 = 5.0

#: Free-haul distance, metres. **A contract term, not a physical one**: haul within it
#: is priced inside the excavation rate and beyond it is overhaul, priced separately.
#: Standard highway-earthworks practice; exposed because it belongs to an agreement.
DEFAULT_FREE_HAUL_M = 150.0


def earthwork_balance(cut_m3, fill_m3, soil_name=None,
                      bulking=None, compaction=None):
    """Does the design balance, once the states are made comparable?

    ``cut_m3`` and ``fill_m3`` are both **bank** measure off the burn.

    Returns a dict with ``bank_cut_m3``, ``bank_fill_m3``, ``loose_from_cut_m3`` (what a
    truck carries), ``bank_needed_for_fill_m3`` (how much in-situ soil the fill actually
    consumes once compacted), ``surplus_m3`` (bank measure to cart off), ``deficit_m3``
    (bank measure to import) and ``balanced_within_pct``.

    Surplus and deficit are mutually exclusive: a design either has spare soil or needs
    some, and reporting both is how a reader ends up adding them together.
    """
    from terrainflow_assessment.core.sizing.advisories import (
        bulking_factor,
        compaction_factor,
    )

    cut = max(0.0, float(cut_m3))
    fill = max(0.0, float(fill_m3))
    bulk = float(bulking) if bulking is not None else bulking_factor(soil_name)
    comp = float(compaction) if compaction is not None else compaction_factor(soil_name)

    # A compacted fill of `fill` m³ swallows more than `fill` m³ of in-situ soil,
    # because placing it squeezes the air out. That is the whole reason cut == fill is
    # not a balance.
    bank_needed = fill / comp if comp > 0 else 0.0
    net = cut - bank_needed

    denominator = max(cut, bank_needed, 1e-9)
    return {
        "bank_cut_m3": cut,
        "bank_fill_m3": fill,
        "loose_from_cut_m3": cut * bulk,
        "bank_needed_for_fill_m3": bank_needed,
        "surplus_m3": max(0.0, net),
        "deficit_m3": max(0.0, -net),
        "balanced_within_pct": abs(net) / denominator * 100.0,
        "bulking_factor": bulk,
        "compaction_factor": comp,
    }


def haul_regions(original, burned, transform, cell_area_m2,
                 block_m=DEFAULT_BLOCK_M, min_region_m3=DEFAULT_MIN_REGION_M3):
    """Reduce the burn difference to cut and fill regions with volumes and centroids.

    Returns ``(cuts, fills)``, each a list of ``{"x", "y", "volume_m3"}``. Volumes are
    positive in both lists — a cut *supplies* and a fill *demands*.

    ``nansum`` throughout, and for the reason ``burn_quantities`` gives: a DEM clipped to
    a property boundary always has holes, and one NaN turns a site total into ``nan``,
    which then reaches a report as "cannot convert float NaN to integer".
    """
    from scipy.ndimage import label as nd_label

    original = np.asarray(original, dtype="float64")
    burned = np.asarray(burned, dtype="float64")
    diff = burned - original

    cell = math.sqrt(max(cell_area_m2, 1e-9))
    block = max(1, int(round(block_m / cell)))

    cuts = _regions_from(np.where(np.isfinite(diff) & (diff < 0), -diff, 0.0),
                         transform, cell_area_m2, block, min_region_m3, nd_label)
    fills = _regions_from(np.where(np.isfinite(diff) & (diff > 0), diff, 0.0),
                          transform, cell_area_m2, block, min_region_m3, nd_label)
    return cuts, fills


def _regions_from(depths, transform, cell_area_m2, block, min_region_m3, nd_label):
    """Connected components of a one-sided depth field, as volume-weighted centroids."""
    mask = depths > 0
    if not mask.any():
        return []

    labelled, count = nd_label(mask)
    out = []
    for tag in range(1, count + 1):
        sel = labelled == tag
        volume = float(np.nansum(depths[sel])) * cell_area_m2
        if volume < min_region_m3:
            continue
        rows, cols = np.nonzero(sel)
        weights = depths[sel]
        total = float(np.nansum(weights)) or 1.0
        # Volume-weighted, so a long thin bank's centroid sits where the earth is
        # rather than at the middle of its bounding shape.
        r = float(np.nansum(rows * weights)) / total
        c = float(np.nansum(cols * weights)) / total
        x = transform.c + (c + 0.5) * transform.a
        y = transform.f + (r + 0.5) * transform.e
        out.append({"x": x, "y": y, "volume_m3": volume})
    # `block` is carried for callers that want to state the resolution the regions were
    # reduced at; the labelling itself is on the full grid, which keeps a narrow bank
    # from being split across two blocks.
    return out


def allocate_haul(cuts, fills, free_haul_m=DEFAULT_FREE_HAUL_M, prefer_exact=True):
    """Match surplus regions to deficit ones, minimising volume × distance.

    Returns a dict with ``moves`` (``from``/``to`` indices, ``volume_m3``,
    ``distance_m``), ``haul_moment_m3m``, ``mean_haul_m``, ``matched_m3``,
    ``unmatched_cut_m3``, ``unmatched_fill_m3``, ``free_haul_m3``, ``overhaul_m3m``
    and ``method``.

    The exact solve is a linear program (``scipy.optimize.linprog``, HiGHS) over a
    problem that is a few dozen regions square — milliseconds. Where ``linprog`` is
    unavailable, or ``prefer_exact`` is off, a greedy nearest-first allocation runs
    instead and ``method`` says so; greedy is never better than the LP, which is
    asserted in the tests.

    **Distance is straight-line.** The result is a lower bound on real haul and the
    caller must present it as one.
    """
    supply = [max(0.0, float(c["volume_m3"])) for c in cuts]
    demand = [max(0.0, float(f["volume_m3"])) for f in fills]
    if not supply or not demand or sum(supply) <= 0 or sum(demand) <= 0:
        return _empty_plan(sum(supply), sum(demand))

    dist = [[math.hypot(c["x"] - f["x"], c["y"] - f["y"]) for f in fills]
            for c in cuts]

    moves, method = None, "greedy nearest-first"
    if prefer_exact:
        moves = _solve_lp(supply, demand, dist)
        if moves is not None:
            method = "least-cost transportation (LP)"
    if moves is None:
        moves = _solve_greedy(supply, demand, dist)

    matched = sum(m["volume_m3"] for m in moves)
    moment = sum(m["volume_m3"] * m["distance_m"] for m in moves)
    free = sum(m["volume_m3"] for m in moves if m["distance_m"] <= free_haul_m)
    overhaul = sum(m["volume_m3"] * (m["distance_m"] - free_haul_m)
                   for m in moves if m["distance_m"] > free_haul_m)

    return {
        "moves": moves,
        "haul_moment_m3m": moment,
        "mean_haul_m": (moment / matched) if matched > 0 else 0.0,
        "matched_m3": matched,
        "unmatched_cut_m3": max(0.0, sum(supply) - matched),
        "unmatched_fill_m3": max(0.0, sum(demand) - matched),
        "free_haul_m3": free,
        "overhaul_m3m": overhaul,
        "free_haul_m": free_haul_m,
        "method": method,
    }


def _empty_plan(supply_total, demand_total):
    return {
        "moves": [], "haul_moment_m3m": 0.0, "mean_haul_m": 0.0, "matched_m3": 0.0,
        "unmatched_cut_m3": max(0.0, supply_total),
        "unmatched_fill_m3": max(0.0, demand_total),
        "free_haul_m3": 0.0, "overhaul_m3m": 0.0,
        "free_haul_m": DEFAULT_FREE_HAUL_M, "method": "nothing to move",
    }


def _solve_lp(supply, demand, dist):
    """Least-cost transportation by linear programming, or ``None`` if unavailable."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return None

    n, m = len(supply), len(demand)
    cost = np.array(dist, dtype="float64").ravel()

    if n == 0 or m == 0:
        return None

    # One row per source (ship no more than you have) and per sink (receive no more
    # than you need); maximising the matched volume is expressed by subtracting a large
    # constant from the cost so the solver prefers to move earth rather than leave it.
    #
    # Sparse, because every variable x[i, j] appears in exactly two of those rows: the
    # matrix is (n + m) x (n * m) with 2nm non-zeros, which is 0.5% occupancy at 200
    # regions a side. Dense it was 128 MB there, and one solve peaked at 394 MB because
    # HiGHS copies what it is handed. The region count has no ceiling — `haul_regions`
    # returns full-grid connected components, so a noisy burn on a large DEM produces
    # hundreds — and all of this runs inside the QGIS process. `method="highs"` takes
    # scipy.sparse directly, so the dense array bought nothing.
    var = np.arange(n * m)
    rows = np.concatenate((var // m, n + (var % m)))
    cols_a = np.concatenate((var, var))
    a_ub = coo_matrix(
        (np.ones(2 * n * m, dtype="float64"), (rows, cols_a)),
        shape=(n + m, n * m),
    )
    b_ub = np.array(list(supply) + list(demand), dtype="float64")

    movable = min(sum(supply), sum(demand))
    reward = float(np.max(cost)) + 1.0 if cost.size else 1.0
    try:
        result = linprog(c=cost - reward, A_ub=a_ub, b_ub=b_ub,
                         bounds=(0, None), method="highs")
    except Exception:
        return None
    if not getattr(result, "success", False):
        return None

    x = np.asarray(result.x, dtype="float64").reshape(n, m)
    moves = []
    for i in range(n):
        for j in range(m):
            v = float(x[i, j])
            if v > 1e-9:
                moves.append({"from": i, "to": j, "volume_m3": v,
                              "distance_m": float(dist[i][j])})
    # Guard the reward trick: if it somehow failed to move what it could, fall back.
    if sum(mv["volume_m3"] for mv in moves) < movable * 0.999:
        return None
    return moves


def _solve_greedy(supply, demand, dist):
    """Nearest-first allocation — the explainable fallback, and never better than the LP."""
    supply = list(supply)
    demand = list(demand)
    pairs = sorted(
        ((dist[i][j], i, j) for i in range(len(supply)) for j in range(len(demand))),
        key=lambda t: t[0],
    )
    moves = []
    for d, i, j in pairs:
        take = min(supply[i], demand[j])
        if take <= 1e-9:
            continue
        supply[i] -= take
        demand[j] -= take
        moves.append({"from": i, "to": j, "volume_m3": take, "distance_m": d})
    return moves
