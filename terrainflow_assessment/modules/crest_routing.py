"""
crest_routing.py — spread a pond's overflow along the crest that holds it back.

A pool fills to its rim and spills. In reality the water surface stays flat as it rises, so
**every metre of rim standing at the pour level passes water at once**, at a uniform
discharge per metre — a broad-crested weir under a uniform head. The flow rasters do not
show that, and not for the reason the field log used to give.

This is a property of **ponds, not of dams.** A keyed swale's companion berm impounds a
dead-flat pool exactly as a dam does (Round 9: Swale 5 stands at 69.60 m across its whole
pond), a basin fills against its rim, and a natural hollow behaves no differently. Nothing
below tests what *type* of feature made the hollow.

What is actually wrong (Round 14, measured on Quail Island's Dam 15)
--------------------------------------------------------------------
``resolve_flats`` (Barnes et al. 2015, in pysheds) resolves a flat with a multi-source
distance transform seeded from **every** low-edge cell, so it is already multi-outlet and
flow really does leave a pool in more than one place — **23 distinct cells** for Dam 15, not
the "single cell" the log claimed. But it gives each flat cell a route to its **nearest** way
out, and that is a different thing from a weir: 87.1% of the flux crossed one crest cell
while 16 of 24 carried nothing.

**A reservoir is a mixing node; a distance transform is a routing node.** Dam 15 receives
essentially its whole catchment as one channel, and nearest-outlet routing hands that channel
to whichever crest cell is closest to where it arrives.

Why this is not fixed in the flow directions
--------------------------------------------
It was tried, and it **cannot** work. Flow routing carries one pointer per cell, so a cell's
whole load goes to exactly one destination; Dam 15's inflow arrives as a single channel of
~36,000 of the pool's 42,000 units, and no partition of *cells* can divide the load of *one*
cell. Balancing the partition evened the pool-facing row (20.9x -> 0.5x even) and still left
84% crossing at one place. One pointer per cell and a weir are incompatible — so
``resolve_terminals``' pointer doubling, ``feature_inflow_m3``'s cell counts and
``longest_flow_path``'s single path keep the pointers they have, untouched.

The split belongs to **accumulation**, which is a flux field and carries no such constraint.

How it is done
--------------
Each pond is **contracted to a single node**: everything arriving is pooled, and leaves
divided equally between the cells that discharge from it. The caller's own routing engine
runs every accumulation, so D-infinity keeps splitting fractionally everywhere outside a
pond, and the whole thing rests on one property of accumulation: **it is linear in its
weights**, so the fields of separate passes simply add.

1. **Absorb.** Point every cell of every pond at itself, and accumulate. Each pond now holds
   exactly what reaches it, and nothing threads through it — which is also the truthful
   picture of a pond, since a channel does not survive crossing one.
2. **Re-emit, then measure what came back.** Inject ``T / n`` at each cell a pond discharges
   *into* and accumulate that injection alone. Sum the field into the running total, then
   read off how much of it landed in a pond again — that is what those ponds have now
   received, and it is emitted on the next pass. Repeat until nothing is left holding.

Why the second pass is not enough (Round 15, the defect this module was parked for)
-----------------------------------------------------------------------------------
The parked version resolved the pond-to-pond hand-off analytically, from a graph built out of
``rid[tgt[exit]]`` — an edge only where an exit's *immediate* D8 neighbour is a pond cell.
Measured on Quail Island, that graph holds **1 of 5** real edges built-gated and **12 of 221**
ungated. The overwhelming case is a pond emitting onto open ground, the water running tens of
cells downhill, and *then* falling into the next hollow, where it was absorbed against a total
already fixed and simply stopped. **16,006.9 cell-units, 1.824% of the site**, which read as
-15.0% at the site's busiest cell.

The direct hand-off it *did* see was never the problem: the unit that dies on the downstream
pond's cell is exactly cancelled by the share added to that pond's total, so it is
mass-neutral by identity. **100% of the loss was the indirect path, 0% the direct one.**

So no pond graph is built here at all. The transfer between ponds is not modelled, it is
**measured**, by the same engine that carries the final answer — which is why an indirect
hand-off, a chain of them, or a pond that receives its own emission back all need no special
case. Stated compactly: the parked version was the Neumann series truncated after one term,
computed on the wrong operator.

Conservation, and why truncating is safe
----------------------------------------
A pond emits exactly what it has received, so at every point in the loop

    (flux reaching a real terminal) + residual + stranded == total weight

*exactly* — not only at convergence. Every quantity in the loop is non-negative (``held`` is a
bincount of an accumulation of non-negative weights), so the iteration is **monotone from
below**: it can only ever under-emit, never over-emit, never drive a cell negative. Stopping
early is therefore one-signed and bounded by a number this module returns, and the map can
only ever show less water downstream than there is — the right direction for a tool that
sizes spillways. Measured across every scope and cap tried: conservation error +0.0000, and
zero negative cells.

It terminates because emission edges strictly descend the conditioned surface and D-infinity
never ascends it (0 of 877,364 live edges ascend). The pond graph is therefore acyclic, the
transfer operator nilpotent — measured spectral radius 0.000000 over all 195 ponds — and the
residual reaches exact zero in ``cascade depth + 1`` passes.

Pure numpy + scipy: no QGIS, no pysheds, no rasterio.
"""

from __future__ import annotations

import numpy as np

# A pond larger than this keeps the default routing. Only ever catches something
# pathological (a burn that dammed a whole tile); it exists so the analysis thread cannot
# stall with no way to report why.
MAX_REGION_CELLS = 2_000_000

# Below a 3x3 pool there is nothing to spread along: the "crest" is one to three cells wide,
# and contracting the hollow divides a channel across a puddle's rim. Measured on Quail
# Island the median pool is **3 cells holding ~17 mm** — ``fill_depressions`` noise rather
# than a reservoir — and 125 of 195 pools are 4 cells or fewer. Dropping them costs 0.03
# percentage points at the site's busiest cell and saves about five accumulation passes,
# because those puddles are what makes the pond cascade deep. This is a raster-resolution
# guard, so it is counted in **cells, not m2**. Raising it to 50 halves the pass count again
# for 2% of the ponded area; the smallest pond anyone built on that design is 561 cells, so
# there is a wide margin before a designed feature is at risk.
MIN_POND_CELLS = 9

# Hard stop on the re-emission loop. Not needed for correctness — the loop converges to an
# exact zero residual on any real surface — but a pathological DEM must not be able to stall
# the analysis thread with no way to report why. When it fires, the water still held is
# returned as ``residual`` rather than quietly dropped.
MAX_SPREAD_PASSES = 64

_STRUCT = np.ones((3, 3), dtype=bool)


class Impoundment:
    """One pond, and the level rim it spills over.

    ``region`` and ``pool`` are deliberately separate. The region is the routing extent —
    pool *plus* the level band through the wall — and is what absorbs. The pool is the
    standing water, and is the only part of it that is a *water body*: painting a dam wall
    as water would be wrong, and 9,325 of the 9,380 cells whose accumulation the contraction
    changes are pool, not rim.
    """

    __slots__ = ("region", "pour_level_m", "n_cells", "pool")

    def __init__(self, region, pour_level_m, pool=None):
        self.region = region                      # bool (rows, cols) — pool + level rim
        self.pool = region if pool is None else pool   # bool — the standing water only
        self.pour_level_m = float(pour_level_m)
        self.n_cells = int(region.sum())


class CrestPlan:
    """The geometry of the contraction: which cells absorb, and where each pond discharges.

    Built once from the pointers and reused for every accumulation field the caller wants
    spread — the plain one and the runoff-weighted one share it, because the ponds and their
    exits do not depend on how much water is falling.
    """

    __slots__ = ("absorb", "pools", "rid", "exits", "targets", "shape", "cascade_depth",
                 "skipped")

    def __init__(self, absorb, pools, rid, exits, targets, shape, cascade_depth, skipped):
        self.absorb = absorb              # bool (rows, cols) — cells to make self-draining
        self.pools = pools                # bool (rows, cols) — the standing water only
        self.rid = rid                    # int32 flat — pond index per cell, 0 = none
        self.exits = exits                # list of flat index arrays, one per pond
        self.targets = targets            # list of flat index arrays, aligned with exits
        self.shape = shape
        self.cascade_depth = int(cascade_depth)
        self.skipped = skipped            # list[str]

    @property
    def ponds(self) -> int:
        return len(self.exits)

    @property
    def outlet_cells(self) -> int:
        return int(sum(int(e.size) for e in self.exits))

    @property
    def changed(self) -> bool:
        return bool(self.exits) and self.outlet_cells > 0

    def default_passes(self) -> int:
        """How many accumulations to allow before giving up and reporting the residual.

        The budget has to scale with the cascade, because the decay is **linear, not
        geometric** — each pass moves water one link down the chain. A synthetic 20-dam
        terrace (a keyline sequence, which is exactly what this plugin is for) still holds
        **61.1%** of its water after six passes, so a small fixed cap is not an option.

        Generous on purpose. The cascade probe reads the **D8** graph while the water moves
        by D-infinity, which fans into ponds a single-pointer chain never enters, so the
        probe under-reads: Quail Island measures a depth of **6** and needs **8** emission
        rounds. Doubling it and flooring at 16 costs nothing when the loop converges — it
        stops on an empty residual, not on the budget — and the cap only ever bites on a
        surface that would otherwise stall the analysis thread.
        """
        return int(min(MAX_SPREAD_PASSES, max(16, 2 * self.cascade_depth + 4)))


class CrestSpread:
    """What :func:`spread_crests` worked out, and what it could not place."""

    __slots__ = ("accumulation", "outlets", "pond_flow", "ponds", "outlet_cells", "passes",
                 "residual", "skipped")

    def __init__(self, accumulation, outlets, pond_flow, ponds, outlet_cells, passes,
                 residual, skipped):
        self.accumulation = accumulation    # float64 (rows, cols) — the spread field
        self.outlets = outlets              # float64 (rows, cols) — each crest cell's share
        self.pond_flow = pond_flow          # float64 (rows, cols) — see below
        self.ponds = ponds                  # int
        self.outlet_cells = outlet_cells    # int
        self.passes = passes                # int — accumulations run, including the absorb
        self.residual = residual            # float — still held when the loop stopped
        self.skipped = skipped              # list[str]

    @property
    def changed(self) -> bool:
        return self.ponds > 0

    @property
    def pond_mask(self):
        """The standing water — every cell of every contracted pool."""
        return self.pond_flow > 0


def _level_rim(filled, pool, pour, tol):
    """Cells at the pour level reachable from *pool* — the pond's own rim and wall.

    The pool-facing row of a wall is not the crest. A 2 m wall on a 1 m grid is two cells
    thick and only the far row has anywhere to drain to: on Dam 15 the pool-facing row is 24
    cells of which **8** can pass water, while the full level band is **65 cells of which 49
    can**. Growing the rim is what makes an even spread reachable at all.

    (This is a routing region, not a length. ``reporting.overtopping_spill`` deliberately
    reports the pool-facing row instead, because that band is length x *thickness* and using
    it as a length would shrink ``q = Q/L`` in the unsafe direction.)

    Worked inside a window around the pool rather than over the whole grid. The obvious
    version — dilate, intersect with the pour level, repeat until it stops growing — is
    correct but runs a full-grid dilation per iteration per pond, and on Quail Island's 195
    pools (median 3 cells) that measured **9.24 s**, three times every accumulation pass in
    this module put together. Confined to a padded bounding box, doubling the pad while the
    rim still runs off the window edge, it is **0.73 s — 13x faster and bit-identical**,
    verified 195 of 195 regions and 637 of 637 rim cells against the loop it replaced.
    """
    from scipy.ndimage import binary_dilation

    rows, cols = filled.shape
    rr, cc = np.nonzero(pool)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1

    pad = 4
    while True:
        a0, a1 = max(0, r0 - pad), min(rows, r1 + pad)
        b0, b1 = max(0, c0 - pad), min(cols, c1 + pad)
        whole = (a0 == 0 and b0 == 0 and a1 == rows and b1 == cols)

        sub_pool = pool[a0:a1, b0:b1]
        at_pour = np.abs(filled[a0:a1, b0:b1] - pour) <= tol
        grown = binary_dilation(sub_pool, structure=_STRUCT,
                                mask=at_pour | sub_pool, iterations=-1)
        rim = grown & ~sub_pool

        # A rim still touching an edge we cropped may continue past it.
        spills_out = (
            (a0 > 0 and rim[0].any()) or (a1 < rows and rim[-1].any())
            or (b0 > 0 and rim[:, 0].any()) or (b1 < cols and rim[:, -1].any())
        )
        if whole or not spills_out:
            out = np.zeros_like(pool)
            out[a0:a1, b0:b1] = rim
            return out
        pad *= 2


def find_impoundments(filled, ground, built=None, min_depth=1e-3, tol=1e-6,
                      min_cells=MIN_POND_CELLS, max_cells=MAX_REGION_CELLS):
    """Every pond on the surface, with the level rim it spills over.

    Parameters
    ----------
    filled : 2-D array — the depression-filled surface (pond tops are level here).
    ground : 2-D array — the same surface **before** depressions were filled. Take the copy
        *before* the call: pysheds' ``fill_depressions`` writes into its input buffer and
        returns it, so ``filled - pit_filled`` afterwards is identically zero.
    built : 2-D bool array or None — when given, only ponds touching raised ground are
        returned. Left as ``None`` a natural hollow is treated the same as an impounded one,
        which is the honest default and the one that ships: a lake spills over its saddle by
        the same physics, and correcting only the design tier would leave baseline and design
        on different hydrology, so a before/after comparison would partly measure the change
        in method.
    min_cells : pools smaller than this keep the default routing — see
        :data:`MIN_POND_CELLS`. Measured on the **pool**, not on pool-plus-rim, because it is
        the standing water that decides whether this is a reservoir or a rounding error.

    Returns ``(impoundments, skipped)``.
    """
    from scipy.ndimage import label

    filled = np.asarray(filled, dtype="float64")
    ground = np.asarray(ground, dtype="float64")
    out, skipped = [], []

    wet = (filled - ground) > min_depth
    if not wet.any():
        return out, skipped
    if built is not None:
        built = np.asarray(built, dtype=bool)
        if built.shape != filled.shape:
            built = None

    # 8-connected because that is how flow moves. ``reporting.overtopping_spill`` labels with
    # scipy's 4-connected default, which is right for measuring a pool's volume and wrong for
    # deciding which cells route together — a wall on a grid diagonal splits a pond into two
    # 4-connected pieces that water crosses freely.
    labels, n_pools = label(wet, structure=_STRUCT)
    pool_sizes = np.bincount(labels.ravel(), minlength=n_pools + 1)

    for pid in range(1, n_pools + 1):
        if pool_sizes[pid] < min_cells:
            continue
        pool = labels == pid
        # The *lowest* level the pool stands at, not the highest. Two depressions
        # that touch on a diagonal are one 8-connected component here while
        # remaining two ponds on a filled surface — each flat at its own pour
        # level. Taking the maximum sizes `_level_rim` for a level the lower
        # sub-pool never reaches, so its crest band is drawn across ground the
        # water does not get to and the spread is shed over cells that stay dry.
        # The minimum is the level the merged component is certainly at, which is
        # the conservative reading and the one the lower pond actually spills at.
        levels = filled[pool]
        pour = float(levels.min())
        spread = float(levels.max()) - pour
        if spread > tol:
            skipped.append(
                f"a {int(pool.sum()):,}-cell pond spans {spread:.2f} m of pour "
                f"level — treated as one pond at {pour:.2f} m, its lower level")
        region = pool | _level_rim(filled, pool, pour, tol)
        imp = Impoundment(region, pour, pool=pool)
        if built is not None and not (built & region).any():
            continue
        if imp.n_cells > max_cells:
            skipped.append(
                f"a {imp.n_cells:,}-cell pond at {pour:.2f} m keeps the default routing "
                f"(over the {max_cells:,}-cell spreading cap)")
            continue
        out.append(imp)

    return out, skipped


def plan_crest_absorption(impoundments, next_flat, is_sink, shape, skipped=None):
    """Which cells absorb, which discharge, and how deep the pond cascade runs.

    ``next_flat``/``is_sink`` are the steepest-descent pointers over the conditioned surface
    (``flow_graph.d8_from_dem``); they are read, never written. A cell discharges when its
    pointer leaves its own pond — which is a statement about the terrain, not about how much
    water is on it, so this plan is built once and reused for every field the caller spreads.

    **A pond with nothing leaving it keeps the default routing.** Contracting one would
    absorb its whole catchment and hand back water nobody could place; leaving it alone hands
    it instead to the mechanism that already exists for water with nowhere to go, where
    ``d8_from_dem`` calls it a sink and ``FlowAnalysis.unrouted_flow`` reports it. It happens
    when the pour-level band reaches every remaining cell — a bowl whose rim is the edge of
    the tile — and the alternative is a second way of saying the same thing, less well.
    """
    rows, cols = shape
    n = rows * cols
    absorb = np.zeros(shape, dtype=bool)
    pools = np.zeros(shape, dtype=bool)
    rid = np.zeros(n, dtype=np.int32)
    skipped = list(skipped or [])
    if not impoundments:
        return CrestPlan(absorb, pools, rid, [], [], shape, 0, skipped)

    idx = np.arange(n, dtype=np.int64)
    sink = np.asarray(is_sink, dtype=bool).ravel()
    tgt = np.where(sink, idx, np.asarray(next_flat).astype(np.int64).ravel())

    for i, imp in enumerate(impoundments, start=1):
        rid[imp.region.ravel()] = i
        absorb |= imp.region

    # Whether a cell discharges does not depend on which *other* ponds survive the next
    # step, so this is worked out once and then filtered.
    found = []
    for i, imp in enumerate(impoundments, start=1):
        cells = np.flatnonzero((rid == i) & (rid[tgt] != i) & (tgt != idx))
        if cells.size == 0:
            skipped.append(
                f"a {imp.n_cells:,}-cell pond at {imp.pour_level_m:.2f} m keeps the default "
                f"routing (its level band reaches every cell around it, so nothing "
                f"discharges)")
            continue
        found.append((imp, cells))

    # Rebuilt rather than patched, so two ponds whose level bands overlap cannot leave a
    # hole in ``absorb`` when one of them drops out. Ids must also stay contiguous —
    # ``np.bincount`` indexes by them.
    rid[:] = 0
    absorb[:] = False
    exits, targets = [], []
    for i, (imp, cells) in enumerate(found, start=1):
        rid[imp.region.ravel()] = i
        absorb |= imp.region
        pools |= imp.pool
        exits.append(cells)
        targets.append(tgt[cells])

    depth = _cascade_depth(rid, tgt, exits, targets)
    return CrestPlan(absorb, pools, rid, exits, targets, shape, depth, skipped)


def _cascade_depth(rid, tgt, exits, targets):
    """Longest chain of ponds water passes through, by pointer doubling.

    Only ever used to size the re-emission budget, so a D8 reading of a D-infinity landscape
    is good enough — it is a *count of links*, and the link count is what the pass count
    follows. Pointer doubling over the absorbing graph gives every cell its ultimate
    destination in log(chain) steps; the pond ids of those destinations are the edges.
    """
    n = rid.size
    idx = np.arange(n, dtype=np.int64)
    # Absorbing graph: a pond swallows whatever reaches it, so the walk stops there.
    ptr = np.where(rid > 0, idx, tgt)
    for _ in range(MAX_SPREAD_PASSES):
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt

    n_ponds = len(exits)
    edges = []
    for i in range(n_ponds):
        down = rid[ptr[targets[i]]]
        edges.append(tuple(sorted({int(d) for d in down if d and int(d) != i + 1})))

    # Longest path, memoised. A cycle cannot arise from terrain (a spill leaves at the pour
    # level and can only reach a pond below it) but the guard costs nothing and keeps a
    # malformed graph from recursing forever.
    depth = [-1] * (n_ponds + 1)
    walking = [False] * (n_ponds + 1)

    def longest(i):
        if depth[i] >= 0:
            return depth[i]
        if walking[i]:
            return 0
        walking[i] = True
        best = 0
        for d in edges[i - 1]:
            best = max(best, 1 + longest(d))
        walking[i] = False
        depth[i] = best
        return best

    return max((longest(i) for i in range(1, n_ponds + 1)), default=0)


def spread_crests(plan, accumulate, base_weights=None, max_passes=None, tol=1e-12):
    """Contract each pond to a mixing node and shed its whole inflow evenly along its crest.

    Parameters
    ----------
    plan : :class:`CrestPlan` from :func:`plan_crest_absorption`.
    accumulate : ``callable(weights) -> 2-D array``. Must run the caller's own accumulation
        over a direction field in which **every cell of** ``plan.absorb`` **self-loops**, so a
        pond holds what reaches it and passes nothing on. ``weights=None`` means the engine's
        own default (one unit per valid cell). Taking it as a callback is what keeps this
        module free of pysheds and testable against a fifteen-line accumulator.
    base_weights : the weights for the first pass, or ``None`` for the engine default.

    Returns a :class:`CrestSpread`. ``accumulation`` is the field; ``outlets`` carries each
    crest cell's own share so the caller can show the crest passing what it actually passes;
    ``pond_flow`` paints each pool with its pond's whole throughput, in the same cell-units
    as the accumulation, so a reader that needs a *through-flow* figure inside a pond has one.

    That last raster exists because contraction makes the accumulation inside a pool stop
    meaning contributing area — measured on Quail Island, the median pool cell falls from
    **22.3 to 1.0**. Anything reading accumulation as catchment size (``keypoint_analysis``
    sizes a keypoint's catchment and a pond site by it) needs the pond's own figure there
    instead. Every cell of a pool carries the same value, so it is a **lookup, not a
    distributable quantity** — summing it over an area multiplies by the pool's cell count.
    """
    acc = np.array(accumulate(base_weights), dtype="float64", copy=True)
    shape = plan.shape
    n = shape[0] * shape[1]
    outlets = np.zeros(n, dtype="float64")
    pond_flow = np.zeros(shape, dtype="float64")
    if not plan.changed:
        return CrestSpread(acc, outlets.reshape(shape), pond_flow, 0, 0, 1, 0.0,
                           list(plan.skipped))

    n_ponds = plan.ponds
    held = np.bincount(plan.rid, weights=acc.ravel(), minlength=n_ponds + 1)[1:]
    stop_at = float(tol) * max(float(held.sum()), 1.0)

    budget = int(max_passes) if max_passes else plan.default_passes()
    passes = 1
    emitted = np.zeros(n_ponds, dtype="float64")
    while passes < budget:
        if held.sum() <= stop_at:
            break
        inj = np.zeros(n, dtype="float64")
        for i in range(n_ponds):
            if held[i] <= 0:
                continue
            cells = plan.exits[i]
            share = held[i] / cells.size
            outlets[cells] += share
            # Injected at what the pond discharges *into*, never at the discharging cell: a
            # cell that also receives flow from outside the pond would otherwise pass that on
            # top of its share, which is the double count the first attempt died of.
            np.add.at(inj, plan.targets[i], share)
        emitted += held
        step = np.asarray(accumulate(inj.reshape(shape)), dtype="float64")
        acc += step
        passes += 1
        held = np.bincount(plan.rid, weights=step.ravel(), minlength=n_ponds + 1)[1:]

    residual = float(held.sum())
    # The crest reads the flux it passes. These cells absorb, so they propagate nothing and
    # no downstream figure can move; what changes is only that the wall stops reporting the
    # pond's whole contents at one cell and nothing at the rest.
    live = outlets > 0
    acc[live.reshape(shape)] = outlets[live]

    # Each pool painted with its own pond's throughput. Only the pool: a wall is not a water
    # body, and its accumulation was already the low figure a local high should have.
    ids = plan.rid.reshape(shape)
    with_water = plan.pools & (ids > 0)
    pond_flow[with_water] = emitted[ids[with_water] - 1]

    return CrestSpread(acc, outlets.reshape(shape), pond_flow, int((emitted > 0).sum()),
                       int(live.sum()), passes, residual, list(plan.skipped))
