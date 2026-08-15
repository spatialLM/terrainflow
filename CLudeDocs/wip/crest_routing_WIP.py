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
out, and that is a different thing from a weir:

| | before |
|---|---|
| share through the busiest crest cell | **87.1%** — 20.9x an even share |
| crest cells carrying < 1% of even | **16 of 24** (median flux **0**) |
| load that is through-flow from upstream | **96.1%** |

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
divided equally between the cells that discharge from it. Two accumulation passes, both run
by the caller's own routing engine so D-infinity keeps splitting fractionally everywhere
outside a pond:

1. **Absorb.** Point every cell of every pond at itself, and accumulate. Each pond now holds
   exactly what reaches it, and nothing threads through it — which is also the truthful
   picture of a pond, since a channel does not survive crossing one.
2. **Re-emit.** Accumulate again with the ponds still absorbing, but inject ``T / n`` as a
   source at each cell a pond discharges into.

Mass is conserved by construction: a pond emits exactly what it received, so every figure
below the first confluence is unchanged, and the site's totals do not move. Ponds that drain
into other ponds are handled by resolving ``T`` in topological order over the pond graph
first, so a chain of dams passes its water down rather than losing it at the first one.

An earlier attempt redistributed flux **among the cells already leaving**, without
contracting. It failed because that set is not a clean cut — sheet flow reconverges down the
wall face, so shares were double-counted, moving the site's busiest cell by -6% to -10% and
driving 64 cells negative. Contraction removes the failure mode rather than patching it.

Pure numpy + scipy: no QGIS, no pysheds, no rasterio.
"""

from __future__ import annotations

import numpy as np

# A pond larger than this keeps the default routing. Only ever catches something
# pathological (a burn that dammed a whole tile); it exists so the analysis thread cannot
# stall with no way to report why.
MAX_REGION_CELLS = 2_000_000

_STRUCT = np.ones((3, 3), dtype=bool)


class Impoundment:
    """One pond, and the level rim it spills over."""

    __slots__ = ("region", "pour_level_m", "n_cells")

    def __init__(self, region, pour_level_m):
        self.region = region                      # bool (rows, cols) — pool + level rim
        self.pour_level_m = float(pour_level_m)
        self.n_cells = int(region.sum())


class CrestSplit:
    """What :func:`plan_crest_split` worked out for the caller to apply."""

    __slots__ = ("absorb", "injection", "outlets", "ponds", "outlet_cells", "skipped")

    def __init__(self, absorb, injection, outlets, ponds, outlet_cells, skipped):
        self.absorb = absorb              # bool (rows, cols) — cells to make self-draining
        self.injection = injection        # float64 (rows, cols) — source term for pass 2
        self.outlets = outlets            # float64 (rows, cols) — each crest cell's share
        self.ponds = ponds                # int
        self.outlet_cells = outlet_cells  # int
        self.skipped = skipped            # list[str]

    @property
    def changed(self) -> bool:
        return self.ponds > 0


def _level_rim(filled, pool, pour, tol):
    """Cells at the pour level reachable from *pool* — the pond's own rim and wall.

    The pool-facing row of a wall is not the crest. A 2 m wall on a 1 m grid is two cells
    thick and only the far row has anywhere to drain to: on Dam 15 the pool-facing row is 24
    cells of which **8** can pass water, while the full level band is **65 cells of which 49
    can**. Growing the rim is what makes an even spread reachable at all.

    (This is a routing region, not a length. ``reporting.overtopping_spill`` deliberately
    reports the pool-facing row instead, because that band is length x *thickness* and using
    it as a length would shrink ``q = Q/L`` in the unsafe direction.)
    """
    from scipy.ndimage import binary_dilation

    at_pour = np.abs(filled - pour) <= tol
    rim = pool.copy()
    while True:
        grown = (binary_dilation(rim, structure=_STRUCT) & at_pour) | rim
        if grown.sum() == rim.sum():
            return grown & ~pool
        rim = grown


def find_impoundments(filled, ground, built=None, min_depth=1e-3, tol=1e-6,
                      max_cells=MAX_REGION_CELLS):
    """Every pond on the surface, with the level rim it spills over.

    Parameters
    ----------
    filled : 2-D array — the depression-filled surface (pond tops are level here).
    ground : 2-D array — the same surface **before** depressions were filled. Take the copy
        *before* the call: pysheds' ``fill_depressions`` writes into its input buffer and
        returns it, so ``filled - pit_filled`` afterwards is identically zero.
    built : 2-D bool array or None — when given, only ponds touching raised ground are
        returned. Left as ``None`` a natural hollow is treated the same as an impounded one,
        which is the honest default: a lake spills over its saddle by the same physics, and
        applying the weir to one tier but not the other would make a before/after comparison
        partly measure the change in method.

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

    for pid in range(1, n_pools + 1):
        pool = labels == pid
        if not pool.any():
            continue
        pour = float(filled[pool].max())
        region = pool | _level_rim(filled, pool, pour, tol)
        if built is not None and not (built & region).any():
            continue
        n_cells = int(region.sum())
        if n_cells > max_cells:
            skipped.append(
                f"a {n_cells:,}-cell pond at {pour:.2f} m keeps the default routing "
                f"(over the {max_cells:,}-cell spreading cap)")
            continue
        out.append(Impoundment(region, pour))

    return out, skipped


def plan_crest_split(impoundments, absorbed_acc, next_flat, is_sink, shape):
    """Work out each pond's outflow and how to shed it evenly.

    Parameters
    ----------
    absorbed_acc : 2-D array — accumulation from the pass where every pond cell drains to
        itself. A pond's cells then hold exactly what reached it and nothing more.
    next_flat, is_sink : the steepest-descent pointers over the conditioned surface
        (``flow_graph.d8_from_dem``), used to find which cells discharge out of each pond.

    Returns a :class:`CrestSplit`. ``injection`` is a source term to add to the accumulation
    weights for the second pass; ``outlets`` carries each crest cell's own share so the
    caller can show the crest passing what it actually passes.
    """
    rows, cols = shape
    n = rows * cols
    absorb = np.zeros(shape, dtype=bool)
    injection = np.zeros(shape, dtype="float64")
    outlets = np.zeros(shape, dtype="float64")
    if not impoundments:
        return CrestSplit(absorb, injection, outlets, 0, 0, [])

    idx = np.arange(n, dtype=np.int64)
    is_sink = np.asarray(is_sink, dtype=bool).ravel()
    tgt = np.where(is_sink, idx, np.asarray(next_flat).astype(np.int64).ravel())

    # Which pond each cell belongs to (0 = none), so a target landing in another pond is
    # recognisable and its water is passed on rather than lost at the first wall.
    rid = np.zeros(n, dtype=np.int64)
    for i, imp in enumerate(impoundments, start=1):
        rid[imp.region.ravel()] = i
        absorb |= imp.region

    held = np.bincount(rid, weights=np.asarray(absorbed_acc, dtype="float64").ravel(),
                       minlength=len(impoundments) + 1)

    # Each pond's discharging cells and where they discharge to.
    exits, downstream = {}, {}
    for i in range(1, len(impoundments) + 1):
        cells = np.flatnonzero((rid == i) & (rid[tgt] != i) & (tgt != idx))
        exits[i] = cells
        downstream[i] = rid[tgt[cells]] if cells.size else np.empty(0, dtype=np.int64)

    # A pond draining into another pond must hand its water on, so resolve the totals in
    # topological order over the pond graph. Without this a chain of dams loses everything
    # above the lowest one.
    order = _pond_order(downstream, len(impoundments))
    total = {i: float(held[i]) for i in range(1, len(impoundments) + 1)}
    for i in order:
        cells = exits[i]
        if cells.size == 0:
            continue
        share = total[i] / cells.size
        for d in downstream[i]:
            if d:
                total[int(d)] += share

    ponds = 0
    outlet_cells = 0
    flat_inj = injection.ravel()
    flat_out = outlets.ravel()
    for i in order:
        cells = exits[i]
        if cells.size == 0 or total[i] <= 0:
            continue
        share = total[i] / cells.size
        flat_out[cells] = share
        # Injected at what the pond discharges *into*, never at the discharging cell: a cell
        # that also receives flow from outside the pond would otherwise pass that on top of
        # its share, which is the double count the first attempt died of.
        np.add.at(flat_inj, tgt[cells], share)
        ponds += 1
        outlet_cells += int(cells.size)

    return CrestSplit(absorb, injection, outlets, ponds, outlet_cells, [])


def _pond_order(downstream, n_ponds):
    """Ponds ordered so each comes before any pond it drains into (Kahn, cycle-safe)."""
    indeg = {i: 0 for i in range(1, n_ponds + 1)}
    edges = {}
    for i, ds in downstream.items():
        outs = {int(d) for d in ds if d and int(d) != i}
        edges[i] = outs
        for d in outs:
            indeg[d] = indeg.get(d, 0) + 1

    ready = [i for i in range(1, n_ponds + 1) if indeg[i] == 0]
    order = []
    while ready:
        i = ready.pop()
        order.append(i)
        for d in edges.get(i, ()):  # noqa: SIM118
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    # Two ponds that drain into each other cannot both be downstream; take them in index
    # order rather than dropping their water on the floor.
    order.extend(i for i in range(1, n_ponds + 1) if i not in set(order))
    return order
