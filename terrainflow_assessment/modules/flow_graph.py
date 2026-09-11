"""
flow_graph.py — pure flow-network graph algorithms on a conditioned DEM.

The shared "walk the flow network" layer beneath the design-tier water balance. Its
job is to answer, cell by cell, **where does this water end up** — which earthwork
intercepts it, or whether it leaves the site — so per-feature catchments are
mutually exclusive and exhaustive and the balance closes.

    d8_from_dem            — steepest-descent next-cell pointers + sink mask
    resolve_terminals      — pointer doubling: every cell → the terminal it reaches
    label_direct_catchments — per-earthwork direct catchments (the headline function)
    walk_downslope         — bounded walk used to resolve a feature's overflow target
    topological_order      — cascade order for the overflow graph (cycle-safe)
    strahler_order         — channel order over a stream mask (order 1 = a primary valley)
    stream_links           — channel links, top-down, split at junctions
    accumulate             — the pointer graph's own contributing-cell count
    main_stem_to_divide    — the valley centreline above a channel head, to the divide
    data_boundary_mask     — cells on the grid edge or beside nodata

Design rules baked in
---------------------
* **Direction comes from steepest descent on the *conditioned* DEM**, never from
  rounding the saved D-infinity angles. A rounded angle's neighbour is not guaranteed
  to be lower, which seeds cycles (measured: ~90k cells trapped in cycles on a 1690²
  grid); steepest descent after ``fill_pits`` + ``fill_depressions`` +
  ``resolve_flats`` is provably acyclic and costs the same. (That middle step is a
  priority-flood **fill** — ``breach_depressions`` does not exist in pysheds 0.5 and has
  never run, whatever the variable it was assigned to was called.)
* **Labelling is storm-independent.** It depends only on terrain + geometry, so the
  caller caches the result and re-derives volumes (× runoff depth) for free when the
  storm changes. Only a geometry edit invalidates it.
* **Exhaustive and mutually exclusive by construction**: every domain cell resolves to
  exactly one of {an earthwork, the site exit, an interior sink, unresolved}. Callers
  rely on ``Σ counts + exit + sink + unresolved == domain.sum()`` — asserted in tests.

Pure numpy: no QGIS, no rasterio, no pysheds.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# Terminal-label sentinels. Real earthwork labels are >= 0.
LABEL_NONE = -1        # ordinary cell / outside the domain — not a terminal, not counted
LABEL_EXIT = -2        # water leaves the site here (grid edge, or flows out of the domain)
LABEL_UNRESOLVED = -3  # trapped in a cycle — should never happen on a conditioned DEM
LABEL_SINK = -4        # interior pit that swallows water (usually a nodata hole)

# Neighbour offsets, scanned in this order so ties break deterministically.
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Pointer doubling converges in ceil(log2(longest path)); this is a runaway guard.
_MAX_DOUBLING_ITERS = 64


class LabelResult:
    """Outcome of :func:`label_direct_catchments`."""

    __slots__ = ("labels", "counts", "exit_cells", "sink_cells",
                 "unresolved_cells", "domain_cells", "iterations")

    def __init__(self, labels, counts, exit_cells, sink_cells,
                 unresolved_cells, domain_cells, iterations):
        self.labels = labels                      # int32 (rows, cols); LABEL_* or >= 0
        self.counts = counts                      # int64 (n_labels,) cells per earthwork
        self.exit_cells = exit_cells              # int
        self.sink_cells = sink_cells              # int
        self.unresolved_cells = unresolved_cells  # int
        self.domain_cells = domain_cells          # int
        self.iterations = iterations              # pointer-doubling rounds used

    @property
    def accounted(self) -> int:
        """Total domain cells assigned somewhere — must equal ``domain_cells``."""
        return (int(self.counts.sum()) + self.exit_cells
                + self.sink_cells + self.unresolved_cells)

    def is_exhaustive(self) -> bool:
        """True when every domain cell resolved to exactly one destination."""
        return self.accounted == self.domain_cells


def _window(dr: int, dc: int, rows: int, cols: int):
    """Aligned (source, neighbour) slice pairs for the neighbour at offset (dr, dc)."""
    r0 = max(0, -dr)
    r1 = rows - max(0, dr)
    c0 = max(0, -dc)
    c1 = cols - max(0, dc)
    src = (slice(r0, r1), slice(c0, c1))
    nbr = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
    return src, nbr


def d8_from_dem(dem, cell_w: float = 1.0, cell_h: float = 1.0, nodata=None):
    """Steepest-descent flow pointers over *dem*.

    Returns ``(next_flat, is_sink)`` where ``next_flat`` is an int32 array of flat
    indices — ``next_flat[i]`` is the cell that cell *i* drains into — and ``is_sink``
    is the boolean mask of cells with no strictly-lower neighbour (which point at
    themselves). Slope is ``(z - z_neighbour) / distance``, so diagonals are correctly
    de-weighted by √2 rather than competing on raw drop.

    *dem* should be **hydrologically conditioned** (pits filled, depressions filled,
    flats resolved). On a conditioned DEM the pointer graph is acyclic and every
    interior cell reaches the boundary; on a raw DEM it will contain pits, which
    :func:`label_direct_catchments` reports as ``LABEL_SINK`` rather than hiding.

    **Ties break by scan position.** ``best`` starts at 0.0 and the test is a strict ``>``,
    so a neighbour must be *strictly* lower to be chosen and, among equal slopes, the
    earliest offset in :data:`_OFFSETS` wins. That matters wherever the surface is level:
    on a resolved flat the synthetic gradient is an integer multiple of a small epsilon, so
    exact ties are common and every one of them resolves the same way. Note pysheds' own
    ``flowdir`` scans N, NE, E, SE, S, SW, W, NW and this scans NW, N, NE, W, E, SW, S, SE,
    so the two tiers lean in different directions on the same tie — pinned by
    ``test_flow_graph.TestTieBreak`` so neither can drift unnoticed.

    Non-finite cells (and *nodata*, when given) are excluded: nothing drains into them
    and they are left as self-pointing sinks.
    """
    z = np.asarray(dem, dtype="float64")
    if z.ndim != 2:
        raise ValueError("dem must be 2-D")
    rows, cols = z.shape
    n = rows * cols

    invalid = ~np.isfinite(z)
    if nodata is not None:
        invalid |= (z == nodata)

    idx = np.arange(n, dtype=np.int32).reshape(rows, cols)
    nxt = idx.copy()
    best = np.zeros((rows, cols), dtype="float64")

    for dr, dc in _OFFSETS:
        dist = math.hypot(dr * cell_h, dc * cell_w)
        if dist <= 0:
            continue
        src, nbr = _window(dr, dc, rows, cols)

        drop = (z[src] - z[nbr]) / dist
        # NaN comparisons are False, so non-finite neighbours lose automatically;
        # the explicit masks also keep nodata cells from claiming or being claimed.
        better = (drop > best[src]) & ~invalid[src] & ~invalid[nbr]

        best[src] = np.where(better, drop, best[src])
        nxt[src] = np.where(better, idx[nbr], nxt[src])

    next_flat = nxt.ravel()
    is_sink = next_flat == np.arange(n, dtype=np.int32)
    return next_flat, is_sink


def resolve_terminals(next_flat, terminal_label, max_iters: int = _MAX_DOUBLING_ITERS):
    """Resolve every cell to the label of the terminal it drains into.

    ``terminal_label`` is a flat int array: ``LABEL_NONE`` for ordinary cells that keep
    flowing, anything else (a real ``>= 0`` label or a ``LABEL_*`` sentinel) marks a
    terminal. Terminals are made self-pointing, then pointer doubling (``nxt = nxt[nxt]``)
    collapses every chain in O(log path length) passes rather than walking cell by cell.

    Returns ``(labels_flat, iterations)``. Cells that never reach a terminal — only
    possible if the pointer graph contains a cycle — come back as ``LABEL_UNRESOLVED``
    instead of hanging or silently joining a neighbour's catchment.
    """
    next_flat = np.asarray(next_flat)
    terminal_label = np.asarray(terminal_label)
    n = next_flat.size

    nxt = next_flat.astype(np.int32, copy=True)
    is_term = terminal_label != LABEL_NONE
    idx = np.arange(n, dtype=np.int32)
    nxt[is_term] = idx[is_term]

    iterations = 0
    for _ in range(max_iters):
        nxt2 = nxt[nxt]
        iterations += 1
        if np.array_equal(nxt2, nxt):
            break
        nxt = nxt2

    labels = terminal_label[nxt].astype(np.int32, copy=True)
    labels[labels == LABEL_NONE] = LABEL_UNRESOLVED
    return labels, iterations


def label_direct_catchments(next_flat, interceptor_labels, domain_mask,
                            is_sink=None, max_iters: int = _MAX_DOUBLING_ITERS):
    """Assign every domain cell to the earthwork that first intercepts its runoff.

    Parameters
    ----------
    next_flat : flat int array from :func:`d8_from_dem`.
    interceptor_labels : (rows, cols) int array — ``>= 0`` on cells covered by an
        earthwork footprint (the label indexes the caller's earthwork list),
        ``LABEL_NONE`` elsewhere.
    domain_mask : (rows, cols) bool — the site. Cells outside it are not counted, and
        water flowing out of it is site exit.
    is_sink : optional bool mask from :func:`d8_from_dem` (recomputed if omitted).

    Returns a :class:`LabelResult`. Every domain cell lands in exactly one bucket:
    an earthwork, ``LABEL_EXIT`` (left the site), ``LABEL_SINK`` (interior pit) or
    ``LABEL_UNRESOLVED``. Cells *on* an earthwork are counted to that earthwork — rain
    landing in a swale drains into it.

    Sinks are split deliberately: one on the grid edge is a genuine outlet and counts
    as exit, one in the interior is a nodata artefact and is reported separately so a
    suspicious number is visible rather than buried in "leaves site".
    """
    interceptor_labels = np.asarray(interceptor_labels)
    domain_mask = np.asarray(domain_mask, dtype=bool)
    if interceptor_labels.shape != domain_mask.shape:
        raise ValueError("interceptor_labels and domain_mask must have the same shape")

    rows, cols = interceptor_labels.shape
    n = rows * cols
    lab_flat = interceptor_labels.ravel()
    dom_flat = domain_mask.ravel()

    if is_sink is None:
        is_sink = np.asarray(next_flat) == np.arange(n, dtype=np.int32)
    is_sink = np.asarray(is_sink, dtype=bool).ravel()

    edge = np.zeros((rows, cols), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    edge_flat = edge.ravel()

    term = np.full(n, LABEL_NONE, dtype=np.int32)
    # Leaving the domain is leaving the site.
    term[~dom_flat] = LABEL_EXIT
    intercepted = (lab_flat >= 0) & dom_flat
    term[intercepted] = lab_flat[intercepted]
    # Sinks that aren't already earthworks: edge = outlet, interior = pit.
    plain_sink = is_sink & dom_flat & ~intercepted
    term[plain_sink & edge_flat] = LABEL_EXIT
    term[plain_sink & ~edge_flat] = LABEL_SINK

    labels_flat, iterations = resolve_terminals(next_flat, term, max_iters=max_iters)
    # Outside the domain is not ours to account for.
    labels_flat[~dom_flat] = LABEL_NONE

    in_dom = labels_flat[dom_flat]
    n_labels = int(lab_flat.max()) + 1 if lab_flat.size and lab_flat.max() >= 0 else 0
    positive = in_dom[in_dom >= 0]
    counts = np.bincount(positive, minlength=n_labels).astype(np.int64)

    return LabelResult(
        labels=labels_flat.reshape(rows, cols),
        counts=counts,
        exit_cells=int((in_dom == LABEL_EXIT).sum()),
        sink_cells=int((in_dom == LABEL_SINK).sum()),
        unresolved_cells=int((in_dom == LABEL_UNRESOLVED).sum()),
        domain_cells=int(dom_flat.sum()),
        iterations=iterations,
    )


def walk_downslope(next_flat, start_flat: int, interceptor_flat,
                   skip_label: int = LABEL_NONE, max_steps: int = 100_000):
    """Follow the flow path from *start_flat* to the next earthwork downslope.

    Returns ``(label, steps)`` — the label of the first earthwork encountered other
    than *skip_label*, or ``None`` if the path reaches a sink or the grid edge first
    (i.e. the overflow leaves the site). This is how a feature's overflow target is
    resolved: from its own outlet cell, follow where water actually goes, rather than
    guessing by centroid elevation.

    A visited set guards against cycles, so a malformed pointer graph costs a bounded
    walk rather than an infinite loop.
    """
    next_flat = np.asarray(next_flat)
    interceptor_flat = np.asarray(interceptor_flat)

    cur = int(start_flat)
    if not (0 <= cur < next_flat.size):
        return None, 0

    seen = {cur}
    for step in range(1, max_steps + 1):
        nxt = int(next_flat[cur])
        if nxt == cur or nxt in seen:
            return None, step          # sink, edge outlet, or a loop → leaves site
        cur = nxt
        seen.add(cur)
        label = int(interceptor_flat[cur])
        if label >= 0 and label != skip_label:
            return label, step
    return None, max_steps


def longest_flow_path(next_flat, mask_flat, cols: int,
                      cell_w: float = 1.0, cell_h: float = 1.0, trace: bool = False):
    """Longest travel distance through *mask_flat*, in metres.

    The hydraulically most distant point of a catchment is what sets its time of
    concentration, and "most distant" means along the flow path, not as the crow
    flies — a long shallow draw takes far longer than its straight-line length.

    Within a catchment the flow pointers form a tree draining to the feature, so this
    is a longest-path problem on a DAG: process cells from the ridges down, and each
    cell records the longest route that reaches it. Kahn's algorithm gives the order
    in O(n) without recursion, which matters because a 285 ha catchment at 1 m is
    millions of cells and Python's stack is not.

    Diagonal steps cost ``√(w² + h²)``, so a staircase path is not counted as if it
    ran along the axes.

    Returns ``(length_m, cells_visited)``, or ``(length_m, cells_visited, path)`` when
    *trace* is set — *path* being the flat cell indices from the most distant point
    down to where the flow leaves the mask.

    The trace exists because one average slope over the whole path is not what TR-55
    asks for, and is not harmless: hillslopes are concave, so a single average is far
    too gentle at the top and too steep at the bottom. Measured on a typical concave
    profile it lengthens Tc by 25%, which lowers the design intensity and undersizes
    the overflow. With the path in hand each leg gets its own slope from the actual
    elevation profile.

    A cell trapped in a cycle is simply never dequeued, so a malformed pointer graph
    under-reports rather than hanging; *cells_visited* against ``mask_flat.sum()`` is
    how the caller can tell.
    """
    next_flat = np.asarray(next_flat)
    mask_flat = np.asarray(mask_flat, dtype=bool).ravel()
    if not mask_flat.any():
        return (0.0, 0, []) if trace else (0.0, 0)

    idx = np.flatnonzero(mask_flat)
    nxt = next_flat[idx]

    # Step length per cell, by whether its outflow is diagonal.
    diag = np.hypot(cell_w, cell_h)
    src_r, src_c = idx // cols, idx % cols
    dst_r, dst_c = nxt // cols, nxt % cols
    dr, dc = np.abs(dst_r - src_r), np.abs(dst_c - src_c)
    step = np.where((dr == 1) & (dc == 1), diag,
                    np.where(dr == 1, float(cell_h), float(cell_w)))

    # Only edges that stay inside the catchment participate; one leaving it is the
    # outflow, and its length belongs to whatever is downstream.
    inside = mask_flat[nxt] & (nxt != idx)
    position = np.full(next_flat.size, -1, dtype=np.int64)
    position[idx] = np.arange(idx.size)

    indeg = np.zeros(idx.size, dtype=np.int64)
    targets = position[nxt]
    valid = inside & (targets >= 0)
    np.add.at(indeg, targets[valid], 1)

    up_len = np.zeros(idx.size, dtype=np.float64)
    # Which upstream cell handed each cell its longest route. Only needed for a trace,
    # but it costs one int array and keeps the traversal a single pass.
    came_from = np.full(idx.size, -1, dtype=np.int64) if trace else None

    queue = deque(np.flatnonzero(indeg == 0).tolist())
    visited = 0
    best = 0.0
    best_at = -1
    while queue:
        i = queue.popleft()
        visited += 1
        if up_len[i] > best:
            best = up_len[i]
            best_at = i
        if not valid[i]:
            continue
        j = int(targets[i])
        candidate = up_len[i] + step[i]
        if candidate > up_len[j]:
            up_len[j] = candidate
            if trace:
                came_from[j] = i
        indeg[j] -= 1
        if indeg[j] == 0:
            queue.append(j)

    if not trace:
        return float(best), int(visited)

    # Walk the predecessors back from the cell that ended up furthest from the outlet,
    # then reverse so the path reads ridge → outlet, the direction water travels.
    path = []
    cur = best_at
    guard = idx.size + 1
    while cur >= 0 and guard > 0:
        path.append(int(idx[cur]))
        cur = int(came_from[cur])
        guard -= 1
    path.reverse()
    return float(best), int(visited), path


def topological_order(edges):
    """Cascade order for an overflow graph, plus any nodes that had to be cycle-broken.

    ``edges`` maps ``node_id -> downstream_node_id or None``. Because each feature has
    at most one outgoing edge, cycles are simple rings; Kahn's algorithm drains
    everything acyclic and whatever remains is exactly the ring members.

    Returns ``(order, broken)``. *order* always contains every node — cycle members are
    appended in stable insertion order so the caller can still run a single top-to-bottom
    pass — and *broken* names the nodes whose links form a cycle, so the caller can warn
    and demote the offending user override rather than silently mis-routing.
    """
    nodes = list(edges.keys())
    indeg = {k: 0 for k in nodes}
    for src, dst in edges.items():
        if dst is not None and dst in indeg and dst != src:
            indeg[dst] += 1

    # Each node has at most one outgoing edge, so its in-degree only ever reaches zero
    # once — a node is therefore enqueued exactly once and needs no visited guard.
    queue = deque(k for k in nodes if indeg[k] == 0)
    order = []
    placed = set()
    while queue:
        node = queue.popleft()
        order.append(node)
        placed.add(node)
        dst = edges.get(node)
        if dst is not None and dst in indeg and dst != node:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)

    broken = [k for k in nodes if k not in placed]
    order.extend(broken)
    return order, broken


# ---------------------------------------------------------------------------
# Stream ordering — which valleys are the *primary* ones
# ---------------------------------------------------------------------------

def strahler_order(next_flat, stream_mask_flat):
    """Strahler (1957) order for every channel cell, as a flat int32 array.

    Order 1 is a headwater link: a channel cell with no channel cell draining into it,
    and everything below it until it meets another. Where two links of order *n* meet
    the result is *n + 1*; where orders differ the larger simply continues.

    Yeomans' **primary valley** is the small upland valley at the head of a
    ridge-and-valley pair, which is what an order-1 link is. The trunk of a catchment
    is not a primary valley, and running the keypoint criterion on it — as the single
    ``argmax(acc)`` stem did — answers a different question from the one the method asks.

    **Run this over the stream mask only.** The channel network is typically under 2% of
    a tile; ordering 50,000 cells is a pure-Python pass of about 0.05 s, and ordering
    2.8 million is 5 s and a temptation to reach for numba for no gain.

    Non-channel cells come back as 0.
    """
    next_flat = np.asarray(next_flat, dtype=np.int64)
    stream = np.asarray(stream_mask_flat, dtype=bool).ravel()
    n = next_flat.size

    order = np.zeros(n, dtype=np.int32)
    cells = np.flatnonzero(stream)
    if cells.size == 0:
        return order

    # How many channel cells drain into each channel cell. A cell with none is a
    # headwater source, and the sweep below can start from it.
    indeg = np.zeros(n, dtype=np.int32)
    for i in cells:
        j = int(next_flat[i])
        if j != i and stream[j]:
            indeg[j] += 1

    # Highest order arriving at each cell, and how many links arrive carrying it —
    # Strahler's rule needs both, because two equal orders promote and unequal ones
    # do not.
    best = np.zeros(n, dtype=np.int32)
    best_count = np.zeros(n, dtype=np.int32)
    pending = indeg.copy()

    from collections import deque
    queue = deque(int(i) for i in cells if indeg[i] == 0)

    seen = 0
    while queue:
        i = queue.popleft()
        seen += 1
        if best_count[i] == 0:
            order[i] = 1                      # a source
        elif best_count[i] >= 2:
            order[i] = best[i] + 1            # two equal orders meet
        else:
            order[i] = best[i]                # the larger continues

        j = int(next_flat[i])
        if j == i or not stream[j]:
            continue
        o = order[i]
        if o > best[j]:
            best[j], best_count[j] = o, 1
        elif o == best[j]:
            best_count[j] += 1
        pending[j] -= 1
        if pending[j] == 0:
            queue.append(j)

    # A conditioned DEM gives an acyclic pointer graph, so everything should drain. If
    # anything is left it is a ring, and it keeps order 0 rather than being guessed at.
    return order


def stream_links(next_flat, stream_mask_flat, order_flat, cols, max_order=1,
                 min_cells=3):
    """Channel links of order ≤ *max_order*, each as an ordered list of ``(row, col)``.

    A *link* runs from a source (or a junction) down to the next junction. Splitting
    there is what makes "one valley" a well-defined thing to trace a profile along, and
    the cells come back already ordered from the top down, which is what the keypoint
    profile needs.

    Links shorter than *min_cells* are dropped: a two-cell stub has no profile to take a
    second derivative of, and fitting one to it produces a keypoint out of noise.
    """
    next_flat = np.asarray(next_flat, dtype=np.int64)
    stream = np.asarray(stream_mask_flat, dtype=bool).ravel()
    order = np.asarray(order_flat, dtype=np.int32).ravel()
    n = next_flat.size

    indeg = np.zeros(n, dtype=np.int32)
    for i in np.flatnonzero(stream):
        j = int(next_flat[i])
        if j != i and stream[j]:
            indeg[j] += 1

    links = []
    for start in np.flatnonzero(stream & (order > 0) & (order <= max_order)):
        start = int(start)
        # Begin only at a source, or immediately below a junction — otherwise every
        # cell along a link would seed its own duplicate of that link's tail.
        upstream_same = indeg[start] == 1
        if upstream_same:
            continue

        path = []
        i = start
        while True:
            r, c = divmod(i, cols)
            path.append((int(r), int(c)))
            j = int(next_flat[i])
            if j == i or not stream[j]:
                break
            if order[j] != order[i]:
                break          # the link ends where the order changes
            if indeg[j] > 1:
                path.append((int(j // cols), int(j % cols)))
                break          # ...and at the junction itself
            i = j

        if len(path) >= min_cells:
            links.append(path)

    return links


def accumulate(next_flat, valid_flat=None):
    """Cells draining through each cell, itself included — an int64 flat array.

    The pointer graph's **own** contributing-cell count. A stream mask thresholded on it
    can be walked by the same pointers without a single cell leaving the mask: along a
    single-successor path the count can only grow, so everything downstream of a channel
    cell is a channel cell too. A mask taken from one routing scheme and walked by
    pointers from another has no such guarantee, and that mixture is what shattered the
    keyline network into 3 m fragments (`KPA-52`).

    Kahn's pass, vectorised per level. The cells with no inflow hand their count to the
    cell they drain into; a target whose last inflow has arrived joins the next level;
    and so on down to the sinks. No elevation is read, so the order is the graph's and
    nothing else — the same reasoning :func:`longest_flow_path` uses. On the 400x400
    fixture this is 603 levels in 0.04 s, against 0.15 s for an elevation-sorted loop.

    A cell on a cycle never sees its last inflow arrive and keeps a partial count rather
    than hanging. A conditioned DEM has no cycles; a caller that wants to know can compare
    the count at the sinks against ``valid_flat.sum()``.

    Cells outside *valid_flat* weigh 0 and pass nothing on: a nodata hole contributes no
    area, and neither does anything that was routed into it.
    """
    next_flat = np.asarray(next_flat, dtype=np.int64).ravel()
    n = next_flat.size
    idx = np.arange(n, dtype=np.int64)
    if valid_flat is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.asarray(valid_flat, dtype=bool).ravel()

    moves = next_flat != idx
    indeg = np.bincount(next_flat[moves], minlength=n)
    acc = valid.astype(np.int64)

    frontier = np.flatnonzero(indeg == 0)
    while frontier.size:
        target = next_flat[frontier]
        moving = target != frontier
        src = frontier[moving]
        if src.size == 0:
            break
        dst, inverse = np.unique(target[moving], return_inverse=True)
        acc[dst] += np.rint(np.bincount(inverse, weights=acc[src])).astype(np.int64)
        indeg[dst] -= np.bincount(inverse)
        frontier = dst[indeg[dst] == 0]
    return acc


def main_stem_to_divide(next_flat, acc_flat, head_flat, cols: int,
                        allowed_flat=None, max_steps=None):
    """The valley centreline above *head_flat*, up to the divide — flat indices, top-down.

    Yeomans' primary valley "starts as a more or less sudden steepening of the side slope
    of a main ridge" (*Water for Every Farm*, p58) — at the divide, well above any point a
    channel-area threshold would call a channel head — and its keypoint sits in the short
    steep reach just below that. A link that begins at the channel head has already
    dropped the reach the keypoint is defined against, so it is walked back up here.

    From the head, step to the 8-neighbour that drains **into** the current cell and
    carries the most accumulation — the main stem — and repeat until a cell nothing
    drains into: the divide. Ties go to the first offset scanned in :data:`_OFFSETS`,
    the same rule :func:`d8_from_dem` uses, so the answer is deterministic on a resolved
    flat. A cell outside *allowed_flat* ends the walk below it.

    No index of inflows is needed: a cell's inflows are among its eight neighbours, so
    each step is eight pointer reads. *max_steps* defaults to the grid size and is only a
    guard against a malformed graph.

    Returns the cells **above** the head only, so the caller prepends them to the link.
    Empty when the head is itself a divide.
    """
    next_flat = np.asarray(next_flat, dtype=np.int64).ravel()
    acc = np.asarray(acc_flat).ravel()
    n = next_flat.size
    rows = n // cols
    allowed = (None if allowed_flat is None
               else np.asarray(allowed_flat, dtype=bool).ravel())
    if max_steps is None:
        max_steps = n

    path = []
    cur = int(head_flat)
    seen = {cur}
    for _ in range(max_steps):
        r, c = divmod(cur, cols)
        best = -1
        best_acc = None
        for dr, dc in _OFFSETS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            j = nr * cols + nc
            if j == cur or int(next_flat[j]) != cur or j in seen:
                continue
            if allowed is not None and not allowed[j]:
                continue
            a = acc[j]
            if best_acc is None or a > best_acc:
                best, best_acc = j, a
        if best < 0:
            break
        path.append(best)
        seen.add(best)
        cur = best

    path.reverse()
    return path


def data_boundary_mask(valid2d):
    """True on valid cells that sit on the grid edge or touch a cell that is not valid.

    Where the data stops, so does what can be said about the ground beyond it. A valley
    whose divide lands here may have its steep upper reach off the map; a channel that
    reaches here has left the site, because a boundary row has no outside for
    :func:`d8_from_dem` to route into and the pointers run *along* it instead. Both are
    decisions for the caller — this only says where the edge of the data is.

    The same neighbourhood drift `terrain_indices.landform_tpi` warns about — a window
    hanging off the data edge — is the reason the mask includes cells *beside* nodata and
    not only the outer ring.
    """
    valid = np.asarray(valid2d, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid2d must be 2-D")
    rows, cols = valid.shape
    out = np.zeros_like(valid)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    for dr, dc in _OFFSETS:
        src, nbr = _window(dr, dc, rows, cols)
        out[src] |= ~valid[nbr]
    return out & valid
