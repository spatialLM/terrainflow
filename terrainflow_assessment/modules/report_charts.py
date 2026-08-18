"""
report_charts.py — the report's one graphic: the flow network diagram.

The map shows *where* features sit; this shows *what connects to what*, which is
the thing no table conveys and the thing a landowner most needs to understand
about a water-harvesting scheme. Features spilling to nowhere become obvious at
a glance.

Kept apart from :mod:`report_model` so that module stays importable without
matplotlib — the content decisions must never depend on whether a chart can be
drawn. Every function here returns ``None`` when matplotlib is missing, and the
model already carries a text fallback for that case.

Greyscale-safe by construction: link type is carried by line style, never by
colour, and every node is directly labelled.
"""

import base64
import io
import logging

_log = logging.getLogger(__name__)

#: The resolution every report figure is rendered at.
#:
#: It has to be the one ``layout_pdf`` lays them out against — that renderer sizes
#: an image as ``pixels / dpi × 25.4 mm``, so a figure saved at any other value
#: prints at the wrong size and there is no fitting rule that can recover it. The
#: hydrograph and fill timeline were saved at 100 and laid out against 200, and so
#: printed at half width with ~5.5 pt axis labels. HTML scales to the column, so
#: the higher figure costs it nothing but sharpness.
REPORT_DPI = 200

_FALLBACK_TYPE_COLOUR = "#7f8c8d"


def type_colour(ew_type):
    """The registry's colour for an earthwork type.

    Read from ``core.registry.earthwork_types`` rather than kept here. This
    module used to carry its own five-colour table under a comment claiming it
    matched the panel — it did not, and had not for some time: a swale was cyan
    on the map and blue on this diagram, a dam brown on the map and green here.
    The registry is where every other part of the plugin gets its colour, and
    it is dependency-free, so reading it costs this module nothing.
    """
    try:
        from terrainflow_assessment.core.registry.earthwork_types import get_type
        return get_type(ew_type).style[1] or _FALLBACK_TYPE_COLOUR
    except Exception:
        return _FALLBACK_TYPE_COLOUR
# Every glyph here must exist in matplotlib's default DejaVu Sans or it prints
# as a tofu box. The panel uses U+2312 ARC for a berm, which DejaVu lacks;
# U+25E0 UPPER HALF CIRCLE reads the same and is present. Guarded by a test.
_GLYPH = {"swale": "∿", "basin": "▢", "dam": "▮", "berm": "◠", "diversion": "↘"}

_INK = "#22302e"
_MUTED = "#5f7176"
_WATER = "#1273b5"
_WARN = "#b9770e"

# Geometry is in millimetres at final printed size, and the axes are drawn with
# an equal aspect, so a point size here is the point size on paper. Working in
# arbitrary units instead makes text scale with the number of nodes: a one-node
# diagram came out with the label three times the width of its own box.
#
# The renderer scales the whole figure down to fit the text column, so every
# millimetre of column pitch costs printed text size on a wide scheme. A
# 42-feature design runs eight ranks deep; at the old 44 + 22 mm pitch that was
# 528 mm of diagram squeezed into a 247 mm landscape page, and the labels came
# out too small to read. Hence the tighter box and the much tighter gap — the
# gap only has to be long enough to see an arrowhead in.
_NODE_W, _NODE_H = 38.0, 15.5
_COL_GAP, _ROW_GAP = 12.0, 5.0
_MAX_PER_COL = 8
_KEY_H = 9.0
_PAD = 3.0

#: Room to the right of a terminal node for its "off the block" stub and label,
#: which are drawn outside the box and so are not in its position.
_TERMINAL_ROOM = _COL_GAP + 12.0

#: The printable area of the landscape network page, less its margins — what
#: ``layout_pdf._render_image`` scales the finished figure down to fit. The
#: packer works against the pair rather than against a width alone: the figure
#: is shrunk by whichever dimension overflows *most*, so packing to a fixed
#: width would trade a scheme that was too wide for one that is too tall and
#: come out with smaller type than it started with.
_PAGE_W_MM, _PAGE_H_MM = 267.0, 170.0

_PT_NAME, _PT_META, _PT_SMALL, _PT_KEY = 7.5, 6.0, 5.2, 6.0


def _pyplot():
    """matplotlib or None. Absence is a normal state, not an error."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover - depends on the environment
        _log.info("matplotlib unavailable, charts will be omitted: %s", exc)
        return None


def render_flow_network(graph, path=None, dpi=REPORT_DPI):
    """Draw the overflow network. Returns PNG bytes, or writes to ``path``.

    ``graph`` is the dict from
    :func:`terrainflow_assessment.modules.report_model.build_flow_graph`.
    Returns ``None`` when matplotlib is unavailable or there is nothing to draw.

    Every node carries its capacity and how full it got, whatever the size of
    the scheme. This used to fall back to name-only above 25 features, on the
    theory that the detail stopped being readable first — but a real 42-feature
    design is exactly where a reader needs to know which features are full, and
    a box with nothing but a name in it answers none of the questions the page
    is there to answer. The geometry above was tightened instead.
    """
    plt = _pyplot()
    if plt is None:
        return None
    nodes = list(graph.get("nodes") or [])
    if not nodes:
        return None

    positions = _layout(nodes)
    # Room for the widest thing on the right: a terminal node's "off the block"
    # stub and its label, which sit outside the node box.
    width = max(p[0] for p in positions.values()) + _NODE_W + _COL_GAP + 12.0
    # The key runs to about 135 mm, so a one-node diagram is sized by its
    # legend rather than by its content.
    width = max(width, 145.0)
    height = (max(p[1] for p in positions.values()) + _NODE_H + _KEY_H + _PAD)

    fig, ax = plt.subplots(figsize=(width / 25.4, height / 25.4), dpi=dpi)
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.invert_yaxis()
    # Equal aspect keeps one millimetre the same size in both directions, so the
    # box geometry and the point sizes stay in the relationship set above.
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    by_id = {n["id"]: n for n in nodes}
    # Obstacles are scoped to the link's own block. A link never leaves its
    # cascade, and blocks do not overlap, so a box in another block can only
    # push a route around something it was never going to touch — most visibly
    # the drop-below fallback, which would dive past a whole neighbouring chain.
    keys = _component_keys(nodes)
    boxes_by_key = {}
    for node in nodes:
        boxes_by_key.setdefault(keys[node["id"]], []).append(
            positions[node["id"]])
    stagger = _channel_stagger(nodes, positions, keys)
    for node in nodes:
        _draw_edge(ax, node, by_id, positions,
                   tuple(boxes_by_key[keys[node["id"]]]), stagger)
    for node in nodes:
        _draw_node(ax, node, positions[node["id"]])

    _draw_key(ax, height)

    if path:
        fig.savefig(path, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
        return path
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _write_b64(b64, path):
    """Persist an existing base64 PNG to a file. Returns the path, or None.

    The hydrograph and fill-timeline builders in :mod:`reporting` already return
    base64 and are already tested; both renderers want a path, so convert rather
    than reimplement two charts.
    """
    if not b64:
        return None
    try:
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(b64))
        return path
    except (OSError, ValueError) as exc:  # pragma: no cover - disk failure
        _log.info("could not write chart %s: %s", path, exc)
        return None


def render_hydrograph(baseline, post, path, dpi=REPORT_DPI):
    """Before/after outflow hydrograph. Simulation-only; None without one."""
    from terrainflow_assessment.modules.reporting import _build_hydrograph_chart

    if baseline is None or post is None:
        return None
    return _write_b64(_build_hydrograph_chart(baseline, post, dpi=dpi), path)


def render_fill_timeline(post, path, dpi=REPORT_DPI):
    """Per-feature fill over the storm. Simulation-only; None without one."""
    from terrainflow_assessment.modules.reporting import (
        _build_fill_timeline_chart,
    )

    if post is None:
        return None
    return _write_b64(_build_fill_timeline_chart(post, dpi=dpi), path)


#: Millimetres of diagram height given to the full spread of the scheme's
#: elevations. Tall enough that a two-metre difference between neighbouring
#: features is visible, short enough that a scheme spanning a whole hillside
#: still fits a landscape page once the figure is scaled to the text column.
_ELEV_SPAN_MM = 110.0

#: Below this, the elevation range is not worth drawing to scale — features
#: within a few centimetres of each other would stack into one illegible band,
#: and the rank layout says more about the chain than a flat line does.
_MIN_ELEV_RANGE_M = 0.5


def _layout(nodes):
    """Positions in millimetres, keyed by node id.

    Elevation-ordered where the DEM gave real heights, because that is what the
    diagram is *about*: water runs downhill, and a reader tracing a cascade is
    asking which feature is above which. Ranked columns answer a different
    question — how many links from the top — and on a scheme where two chains
    interleave in height the two answers look identical and are not.

    Falls back to the rank layout when the heights cannot support the claim:
    fewer than two known elevations, or a spread too small to draw to scale.
    A feature whose centroid fell on nodata has ``elevation_known`` False and is
    never placed by a stand-in value — see ``EarthworkStore.elevation_known``.

    **A cascade is laid out as a block, and the blocks are packed.** Placing
    every feature on one grid put features that feed each other at opposite ends
    of the page, and their connectors then crossed a dozen links they had nothing
    to do with. Each cascade — a weakly-connected group of the routing graph — is
    laid out on its own and the finished blocks are packed across the page, so a
    link can no longer cross a feature outside its own chain. Millimetres per
    metre are measured once over every node, so a metre of fall is the same
    distance in every block; only the block's own top is rebased, which means
    heights compare *within* a chain and not between them. That is the trade the
    grouping makes, and it is the right way round: nobody traces a cascade
    between two chains that never meet.
    """
    scale = _elevation_scale(nodes)
    chains, loose = _components(nodes)
    if not chains:
        return _sub_layout(nodes, scale)

    def laid_out(factor):
        squashed = None if scale is None else (scale[0], scale[1] * factor)
        blocks = [_block(m, squashed, i) for i, m in enumerate(chains)]
        blocks.sort(key=lambda b: (-b.width, b.order))
        if loose:
            blocks.append(_block(loose, squashed, len(chains), shelf_break=True))
        return _pack(blocks)

    best = None
    for factor in _EXAGGERATIONS:
        positions, fit = laid_out(factor)
        # Ties go to the *largest* factor that prints this well: shrinking the
        # exaggeration past the point where it buys printed size only throws
        # away the fall the diagram is drawn to show.
        rank = (round(fit, 4), factor)
        if best is None or rank > best[0]:
            best = (rank, positions)
        if scale is None:
            break
    return best[1]


def _elevation_scale(nodes):
    """``(top, mm per metre)`` for the whole figure, or None to lay out by rank.

    Measured once over every node rather than per block: the scale is what makes
    two boxes a metre apart look a metre apart, and a per-block scale would draw
    the same fall at a different size in each chain.
    """
    known = [n for n in nodes
             if n.get("elevation_known") and n.get("elevation") is not None]
    if len(known) < 2:
        return None
    heights = [float(n["elevation"]) for n in known]
    top, bottom = max(heights), min(heights)
    if top - bottom < _MIN_ELEV_RANGE_M:
        return None
    return top, _ELEV_SPAN_MM / (top - bottom)


#: How much of the full elevation exaggeration to try. Banding the cascades
#: costs height — each chain gets a strip of its own instead of every chain
#: sharing one column stack — and height is what the page shrinks the figure by.
#: So the exaggeration stops being a constant and becomes a budget: the ladder
#: is walked and the largest factor that still prints at the best available size
#: wins. At the bottom of it the boxes are held apart by ``_spread``'s minimum
#: pitch alone, which is a flat diagram; that is only ever chosen when the fall
#: was buying no printed size at all.
_EXAGGERATIONS = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.06, 0.0)

#: The block everything unconnected shares. A feature with no link either way is
#: not a cascade, and giving each one a block of its own would scatter a scheme
#: that has not been wired up across the whole page — which is the complaint the
#: grouping exists to answer, not a shape of it.
_LOOSE = "\x00loose"


def _component_keys(nodes):
    """``{node id: block key}`` — one key per cascade, ``_LOOSE`` for the rest.

    Union-find over the routing edges. The graph is functional (``target_id`` is
    the single outgoing edge) so there is at most one union per node, and a ring
    — which ``simulation.resolve_targets`` can still hand over — merges into one
    component like any other cycle rather than looping here.
    """
    parent = {n["id"]: n["id"] for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for node in nodes:
        target = node.get("target_id")
        if target in parent and target != node["id"]:
            ra, rb = find(node["id"]), find(target)
            if ra != rb:
                parent[rb] = ra

    sizes = {}
    for node in nodes:
        root = find(node["id"])
        sizes[root] = sizes.get(root, 0) + 1
    return {n["id"]: (find(n["id"]) if sizes[find(n["id"])] > 1 else _LOOSE)
            for n in nodes}


def _components(nodes):
    """``(chains, loose)`` — connected cascades in order, then everything else."""
    keys = _component_keys(nodes)
    chains = {}
    loose = []
    for node in nodes:
        key = keys[node["id"]]
        if key == _LOOSE:
            loose.append(node)
        else:
            chains.setdefault(key, []).append(node)
    return list(chains.values()), loose


def _sub_layout(nodes, scale):
    """One group of nodes on its own grid — by height where there is one."""
    if scale is None:
        return _layout_by_rank(nodes)
    return _layout_by_elevation(nodes, scale)


class _Block:
    """One laid-out group, rebased to the origin so it can be moved."""

    __slots__ = ("width", "height", "positions", "order", "shelf_break")

    def __init__(self, width, height, positions, order, shelf_break):
        self.width = width
        self.height = height
        self.positions = positions
        self.order = order
        self.shelf_break = shelf_break


def _block(members, scale, order, shelf_break=False):
    """Lay one group out and rebase it to the origin.

    For a scheme with a single group the rebase is a no-op — the first column is
    already at x=0 and the highest feature at y=0 — so a one-cascade diagram
    comes out exactly as it did before the grouping existed.
    """
    positions = _sub_layout(members, scale)
    min_x = min(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    rebased = {nid: (x - min_x, y - min_y) for nid, (x, y) in positions.items()}
    ids = set(rebased)
    terminal = any(not n.get("target_id") or n["target_id"] not in ids
                   for n in members)
    width = (max(x for x, _ in rebased.values()) + _NODE_W
             + (_TERMINAL_ROOM if terminal else 0.0))
    # A blocked link drops below every box in its way before coming back, so a
    # block is one row gap taller than its lowest box.
    height = max(y for _, y in rebased.values()) + _NODE_H + _ROW_GAP
    return _Block(width, height, rebased, order, shelf_break)


def _shelves(blocks, budget):
    """Lay the blocks out left to right, wrapping past ``budget`` millimetres.

    Returns ``(positions, width, height)``. The loose block takes a shelf of its
    own: it is the one group with no internal order to protect, and on a
    part-wired scheme it is the tallest thing on the page.
    """
    positions = {}
    x = y = shelf_h = width = 0.0
    for block in blocks:
        if x > 0.0 and (block.shelf_break or x + block.width > budget):
            x = 0.0
            y += shelf_h + _ROW_GAP * 2.0
            shelf_h = 0.0
        for nid, (bx, by) in block.positions.items():
            positions[nid] = (x + bx, y + by)
        width = max(width, x + block.width)
        x += block.width + _COL_GAP
        shelf_h = max(shelf_h, block.height)
    return positions, width, y + shelf_h


def _pack(blocks):
    """Shelf-pack the blocks at whichever width prints largest.

    Returns ``(positions, fit)``, where ``fit`` is the fraction the page will
    shrink the finished figure by.

    Widest first, because a shelf opened by a narrow block wastes the rest of
    its width. The width to wrap at is not obvious and is not fixed: the page
    shrinks the figure by ``min(page_w / w, page_h / h)``, so a scheme wrapped
    at the page width can come back *taller* than it was wide and print smaller
    than it would have unwrapped. Every wrap point that changes the answer is a
    prefix sum of the block widths, so there are only as many candidates as
    there are blocks — cheap enough to lay all of them out and keep the one that
    survives the page best.
    """
    budgets = []
    running = 0.0
    for block in blocks:
        running += block.width + _COL_GAP
        budgets.append(running)
    best = None
    for budget in budgets:
        positions, width, height = _shelves(blocks, budget)
        # Clamped at 1.0 because ``layout_pdf._render_image`` only ever
        # shrinks a figure. Chasing a fit above 1 optimises headroom the
        # page will not give back, and it does it by flattening the fall
        # out of a diagram that already fitted.
        fit = min(1.0, _PAGE_W_MM / max(width, 1e-6),
                  _PAGE_H_MM / max(height, 1e-6))
        # Ties go to the narrower packing: same printed size, less white space.
        rank = (round(fit, 6), -round(width, 6))
        if best is None or rank > best[0]:
            best = (rank, positions, fit)
    return best[1], best[2]


def _layout_by_rank(nodes):
    """Rank → column, order → row, wrapping columns that run off the page."""
    cols = {}
    for n in sorted(nodes, key=lambda n: (n.get("rank", 0), n.get("order", 0))):
        cols.setdefault(n.get("rank", 0), []).append(n)

    positions = {}
    x = 0.0
    for rank in sorted(cols):
        members = cols[rank]
        # A rank taller than the page wraps into side-by-side sub-columns rather
        # than running off the bottom.
        chunks = [members[i:i + _MAX_PER_COL]
                  for i in range(0, len(members), _MAX_PER_COL)]
        for chunk in chunks:
            for row, node in enumerate(chunk):
                positions[node["id"]] = (x, row * (_NODE_H + _ROW_GAP))
            x += _NODE_W + _COL_GAP
    return positions


def _layout_by_elevation(nodes, scale):
    """x from the chain's rank, y from real height — high ground at the top.

    The two axes answer the two questions the page asks: across is *how far down
    the chain*, down is *how far down the hill*. Keeping rank on x means the
    arrows still read left-to-right in flow order, so the elevation axis is
    added information rather than a replacement for it.

    ``scale`` is ``(top, mm per metre)`` from :func:`_elevation_scale`, measured
    over every node in the figure rather than over these ones, so two blocks draw
    the same fall at the same size even though each is rebased to its own top.

    Features with no known elevation are parked below the ones that have them,
    in rank order, rather than being dropped or given a fabricated height.
    """
    top, mm_per_m = scale

    cols = {}
    for n in sorted(nodes, key=lambda n: (n.get("rank", 0), n.get("order", 0))):
        cols.setdefault(n.get("rank", 0), []).append(n)

    # Parked below the *placed* features, not below a fixed span. For a scheme
    # laid out as one group the two are the same number, because its lowest
    # feature is what the span was measured to; for a block that covers part of
    # the hill, or one drawn at a reduced exaggeration, the fixed span would
    # leave a hand's breadth of empty paper inside the block and then charge the
    # page for it.
    placed_span = max(
        [(top - float(n["elevation"])) * mm_per_m for n in nodes
         if n.get("elevation_known") and n.get("elevation") is not None],
        default=0.0)
    unknown_y = placed_span + _NODE_H + _ROW_GAP * 2
    positions = {}
    x = 0.0
    for rank in sorted(cols):
        placed = []
        for node in cols[rank]:
            if node.get("elevation_known") and node.get("elevation") is not None:
                y = (top - float(node["elevation"])) * mm_per_m
            else:
                y = unknown_y
                unknown_y += _NODE_H + _ROW_GAP
            placed.append([node, y])
        for node, y in _spread(placed):
            positions[node["id"]] = (x, y)
        x += _NODE_W + _COL_GAP
    return positions


def _spread(placed):
    """Push overlapping boxes in one column apart, keeping their height order.

    Two features a hand's breadth apart in elevation land a hand's breadth apart
    on the page, which is one illegible stack of boxes. Sorting by y and walking
    down enforces a minimum pitch while preserving which is above which — the
    only property of the elevation axis a reader actually reads off it.
    """
    pitch = _NODE_H + _ROW_GAP
    ordered = sorted(placed, key=lambda item: item[1])
    for i in range(1, len(ordered)):
        gap = ordered[i][1] - ordered[i - 1][1]
        if gap < pitch:
            ordered[i][1] = ordered[i - 1][1] + pitch
    return [(node, y) for node, y in ordered]


def _draw_node(ax, node, pos):
    from matplotlib.patches import FancyBboxPatch, Rectangle

    x, y = pos
    colour = type_colour(node.get("ew_type"))
    # A feature with nowhere to send its overflow gets a heavy border: it is the
    # thing on this page most worth noticing.
    lw = 1.8 if node.get("is_terminal") else 0.9

    ax.add_patch(FancyBboxPatch(
        (x, y), _NODE_W, _NODE_H, boxstyle="round,pad=0,rounding_size=1.5",
        linewidth=lw, edgecolor=_INK, facecolor="white", zorder=3))
    ax.add_patch(Rectangle((x, y), 2.2, _NODE_H, linewidth=0,
                           facecolor=colour, zorder=4))

    glyph = _GLYPH.get(node.get("ew_type"), "●")
    left = x + 4.0
    ax.text(left, y + 4.6, f"{glyph} {node['name']}", fontsize=_PT_NAME,
            color=_INK, va="center", ha="left", zorder=5, fontweight="bold")
    ax.text(left, y + 9.2, f"{node.get('capacity_m3', 0):,.0f} m³ dug",
            fontsize=_PT_META, color=_MUTED, va="center", ha="left", zorder=5)

    # The fill bar lives inside the node so one graphic answers both "where does
    # the water go" and "how hard is this feature working".
    pct = max(0.0, min(100.0, float(node.get("fill_pct") or 0.0)))
    bar_x, bar_y = left, y + 11.6
    bar_w, bar_h = _NODE_W - 8.0 - 8.0, 2.2
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, linewidth=0,
                           facecolor="#dde4e5", zorder=5))
    if pct > 0:
        ax.add_patch(Rectangle(
            (bar_x, bar_y), bar_w * pct / 100.0, bar_h, linewidth=0,
            facecolor=_WARN if node.get("overflowed") else _WATER, zorder=6))
    ax.text(x + _NODE_W - 3.0, bar_y + bar_h / 2.0, f"{pct:.0f}%",
            fontsize=_PT_SMALL, color=_MUTED, va="center", ha="right", zorder=7)


def _edge_width(overflow_m3):
    """Line weight for a link carrying ``overflow_m3``, in printed millimetres.

    Stepped rather than continuous: the eye reads three weights reliably and a
    smooth function off a volume that spans two orders of magnitude just makes
    every link look the same.
    """
    volume = float(overflow_m3 or 0.0)
    if volume >= 500.0:
        return 2.2
    if volume >= 100.0:
        return 1.6
    return 1.1


#: How far a link may be nudged off the centre of a channel so that two links
#: sharing one do not draw over each other. The channel is ``_COL_GAP`` wide and
#: a box edge sits at each side of it, so this has to stay well inside half of
#: that or the staggering would put a line through the thing it is avoiding.
_CHANNEL_STAGGER_MM = 3.2


def _clear_horizontal(y, x_from, x_to, boxes):
    """Can a horizontal run at *y* cross from *x_from* to *x_to* untouched?

    ``boxes`` is every node's ``(x, y)``. A run is blocked when it passes through
    a box's vertical extent while inside its column band — which is the only way
    a link can end up appearing to enter a feature it does not connect to.
    """
    lo, hi = (x_from, x_to) if x_from <= x_to else (x_to, x_from)
    for bx, by in boxes:
        if by <= y <= by + _NODE_H and bx < hi and bx + _NODE_W > lo:
            return False
    return True


def _edge_route(start, end, boxes=(), stagger=0.0):
    """Waypoints from *start* to *end* that stay out of every node box.

    A straight line between two boxes at different heights cuts diagonally across
    whatever column lies between them, and the boxes are drawn over the top of
    it — so a link ran into one feature and out the other side of another it had
    nothing to do with, which reads as a connection that does not exist.

    The fix is the one a circuit diagram uses: leave horizontally, turn in the
    empty channel between two columns, and arrive horizontally. Boxes only ever
    occupy the column bands and a channel is the gap between two of them, so the
    vertical run cannot cross one. The corner is where the eye picks the line up
    again on the far side of a crossing, which a diagonal never gives it.

    Which channel is not arbitrary. Turning late (just before the target) keeps
    the horizontal on the source's own row; turning early (just after the source)
    puts it on the target's. Either can be blocked when a link skips a column, so
    both are tested against the boxes and the first clear one wins. When neither
    is — a crowded scheme with a long skip — the link drops below every box in
    its way, which is longer but unambiguous.

    ``stagger`` offsets the vertical run so two links sharing a channel stay
    separately readable; without it they superimpose and read as one link.
    """
    (sx, sy), (ex, ey) = start, end
    boxes = tuple(boxes)
    if abs(sy - ey) < 0.5 and _clear_horizontal(sy, sx, ex, boxes):
        # Already in line and nothing in between: corners would imply a detour
        # that is not there.
        return [start, end]
    if ex > sx:
        late = ex - _COL_GAP / 2.0 + stagger
        if _clear_horizontal(sy, sx, late, boxes):
            return [start, (late, sy), (late, ey), end]
        early = sx + _COL_GAP / 2.0 + stagger
        if _clear_horizontal(ey, early, ex, boxes):
            return [start, (early, sy), (early, ey), end]
    # Backwards, within one column, or a skip with both channels blocked. Go out
    # to the right, drop below everything in the way, and come back in.
    out = sx + _COL_GAP / 2.0 + stagger
    back = ex - _COL_GAP / 2.0 + stagger
    below = max([sy, ey] + [by + _NODE_H for bx, by in boxes
                            if min(sx, ex) - _NODE_W <= bx <= max(sx, ex)]
                ) + _ROW_GAP
    return [start, (out, sy), (out, below), (back, below), (back, ey), end]


def _channel_stagger(nodes, positions, keys=None):
    """A per-link nudge, so links sharing one channel stay distinguishable.

    Every link into the same column turns in the same gap, so on a scheme where
    two features feed a third the two verticals land on the same millimetre and
    print as a single line — the diagram then shows one inflow where there are
    two. Grouping by target column and fanning the group across the channel is
    what separates them; ordering by the source's height keeps the fan in the
    same order as the boxes it comes from, so the lines do not cross each other
    on the way in.

    ``keys`` is the block each node belongs to, from :func:`_component_keys`.
    Two packed blocks can share a column position without sharing a channel, and
    fanning their links as one group would nudge each of them off the centre of
    a gap it has entirely to itself.
    """
    groups = {}
    for node in nodes:
        target = node.get("target_id")
        if not target or target not in positions or node["id"] not in positions:
            continue
        key = keys.get(node["id"]) if keys else None
        groups.setdefault((key, positions[target][0]), []).append(node)
    out = {}
    for members in groups.values():
        members.sort(key=lambda n: positions[n["id"]][1])
        if len(members) == 1:
            out[members[0]["id"]] = 0.0
            continue
        step = 2.0 * _CHANNEL_STAGGER_MM / (len(members) - 1)
        for i, node in enumerate(members):
            out[node["id"]] = -_CHANNEL_STAGGER_MM + i * step
    return out


def _draw_edge(ax, node, by_id, positions, boxes=(), stagger=None):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path

    x, y = positions[node["id"]]
    start = (x + _NODE_W, y + _NODE_H / 2.0)
    target = by_id.get(node.get("target_id"))
    # A link only carries water if its source actually spilled — the graph is
    # functional, one link out per feature, so the edge's volume *is* the
    # source node's overflow. Most links on a real scheme carry nothing; the
    # few that do are the ones the design stands or falls on, and they were
    # previously drawn identically to the dry ones.
    overflow = float(node.get("overflow_m3") or 0.0)
    wet = overflow > 0

    if target is not None and target["id"] in positions:
        tx, ty = positions[target["id"]]
        end = (tx, ty + _NODE_H / 2.0)
        # Its own box and the target's are not obstacles — the route starts and
        # ends on them.
        others = tuple(p for p in boxes
                       if p != positions[node["id"]] and p != (tx, ty))
        route = _edge_route(start, end, others,
                            (stagger or {}).get(node["id"], 0.0))
        # On the vertical run, where there is one: it is the longest clear stretch
        # of the line and the one place a volume can sit without landing on a box
        # or on another link's horizontal.
        label_at = _label_point(route, others)
    else:
        # Off the block: a stub arrow to open air, labelled, so a terminal
        # feature never just stops with no explanation.
        end = (start[0] + _COL_GAP * 0.55, start[1])
        route, label_at = [start, end], None
        ax.text(end[0] + 1.2, end[1], "off the block", fontsize=_PT_SMALL,
                color=_WARN if wet else _MUTED, va="center", ha="left",
                zorder=2)

    # Style still carries who chose the link — solid for one you drew, dashed
    # for one followed downhill — so the diagram stays readable in greyscale.
    # Weight and colour carry whether water goes down it, which is a different
    # question and needed a different channel.
    # zorder below the boxes either way. A link that ended up over a box would
    # read as passing through it, which is the same misreading from the other
    # side — the routing is what keeps the line clear, not the stacking.
    ax.add_patch(FancyArrowPatch(
        path=Path(route, [Path.MOVETO] + [Path.LINETO] * (len(route) - 1)),
        arrowstyle="-|>", mutation_scale=9 if wet else 7,
        linewidth=_edge_width(overflow) if wet else 0.9,
        linestyle="-" if node.get("is_user_link") else (0, (3, 2)),
        color=(_WATER if wet
               else (_INK if node.get("is_user_link") else _MUTED)),
        alpha=1.0 if wet else 0.75,
        shrinkA=1.0, shrinkB=1.0, zorder=2,
        joinstyle="miter", capstyle="butt"))

    if label_at and wet:
        ax.text(label_at[0] + 1.0, label_at[1],
                f"{overflow:,.0f} m³", fontsize=_PT_SMALL,
                color=_WATER, va="center", ha="left", zorder=4)


#: Roughly how much room "1,234 m³" needs to the right of the line it labels.
_LABEL_W_MM = 14.0


def _label_point(route, boxes=()):
    """Where a link's volume goes: a clear stretch of its longest vertical run.

    The midpoint is the natural place and usually the right one, but a long
    vertical passes the full height of a column, so on a scheme of any size its
    midpoint is level with somebody's box and the text lands on top of it. The
    candidates walk outward from the middle and the first one with room wins.

    On a straight horizontal link there is no vertical run, so it falls back to
    the midpoint of the whole line. Labelling the geometric centre of an
    L-shaped route would put the text on the corner, which is the one part of
    the line the reader is using to follow it.
    """
    best, best_len = None, 0.0
    for (x1, y1), (x2, y2) in zip(route, route[1:]):
        if abs(x2 - x1) < 0.5 and abs(y2 - y1) > best_len:
            best_len = abs(y2 - y1)
            best = (x1, min(y1, y2), max(y1, y2))
    if best is None:
        (sx, sy), (ex, ey) = route[0], route[-1]
        return ((sx + ex) / 2.0, (sy + ey) / 2.0)

    x, lo, hi = best
    for fraction in (0.5, 0.35, 0.65, 0.22, 0.78):
        y = lo + (hi - lo) * fraction
        # A box's whole row counts as busy, not just the box itself: every
        # feature's own link leaves horizontally at its mid-height and runs right
        # to whichever channel it turns in, so the band beside a box is occupied
        # by a line even where the box has ended. Conservative on purpose — the
        # cost of a false positive is the next candidate down.
        if all(not (by - 1.0 <= y <= by + _NODE_H + 1.0
                    and bx <= x + _LABEL_W_MM)
               for bx, by in boxes):
            return (x, y)
    return (x, (lo + hi) / 2.0)


def _draw_key(ax, height):
    """Link style carries meaning, so it has to be stated — and in greyscale."""
    from matplotlib.lines import Line2D

    y = height - _KEY_H / 2.0
    x = 1.0
    for label, kwargs in (
        ("a link you drew", {"color": _INK, "linewidth": 1.0}),
        ("followed automatically downhill",
         {"color": _MUTED, "linewidth": 0.9, "linestyle": (0, (3, 2))}),
        ("carrying overflow — thicker is more",
         {"color": _WATER, "linewidth": 2.0}),
    ):
        ax.add_line(Line2D([x, x + 8.0], [y, y], **kwargs))
        ax.text(x + 9.5, y, label, fontsize=_PT_KEY, color=_MUTED,
                va="center", ha="left")
        # Advance by the drawn rule plus the text, at the same average-advance
        # approximation the layout uses elsewhere.
        x += 9.5 + len(label) * _PT_KEY * 0.5 * 25.4 / 72.0 + 6.0
