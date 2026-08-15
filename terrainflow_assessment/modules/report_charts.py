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
    for node in nodes:
        _draw_edge(ax, node, by_id, positions)
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


def _layout(nodes):
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


def _draw_edge(ax, node, by_id, positions):
    from matplotlib.patches import FancyArrowPatch

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
        label_at = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    else:
        # Off the block: a stub arrow to open air, labelled, so a terminal
        # feature never just stops with no explanation.
        end = (start[0] + _COL_GAP * 0.55, start[1])
        label_at = None
        ax.text(end[0] + 1.2, end[1], "off the block", fontsize=_PT_SMALL,
                color=_WARN if wet else _MUTED, va="center", ha="left",
                zorder=2)

    # Style still carries who chose the link — solid for one you drew, dashed
    # for one followed downhill — so the diagram stays readable in greyscale.
    # Weight and colour carry whether water goes down it, which is a different
    # question and needed a different channel.
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9 if wet else 7,
        linewidth=_edge_width(overflow) if wet else 0.9,
        linestyle="-" if node.get("is_user_link") else (0, (3, 2)),
        color=(_WATER if wet
               else (_INK if node.get("is_user_link") else _MUTED)),
        alpha=1.0 if wet else 0.75,
        shrinkA=1.0, shrinkB=1.0, zorder=3 if wet else 2))

    if label_at and wet:
        ax.text(label_at[0], label_at[1] - 1.2,
                f"{overflow:,.0f} m³", fontsize=_PT_SMALL,
                color=_WATER, va="bottom", ha="center", zorder=4)


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
