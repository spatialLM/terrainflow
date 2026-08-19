"""
plan_geometry.py — plan-view placement geometry.

Where something sits *on* a drawn feature, and which way it faces. Pure
coordinate maths: no QGIS, no Qt, so it is testable under the coverage gate and
usable from either side of the controller boundary.

Snapping itself is deliberately NOT here. QGIS already answers "nearest point on
this alignment, and which segment" via ``QgsGeometry.closestSegmentWithContext``,
and the map tools use it. A second implementation in Python would be a second
answer to the same question, and the two would drift. This module takes the
snapped point and the segment index QGIS returns and does the part QGIS does not:
the local tangent, and the two bars that hang off it — the crest along the
alignment (``crest_bar``, what gets cut) and the breach axis square across it
(``perpendicular_sill``, which way the water goes through).
"""

from __future__ import annotations

import math

# A weir narrower than this is drawn at this width anyway. Below roughly half a
# metre the sill stops reading as a structure and starts reading as a dropped
# pin, and a 0.2 m notch still has to be findable on the map.
#
# **A drawing floor, not a design one.** The burn rounds the built width up to a
# whole number of DEM cells before it cuts, so on any grid at or above 0.5 m this
# never binds there — the burned crest is at least one cell wide by construction
# and is wider than this floor whenever the cell is. Callers that are cutting
# rather than drawing pass the already-rounded width; callers that are drawing
# get the floor, so a sub-cell notch is still visible on the map.
MIN_SILL_M = 0.5


def bearing_at(points, seg_index):
    """Direction of the alignment at *seg_index*, in radians, measured
    anticlockwise from east.

    *points* is a sequence of ``(x, y)``. *seg_index* is the index of the vertex
    starting the segment, as returned by ``closestSegmentWithContext`` (which
    reports the vertex *after* the segment, so callers pass ``idx - 1``).

    Returns ``None`` when there is no segment to take a bearing from — a
    single-vertex or empty geometry. Callers must handle that rather than
    receive a plausible zero.
    """
    pts = _clean(points)
    if len(pts) < 2:
        return None

    i = max(0, min(int(seg_index), len(pts) - 2))
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    dx, dy = x1 - x0, y1 - y0
    if dx == 0.0 and dy == 0.0:
        # Duplicate vertices: walk outward for the nearest segment with length.
        for j in range(1, len(pts)):
            lo, hi = i - j, i + j
            for k in (hi, lo):
                if 0 <= k < len(pts) - 1:
                    ax, ay = pts[k]
                    bx, by = pts[k + 1]
                    if (bx - ax, by - ay) != (0.0, 0.0):
                        return math.atan2(by - ay, bx - ax)
        return None
    return math.atan2(dy, dx)


def crest_bar(points, seg_index, centre, width_m):
    """The two ends of a weir crest of *width_m*, centred on *centre*, running
    **along** the alignment at *seg_index*.

    This is the geometry that gets cut. A weir's crest is the line the water
    crosses as it leaves, and on a swale, a basin rim or a dam that line lies
    along the bank — the flow goes *across* it. ``width_m`` is the length of
    crest the flow passes over, so it is measured along the alignment, and the
    burned notch is the run of bank lowered to the sill.

    :func:`perpendicular_sill` is the same maths a quarter turn round, and it
    answers the other question — which way the water travels through the notch.
    Both are needed; only this one is a crest.

    Returns ``((x1, y1), (x2, y2), bearing_rad)``, or ``None`` if no bearing can
    be taken.
    """
    return _bar(points, seg_index, centre, width_m, along=True)


def perpendicular_sill(points, seg_index, centre, width_m):
    """The **breach axis** through a sill of *width_m*, square across the
    alignment at *seg_index*.

    This is the direction water travels as it leaves — from the pond, over the
    crest, onto the ground below. It is the right shape for the cartographic
    gate symbol (a marker that reads as a way *through* the bank) and the wrong
    shape for the cut: lowering ground along this line runs the notch down the
    flow path rather than across the bank. The crest itself is
    :func:`crest_bar`.

    Returns ``((x1, y1), (x2, y2), bearing_rad)``, or ``None`` if no bearing can
    be taken.
    """
    return _bar(points, seg_index, centre, width_m, along=False)


def _bar(points, seg_index, centre, width_m, along):
    """Shared endpoint maths for :func:`crest_bar` / :func:`perpendicular_sill`."""
    bearing = bearing_at(points, seg_index)
    if bearing is None:
        return None

    half = max(float(width_m or 0.0), MIN_SILL_M) / 2.0
    axis = bearing if along else bearing + math.pi / 2.0
    dx, dy = math.cos(axis) * half, math.sin(axis) * half
    cx, cy = float(centre[0]), float(centre[1])
    return (cx - dx, cy - dy), (cx + dx, cy + dy), bearing


def _clean(points):
    """Coordinate pairs only, tolerating QgsPointXY-ish objects and empties."""
    out = []
    for p in points or ():
        if p is None:
            continue
        if hasattr(p, "x") and callable(p.x):
            out.append((float(p.x()), float(p.y())))
        else:
            out.append((float(p[0]), float(p[1])))
    return out
