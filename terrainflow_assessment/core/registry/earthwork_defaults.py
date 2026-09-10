"""
earthwork_defaults.py — the user's standard earthwork dimensions, layered over the
shipped registry.

A landowner's earthwork size is set by the implement they own: a trough that digs to a
fixed depth, a bucket of a fixed width. This module holds the *user's* numbers and
composes them with ``earthwork_types``' shipped ones, so a feature they draw starts at
the size they actually build.

**The registry is never mutated.** ``_REGISTRY`` stays the record of what the plugin
ships, which is what keeps "what was standard before I changed it" answerable — and what
keeps a personal preference out of the ~20 readers of ``get_type()`` that want a style
colour or a capability flag, not a dimension.

**Pure.** stdlib + ``earthwork_types`` only. The QGIS layer reads ``QgsSettings``, decodes
with :func:`decode`, and passes the result in; nothing here learns that a settings store
exists.

Units are metres throughout. The stored triple is **depth, top width, bottom width** —
never the side slope, which is derived from them. That is the same choice
``modules/project_io`` made for the design file and ``panel.py`` for the swale criteria:
three measurements you can take with a tape at the machine, one consequence you cannot.
It also means the ``max(0.1, ...)`` floor below is never reached by a user's own numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .earthwork_types import get_type

# The historical unknown-type triple, which lived duplicated in Earthwork.__init__ and
# the properties dialog before this module existed. One home.
_FALLBACK_DEPTH = 0.5
_FALLBACK_TOP_WIDTH = 2.0
_FALLBACK_SIDE_SLOPE = 1.0

# The floor Earthwork.__init__ has always applied to a derived bottom width. A footprint
# whose batters would meet before the drawn depth cannot be dug as a trench, and this
# keeps it a (very narrow) floored one rather than a negative number.
_MIN_BOTTOM_WIDTH = 0.1

# Storage keys, and the order the dialog reads them in.
SETTABLE_DIMS = ("depth", "top_width_m", "bottom_width_m")

# Spin boxes carry two decimals, so two decimals is the grain at which "the user changed
# something" is a real question rather than float noise.
_COMPARE_DP = 2


@dataclass(frozen=True)
class DimensionDefaults:
    """One type's stored standard. ``None`` means "no preference for this dimension"."""

    depth: float | None = None
    top_width_m: float | None = None
    bottom_width_m: float | None = None

    def is_empty(self) -> bool:
        return (self.depth is None
                and self.top_width_m is None
                and self.bottom_width_m is None)


@dataclass(frozen=True)
class ResolvedDims:
    """A concrete cross-section: what a fresh feature of this type starts at."""

    depth: float
    top_width_m: float
    bottom_width_m: float


def _derive_bottom_width(top_width, depth, side_slope):
    """Reproduce ``Earthwork.__init__``'s derivation exactly, floor included."""
    return max(_MIN_BOTTOM_WIDTH, top_width - 2 * side_slope * depth)


def shipped_dims(ew_type):
    """The dimensions this type ships with.

    Never raises — an unknown type gets the historical fallback, which is the branch
    ``Earthwork.__init__`` has always carried for it.
    """
    try:
        cfg = get_type(ew_type)
        depth = cfg.default_depth
        top_width = cfg.default_top_width
        side_slope = cfg.default_side_slope
    except KeyError:
        depth = _FALLBACK_DEPTH
        top_width = _FALLBACK_TOP_WIDTH
        side_slope = _FALLBACK_SIDE_SLOPE
    return ResolvedDims(
        depth=depth,
        top_width_m=top_width,
        bottom_width_m=_derive_bottom_width(top_width, depth, side_slope),
    )


def _shipped_side_slope(ew_type):
    try:
        return get_type(ew_type).default_side_slope
    except KeyError:
        return _FALLBACK_SIDE_SLOPE


def settable_dims(ew_type):
    """Which dimensions this type lets a user set, straight off the registry.

    Falls out per type with no special-casing: a basin declares only ``depth`` as
    independent, so its footprint is the drawn polygon; a dam declares ``top_width``
    (its wall thickness) but takes its height from the crest elevation sampled off the
    DEM — a property of the valley, not of the implement.
    """
    try:
        cfg = get_type(ew_type)
    except KeyError:
        return SETTABLE_DIMS
    dims = []
    if "depth" in cfg.independent_dims:
        dims.append("depth")
    if "top_width" in cfg.independent_dims:
        dims.append("top_width_m")
    if "bottom_width" in cfg.derived_dims:
        dims.append("bottom_width_m")
    return tuple(dims)


def resolve_dimensions(ew_type, prefs=None):
    """Compose the shipped dimensions with the user's standard.

    With no preference for *ew_type* this returns :func:`shipped_dims` unchanged, which
    is the property that keeps the whole feature invisible to a user who never sets one.
    """
    shipped = shipped_dims(ew_type)
    pref = prefs.get(ew_type) if prefs else None
    if pref is None or pref.is_empty():
        return shipped

    depth = pref.depth if pref.depth is not None else shipped.depth
    top_width = pref.top_width_m if pref.top_width_m is not None else shipped.top_width_m
    if pref.bottom_width_m is not None:
        bottom_width = pref.bottom_width_m
    else:
        # A partial preference (a hand-edited blob) must not pair a new depth with the
        # shipped floor — the section would describe a batter nobody chose.
        bottom_width = _derive_bottom_width(
            top_width, depth, _shipped_side_slope(ew_type))
    return ResolvedDims(depth=depth, top_width_m=top_width, bottom_width_m=bottom_width)


def dims_match(a, b, dp=_COMPARE_DP):
    """Whether two cross-sections are the same at spin-box precision.

    Drives the "Save as my standard size" toggle: it stays ticked while the dialog still
    describes the stored standard, and unticks the moment the user departs from it.
    """
    return (round(a.depth, dp) == round(b.depth, dp)
            and round(a.top_width_m, dp) == round(b.top_width_m, dp)
            and round(a.bottom_width_m, dp) == round(b.bottom_width_m, dp))


def _clean_value(value):
    """A dimension is a positive, finite number or it is not a dimension."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0:
        return None
    return out


def _as_field_dict(fields):
    """Accept either shape this gate is handed, or None for neither.

    :func:`decode` arrives with the raw ``{dim: value}`` mapping off the JSON blob,
    while :func:`encode` and the controller hold typed :class:`DimensionDefaults`.
    Normalising here keeps one validation gate rather than two that can drift.
    """
    if isinstance(fields, DimensionDefaults):
        return {dim: getattr(fields, dim) for dim in SETTABLE_DIMS}
    if isinstance(fields, dict):
        return fields
    return None


def sanitise(raw):
    """Drop anything that is not a usable standard, and anything already standard.

    Sparse **at the type level, not the field level**: a type whose whole triple matches
    the shipped one is dropped, and any other type keeps all three of its numbers. Per
    *field* sparseness would be a bug — storing only a changed depth lets
    :func:`resolve_dimensions` re-derive a bottom width the user never saw.

    Dropping a matching type is also how a standard is cleared: draw one at the shipped
    size with the toggle on, and the entry goes away rather than pinning the user to a
    number that a later release may ship differently.

    Note what is *not* dropped: a value outside the type's ``depth_range`` /
    ``top_width_range``. Those envelopes are advisory (see ``core/sizing/advisories``),
    and a machine that digs outside one is a fact about the machine. It is flagged, not
    overridden.
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    for ew_type, fields in raw.items():
        fields = _as_field_dict(fields)
        if fields is None:
            continue
        allowed = settable_dims(ew_type)
        values = {}
        for dim in SETTABLE_DIMS:
            if dim not in allowed:
                continue
            cleaned = _clean_value(fields.get(dim))
            if cleaned is not None:
                values[dim] = cleaned
        if not values:
            continue
        pref = DimensionDefaults(**values)
        if dims_match(resolve_dimensions(ew_type, {ew_type: pref}),
                      shipped_dims(ew_type)):
            continue
        out[ew_type] = pref
    return out


def encode(prefs):
    """Serialise to the single JSON blob the settings store holds."""
    payload = {}
    for ew_type, pref in sanitise(prefs).items():
        payload[ew_type] = {dim: getattr(pref, dim) for dim in SETTABLE_DIMS
                            if getattr(pref, dim) is not None}
    return json.dumps(payload, sort_keys=True)


def decode(text):
    """Parse the stored blob, degrading to "no standard set" rather than raising.

    Corrupt, hand-edited, or written by a build that knew a type this one does not:
    every one of those is answered by falling back to the shipped defaults, because a
    preference is never worth an exception on plugin load.
    """
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return sanitise(raw)
