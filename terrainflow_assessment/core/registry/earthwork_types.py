"""
earthwork_types.py — Registry of supported earthwork types.

Adding a new type (e.g. "terrace") means registering it here:
    register_type(EarthworkTypeConfig("terrace", ...))
and giving it a row in each per-type table — the burn dispatch and the three glyph
tables — which `tests/test_registry_completeness.py` checks for every key.

EarthworkTypeConfig fields
--------------------------
key          : str  — internal identifier used as Earthwork.type
label        : str  — display name shown in the UI
geom_type    : str  — "LineString" or "Polygon"
has_storage  : bool — True if the type accumulates water
has_capacity : bool — True if calculate_capacity returns non-zero values
has_cut      : bool — True if calculate_cut_volume returns non-zero values
has_fill     : bool — True if calculate_fill_volume returns non-zero values
burn_method  : str  — key into DEMBurner._BURN_DISPATCH
style        : tuple[str, str, str] — (symbol_type, hex_colour, line_width_or_opacity)
category     : str  — UI grouping: "storage" (holds water — swale/basin/dam; note a
               dam's has_storage=False is a capacity-*path* flag, its volume comes via
               stage-storage) or "control" (moves/blocks flow — berm/diversion)
tooltip      : str  — draw-button tooltip; the panel falls back to "Draw a <label>"
short_label  : str  — stem of a fresh feature's name ("Diversion 3"); the label when ""
depth_label  : str  — properties-dialog row label for ``depth`` ("Depth:" — a bench
               that keeps its dyke height there says "Dyke height:")
width_label  : str  — row label for ``top_width`` ("Width:" — a dam says "Wall thickness:")
bench_mode   : str | None — a bench-shaped type: "level" (a cutback swale, dyke along
               the outer edge) or "reverse" (a bench terrace, 5 % back into the hill);
               None for everything else. Keys the FAO chain (core/sizing/bench.py), the
               bench burn and the bench branches of capacity, cut and fill.
riser_slope  : float | None — the riser as H:V run per unit rise (1.0 machine-built
               earth, 0.75 hand-made earth, 0.5 hand-made rock; FAO 13/3 §6.1)
dyke_top_width_m : float — the dyke along a level bench's outer edge, m; 0 = none
default_side_slope : float — default wall/side batter as an H:V ratio (horizontal run
               per unit vertical rise). 1.0 == 1:1 (today's implicit assumption); 0.0 ==
               vertical / not modelled. Seeds Earthwork.bottom_width_m / batter_run_m defaults.

Sizing policy fields (consumed by core.sizing + the properties dialog)
----------------------------------------------------------------------
default_depth    : float — seed depth (m) for a fresh feature of this type
depth_range      : (min, max) — advisory depth bounds (m) for the UI spinbox
default_top_width: float — seed top width (m)
top_width_range  : (min, max) — advisory top-width bounds (m)
independent_dims : tuple[str, ...] — dims the user sets directly
derived_dims     : tuple[str, ...] — dims computed from the independent ones
                   (e.g. a channel's bottom_width follows top_width + side slope)
soil_group       : str | None — default soil texture association, or None to use the
                   site soil. Keys the batter/grade advisories in core.sizing.advisories.

Spillway policy fields (resolved by modules.earthwork_design.spillway_policy)
-----------------------------------------------------------------------------
spillway_freeboard_m : float — clear height required between the design nappe and the
                   lowest containing ground, in metres. Per type because 0.30 m is an
                   *embankment* standard (NRCS CPS-378) and a cut channel has no
                   embankment; see the comment on the field.
spillway_head_m  : float — default design head over the crest (m) for a fresh spillway.
spillway_head_band : (min, max) — the head range this type treats as ordinary. Outside
                   it the weir formula still holds, it simply stops being routine, so
                   this drives an advisory and never a constraint. Per type, or a swale
                   designed at its own default head would warn permanently — and an
                   advisory that always fires is one the user learns to skip.

Predicates (the questions the UI asks about a type — never compare a key by hand)
----------------------------------------------------------------------------------
is_crest_type(key)   — built to an absolute crest and holds water behind it (a dam, a
                   detainment bund, a WASCOB): the crest row, the wall metrics, keyed-in
                   ends and stage-storage capacity all follow from this, not from the key.
offers_spillway(key) — can be given a designed overflow: has_storage, or a crest type.
is_linear_store(key) — a line-drawn feature that holds water along its run (a swale, a
                   cutback bench): the length row, the demand check, keyed-in berm ends.
name_stem(key)       — what "Swale 3" is named from.
bench_mode_of(key)   — "level" / "reverse" for a bench-shaped type, else None.
`tests/test_architecture.py` forbids comparing a type key against "dam" anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EarthworkTypeConfig:
    key: str
    label: str
    geom_type: str            # "LineString" | "Polygon"
    has_storage: bool
    has_capacity: bool
    has_cut: bool
    has_fill: bool
    burn_method: str
    style: tuple[str, str, str]  # (symbol_type, hex_colour, line_width)
    default_side_slope: float = 1.0  # H:V run-per-rise; 1.0 == 1:1, 0.0 == vertical
    category: str = "storage"        # "storage" | "control" — UI grouping
    tooltip: str = ""                # draw-button tooltip ("" → panel fallback)
    #: Stem of a fresh feature's name — "Diversion" in "Diversion 3" — where the label
    #: is too long to be one; the label itself when "".
    short_label: str = ""
    #: Row labels in the properties dialog for the two aliased dimensions. A type may
    #: keep another quantity in ``depth`` or ``top_width`` (a bench keeps its dyke height
    #: and its bench width there) so that defaults, standards, serialisation and the
    #: dialog read-back are all reused; the label is what tells the user which quantity
    #: the row holds.
    depth_label: str = "Depth:"
    width_label: str = "Width:"

    # --- bench geometry (FAO 13/3 §6.1; core/sizing/bench.py) ---
    #: "level" or "reverse" for a bench-shaped type; None for a channel, wall or basin.
    bench_mode: str | None = None
    #: Riser slope as run per unit rise. A registry constant rather than a per-feature
    #: dimension, following `berm_batter_run`: the riser is built by whatever builds the
    #: bench, and a dialog row for it would be one more number nobody sets.
    riser_slope: float | None = None
    #: Top width of the dyke along a level bench's outer edge; 0.0 when there is none.
    dyke_top_width_m: float = 0.0

    # --- sizing policy (per-feature dimension defaults/limits + soil) ---
    default_depth: float = 0.5
    depth_range: tuple[float, float] = (0.1, 10.0)
    default_top_width: float = 2.0
    top_width_range: tuple[float, float] = (0.1, 100.0)
    independent_dims: tuple[str, ...] = ("depth", "top_width")
    derived_dims: tuple[str, ...] = ("bottom_width",)
    soil_group: str | None = None

    # --- spillway policy (per-type; see modules.earthwork_design.spillway_policy) ---
    # Freeboard and head are NOT one number across types. NRCS CPS-378's 0.30 m is a
    # *pond embankment* figure — the margin between the design nappe and the top of a
    # settled wall. A cut swale has no embankment to breach; its failure mode is water
    # leaving somewhere unarmoured. Applying 0.30 m to a 0.5 m swale spends 60% of the
    # dig before any head is added, which is how a tool teaches people to over-excavate.
    # Defaults here are the embankment figures, so an unlisted type stays conservative.
    spillway_freeboard_m: float = 0.30
    spillway_head_m: float = 0.30
    spillway_head_band: tuple[float, float] = (0.20, 0.50)


# ---------------------------------------------------------------------------
# Built-in types
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, EarthworkTypeConfig] = {}


def _add(cfg: EarthworkTypeConfig) -> None:
    _REGISTRY[cfg.key] = cfg


_add(EarthworkTypeConfig(
    key="swale",
    label="Swale",
    geom_type="LineString",
    has_storage=True,
    has_capacity=True,
    has_cut=True,
    has_fill=True,
    burn_method="swale",
    # Violet, not cyan. A swale used to be drawn in the same blue-green family
    # as the natural streams and the water rasters it sits among, so on the
    # design map the thing being proposed was hard to pick out from the
    # hydrology that was already there. Nothing else on these maps is violet.
    style=("line", "#7E57C2", "2.5"),
    category="storage",
    tooltip=(
        "On-contour channel that captures and infiltrates runoff.\n"
        "Set Draw along to Segment or Contour so it sits level and holds\n"
        "water evenly, or leave it Free to draw by hand."
    ),
    default_side_slope=1.0,
    default_depth=0.5,
    depth_range=(0.1, 2.0),
    default_top_width=2.0,
    top_width_range=(0.5, 10.0),
    independent_dims=("depth", "top_width"),
    derived_dims=("bottom_width",),
    soil_group=None,
    # A swale spills over a low sill in its own bank, not over a dam wall. Both figures
    # are half the embankment ones: at 0.15 + 0.15 the registry's default 0.5 m swale
    # keeps a usable 0.20 m crest band, where 0.30 + 0.30 needs 0.60 m of a 0.50 m dig
    # and reports the most ordinary swale in the plugin as impossible.
    spillway_freeboard_m=0.15,
    spillway_head_m=0.15,
    spillway_head_band=(0.10, 0.30),
))

_add(EarthworkTypeConfig(
    key="berm",
    label="Berm",
    geom_type="LineString",
    has_storage=False,
    has_capacity=False,
    has_cut=False,
    has_fill=True,
    burn_method="berm",
    style=("line", "#8BC34A", "2.5"),
    category="control",
    tooltip=(
        "Raised ridge that blocks or redirects surface flow\n"
        "(no storage of its own)."
    ),
    # **The berm's batter, and live.** `derived_dims=()` and the panel exposes only
    # depth and top width, so a berm carries no side slope of its own — this is the
    # one. `earthwork_design.berm_batter_run` reads it, and both the price
    # (`calculate_fill_volume`) and the build (`DEMBurner._burn_berm`) go through
    # that, so moving this number moves the drawn section and the burned bank
    # together. It used to be inert: the formula assumed a fixed 1:1 triangle and
    # the burn placed a vertical prism, and they disagreed by 4x.
    #
    # `default_top_width` is therefore the **crest**; the base is
    # `top_width + 2 * depth * side_slope`.
    default_side_slope=1.0,
    default_depth=0.5,
    depth_range=(0.1, 2.0),
    default_top_width=2.0,
    top_width_range=(0.5, 10.0),
    independent_dims=("depth", "top_width"),
    derived_dims=(),
    soil_group=None,
))

_add(EarthworkTypeConfig(
    key="basin",
    label="Basin",
    geom_type="Polygon",
    has_storage=True,
    has_capacity=True,
    has_cut=True,
    has_fill=False,
    burn_method="basin",
    style=("fill", "#2196F3", "1.0"),
    category="storage",
    tooltip=(
        "Excavated detention/infiltration basin — draw its footprint\n"
        "as a polygon. Wall batter is settable in its properties."
    ),
    default_side_slope=0.0,  # vertical walls by default; batter settable per feature
    default_depth=1.5,
    depth_range=(0.2, 5.0),
    default_top_width=0.0,  # footprint comes from the drawn polygon, not a width
    top_width_range=(0.0, 0.0),
    independent_dims=("depth",),
    derived_dims=(),
    soil_group=None,
))

_add(EarthworkTypeConfig(
    key="dam",
    label="Dam",
    geom_type="LineString",
    has_storage=False,
    has_capacity=False,
    has_cut=False,
    has_fill=True,
    burn_method="dam",
    style=("line", "#795548", "3.5"),
    category="storage",
    tooltip=(
        "Wall across a valley that impounds water behind its crest.\n"
        "Draw the wall line; storage is computed by flooding the DEM."
    ),
    default_side_slope=0.0,  # rectangular wall approximation today
    default_depth=2.0,       # nominal wall height when no crest is sampled
    depth_range=(0.2, 10.0),
    default_top_width=2.0,   # wall thickness (the drawn line is the inner/wet-side wall)
    width_label="Wall thickness:",
    top_width_range=(0.5, 20.0),
    independent_dims=("crest_elevation", "top_width"),
    derived_dims=(),
    soil_group=None,
))

_add(EarthworkTypeConfig(
    key="diversion",
    label="Diversion Drain",
    short_label="Diversion",
    geom_type="LineString",
    has_storage=False,
    has_capacity=False,
    has_cut=True,
    has_fill=False,
    burn_method="diversion",
    style=("line", "#FF9800", "2.0"),
    category="control",
    tooltip=(
        "Gently-graded channel that carries water across the slope\n"
        "to a storage feature or safe outlet."
    ),
    default_side_slope=1.0,
    default_depth=0.3,
    depth_range=(0.1, 1.5),
    default_top_width=1.0,
    top_width_range=(0.3, 5.0),
    independent_dims=("depth", "top_width", "gradient_pct"),
    derived_dims=("bottom_width",),
    soil_group=None,
))

_add(EarthworkTypeConfig(
    key="cutback_swale",
    label="Cutback swale",
    geom_type="LineString",
    has_storage=True,
    has_capacity=True,
    has_cut=True,
    has_fill=True,
    burn_method="bench",
    # Magenta: nothing else on these maps is, and it has to read apart from the
    # violet swale it sits beside in the menu.
    style=("line", "#D81B60", "2.5"),
    category="storage",
    tooltip=(
        "Level bench cut into the hillside with a low dyke along its outer\n"
        "edge — FAO's cutback terrace. Holds a shallow pond on the platform\n"
        "where a swale would be too steep to dig. Draw it along a contour."
    ),
    # A level platform: the pond is a rectangle, not a trench. `depth` holds the
    # dyke height and `top_width` the bench width, so every path that moves a
    # dimension — defaults, standards, serialisation, the dialog read-back — is
    # reused unchanged, and the row labels say which quantity is which.
    default_side_slope=0.0,
    default_depth=0.20,
    depth_range=(0.10, 0.30),
    default_top_width=4.0,
    top_width_range=(2.5, 8.0),
    independent_dims=("depth", "top_width"),
    derived_dims=(),
    depth_label="Dyke height:",
    width_label="Bench width:",
    soil_group=None,
    bench_mode="level",
    riser_slope=1.0,            # machine-built earth riser, FAO 13/3 §6.1
    dyke_top_width_m=0.30,
    # A dyke 0.10–0.30 m tall spills over its own crest, not an embankment. Half the
    # swale's figures again: at 0.05 + 0.10 the default 0.20 m dyke keeps 0.05 m of
    # crest, where the swale's 0.15 + 0.15 exceeds the dyke outright.
    spillway_freeboard_m=0.05,
    spillway_head_m=0.10,
    spillway_head_band=(0.05, 0.15),
))

_add(EarthworkTypeConfig(
    key="bench_terrace",
    label="Bench terrace",
    geom_type="LineString",
    # A drainage type: it breaks the slope and sheds, it is not a store — so no
    # capacity, no spillway, no demand check. A bench that impounds is the cutback.
    has_storage=False,
    has_capacity=False,
    has_cut=True,
    has_fill=True,
    burn_method="bench",
    # Earth brown: a cut bench, and it has to read apart from the magenta cutback.
    style=("line", "#8D6E63", "2.5"),
    category="control",
    tooltip=(
        "Bench cut across the slope and tilted 5 % back into the hill — FAO's\n"
        "reverse-sloped bench terrace. Breaks a long slope into short steps.\n"
        "Draw it along a contour. In this model it holds what it catches\n"
        "until that soaks in; it has no graded outlet yet."
    ),
    default_side_slope=0.0,
    # No depth of its own: the rise across the bench is FAO's fixed grade times the
    # width, and the riser follows from the ground. The dialog shows no depth row.
    default_depth=0.0,
    depth_range=(0.0, 0.0),
    default_top_width=4.0,
    # FAO 13/3 §6.1: 3.5–8 m machine-built, 2.5–5 m by hand. Advisory, as every range.
    top_width_range=(2.5, 8.0),
    independent_dims=("top_width",),
    derived_dims=(),
    width_label="Bench width:",
    soil_group=None,
    bench_mode="reverse",
    riser_slope=1.0,            # machine-built earth riser, FAO 13/3 §6.1
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_type(key: str) -> EarthworkTypeConfig:
    """Return the config for *key*, raising KeyError for unknown types."""
    return _REGISTRY[key]


def all_types() -> dict[str, EarthworkTypeConfig]:
    """Return a snapshot of the full registry."""
    return dict(_REGISTRY)


def register_type(config: EarthworkTypeConfig) -> None:
    """Register a new earthwork type (or override an existing one)."""
    _REGISTRY[config.key] = config


# ---------------------------------------------------------------------------
# Predicates — the questions the UI asks about a type
# ---------------------------------------------------------------------------
# Each of these used to be asked as `ew.type == "dam"` — thirty-three times, in six files —
# and every one of those would have made a detainment bund behave as a swale: no crest
# row, no wall metrics, no spillway, a network node labelled by the ground under its
# wall. A question about a *kind* of type is answered here, once, and an unknown key
# answers "no" rather than raising, because every caller is a display path that would
# otherwise have to carry its own try/except.

def is_crest_type(key: str) -> bool:
    """Built to an absolute crest and holds water behind it — wall mechanics."""
    try:
        return "crest_elevation" in get_type(key).independent_dims
    except KeyError:
        return False


def offers_spillway(key: str) -> bool:
    """Can be given a designed overflow: it holds water, or it stands a wall."""
    try:
        cfg = get_type(key)
    except KeyError:
        return False
    return cfg.has_storage or is_crest_type(key)


def is_linear_store(key: str) -> bool:
    """A line-drawn feature that holds water along its run (a swale, a cutback bench)."""
    try:
        cfg = get_type(key)
    except KeyError:
        return False
    return cfg.has_storage and cfg.geom_type == "LineString"


def bench_mode_of(key: str) -> str | None:
    """"level" or "reverse" for a bench-shaped type; None for anything else."""
    try:
        return get_type(key).bench_mode
    except KeyError:
        return None


def name_stem(key: str) -> str:
    """What a fresh feature of this type is named from — "Swale" in "Swale 3"."""
    try:
        cfg = get_type(key)
    except KeyError:
        return key.capitalize()
    return cfg.short_label or cfg.label
