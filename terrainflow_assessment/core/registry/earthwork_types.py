"""
earthwork_types.py — Registry of supported earthwork types.

Adding a new type (e.g. "terrace") requires touching exactly this one file:
    register_type(EarthworkTypeConfig("terrace", ...))

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
        "Draw from a contour (Pick Segment / Full Contour) so it holds\n"
        "water evenly, or freehand."
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
    top_width_range=(0.5, 20.0),
    independent_dims=("crest_elevation", "top_width"),
    derived_dims=(),
    soil_group=None,
))

_add(EarthworkTypeConfig(
    key="diversion",
    label="Diversion Drain",
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
