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
    style=("line", "#00BCD4", "2.5"),
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
