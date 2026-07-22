"""
core.sizing — shared earthwork sizing engine.

One pure primitive per hydraulic/geometric formula, the single source of truth
for cross-section, volume, conveyance and drawdown maths (mirrors the pattern of
``core/registry`` and ``qgis/adapters/geom.py``). The QGIS-facing modules and the
per-feature capacity/cut helpers in ``modules/earthwork_design.py`` all call into
here rather than re-deriving the trapezoid formula per feature (the historical
copy-paste divergence this package retires).

Conventions
-----------
* Internal units are SI decimals — metres, m², m³, m/s, and slopes as **decimal
  ratios** (rise/run or run/rise as documented per function). The UI converts to
  percent grades / ``z:1`` batters at display time only.
* Every result dataclass exposes ``min_dimension`` — the narrowest physical width
  (channel bottom / pond bottom / wall thickness) of the sized feature, or ``None``
  where the result is not a physical cross-section (spacing, drawdown time). It is
  a first-class output that drives the burn-stage sub-cell warning and the step-3
  export payload; it is *not* a render-resolution selector.
"""

from .primitives import (  # noqa: F401
    ContourSpacingResult,
    DrawdownResult,
    FlowResult,
    PondResult,
    SectionResult,
    VolumeResult,
    contour_spacing,
    drawdown_time,
    manning_flow,
    pond_volume_frustum,
    prismatic_volume,
    trapezoid_section,
)
