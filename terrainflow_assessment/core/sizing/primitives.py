"""
primitives.py — pure sizing formulas for the earthwork engine.

Single source of truth for the six shared calculations, extracted from the six
divergent trapezoid copies that previously lived in ``modules/earthwork_design.py``:

    trapezoid_section     — open-channel trapezoid geometry (A, P, R, side slope)
    prismatic_volume      — cross-section × length (swale / channel storage)
    contour_spacing       — horizontal interval between contour features (terrace HI)
    manning_flow          — Manning's uniform-flow discharge + velocity (bankfull ceiling)
    pond_volume_frustum   — prismoidal volume of an excavated rectangular frustum pond
    basin_volume_battered — battered-wall volume of an arbitrary-footprint basin
    drawdown_time         — time to infiltrate a stored volume away

All functions are pure and unit-consistent (SI decimals; slopes as decimal ratios).
Each returns a frozen dataclass exposing ``min_dimension`` (§8 of the spec).

Spec §1 fixes baked in here
---------------------------
1. ``drawdown_time`` uses ``storage ÷ (infiltration_rate × infiltrating_area)``
   (≡ ``depth ÷ infiltration_rate``), not the dimensionally-wrong ``storage ÷ rate``.
3. Divide-by-zero guards: ``trapezoid_section`` guards ``depth → 0``;
   ``contour_spacing`` (terrace HI) guards ``slope → 0``; ``manning_flow`` guards
   ``slope → 0`` and empty section; ``pond_volume_frustum`` enforces
   ``2·z·d < min(L, W)`` *before* computing the mid/bottom areas.
4. Units are internal decimals; conversion to percent / ``z:1`` happens at the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result types — every one exposes ``min_dimension`` (§8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectionResult:
    """Trapezoidal open-channel cross-section geometry."""
    area: float                 # flow area A (m²)
    wetted_perimeter: float     # P (m)
    hydraulic_radius: float     # R = A / P (m)
    side_slope: float           # z = (T − b) / 2d, decimal run-per-rise
    top_width: float            # T (m)
    bottom_width: float         # b (m)
    depth: float                # d (m)
    min_dimension: float        # narrowest width = min(T, b) (m)


@dataclass(frozen=True)
class VolumeResult:
    """Prismatic (cross-section × length) volume."""
    volume: float               # m³
    section_area: float         # m²
    length: float               # m
    min_dimension: float | None  # carried through from the section, if known


@dataclass(frozen=True)
class ContourSpacingResult:
    """Horizontal interval between contour-aligned features (terrace HI)."""
    spacing: float              # horizontal interval (m); inf when ground is flat
    vertical_interval: float    # VI (m)
    ground_slope: float         # decimal rise/run
    min_dimension: float | None  # not a physical width → None


@dataclass(frozen=True)
class FlowResult:
    """Manning uniform-flow result. ``discharge`` is the bankfull *ceiling* (max capacity)."""
    discharge: float            # Q (m³/s) — full-section capacity, NOT storm flow
    velocity: float             # v = Q / A (m/s) — bankfull ≈ peak, for scour check
    min_dimension: float | None  # carried through from the section, if known


@dataclass(frozen=True)
class PondResult:
    """Prismoidal volume of an excavated rectangular frustum pond."""
    volume: float               # m³
    top_area: float             # A_top (m²)
    mid_area: float             # A_mid at mid-depth (m²)
    bottom_area: float          # A_bot (m²)
    bottom_length: float        # m
    bottom_width: float         # m
    min_dimension: float        # min(bottom_length, bottom_width) (m)


@dataclass(frozen=True)
class BasinResult:
    """Battered-wall volume of an arbitrary-footprint basin (inset-prism model)."""
    volume: float               # m³
    top_area: float             # surface area A (m²)
    bottom_area: float          # floor area at effective depth (m²)
    effective_depth: float      # ≤ design depth; smaller when walls converge (m)
    side_slope: float           # z, decimal run-per-rise
    min_dimension: float | None  # arbitrary footprint has no single width → None


@dataclass(frozen=True)
class DrawdownResult:
    """Time to infiltrate a stored volume away."""
    time_hr: float              # hours
    storage_m3: float           # m³
    infiltration_rate_m_hr: float  # m/hr
    infiltrating_area_m2: float    # m²
    min_dimension: float | None  # not a physical width → None


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def trapezoid_section(top_width: float, bottom_width: float,
                      depth: float) -> SectionResult:
    """Trapezoidal open-channel section geometry.

    ``z = (T − b) / 2d``, ``A = (T + b)/2 · d``, ``P = b + 2d·√(1 + z²)``,
    ``R = A / P``.

    Guards ``depth → 0`` (§1.3): a zero/negative depth yields a degenerate section
    (zero area, vertical walls) rather than dividing by ``d``.
    """
    if depth <= 0:
        side_slope = 0.0
        area = 0.0
        wetted_perimeter = max(bottom_width, 0.0)
        hydraulic_radius = 0.0
    else:
        side_slope = (top_width - bottom_width) / (2.0 * depth)
        area = ((top_width + bottom_width) / 2.0) * depth
        wetted_perimeter = bottom_width + 2.0 * depth * math.sqrt(1.0 + side_slope ** 2)
        hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0.0

    return SectionResult(
        area=area,
        wetted_perimeter=wetted_perimeter,
        hydraulic_radius=hydraulic_radius,
        side_slope=side_slope,
        top_width=top_width,
        bottom_width=bottom_width,
        depth=depth,
        min_dimension=min(top_width, bottom_width),
    )


def prismatic_volume(section_area: float, length: float,
                     min_dimension: float | None = None) -> VolumeResult:
    """Volume of a prismatic (constant cross-section) feature = ``A · length``.

    Freeboard / compaction factors are the caller's responsibility (they are
    feature-specific policy, not geometry). ``min_dimension`` is carried through
    from the driving section when known.
    """
    volume = max(0.0, section_area) * max(0.0, length)
    return VolumeResult(
        volume=volume,
        section_area=section_area,
        length=length,
        min_dimension=min_dimension,
    )


def contour_spacing(vertical_interval: float,
                    ground_slope: float) -> ContourSpacingResult:
    """Horizontal interval between contour-aligned features (terrace HI = VI / S).

    ``ground_slope`` is a decimal rise/run. Guards ``slope → 0`` (§1.3): flat
    ground gives an infinite interval (``math.inf``) rather than dividing by zero.
    """
    if ground_slope <= 0:
        spacing = math.inf
    else:
        spacing = max(0.0, vertical_interval) / ground_slope
    return ContourSpacingResult(
        spacing=spacing,
        vertical_interval=vertical_interval,
        ground_slope=ground_slope,
        min_dimension=None,
    )


def manning_flow(area: float, hydraulic_radius: float, slope: float,
                 n: float, min_dimension: float | None = None) -> FlowResult:
    """Manning's uniform-flow discharge ``Q = (1/n)·A·R^(2/3)·√S`` and velocity.

    ``slope`` is the decimal energy/bed slope (rise/run). Guards ``slope → 0`` and a
    degenerate section (§1.3): either yields zero discharge.

    ``discharge`` is the **bankfull ceiling** ("max capacity"), never storm flow.
    """
    if slope <= 0 or area <= 0 or hydraulic_radius <= 0 or n <= 0:
        return FlowResult(discharge=0.0, velocity=0.0, min_dimension=min_dimension)
    discharge = (1.0 / n) * area * (hydraulic_radius ** (2.0 / 3.0)) * (slope ** 0.5)
    velocity = discharge / area
    return FlowResult(discharge=discharge, velocity=velocity, min_dimension=min_dimension)


def pond_volume_frustum(top_length: float, top_width: float, depth: float,
                        side_slope: float) -> PondResult:
    """Prismoidal volume of an excavated rectangular frustum pond.

    Walls batter inward at ``side_slope`` (decimal run-per-rise) from a ``top_length ×
    top_width`` surface. Volume by the prismoidal rule ``(d/6)(A_top + 4·A_mid +
    A_bot)`` with ``A_mid`` taken at mid-depth.

    §1.3 guard: enforce ``2·z·d < min(L, W)`` **before** computing the mid/bottom
    areas — otherwise the battered walls converge before reaching full depth and the
    "areas" go negative. A converging geometry raises ``ValueError`` (a genuine
    invalid design, distinct from the advisory batter guidance in ``advisories.py``).
    """
    if depth <= 0:
        raise ValueError("pond depth must be positive")
    if side_slope < 0:
        raise ValueError("pond side slope must be non-negative")

    full_inset = 2.0 * side_slope * depth
    if full_inset >= min(top_length, top_width):
        raise ValueError(
            f"battered walls converge before reaching depth: "
            f"2·z·d = {full_inset:.3g} m ≥ min(L, W) = {min(top_length, top_width):.3g} m"
        )

    top_area = top_length * top_width

    mid_length = top_length - side_slope * depth
    mid_width = top_width - side_slope * depth
    mid_area = mid_length * mid_width

    bottom_length = top_length - full_inset
    bottom_width = top_width - full_inset
    bottom_area = bottom_length * bottom_width

    volume = (depth / 6.0) * (top_area + 4.0 * mid_area + bottom_area)

    return PondResult(
        volume=volume,
        top_area=top_area,
        mid_area=mid_area,
        bottom_area=bottom_area,
        bottom_length=bottom_length,
        bottom_width=bottom_width,
        min_dimension=min(bottom_length, bottom_width),
    )


def basin_volume_battered(area_m2: float, perimeter_m: float, depth_m: float,
                          side_slope: float) -> BasinResult:
    """Battered-wall volume of an arbitrary-footprint basin (inset-prism model).

    The drawn footprint is an arbitrary polygon, so the rectangular frustum
    (:func:`pond_volume_frustum`) does not apply. Model the walls battering inward
    at ``side_slope`` (decimal run-per-rise) by shrinking the area linearly with
    depth via the perimeter: ``A(t) = max(0, A − P·z·t)``, ``V = ∫₀^d A(t) dt``.

    - No convergence (``P·z·d ≤ A``): ``V = A·d − P·z·d²/2``;
      ``bottom_area = A − P·z·d``; ``effective_depth = d``.
    - Convergence at ``t* = A/(P·z) < d``: the walls meet before design depth —
      ``V = A²/(2·P·z)``, ``bottom_area = 0``, ``effective_depth = t*``. Clamped,
      not an error: this runs live during vertex drags and must degrade gracefully.
    - ``z = 0``: exact vertical prism ``A·d``.

    The linear-inset model omits the positive corner term (a square 10×10 m,
    z = 1, d = 1 m gives 80.0 m³ vs the exact 81.33 m³), so it slightly
    *underestimates* — conservative for storage claims.

    Guards (§1.3): non-positive depth or area → graceful zero-volume degenerate
    result (never raises mid-drag). ``side_slope < 0`` → ``ValueError``
    (programmer error; the UI floor is 0).
    """
    if side_slope < 0:
        raise ValueError("basin side slope must be non-negative")
    if depth_m <= 0 or area_m2 <= 0:
        return BasinResult(
            volume=0.0,
            top_area=max(0.0, area_m2),
            bottom_area=max(0.0, area_m2),
            effective_depth=0.0,
            side_slope=side_slope,
            min_dimension=None,
        )

    shrink_rate = perimeter_m * side_slope  # m² lost per metre of depth
    if shrink_rate * depth_m <= area_m2:
        volume = area_m2 * depth_m - shrink_rate * depth_m ** 2 / 2.0
        bottom_area = area_m2 - shrink_rate * depth_m
        effective_depth = depth_m
    else:
        # Walls converge at t* = A / (P·z) before reaching the design depth.
        effective_depth = area_m2 / shrink_rate
        volume = area_m2 ** 2 / (2.0 * shrink_rate)
        bottom_area = 0.0

    return BasinResult(
        volume=volume,
        top_area=area_m2,
        bottom_area=bottom_area,
        effective_depth=effective_depth,
        side_slope=side_slope,
        min_dimension=None,
    )


def drawdown_time(storage_m3: float, infiltration_rate_m_hr: float,
                  infiltrating_area_m2: float) -> DrawdownResult:
    """Time (hours) to infiltrate ``storage_m3`` away through the pond floor.

    §1.1 fix: ``t = storage ÷ (infiltration_rate × infiltrating_area)``, which is
    dimensionally ``m³ / (m/hr · m²) = hr`` and reduces to ``depth ÷ rate`` — matching
    how ``simulation.py`` already loses water (``rate × area × dt``). The spec's
    ``storage ÷ rate`` (``m³ / (m/hr) = m²·hr``) was dimensionally wrong.

    ``infiltration_rate_m_hr`` is a decimal metres/hour (convert mm/hr ÷ 1000 first).
    Guards a zero rate or area → infinite drawdown (``math.inf``) rather than
    dividing by zero.
    """
    denom = infiltration_rate_m_hr * infiltrating_area_m2
    if denom <= 0:
        time_hr = math.inf
    else:
        time_hr = max(0.0, storage_m3) / denom
    return DrawdownResult(
        time_hr=time_hr,
        storage_m3=storage_m3,
        infiltration_rate_m_hr=infiltration_rate_m_hr,
        infiltrating_area_m2=infiltrating_area_m2,
        min_dimension=None,
    )
