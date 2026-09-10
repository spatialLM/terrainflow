"""
advisories.py — soil batter/grade guidelines and a CN-based slope cross-check.

**Advisory only — never a silent clamp (§1.2).** Every function returns
``(within_envelope, guideline_text)`` describing whether the user's geometry sits
inside the recommended envelope for the soil, and *why*. The UI surfaces the text as
a hover tooltip and flags out-of-envelope geometry; it never rewrites the user's
dimensions. Batters that are steeper than recommended, or grades that risk erosion,
are the user's call to make.

Reference envelopes are rules of thumb for small-scale (permaculture / lifestyle
block) earthworks, keyed to the same soil textures as ``modules/swale_design`` and
the SCS ``SOIL_REFERENCE`` curve numbers. Units: batters as ``z:1`` decimal
run-per-rise (larger = flatter = more stable); grades as percent.
"""

from __future__ import annotations

import math

# Minimum *stable* side-slope batter, as decimal run-per-rise (z:1). Sandier soils
# slump and need a flatter (larger-z) batter; cohesive clays hold a steeper wall.
# A user batter with z below the soil's value is flagged (too steep to be stable).
SOIL_MIN_BATTER: dict[str, float] = {
    "Sand":       2.0,   # 2:1
    "Sandy loam": 1.5,
    "Loam":       1.5,
    "Clay loam":  1.0,
    "Clay":       1.0,   # 1:1 — clay holds a steeper face
}

# Maximum *non-eroding* channel grade (percent) before erosion protection is needed.
# Erodible sands scour at low grades; cohesive soils tolerate more.
SOIL_MAX_GRADE_PCT: dict[str, float] = {
    "Sand":       0.5,
    "Sandy loam": 1.0,
    "Loam":       1.5,
    "Clay loam":  2.0,
    "Clay":       2.0,
}

# Curve numbers by soil texture (AMC II) — mirror of SCSRunoff.SOIL_REFERENCE, kept
# duplicated so core/sizing stays free of a modules/ import. TR-55 Table 2-2, pasture
# in GOOD hydrologic condition; see the provenance note on SCSRunoff.SOIL_REFERENCE.
# Used only to infer the soil group from an entered CN for the cross-check, so the
# condition assumption costs nothing here: it is a nearest-match lookup, not a runoff
# figure. Any edit to one table must be mirrored in the other.
_SOIL_CN: dict[str, int] = {
    "Sand":       39,
    "Sandy loam": 49,
    "Loam":       61,
    "Clay loam":  74,
    "Clay":       80,
}

_DEFAULT_SOIL = "Loam"


def _resolve_soil(soil_name: str | None) -> str:
    """Return a known soil key, defaulting to Loam for unknown/None."""
    if soil_name in SOIL_MIN_BATTER:
        return soil_name
    return _DEFAULT_SOIL


def soil_from_cn(cn: float) -> str:
    """Infer the soil texture whose reference CN is nearest to *cn*.

    The entered curve number implies a hydrologic soil group; this maps it back to
    the nearest texture so the batter/grade envelope can be cross-checked against the
    geometry the user drew (§1.2).
    """
    return min(_SOIL_CN, key=lambda name: abs(_SOIL_CN[name] - cn))


def batter_advisory(soil_name: str | None,
                    side_slope: float) -> tuple[bool, str]:
    """Is *side_slope* (z:1 decimal run-per-rise) stable for *soil_name*?

    Returns ``(within_envelope, guideline_text)``. ``within_envelope`` is False when
    the batter is steeper (smaller z) than the soil's minimum stable slope. Advisory
    only — the caller decides whether to honour or override it.
    """
    soil = _resolve_soil(soil_name)
    min_z = SOIL_MIN_BATTER[soil]
    within = side_slope >= min_z
    if within:
        text = (
            f"{soil}: batter {side_slope:.2g}:1 is at or flatter than the "
            f"recommended {min_z:.2g}:1 minimum — stable."
        )
    else:
        text = (
            f"{soil}: batter {side_slope:.2g}:1 is steeper than the recommended "
            f"{min_z:.2g}:1 minimum. Steeper walls may slump; flatten the batter or "
            f"stabilise the face. (Advisory — not enforced.)"
        )
    return within, text


def grade_advisory(soil_name: str | None,
                   grade_pct: float) -> tuple[bool, str]:
    """Is *grade_pct* (channel grade, percent) below the erosion threshold for the soil?

    Returns ``(within_envelope, guideline_text)``. ``within_envelope`` is False when
    the grade exceeds the soil's maximum non-eroding grade. Advisory only.
    """
    soil = _resolve_soil(soil_name)
    max_pct = SOIL_MAX_GRADE_PCT[soil]
    within = grade_pct <= max_pct
    if within:
        text = (
            f"{soil}: grade {grade_pct:.2g}% is within the recommended "
            f"≤ {max_pct:.2g}% envelope — low scour risk."
        )
    else:
        text = (
            f"{soil}: grade {grade_pct:.2g}% exceeds the recommended {max_pct:.2g}% "
            f"maximum. Expect scour; add erosion protection (rock mulch / vegetation) "
            f"or drop structures. (Advisory — not enforced.)"
        )
    return within, text


def cn_slope_crosscheck(cn: float, side_slope: float) -> tuple[bool, str]:
    """Cross-check a batter against the soil group implied by the entered CN (§1.2).

    Infers the soil texture from *cn*, then flags (does not block) an out-of-envelope
    batter. Returns ``(within_envelope, guideline_text)``.
    """
    soil = soil_from_cn(cn)
    within, batter_text = batter_advisory(soil, side_slope)
    text = f"CN {cn:.0f} implies {soil}. {batter_text}"
    return within, text


# ---------------------------------------------------------------------------
# Spacing between contour-aligned features
# ---------------------------------------------------------------------------

# Vertical-interval constants for the terrace rule ``VI(ft) = X·S% + Y``, the SCS /
# Ramser form carried in the USDA-NRCS *Engineering Field Handbook* Part 650, Ch. 8
# (Terraces) and in Schwab et al., *Soil and Water Conservation Engineering*.
#
# **This is an American table applied to New Zealand ground, and the advisory says so.**
# X covers rainfall erosivity and soil erodibility (0.4 in gentler climates through 0.8
# where storms are intense); Y covers tillage, cover and equipment (1.0 for clean-tilled
# erodible ground through 4.0 where cover is good and the implement is wide). Both are
# exposed rather than buried, because ADV-02 already records one unsourced table in this
# file and a second would be worse.
TERRACE_X_DEFAULT = 0.6
TERRACE_Y_DEFAULT = 2.0

#: Cover/tillage constant by soil, on the same five textures as everything else here.
#: Erodible sands want the closest spacing (lowest Y); cohesive clays tolerate more.
TERRACE_Y_BY_SOIL: dict[str, float] = {
    "Sand":       1.0,
    "Sandy loam": 1.5,
    "Loam":       2.0,
    "Clay loam":  2.5,
    "Clay":       3.0,
}

_FT_TO_M = 0.3048


def terrace_vertical_interval(slope_pct: float,
                              soil_name: str | None = None,
                              x: float = TERRACE_X_DEFAULT,
                              y: float | None = None) -> float:
    """Recommended vertical interval between contour features, in metres.

    ``VI(ft) = X · S% + Y``, converted to metres. See the note on the constants above:
    a published *rule*, not a derivation, and its provenance is American.

    Flat ground returns the smallest interval the rule gives (``Y`` alone) rather than
    zero — the rule controls erosion down a slope, and on the level there is no slope
    for it to control.
    """
    if y is None:
        y = TERRACE_Y_BY_SOIL.get(_resolve_soil(soil_name), TERRACE_Y_DEFAULT)
    return (x * max(0.0, float(slope_pct)) + y) * _FT_TO_M


def capture_spacing(runoff_mm: float, capacity_m3_per_m: float) -> float:
    """Widest upslope strip one metre of the drawn section can hold, in metres.

    The other half of the question, and a different one: the terrace rule asks how far
    apart features must be so the slope between them does not scour, while this asks how
    far apart they can be before each receives more water than it holds.

    ``capacity_m3_per_m`` is storage plus soakage per metre of feature — the same figure
    ``swale_design.recommend_swale_length`` sizes against, transposed.

    Returns ``math.inf`` when there is no runoff to catch.
    """
    depth_m = max(0.0, float(runoff_mm)) / 1000.0
    if depth_m <= 0.0:
        return math.inf
    return max(0.0, float(capacity_m3_per_m)) / depth_m


def spacing_advisory(slope_pct: float,
                     soil_name: str | None = None,
                     runoff_mm: float | None = None,
                     capacity_m3_per_m: float | None = None,
                     x: float = TERRACE_X_DEFAULT,
                     y: float | None = None) -> dict:
    """Both spacing rules at one slope, and which of the two governs.

    Returns a dict carrying ``vertical_interval_m``, ``erosion_spacing_m``,
    ``capture_spacing_m`` (``None`` when no storm or section was supplied),
    ``recommended_spacing_m``, ``governing`` (``"erosion"`` / ``"capture"`` /
    ``"none"``) and a ``text`` line that names its own basis.

    **The smaller of the two wins**, because they are different failure modes and
    neither excuses the other: a spacing that holds the water can still let the slope
    between features scour, and one that protects the slope can still overtop.

    Flat ground gives an infinite erosion spacing (``contour_spacing`` guards the
    division), reported as "not governed by erosion" rather than as a number — putting
    ``inf`` in a spin box is how a recommendation becomes a bug report.
    """
    from .primitives import contour_spacing

    soil = _resolve_soil(soil_name)
    vi_m = terrace_vertical_interval(slope_pct, soil, x=x, y=y)
    grade = max(0.0, float(slope_pct)) / 100.0

    # The first production caller of ``contour_spacing`` — HI = VI / S, the terrace
    # relation it was written for and has been waiting on.
    erosion_m = contour_spacing(vi_m, grade).spacing

    capture_m = None
    if runoff_mm is not None and capacity_m3_per_m is not None:
        capture_m = capture_spacing(runoff_mm, capacity_m3_per_m)

    candidates = [v for v in (erosion_m, capture_m)
                  if v is not None and math.isfinite(v) and v > 0]
    recommended = min(candidates) if candidates else None

    if recommended is None:
        governing = "none"
    elif capture_m is not None and math.isclose(recommended, capture_m):
        governing = "capture"
    else:
        governing = "erosion"

    if recommended is None:
        text = (
            f"{soil}: ground at {slope_pct:.1f}% is effectively level, so spacing is "
            "not governed by erosion, and no storm or section was supplied to size "
            "capture against."
        )
    elif governing == "erosion":
        text = (
            f"{soil}: at {slope_pct:.1f}% the terrace rule gives a {vi_m:.2f} m "
            f"vertical interval — {erosion_m:.0f} m apart on the ground. Erosion "
            "governs."
        )
    else:
        text = (
            f"{soil}: the drawn section holds the storm off a {capture_m:.0f} m strip, "
            f"against {erosion_m:.0f} m from the terrace rule at {slope_pct:.1f}%. "
            "Capture governs."
        )

    return {
        "soil": soil,
        "slope_pct": float(slope_pct),
        "vertical_interval_m": vi_m,
        "erosion_spacing_m": erosion_m,
        "capture_spacing_m": capture_m,
        "recommended_spacing_m": recommended,
        "governing": governing,
        "text": text,
    }
