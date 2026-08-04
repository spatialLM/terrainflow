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
