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
# Earthmoving: bulking and compaction
# ---------------------------------------------------------------------------

# **The number this replaces was anonymous, and there were two of them.**
# `earthwork_design.berm_spoil_per_metre` multiplies a trench section by a bare 0.75,
# and `DEMBurner` does the same to a cut-depth sum, with no source on either and
# MATHS_AUDIT §5 still listing shrink/swell as unverified. A third differently-sourced
# factor would be the "four incompatible derivations of berm height" trap the project
# already documents, so this is the one place it is named.
#
# Two factors, because cut and fill are measured in different states and the difference
# is the whole point of a balance:
#
#   BULKING     bank (in situ) → loose. What a truck carries.
#   COMPACTION  bank → placed and compacted. What a fill consumes.
#
# Values are the usual small-earthworks ranges (Caterpillar *Performance Handbook*
# load-and-swell tables; Church, *Excavation Handbook*), on the same five textures as
# every other table here. They are **typical figures, not a soil test**, and the
# advisory says so.

#: bank → loose. Excavated soil takes up more room than it did in the ground.
SOIL_BULKING: dict[str, float] = {
    "Sand":       1.10,
    "Sandy loam": 1.18,
    "Loam":       1.25,
    "Clay loam":  1.30,
    "Clay":       1.35,
}

#: bank → compacted fill. Placed and rolled, soil occupies less than it did in situ.
SOIL_COMPACTION: dict[str, float] = {
    "Sand":       0.95,
    "Sandy loam": 0.90,
    "Loam":       0.88,
    "Clay loam":  0.87,
    "Clay":       0.85,
}

#: The historical factor, kept as a named constant so the existing spoil figures do
#: not move. `berm_spoil_per_metre` and the burner still use **this**, not the
#: soil-keyed table: wiring those onto soil changes berm sizing on every design, which
#: is a decision to take on its own with `checks_fixture_regression` in hand.
DEFAULT_COMPACTION = 0.75


def bulking_factor(soil_name: str | None) -> float:
    """bank → loose, for the soil. See :data:`SOIL_BULKING`."""
    return SOIL_BULKING[_resolve_soil(soil_name)]


def compaction_factor(soil_name: str | None) -> float:
    """bank → compacted fill, for the soil. See :data:`SOIL_COMPACTION`."""
    return SOIL_COMPACTION[_resolve_soil(soil_name)]


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


# ---------------------------------------------------------------------------
# Published storage rules for crest types
# ---------------------------------------------------------------------------

#: NZ Building (Dam Safety) Regulations 2022, in force 13 May 2024: a dam is classifiable
#: when it is **four metres or higher and holds 20,000 m³ or more** — both together.
#: Fetched 2026-09-13 from NZSOLD's summary of the MBIE regulations
#: (nzsold.org.nz/mbie-dam-safety-regulations). Past it the design needs a Producer
#: Statement and a CPEng, which no plugin figure replaces.
CLASSIFIABLE_DAM_HEIGHT_M = 4.0
CLASSIFIABLE_DAM_VOLUME_M3 = 20_000.0


def storage_rule_advisory(*, catchment_m2: float | None = None,
                          storage_m3: float | None = None,
                          max_height_m: float | None = None,
                          min_storage_m3_per_ha: float | None = None,
                          max_storage_m3: float | None = None,
                          max_storage_verified: bool = True,
                          max_catchment_ha: float | None = None,
                          max_drawdown_hr: float | None = None,
                          infiltration_mm_hr: float | None = None,
                          pond_area_m2: float | None = None) -> dict:
    """Judge a crest type's measured pond against the rules published for its kind.

    Plain numbers in, so the registry stays the one place the rules live and this stays
    math: the caller passes the type's rule fields beside the measurements. A rule left
    ``None`` is not checked; a measurement left ``None`` gives no verdict on the rules
    that need it, rather than a verdict on zero.

    * **Minimum storage** — ``min_storage_m3_per_ha × catchment ha`` against
      ``storage_m3``, the pond measured at the crest.
    * **Drawdown** — :func:`drawdown_time` over ``pond_area_m2`` at
      ``infiltration_mm_hr``. **Infiltration only**: a bund drains through a decant, so
      this is the slow case, and it is said to be.
    * **Ceilings** — ``max_storage_m3`` (quoted as unverified when
      ``max_storage_verified`` is False) and ``max_catchment_ha``.
    * **Classifiable dam** — :data:`CLASSIFIABLE_DAM_HEIGHT_M` and
      :data:`CLASSIFIABLE_DAM_VOLUME_M3` together, inclusive.

    Returns ``required_m3``, ``holds`` (True / False / None), ``drawdown_hr``,
    ``classifiable_dam`` (True / False / None), ``flags`` (one sentence per breach) and
    ``text`` (the requirement and every flag, for a label).
    """
    from .primitives import drawdown_time

    catchment_ha = (float(catchment_m2) / 10_000.0
                    if catchment_m2 is not None and catchment_m2 > 0 else None)
    flags = []

    required = None
    holds = None
    if min_storage_m3_per_ha is not None and catchment_ha is not None:
        required = float(min_storage_m3_per_ha) * catchment_ha
        if storage_m3 is not None:
            holds = float(storage_m3) >= required
            if not holds:
                flags.append(
                    f"Holds {storage_m3:,.0f} m³ against the {required:,.0f} m³ its "
                    f"{catchment_ha:.1f} ha catchment needs at "
                    f"{min_storage_m3_per_ha:g} m³/ha — raise the crest or lengthen "
                    f"the bund.")

    drawdown = None
    if (max_drawdown_hr is not None and storage_m3 is not None
            and infiltration_mm_hr is not None and pond_area_m2 is not None):
        drawdown = drawdown_time(float(storage_m3), float(infiltration_mm_hr) / 1000.0,
                                 float(pond_area_m2)).time_hr
        if drawdown > float(max_drawdown_hr):
            shown = "never" if math.isinf(drawdown) else f"{drawdown:,.0f} h"
            flags.append(
                f"Soaking away alone it would drain in {shown}, past the "
                f"{max_drawdown_hr:g} h limit. That is an infiltration-only bound: a "
                f"decant outlet is what brings it inside the limit.")

    if max_storage_m3 is not None and storage_m3 is not None \
            and float(storage_m3) > float(max_storage_m3):
        basis = "" if max_storage_verified else " (an unverified figure)"
        flags.append(
            f"Holds {storage_m3:,.0f} m³, above the {max_storage_m3:,.0f} m³ this kind "
            f"of structure is kept under{basis}.")

    if max_catchment_ha is not None and catchment_ha is not None \
            and catchment_ha > float(max_catchment_ha):
        flags.append(
            f"Takes {catchment_ha:.1f} ha of catchment, over the {max_catchment_ha:g} ha "
            f"(30 ac) NRCS CPS 638 allows one basin — split it with a basin upslope.")

    classifiable = None
    if storage_m3 is not None and max_height_m is not None:
        classifiable = (float(max_height_m) >= CLASSIFIABLE_DAM_HEIGHT_M
                        and float(storage_m3) >= CLASSIFIABLE_DAM_VOLUME_M3)
        if classifiable:
            flags.append(
                f"At {max_height_m:.1f} m and {storage_m3:,.0f} m³ this is a classifiable "
                f"dam under NZ's Building (Dam Safety) Regulations 2022 (4 m and "
                f"20,000 m³): it needs an engineer, not a plugin figure.")

    if required is None:
        lead = ""
    elif storage_m3 is None:
        lead = (f"Needs {required:,.0f} m³ ({min_storage_m3_per_ha:g} m³/ha × "
                f"{catchment_ha:.1f} ha); storage not measured yet — run Re-analyse "
                f"with Earthworks.")
    elif holds:
        lead = (f"✓ Holds {storage_m3:,.0f} m³ against the {required:,.0f} m³ its "
                f"catchment needs.")
    else:
        lead = ""
    text = " ".join(part for part in [lead, *flags] if part)

    return {
        "required_m3": required,
        "holds": holds,
        "drawdown_hr": drawdown,
        "classifiable_dam": classifiable,
        "flags": flags,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Spacing for a continuous bench system
# ---------------------------------------------------------------------------

def bench_spacing_advisory(slope_pct: float, bench_width: float, *,
                           riser_slope: float = 1.0,
                           mode: str = "reverse",
                           dyke_height: float = 0.0,
                           runoff_mm: float | None = None,
                           capacity_m3_per_m: float | None = None) -> dict:
    """The layout a bench section supports on this ground, and the basis for it.

    A sibling of :func:`spacing_advisory`, not a flag on it. That function has two arms
    — the NRCS terrace rule against scour on the slope *between* features, and capture
    — and for a bench system the first does not exist: benches and risers replace the
    natural slope, so there is no hillside left between them to scour, and the spacing
    of continuous benching simply *is* FAO's terrace width ``W_t``. What remains is the
    water question. FAO answers it in the field with intermittent layouts — one bench
    in every two terrace widths, in every three — and :func:`capture_spacing` is the
    same question transposed: the strip one metre of section holds. The widest layout
    the section supports is therefore ``⌊strip ÷ W_t⌋ + 1``: one bench in every *k*
    terrace widths, the *k − 1* unbenched widths above it draining onto it.

    Returns ``slope_pct``, ``vertical_interval_m`` (FAO's VI — the drop between benches,
    not the NRCS interval), ``terrace_width_m``, ``capture_spacing_m`` (``None`` without
    a storm and a section), ``layout_every`` (*k*; 1 is continuous benching),
    ``recommended_spacing_m`` (``k × W_t``), ``governing`` (``"capture"`` when a storm
    sized it, ``"none"`` otherwise) and a ``text`` that names its basis. Ground the
    bench cannot be cut into (see :func:`bench_geometry`) is reported in ``text`` with
    the geometry fields ``None``, never raised — an advisory has nowhere to raise to.

    A bench drawn alone still reads short in its demand check against the measured
    catchment, exactly as a swale drawn alone does, because until benches are added
    above it the whole hill drains to it. This is what says how many to add.
    """
    from .bench import bench_geometry

    try:
        geom = bench_geometry(bench_width, slope_pct, riser_slope=riser_slope,
                              mode=mode, dyke_height=dyke_height)
    except ValueError as exc:
        return {
            "slope_pct": float(slope_pct),
            "vertical_interval_m": None,
            "terrace_width_m": None,
            "capture_spacing_m": None,
            "layout_every": None,
            "recommended_spacing_m": None,
            "governing": "none",
            "text": f"No bench layout at {slope_pct:.1f}%: {exc}.",
        }

    w_t = geom.terrace_width
    capture_m = None
    if runoff_mm is not None and capacity_m3_per_m is not None:
        capture_m = capture_spacing(runoff_mm, capacity_m3_per_m)

    if capture_m is not None and math.isfinite(capture_m) and w_t > 0:
        every = int(capture_m // w_t) + 1
        governing = "capture"
    else:
        every = 1
        governing = "none"
    recommended = every * w_t

    basis = (
        f"FAO 13/3: a {bench_width:.1f} m bench on {slope_pct:.1f}% ground drops "
        f"{geom.vertical_interval:.2f} m to the next and, with its riser, takes "
        f"{w_t:.2f} m of hillside."
    )
    if governing == "capture":
        text = (
            f"{basis} The section holds the storm off a {capture_m:.1f} m strip, so one "
            f"bench in every {every} terrace widths ({recommended:.1f} m apart) is the "
            f"widest layout it supports."
        )
    elif capture_m is not None:
        text = (
            f"{basis} There is no runoff to catch, so capture does not govern; "
            f"continuous benching — one bench in every terrace width — is assumed."
        )
    else:
        text = (
            f"{basis} No storm or section was supplied, so continuous benching — one "
            f"bench in every terrace width — is assumed."
        )

    return {
        "slope_pct": float(slope_pct),
        "vertical_interval_m": geom.vertical_interval,
        "terrace_width_m": w_t,
        "capture_spacing_m": capture_m,
        "layout_every": every,
        "recommended_spacing_m": recommended,
        "governing": governing,
        "text": text,
    }
