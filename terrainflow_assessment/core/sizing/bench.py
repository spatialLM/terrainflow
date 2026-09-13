"""
bench.py — the FAO continuous-bench geometry chain (Conservation Guide 13/3, §6.1).

One chain for every bench-shaped earthwork: the cutback swale (a level bench with a
dyke along its outer edge) and the bench terrace (reverse-sloped, or outward-sloped in
dry country). Given the bench width, the ground slope and the riser slope it returns the
vertical interval between benches, the riser, the terrace width one bench and its riser
occupy, the metres of bench per hectare, and FAO's cut-and-fill figures.

    VI  = S·W_b / (100 − S·U)          vertical interval between benches, m
    RH  = 0.05·W_b   (reverse)          rise across the bench, m
        = −0.03·W_b  (outward)
        = 0          (level)
    H_r = VI + RH + DH                  riser height, m (DH: the dyke of a level bench)
    D_c = (W_b·S/100 + RH) / 2          depth of cut at the inner edge, m
    W_r = H_r · U                       riser width, m
    W_t = W_b + W_r                     terrace width — one bench and one riser, m
    L   = 10 000 / W_t                  bench length per hectare, m/ha
    C   = W_b · H_r / 8                 FAO's cut section, m²/m
    V   = L · C                         cut (= fill) per hectare, m³/ha

``S`` is the ground slope in percent, ``U`` the riser slope as horizontal run per unit
rise (1.0 for a machine-built earth riser, 0.75 hand-made earth, 0.5 hand-made rock),
``DH`` the dyke height of a level bench and 0 otherwise. The dyke is fill placed on the
finished platform, so it raises the riser but takes no part in the cut balance, which is
why it appears in ``H_r`` and not in ``D_c``.

**Rounding is part of the definition.** FAO's Table 1 was worked by hand to two
decimals, and it reconciles only when each step is rounded to two decimals before it
feeds the next: carry ``W_t`` unrounded and the 3.00 m bench at 30 % gives 2510 m/ha
against the table's 2513. Every step here is therefore computed in ``Decimal`` and
rounded half-up to 2 dp — never ``round()`` on a float, whose ties fall either way
(1.0275 rounds to 1.02 as a float and to 1.03 on paper) — with ``L`` and ``V`` rounded to
whole numbers. The one figure the table itself gets wrong is pinned in the tests: its
4.00 m row prints ``W_t`` 5.03 and ``L`` 1989, and 10 000 / 5.03 is 1988.

This is the **continuous** chain only. FAO's discontinuous chain (orchard terraces,
hillside ditches) shares its symbols and differs in three places, and applying it to a
continuous bench halves the cut; a discontinuous type would need its own chain.

Deliberately not built on ``primitives.contour_spacing``. That is the NRCS terrace rule
``HI = VI / S`` — a spacing *between* independent features on the natural slope — and
a continuous bench system has no natural slope left between its benches: the spacing
*is* ``W_t``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

#: The three ways a bench can be tilted across its width.
BENCH_MODES = ("level", "reverse", "outward")

#: FAO's fixed cross-bench grades: 5 % back into the hill, 3 % out over the riser.
REVERSE_GRADE = 0.05
OUTWARD_GRADE = 0.03

#: Riser slopes, horizontal run per unit rise, by how the riser is built (FAO 13/3 §6.1).
RISER_MACHINE_EARTH = 1.0
RISER_HAND_EARTH = 0.75
RISER_HAND_ROCK = 0.5

_CENT = Decimal("0.01")
_ONE = Decimal("1")


def _dec(x) -> Decimal:
    """A float as the decimal its shortest repr names — 1.0275, not 1.02749999…"""
    return Decimal(repr(float(x)))


def _r2(x: Decimal) -> Decimal:
    return x.quantize(_CENT, rounding=ROUND_HALF_UP)


def _r0(x: Decimal) -> Decimal:
    return x.quantize(_ONE, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class BenchResult:
    """One bench and its riser, by FAO's continuous chain. All lengths in metres."""
    bench_width: float          # W_b — the platform, as drawn
    slope_pct: float            # S — the natural ground across the bench
    riser_slope: float          # U — run per unit rise of the riser face
    mode: str                   # "level" | "reverse" | "outward"
    dyke_height: float          # DH — the dyke of a level bench; 0 otherwise
    vertical_interval: float    # VI — drop from one bench to the next
    edge_rise: float            # RH — rise across the bench (negative when outward)
    riser_height: float         # H_r
    depth_of_cut: float         # D_c — at the inner (uphill) edge
    riser_width: float          # W_r
    terrace_width: float        # W_t — one bench and one riser
    length_per_ha: int          # L — metres of bench in a hectare of hillside
    cut_section: float          # C — FAO's cut per metre of bench, m²
    volume_per_ha: int          # V — cut (= fill) per hectare, m³
    min_dimension: float        # the bench width — the narrowest physical width here


def bench_geometry(bench_width: float, slope_pct: float, *,
                   riser_slope: float = RISER_MACHINE_EARTH,
                   mode: str = "reverse",
                   dyke_height: float = 0.0) -> BenchResult:
    """FAO's continuous-bench chain for one bench of ``bench_width`` on ``slope_pct``.

    Raises ``ValueError`` for a non-positive width, an unknown ``mode``, or ground at
    least as steep as the riser (``S·U ≥ 100``): FAO's formula puts the next bench's
    riser toe at or beyond the top of this one, so the design does not exist. A caller
    showing derived rows catches that and shows a dash, the way ``pond_volume_frustum``'s
    converging geometry is handled.

    Flat ground is not an error — the vertical interval is zero and the riser is only
    whatever the bench itself rises by.
    """
    if mode not in BENCH_MODES:
        raise ValueError(f"unknown bench mode {mode!r}; one of {BENCH_MODES}")
    if bench_width <= 0:
        raise ValueError("bench width must be positive")
    if dyke_height < 0:
        raise ValueError("dyke height cannot be negative")
    if riser_slope < 0:
        raise ValueError("riser slope cannot be negative")

    w_b = _dec(bench_width)
    s = _dec(max(0.0, float(slope_pct)))
    u = _dec(riser_slope)
    d_h = _dec(dyke_height)

    denominator = Decimal(100) - s * u
    if denominator <= 0:
        raise ValueError(
            f"a {riser_slope:g}:1 riser cannot be cut into ground at {slope_pct:g} % — "
            f"the riser is no steeper than the hillside, so the benches never meet"
        )

    vi = _r2(s * w_b / denominator)
    if mode == "reverse":
        rh = _r2(w_b * _dec(REVERSE_GRADE))
    elif mode == "outward":
        rh = -_r2(w_b * _dec(OUTWARD_GRADE))
    else:
        rh = Decimal(0)
    h_r = _r2(vi + rh + d_h)
    fall = _r2(w_b * s / Decimal(100))
    d_c = _r2((fall + rh) / Decimal(2))
    w_r = _r2(h_r * u)
    w_t = _r2(w_b + w_r)
    length = _r0(Decimal(10000) / w_t)
    section = _r2(w_b * h_r / Decimal(8))
    volume = _r0(length * section)

    return BenchResult(
        bench_width=float(w_b),
        slope_pct=float(s),
        riser_slope=float(u),
        mode=mode,
        dyke_height=float(d_h),
        vertical_interval=float(vi),
        edge_rise=float(rh),
        riser_height=float(h_r),
        depth_of_cut=float(d_c),
        riser_width=float(w_r),
        terrace_width=float(w_t),
        length_per_ha=int(length),
        cut_section=float(section),
        volume_per_ha=int(volume),
        min_dimension=float(w_b),
    )
