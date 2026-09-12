"""
reporting.py — report data containers, verification maths and shared wording.

Provides:
  BaselineReport         — data from a baseline (no earthworks) analysis
  PostInterventionReport — data from a post-earthworks analysis + simulation
  VerificationResult     — terrain-vs-analytic storage check
  ComparisonResult       — computed before/after metrics
  compare()              — compute comparison metrics
  build_verification()   — the non-circular storage check
  format_live_assessment() — the panel's Live Assessment readout
  round_volume/fmt_*/drain_wording/... — wording and rounding shared by the
                           panel and both report renderers

**There is no HTML generator here any more.** It used to hold ``export_html()``,
a 300-line f-string that built its own document from a ``ComparisonResult`` —
a second, divergent report that needed a fill simulation and said different
things from the PDF. Both formats now render the one document model:

    modules/report_model.py   what the report says      (pure)
    modules/report_charts.py  its charts                (pure)
    modules/report_html.py    -> HTML                   (pure)
    qgis/adapters/layout_pdf.py -> PDF                  (QGIS)

``tests/test_report_renderer_parity.py`` asserts the two renderers cover exactly
the same section types, so they cannot drift apart again.
"""

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report data containers
# ---------------------------------------------------------------------------

@dataclass
class BaselineReport:
    """Results from the baseline (no earthworks) analysis run."""
    site_name: str = "Unnamed Site"
    dem_path: str = ""
    crs: str = ""
    cell_size_m: float = 1.0
    catchment_area_ha: float = 0.0
    rainfall_mm: float = 0.0
    duration_hr: float = 1.0
    cn: float = 70.0
    runoff_mm: float = 0.0
    total_runoff_m3: float = 0.0         # total runoff generated
    exit_volume_m3: float = 0.0          # total volume exiting the site
    peak_outflow_ls: float = 0.0         # peak flow rate at exit (L/s)
    peak_outflow_time_hr: float = 0.0    # time of peak (hr from storm start)
    exit_points: list[dict] = field(default_factory=list)
    # Timestep series for hydrograph
    timestep_table: list[dict] = field(default_factory=list)


@dataclass
class PostInterventionReport:
    """Results from the post-earthworks analysis + fill simulation."""
    exit_volume_m3: float = 0.0
    peak_outflow_ls: float = 0.0
    peak_outflow_time_hr: float = 0.0
    total_infiltrated_m3: float = 0.0
    earthwork_summary: list[dict] = field(default_factory=list)
    timestep_table: list[dict] = field(default_factory=list)
    exit_points: list[dict] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Non-circular check: terrain-derived ponding vs analytic capacity (spec §4)."""
    analytic_total_m3: float = 0.0       # Σ analytic capacity over storage features
    terrain_total_m3: float = 0.0        # Σ(earthworks ponding − baseline ponding), floored ≥0
    delta_m3: float = 0.0                # terrain − analytic
    delta_pct: float = 0.0               # delta as % of analytic (0 when analytic == 0)
    unattributed_m3: float = 0.0         # terrain ponding not tied to any feature footprint
    per_feature: list[dict] = field(default_factory=list)  # {name, analytic_m3, terrain_m3|None, delta_pct|None, routing_only}
    # Features whose pools are one pool. Each entry is the only place that water is
    # counted — its members carry terrain_m3 None, because neither of them holds it
    # alone. {names, rasterisable_m3, terrain_m3, delta_pct|None}
    merged_groups: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    # None when baseline ponding was subtracted; otherwise why it could not be, in
    # which case every terrain figure still carries the water that ponded naturally.
    baseline_uncorrected: Optional[str] = None


@dataclass
class ComparisonResult:
    """Computed before/after metrics."""
    captured_pct: float = 0.0           # % of runoff now retained on-site
    exit_reduction_pct: float = 0.0     # % reduction in exit volume
    peak_reduction_pct: float = 0.0     # % reduction in peak outflow rate
    peak_delay_hr: float = 0.0          # hours by which peak is delayed
    baseline: Optional[BaselineReport] = None
    post: Optional[PostInterventionReport] = None
    net_cut_m3: float = 0.0             # total soil excavated
    net_fill_m3: float = 0.0            # total material placed
    # cut − fill, both drawn from the sections. A **geometric difference, not a
    # balance**: the two are quoted in different states, because a compacted fill
    # swallows more in-situ soil than its own placed volume. The balance is
    # ``mass_haul.earthwork_balance``, which converts between the states first, and it
    # is what the report prints. Kept because callers read it — but it must not be
    # labelled "balance" anywhere a user can see.
    net_cut_fill_m3: float = 0.0
    verification: Optional[VerificationResult] = None  # terrain-vs-analytic (§4)


# ---------------------------------------------------------------------------
# Non-circular verification — terrain-derived ponding vs analytic capacity (§4)
# ---------------------------------------------------------------------------

def raster_ponding_volume(ponding, cell_area_m2, min_depth=0.001):
    """Total ponded volume (m³) from a depth raster = Σ depth·cell_area over ponded cells."""
    import numpy as np
    arr = np.asarray(ponding, dtype="float64")
    ponded = np.where(arr >= min_depth, arr, 0.0)
    return float(ponded.sum() * cell_area_m2)


def impounded_volume(baseline_ponding, dammed_ponding, cell_area_m2):
    """Volume (m³) a dam impounds = the *new* ponding it creates on the DEM.

    ``Σ max(dammed − baseline, 0) × cell_area`` over the two ponding-depth rasters
    (before/after burning the dam to its crest). The positive clip counts only cells
    the dam newly floods; a wall too short to hold water yields ~0. Pure and testable.
    """
    import numpy as np
    dammed = np.asarray(dammed_ponding, dtype="float64")
    baseline = np.asarray(baseline_ponding, dtype="float64")
    new_ponding = np.clip(dammed - baseline, 0.0, None)
    return float(new_ponding.sum() * cell_area_m2)


# ---------------------------------------------------------------------------
# Live Assessment panel readout (design tier)
# ---------------------------------------------------------------------------

# Capture-% traffic light: green ≥ 80, amber ≥ 40, red below.
_LIVE_GOOD, _LIVE_MID, _LIVE_BAD = "#1e8449", "#b9770e", "#c0392b"
_LIVE_MUTED = "#566573"
_LIVE_FILL = "#2e86c1"

# ---------------------------------------------------------------------------
# The capture band — one definition, three readouts
# ---------------------------------------------------------------------------
# The panel scorecard, the live assessment and the printed report all grade the
# same capture percentage, and a design that reads green on screen and amber on
# paper is a bug the reader has no way to resolve. The thresholds and the three
# colours live here, beside the wording they are graded alongside, for the same
# reason ``round_volume`` does.

CAPTURE_GOOD_PCT = 80.0     # at or above: the design holds the storm
CAPTURE_FAIR_PCT = 40.0     # at or above: it holds a useful share of it

CAPTURE_COLOURS = {"good": _LIVE_GOOD, "warn": _LIVE_MID, "bad": _LIVE_BAD}


def capture_tone(pct):
    """Grade a capture percentage: ``"good"`` / ``"warn"`` / ``"bad"``."""
    if pct >= CAPTURE_GOOD_PCT:
        return "good"
    if pct >= CAPTURE_FAIR_PCT:
        return "warn"
    return "bad"


def capture_colour(pct):
    """The hex colour for a capture percentage's grade."""
    return CAPTURE_COLOURS[capture_tone(pct)]


def _mini_bar(pct, colour, back="#d6dbdf"):
    """A thin horizontal bar as a Qt-rich-text table (QLabel has no CSS widths)."""
    p = int(max(0.0, min(100.0, pct)))
    cells = []
    if p > 0:
        cells.append(
            f"<td bgcolor='{colour}' width='{p}%'>"
            "<span style='font-size:3px;'>&nbsp;</span></td>"
        )
    if p < 100:
        cells.append(
            f"<td bgcolor='{back}' width='{100 - p}%'>"
            "<span style='font-size:3px;'>&nbsp;</span></td>"
        )
    return (
        "<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        + "".join(cells) + "</tr></table>"
    )


# ---------------------------------------------------------------------------
# Shared wording and rounding — the panel and the report say the same thing
# ---------------------------------------------------------------------------
# These live beside format_live_assessment rather than in a report-only module
# on purpose: both readouts describe the same BalanceResult, and two copies of
# "how do we phrase a fill percentage" is how the panel and the printed report
# end up disagreeing about the same number.

def round_volume(m3):
    """Round a volume to the precision the model has actually earned.

    The balance reports to 0.1 m³. Printing that claims a survey nobody did, and
    one over-precise figure invites the reader to distrust every other one. Below
    10,000 m³ round to the nearest 10; above it, to the nearest 100.

    A NaN or infinity is treated as no figure at all — the same as ``None``, so it
    prints as an em dash. Every volume in the document comes through here, and
    ``int(round(nan))`` raises, so one unmeasurable number used to abort the whole
    export with "cannot convert float NaN to integer" rather than leave one cell
    blank. A figure that is not a number is not one this document can print.
    """
    if m3 is None:
        return None
    v = float(m3)
    if not math.isfinite(v):
        return None
    step = 10.0 if abs(v) < 10000.0 else 100.0
    return int(round(v / step) * step)


def fmt_volume(m3, unit="m³"):
    """A rounded volume with thousands separators, or an em dash for no figure.

    "No figure" is ``None`` *or* anything :func:`round_volume` cannot round — a NaN
    volume is not a volume.
    """
    rounded = round_volume(m3)
    if rounded is None:
        return "—"
    return f"{rounded:,}{(' ' + unit) if unit else ''}"


def fmt_pct(pct, places=0, signed=False):
    """A percentage as a whole number by default. None becomes an em dash.

    *signed* forces an explicit ``+`` on positives, for a column where the reader
    has to tell an improvement from a regression at a glance. Off by default so
    every existing caller is unchanged; used by the before/after "Change" column,
    whose third row already prints ``{:+.1f} hr`` and whose other two rows would
    otherwise show a 30 % reduction as "-30%" and a 30 % *increase* as "30%".
    """
    if pct is None:
        return "—"
    return f"{float(pct):{'+' if signed else ''}.{places}f}%"


def fmt_area_ha(m2):
    """Square metres in as hectares out, one decimal."""
    if m2 is None:
        return "—"
    return f"{float(m2) / 10000.0:.1f} ha"


def drain_wording(drain_hours):
    """Plain English for ``per_feature['drain_hours']``.

    ``None`` means the feature never empties by soaking — a dam is supposed to
    hold water. Rendering that as ``0``, ``—`` or a blank cell has it read as
    "drains instantly" or "unknown", which is the opposite of what it means.
    """
    if drain_hours is None:
        return "Holds water — does not empty by soaking"
    hours = float(drain_hours)
    if hours < 1.0:
        return "empties within the hour"
    if hours < 24.0:
        return f"empties in about {hours:.0f} hours"
    if hours <= 168.0:
        return f"empties in about {hours / 24.0:.0f} days"
    return "more than a week to empty"


def fill_wording(fill_pct, overflowed):
    """Plain English for how hard a feature is working, plus the number.

    The words alone put three quite different features — 62%, 80% and 94% — in
    one bucket called "well used", and the reader has no way to tell which of
    them is nearly out of room. The percentage is what they act on, so it goes
    in the same cell rather than being left to be inferred from Held ÷ Capacity.
    """
    pct = float(fill_pct or 0.0)
    if overflowed:
        return f"fills and spills · {pct:.0f}%"
    if pct >= 95.0:
        return f"full at this storm · {pct:.0f}%"
    if pct >= 60.0:
        return f"well used · {pct:.0f}%"
    return f"room to spare · {pct:.0f}%"


# Spillway review states, as the reader should see them. The raw keys are
# internal vocabulary; a status column reading "no_datum" tells nobody anything.
SPILLWAY_STATE_WORDING = {
    "ok": "Sized",
    "fail": "Needs attention",
    "no_flow": "No inflow calculated",
    "no_datum": "No ground level yet",
    "undesigned": "No spillway designed",
    "unsited": "Not placed on the map",
    "disabled": "Turned off",
}


def spillway_state_wording(state):
    return SPILLWAY_STATE_WORDING.get(state, str(state or "—"))


def type_label(key):
    """The registry's display label for an earthwork type key.

    ``per_feature`` carries the raw key, and a column of "swale"/"dam" reads as
    a database dump rather than a document.
    """
    try:
        from terrainflow_assessment.core.registry.earthwork_types import get_type
        return get_type(key).label
    except Exception:
        return str(key or "").replace("_", " ").title()


def cut_fill_sentence(cut_m3, fill_m3):
    """The cut/fill balance in words.

    ``+340 m³`` reads as a credit to anyone who is not an engineer, and the sign
    convention is not stated anywhere a landowner would look. Say which way the
    soil goes instead.
    """
    cut = round_volume(cut_m3 or 0.0)
    fill = round_volume(fill_m3 or 0.0)
    if cut is None or fill is None:
        # One of the two is not a number — say so, rather than subtracting it.
        return ("The earthmoving quantities could not be measured for this "
                "design.")
    net = cut - fill
    if abs(net) < 10:
        return (f"About {cut:,} m³ comes out and {fill:,} m³ goes back in — "
                "close enough to balance on site.")
    if net > 0:
        return (f"About {net:,} m³ more soil comes out than goes back in; "
                "that surplus needs somewhere to go on site.")
    return (f"About {abs(net):,} m³ more soil is needed than the excavation "
            "produces; it has to be won from somewhere else on the block.")


def unique_names(rows, key="name"):
    """Feature names, disambiguated where they collide.

    The default name is ``f"{type} {len(manager) + 1}"`` counting *all*
    earthworks, so deleting one and drawing another reproduces an existing name
    — as the earthworks controller already documents. The balance is keyed by
    id, but the verification result is keyed by name, so a collision silently
    attributes one feature's measured storage to another. Returns a list of
    display names positionally matching ``rows``.
    """
    return _lettered([row.get(key) or "Unnamed" for row in rows])


def _suffix(n):
    """0 -> 'a', 25 -> 'z', 26 -> 'aa', 27 -> 'ab' — spreadsheet-column lettering.

    `chr(ord('a') + n)` ran off the end of the alphabet at the 27th duplicate and
    printed `{`, `|`, `}`. Twenty-seven features sharing one name is unlikely; a
    document with a brace in a feature label is not something to ship on the
    strength of unlikely.
    """
    out = ""
    n = int(n)
    while True:
        out = chr(ord("a") + n % 26) + out
        n = n // 26 - 1
        if n < 0:
            return out


def _lettered(names):
    """Disambiguate a list of names positionally, leaving unique ones untouched."""
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    used = {}
    out = []
    for name in names:
        if counts[name] == 1:
            out.append(name)
            continue
        used[name] = used.get(name, 0) + 1
        out.append(f"{name} ({_suffix(used[name] - 1)})")
    return out


def disambiguate(pairs):
    """``[(id, name), ...]`` -> ``{id: display name}``, built **once** per document.

    The report used to call :func:`unique_names` in three places on three different
    inputs — the balance rows twice and the earthwork list once — and not at all in
    the volume ladder. Three inputs means three letterings: "Swale 3 (a)" on one
    page could be a different feature from "(a)" on the next, and the ladder showed
    two identical rows. One map keyed on identity settles it for the whole document.

    Ordering is the caller's, so the letters follow the order the reader meets the
    features in. A duplicate id keeps its first name.
    """
    ordered, seen = [], set()
    for fid, name in pairs:
        if fid in seen:
            continue
        seen.add(fid)
        ordered.append((fid, name or "Unnamed"))
    display = _lettered([name for _fid, name in ordered])
    return {fid: display[i] for i, (fid, _name) in enumerate(ordered)}


def format_live_assessment(result, have_flow):
    """Qt-rich-text HTML for the Live Assessment panel readout.

    ``result`` is a :class:`~terrainflow_assessment.modules.water_balance.BalanceResult`.
    With flow data: a colour-coded capture-% headline + bar, the held/leaves split,
    and a per-feature table (inflow → stored, fill %, ⚠ when overflowing). Without:
    per-feature capacities and a run-baseline hint. Pure string building — testable.
    """
    r = result
    parts = []

    if have_flow:
        colour = capture_colour(r.capture_pct)
        parts.append(
            f"<span style='font-size:20px;font-weight:bold;color:{colour};'>"
            f"{r.capture_pct:.0f}%</span> "
            f"<span style='font-size:11px;color:{_LIVE_MUTED};'>of storm runoff "
            f"captured</span>"
        )
        parts.append(_mini_bar(r.capture_pct, colour))
        stored = max(0.0, r.total_captured_m3 - r.total_infiltration_m3)
        parts.append(
            f"<span style='font-size:11px;color:#2c3e50;'>"
            f"{r.total_captured_m3:,.0f} m³ held ({stored:,.0f} stored + "
            f"{r.total_infiltration_m3:,.0f} soaked in) · "
            f"{r.site_exit_m3:,.0f} m³ leaves site</span>"
        )
    else:
        parts.append(
            f"<span style='font-size:11px;color:{_LIVE_MUTED};'><i>Run baseline "
            "analysis to see storm capture %.</i></span>"
        )

    if r.per_feature:
        rows = []
        for f in r.per_feature:
            if have_flow:
                # water_balance emits direct_/upstream_/total_inflow_m3 — there is no
                # bare 'inflow_m3', and subscripting one crashed this row whenever flow
                # data was present. Total inflow is the figure this row means.
                inflow = f.get("total_inflow_m3", 0.0)
                detail = f"{inflow:,.0f} → {f['stored_m3']:,.0f} m³"
                if f["overflowed"] or f["fill_pct"] >= 100:
                    status = (
                        f"<span style='color:{_LIVE_MID};font-weight:bold;'>⚠ full</span>"
                    )
                else:
                    status = (
                        f"<span style='color:{_LIVE_FILL};'>{f['fill_pct']:.0f}%</span>"
                    )
            else:
                detail = f"{f.get('capacity_m3', 0.0):,.0f} m³"
                status = ""
            rows.append(
                "<tr>"
                f"<td>{f['name']}</td>"
                f"<td align='right'>{detail}</td>"
                f"<td align='right' width='40'>{status}</td>"
                "</tr>"
            )
        parts.append(
            "<table width='100%' cellspacing='0' cellpadding='1' "
            "style='font-size:11px;color:#2c3e50;'>" + "".join(rows) + "</table>"
        )

    parts.append(
        f"<span style='font-size:11px;color:{_LIVE_MUTED};'>"
        f"Capacity {r.total_capacity_m3:,.0f} m³ · Cut {r.total_cut_m3:,.0f} · "
        f"Fill {r.total_fill_m3:,.0f} m³</span>"
    )
    parts.append(
        "<span style='font-size:10px;color:#95a5a6;'>Analytical estimate — verify "
        "with Re-analyse with Earthworks.</span>"
    )
    return "<br>".join(parts)


class PondAttribution(NamedTuple):
    """How a ponding raster divided between the features that made it.

    Three disjoint buckets, and they conserve: ``Σ per_name + Σ group volumes +
    unattributed_m3`` is the whole ponded volume of the raster.
    """
    per_name: dict           # {name: m³} — regions this feature alone touches
    unattributed_m3: float   # regions touching no footprint at all
    groups: list             # [{"names": (…), "volume_m3": …, "overlaps": {name: cells}}]


class PoolGrouping(NamedTuple):
    """Which features share which pools, on one grid.

    :func:`attribute_ponding_volume` and :func:`event_pond_depth` are two views of
    the same water — one totals it, the other draws it — and their docstrings say
    they must agree about what a pool is. They agreed by hand-copy: the same
    labelling, the same overlap counts and the same union-find, written out twice.
    Now there is one of each, and agreement is structural.
    """
    labels: object           # int region ids, 0 = the dry background
    n_regions: int
    touched: list            # rid → {name: overlapping cells}
    members: dict            # root → tuple of names, sorted
    root_of_region: list     # rid → root, or None where the region touches nothing
    regions_of: dict         # root → [rid] this component owns


def group_pools(depth, footprints, min_depth=0.001):
    """Label the ponded regions and join every feature that shares one.

    Joining is transitive: if A and B share one pool and B and C share another, all
    three are one component, because no cut separates A's water from C's. Every
    region a component's members touch belongs to that component, including the ones
    a member holds alone — see :func:`attribute_ponding_volume` for why.
    """
    import numpy as np
    from scipy.ndimage import label

    arr = np.asarray(depth, dtype="float64")
    labels, n_regions = label(arr >= min_depth)

    overlaps = {name: np.bincount(labels[mask], minlength=n_regions + 1)
                for name, mask in footprints}
    touched = [
        {name: int(counts[rid])
         for name, counts in overlaps.items() if counts[rid] > 0}
        for rid in range(n_regions + 1)
    ]

    parent = {name: name for name, _ in footprints}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid in range(1, n_regions + 1):
        names = list(touched[rid])
        for other in names[1:]:
            ra, rb = find(names[0]), find(other)
            if ra != rb:
                parent[rb] = ra

    members = {}
    for name in parent:
        members.setdefault(find(name), []).append(name)
    members = {root: tuple(sorted(names)) for root, names in members.items()}

    root_of_region = [None] * (n_regions + 1)
    regions_of = {}
    for rid in range(1, n_regions + 1):
        if not touched[rid]:
            continue
        root = find(next(iter(touched[rid])))
        root_of_region[rid] = root
        regions_of.setdefault(root, []).append(rid)

    return PoolGrouping(labels=labels, n_regions=n_regions, touched=touched,
                        members=members, root_of_region=root_of_region,
                        regions_of=regions_of)


def attribute_ponding_volume(ponding_diff, cell_area_m2, footprints, min_depth=0.001):
    """Attribute a ponding-difference raster to earthwork footprints.

    Connected ponded regions (``>= min_depth``) are labelled, then sorted by how many
    footprints each one touches:

    * **none** → ``unattributed_m3``.
    * **one** → the whole region, that feature's. Deliberately the whole of it, including
      cells outside the footprint: a dam's pool lies upstream of its wall, and a basin's
      pool spreads past the polygon that was drawn. Clipping to the footprint would
      under-report exactly the features whose storage is impounded rather than excavated.
    * **two or more** → the features are joined, and the region belongs to the set.

    That last case used to award the region entire to whichever footprint overlapped it
    most, which is how a single pool came to be reported twice over: on the Quail Island
    design, Basin 39 and Dam 40 are one structure — the wall sits on the basin's downhill
    lip — and the 3,593 m³ they impound together was credited wholly to Basin 39
    (Δ +211%) while Dam 40, holding 70 cells of the same water, reported Δ −100%. Two more
    pairs did the same. The volume was never wrong; it was in the wrong box, and the
    per-feature deltas were reporting the box, not the burn.

    A set is not split by overlap share either. There is no defensible share: a dam
    contributes a wall, a basin contributes a hole, and the pool is a property of the
    pair. The honest report is the set's total against the set's capacity, which is
    what :func:`build_verification` prints.

    **Joining is transitive, and a joined feature brings its solo pools with it.** The
    sets are the connected components of the graph whose edges are shared regions — so
    if A and B share one pool and B and C share another, all three are one set, because
    there is no cut that separates A's water from C's. And once a feature is in a set,
    *every* region it touches goes to that set, including the ones it holds by itself.
    Leaving those on the feature's own row would compare a part of its water against the
    whole of its capacity, which reads as a deficit that is not there: Dam 3 holds 158 m³
    alone and 321 m³ jointly with Swale 6, and scoring the joint pool against both
    capacities gave Δ −33% when the pair is in fact within 1% of what the grid holds.

    Parameters
    ----------
    ponding_diff : 2-D array — terrain ponding depth (typically earthworks − baseline)
    cell_area_m2 : float
    footprints   : list of (name, bool_mask) — one boolean footprint per feature

    Returns
    -------
    :class:`PondAttribution`
    """
    import numpy as np

    arr = np.asarray(ponding_diff, dtype="float64")
    per_name = {name: 0.0 for name, _ in footprints}
    unattributed = 0.0

    pools = group_pools(arr, footprints, min_depth)
    if pools.n_regions == 0:
        return PondAttribution(per_name, 0.0, [])

    # One pass over the array rather than one per (region × feature). The old form
    # built a full-grid boolean per region and again per feature inside it — on this
    # site, 669 regions × 40 features × 2.8 M cells. Label 0 is the unponded
    # background, so cells under min_depth cannot contribute.
    volumes = np.bincount(pools.labels.ravel(), weights=arr.ravel(),
                          minlength=pools.n_regions + 1) * cell_area_m2

    # Every region goes to the component of whatever it touches. A feature in a
    # multi-feature component contributes its solo pools to that component too, which
    # is what keeps the set's measured volume comparable with the set's capacity.
    grouped = {}
    for rid in range(1, pools.n_regions + 1):
        volume = float(volumes[rid])
        root = pools.root_of_region[rid]
        if root is None:
            unattributed += volume
            continue
        names = pools.members[root]
        if len(names) == 1:
            per_name[names[0]] += volume
            continue
        entry = grouped.setdefault(
            root, {"names": names, "volume_m3": 0.0,
                   "overlaps": {n: 0 for n in names}})
        entry["volume_m3"] += volume
        for name, cells in pools.touched[rid].items():
            entry["overlaps"][name] += cells

    groups = sorted(grouped.values(), key=lambda g: -g["volume_m3"])
    return PondAttribution(per_name, unattributed, groups)


class SpillOver(NamedTuple):
    """Where a pool leaves over the barrier that made it, and along how much of it."""
    name: str
    pour_level_m: float        # the elevation the pool fills to before it escapes
    length_m: float            # metres of this barrier's own crest at that level
    alt_saddle_m: float        # lowest rim point that is NOT this barrier (inf if none)
    pool_volume_m3: float
    mask: object               # bool array — the crest cells water goes over
    # Where the water surface actually stands this event, or None when no event
    # pond was supplied. Everything above is a property of the ground and the
    # structure and holds whatever the storm does; this is the one figure that
    # depends on the storm.
    event_level_m: float = None

    @property
    def overtops_this_event(self):
        """Does the modelled event reach the pour level, or only capacity would?

        None when there is no event pond to answer from — which is not False. A
        barrier that has not been asked is not a barrier that has been cleared.
        """
        if self.event_level_m is None:
            return None
        # A millimetre, matching ``min_depth``: below that the pool is at the pour
        # level as far as anything measured off a metre-grid DEM can tell.
        return self.event_level_m >= self.pour_level_m - 1e-3


def overtopping_spill(ponding, ground, cell_size_m, barriers, built=None,
                      min_depth=0.001, cell_area_m2=None, event_depth=None):
    """Which barriers their own pools pour over, and along what length of crest.

    A pool fills to the lowest point of its rim and leaves there. Where that low point
    is the structure's own crest, the water goes over the wall — the failure mode an
    earth dam has. This finds those cases by measuring the rim, so it answers from the
    burned ground rather than from intent.

    **The length matters as much as the fact.** A level crest does not spill at a point:
    the water surface stays flat as it rises, so it goes over every metre of crest that
    stands at the pour level at once. That length is the crest cells on the pool's rim at
    the pour elevation, and it is what unit discharge (and therefore erosion) has to be
    figured against.

    This used to say the stream layer draws one thread "because D8 sends the whole overflow
    through one cell, steepest descent picking a single neighbour". **Both halves are
    wrong**, measured in Round 14. Flat resolution is already multi-outlet, and Dam 15's
    pool leaves at **23 distinct cells**. What concentrated the flow was that a flat is
    resolved by distance — every cell routes to its *nearest* way out — so the crest cell
    closest to where the inflow channel arrived took **87% of the flux** while 16 of 24
    carried nothing.

    **That half is now fixed** (Round 15). ``crest_routing`` contracts each pond to a mixing
    node and sheds its whole inflow evenly over the cells that discharge, so Dam 15's level
    band reads a uniform 962.1 — a ratio of exactly 1.00x, a true weir — against
    ``min 1 / median 1 / max 36,782`` before. What remains is the **threshold**: the Streams
    layer draws ``accumulation > stream_threshold``, and a crest cell carrying its 1/n share
    clears a contributing-area threshold the whole catchment only just exceeds even less
    easily than before. A wall that draws no channel is now the layer's cut-off saying so,
    and nothing else.

    The length measured here is unaffected by any of that — it is read off the burned rim,
    never from flow direction — which is why it was worth reporting even while the
    explanation beside it was wrong.

    **Walk the pool-facing row, not the whole level band** — this is a length, and the band
    is not one. Dam 15's wall stands 24 cells along the water and **49** cells of its level
    band can discharge, but that larger figure is length × *thickness*: the wall is ~25 cells
    long and 2–3 cells thick, and both faces drain. Counting it would inflate ``length_m``,
    shrink the unit discharge that erosion is judged by, and do so in the unsafe direction.
    The pool-facing row is the crest run; the ``drawn_len`` cap then absorbs the √2 a
    diagonal cell path over-counts by.

    ``barriers`` is ``[(name, crest_mask, drawn_length_m)]`` — *crest_mask* being the
    cells this feature raised, not its whole footprint. The reported length is capped at
    ``drawn_length_m`` because a wall cannot overtop along more of itself than it has;
    without the cap a diagonal run of cells over-counts by up to √2.

    ``built`` is every cell the burn raised anywhere. It is what ``alt_saddle_m`` — the
    next way out if this crest were lifted — is measured against, and passing it matters:
    a dam's keyed returns are raised ground that its recorded mask does not cover, so
    without it the "natural saddle" comes back as the dam's own wall at its own level,
    and the advice reads *raise the crest 0.00 m*. Another feature's embankment is not a
    natural saddle either. Defaults to the barrier's own crest, which is the weaker
    answer but never a wrong-shaped one.

    ``ponding`` is the **full-capacity** raster, and everything above is therefore an
    answer about the structure rather than about the storm: *filled to its spill point,
    this pool leaves over its own crest*. That is the right reference state for the
    question the layer exists to ask — a wall with no freeboard is a fault whether or
    not this particular event finds it — but on the map it sat one row from an event
    water line drawn well below the crest, and read as a claim that the modelled storm
    was going over the top. It was not.

    ``event_depth`` is what closes that gap: the event pond depth raster
    (:func:`event_pond_depth`) on the same grid. Each spill then also carries
    ``event_level_m``, the elevation the water actually reaches, so a caller can tell
    *this event overtops it* from *filling it would*. Omit it and ``event_level_m`` is
    None, which the callers read as "not asked" rather than as "no".

    Returns a list of :class:`SpillOver`, one per barrier that pours over itself,
    deepest pool first. A barrier whose pool escapes elsewhere is simply absent.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, label

    pond = np.asarray(ponding, dtype="float64")
    bed = np.asarray(ground, dtype="float64")
    # A length and an area are two different questions and one number answers both
    # only on a square grid: `cell_size_m` measures crest length, `cell_area_m2`
    # converts a pool's cells to cubic metres. Defaulted from the length so callers
    # on square grids are unchanged.
    cell_area = (float(cell_area_m2) if cell_area_m2 is not None
                 else float(cell_size_m) ** 2)
    out = []
    if pond.shape != bed.shape:
        return out

    labels, n_pools = label(pond >= min_depth)
    if n_pools == 0:
        return out

    made_ground = None if built is None else np.asarray(built, dtype=bool)
    if made_ground is not None and made_ground.shape != pond.shape:
        made_ground = None

    event = None if event_depth is None else np.asarray(event_depth, dtype="float64")
    if event is not None and event.shape != pond.shape:
        event = None

    for name, crest, drawn_len in barriers:
        crest = np.asarray(crest, dtype=bool)
        if crest.shape != pond.shape or not crest.any():
            continue
        # Pools this barrier stands against: the ones its crest cells touch.
        touching = set(np.unique(labels[binary_dilation(crest)])) - {0}
        for pid in touching:
            pool = labels == pid
            rim = binary_dilation(pool) & ~pool
            if not rim.any():
                continue
            pour = float(bed[rim].min())
            # Its own crest, at the level the pool actually leaves at.
            own = rim & crest & (bed <= pour + 1e-6)
            if not own.any():
                continue          # this pool escapes somewhere else — not this barrier
            other = rim & ~(crest if made_ground is None else made_ground)
            alt = float(bed[other].min()) if other.any() else float("inf")
            length = min(float(own.sum()) * float(cell_size_m),
                         float(drawn_len) if drawn_len else float("inf"))
            # The water surface this event puts in *this* pool. A pool is level, so
            # any wet cell answers it and the max is taken only to be immune to the
            # pool's edge cells. Dry this event → the floor, which is below the pour
            # level by construction and so reads as "does not reach the crest".
            event_level = None
            if event is not None:
                filled = pool & (event > min_depth)
                event_level = (float((bed + event)[filled].max()) if filled.any()
                               else float(bed[pool].min()))
            out.append(SpillOver(
                name=name,
                pour_level_m=pour,
                length_m=length,
                alt_saddle_m=alt,
                pool_volume_m3=float(pond[pool].sum() * cell_area),
                mask=own,
                event_level_m=event_level,
            ))
    out.sort(key=lambda s: -s.pool_volume_m3)
    return out


def level_for_volume(ground, cell_area_m2, volume_m3, ceiling=None):
    """The water-surface elevation at which *volume_m3* stands over *ground*.

    A stage-storage curve inverted, solved exactly rather than by bisection. Sort the
    bed elevations; the volume held when the surface sits on the k-th of them is
    ``cell_area × Σ_{i<k} (g_k − g_i)``. That is monotone and piecewise linear in
    level, so the bracketing pair is one ``searchsorted`` away and the level between
    them follows from the count of cells wet there — no tolerance, no iteration count.

    ``ceiling`` caps the answer at the pool's own spill level. A feature the balance
    credits with more water than its pond can hold is then drawn brim-full rather than
    standing above the ground the surplus would in fact run over: that surplus is
    overflow, and overflow is somewhere else's water.

    Returns None for an empty region, which the caller skips.
    """
    import numpy as np

    g = np.sort(np.asarray(ground, dtype="float64").ravel())
    if g.size == 0:
        return None
    if volume_m3 <= 0 or cell_area_m2 <= 0:
        return float(g[0])

    below = np.arange(g.size)                      # cells strictly under g[k]
    held = (g * below - np.concatenate(([0.0], np.cumsum(g)[:-1]))) * cell_area_m2
    idx = max(int(np.searchsorted(held, float(volume_m3), side="right")) - 1, 0)
    wet = idx + 1                                  # cells wet once the surface passes
    level = g[idx] + (float(volume_m3) - held[idx]) / (wet * cell_area_m2)
    if ceiling is not None:
        level = min(level, float(ceiling))
    return float(level)


def event_pond_depth(ponding, ground, cell_area_m2, footprints, stored_by_name,
                     existing=None, min_depth=0.001, pools=None):
    """Where the water actually stands for *this* event, as a depth raster.

    ``ponding`` is the full-capacity depth raster — every hollow filled to its spill
    point, which is what the burn's ponding layer is — and ``ground`` the surface under
    it. Each pool is re-filled with only the water the balance delivers, by solving for
    the level that holds it (:func:`level_for_volume`). A part-full pond therefore comes
    out **smaller and shallower**, standing in the bottom of its basin. Scaling the full
    pond's depth by a fill fraction instead, as the simulation's frame layer does, keeps
    the full pond's footprint and paints water up banks it never reaches.

    Pools are attributed exactly as :func:`attribute_ponding_volume` attributes them —
    same labelling, same transitive joining — because the two must agree about what a
    pool is. Where a joined set owns more than one pool there is no defensible way to
    say which of them the water is in, so the set's volume is split between them in
    proportion to their capacity. That split is a presentational choice and nothing
    downstream reads it as a measurement.

    ``existing`` is the pre-earthwork ponding on the same grid. Its volume is added back
    per pool, because ``stored_by_name`` is water held *over and above* what ponded
    there naturally (that is the basis ``feature_storage`` measures capacity on) and a
    dam sitting in a wet hollow would otherwise be drawn emptier than the ground is.

    Parameters
    ----------
    ponding        : 2-D array — full-capacity ponding depth (m)
    ground         : 2-D array — bed elevation under it (m), the burned DEM
    cell_area_m2   : float
    footprints     : list of (name, bool_mask), as :func:`attribute_ponding_volume` takes
    stored_by_name : {name: m³} — what the balance says each feature holds
    existing       : 2-D array or None — baseline ponding depth (m)
    pools          : a :class:`PoolGrouping` over the same ``ponding`` and
        ``footprints``, when the caller already has one. The grouping depends on
        geometry alone, so a caller redrawing the same pools at many fill levels —
        the simulation's playback does, twice a second — computes it once instead of
        relabelling the whole grid per frame. Omit and it is built here.

    Returns
    -------
    2-D float array — event ponding depth (m), zero where the water does not reach.
    """
    import numpy as np

    arr = np.asarray(ponding, dtype="float64")
    bed = np.asarray(ground, dtype="float64")
    out = np.zeros_like(arr)
    if arr.shape != bed.shape:
        return out

    # The same pools, from the same helper, as attribute_ponding_volume — which is
    # what makes "the two must agree about what a pool is" true by construction.
    if pools is None:
        pools = group_pools(arr, footprints, min_depth)
    if pools.n_regions == 0:
        return out
    labels, n_regions = pools.labels, pools.n_regions
    members, regions_of = pools.members, pools.regions_of

    capacity = np.bincount(labels.ravel(), weights=arr.ravel(),
                           minlength=n_regions + 1) * cell_area_m2
    if existing is None:
        natural = np.zeros_like(capacity)
    else:
        natural = np.bincount(
            labels.ravel(), weights=np.clip(
                np.asarray(existing, dtype="float64"), 0.0, None).ravel(),
            minlength=n_regions + 1) * cell_area_m2

    # One sort of the whole grid buys every region's cells. Masking per region instead
    # is the trap ``attribute_ponding_volume`` documents: hundreds of regions each
    # walking millions of cells.
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(n_regions + 2))

    bed_flat, pond_flat, out_flat = bed.ravel(), arr.ravel(), out.ravel()
    for root, rids in regions_of.items():
        stored = sum(float(stored_by_name.get(n, 0.0) or 0.0) for n in members[root])
        caps = np.array([capacity[r] for r in rids], dtype="float64")
        total = float(caps.sum())
        for rid, cap in zip(rids, caps):
            share = natural[rid] + (stored * (cap / total) if total > 0 else 0.0)
            if share <= 0:
                continue
            cells = order[starts[rid]:starts[rid + 1]]
            g = bed_flat[cells]
            # A cell with no ground elevation cannot be said to hold water, and it
            # must not decide the answer for the cells around it either: one NaN
            # made ``spill`` NaN for the whole pool, and left a NaN in the depth
            # raster that every downstream sum then had to know to avoid. The rest
            # of this module masks nodata rather than propagating it — `raster_
            # ponding_volume` and `attribute_ponding_volume` both do — so this
            # does the same and the pool is solved over the ground that exists.
            solid = np.isfinite(g)
            if not solid.all():
                cells, g = cells[solid], g[solid]
                if cells.size == 0:
                    continue
            spill = float((g + pond_flat[cells]).max())
            level = level_for_volume(g, cell_area_m2, share, ceiling=spill)
            if level is None:
                continue
            out_flat[cells] = np.clip(level - g, 0.0, None)
    return out


# A per-feature gap between the drawn geometry and what the grid can represent,
# beyond this fraction, is worth naming explicitly in the table. It gates two things
# that must agree: the caveat sentences, and the per-row ``section_overstated`` flag
# the panel and the report style from. A measured gap is the right trigger rather than
# a width-vs-cell-size proxy — the burner levels a flat floor at full depth for any
# feature carrying no batter run, so a wide swale is misrepresented exactly as much as
# a narrow one, while a genuinely vertical-walled channel of any width is not
# misrepresented at all.
_RESOLUTION_CAVEAT_THRESHOLD = 0.10


def build_verification(analytic_by_name, terrain_by_name, baseline_total_m3,
                       earthworks_total_m3, min_dims, cell_size,
                       breakdowns=None, existing_by_name=None, merged_groups=None,
                       unattributed_m3=0.0):
    """Assemble the terrain-vs-analytic verification (site headline + per-feature).

    Site terrain-derived storage = ``earthworks_total − baseline_total`` (floored ≥0) —
    isolates the earthwork effect (step-5 before/after integrity). Per feature, a
    ``min_dimension`` below ``cell_size`` is flagged ``routing_only`` and gets no
    independent volume claim (spec §4): sub-cell features validate placement/routing
    only, not storage.

    ``breakdowns`` maps each key to
    :func:`~terrainflow_assessment.modules.earthwork_design.capacity_breakdown`. When
    supplied, the delta is measured against what the **grid can represent**
    (``rasterisable``) rather than the design capacity, which is what makes it
    interpretable: a non-zero delta then means the burn is wrong, and nothing else.

    Unrelated gaps used to be summed into one percentage — a burn error and the grid's
    distortion of the drawn shape. That is why a headline of "Verified · Δ −38%" carried
    no actionable meaning. Each is now reported separately, and the first is *calculated*
    while the rest are *measured*, which is the division that matters:

    ``section`` → ``cut``          did the grid hold the section you drew?
    ``section`` → ``rasterisable`` what the bank and the hillside add (``impoundment_m3``)
    ``rasterisable`` → ``terrain``   interaction with neighbouring features

    The third of those is routinely the largest and it is not an error. A companion berm
    keyed into its banks holds water above natural ground, so what a swale impounds is
    commonly half again what its cross-section says — which no cross-section can predict,
    because it depends on the hillside.

    Past ``_RESOLUTION_CAVEAT_THRESHOLD`` the *grid* gap is carried on the row itself as
    ``section_overstated`` / ``section_gap_pct``, because at that size the burn did not
    cut the section it was given and the geometry is the capacity to read. The panel and
    the report both style from those two keys, so a near-zero Δ on such a row is never
    presented as reassurance about a section it did not hold.

    ``existing_by_name`` is the *baseline* ponding attributed to the same footprints —
    water already sitting there before any earthwork. Every other figure here is
    marginal (what the design adds), which is the right default for a before/after
    tool, but it leaves a fair question unanswered for a feature built in a hollow:
    a dam reported at 726 m³ may sit in ground already holding 555 m³, so the pool
    the owner actually sees is 1,281 m³. Carrying ``existing_m3`` and ``total_m3``
    alongside lets the table say all three rather than make the reader choose which
    one "storage" meant. ``total = existing + terrain`` by construction, so the three
    can never disagree.

    ``merged_groups`` comes from :class:`PondAttribution` and names the sets of features
    that share one pool. A member of such a set gets ``terrain_m3`` / ``delta_pct`` of
    ``None`` — the same treatment a sub-cell feature gets, and for the same reason: there
    is a measurement, but it is not a measurement *of this feature*. The comparison is
    made once, for the set, in ``merged_groups``. Anything else double-counts the water
    or credits it to whichever member happens to overlap it most, which is how one pool
    reported +211% on a basin and −100% on the dam holding it back.
    """
    breakdowns = breakdowns or {}
    existing_by_name = existing_by_name or {}
    merged_groups = list(merged_groups or [])
    merged_by_name = {}
    for group in merged_groups:
        for name in group.get("names", ()):
            merged_by_name[name] = tuple(
                n for n in group["names"] if n != name)

    def _reference(key, fallback):
        b = breakdowns.get(key)
        if b and b.get("rasterisable"):
            return float(b["rasterisable"])
        return float(fallback)

    analytic_total = float(sum(analytic_by_name.values()))
    reference_total = float(sum(
        _reference(k, v) for k, v in analytic_by_name.items()
    ))
    terrain_total = max(0.0, earthworks_total_m3 - baseline_total_m3)
    delta_m3 = terrain_total - reference_total
    delta_pct = (delta_m3 / reference_total * 100.0) if reference_total > 0 else 0.0

    per_feature = []
    resolution_flagged = []
    for name, analytic_m3 in analytic_by_name.items():
        min_dim = min_dims.get(name)
        routing_only = min_dim is not None and cell_size > 0 and min_dim < cell_size
        b = breakdowns.get(name) or {}
        reference = _reference(name, analytic_m3)
        existing = float(existing_by_name.get(name, 0.0))

        shares_pool_with = merged_by_name.get(name)

        if routing_only or shares_pool_with:
            # Two different reasons to withhold a measured volume, one rule: report a
            # figure only where it describes this feature. A sub-cell feature has no
            # measurement of its own; a feature sharing a pool has one that is not its
            # own. The group row below makes the comparison the members cannot.
            terrain_m3 = None
            feat_delta_pct = None
        else:
            terrain_m3 = float(terrain_by_name.get(name, 0.0))
            feat_delta_pct = (
                (terrain_m3 - reference) / reference * 100.0 if reference > 0 else None
            )

        geometric = float(b.get("geometric", analytic_m3))
        # **The trench, against the trench.** ``resolution_penalty_m3`` is now
        # ``cut − section``: the excavation the burn actually made against the section
        # that was drawn, and nothing else. Measured against ``geometric`` instead it
        # would also carry the companion berm, and against ``rasterisable`` it would
        # carry the whole impoundment — a berm doing exactly its job would report as a
        # resolution failure of +80% and upward, which is the noise this flag has twice
        # had to be rescued from.
        section = float(b.get("section_m3", geometric))
        berm_credit = float(b.get("berm_credit_m3", 0.0))
        penalty = float(b.get("resolution_penalty_m3", 0.0))
        cut = b.get("cut_m3")
        excavation = b.get("excavation_m3")
        # What the bank and the hillside add beyond the drawn trench. Positive and often
        # large; not an error, and deliberately its own key so it can never be mistaken
        # for one.
        impoundment = float(b.get("impoundment_m3", 0.0))
        barrier = bool(b.get("barrier_impounded", False))
        gap_pct = penalty / section * 100.0 if section > 0 else None
        gap_material = (gap_pct is not None
                        and abs(penalty) / section > _RESOLUTION_CAVEAT_THRESHOLD)
        if gap_material:
            resolution_flagged.append((name, gap_pct))

        # One predicate, two consumers: the caveat sentences above and the per-row flag
        # below cannot disagree about which features are affected.
        #
        # Past this gap the at-grid and measured columns have stopped describing storage
        # and describe the trench the burn cut. Flagging the row is what stops a
        # near-zero Δ — which only ever compared measured against the grid — from
        # reading as reassurance about a volume it never tested.
        #
        # A dam is excluded because it has no drawn cross-section to be overstated
        # against (its penalty is forced to zero anyway), a sub-cell feature because it
        # claims no measured volume in the first place, and a feature sharing a pool
        # because it has no Δ of its own to qualify. The flag styles a delta and names
        # itself in a roll-call under the table; setting it on a row whose delta reads
        # "—" would send the reader hunting for a mark that is not there.
        overstated = bool(gap_material and not routing_only and not barrier
                          and not shares_pool_with)

        per_feature.append({
            "name": name,
            # The whole drawn shape, brim-full. It carried a blanket 20% freeboard
            # deduction until that allowance was removed, so this and ``geometric_m3``
            # are now the same quantity for anything with a drawn section.
            "analytic_m3": float(analytic_m3),
            "geometric_m3": geometric,                   # the drawn shape
            # What geometric is made of. The grid columns describe the trench only, so
            # a reader comparing them needs to know how much of Geometric is not trench.
            "section_m3": section,
            "berm_credit_m3": berm_credit,
            "rasterisable_m3": reference,                # what it impounds, alone
            "terrain_m3": terrain_m3,                    # what the finished site ponds
            "delta_pct": feat_delta_pct,                 # interaction with neighbours
            # The trench as cut, and the grid-fidelity gap it implies. None where no burn
            # has run — a claim about the cut needs a cut.
            "cut_m3": float(cut) if cut is not None else None,
            # The earth that comes out, which is a different question from the one above
            # and the one the build schedule prices from: `cut_m3` is the trench filled to
            # its own pour point, this is `original − burned`. They part company by the
            # over-dig a level invert makes on falling ground. Kept as separate keys
            # because the panel narrates the first and the report prints the second, and
            # the single name they used to share is how one came to be printed as the other.
            "excavation_m3": float(excavation) if excavation is not None else None,
            "resolution_penalty_m3": penalty,
            "impoundment_m3": impoundment,
            "routing_only": routing_only,
            # The other features this one's pool is continuous with, or None. Both
            # renderers use it to say why the last two columns are blank.
            "merged_with": list(shares_pool_with) if shares_pool_with else None,
            # A dam impounds against the terrain rather than a drawn section, so its
            # three "design" columns are one number and only Measured is independent.
            "barrier_impounded": barrier,
            # The terrain model does not hold this feature's drawn section, so At-grid
            # and Measured are a placement/routing check here and not a capacity. Both
            # renderers style from these two: the flag decides *whether* to say so, the
            # signed gap decides *what* to say.
            "section_overstated": overstated,
            "section_gap_pct": gap_pct,
            # Water already ponding here before any earthwork, and the pool actually
            # standing on the ground afterwards. terrain_m3 remains the marginal figure
            # every delta is measured against — these two only add context.
            "existing_m3": existing,
            "total_m3": None if terrain_m3 is None else terrain_m3 + existing,
        })

    # One row per shared pool, carrying the comparison its members individually cannot.
    # The reference is the members' at-grid figures summed, so the group is tested on
    # exactly the same basis as a lone feature: measured against what the grid holds.
    group_rows = []
    for group in merged_groups:
        names = [n for n in group.get("names", ()) if n in analytic_by_name]
        if len(names) < 2:
            continue
        reference = sum(_reference(n, analytic_by_name[n]) for n in names)
        measured = float(group.get("volume_m3", 0.0))
        group_rows.append({
            "names": names,
            "rasterisable_m3": reference,
            "terrain_m3": measured,
            "delta_pct": ((measured - reference) / reference * 100.0
                          if reference > 0 else None),
            "existing_m3": sum(float(existing_by_name.get(n, 0.0)) for n in names),
        })

    caveats = [
        "Δ compares measured terrain storage against what a "
        f"{cell_size:.2f} m grid can represent — so a non-zero Δ is a burn issue, not "
        "a resolution effect.",
        "Terrain volume is attributed by connected depression. A pool that reaches "
        "two features belongs to neither alone, so it is reported once for the pair.",
        "Terrain total includes barrier-impounded storage (e.g. dams) that has no "
        "analytic counterpart.",
        "Sub-cell features (narrower than one DEM cell) are validated for placement and "
        "routing only, not independent storage volume.",
    ]
    for name, pct in resolution_flagged[:3]:
        # The burn now cuts the drawn section, so a material gap here is the cell size
        # failing to hold it rather than the burn declining to try: a footprint two
        # cells across has no cell more than half a cell from its own edge, and so
        # cannot reach full depth however the batter is specified.
        caveats.append(
            f"{name}: a {cell_size:.2f} m cell cannot hold its drawn section, so the "
            f"terrain model cut {pct:+.0f}% against it. Read Geometric, not At-grid, "
            f"as its capacity."
        )

    for group in group_rows:
        joined = " + ".join(group["names"])
        caveats.append(
            f"{joined} impound one continuous pool, so it is measured once for the set: "
            f"{fmt_volume(group['terrain_m3'])} against "
            f"{fmt_volume(group['rasterisable_m3'])} "
            f"at grid. Neither holds it alone, so neither carries a Δ of its own."
        )

    return VerificationResult(
        analytic_total_m3=analytic_total,
        terrain_total_m3=terrain_total,
        delta_m3=delta_m3,
        delta_pct=delta_pct,
        unattributed_m3=float(unattributed_m3 or 0.0),
        per_feature=per_feature,
        merged_groups=group_rows,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare(baseline: BaselineReport,
            post: PostInterventionReport) -> ComparisonResult:
    """
    Compute before/after comparison metrics.

    Parameters
    ----------
    baseline : BaselineReport
    post : PostInterventionReport

    Returns
    -------
    ComparisonResult
    """
    result = ComparisonResult(baseline=baseline, post=post)

    # Nothing below is clamped to zero from beneath, and that is the point of this
    # block. These quantities were each `max(0.0, ...)`, so a design that made
    # things *worse* — a diversion delivering water to the boundary faster, a post
    # exit volume above the baseline's — came out as 0.0 and rendered on a page
    # headed "Before and after", in a column headed "Change", as "0%" or an em
    # dash. Reporting no change for an adverse change is worse than reporting
    # nothing, and it contradicts this project's own rule for the sibling
    # quantity: `unrouted_flow` is "printed and flagged, never clamped".
    #
    # `captured_pct` keeps its *upper* clamp and loses only the lower one. The
    # asymmetry is deliberate: capturing more than fell is unphysical and could
    # only be an accounting defect, while a negative capture is a real design —
    # one that exports more than it received, having drained storage that was
    # already there.
    total = baseline.total_runoff_m3
    if total > 0:
        captured = total - post.exit_volume_m3
        result.captured_pct = min(100.0, captured / total * 100.0)

    if baseline.exit_volume_m3 > 0:
        result.exit_reduction_pct = (
            (baseline.exit_volume_m3 - post.exit_volume_m3)
            / baseline.exit_volume_m3 * 100.0
        )

    if baseline.peak_outflow_ls > 0:
        result.peak_reduction_pct = (
            (baseline.peak_outflow_ls - post.peak_outflow_ls)
            / baseline.peak_outflow_ls * 100.0
        )

    result.peak_delay_hr = post.peak_outflow_time_hr - baseline.peak_outflow_time_hr

    result.net_cut_m3 = sum(
        s.get("cut_vol_m3", 0.0) for s in post.earthwork_summary
    )
    result.net_fill_m3 = sum(
        s.get("fill_vol_m3", 0.0) for s in post.earthwork_summary
    )
    result.net_cut_fill_m3 = result.net_cut_m3 - result.net_fill_m3

    return result


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig, dpi=None):
    """Convert a matplotlib Figure to a base64-encoded PNG string.

    The default is the report's export resolution, not matplotlib's. These were
    saved at 100 dpi and then laid out by ``layout_pdf`` as
    ``pixels / self.dpi * 25.4`` with ``self.dpi`` at 200 — so every one of them
    printed at half the intended width, with axis labels around 5.5 pt, and the
    ``min(1.0, ...)`` fit could not correct an image that was already too small.
    """
    from terrainflow_assessment.modules.report_charts import REPORT_DPI

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi or REPORT_DPI, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_hydrograph_chart(baseline: BaselineReport,
                             post: PostInterventionReport, dpi=None):
    """
    Build a before/after outflow hydrograph chart.
    Returns base64 PNG string, or None when there is nothing to draw.

    Nothing to draw means neither side has a timestep table. Without this guard the
    function drew the axes anyway and returned them, so a comparison with no
    simulation behind it put a blank full-width panel in the report captioned
    "Outflow Hydrograph — Before vs After" — which reads as a measurement showing no
    flow rather than as a measurement that was never taken. ``_build_fill_timeline_
    chart`` has had the same guard all along; matplotlib flagged the difference on
    every run ("No artists with labels found to put in legend") and nothing was
    listening.
    """
    if not baseline.timestep_table and not post.timestep_table:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    if baseline.timestep_table:
        times_b = [r["time_hr"] for r in baseline.timestep_table]
        flows_b = [r.get("outflow_ls", 0.0) for r in baseline.timestep_table]
        ax.plot(times_b, flows_b, color="#c0392b", linewidth=2,
                label="Baseline (no earthworks)", zorder=3)

    if post.timestep_table:
        times_p = [r["time_hr"] for r in post.timestep_table]
        flows_p = [r.get("outflow_ls", 0.0) for r in post.timestep_table]
        ax.plot(times_p, flows_p, color="#2980b9", linewidth=2,
                label="With earthworks", zorder=3)

    ax.set_xlabel("Time (hr)", fontsize=11)
    ax.set_ylabel("Site exit flow (L/s)", fontsize=11)
    ax.set_title("Outflow Hydrograph — Before vs After", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    b64 = _fig_to_base64(fig, dpi)
    plt.close(fig)
    return b64


def _build_fill_timeline_chart(post: PostInterventionReport, dpi=None):
    """
    Build a stacked fill-% chart showing each earthwork filling over time.
    Returns base64 PNG string or None.
    """
    if not post.timestep_table or not post.earthwork_summary:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    # Join on identity, label with the name. `timestep_table` is keyed by
    # `store.id`, because the default name counter reproduces a deleted feature's
    # name and two features can share one; `earthwork_summary` carries `"id"`
    # alongside `"name"` for exactly this join. The `or s.get("name")` fallback is
    # what a summary assembled without ids resolves to, which is the old key.
    ew_names = [s["name"] for s in post.earthwork_summary]
    ew_keys = [s.get("id") or s.get("name") for s in post.earthwork_summary]
    times = [r["time_hr"] for r in post.timestep_table]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    colours = cm.Blues(
        [0.4 + 0.5 * i / max(len(ew_names) - 1, 1) for i in range(len(ew_names))]
    )

    for i, name in enumerate(ew_names):
        col_key = f"{ew_keys[i]}_fill_pct"
        if col_key in post.timestep_table[0]:
            fill_series = [r.get(col_key, 0.0) for r in post.timestep_table]
            ax.plot(times, fill_series, linewidth=1.8, color=colours[i], label=name)
            # Mark overflow event
            s = post.earthwork_summary[i]
            if s.get("overflowed") and s.get("first_overflow_hr") is not None:
                ax.axvline(s["first_overflow_hr"], color=colours[i],
                           linestyle="--", alpha=0.6, linewidth=1.2)

    ax.axhline(100, color="#e74c3c", linewidth=1.5, linestyle=":", alpha=0.8,
               label="100% full (overflow)")
    ax.set_xlabel("Time (hr)", fontsize=11)
    ax.set_ylabel("Fill (%)", fontsize=11)
    ax.set_title("Earthwork Fill Timeline", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    b64 = _fig_to_base64(fig, dpi)
    plt.close(fig)
    return b64

