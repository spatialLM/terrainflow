"""
report_model.py — what goes in the Site Water Plan, and in what order.

The renderer-independent half of the report. ``build_report(data)`` turns a
:class:`ReportData` bundle into a flat list of :class:`Section` objects; a
renderer (PDF today, HTML later) walks that list and draws it. Content decisions
— which sections appear, what they say when their data is missing, how a number
is worded — all live here, where they are pure and testable without QGIS.

**No simulation required.** The design tier alone answers everything a landowner
needs: :class:`~terrainflow_assessment.modules.water_balance.BalanceResult`
carries capture %, per-feature water, cut/fill and the mass-balance flag, and it
is recomputed on every design edit. A simulation, when one has been run, is an
optional enrichment — never a precondition.

The document degrades rather than disappearing. A section with no data keeps its
heading and states its reason and which button produces it, because silent
omission is how a reader fails to notice that verification never ran.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from terrainflow_assessment.core.registry.map_palette import (
    AREA_OUTLINES,
    STREAMS,
    hex_of,
    stop_colour,
)
from terrainflow_assessment.modules.reporting import (
    CAPTURE_GOOD_PCT,
    capture_tone,
    cut_fill_sentence,
    disambiguate,
    drain_wording,
    fill_wording,
    fmt_area_ha,
    fmt_pct,
    fmt_volume,
    spillway_state_wording,
    type_label,
    unique_names,
)

# Words that would make this document a claim it cannot support. A landowner
# takes this to a bank or a council; an auto-generated PDF must not imply that
# anything in it has been certified. Enforced by test over the rendered strings.
FORBIDDEN_WORDS = (
    "certified", "certifies", "certification", "compliant", "compliance",
    "approved", "guaranteed", "warranted",
)

# The one context in which those words may appear is a sentence *denying* the
# claim. Disclaimers are registered here rather than pattern-matched for
# negation, so adding one is a deliberate act and the test stays exact.
STANDING_CAVEAT = (
    "Design estimate from a terrain model. Not a survey and not an engineering "
    "certification — see the final page."
)

_NOT_A_SURVEY = (
    "This is a design estimate produced from a terrain model. It is not a "
    "survey: every level, length and volume comes from a digital elevation "
    "model, not from instrument levels. It is not an engineering assessment "
    "of any structure — embankment stability, foundations, seepage and "
    "spillway erosion are all outside its scope. It makes no statement "
    "about any regional plan or building requirement. Anything impounding "
    "water may be a classifiable dam under New Zealand regulation and needs "
    "a suitably qualified engineer. Have your contractor set out on the "
    "ground before digging."
)

ALLOWED_DISCLAIMERS = (STANDING_CAVEAT, _NOT_A_SURVEY)

MAP_CAPTION_SUFFIX = "Terrain model, not a survey."


# ---------------------------------------------------------------------------
# Section types — the vocabulary a renderer has to understand
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """Base: everything carries an optional anchor for cross-references."""
    anchor: str = ""


@dataclass
class Heading(Section):
    text: str = ""
    level: int = 1


@dataclass
class Paragraph(Section):
    text: str = ""


@dataclass
class Callout(Section):
    """A boxed note. ``tone`` is one of info | warn | bad."""
    text: str = ""
    tone: str = "info"
    title: str = ""


@dataclass
class Hero(Section):
    """The page-1 headline figure. ``value`` is pre-formatted."""
    value: str = ""
    label: str = ""
    tone: str = "info"
    sub: str = ""


@dataclass
class StatGrid(Section):
    """Cards of (label, value, sub)."""
    cards: list = field(default_factory=list)


@dataclass
class KeyValueTable(Section):
    title: str = ""
    rows: list = field(default_factory=list)          # [(key, value), ...]


@dataclass
class DataTable(Section):
    title: str = ""
    headers: list = field(default_factory=list)
    rows: list = field(default_factory=list)          # [[cell, ...], ...]
    note: str = ""
    # Rendered landscape when the column count needs it. The layout table
    # overflows its frame silently rather than wrapping, so width is the
    # renderer's problem and this is how it is told.
    wide: bool = False


@dataclass
class ImageRef(Section):
    """A chart the renderer supplies by key. ``fallback`` prints when absent."""
    key: str = ""
    caption: str = ""
    fallback: Optional[Section] = None


@dataclass
class LegendEntry:
    """One row of a map key.

    ``kind`` is ``line`` | ``fill`` | ``point`` — the shape of the swatch — or
    ``ramp``, in which case ``colours`` carries the whole gradient low-to-high
    and ``colour`` is ignored. A renderer that cannot draw a gradient may fall
    back to the first and last stops.
    """
    label: str = ""
    colour: str = ""
    kind: str = "line"
    colours: tuple = ()


@dataclass
class MapRef(Section):
    """A map the renderer supplies by key. ``reason`` set when unavailable.

    ``legend`` is built here rather than read out of QGIS so that what the key
    says is a content decision like every other, testable without a map. Both
    renderers draw it; the parity test is what keeps them agreeing.
    """
    key: str = ""
    caption: str = ""
    reason: str = ""
    legend: list = field(default_factory=list)


@dataclass
class PageBreak(Section):
    orientation: str = "portrait"                     # portrait | landscape


@dataclass
class Report:
    title: str = ""
    subtitle: str = ""
    footer: str = ""
    sections: list = field(default_factory=list)
    # Which stages actually ran — printed on the cover so the document states
    # its own completeness rather than leaving the reader to infer it.
    completeness: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input bundle
# ---------------------------------------------------------------------------

@dataclass
class ReportData:
    """Everything the report can draw on. Every field may be ``None``."""
    site_name: str = "Unnamed Site"
    generated_at: str = ""
    plugin_version: str = ""
    run_tag: str = ""
    current_tag: str = ""            # re-derived at export; differs when stale

    baseline: Any = None             # BaselineReport
    balance: Any = None              # BalanceResult (design tier)
    balance_stores: Optional[list] = None
    earthworks: Optional[list] = None  # Earthwork objects, for the build schedule
    verification: Any = None         # VerificationResult
    comparison: Any = None           # ComparisonResult — optional enrichment
    spillway_rows: Optional[list] = None
    spillway_context: Optional[dict] = None
    edits_since_verify: Optional[int] = None
    # {feature id: display name}, built once by build_report so a letter means the
    # same feature on every page. See _display_names.
    display_names: dict = field(default_factory=dict)
    natural_ponding_m3: Optional[float] = None
    # {cut_m3, fill_m3} the burn moved. None before a burn — the measured
    # earthmoving does not exist until the design has been cut into the terrain.
    burn_quantities: Optional[dict] = None

    inputs: dict = field(default_factory=dict)   # panel settings, for the appendix
    dem: dict = field(default_factory=dict)      # provenance
    maps: dict = field(default_factory=dict)     # key -> reason-if-unavailable
    # Whether the summary map got a catchment outline. Set by the exporter, which
    # is the only thing that knows whether the labelling traced to anything — a
    # key naming a line the map does not draw is worse than no key.
    catchment_outline: bool = False

    # ---- derived predicates, so the rules below read as prose ----
    @property
    def has_baseline(self):
        return self.baseline is not None

    @property
    def has_design(self):
        return bool(self.balance is not None
                    and getattr(self.balance, "per_feature", None))

    @property
    def has_verification(self):
        return self.verification is not None

    @property
    def stale_storm(self):
        """Storm inputs changed since the baseline was run."""
        return bool(self.run_tag and self.current_tag
                    and self.run_tag != self.current_tag)


# ---------------------------------------------------------------------------
# Reasons — one place, so an absent section always explains itself the same way
# ---------------------------------------------------------------------------

REASON_NO_BASELINE = (
    "No baseline analysis has been run, so nothing is known about how water "
    "moves across this block. Run Baseline to fill this in."
)
REASON_NO_DESIGN = (
    "No earthworks have been drawn yet, so there is no design to describe. "
    "Draw a swale, basin or dam on the Design stage to fill this in."
)
REASON_NO_VERIFY = (
    "You have not re-analysed with earthworks. Every storage figure in this "
    "report is arithmetic on the shapes you drew — nothing has been measured "
    "against the ground. Run Re-analyse with Earthworks to fill this in."
)
REASON_ALL_DISABLED = (
    "Every earthwork on this design is currently turned off, so there is no "
    "active scheme to assess. Re-enable at least one feature to fill this in."
)


def _reason_for_design(data):
    """Why the design sections are empty — 'all disabled' is not 'none drawn'."""
    if data.earthworks:
        return REASON_ALL_DISABLED
    return REASON_NO_DESIGN


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------

def _display_names(data):
    """One ``{id: display name}`` for the whole document.

    The enabled earthworks first, because that is the population and the order the
    reader meets them in; any balance row for a feature not in that list follows.
    Built once so a letter means the same feature on every page — three separate
    letterings meant "(a)" on the design page could be a different swale from "(a)"
    on the network page, and the volume ladder, which lettered nothing at all,
    could print two identical rows.
    """
    pairs = []
    for e in (data.earthworks or []):
        if getattr(e, "enabled", True):
            pairs.append((getattr(e, "id", None), getattr(e, "name", "")))
    for f in (getattr(data.balance, "per_feature", None) or []):
        pairs.append((f.get("id"), f.get("name")))
    return disambiguate(pairs)


def _by_raw_name(data):
    """``{raw name: display name}`` for names that identify exactly one feature.

    The verification rows are keyed by name and carry no id, so they could not reach
    the document's ``{id: name}`` map and printed the raw name instead — while every
    other page printed the disambiguated one. A reader then met "Swale 3" in the
    caveats and "Swale 3 (b)" in the table above it.

    Ambiguous raw names are deliberately absent: where two features share one, there
    is no mapping that is true of either, and guessing would attribute a caveat to
    the wrong earthwork. Those keep the raw name, which is the honest answer.
    """
    cached = getattr(data, "_raw_name_map", None)
    if cached is not None:
        return cached
    names = getattr(data, "display_names", None) or {}
    pairs = []
    for e in (data.earthworks or []):
        if getattr(e, "enabled", True):
            pairs.append((getattr(e, "id", None), getattr(e, "name", "")))
    for f in (getattr(data.balance, "per_feature", None) or []):
        pairs.append((f.get("id"), f.get("name")))
    counts = {}
    for _, raw in pairs:
        counts[raw] = counts.get(raw, 0) + 1
    mapping = {raw: names[fid] for fid, raw in pairs
               if counts.get(raw) == 1 and fid in names}
    data._raw_name_map = mapping
    return mapping


def _shown(data, row, fallback_key="name"):
    """A row's display name: the document's, or its own where it has no id."""
    fid = row.get("id") if isinstance(row, dict) else getattr(row, "id", None)
    names = getattr(data, "display_names", None) or {}
    if fid in names:
        return names[fid]
    raw = (row.get(fallback_key) if isinstance(row, dict)
           else getattr(row, fallback_key, "")) or "Unnamed"
    # No id — a verification row. Reach the document's map through the name.
    return _by_raw_name(data).get(raw, raw)


def build_report(data):
    """Assemble the whole document. Never raises on missing data."""
    # Before any section is built: every page reads the same map.
    data.display_names = _display_names(data)
    report = Report(
        title=f"{data.site_name} — Site Water Plan",
        subtitle="Drainage and earthwork assessment — TerrainFlow",
        footer=_footer(data),
        completeness={
            "baseline": data.has_baseline,
            "design": data.has_design,
            "verify": data.has_verification,
        },
    )
    s = report.sections
    s.extend(_page_summary(data))
    s.extend(_page_how_to_read(data))
    s.extend(_page_site_today(data))
    s.extend(_page_design(data))
    s.extend(_page_network(data))
    s.extend(_page_water_shared(data))
    s.extend(_page_spillways(data))
    s.extend(_page_build_schedule(data))
    s.extend(_page_verification(data))
    s.extend(_page_simulation(data))
    s.extend(_page_appendix(data))
    return report


def _footer(data):
    bits = [data.site_name]
    if data.run_tag:
        bits.append(data.run_tag)
    if data.generated_at:
        bits.append(f"Generated {data.generated_at}")
    if data.plugin_version:
        bits.append(f"TerrainFlow {data.plugin_version}")
    return " · ".join(bits) + " — design estimate from a terrain model, not a survey."


# ---- page 1 ---------------------------------------------------------------

def _page_summary(data):
    out = [Heading(text="Your scheme in one page", level=1, anchor="summary")]

    # The site before any number about it. A reader who has never seen the block
    # cannot place a single figure on this page until they know what shape the
    # property is and where the features sit in it — and the reader this document
    # is handed to is often exactly that person. Framed wider than the design map
    # on page 3 so the block sits in its surroundings rather than filling the
    # frame; that map is the one to work from, this one is to get oriented by.
    out.append(_map_ref(data, "overview",
                        "The block, its water and what the scheme catches."))

    if data.stale_storm:
        out.append(Callout(
            tone="warn", title="The storm has changed since Baseline ran",
            text=(f"The baseline was run for {data.run_tag}; the design figures "
                  f"below are calculated for {data.current_tag}. Re-run Baseline "
                  "before relying on the capture figure."),
        ))

    b, bal = data.baseline, data.balance

    if not data.has_baseline:
        out.append(Callout(tone="warn", text=REASON_NO_BASELINE))
        return out

    if bal is None or not data.has_design:
        out.append(Hero(
            value=fmt_volume(getattr(b, "total_runoff_m3", 0.0)),
            label="of runoff lands on this block in the storm you chose",
            tone="info",
            sub=_storm_line(b),
        ))
        out.append(Callout(tone="info", text=_reason_for_design(data)))
    elif not bal.mass_balance_ok:
        # Deliberately still printed, small and unclamped, but the headline slot
        # says not to trust it. mass_balance_ok is the flag that says whether
        # the capture figure is bookkeeping-sound at all.
        out.append(Callout(
            tone="bad", title="These figures do not balance",
            text=("The water going in and the water coming out differ by more "
                  "than the tolerance allows. Do not rely on any number in this "
                  "report until they balance — re-run Baseline, and check the "
                  "routing warnings on the flow network page. The capture "
                  f"figure as computed is {fmt_pct(bal.capture_pct)}."),
        ))
    elif bal.per_feature and not sum(f.get("total_inflow_m3") or 0.0
                                     for f in bal.per_feature):
        # A baseline exists and every feature was measured, so this is not "no
        # baseline" — it is "nothing drains into anything you drew", a design
        # finding. Conditioned on the per-feature figures, not on
        # `total_inflow_m3`: that is site-wide runoff and it is 0 whenever the
        # design tier simply has no flow grid yet, which would print a diagnosis
        # of a fault that may not exist.
        out.append(Hero(
            value=fmt_volume(bal.total_capacity_m3),
            label="of storage built — but no part of the block drains into it",
            tone="warn",
            sub=("Check where your features sit relative to the flow paths on "
                 "the site map."),
        ))
    else:
        out.append(Hero(
            value=fmt_pct(bal.capture_pct),
            label="of the storm's runoff stays on your block",
            tone=_capture_tone(bal.capture_pct),
            sub=_storm_line(b),
        ))
        note = _capture_footnote(data)
        if note:
            out.append(Paragraph(text=note))
        out.append(_water_fate_table(bal, data))

    out.append(_summary_cards(data))
    checks = _condition_checks(data)
    if checks:
        out.append(checks)
    nxt = _next_three_things(data)
    if nxt:
        out.append(nxt)
    out.append(Paragraph(text=STANDING_CAVEAT))
    return out


# The two provenances every figure in this document has, named once here and tagged on
# every table that carries numbers. Kept as constants rather than typed out per table so
# the words cannot drift between sections, and short enough to sit in front of a note
# without pushing it onto another line.
CALCULATED = "Geometric calculated"
MEASURED = "Measured"


def _tag(kind, note=""):
    """Prefix *note* with which kind of figure the table holds."""
    return f"{kind} · {note}" if note else kind


def _page_how_to_read(data):
    """Which numbers are worked out and which are read off the ground.

    Every figure in this report is one or the other, and they fail in opposite
    directions: a calculated volume is exactly right about a shape that may not be
    buildable here, and a measured one is exactly right about a terrain model that is
    not a survey. Readers were treating them as competing estimates of one number and
    reading the disagreement as an error — so the distinction is stated once, up front,
    and then tagged on every table rather than left to be inferred.
    """
    out = [PageBreak(),
           Heading(text="How to read the numbers in this report", level=1,
                   anchor="how-to-read"),
           Paragraph(text=(
               "Every figure here is one of two kinds. They answer different "
               "questions, and where they disagree that is information, not a "
               "mistake — so each table says which kind it holds."))]

    # Callouts, not a two-column table: these are paragraphs, and a layout table
    # silently overflows its frame rather than wrapping (see layout_pdf), so the
    # definitions were printed cut off mid-sentence at the page edge.
    out.append(Callout(
        tone="info", title=CALCULATED,
        text=("Worked out from the dimensions you drew — lengths, widths, depths, "
              "batters — using geometry and standard hydraulic formulae. It never "
              "looks at the ground: it assumes a completely flat gradient under the "
              "feature. You can check any of it with a tape measure and a "
              "calculator, and it would be the same on any site with the same "
              "dimensions. Capacity, cut and fill quantities, spillway sizes and berm "
              "heights are all geometric calculated figures.")))
    out.append(Callout(
        tone="info", title=MEASURED,
        text=("Read off the elevation model — either the ground as it is, or the "
              "ground with your design cut into it and then flooded. It is specific to "
              "this block and this survey, and it accounts for slope, hollows and "
              "where water actually runs. Catchment areas, rim levels, the storage the "
              "terrain holds and the earth the design moves are all measured.")))

    out.append(Paragraph(text=(
        "On the earth being moved, the gap between the two runs one way. "
        "A calculated volume assumes the ground is flat under the feature. Real "
        "ground is not, and these features are built level — a swale floor and a "
        "berm crest each sit at one elevation, which is what lets them hold water "
        "evenly instead of draining to one end. On sloping ground that means "
        "cutting deeper at the high end and building higher at the low end than the "
        "drawn section implies. The calculated figure is what you would move on a "
        "flat site; the measured figure is what this site asks for.")))

    out.append(Callout(
        tone="info", title="Storage is measured, and that matters",
        text=("How much a feature holds is measured, not calculated — the design is "
              "flooded on the elevation model and the pond is counted. This is a "
              "deliberate choice and it usually reports more storage than the drawn "
              "section does, because a bank keyed into its ends holds water above "
              "natural ground and up the slope behind it. No cross-section predicts "
              "that; it depends on the hillside. Sized on the drawn section instead, a "
              "swale like that reports full while most of its pond is still empty, and "
              "the design gets enlarged when it did not need to be.")))

    out.append(Paragraph(text=(
        "The headline percentage of the storm held on site is measured using "
        "topographical data on both halves: how much your features hold and how much "
        "water reaches each of them are both read off the terrain. Where this report "
        "gives a geometric calculated figure beside a measured one, it says so.")))

    cell = getattr(data.baseline, "cell_size_m", None) if data.baseline else None
    grid = (f" — {cell:.2f} m across per cell on this site" if cell else "")
    out.append(Callout(
        tone="info", title="What measured does not mean",
        text=(f"Measured here means measured from the elevation model, not surveyed on "
              f"the ground. That model is a grid of heights{grid}, so it cannot see "
              f"anything smaller than one cell, and it carries whatever error the "
              f"survey behind it carries. It is the best available description of the "
              f"block; it is not the block. " + STANDING_CAVEAT)))
    return out


def _capture_tone(pct):
    # One band for the panel and the page — see reporting.CAPTURE_GOOD_PCT.
    return capture_tone(pct)


def _storm_line(b):
    if b is None:
        return ""
    return (f"{b.rainfall_mm:.0f} mm over {b.duration_hr:.0f} hours · "
            f"{b.runoff_mm:.0f} mm of that runs off")


def _capture_footnote(data):
    """The simulated capture figure, where one exists, and why it differs.

    The two numbers are computed from different things and neither is a
    correction of the other: ``capture_pct`` counts captured water cell by cell
    against site-wide runoff and exists as soon as a design does, while
    ``captured_pct`` is runoff minus the volume the routed simulation actually
    pushed over the boundary. Printing the headline with no acknowledgement of
    the other figure is what makes a reader treat the gap as an error.
    """
    c = data.comparison
    if c is None:
        return ""
    return (
        f"An hour-by-hour simulation of the same storm put this at "
        f"{fmt_pct(getattr(c, 'captured_pct', 0.0))}. That is a different "
        "measurement, not a correction: this figure counts captured water "
        "across every cell of the block, the simulated one is what was left "
        "after routing the storm through the design over time. Quote whichever "
        "you mean and say which it is.")


def _simulated_fate(data):
    """Held / soaked / leaves as the simulation had them. None without one.

    Derived rather than stored: the simulation reports what left the block and
    what soaked away, so what stayed is the remainder. Clamped at zero because
    a routed exit volume can exceed the depth-derived runoff total on a site
    where water arrives from off-block.
    """
    c = data.comparison
    post = getattr(c, "post", None) if c is not None else None
    if post is None or data.baseline is None:
        return None
    total = float(getattr(data.baseline, "total_runoff_m3", 0.0) or 0.0)
    leaves = float(getattr(post, "exit_volume_m3", 0.0) or 0.0)
    soaked = float(getattr(post, "total_infiltrated_m3", 0.0) or 0.0)
    return {"held": max(0.0, total - leaves - soaked), "soaked": soaked,
            "leaves": leaves, "total": total}


def _water_fate_table(bal, data=None):
    """Where the storm's water ends up. A table, not a chart: three numbers do
    not need a graphic, and a table survives a missing matplotlib.

    Where a simulation has run, its figures sit alongside in their own columns.
    The two share columns divide by different totals — see the note — so they
    are labelled separately rather than being presented as one percentage.
    """
    stored = max(0.0, bal.total_captured_m3 - bal.total_infiltration_m3)
    total = bal.total_inflow_m3 or 0.0

    def share(v, denominator=None):
        denominator = total if denominator is None else denominator
        return fmt_pct(100.0 * v / denominator) if denominator else "—"

    rows = [
        ["Held in your earthworks", fmt_volume(stored), share(stored)],
        ["Soaked into the ground", fmt_volume(bal.total_infiltration_m3),
         share(bal.total_infiltration_m3)],
        ["Leaves the block", fmt_volume(bal.site_exit_m3),
         share(bal.site_exit_m3)],
    ]
    headers = ["", "Volume", "Share of runoff"]

    notes = []
    # The same three volumes against the ground the design actually intercepts.
    # Against the whole catchment, a feature that works its own slope perfectly is
    # scored down by every hectare that drains past it — two different faults with
    # opposite remedies, reported as one number. Omitted where the design works
    # essentially all of the catchment, since the column would then restate the one
    # beside it.
    managed = _managed_catchment(data) if data is not None else None
    if managed is not None and managed["runoff_m3"] > 0 \
            and managed["managed_pct"] < 99.0:
        headers.append("Share of worked catchment")
        # The exit row gets an em dash rather than a percentage, and that is the
        # whole design of this column. Its denominator is the runoff on ground a
        # feature intercepts, so the matching numerator would be the overflow that
        # ran past the last feature — not ``site_exit_m3``, which also carries
        # runoff from ground no feature touches. Printing that share beside the
        # site exit volume in the same row puts two different "water leaving"
        # figures on one line, which is the misreading Rule 2 exists to stop. The
        # complement is visible anyway: what is not held or soaked, left.
        for row, value in zip(rows, (stored, bal.total_infiltration_m3, None)):
            row.append("—" if value is None
                       else share(value, managed["runoff_m3"]))
        notes.append(
            f"The last column divides only the runoff falling on the "
            f"{fmt_area_ha(managed['managed_m2'])} that drains into a feature "
            f"({fmt_pct(managed['managed_pct'])} of the catchment) — so it answers "
            "how well the features work the ground they actually command, which "
            "the headline percentage cannot separate from how much ground they "
            "command in the first place. The exit row is left blank there on "
            "purpose: water leaving from ground no feature intercepts was never "
            "in that denominator, and the two kinds of exit are never mixed in "
            "one figure.")
    sim = _simulated_fate(data) if data is not None else None
    if sim is not None:
        headers += ["Volume (simulated)", "Share (simulated)"]
        for row, key in zip(rows, ("held", "soaked", "leaves")):
            row += [fmt_volume(sim[key]), share(sim[key], sim["total"])]
        # Both denominators are site-wide runoff (water_balance measures the
        # calculated one over the whole domain). The note used to say the
        # calculated shares divided "the water that reaches your earthworks",
        # which is the one paragraph here meant to prevent a misreading and it
        # instructed the reader wrongly. What actually differs is the method.
        notes.append(
            "Both share columns divide the runoff the whole block generates. "
            f"They differ in method, not in denominator: the calculated column "
            f"({fmt_volume(total)}) is the design-tier balance, which routes the "
            f"event as one total; the simulated column ({fmt_volume(sim['total'])}) "
            "is the same event stepped through time, so a feature that fills and "
            "spills part-way through holds less than its capacity suggests. Where "
            "the two disagree, the difference is timing.")

    if not bal.counts_infiltration:
        notes.append(
            f"Soakage is not being counted towards capture on this design. "
            f"A further {fmt_volume(bal.infiltration_buffer_m3)} could soak "
            "away, but the features are sized to hold their water without "
            "relying on it.")
    return DataTable(
        title="Where the storm's water goes", headers=headers, rows=rows,
        note=_tag("Geometric calculated storage against measured inflow",
                  " ".join(notes)))


def _managed_catchment(data):
    """How much of the catchment the design actually works, and how well.

    The headline capture percentage answers one question — what share of the whole
    storm stays on the block — and it conflates two quite different ways of scoring
    badly. A design can capture little because its features are too small, or
    because most of the block drains past them entirely. Those call for opposite
    responses, and the single figure cannot tell them apart.

    So: ``managed_pct`` is the share of the contributing catchment that drains into
    some feature, and ``capture_pct`` is how much of the runoff *falling on that
    ground* the design holds. Both come off ``per_feature``, whose direct catchments
    are mutually exclusive by construction (``flow_graph`` labels each cell with the
    feature that intercepts it **first**), so summing them never double-counts a
    hillside and the runoff total is the water genuinely entering the network.

    Neither figure is clamped, for the same reason ``capture_pct`` is not: a value
    over 100% is a labelling bug and has to be visible rather than rounded away.
    """
    bal, b = data.balance, data.baseline
    if bal is None or b is None or not bal.per_feature:
        return None
    catchment_m2 = float(getattr(b, "catchment_area_ha", 0.0) or 0.0) * 10000.0
    managed_m2 = sum(float(f.get("direct_catchment_m2") or 0.0)
                     for f in bal.per_feature)
    runoff_m3 = sum(float(f.get("direct_inflow_m3") or 0.0)
                    for f in bal.per_feature)
    if catchment_m2 <= 0 or managed_m2 <= 0:
        return None
    return {
        "managed_m2": managed_m2,
        "catchment_m2": catchment_m2,
        "managed_pct": 100.0 * managed_m2 / catchment_m2,
        "runoff_m3": runoff_m3,
        "capture_pct": (100.0 * bal.total_captured_m3 / runoff_m3
                        if runoff_m3 > 0 else None),
    }


def _summary_cards(data):
    b, bal = data.baseline, data.balance
    cards = []
    if b is not None:
        cards.append(("Your catchment", f"{b.catchment_area_ha:.1f} ha",
                      "the ground draining through this block"))
        cards.append(("The storm", f"{b.rainfall_mm:.0f} mm / {b.duration_hr:.0f} hr",
                      f"{b.runoff_mm:.0f} mm of runoff"))
    managed = _managed_catchment(data)
    if managed is not None:
        cards.append((
            "Catchment worked", fmt_pct(managed["managed_pct"]),
            f"{fmt_area_ha(managed['managed_m2'])} of "
            f"{fmt_area_ha(managed['catchment_m2'])} drains into a feature"))
        if managed["capture_pct"] is not None:
            cards.append((
                "Capture within it", fmt_pct(managed["capture_pct"]),
                "of the runoff falling on that ground is held"))
    if bal is not None:
        n = len(bal.per_feature)
        cards.append(("Storage built", fmt_volume(bal.total_capacity_m3),
                      f"across {n} feature{'s' if n != 1 else ''}"))
        cards.append(("Soil to move", fmt_volume(bal.total_cut_m3),
                      f"cut · {fmt_volume(bal.total_fill_m3)} fill"))
    return StatGrid(cards=cards)


def _condition_checks(data):
    """Named conditions, never an overall grade.

    Deliberately not a score, a dial or a pass/fail seal: this document has no
    standing to grade a design, and a single green tick is exactly what a reader
    would quote onward.
    """
    bal = data.balance
    if bal is None:
        return None
    rows = []

    if bal.total_inflow_m3:
        held = bal.capture_pct >= CAPTURE_GOOD_PCT
        rows.append(["Holds the storm",
                     "yes" if held else "partly",
                     f"{fmt_pct(bal.capture_pct)} of runoff stays on the block"])

    terminal = [f for f in bal.per_feature
                if f.get("is_terminal") and (f.get("overflow_m3") or 0) > 0]
    if terminal:
        rows.append(["Overflow has somewhere to go", "no",
                     f"{len(terminal)} feature(s) spill off the block "
                     f"({fmt_volume(bal.terminal_deficit_m3)})"])
    else:
        rows.append(["Overflow has somewhere to go", "yes",
                     "every feature that fills hands its water on"])

    if data.spillway_rows:
        bad = [r for r in data.spillway_rows if r.get("state") == "fail"]
        missing = [r for r in data.spillway_rows
                   if r.get("state") in ("undesigned", "unsited")]
        if bad or missing:
            detail = []
            if bad:
                detail.append(f"{len(bad)} too small")
            if missing:
                detail.append(f"{len(missing)} not designed or placed")
            rows.append(["Every weir big enough", "no", ", ".join(detail)])
        else:
            rows.append(["Every weir big enough", "yes",
                         "each spillway passes its design flow"])
    return DataTable(title="Three things worth checking",
                     headers=["Check", "", "Detail"], rows=rows)


def _next_three_things(data):
    """The highest-severity findings, in the reader's words."""
    items = []
    bal = data.balance
    if bal is not None:
        for w in (bal.routing_warnings or []):
            items.append(str(w))
        if bal.terminal_deficit_m3 > 0:
            items.append(
                f"{fmt_volume(bal.terminal_deficit_m3)} overflows from features "
                "with nowhere to send it. Decide where that water goes before "
                "you dig, not afterwards.")
    for row in (data.spillway_rows or []):
        for p in (row.get("problems") or []):
            items.append(f"{row.get('name', 'A feature')}: {p}")
    if not items:
        return None
    return DataTable(title="The next three things",
                     headers=["", ""],
                     rows=[[str(i + 1), t] for i, t in enumerate(items[:3])])


# ---- page 2 ---------------------------------------------------------------

def _page_site_today(data):
    out = [PageBreak(),
           Heading(text="Where your water goes today", level=1, anchor="site")]
    if not data.has_baseline:
        out.append(Callout(tone="info", text=REASON_NO_BASELINE))
        return out

    b = data.baseline
    out.append(Paragraph(text=(
        f"Before you build anything, a {b.rainfall_mm:.0f} mm storm puts about "
        f"{fmt_volume(b.total_runoff_m3)} of moving water on this block. This "
        "map shows the paths it takes and where it crosses your boundary.")))
    out.append(_map_ref(data, "flow",
                        "Flow paths, watercourses and boundary crossings."))

    rows = [
        ("Catchment area", f"{b.catchment_area_ha:.1f} ha"),
        ("Design storm", f"{b.rainfall_mm:.0f} mm over {b.duration_hr:.0f} hours"),
        ("Runoff depth", f"{b.runoff_mm:.1f} mm"),
        ("Total runoff", fmt_volume(b.total_runoff_m3)),
        ("Terrain model", f"{b.cell_size_m:.2f} m grid · {b.crs}"),
    ]
    basis = (data.inputs or {}).get("sizing_basis")
    if basis == "coefficient":
        rows.append(("Runoff basis",
                     f"runoff coefficient {(data.inputs or {}).get('runoff_coefficient', 0):.2f}"))
    elif basis == "runoff":
        rows.append(("Runoff basis", f"SCS curve number {b.cn:.0f}"))
    elif basis == "rainfall":
        rows.append(("Runoff basis", "all rain runs off"))
    out.append(KeyValueTable(title="This site", rows=rows))

    if data.natural_ponding_m3 and data.natural_ponding_m3 >= 10:
        out.append(Paragraph(text=(
            f"About {fmt_volume(data.natural_ponding_m3)} already collects in "
            "hollows here without any earthworks. That is context, not part of "
            "what your design captures.")))

    out.extend(_exit_points(data))
    return out


def _exit_points(data):
    """Boundary crossings, and the box that stops the two exit figures merging."""
    b = data.baseline
    pts = sorted((b.exit_points or []),
                 key=lambda p: p.get("volume_m3", 0), reverse=True)
    if not pts:
        return []
    # Every crossing, not the top five. The map above draws all of them, so a
    # five-row table under it left the reader counting markers they could not
    # look up — and the "and N smaller crossings" line it printed instead named
    # none of them. The list and the figure now hold the same set.
    rows = [[p.get("label", "").split(":")[0] or f"Exit {i + 1}",
             f"{p.get('flow_ls', 0):.1f} L/s",
             fmt_volume(p.get("volume_m3", 0))]
            for i, p in enumerate(pts)]
    notes = [_exit_threshold_note(data)]
    # "Peak flow" until the panel grew the same table and had to name the column
    # honestly. ``flow_ls`` is the crossing's event volume over the event duration —
    # an **average rate**, spatially maxed across the cells of the crossing. The peak a
    # structure is sized against is Q at the time of concentration, comes from
    # ``peak_flow.py``, is larger, and is on the spillway page. Two documents printing
    # one number under two names, one of which is the name of a different quantity the
    # same document also reports, is the divergence this table exists to avoid.
    out = [DataTable(title="Where water leaves the boundary",
                     headers=["Crossing", "Average rate over the event",
                              "Volume over the event"],
                     rows=rows,
                     note=_tag(MEASURED, " ".join(n for n in notes if n)))]

    # Rule 2: the two exit volumes are never in one table and never subtracted.
    if data.has_design and data.balance is not None:
        out.append(Callout(
            tone="info", title="Two different 'water leaving' figures",
            text=(
                # No emphasis markup here: the PDF draws this straight into a
                # layout label and the HTML escapes it, so asterisks would print
                # as asterisks in both. The wording has to carry it.
                "This page counts water crossing the user's boundary. It answers "
                "where the water goes — and it only counts the crossings big "
                f"enough to draw ({_exit_threshold_phrase(data)}). The capture "
                "figure on page one is a different measurement "
                f"({fmt_volume(data.balance.site_exit_m3)}), and it answers how "
                "much of the storm was not captured, counted cell by cell "
                "across the whole site — including everywhere water seeps out "
                "below that threshold.\n\n"
                "That is why more water leaves the block than the crossings "
                "above add up to. They are not two estimates of one number and "
                "the difference between them is not an error. Read them "
                "separately, say which one is meant, and never subtract one "
                "from the other."),
        ))
    return out


def _exit_threshold_ls(data):
    """The panel's exit display threshold, in L/s, or None if not recorded."""
    try:
        return float((data.inputs or {}).get("exit_flow_ls"))
    except (TypeError, ValueError):
        return None


def _exit_threshold_phrase(data):
    """'above 0.5 L/s' — or a wording that does not invent a number."""
    threshold = _exit_threshold_ls(data)
    if threshold is None:
        return "above your display threshold"
    return f"above your display threshold of {threshold:g} L/s"


def _exit_threshold_note(data):
    """Why the listed crossings do not account for all the water that leaves.

    A reader who adds the column up and compares it with the site figure needs
    to be told, on the table itself, that the list is a filtered one. Water
    also crosses the boundary as sheet flow and through crossings below the
    threshold, none of which is drawn and none of which is listed here.
    """
    threshold = _exit_threshold_ls(data)
    if threshold is None:
        return ("Only crossings above your display threshold are drawn and "
                "listed. Water also leaves through smaller crossings and as "
                "sheet flow, so the block loses more than these rows total.")
    return (f"Only crossings carrying more than {threshold:g} L/s are drawn "
            "and listed. Water also leaves through smaller crossings and as "
            "sheet flow between them, so the block loses more than these rows "
            "total — lower the threshold on the Baseline stage to see them.")


# ---- page 3 ---------------------------------------------------------------

def _page_design(data):
    out = [PageBreak(),
           Heading(text="The design", level=1, anchor="design")]
    if not data.has_design:
        out.append(Callout(tone="info", text=_reason_for_design(data)))
        return out

    bal = data.balance
    by_type = {}
    for f in bal.per_feature:
        by_type[f.get("ew_type", "?")] = by_type.get(f.get("ew_type", "?"), 0) + 1
    parts = ", ".join(f"{n} {type_label(t).lower()}{'s' if n != 1 else ''}"
                      for t, n in sorted(by_type.items()))
    out.append(Paragraph(text=(
        f"{len(bal.per_feature)} features: {parts}. Together they hold "
        f"{fmt_volume(bal.total_capacity_m3)} and pass the rest on deliberately "
        "rather than by accident.")))
    # Spillway markers and overflow links used to be drawn on this map and named
    # in its key. They came off with everything else that was not the design
    # itself: this is the sheet somebody stands in a paddock holding, and at the
    # scale it is now framed at, a second set of markers over every feature was
    # competing with the names. Both are still in the document — the spillways
    # have a page and a table of their own, and the links are on the flow
    # diagram, which is where a routing question is actually answered.
    out.append(_map_ref(data, "design",
                        "Every feature as drawn, over the ground it is dug in."))
    # The map is labelled with names alone — a volume against every line was
    # unreadable wherever features cluster. Every capacity is in one place, on
    # the "How the water is shared out" page, rather than in a second feature
    # list that repeated the same names here.
    out.append(Paragraph(text=(
        "Features are labelled by name. What each one holds, what drains into "
        "it and where it spills are all on the 'How the water is shared out' "
        "page.")))
    return out


# ---- page 4: the flow network --------------------------------------------

def _page_network(data):
    out = [PageBreak(orientation="landscape"),
           Heading(text="Where the water goes when things fill", level=1,
                   anchor="network")]
    if not data.has_design:
        out.append(Callout(tone="info", text=_reason_for_design(data)))
        return out

    bal = data.balance
    out.append(Paragraph(text=(
        "Full features are normal — they are meant to fill. What matters is the "
        "chain: this shows where each one hands its water on, and which ones "
        "hand it to nowhere in particular.")))

    graph = build_flow_graph(bal, getattr(data, "display_names", None))

    # Findings before the picture: they are what the reader has to act on, and
    # putting them after a full-page diagram orphans them onto a page of their
    # own.
    if graph["all_terminal"]:
        out.append(Paragraph(text=(
            "No feature overflows into another — each one drains independently. "
            "That is worth knowing: nothing here is backed up by anything else.")))
    if bal.routing_warnings:
        out.append(Callout(tone="warn", title="Routing warnings",
                           text="\n".join(str(w) for w in bal.routing_warnings)))
    if bal.terminal_deficit_m3 > 0:
        out.append(Callout(tone="warn", text=(
            f"{fmt_volume(bal.terminal_deficit_m3)} of overflow leaves features "
            "that have no downstream destination. It is counted as leaving the "
            "block.")))

    out.append(ImageRef(
        key="network", caption=(
            "Features are placed down the page by height, so the cascade runs "
            "downhill as it runs across. Solid arrows are links you drew; dashed "
            "arrows are followed automatically downhill. Blue arrows are the ones "
            "water actually goes down at this storm, thicker where more of it "
            "does. Each box shows what the feature holds when dug and how full it "
            "gets."),
        fallback=DataTable(title="Overflow chains", headers=["Chain"],
                           rows=[[c] for c in graph["chains"]])))
    return out


def build_flow_graph(balance, display_names=None):
    """The overflow network as pure data — nodes, edges and readable chains.

    ``per_feature`` already carries the whole graph: ``target_id`` is the next
    feature, ``is_user_link`` says whether you drew that link or it was followed
    downhill, and ``is_terminal`` marks water leaving the block. Ranks come from
    ``simulation.layer_nodes``, which exists for exactly this.

    ``display_names`` is the document's one ``{id: name}`` map. Passed in rather
    than derived here, so the diagram labels a feature the same way the pages
    around it do; without one it letters the balance rows on their own, which is
    right for a caller that has nothing else to be consistent with.
    """
    from terrainflow_assessment.modules.simulation import layer_nodes

    per = list(getattr(balance, "per_feature", None) or [])
    if display_names:
        names = [display_names.get(f.get("id")) or f.get("name") or "Unnamed"
                 for f in per]
    else:
        names = unique_names(per)
    by_id = {}
    nodes = []
    for i, f in enumerate(per):
        node = {
            "id": f.get("id"),
            "name": names[i],
            "ew_type": f.get("ew_type", ""),
            "capacity_m3": f.get("capacity_m3", 0.0),
            "stored_m3": f.get("stored_m3", 0.0),
            "fill_pct": f.get("fill_pct", 0.0),
            "overflowed": bool(f.get("overflowed")),
            "overflow_m3": f.get("overflow_m3", 0.0),
            "is_terminal": bool(f.get("is_terminal")),
            "target_id": f.get("target_id"),
            "is_user_link": bool(f.get("is_user_link")),
            # For the cascade diagram, which places a feature down the page by
            # height. Absent on an older balance, so both are read defensively.
            "elevation": f.get("elevation"),
            "elevation_known": bool(f.get("elevation_known")),
        }
        nodes.append(node)
        by_id[node["id"]] = node

    edges = {n["id"]: (n["target_id"], n["is_user_link"]) for n in nodes}
    ranks = layer_nodes([n["id"] for n in nodes],
                        {k: v[0] for k, v in edges.items()})
    for n in nodes:
        rank, order = ranks.get(n["id"], (0, 0))
        n["rank"], n["order"] = rank, order

    return {
        "nodes": nodes,
        "edges": edges,
        "chains": _chains(nodes, by_id),
        "all_terminal": bool(nodes) and all(n["is_terminal"] for n in nodes),
    }


def _chains(nodes, by_id):
    """Readable cascades: 'Swale 1 -> Basin 5 -> off the block'.

    Doubles as the fallback when no chart can be drawn. Walks with a visited set
    because ``resolve_targets`` can leave a cycle in place (it warns rather than
    rejecting), and a naive walk would spin forever.
    """
    targets = {n["target_id"] for n in nodes if n["target_id"]}
    heads = [n for n in nodes if n["id"] not in targets] or nodes
    out = []
    for head in heads:
        seen, parts, cur = set(), [], head
        while cur is not None and cur["id"] not in seen:
            seen.add(cur["id"])
            parts.append(cur["name"])
            nxt = cur.get("target_id")
            cur = by_id.get(nxt) if nxt else None
        # Falling out with cur set means we walked back onto something already
        # visited — a ring. resolve_targets warns about those rather than
        # rejecting them, so the walk has to survive one.
        parts.append(f"back to {cur['name']} (loop)" if cur is not None
                     else "off the block")
        out.append(" → ".join(parts))
    return out


# ---- page 5 ---------------------------------------------------------------

def _page_water_shared(data):
    out = [PageBreak(orientation="landscape"),
           Heading(text="How the water is shared out", level=1, anchor="water")]
    if not data.has_design:
        out.append(Callout(tone="info", text=_reason_for_design(data)))
        return out

    bal = data.balance
    out.append(Paragraph(text=(
        "Each feature catches the slope directly above it, plus whatever the "
        "feature above sends down when it fills. 'Spills to' is where that water "
        "goes next — if it says 'off the block', that is a decision worth making "
        "on purpose.")))

    names = [_shown(data, f) for f in bal.per_feature]
    by_id = {f.get("id"): names[i] for i, f in enumerate(bal.per_feature)}
    soak_header = ("Soaked away" if bal.counts_infiltration
                   else "Could soak away")
    rows = []
    for i, f in enumerate(bal.per_feature):
        target = f.get("target_id")
        if f.get("is_terminal") or not target:
            dest = "off the block"
        else:
            dest = by_id.get(target, "—")
            if f.get("is_user_link"):
                dest += " (your link)"
        soak = (f.get("infiltration_m3") if bal.counts_infiltration
                else f.get("infiltration_buffer_m3"))
        rows.append([
            names[i],
            fmt_volume(f.get("capacity_m3", 0)),
            fmt_area_ha(f.get("direct_catchment_m2", 0)),
            fmt_volume(f.get("total_inflow_m3", 0)),
            fmt_volume(f.get("upstream_inflow_m3", 0)),
            fmt_volume(f.get("stored_m3", 0)),
            fmt_volume(soak),
            fill_wording(f.get("fill_pct"), f.get("overflowed")),
            dest,
            fmt_volume(f.get("overflow_m3", 0)),
            drain_wording(f.get("drain_hours")),
        ])
    # No Type column: the name already carries it ("Dam 1" is a dam), and this
    # is the widest table in the document — the space buys Capacity, which was
    # previously only in a second feature list that said nothing else.
    out.append(DataTable(
        title="Water arriving at each feature",
        headers=["Feature", "Capacity", "Catchment", "Water in", "From above",
                 "Held", soak_header, "How full", "Spills to", "Spills",
                 "Empties in"],
        rows=rows, wide=True,
        note=_tag("Geometric calculated storage against measured inflow",
                  "Capacity comes from the dimensions you drew; catchment and the "
                  "water arriving are read off the terrain.")))
    return out


# ---- page 6 ---------------------------------------------------------------

def _page_spillways(data):
    out = [PageBreak(),
           Heading(text="Overflow safety", level=1, anchor="spillways")]
    if not data.has_design:
        out.append(Callout(tone="info", text=_reason_for_design(data)))
        return out
    if not data.spillway_rows:
        out.append(Callout(tone="info", text=(
            "No overflow structures have been designed on this scheme yet.")))
        return out

    # Rule: volume and rate are different questions with different answers.
    ctx = data.spillway_context or {}
    out.append(Callout(tone="info", title="Two different questions", text=(
        "Storage is sized by how much rain falls. Spillways are sized by how "
        "fast it falls. Those are different questions and this report answers "
        "them separately — which is why a small basin can still need a wide "
        "weir.")))
    if ctx.get("intensity_is_default"):
        out.append(Callout(tone="warn", title="Design intensity is a placeholder", text=(
            f"These weirs are sized on {ctx.get('intensity_mm_hr', 0):.0f} mm/hr, "
            "which is the tool's default rather than this site's rainfall. There "
            "is no way to work a peak intensity out from a storm depth and "
            "duration. Enter a HIRDS rainfall table and these widths will move.")))

    rows = []
    for r in data.spillway_rows:
        rows.append([
            r.get("name", ""),
            _ls(r.get("peak_flow_m3s")),
            _m(r.get("required_width_m")),
            _m(r.get("built_width_m")),
            _m(r.get("actual_head_m")),
            _m(r.get("freeboard_m")),
            spillway_state_wording(r.get("state")),
        ])
    out.append(DataTable(
        title="Spillways",
        headers=["Feature", "Fast flow to pass", "Width needed", "Width designed",
                 "Water depth over weir", "Margin above", "Status"],
        rows=rows, wide=True,
        note=_tag("Geometric calculated storage against measured inflow",
                  "Widths and water depths come from the weir equation; the flow "
                  "each has to pass is the peak off a catchment read from the "
                  "terrain.")))

    # What each sill costs, in storage. Shown only where something has actually been
    # measured: the figures come from flooding each feature alone on the terrain model,
    # and a table of dashes says nothing a sentence cannot say better.
    levels = [r for r in data.spillway_rows
              if r.get("crest_elevation") is not None
              and r.get("rim_elevation") is not None]
    if levels:
        out.append(Paragraph(text=(
            "A spillway buys you the choice of where water leaves, and it pays for "
            "that in storage: everything above the sill goes out rather than being "
            "held. These are the levels each feature is working between.")))
        out.append(DataTable(
            title="Where each feature lets go, and what that costs",
            headers=["Feature", "Sill (designed)", "Held to", "Natural ground",
                     "Holds to the sill", "Would hold to the top", "Given up"],
            rows=[[
                r.get("name", ""),
                _m(r.get("crest_elevation")),
                _m(r.get("rim_elevation")),
                _m(r.get("lip_elevation")),
                fmt_volume(r.get("sill_storage_m3")),
                fmt_volume(r.get("containment_storage_m3")),
                _giveup(r),
            ] for r in levels],
            wide=True,
            note=_tag(MEASURED,
                      "The sill is the level you set. The other two come from the "
                      "terrain model: 'held to' is where this feature was measured to "
                      "pond to, and 'natural ground' is the lowest bare ground round "
                      "it. Where those differ, the gap is water standing on ground you "
                      "built.")))

    out.extend(_spillway_gauge_table(data))
    out.extend(_spillway_as_burned_table(data))

    notes = []
    for r in data.spillway_rows:
        for n in (r.get("notes") or []):
            notes.append(f"{r.get('name', 'A feature')}: {n}")
    if notes:
        out.append(Callout(tone="info", title="Worth knowing", text="\n".join(notes)))

    problems = []
    for r in data.spillway_rows:
        for p in (r.get("problems") or []):
            problems.append(f"{r.get('name', 'A feature')}: {p}")
    if problems:
        out.append(Callout(tone="warn", title="What needs attention",
                           text="\n".join(problems)))
    return out


def _spillway_gauge_table(data):
    """Three volumes off one curve: the sill, the surcharge, and the top.

    "% full" used to divide by the sill volume, so a spillway pinned its feature at 100%
    exactly when it started doing its job. These are the three marks that make that
    band readable: what the feature holds to its sill, where the water actually stands
    while the design storm passes the weir, and what it would hold before leaving over
    the structure itself. Reaching the third means the spillway is not sufficient — the
    same condition the freeboard check fires on, so the two cannot disagree.
    """
    rows = [r for r in (data.spillway_rows or [])
            if r.get("surcharge_storage_m3") is not None
            and r.get("containment_storage_m3") is not None]
    if not rows:
        return []
    return [
        Paragraph(text=(
            "While the storm passes, the water in a spillwayed feature stands above "
            "the sill — that is what drives it over the weir. These are the three "
            "levels that matter: what it holds to the sill, where it actually stands "
            "at the design flow, and the point at which water would start leaving over "
            "the structure instead of through it.")),
        DataTable(
            title="How full each feature gets while the storm passes",
            headers=["Feature", "Holds to the sill", "Storm water level",
                     "Holds at that level", "Would hold to the top", "Verdict"],
            rows=[[
                r.get("name", ""),
                fmt_volume(r.get("sill_storage_m3")),
                _m(r.get("surcharge_level_m")),
                fmt_volume(r.get("surcharge_storage_m3")),
                fmt_volume(r.get("containment_storage_m3")),
                ("Spillway is not passing enough — water reaches the top"
                 if r.get("spillway_insufficient")
                 else "Spillway holds the storm below the top"),
            ] for r in rows],
            wide=True,
            note=_tag(MEASURED,
                      "Volumes come off the stage-storage curve measured by flooding "
                      "each feature alone; the storm water level is the sill plus the "
                      "depth the built width actually produces at the peak flow.")),
    ]


def _spillway_as_burned_table(data):
    """Did the terrain model take the sill, and does the pond agree?

    Three elevations, and the disagreements are the content. The designed sill is what
    the user set; the as-burned sill is what the notch cut into the terrain model came
    out at; the measured spill level is where the finished pond was actually found to
    let go once the whole site was burned. Each way they can differ names a different
    fault, and none of them is visible from the design alone.

    Absent until an earthworks re-analysis has run, because until then there is no burn
    to measure — and a table of dashes asserts a question was asked and answered.
    """
    rows = [r for r in (data.spillway_rows or [])
            if r.get("crest_elevation") is not None
            and (r.get("burned_sill_m") is not None
                 or r.get("actual_spill_level_m") is not None)]
    if not rows:
        return []
    return [
        Paragraph(text=(
            "A designed spillway is cut into the terrain model as a notch, so the model "
            "routes water through it rather than over the bank. These three levels say "
            "whether that worked. They should agree; where they do not, the row says "
            "what went wrong.")),
        DataTable(
            title="Did the model take the sill?",
            headers=["Feature", "Sill designed", "Sill as cut", "Pond lets go at",
                     "Reading"],
            rows=[[
                r.get("name", ""),
                _m(r.get("crest_elevation")),
                _m(r.get("burned_sill_m")),
                _m(r.get("actual_spill_level_m")),
                _sill_reading(r),
            ] for r in rows],
            wide=True,
            note=_tag(MEASURED,
                      "The sill designed is the level you set; the other two are read "
                      "off the burned terrain model after the last earthworks "
                      "re-analysis.")),
    ]


#: Elevations print to two decimals, so a disagreement finer than a centimetre is one
#: nobody can act on and the reading would contradict its own figures.
_SILL_TOLERANCE_M = 0.01


def _sill_reading(row):
    """One sentence naming which of the three elevations disagree, and why it matters."""
    crest = row.get("crest_elevation")
    burned = row.get("burned_sill_m")
    actual = row.get("actual_spill_level_m")
    if burned is None:
        return "No notch was cut here — the spillway is not sited on the feature."
    if burned > crest + _SILL_TOLERANCE_M:
        return ("The notch was refused: the bank still stands above the sill. See the "
                "warnings from the last re-analysis for which guard stopped it.")
    if burned < crest - _SILL_TOLERANCE_M:
        return ("The ground along the notch was already below the sill, so cutting it "
                "moved nothing. The water was leaving here anyway.")
    if actual is None:
        return "Cut to the designed level."
    if actual > burned + _SILL_TOLERANCE_M:
        return ("Cut, but the pond lets go higher — the notch does not daylight, so "
                "the water is going out somewhere else.")
    if actual < crest - _SILL_TOLERANCE_M:
        return ("A lower point on the rim is the control, so this spillway never comes "
                "into play. Raise that point or move the sill.")
    return "Cut to the designed level, and the pond lets go there."


def _giveup(row):
    """The given-up figure as ``856 m³ (32%)`` — or an em dash where nothing is measured.

    Never "0 m³" for an unmeasured feature: a spillway that gives up nothing is a real
    and unusual state, and printing it for a feature nobody has flooded claims it.
    """
    given = row.get("given_up_m3")
    if given is None:
        return "—"
    pct = row.get("given_up_pct")
    if pct is None:
        return fmt_volume(given)
    return f"{fmt_volume(given)} ({pct:.0f}%)"


def _ls(m3s):
    return "—" if m3s is None else f"{float(m3s) * 1000:.0f} L/s"


def _m(v):
    return "—" if v is None else f"{float(v):.2f} m"


# ---- page 7 ---------------------------------------------------------------

def _page_build_schedule(data):
    out = [PageBreak(orientation="landscape"),
           Heading(text="Build schedule", level=1, anchor="build")]
    if not data.has_design or not data.earthworks:
        out.append(Callout(tone="info", text=_reason_for_design(data)))
        return out

    out.append(Paragraph(text=(
        "Every feature, with the geometric calculated figures from the dimensions "
        "you drew — brim-full, with no allowance taken off — set beside what the "
        "terrain model measures for the same feature. "
        "The measured columns are blank until an earthworks re-analysis has been "
        "run, and they are the ones to price the job from.")))

    cut_by_id, fill_by_id = {}, {}
    for st in (data.balance_stores or []):
        cut_by_id[getattr(st, "id", None)] = getattr(st, "cut_vol_m3", 0.0)
        fill_by_id[getattr(st, "id", None)] = getattr(st, "fill_vol_m3", 0.0)
    measured = _measured_by_name(data)

    enabled = [e for e in data.earthworks if getattr(e, "enabled", True)]
    names = [_shown(data, e) for e in enabled]
    rows = []
    for i, e in enumerate(enabled):
        m = measured.get(getattr(e, "name", None)) or {}
        rows.append([
            names[i],
            f"{getattr(e, 'length_m', 0) or 0:.0f} m",
            f"{getattr(e, 'top_width_m', 0) or 0:.2f} m",
            f"{getattr(e, 'bottom_width_m', 0) or 0:.2f} m",
            f"{getattr(e, 'depth', 0) or 0:.2f} m",
            _batter(e),
            _type_extra(e),
            getattr(e, "soil_name", "") or "site default",
            fmt_volume(getattr(e, "capacity_m3", 0)),
            fmt_volume(m.get("terrain_m3")),
            fmt_volume(cut_by_id.get(getattr(e, "id", None))),
            # `excavation_m3`, not `cut_m3`. This column used to print the latter — the
            # trench filled to its own pour point — which on falling ground is the earth
            # that fits in the hole rather than the earth taken out of it, understating a
            # swale on 10% cross-slope by 23% in the one column the paragraph above tells
            # the reader to price from, and disagreeing with the site total in the very
            # next table. `cut_m3` keeps its job as the grid-fidelity yardstick, where
            # being measured to the pour point is the point.
            fmt_volume(m.get("excavation_m3")),
            fmt_volume(fill_by_id.get(getattr(e, "id", None))),
        ])
    # "Dam 1" already says it is a dam, so a Type column beside the name spent
    # a column of a twelve-column table restating it. "Type detail" stays —
    # that carries the one dimension that matters for the type and no other.
    #
    # There is no measured fill column, and that is deliberate rather than an
    # omission: the burn builds banks that lie outside their own footprints and
    # cuts that overlap, so per-feature fill could only be produced by an
    # attribution rule, and the figure would report the rule. It is a site total
    # in the table below.
    out.append(DataTable(
        title="Every feature — drawn against measured",
        headers=["Feature", "Length", "Top width", "Bottom width",
                 "Depth", "Batter", "Type detail", "Soil",
                 "Capacity (geometric)", "Capacity (measured)",
                 "Cut (geometric)", "Cut (measured)", "Fill (geometric)"],
        rows=rows, wide=True,
        note=_tag(f"{CALCULATED} and {MEASURED}",
                  cut_fill_sentence(data.balance.total_cut_m3,
                                    data.balance.total_fill_m3)
                  + " Measured fill is a site total only — banks sit outside their "
                    "own footprints and cuts overlap, so splitting it between "
                    "features would report the splitting rule rather than the job.")))
    out.extend(_earthmoving(data))
    return out


def _measured_by_name(data):
    """``{raw name: verification row}`` for the features measured on the terrain.

    Keyed by name because that is the only key the verification carries — it is
    built from the burn's footprint masks, which are named, not from the balance
    rows. A name shared by two features is dropped rather than guessed at: the two
    would otherwise take each other's measured cut, and a wrong figure in a column
    headed "measured" is worse than a blank one.
    """
    v = data.verification
    rows = list(getattr(v, "per_feature", None) or []) if v is not None else []
    counts = {}
    for f in rows:
        counts[f.get("name")] = counts.get(f.get("name"), 0) + 1
    return {f.get("name"): f for f in rows if counts.get(f.get("name")) == 1}


def _earthmoving(data):
    """The drawn quantities against the ones the terrain model actually moves.

    Both, because they answer different questions and the gap between them is the
    point. ``calculate_cut_volume`` is ``section × length`` — the earth that comes out
    if the ground under the feature is flat. The burn cuts to a level invert and builds
    to a level crest, so on real ground the trench runs deeper than its design depth
    almost everywhere and the bank stands taller where the ground falls away. On the
    design this was written against, the drawn sections imply 11,948 m³ and the terrain
    model moves 20,096 m³: quoting the first to a contractor is a 68% shortfall.

    Nothing to print before a burn has run — the measured figure does not exist yet,
    and the calculated one is already in the table above.
    """
    q = data.burn_quantities or {}
    # NaN is what a burn over a DEM with nodata holes used to produce, and it is
    # truthy — so it passed the guard below and printed as "nan" in the measured
    # column. An unmeasurable figure is an absent one.
    cut, fill = _measured(q.get("cut_m3")), _measured(q.get("fill_m3"))
    if not cut and not fill:
        return []

    drawn_cut = _measured(getattr(data.balance, "total_cut_m3", 0.0)) or 0.0
    drawn_fill = _measured(getattr(data.balance, "total_fill_m3", 0.0)) or 0.0
    rows = [
        ["Cut — soil out", fmt_volume(drawn_cut), fmt_volume(cut),
         _ratio(drawn_cut, cut)],
        ["Fill — soil placed", fmt_volume(drawn_fill), fmt_volume(fill),
         _ratio(drawn_fill, fill)],
    ]
    return [DataTable(
        title="Earthmoving — drawn against measured",
        headers=["", "Geometric calculated (flat ground)", "Measured (this terrain)",
                 "Difference"],
        rows=rows,
        note=("Calculated against measured · Price the job on the measured column. "
              "Both features are built level — a swale floor and a berm crest each at "
              "one elevation — which is what makes them hold water evenly, and it is "
              "why sloping ground costs more earth than the drawn section implies. "
              "Site totals only: earthworks overlap and a berm sits outside its own "
              "footprint, so splitting the measured figure between features would "
              "report the splitting rule rather than the job."))]


def _measured(v):
    """A figure, or ``None`` where there is not a number to print."""
    if v is None:
        return None
    try:
        return v if math.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None


def _ratio(drawn, measured):
    """How much more the ground asks for than the drawing implies."""
    if not drawn or measured is None:
        return "—"
    return f"{measured / drawn:.2f}× the drawn figure"


def _batter(e):
    slope = getattr(e, "side_slope", None)
    if not slope:
        return "vertical"
    return f"1 in {float(slope):.1f}"


def _type_extra(e):
    """The one dimension that matters for this type and no other."""
    t = getattr(e, "type", "")
    if t == "dam":
        crest = getattr(e, "crest_elevation", None)
        keyed = "keyed into banks" if getattr(e, "key_into_banks", False) else "as drawn"
        return f"crest {crest:.2f} m ({keyed})" if crest else keyed
    if t == "diversion":
        return f"grade {getattr(e, 'gradient_pct', 0) or 0:.1f}%"
    if t == "swale" and getattr(e, "companion_berm", False):
        return "with berm"
    return ""


# ---- page 8 ---------------------------------------------------------------

def _page_verification(data):
    out = [PageBreak(),
           Heading(text="Checked against the ground", level=1, anchor="verify")]
    if not data.has_verification:
        out.append(Callout(tone="info", text=REASON_NO_VERIFY))
        return out

    v = data.verification
    if data.edits_since_verify:
        out.append(Callout(tone="warn", title="These measurements are out of date",
                           text=(f"The design has been edited "
                                 f"{data.edits_since_verify} time(s) since it was "
                                 "checked. The measured figures below describe an "
                                 "earlier version of it.")))

    out.append(Paragraph(text=(
        "The storage figures elsewhere in this report come from flooding each feature "
        "on the terrain model on its own. This page rebuilt the whole design at once "
        "and measured the water that stood in it, so it is the check on whether the "
        "features interfere with one another.")))
    out.append(_map_ref(data, "ponding",
                        "Water held before and after the earthworks."))
    out.extend(_volume_ladder(v, data))

    if v.caveats:
        out.append(Callout(tone="info", title="Attribution caveats",
                           text="\n".join(_renamed_caveat(str(c), data)
                                          for c in v.caveats)))
    if v.baseline_uncorrected:
        out.append(Callout(tone="warn",
                           title="Water that was already there could not be subtracted",
                           text=(f"{v.baseline_uncorrected} Every measured figure "
                                 "below still includes water that ponded here "
                                 "naturally before any earthwork.")))
    return out


#: The one group caveat that leads with feature names rather than a single name.
_POOL_CAVEAT = " impound one continuous pool"


def _renamed_caveat(text, data):
    """A caveat naming the features the way the tables above it do.

    ``build_verification`` writes these sentences from the raw names it was given —
    it has no access to the document's name map, and giving it one would make a pure
    verification function depend on how a report letters its rows. So the rename
    happens here, and only on the leading name: these two forms put the feature
    first ("Swale 3: a 1.00 m cell…", "Swale 3 + Dam 4 impound one continuous
    pool…"). Substituting anywhere in the prose would rewrite words that happen to
    match a feature name.
    """
    mapping = _by_raw_name(data)
    if not mapping:
        return text
    head, sep, rest = text.partition(": ")
    if sep and head in mapping:
        return f"{mapping[head]}{sep}{rest}"
    head, sep, rest = text.partition(_POOL_CAVEAT)
    if sep:
        renamed = " + ".join(mapping.get(n, n) for n in head.split(" + "))
        return f"{renamed}{sep}{rest}"
    return text


#: Marks a Δ whose at-grid reference is not the feature's capacity. In DejaVu Sans,
#: so it prints rather than boxing.
_OVERSTATED_MARK = "†"


def _overstated_note(v, data=None):
    """The sentence the dagger stands for, naming the features it marks.

    Print has no hover, so the explanation cannot live in a tooltip the way the panel's
    does; it goes under the table instead, where it can wrap. Naming the features keeps
    the marker from being a symbol the reader has to hunt for — which means naming them
    the way the table above does, hence ``_shown`` rather than the raw name.
    """
    flagged = [_shown(data, f) if data is not None else f.get("name", "")
               for f in v.per_feature if f.get("section_overstated")]
    if not flagged:
        return ""
    named = ", ".join(flagged[:4])
    more = f" and {len(flagged) - 4} more" if len(flagged) > 4 else ""
    plural = len(flagged) != 1
    return (f"  {_OVERSTATED_MARK} {named}{more}: this cell size cannot hold "
            f"{'their drawn sections' if plural else 'its drawn section'}, so read "
            f"Geometric for whether {'they are' if plural else 'it is'} big enough. "
            f"A footprint two cells across has no cell more than half a cell from its "
            f"own edge, so it cannot reach the depth it was drawn at.")


def _impoundment_note(v):
    """How much of the measured storage the banks hold above natural ground.

    The single most surprising number in the document, so it is stated rather than left
    to be inferred from a column that is larger than the one before it. On the Quail
    Island design it is roughly half the total: the drawn sections come to 21,500 m³ and
    the ground holds about 31,000 m³.
    """
    impounded = sum(f.get("impoundment_m3") or 0.0 for f in v.per_feature)
    sections = sum(f.get("section_m3") or 0.0 for f in v.per_feature)
    if sections <= 0 or impounded <= sections * 0.05:
        return ""
    return (f"  Across the design these features impound {fmt_volume(impounded)} more "
            f"than the {fmt_volume(sections)} of trench drawn for them — water the banks hold "
            f"above natural ground, and the reason At this grid exceeds Geometric.")


def _volume_ladder(v, data=None):
    """Rule 1: three figures, in derivation order, never merged.

    ``data`` carries the document's ``display_names``. This table lettered nothing
    at all while two other pages lettered from two different inputs, so it could
    print two rows reading "Swale 3" with no way to tell which was which.

    They are three different questions about one earthwork, not three estimates of one
    number. The first is **geometric calculated** from the drawn dimensions; the last
    two are **measured** by flooding the terrain. That division is the one the reader
    most needs, and it is why the middle column is usually the largest: a companion
    berm keyed into its banks holds water above natural ground, which no cross-section
    can predict.

    **Design storage is gone, and so is the allowance it was made of.** It was
    ``geometric`` less a blanket 20% freeboard, and a single site-wide fraction is not
    what determines freeboard on a real feature — the spillway does, and the spillway is
    sized separately on page 6 from a peak flow this column knows nothing about. Dropping
    the column left the deduction itself inside ``calculate_capacity``, where it still
    reached the build schedule under a heading that said *geometric*; every storage
    figure in the document is now the whole drawn shape, brim-full, and no rung of the
    ladder is a rule of thumb.

    Δ appears once in the whole document and compares two floods — what a feature holds
    on its own against what the finished site ponds there — so it isolates interaction
    between features and nothing else.
    """
    out = [Callout(tone="info", title="Reading this table", text=(
        "The first figure is geometric calculated from the dimensions you drew, and "
        "you can check it by hand: it is the cross-section volume brim-full — for a "
        "swale with a companion berm, the trench plus the berm's own section. The "
        "last two are "
        "measured by flooding the terrain: At this grid is what the feature impounds "
        "on this hillside on its own, and Measured is the pond it ends up with once "
        "everything is built. At this grid is usually the largest of the three, and "
        "that is not an error — a bank keyed into its ends holds water above natural "
        "ground and up the slope behind it, which is real storage no drawn section "
        "accounts for. Only Measured minus At-this-grid is an error term, and it means "
        "a neighbouring feature is changing where the water goes."))]

    rows = []
    for f in v.per_feature:
        if f.get("routing_only"):
            rows.append([_shown(data, f), fmt_volume(f.get("geometric_m3")),
                         "n/a — sub-cell", "n/a — sub-cell", "n/a — sub-cell"])
            continue
        if f.get("merged_with"):
            # One pool, two owners. Printing a share of it under either name would
            # report the sharing rule; the pool gets its own table below.
            rows.append([_shown(data, f), fmt_volume(f.get("geometric_m3")),
                         fmt_volume(f.get("rasterisable_m3")),
                         "shared pool", "see below"])
            continue
        if f.get("barrier_impounded"):
            # A dam holds against the hillside, not against a drawn section:
            # geometric and grid are one computation. Printing two identical
            # numbers would imply two independent derivations.
            rows.append([_shown(data, f),
                         f"{fmt_volume(f.get('analytic_m3'))} (barrier-impounded)",
                         "", fmt_volume(f.get("terrain_m3")),
                         fmt_pct(f.get("delta_pct"))])
            continue
        delta = (fmt_pct(f.get("delta_pct"))
                 if f.get("delta_pct") is not None else "—")
        # The marker rather than the sentence: this column shares a fixed page width
        # with four others in proportion to its longest cell, so spelling it out per
        # row would squeeze the volumes it is meant to qualify. The note below the
        # table names every marked feature, so the dagger is never left orphaned.
        if f.get("section_overstated"):
            delta = f"{delta} {_OVERSTATED_MARK}"
        rows.append([
            _shown(data, f),
            fmt_volume(f.get("geometric_m3")),
            fmt_volume(f.get("rasterisable_m3")),
            fmt_volume(f.get("terrain_m3")),
            delta,
        ])
    out.append(DataTable(
        title="Drawn storage against measured storage",
        headers=["Feature", "Geometric (drawn)", "At this grid (held)",
                 "Measured", "Δ vs grid"],
        rows=rows, wide=True,
        note=("The first column is geometric calculated, the last two measured. Δ "
              "compares what each feature holds alone against what the finished site "
              "ponds there, so it is diagnostic of interaction between features and "
              "not a pass or a fail." + _impoundment_note(v)
              + _overstated_note(v, data))))

    # Where two features hold one sheet of water, this is the row that tests the burn.
    # It has to appear, and appear with a Δ, or the "shared pool" cells above look like
    # a measurement that failed rather than one made at the right scale.
    group_rows = [[" + ".join(g.get("names", ())),
                   fmt_volume(g.get("rasterisable_m3")),
                   fmt_volume(g.get("terrain_m3")),
                   fmt_pct(g.get("delta_pct")) if g.get("delta_pct") is not None else "—"]
                  for g in (getattr(v, "merged_groups", None) or [])]
    if group_rows:
        out.append(DataTable(
            title="Features holding one pool between them",
            headers=["Features", "At this grid (check)", "Measured", "Δ vs grid"],
            rows=group_rows,
            note=("Their pools are continuous, so the water is measured once for the "
                  "set. Dividing it between them would report the division rather "
                  "than the burn: one contributes a wall and another a hole, and "
                  "there is no share that is true of either. This Δ tests the burn "
                  "for all of them together.")))

    # The "Standing water — context only" table lived here. It carried existing_m3
    # and total_m3 per feature, and total_m3 is the largest number in the section —
    # so a table explicitly labelled "not a design claim" was the one a reader was
    # most likely to quote. Both figures remain on the rows for anything that wants
    # them; the document no longer prints a table whose whole caption is a warning.
    return out


# ---- optional: simulation enrichment --------------------------------------

def _page_simulation(data):
    """Timing, but only where a routed simulation actually produced it.

    Everything else in this report is an event-total balance and makes no claim
    about *when* water arrives. These sections do, so they appear only when a
    simulation has run and they say plainly where they came from.
    """
    c = data.comparison
    if c is None:
        return []

    out = [PageBreak(),
           Heading(text="How the storm plays out over time", level=1,
                   anchor="simulation"),
           Callout(tone="info", title="These figures come from the simulation",
                   text=("The rest of this report totals the whole event. This "
                         "section is the only part that models the storm hour "
                         "by hour, so it is the only part that says anything "
                         "about timing or peak flow."))]

    rows = [
        ["Water leaving the block",
         fmt_volume(getattr(c.baseline, "exit_volume_m3", 0) if c.baseline else 0),
         fmt_volume(getattr(c.post, "exit_volume_m3", 0) if c.post else 0),
         fmt_pct(-c.exit_reduction_pct) if c.exit_reduction_pct else "—"],
        ["Fastest flow at the boundary",
         f"{getattr(c.baseline, 'peak_outflow_ls', 0):.0f} L/s" if c.baseline else "—",
         f"{getattr(c.post, 'peak_outflow_ls', 0):.0f} L/s" if c.post else "—",
         fmt_pct(-c.peak_reduction_pct) if c.peak_reduction_pct else "—"],
        ["When that peak arrives",
         f"{getattr(c.baseline, 'peak_outflow_time_hr', 0):.1f} hr" if c.baseline else "—",
         f"{getattr(c.post, 'peak_outflow_time_hr', 0):.1f} hr" if c.post else "—",
         f"{c.peak_delay_hr:+.1f} hr" if c.peak_delay_hr else "—"],
    ]
    out.append(DataTable(
        title="Before and after the earthworks",
        headers=["", "As it is now", "With your design", "Change"], rows=rows))

    out.append(ImageRef(key="hydrograph", caption=(
        "Flow leaving the block through the storm, before and after.")))
    out.append(ImageRef(key="fill_timeline", caption=(
        "How full each feature gets as the storm runs, and when it spills.")))
    return out


# ---- page 9 ---------------------------------------------------------------

def _page_appendix(data):
    out = [PageBreak(),
           Heading(text="Inputs, method and limits", level=1, anchor="appendix"),
           Paragraph(text=("Everything above came from these settings. If a "
                           "number looks wrong, this is the page that tells you "
                           "which input to change."))]

    inputs = data.inputs or {}
    if inputs:
        rows = [("Runoff method", _basis_label(inputs))]
        rows += [(k.replace("_", " ").capitalize(), str(v))
                 for k, v in sorted(inputs.items())
                 if k not in _UNUSED_BY_BASIS.get(
                     inputs.get("sizing_basis"), ())]
        out.append(KeyValueTable(title="Settings used", rows=rows))

    out.extend(_methodology())

    dem = data.dem or {}
    if dem:
        # The digest prefix goes through verbatim: 'sha256-sampled:' means the
        # hash covered a sample, which is enough to spot a substitution but not
        # to claim bit-identity. Relabelling it 'sha256' would be an overclaim.
        out.append(KeyValueTable(
            title="Terrain model",
            rows=[(k.replace("_", " ").capitalize(), str(v))
                  for k, v in dem.items()]))

    out.append(Heading(text="What this document is not", level=2))
    out.append(Paragraph(text=_NOT_A_SURVEY))
    return out


#: How the runoff depth was arrived at, in the words the dialog offers it under.
#: Keyed off the constants themselves so a renamed basis cannot silently fall
#: through to "not recorded" here while the panel still shows a label for it.
_BASIS_LABELS = {
    "coefficient": "Runoff coefficient (Lancaster)",
    "rainfall": "Total rainfall — all rain runs off",
    "runoff": "Surface runoff (SCS curve number)",
}

#: Inputs a given basis never reads. The settings table used to print every field
#: it was handed, alphabetically, so a site sized on a Lancaster runoff coefficient
#: still listed "Cn 61" — a curve number nothing in that run consulted. A reader
#: cannot tell a live input from a dormant default, and the appendix exists
#: precisely to say which input to change.
_UNUSED_BY_BASIS = {
    "coefficient": ("cn", "ground_condition", "moisture"),
    "rainfall": ("cn", "ground_condition", "moisture", "runoff_coefficient"),
    "runoff": ("runoff_coefficient",),
}


def _basis_label(inputs):
    basis = (inputs or {}).get("sizing_basis")
    return _BASIS_LABELS.get(basis, str(basis or "not recorded"))


def _methodology():
    """How the two kinds of figure are actually produced.

    The reading guide on page 2 says *which* figures are which. This says how each
    one is arrived at, because a reader who has to defend a number needs the method
    and not just the label. It sits in the appendix rather than up front: it is
    reference material, and page 2 has to stay short enough to be read.
    """
    return [
        Heading(text="How this model works", level=2),
        Paragraph(text=(
            "Two independent methods run over the same design, and the report keeps "
            "them apart everywhere.")),
        Callout(tone="info", title="The geometric calculated method", text=(
            "Each feature's drawn dimensions — length, top and bottom width, depth "
            "and batter — are turned into a cross-section, and the section is "
            "multiplied along the feature's length. Capacity, cut and fill all come "
            "from that one calculation, and spillway widths come from the weir "
            "equation on top of it. It assumes the ground under the feature is a "
            "flat plane. Nothing about this site enters the arithmetic, which is "
            "what makes it checkable by hand and what makes it wrong in a knowable "
            "direction: a level structure on sloping ground always costs more earth "
            "than its drawn section implies.")),
        Callout(tone="info", title="The measured method", text=(
            "The design is cut into the elevation model — inverts levelled, crests "
            "raised — and the resulting surface is flooded by filling every "
            "depression to its pour point. What each feature holds is the water "
            "standing in it, counted cell by cell. The catchment feeding it is "
            "traced by following flow directions over the same grid, so each cell "
            "is credited to the first feature that intercepts it and no hillside is "
            "counted twice. This accounts for slope, for banks that hold water above "
            "natural ground, and for features that interfere with one another — none "
            "of which any cross-section can predict.")),
        Paragraph(text=(
            "Where the two disagree, the disagreement is the finding. A measured "
            "storage above the geometric figure is usually a bank impounding water "
            "up the slope behind it. A measured storage below it usually means the "
            "grid cannot hold the section that was drawn — the report marks those "
            "features rather than leaving the gap to be read as an error.")),
    ]


def _map_ref(data, key, caption):
    """A map, or an honest note about why there isn't one."""
    reason = (data.maps or {}).get(key)
    full = f"{caption} {MAP_CAPTION_SUFFIX}"
    if reason:
        return MapRef(key=key, caption=full, reason=str(reason))
    return MapRef(key=key, caption=full, legend=_map_legend(data, key))


# Colours for the things on a map that are not earthworks and not a raster ramp.
# A key that names a colour the map does not use is worse than no key: this said
# "Site boundary" in red while the map drew it bright blue, under a comment
# claiming a test asserted the pair. It did not. Every one of these is now
# asserted against its source in tests/test_map_palette.py, and the boundary is
# read from the palette rather than transcribed at all.
_BOUNDARY_COLOUR = hex_of(AREA_OUTLINES["boundary"])
_EXIT_COLOUR = "#DC0000"          # baseline.py exit-point marker, 220,0,0
_SPILLWAY_COLOUR = "#1273B5"      # _symbols.spillway_symbol
_CONNECTION_COLOUR = "#3A608C"    # _symbols.connection_symbol, 58,96,140


def _type_colour(key):
    """The registry's colour for an earthwork type. Grey if it has none."""
    try:
        from terrainflow_assessment.core.registry.earthwork_types import get_type
        return get_type(key).style[1]
    except Exception:
        return "#7F8C8D"


def _ramp_entry(label, ramp):
    """A gradient legend row from a palette ramp, low to high."""
    from terrainflow_assessment.core.registry.map_palette import visible_stops

    stops = visible_stops(ramp)
    return LegendEntry(label=label, kind="ramp",
                       colour=stops[0][0] if stops else "",
                       colours=tuple(c for c, _l in stops))


def _types_present(data):
    """Earthwork types on this block, in the order the balance lists them.

    A key listing five types on a scheme with two swales is its own kind of
    wrong, so every map key is built from what the design actually contains.
    """
    seen = []
    for f in ((data.balance.per_feature if data.balance is not None
               else None) or []):
        t = f.get("ew_type")
        if t and t not in seen:
            seen.append(t)
    return seen


def _map_legend(data, key):
    """What the reader needs to be told the colours mean.

    Built from the same registry the map is drawn from, and only for the things
    actually on this map — a key listing five earthwork types on a scheme with
    two swales is its own kind of wrong.
    """
    from terrainflow_assessment.core.registry.map_palette import (
        WATER_CAPTURED,
        surface_runoff_ramp,
    )

    if key == "design":
        # Every earthwork type on the block, and nothing else. The spillway and
        # overflow-link entries came off with the layers they named — see
        # ``_page_design``.
        out = [LegendEntry(label=type_label(t), colour=_type_colour(t),
                           kind="fill" if t == "basin" else "line")
               for t in _types_present(data)]
        out.append(LegendEntry(label="Site boundary",
                               colour=_BOUNDARY_COLOUR, kind="line"))
        return out

    if key == "overview":
        # The summary map carries no earthworks, so it names none of them. What
        # it does carry is water — held, moving, and the ground that feeds it —
        # and none of that is self-explanatory over an aerial photograph.
        out = [
            _ramp_entry("Water held — shallow to deepest", WATER_CAPTURED),
            LegendEntry(label="Watercourse",
                        colour=stop_colour(STREAMS, "channel"), kind="line"),
        ]
        if data.catchment_outline:
            # One line per type, in the colour of the feature that catches it,
            # because that is how the map draws them.
            for t in _types_present(data):
                out.append(LegendEntry(
                    label=f"Catchment — {type_label(t).lower()}",
                    colour=_type_colour(t), kind="line"))
        out.append(LegendEntry(label="Site boundary",
                               colour=_BOUNDARY_COLOUR, kind="line"))
        return out

    if key == "flow":
        # The default ramp regardless of the panel's scale mode: the modes move
        # where the stops sit, not what colour each one is, so the key says the
        # same thing either way.
        return [
            _ramp_entry("Surface runoff — diffuse to channel",
                        surface_runoff_ramp()),
            LegendEntry(label="Watercourse",
                        colour=stop_colour(STREAMS, "channel"), kind="line"),
            _ramp_entry("Water held — shallow to deepest", WATER_CAPTURED),
            LegendEntry(label="Boundary crossing", colour=_EXIT_COLOUR,
                        kind="point"),
            LegendEntry(label="Site boundary", colour=_BOUNDARY_COLOUR,
                        kind="line"),
        ]

    if key == "ponding":
        return [
            _ramp_entry("Water captured — shallow to deepest", WATER_CAPTURED),
            LegendEntry(label="Site boundary", colour=_BOUNDARY_COLOUR,
                        kind="line"),
        ]
    return []
