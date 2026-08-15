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
    baseline_result: Optional[dict] = None
    balance: Any = None              # BalanceResult (design tier)
    balance_stores: Optional[list] = None
    earthworks: Optional[list] = None  # Earthwork objects, for the build schedule
    verification: Any = None         # VerificationResult
    comparison: Any = None           # ComparisonResult — optional enrichment
    spillway_rows: Optional[list] = None
    spillway_context: Optional[dict] = None
    area_subtotals: Optional[list] = None
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
    charts: dict = field(default_factory=dict)   # key -> True when renderable

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


def _shown(data, row, fallback_key="name"):
    """A row's display name: the document's, or its own where it has no id."""
    fid = row.get("id") if isinstance(row, dict) else getattr(row, "id", None)
    names = getattr(data, "display_names", None) or {}
    if fid in names:
        return names[fid]
    if isinstance(row, dict):
        return row.get(fallback_key) or "Unnamed"
    return getattr(row, fallback_key, "") or "Unnamed"


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
CALCULATED = "Calculated"
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
              "looks at the ground. You can check any of it with a tape measure and a "
              "calculator, and it would be the same on any site with the same "
              "dimensions. Capacity, cut and fill quantities, spillway sizes and berm "
              "heights are all calculated.")))
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

    out.append(Callout(
        tone="info", title="Which to quote",
        text=("Quote the calculated figures when you are setting out or pricing the "
              "job: they are the dimensions a contractor builds to and the earth that "
              "has to be shifted, and they do not move when the terrain model is "
              "re-run. Quote the measured storage when you are asking whether the "
              "scheme holds enough water — but read it as conditional on the banks "
              "being built as modelled, because most of the difference between the two "
              "is what those banks retain.")))

    out.append(Paragraph(text=(
        "The headline percentage of the storm held on site is measured on both halves: "
        "how much your features hold and how much water reaches each of them are both "
        "read off the terrain. Where this report gives a calculated figure beside a "
        "measured one, it says so.")))

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
        note=_tag("Calculated storage against measured inflow",
                  " ".join(notes)))


def _summary_cards(data):
    b, bal = data.baseline, data.balance
    cards = []
    if b is not None:
        cards.append(("Your catchment", f"{b.catchment_area_ha:.1f} ha",
                      "the ground draining through this block"))
        cards.append(("The storm", f"{b.rainfall_mm:.0f} mm / {b.duration_hr:.0f} hr",
                      f"{b.runoff_mm:.0f} mm of runoff"))
    if bal is not None:
        n = len([f for f in bal.per_feature])
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
    top = pts[:5]
    rows = [[p.get("label", "").split(":")[0] or f"Exit {i + 1}",
             f"{p.get('flow_ls', 0):.1f} L/s",
             fmt_volume(p.get("volume_m3", 0))]
            for i, p in enumerate(top)]
    notes = []
    if len(pts) > 5:
        rest = sum(p.get("volume_m3", 0) for p in pts[5:])
        notes.append(f"and {len(pts) - 5} smaller crossings, together about "
                     f"{fmt_volume(rest)}.")
    notes.append(_exit_threshold_note(data))
    out = [DataTable(title="Where water leaves the boundary",
                     headers=["Crossing", "Peak flow", "Volume over the event"],
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
                "This page counts water crossing your boundary. It answers "
                "where the water goes — and it only counts the crossings big "
                f"enough to draw ({_exit_threshold_phrase(data)}). The capture "
                "figure on page one is a different measurement "
                f"({fmt_volume(data.balance.site_exit_m3)}), and it answers how "
                "much of the storm was not captured, counted cell by cell "
                "across the whole site — including everywhere water seeps out "
                "below that threshold.\n\n"
                "That is why more water leaves the block than the crossings "
                "above add up to. They are not two estimates of one number and "
                "the difference between them is not an error. Quote them "
                "separately, say which one you mean, and never subtract one "
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
    out.append(_map_ref(data, "design",
                        "Every feature as drawn, with spillways and overflow links."))
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
            "Solid arrows are links you drew; dashed arrows are followed "
            "automatically downhill. Blue arrows are the ones water actually "
            "goes down at this storm, thicker where more of it does. Each box "
            "shows what the feature holds when dug and how full it gets."),
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
        note=_tag("Calculated storage against measured inflow",
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
        note=_tag("Calculated storage against measured inflow",
                  "Widths and water depths come from the weir equation; the flow "
                  "each has to pass is the peak off a catchment read from the "
                  "terrain.")))

    problems = []
    for r in data.spillway_rows:
        for p in (r.get("problems") or []):
            problems.append(f"{r.get('name', 'A feature')}: {p}")
    if problems:
        out.append(Callout(tone="warn", title="What needs attention",
                           text="\n".join(problems)))
    return out


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
        "Take this page to your contractor. Capacity is the working volume with "
        "the standard freeboard already taken off — what the feature holds in "
        "service, not the size of the hole.")))

    cut_by_id, fill_by_id = {}, {}
    for st in (data.balance_stores or []):
        cut_by_id[getattr(st, "id", None)] = getattr(st, "cut_vol_m3", 0.0)
        fill_by_id[getattr(st, "id", None)] = getattr(st, "fill_vol_m3", 0.0)

    enabled = [e for e in data.earthworks if getattr(e, "enabled", True)]
    names = [_shown(data, e) for e in enabled]
    rows = []
    for i, e in enumerate(enabled):
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
            fmt_volume(cut_by_id.get(getattr(e, "id", None))),
            fmt_volume(fill_by_id.get(getattr(e, "id", None))),
        ])
    # "Dam 1" already says it is a dam, so a Type column beside the name spent
    # a column of a twelve-column table restating it. "Type detail" stays —
    # that carries the one dimension that matters for the type and no other.
    out.append(DataTable(
        title="Every feature, as drawn",
        headers=["Feature", "Length", "Top width", "Bottom width",
                 "Depth", "Batter", "Type detail", "Soil", "Capacity",
                 "Cut", "Fill"],
        rows=rows, wide=True,
        note=_tag(CALCULATED, cut_fill_sentence(data.balance.total_cut_m3,
                                                data.balance.total_fill_m3))))
    out.extend(_earthmoving(data))
    return out


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
    cut, fill = q.get("cut_m3"), q.get("fill_m3")
    if not cut and not fill:
        return []

    drawn_cut = getattr(data.balance, "total_cut_m3", 0.0) or 0.0
    drawn_fill = getattr(data.balance, "total_fill_m3", 0.0) or 0.0
    rows = [
        ["Cut — soil out", fmt_volume(drawn_cut), fmt_volume(cut),
         _ratio(drawn_cut, cut)],
        ["Fill — soil placed", fmt_volume(drawn_fill), fmt_volume(fill),
         _ratio(drawn_fill, fill)],
    ]
    return [DataTable(
        title="Earthmoving — drawn against measured",
        headers=["", "Calculated (flat ground)", "Measured (this terrain)",
                 "Difference"],
        rows=rows,
        note=("Calculated against measured · Price the job on the measured column. "
              "Both features are built level — a swale floor and a berm crest each at "
              "one elevation — which is what makes them hold water evenly, and it is "
              "why sloping ground costs more earth than the drawn section implies. "
              "Site totals only: earthworks overlap and a berm sits outside its own "
              "footprint, so splitting the measured figure between features would "
              "report the splitting rule rather than the job."))]


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
                           text="\n".join(str(c) for c in v.caveats)))
    if v.baseline_uncorrected:
        out.append(Callout(tone="warn",
                           title="Water that was already there could not be subtracted",
                           text=(f"{v.baseline_uncorrected} Every measured figure "
                                 "below still includes water that ponded here "
                                 "naturally before any earthwork.")))
    return out


#: Marks a Δ whose at-grid reference is not the feature's capacity. In DejaVu Sans,
#: so it prints rather than boxing.
_OVERSTATED_MARK = "†"


def _overstated_note(v):
    """The sentence the dagger stands for, naming the features it marks.

    Print has no hover, so the explanation cannot live in a tooltip the way the panel's
    does; it goes under the table instead, where it can wrap. Naming the features keeps
    the marker from being a symbol the reader has to hunt for.
    """
    flagged = [f.get("name", "") for f in v.per_feature if f.get("section_overstated")]
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
    """Rule 1: four figures, in derivation order, never merged.

    ``data`` carries the document's ``display_names``. This table lettered nothing
    at all while two other pages lettered from two different inputs, so it could
    print two rows reading "Swale 3" with no way to tell which was which.

    They are four different questions about one earthwork, not four estimates of one
    number. The first two are **calculated** from the drawn dimensions; the last two are
    **measured** by flooding the terrain. That division is the one the reader most needs,
    and it is why the third column is usually the largest: a companion berm keyed into
    its banks holds water above natural ground, which no cross-section can predict.

    Δ appears once in the whole document and compares two floods — what a feature holds
    on its own against what the finished site ponds there — so it isolates interaction
    between features and nothing else.
    """
    out = [Callout(tone="info", title="Reading this table", text=(
        "The first two figures are calculated from the dimensions you drew, and you can "
        "check them by hand. Geometric is the cross-section volume — for a swale with a "
        "companion berm, the trench plus the berm's own section — and Design storage is "
        "that less your freeboard allowance. The last two are measured by flooding the "
        "terrain: At this grid is what the feature impounds on this hillside on its own, "
        "and Measured is the pond it ends up with once everything is built. At this grid "
        "is usually the larger of all four, and that is not an error — a bank keyed into "
        "its ends holds water above natural ground and up the slope behind it, which is "
        "real storage no drawn section accounts for. Only Measured minus At-this-grid is "
        "an error term, and it means a neighbouring feature is changing where the water "
        "goes."))]

    rows = []
    for f in v.per_feature:
        if f.get("routing_only"):
            rows.append([_shown(data, f), fmt_volume(f.get("analytic_m3")),
                         fmt_volume(f.get("geometric_m3")),
                         "n/a — sub-cell", "n/a — sub-cell", "n/a — sub-cell"])
            continue
        if f.get("merged_with"):
            # One pool, two owners. Printing a share of it under either name would
            # report the sharing rule; the pool gets its own table below.
            rows.append([_shown(data, f), fmt_volume(f.get("analytic_m3")),
                         fmt_volume(f.get("geometric_m3")),
                         fmt_volume(f.get("rasterisable_m3")),
                         "shared pool", "see below"])
            continue
        if f.get("barrier_impounded"):
            # A dam holds against the hillside, not against a drawn section:
            # design, geometric and grid are one computation. Printing three
            # identical numbers would imply three independent derivations.
            rows.append([f.get("name", ""),
                         f"{fmt_volume(f.get('analytic_m3'))} (barrier-impounded)",
                         "", "", fmt_volume(f.get("terrain_m3")),
                         fmt_pct(f.get("delta_pct"))])
            continue
        delta = (fmt_pct(f.get("delta_pct"))
                 if f.get("delta_pct") is not None else "—")
        # The marker rather than the sentence: this column shares a fixed page width
        # with five others in proportion to its longest cell, so spelling it out per
        # row would squeeze the volumes it is meant to qualify. The note below the
        # table names every marked feature, so the dagger is never left orphaned.
        if f.get("section_overstated"):
            delta = f"{delta} {_OVERSTATED_MARK}"
        rows.append([
            f.get("name", ""),
            fmt_volume(f.get("analytic_m3")),
            fmt_volume(f.get("geometric_m3")),
            fmt_volume(f.get("rasterisable_m3")),
            fmt_volume(f.get("terrain_m3")),
            delta,
        ])
    out.append(DataTable(
        title="Drawn storage against measured storage",
        headers=["Feature", "Design storage", "Geometric (drawn)",
                 "At this grid (held)", "Measured", "Δ vs grid"],
        rows=rows, wide=True,
        note=("The first two columns are calculated, the last two measured. Δ compares "
              "what each feature holds alone against what the finished site ponds "
              "there, so it is diagnostic of interaction between features and not a "
              "pass or a fail." + _impoundment_note(v) + _overstated_note(v))))

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

    # Context, physically separated: total_m3 is the largest number here and
    # would otherwise be read as the capacity.
    ctx_rows = [[f.get("name", ""), fmt_volume(f.get("existing_m3")),
                 fmt_volume(f.get("total_m3"))]
                for f in v.per_feature
                if f.get("existing_m3") or f.get("total_m3")]
    if ctx_rows:
        out.append(DataTable(
            title="Standing water — context only, not a design claim",
            headers=["Feature", "Already ponding before", "Pool you would see"],
            rows=ctx_rows))
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
        out.append(KeyValueTable(
            title="Settings used",
            rows=[(k.replace("_", " ").capitalize(), str(v))
                  for k, v in sorted(inputs.items())]))

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
        out = []
        seen = []
        for f in ((data.balance.per_feature if data.balance is not None
                   else None) or []):
            t = f.get("ew_type")
            if t and t not in seen:
                seen.append(t)
        for t in seen:
            kind = "fill" if t == "basin" else "line"
            out.append(LegendEntry(label=type_label(t), colour=_type_colour(t),
                                   kind=kind))
        if data.spillway_rows:
            out.append(LegendEntry(label="Spillway",
                                   colour=_SPILLWAY_COLOUR, kind="point"))
        out.append(LegendEntry(label="Overflow link",
                               colour=_CONNECTION_COLOUR, kind="line"))
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
