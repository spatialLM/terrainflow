"""
verification_table.py — per-earthwork design vs measured storage.

Replaces a single site-wide "Verified · Δ −38%" chip, which conflated three unrelated
gaps and so meant nothing. Three columns, each answering one question:

  Geometric    the drawn shape, exactly          "How big is the hole I drew?"
  At grid      that shape as the terrain model   "What did the terrain model make of it?"
               represents it
  Measured     ponding read off the burned DEM   "What did the burn actually produce?"

**Geometric is the capacity figure.** It is computed from the cross-section and never
touches the DEM, so it is the number to trust for "will this hold enough". At-grid and
Measured check placement, routing and the burn against real terrain — for a feature the
terrain model cannot hold, they are not storage volumes at all.

There was a fourth column, **Design**, and it led: ``geometric`` less a blanket 20%
freeboard. It is gone, and so is the deduction it was made of. Freeboard on a real
feature is set by its spillway, which is sized on the Spillways table from a peak flow
this column knew nothing about — so a figure from a rule of thumb sat first in a run of
columns meant to be compared against each other, and invited exactly the comparison it
could not support. Dropping the column left the 20% deduction itself inside
``calculate_capacity``, still reaching the build schedule under a heading that said
*geometric*, so it is gone too: **Geometric is the whole drawn section, brim-full.**
The panel and the report now say the same thing.

Only the last comparison is a correctness check. **Δ = Measured vs At-grid**, and it
should sit near zero — a non-zero Δ there is a burn bug and nothing else. The remaining
expected gap is Geometric → At-grid, the terrain model cutting the section that was
drawn. The burner tapers the walls to the feature's own two widths, so the trench it cuts
is the trapezoid rather than a full-depth rectangle a third to a half larger; and At-grid
is *measured* off that trench rather than modelled beside it.

Where the gap is still material the grid genuinely cannot hold the section — a footprint
two cells across has no cell more than half a cell from its own edge, so it cannot reach
full depth — and ``section_overstated`` says so, withholding the green from a Δ that did
not test that volume.

For a swale with a companion berm, Geometric is trench **plus** the berm's own section,
while the last two columns are the trench alone. ``section_m3`` and ``berm_credit_m3``
carry the split, so the comparison is trench against trench.

Rows for sub-cell features carry no measured figure at all. A feature narrower than
one DEM cell is validated for placement and routing only; claiming a storage volume
for it would be inventing precision the grid does not have.

Two features whose pools run together are blank in the same two columns, for the
neighbouring reason: there *is* a measurement, but it is not a measurement of either
one. A basin and the dam on its lip impound a single sheet of water, and splitting it
between them — by overlap, or to whichever touches more of it — reports the split
rather than the burn. The pair gets one line in the footer instead.
"""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from terrainflow_assessment.qgis import _theme
from terrainflow_assessment.qgis import help_text as H
from terrainflow_assessment.qgis._theme import brush as _brush

_INK = _theme.INK
_MUTED = _theme.MUTED
_FAINT = _theme.FAINT
_HAIRLINE = _theme.HAIRLINE
_HAIRLINE_STRONG = _theme.HAIRLINE_STRONG
_GROUND = _theme.GROUND
_SURFACE = _theme.SURFACE
_GOOD = _theme.GOOD
_WARN = _theme.WARN
_BAD = _theme.BAD
# Informational, not a fault — the outflow-spillway blue, so the two review tables in
# the Design/Verify stages read as one design language rather than two.
_INFO = _theme.WATER

# Δ bands. Under 5% the burn reproduces what the grid can represent; past 15% it is
# not a rounding artefact and something in the burn is wrong.
_DELTA_GOOD = 5.0
_DELTA_WARN = 15.0

# Marks a Δ measured against a grid figure that is not this feature's capacity. Kept to
# one character because the column is the narrowest in the table; the footer and the row
# tooltip carry the sentence.
_OVERSTATED_MARK = "†"

# Below this, pre-existing ponding is a stray cell or two of grid noise and saying
# "1 m³ was already ponding there" would be more distracting than informative. A
# feature genuinely built into a hollow clears it by orders of magnitude.
_EXISTING_PONDING_FLOOR = 10.0

_HEADERS = ("Feature", "Geometric", "At grid", "Measured", "Δ")

# Per-column hover copy. The header row has no space for the distinction that matters
# most — which columns are calculated and which are measured — so it is said here and
# summarised in one line above the table.
_HEADER_TIPS = H.VERIFY_TABLE_HEADER_TIPS

_SUBHEAD = H.VERIFY_TABLE_SUBHEAD

_QSS = f"""
QTableWidget {{
    background: {_SURFACE}; border: 1px solid {_HAIRLINE};
    border-radius: 6px; gridline-color: {_HAIRLINE};
    selection-background-color: #e9f3ee; selection-color: {_INK};
}}
QHeaderView::section {{
    background: {_GROUND}; border: none;
    border-bottom: 1px solid {_HAIRLINE_STRONG};
    padding: 4px 5px; font-size: 10px; font-weight: 700; color: {_MUTED};
}}
"""


class VerificationTable(QWidget):
    """Per-feature verification, with a footer that explains the Δ in words."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)

        # Above the table, not below it: a reader who takes At-grid for the capacity has
        # already done so by the time they reach the footer.
        self.subhead = QLabel(_SUBHEAD)
        self.subhead.setWordWrap(True)
        self.subhead.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
        lay.addWidget(self.subhead)

        self.table = QTableWidget(0, len(_HEADERS))
        self.table.setHorizontalHeaderLabels(list(_HEADERS))
        for c, tip in enumerate(_HEADER_TIPS):
            item = self.table.horizontalHeaderItem(c)
            if item is not None:
                item.setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(_QSS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        lay.addWidget(self.table)

        self.footer = QLabel("")
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(f"color: {_MUTED}; font-size: 10.5px;")
        lay.addWidget(self.footer)

        self.setVisible(False)

    # ------------------------------------------------------------------ API

    def set_result(self, result, cell_size_m=1.0):
        """Populate from a :class:`~terrainflow_assessment.modules.reporting.
        VerificationResult`, or hide when there is nothing to show."""
        rows = list(getattr(result, "per_feature", None) or []) if result else []
        if not rows:
            self.setVisible(False)
            self.table.setRowCount(0)
            self.footer.setText("")
            return

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._fill_row(r, row)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._size_to_contents(len(rows))
        self.footer.setText(self._footer_text(result, rows, cell_size_m))
        self.setVisible(True)

    # ------------------------------------------------------------------ internals

    def _fill_row(self, r, row):
        routing_only = bool(row.get("routing_only"))
        measured = row.get("terrain_m3")
        delta = row.get("delta_pct")

        # A dam has no drawn cross-section — the shape of the water is the shape of
        # the valley — so repeating one number across two columns would imply an
        # agreement that was never tested. Show it once, in Geometric, tagged with why
        # it is not a drawn section. That is the report's own convention for this row
        # (``_volume_ladder`` prints "N (barrier-impounded)"), and following it here is
        # what stopped a dam losing its only volume figure when Design went.
        barrier = bool(row.get("barrier_impounded"))

        # The terrain model does not hold this feature's drawn section, so neither of
        # the last two columns is a capacity and the Δ between them is not a verdict on
        # one. Marked rather than hidden — the burn check itself is still worth reading.
        overstated = bool(row.get("section_overstated"))

        # This feature's pool is continuous with a neighbour's. The water is measured,
        # once, for the set; naming a number here would be naming a share of it.
        merged = bool(row.get("merged_with"))
        if routing_only:
            measured_text = "sub-cell"
        elif merged:
            measured_text = "shared"
        else:
            measured_text = _m3(measured)

        cells = [
            (row.get("name", "—"), _INK, Qt.AlignmentFlag.AlignLeft),
            (f"{_m3(row.get('analytic_m3'))} impounded" if barrier
             else _m3(row.get("geometric_m3")),
             _FAINT if barrier else _MUTED, Qt.AlignmentFlag.AlignRight),
            ("—" if barrier else _m3(row.get("rasterisable_m3")),
             _FAINT if barrier else _INK, Qt.AlignmentFlag.AlignRight),
            (measured_text,
             _FAINT if (routing_only or merged) else _INK,
             Qt.AlignmentFlag.AlignRight),
            (self._delta_text(delta, routing_only or merged, overstated),
             self._delta_colour(delta, routing_only or merged, overstated),
             Qt.AlignmentFlag.AlignRight),
        ]
        for c, (text, colour, align) in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setForeground(_brush(colour))
            item.setTextAlignment(int(align | Qt.AlignmentFlag.AlignVCenter))
            if c == 0:
                item.setToolTip(self._row_tooltip(row))
            self.table.setItem(r, c, item)

    @staticmethod
    def _delta_text(delta, routing_only, overstated=False):
        if routing_only:
            return "—"
        if delta is None:
            return "—"
        return f"{delta:+.0f}% {_OVERSTATED_MARK}" if overstated else f"{delta:+.0f}%"

    @staticmethod
    def _delta_colour(delta, routing_only, overstated=False):
        """Green means "the burn is right AND that means something here".

        Only the pass colour is withheld on a flagged row: a Δ of +0% against a grid
        figure that is not the feature's capacity is the reassurance this table used to
        give wrongly, so it reads informational blue instead. A genuine burn error keeps
        its amber or red — the flag is about which volume was compared, not about how
        badly the burn missed it, and suppressing a real fault to make a caveat would
        trade one misreading for a worse one.
        """
        if routing_only or delta is None:
            return _FAINT
        magnitude = abs(delta)
        if magnitude <= _DELTA_GOOD:
            return _INFO if overstated else _GOOD
        return _WARN if magnitude <= _DELTA_WARN else _BAD

    def _row_tooltip(self, row):
        """Spell out this feature's three gaps, since only one of them is a defect."""
        name = row.get("name", "This feature")
        if row.get("barrier_impounded"):
            measured = row.get("terrain_m3")
            design = row.get("analytic_m3") or 0.0
            return (
                f"{name} impounds water against the terrain, so there is no drawn "
                f"cross-section to rasterise — the shape of the pool is the shape of "
                f"the valley. The {design:,.0f} m³ under Geometric is already a "
                f"flooded-volume calculation over this same grid, which is why it is "
                f"marked impounded rather than read as a drawn section.\n\n"
                f"Δ therefore compares the burn ({_m3(measured)} m³) against that "
                f"calculation directly. Near zero means the two agree; there is no "
                f"resolution term to separate out."
            )
        if row.get("routing_only"):
            return (f"{name} is narrower than one DEM cell. It is verified for "
                    f"placement and routing only — a storage volume measured off the "
                    f"grid would be an artefact of the grid, not of the design.")
        merged = row.get("merged_with")
        if merged:
            others = ", ".join(merged)
            return (f"{name} impounds one continuous pool with {others}. The water is "
                    f"real and it is measured — but it is held by the set, not by "
                    f"{name}, and there is no honest way to divide it: one contributes "
                    f"a wall, another a hole. Splitting it would report the split. The "
                    f"footer gives the pair's measured volume against the pair's "
                    f"at-grid capacity, which is the comparison that tests the burn.")

        geometric = row.get("geometric_m3") or 0.0
        at_grid = row.get("rasterisable_m3") or 0.0
        measured = row.get("terrain_m3")
        penalty = row.get("resolution_penalty_m3") or 0.0

        overstated = bool(row.get("section_overstated"))
        gap_pct = row.get("section_gap_pct")

        section = row.get("section_m3") or geometric
        impoundment = row.get("impoundment_m3") or 0.0
        cut = row.get("cut_m3")
        excavation = row.get("excavation_m3")

        parts = [f"{name}"]
        if geometric > 0:
            parts.append(
                f"Geometric {geometric:,.0f} m³ is the shape you drew, calculated from "
                f"your dimensions and brim-full; the two columns after it are measured "
                f"off the ground.")
        if at_grid > 0 and section > 0 and impoundment > section * 0.05:
            parts.append(
                f"At grid {at_grid:,.0f} m³ is what this feature impounds on this "
                f"hillside, flooded on the DEM on its own. That is {impoundment:,.0f} m³ "
                f"more than the {section:,.0f} m³ trench you drew — water the bank holds "
                f"above natural ground and up the slope behind it. No cross-section "
                f"predicts it, which is why it is measured rather than calculated.")
        elif at_grid > 0:
            parts.append(
                f"At grid {at_grid:,.0f} m³ is what this feature impounds on this "
                f"hillside, flooded on the DEM on its own — close to the "
                f"{section:,.0f} m³ you drew, so the ground here is adding little.")
        # Trust the model. The fallback here was `penalty / section * 100.0`, and it
        # was reached in precisely the case `build_verification` had already decided
        # the divisor was zero (`gap_pct = penalty / section * 100.0 if section > 0
        # else None`) — so the guard was a ZeroDivisionError waiting for a non-barrier
        # feature with a non-zero resolution penalty and no drawn section. Inside a
        # tooltip builder, which runs from a Qt paint path.
        if cut is not None and penalty and gap_pct is not None:
            parts.append(
                f"The trench holds {cut:,.0f} m³ to its own rim against the "
                f"{section:,.0f} m³ drawn, {gap_pct:+.0f}%. That is purely whether the grid "
                f"could hold the section — a feature narrower than about three cells "
                f"cannot reach full depth at any cell size.")
        # Deliberately a separate sentence from the one above, and never folded into it.
        # They are two measurements of different things that agree only on flat ground:
        # the rim figure is what fits in the hole, this is what comes out of the hillside,
        # and the gap is the over-dig a level invert makes as the ground rises away from
        # the pour point. Narrating the second against the first's percentage was the
        # defect this pair was split to prevent.
        if excavation is not None and excavation > 0:
            parts.append(
                f"Excavation {excavation:,.0f} m³ is the earth that comes out — measured "
                f"off the ground rather than the section, so it carries the extra dig "
                f"where the hillside rises away from the trench's level floor. This is "
                f"the figure the Site Water Plan prices from.")
        if measured is not None and at_grid > 0:
            parts.append(
                f"Measured {measured:,.0f} m³ against {at_grid:,.0f} m³ at grid is the "
                f"only comparison that tests anything. Both are floods, so a gap here "
                f"means the finished site ponds differently from this feature on its "
                f"own — a neighbour, and nothing else.")
        berm = row.get("berm_credit_m3") or 0.0
        if berm:
            parts.append(
                f"Of the drawn {geometric:,.0f} m³, {berm:,.0f} m³ is the companion "
                f"berm's own section — an estimate from the spoil. What the bank "
                f"actually holds back is in the two measured columns, and it is usually "
                f"a good deal more than its own section.")
        if overstated:
            parts.append(
                f"{_OVERSTATED_MARK} The grid could not hold the section you drew here, "
                f"so read Geometric ({geometric:,.0f} m³) for whether it is big enough. "
                f"A near-zero Δ above is only the two floods agreeing.")
        return "\n\n".join(parts)

    def _footer_text(self, result, rows, cell_size_m):
        real = [r for r in rows if not r.get("routing_only")
                and r.get("delta_pct") is not None]
        sub_cell = len(rows) - len([r for r in rows if not r.get("routing_only")])

        bits = []

        # Leads, because it invalidates every other sentence below it: without the
        # baseline subtraction each measured figure includes water that ponded there
        # naturally, so a feature in a hollow reads high and the delta is not a burn
        # error at all.
        uncorrected = getattr(result, "baseline_uncorrected", None)
        if uncorrected:
            bits.append(
                f"Measured storage still includes natural ponding: {uncorrected}. "
                f"Deltas below overstate storage for any feature sitting in ground "
                f"that already ponds."
            )
        if real:
            worst = max(real, key=lambda r: abs(r["delta_pct"]))
            worst_pct = worst["delta_pct"]
            if abs(worst_pct) <= _DELTA_GOOD:
                bits.append(
                    f"Every feature ponds within {abs(worst_pct):.0f}% of what it holds "
                    f"on its own, so nothing here is interfering with anything else.")
            else:
                bits.append(
                    f"{worst['name']} ponds {worst_pct:+.0f}% against what it holds on "
                    f"its own. Both figures are floods, so that is a neighbouring "
                    f"feature changing where its water goes — not resolution "
                    f"and not the burn.")

        # The headline the two measured columns cannot show on their own: how much of the
        # storage is the hillside rather than the cross-section. On a keyed design this is
        # most of it, and it is the reason the live score no longer reads "full" on
        # features that have half their pond in hand.
        impounded = sum(r.get("impoundment_m3") or 0.0 for r in rows)
        sections = sum(r.get("section_m3") or 0.0 for r in rows)
        if sections > 0 and impounded > sections * 0.05:
            bits.append(
                f"Across the design your features impound {impounded:,.0f} m³ more than "
                f"the {sections:,.0f} m³ of trench you drew — water the banks hold above "
                f"natural ground. That is real storage and it is what the live score "
                f"sizes against.")

        # Purely the grid-fidelity term: cut against drawn section. It has nothing to say
        # about the berm, which is why it is measured against section_m3 and not against
        # a Geometric that also carries one.
        flagged = [r for r in rows if r.get("section_overstated")]
        penalties = [(r["name"],
                      r["resolution_penalty_m3"] / (r.get("section_m3")
                                                    or r["geometric_m3"]) * 100.0)
                     for r in rows
                     if (r.get("section_m3") or r.get("geometric_m3"))
                     and r.get("resolution_penalty_m3")]
        if penalties:
            name, pct = max(penalties, key=lambda p: abs(p[1]))
            bits.append(
                f"The widest gap between the trench you drew and the one the grid could "
                f"cut is {pct:+.0f}% on {name}.")

        if flagged:
            names = ", ".join(r.get("name", "?") for r in flagged[:3])
            more = f" and {len(flagged) - 3} more" if len(flagged) > 3 else ""
            bits.append(
                f"{_OVERSTATED_MARK} marks {names}{more}: {_grid(cell_size_m)} cannot "
                f"hold {'their sections' if len(flagged) != 1 else 'its section'}, so "
                f"read Geometric for capacity there.")

        # The comparison the member rows could not make. Placed before the existing-
        # ponding lines because it explains two blank columns the reader has already
        # hit; without it, "shared" reads as a failure to measure.
        for group in (getattr(result, "merged_groups", None) or [])[:3]:
            joined = " + ".join(group.get("names", ()))
            gd = group.get("delta_pct")
            verdict = "" if gd is None else f" — Δ {gd:+.0f}%"
            bits.append(
                f"{joined} hold one pool between them: {group.get('terrain_m3', 0.0):,.0f} "
                f"m³ measured against {group.get('rasterisable_m3', 0.0):,.0f} m³ at "
                f"grid{verdict}. Neither row above claims it.")

        # Every figure in the table is marginal — what the design adds. For a feature
        # built into ground that already ponds, that under-describes the pool someone
        # standing there would see, so name all three: existing, added, total.
        for r in rows:
            existing = r.get("existing_m3") or 0.0
            total = r.get("total_m3")
            if existing < _EXISTING_PONDING_FLOOR or total is None:
                continue
            bits.append(
                f"{r.get('name', 'This feature')} holds {total:,.0f} m³ in total — "
                f"{existing:,.0f} m³ was already ponding there, and it adds "
                f"{r.get('terrain_m3') or 0.0:,.0f} m³."
            )

        if sub_cell:
            bits.append(
                f"{sub_cell} feature{'s' if sub_cell != 1 else ''} narrower than one "
                f"cell ({_cell_width(cell_size_m)}) "
                f"{'are' if sub_cell != 1 else 'is'} checked for placement and "
                f"routing only.")
        return " ".join(bits)

    def _size_to_contents(self, n_rows):
        header = self.table.horizontalHeader().height()
        row_h = self.table.rowHeight(0) if n_rows else 20
        self.table.setFixedHeight(header + row_h * min(n_rows, 8) + 6)


def _grid(cell_size_m):
    """"the 1.0 m grid", or "this cell size" when the grid is not known.

    R-8: `set_result` has threaded `cell_size_m` from `panel.py` all along and
    `_footer_text` never read it, so both caveats below talked about "one cell" and
    "this cell size" without ever saying how big a cell is. That is the one number
    that makes either sentence actionable — "narrower than one cell" means something
    different on a 1 m grid and a 5 m one, and the reader cannot tell which they have.
    """
    try:
        size = float(cell_size_m)
    except (TypeError, ValueError):
        return "this cell size"
    return f"the {size:g} m grid" if size > 0 else "this cell size"


def _cell_width(cell_size_m):
    """"1.0 m" for the parenthetical beside "narrower than one cell"."""
    try:
        size = float(cell_size_m)
    except (TypeError, ValueError):
        return "this grid"
    return f"{size:g} m" if size > 0 else "this grid"


def _m3(value):
    return "—" if value is None else f"{value:,.0f}"

