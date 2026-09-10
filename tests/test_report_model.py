"""Tests for the pure report model — content decisions, wording, degradation.

The headline capability under test is that a complete report builds from the
design tier alone, with no simulation anywhere in sight.
"""

import pytest

from terrainflow_assessment.modules.report_model import (
    ALLOWED_DISCLAIMERS,
    FORBIDDEN_WORDS,
    REASON_ALL_DISABLED,
    REASON_NO_BASELINE,
    REASON_NO_DESIGN,
    REASON_NO_VERIFY,
    Callout,
    DataTable,
    Hero,
    ImageRef,
    KeyValueTable,
    MapRef,
    Paragraph,
    ReportData,
    StatGrid,
    build_flow_graph,
    build_report,
)
from terrainflow_assessment.modules.reporting import (
    BaselineReport,
    VerificationResult,
    cut_fill_sentence,
    drain_wording,
    fill_wording,
    fmt_volume,
    round_volume,
    unique_names,
)
from terrainflow_assessment.modules.water_balance import BalanceResult

# ---------------------------------------------------------------- fixtures

def _feature(fid="a", name="Swale 1", **kw):
    row = {
        "id": fid, "name": name, "ew_type": "swale",
        "direct_catchment_m2": 34000.0, "direct_inflow_m3": 900.0,
        "upstream_inflow_m3": 300.0, "total_inflow_m3": 1200.0,
        "stored_m3": 1100.0, "infiltration_m3": 60.0,
        "infiltration_buffer_m3": 90.0, "drain_hours": 18.0,
        "overflow_m3": 0.0, "capacity_m3": 1250.0, "fill_pct": 88.0,
        "overflowed": False, "target_id": None, "is_user_link": False,
        "is_terminal": True,
    }
    row.update(kw)
    return row


def _balance(**kw):
    kwargs = {
        "capture_pct": 78.0, "total_inflow_m3": 2521.0,
        "total_captured_m3": 1966.0, "site_exit_m3": 555.0,
        "uncaptured_m3": 402.0, "routed_exit_m3": 153.0,
        "total_capacity_m3": 2500.0, "total_infiltration_m3": 120.0,
        "infiltration_buffer_m3": 180.0, "counts_infiltration": True,
        "total_cut_m3": 7944.0, "total_fill_m3": 3533.0,
        "terminal_deficit_m3": 0.0, "mass_balance_ok": True,
        "routing_warnings": [], "per_feature": [_feature()],
    }
    kwargs.update(kw)
    return BalanceResult(**kwargs)


def _baseline(**kw):
    kwargs = {
        "site_name": "Quail Island", "crs": "EPSG:2193", "cell_size_m": 1.0,
        "catchment_area_ha": 36.2, "rainfall_mm": 120.0, "duration_hr": 24.0,
        "cn": 61, "runoff_mm": 52.3, "total_runoff_m3": 21715.0,
        "exit_volume_m3": 10767.0,
        "exit_points": [
            {"label": f"Exit {i}: x L/s", "flow_ls": 40.0 - i,
             "volume_m3": 3000.0 - i * 100} for i in range(1, 9)],
    }
    kwargs.update(kw)
    return BaselineReport(**kwargs)


def _data(**kw):
    kwargs = {
        "site_name": "Quail Island", "generated_at": "2026-08-09 14:22",
        "plugin_version": "0.2.0", "run_tag": "120mm·24h·C0.50·0.5ha",
        "baseline": _baseline(), "balance": _balance(),
    }
    kwargs.update(kw)
    return ReportData(**kwargs)


def _text_of(report):
    """Every rendered string in the document, flattened."""
    out = []
    for s in report.sections:
        for attr in ("text", "title", "caption", "label", "value", "sub",
                     "note", "reason"):
            v = getattr(s, attr, None)
            if isinstance(v, str):
                out.append(v)
        for row in getattr(s, "rows", []) or []:
            out.extend(str(c) for c in row)
        for card in getattr(s, "cards", []) or []:
            out.extend(str(c) for c in card)
        out.extend(str(h) for h in getattr(s, "headers", []) or [])
    return "\n".join(out)


def _sections(report, kind):
    return [s for s in report.sections if isinstance(s, kind)]


def _table(report, title):
    """The one DataTable with this title. Raises if the title has moved."""
    found = [t for t in _sections(report, DataTable) if t.title == title]
    assert found, f"no table titled {title!r}"
    return found[0]


def _cell(table, row, header):
    """A cell by column *heading* rather than by index.

    Columns get added and removed — a Type column dropped here, a Capacity
    column added there — and an index-based assertion silently starts reading
    its neighbour rather than failing.
    """
    assert header in table.headers, f"{header!r} not in {table.headers}"
    return table.rows[row][table.headers.index(header)]


# ---------------------------------------------------------------- the matrix

class TestDegradation:
    def test_full_stack_builds(self):
        r = build_report(_data(verification=VerificationResult(
            analytic_total_m3=1000.0, terrain_total_m3=950.0,
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                          "terrain_m3": 950.0, "delta_pct": -13.6}])))
        assert r.completeness == {"baseline": True, "design": True, "verify": True}
        assert REASON_NO_VERIFY not in _text_of(r)

    def test_balance_only_no_simulation(self):
        """The headline new capability: a full report with comparison=None."""
        r = build_report(_data())
        assert r.completeness["design"] is True
        text = _text_of(r)
        assert "78%" in text
        # Nothing may imply timing — that is the simulation's claim, not ours.
        for banned in ("hydrograph", "peak flow reduction", "by hour"):
            assert banned.lower() not in text.lower()

    def test_baseline_only(self):
        r = build_report(_data(balance=None))
        assert r.completeness == {"baseline": True, "design": False, "verify": False}
        text = _text_of(r)
        assert REASON_NO_DESIGN in text
        assert REASON_NO_VERIFY in text
        # The site pages still carry real content — that is what makes a
        # baseline-only report worth printing at all.
        assert "36.2 ha" in text

    def test_all_earthworks_disabled_is_not_none_drawn(self):
        r = build_report(_data(balance=None, earthworks=[object()]))
        text = _text_of(r)
        assert REASON_ALL_DISABLED in text
        assert REASON_NO_DESIGN not in text

    def test_empty_data_does_not_raise(self):
        r = build_report(ReportData())
        assert r.sections
        assert REASON_NO_BASELINE in _text_of(r)

    def test_absent_sections_keep_their_heading(self):
        """Silent omission is how a reader misses that verification never ran."""
        from terrainflow_assessment.modules.report_model import Heading
        headings = {h.text for h in _sections(build_report(ReportData()), Heading)}
        for expected in ("The design", "Checked against the ground",
                         "Overflow safety", "Build schedule"):
            assert expected in headings


# ---------------------------------------------------------------- rule 3

class TestSuppression:
    def test_mass_balance_failure_replaces_the_hero(self):
        r = build_report(_data(balance=_balance(mass_balance_ok=False)))
        assert not _sections(r, Hero)
        bad = [c for c in _sections(r, Callout) if c.tone == "bad"]
        assert bad and "do not balance" in bad[0].title.lower()
        # Still printed, unclamped, just not as the headline.
        assert "78%" in _text_of(r)

    def test_zero_inflow_with_baseline_is_a_drainage_finding(self):
        """Every feature measured, and not one of them receives anything."""
        r = build_report(_data(balance=_balance(
            total_inflow_m3=0.0, capture_pct=0.0,
            per_feature=[_feature(total_inflow_m3=0.0)])))
        hero = _sections(r, Hero)[0]
        assert "0%" not in hero.value
        assert "drains into it" in hero.label
        assert "flow paths" in hero.sub

    def test_no_flow_grid_is_not_reported_as_a_drainage_fault(self):
        """`total_inflow_m3` is site-wide runoff and is 0 whenever the design tier
        has no flow grid yet. Reading that as "nothing drains into your features"
        diagnoses a fault that may not exist."""
        r = build_report(_data(balance=_balance(total_inflow_m3=0.0,
                                                capture_pct=0.0,
                                                per_feature=[])))
        heroes = _sections(r, Hero)
        assert not any("drains into it" in h.label for h in heroes), (
            "an unmeasured design was reported as a badly placed one")

    def test_drain_hours_none_never_renders_as_zero(self):
        r = build_report(_data(balance=_balance(
            per_feature=[_feature(drain_hours=None)])))
        text = _text_of(r)
        assert "Holds water — does not empty by soaking" in text

    def test_capture_pct_is_not_clamped(self):
        r = build_report(_data(balance=_balance(capture_pct=104.0)))
        assert "104%" in _text_of(r)

    def test_certification_words_appear_only_in_disclaimers(self):
        """A landowner takes this to a bank. It must not imply it certifies
        anything — but the disclaimers legitimately say 'not a certification'."""
        r = build_report(_data(verification=VerificationResult(
            caveats=["a caveat"],
            per_feature=[{"name": "Swale 1", "analytic_m3": 1.0}])))
        text = _text_of(r)
        for allowed in ALLOWED_DISCLAIMERS:
            text = text.replace(allowed, "")
        text = text.lower()
        for word in FORBIDDEN_WORDS:
            assert word not in text, f"{word!r} leaked outside a disclaimer"

    def test_the_disclaimers_are_actually_present(self):
        """Otherwise the exemption above could hide their removal."""
        text = _text_of(build_report(_data()))
        for allowed in ALLOWED_DISCLAIMERS:
            assert allowed in text

    def test_no_uuid_reaches_the_page(self):
        uid = "af46cfbb9c1d4e2fa0b3c5d7e9f10234"
        r = build_report(_data(balance=_balance(per_feature=[
            _feature(fid=uid, name="Swale 1", target_id="other",
                     is_terminal=False),
            _feature(fid="other", name="Basin 2", is_terminal=True)])))
        assert uid not in _text_of(r)

    def test_infiltration_buffer_shown_only_when_not_credited(self):
        credited = _text_of(build_report(_data(
            balance=_balance(counts_infiltration=True))))
        assert "Soaked away" in credited
        not_credited = _text_of(build_report(_data(
            balance=_balance(counts_infiltration=False))))
        assert "Could soak away" in not_credited
        assert "without relying on it" in not_credited

    def test_terminal_deficit_warns(self):
        r = build_report(_data(balance=_balance(terminal_deficit_m3=430.0)))
        warns = [c.text for c in _sections(r, Callout) if c.tone == "warn"]
        assert any("no downstream destination" in w for w in warns)

    def test_stale_storm_banner(self):
        r = build_report(_data(run_tag="120mm·24h", current_tag="80mm·6h"))
        warns = [c for c in _sections(r, Callout) if c.tone == "warn"]
        assert any("storm has changed" in c.title.lower() for c in warns)

    def test_stale_verification_is_flagged(self):
        r = build_report(_data(edits_since_verify=7,
                               verification=VerificationResult(per_feature=[])))
        text = _text_of(r)
        assert "out of date" in text
        assert "7 time(s)" in text


# ---------------------------------------------------------------- rules 1 & 2

class TestPresentationRules:
    def test_delta_is_only_ever_against_the_grid(self):
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                          "terrain_m3": 950.0, "delta_pct": -13.6}])))
        headers = [h for t in _sections(r, DataTable) for h in t.headers]
        assert "Δ vs grid" in headers
        assert "Δ vs design" not in headers
        assert sum(1 for h in headers if h.startswith("Δ")) == 1

    def test_volume_ladder_is_in_derivation_order(self):
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                          "terrain_m3": 950.0, "delta_pct": -13.6}])))
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert ladder.headers == ["Feature", "Geometric (drawn)",
                                  "At this grid (held)", "Measured", "Δ vs grid"]

    def test_the_ladder_does_not_print_a_freeboard_derived_storage(self):
        """Design storage was geometric less a blanket freeboard fraction. Freeboard
        on a real feature is set by its spillway, sized on page 6 from a peak flow
        this column knows nothing about — so it was a fourth storage figure from a
        rule of thumb, sitting in a run of columns meant to be compared."""
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                          "terrain_m3": 950.0, "delta_pct": -13.6}])))
        assert "Design storage" not in [h for t in _sections(r, DataTable)
                                        for h in t.headers]
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert all(len(row) == len(ladder.headers) for row in ladder.rows)

    def test_the_blurb_splits_calculated_columns_from_measured_ones(self):
        """The division a reader needs, and the reason column three is the largest.

        The old copy told them the last two columns "should track the drawn trench
        closely". They do not, and should not: a bank keyed into its ends holds water
        above natural ground, so what a swale impounds is routinely half again its
        section. Presenting that as the expected case is what stops it reading as an
        error — and stops a designer enlarging a feature that already has the storage.
        """
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1100.0,
                          "terrain_m3": 950.0, "delta_pct": -13.6}])))
        blurb = " ".join(c.text for c in _sections(r, Callout)
                         if c.title == "Reading this table")
        assert "calculated from the dimensions you drew" in blurb
        assert "measured by flooding the terrain" in blurb
        assert "above natural ground" in blurb
        # Neither of the framings this replaced: the cell size does not cause the gap,
        # and the two grid columns do not track the drawn trench.
        assert "grid artefact" not in blurb
        assert "track the drawn trench" not in blurb

    def test_an_overstated_row_is_marked_and_explained_under_the_table(self):
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 33", "analytic_m3": 291.0,
                          "geometric_m3": 364.0, "rasterisable_m3": 453.0,
                          "terrain_m3": 453.0, "delta_pct": 0.0,
                          "section_overstated": True, "section_gap_pct": 24.5}])))
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert ladder.rows[0][-1].endswith("†")
        # Print has no hover, so the dagger must be resolved on the page itself.
        assert "† Swale 33" in ladder.note
        assert "read Geometric" in ladder.note
        assert "cannot hold" in ladder.note

    def test_a_clean_row_carries_no_marker_and_no_note(self):
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "geometric_m3": 1250.0, "rasterisable_m3": 1260.0,
                          "terrain_m3": 1255.0, "delta_pct": -0.4}])))
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert "†" not in ladder.rows[0][-1]
        assert "†" not in ladder.note

    def test_sub_cell_feature_claims_no_storage(self):
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 8", "analytic_m3": 80.0,
                          "geometric_m3": 100.0, "routing_only": True}])))
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert ladder.rows[0][2:] == ["n/a — sub-cell"] * 3
        assert "0" not in ladder.rows[0][3]

    def test_dam_collapses_its_design_figures(self):
        """A dam impounds against the hillside, so geometric and at-grid are one
        computation — printing them twice would imply two derivations."""
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Dam 4", "analytic_m3": 1838.0,
                          "terrain_m3": 1838.0, "delta_pct": 0.0,
                          "barrier_impounded": True}])))
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert "barrier-impounded" in ladder.rows[0][1]
        assert ladder.rows[0][2] == ""

    def test_standing_water_is_not_printed_at_all(self):
        """It was a table whose caption was a warning about itself, carrying the
        largest number in the section — so it was the one most likely to be quoted."""
        r = build_report(_data(verification=VerificationResult(
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "terrain_m3": 950.0, "existing_m3": 420.0,
                          "total_m3": 1370.0}])))
        assert not [t for t in _sections(r, DataTable) if "context only" in t.title]
        # And it has not simply moved into the ladder's column run.
        ladder = [t for t in _sections(r, DataTable)
                  if "Δ vs grid" in t.headers][0]
        assert "1,370" not in str(ladder.rows)

    def test_uncorrected_baseline_ponding_is_disclosed(self):
        """Left unsaid, every measured figure overstates what the design adds."""
        r = build_report(_data(verification=VerificationResult(
            baseline_uncorrected="The baseline ponding raster was unreadable.",
            per_feature=[{"name": "Swale 1", "analytic_m3": 1000.0,
                          "terrain_m3": 950.0}])))
        warn = [c for c in _sections(r, Callout)
                if "already there" in c.title]
        assert warn
        assert "unreadable" in warn[0].text
        assert "ponded here naturally" in warn[0].text

    def test_verification_caveats_are_quoted_verbatim(self):
        caveat = "Swale 1: the grid represents it +114% differently."
        r = build_report(_data(verification=VerificationResult(
            caveats=[caveat], per_feature=[])))
        assert caveat in _text_of(r)

    def test_two_exit_volumes_are_never_in_one_table(self):
        r = build_report(_data())
        for t in _sections(r, DataTable):
            joined = " ".join(t.headers).lower()
            assert not ("boundary" in joined and "not captured" in joined)
        box = [c for c in _sections(r, Callout)
               if "two different" in c.title.lower()]
        assert box and "never subtract" in box[0].text.lower()


# ------------------------------------------- the two capture figures (item 1)

def _comparison(**kw):
    from terrainflow_assessment.modules.reporting import (
        ComparisonResult,
        PostInterventionReport,
    )

    kwargs = {"captured_pct": 47.0,
              "baseline": _baseline(),
              "post": PostInterventionReport(exit_volume_m3=11500.0,
                                             total_infiltrated_m3=800.0)}
    kwargs.update(kw)
    return ComparisonResult(**kwargs)


class TestTwoCaptureFigures:
    def test_headline_stays_the_calculated_figure(self):
        """It is the one that always exists and the one every other page is
        built from; swapping it for the simulated figure when a simulation
        happens to have run would change what the headline means."""
        r = build_report(_data(comparison=_comparison()))
        assert _sections(r, Hero)[0].value == "78%"

    def test_the_simulated_figure_is_named_under_the_headline(self):
        r = build_report(_data(comparison=_comparison()))
        text = _text_of(r)
        assert "47%" in text
        assert "different measurement, not a correction" in text

    def test_no_footnote_without_a_simulation(self):
        assert "different measurement, not a correction" not in _text_of(
            build_report(_data()))

    def test_water_fate_table_carries_both(self):
        table = _table(build_report(_data(comparison=_comparison())),
                       "Where the storm's water goes")
        assert "Volume (simulated)" in table.headers
        # runoff 21,715 - exit 11,500 - soaked 800
        assert _cell(table, 0, "Volume (simulated)") == fmt_volume(9415.0)
        assert _cell(table, 2, "Volume (simulated)") == fmt_volume(11500.0)

    def test_the_two_share_columns_say_what_actually_differs(self):
        """Both divide site-wide runoff. The note used to claim the calculated
        column divided "the water that reaches your earthworks" — the one
        paragraph meant to prevent a misreading, instructing the reader wrongly."""
        table = _table(build_report(_data(comparison=_comparison())),
                       "Where the storm's water goes")
        assert "divide the runoff the whole block generates" in table.note
        assert "reaches your earthworks" not in table.note
        assert "timing" in table.note

    def test_no_simulated_columns_without_a_simulation(self):
        table = _table(build_report(_data()), "Where the storm's water goes")
        assert not [h for h in table.headers if "simulated" in h.lower()]
        assert table.headers[:3] == ["", "Volume", "Share of runoff"]

    def test_simulated_held_never_goes_negative(self):
        """A routed exit volume can exceed the depth-derived runoff total on a
        site that takes water from off-block."""
        from terrainflow_assessment.modules.reporting import (
            PostInterventionReport,
        )

        table = _table(build_report(_data(comparison=_comparison(
            post=PostInterventionReport(exit_volume_m3=99999.0)))),
            "Where the storm's water goes")
        assert _cell(table, 0, "Volume (simulated)") == fmt_volume(0.0)


class TestExitThreshold:
    def test_the_threshold_is_stated_as_a_number(self):
        """Adding the listed crossings up and comparing them with the site
        figure is the obvious thing to do, and it does not reconcile."""
        r = build_report(_data(inputs={"exit_flow_ls": 0.5}))
        table = _table(r, "Where water leaves the boundary")
        assert "0.5 L/s" in table.note
        assert "sheet flow" in table.note

    def test_it_says_why_the_rows_do_not_add_up(self):
        r = build_report(_data(inputs={"exit_flow_ls": 0.5}))
        text = _text_of(r)
        assert "more water leaves the block than the crossings" in text

    def test_no_number_is_invented_when_it_was_not_recorded(self):
        r = build_report(_data(inputs={}))
        table = _table(r, "Where water leaves the boundary")
        assert "L/s" not in table.note
        assert "display threshold" in table.note


class TestMapLegends:
    def test_the_design_map_keys_only_the_types_it_draws(self):
        """A key listing five earthwork types on a scheme with two swales is
        its own kind of wrong."""
        r = build_report(_data(balance=_balance(per_feature=[
            _feature(fid="a", name="Swale 1", ew_type="swale"),
            _feature(fid="b", name="Dam 2", ew_type="dam")])))
        legend = [m for m in _sections(r, MapRef) if m.key == "design"][0]
        labels = [e.label for e in legend.legend]
        assert "Swale" in labels and "Dam" in labels
        assert "Basin" not in labels and "Berm" not in labels

    def test_earthwork_swatches_are_the_registry_colours(self):
        from terrainflow_assessment.core.registry.earthwork_types import get_type

        r = build_report(_data())
        legend = [m for m in _sections(r, MapRef) if m.key == "design"][0]
        swale = [e for e in legend.legend if e.label == "Swale"][0]
        assert swale.colour == get_type("swale").style[1]

    def test_the_flow_map_keys_the_runoff_ramp(self):
        from terrainflow_assessment.core.registry.map_palette import (
            surface_runoff_ramp,
            visible_stops,
        )

        r = build_report(_data())
        legend = [m for m in _sections(r, MapRef) if m.key == "flow"][0]
        ramp = [e for e in legend.legend if e.kind == "ramp"][0]
        assert list(ramp.colours) == [
            c for c, _l in visible_stops(surface_runoff_ramp())]

    def test_a_map_that_cannot_be_drawn_has_no_key(self):
        """A key for a figure that is not there describes nothing."""
        r = build_report(_data(maps={"design": "No earthwork layers."}))
        legend = [m for m in _sections(r, MapRef) if m.key == "design"][0]
        assert legend.reason and not legend.legend


# ---------------------------------------------------------------- D1

class TestFlowGraph:
    def test_edges_come_from_per_feature(self):
        g = build_flow_graph(_balance(per_feature=[
            _feature(fid="a", name="Swale 1", target_id="b",
                     is_terminal=False, is_user_link=True),
            _feature(fid="b", name="Basin 2", is_terminal=True)]))
        assert g["edges"]["a"] == ("b", True)
        assert g["edges"]["b"] == (None, False)
        assert g["chains"] == ["Swale 1 → Basin 2 → off the block"]
        assert g["all_terminal"] is False

    def test_ranks_put_downstream_later(self):
        g = build_flow_graph(_balance(per_feature=[
            _feature(fid="a", name="A", target_id="b", is_terminal=False),
            _feature(fid="b", name="B", is_terminal=True)]))
        ranks = {n["name"]: n["rank"] for n in g["nodes"]}
        assert ranks["B"] > ranks["A"]

    def test_cycle_terminates(self):
        """resolve_targets warns about rings rather than rejecting them."""
        g = build_flow_graph(_balance(per_feature=[
            _feature(fid="a", name="A", target_id="b", is_terminal=False),
            _feature(fid="b", name="B", target_id="a", is_terminal=False)]))
        assert g["chains"]
        assert any("loop" in c for c in g["chains"])

    def test_single_feature(self):
        g = build_flow_graph(_balance(per_feature=[_feature()]))
        assert len(g["nodes"]) == 1
        assert g["all_terminal"] is True

    def test_empty_balance(self):
        g = build_flow_graph(_balance(per_feature=[]))
        assert g["nodes"] == [] and g["chains"] == []
        assert g["all_terminal"] is False

    def test_all_terminal_is_reported_as_a_finding(self):
        r = build_report(_data(balance=_balance(per_feature=[
            _feature(fid="a", name="A"), _feature(fid="b", name="B")])))
        assert "drains independently" in _text_of(r)

    def test_chart_has_a_text_fallback(self):
        r = build_report(_data())
        img = _sections(r, ImageRef)
        assert img and img[0].key == "network"
        assert isinstance(img[0].fallback, DataTable)
        assert img[0].fallback.rows, "the cascade must survive a missing chart"


# ---------------------------------------------------------------- wording

class TestWording:
    @pytest.mark.parametrize("raw,expected", [
        (1247.3, 1250), (9994.0, 9990), (21715.0, 21700), (0.0, 0), (4.0, 0),
    ])
    def test_round_volume(self, raw, expected):
        assert round_volume(raw) == expected

    def test_fmt_volume_none(self):
        assert fmt_volume(None) == "—"
        assert fmt_volume(21715.0) == "21,700 m³"

    def test_a_figure_that_is_not_a_number_prints_as_no_figure(self):
        """Every volume in the document comes through here, and ``int(round(nan))``
        raises — so one unmeasurable number aborted the whole export instead of
        leaving one cell blank."""
        assert round_volume(float("nan")) is None
        assert round_volume(float("inf")) is None
        assert fmt_volume(float("nan")) == "—"

    @pytest.mark.parametrize("hours,fragment", [
        (None, "does not empty"), (0.5, "within the hour"), (18.0, "18 hours"),
        (72.0, "3 days"), (400.0, "more than a week"),
    ])
    def test_drain_wording(self, hours, fragment):
        assert fragment in drain_wording(hours)

    @pytest.mark.parametrize("pct,over,expected", [
        (10.0, False, "room to spare"), (70.0, False, "well used"),
        (99.0, False, "full at this storm"), (40.0, True, "fills and spills"),
    ])
    def test_fill_wording(self, pct, over, expected):
        assert fill_wording(pct, over).startswith(expected)

    @pytest.mark.parametrize("pct,over,expected", [
        (10.0, False, "room to spare · 10%"),
        (70.4, False, "well used · 70%"),
        (99.0, False, "full at this storm · 99%"),
        (40.0, True, "fills and spills · 40%"),
        (None, False, "room to spare · 0%"),
    ])
    def test_fill_wording_carries_the_number(self, pct, over, expected):
        """The words bucket 62%, 80% and 94% together; the reader acts on the
        number, so it belongs in the same cell rather than being inferred."""
        assert fill_wording(pct, over) == expected

    def test_cut_fill_sentence_survives_an_unmeasurable_quantity(self):
        s = cut_fill_sentence(float("nan"), float("nan"))
        assert "could not be measured" in s

    def test_cut_fill_sentence_avoids_signed_numbers(self):
        s = cut_fill_sentence(7944.0, 3533.0)
        assert "+" not in s and "more soil comes out" in s
        assert "more soil is needed" in cut_fill_sentence(100.0, 900.0)
        assert "balance on site" in cut_fill_sentence(500.0, 500.0)

    def test_duplicate_names_are_disambiguated(self):
        """The default name repeats after a delete — the code says so itself."""
        rows = [{"name": "Swale 3"}, {"name": "Basin 1"}, {"name": "Swale 3"}]
        assert unique_names(rows) == ["Swale 3 (a)", "Basin 1", "Swale 3 (b)"]

    def test_unique_names_leaves_distinct_names_alone(self):
        rows = [{"name": "Swale 1"}, {"name": "Basin 2"}]
        assert unique_names(rows) == ["Swale 1", "Basin 2"]

    def test_type_keys_are_never_printed_raw(self):
        """A column of "swale"/"dam" reads as a database dump, not a document.

        No table prints the type any more — the feature name carries it — so
        this asserts the raw registry keys reach no cell of any table at all.
        """
        r = build_report(_data(balance=_balance(per_feature=[
            _feature(fid="a", name="A", ew_type="swale"),
            _feature(fid="b", name="B", ew_type="diversion")])))
        cells = [str(cell) for t in _sections(r, DataTable)
                 for row in t.rows for cell in row]
        assert "swale" not in cells and "diversion" not in cells

    def test_unknown_type_key_still_reads_as_words(self):
        from terrainflow_assessment.modules.reporting import type_label
        assert type_label("pond_site") == "Pond Site"
        assert type_label(None) == ""


# ------------------------------------------------- one feature list, not two

class TestFeatureTables:
    def test_there_is_only_one_list_of_features(self):
        """"Features" (name/type/capacity) restated a subset of "Water
        arriving at each feature" two pages later, so the same forty names were
        printed twice and neither table was the one to quote."""
        r = build_report(_data())
        titles = [t.title for t in _sections(r, DataTable)]
        assert "Features" not in titles
        assert titles.count("Water arriving at each feature") == 1

    def test_capacity_survived_the_merge(self):
        """Dropping the other table must not lose what it carried."""
        table = _table(build_report(_data()), "Water arriving at each feature")
        assert _cell(table, 0, "Capacity") == fmt_volume(1250.0)

    def test_the_design_page_points_at_the_surviving_table(self):
        text = _text_of(build_report(_data()))
        assert "How the water is shared out" in text

    def test_no_table_repeats_the_type_beside_the_name(self):
        """"Dam 1" already says it is a dam. The column spent width on a
        twelve-column table restating the first word of the name."""
        r = build_report(_data(earthworks=[_FakeEarthwork()]))
        for title in ("Water arriving at each feature", "Every feature — drawn against measured"):
            assert "Type" not in _table(r, title).headers

    def test_the_build_schedule_keeps_type_detail(self):
        """That column carries the one dimension that matters *for* the type —
        a dam's crest, a diversion's grade — which the name does not."""
        r = build_report(_data(earthworks=[_FakeEarthwork()]))
        assert "Type detail" in _table(r, "Every feature — drawn against measured").headers


class TestPageOneDetail:
    @pytest.mark.parametrize("pct,tone", [(90.0, "good"), (55.0, "warn"),
                                          (12.0, "bad")])
    def test_hero_tone_bands(self, pct, tone):
        r = build_report(_data(balance=_balance(capture_pct=pct)))
        assert _sections(r, Hero)[0].tone == tone

    def test_terminal_overflow_fails_the_overflow_check(self):
        r = build_report(_data(balance=_balance(
            terminal_deficit_m3=430.0,
            per_feature=[_feature(is_terminal=True, overflow_m3=430.0)])))
        checks = [t for t in _sections(r, DataTable)
                  if t.title == "Three things worth checking"][0]
        row = [r for r in checks.rows if "Overflow" in r[0]][0]
        assert row[1] == "no" and "spill off the block" in row[2]

    def test_undesigned_spillway_fails_the_weir_check(self):
        r = build_report(_data(spillway_rows=[_spill(state="unsited")]))
        checks = [t for t in _sections(r, DataTable)
                  if t.title == "Three things worth checking"][0]
        row = [r for r in checks.rows if "weir" in r[0]][0]
        assert row[1] == "no" and "not designed or placed" in row[2]

    def test_routing_warnings_surface_on_page_one_and_the_network(self):
        r = build_report(_data(balance=_balance(
            routing_warnings=["Swale 1 overflows into a ring."])))
        text = _text_of(r)
        assert text.count("Swale 1 overflows into a ring.") >= 2
        assert [c for c in _sections(r, Callout) if c.title == "Routing warnings"]

    def test_next_three_things_is_capped_at_three(self):
        r = build_report(_data(balance=_balance(
            routing_warnings=[f"warning {i}" for i in range(9)])))
        table = [t for t in _sections(r, DataTable)
                 if t.title == "The next three things"][0]
        assert len(table.rows) == 3

    def test_no_storm_line_without_a_baseline(self):
        r = build_report(_data(baseline=None, balance=None,
                               earthworks=[object()]))
        assert REASON_NO_BASELINE in _text_of(r)

    @pytest.mark.parametrize("basis,fragment", [
        ("coefficient", "runoff coefficient 0.50"),
        ("runoff", "SCS curve number 61"),
        ("rainfall", "all rain runs off"),
    ])
    def test_runoff_basis_is_named(self, basis, fragment):
        r = build_report(_data(inputs={"sizing_basis": basis,
                                       "runoff_coefficient": 0.5}))
        assert fragment in _text_of(r)

    def test_natural_ponding_is_context_not_capture(self):
        r = build_report(_data(natural_ponding_m3=320.0))
        text = _text_of(r)
        assert "already collects in hollows" in text
        assert "not part of what your design captures" in text

    def test_trivial_natural_ponding_is_not_mentioned(self):
        # Matched on the sentence, not on the bare word "hollows": the reading guide
        # also has to describe what a terrain model can see, and a negative assertion
        # over one common noun turns any other use of it into a failure somewhere else.
        assert "already collects in hollows" not in _text_of(build_report(_data(
            natural_ponding_m3=4.0)))

    def test_no_exit_points_means_no_exit_table(self):
        r = build_report(_data(baseline=_baseline(exit_points=[])))
        assert not [t for t in _sections(r, DataTable)
                    if "leaves the boundary" in t.title]

    def test_every_crossing_the_map_draws_is_listed(self):
        """The table was capped at five under a map that draws all of them, so the
        reader was left counting markers with no row to look them up in."""
        r = build_report(_data())          # the fixture carries eight crossings
        table = _table(r, "Where water leaves the boundary")
        assert len(table.rows) == 8
        # The rolled-up "and N smaller crossings, together about X" line is gone.
        # The threshold note stays: crossings below the display threshold are a
        # different set, and they are still not drawn and still not listed.
        assert "together about" not in (table.note or "")
        assert "display threshold" in (table.note or "")

    def test_user_drawn_links_are_marked(self):
        r = build_report(_data(balance=_balance(per_feature=[
            _feature(fid="a", name="Swale 1", target_id="b",
                     is_terminal=False, is_user_link=True),
            _feature(fid="b", name="Basin 2")])))
        table = _table(r, "Water arriving at each feature")
        assert _cell(table, 0, "Spills to") == "Basin 2 (your link)"


# ---------------------------------------------------------------- spillways

def _spill(**kw):
    row = {
        "name": "Dam 4", "ew_type": "dam", "state": "ok",
        "peak_flow_m3s": 0.393, "upstream_m3s": 0.109,
        "required_width_m": 1.65, "built_width_m": 1.20,
        "target_head_m": 0.30, "actual_head_m": 0.37, "freeboard_m": 0.33,
        "crest_elevation": 213.9, "rim_elevation": 214.6, "problems": [],
    }
    row.update(kw)
    return row


class TestSpillways:
    def test_table_translates_state_into_words(self):
        r = build_report(_data(spillway_rows=[
            _spill(state="no_datum"), _spill(name="Dam 5", state="fail")]))
        table = [t for t in _sections(r, DataTable) if t.title == "Spillways"][0]
        states = [row[-1] for row in table.rows]
        assert "No ground level yet" in states
        assert "Needs attention" in states
        assert "no_datum" not in _text_of(r)

    def test_volume_versus_rate_box_is_present(self):
        r = build_report(_data(spillway_rows=[_spill()]))
        box = [c for c in _sections(r, Callout)
               if c.title == "Two different questions"]
        assert box and "how fast it falls" in box[0].text

    def test_default_intensity_is_flagged(self):
        r = build_report(_data(
            spillway_rows=[_spill()],
            spillway_context={"intensity_is_default": True,
                              "intensity_mm_hr": 40.0}))
        warn = [c for c in _sections(r, Callout)
                if "placeholder" in c.title.lower()]
        assert warn and "40 mm/hr" in warn[0].text

    def test_real_intensity_is_not_flagged(self):
        r = build_report(_data(
            spillway_rows=[_spill()],
            spillway_context={"intensity_is_default": False,
                              "intensity_mm_hr": 87.4}))
        assert not [c for c in _sections(r, Callout)
                    if "placeholder" in c.title.lower()]

    def test_problems_are_quoted_verbatim(self):
        problem = "Spillway is 1.20 m wide but needs 1.65 m for the design flow."
        r = build_report(_data(spillway_rows=[
            _spill(state="fail", problems=[problem])]))
        assert problem in _text_of(r)

    def test_no_spillways_says_so(self):
        r = build_report(_data(spillway_rows=[]))
        assert "No overflow structures" in _text_of(r)

    def test_missing_numbers_print_a_dash_not_a_zero(self):
        r = build_report(_data(spillway_rows=[
            _spill(state="no_datum", peak_flow_m3s=None, freeboard_m=None,
                   required_width_m=None, built_width_m=None,
                   actual_head_m=None)]))
        table = [t for t in _sections(r, DataTable) if t.title == "Spillways"][0]
        assert table.rows[0][1:6] == ["—"] * 5

    def test_flow_is_shown_in_litres_per_second(self):
        r = build_report(_data(spillway_rows=[_spill(peak_flow_m3s=0.393)]))
        table = [t for t in _sections(r, DataTable) if t.title == "Spillways"][0]
        assert table.rows[0][1] == "393 L/s"


class TestSpillwayGauge:
    """The band between the sill volume and the brim, and what it means.

    "% full" divided by the sill volume, so a spillway pinned its feature at 100%
    exactly when it started doing its job. These three marks are what make that band
    readable — and the verdict has to agree with the freeboard check, because the two
    are the same condition said twice.
    """

    def _gauged(self, **kw):
        row = _spill(sill_storage_m3=1276.6, containment_storage_m3=2132.8,
                     surcharge_level_m=214.27, surcharge_storage_m3=1700.0,
                     spillway_insufficient=False)
        row.update(kw)
        return row

    def _table(self, r):
        return [t for t in _sections(r, DataTable)
                if t.title.startswith("How full each feature")]

    def test_the_three_marks_reach_the_page(self):
        r = build_report(_data(spillway_rows=[self._gauged()]))
        table = self._table(r)
        assert table, "the gauge table was not built"
        row = table[0].rows[0]
        # Through `fmt_volume`, which rounds like every other volume on the page.
        assert "1,280" in row[1]
        assert "214.27 m" in row[2]
        assert "below the top" in row[5]

    def test_reaching_the_top_is_named_as_the_spillway_not_coping(self):
        r = build_report(_data(spillway_rows=[
            self._gauged(spillway_insufficient=True)]))
        assert "not passing enough" in self._table(r)[0].rows[0][5]

    def test_nothing_measured_means_no_table_rather_than_a_row_of_dashes(self):
        r = build_report(_data(spillway_rows=[_spill()]))
        assert not self._table(r)


class TestSpillwayAsBurned:
    """Three elevations, and the disagreements are the content."""

    def _burned(self, **kw):
        row = _spill(crest_elevation=55.52, burned_sill_m=55.52,
                     actual_spill_level_m=55.52)
        row.update(kw)
        return row

    def _table(self, r):
        return [t for t in _sections(r, DataTable)
                if t.title == "Did the model take the sill?"]

    def test_absent_until_a_burn_has_run(self):
        r = build_report(_data(spillway_rows=[
            _spill(burned_sill_m=None, actual_spill_level_m=None)]))
        assert not self._table(r)

    def test_a_clean_cut_says_so(self):
        r = build_report(_data(spillway_rows=[self._burned()]))
        assert "pond lets go there" in self._table(r)[0].rows[0][4]

    def test_a_refused_notch_is_named(self):
        r = build_report(_data(spillway_rows=[self._burned(burned_sill_m=57.10)]))
        assert "refused" in self._table(r)[0].rows[0][4]

    def test_ground_already_below_the_sill_is_named(self):
        r = build_report(_data(spillway_rows=[self._burned(burned_sill_m=53.99)]))
        assert "already below the sill" in self._table(r)[0].rows[0][4]

    def test_a_notch_that_does_not_daylight_is_named(self):
        r = build_report(_data(spillway_rows=[
            self._burned(actual_spill_level_m=56.12)]))
        assert "does not daylight" in self._table(r)[0].rows[0][4]

    def test_a_lower_saddle_taking_the_control_is_named(self):
        r = build_report(_data(spillway_rows=[
            self._burned(actual_spill_level_m=55.00)]))
        assert "never comes into play" in self._table(r)[0].rows[0][4]

    def test_a_cut_with_no_measured_pond_still_reports_the_cut(self):
        r = build_report(_data(spillway_rows=[
            self._burned(actual_spill_level_m=None)]))
        assert self._table(r)[0].rows[0][4] == "Cut to the designed level."

    def test_a_sill_that_was_never_sited_says_so(self):
        r = build_report(_data(spillway_rows=[
            self._burned(burned_sill_m=None)]))
        assert "not sited on the feature" in self._table(r)[0].rows[0][4]


# ---------------------------------------------------------------- build sheet

class _FakeEarthwork:
    def __init__(self, **kw):
        self.id = kw.get("id", "a")
        self.name = kw.get("name", "Swale 1")
        self.type = kw.get("type", "swale")
        self.enabled = kw.get("enabled", True)
        self.length_m = kw.get("length_m", 148.2)
        self.top_width_m = kw.get("top_width_m", 2.4)
        self.bottom_width_m = kw.get("bottom_width_m", 1.2)
        self.depth = kw.get("depth", 0.45)
        self.side_slope = kw.get("side_slope", 1.5)
        self.soil_name = kw.get("soil_name", "Loam")
        self.capacity_m3 = kw.get("capacity_m3", 96.0)
        self.crest_elevation = kw.get("crest_elevation", None)
        self.key_into_banks = kw.get("key_into_banks", False)
        self.gradient_pct = kw.get("gradient_pct", 1.0)
        self.companion_berm = kw.get("companion_berm", False)


class _FakeStore:
    def __init__(self, sid, cut, fill):
        self.id, self.cut_vol_m3, self.fill_vol_m3 = sid, cut, fill


class TestBuildSchedule:
    def test_per_feature_cut_and_fill_come_from_the_stores(self):
        """Not recomputed: the diversion branch treats width as the bed width,
        and getting that wrong would make this page contradict page one."""
        r = build_report(_data(
            earthworks=[_FakeEarthwork(id="a")],
            balance_stores=[_FakeStore("a", 201.7, 0.0)]))
        table = [t for t in _sections(r, DataTable)
                 if t.title == "Every feature — drawn against measured"][0]
        cut = table.headers.index("Cut (geometric)")
        fill = table.headers.index("Fill (geometric)")
        assert [table.rows[0][cut], table.rows[0][fill]] == ["200 m³", "0 m³"]

    def test_measured_columns_sit_beside_the_drawn_ones(self):
        """The gap between the two is the point, and it only reads as a gap when
        they are adjacent — the measured figure is the one to price from."""
        r = build_report(_data(
            earthworks=[_FakeEarthwork(id="a", name="Swale 1")],
            balance_stores=[_FakeStore("a", 201.7, 0.0)],
            verification=VerificationResult(per_feature=[
                {"name": "Swale 1", "terrain_m3": 1930.0, "cut_m3": 340.0,
                 "excavation_m3": 410.0}])))
        table = _table(r, "Every feature — drawn against measured")
        h = table.headers
        assert h.index("Capacity (measured)") == h.index("Capacity (geometric)") + 1
        assert h.index("Cut (measured)") == h.index("Cut (geometric)") + 1
        assert table.rows[0][h.index("Capacity (measured)")] == "1,930 m³"
        assert table.rows[0][h.index("Cut (measured)")] == "410 m³"

    def test_cut_measured_is_the_earth_moved_not_the_trench_void(self):
        """What the column *means*, which nothing pinned until it meant the wrong thing.

        ``cut_m3`` is the trench filled to its own bare pour point — what fits in the
        hole. ``excavation_m3`` is ``original − burned`` — what comes out of the
        hillside. On falling ground the second is larger, and it is the second the
        paragraph above this table tells the reader to price the job from.

        This column printed ``cut_m3`` for several releases. Every test around it
        supplied one number and asserted that number came back, so the plumbing was
        covered end to end and the meaning was covered nowhere: the column disagreed
        with the site total one table below it and the suite stayed green.
        """
        r = build_report(_data(
            earthworks=[_FakeEarthwork(id="a", name="Swale 1")],
            balance_stores=[_FakeStore("a", 201.7, 0.0)],
            verification=VerificationResult(per_feature=[
                {"name": "Swale 1", "terrain_m3": 1930.0,
                 "cut_m3": 340.0, "excavation_m3": 410.0}])))
        table = _table(r, "Every feature — drawn against measured")
        cell = table.rows[0][table.headers.index("Cut (measured)")]
        assert cell == "410 m³", (
            f"the column shows {cell}; 340 m³ is the trench void, which is a different "
            f"question and belongs to the resolution penalty"
        )

    def test_an_unmeasured_cut_is_suppressed_rather_than_zeroed(self):
        """No burn, no claim about earthmoving — an em dash, not a nought.

        A zero in a run of volume columns reads as "this feature moves no earth", which
        is never true of something that has been dug.
        """
        r = build_report(_data(
            earthworks=[_FakeEarthwork(id="a", name="Swale 1")],
            balance_stores=[_FakeStore("a", 201.7, 0.0)],
            verification=VerificationResult(per_feature=[
                {"name": "Swale 1", "terrain_m3": 1930.0, "cut_m3": 340.0}])))
        table = _table(r, "Every feature — drawn against measured")
        assert table.rows[0][table.headers.index("Cut (measured)")] == "—"

    def test_no_measured_fill_column_is_invented(self):
        """Banks sit outside their own footprints and cuts overlap, so a per-feature
        split would report the splitting rule. It stays a site total."""
        table = _table(build_report(_data(earthworks=[_FakeEarthwork(id="a")])),
                       "Every feature — drawn against measured")
        assert "Fill (measured)" not in table.headers
        assert "Measured fill is a site total only" in table.note

    def test_an_unmeasured_feature_leaves_the_column_blank(self):
        table = _table(build_report(_data(earthworks=[_FakeEarthwork(id="a")])),
                       "Every feature — drawn against measured")
        assert table.rows[0][table.headers.index("Capacity (measured)")] == "—"

    def test_a_duplicate_name_is_not_given_another_features_measurement(self):
        """Verification is name-keyed; two features sharing a name would take each
        other's measured cut, and a wrong number under "measured" is worse than none."""
        r = build_report(_data(
            earthworks=[_FakeEarthwork(id="a", name="Swale 1"),
                        _FakeEarthwork(id="b", name="Swale 1")],
            verification=VerificationResult(per_feature=[
                {"name": "Swale 1", "terrain_m3": 1930.0, "cut_m3": 340.0},
                {"name": "Swale 1", "terrain_m3": 45.0, "cut_m3": 12.0}])))
        table = _table(r, "Every feature — drawn against measured")
        col = table.headers.index("Capacity (measured)")
        assert [row[col] for row in table.rows] == ["—", "—"]

    def test_the_contractor_instruction_is_gone(self):
        assert "Take this page to your contractor" not in _text_of(
            build_report(_data(earthworks=[_FakeEarthwork(id="a")])))

    def test_disabled_features_are_left_off_the_quote(self):
        r = build_report(_data(earthworks=[
            _FakeEarthwork(id="a", name="Swale 1"),
            _FakeEarthwork(id="b", name="Swale 2", enabled=False)]))
        table = [t for t in _sections(r, DataTable)
                 if t.title == "Every feature — drawn against measured"][0]
        assert [row[0] for row in table.rows] == ["Swale 1"]

    def test_cut_fill_balance_is_in_words(self):
        r = build_report(_data(earthworks=[_FakeEarthwork()]))
        table = [t for t in _sections(r, DataTable)
                 if t.title == "Every feature — drawn against measured"][0]
        assert "more soil comes out" in table.note
        assert "+" not in table.note

    @pytest.mark.parametrize("kw,expected", [
        ({"type": "dam", "crest_elevation": 213.9, "key_into_banks": True},
         "crest 213.90 m (keyed into banks)"),
        ({"type": "dam", "crest_elevation": 213.9}, "crest 213.90 m (as drawn)"),
        ({"type": "dam", "crest_elevation": None}, "as drawn"),
        ({"type": "diversion", "gradient_pct": 1.5}, "grade 1.5%"),
        ({"type": "swale", "companion_berm": True}, "with berm"),
        ({"type": "basin"}, ""),
    ])
    def test_type_detail_names_the_dimension_that_matters(self, kw, expected):
        r = build_report(_data(earthworks=[_FakeEarthwork(**kw)]))
        table = _table(r, "Every feature — drawn against measured")
        assert _cell(table, 0, "Type detail") == expected

    @pytest.mark.parametrize("slope,expected", [
        (1.5, "1 in 1.5"), (0, "vertical"), (None, "vertical"),
    ])
    def test_batter_is_stated_the_way_a_contractor_reads_it(self, slope, expected):
        r = build_report(_data(earthworks=[_FakeEarthwork(side_slope=slope)]))
        table = _table(r, "Every feature — drawn against measured")
        assert _cell(table, 0, "Batter") == expected

    def test_missing_soil_says_site_default(self):
        r = build_report(_data(earthworks=[_FakeEarthwork(soil_name="")]))
        table = _table(r, "Every feature — drawn against measured")
        assert _cell(table, 0, "Soil") == "site default"

    def test_schedule_needs_earthworks_not_just_a_balance(self):
        r = build_report(_data(earthworks=None))
        assert "Take this page to your contractor" not in _text_of(r)


# ---------------------------------------------------------------- structure

class TestStructure:
    def test_maps_state_why_they_are_missing(self):
        r = build_report(_data(maps={"design": "the layer was deleted"}))
        design = [m for m in _sections(r, MapRef) if m.key == "design"]
        assert design and design[0].reason == "the layer was deleted"

    def test_map_captions_carry_the_disclaimer(self):
        r = build_report(_data())
        for m in _sections(r, MapRef):
            assert "not a survey" in m.caption

    def test_footer_carries_provenance(self):
        r = build_report(_data())
        assert "Quail Island" in r.footer
        assert "120mm·24h·C0.50·0.5ha" in r.footer
        assert "TerrainFlow 0.2.0" in r.footer
        assert "not a survey" in r.footer

    def test_wide_tables_are_marked_for_landscape(self):
        """The layout table overflows its frame silently rather than wrapping."""
        r = build_report(_data())
        wide = [t for t in _sections(r, DataTable) if t.wide]
        assert wide, "the per-feature tables must ask for a landscape page"

    def test_appendix_prints_the_digest_prefix_verbatim(self):
        r = build_report(_data(dem={"fingerprint": "sha256-sampled:4f3a9c1"}))
        rows = [row for t in _sections(r, KeyValueTable) for row in t.rows]
        assert any("sha256-sampled:" in str(v) for _, v in rows)

    def test_standing_caveat_is_on_page_one(self):
        r = build_report(_data())
        paras = [p.text for p in _sections(r, Paragraph)]
        assert any("not an engineering certification" in p for p in paras)


class TestTheTwoKindsOfNumber:
    """Which figures are worked out and which are read off the ground.

    They fail in opposite directions — a calculated volume is exactly right about a
    shape that may not be buildable here, a measured one is exactly right about a
    terrain model that is not a survey — and readers were treating them as competing
    estimates of one number and reading the disagreement as an error. The document has
    to define the two and then say which each table holds.
    """

    def test_the_two_kinds_are_defined_up_front(self):
        text = _text_of(build_report(_data()))
        assert "How to read the numbers in this report" in text
        head = text[:text.index("Site today")] if "Site today" in text else text
        assert "Geometric calculated" in head and "Measured" in head

    def test_it_comes_before_the_detail_pages(self):
        from terrainflow_assessment.modules.report_model import Heading
        r = build_report(_data())
        titles = [s.text for s in _sections(r, Heading) if s.level == 1]
        assert titles.index("How to read the numbers in this report") == 1, titles

    def test_calculated_is_defined_as_independent_of_the_ground(self):
        text = _text_of(build_report(_data()))
        assert "never looks at the ground" in text
        assert "tape measure" in text

    def test_measured_is_defined_as_from_the_elevation_model(self):
        text = _text_of(build_report(_data()))
        assert "Read off the elevation model" in text

    def test_measured_is_not_allowed_to_read_as_surveyed(self):
        """The word has to be fenced or it promises instrument levels."""
        text = _text_of(build_report(_data()))
        assert "not surveyed on the ground" in text

    def test_the_hybrid_figure_is_named_rather_than_left_to_be_guessed(self):
        """Capture % is calculated storage against measured inflow — neither alone."""
        text = _text_of(build_report(_data()))
        assert "calculated storage against measured inflow" in text.lower()

    def test_the_cell_size_is_stated_so_the_limit_is_concrete(self):
        text = _text_of(build_report(_data(baseline=_baseline(cell_size_m=1.0))))
        assert "1.00 m across per cell" in text

    def test_every_figure_table_says_which_kind_it_holds(self):
        r = build_report(_data(
            spillway_rows=[_spill()],
            burn_quantities={"cut_m3": 20096.0, "fill_m3": 10230.0},
            earthworks=[_FakeEarthwork(id="a")],
            balance_stores=[_FakeStore("a", 201.7, 0.0)]))
        untagged = [
            t.title for t in _sections(r, DataTable)
            if t.title in ("Where the storm's water goes",
                           "Water arriving at each feature", "Spillways",
                           "Every feature — drawn against measured", "Where water leaves the boundary")
            and "calculated" not in (t.note or "").lower()
            and "Measured" not in (t.note or "")
        ]
        assert not untagged, f"tables carrying figures with no kind stated: {untagged}"


class TestOverviewMap:
    """A reader who has never seen the block cannot place a figure on page one
    until they know what shape the property is and where the features sit."""

    def _overview(self, report):
        found = [m for m in _sections(report, MapRef) if m.key == "overview"]
        return found[0] if found else None

    def test_it_is_on_page_one(self):
        r = build_report(_data())
        keys = [m.key for m in _sections(r, MapRef)]
        assert keys[0] == "overview", keys

    def test_a_missing_overview_states_its_reason(self):
        r = build_report(_data(maps={"overview": "No DEM is loaded."}))
        assert self._overview(r).reason == "No DEM is loaded."

    def _design(self, report):
        return [m for m in _sections(report, MapRef) if m.key == "design"][0]

    def test_its_key_does_not_name_things_it_does_not_draw(self):
        """Neither map carries spillway markers or overflow links any more, so
        a key listing them would describe a map the reader is not looking at.
        Both are still in the document — the spillways have a page of their own,
        the links are on the flow diagram."""
        r = build_report(_data(spillway_rows=[_spill()]))
        for legend in (self._overview(r).legend, self._design(r).legend):
            labels = [e.label for e in (legend or [])]
            assert "Site boundary" in labels
            assert "Spillway" not in labels
            assert "Overflow link" not in labels

    def test_it_names_the_water_it_draws_and_not_the_earthworks(self):
        """The summary map is a photograph of the block with its water on it.
        The earthwork types are the design map's key, one page on."""
        r = build_report(_data())
        labels = [e.label for e in (self._overview(r).legend or [])]
        assert any(label.startswith("Water held") for label in labels), labels
        assert "Watercourse" in labels
        assert not any(label.startswith("Swale") for label in labels), labels
        assert "Swale" in [e.label for e in (self._design(r).legend or [])]

    def test_the_catchment_outline_is_named_only_when_it_is_drawn(self):
        """The labelling can trace to nothing — no design, or a design whose
        catchments have not been computed — and the exporter is the only thing
        that knows which."""
        without = [e.label for e in (self._overview(
            build_report(_data())).legend or [])]
        assert not any(label.startswith("Catchment") for label in without)

        r = build_report(_data(catchment_outline=True))
        labels = [e.label for e in (self._overview(r).legend or [])]
        assert "Catchment — swale" in labels, labels

    def test_the_catchment_outline_is_keyed_in_the_feature_colour(self):
        """The map draws a catchment in the colour of the thing that catches
        it, so the key has to say so type by type."""
        r = build_report(_data(catchment_outline=True))
        entries = {e.label: e for e in (self._overview(r).legend or [])}
        swale = [e for e in (self._design(r).legend or [])
                 if e.label == "Swale"][0]
        assert entries["Catchment — swale"].colour == swale.colour
        assert entries["Catchment — swale"].kind == "line"


class TestWorkedCatchment:
    """Capture % conflates two faults with opposite remedies: features too small,
    and a block that mostly drains past them. These separate the two."""

    def _report(self, **kw):
        per = kw.pop("per_feature", [
            _feature(fid="a", name="Swale 1", direct_catchment_m2=150000.0,
                     direct_inflow_m3=7500.0),
            _feature(fid="b", name="Basin 2", direct_catchment_m2=50000.0,
                     direct_inflow_m3=2500.0),
        ])
        bal = _balance(per_feature=per, total_captured_m3=8000.0,
                       routed_exit_m3=2000.0, uncaptured_m3=1200.0, **kw)
        return build_report(_data(balance=bal))

    def _cards(self, report):
        grids = _sections(report, StatGrid)
        return {c[0]: c[1] for g in grids for c in g.cards}

    def test_the_worked_share_of_the_catchment_is_reported(self):
        # 20 ha of direct catchment against the fixture's 36.2 ha contributing area.
        cards = self._cards(self._report())
        assert cards["Catchment worked"] == "55%"

    def test_capture_within_the_worked_ground_is_reported(self):
        # 8,000 m3 captured of the 10,000 m3 falling on ground that drains in.
        cards = self._cards(self._report())
        assert cards["Capture within it"] == "80%"

    def test_the_exit_row_carries_no_worked_share(self):
        """Its denominator is runoff on worked ground, so the matching numerator
        is the routed overflow — not the site exit sitting in the same row. A
        share of one beside the volume of the other is two different 'water
        leaving' figures on one line, which is what Rule 2 forbids."""
        table = _table(self._report(), "Where the storm's water goes")
        col = table.headers.index("Share of worked catchment")
        assert table.rows[2][0] == "Leaves the block"
        assert table.rows[2][col] == "—"

    def test_the_shares_it_does_print_are_of_the_worked_runoff(self):
        table = _table(self._report(), "Where the storm's water goes")
        col = table.headers.index("Share of worked catchment")
        # 7,880 m3 held and 120 m3 soaked, of the 10,000 m3 falling on worked ground.
        assert [table.rows[0][col], table.rows[1][col]] == ["79%", "1%"]

    def test_the_note_says_why_the_exit_row_is_blank(self):
        table = _table(self._report(), "Where the storm's water goes")
        assert "never mixed in one figure" in table.note

    def test_a_design_working_the_whole_catchment_gets_no_second_column(self):
        """The column would restate the one beside it."""
        table = _table(self._report(per_feature=[
            _feature(fid="a", direct_catchment_m2=362000.0,
                     direct_inflow_m3=10000.0)]), "Where the storm's water goes")
        assert "Share of worked catchment" not in table.headers

    def test_no_catchment_data_means_no_claim(self):
        r = build_report(_data(baseline=_baseline(catchment_area_ha=0.0)))
        assert "Catchment worked" not in self._cards(r)

    def test_the_card_and_the_panel_readout_agree(self):
        """The report and the Design-panel readout share one formula but not one set
        of inputs: the card divides summed per-feature m2 by
        ``BaselineReport.catchment_area_ha``, while the panel divides raw cell counts
        by ``flow_domain_mask``. Sharing ``catchment_coverage`` removes the formula
        divergence; only this pins the inputs, which is the half that can still drift."""
        from terrainflow_assessment.modules.report_model import _managed_catchment
        from terrainflow_assessment.modules.water_balance import catchment_coverage

        # What the panel sees: cell counts over domain cells, at 1 m cells.
        counts = {"a": 150000, "b": 50000}
        cell_area_m2 = 1.0
        domain_cells = 362000
        panel = catchment_coverage(sum(counts.values()) * cell_area_m2,
                                   domain_cells * cell_area_m2)

        # What the report sees: the same ground, reached by the other route.
        card = _managed_catchment(_data(balance=_balance(per_feature=[
            _feature(fid="a", direct_catchment_m2=150000.0, direct_inflow_m3=7500.0),
            _feature(fid="b", direct_catchment_m2=50000.0, direct_inflow_m3=2500.0),
        ])))

        assert round(panel["managed_pct"]) == round(card["managed_pct"])


class TestAppendixNamesTheMethodItRan:
    """The settings table printed every field it was handed, alphabetically — so a
    site sized on a Lancaster runoff coefficient still listed 'Cn 61', a curve number
    nothing in that run consulted. The appendix exists to say which input to change."""

    def _settings(self, **inputs):
        base = {"sizing_basis": "coefficient", "runoff_coefficient": 0.5,
                "cn": 61, "ground_condition": "good", "rainfall_mm": 120.0}
        base.update(inputs)
        r = build_report(_data(inputs=base))
        found = [t for t in _sections(r, KeyValueTable)
                 if t.title == "Settings used"]
        return dict(found[0].rows) if found else {}

    def test_the_runoff_method_is_named(self):
        assert self._settings()["Runoff method"] == "Runoff coefficient (Lancaster)"
        assert self._settings(sizing_basis="runoff")["Runoff method"] == (
            "Surface runoff (SCS curve number)")

    def test_the_coefficient_basis_does_not_list_a_curve_number(self):
        rows = self._settings()
        assert "Cn" not in rows and "Ground condition" not in rows
        assert "Runoff coefficient" in rows

    def test_the_scs_basis_does_not_list_a_coefficient(self):
        rows = self._settings(sizing_basis="runoff")
        assert "Runoff coefficient" not in rows
        assert "Cn" in rows

    def test_an_unrecognised_basis_is_reported_rather_than_guessed(self):
        assert self._settings(sizing_basis="whatever")["Runoff method"] == "whatever"


class TestMethodologyIsExplained:
    def test_both_methods_get_their_own_explanation(self):
        text = _text_of(build_report(_data()))
        assert "How this model works" in text
        assert "The geometric calculated method" in text
        assert "The measured method" in text

    def test_it_says_what_a_disagreement_between_them_means(self):
        text = _text_of(build_report(_data()))
        assert "the disagreement is the finding" in text


class TestCaveatsNameFeaturesTheWayTheTablesDo:
    """A reader met 'Swale 3' in the caveats and 'Swale 3 (b)' in the table above."""

    def _report(self, caveat):
        return build_report(_data(
            earthworks=[_FakeEarthwork(id="a", name="Swale 3")],
            balance=_balance(per_feature=[_feature(fid="a", name="Swale 3"),
                                          _feature(fid="b", name="Basin 9")]),
            verification=VerificationResult(
                caveats=[caveat],
                per_feature=[{"name": "Swale 3", "analytic_m3": 1000.0,
                              "terrain_m3": 950.0}])))

    def _caveat_text(self, report):
        return [c for c in _sections(report, Callout)
                if c.title == "Attribution caveats"][0].text

    def test_a_leading_feature_name_is_renamed(self):
        r = self._report("Swale 3: a 1.00 m cell cannot hold its drawn section.")
        # One feature named "Swale 3" among the earthworks, so it is unambiguous
        # and the caveat gets whatever heading the tables use for it.
        assert self._caveat_text(r).startswith("Swale 3:")

    def test_prose_that_merely_contains_a_name_is_left_alone(self):
        text = "Terrain volume is attributed by connected depression."
        assert self._caveat_text(self._report(text)) == text

    def test_an_ambiguous_name_keeps_its_raw_form(self):
        """Two features share the name, so no mapping is true of either."""
        r = build_report(_data(
            earthworks=[_FakeEarthwork(id="a", name="Swale 3"),
                        _FakeEarthwork(id="b", name="Swale 3")],
            verification=VerificationResult(
                caveats=["Swale 3: a 1.00 m cell cannot hold its drawn section."],
                per_feature=[{"name": "Swale 3", "analytic_m3": 1000.0}])))
        assert self._caveat_text(r).startswith("Swale 3:")


class TestEarthmoving:
    """Both quantities, because they answer different questions.

    ``calculate_cut_volume`` is section x length — the earth that comes out if the
    ground is flat. The burn cuts a level invert and builds a level crest, so on real
    ground it moves more. On the design this was written against, 11,948 m3 drawn
    against 20,096 m3 measured: quoting the first to a contractor is 68% short.
    """

    def _report(self, **kw):
        kwargs = {"earthworks": [_FakeEarthwork(id="a")],
                  "balance_stores": [_FakeStore("a", 201.7, 0.0)],
                  "burn_quantities": {"cut_m3": 20096.0, "fill_m3": 10230.0}}
        kwargs.update(kw)
        return build_report(_data(**kwargs))

    def _table(self, report):
        found = [t for t in _sections(report, DataTable)
                 if t.title.startswith("Earthmoving")]
        return found[0] if found else None

    def test_both_figures_are_printed(self):
        t = self._table(self._report())
        assert t is not None
        flat = " ".join(str(c) for row in t.rows for c in row)
        # Through the document's own rounding, like every other volume in it.
        assert fmt_volume(20096.0) in flat and fmt_volume(10230.0) in flat
        assert fmt_volume(7944.0) in flat and fmt_volume(3533.0) in flat

    def test_the_difference_is_stated_rather_than_left_to_arithmetic(self):
        t = self._table(self._report())
        flat = " ".join(str(c) for row in t.rows for c in row)
        assert "the drawn figure" in flat

    def test_it_says_which_column_to_price_from(self):
        t = self._table(self._report())
        assert "Price the job on the measured column" in t.note

    def test_nothing_is_printed_before_a_burn(self):
        assert self._table(self._report(burn_quantities=None)) is None

    def test_an_unmeasurable_quantity_is_absent_rather_than_nan(self):
        """NaN is truthy, so it passed the "has a burn run" guard and reached the
        rounding, which raised — the export died rather than the cell going blank."""
        nan = float("nan")
        r = self._report(burn_quantities={"cut_m3": nan, "fill_m3": nan})
        assert self._table(r) is None
        assert "nan" not in _text_of(r).lower()

    def test_it_does_not_invent_a_per_feature_split(self):
        """Banks lie outside their own footprints and cuts overlap, so a split would
        report the splitting rule rather than the job."""
        t = self._table(self._report())
        assert "Site totals only" in t.note
        assert len(t.rows) == 2


class TestOneNameMapForTheWholeDocument:
    """A letter has to mean the same feature on every page.

    The report called `unique_names` three times on three different inputs — the
    balance rows twice, the earthwork list once — and not at all in the volume
    ladder. So "Swale 3 (a)" on the design page could be a different feature from
    "(a)" on the network page, and the ladder printed two identical rows.
    """

    class _Ew:
        enabled = True

        def __init__(self, fid, name):
            self.id = fid
            self.name = name
            self.length_m = 100.0
            self.top_width_m = 2.0
            self.bottom_width_m = 1.0
            self.depth = 0.5
            self.type = "swale"

    def _colliding(self):
        return _data(
            earthworks=[self._Ew("id-a", "Swale 3"), self._Ew("id-b", "Swale 3")],
            balance=_balance(per_feature=[
                _feature(id="id-b", name="Swale 3"),
                _feature(id="id-a", name="Swale 3"),
            ]),
        )

    def test_the_map_letters_each_id_once(self):
        from terrainflow_assessment.modules.report_model import _display_names

        names = _display_names(self._colliding())
        assert set(names) == {"id-a", "id-b"}
        assert sorted(names.values()) == ["Swale 3 (a)", "Swale 3 (b)"]

    def test_the_letter_survives_a_different_row_order(self):
        """The balance lists them b-then-a; the letters follow the earthworks."""
        from terrainflow_assessment.modules.report_model import _display_names

        names = _display_names(self._colliding())
        assert names["id-a"] == "Swale 3 (a)"
        assert names["id-b"] == "Swale 3 (b)"

    def test_every_page_uses_the_same_letters(self):
        from terrainflow_assessment.modules.report_model import build_report

        data = self._colliding()
        report = build_report(data)
        text = _text_of(report)
        assert "Swale 3 (a)" in text and "Swale 3 (b)" in text
        # And never the bare colliding name on its own line in a table cell.
        for section in report.sections:
            for row in getattr(section, "rows", []) or []:
                if isinstance(row, (list, tuple)) and row:
                    assert row[0] != "Swale 3", (
                        f"an undisambiguated row survived: {row}")

    def test_a_unique_name_is_left_alone(self):
        from terrainflow_assessment.modules.report_model import _display_names

        data = _data(earthworks=[self._Ew("id-a", "North Swale")],
                     balance=_balance(per_feature=[_feature(id="id-a",
                                                            name="North Swale")]))
        assert _display_names(data) == {"id-a": "North Swale"}


class TestLetteringPastTheAlphabet:
    def test_the_27th_duplicate_is_aa_not_a_brace(self):
        from terrainflow_assessment.modules.reporting import _suffix

        assert _suffix(0) == "a"
        assert _suffix(25) == "z"
        assert _suffix(26) == "aa"
        assert _suffix(27) == "ab"
        assert _suffix(51) == "az"
        assert _suffix(52) == "ba"

    def test_no_suffix_falls_outside_the_alphabet(self):
        from terrainflow_assessment.modules.reporting import _suffix

        for n in range(200):
            assert _suffix(n).isalpha(), (n, _suffix(n))


class TestEarthworkBalanceSection:
    """The balance and haul rows, which extend the existing earthmoving section.

    Built from existing section types on purpose — a new `Section` subclass would cost
    a handler in both renderers and the parity test, for a table.
    """

    @staticmethod
    def _report(**kwargs):
        # The build-schedule page needs a design; without one the whole page — and so
        # the earthmoving section it ends with — is correctly absent.
        kwargs.setdefault("earthworks", [_feature()])
        kwargs.setdefault("burn_quantities", {"cut_m3": 1000.0, "fill_m3": 900.0})
        return build_report(_data(**kwargs))

    @staticmethod
    def _balance_table(report):
        found = [t for t in _sections(report, DataTable)
                 if t.title == "Earthwork balance and haul"]
        return found[0] if found else None

    def test_the_section_appears_once_a_burn_has_run(self):
        table = self._balance_table(self._report())
        assert table is not None, "no balance table after a burn"
        labels = [row[0] for row in table.rows]
        assert "Cut (bank measure)" in labels
        assert "In-situ soil the fill consumes" in labels

    def test_a_balance_is_not_cut_minus_fill(self):
        """1000 cut against 900 placed fill is a DEFICIT once compaction is applied,
        even though the raw difference is +100."""
        table = self._balance_table(self._report())
        labels = [row[0] for row in table.rows]
        assert any("Deficit" in label for label in labels), labels

    def test_the_factors_are_named_on_the_page(self):
        """A figure derived from a factor the reader cannot see is a rumour."""
        table = self._balance_table(self._report())
        assert "bulking" in table.note and "compaction" in table.note
        assert "not a soil test" in table.note

    def test_haul_rows_appear_when_a_plan_exists(self):
        plan = {
            "matched_m3": 500.0, "mean_haul_m": 40.0, "haul_moment_m3m": 20_000.0,
            "free_haul_m3": 500.0, "overhaul_m3m": 0.0, "free_haul_m": 150.0,
            "method": "least-cost transportation (LP)",
        }
        table = self._balance_table(self._report(haul_plan=plan))
        labels = [row[0] for row in table.rows]
        assert "Mean haul distance" in labels
        assert "Haul moment" in labels

    def test_the_straight_line_caveat_is_stated(self):
        """Real haul follows a track, around a gully, up a grade. This is a lower
        bound and the page has to say so."""
        plan = {
            "matched_m3": 500.0, "mean_haul_m": 40.0, "haul_moment_m3m": 20_000.0,
            "free_haul_m3": 500.0, "overhaul_m3m": 0.0, "free_haul_m": 150.0,
            "method": "least-cost transportation (LP)",
        }
        table = self._balance_table(self._report(haul_plan=plan))
        assert "STRAIGHT LINE" in table.note
        assert "lower bound" in table.note
        assert "contract term" in table.note

    def test_without_a_haul_plan_the_balance_still_prints_and_says_why(self):
        table = self._balance_table(self._report(haul_plan=None))
        assert table is not None
        assert "No haul figure" in table.note
        labels = [row[0] for row in table.rows]
        assert "Mean haul distance" not in labels
