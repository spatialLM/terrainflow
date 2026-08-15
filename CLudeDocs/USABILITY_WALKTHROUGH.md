# TerrainFlow — end-to-end usability walkthrough

A guided run of a full design, **Load DEM → Baseline → Analysis → Design → Re-analyse →
Simulate → Report**, to find what's working and what isn't before polishing.

This doc is **pre-annotated with predicted friction** (⚠) pulled straight from the code, so
you can confirm/deny each one as you go instead of hunting blind. Record what actually
happens in the **Findings table** at the bottom.

---

## How to use

1. Deploy + reload the plugin (`deploy.ps1`, then disable/re-enable in Plugin Manager).
2. Work top-to-bottom through the stages. For each step:
   - do the **Action**,
   - compare against **Expected**,
   - check the **⚠ Watch-for** items (these are predicted snags — mark each ✓ confirmed
     working / ✗ confirmed broken / — not observed),
   - jot anything odd in the **Findings table**.
3. Symbols: **✓** works · **✗** broken/wrong · **⚠** predicted snag to verify · **?** unclear.

### Prerequisites
- **A projected DEM.** Use the **BX24** site (1 m, EPSG:2193, ~284 ha) or any projected
  DEM. A geographic (lat/long) CRS is rejected on load by design.
- **matplotlib in the QGIS Python.** Without it the report's charts silently become
  "Chart unavailable" placeholders (no warning). Quick check in the QGIS Python console:
  `import matplotlib` — if it errors, the report will have no hydrograph/fill charts.
- Confirm you're on the current build: after keypoint analysis the status reads
  "*N valley points*" (new), not "*N keypoints*" (old).

---

## Stage 0 — First impression

| | |
|---|---|
| **Action** | Open the TerrainFlow panel fresh. |
| **Expected** | You'd expect to land on **Baseline** (the start of the pipeline). |
| **⚠ Watch-for** | The panel **opens on the *Design* stage**, not Baseline — you start two stages into the pipeline on tools whose scoring is inert until Baseline runs. Note whether this is confusing as a first-run experience. |

---

## Stage 1 — Data input (load the site)

| | |
|---|---|
| **Action** | Baseline stage → Data Input. Pick the **DEM Layer** (BX24). Set **Site Boundary** (pick a polygon layer *or* click **✏ Draw** and draw one). Enter a **Site name**. Optionally set **Analysis area** and **Earthworks area** (dropdown or ✏ Draw). |
| **Expected** | DEM info line shows `Cell / Area / EPSG`. Drawn polygons become a layer auto-selected in their picker. Site name will appear on the report. |
| **⚠ Watch-for** | (a) **✏ Draw** on each of the three area pickers — does it draw, add a layer, and auto-select it? (b) Does the site name actually reach the report? (the export filename ignores it regardless). (c) Does a drawn polygon land in the DEM's CRS? |

---

## Stage 2 — Baseline analysis (the keystone run)

| | |
|---|---|
| **Action** | Baseline stage → set storm (rainfall mm, duration hr, soil/CN, moisture, routing), then **Run Baseline Analysis**. |
| **Expected** | Progress bar → completes with a summary (runoff mm, CN, exit points). Stream/ponding/exit-point layers load. Baseline stepper turns **✓**. Terrain-tools buttons become enabled. |
| **⚠ Watch-for** | (a) **If the run fails, the Baseline stage still turns green ✓** and the summary reads "Analysis failed…" — a crash looks like success. (b) After completing, **change the rainfall or soil** and note that Baseline/Analysis **do not go "stale"** even though ponding is storm-dependent — later scores may quietly reflect the *old* storm. (c) Confirm Terrain Tools were disabled *before* this run. |

**Baseline is the keystone** — it gates contour/keypoint analysis, live scoring, the
simulation's flow direction, and the report. Skipping it degrades everything downstream.

---

## Stage 3 — Analysis (contours · keypoints · keyline)

### 3a. Contours + swale segments
| | |
|---|---|
| **Action** | Analysis stage → Contours tab. **Analyse Contours** → **Select Top 5 Swales** → set catchment/depth/width → **Find Best Swale Segments**. |
| **Expected** | Ranked candidate-contour layer (gold #1 → grey), the rank legend matches, and swale segments appear labelled with inflow + required length. |
| **⚠ Watch-for** | (a) **Reworked sizing:** required swale length should now be *shorter/sane* (trapezoidal storage + infiltration), not runaway kilometres. (b) Segments flagged **⚠** where the contour is too short to hold the design inflow — does that show in the label + success message? (c) Does the analysis honour the **Usable area** clip if set? (d) If contour/segment analysis **errors, does the stage still turn green ✓?** |

### 3b. Keypoints + pond sites
| | |
|---|---|
| **Action** | Keypoints tab → set count → **Find Keypoints + Ridgelines** → **Recommend Pond Sites**. |
| **Expected** | Diamond keypoints + dashed ridgelines + pond-site markers. The result list is now **clickable — a row zooms the canvas to that feature**. Status reads "*N valley points…*". |
| **⚠ Watch-for** | (a) Does clicking a list row actually zoom/flash the map? (b) **Doing only keypoint work leaves the Analysis stepper grey** (only Contours marks it done) — confirm. (c) Are keypoints landing at sensible valley inflections? |

### 3c. Keyline design (the new feature — expected to need polish)
| | |
|---|---|
| **Action** | Keypoints tab → set guides/spacing/grade → **Generate Keylines**. Then try **✏ Draw Keyline** (freehand, watch the status-bar slope cue). Then **Convert Keyline → Swale**. |
| **Expected** | A solid keyline following the valley contour + dashed parallel guides either side + a star keypoint marker. Draw adds a line to "Drawn Keylines". Convert creates a swale in the Design stage, reshape-locked to the keyline. |
| **⚠ Watch-for** | This is flagged for polish — capture specifics: (a) does the keyline follow the contour or look wrong/ragged? (b) do the parallel guides drift sensibly (valley→ridge)? (c) does self-intersection happen on tight bends? (d) does Convert → Swale land a usable swale? (e) is the workflow (generate vs draw vs convert) clear? |

---

## Stage 4 — Design (draw the water-capture earthworks)

| | |
|---|---|
| **Action** | Design stage. Draw a **Swale** (try Segment / Contour / Free modes), a **Basin** (polygon), and a **Dam** (line across a valley). Set each one's properties (depth/width/crest/etc.) in the dialog. Watch the **Scorecard** and **Live Assessment** network update. |
| **Expected** | Each earthwork renders at real width; the scorecard shows capture % + stored/soaked/leaves; the network shows nodes high→low with fill bars. |
| **⚠ Watch-for** | (a) Swale **Contour / Full-contour modes hard-stop** if you haven't run contour analysis — is the message clear? (b) With earthworks drawn but **no baseline**, the scorecard says "run Baseline" and the network shows capacity-but-no-water — is the "do baseline next" cue clear? (c) **Edit/Delete/Toggle/Reshape do nothing silently if no network node is selected** — confirm. (d) Draw a feature **at the DEM edge** and note if it later reads 0 % fill forever (centroid-outside-DEM). (e) Does the shortcut-badge removal look clean (no leftover gaps)? |

---

## Stage 5 — Re-analyse with Earthworks (design → verify bridge)

| | |
|---|---|
| **Action** | Design stage → **Re-analyse with Earthworks**. Then toggle **Show: with earthworks** to compare before/after. |
| **Expected** | Burns the earthworks into the DEM, re-runs flow, loads "Earthworks — Burned DEM" + updated layers, marks **Verify ✓**, sets the verified chip green. |
| **⚠ Watch-for** | (a) The warning says **"Load a DEM and run baseline first"** but it will actually run with **no baseline** — and if you do, the verification Δ is silently computed against an **all-zero baseline** (meaningless). Confirm whether skipping baseline is (wrongly) allowed. (b) **Verify turns ✓ here — before you've run any simulation.** Note that this is misleading. (c) Does the before/after toggle do anything meaningful? (d) Do the burn honesty warnings (sub-cell/resolution) appear? |

---

## Stage 6 — Verify (time-step simulation)

| | |
|---|---|
| **Action** | Verify stage → Fill Simulation. Set mode (Uniform), rainfall/duration/timestep → **Run Simulation**. Then use the **playback slider / play** and the results table. |
| **Expected** | Progress → playback controls + results table appear. Frames animate incremental/cumulative flow; earthwork fill points scale green→red and flag overflow. |
| **⚠ Watch-for** | (a) **Run Simulation is never disabled** — click it *before* baseline and note you only get a transient warning, no gating. (b) If you only half-completed Re-analyse, does it **silently simulate the baseline DEM** instead of your earthworks (no notice)? (c) Does an edge-drawn earthwork read **0 % fill** all the way through? (d) If the sim errors, is it just "see Python console" with no detail? (e) Does the earthwork-fill layer + ponding animate correctly? |

---

## Stage 7 — Report (export)

| | |
|---|---|
| **Action** | Report stage → **Export Report** → choose a path (PDF or HTML in the file dialog). |
| **Expected** | A ~9-page A4 PDF, *Site Water Plan*: one-page summary (capture %, water-fate table, stat cards, three condition checks, next three things) · site & flow map with the two-exit-volumes note · design plan map + feature list · flow-network diagram · per-feature water table (landscape) · spillway review · build schedule (landscape) · terrain check when Re-analyse has run · inputs, provenance and limits. Opens automatically (Windows). |
| **⚠ Watch-for** | (a) Does the **site name** reach the cover? A blank field falls back to the QGIS project title, then the DEM name, and the message bar says so. (b) Do the **landscape tables fit** at your feature count — they overflow silently rather than wrapping. (c) Do the **maps draw**, or is a stated reason printed instead? (d) Does the **scale bar** read sensibly for your extent? (e) With matplotlib missing, does the flow network degrade to the text cascade? |
| **Note** | **Neither format needs a simulation** — both are built from the design tier and render the same document, so a figure in one is a figure in the other. When a simulation *has* run, both gain the same extra section (before/after table, hydrograph, fill timeline), labelled as the only part that models timing. |

---

## Pre-seeded issue register

Confirm/annotate these during the run; add any new ones you find. (File refs for the
fixing session.)

### Blockers / misleading
- [ ] Panel opens on **Design**, not Baseline — `panel.py:134`.
- [ ] **Verify marked "done" by Re-analyse, not by Run Simulation** — `panel.py:1128`.
- [ ] **Design & Report stepper stages never advance** — no `mark_stage` for them.
- [x] **Failed runs still show green ✓** → fixed. `set_baseline_failed` /
      `set_earthworks_failed` leave the button idle and the stage un-ticked: amber when an
      earlier run left usable output behind, quiet when nothing has ever succeeded. The
      baseline failure path no longer switches on the downstream results tools either —
      it used to enable ponding query, slope, throughflow and contours for a run that
      produced no data. Covered by four checks in `tests_qgis/checks_report.py`.
- [ ] **Re-analyse warning text vs actual precondition mismatch** → zeroed verification Δ if
      baseline skipped — `earthworks.py` `run_with_earthworks` / `_compute_verification`.

### Confusing / silent
- [ ] **Run Simulation never gated** — `simulation.py` (only a transient warning).
- [ ] **Silent fallback to baseline DEM** when burn/re-analyse half-done — `simulation.py:52-66`.
- [ ] **Baseline/Analysis never go "stale"** on storm/soil change.
- [ ] **Analysis "done" set by Contours only, not Keypoints** — `panel.py` `set_keypoint_complete`.
- [ ] **Edit/Delete/Toggle/Reshape silent no-ops** with no selection — `earthworks.py`.
- [ ] **Centroid-outside-DEM stores get 0 inflow silently** — `simulation.py:360`.
- [ ] **Broad `except Exception: pass`** hides simulation/ponding failures.

### Report gaps — closed by the PDF Site Water Plan
- [x] **No map/plan-view images** → three maps (design plan, flow, before/after ponding),
      each with a scale bar and a stated reason when its layers are unavailable.
- [x] **Charts vanish silently without matplotlib** → the flow network degrades to a text
      cascade; nothing is ever the sole carrier of a number.
- [x] **No baseline-only report** → a baseline is now the only precondition. Sections
      without data keep their heading and say which button produces them.
- [x] **Default filename ignores site name** → `<Site>_WaterPlan_<date>.pdf`, sanitised,
      with a fallback chain and a message-bar nudge when it lands on "Unnamed Site".
- [x] **Misleading baseline peak-timing (0.0 hr)** → every simulation-derived figure is cut
      from the PDF rather than degraded; no timing claim is made anywhere.
- [x] **Report stepper stage never advances** → `mark_stage("report", "done")` on export.
- [x] **Export button stayed lit after Open Design** and dead-ended on a warning → cleared
      with the derived results, along with the stale summary label.
- [x] **Methodology text hardcoded/generic** → the old HTML generator is gone; both formats
      print the real inputs, DEM provenance and limits appendix.
- [x] **PDF and HTML can disagree** → converged. One `Report`, two renderers, and
      `tests/test_report_renderer_parity.py` fails if a section type is handled by only one
      of them or a model figure fails to reach the HTML. HTML no longer needs a simulation.

_Verified non-issue: simulation CN via `SOIL_REFERENCE` is a real CN table
(`SCSRunoff.SOIL_REFERENCE`) — not a unit mix-up._

---

## Adding screenshots

Images make findings much clearer. Keep them in a **`CLudeDocs/walkthrough_img/`** folder
(create it, or let VS Code's paste-into-workspace create it) so paths stay relative.

- **In a table cell** — size it and caption with `<br>` so the column doesn't blow out:
  `<img src="walkthrough_img/stage7-report.png" width="240"><br>no site map`
- **Larger / side-by-side evidence** — drop a full-width shot with a caption in the
  **Evidence** section below rather than cramming the table.

### Evidence (full-size screenshots, captioned)
<!-- e.g.
**Stage 7 — report has no map**
![report top](walkthrough_img/stage7-report.png)
-->

## Findings table (fill in as you go)

| Stage | Step / ⚠ item | Result (✓/✗/⚠/?) | Note |
|---|---|---|---|
| 0 | Opens on Design | | |
| 1 | ✏ Draw area pickers | | |
| 1 | Site name → report | | |
| 2 | Baseline run + summary | | |
| 2 | Failed run shows ✓? | | |
| 2 | Storm change → stale? | | |
| 3a | Swale sizing sane? | | |
| 3a | Capped ⚠ shown? | | |
| 3b | Result list zoom-on-click | | |
| 3b | Keypoint-only leaves Analysis grey? | | |
| 3c | Keyline follows contour? | | |
| 3c | Guides drift sensibly? | | |
| 3c | Convert → Swale usable? | | |
| 4 | Contour swale modes gated clearly? | | |
| 4 | No-selection buttons silent? | | |
| 4 | Edge-drawn feature | | |
| 5 | Re-analyse without baseline allowed? | | |
| 5 | Verify ✓ before simulate? | | |
| 5 | Before/after toggle | | |
| 6 | Run Sim ungated? | | |
| 6 | Silent baseline fallback? | | |
| 6 | Playback + fill layer | | |
| 7 | Export gated to full cycle? | | |
| 7 | No map images gap? | | |
| 7 | Charts present (matplotlib)? | | |

### New issues found (not in the register)
1.
2.
3.

---

_Next: run this end-to-end, fill the table, then we triage the confirmed items into a fix
backlog (walkthrough first, then fix)._
