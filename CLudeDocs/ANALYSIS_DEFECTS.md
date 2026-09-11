# Analysis-tier defect register (2026-09-11)

Thirty-one findings against the analysis tier as rebuilt on 2026-09-10
(`1089c3a` … `d40a2c0`, plus `ccdbcd6`, `8bf319b`, `c288a6a`), measured on a real DEM.
Companion to `MATHS_AUDIT.md`, in its dialect and under its rules.

**§1 through §6 are documentation, written before anything was changed.** Where a fix is
obvious it is named in `root cause` so a later pass does not have to re-derive it.

**§7 records what has since been fixed, and is the authority on current state.** Three
findings are closed and one new one (`TIX-01`) was opened and closed in the same pass. The
§1 rows are left as written, which is `MATHS_AUDIT`'s convention and this register's: the
table stays a record of what was found. Read §7 before acting on any row above it.

**HEAD at write time:** `c288a6a`, branch `wip/core-qgis-refactor`, tree clean.
**Fixture:** `tests/fixtures/quail_island_catchment.tif` — 400 × 400 @ 1 m (16 ha), EPSG:2193,
0 nodata cells, z −0.10 … 84.78 m.
**Second surface:** `_harness.build_synthetic_dem()` default — 300 × 300 @ 2 m (36 ha).

---

## §0 Method, status, and how to read this document

### §0.1 Legend

Copied verbatim from `MATHS_AUDIT.md:61-69` so the two documents cannot drift apart:

> **Verdict tokens:** OK (correct) · WRONG (incorrect) · DISC (differs from
> source/intent — explained; deliberate divergence is still DISC) · UNVER
> (no locatable published source) · UNVER-B (verification blocked mid-audit) ·
> HAZ (correct today, silently breaks under a stated condition).
> **Direction:** U = can UNDER-size (dangerous: breached earthwork), O = OVER-size
> (wasted excavation), n = neutral, i-d = input-dependent.
> **Severity:** C critical · H high · M medium · L low · I info. Per the audit brief,
> anything that can under-size in a production sizing path ranks ≥ H.
> **Confidence:** H/M/L.

Two consequences of that scale are worth stating, because both were got wrong in drafting
and corrected in review:

- **No finding here is `H`.** The escalator is narrow — *anything that can under-size in a
  production sizing path ranks ≥ H*. The keyline path emits **geometry, not dimensions**, so
  it does not fire there. `MATHS_AUDIT` has no `C` rows and three `H` rows
  (`EWD-32`, `EWD-35`, `EWD-05`), all sizing constants; every confirmed-`WRONG`
  analysis defect in it sits at `M`. These do too.

  **Updated 2026-09-11.** True of §1 and §2 as written, and still the right reading of
  them. Step C then opened exactly one `H`: `FLA-26` (§8.2), where selecting "D8" in the
  Routing combo raises out of pysheds and kills the baseline run. It is `H` **not** through
  the escalator above — nothing is under-sized, because nothing is sized — but because a
  labelled control in the shipped UI takes the plugin's entry point down with an unhandled
  exception, after which no sizing question can be asked at all. Filed knowingly, and the
  paragraph stands for everything it was written about.
- **Bare `U` is reserved.** The legend spends it on *"can under-size (dangerous: breached
  earthwork)"*. The register's own practice for other under-reads is a **qualified** `U`
  (`U(ridge detect)`, `U(warning)`) or `i-d`. So `KPA-41` is `U(drift under-read)`,
  `KPA-42`/`KPA-46` are `i-d`, and `SWL-22` is **`O`** — omitting infiltration lowers the
  capacity handed to `capture_spacing`, which **tightens** the advised interval.

### §0.2 New prefixes

Prefixes are keyed to **files**, and the prefix is the module whose function or contract is
the subject, with the caller cited in `where`. The precedent is `SWL-17` (§1 #27 of
`MATHS_AUDIT`), which cites `swale_design.py:259` *and*
`qgis/controllers/earthworks.py:1755` and is filed under `SWL`. Honouring that rule is why
the inert `acc_path` is `KPA-48` rather than a controller finding, and why the omitted
`infiltration_mm_hr` is `SWL-22`.

| Prefix | Covers | Note |
|---|---|---|
| `CTL` | `qgis/controllers/contour.py`, `core/registry/map_palette.py`, `qgis/controllers/_symbols.py`, `qgis/controllers/terrain.py` | **New.** These four files have no prefix in `MATHS_AUDIT` at all |
| `IMP` | `modules/impoundment_sites.py` | **New**, and free |
| `MHL` | `modules/mass_haul.py` | **New**, and free |
| `RPT` | **extended** to `modules/report_model.py` as well as `modules/reporting.py` (§2.5) | Siblings; inventing a second prefix for one of them buys nothing |
| `TIX` | `modules/terrain_indices.py` | Reserved when this register was written; **`TIX-01` was opened and closed on 2026-09-11** (§7.3), and **`TIX-02` opened the same day at `I`** (§8.5). The aspect ramp is `CTL-02`, because the defect is the *pairing* at `terrain.py:43,190`, not anything inside `terrain_indices.py` — and `KPA-52` (§8.1) is filed under `KPA` for the same reason, the pairing being in `find_keypoints` |

The register already carries a second, non-file ID family (`NEW-W5-01/02`,
`NEW-W8-01..05`), which is the precedent for adding one.

**Highest number in use per prefix at write time** (`MATHS_AUDIT` §2, verified by grep):
`ADV 08 · ALG 14 · BRN 09 · CAT 27 · CTA 29 · DEM 05 · EWD 58 · FLA 25 · FLG 17 · FLL 17 ·
FTP 11 · IDF 14 · INC 24 · KPA 37 · PIO 01 · PKF 09 · PRM 19 · REG 06 · RPT 25 · SIM 29 ·
SWL 21 · TOC 23 · UNI 18 · WBL 12`. `MHL`, `IMP`, `CTL` and `TIX` appear nowhere. **No
`EWD` ID is claimed** — the `EWD-59` this campaign drafted was withdrawn (§3), so `EWD`
stays at 58.

### §0.3 The owner-attestation precedent

`KPA-31` was resolved on a **domain ruling by the repo owner**, which is a kind of grounds
`MATHS_AUDIT` had never used. §7's rule is *"sources actually fetched and read this audit"*,
and every §5 resolution to date rests on a §7 entry — `ADV-02` established *"no source
exists"* by fetching `[FAO-WT]`, `[FAO-WM]` and `[HEC15]` and recording what the named
chapters lack.

So the ruling was **not** allowed to stand as grounds on its own. Two Yeomans texts were
fetched and read on 2026-09-11 (`[YEO-WFEF]`, 368 pp.; `[YEO-MKIV]`, 14 pp.) and they
establish the same absence independently. The ruling is recorded under its own key,
`[OWNER-2026-09-11]`, explicitly labelled *an owner attestation, not a fetched source*, so
a later reader cannot mistake it for one. See `MATHS_AUDIT` §9.8 for the resolution and §7
for the rows.

**This is a new precedent and is labelled as one.** It says: an owner ruling may direct the
audit's attention and may corroborate — and, per the ratification below, may *carry* a row
when nothing is fetchable, provided it is filed under its own token and never as a source.

~~The owner still owes a citation for Doherty~~ — **DROPPED 2026-09-11 on the owner's
decision.** `[DOHERTY]` is not added and no row is opened for it. The reasoning is recorded
because "we decided not to cite something" is the kind of thing a later reader re-opens:
Darren Doherty's published material is Regrarians course handbooks and recorded workshops,
which are weak as citations — typically undated, revised per delivery, and not pinnable to
an edition. Nothing in the tree rested on it. Any constant that would have been attributed
to him is instead declared a **TerrainFlow convention**, which is what the 50 m drift window
(§9.6) and `MIN_SLOPE_EASE` already are — and, since §10, `MIN_REACH_CELLS`, the fewest
cells a reach may have on either side of a two-slope break.

#### §0.3.1 The rule an attestation is admitted under (ratified 2026-09-11)

The owner has ratified the precedent: **an owner attestation may carry a register row.**
It is admitted under conditions, because grounds that can close anything close nothing.

**An attestation may carry a row only when all four hold:**

1. **Nothing fetchable would settle it.** A published source that exists must be fetched
   and read — that rule is unchanged and is §7's whole purpose. An attestation is for the
   case where the question is about practice, local convention, or an absence that no text
   states. If a fetch was not attempted, the row is not eligible.
2. **It is filed under its own token**, `[OWNER-<date>]`, never as a `[SOURCE]`. §7 remains
   *"sources actually fetched and read"* and an attestation never appears in it as one.
3. **It is dated and quoted.** What was ruled, in the owner's terms, on what day — so a
   later reader can see the claim rather than a summary of it, and can tell when it was
   made.
4. **The row says it rests on an attestation**, in the row itself, not only in §7. A reader
   scanning §1 must be able to see which rows are attested without cross-referencing.

**What an attestation cannot do:**

- It cannot make a row `OK`. The strongest verdict it supports is `UNVER` — *"no locatable
  published source"* — which is the honest description of a claim resting on domain
  authority. `OK` means checked against something external.
- It cannot override a measurement. Where an attestation and a measurement disagree, the
  measurement stands and the disagreement is recorded. This has already happened once in
  spirit: the owner's ruling on Yeomans was corroborated by fetching the texts, and had they
  contradicted it, the texts would have won.
- It cannot be inherited. An attestation closes the row it is written for. A later row
  relying on the same domain fact needs its own, or a fetch.

**Why admit them at all.** The alternative is that a question nobody can settle from a text
stays `UNVER-B` — "verification blocked" — forever, which reads as an outstanding defect
when it is actually a settled matter of practice. `KPA-31` was exactly that shape. Letting
the owner close it, visibly and under a token a reader can discount, is more honest than
leaving a permanent open row that quietly means "we know the answer but may not write it
down".

### §0.4 Rules of entry

- **A finding contradicted by a passing test is `DISC` by default**, and needs an explicit
  argument in `decided by` to become `WRONG`. Running the grep that establishes this is an
  entry condition on writing any entry, as is the production-caller screen.
- `not exercised` is printed as an honest third state, distinct from pass and fail.
- Every number here was measured by a probe under `tests_qgis/probes/`, and every entry's
  `reproduce` names the probe and the evidence file. Numbers that Step G pins were measured
  **twice** — once by the probe, once by the check.

### §0.5 Environment for every measurement

```
$env:QT_QPA_PLATFORM = 'offscreen'
& 'F:\bin\python-qgis-ltr.bat' tests_qgis\probes\p_<name>.py
```

QGIS 3.44 LTR at the root of `F:`; pysheds 0.5; shapely 2.1.2 / GEOS 3.13.1.
Pure suite at write time: **2,889 passed, 2 xfailed, 5 xpassed in 39.75 s** — so
`CLAUDE.md`'s *"~2,590 tests, ~60 s"* is stale in both figures.

---

## §1 Findings summary (severity-ranked) — the triage surface

Confirmed findings only. Unconfirmed candidates are in §4; refuted claims are in §3.

| # | ID | Sev | Dir | Where | Finding | Verdict | Conf |
|---|----|-----|-----|-------|---------|---------|------|
| 1 | KPA-39 | M | n | keypoint_analysis.py:807-811 | Every `keypoint_on_path` `None` is reported as *"no break in the floor clearing 2% of grade change"*. Traced over 127 real links: `len(thalweg)<5` **78**, guard-window **48**, prominence **0** — the stated reason is false **126/126**. On the 2 m synthetic DEM it is false 2/28 | WRONG | H |
| 2 | KPA-40 | M | i-d | keypoint_analysis.py:699-722 | No keypoint is possible below `7·min(5·cell,10)` m of thalweg — measured **35.0 m (36 cells) on 1 m**, **70.0 m (36 cells) on 2 m**. Undocumented. Qualifies `KPA-28`'s published `OK` | WRONG | H |
| 3 | KPA-41 | M | U(drift under-read) | keypoint_analysis.py:911-924 | `drift_1_in_n` is net end-to-end fall. Measured: a ridge guide reporting **1:544.8, `over_limit` False** runs **1:6.3 over its steepest 20 m** — an **86.4×** understatement, so the docstring's promise that steeper-than-`max_grade_n` guides are flagged fails | WRONG | H |
| 4 | KPA-42 | M | i-d | keypoint_analysis.py:1076-1099 | `_sample_dem` returns the keypoint elevation for any off-grid sample and nothing clips a guide to the data extent. Measured **9/151, 19/146, 27/143** fabricated vertices on the three valley guides; all three report `drift_fall_m` of exactly **0.000**, which is the default, not terrain. **Supersedes `KPA-35`** | WRONG | H |
| 5 | KPA-38 | M | i-d | keypoint_analysis.py:789 | `find_keypoints` builds D8 pointers from the **raw** DEM while its stream mask comes from conditioned accumulation. Measured at 0.2 ha: **127 order-1 links raw vs 93 conditioned**; **525 sinks vs 65**. Withdraws the precondition warrant under `FLG-06` | WRONG | H |
| 6 | KPA-46 | M | i-d | contour.py:1398-1404 | The no-keypoint fallback applies **no prominence bar**, and its stated scope is *"a small or single-valley DEM"*. On the 36 ha / 28-link synthetic DEM it fires on **every** run. On the real fixture it does **not** fire — production finds 1 keypoint | WRONG | H |
| 7 | KPA-48 | M | i-d | contour.py:1390 vs keypoint_analysis.py:1117 | `acc_path` is passed alone but the gate needs **both** paths, so it is an inert parameter: measured **0.48 s recomputing** against **0.022 s** when both are supplied, and the recomputed field differs from the supplied one by up to **65,086 cells**. The supplied field is crest-split and the recompute is not | WRONG | H |
| 8 | KPA-50 | M | n | keypoint_analysis.py:591, :9 | Two stale docstrings still present the deleted `cross_grade` generator as **step 5 of Yeomans' method**, under a heading reading *"True Yeomans keyline design"* — the closest thing in the repo to an implied attribution, describing a parameter that now only raises `DeprecationWarning` | WRONG | H |
| 9 | SWL-22 | M | O | contour.py:176-180 vs :806 | `suggest_spacing` calls `capacity_per_metre` without `infiltration_mm_hr` while `find_swale_segments` passes a real soil rate. Measured shortfall up to **85.7 %** (Sand / 24 h, 0.6 × 2.0 m section). Lower capacity ⇒ **tighter** advised interval ⇒ over-sizes | WRONG | H |
| 10 | CTL-02 | M | n | core/registry/map_palette.py:280 + terrain.py:43,190 | `ASPECT_CLASSES` declares **compass degrees** and is handed to `apply_raster_ramp`, whose contract is *fraction of max*. With `symmetric=False`, `top ≈ 359`, so the nine stops land at −359, 0, 16155 … 113085: every real value sits in the first **2.2 %** of the first interval | WRONG | H |
| 11 | CTL-03 | M | n | checks_terrain.py:138-150; test_map_palette.py:25-26 | The check that should catch `CTL-02` asserts less than its own name: `check_every_terrain_index_renders` asserts only that a layer id exists. The unit `RAMPS` list excludes `ASPECT_CLASSES`, `CURVATURE`, `WETNESS_INDEX`, `EROSIVE_POWER` | WRONG | H |
| 12 | CTL-04 | M | n | contour.py:1440-1441, :1608-1616 | **The "only one keypoint shown" symptom.** `_on_keyline_ready` passes `keypoints[0]` to `_display_keylines`, which builds the marker layer with **no loop** and names it in the singular — N keylines, one star — while the panel prints the true count at `:1448`. `_state.keyline_keypoints` has no production reader | WRONG | H |
| 13 | CTL-05 | M | n | contour.py:1598-1605; earthworks.py:232-233 | `break` on the first keyline run sets `keyline_master_coords`/`_geom`, so **"Convert Keyline → Swale" can only ever convert valley 1** | WRONG | H |
| 14 | RPT-26 | M | i-d | report_model.py:1608-1609 | `_earthwork_balance_section` reads `earthwork_soil_name`/`soil_name` off `ReportData`, which declares **neither**, so every report resolves to Loam. ≤ **8.0 %** on `bank_needed_for_fill_m3`, **13.6 %** on "Loose to cart"; surplus/deficit can flip | WRONG | H |
| 15 | IMP-01 | M | O | impoundment_sites.py:119-123, :334, :161-183 | `transect_cells` dedupes cells, so a diagonal crest yields **0.707 cells per map-space step** while `wall_len = len(run)·step` and `embankment_volume` both count one step per cell. Measured on production candidates: **0.713–0.933** cells/step, crest understated **14.1 %** (139.0 m counted vs 161.8 m), `storage_ratio` inflated **1.16×** | WRONG | H |
| 16 | IMP-03 | M | n | impoundment_sites.py:327-348 | The refusal `reason` is one slot rebound inside the ascending trial-height loop, so only the **last failing height** survives — and wall length grows with height, so the common case reports *"the wall would run over 200 m"* while hiding that every buildable height failed as unenclosed. Sibling of `KPA-39` | WRONG | H |
| 17 | IMP-04 | M | n | impoundment_sites.py:69-70, :217+ | The "not enclosed within the pond window" refusal names two causes and omits the one that often applies: **the DEM ran out**. Also falsifies `:69-70`'s claim that *"the window is the same size whatever the DEM is"* | WRONG | H |
| 18 | FLG-18 | M | i-d | flow_graph.py:507-508; keypoint_analysis.py:791 | `stream_links` emits at `min_cells=3` while its only consumer needs ≥5 cells **and** 35 m. Measured at 0.2 ha on the production surface: **127 links emitted, 1 can clear the floor** — 126 are profiled and refused by construction. No integer `min_cells` can express a metric bar when steps vary 1.0–1.414 m | WRONG | H |
| 19 | KPA-43 | L | n | keypoint_analysis.py:804 | The `break` drops every remaining link from *both* returned lists. Measured at 0.2 ha, `max_valleys=1`: `1 + 2 = 3` against **127** links — **124 never examined and never reported**. At `max_valleys=8` the identity holds only because just one keypoint exists | WRONG | H |
| 20 | FLG-19 | L | n | flow_graph.py:101-104 | `d8_from_dem`'s docstring claims a conditioned DEM gives an acyclic graph where every interior cell reaches the boundary. Measured on the real conditioned surface: **65 sinks and 72 mask-leaving pointers** survive | WRONG | H |
| 21 | CTA-30 | L | n | contour_analysis.py:206-207, :787 | `extract_contours` leaks `tempfile.mkdtemp` on every call — no `finally`, `rmtree`, `TemporaryDirectory` or `atexit` anywhere in the file, and both early-return fallbacks leak too. **Measured 2026-09-11: 456 `tfa_contours_*` directories, 138.7 MB**; across all `tfa_*` prefixes, **5,689 directories and 2,365 MB** | WRONG | H |
| 22 | ADV-09 | L | n | advisories.py (spacing_advisory) | With `capacity_m3_per_m=0.0`, `capture_spacing` returns 0.0, the `v>0` filter discards it, and `governing` reports `"erosion"` — where a section holding nothing is precisely capture-governed. **Unreachable from the UI** behind two guards, so latent | WRONG | H |
| 23 | KPA-51 | L | n | CLudeDocs/STRETCH_GOALS.md:140; CLudeDocs/USABILITY_WALKTHROUGH.md:87 | Shipped docs still describe the one-keypoint era and match what the code draws: *"finds **the keypoint**"*, *"a **star** keypoint marker"* — both singular. Stage 3c's watch-for list never asks how many keypoints appeared | WRONG | H |
| 24 | CTL-06 | L | n | contour.py:1601-1603 | The keyline→swale conversion **discards the Z that `keypoint_analysis` went to trouble to sample**: `keyline_master_geom` is built with `QgsGeometry.fromPolylineXY`. Adjacent to `KPA-42`, which is about Z being *fabricated*; here it is measured and then thrown away | WRONG | L |
| 25 | KPA-44 | L | n | checks_contour.py:487; test_keyline_yeomans.py | The prominence bar's **accept** branch and the `skipped` list have no end-to-end QGIS coverage, and the check that looks like it covers them is **tautologically satisfied** because the fallback sets `keypoints = [one]`. Re-traced: on the synthetic DEM **26 of 28** refusals are the prominence test, not 28 | WRONG | H |
| 26 | KPA-45 | L | n | keypoint_analysis.py:997-1053 | `offset_parts`' fold guard is **vestigial on this stack** — under shapely 2.1.2 / GEOS 3.13.1 `offset_curve` already removes self-intersections and no fold was constructible. Its test asserts only a 20 m spread on whatever is kept; measured spread **0.02 m**. The 24-sample cap is real but gives 6.37 m spacing at 150 m, biting only from ~600 m | DISC | M |
| 27 | KPA-49 | L | n | keypoint_analysis.py:822; panel.py:1269-1271; project_io.py:124 | **1:500 is unattributed, not miscited.** Every occurrence is a bare literal; no 1:400 *grade* exists in the repo; the docstring that cites Yeomans deliberately excludes the threshold. The defect is **omission** — nothing says it is a TerrainFlow convention. Sharpened by the source pass: **1 in 500 *does* appear in Yeomans**, as a channel's rate of fall. `MIN_SLOPE_EASE = 0.02` is unattributed on the same footing | DISC | H |
| 28 | CTL-01 | L | n | contour.py:137 vs :168 | `suggest_spacing`'s docstring says it reads slope *"over the usable area"*; the call passes no mask though the parameter exists and is unit-tested. Measured on the fixture: site-wide median **8.06°** against **6.81°** over a middle-half mask, a **−15.5 %** change | DISC | H |
| 29 | IMP-02 | L | n | impoundment_sites.py:60-62 vs :335 | `DEFAULT_MAX_WALL_M`'s docstring says *"either side"* and `transect_cells` searches ±200 m — measured span **400 m** — but the refusal compares the **total** run against 200 m. Code and message agree; the docstring describes a limit twice as permissive | DISC | H |
| 30 | MHL-01 | L | n | mass_haul.py:124; earthworks.py:5050 | `haul_regions` computes `block` from `block_m` and passes it to `_regions_from`, which never reads it: **one distinct output across a 100,000× sweep**. And the only caller never passes `block_m` at all, so it is inert twice over | DISC | H |
| 31 | KPA-47 | I | n | keypoint_analysis.py:898,907 | `offset_m`'s sign comes from measured elevation, not the geometric offset. **0 duplicates in 168 runs.** `checks_contour.py:499-505` pins the classification this sign is derived from. **Closed — recorded so it is not re-raised** | DISC | H |

**Tally: 31 findings — 25 `WRONG` and 6 `DISC`, one of the six (`KPA-47`) closed on entry.**
No `C` and no `H`. Severity `M` × 18, `L` × 12, `I` × 1.

---

## §2 Per-file findings

Each entry carries: **tokens · where · decided by** (every `DISC`) **· what was run ·
observed vs expected · quoted contract · root cause · blast radius · holds while ·
blocked by / blocks · reproduce · status · supersedes**. Fields that do not apply are
omitted rather than filled with "n/a".

### 2.1 `modules/keypoint_analysis.py`

#### KPA-38 — the pointer graph is built on the raw DEM while the mask is conditioned

- **Tokens:** `WRONG / M / i-d / H`
- **Where:** `keypoint_analysis.py:789` (`d8_from_dem(self.dem, …)`), against `:783` where
  the stream mask is built from `acc_arr`, and `:1127-1151` where `_ensure_flow_data`
  builds a conditioned surface and discards it.
- **What was run:** `p_flow_graph.py`, stage `link_populations`. The full link pipeline —
  `d8_from_dem` → `strahler_order` → `stream_links` — run twice at each of three
  thresholds, once over `self.dem` (raw, which is what production does) and once over the
  conditioned surface, with the same `acc_arr` mask both times.
- **Observed.** At **0.2 ha, the production threshold**:

  | Surface | order-1 links | sinks on finite ground | stream cells that are sinks | links clearing the 35 m floor |
  |---|---|---|---|---|
  | raw (production) | **127** | **525** | **42** | 1 |
  | conditioned | **93** | **65** | **0** | 2 |

  Sensitivity at 0.5 ha: 77 raw / 52 conditioned. At 1.0 ha: 55 / 38. **Those two rows are
  sensitivity, not evidence of production behaviour** — `contour.py:1397` passes
  `max_valleys` alone, so `stream_threshold_cells` takes its `max(20, round(2000/cell_area))`
  default (`keypoint_analysis.py:781`), which is 0.2 ha on the 1 m fixture **and** on the
  2 m synthetic DEM.
- **Expected:** the surface the pointer graph is built on should be the surface the mask
  was derived from, and `flow_graph.py:101-104` says which one that has to be.
- **Quoted contract** (`flow_graph.py:101-104`): *"`dem` should be **hydrologically
  conditioned** (pits filled, depressions filled, flats resolved). On a conditioned DEM the
  pointer graph is acyclic and every interior cell reaches the boundary."*
- **Root cause:** `_ensure_flow_data` conditions a surface to get `acc`, returns only
  `(fdir, acc)`, and the conditioned array goes out of scope. `find_keypoints` then reaches
  for `self.dem`, which is the raw float32 read in `__init__`. The fix is to return or cache
  the conditioned surface — and it is genuinely *one dict key away* on the controller side,
  which `KPA-48` measures.
- **Blast radius.** **Not the count.** `:800` sorts by `acc_arr[link[-1]]` — a
  conditioned-accumulation sample taken at a **raw-graph terminus** — so a fragment ending
  at a mid-valley pit is ranked on under-stated accumulation and can out-rank a real valley.
  Measured on the fixture: the **rank-1 link on the production path is 4 cells / 4.24 m
  long** and yields no keypoint, while the rank-1 link on the conditioned surface is 55
  cells / 54 m and yields one. `keypoints[0]` becomes `keyline_master_geom`
  (`contour.py:1598-1605`), which *"Convert Keyline → Swale"* turns into a real earthwork,
  and the `valley` layer attribute **is** that rank. Binds at 2 keypoints. **Not reached:**
  `valley_cells` is written and never read; `catchment_ha` feeds `label`, which has no
  production reader.
- **Why `WRONG`/`M` and not `HAZ`/`L`.** `HAZ` means "correct today". The link set already
  differs today, 127 against 93. `M` is the band every confirmed-`WRONG` production-analysis
  defect occupies in `MATHS_AUDIT` (`KPA-21`, `KPA-14`, `DEM-05`, `SIM-26`, `CAT-27`); `H` is
  reserved for under-sizing in a sizing path; `L` belongs to `HAZ` rows, dead code, and
  `KPA-33`, which earned `L` for firing **only on a fallback** — this fires on 100 % of
  keyline runs. `FIELD_TEST_LOG.md:1519` is the reason "no change on the fixture" would have
  been a bad inference here: the `resolve_flats_safely` fix took real terrain from **516
  sinks to 0** while moving the fixture's nine pinned figures by 0.00 %.
- **Not the finding:** `acc` is dinf and `d8_from_dem` is D8 **deliberately**
  (`flow_graph.py:17-23` — ~90k cells trapped in cycles when dinf angles are rounded). The
  fix restores a shared **surface**, not a shared **scheme**. And `keypoint_on_path` must
  keep sampling **raw** ground, because a keypoint is a real elevation.
- **Holds while** `find_keypoints` reads `self.dem` for its pointer graph.
- **Blocks:** the `FLG-06` precondition warrant — `MATHS_AUDIT` publishes that row `OK`,
  and this is the only raw call site in the tree.
- **Reproduce:** `p_flow_graph.py` → `evidence/p_flow_graph.json`, stage `link_populations`.
- **Status:** open, documented, not fixed.

#### KPA-48 — `acc_path` is an inert parameter, and the keyline runs on an uncorrected field

- **Tokens:** `WRONG / M / i-d / H`
- **Where:** `contour.py:1390` (`YeomansKeylineAnalysis(dem_path, acc_path=acc_path)`)
  against `keypoint_analysis.py:1117` (`if self._fdir_path and self._acc_path:`).
- **What was run:** `p_controllers.py`, stage `kpa48_acc_path_is_inert`. A crest-split
  baseline accumulation was written to disk, handed in exactly as the controller hands it
  in, and the returned field compared against it — then the same with **both** paths
  supplied, which is the branch the gate was written for.
- **Observed:**

  | Call | wall time | reads the supplied raster? |
  |---|---|---|
  | `acc_path` only — what production does | **0.48 s** | **no** |
  | `fdir_path` **and** `acc_path` | **0.022 s** | yes |

  The recomputed field differs from the supplied one by up to **65,086 cells** (both peak at
  67,199, so the disagreement is in the body of the field, not its scale).
- **Expected:** a caller that has already computed accumulation and hands it over should not
  pay for it again, and should get the field it handed over.
- **Root cause:** the gate is an `and` over two paths, and the caller supplies one. The
  second consequence is the sharper one: **`fa.acc` is crest-split and this recompute is
  not**, so the keyline is the only analysis tool in the plugin running on a
  pond-uncorrected accumulation.
- **Blast radius:** every keyline press — a full pysheds condition-and-accumulate — plus a
  silent divergence between the stream network the user sees and the one the keyline used.
- **Fix note, now measured rather than asserted.** `FlowAnalysis.run(crest_split=False)`
  does return a `conditioned_dem` key, and the surface `_ensure_flow_data` rebuilds is
  **bit-identical** to it: max |Δ| = **0.000e+00** over all 160,000 cells, 0 cells differing
  at all. Dropping `_ensure_flow_data`'s float32 GeoTIFF round-trip changes nothing either
  (max |Δ| also 0.0 on this fixture, whose z_max is 84.78 m). So *"the input is one dict key
  away"* is **true as stated**, and handing the key over is numerically free.
- **Holds while** the gate requires both paths and the controller supplies one.
- **Blocks:** nothing. **Blocked by:** nothing.
- **Reproduce:** `p_controllers.py` → `evidence/p_controllers.json`, stages
  `kpa48_acc_path_is_inert`; the bit-identity is `p_flow_graph.json` →
  `conditioned_surfaces.replica_vs_run`.
- **Status:** open, documented, not fixed.

#### KPA-39 — every refusal is reported as the one guard that refused nothing

- **Tokens:** `WRONG / M / n / H`
- **Where:** `keypoint_analysis.py:807-811` (the `skipped.append(...)` message, built at `:808-810`) against the
  five `return None` sites in `keypoint_on_path`.
- **What was run:** `p_keypoints.py`, stage `guard_histogram_*`. `sys.settrace` attributes
  every `None` to the line it returned from — nothing short of a line tracer can separate
  five `return None` statements inside one function from outside it.
- **Observed.** Real fixture, 0.2 ha, 127 order-1 links:

  | Line | Guard | Count |
  |---|---|---|
  | `:674` | `thalweg is None or len(thalweg) < 5` | **78** |
  | `:688` | `total_len <= 0` | 0 |
  | `:697` | `len(arc) < 5` — too little finite ground | 0 |
  | `:722` | `n_samp - 2*guard < 3` — profile too short to have an interior | **48** |
  | `:734` | `slope_ease < MIN_SLOPE_EASE` — **the only one reported** | **0** |

  126 refused, 1 accepted. **The stated reason is false 126 times out of 126.** `:688` and
  `:697` are dead code on this fixture, which has no nodata.

  Default synthetic DEM (2 m, 28 order-1 links): `:722` **2**, `:734` **26**. All 28
  refused; the message is false 2/28. **This is the re-trace `KPA-44` needed** — the earlier
  claim that all 28 were refused on prominence was read off this very message, and is
  **26/28, not 28/28**.
- **Expected:** the register's own rule at `keypoint_analysis.py:766` — *"the house style is
  to say what was refused **and why**, not to return a shorter list."*
- **Root cause:** one hard-coded message for a function with five exits.
- **Blast radius:** the message is surfaced verbatim in the panel and the message bar
  (`contour.py:1459-1463`), so a user tuning `MIN_SLOPE_EASE` on the fixture would be
  tuning the one guard that never fires. It is pushed as `pushInfo`, and
  `_harness.py:755-769`'s `assert_no_errors` fails only on **criticals and blocking modal
  dialogs** — so the message appears on every run and nothing fails.
- **A test pins the false message.** `test_keyline_yeomans.py:439` is
  `assert "grade" in skipped[0]`. It pins the **unconditional** string, passes today, and
  **will fail the moment the reason becomes truthful**. Recorded so a fixer expects it.
- **Holds while** the message is built outside the function that decided.
- **Blocks:** `KPA-44`'s attribution, which had to be re-established here.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stages
  `guard_histogram_fixture` and `guard_histogram_synthetic`.
- **Status:** open, documented, not fixed.

#### KPA-40 — an undocumented 35 m floor on the shortest valley that can have a keypoint

- **Tokens:** `WRONG / M / i-d / H`
- **Where:** `keypoint_analysis.py:699-702` (resample), `:704-710` (savgol window),
  `:719-722` (the interior test).
- **What was run:** `p_keypoints.py`, stage `profile_floor_*`. Bisection against the **real
  function** on a synthetic straight profile carrying a genuine slope break, with
  `require_prominence=False` so only the length guards can refuse — the answer is the
  function's, not the algebra's.
- **Observed:** first pass at **36 cells** on both grids — **35.0 m on the 1 m fixture** and
  **70.0 m on the 2 m synthetic DEM**. The algebra agrees: `spacing = min(5·cell, 10)`,
  `n_samp = max(5, int(L/spacing))`, `win ≥ 5` so `guard ≥ 2`, and `n_samp - 2·guard ≥ 3`
  first holds at `n_samp = 7`, i.e. `L ≥ 7·min(5·cell, 10)`.
- **Expected:** a limit this sharp should be documented where the resample constant is
  chosen. Nothing in the module states it.
- **Root cause:** three constants chosen independently — the 5-cell/10 m resample
  (`KPA-27`), the 20 % savgol window (`KPA-28`) and the half-window guard — whose product is
  a metric floor nobody wrote down.
- **Relation to published rows.** *Related to* `KPA-27`, whose design-constant question
  survives, so its §5 listing must not be retired. **It qualifies `KPA-28`'s published
  `OK`**, which reads *"all satisfied for every `n_samp ≥ 5`"*: `n_samp` 5 and 6 satisfy
  savgol and are still refused at `:722`, so that `OK` is true of the filter and false of
  the function. `KPA-40` is the joint consequence of the two.
- **Blast radius:** with `FLG-18`, it is why 126 of 127 links on the fixture are profiled
  and refused by construction.
- **Holds while** the resample constant stands. **The 70 m arm is the `holds while` test**,
  and it passed.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stages
  `profile_floor_fixture` and `profile_floor_synthetic`.
- **Status:** open, documented, not fixed.

#### KPA-43 — `max_valleys` drops links from both returned lists

- **Tokens:** `WRONG / L / n / H`
- **Where:** `keypoint_analysis.py:804` (`if len(keypoints) >= max_valleys: break`).
- **What was run:** `p_keypoints.py`, stage `accounting_identity_*`; `len(keypoints) +
  len(skipped)` against `len(links)` at `max_valleys` 1 and 8.
- **Observed:**

  | DEM | `max_valleys` | keypoints | skipped | sum | links | unaccounted |
  |---|---|---|---|---|---|---|
  | fixture (1 m) | 1 | 1 | 2 | **3** | **127** | **124** |
  | fixture (1 m) | 8 | 1 | 126 | 127 | 127 | 0 |
  | synthetic (2 m) | 1 | 0 | 28 | 28 | 28 | 0 |
  | synthetic (2 m) | 8 | 0 | 28 | 28 | 28 | 0 |

  The identity holds at `max_valleys=8` on the fixture **only because just one keypoint
  exists**, so the `break` never fires. That is a coincidence of this fixture, not a
  property of the code.
- **Expected:** `|accepted| + |refused| == |input|` for a screening tool.
- **Root cause:** the loop stops before examining the remainder, and the remainder is
  reported nowhere.
- **Reported as a count, deliberately.** Fabricating reasons for links that were never
  examined would be `KPA-39` again.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stages
  `accounting_identity_fixture` / `accounting_identity_synthetic`.
- **Status:** open, documented, not fixed.

#### KPA-41 — net end-to-end fall is not the drift the docstring promises to flag

- **Tokens:** `WRONG / M / U(drift under-read) / H`
- **Where:** `keypoint_analysis.py:911-924` (`_run_record`), against the docstring promise
  at `:848-850`.
- **Decided by** — this is a `WRONG` that needed an argument, because the metric itself is
  honestly described. `_run_record:914-917` states plainly that it computes net fall, so by
  the legend's *"deliberate divergence is still DISC"* the **metric** is `DISC`. What is
  `WRONG` is the **contract** at `:848-850`: *"`max_grade_n` is a **threshold**: every run
  reports the drift it actually achieves, and **guides steeper than 1:`max_grade_n` are
  flagged**."* A ridge guide running 1:6.3 over 20 m is not flagged. With no published
  definition of drift there is nothing for the *metric* to be `WRONG` against; the promise
  is a different matter.
- **What was run:** `p_keypoints.py`, stage `drift_table`. For each of the 7 guides on the
  fixture's one keypoint, the steepest sustained grade over 10 / 20 / 50 m windows,
  measured on **real ground only** (windows containing a fabricated vertex are excluded —
  see `KPA-42`).
- **Observed,** worst case: `ridge_guide` at offset **+15.0 m** reports
  `drift_1_in_n = 544.8` and `over_limit = False`, and runs **1:6.3 over its steepest 20 m**
  — an **86.4×** understatement. The guide at +5.0 m reports 1:432.7 and *is* flagged
  `over_limit = True`, so the flag is not dead; it simply fires on the wrong guides.
- **Expected:** a number that a plough actually experiences.
- **Root cause:** drift varies along the line **by construction** — a parallel offset of a
  curved contour is not a contour — so near-zero *net* fall can conceal mid-run reversal.
  Averaging over the whole line is the wrong estimator for a maximum.
- **Fails on Yeomans' own terms.** `[YEO-WFEF]`: *"any cultivation which is done parallel to
  any contour line marked in on the land surface, **must inevitably drift off the true
  contour as the cultivation continues**"* — drift is the **device**, not an error term. So
  the right shape for a limit is **steepest sustained grade over a window** as the pass/fail,
  with **fraction-of-length-over-limit as a reporting figure only**, never a gate.
- **The window is a TerrainFlow convention and is labelled one.** No source supplies it —
  see `KPA-49` and `MATHS_AUDIT` §9.8 — so it was **sized** rather than chosen, on
  `_harness.build_synthetic_dem`'s correlated roughness at a fixed seed:

  | Window | spread of steepest grade across `roughness_m` 0.00 → 0.10 | as a multiple of `MIN_SLOPE_EASE` |
  |---|---|---|
  | 10 m | 0.0435 | **2.18×** |
  | 20 m | 0.0488 | **2.44×** |
  | 50 m | 0.0165 | **0.82×** |

  **50 m is the shortest window whose spread stays inside `MIN_SLOPE_EASE`** and is the
  recommendation. It is sized on **synthetic** roughness of 0.05–0.10 m, which is the
  transfer a reader has to judge; the real fixture's own figures are reported beside it in
  the drift table so they can. **The decision is the owner's and is still owed** — see §6.
- **Relation:** *related to* `KPA-31`; does **not** supersede it.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stages `drift_table` and
  `roughness_sweep`.
- **Status:** open, documented, not fixed.

#### KPA-42 — off-grid samples are given the keypoint's elevation, and nothing clips a guide

- **Tokens:** `WRONG / M / i-d / H` — **supersedes `KPA-35`**
- **Where:** `keypoint_analysis.py:1088-1094` (`_sample_dem` returns `default`),
  `:1076-1086` (`_sample_z` passes `base_elev` as that default).
- **What was run:** `p_keypoints.py`, stage `drift_table`. `_sample_dem`'s own bounds-and-NaN
  test was **reproduced**, not patched, so the count cannot drift from the function it
  describes; every vertex of every guide classified real or fabricated.
- **Observed:** 3 of 7 guides carry fabricated Z —

  | Guide | fabricated / total vertices | `drift_fall_m` as reported |
  |---|---|---|
  | `valley_guide` @ −5 m | **9 / 151** | 0.000 |
  | `valley_guide` @ −10 m | **19 / 146** | 0.000 |
  | `valley_guide` @ −15 m | **27 / 143** | 0.000 |

  All three report **exactly** `0.000` net fall and `drift_1_in_n = null`. That is not
  symmetry and not a closed ring: the traced contour runs west-edge → north-edge, so inward
  offsets leave the grid at **both** ends and both endpoints receive `base_elev`. **The
  0.000 is the default, not terrain.** The three ridge guides, whose Z is entirely real,
  report 1:432.7, 1:586.6 and 1:544.8.
- **Expected:** a sample off the raster is missing data, not an elevation.
- **Quoted contract** (`:853-856`): *"`geometry` (3D LineString, **Z sampled from the
  ground** — a plough guide sits *on* the ground…)"*.
- **Root cause:** a default-valued sampler plus no extent clip. The default is the
  keypoint's own elevation, which is the one value guaranteed to make a drift computation
  look clean.
- **Blast radius:** the Z profile, the `mean_elev` that drives valley/ridge classification
  (`:898`), `drift_1_in_n`, `over_limit`, and exported 3D geometry.
- **Superseded row:** `KPA-35` (`MATHS_AUDIT:640`, `HAZ / i-d / L`, *"z-sampling nearest;
  fallback `base_elev`"*) is literally `_sample_dem`'s description. This raises it to
  `WRONG / M` with measurement. *Related to* `KPA-26` and `NEW-W8-01` only — the NaN half of
  `KPA-26` was re-filed as `NEW-W8-01` and **fixed**.
- **Numbers withdrawn:** an earlier draft reported 3.76–8.60 m of drift and 1:3 grades.
  Those were artefacts of the fabrication itself. Real ground gives 2.11–2.87 m and roughly
  1:12–1:14 over 20 m.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stage `drift_table`,
  field `fabricated_z_vertices`.
- **Status:** open, documented, not fixed.

#### KPA-44 — the prominence bar's accept branch has no QGIS coverage, and the check that looks like it does is tautological

- **Tokens:** `WRONG / L / n / H`
- **Where:** `checks_contour.py:487`
  (`len(by_type["keyline"]) == len(h.state.keyline_keypoints)`).
- **What was run:** the re-trace under `KPA-39`, plus a grep over `tests_qgis/` for any
  check that mentions the prominence bar or reads `skipped`.
- **Observed:** on the harness DEM `find_keypoints` returns empty, so `contour.py:1398`
  takes the fallback **every time** and sets `keypoints = [one]`. The assertion then compares
  1 against 1 and is **satisfied by construction** — it would pass on a broken
  implementation. No `checks_*` module mentions the bar or reads `skipped`.
- **Attribution, corrected.** The earlier claim that *all 28 links are refused on
  prominence* was read off the skipped message that `KPA-39` shows is unconditional. Re-traced
  with `settrace`: **26 of 28** are the prominence test and **2** are the length guard at
  `:722`. The emptiness, and therefore the fallback, was never in doubt; the attribution was.
- **The unit layer does cover the bar both ways** — `test_keyline_yeomans.py:413` accept,
  `:426-439` refuse — so the gap is **QGIS-level only**, and that is the whole finding.
- **Relation:** *related to* `CTL-04`, whose fix should also assert the marker layer's
  `featureCount()`.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stage
  `guard_histogram_synthetic`.
- **Status:** open, documented, not fixed.

#### KPA-45 — the fold guard is vestigial on this stack, and its test cannot tell

- **Tokens:** `DISC / L / n / M`
- **Where:** `keypoint_analysis.py:997-1053` (`offset_parts`), and
  `test_a_fold_is_refused_rather_than_returned`.
- **Decided by:** a passing test covers this code, so `DISC` is the default and it stays.
  The guard is not wrong; it is unreachable on the installed stack, and the test that
  appears to exercise it does not.
- **Observed:** under shapely 2.1.2 / GEOS 3.13.1, `offset_curve` already removes
  self-intersections, and no fold was constructible. The test asserts only a 20 m spread on
  whatever is kept, never `parts == []`; measured spread **0.02 m** — it passes because
  there is no fold to refuse. The 24-sample cap is real but gives **6.37 m** spacing at
  150 m, not "tens of metres"; it only bites from ~600 m.
- **The finding is the guard and its test**, not the cap — the cap needs no probe.
- **Holds while** GEOS keeps removing self-intersections in `offset_curve`. A GEOS
  downgrade makes the guard live again, and the test still would not prove it.
- **Status:** open, documented, not fixed.

#### KPA-46 — the fallback's stated scope is a rare case; it is the ordinary path

- **Tokens:** `WRONG / M / i-d / H`
- **Where:** `contour.py:1398-1404`.
- **What was run:** `p_keypoints.py` on both surfaces, and `p_flow_graph.py` for the link
  populations.
- **Observed — and this is where the earlier draft was wrong.**
  - **Synthetic harness DEM (2 m, 36 ha, 28 order-1 links):** `find_keypoints` returns
    **empty at the production threshold**, so the fallback fires on **every run**, drawing
    its one un-prominenced keyline. Neither small nor single-valley.
  - **Real fixture (1 m, 16 ha):** production finds **1 keypoint**, so **the fallback does
    not fire**. An earlier draft claimed *"the real fixture also falls back at 2.0 and
    5.0 ha"* — those thresholds are configurations **production never runs**, because
    `contour.py:1397` passes `max_valleys` alone and the default is 0.2 ha on both grids.
    The claim is withdrawn.
  - The roughness sweep strengthens the synthetic arm considerably: at `roughness_m`
    0.00 / 0.05 / 0.10 / 0.25 the number of keypoints clearing the 2 % bar is
    **0 / 0 / 0 / 0**. The fallback is not marginal on that surface; nothing ever clears.
- **Expected** (`contour.py:1399-1401`): *"on a small or single-valley DEM there may be no
  order-1 link long enough to profile, and the old answer is still an answer."*
- **Root cause:** the fallback applies **no prominence bar** — it calls `find_keypoint()`,
  which is `argmax` over one stem. `MIN_SLOPE_EASE`'s own docstring (`:631-640`) says the
  un-prominenced `argmax` was *"survivable **while one keypoint was found on one stem**; run
  per primary valley it fabricates them at scale."* So the fallback hands the user the exact
  output the same commit declared invented, while `:1459-1463` simultaneously reports that
  28 valleys had no keypoint clearing 2 % of grade change.
- **Why `i-d` and not `U`:** an invented keypoint places a pattern on the wrong ground,
  which is `KPA-21`/`KPA-33` precedent, not an under-sized structure.
- **Blast radius:** on the synthetic DEM the fallback's single keyline is drawn at **row 4 —
  the top edge of the DEM**.
- **Reproduce:** `p_keypoints.py` → `evidence/p_keypoints.json`, stages
  `guard_histogram_synthetic` and `roughness_sweep`; `p_flow_graph.json` for the fixture's
  single keypoint.
- **Status:** open, documented, not fixed.

#### KPA-47 — `offset_m`'s sign is derived from measured elevation, and a green check pins it

- **Tokens:** `DISC / I / n / H` — **closed on entry**
- **Where:** `keypoint_analysis.py:898` (`above = mean_elev >= keyline_elev`), with its reasoning at `:893-897`.
- **Decided by:** `checks_contour.py:499-505`. **The citation in the campaign plan was
  wrong and is corrected here:** that block does **not** assert `offset_m`'s sign —
  `offset_m` does not occur in `checks_contour.py` at all. It asserts `line_type` against
  **measured elevation** (a `ridge_guide` must sit at or above the keyline elevation), which
  is the classification `:898` derives the sign from. The check is therefore still what
  decides this row; it decides it one step upstream.
- **Observed:** 0 duplicates in 168 runs.
- **Recorded so it is not re-raised.** Filing this as a bug would send a fixer to break a
  green check. The code's own comment at `:893-897` explains the choice: `offset_curve`'s
  sign means "left of the direction of travel", and the traced contour's winding is not
  normalised, so measuring is the only stable basis.
- **Status:** closed.

#### KPA-49 — 1:500 is unattributed, and the source pass makes labelling it urgent

- **Tokens:** `DISC / L / n / H`
- **Where:** `keypoint_analysis.py:822` (`max_grade_n=500`), `panel.py:1269`
  (`setValue(500)`) and `:1271` (the tooltip), `project_io.py:124`.
- **Decided by:** the code is internally consistent and no source is misquoted, so this is
  `DISC` — a divergence between what a reader will infer and what is actually claimed.
- **What was run:** a grep over every occurrence of 500 and 400 in a grade context, plus the
  source pass recorded in `MATHS_AUDIT` §9.8.
- **Observed:**
  - Every occurrence is a **bare literal**. No 1:400 *grade* exists anywhere in the repo —
    the single literal hit, `checks_symbology.py:289`, is a **map scale**. So **no transplant
    and no sign error was committed.**
  - The docstring that *does* cite Yeomans deliberately excludes the threshold: `:826`
    attributes only the keyline definition, and `:848-851` calls the drift *"the number
    nobody had ever measured"*.
  - `KEYLINE_GRADE`'s help text **argues the repo's own position** — Yeomans is invoked to
    explain why **no** grade is imposed.
- **Sharpened by the source pass, and this is new.** **1 in 500 *does* occur in Yeomans.**
  `[YEO-WFEF]`, on a channel in flat country: *"According to circumstances, **its rate of
  fall may be anything from 1 in 500 to 1 in 5,000**."* It is a **channel's rate of fall** —
  a conveyance minimum — not a drift tolerance, and therefore does not source the repo's
  constant. But it means a reader who goes looking **will find 1:500 in Yeomans** and may
  reasonably conclude the constant is attributed. It is not.
- **Root cause:** omission. Nothing states that 1:500 is a TerrainFlow convention with no
  published basis. `MIN_SLOPE_EASE = 0.02` is unattributed on exactly the same footing.
- **Fix:** one line in `KEYLINE_GRADE`, and one beside `MIN_SLOPE_EASE`.
- **Status:** open, documented, not fixed.

#### KPA-50 — two stale docstrings present the deleted generator as Yeomans' own method

- **Tokens:** `WRONG / M / n / H`
- **Where:** `keypoint_analysis.py:591` (inside the class docstring headed *"True Yeomans
  keyline design"*) and `:9`.
- **Observed,** quoted verbatim from `:591`, given as **step 5 of Yeomans' method**:
  *"Generate cultivation runs: contour-parallel lines with a deliberate `cross_grade`
  (default 1/500) so water is gently directed across the slope rather than flowing straight
  downhill."* That is the generator framing the 2026-09-10 rebuild repudiated, describing a
  parameter that now only raises `DeprecationWarning` (`:841-846`). `:9` repeats it.
- **Root cause:** staleness, not citation. But it sits **exactly where a reader would look
  for provenance**, under a Yeomans heading, which is why it is `M` and not `L`, and why it
  is the closest thing in the repo to an implied attribution — `KPA-49`'s omission plus this
  staleness is how a reader ends up believing 1:500 is Yeomans'.
- **Status:** open, documented, not fixed.

#### KPA-51 — the shipped docs still describe the one-keypoint era

- **Tokens:** `WRONG / L / n / H`
- **Where:** `CLudeDocs/STRETCH_GOALS.md:140`, `CLudeDocs/USABILITY_WALKTHROUGH.md:87`.
  **Both are under `CLudeDocs/`** — an earlier draft cited them as repo-root paths.
- **Observed:** *"finds **the keypoint**"* and *"a **star** keypoint marker"*, both
  singular, both confirmed stale at HEAD. They match what the code actually **draws**, which
  is `CTL-04`.
- **Root cause:** Stage 3c's watch-for list never asks how many keypoints appeared, which is
  why the walkthrough could not catch `CTL-04` either.
- **Status:** open, documented, not fixed.

### 2.2 `modules/flow_graph.py`

#### FLG-18 — the link filter is a cell count; its only consumer needs metres

- **Tokens:** `WRONG / M / i-d / H`
- **Where:** `flow_graph.py:507-508` (`stream_links(..., min_cells=3)`) against
  `keypoint_analysis.py:791-792`, where the only production caller never passes it.
- **What was run:** `p_flow_graph.py`, stage `link_populations`: every emitted link's arc
  length against the floor `KPA-40` measures.
- **Observed,** production surface at 0.2 ha:

  | | count |
  |---|---|
  | order-1 links emitted at `min_cells=3` | **127** |
  | of those, shorter than 5 cells (refused at `:674` before any profile) | **78** |
  | of those, able to clear the 35 m floor | **1** |
  | profiled and refused by construction | **126** |

  Link lengths: min 2.0 m, median **3.4 m**, mean 5.2 m, max 54.0 m. The median link is an
  order of magnitude below the floor.
- **Expected:** the filter and the consumer should express the same bar.
- **Root cause:** a **unit** mismatch, not a badly chosen number. No integer `min_cells` can
  express a metric bar when a step is 1.0 m axis-aligned and 1.414 m diagonal. The fix is a
  `min_length_m = 7 · min(5·cell_size, 10)` parameter, and for `find_keypoints` to pass it.
- **Quoted contract** (`flow_graph.py:516-517`): *"Links shorter than `min_cells` are
  dropped: a two-cell stub has no profile to take a second derivative of, and fitting one to
  it produces a keypoint out of noise."* The intent is exactly right; the unit is wrong, so
  the guard stops three-cell stubs and passes thirty-cell ones that are equally unprofilable.
- **Blast radius:** 126 wasted savgol fits per keyline run on this fixture, and — with
  `KPA-39` — 126 refusals reported under a reason that did not apply.
- **Corroborated from an unexpected direction.** The roughness sweep (`KPA-41`) shows the
  failure mode scaling: as `roughness_m` goes 0.00 → 0.25 the order-1 link count rises
  **28 → 263 → 327 → 356** while the links long enough to profile *fall* **26 → 3 → 1 → 0**.
  Noise shatters the network into stubs that this floor then refuses.
- **Reproduce:** `p_flow_graph.py` → `evidence/p_flow_graph.json`, fields
  `links_clearing_floor` / `links_below_floor`.
- **Status:** open, documented, not fixed.

#### FLG-19 — sinks and mask-leaving pointers survive real conditioning

- **Tokens:** `WRONG / L / n / H`
- **Where:** `flow_graph.py:101-104` (the docstring's precondition claim).
- **What was run:** `p_flow_graph.py`, stage `link_populations`, `pointers` block —
  `is_sink` and the pointer targets of every stream cell, over both surfaces.
- **Observed,** on the **conditioned** surface, which is the one the docstring is about:
  **65 sinks on finite ground** and **72 stream cells whose pointer leaves the stream mask**,
  out of 1,688 stream cells. Zero stream cells are themselves sinks, and zero point at
  non-finite ground. On the raw surface — production's actual input — it is **525 sinks** and
  **42 stream cells that are sinks**.
- **Quoted contract** (`:101-104`): *"On a conditioned DEM the pointer graph is acyclic and
  **every interior cell reaches the boundary**."*
- **Root cause:** the claim is true of acyclicity and false of reachability. `d8_from_dem` is
  acyclic **by construction** — `best` starts at 0.0 and the test is a strict `>`, so a
  neighbour must be strictly lower — but pysheds' conditioning leaves cells with no strictly
  lower neighbour at the resolved-flat epsilon, and those point at themselves.
- **Not the finding:** acyclicity. Measuring that would measure the construction, not the
  surface. The number that means something is `is_sink.sum()`.
- **Severity `L`:** the consumer (`label_direct_catchments`) reports these as `LABEL_SINK`
  rather than hiding them, so the docstring over-promises but nothing downstream is misled.
- **Reproduce:** `p_flow_graph.py` → `evidence/p_flow_graph.json`, field `pointers`.
- **Status:** open, documented, not fixed.

---

### 2.3 `qgis/controllers/contour.py`

#### CTL-01 — the spacing advice is read site-wide, not over the usable area

- **Tokens:** `DISC / L / n / H`
- **Where:** `contour.py:137` (docstring) against `:168` (`slope_statistics(slope)`), and
  `terrain_indices.py:279` (`slope_statistics(slope_deg, mask=None, …)`).
- **Decided by:** the parameter exists, is unit-tested, and the call simply does not use it.
  Nothing is computed incorrectly — the advice is a correct answer to a different question —
  so `DISC`.
- **Observed** on the fixture: site-wide median slope **8.06°**; over a middle-half mask
  standing in for a property boundary, **6.81°** — a **−15.5 %** change. The median, the
  printed quartiles and the advised interval are all affected.
- **Quoted contract** (`:137`): *"Reads the slope raster **over the usable area**, then asks
  both spacing rules…"*
- **Root cause:** an omitted argument, the same shape as `SWL-22` three lines below it.
- **Reproduce:** `p_controllers.py` → `evidence/p_controllers.json`, stage
  `ctl01_unmasked_slope`.
- **Status:** open, documented, not fixed.

#### CTL-04 — N keylines, one star: the count is reported and not drawn

- **Tokens:** `WRONG / M / n / H`
- **Where:** `contour.py:1440-1441` (`self._state.keyline_keypoints = keypoints`, then
  `self._display_keylines(runs, keypoints[0])`), `:1608` (`"Keyline Keypoint"`, singular),
  `:1613-1616` (one feature).
- **This is the "only one keypoint shown" symptom, and it is live independently of the
  fallback.** `run_keyline_analysis` loops every keypoint and tags `run["valley"]`, so the
  **line** layer is correct. The **marker** layer is built from a single dict with no loop.
  On a multi-keypoint DEM the map shows N keylines and exactly one star, while the panel
  prints `f"{len(keypoints)} primary valley(s)"` at `:1448`.
- **Blast radius, folded in from what was briefly filed as a separate `CTL-06`:**
  `_state.keyline_keypoints` (`:1440`) is written and read by **nothing in the plugin** — its
  only consumer is `checks_contour.py:487`. That is the same gap, not a second one: a fix
  that has `_display_keylines` read the plural list closes both. One defect, one ID.
- **Root cause:** `_display_keypoints` already has the loop this needs (`:1641-1647`).
- **Relation:** *related to* `KPA-44`, whose coverage gap should also assert the marker
  layer's `featureCount()`.
- **Status:** open, documented, not fixed.

#### CTL-05 — "Convert Keyline → Swale" can only ever convert valley 1

- **Tokens:** `WRONG / M / n / H`
- **Where:** `contour.py:1598-1605` (the `for run in runs: … break`), consumed at
  `earthworks.py:232-233`.
- **Observed:** the loop takes the first run whose `line_type == "keyline"` and breaks, so
  `keyline_master_coords` and `keyline_master_geom` always describe valley 1 whatever the
  user selected.
- **Root cause:** a single-valley data model behind a multi-valley producer — the same era
  as `CTL-04`.
- **Status:** open, documented, not fixed.

#### CTL-06 — the conversion discards the Z the analysis went to trouble to sample

- **Tokens:** `WRONG / L / n / H`
- **Where:** `contour.py:1601-1603` (`QgsGeometry.fromPolylineXY`).
- **Observed:** `get_cultivation_runs` returns a **3D** LineString whose Z is draped on the
  ground (`keypoint_analysis.py:853-856`), and the conversion rebuilds it from XY alone.
- **Adjacent to `KPA-42`,** and worth reading beside it: there, Z is *fabricated*; here it is
  carefully measured and then thrown away.
- **Note on the ID:** this number previously held the unread-state-field point now folded
  into `CTL-04`. It was reassigned rather than left as a gap.
- **Status:** open, documented, not fixed.

---

### 2.4 `core/registry/map_palette.py`, `qgis/controllers/_symbols.py`, `qgis/controllers/terrain.py`

#### CTL-02 — compass degrees handed to a ramp whose contract is fraction-of-max

- **Tokens:** `WRONG / M / n / H`
- **Where:** `core/registry/map_palette.py:280` (`ASPECT_CLASSES`), paired at
  `qgis/controllers/terrain.py:43` (`"aspect": ("Aspect", P.ASPECT_CLASSES, False)`) and
  applied at `:190` (`if not symmetric:`).
- **Observed:** `ASPECT_CLASSES` declares **compass degrees**. `apply_raster_ramp`'s contract
  is *fraction of max*. Because `aspect` is registered `symmetric=False`, `top = band_max ≈
  359`, so the nine stops land at **−359, 0, 16155 … 113085**. Every real value therefore
  sits inside the first **2.2 %** of the first interval, so the whole map renders within
  2.2 % of the north colour, and the `-1` flat sentinel lands at t ≈ 99.7 % — the far end of
  the ramp.
- **Root cause:** two correct components with incompatible units at the seam. And the seam
  cannot be fixed by declaring intent: `_symbols.py:105` sets
  `QgsColorRampShader.Interpolated` and has **no** `Discrete` or `Exact` path, so
  `map_palette`'s own "named classes, not a ramp" rule is currently unimplementable through
  it.
- **Filed `CTL`, not `TIX`:** the defect is the **pairing** at `terrain.py:43,190`, not
  anything inside `terrain_indices.py`.
- **Status:** open, documented, not fixed.

#### CTL-03 — the check that should catch CTL-02 asserts less than its own name

- **Tokens:** `WRONG / M / n / H`
- **Where:** `tests_qgis/checks_terrain.py:138-150`; `tests/test_map_palette.py:25-26`.
- **Observed:** `check_every_terrain_index_renders` has the docstring *"a ramp that resolves
  to one flat colour is indistinguishable from a broken layer"* and a body that asserts only
  that a layer id exists. It would pass on `CTL-02`. At the unit level, `RAMPS` is built from
  three ramps and **excludes** `ASPECT_CLASSES`, `CURVATURE`, `WETNESS_INDEX` and
  `EROSIVE_POWER`.
- **Cheap fix, and it needs no image:** read
  `QgsColorRampShader.colorRampItemList()` for each `INDEX_SPECS` layer and assert the stops
  span `[band_min, band_max]`.
- **Status:** open, documented, not fixed.

---

### 2.5 `modules/impoundment_sites.py`

#### IMP-01 — a diagonal crest is measured in cells and paid for in metres

- **Tokens:** `WRONG / M / O / H`
- **Where:** `impoundment_sites.py:119-123` (the dedupe), `:334` (`wall_len = len(run) *
  step`), `:161-183` (`embankment_volume`, one prismatic section per surviving cell).
- **What was run:** `p_impoundment.py`. The harness was driven through the **production
  path** — baseline, then the find-keypoints signal, then the recommend-ponds signal — and
  `_state.pond_sites` read back. For each site the bearing was recomputed with
  `horn_gradient` + `flow_bearing`, `transect_cells` re-run, and the counted crest compared
  with the centre-to-centre polyline length through the same cells.
- **Observed:** 5 heuristic keypoints seeded **4** candidates, of which **2** produced a wall.

  | | value |
  |---|---|
  | distinct cells per map-space step, across the 4 candidates | **0.713 – 0.933** (1.000 is axis-aligned, 0.707 is 45°) |
  | worst crest under-statement | **14.1 %** — 139.0 m counted against **161.8 m** on the ground |
  | crest bearing there | 66.1°, i.e. **23.9° off axis** |
  | `storage_ratio` inflation from the matching fill under-count | **1.16×** |
  | rank order changes when corrected | **no**, on this fixture |
  | corrected crest still inside `max_wall_m` | yes (161.8 m < 200 m) |

- **Expected:** crest length in metres, measured in map space.
- **Root cause:** `transect_cells` dedupes so that a cell is measured once, which is right for
  *sampling* and wrong for *length*. Both consumers then treat the surviving list as evenly
  spaced at `step`.
- **Blast radius:** `wall_len` gates the `max_wall_m` refusal and is printed in the site
  label; `fill_m3` is the denominator of the rank. The bias is systematically toward diagonal
  necks. The theoretical worst case is **29.3 %** (at exactly 45°); the worst *measured* on
  this fixture is 14.1 %, because no ranked candidate sat at 45° — the 0.713 cells/step
  candidate produced no wall at any trial height.
- **The rotation test cannot see this:** it compares north–south against east–west, and both
  are axis-aligned.
- **Reproduce:** `p_impoundment.py` → `evidence/p_impoundment.json`, stage
  `transect_undercount`.
- **Status:** open, documented, not fixed.

#### IMP-02 — "either side" describes a limit twice as permissive as the one enforced

- **Tokens:** `DISC / L / n / H`
- **Where:** `impoundment_sites.py:60-62` (*"How far **either side** of the candidate the
  wall may run before the site is refused"*, `DEFAULT_MAX_WALL_M = 200.0`) against `:334-335`,
  which compares the **total** run.
- **Decided by:** the code and the user-facing message agree with each other — the refusal
  says *"the wall would run over 200 m"*, and 200 m total is what is enforced. Only the
  docstring diverges, so `DISC`.
- **Observed:** `transect_cells` searches ±200 m and returns a crest line spanning a measured
  **400 m**, of which at most 200 m can ever be accepted.
- **Reproduce:** `p_controllers.py` → `evidence/p_controllers.json`, stage
  `imp02_wall_limit`.
- **Status:** open, documented, not fixed.

#### IMP-03 — the refusal reported is the one from the least informative trial height

- **Tokens:** `WRONG / M / n / H`
- **Where:** `impoundment_sites.py:327-348` — `reason` is a single slot rebound inside the
  ascending `for height in crest_heights_m` loop.
- **Observed:** only the **last failing height** survives. Wall length grows with height, so
  the tallest trial is the one most likely to trip the length test — and the common case
  therefore reports *"the wall would run over 200 m — not a neck"* while hiding that every
  buildable height failed as **unenclosed**. Worse, `if not run: continue` sets no reason at
  all, so a NaN cell falls through to the initial value and reports *"no enclosed pool at any
  trial wall height"*.
- **Root cause:** one slot for five answers. **Sibling of `KPA-39`** — the same defect shape
  in a different module, which is why both are in this register.
- **Blast radius:** surfaced as the map label (`contour.py:1742` renders `"label"`, and
  `_refused` writes the reason into it).
- **Status:** open, documented, not fixed.

#### IMP-04 — a site refused for running out of DEM is reported as a landform judgement

- **Tokens:** `WRONG / M / n / H`
- **Where:** `impoundment_sites.py:217+` (`impounded_volume`'s refusal text), and the claim
  at `:69-70`.
- **Observed:** the "not enclosed within the pond window" refusal names two causes —
  *"runs round the wall or further upstream than a pond of this height should"* — and omits
  the one that often applies: **the DEM ran out**. A candidate 150 m from the raster edge is
  told its landform is wrong.
- **And it falsifies `:69-70`.** The comment claims *"the window is the same size whatever
  the DEM is, so a sweep over twenty candidates has a predictable price."* Clipping means
  neither the size nor the semantics are DEM-independent.
- **Status:** open, documented, not fixed.

---

### 2.6 `modules/swale_design.py`

#### SWL-22 — the spacing advisor omits the infiltration the segment finder passes

- **Tokens:** `WRONG / M / O / H`
- **Where:** `contour.py:176-180` (`capacity_per_metre(...)` with `duration_hr` but no
  `infiltration_mm_hr`) against `contour.py:806`
  (`infiltration_mm_hr=get_infiltration_rate(self._panel.earthwork_soil_name)`), filed under
  `SWL` because the contract is `swale_design.capacity_per_metre`'s.
- **Decided by** — this is `WRONG`, not `DISC`, because the two callers of one extracted
  function disagree, and the extraction exists precisely to stop that. `swale_design.py:128-131`:
  *"Extracted so the spacing advisor can ask the **transposed** question… Two copies of this
  would be two answers to 'does the swale hold its storm', which is the divergence
  `core/sizing` exists to stop."* `duration_hr` **is** passed, so infiltration is the sole
  missing factor — not an oversight of the whole soakage model.
- **Observed,** 0.6 m × 2.0 m section, side slope 1.0:

  | Soil | infiltration (mm/hr) | 1 h | 6 h | 24 h |
  |---|---|---|---|---|
  | Sand | 15.0 | +3.6 % | +21.4 % | **+85.7 %** |
  | Loam | 4.0 | +1.0 % | +5.7 % | +22.9 % |
  | Clay | 1.5 | +0.4 % | +2.1 % | +8.6 % |

  These are how much the capacity **rises** when the omitted argument is supplied — the
  advisory works from 0.840 m³/m in every cell of that table, against 1.560 m³/m for
  Sand at 24 h.
- **Direction is `O`, and it is load-bearing.** Omitting infiltration *lowers* the capacity
  handed to `capture_spacing`, so the advised interval comes out **tighter** — more swales
  than needed. Had this been `U`, §0's escalator would force `≥ H` on a sizing advisory.
- **This is `SWL-17`'s exact shape**, one file over: a controller calling a
  `swale_design` function with an argument omitted. `SWL-17` is the precedent for filing it
  under `SWL` with the caller cited.
- **Reproduce:** `p_controllers.py` → `evidence/p_controllers.json`, stage
  `swl22_omitted_infiltration`.
- **Status:** open, documented, not fixed.

---

### 2.7 `modules/report_model.py`

#### RPT-26 — every report resolves the earthwork soil to Loam

- **Tokens:** `WRONG / M / i-d / H`
- **Where:** `report_model.py:1608-1609` —
  `soil = (getattr(data, "earthwork_soil_name", None) or getattr(data, "soil_name", None))`,
  where `ReportData` declares **neither** attribute.
- **Observed:** both `getattr` calls return `None`, so the lookup falls through to its Loam
  default on every report, whatever the user selected.
- **Magnitude for one user:** ≤ **8.0 %** on `bank_needed_for_fill_m3` and **13.6 %** on
  "Loose to cart". (The 11.8 % figure that appeared in drafting is the Clay-to-Sand
  *spread*, not the error any single user sees.) Surplus and deficit can flip sign: at
  `fill = 1000 m³`, `bank_needed` is 1053 on Sand against 1136 on Loam.
- **Root cause:** the value exists and is reachable — `data.inputs["earthwork_soil_name"]`
  already carries it. The fix is one line.
- **Why `getattr` hid it:** a defaulted `getattr` cannot fail, so a missing attribute reads
  as a legitimate "not supplied".
- **Status:** open, documented, not fixed.

---

### 2.8 `modules/contour_analysis.py`

#### CTA-30 — `extract_contours` leaks a temporary directory on every call

- **Tokens:** `WRONG / L / n / H`
- **Where:** `contour_analysis.py:206-207`
  (`output_path = os.path.join(tempfile.mkdtemp(prefix="tfa_contours_"), "contours.gpkg")`).
  `analyse_contours:787` is the sole production caller and never passes `output_path`.
- **Observed:** no `finally`, no `rmtree`, no `TemporaryDirectory` and no `atexit` anywhere
  in the file — grepped, not assumed. Both early-return fallbacks leak too.
- **Measured on this machine, 2026-09-11:**

  | Prefix | directories | bytes |
  |---|---|---|
  | `tfa_contours_*` | **456** | **138.7 MB** |
  | `tfa_qgis_*` | 65 | — |
  | all `tfa_*` | **5,689** | **2,365 MB** |

  Every `tfa_contours_*` directory was created between **2026-09-10 16:52** and
  **2026-09-11 14:45**, i.e. inside this campaign's own test window.
- **The count is a test-run artefact and must be read as one.** The **finding is the code
  path** — a `mkdtemp` with no owner. The magnitude is whatever the machine's test history
  happens to be, and this session's own probe runs added to it. An earlier draft quoted
  "443 directories ≈ 311 MB"; the byte figure was wrong (it was ~137.8 MB at that moment) and
  the directory count moves with every run. Quoting a *stable* number here would be the
  mistake.
- **The 2.3 GB across all `tfa_*` prefixes is not all CTA-30.** The bulk is the per-run
  output directory the plugin creates for each session (`_state.output_dir`, `F:/Temp/tfa_*`),
  which is a **different** leak in test infrastructure. It is filed in §4, not here, because
  it is not `extract_contours`.
- **Status:** open, documented, not fixed.

---

### 2.9 `modules/mass_haul.py`

#### MHL-01 — `block_m` is inert twice over

- **Tokens:** `DISC / L / n / H`
- **Where:** `mass_haul.py:124` (`block = max(1, int(round(block_m / cell)))`), passed to
  `_regions_from`, which never reads it; and `earthworks.py:5050`, the only caller, which
  never passes `block_m` at all.
- **Decided by:** nothing computes a wrong answer — the regions are correct, they are simply
  not blocked. The divergence is between `DEFAULT_BLOCK_M`'s docstring and the code, so
  `DISC`.
- **Observed:** `block_m` swept over **0.01, 1.0, 10.0, 20.0, 100.0, 1000.0 m** — a
  **100,000×** range — produces **one distinct output**, by SHA-256 over the returned cut and
  fill lists.
- **Quoted contract:** `DEFAULT_BLOCK_M`'s docstring says the 10 m reduction is what keeps
  matching tractable. No reduction happens.
- **Inert in two independent ways**, which is why it survived: even a caller that did pass it
  would see nothing, and no caller passes it.
- **Reproduce:** `p_controllers.py` → `evidence/p_controllers.json`, stage
  `mhl01_block_m_is_inert`.
- **Status:** open, documented, not fixed.

---

### 2.10 `core/sizing/advisories.py`

#### ADV-09 — a section that holds nothing is reported as erosion-governed

- **Tokens:** `WRONG / L / n / H`
- **Where:** `advisories.py:260` (`spacing_advisory`), with `capacity_m3_per_m=0.0`.
- **Observed:** `capture_spacing` returns `0.0`, the `v > 0` filter discards it, and
  `governing` reports `"erosion"` — where a section holding nothing is precisely
  **capture**-governed.
- **Latent, and the reason matters.** It is **unreachable from the UI** behind two guards:
  `capacity or None` coerces `0.0` to `None`, and the spin boxes are floored at 0.1. No other
  caller exists. So this is a latent defect in a library function, not a live one.
- **`capacity or None` papers over rather than fixes:** the dict then reports
  `capture_spacing_m = None`, which is documented as *"no storm or section supplied"* — a
  different and equally untrue statement.
- **A test would have caught it, and asserts the opposite.**
  `tests/test_sizing.py:451-458` (`test_the_recommendation_never_exceeds_either_rule`)
  asserts `recommended <= both`, an invariant this input would **violate**. It is not
  exercised with a zero capacity.
- **Status:** open, documented, not fixed.

---

## §3 Refuted claims — the gate earning its place

Five claims were drafted, verified, and **withdrawn**. They are recorded because a refuted
claim that is merely deleted gets re-raised, and because one of them would have had a fixer
break working maps.

| # | Claim as drafted | What verification found |
|---|---|---|
| 1 | `_apply_index_ramp` discards `min_value`, so signed indices are mis-anchored | **The consequence is the opposite of the claim.** `CURVATURE`'s fractions are already **signed**, so `floor=0, span=bound` gives exactly the symmetric ±bound anchoring the docs describe. Honouring the floor would put stops at −3·bound … +bound and paint **zero curvature as "gathering"**. Anyone fixing this from the write-up would break working maps. Residue: the argument is dead and misleading. `OK (cleanliness)` |
| 2 | `event_yield_m3` applies rainfall where it should apply runoff | **REFUTED.** The caller passes a loss-adjusted runoff **depth**: `analysis_worker.py:408` emits `runoff_mm`, computed at `:108-131` by `coefficient_runoff_depth` (C×P) or full SCS-CN, and `impoundment_sites.py:373` applies it. `checks_fixture_regression.py` proves it — `rainfall_mm 120.0`, `runoff_coefficient 0.50`, `EXPECTED["runoff_mm"] == 60.0`. Adding a coefficient inside `impoundment_sites` would **double-count losses**. Residue: one docstring line stating the caller owes a runoff depth. `OK` |
| 3 | The mechanism behind `KPA-44` — `stream_links` returns ~1 link on the harness DEM | **REFUTED.** It returns **28**. D-infinity disperses flow across a ~2,944-cell band. The *coverage* gap `KPA-44` records is real; this explanation of it was not |
| 4 | Two `pool_reach_m` sub-claims: "the window is always the whole DEM" and "the refusal can never fire" | **REFUTED, both.** Full-axis only for indices in [149, 250] on a 400-cell axis, and the refusal does fire. The real finding in this area is `IMP-04` |
| 5 | float64 for the conditioned surface is a live fault on this fixture | **REFUTED, and the drafting arithmetic was wrong.** ε is already measured at **5.99e-06** (`FIELD_TEST_LOG.md:1581` — a document this campaign quotes elsewhere), so there was never a conflict to settle; this campaign re-measured it independently at **5.994524e-06**. And float32 ulp in [64, 128) is 7.63e-06, so 2ε and 3ε both round to 2 ulp: the gradient is **not** monotone above 64 m, i.e. "5.99e-06 survives the round trip" is false at the summit. Production writes float64 anyway (`analysis_worker.py:216-218`, `dtype="float64", nodata=fa.nodata`), so what remains is a `LOW` docstring finding about a stale "near sea level" claim, not a live defect. **Measured directly this pass:** on this fixture (z_max 84.78 m) the float32 round-trip preserves the sink count exactly — 65 either way, max |Δ| 3.8e-06 m |

**And one ID withdrawn entirely.** A drafted `EWD-59` compared TerrainFlow's diversion-drain
gradient against Yeomans'. The owner's challenge was correct and the code refuses the
conflation explicitly in two places: TerrainFlow's diversion drain is a **constructed
channel**, Yeomans' is a **plough-formed line**. Comparing their grades was a category error,
and `EWD-14` already covers the 1.0 % default. `EWD` stays at **58**.

---

## §4 Candidates, unfiled

Following the audit's own precedent at `MATHS_AUDIT.md:74-75` — candidates that are not yet
grounded well enough for §1 are listed rather than dropped.

> **The coverage rows Step E owed are in §8.9**, measured by `p_coverage.py` over the 30
> subjects that step named. Short version: two of the thirty are called by neither suite
> (`bulking_factor`, `UsableAreaDisjoint`), eight are reached only from inside their own
> module, and `help_text.py`'s 155 tooltips have no test of any kind — four of them quote
> a value the code holds and all four agree with it.

1. **`keypoint_on_path:740` — a NaN keypoint cell reports 0.0 m.**
   `elev = float(self.dem[kr, kc]) if not np.isnan(...) else 0.0`. Sibling of `NEW-W8-01`,
   which was fixed in the *profile*; this is the same substitution in the returned
   **elevation**. Unreachable on the finite fixture, so unmeasured. Needs a nodata DEM to
   exercise.
2. **`advisories.py:301` — a tie reports "capture".** The governing test uses `isclose`, so
   an exact tie between the two rules resolves to capture rather than being reported as a
   tie. Latent; no magnitude measured.
3. **`find_keypoints`' 0.2 ha default is decoupled from the panel's 5 ha stream threshold.**
   The panel's `stream_threshold_ha` defaults to 5.0 and drives every other tool; the keyline
   silently uses `max(20, round(2000/cell_area))` = 0.2 ha because `contour.py:1397` passes
   no threshold. `tests_qgis/README.md:278` documents the 5 ha figure. Two thresholds for one
   user-visible idea. **Not filed** because it is arguably the intended design — a primary
   valley is *not* a 5 ha stream — but it is undocumented either way, and it is why the
   knob sweep found `stream_threshold_ha` live and `keyline_max_valleys` inert.
4. **The QGIS suite leaks its own temp directories.** `run_all.py:131` creates
   `tfa_qgis_checks_*` and `checks_robustness.py:33` (`tfa_nasty_`) and `:105` (`tfa_seed_`) create more; `_state.output_dir`
   creates one `F:/Temp/tfa_*` per harness run. Measured 2026-09-11: **65 `tfa_qgis_*`** and
   **5,689 `tfa_*` totalling 2,365 MB**. This is **test infrastructure, not `CTA-30`**, and
   it dwarfs `CTA-30`'s own 138.7 MB — which is exactly why the two must not be conflated in
   one number.
5. **The nasty synthetic DEM's docstring comment is stale.** `_harness.py:193` says
   `build_synthetic_dem(p, rough=True, pits=6, voids=2)` gives *"18 sinks, 180 nodata cells"*.
   Measured: **30 sinks**, 180 nodata cells. The comment is asserted nowhere, so this is a
   stale comment rather than a regression — but the register must not quote it as a
   measurement, and something did move it.
6. **`CLAUDE.md`'s test-suite figures are stale.** It says ~2,590 tests in ~60 s; measured
   **2,889 passed, 2 xfailed, 5 xpassed in 39.75 s**. Documentation only.

---

## §5 Sources

In §7's four-column shape. The two Yeomans rows and the owner attestation are carried in
`MATHS_AUDIT.md` §7 as well, since they resolve published rows there.

| Key | Title / edition | URL | Grounds |
|-----|-----------------|-----|---------|
| [YEO-WFEF] | Yeomans, K. B. & Yeomans, P. A. (dec.), *Water for Every Farm — Yeomans Keyline Plan*, ISBN 1438225784. Full 368-page PDF fetched and text-extracted 2026-09-11 | cheiodasideia.libertar.org/wp-content/uploads/2022/11/Water-for-Every-Farm-Yeomans-Keyline-Plan-Ken-B.-Yeomans-P.A.dec_.-Yeomans.pdf | KPA-41, KPA-49; and `MATHS_AUDIT` KPA-29/31 |
| [YEO-MKIV] | Yeomans, *Keyline Design Mark IV — "Soil, Water & Carbon for Every Farm"*, 14 pp., fetched and read 2026-09-11 | agwaterstewards.org/wp-content/uploads/2016/08/KeylineArticle.pdf | KPA-41; and `MATHS_AUDIT` KPA-29/31 |
| [OWNER-2026-09-11] | Repo owner's domain ruling — **an attestation, not a fetched source**. See §0.3 | — | corroboration only |
| [CODE] | Source re-read against HEAD `c288a6a` while writing this register; every line citation above re-checked at write time | — | all rows |
| [PROBE] | `tests_qgis/probes/p_{smoke,flow_graph,keypoints,impoundment,controllers}.py` and `tests_qgis/probes/evidence/*.json`, 2026-09-11 | — | every measured number |

**Failed fetch:** yeomansplow.com.au *"Yeomans Keyline Systems Explained"* — 301 to
yeomansplow.com, which returned **403 Forbidden**. It is therefore **not cited**, and
[YEO-MKIV] and [YEO-WFEF] carry the source question on their own.

---

## §6 What Steps B–G still owe

Step A (measure), A1 (probes), A2 (`MATHS_AUDIT` re-anchoring and sources) and A3 (this
register) are complete, and Step G (pinning) is done for the production-path numbers. What
follows is scoped and not started.

> **Superseded 2026-09-11 — see §8.** Steps **B, C, D, E and F are now complete**. The
> table below is left exactly as written, per this document's own convention, and §8 is
> the authority on what they found. Two of the criteria stated in it were measured and
> turned out to be **wrong**, which §8.7 records: the aspect witness in row **D** has its
> sign inverted (`dz_dy > 0` should read `dz_dy < 0` — as written it agrees 0.01 % of the
> time on correct code), and both tolerances in row **F** ask for more precision than a
> float32 return value can carry. Row **C**'s `earthworks.py:2219` does not resolve
> either; the float32 sites in that file are `earthwork_design.py:2137` and `:2250-2266`.
> What remains open from this table is row **G**'s three deliberate exclusions.

| Step | Owes | Probe |
|---|---|---|
| **B** | Re-run `p_flow_graph` and `p_keypoints` as regressions once any fix lands. The roughness sweep moved **into A1** and is done | existing |
| **C** | Two invariance arms: **Z + 600 m** (every output identical under a constant elevation offset — the only arm that reaches the float32 regime `save_result`, `earthworks.py:2219` and `KPA-38`'s fix note all reason about) and **mirror / transpose** (catches `KPA-33`'s row→y flip, `CTA-08`'s rc→map asymmetry, `IMP-01`'s diagonal count and `UNI-15`'s convention in one arm — a **regression** guard on shipped fixes, not an investigation). Plus a `routing='d8'` run, currently tested nowhere | `p_invariance.py` |
| **D** | The reduced battery: **signed witnesses** (plan curvature negative over the top-5 % accumulation network, positive over the top-5 % TPI; `aspect ∈ (90,270)` coincides with `dz_dy > 0` for ≥90 % of cells above 2°), **dimensional identities** (`catchment_ha·10_000 == (acc+1)·cell_area`; `specific_catchment_area(cell 2)/(cell 1) == 2.0` exactly; `erosion_spacing_m == VI(p50)/p50_grade`), and **conservation** (`|accepted| + |refused| == |input|` for every screening tool — which is `KPA-39`, `KPA-43` and `IMP-03` as arithmetic). Honest pass criterion for curvature vs TPI: measured agreement is **64.5 %** at the default 15 m window, so assert `mean(plan[cls==1]) > 0 > mean(plan[cls==-1])` and point-biserial ≥ **+0.25**; a **negative** correlation is the unambiguous failure | `p_battery.py` |
| **E** | Coverage rows for `strahler_order`, `topographic_wetness_index`, `stream_power_index`, `sediment_transport_index`, `landform_classes`, `slope_statistics`, `terrace_vertical_interval`, `bulking_factor`/`compaction_factor`, `embankment_volume`, `UsableAreaDisjoint`/`_sample_line`/`classify_contour_inflow`, `recommend_swale_length`, `ComparisonResult`, `report_model._earthmoving`, `project_io.INPUT_FIELDS` additions, three `panel` properties, and **`help_text.py` (+126 lines of user-facing claims, zero coverage)** — for which the cheap check is text-versus-constant. **`TIX` may be opened here** — `TIX-01` already was, on 2026-09-11 (§7.3). ~~Record separately that `landform_tpi` has no production caller~~ **RESOLVED**: `find_ridgelines` now calls it and `landform_classes` (§7.4). `terrain.py:249-253` still writes six bands with no TPI among them, so a TPI *layer* remains unavailable to the user — that part stands | — |
| **F** | Four exact-tolerance cross-checks: `haul_regions` totals vs `burn_quantities` (**0.1 %**); `slope_degrees` / `aspect_degrees` / `horn_gradient` from one stencil (**1e-12**); `flow_bearing` vs `aspect_degrees` (**1e-6°**); `stream_links` emitted vs consumable vs `keypoints + skipped` (**exact integers**) | `p_crosscheck.py` |
| **G** | **Done** for the production path — see `checks_fixture_regression.check_keyline_network_numbers_have_not_moved`. Still out of scope by decision: the conditioned-surface numbers (unreachable from production — that is `KPA-38`), the 35.0 m floor as a pinned value (it is analytic, not a fixture measurement, and the file's 0.5 % relative tolerance is the wrong instrument for it), and the five-threshold keypoint vector | existing |

### Still owed by the owner

1. ~~**Which Doherty text and edition.**~~ — **DROPPED 2026-09-11 on the owner's decision.**
   `[DOHERTY]` is not added, nothing rested on it, and any constant that would have carried
   it is declared a TerrainFlow convention instead. See §0.3.
2. ~~**Confirmation that an owner attestation may stand as grounds**~~ — **RATIFIED
   2026-09-11 by the owner.** An attestation **may** carry a row. See §0.3.1 for the rule
   it is admitted under.
3. ~~**The `KPA-41` window decision**~~ — **DECIDED 2026-09-11: 50 m**, and implemented;
   see §9.6. The reasoning that produced the recommendation is kept below.

   The measurement
   recommends **50 m** — the shortest window whose steepest-grade spread stays inside
   `MIN_SLOPE_EASE` across `roughness_m` 0.00 → 0.10. 10 m and 20 m are 2.18× and 2.44×
   `MIN_SLOPE_EASE` and are not defensible on noisy ground. The window is a **TerrainFlow
   convention** sized on **synthetic** roughness, and the decision is the owner's.

---

## §7 Fixed since this register was written (2026-09-11)

Four rows are closed. §1 and §2 are left as written; this section is the authority on
current state.

Everything here was **measured before and after**, on the real fixture, and every claim
below is a number from that pair of runs rather than a reading of the diff.

| ID | Was | Now |
|---|---|---|
| `CTL-02` | aspect ramp laid at −360 … 113400 over data spanning [−1, 360]; **0.32 %** of the ramp occupied | stops at −1, 0, 45 … 360 over the same data; **100.00 %** occupied |
| `CTL-03` | the check that should catch it asserted only that a layer id existed | asserts the ramp resolves its band; **verified to fail on the pre-fix code** |
| `TIX-01` | **new** — `landform_classes(tpi, slope_deg, …)` never read `slope_deg` | parameter removed |
| `KPA-12` (published, `MATHS_AUDIT` §1 #18) | ridgeline TPI window in **cells**, cut at an absolute **1.5 m** | window in metres, cut in standard deviations — see `MATHS_AUDIT` §9.9 |

### 7.1 `CTL-02` — the aspect map is a gradient again

**Root cause, confirmed:** `ASPECT_CLASSES` is the only palette in `map_palette.py` whose
first element is a **value** (compass degrees) rather than a **fraction of the band
maximum**. `apply_raster_ramp` had one path and it multiplied. With `top = band_max ≈ 360`
the nine stops landed at −360, 0, 16200 … 113400, so every real value fell inside the first
stop and the layer drew as one wash.

**Fix:** `apply_raster_ramp` gains `absolute=False`. When true, stop values are laid down
untouched. `INDEX_SPECS` gains a fourth field saying which palette is which, and aspect is
the only one that sets it. Existing callers are untouched and keep the fractional path, so
nothing else moved — the 49 screenshot baselines are still pixel-identical.

**And the compass now closes.** A stop was added at 360° carrying the north colour. Aspect
is circular: a face at 359° is north-facing, and without the closing stop everything from
315° to 360° clamped flat onto the NW colour. The register's `WRONG` verdict stands as
written; it is now `WRONG (fixed)`.

**Measured, real fixture:**

| | stops | band | ramp occupied |
|---|---|---|---|
| before | −360.0, 0.0, 16200.0 … 129599.9 | [−1.000, 360.000] | **0.32 %** |
| after | −1.0, 0.0, 45.0 … 360.0 | [−1.000, 360.000] | **100.00 %** |

### 7.2 `CTL-03` — the check now makes the claim its name makes

`check_every_terrain_index_renders` carried the docstring *"a ramp that resolves to one
flat colour is indistinguishable from a broken layer"* over a body that asserted a layer id
existed. It would have passed on `CTL-02` — and did, for as long as `CTL-02` was live.

It now measures the ramp against the band it paints and refuses a ramp too wide to resolve
it. The bar is **10×**; measured ratios at the time of writing are aspect 1.0× (was 360×),
TWI/SPI/STI ~1.0×, and the two curvature ramps 0.14× and 0.10×. It is deliberately
one-sided: a ramp *narrower* than its band is the documented, intended behaviour for
curvature, which anchors on ±p95 so that a couple of cliff-edge cells cannot flatten
everything else.

**The check was verified against the defect, not just added.** Re-registering aspect as
non-absolute — the exact pre-fix state — makes it fail with:

```
aspect: the colour ramp spans 129,959.901 over data spanning 361.000 — 360.0x too wide,
so the layer resolves to roughly one colour.
```

A check that would not have caught the bug it is named for is what `CTL-03` *was*.

### 7.3 `TIX-01` — an inert parameter, found while wiring the orphans

- **Tokens:** `DISC / I / n / H` — opened and closed in the same pass
- **Where:** `terrain_indices.py:250`, `landform_classes(tpi, slope_deg, sd, mask)`
- **Observed:** `slope_deg` was accepted and never read. Every call site in the tree passed
  `None` for it — including all four in `tests/test_terrain_indices.py`, which is how it
  survived: the tests documented the parameter as unused and nobody read them that way.
- **Fix:** removed, rather than wired. Weiss's *fuller* scheme does use slope, but only to
  split a fourth class (`plains`) out of `midslope`, and this function returns three
  classes. Restoring it means implementing that split, not re-adding an argument. The
  docstring now says so, so the next reader does not re-add it.
- **Why it matters here:** `landform_classes` was about to gain its first production
  caller. Wiring a dead parameter into production is how `MHL-01` and `KPA-48` happened.

**This is the first `TIX` row.** §0.2 reserved the prefix for `terrain_indices.py` and
recorded that none existed yet.

### 7.4 The two orphaned functions now have a caller

§4 and the closing summary of this register recorded `landform_tpi` and `landform_classes`
as having no production caller: the terrain tool writes six bands and neither is among
them, while `find_ridgelines` computed its own TPI inline.

Both are now called from `find_ridgelines`, and the inline copy is deleted. The
consolidation runs in the direction the repo owner set — ridgeline analysis stays where it
lives, alongside the Yeomans keyline work, and the library functions serve it rather than
competing with it.

**The orphans were the *corrected* implementations, which is the part worth remembering.**
`landform_tpi`'s docstring records the cell-count window as a fault it had already fixed;
the shipping copy still had it. `landform_classes`'s docstring says an absolute metre
threshold cannot transfer between sites; the shipping copy used one. Someone fixed this
properly in `terrain_indices.py` and never rewired the caller, so the repo has been running
the superseded code and testing the replacement. See `MATHS_AUDIT` §9.9 for the
measurements.

### 7.5 What this pass did **not** change

- **The report still has no analysis-tier page** (§4, item in the closing summary). Eleven
  pages, and `keyline` does not appear in `report_model.py`. Deferred by the owner.
- **`acc <= 2`** — the ridge test itself is untouched, and it is why real ridges fragment
  into short runs where the synthetic surface gives 592 m spines. Whether one 77 m ridgeline
  on a 16 ha clip is a *useful* answer is a design question, not a correctness one.
- **`tpi_window_m`, `min_tpi_sd` and `min_length_m` are still not exposed in the panel** or
  persisted in `project_io.INPUT_FIELDS`. A user on ground the defaults do not suit still
  cannot reach them. That is feature work, and it is not done.
- **Terrain indices have no screenshot coverage.** The 49 baseline images cover the baseline
  layers; none renders a terrain index, which is why `CTL-02` could not have been caught by
  the visual tier either. `CTL-03`'s strengthened assertion is now the only thing standing
  between that ramp and another flat map.
- Every other row in §1 is open exactly as written.

---

## §8 Steps B–F (2026-09-11)

Steps B, C, D, E and F are complete. Four new probes (`p_invariance`, `p_battery`,
`p_coverage`, `p_crosscheck`) and five permanent checks in
`tests_qgis/checks_fixture_regression.py`.

This section is the authority on what those steps found, exactly as §7 is the authority
on what was fixed. §1 and §2 are still not edited — that convention is why `MATHS_AUDIT`
survived a rebuild.

**Four new findings, and one of them explains six published rows.** They are `KPA-52`
(the mask and the pointers come from different routing schemes), `FLA-26` (`routing='d8'`
is a two-click crash), `KPA-53` (the routing setting never reaches the keyline tier) and
`KPA-54` (the keypoint label quotes the wrong catchment). `TIX-02` is opened at `I`.

Everything else these steps ran **passed**, and the passes are recorded too: an
invariance arm that finds nothing is evidence about the code, not a wasted afternoon.

**A note on the four new IDs, because the campaign has been bitten by this twice now.**
Two separate mistakes in one draft:

1. **`KPA` was already at 51 in this register, not 48.** §1 rows 8, 23 and 27 are
   `KPA-50`, `KPA-51` and `KPA-49`. The first draft filed the new findings as 49, 50 and 51
   and collided with all three. They are `KPA-52`, `KPA-53` and `KPA-54`.
2. **`FLG` is `flow_graph.py`; the crash is in `flow_analysis.py`.** The first draft filed
   it as `FLG-20`. `flow_analysis.py`'s prefix is **`FLA`** (`MATHS_AUDIT` §2.1), which runs
   to 25 — so the crash is `FLA-26`, and, checking the right family for ancestors turned up
   `FLA-02`, an `UNVER-B` row about *this exact except clause*. Filing under the wrong
   prefix would have hidden a supersession, not just misnamed a row.

`TIX` was at 01, so `TIX-02` was free. The original plan's own review caught 13 of 24
proposed IDs colliding; the lesson does not stop applying because the register is now the
thing being extended rather than the thing being cited.

### §8.1 `KPA-52` — the channel mask and the pointer graph disagree about routing

> **STILL OPEN — see §9.5 for the routing table and §9.7 for the routing-free option.** Since `KPA-48` was fixed the
> mask follows the panel's routing rather than a hard-coded literal, so the mismatch is
> now a function of a user setting with a measured cost: the D-infinity default yields
> **1** keypoint where D8 yields **5**. §9.5 carries the three-way table and the
> decision it needs.

| Id | Sev | Dir | Where | Claim | Verdict | Conf |
|---|---|---|---|---|---|---|
| `KPA-52` | M | U(valley detect) | `keypoint_analysis.py:803,809` | `find_keypoints` builds its stream mask from pysheds **D-infinity** accumulation and then traces links along **D8** pointers. On the fixture that shatters a 1,688-cell network into 127 fragments of median 3.4 m, of which **1** is long enough to profile. Mask taken from the same graph as the pointers: 13 links, median 39.8 m, **7** long enough | WRONG | H |

**Where.** `_ensure_flow_data` asks pysheds for `routing="dinf"` (`:1173`) and returns
that accumulation. `find_keypoints` thresholds it into `stream` (`:803`), then calls
`flow_graph.d8_from_dem(self.dem, …)` (`:809`) for the pointers `stream_links` walks. The
two are different routing schemes over the same ground.

**What was run.** `p_crosscheck.stage_mask_and_pointers_agree`, at the production 0.2 ha
threshold on the real fixture, holding the pointers fixed and changing only where the mask
comes from.

| | D-infinity mask (production) | D8 mask, same graph as the pointers |
|---|---|---|
| stream cells | 1,688 | 684 |
| order-1 links | 127 | 13 |
| stream cells inside a link | 717 | 630 |
| stream cells in **no** link | **971** | 54 |
| cells whose D8 pointer leaves the mask | **125** | **0** |
| median link length | 3.4 m | 39.8 m |
| longest link | 54 m | 157 m |
| links clearing the 35 m profile floor | **1** | **7** |

**Root cause.** D-infinity divides a cell's flow between two downslope neighbours, so it
wets a broader and more diffuse network than D8 concentrates into. A D8 pointer traced
through that wider mask can leave it — 125 cells do — and `stream_links` ends a link
wherever that happens. The result is not a channel network; it is the D8 skeleton cut
into pieces wherever the two schemes disagree, with 971 of the 1,688 masked cells in no
link at all.

**Blast radius.** This is the mechanism under six published rows, all of which measured a
*symptom* of it:

- `KPA-39` — the guard histogram. 78 refusals at *"thalweg is None or len(thalweg) < 5"*
  and 48 at *"profile too short to have an interior"* are both fragment-length refusals.
- `KPA-40` — the 35 m floor. The floor is correct; almost nothing reaches it because the
  links are 3.4 m.
- `FLG-18` — emitted against consumable. 127 against 1.
- `FLG-19` — mask-leaving pointers. 125 on the raw surface `find_keypoints` actually uses
  (the register's published 72 is the *conditioned* surface, which production does not
  reach — that is `KPA-38`). Both figures are in `p_flow_graph.json`; neither is wrong.
- `KPA-43`, `KPA-44` — the accounting identity and the refusal message, both counted over
  a link list this produces.

**What this does not say.** It does not say D-infinity is the wrong accumulation. It is
the more physical of the two and is the default for good reasons. The finding is that
**mixing** them is what fragments the network, and nothing in the tree says which of the
two `find_keypoints` means. Either half is a defensible fix; running both at once is not.

**holds while** `_ensure_flow_data` hard-codes `routing="dinf"` and `find_keypoints`
calls `d8_from_dem`. **blocks** a truthful `KPA-39` refusal message: until the mask and
the pointers agree, most refusals are about fragmentation, not about the valley.
**reproduce** `p_crosscheck.py` → `evidence/p_crosscheck.json`,
`crosscheck_mask_routing_vs_pointer_routing`. **supersedes** nothing; it is the *cause* of
six rows, not a replacement for them.

### §8.2 `FLA-26` — `routing='d8'` is two clicks from an unhandled crash

> **CLOSED — see §9.1.** Left as written, including its proposed fix, which was **wrong**:
> widening the `except` clause would have moved the crash one line down, because
> `Grid.accumulation` defaults to `routing='d8'` and the fallback call re-enters the same
> function. §9.1 has what the fix actually is and why the proposal failed.

| Id | Sev | Dir | Where | Claim | Verdict | Conf |
|---|---|---|---|---|---|---|
| `FLA-26` | H | n | `flow_analysis.py:514-517` | Selecting "D8" in the Routing combo raises `AttributeError: module 'numpy' has no attribute 'in1d'` out of pysheds and kills the baseline run. The same hazard is guarded in the two paths that never request d8 and unguarded in the one whose routing the user controls | WRONG | H |

**Severity.** `H`, and the escalator is not what puts it there — the keyline path emits
geometry, so §0.1's under-sizing rule does not fire. It is `H` because a labelled option
in the shipped UI takes the plugin's entry point down with an unhandled exception, and no
sizing question can be asked at all afterwards. This is the register's first `H` outside
`MATHS_AUDIT`'s three sizing constants, and it is filed knowingly.

**The path, end to end.** `panel.py:773` — `addItems(["D-infinity (recommended)", "D8"])`.
`panel.py:2550-2551` — `routing` returns `"d8"` when the combo text contains `D8`.
`baseline.py:427` passes `routing=self._panel.routing` to the worker;
`analysis_worker.py:147` calls `fa.run(routing=self.routing, …)`; `flow_analysis.py:515`
calls `self.grid.accumulation(self.fdir, routing="d8")`; `pysheds/sgrid.py:904` calls
`np.in1d`, removed in NumPy 2.

**Why the guard does not catch it.** `flow_analysis.py:514-517` is
`except TypeError`. The raise is an `AttributeError`. Both sibling call sites
already write the wider clause:

| Call site | Guard | Requests d8? |
|---|---|---|
| `catchment.py:223-226` | `except (TypeError, AttributeError)` | never — hard-codes dinf |
| `keypoint_analysis.py:1178-1182` | `except (TypeError, AttributeError)` | never — hard-codes dinf |
| **`flow_analysis.py:514-517`** | **`except TypeError`** | **yes — the panel's value** |

**Previously known, never measured.** `project_io.py:85` says in a comment that *"a file
restoring as d8 can raise outright"*, and the persisted default was set to `"dinf"` on
that basis. The comment is correct and the default is the right one; what neither did was
stop the combo from offering d8, or the run from dying when it is chosen. The memory of
this divergence is recorded as a known-open item — "left alone because it changes every
assessment" — which is true of the *numerical* difference between the two schemes and not
true of the crash.

**The fix is the except clause**, one word wider, matching its two siblings — after which
d8 falls back to whatever pysheds' default accumulation gives, exactly as the other two
paths already do. Not applied here: this campaign documents and pins.

**`supersedes` — `MATHS_AUDIT` `FLA-02`**, and this is a genuine supersession rather than a
family resemblance. That row reads *"flowdir+accumulation; **silent routing downgrade on
TypeError**"* at the pre-rebuild `:90-99`, verdict `UNVER-B` — verification blocked on
pysheds documentation. Running it answers what reading the docs could not: the downgrade is
not silent and does not happen at all, because the exception raised is not the one caught.
`FLA-02` closes to this row.

**holds while** pysheds calls `np.in1d` and NumPy 2 is installed. A pysheds release that
drops the call would close this without anyone touching TerrainFlow, which is why the
check pins the *exception*, not the behaviour. **reproduce** `p_invariance.py` →
`evidence/p_invariance.json`, `routing_dinf_vs_d8`. **pinned by**
`check_d8_routing_is_still_the_documented_crash`, which asserts the crash on purpose and
carries instructions to invert it when the fix lands.

### §8.3 `KPA-53` — the routing setting never reaches the keyline tier

> **CLOSED — see §9.3**, and not by the route this entry implies. The fix is not to thread
> the panel's routing down a second path: the accumulation raster `contour.py` already
> hands over came from `FlowAnalysis.run(routing=panel.routing)`, so honouring it (which is
> `KPA-48`) carries the routing choice in the data. Both rows close on one change.

| Id | Sev | Dir | Where | Claim | Verdict | Conf |
|---|---|---|---|---|---|---|
| `KPA-53` | M | i-d | `keypoint_analysis.py:1173` | `_ensure_flow_data` hard-codes `routing="dinf"`, takes no routing argument, and `find_keypoints` has no parameter to pass one. A user who selects D8 gets D-infinity under every keypoint and keyline regardless | WRONG | H |

Measured by signature, not by reading: `_ensure_flow_data(self)` accepts no `routing`
argument (`p_invariance.routing_reaches_keyline_tier`), and
`find_keypoints(self, max_valleys=8, stream_threshold_cells=None, max_order=1,
boundary_mask=None)` has none to forward.

Today this is masked by `FLA-26`: the run dies before the keyline tier is reached. Fix
`FLA-26` alone and the plugin acquires a *silent* divergence in its place — the baseline
raster the user sees routed one way, the keypoints drawn on top of it routed the other.
Recorded now so that the second defect is not created while removing the first.

`i-d`, not `n`: whether the two disagree at all depends on the terrain.

### §8.4 `KPA-54` — the keypoint label quotes the valley's catchment, not the keypoint's

> **CLOSED — see §9.4.**

| Id | Sev | Dir | Where | Claim | Verdict | Conf |
|---|---|---|---|---|---|---|
| `KPA-54` | M | O | `keypoint_analysis.py:816-818, 833-837` | `catchment_ha` is read at the **link's outlet cell** and then labelled *"N ha above"* on a keypoint that sits partway up the link. On the rank-1 keypoint that is 5.867 ha against 2.116 ha actually above it — an overstatement of 3.751 ha, **177%** | WRONG | H |

`_catchment(link)` (`:816-818`) returns `acc_arr[link[-1]]`, the accumulation at the bottom of the
valley link. That is the right figure for **ranking** valleys, which is what
`links.sort(key=_catchment)` uses it for and why it is computed at all. It is then carried
onto the keypoint as `kp["catchment_ha"]` and rendered into
`f"Keypoint at {elevation:.1f} m — {catchment_ha:.1f} ha above"`.

A keypoint is a point on the profile; the ground above it is the ground above *it*. The
label says "above" and means "above the bottom of the valley this keypoint is on".

`O`, per §0.1's practice: it over-reads the catchment. It is not a sizing path — nothing
is dimensioned from this number — which is why it is `M` and not higher. What it does is
tell the user, on the map, that a keypoint commands nearly three times the country it
commands.

**reproduce** `p_battery.py` → `evidence/p_battery.json`,
`identity_catchment_ha_plus_one`. The probe pairs each keypoint to its own link by cell
membership and checks `valley_cells` against the link length, so a mis-pairing would show
up rather than skew the number quietly.

### §8.5 `TIX-02` — two conventions for whether a cell drains itself

| Id | Sev | Dir | Where | Claim | Verdict | Conf |
|---|---|---|---|---|---|---|
| `TIX-02` | I | n | `terrain_indices.py:77` vs `keypoint_analysis.py:833` | `specific_catchment_area` adds `+1` to the accumulation and documents *measuring* that pysheds excludes the cell itself; `find_keypoints` computes `catchment_ha` from the same raster without it. One cell — 1 m² on the fixture | DISC | H |

Filed at `I` because the magnitude is one cell and neither figure is used where that
matters. Filed at all because the two are the same question answered two ways in one
tree, and `specific_catchment_area`'s docstring is emphatic about why the `+1` is there
(*"without it every ridge cell has zero catchment and `ln(a)` is `-inf` along the top of
every hill"*). A reader who trusts that docstring and then reads `find_keypoints` learns
the opposite.

### §8.6 What Steps C, D and F ran that **passed**

Recorded because a passing oracle is a measurement. All on the real fixture, 2026-09-11.

**Z + 600 m (Step C).** Slope, plan and profile curvature, TPI, the D8 pointer graph
(0 disagreements), sink count (525 both), order-1 link count (13 both), keypoint positions
(`[(70, 51)]` both) and the skipped count (126 both) are all unchanged; the keypoint
elevation rises by exactly 600. Worst index delta 4.657e-10 on TPI, everything else
exactly 0.

The interesting half is *why* it passes. At the fixture's 84.78 m float32 spacing is
7.63e-06 m and `resolve_flats`' 1e-05 m inflation step survives it; at 684.78 m the
spacing is 6.10e-05 m and it does not — which is exactly the failure
`flow_analysis.save_result`'s docstring documents. Through the keyline tier's own float32
temp raster (`keypoint_analysis.py:1147-1171`, which bypasses `save_result` entirely),
**15,622 distinct elevation levels are quantised away** at +600 m — and the sink count on
the conditioned surface is **65 at both elevations**. `safe_flat_epsilon` derives its step
from the surface's own elevation instead of taking pysheds' fixed 1e-5, and that
derivation is the whole reason the arm passes. The permanent check guards it.

**Mirror (Step C).** Slope, plan curvature, dz_dx (negated), dz_dy (unchanged) all
exactly 0; TPI 3.6e-12; aspect reflects to within 1.5e-05 deg with **0** flat-sentinel
flips. Sinks and links identical. D8 pointers: 31 disagreements, **all 31 exact steepness
ties** — expected, because a flip reverses `_OFFSETS`' scan order
(`flow_graph.py:106-113`) — and **0** on cells where the two candidates differ in
steepness. Keypoints mirror. `UNI-15` and its `CTA-08` component stay fixed.

**Signed witnesses (Step D).** Plan curvature averages **-4.357e-02 /m** over the
top-5 % flow network (hollows converge: must be negative) and **+4.088e-02 /m** over the
top-5 % TPI (noses diverge: must be positive). Against TPI's own classes: **+3.658e-02**
on ridge, **-4.401e-02** in valley, point-biserial **+0.489** raw and **+0.680** after a
15-cell pre-smooth. Aspect in (90, 270) coincides with `dz_dy < 0` on **100.00 %** of
152,044 cells steeper than 2 deg — 0 disagreements.

**Dimensional identities (Step D).** `specific_catchment_area` at 2 m over 1 m is
**exactly 2.0** at every accumulation tested. The terrace unit chain closes exactly:
at the site's p50 slope of 8.062 deg (14.165 %), `spacing_advisory`'s
`erosion_spacing_m` is 22.591630 m against `VI/grade` = 22.591630 m, relative error
**0.00e+00**; and `VI` at zero slope is 0.609600 m against `Y = 2.0 ft = 0.609600 m`, so
the feet-to-metres conversion the rule needs is exact and no slip is hiding in it.

**Conservation (Step D).** `keypoints + skipped == links` holds at `max_valleys` 8 and
127 (1 + 126 = 127) and fails at `max_valleys=1` with **124 links never examined** —
`KPA-43` exactly as published. `clip_to_usable_area` conserves 10 = 5 kept + 5 dropped.

**Haul volumes (Step F).** `haul_regions` with `min_region_m3=0` sums to
**497.50 m³ cut / 398.80 m³ fill** against `burn_quantities`' identical figures —
relative error 0.00e+00 and 1.43e-16, against a 0.1 % bar. At the shipped
`min_region_m3=5 m³` the haul plan omits **0.80 m³ of fill across 8 regions**, which the
site total still counts; that is the threshold working, not an error, and it is now a
known quantity.

**Link accounting (Step F).** 127 links emitted at `min_cells=3`, **1** clearing the 35 m
profile floor; `keypoints + skipped = 127` matches emitted exactly. See `KPA-52` for why
the first two numbers are so far apart.

### §8.7 Two of the plan's own criteria were wrong, and are corrected here

Both were stated in the original plan and repeated in §6 of this register. Both were
measured rather than assumed, which is how they were caught.

**The aspect witness had its sign inverted.** §6 and the plan both state it as
*"`aspect ∈ (90,270)` coincides with `dz_dy > 0` for ≥90 % of cells above 2°"*. Measured
on correct code, that form agrees **0.01 %** of the time. The correct statement is
`dz_dy < 0`, which agrees **100.00 %**: `horn_gradient`'s `dz_dy` rises toward increasing
row and row increases southward, so ground *falling* to the south has `dz_dy < 0` — and
aspect, which points downslope, lands in (90, 270) exactly there. Algebraically
`aspect = atan2(-dz_dx, dz_dy)` has `dz_dy` as its northward term. The plan carried the
docstring's "increasing row is southward" through without its negation.

Worth keeping as a pattern: a witness stated backwards fails at *near zero*, not at
"below the bar". An agreement of 0.01 % is not a marginal result, and reading it as one
would have filed a defect against `aspect_degrees`.

**Two Step F tolerances ask for more precision than the values carry.** The plan sets
1e-12 for the shared-stencil check and 1e-6 deg for `flow_bearing` against
`aspect_degrees`. Both `slope_degrees` and `aspect_degrees` return **float32**, whose
spacing is 3.05e-05 deg at 360 deg. Measured: the stencil re-derivation agrees to
1.907e-06 deg (slope) and 1.526e-05 deg (aspect), and `flow_bearing` agrees with
`aspect_degrees` to 1.513e-05 deg over 2,000 sampled cells. Every one of those is inside
float32's own spacing and outside the plan's bar. The corrected criterion is
**"no worse than the cast"**, which all three meet. Filing these as failures would have
filed a defect against `.astype("float32")`.

**One figure the plan quotes is reproduced with a different denominator.** The plan says
curvature-versus-TPI agreement is 64.5 % at the default 15 m window. Measured here over
the cells TPI actually *classes* (ridge or valley, excluding midslope): **73.2 %**. Both
are descriptions of the same correct behaviour; neither is the pass criterion, for the
reason §6 already gives.

### §8.8 What this pass corrected in its own instruments

**`p_keypoints.py`'s guard map had rotted, and its evidence file was wrong.** The probe
carried a hand-written `{lineno: guard}` dict for the five `return None` sites in
`keypoint_on_path`. The ridgeline fix (`ac2966b`, §7.4) moved all five down 20 lines. The
tracer went on counting correctly — 78 / 48 / 0 on the fixture, 2 / 26 on the synthetic
DEM, unchanged — but attributed every one to `"UNKNOWN LINE"`, which inverted the two
figures derived from that map: `refusals_actually_from_prominence` read 26 → 0 and
`message_is_false_for` read 2 → 28. The committed evidence would have supported the claim
that `KPA-44`'s message is false in all 28 synthetic cases, when it is false in 2.

`guard_lines()` now walks the function's AST on every run and labels each `return None`
by the source of its enclosing `if`. The gloss table is keyed by the **condition**, not
the line, so a guard that moves keeps its name and a guard that is *rewritten* honestly
loses it.

This is the whole argument for Step B. Both probes were re-run as regressions;
`p_flow_graph` reproduced bit-identically, and this is what `p_keypoints` turned up.

**`p_coverage.py`'s first two answers were both artefacts of its own filters.** It
excluded each subject's defining module when counting production callers, which reported
`specific_catchment_area`, `transect_cells`, `flow_bearing`, `embankment_volume` and
`clip_to_usable_area` as having *no production caller* when each is called by its own
siblings. And its help-text constant parser stopped at the first `)` that ended a line,
truncating any tooltip containing a parenthetical aside. Both are fixed; the corrected
run distinguishes "called only from inside its own module" as its own state.

### §8.9 Step E — coverage

Measured by `p_coverage.py` over the 30 subjects Step E named, counting **call sites**
rather than mentions.

- **Not exercised by either suite: `bulking_factor`, `UsableAreaDisjoint`.** Two of
  thirty. `compaction_factor`, its sibling one line away, *is* exercised.
- **Called only from inside their own module (8):** `specific_catchment_area`,
  `terrace_vertical_interval`, `capture_spacing`, `embankment_volume`, `transect_cells`,
  `flow_bearing`, `UsableAreaDisjoint`, `clip_to_usable_area`. All are wired — each has a
  caller in its own file — but none is reached across a module boundary, so a change to
  any of their signatures is invisible outside one file.
- **Panel properties:** `keyline_max_grade_n`, `keyline_max_valleys` and
  `set_spacing_advice` are all exercised. `set_spacing_advice` only by the QGIS suite.
- **`help_text.py`: 1,475 lines, 155 tooltip constants, zero tests in either suite.**
  Four numeric claims that quote a value the code holds were checked against it —
  the swale placement default (0.5 ha), the surface-runoff fade top (2 m³), the analysis
  max-slope default (18°) and the berm's spoil compaction (75 %) — and **all four agree**.
  The remaining 151 state geography, method or advice rather than a value, and no
  mechanical rule separates those from the ones that do; the table in `p_coverage.py` is
  the honest instrument and it is extended by hand.
- ~~`landform_tpi` has no production caller~~ — **RESOLVED** in §7.4, and the corrected
  probe confirms it: `find_ridgelines` calls both it and `landform_classes`.

### §8.10 Five permanent checks, and why they are not recorded numbers

Added to `tests_qgis/checks_fixture_regression.py`. Total added runtime: the module runs
9 checks in **18 s**.

| Check | Guards |
|---|---|
| `check_terrain_answers_do_not_depend_on_absolute_elevation` | Z+600 m over indices, pointers, sinks, links and keypoint positions — the only arm in either suite that reaches the float32 regime, and the guard on `safe_flat_epsilon`'s elevation-derived floor |
| `check_terrain_answers_mirror_under_a_horizontal_flip` | The whole sign-and-axis class at once, with tied D8 pointers separated from real disagreement |
| `check_signed_rasters_point_the_right_way` | An inverted sign, which every percentile and finite-fraction check in the suite passes unchanged |
| `check_haul_volumes_agree_with_the_burn` | `haul_regions` against `burn_quantities`, two routes to one sum |
| `check_d8_routing_is_still_the_documented_crash` | `FLA-26`, asserted as broken on purpose, with instructions to invert it when fixed — **inverted the same day** to `check_d8_routing_runs_and_differs_from_dinf` (§9.1) |

Everything above them in that file pins a **measurement** and must be re-recorded when the
maths deliberately changes. These pin a **property** — a relation that holds on any
terrain at any elevation — so they never need re-recording and a failure is always a
defect. That is why they can share the file without adding to its maintenance burden.

One number in their output will look wrong beside the recorded constants and is not:
the invariance arms report **13** order-1 links where `EXPECTED_KEYLINE` records **127**.
Both are right. The arms derive their stream mask from the same D8 graph the pointers come
from; production derives it from pysheds' D-infinity accumulation. That gap is `KPA-52`.

### §8.11 What Steps B–F did **not** change

No behaviour was changed in this pass. Everything in §8.1 through §8.5 is open.

*(`FLA-26` was fixed straight afterwards, on the owner's instruction — §9.1. Everything
else in this paragraph stands.)*

Beyond those: `FLA-26` is a live crash on a labelled UI control and the fix is one word in
one `except` clause — it is left open only because this campaign documents and pins, and
the decision to ship a fix is the owner's. The three ridgeline thresholds are still
unreachable from the panel. The report still has no analysis-tier page. Terrain indices
still have no screenshot coverage. `bulking_factor` and `UsableAreaDisjoint` still have no
test. And 151 of `help_text.py`'s 155 tooltips are still unchecked against anything.

---

## §9 Fixed after Steps B–F (2026-09-11)

§8 is left exactly as written; this section is the authority on what has since changed.
One row closes.

### §9.1 `FLA-26` — D8 routing runs

**Closed.** `routing='d8'` now completes on the real fixture. `check_d8_routing_is_still_the_documented_crash`, which asserted the crash on purpose and carried instructions to invert itself, has been inverted: it is now `check_d8_routing_runs_and_differs_from_dinf`.

**The fix is not the one §8.2 proposed, and §8.2 was wrong about it.** That entry said
*"The fix is the except clause, one word wider, matching its two siblings"*. It is not, and
the reason is worth recording because it is the kind of mistake that ships:
`Grid.accumulation`'s signature defaults to `routing='d8'` (`pysheds/sgrid.py:822`), so the
fallback call inside that `except` block — `self.grid.accumulation(self.fdir)` — re-enters
`_d8_accumulation` and raises the identical `AttributeError` one line further down. Widening
the clause would have **moved** the crash from `flow_analysis.py:515` to `:517` and left a
check passing for the wrong reason. The proposal was made from reading the guard and not the
line it guards.

**What the fix actually is.** pysheds 0.5 calls `np.in1d` at **nine** sites in `sgrid.py`,
not one, and NumPy removed the name in 2.0 (this environment: 2.5.2). Every site is on a D8
code path, which is why only the D8 option ever met it. A new module,
`modules/pysheds_compat.py`, restores `np.in1d = np.isin` when the attribute is absent and
re-exports `Grid`; the **five** modules that build a pysheds `Grid` — `flow_analysis.py`,
`catchment.py`, `keypoint_analysis.py`, `earthwork_design.py` and `simulation.py` — now
import `Grid` from there.

Five, not four: `simulation.py:565` was missed by the first grep of this pass, which ran
under a `head -10` that cut the list. Importing `Grid` **from the compat module** rather
than applying a shim as a bare side effect is what makes that class of miss visible — a
sixth pysheds importer added tomorrow either imports from here and is shimmed, or imports
from `pysheds.grid` and shows up in one grep.

The substitution is exact, not approximate. `np.isin` is NumPy's own documented replacement;
the two differ only in that `in1d` flattened its result while `isin` preserves shape, and
every pysheds call site passes `fdir.ravel()` and then calls `.reshape(fdir.shape)` on what
comes back. The shim **adds** a name rather than replacing one — nothing on NumPy 2 can be
depending on different behaviour from `np.in1d`, because anything calling it already fails —
and it is guarded by `hasattr`, not a version test, so a NumPy that restores the name keeps
its own implementation.

**The `except TypeError` clause is deliberately left narrow.** With the shim in place there
is nothing for a wider clause to catch, and widening it would mean the *next* removed alias
produces a silent routing downgrade instead of a traceback — which is precisely what
`MATHS_AUDIT`'s `FLA-02` was opened about. The guard keeps its documented job: tolerating a
pysheds too old to accept a `routing=` keyword.

**Measured on the real fixture**, dinf against d8 on the same conditioned surface:

| | dinf | d8 |
|---|---|---|
| runs | yes | **yes** — was `AttributeError` |
| accumulation max | 67,198.9 cells | **105,167.0 cells** |
| unrouted cells | 0 | 0 |
| cells where accumulation differs | — | **152,647 of 160,000 (95.4 %)** |
| conditioned surface | identical, max delta **0.0 m** | |

The two schemes disagree about where water goes on 95 % of the tile, and D8's trunk carries
**56 % more** than D-infinity's because nothing is divided off it along the way. That
divergence is the entire reason for offering the option, and it is now reachable.

The conditioned surfaces being bit-identical is not incidental — pit-filling, depression
filling and flat resolution all run before the routing branch (`flow_analysis.py:503`), so
anything other than 0.0 would mean the conditioning had silently become
routing-dependent. The check asserts it.

**`KPA-53` is not closed by this and becomes live because of it.** With the crash gone, a
user who selects D8 gets a D8 baseline raster and D-infinity accumulation under every
keypoint and keyline, because `_ensure_flow_data` still hard-codes `routing="dinf"` and
takes no argument. §8.3 predicted exactly this — *"fix `FLA-26` alone and the plugin
acquires a silent divergence in its place"* — and it is now the standing state rather than a
prediction. It is the next thing to fix on this path.

### §9.2 What the fix uncovered in the pure suite

The `np.in1d` breakage was not only worked around in production code. It was worked around
in **six places** in `tests/`, and restoring the alias turned all of them into statements
that are no longer true.

**Two `xfail` markers, covering seven tests, are removed.** `_PYSHEDS_NUMPY2_COMPAT_PON`
(five tests over `DEMBurner.get_ponding_layer` and the plugin's wrapper) and
`_PYSHEDS_NUMPY2_COMPAT` (two over `_run_simulation`) both carried
`reason="pysheds 0.5 uses np.in1d removed in NumPy 2.0"`. Six of the seven xpassed the
moment the shim landed and now assert properly.

**The seventh was never about `np.in1d` at all.** `test_run_simulation_with_stores` failed
on a deliberate `ValueError` from `simulation.py:574-578` — it passed `earthwork_stores`
with no `catchment_labels`, which that function refuses outright, because there is no
correct fallback (sampling the cumulative accumulation raster double-counts every upstream
feature's catchment — the bug `water_balance.py` removed). The test had been calling the API
in a way the API explicitly rejects, and **`strict=False` on a marker blaming NumPy hid it**,
because a non-strict `xfail` accepts a failure for any reason whatsoever.

That is the general lesson and it is worth more than the fix: *an `xfail` whose stated
reason is wrong does not merely mis-document, it silently accepts a different failure.* This
one had a test dead in the tree for as long as that guard has existed.

The test now supplies a labelling — indices into `catchment_label_ids`, negative for cells
that drain to nothing — and gains the assertion it was missing: that the store actually
**receives water**. Without it the test passed on a summary row for a feature nothing drains
to, which is all it would ever have checked.

**Four comments and a docstring still route around a bug that no longer exists.**
`test_flow_analysis_assessment.py:507` runs dinf and then sets `fa.routing = "d8"` by hand
*"so we run dinf then flip fa.routing to hit the d8 bearing path without running d8
accumulation"*; `:1341`, `:1359` and `:1377` each carry
`# pysheds d8 accumulation needs np.in1d (gone in NumPy 2)`; `:1394` says so in a class
docstring; `test_project_io.py:147` repeats it. These are **left alone** — every one of them
passes, and rewriting working tests to exercise a newly available path is its own piece of
work with its own risk. They are recorded here so the next reader knows the comments are
stale rather than believing them, and so the simplification is a known, scoped task rather
than a discovery.

**Suite figures.** 2,889 passed / 2 xfailed / 5 xpassed before; **2,897 passed / 0 xfailed /
0 xpassed** after. The disappearance of every xfail and xpass is the point: nothing in the
pure suite is now passing *while declared to be failing*.

Collection moved 2,896 → 2,897, which is one test appearing rather than a miscount —
`test_architecture.py::test_tooltip_copy_lives_in_help_text` is parametrized over the files
in `modules/`, so `pysheds_compat.py` added a case. It passes.

### §9.3 `KPA-48` and `KPA-53` — the keyline tier uses the field it was handed

**Both closed by one change**, and they turned out to be one defect seen from two sides.

`_ensure_flow_data`'s gate was `if self._fdir_path and self._acc_path:` — an `and` over two
paths when `contour.py:1390` supplies exactly one (`acc_path=acc_path`, from
`self._state.baseline_result["flow_accumulation"]`). The gate therefore never opened, the
supplied raster was discarded, and the tier recomputed its own field from the DEM with
`routing="dinf"` written as a literal.

That literal is what §8.3 filed as `KPA-53`. The fix is **not** to thread the panel's
routing setting down a second path: the raster already handed over came from
`FlowAnalysis.run(routing=panel.routing)`, so **honouring it carries the user's routing
choice in the data**. `KPA-53` closes as a consequence of `KPA-48`, not beside it.

**The gate now keys on accumulation alone**, because accumulation is the only half anybody
reads. `_ensure_flow_data` returns `(fdir_arr, acc_arr)`; `find_keypoint` passes `fdir_arr`
straight into `_trace_thalweg`, which takes it as a parameter and **never touches it** —
that walk goes by elevation and accumulation deliberately, to avoid the non-monotone
accumulation artefacts D-infinity produces on flat ground (its own docstring says so). So
`fdir_arr` may now come back `None`, and the docstring says that too. The cache test had
the same `and` and was fixed with it.

A `routing` parameter is added to `__init__` for the one case with nothing to inherit — a
user who presses Keyline before Baseline — and `contour.py` passes `self._panel.routing`
into it.

**Measured on the fixture**, reproducing `KPA-48`'s own table and extending it:

| | supplied (what production does) | recomputed (the old behaviour) |
|---|---|---|
| time to obtain flow data | **0.0196 s** | 0.5000 s |
| uses the raster it was handed | **yes — 0 cells differ** | no |
| valleys refused | **122** | 126 |
| keypoints found | 1 | 1 |

**25.5x faster, and a different answer.** The two fields differ on **510 cells**, by up to
**65,086** accumulation units — the supplied one is crest-split and the recompute is not,
so the keyline was the only tool in the plugin drawing on a pond-uncorrected accumulation.
Four valleys that the recompute refuses are not refused on the corrected field.

**One figure in `KPA-48` is sharpened rather than confirmed.** Its §2 entry reads *"the
recomputed field differs from the supplied one by up to 65,086 cells"*, which reads as a
count of cells. It is a magnitude: **510 cells differ**, by up to 65,086 accumulation units.
Both numbers are now recorded so neither can be quoted as the other.

**Pinning, and a claim this fix made false.**
`check_keyline_network_numbers_have_not_moved` described itself as running *"the production
path"*. After this fix production takes the other branch, so that sentence had to go rather
than be left to mislead — it now says it pins the **no-baseline** branch, which still
matters because a user can press Keyline before Baseline and get exactly that.

`check_keyline_network_with_a_baseline_has_not_moved` pins what users actually get, with
`EXPECTED_KEYLINE_WITH_BASELINE` = 1 keypoint, 122 skipped, 123 links accounted for. Its
load-bearing assertion is **zero cells differing from the supplied raster**: a count that
drifts is ambiguous, but a reopened gate cannot produce a zero there. Wall time is
deliberately *not* asserted — the 25x speed-up is the reason for the fix, but a timing
assertion on a shared machine is a flake generator, and the zero-differing-cells test
catches the same regression deterministically.

`EXPECTED_KEYLINE` itself does **not** move: that check constructs the analysis without
`acc_path`, so it takes the branch this change leaves alone. Verified, not assumed — 10
checks pass in `checks_fixture_regression` and every recorded integer in it is unchanged.

### §9.4 `KPA-54` — the keypoint label quotes the ground above the keypoint

**Closed.** The label read `"Keypoint at {elevation} m — {catchment_ha} ha above"`, where
`catchment_ha` is the accumulation at the **link's outlet**. A keypoint sits partway up its
link, so on the fixture's rank-1 keypoint that label claimed 5.9 ha over a point commanding
2.1 ha — an overstatement of **177%**, on the map, at the point itself.

`catchment_ha` **keeps its name and its meaning**, because it is the valley ranking basis
(`links.sort(key=_catchment)`) and is what the keypoint layer's attribute of that name has
carried since the layer existed; renaming it would silently change exported data. A second
key, `keypoint_catchment_ha`, carries the ground above the keypoint itself, and the label
now quotes that and names the other:

    Keypoint at 41.2 m — 2.1 ha above (5.9 ha in the valley)

Both numbers are useful and they answer different questions — how much water reaches this
point, and how big the valley this point belongs to is. The defect was never that the
valley figure was computed; it was that it was presented as the other one.

### §9.5 `KPA-52` — sharpened by the `KPA-48` fix, and now an owner decision

`KPA-52` is **not closed**, and the `KPA-48` fix changed what it is rather than removing it.

`find_keypoints` traces links along `flow_graph.d8_from_dem` pointers. Its mask used to come
from a hard-coded D-infinity recompute; it now comes from the baseline's accumulation, whose
routing **the user chooses**. So the mask/pointer mismatch is now a function of a panel
setting, and the cost of each choice is measurable. On the fixture at the production 0.2 ha
threshold:

| | D-infinity mask (the default) | D8 mask |
|---|---|---|
| stream cells | 1,638 | 1,533 |
| order-1 links | 123 | **72** |
| stream cells inside a link | 700 | **1,168** |
| stream cells in **no** link | 938 | **365** |
| pointers leaving the mask | 108 | **35** |
| median link length | 3.41 m | **9.21 m** |
| longest link | 54.0 m | **157.3 m** |
| links clearing the 35 m profile floor | 1 | **8** |
| **keypoints found** | **1** | **5** |
| valleys refused | 122 | **67** |

**The D-infinity default costs four of five keypoints.** Not because D-infinity is a worse
accumulation — it is the more physical of the two — but because it wets a broader, more
diffuse network than the single-successor D8 pointers can trace through, so links end
wherever the two disagree.

**D8 is closer to consistent but is not consistent.** 35 pointers still leave the mask and
365 stream cells still fall in no link, because the mask comes from *pysheds'* D8
accumulation while the pointers come from *TerrainFlow's own* `d8_from_dem` — two D8
implementations whose tie-breaks differ, as `flow_graph.py:106-113` already documents
(pysheds scans N, NE, E, SE, S, SW, W, NW; `d8_from_dem` scans NW, N, NE, W, E, SW, S, SE).

So there are three candidate masks, not two, and Step F measured the third:

| Mask source | order-1 links | clearing the floor | pointers leaving the mask |
|---|---|---|---|
| pysheds D-infinity (default today) | 123 | 1 | 108 |
| pysheds D8 (panel's other option) | 72 | 8 | 35 |
| `d8_from_dem`'s own accumulation | 13 | 7 | **0** |

**This is left open deliberately and is an owner decision**, because every option changes
how many keypoints every assessment produces, and that is exactly the class of change the
project has previously ruled must not be made quietly. The three readings are:

1. **Leave the default.** The tier keeps answering one valley on this terrain.
2. **Recommend D8 in the panel**, making the combo's second entry the one that suits the
   keyline tier. Cheap, reversible, and does not touch the maths — but it also changes the
   baseline raster the user sees, which is a separate question.
3. **Build the mask from `d8_from_dem`'s own accumulation** inside `find_keypoints`, making
   the tier internally consistent whatever the panel says. Zero pointers leave the mask.
   Fewest links, but seven of thirteen are profilable against one of 123.

Nothing here recommends one. What the campaign can say is that option 3 is the only one
where the number of mask-leaving pointers is **zero**, and that "a primary valley" has to
mean something specific before any of the three is defensible.

### §9.6 `KPA-41` — the drift limit is judged on a 50 m window

**Closed**, on the owner's decision of 2026-09-11 to adopt the measurement's recommended
window.

`drift_1_in_n` was net end-to-end fall: `(z_first - z_last) / length`. `over_limit` tested
it against `max_grade_n`, under a docstring promising to flag *"guides whose measured drift
is steeper than 1:N"*. A guide wanders up and down its own length by construction, so a net
figure cannot keep that promise.

`_steepest_drift` measures the steepest sustained fall over any **50 m** of the guide, and
`over_limit` is judged on that. `drift_1_in_n` keeps its name, its value and its layer
attribute — it is still the right answer to *"where does this guide start and finish"* —
and `steepest_1_in_n` is the new figure the limit tests. The panel readout now quotes the
same number it flags on, which it previously did not.

**50 m is a TerrainFlow convention and is labelled as one.** The Yeomans texts were fetched
and read (`MATHS_AUDIT` §9.8) and publish no drift tolerance at all, so there is nothing to
cite. It was *sized* rather than chosen, by the roughness sweep in `p_keypoints.py`: it is
the shortest window whose steepest-grade reading stays inside `MIN_SLOPE_EASE` across
synthetic correlated roughness of 0.00 → 0.10 m, the band `_box_blur`'s docstring puts real
LiDAR noise in. At 10 m and 20 m the reading is 2.18x and 2.44x `MIN_SLOPE_EASE` — those
windows measure the DEM's noise, not the guide's drift.

**Measured on the fixture at the panel's 1:500 default**, six guides on one keypoint:

| Guide | net `1:N` | steepest over 50 m | understatement |
|---|---|---|---|
| valley | *none measurable* | 1:26.3 | unbounded |
| ridge | 1:432.7 | 1:32.8 | 13.2x |
| valley | *none measurable* | 1:22.2 | unbounded |
| ridge | 1:586.6 | 1:20.9 | 28.1x |
| valley | *none measurable* | 1:18.9 | unbounded |
| ridge | 1:544.8 | **1:13.0** | **41.9x** |

**1 of 6 guides was flagged; 6 of 6 are flagged now.** Median understatement 28.1x on the
guides where a net figure existed at all.

The three valley guides are worse than `KPA-41` described. Their net fall is **zero** —
both ends at the same height — so `drift_1_in_n` was `None` and the UI reported no
measurable drift on a guide running at 1:18.9. That is not an understatement, it is a
blind spot: the net measure cannot see a guide that returns to its starting height however
steeply it gets there.

**That every guide now exceeds 1:500 is a result, not a bug in the new measure**, and it is
left standing rather than tuned away. `KPA-41` only ever claimed the flag was broken. What
the corrected flag says about this terrain — that the limit is wrong for it, or that these
guides genuinely drift steeply — is a question for the owner and is not answered here.

**A correction made during implementation, because the first version was wrong.**
`_steepest_drift` initially took "real ground" to mean `isfinite(z)`. That is not the test:
`_sample_dem` returns the **keypoint's own elevation** for any sample off the grid or on a
nodata cell (`KPA-42`), which is a perfectly finite number. So fabricated vertices were
being counted as real.

The effect was the opposite of what was predicted. A fabricated vertex is not *flat* — it
carries a constant lifted from somewhere else on the hill — so pairing one with real ground
at a different height manufactures a fall that is not there. Excluding them made the valley
guides read **gentler**, not steeper: 1:19.3 / 1:8.8 / 1:5.8 became 1:26.3 / 1:22.2 /
1:18.9. The ridge guides did not move at all, so they lie entirely on-grid.

The mask is now recomputed from the geometry against the raster the same way `_sample_dem`
decides it, and a window is measured only when every vertex in it is real ground. The
numbers in the table above are the corrected ones.

### §9.7 Can a keypoint be found without routing? Measured — yes, and it does not settle `KPA-52`

The owner's position, put on 2026-09-11: *a keypoint is a topography-determined point — the
inflection based on the contours, not the flow of water.*

**The position is right, and the texts support it.** Yeomans locates the keypoint on a
contour map and on the ground: the break where a primary valley's steep upper floor eases
into its gentler lower floor. No flow algorithm appears in that definition.

**And the inflection itself is already routing-free.** `keypoint_on_path` resamples the long
profile, smooths it and takes the argmax of the second derivative — pure topography along a
line. Routing never touches that computation. What it touches is **which line the profile is
taken along**, and the word carrying the damage is *primary*: today a primary valley is a
**Strahler order-1 link**, which is a flow concept by construction.

So `p_topographic_valleys.py` was written to ask whether the routing can be *removed* rather
than chosen between. Valley floors come from TPI — `z - mean(z)` over a neighbourhood, a
pure landform measure — classified by `landform_classes`, skeletonised, and **cut at
junctions**, so a leaf branch is a valley nothing else joins from above. That is the exact
landform analogue of order-1, expressed in the graph of the land rather than the graph of
the water. `keypoint_on_path` is then run **unchanged** along each branch, ordered downhill
because `slope_ease` requires it.

**Result: it works, and it is routing-independent by construction** — the extraction takes a
DEM and returns keypoints with no accumulation, no pointers and no cell threshold anywhere
in it. On the fixture: **3 keypoints**, at `(72, 44)`, `(103, 145)`, `(125, 161)`.

Three findings came with it, and the last two are why this does not settle `KPA-52`.

**1. A naive extraction picks up the clip edge.** The first run returned 7 keypoints, of
which **4** sat on rows 1 and 397 or column 1 of a 400x400 grid. `landform_tpi`'s own
docstring predicts this — a neighbourhood mean whose window hangs off the data edge is taken
over fewer cells and drifts, *"fabricating a ridge line all the way round the data
boundary"* — and valleys fabricate identically. A half-window guard removed exactly those 4
and left the 3 interior ones untouched. Any production version needs that guard.

**2. The topographic skeleton is shattered too.** 6,171 valley cells thin to 1,313, cut at
118 junctions into **217 branches — of which 3 exceed 50 m.** That is the same failure mode
as `KPA-52` describes for the flow network, arrived at by a completely different route.
Removing the routing does **not** remove the fragmentation; a valley floor picked out by a
15 m TPI window is broken wherever the floor briefly widens or flattens, just as a channel
is broken wherever two routing schemes disagree. Whatever fixes one has to fix the other.

**3. Three methods, three disjoint answers.** Of the 3 topographic keypoints, **0** are
within 3 cells of a D-infinity keypoint (there is 1) and **0** of a D8 one (there are 5).

That third figure needs stating carefully, because the obvious reading overclaims. The three
methods do not merely disagree about *where the inflection is*; they disagree about **what
the valleys are** — 3 branches over 50 m against 72 and 123 links. So the disjoint keypoints
are at least partly a disagreement about valleys rather than about inflections, and this
probe does not separate the two. Doing so means comparing the *lines*, not the points, and
has not been done.

**What this means for `KPA-52`.** Adopting a topographic extraction is the architecturally
right answer — it makes the question "which routing?" disappear rather than answering it,
and it matches what the method actually is. It is **not** a drop-in fix, and shipping it as
one would replace a known defect with an unknown one:

- It needs the boundary guard, or it invents keypoints on the clip edge.
- It needs an answer to fragmentation, or it finds 3 valleys where the terrain has more.
- It would change every assessment's keypoint set to a third disjoint answer, and nothing
  measured so far says that answer is *right* — only that it is arrived at honestly.

The stability question underneath all three is the one worth naming: `keypoint_on_path`
takes an **argmax**, so it always returns something, and on any given line it returns
exactly one thing. `MIN_SLOPE_EASE` is the only guard against that being noise, and `KPA-39`
measured it refusing **0 of 126** candidates on real ground. A method that gives a different
answer for every plausible valley line, with a prominence test that refuses almost nothing,
is fragile wherever the line comes from.

**Status: `KPA-52` stays open.** The routing-free path is demonstrated and the probe is
committed so the numbers can be re-run, but the decision now has a third option and two new
prerequisites rather than a resolution.

*(Closed later the same day — §10.)*

---

## §10 `KPA-52` closed — a primary valley that starts at the divide, traced on one graph (2026-09-11)

`KPA-52` is **closed**, on six owner decisions taken before and during the work, and the
fix went further than any of §9.5's three options because the sources asked it to. §1 and
§2 stay as written; this section is the authority on what changed.

### §10.1 What the texts say a primary valley is

The owner asked for a fix "based in reality as to how keylines are determined in the real
world", so the two Yeomans texts already in §7 were re-fetched and read for the definition
of the *valley*, which §9.7 had found was the word carrying the damage. Page-cited, from
[YEO-WFEF] unless marked:

- "Valleys form into the side of the main ridge. They are named primary valleys… The primary
  valley is the smallest of the three shapes of land. It is the first valley and the only
  true valley shape in the landscape." (p40)
- "A primary valley head generally starts as a more or less sudden steepening of the side
  slope of a main ridge. Further down, the valley changes to a flatter sloping floor which
  continues more or less uniformly to the stream course below it. The primary valley … does
  not usually have a washed out or channelled water course down the middle of it." (p58)
- "The steepest slopes in the landscape usually occur in the centre of the valley above the
  Keypoint. This first steep slope at the head of the valley is short, then the slope
  changes to a more gradual and longer slope that extends to the creek (or valley junction)
  below." (p40)
- "The primary valley has two slopes; the upper slope is steep and changes to a much flatter
  slope at the Keyline of the valley." (p44) "The Keypoint of the valley is the point of
  change in the two slopes of the primary valley." (p60–61)
- "A primary valley, bounded by the portion of the water divide of the main ridge above it
  and by the water-divides of the primary ridges on either side of it, is the primary, the
  smallest, or the first catchment area." (p61)
- Runoff reaches the floor "by the steepest path and the fastest route" (p45), "at right
  angles to the contours" (p43). "The creek is the lower boundary of its tributary primary
  valleys." (p45)
- [YEO-MKIV]: "On a contour map, the Keypoint is apparent, because the contour lines are
  closer together above it, and further apart below it."

Against the code as it stood, that is three disagreements, not one:

| Yeomans | The code before |
|---|---|
| The valley starts at the divide, above any channel | A link started at the 0.2 ha **channel head** |
| The keypoint is the change between **two slopes**, the short steep upper reach and the long gentle lower one | The short steep reach was above the link and never profiled; the criterion was the sharpest local easing of a smoothed profile |
| The floor is the steepest-descent line — a *line* | The mask was a D-infinity *area partition* walked by D8 pointers on the raw DEM (`KPA-38`, `KPA-52`) |

### §10.2 What changed

**One graph** (`modules/flow_graph.py`, `modules/keypoint_analysis.py`). Steepest-descent
pointers from `d8_from_dem` on the **conditioned** surface — the baseline's
`conditioned_dem` raster when there is one (`YeomansKeylineAnalysis(conditioned_path=…)`,
which `contour.py` now passes, read float64 with the DEM's nodata exactly as
`earthworks._ensure_flow_graph` reads it), else conditioned in place and **kept**
(`_condition_surface`, `_ensure_conditioned`). The stream mask is that graph's **own**
accumulation (new `flow_graph.accumulate`: Kahn's pass vectorised per level, 603 levels in
0.04 s on the fixture, equal cell for cell to the elevation-sorted oracle
`checks_fixture_regression._accumulate`). Mask-leaving pointers are therefore **zero by
construction**, which the regression check now asserts as a property.

**The divide** (`flow_graph.main_stem_to_divide`, `YeomansKeylineAnalysis.primary_valleys`).
Each order-1 link is walked back up its main stem — the inflowing neighbour with the most
accumulation, ties to the first offset scanned as in `d8_from_dem` — to a cell nothing
drains into. On the fixture the extension is 59–168 cells per valley (median 91); profiles
go from 4–157 m to 90–322 m. `stream_links`' `min_cells=3` still applies to the channel
part only, which is the point: the threshold says which valleys exist, not where they start.

**The data edge** (`flow_graph.data_boundary_mask`, `_edge_rule`). A boundary row has no
outside for `d8_from_dem` to route into, so its pointers run *along* the row and fabricate
a channel there — 455 cells on the fixture, and the largest valley had 128 of its 183 m and
its keypoint on that run. A valley is now **cut at the first grid-edge cell below its
divide** ("the creek is the lower boundary"; `flow_graph.LABEL_EXIT`), and a leading run
*along* the edge is trimmed to the divide. A divide on the edge is kept and **flagged**
(`head_on_boundary`, in the label and the panel summary), not refused: the break may still
be on the map. A valley whose channel exists **only** on the boundary row — the row's own
collecting artefact, nothing of it left after the cut but the edge cell — has no channel
on the map and is refused with that reason (`channel_on_map`). On the fixture that is 5
valleys, none of which had a keypoint; on the synthetic harness DEM it is the 2 corner
"valleys" whose channel began on the bottom row.

**The criterion** (`keypoint_on_path_with_reason`; `keypoint_on_path` is now a wrapper).
The keypoint is the break of a **continuous two-slope least-squares fit** to the cell
profile — `z = a + b·s + c·max(0, s − s_k)` at every candidate with at least
`MIN_REACH_CELLS = 3` cells on each side, closed form through suffix sums, one batched 3×3
solve — accepted when `grade_above − grade_below ≥ MIN_SLOPE_EASE` (2 %, value unchanged).
Why it replaced the argmax of a smoothed second derivative is a measurement, in §10.4.
The resampling (`KPA-27`), the Savitzky–Golay window (`KPA-28`), the argmax with its guard
band (`KPA-29`) and the 35 m profile floor (`KPA-40`) are gone with it. `MATHS_AUDIT` §9.10
records the re-expression against the rows.

**Truthful refusals** (`KPA-39`). Every refusal names its guard — too few cells, zero
length, too little finite ground, no two-slope break (quoting both grades), runs off the
DEM, channel begins on the data edge — as class constants a caller can match on.
`p_keypoints` traces the returning line *and* reads the reason string and asserts they
agree: on the fixture 13 refused, 13 by line, 13 by string.

**Ranking and labels** stay on the *supplied* accumulation (owner decision 4): the
baseline's pond-corrected field ranks the valleys and quotes the "N ha" figures, so the list
order and the numbers agree, `acc_path` stays live and the with-baseline check's
zero-differing-cells assertion keeps its meaning. `own_catchment_cells` is carried beside
it: on the fixture the two disagree by more than 10 % at the foot of **20 of 23** valleys,
because D-infinity divides flow at every cell and a D8 count does not. Nothing here is
wrong; the two fields answer different questions and the register now says which is used
for what.

**Not changed.** The 0.2 ha default threshold; `MIN_SLOPE_EASE`'s 2 % value (`KPA-49`);
`find_keypoint`'s single-stem walk and `_trace_thalweg` (the controller's fallback, which
inherits the criterion through the wrapper); `strahler_order` and `stream_links`; the
keypoint dict's keys (`grade_above`, `grade_below`, `channel_cells`, `extension_cells`,
`head_on_boundary`, `runs_off_dem_m` are added); every caller's interface; `panel.py`.

### §10.3 Before and after, on the fixture at the production threshold

| | dinf mask, D8 pointers, raw DEM (§9.5, was production) | D8 mask, same (§9.5) | own-graph mask, raw DEM (§9.5 option 3) | **production now** |
|---|---|---|---|---|
| primary valleys | 123 | 72 | 13 | **23** |
| stream cells | 1,638 | 1,533 | 684 | 1,571 |
| pointers leaving the mask | 108 | 35 | 0 | **0** (asserted) |
| median valley length | 3.4 m | 9.2 m | 39.8 m | **138 m**, divide to foot |
| valleys long enough to fit | 1 | 8 | 7 | 23 |
| cut at the data edge / divide on the edge | — | — | — | 7 / 3 |
| keypoints, uncapped | 1 | 5 | — | **10** |
| refused: before the fit / no two-slope break | 122 (all reported as "no break") | 67 | — | 5 / 8, each with its reason |
| at the panel cap of 8 | 1 keyed, 122 refused | — | — | 8 keyed, 5 refused, 10 unexamined (`KPA-43`) |

**Routing independence, the thing `KPA-52` was about.** With a D-infinity baseline and
with a D8 baseline the keypoint sets are **identical, 10 of 10**, because the valley network
no longer reads the baseline's routing at all — only its accumulation for ranking. Before,
the two shared none.

**Invariance.** Under +600 m the 8 keypoint positions are identical (so the constructor's
float32 cast, owner decision 3, did not need changing and was not). Under a horizontal flip
7 of 8 mirror exactly and one lands one row off, on one of the 31 pointers that are exact
steepness ties and reverse with the scan order; the mirror check does not assert keypoints
and this is recorded here rather than tuned.

**Against §9.7's topographic set.** Re-run with the new criterion the TPI extraction finds 2
keypoints; 0 are within 3 cells of the 10. That comparison is between different *valleys*
and stays where §9.7 left it.

**Cost.** A keyline press with a baseline: 0.14 s (the conditioned surface is read, not
recomputed). Without one: 0.81 s, of which 0.47 s is pysheds conditioning the DEM, as before.

### §10.4 Why the argmax criterion was replaced — measured, not argued

Before approval the plan was reviewed fresh and its assumptions measured on the fixture
with the proposed extraction and the **unchanged** argmax criterion. Two of the numbers
changed the plan:

- With the edge cut, argmax accepted **15** of 23 valleys — but **3** of the 15 sat at or one
  sample inside its own smoothing guard, 10–16 m below the divide on 0.7–3 m of fall, and
  **4** more were on valleys whose grade *increases* downhill (above 0.08–0.16, below
  0.28–0.38 — the "nosed over" ridge p44 says is a ridge shape, not a valley), where the
  argmax had found a kink at the foot. That is §9.7's fragility exposed by longer profiles:
  the sharpest local easing is not the change between two slopes.
- The two-slope fit accepted **9** of the same 23 (10 once the edge rule was finalised),
  every one with grade above greater than grade below, none within 1 m of the divide; **8**
  of its 9 were within 15 cells of the argmax point on the same valleys, and it refused all
  4 convex valleys and the uniform one. On the synthetic test DEMs it lands **exactly** on
  the built-in break rows — 22 on the single-stem valley (argmax: 22), 60 and 60 on the two
  parallel valleys (argmax: 60 and 62) — and refuses the uniform valley with an easing of
  0.000.

The owner chose the replacement over keeping argmax with a two-slope gate (decision 6).
`test_keypoint_on_synthetic_valley`'s ±1-cell bar and
`test_keypoint_on_path_keeps_the_verified_criterion` pass unchanged.

### §10.5 Rows moved

| Row | Was | Now |
|---|---|---|
| `KPA-52` | open, owner decision | **closed** — mask, pointers and accumulation are one graph; the network cannot read the routing setting |
| `KPA-38` | open | **closed** — the raw DEM is no longer walked; the conditioned surface is supplied or kept |
| `KPA-39` | open | **closed** — every refusal names its guard, and the probe checks line against string |
| `KPA-40` | open | **no longer binds** — the 35 m floor went with the filter; the only length rule is `2·MIN_REACH_CELLS + 1` cells |
| `KPA-43` | open | **unchanged and now visible** — at the cap of 8, 10 valleys go unexamined; pinned as a count |
| `KPA-44` | open | **re-established** — on the synthetic DEM 0 of 6 valleys are refused by prominence, not 28 of 28 |
| `FLG-18`, `FLG-19` | open | **superseded on the production path** — kept in the probes as the before picture |
| `KPA-27`, `KPA-28`, `KPA-29` (`MATHS_AUDIT`) | OK | **re-expressed** — see `MATHS_AUDIT` §9.10 |
| `KPA-49` | open | unchanged — 2 % is still a convention, now over two reaches rather than ±guard |
| `KPA-54` | fixed | unchanged in meaning; the keypoint now sits on the extension, so "ha above" is small by construction |

`MIN_REACH_CELLS` is declared a **TerrainFlow convention** under §0.3, like the 50 m window
and `MIN_SLOPE_EASE`.

### §10.6 Left open, on purpose

- **Saddle-headed valleys.** "When the saddle is deep the first steep slope of the primary
  valley may be gone. The Keypoint of such a primary valley is the saddle." (p41) A valley
  with no steep upper reach refuses on prominence; nothing looks for a saddle. Sourced, and
  not handled.
- **A nodata-clipped DEM.** The fixture has no nodata. `data_boundary_mask` flags cells
  beside a hole as well as the outer ring, and the edge cut keys on the grid edge only, so
  a catchment clipped to its divide by nodata will flag every valley and cut none. The
  right behaviour there is unmeasured.
- **Symmetric duplicates.** The synthetic harness DEM's central valley floor is two cells
  wide at exactly equal height, so it yields two parallel primary valleys and two coincident
  keypoints (cols 149 and 150). A synthetic artefact; real ground does not tie like that.
- **The ranking field**, decision 4, is a choice with a measured consequence (20 of 23
  disagree); reversible in one line if the labels prove confusing.
- **Noise.** `p_keypoints`' roughness sweep (synthetic correlated roughness 0.00 → 0.25 m
  on the 2 m harness DEM) now reads: primary valleys 6 → 71 → 68 → 62, and valleys clearing
  the 2 % bar **6 → 68 → 64 → 59**. Before, the 35 m floor refused every noise fragment and
  nothing cleared the bar at any roughness; now every fragment reaches its divide and is
  fitted, and a two-slope fit to a noisy profile usually finds *some* break easing by 2 %.
  Whether those are valleys or noise is exactly §9.7's worry, unmeasured on real ground
  (the fixture refuses 8 of 18 fitted valleys, so it is not accepting everything). The
  natural next guard is a prominence stated against the fit's own residual — a two-slope
  model has to explain the profile *better* than one slope by a margin — which is a
  `MIN_SLOPE_EASE`-class convention and an owner decision, not a quiet tweak.

### §10.7 Suites

Pure suite **2,928 passed** (2,897 before; the new tests cover `accumulate`,
`main_stem_to_divide`, `data_boundary_mask`, the divide extension, both halves of the edge
rule, the channel-on-map refusal, the two-slope fit against a least-squares oracle, the
convex and uniform refusals, nodata handling, and the supplied conditioned surface).
`checks_fixture_regression` re-recorded — `EXPECTED_KEYLINE` and
`EXPECTED_KEYLINE_WITH_BASELINE` — and now asserts zero mask-leaving pointers and the
uncapped `keypoints + skipped == valleys` identity as properties; `checks_contour`'s
keyline check compared every guide against the *first* keyline's elevation and was fixed to
compare per valley, which the synthetic DEM's six keypoints exposed. Probe evidence re-run
and committed for `p_flow_graph`, `p_keypoints`, `p_crosscheck`, `p_topographic_valleys`,
`p_invariance`, `p_battery`, `p_controllers`.
