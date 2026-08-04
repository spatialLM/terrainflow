# TerrainFlow — Maths Correctness Audit

Audit date: 2026-07-30. Scope: every formula and numerical method in
`terrainflow_assessment/modules/` (17 files), `terrainflow_assessment/core/sizing/`
(primitives, advisories), and the numeric defaults in
`terrainflow_assessment/core/registry/earthwork_types.py`. The `qgis/` layer is
excluded (separate pass). **Tests were never used as evidence** — every verdict rests
on a published source actually read during this audit, an internal step-by-step
derivation (Appendix §6), or a code re-read; where none of those was possible the
verdict is honestly UNVERIFIED.

## §0 Method, status, and how to read this document

**Method.** A four-agent inventory pass enumerated ~400 formula sites (master
checklist IDs below). Verification was then fanned out to 13 specialised verifiers:
nine web-grounded groups (SCS-CN, TR-55 Tc, rational+IDF, Lancaster+Yeomans,
open-channel, spillway/dam, volumes, terrain derivatives, flow routing), one pure
algebra prover, two internal-consistency auditors, and a unit/basis sweeper.

**⚠ Verification status: PARTIAL.** Mid-run, the account hit its monthly spend
limit and 10 of the 13 verifiers were terminated before returning results. What
completed with full grounding:

- **A1 (algebra)** — all 23 derivation checks completed (Appendix §6).
- **W5 (open-channel hydraulics)** — all 19 checks completed against fetched
  TR-55 / HEC-15 / FAO / Nashville-SWMM sources (§7 source register).
- **W8 (terrain derivatives)** — all 16 checks completed against Horn 1981,
  GDAL/ESRI docs, numpy/scipy/skimage official docs, TauDEM/pysheds, Jenness-TPI;
  found 2 INCORRECT + 9 DISCREPANCY + 5 NEW sites.
- **Local TR-55 salvage pass** — the assembling session extracted verbatim text
  from the TR-55 PDF already downloaded by the verifiers, resolving: eq. 3-3
  (0.007 form, quoted), the 300-ft sheet statement, Figure 3-1 velocity equations
  (unpaved 16.1345·s^0.5 / paved 20.3282·s^0.5 ft/s — assignment matches code),
  Tc "minimum, 0.1; maximum, 10.0" hr, the full Table 3-1 n-list (all ten rows
  match the code exactly), the bankfull-velocity sentence, Ia = 0.2S, and the
  Table 2-2 pasture CN row that explains the code's texture→CN table (§1 #11).
- All ~150 MAIN-tier rows (trivial arithmetic, display, design constants) were
  audited by the main session against code excerpts read this session.

Rows owned by terminated verifiers carry verdict **UNVER-B** ("unverified —
verification blocked") with the intended source named, so the audit can be resumed
row-by-row once the limit is lifted. §5 is the full blocked/unverified register.
**An UNVER-B verdict is not an all-clear** — several UNVER-B rows are flagged
failure candidates (listed in §4) whose confirmation is pending.

**2026-07-31 completion pass (in-loop):** the high-stakes remainder was finished
from the main session: EWD-35 (weir C — resolved DISC via Brater & King), EWD-05
(freeboard vs NRCS-378), the IDF-07 mean-vs-peak adjudication (code correct),
CAT-18 AMC constants, CAT-13/15 Lancaster table, and the priority code-only
candidates (CAT-27, SIM-27, SWL-17 confirmed; SWL-21 and the dead primitives
confirmed retired; INC-01/07 duplication checks closed). Remaining UNVER-B rows
are lower-stakes corroborations (pysheds semantics, Yeomans/FAO texts, HIRDS,
registry defaults) — see §5.

**2026-07-31 remediation pass.** The findings below have been **fixed in code** —
see §9 for the per-ID register, the one finding withdrawn on re-check (SWL-17), and
the three items raised during remediation that the audit had not caught. Rows in §1
and §2 are left as written so the audit stays a record of what was found; §9 is the
authority on current state.

**Verdict tokens:** OK (correct) · WRONG (incorrect) · DISC (differs from
source/intent — explained; deliberate divergence is still DISC) · UNVER
(no locatable published source) · UNVER-B (verification blocked mid-audit) ·
HAZ (correct today, silently breaks under a stated condition).
**Direction:** U = can UNDER-size (dangerous: breached earthwork), O = OVER-size
(wasted excavation), n = neutral, i-d = input-dependent.
**Severity:** C critical · H high · M medium · L low · I info. Per the audit brief,
anything that can under-size in a production sizing path ranks ≥ H.
**Confidence:** H/M/L.

### §0.1 Second-look rule
Every WRONG/DISC verdict published in §1 from a completed verifier was
cross-checked by the assembling session against the verifier's quoted code and, for
the two HIGH findings, against a fresh read of the cited lines. Findings from
*terminated* verifiers appear only in §4 as unconfirmed candidates, never in §1.

---

## §1 Findings summary (severity-ranked) — the triage surface

Confirmed findings only (grounded in a fetched source or a full derivation).
Unconfirmed candidates are in §4.

| # | ID | Sev | Dir | Where | Finding | Verdict | Conf |
|---|----|-----|-----|-------|---------|---------|------|
| 1 | EWD-32 | H | U | earthwork_design.py:888-899 | Legacy diversion hydraulic radius: area from a z=2 section but wetted perimeter with a √2 (z=1) slant → R over-stated ×1.40, Manning Q over-stated ~+25% (w=1 m, d=0.3 m; worse deeper/narrower) | DISC (deliberate, comment-acknowledged) | H |
| 2 | NEW-W5-01 | M | U | earthwork_design.py:490 vs earthwork_properties_dialog.py:803 | `Earthwork.summary()` omits `bottom_width` → always the legacy path: feature-list label shows ~+45% more discharge than the dialog for the same stored diversion | DISC | H |
| 3 | EWD-26 | M | U(cut) | earthwork_design.py:805-814 | Diversion cut section: comment says "1:1" but builds z=2; declared bed width lands at mid-depth → cut volume under-stated by d² per metre (~23% at w=1, d=0.3) | DISC | H |
| 4 | ALG-06 / EWD-19 | M | O(storage claim) | earthwork_design.py:684-687 vs :841-848 | Companion-berm capacity credit h·T/2 exceeds the berm's own h² fill section by T/(2h) (×1.33 at registry defaults) → ~+14% claimed swale capacity vs the fill-consistent model; non-conservative storage claim | DISC | H (algebra) / M (intent) |
| 5 | CAT-17 / PRM-19 / ALG-11 | M | U(drain hrs) | catchment.py:349-368; primitives.py:317-321 | Drain-down assumes constant (full) wetted area; true drain time strictly exceeds the estimate for battered sections → 24/48-h puddling warnings under-trigger | DISC (simplification, definite direction) | H |
| 6 | SWL-04 | L | U(lean) | swale_design.py:98-100 | Infiltration credited over the TOP width for the full storm (vs part-full wetted width) → soakage over-credited ×T/b (~×1.67 typical); small in absolute terms (~4% of capacity at NZ-loam rates); 0.8 area-fraction freeboard is nonstandard (published practice is depth-based ~20%) but conservative | DISC | M |
| 7 | SWL-03 | L | U(edge) | swale_design.py:95-96 | When 2zd > T the bottom clamps to 0 but depth is kept → triangle section exceeds the true max at that batter → over-stated storage in the converged-walls corner | edge-case | H |
| 8 | NEW-W5-02 | I | O | time_of_concentration.py:25-27 | 100-ft sheet cap attributed to "TR-55", but TR-55 (1986, p.3-3) says 300 ft — the 100 ft figure is the later NEH-630 Ch.15 revision. Behaviour is current-guidance and conservative; attribution wrong | DISC (attribution) | H |
| 9 | ALG-05 note | I | i-d | earthwork_design.py:918-919 | Spillway width returns 0.0 for zero/negative head — callers must treat 0 as "invalid input", not as a designed width | polarity note | H |
| 10 | ADV-02 | L | n | advisories.py:32-38 | Max non-eroding grade %-by-texture matches no published table (published limits are velocity/shear-based); ordering consistent with FAO velocity tables; advisory-only | UNVER | M |
| 11 | CAT-19 / ADV-03 | M | U(SCS basis) | catchment.py:387-394; advisories.py:42-48 | Texture→CN table (39/49/61/74/80) is TR-55 Table 2-2 "Pasture, grassland — **Good** hydrologic condition" across HSG A–D with texture standing in for soil group (49 = the Fair/A value, apparently interpolated). Bakes in Good condition: Poor pasture on the same soils is CN 68-89 → SCS runoff under-read on degraded/heavily-grazed ground | DISC | H |
| 12 | DEM-05 | M | i-d | dem_loader.py:187-200 | Slope raster declares nodata −9999 but never writes it — former nodata regions render as valid slope (~0° interior, near-90° boundary ring), indistinguishable from real data | WRONG | H |
| 13 | KPA-14 | M | U(ridge detect) | keypoint_analysis.py:246-255 | "Thinning" is 3× binary erosion, not skeletonisation — ridges ≤6 cells wide are silently annihilated, ends retreat ~3 cells; surviving remnant is not a centreline. Whole ridge features can vanish or mislocate | WRONG | H |
| 14 | NEW-W8-01 | M | i-d | keypoint_analysis.py:526-529 | Thalweg profile substitutes 0.0 m for NaN cells — one nodata cell injects a full-terrain-height cliff; after smoothing, argmax d²E/ds² can lock the Yeomans keypoint onto the artefact instead of the true inflection | DISC | H |
| 15 | DEM-04 | M | i-d | dem_loader.py:132-159 | Horn stencil itself exact (matches Horn 1981/ESRI verbatim), but NaN→0.0 m before differencing fabricates near-90° slopes in the 1-cell ring beside nodata (GDAL/ESRI emit nodata there); edge replication ~halves border-row slope | DISC | H |
| 16 | CTA-07 / KPA-32 | M | i-d | contour_analysis.py:162; keypoint_analysis.py:653-674 | Marching-squares nodata pre-fill (elev−1 / elev−1e6) closes contours along the data boundary — skimage natively masks NaN and leaves them open. Fabricated edge-hugging contours are selectable for swale alignment; keyline can snap to a boundary artefact (nearest-component test also mixes row/col cell units) | DISC | H |
| 17 | FLL-04/08/09/15 | M | i-d | flow_lines.py:57-58,81-90,139-149 | np.gradient without spacing → gradients in m-per-CELL (per numpy docs), min_grad threshold scale-dependent; NaN→whole-DEM-minimum creates "gradient walls" that drag flow lines into nodata boundaries (viz layer only, but wrong exactly where users clip to a property boundary) | DISC | H |
| 18 | KPA-12 | L | i-d | keypoint_analysis.py:207-233 | TPI formula matches Weiss/Jenness exactly, but NaN→whole-DEM-mean contaminates the neighbourhood mean within 7 cells of nodata (fabricates boundary ridges), and the fixed 15-CELL window makes landform scale resolution-dependent | DISC | H |
| 19 | TOC-09 note | I | U(if Tc>10 h) | time_of_concentration.py:83-85 | TR-55 App. F states Tc validity "minimum, 0.1; **maximum, 10.0**" hr — the code enforces the floor but not the cap; a computed Tc above 10 h would sit outside the method's calibration (long-Tc → low intensity is the under-sizing direction) | DISC (partial) | H |
| 20 | KPA-21 | M | i-d | keypoint_analysis.py:372-375 | **CONFIRMED BUG** (code re-read): `for dc in range(-200, 201)` + `break` on `nc < 0` — first iteration is dc=−200, so any candidate within 200 columns of the west edge exits immediately → valley width 0 m → dam-site score acc/(width+1) maximised → pond-site recommendations biased to the raster's west edge. Fix: `continue` (or clamp the scan range) | WRONG | H |
| 21 | RPT-07 | M | n (fails loud) | reporting.py:186 vs water_balance.py:142-160 | **CONFIRMED BUG** (both sides re-read): live-assessment row does subscript `f['inflow_m3']`, but water_balance emits only `direct_/upstream_/total_inflow_m3` → KeyError at render whenever flow data is present. A crash, not a silent wrong number | WRONG | H |
| 22 | SIM-26 | M | i-d | simulation.py:677-686 | **CONFIRMED** (runtime check): shapely LineString has `.area == 0.0`, so `hasattr(...,"area")` is always True and the `length×width` fallback is unreachable → line-drawn features (swales are polylines) get area_m2=0 → zero infiltration credited and drain_hours=None in the simulation path | WRONG | H |
| 23 | EWD-35 | H | U | earthwork_design.py:910-920 | **RESOLVED (was the top open item):** Brater & King (1976) broad-crested table [BK-MODOT]: for crest breadth ≥ 0.6 m at the code's own 0.20-0.50 m head band, C = 2.60-2.70 English = **1.44-1.49 SI**. The code's C=1.7 (≈3.08 English) occurs only at high head on short crests (approaching sharp-crested). Multiple secondary sources call 1.7 the metric *upper limit* (ideal ≈1.705, ALG-05/A1). Using 1.7 for a wide earthen dam-crest spillway under-sizes width by ≈1.7/1.46 ≈ **16%** | DISC | H |
| 24 | EWD-05 | H | U | earthwork_design.py:184 | **RESOLVED:** NRCS CPS-378 Pond requires "a minimum of 1.0 feet [0.30 m] of freeboard between design high-water-flow elevation in the auxiliary spillway and the top of the settled embankment" [NRCS378]; the code's SPILLWAY_MIN_FREEBOARD_M = 0.15 m is **half** the NRCS minimum | DISC | M |
| 25 | CAT-27 | M | U(sim) | catchment.py:521-523 | **CONFIRMED** (re-read): `is_cumulative = all(non-decreasing)` misclassifies any monotone incremental hyetograph — a constant-rate storm (5,5,5,5 mm) reads as cumulative → 50 mm becomes 5 mm → simulation runoff collapses. Constant-rate CSVs are the most natural user input | WRONG | H |
| 26 | SIM-27 | M | i-d | simulation.py:696-705 | **CONFIRMED** (re-read): centroid elevation read with no nodata check (and default 0.0 on any exception) — a −9999/0.0 elevation sorts below every real store in the elevation-ordered cascade → that store becomes the universal overflow receiver and SIM-17 treats all others as its upstream | WRONG | H |
| 27 | SWL-17 | M | U(warning) | swale_design.py:259 + qgis/controllers/earthworks.py:1755 | **CONFIRMED**: the only production caller passes `freeboard=1.0` while the sizing paths use 0.8 → overtopping checked against 25% more capacity than sizing credits → overtopping under-warned | DISC | H |
| 28 | KPA-33 | L | i-d | keypoint_analysis.py:676-692 | **CONFIRMED by derivation** (map-space trace): the fallback keyline direction (−dz_dc, +dz_dr) equals **−∇z in map space** — the downhill axis, not the contour perpendicular. On a south-facing slope the fallback keyline draws N-S instead of E-W (90° wrong). Fallback path only (fires when find_contours fails); its sibling `_offset_line` applies the row→y flip correctly | WRONG (fallback) | M |
| 29 | IDF-07 / PKF-05 | I | n | rainfall_idf.py:108; peak_flow.py:80-85 | **ADJUDICATED:** standard rational-method practice is average intensity for duration = Tc read from an IDF curve ([TXDOT] + corroborating sources) — exactly what the code pipeline does. The `rational_peak_flow` docstring's "not the storm depth divided by its duration" is correct only as a warning against whole-storm averaging (120 mm/24 h); as written it contradicts the correct pipeline. Code OK; docstring wording misleading | OK (code) / DISC (docstring) | H |

Notable **confirmed-correct** high-stakes items: the trapezoid element set and SI
Manning form (PRM-01..04, PRM-08/09 — verified against HEC-15/TR-55/Nashville
worked examples); the four preset hydraulic radii (recomputed independently to 3 dp);
the bankfull-velocity convention (verbatim TR-55 procedure, conservative direction);
the SCS marginal dQ/dP derivative including all three docstring numerics; the
prismoidal pond path (exact for linear batters); the battered-basin linear-inset
model (under-states storage by (4/3)z²d³ — conservative, and its code comment's
80.0-vs-81.33 example is exact); the rational-method 3.6×10⁶ unit constant; the
metric SCS constant 25400/254; the metric sheet-flow coefficient 0.091265 ≈ 0.0913;
marginal ≥ average C (proven convexity → spillway-conservative); the weighted
circular-mean bearing; the quadratic swale-depth inversion including root choice
and reachability condition.

---

## §2 Per-module audit tables

Columns: ID | computes | line(s) | source | dimensional check | verdict | dir | sev | conf.
Source keys in §7. [CODE] = code re-read this session. [ALG] = Appendix §6 derivation.
Blocked rows name the intended source as "→ source".

### 2.1 flow_analysis.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| FLA-01 | pit-fill → breach (silent fallback to fill) → resolve flats | 78-88 | → pysheds docs | elevation m | UNVER-B | — | — | — |
| FLA-02 | flowdir+accumulation; silent routing downgrade on TypeError | 90-99 | → pysheds docs | fdir rad/codes; acc cells | UNVER-B | — | — | — |
| FLA-03 | weighted accumulation (m³ weights) | 107-122 | → pysheds docs | m³ | UNVER-B | — | — | — |
| FLA-04 | stream threshold 1000 cells | 138-142 | design constant | cells (cell-size dependent) | UNVER | n | I | H |
| FLA-05 | runoff volume = acc × runoff_m × cell_area | 144-151 | → pysheds self-inclusion semantics | cells·m·m²=m³ ✓ | UNVER-B | — | — | — |
| FLA-06 | duration hr→s ×3600 | 190 | [SI] | hr·3600=s ✓ | OK | n | I | H |
| FLA-07 | cell w/h from \|a\|,\|e\|; area w·h; mm→m | 211-214 | [SI]/[CODE] | m² ; mm/1000=m ✓ | OK | n | I | H |
| FLA-08 | event m³ → event-average L/s (×1000/dur_s) | 216-221 | [SI] | m³·1000/s = L/s ✓ | OK | n | I | H |
| FLA-09 | 3×3 max-pool to recover dinf-split channels | 222-225 | → Tarboton 1997 | L/s | UNVER-B | — | — | — |
| FLA-10 | cell-centre (col+0.5)·a, (row+0.5)·e | 230-233 | [CODE] GDAL centre convention | map units | OK | n | I | H |
| FLA-11 | exit clustering radius 20 cells | 236-250 | design constant | map units | UNVER | n | I | H |
| FLA-12 | flow-weighted exit centroid | 251-258 | [CODE] | L/s-weighted mean of coords | OK | n | I | H |
| FLA-13 | pooled peak L/s → m³ (×dur/1000) | 259-260 | [SI] | ✓; note basis = POOLED max, not cell volume | OK | n | I | H |
| FLA-14 | exit label rounding/format | 261-274 | [CODE] | display | OK | n | I | H |
| FLA-15 | fallback outlet = global acc max; mixed centre-offset conventions | 300-307 | → convention check (blocked) | map units | UNVER-B | — | — | — |
| FLA-16 | inverse affine sampling, int() truncation | 309-314 | [CODE] | floor for x≥origin only | OK | n | I | H |
| FLA-17 | non-overlapping catchment partition, claim order ascending acc | 335-338 | [CODE] | boolean | OK | n | I | H |
| FLA-18 | m²→ha ÷10 000 | 351-353 | [SI] | ✓ | OK | n | I | H |
| FLA-19 | dinf radians→compass (90−deg)%360; nan→east | 357-359 | [TAUDEM] "angle in radians counter-clockwise from east"; W8 test angles 0→90° E, π/2→0° N, π→270° W ✓; NaN→east bias where weighted | deg ✓ | OK (formula) | n | L | H |
| FLA-20 | ESRI D8 code→bearing map; unknown→north | 360-365 | [PYSHEDS] dirmap (64,128,1,2,4,8,16,32)=(N,NE,E,SE,S,SW,W,NW) — all 8 pairs match; unknown→0° N bias where weighted | deg | OK (map) | n | L | H |
| FLA-21 | acc-weighted circular mean bearing | 366-371 | [ALG] FLA-21 | deg ✓ | OK | n | I | H |
| FLA-22 | catchment rounding | 375-388 | [CODE] | display | OK | n | I | H |
| FLA-23 | boundary outlets, min_acc 500, 20-cell spacing | 405-421 | design constants | cells/map units | UNVER | n | I | H |
| FLA-24 | fdir unit declaration (dinf rad CCW-east; D8 codes) | 425-429 | → Tarboton/pysheds | — | UNVER-B | — | — | — |
| FLA-25 | float32 write of m³ rasters | 440-446 | [CODE] | ~7 sig figs: ≥10⁷ m³ → ~1 m³ resolution | HAZ | i-d | L | H |

### 2.2 catchment.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| CAT-01 | preview budget 500k cells | 23 | design constant | cells | UNVER | n | I | H |
| CAT-02 | extent ha from full rectangle, pre-mask | 78-82 | [CODE] | cells→m²→ha ✓; denominator bias for CAT-10 | HAZ | i-d | L | H |
| CAT-03 | preview downsample √scale, bilinear elevations | 87-94 | → resampling check (blocked) | m | UNVER-B | — | — | — |
| CAT-04 | nodata −9999 | 147 | [CODE] | sentinel | OK | n | I | H |
| CAT-05 | conditioning + dinf (docstring says D8) | 156-176 | → pysheds; doc mismatch → §4 | — | UNVER-B | — | — | — |
| CAT-06 | pour point = boundary argmax acc | 178-187 | [CODE] | cells | OK | n | I | H |
| CAT-07 | catchment ∪ site mask | 190-199 | → pysheds catchment() | boolean | UNVER-B | — | — | — |
| CAT-08 | polygon area ÷10 000 from downsampled grid | 215-216 | [SI]/[CODE] | ✓; carries resample error | OK | n | I | H |
| CAT-09 | clip bbox +20 %/axis | 218-221 | design constant | m | UNVER | n | I | H |
| CAT-10 | coverage % downsampled-poly ÷ full-extent | 223 | [CODE] | mixed bases (see CAT-02) | HAZ | i-d | L | H |
| CAT-11 | rounding | 226-233 | [CODE] | display | OK | n | I | H |
| CAT-12 | clip nodata −9999 | 241-252 | [CODE] | sentinel | OK | n | I | H |
| CAT-13 | Lancaster coefficient table | 269-291 | [LANC] fetched: Grass/Lawn 5-35 ✓, Bare Earth 20-75 ✓, Concrete/Asphalt 80-95 ✓, Metal 95 ✓; code's "Healthy indigenous landscape 20-70" is Lancaster's **"Sonoran Desert Uplands 20-70"** relabelled for NZ (transferability caveat); code's "typical" midpoints are project-chosen, not on the page | dimensionless | OK (ranges) / DISC (relabel+typicals) | i-d | L | H |
| CAT-14 | default C 0.50 ("design", not Lancaster — own comment) | 297 | project decision (STRETCH_GOALS §ref) | dimensionless | UNVER | i-d | M | H |
| CAT-15 | runoff depth = P × C, clamp 0-1 | 300-309 | [LANC] page's model (coefficient × rainfall on the catchment) — the water-harvesting form of the volumetric rational method | mm·(–)=mm ✓ | OK | n | I | H |
| CAT-16 | SCS marginal dQ/dP = u(u+2S)/(u+S)² | 312-346 | [ALG] ALG-02 (derivative exact; docstring numerics reproduce) | dimensionless ✓ | OK (algebra) | n | I | H |
| CAT-17 | drain-down V/(rate·area), constant wetted area | 349-368 | [ALG] ALG-11 | m³/((m/hr)·m²)=hr ✓ | DISC | U(hrs) | M | H |
| CAT-18 | AMC dry/wet CN conversions | 381-385 | Constants match the Chow et al. (1988) family (corroborated [TXDOT-era secondary]); alternative Hawkins-1985 family fetched verbatim [PONCE] agrees within ≤~1 CN over the valid range, bounding any error at negligible | dimensionless ✓ | OK | n | I | M |
| CAT-19 | soil-texture→CN table 39/49/61/74/80 | 387-394 | [TR55] Table 2-2 (local pass): "Pasture, grassland… Good 39 61 74 80" across HSG A-D; texture used as HSG proxy; 49 = Fair/A (interpolated); Good condition baked in → §1 #11 | dimensionless | DISC | U | M | H |
| CAT-20 | storm presets (depth, 1 hr) labelled mm/hr | 396-402 | [CODE] | label = rate only at 1 h | HAZ | i-d | L | H |
| CAT-21 | adjusted-CN clamp [1,100] | 404-407 | [CODE] | dimensionless | OK | n | I | H |
| CAT-22 | SCS Q=(P−Ia)²/(P−Ia+S), S=25400/CN−254, Ia=0.2S | 409-426 | [TR55] eq. 2-1..2-4 + [ALG] ALG-01 | mm ✓ | OK | n | I | H |
| CAT-23 | runoff ratio Q/P | 428-432 | [CODE] arithmetic on CAT-22 | dimensionless | OK | n | I | H |
| CAT-24 | catchment volume mm→m³ | 434-436 | [SI] | (mm/1000)·m²=m³ ✓ | OK | n | I | H |
| CAT-25 | CN raster burn (zone cn=0 = fill; AMC after burn) | 438-468 | [CODE]; hazard → §4 | CN | HAZ | i-d | L | H |
| CAT-26 | inline duplicate of SCS Q per cell | 470-480 | [TR55] + [ALG] ALG-01; duplication → §4 INC-01 | mm ✓ | OK | n | I | H |
| CAT-27 | hyetograph cumulative-vs-incremental auto-detect | 519-537 | [CODE] re-read — **CONFIRMED** false-positive on monotone incremental storms → §1 #25 | mm/min | WRONG | U(sim) | M | H |

### 2.3 water_balance.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| WBL-01 | model + declared field units | 15-74 | [CODE] | m³/%/hr declared | OK | n | I | H |
| WBL-02 | single-shot cascade dt = duration | 106-112 | infiltration-model structure → §4 | hr | UNVER-B | — | — | — |
| WBL-03 | captured = stored+infiltration; exit = uncaptured+routed | 114-117 | mass-balance trace → §4 | m³ | UNVER-B | — | — | — |
| WBL-04 | capture % deliberately unclamped | 119-122 | pairs with RPT-17 → §4 INC-02 | % | UNVER-B | — | — | — |
| WBL-05 | mass-balance tol max(1 m³, 1 %) | 125-130 | design constant | m³ | UNVER | n | I | H |
| WBL-06 | fill % | 135 | [CODE] | % | OK | n | I | H |
| WBL-07 | upstream = total − direct, floored 0 | 137-148 | [CODE] | m³ | OK | n | I | H |
| WBL-08 | terminal deficit Σ | 139-141 | [CODE] | m³ | OK | n | I | H |
| WBL-09 | drain-down call basis (footprint area; fill-only → None) | 153-154 | → §4 | m² | UNVER-B | — | — | — |
| WBL-10 | fill % display clamp vs raw | 157 | → §4 | % | UNVER-B | — | — | — |
| WBL-11 | site totals | 164-181 | [CODE] | m³ | OK | n | I | H |
| WBL-12 | area subtotals cells→m³; bucket partition claim | 210-237 | [CODE]; partition caveat → §3 | n·m²·(mm/1000)=m³ ✓ | OK w/ note | n | L | H |

### 2.4 simulation.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| SIM-01 | default infiltration 4.0 mm/hr (Loam) | 40 | → FAO/NRCS texture tables (blocked) | mm/hr | UNVER-B | — | — | — |
| SIM-02 | infiltration = rate×area×dt, no saturation | 75-90 | → Lancaster (blocked); mm/hr→m/hr ✓ | (m/hr)·m²·hr=m³ ✓ | UNVER-B | — | — | — |
| SIM-03 | elevation-only downslope receiver | 97-127 | legacy path → §4 | m | UNVER-B | — | — | — |
| SIM-04 | cascade order topo-else-elevation | 260-266 | [CODE] | — | OK | n | I | H |
| SIM-05 | infiltration min(potential, available) | 271-274 | accumulator naming → §4 | m³ | UNVER-B | — | — | — |
| SIM-06 | infiltration before capacity clamp | 276-280 | ordering → §4 | m³ | UNVER-B | — | — | — |
| SIM-07 | overflow = stored − capacity, routed | 282-300 | [CODE] | m³ | OK | n | I | H |
| SIM-08 | peak fill % post-clamp | 302-304 | [CODE] | % ≤100 | OK | n | I | H |
| SIM-09 | rank layout | 330-347 | [CODE] | integer | OK | n | I | H |
| SIM-10 | cell area; nodata −9999; float32 | 393-399 | [CODE] | m² | OK | n | I | H |
| SIM-11 | uniform CN raster | 419-421 | [CODE] | CN | OK | n | I | H |
| SIM-12 | min→hr ÷60 | 431-452 | [SI] | ✓ | OK | n | I | H |
| SIM-13 | incremental SCS runoff by differencing cumulative Q | 457-464 | → NEH-630 temporal distribution (blocked); code documents "no travel time" | mm→m³ ✓ | UNVER-B | — | — | — |
| SIM-14 | float32 downcast before float64 accumulation | 466-479 | [CODE] | precision loss point | HAZ | i-d | L | H |
| SIM-15 | outflow proxy = domain-wide max (comment says boundary) | 481-486 | comment/code mismatch → §4 | m³→L/s ✓ | UNVER-B | — | — | — |
| SIM-16 | step runoff Σ | 490 | [CODE] | m³ | OK | n | I | H |
| SIM-17 | centroid-cell inflow; elevation-based upstream subtraction; name-keyed | 492-529 | flagged → §4 | m³ | UNVER-B | — | — | — |
| SIM-18 | cascade without routing= (legacy heuristic governs simulation) | 531-540 | BY-DESIGN per STRETCH_GOALS §7 | m³ | DISC (by design) | i-d | L | H |
| SIM-19 | baseline outflow instant-exit | 547 | [CODE] | m³→L/s ✓ | OK | n | I | H |
| SIM-20 | row rounding | 550-562 | [CODE] | display | OK | n | I | H |
| SIM-21 | negatives→nodata on write | 567-574 | [CODE] | m³ | OK | n | I | H |
| SIM-22 | overflow-event tolerance 1e-6 hr | 577-588 | [CODE] | hr | OK | n | I | H |
| SIM-23 | "peak" raster = per-step volume, not a rate | 593-604 | [CODE] | m³/step; cross-step comparable only at constant dt | HAZ | i-d | L | H |
| SIM-24 | 0.0 hr falsy → first_overflow None | 607-624 | flagged → §4 | hr | UNVER-B | — | — | — |
| SIM-25 | site totals rounding | 636-638 | [CODE] | display | OK | n | I | H |
| SIM-26 | LineString .area==0 → unreachable length×width branch → area 0 | 677-686 | [CODE] + runtime shapely check — **CONFIRMED** → §1 #22 | m² | WRONG | i-d | M | H |
| SIM-27 | centroid elevation, no nodata check | 692-705 | [CODE] re-read — **CONFIRMED** (default 0.0 on exception too) → §1 #26 | m | WRONG | i-d | M | H |
| SIM-28 | cut/fill delegation | 707-709 | [CODE] | m³ | OK | n | I | H |
| SIM-29 | infiltration zeroed for fill-only (has_cut); exception→True | 711-728 | flagged → §4 | mm/hr | UNVER-B | — | — | — |

### 2.5 reporting.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| RPT-01 | container defaults (cn 70, cell 1.0, 1 hr) | 31-82 | design defaults | declared | UNVER | n | I | H |
| RPT-02 | ponding Σdepth·cell_area, min_depth 1 mm | 89-94 | [SI]/[CODE] | m·m²=m³ ✓ | OK | n | I | H |
| RPT-03 | impounded = Σmax(dammed−baseline,0)·cell_area | 97-108 | [CODE] | m³ ✓ | OK | n | I | H |
| RPT-04 | traffic lights 80/40 % | 115-126 | display policy | % | UNVER | n | I | H |
| RPT-05 | mini-bar clamp/int | 129-146 | [CODE] | % | OK | n | I | H |
| RPT-06 | stored = captured − infiltration back-calc | 169 | inversion check → §4 | m³ | UNVER-B | — | — | — |
| RPT-07 | reads f['inflow_m3'] vs emitted key names | 185-194 | [CODE] both sides re-read — **CONFIRMED KeyError** → §1 #21 | m³ | WRONG | n | M | H |
| RPT-08 | region attribution: thresholded mask, raw-value sum | 240-266 | divergence from RPT-02 → §4 | m³ | UNVER-B | — | — | — |
| RPT-09 | resolution caveat 0.10 | 269-271 | display policy | fraction | UNVER | n | I | H |
| RPT-10 | Δ vs rasterisable; 0.0 falsy fallback | 302-314 | flagged → §4 | m³/% | UNVER-B | — | — | — |
| RPT-11 | routing_only min_dim<cell | 320 | [CODE] | m | OK | n | I | H |
| RPT-12 | per-feature Δ % | 322-331 | [CODE] | % | OK | n | I | H |
| RPT-13 | resolution flag abs/signed | 333-336 | [CODE] | % | OK | n | I | H |
| RPT-14 | design→geometric→rasterisable→terrain chain | 338-351 | [CODE]; matches STRETCH_GOALS §1 | m³ | OK | n | I | H |
| RPT-15 | caveats; only first 3 features named | 353-368 | [CODE] | display | OK | n | I | M |
| RPT-16 | unattributed_m3 hard-wired 0.0 | 375 | dead feature → §4 | m³ | UNVER-B | — | — | — |
| RPT-17 | captured % CLAMPED [0,100] | 401-404 | contradiction with WBL-04 → §4 INC-02 | % | UNVER-B | — | — | — |
| RPT-18 | exit reduction floored at 0 (masks worsening) | 406-410 | flagged → §4 | % | UNVER-B | — | — | — |
| RPT-19 | peak reduction floored at 0 | 412-416 | flagged → §4 | % | UNVER-B | — | — | — |
| RPT-20 | peak delay floored at 0 | 418 | flagged → §4 | hr | UNVER-B | — | — | — |
| RPT-21 | net cut/fill, no bulking factor | 420-426 | → shrink/swell tables (blocked) | m³ | UNVER-B | — | — | — |
| RPT-22 | display thresholds ±25 %, 0.05 m³ | 608-614 | display policy | %/m³ | UNVER | n | I | H |
| RPT-23 | chart ramp | 507-509 | [CODE] | cosmetic | OK | n | I | H |
| RPT-24 | fill-chart y-cap 110 % | 438,522-527 | [CODE] | % axis; >110 % visually clipped | HAZ | n | L | H |
| RPT-25 | methodology prose (sections/routing/runoff-basis claims) | 809-833 | contradictions → §4 | — | UNVER-B | — | — | — |

### 2.6 peak_flow.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| PKF-01 | basis keys | 33-35 | [CODE] | — | OK | n | I | H |
| PKF-02 | harvesting ceiling 0.35 | 36-42 | project decision → Lancaster (blocked) | dimensionless | UNVER-B | — | — | — |
| PKF-03 | default peak intensity 40 mm/hr (self-declared placeholder) | 44-48 | code documents "not a computed answer" | mm/hr | UNVER | i-d | M | H |
| PKF-04 | C selection by basis; SCS→marginal | 66-76 | [ALG] ALG-10 for marginal choice | dimensionless | OK (conservative) | O | I | H |
| PKF-05 | Q = C·(i/3.6e6)·A | 87-89 | [ALG] ALG-04; → HEC-22 statement (blocked) | mm/hr→m/s→m³/s ✓ | OK (constant) / UNVER-B (method statement) | n | I | H |
| PKF-06 | harvesting-grade test | 98-100 | [CODE] | dimensionless | OK | n | I | H |
| PKF-07 | peak cascade, no attenuation (documented deliberate) | 120-133 | → routing refs (blocked); direction reasoning: ignoring attenuation raises downstream peak | m³/s | UNVER-B (DISC-by-design expected, O) | O | L | M |
| PKF-08 | own/upstream split | 142-147 | [CODE] | m³/s | OK | n | I | H |
| PKF-09 | dQ/dP delegation | 73 | [ALG] ALG-02 | dimensionless | OK | n | I | H |

### 2.7 rainfall_idf.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| IDF-01 | HIRDS durations 10…1440 min | 30 | → HIRDS v4 docs (blocked) | min | UNVER-B | — | — | — |
| IDF-02 | ARIs 2…100 yr | 33 | → HIRDS v4 docs (blocked) | yr | UNVER-B | — | — | — |
| IDF-03 | P2 = ARI-2 / 1440-min row | 38-39 | [TR55] sheet-flow P₂ definition = 2-yr 24-hr depth ✓ (§0.1) | mm | OK | n | I | H |
| IDF-04 | table cleaning | 52-58 | [CODE] | min/mm | OK | n | I | H |
| IDF-05 | end clamping, no extrapolation | 79-88 | [CODE]; conservative-honest policy documented | mm | OK | n | I | H |
| IDF-06 | log-log power-law interpolation | 89-96 | [ALG] ALG-13 (exact at knots, monotone); → HIRDS behaviour (blocked) | mm ✓ | OK (form) / UNVER-B (choice vs HIRDS) | n | I | H |
| IDF-07 | intensity = depth×60/duration (mean over Tc-duration storm) | 108 | [TXDOT]+corroborators: rational i = average intensity for duration = Tc from IDF — the code pipeline IS standard practice → §1 #29 | mm/hr ✓ | OK | n | I | H |
| IDF-08 | sheet_flow_p2 accessor | 112 | [CODE] | mm | OK | n | I | H |
| IDF-09 | extrapolation flag pooled across ARIs | 119-122 | [CODE] | min | HAZ | n | L | H |
| IDF-10 | duration parsing ×60; bare = minutes | 250-261 | [SI]/[CODE]; suffix order safe ('mins/min' before 'm') | ✓ | OK | n | I | H |
| IDF-11 | thousands-separator hazard + tab-first guard | 204-234,266 | residual-risk check → §4 | mm | UNVER-B | — | — | — |
| IDF-12 | ARI integer test | 240-245 | [CODE] | yr | OK | n | I | H |
| IDF-13 | dict round-trip | 130-140 | [CODE] | min/yr | OK | n | I | H |
| IDF-14 | "HIRDS v4" label | 223 | [CODE] | — | OK | n | I | H |

### 2.8 time_of_concentration.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| TOC-01 | sheet cap 100 ft → 30.48 m | 27 | [TR55] says 300 ft; 100 ft = NEH-630 Ch.15 → see NEW-W5-02 (§1 #8) | m ✓ | DISC (attribution; behaviour conservative) | O | I | H |
| TOC-02 | ft/m, in/mm inverses | 29-30 | [ALG] TOC-02 (exact definitions) | ✓ | OK | n | I | H |
| TOC-03 | metric sheet coeff ≈0.0913 | 32-34 | [ALG] ALG-03 = 0.091265; → TR-55 eq.3-3 base (§0.1) | ✓ | OK (conversion) | n | I | H |
| TOC-04 | shallow V=16.1345/20.3282·√s ×0.3048 | 36-38 | [TR55] App. F verbatim: "Unpaved V = 16.1345 (s) 0.5 / Paved V = 20.3282 (s) 0.5 … V ft/s, s ft/ft" — assignment matches; s dimensionless so ×0.3048 → m/s valid | m/s per √(m/m) ✓ | OK | n | I | H |
| TOC-05 | sheet n table vs TR-55 Table 3-1 | 43-55 | [TR55] Table 3-1 (local pass): all ten rows match exactly (0.011/0.05/0.06/0.17/0.15/0.24/0.41/0.13/0.40/0.80) | s/m^⅓ | OK | n | I | H |
| TOC-06 | defaults n 0.24/0.05/0.025; R 0.211 | 55-75 | n=0.025 verified [HEC15] Table 2.1 (bare soil max) + [LMNO]; others design defaults | — | OK (0.025) / UNVER (rest) | n | I | H/M |
| TOC-07 | preset radii 0.125/0.211/0.366/0.761 | 64-69 | [W5 arithmetic] recomputed to 3 dp from trapezoid elements | m ✓ | OK | n | I | H |
| TOC-08 | channel threshold 1.0 ha | 77-81 | design constant; code documents the direction rationale | ha | UNVER | n | I | H |
| TOC-09 | Tc floor 0.1 hr | 83-85 | [TR55] App. F verbatim: "Tc … (minimum, 0.1; maximum, 10.0)"; floor correct; the 10-hr CAP is not enforced → §1 #19 | hr | OK (floor) / DISC (missing cap) | U(if Tc>10 h) | I | H |
| TOC-10 | leg sum; design=max(total, 0.1); ×60 | 99-117 | [CODE] | hr; hr→min ✓ | OK | n | I | H |
| TOC-11 | summary: clamped header, raw legs | 120-123 | [CODE] | min | HAZ | n | L | H |
| TOC-12 | sheet Tt=0.0913(nL)^0.8/(P2^0.5 s^0.4); floors 1e-4/1e-3 | 138-143 | [ALG] ALG-03 + → TR-55 eq.3-3 (§0.1) | hr ✓ | OK (form+conversion) | n | I | H |
| TOC-13 | shallow t = L/(k√s·3600) | 148-153 | [SI] ÷3600 ✓; k values → TOC-04 | hr ✓ | OK (arith) | n | I | H |
| TOC-14 | channel Manning V=(1/n)R^(2/3)√s | 165-172 | [HEC15] Eq 2.1 SI (α=1) | m/s; hr ✓ | OK | n | I | H |
| TOC-15 | leg assembly + warnings | 187-232 | [CODE]; direction claims verified | hr | OK | n | I | H |
| TOC-16 | bankfull R (bottom floor 0.05) | 254-260 | [TR55] verbatim "determined for bank-full elevation" | m ✓ | OK (published procedure, conservative) | O | I | H |
| TOC-17 | per-run slope abs() (adverse reads positive) | 289-295 | [CODE]; adverse-reach behaviour noted | m/m | HAZ | i-d | L | H |
| TOC-18 | falsy radius/roughness substitution | 297-299 | flagged → §4 | m | UNVER-B | — | — | — |
| TOC-19 | run segmentation downstream-indexed | 305-317 | [CODE] | m/hr | OK | n | I | H |
| TOC-20 | channel length from area threshold; ha→m² | 341-346 | [SI]/[CODE] | ✓ | OK | n | I | H |
| TOC-21 | leg slopes end-to-end abs() | 374-392 | [CODE]; rationale documented | m/m | OK | n | I | H |
| TOC-22 | linear interp end-clamped | 396-408 | [CODE] | m | OK | n | I | H |
| TOC-23 | three-leg split (sheet 30.48 m, channel downstream) | 424-440 | [CODE]; partition exhaustive | m | OK | n | I | H |

### 2.9 flow_graph.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| FLG-01 | sentinels −1…−4 | 40-43 | [CODE] | — | OK | n | I | H |
| FLG-02 | fixed D8 scan order | 46 | [CODE] | — | OK | n | I | H |
| FLG-03 | doubling cap 64 | 49 | [CODE] | log₂(path) | OK | n | I | H |
| FLG-04 | accounting invariant Σ==domain | 71-76 | trace → §4 | cells | UNVER-B | — | — | — |
| FLG-05 | window slices | 81-87 | [CODE] | idx | OK | n | I | H |
| FLG-06 | best=0 → only positive drops claim | 107-119 | [CODE]; consistent with conditioned-DEM precondition | m/m | OK | n | I | H |
| FLG-07 | drop=(z−z_nbr)/hypot(dr·h,dc·w) | 121-137 | → O'Callaghan-Mark (blocked); diagonal de-weighting documented | m/m ✓ | UNVER-B | — | — | — |
| FLG-08 | pointer doubling | 152-171 | → algorithm ref (blocked) | idx | UNVER-B | — | — | — |
| FLG-09 | edge mask | 211-215 | [CODE] | cells | OK | n | I | H |
| FLG-10 | terminal classification priority | 216-228 | → §4 trace (blocked) | — | UNVER-B | — | — | — |
| FLG-11 | bincount counting | 230-242 | [CODE] | cells | OK | n | I | H |
| FLG-12 | walk_downslope 100k cap | 246-276 | [CODE] | cells | OK | n | I | H |
| FLG-13 | step length: diag √(w²+h²) | 316-325 | [CODE]; geometry sound | m ✓ | OK | n | I | H |
| FLG-14 | in-mask edge filter | 328-336 | [CODE] | graph | OK | n | I | H |
| FLG-15 | Kahn longest path on DAG | 343-363 | → CLRS-style ref (blocked); recurrence standard | m | UNVER-B | — | — | — |
| FLG-16 | path back-walk guard | 368-378 | [CODE] | cells | OK | n | I | H |
| FLG-17 | topo order, ring append | 393-416 | [CODE]; ≤1 out-edge argument documented | — | OK | n | I | H |

### 2.10 flow_lines.py (visualisation only)

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| FLL-01 | defaults 60 m / 400 / 1e-4 | 21-22 | viz tuning | mixed (see FLL-08) | UNVER | n | I | H |
| FLL-02 | nodata→NaN | 47-48 | [CODE] | m | OK | n | I | H |
| FLL-03 | cell_size=(w+h)/2; min_len 1.5× | 51-55 | [CODE] | anisotropy averaged | HAZ | i-d | L | H |
| FLL-04 | np.gradient NO spacing → m per CELL; NaN→global min | 57-58 | [NUMPY] "default unitary spacing" confirmed; nodata fill creates gradient walls → §1 #17 | m/cell (mixed per-axis units if w≠h) | DISC | i-d | M | H |
| FLL-05 | seed step m→cells | 60 | [CODE] | ✓ | OK | n | I | H |
| FLL-06 | (col+0.5) signed centre | 62-65 | [CODE] | map units | OK | n | I | H |
| FLL-07 | seed offsets | 68-69 | [CODE] | cells | OK | n | I | H |
| FLL-08 | min_grad 1e-4 in m-per-cell (scale-dependent) | 81-86 | [NUMPY] units per-cell → physical threshold = 1e-4/cell_size (behavioural impact negligible) | m/cell | DISC | i-d | L | H |
| FLL-09 | degrees(arctan(mag/cell)) | 88 | [ESRI] arctan relation; exact iff square cells; inherits FLL-04 near-nodata inflation | rise/run→deg | OK (square cells) / DISC (aniso) | i-d | L | H |
| FLL-10 | unit-step advance in index space | 89-90 | [CODE] | 1 cell/step | OK | n | I | H |
| FLL-11 | bounds/NaN termination | 76-92 | [CODE] | idx | OK | n | I | H |
| FLL-12 | filters; unweighted mean slope | 95-104 | [CODE] | deg; step- not length-weighted | HAZ | n | L | H |
| FLL-13 | slope_vectors defaults 50 m/0.5° | 109 | viz tuning | m/deg | UNVER | n | I | H |
| FLL-14 | duplicate gradient block | 136-141 | [CODE] = FLL-04 semantics | — | OK (dup) | n | I | H |
| FLL-15 | magnitude/slope duplicate | 147-149 | as FLL-04/09 (W8) | — | DISC | i-d | L | H |
| FLL-16 | bearing atan2(−gx, gy) % 360 | 151-153 | [GDAL] compass convention; W8 four-direction check E 90/N 0/S 180/W 270 all ✓ | deg compass ✓ | OK | n | L | H |
| FLL-17 | coords/rounding | 154-160 | [CODE] | map units | OK | n | I | H |

### 2.11 footprint.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| FTP-01 | all_touched=True policy | 29-32 | → rasterio docs (blocked); over-claim for thin features documented+deliberate | cells | UNVER-B | — | — | — |
| FTP-02 | rasterize | 42-51 | [CODE] | mask | OK | n | I | H |
| FTP-03 | domain fallback chain (…→ whole grid) | 69-92 | denominator trace → §4 | mask | UNVER-B | — | — | — |
| FTP-04 | dilate8 | 97-103 | [CODE] | cells | OK | n | I | H |
| FTP-05 | outer ring | 112-115 | [CODE] | cells | OK | n | I | H |
| FTP-06 | valid filter | 120-123 | [CODE] | mask | OK | n | I | H |
| FTP-07 | pour point rim-min; fallback highest-inside | 138-152 | direction trace → blocked (see §4 note) | m | UNVER-B | — | — | — |
| FTP-08 | outlet argmin inside | 162-167 | [CODE] | m | OK | n | I | H |
| FTP-09 | internal relief max−min | 177-182 | [CODE] | m | OK | n | I | H |
| FTP-10 | min_dimension 2A/P | 194-202 | [ALG] FTP-10 (both docstring limits exact; warning polarity safe) | m ✓ | OK | n | I | H |
| FTP-11 | (no stage-storage here) | — | [CODE] | — | OK | n | I | H |

### 2.12 core/sizing/primitives.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| PRM-01 | z=(T−b)/2d | 130 | [NASH] (Chow elements) — exact inversion of T=b+2zd | m/m ✓ | OK | n | I | H |
| PRM-02 | A=(T+b)/2·d | 131 | [NASH] A=(b+zd)d — identical | m² ✓ | OK | n | I | H |
| PRM-03 | P=b+2d√(1+z²) | 132 | [NASH] worked example 16.32 ft reproduces | m ✓ | OK | n | I | H |
| PRM-04 | R=A/P | 133 | [HEC15] glossary; [TR55] r=a/p_w | m ✓ | OK | n | I | H |
| PRM-05 | min_dim=min(T,b) | 143 | [CODE] | m | OK | n | I | H |
| PRM-06 | prismatic V=A·L (dead code) | 155 | [CODE]; dead → §4 INC-06 | m³ ✓ | OK (dead) | n | L | H |
| PRM-07 | contour spacing HI=VI/S (dead code) | 171-174 | → terrace refs (blocked); dead | m ✓ | UNVER-B (dead) | — | L | — |
| PRM-08 | Manning Q=(1/n)AR^(2/3)√S SI | 194 | [HEC15] Eq 2.1, α=1.0 SI | m³/s ✓ | OK | n | I | H |
| PRM-09 | v=Q/A | 195 | [HEC15]/[TR55] | m/s ✓ | OK | n | I | H |
| PRM-10 | Manning guards | 192-193 | [CODE] | — | OK | n | I | H |
| PRM-11 | convergence guard 2zd ≥ min(L,W) | 217-222 | [ALG] PRM-11 (exact wall-meet condition) | m ✓ | OK | n | I | H |
| PRM-12 | mid-area at L−z·d | 224-232 | [ALG] PRM-12 (true mid-depth section) | m² ✓ | OK | n | I | H |
| PRM-13 | prismoidal (d/6)(A₁+4Am+A₂) (dead code) | 234 | [ALG] ALG-08 (EXACT for linear batters — reproduces 81.333) | m³ ✓ | OK (dead) | n | L | H |
| PRM-14 | frustum min dim | 243 | [CODE] | m | OK | n | I | H |
| PRM-15 | shrink rate P·z | 283 | [ALG] PRM-15 (first-order offset loss) | m²/m ✓ | OK | U(storage, conservative) | I | H |
| PRM-16 | battered V=A·d−Pzd²/2 | 284-287 | [ALG] ALG-07 (under-states by (4/3)z²d³; code's 80.0-vs-81.33 exact) | m³ ✓ | DISC (documented conservative simplification) | U(storage) | I | H |
| PRM-17 | converging V=A²/(2Pz), t*=A/(Pz) | 288-292 | [ALG] PRM-17 (integral + continuity at t*, 125=125) | m³ ✓ | OK | n | I | H |
| PRM-18 | degenerate branches | 271-281 | [CODE] | — | OK | n | I | H |
| PRM-19 | drawdown t=V/(rate·area) (dead code) | 317-321 | [ALG] PRM-19/ALG-11 | hr ✓ | OK (algebra; constant-area U-bias) | U(hrs) | L | H |

### 2.13 core/sizing/advisories.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| ADV-01 | min stable batter by soil | 22-28 | → NRCS/cut-slope refs (blocked) | z (H:V) | UNVER-B | — | — | — |
| ADV-02 | max non-eroding grade % by soil | 32-38 | [FAO-WT]/[FAO-WM]/[HEC15] — no grade-by-texture table exists; ordering consistent; advisory only | % | UNVER | n | L | M |
| ADV-03 | CN table (dup) | 42-48 | as CAT-19 ([TR55] Table 2-2) + §4 INC-07 | CN | DISC | U | M | H |
| ADV-04 | nearest-CN argmin | 67 | [CODE] | CN | OK | n | I | H |
| ADV-05 | batter test | 79-80 | [CODE] | z | OK | n | I | H |
| ADV-06 | grade test | 103-104 | [CODE] | % | OK | n | I | H |
| ADV-07 | composition | 125-127 | [CODE] | — | OK | n | I | H |
| ADV-08 | soil fallback | 53-57 | [CODE] | — | OK | n | I | H |

### 2.14 core/registry/earthwork_types.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| REG-01 | dataclass defaults (z 1.0, d 0.5, T 2.0, ranges) | 55-66 | → NRCS 378/EFH-11 (blocked) | m / H:V | UNVER-B | — | — | — |
| REG-02 | swale defaults | 96-103 | → (blocked) | m | UNVER-B | — | — | — |
| REG-03 | berm defaults | 122-128 | → (blocked); note berm fill maths ignores these (EWD-27) | m | UNVER-B | — | — | — |
| REG-04 | basin defaults (vertical, 1.5 m) | 146-152 | → (blocked) | m | UNVER-B | — | — | — |
| REG-05 | dam defaults (rect wall, 2.0 m) | 171-177 | → (blocked) | m | UNVER-B | — | — | — |
| REG-06 | diversion defaults + top-vs-bed width clash | 196-203 | → §4 (documented TODO; live consequence at EWD-26/32) | m | UNVER-B | — | — | — |

### 2.15 swale_design.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| SWL-01 | infiltration rates 15/8/4/2.5/1.5 mm/hr | 30-36 | → FAO/NRCS tables (blocked) | mm/hr | UNVER-B | — | — | — |
| SWL-02 | lookup, Loam default | 41 | [CODE] | mm/hr | OK | n | I | H |
| SWL-03 | bottom=max(0,T−2zd) | 95-96 | [NASH] identity; converged-walls edge → §1 #7 | m ✓ | OK w/ edge | U(edge) | L | H |
| SWL-04 | capacity/m = A·0.8 + (f/1000)·dur·T | 98-100 | [NASH]/[HEC15]/[FAO-WT] freeboard refs; infiltration width unpublished — §1 #6 | m³/m ✓ | DISC | U(lean) | L | M |
| SWL-05 | L = inflow/capacity_per_m | 104 | arithmetic | m ✓ | OK | n | I | H |
| SWL-06 | freeboard 0.8 usable fraction | 48-50 | published practice is depth-based (≈20 % or 0.2–0.5 m) — area-fraction form nonstandard but conservative | fraction | DISC | O | I | M |
| SWL-07 | StorageCheck fields | 107-119 | [CODE] | declared | OK | n | I | H |
| SWL-08 | section at drawn dims | 157-158 | [NASH] | m² | OK | n | I | H |
| SWL-09 | storage at length + deficit | 160-167 | arithmetic on SWL-04 terms | m³ ✓ | OK | n | L | H |
| SWL-10 | required section back-out ÷0.8 | 175-177 | exact inversion of SWL-04 | m² ✓ | OK | n | I | H |
| SWL-11 | quadratic depth inversion | 179-192 | [ALG] SWL-11 (root choice, discriminant ⇔ A≤T²/4z, z→0 limit) | m ✓ | OK | n | I | H |
| SWL-12 | delegation | 169-172 | [CODE] | m | OK | n | I | H |
| SWL-13 | 24 stations, midpoint | 221-223 | discretisation choice | m | UNVER | n | I | H |
| SWL-14 | per-cell→station binning | 240-241 | [CODE] | m³ ✓ | OK | n | I | H |
| SWL-15 | uniformity mean/max; ~0.5 threshold | 247 | heuristic | dimensionless | UNVER | n | I | H |
| SWL-16 | cumulative/peak station | 251-255 | [CODE] | m³/m | OK | n | I | H |
| SWL-17 | overtopping march; freeboard default 1.0 vs 0.8 | 259-285 | [CODE] caller trace: sole production caller (earthworks.py:1755) passes freeboard=1.0 → §1 #27 | m³ ✓ | DISC | U(warning) | M | H |
| SWL-18 | contour substring | 336-346 | [CODE] | m | OK | n | I | H |
| SWL-19 | world→pixel truncation; isclose nodata | 375-376 | [CODE] | idx | OK | n | I | H |
| SWL-20 | peak acc sampling 30 pts | 425-434 | → (blocked) | cells | UNVER-B | — | — | — |
| SWL-21 | total-inflow overcount (documented ~×L/cell) | 440-512 | [CODE] Grep: **zero production callers** — the documented over-counter is dead code | m³ | OK (DEAD) | n | L | H |

### 2.16 earthwork_design.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| EWD-01 | caps 4M cells / 64 pad | 57-58 | memory guards | cells | UNVER | n | I | H |
| EWD-02 | abutment bearing vector | 102-111 | [CODE] | m | OK | n | I | H |
| EWD-03 | abutment march 1 m/250 m | 112-123 | search params; failure semantics documented | m | UNVER | n | L | H |
| EWD-04 | pond edge test, eps 1e-6 | 167-172 | [CODE] | m | OK | n | I | H |
| EWD-05 | spillway freeboard 0.15 m | 184 | [NRCS378] minimum 1.0 ft (0.30 m) — code is half the standard → §1 #24 | m | DISC | U | M | M |
| EWD-06 | head band 0.20-0.50 m | 188 | → USBR/Brater-King (blocked) | m | UNVER-B | — | — | — |
| EWD-07 | elev eps 1 mm | 194 | [CODE] | m | OK | n | I | H |
| EWD-08 | default head 0.30 m | 211 | → (blocked) | m | UNVER-B | — | — | — |
| EWD-09 | crest ceiling rim−head−freeboard | 266-269 | → (blocked); internal algebra consistent | m | UNVER-B | — | — | — |
| EWD-10 | crest↔drop binding | 288-303 | [CODE] exact inverses | m | OK | n | I | H |
| EWD-11 | sufficiency rim−crest ≥ head+freeboard | 323-334 | → (blocked); algebra consistent with EWD-09 | m | UNVER-B | — | — | — |
| EWD-12 | invert/band/width checks | 320-355 | [CODE] | m | OK | n | I | H |
| EWD-13 | registry seeds; 0.1 floor | 383-392 | [CODE] | m | OK | n | I | H |
| EWD-14 | batter 0.0; gradient 1.0 % defaults | 393,401 | defaults | m/% | UNVER | n | I | H |
| EWD-15 | side_slope property + lossy setter (0.1 clamp) | 433-440 | flagged → §4 | m/m | UNVER-B | — | — | — |
| EWD-16 | wall_slope ONE-sided vs side_slope TWO-sided | 448-455 | RESOLVED consistent: basin_volume_battered's z is a PER-SIDE inset rate (PRM-15 derivation: A(t)=A−P·z·t needs per-side δ=z·t), and both production callers pass batter_run/depth = per-side (W5-EXTRA-01). Different conventions per feature type, each consumer correct | m/m ✓ | OK | n | I | H |
| EWD-17 | buffer radius T/2 | 463-465 | [CODE] | m | OK | n | I | H |
| EWD-18 | _resolve_bottom_width (implicit z=1; 0.1 floor) | 646-649 | [CODE]; floors → §3 | m | OK | n | I | H |
| EWD-19 | swale capacity ×0.8 + berm h·T/2 credit | 678-689 | [ALG] ALG-06 — berm credit inconsistent with fill section → §1 #4 | m³ ✓ | DISC | O(claim) | M | H |
| EWD-20 | basin capacity battered ×0.8 | 696-699 | [ALG] ALG-07 basis (conservative model) | m³ ✓ | OK | n | I | H |
| EWD-21 | m³→L ×1000 | 703 | [SI] | ✓ | OK | n | I | H |
| EWD-22 | FREEBOARD 0.8 constant | 710 | as SWL-06 | fraction | DISC (nonstandard form, conservative) | O | I | M |
| EWD-23 | breakdown ÷0.8 back-out; cell_size² | 731-772 | [CODE] | m³ | OK | n | I | H |
| EWD-24 | swale cut = area × length (no freeboard) | 794-799 | [NASH] elements; correct that excavation has no freeboard | m³ ✓ | OK | n | I | H |
| EWD-25 | basin cut = A·d ignores batter | 801-803 | vs battered capacity path — three-representation issue → §4 INC-11 | m³ | UNVER-B | O(cut est) | — | — |
| EWD-26 | diversion cut ±2d "1:1" (actually z=2; bed at mid-depth) | 805-814 | [W5 geometry] → §1 #3 | m³ | DISC | U(cut) | M | H |
| EWD-27 | berm fill = d²·L (1:1 triangle; width ignored) | 834-839 | [ALG] ALG-06(a) — 1:1 triangle identity exact | m³ ✓ | OK (formula; registry width unused — noted) | n | L | H |
| EWD-28 | companion berm h=√(0.75A), section h² | 841-848 | [ALG] ALG-06(b) — volume-conserving w/ 25 % loss, self-consistent | m³ ✓ | OK | n | I | H |
| EWD-29 | dam fill = w·d·L rectangle | 850-852 | → embankment-section refs (blocked); under-states fill vs battered wall | m³ | UNVER-B | U(cost) | — | — |
| EWD-30 | berm height √(0.75·A) | 863-866 | [ALG] ALG-06 | m ✓ | OK | n | I | H |
| EWD-31 | diversion n=0.025; %→decimal | 884-887 | [HEC15] Table 2.1 max bare-soil; [LMNO]; [NASH] example | ✓ | OK | n | I | H |
| EWD-32 | legacy R: z=2 area, √2 perimeter | 888-899 | [W5 arithmetic + NASH/HEC15 elements] → §1 #1 | R ×1.40, Q ×1.25 | DISC | U | H | H |
| EWD-33 | stored-geometry path | 900-905 | [NASH]/[HEC15] | ✓ | OK | n | I | H |
| EWD-34 | Manning call, 4 dp | 906-907 | [HEC15] | m³/s ✓ | OK | n | I | H |
| EWD-35 | spillway L = Q/(C·H^1.5), C=1.7 | 910-920 | [ALG] ALG-05 (inversion+units exact) + [BK-MODOT] Brater & King 1976 table: broad-crest C at 0.2-0.5 m head, breadth ≥0.6 m = 1.44-1.49 SI; code's 1.7 ≈ ideal ceiling → width under-sized ~16% → §1 #23 | m ✓ | DISC | U | H | H |
| EWD-36 | cell_size = \|a\| (assumes square) | 946 | [CODE]; → §3 | m | HAZ | i-d | L | H |
| EWD-37 | wall offset to lower-mean side | 1030-1043 | [CODE] | m | OK | n | I | H |
| EWD-38 | swale footprint buffer T/2 | 1062 | [CODE] | m | OK | n | I | H |
| EWD-39 | swale burn level_invert | 1076-1080 | [ALG] ALG-14 identity conditions | m | OK | n | I | H |
| EWD-40 | berm offsets; berm width := T | 1087-1092 | design choice | m | UNVER | n | L | H |
| EWD-41 | berm raise = Σcut·0.75/n_berm | 1112-1125 | [ALG] cell-area cancellation valid (same grid); 0.75 → shrink tables blocked | m ✓ | OK (arith) / UNVER-B (0.75) | i-d | M | M |
| EWD-42 | berm burn += depth | 1131-1138 | [CODE] | m | OK | n | I | H |
| EWD-43 | centroid fallback | 1160-1165 | [CODE] | idx | OK | n | I | H |
| EWD-44 | basin burn branch | 1180-1184 | [CODE] | m | OK | n | I | H |
| EWD-45 | batter steps n=3 | 1189-1215 | discretisation choice; degradation documented | m | UNVER | n | L | H |
| EWD-46 | cut/storage from burned grid | 1219-1224 | [SI]/[CODE] | Σm·m²=m³ ✓ | OK | n | I | H |
| EWD-47 | dam crest absolute max() | 1236-1243 | [CODE] | m | OK | n | I | H |
| EWD-48 | diversion chainage | 1253-1265 | [CODE] | m (planar) | OK | n | I | H |
| EWD-49 | graded invert %→decimal; 3/cell samples | 1269-1290 | [SI]/[CODE] | ✓ | OK | n | I | H |
| EWD-50 | snap+monotonic+warn | 1296-1307 | [CODE]; breach maths at BRN-03 | m | OK | n | I | H |
| EWD-51 | ponding downsample | 1339-1349 | → resample check (blocked) | m | UNVER-B | — | — | — |
| EWD-52 | ponding = clip(filled−dem, 0) | 1375-1376 | [CODE] | m ✓ | OK | n | I | H |
| EWD-53 | ponding bilinear upsample (not volume-conserving) | 1381-1388 | [CODE] | m | HAZ | i-d | L | H |
| EWD-54 | dam-key march ½ cell | 1411-1438 | duplicate walk → §4 INC-04 | m | UNVER | n | L | H |
| EWD-55 | cell_area = \|a·e\| | 1483 | [CODE]; only rectangular-pixel-safe site → §3 | m² | OK | n | I | H |
| EWD-56 | dam stage-storage windowed pad-doubling | 1491-1514 | → NRCS EFH-11 stage-storage (blocked); termination at full DEM confirmed in code | m³ ✓ | UNVER-B | — | — | — |
| EWD-57 | bbox→cell bounds | 1518-1533 | [CODE] | idx | OK | n | I | H |
| EWD-58 | summary() always legacy discharge path | 490 | [W5] NEW-W5-01 → §1 #2 | m³/s | DISC | U | M | H |

### 2.17 burn_strategy.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| BRN-01 | Bresenham 8-connected | 39-59 | → algorithm ref (blocked) | cells | UNVER-B | — | — | — |
| BRN-02 | map→cells truncation, no offset | 74-98 | → GDAL convention (blocked) | idx | UNVER-B | — | — | — |
| BRN-03 | monotonic breach min_drop 1e-3/CELL | 101-128 | grade = 1e-3/cell_size (cell-size dependent) → §3 | m per cell | HAZ | i-d | L | H |
| BRN-04 | level floor spill−depth, cut-only | 131-156 | [ALG] ALG-14 (identity conditions + deviation signs) | m ✓ | OK | i-d | I | H |
| BRN-05 | battered steps depth-sorted | 169-172 | [CODE] | m | OK | n | I | H |
| BRN-06 | rasterisable capacity, 3-cell threshold, shape factor | 175-201 | novel heuristic; worked example internally consistent | cells·m²·m ✓ | UNVER | i-d | L | H |
| BRN-07 | steep-ground relief>depth | 214-227 | [CODE] | m | OK | n | I | H |
| BRN-08 | sub-cell min_dim<cell | 237-243 | [CODE] | m | OK | n | I | H |
| BRN-09 | cap advisory | 253-258 | [CODE] | — | OK | n | I | H |

### 2.18 contour_analysis.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| CTA-01 | acc→ha | 51 | [SI] | cells·m²/10⁴=ha ✓ | OK | n | I | H |
| CTA-02 | inflow m³ from acc | 54 | [SI] | cells·m²·mm/10³=m³ ✓ | OK | n | I | H |
| CTA-03 | fallback label "cells" | 63-66 | [CODE] | honest unit | OK | n | I | H |
| CTA-04 | gdal_contour 1.0 m | 96-104 | [CODE] | m | OK | n | I | H |
| CTA-05 | nodata→NaN | 144 | [CODE] | m | OK | n | I | H |
| CTA-06 | level generation | 150-156 | [CODE] | m | OK | n | I | H |
| CTA-07 | marching nodata elev−1 | 162 | [SKIMAGE] find_contours masks NaN natively, leaves boundary contours open — pre-fill closes them → fabricated edge-hugging contours, selectable for swale alignment → §1 #16 | m | DISC | i-d | M | H |
| CTA-08 | rc→map \|a\|-x / signed-e-y asymmetry | 158,167-169 | → §3 convention row | map | HAZ | i-d | L | H |
| CTA-09 | 18° filter ("≈1:3") | 186,198 | [ALG] ALG-12 (tan 18° = 1/3.08 — approx as claimed); threshold itself design choice | deg | UNVER (threshold) / OK (gloss) | n | I | H |
| CTA-10 | masked np.gradient stencil (deliberate vs Horn) | 217-237 | [NUMPY] central-difference semantics + spacing pairing confirmed correct; a published alternative estimator (GDAL ZevenbergenThorne family); noisier than Horn but mean-of-20-samples damps it; old-numpy fallback reintroduces the border spike | m/m→deg ✓ | DISC (deliberate, documented) | n | L | H |
| CTA-11 | sample count heuristic | 244 | [CODE] | count | OK | n | I | H |
| CTA-12 | map→cell truncation | 248-249 | [CODE]; → §3 | idx | OK | n | I | H |
| CTA-13 | all-NaN samples → 0° → PASSES filter | 254 | polarity check → §4 | deg | UNVER-B | — | — | — |
| CTA-14 | acc nodata→0 | 297 | [CODE] | cells | OK | n | I | H |
| CTA-15 | peak ranking | 306-320 | [CODE] | cells | OK | n | I | H |
| CTA-16 | sliver 0.5 m | 368 | tolerance | m | UNVER | n | I | H |
| CTA-17 | pipeline defaults | 388-390 | defaults | — | UNVER | n | I | H |
| CTA-18 | segment defaults (0.5 ha/0.25/0.3×0.6 m/≤3) | 508-515 | defaults; 0.6×0.3 z=1 → bottom clamps 0 (degenerate triangle feeds SWL-04) | — | UNVER | i-d | L | H |
| CTA-19 | ha→cells | 567-573 | [SI] | ✓ | OK | n | I | H |
| CTA-20 | per-seg slope, no NaN filter (opposite polarity to CTA-13) | 594-605 | → §4 | deg | UNVER-B | — | — | — |
| CTA-21 | stride/min-length floors 1.0 m | 614-620 | [CODE] | m | UNVER | n | I | H |
| CTA-22 | peak window ±5/skip 10 samples | 635-655 | window scales with cell size | samples | UNVER | n | I | H |
| CTA-23 | contrib ha + inflow per peak | 665-667 | [SI] as CTA-01/02 | ✓ | OK | n | I | H |
| CTA-24 | centring/shortfall/cap tolerance | 670-696 | [CODE]; freeboard 0.8 implicit | m | OK | n | I | H |
| CTA-25 | landscape walk 0.25× | 698-708 | heuristic | cells | UNVER | n | I | H |
| CTA-26 | rounding | 710,732-739 | [CODE] | display | OK | n | I | H |
| CTA-27 | mm→m; hr→s | 799-800 | [SI] | ✓ | OK | n | I | H |
| CTA-28 | window mean; L/s; raw-cells fallback in m³ field | 811-846 | unit-mixing check → §4 | mixed | UNVER-B | — | — | — |
| CTA-29 | global max stamp | 856-859 | [CODE] | m³ (or cells) | OK | n | I | H |

### 2.18b modules/project_io.py (post-inventory addition)

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| PIO-01 | persistence schema — coercion types + default values only, no formulas. NOTE the defaults diverge from module signatures: max_slope_deg 15.0 here vs 18.0 in contour_analysis:186; swale 0.6 m × 2.0 m here vs 0.3 × 0.6 m in find_swale_segments; stream_threshold_ha 5.0 (a ha-based threshold — the analysis-side FLA-04 threshold is in CELLS) | ~54-80 | [CODE] grep this session | mixed defaults | HAZ (drift between persisted defaults and code defaults) | i-d | L | M |

### 2.19 dem_loader.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| DEM-01 | cell metrics; mean size; extent ha | 81-85 | [SI]/[CODE]; extent over full rect | ✓ | OK w/ note | n | I | H |
| DEM-02 | projected-CRS guard (refuses geographic) | 66-73 | [CODE]; ONLY such guard — other loaders unchecked → §3 | — | OK | n | I | H |
| DEM-03 | clip nodata −9999; mask+crop only | 113-124 | [CODE] | m | OK | n | I | H |
| DEM-04 | Horn slope stencil (1-2-1 / 8Δ) | 132-159 | [HORN81] weighted-central-difference formula quoted + [ESRI]/[GDAL] — stencil, divisor 8Δ, arctan all EXACT; NaN→0.0 m fabricates ~90° ring beside nodata; edge replication halves border slope → §1 #15 | m/m→deg ✓ | DISC (nodata handling) | i-d | M | H |
| DEM-05 | slope pipeline NaN→0; nodata declared not written | 187-200 | [GDAL] emits nodata over nodata input — this pipeline never writes its declared −9999 → former-nodata renders as valid slope → §1 #12 | deg | WRONG | i-d | M | H |

### 2.20 keypoint_analysis.py

| ID | Computes | Line | Source | Dim | Verdict | Dir | Sev | Conf |
|----|----------|------|--------|-----|---------|-----|-----|------|
| KPA-01 | DEM nodata→NaN; acc unhandled | 31-35 | [CODE] | m; cells | HAZ | i-d | L | H |
| KPA-02 | mean cell size | 37-39 | [CODE] | m | OK | n | I | H |
| KPA-03 | rc→xy version A (\|e\|/2 offset) | 44-47 | → §3 convention row (one full cell vs version B — pending confirm) | map | UNVER-B | — | M? | — |
| KPA-04 | cached Horn slope | 49-57 | as DEM-04 | deg | UNVER-B | — | — | — |
| KPA-05 | min_acc 500 CELLS (0.05 ha @1 m, 1.25 ha @5 m) | 95-118 | [CODE] scale-dependence quantified | cells | HAZ | i-d | L | H |
| KPA-06 | ~10 m smoothing window | 125-127 | [CODE] | m→cells | UNVER | n | I | H |
| KPA-07 | valley mask | 130 | [CODE] | cells | OK | n | I | H |
| KPA-08 | 75th-pct ceiling | 134-138 | heuristic | cells | UNVER | n | I | H |
| KPA-09 | proxy score acc/(slope°+1) | 151-153 | self-described NOT strict Yeomans → W4 blocked | cells/deg (mixed) | UNVER-B | — | — | — |
| KPA-10 | separation 10 % grid | 155-181 | [CODE]; index-space distance | cells | UNVER | n | I | H |
| KPA-11 | catchment ha | 188 | [SI] | ✓ | OK | n | I | H |
| KPA-12 | TPI 15-cell/1.5 m/acc≤2; NaN→DEM mean | 207-233 | [JENNESS/Weiss] TPI = z − neighbourhood mean, EXACT match; NaN→DEM-mean contaminates mean near nodata; cell-count window resolution-dependent (per Weiss's own scale statement); thresholds unpublished → §1 #18 | m ✓ | DISC | i-d | L | H |
| KPA-13 | border strip | 235-236 | [CODE] | cells | OK | n | I | H |
| KPA-14 | erosion ×3 as "thinning" | 246-255 | [SCIPY]/[SKIMAGE] definitions: erosion shrinks shapes, skeletonize extracts centrelines — ridges ≤6 cells wide annihilated, ends retreat, remnant not a centreline; early-stop only fires when the WHOLE raster empties → §1 #13 | cells | WRONG | U(ridge detect) | M | H |
| KPA-15 | min ridge length as cell COUNT | 258,289 | flagged → §4 | cells vs m | UNVER-B | — | — | — |
| KPA-16 | greedy pixel ordering | 59-91 | [CODE] | cells | OK | n | I | H |
| KPA-17 | 30 m search, square scan | 310-316 | [CODE] | m→cells | UNVER | n | I | H |
| KPA-18 | 4× acc ceiling | 322-323 | heuristic | ratio | UNVER | n | I | H |
| KPA-19 | dam crest = candidate+2.0 m (doc says keypoint) | 330 | doc-code mismatch → §4 | m | UNVER-B | — | — | — |
| KPA-20 | dam score cells/(m+1) | 335-343 | dimensionally mixed regulariser | mixed | HAZ | n | L | H |
| KPA-21 | ±200-col scan `break` on nc<0 → width 0 near left edge | 365-380 | [CODE] re-read this session — **CONFIRMED** → §1 #20 | m | WRONG | i-d | M | H |
| KPA-22 | spacing_m as VERTICAL 5 m increment | 384-428 | same-name/two-meanings → §4 | m elevation | UNVER-B | — | — | — |
| KPA-23 | class doc claims D8; code requests dinf; fdir unused | 455-480 | doc-code mismatch → §4 | — | UNVER-B | — | — | — |
| KPA-24 | rc→xy version B ((row+0.5)·e) | 491-493,750-753 | GDAL-consistent form; pairs with KPA-03 → §3 | map | OK (this version) | n | I | M |
| KPA-25 | outlet argmax; ≥5-cell guard | 516-521 | [CODE] | cells | OK | n | I | H |
| KPA-26 | arc length via MEAN cell size; NaN→0.0 m in profile | 526-535 | [CODE]; anisotropy + sea-level-spike noted | m | HAZ | i-d | L | H |
| KPA-27 | resample min(5·cell, 10 m) linear | 537-541 | [CODE] | m | UNVER | n | I | H |
| KPA-28 | savgol 20 % window, poly 3 | 543-548 | [SCIPY] constraints (window ≤ len, odd, > polyorder) all satisfied for every n_samp ≥ 5 (exhaustive small-n trace); lines 546-547 dead code | m | OK | n | I | H |
| KPA-29 | keypoint = argmax d²E/ds² | 550-556 | → Yeomans definition (blocked); derivative units 1/m correct | 1/m ✓ | UNVER-B | — | — | — |
| KPA-30 | nearest-cell map-back | 558-571 | [CODE] | m | OK | n | I | H |
| KPA-31 | cultivation runs; 1/500 advisory only | 573-647 | → Yeomans (blocked); cross_grade never enters geometry (confirmed in inventory excerpt) | m | UNVER-B | — | — | — |
| KPA-32 | contour trace elev−1e6; cell-unit distance | 653-674 | [SKIMAGE] native NaN masking bypassed (fabricated boundary contours) + row²/col² in cell units assumes square cells → keyline can snap to artefact or metrically-farther component → §1 #16 | cells | DISC | i-d | M | H |
| KPA-33 | fallback direction missing row→y sign flip vs sibling | 676-692 | [ALG-style map-space derivation] — **CONFIRMED**: output = −∇z (downhill axis), 90° off the contour → §1 #28 | m/m | WRONG (fallback path) | i-d | L | M |
| KPA-34 | degenerate-gradient defaults 0.0 vs 1.0 | 694-723 | inconsistency → §4 | — | UNVER-B | — | — | — |
| KPA-35 | z-sampling nearest; fallback base_elev | 725-744 | [CODE] | m | HAZ | i-d | L | H |
| KPA-36 | dinf fdir (radians) cast to int32 | 776-806 | **latent hazard** — consumer analysis → §4 | rad→int | UNVER-B | — | — | — |
| KPA-37 | thalweg walk (combined elev+acc test; fdir arg unused) | 812-859 | → §4 | cells | UNVER-B | — | — | — |
| NEW-W8-01 | thalweg profile NaN→0.0 m (cliff artefact → keypoint mislocation) | 526-529 | [W8] → §1 #14 | m | DISC | i-d | M | H |
| NEW-W8-02 | keypoint elevation label falls back to 0.0 m | 186,562 | [W8] | m | HAZ | n | L | H |
| NEW-W8-03 | np.nanmin on all-NaN DEM unguarded (flow_lines.py:58,139) | — | [W8] | — | HAZ | n | L | H |
| NEW-W8-04 | keypoint separation metric mixes row/col cell units (square-cell assumption) | 170-181 | [W8] | cells | HAZ | i-d | I | H |
| NEW-W8-05 | contour rc→map x uses \|a\| — breaks only for negative a (exotic) | contour_analysis.py:167-168 | [W8] | map | HAZ | i-d | I | H |

---

## §3 Unit & measurement-basis mixing register

Confirmed conversion sites are marked ✓ in §2. Register of cross-cutting boundaries
(assembled from the inventory's verbatim excerpts; the dedicated A3 sweep was
blocked, so treat as inventory-grade — locations verified, judgements pending):

| UNI | Boundary | Sites | Guarded? | Silent-failure trigger | Dir |
|-----|----------|-------|----------|------------------------|-----|
| UNI-01 | mm→m ÷1000 | flow_analysis:150,214 · catchment:365 · water_balance:212 · simulation:89,464 · swale_design:99,162,175 · contour:799 | no | none (constant) | n |
| UNI-02 | m³↔L ×1000 | flow_analysis:221,260 · simulation:486,537,547 · earthwork_design:703 · contour:842 | no | none | n |
| UNI-03 | hr↔s ×3600; min↔hr ÷60 | flow_analysis:190 · simulation:485 · contour:800 · toc:153,172 · idf:108,252-261 | no | none | n |
| UNI-04 | m²↔ha ÷10 000 | flow_analysis:353 · catchment:82,216 · contour:51,667 · keypoint:188 · toc:341 · contour:567-573 (inverse) | no | none | n |
| UNI-05 | mm/hr→m/s ÷3.6e6 | peak_flow:89 | no | none — [ALG] ALG-04 | n |
| UNI-06 | imperial 0.3048 / 25.4 | toc:27-38 | no | none — [ALG] ALG-03/TOC-02 | n |
| UNI-07 | %↔decimal slope ÷100 | earthwork_design:885,1269 · advisories (grade %) | no | passing % where decimal expected — W5-EXTRA-01 found **zero** live mix-ups | n |
| UNI-08 | rise/run vs run/rise (z) | primitives callers | docstring only | W5-EXTRA-01: all production call sites verified correct; EWD-16 (basin one-sided z) pending → §4 | n |
| UNI-09 | per-cell vs per-site (acc cells → area/volume) | flow_analysis:144-151 · contour:51,54,665-667 · keypoint:188 · swale:425-434,497-509 | no | pysheds self-inclusion semantics UNVER-B; SWL-21 documented over-count | i-d |
| UNI-10 | freeboard THREE bases: 0.8 area-fraction / 1.0 fraction / 0.15 m absolute | swale:49,259 · earthwork:689,699,710,184 | no | caller confusing fraction with metres | i-d |
| UNI-11 | cell size/area: \|a\| only (ewd:946) vs mean(w,h) (dem:83, kpa:38, fll:53) vs w·h (dem:84, fla:213, sim:396) vs \|a·e\| (ewd:1483) vs cell_size² (ewd:759,1223) | — | no | rectangular (non-square) pixels: \|a\| and cell_size² paths silently wrong | i-d |
| UNI-12 | CRS-in-metres | dem_loader:66-73 ONLY | partial | any module opening rasters directly (flow_analysis, catchment, simulation, flow_lines, keypoint, swale, contour all call rasterio.open) skips the guard if fed a path that never went through load_dem | i-d |
| UNI-13 | float32 narrowing of m³ | flow_analysis:440-446 · simulation:475-479 | no | ≥10⁷ m³ accumulations lose ~1 m³ resolution | i-d |
| UNI-14 | nodata conventions (NINE): NaN / 0.0-acc / elev−1 / elev−1e6 / DEM-mean / DEM-min / 0.0 m / −9999 / −inf | cat C-5,C-7,C-14 · kpa K-1,K-12,K-32,K-37 · fll FL-2 · dem D-3,D-4 · sim:398 | no | value-dependent artefacts (gradient walls at DEM-min fills; mean-fill TPI bias; 0.0-m elevations in profiles) | i-d |
| UNI-15 | coordinate conventions: two rc→xy versions (KPA-03 vs KPA-24, differ by ~one full cell N-S — pending confirm) + \|a\|-x/signed-e-y mix (CTA-08) + int() truncation (~7 sites, no half-cell offset) | — | no | sub-cell positional bias; systematic ~½–1 cell offsets between layers | i-d |
| UNI-16 | depth vs volume | reporting:89-94,240-266 (Σdepth·cellarea) · ewd:1219-1224,1513 | no | RPT-08 raw-sum vs RPT-02 clipped-sum divergence (pending) | i-d |
| UNI-17 | min_drop per CELL not per metre | burn_strategy:101-128 | no | breach grade = 1e-3/cell_size — 10× steeper on 0.1 m grid than 1 m grid | i-d |
| UNI-18 | per-step volume vs rate | simulation:593-604 | no | non-constant dt makes "peak" raster incomparable across steps | i-d |

---

## §4 Internal inconsistencies & cross-file divergences

**Confirmed this audit** (from completed verifiers):
- **INC-A (EWD-32/EWD-26/NEW-W5-01):** the diversion geometry family — see §1 #1-3.
- **INC-B (ALG-06):** companion-berm section h·T/2 (capacity) vs h² (fill) — §1 #4.
- **INC-C (NEW-W5-02):** 100-ft sheet cap attributed to TR-55 which states 300 ft — §1 #8.
- **INC-D (W5-EXTRA-01):** slope-convention audit across all primitive call sites — **no mismatches found** (positive confirmation). `contour_spacing` and `pond_volume_frustum` have zero production callers.

**Flagged candidates — NOT yet independently confirmed** (owners terminated;
each listed with the checklist evidence and what confirmation requires):

| INC | Claim | Evidence status |
|-----|-------|-----------------|
| INC-01 | Duplicated SCS runoff code (catchment 409-426 vs 470-480) | **RESOLVED — numerically identical** (int vs float literals only, same values in Py3 division); duplication remains a drift risk |
| INC-02 | Capture-% split: water_balance deliberately UNCLAMPED (its docstring calls clamping a removed bug) vs reporting.py:401-404 clamped [0,100] | both excerpts verbatim in inventory; consequence trace pending |
| INC-03 | Freeboard three bases (0.8 fraction / 1.0 fraction / 0.15 m) | see UNI-10 |
| INC-04 | Two abutment walks (1 m/250 m vs ½-cell/(rows+cols)·cell) answering the same physical question | both implementations quoted; agreement test pending |
| INC-05 | Nine nodata conventions | see UNI-14 |
| INC-06 | Dead primitives: prismatic_volume, contour_spacing, pond_volume_frustum, drawdown_time never called by production (advertised in module docstring) | **CONFIRMED for all four** — Grep this session: prismatic_volume + drawdown_time appear only in core/sizing definitions/re-exports; W5 confirmed the other two. Note catchment.drain_down_hours duplicates drawdown_time's formula (identical algebra, mm/hr vs m/hr input convention) |
| INC-07 | CN table triplication (catchment / advisories / swale alias) | **CONFIRMED byte-identical** (both inventories quoted both tables verbatim: 39/49/61/74/80); dual ownership remains a drift risk |
| INC-08 | Bottom-width floor zoo: max(0,·) swale:95,157 vs max(0.1,·) ewd:392,440,648 vs max(0.05,·) ewd:811,894,1307 + toc:257 — same feature can get different bottom widths in different paths | quotes verbatim; divergent-example construction pending |
| INC-09 | RPT-07 key mismatch | **CONFIRMED** — subscript access → KeyError at live-panel render with flow data; §1 #21 |
| INC-10 | RPT-16: unattributed_m3 hard-wired 0.0; its display row can never render | quote verbatim; caller sweep pending |
| INC-11 | Basin three-representation split: capacity honours batter, cut ignores it, burn has its own battered path | quotes verbatim; per-surface number trace pending |
| INC-12 | RPT-18/19/20 floors mask designs that WORSEN exit volume / peak / timing (display reads 0 % reduction) | quotes verbatim; polarity check pending |
| INC-13 | RPT-25 methodology prose: claims both trapezoidal and rectangular sections; claims elevation-centroid routing that only the simulation path still uses; omits the Lancaster default basis | quotes verbatim; sentence-by-sentence check pending |
| INC-14 | KPA-21 valley-scan break bug | **CONFIRMED** by re-read; §1 #20. Fix: `continue` instead of `break` on the left-bound check (right-bound break is fine) |
| INC-15 | KPA-33 fallback keyline direction | **CONFIRMED** by map-space derivation — output is −∇z (downhill), 90° off the contour; §1 #28 |
| INC-16 | **KPA-36 latent hazard:** D-infinity fdir (continuous radians) cast to int32; currently unread (thalweg walk ignores fdir) — becomes live if any consumer starts using `_fdir_arr` | quotes verbatim; consumer sweep pending |
| INC-17 | SIM-26 unreachable fallback | **CONFIRMED** by runtime shapely check (area attr exists, == 0.0); §1 #22 |
| INC-18 | SIM-27 nodata elevation in cascade | **CONFIRMED** by re-read (also default 0.0 on any exception); §1 #26 |
| INC-19 | EWD-16 one-sided vs two-sided z | **RESOLVED — consistent.** basin_volume_battered expects a per-side inset rate (PRM-15) and both production callers supply exactly that (W5-EXTRA-01). No factor-2 error. The naming hazard remains for future callers |
| INC-20 | CAT-27 hyetograph misclassification | **CONFIRMED** by re-read; constant-rate storms are the canonical false positive; §1 #25 |
| INC-21 | IDF-07 mean-vs-peak tension | **ADJUDICATED — code correct** (average intensity at duration = Tc is the standard rational method); the docstring sentence is the misleading half; §1 #29 |
| INC-22 | CAT-05/KPA-23 docstrings say D8 while code requests dinf; KPA-19 crest doc/code mismatch; SIM-15 comment says boundary cell, code takes domain max | quotes verbatim; LOW doc-fix items |
| INC-23 | SWL-17 overtopping freeboard | **CONFIRMED** — sole production caller passes freeboard=1.0 explicitly (earthworks.py:1755); §1 #27 |
| INC-24 | SWL-21 over-counter still called? | **RESOLVED — dead code.** Grep found zero production callers; the "100%-captured" bug source is fully retired |

---

## §5 Unverified register

**UNVER-B (blocked)** — rows whose named source was assigned to a terminated
verifier. Resume list, grouped by the source that would resolve them:

- ~~AMC conversions~~ **RESOLVED** (Chow-1988 family corroborated; Hawkins-1985
  family fetched, agreement ≤~1 CN). Still blocked: CAT-25 zone semantics, SIM-13
  temporal distribution (NEH-630).
- ~~TR-55 local PDF group~~ **RESOLVED** in the §0.1 local pass: CAT-19/ADV-03
  (DISC — Good-condition pasture row), TOC-04 (OK), TOC-05 (OK), TOC-09
  (OK floor / missing 10-hr cap).
- ~~EWD-35 weir C~~ **RESOLVED** (Brater & King via [BK-MODOT]: DISC, ~16 % under-width);
  ~~EWD-05 freeboard~~ **RESOLVED** ([NRCS378]: code is half the 0.30 m minimum).
  Still blocked: EWD-06/08/09/11 head-band corroboration, EWD-29, REG-01..06, ADV-01.
- ~~CAT-13/15 Lancaster~~ **RESOLVED** (ranges match; Sonoran-relabel caveat).
  Still blocked: PKF-02, SWL-01, SIM-01/02, PRM-07, KPA-09/29/31 (Yeomans/FAO).
- ~~IDF-07 / PKF-05 method statement~~ **ADJUDICATED** (code correct). Still
  blocked: PKF-07 attenuation refs, IDF-01/02/06 HIRDS corroboration.
- **pysheds / Tarboton / O'Callaghan-Mark / rasterio / GDAL docs (still blocked):**
  FLA-01/02/03/05/09/15/24, CAT-03/05/07, FLG-04/07/08/10/15, FTP-01/03/07,
  KPA-36, EWD-51, BRN-01/02, SWL-20. (W8 RESOLVED: FLA-19/20, DEM-04/05,
  CTA-07/10, FLL-04/08/09/15/16, KPA-12/14/28/32 + KPA-04 by extension of DEM-04.)
- **Shrink/swell & earthwork manuals:** EWD-41's 0.75, RPT-21, EWD-56, BRN-06 corroboration.
- **Code-only (no web needed — resumable immediately):** every §4 INC-## candidate,
  WBL-02/03/04/09/10, SIM-03/05/06/15/17/24/26/27/29, RPT-06/07/08/10/16-20/25,
  IDF-11, TOC-18, CTA-13/20/28, KPA-15/19/21/22/23/33/34/37, EWD-15/16/25/58 detail,
  SWL-17/21, REG-06, FLA-15 offset check, KPA-03 offset confirm.

**UNVER (no publishable source exists — design constants):** FLA-04/11/23, CAT-01/09/14,
WBL-05, PKF-03, TOC-08, SWL-13/15, EWD-01/03/14/40/45/54, BRN-06, CTA-16/17/18/21/22/25,
KPA-06/08/10/17/18/27, FLL-01/13, RPT-01/04/09/22, ADV-02. These are project decisions;
the audit records them so they are chosen knowingly, not silently.

---

## §6 Derivation appendix (A1 — completed in full)

The full step-by-step derivations are preserved verbatim below the fold of this
section in the assembling session's records; the load-bearing results:

- **ALG-01** S = 25400/CN − 254 mm is exactly (1000/CN − 10) in × 25.4. CN 61 → 162.393 mm.
- **ALG-02** dQ/dP = u(u+2S)/(u+S)² derived by quotient rule from Q = u²/(u+S), u = P−0.2S,
  du/dP = 1. Docstring numerics all reproduce: 0.578 / 0.255 / 0.269.
- **ALG-03** 0.007·(1/0.3048)^0.8·25.4^0.5 = 0.091265 (code's ≈0.0913 correct to stated s.f.;
  dividing by (1/25.4)^0.5 ≡ multiplying by 25.4^0.5, bit-identical).
- **ALG-04** mm/hr → m/s = ÷3.6×10⁶ exactly; worked check 36 mm/hr on 100 ha → 10 m³/s both routes.
- **ALG-05** L = Q/(C·H^1.5) exact inversion; C carries m^0.5/s; L in metres. Zero/neg head → 0.0 return is an input-validity polarity to watch.
- **ALG-06** 1:1 triangle of height h: base 2h, area h² (EWD-27 exact). Companion berm h=√(0.75A) → h² = 0.75A conserves volume with 25 % loss (EWD-28 self-consistent). Capacity credit h·T/2 differs by T/(2h) — ×1.33 at registry defaults → ~+14 % capacity claim (the §1 #4 finding).
- **ALG-07** Exact battered rectangle: V = LWd − (L+W)zd² + (4/3)z²d³; linear-inset model omits +(4/3)z²d³ → under-estimate; 10×10 z=1 d=1: 80.0 vs 81.333 (code comment exact).
- **ALG-08** Prismoidal is Simpson on quadratic A(t) → EXACT for linear batters; reproduces 81.333 with the code's own mid-area; non-square check 171.333 ✓.
- **ALG-09** Cone-frustum formula exact only for similar sections; on 20×10 z=1 d=1 under-reads by 0.06 % (not used by shipped code — prismoidal is).
- **ALG-10** g′(u) = 2S²/(u+S)³ > 0 → Q convex through (Ia, 0) → dQ/dP ≥ Q/P strictly → marginal-C spillway sizing is conservative; docstring's "undersize by more than half" checks out (0.255/0.578 = 0.44).
- **ALG-11** T_true = ∫dV/(f·A(h)) > V/(f·A_top) whenever the pond narrows on emptying → drain-down hours under-stated → puddling warnings under-trigger (§1 #5).
- **ALG-12** tan 18° = 0.3249 = 1/3.08 — "approx 1:3" accurate to ~2.5 %.
- **ALG-13** log-log interpolation ≡ p₀(d/d₀)^k, exact at knots, monotone, no overshoot; guards cover the log-domain failure cases.
- **ALG-14** Level-invert identity: burned volume = A·d exactly iff interior ground ≥ floor everywhere; hollows below floor ADD storage (analytic under-states, conservative); a rim gap below spill flips the sign (over-statement) — the one condition to watch.
- **FLA-21 / PRM-11/12/15/17/19 / FTP-10 / SWL-11 / TOC-02** — all CORRECT; see §2 rows.

## §7 Source register (sources actually fetched and read this audit)

| Key | Title / edition | URL | Grounds |
|-----|-----------------|-----|---------|
| [TR55] | USDA-NRCS TR-55, *Urban Hydrology for Small Watersheds*, 2nd ed., June 1986 (210-VI-TR-55) — full PDF (local copy in session scratchpad: tr55.pdf) | hydrocad.net/pdf/TR-55%20Manual.pdf | CAT-22/26, IDF-03, TOC-01/03/12/16, NEW-W5-02, PRM-04 |
| [HEC15] | FHWA HEC-15, *Design of Roadside Channels with Flexible Linings*, 3rd ed., Sept 2005 (FHWA-NHI-05-114) — full PDF (local: hec15.pdf) | fhwa.dot.gov/engineering/hydraulics/pubs/05114/05114.pdf | PRM-04/08/09, TOC-14, EWD-31, ADV-02 |
| [NASH] | Metropolitan Nashville–Davidson County Stormwater Management Manual, Vol. 2 Ch. 3 *Open Channel Hydraulics*, May 2000 (reproduces Chow's elements + worked examples) | nashville.gov/sites/default/files/2021-08/SWMM-Vol2_Ch3.pdf | PRM-01/02/03, SWL-03/04/08, EWD-24/26/31/32/33 |
| [FAO-WT] | FAO, *Water Transport Structures* (Simple Methods for Aquaculture, x6708e ch. 8) — Table 35 permissible velocities, slope + freeboard guidance | fao.org/fishery/static/FAO_Training/FAO_Training/General/x6708e/x6708e08.htm | ADV-02, SWL-04/06 |
| [FAO-WM] | FAO Watershed Management Field Manual (AD083e), diversion ditch chapter | fao.org/4/ad083e/AD083e11.htm | ADV-02 |
| [LMNO] | LMNO Engineering, Manning's n coefficients (cites Chow 1959) | lmnoeng.com/manningn.htm | EWD-31, TOC-06 |
| [BK-MODOT] | Brater & King (1976) broad-crested weir C table (English units, C 2.34-3.32 as f(head, crest breadth)), reproduced by Missouri DOT EPG Fig. 749 — full table text extracted this session; SI = English × 0.552 | epg.modot.org/files/b/bc/749_Broad-Crested_Weir_Coefficients.pdf | EWD-35 |
| [NRCS378] | NRCS Conservation Practice Standard 378 Pond (NHCP 2022 + state editions) — "minimum of 1.0 feet of freeboard between design high-water-flow elevation in the auxiliary spillway and the top of the settled embankment" | nrcs.usda.gov/sites/default/files/2022-09/Pond_378_NHCP_CPS_2022.pdf | EWD-05 |
| [LANC] | Brad Lancaster, "Water Harvesting Calculations" resource page (fetched live) — coefficient ranges: Metal 95 %, Concrete/Asphalt 80-95 %, Tar Roof 85 %, Sonoran Desert Uplands 20-70 %, Bare Earth 20-75 %, Grass/Lawn 5-35 % | harvestingrainwater.com/resource/water-harvesting-calculations/ | CAT-13/15 |
| [PONCE] | Ponce, Hawkins et al. (1985) AMC conversion reproductions ("Runoff Curve Number: Has It Reached Maturity?") | ponce.sdsu.edu/runoff_curve_number_has_it_reached_maturity.html | CAT-18 |
| [TXDOT] | TxDOT Hydraulic Design Manual, Rational Method / Rainfall Intensity (HEC-22-family practice: i = average intensity for duration = Tc from IDF) | txdot.gov/manuals/des/hyd/…/rational-method | IDF-07, PKF-05 |
| [ALG] | Internal derivations, this audit (§6) — every step shown; short numerics spot-recomputed by the assembling session | — | all ALG rows + 9 assigned IDs |
| [HORN81] | Horn, B.K.P. (1981) "Hill Shading and the Reflectance Map", Proc. IEEE 69(1) pp. 14-47 | people.csail.mit.edu/bkph/papers/Hill-Shading.pdf | DEM-04 |
| [GDAL] | GDAL gdaldem documentation (Horn default; aspect/nodata conventions) | gdal.org/en/latest/programs/gdaldem.html | DEM-04/05, FLL-16, CTA-10 |
| [ESRI] | ESRI ArcGIS Pro "How Slope works" | doc.esri.com …/how-slope-works.html | DEM-04, FLL-09 |
| [NUMPY] | numpy.gradient reference (default unitary spacing; central differences) | numpy.org/doc/stable/…/numpy.gradient.html | FLL-04/08, CTA-10 |
| [SCIPY] | scipy savgol_filter + ndimage.binary_erosion references | docs.scipy.org | KPA-14/28 |
| [SKIMAGE] | skimage.measure find_contours + morphology references (native NaN masking; skeletonize definition) | scikit-image.org/docs/stable/api | CTA-07, KPA-14/32 |
| [TAUDEM] | TauDEM 5 "D-Infinity Flow Directions" (Tarboton 1997 convention) | hydrology.usu.edu/taudem/taudem5/help53/DInfinityFlowDirections.html | FLA-19 |
| [PYSHEDS] | pysheds README (default dirmap, dinf) | github.com/mdbartos/pysheds | FLA-20 |
| [JENNESS] | Jenness (2006) TPI documentation v1.2, citing Weiss (2001) | jennessent.com/downloads/TPI_Documentation_online.pdf | KPA-12 |
| [CODE] | Code re-read during this audit (inventory agents' verbatim excerpts + assembling session reads) | — | MAIN rows |
| [SI] | Definitional unit conversions (SI/metric definitions; international ft/in definitions are exact per the 1959 agreement) | — | conversion rows |

Failed fetches recorded by completed verifiers: Oregon State FSL Manning tables (404);
Colorado State CIVE-401 notes (403); nrcs.usda.gov TR-55 mirror (timeout — hydrocad
mirror used instead). A directly quotable reproduction of Chow Table 2-1 was not
obtained; the trapezoid elements were verified against [NASH]'s printed formulas and
worked example plus [TR55]'s r = a/p_w instead.

---

## §8 Completeness reconciliation

Master checklist: 401 formula-site IDs across 20 files, plus UNI/INC/ALG series.
Every checklist ID appears exactly once in §2 (per-module tables). Coverage:
~150 MAIN rows audited by the assembling session; 19 verified by W5 (open-channel);
16 by W8 (terrain derivatives); 9 + 14 ALG by A1 (algebra); ~10 more resolved by
the local TR-55 pass (velocity constants, n-table, Tc floor/cap, CN-table
provenance, eq. 3-3, 300-ft statement, bankfull convention, Ia = 0.2S). The
remainder carry UNVER-B/UNVER with named resume sources in §5. No checklist ID was
dropped. NEW findings raised during verification: NEW-W5-01/02, NEW-W8-01..05
(§1/§2). One file (`modules/project_io.py`) post-dates the inventory sweep; it
contains no formulas (persistence defaults only) and is covered as PIO-01
(§2.18b) with a defaults-drift hazard note.

**Confirmed-findings count (after the 2026-07-31 completion pass):** 9 WRONG —
DEM-05 (slope nodata never written), KPA-14 (erosion erases ridges), KPA-21
(valley-scan `break` biasing dam sites west), RPT-07 (KeyError with flow data),
SIM-26 (line features get zero infiltration area), CAT-27 (constant-rate
hyetographs misread as cumulative), SIM-27 (nodata/0.0 elevation becomes the
universal cascade receiver), KPA-33 (fallback keyline drawn down the fall line),
and SWL-17's caller inconsistency — plus ~20 DISC, led by **EWD-35 (weir C=1.7
→ spillway width under-sized ~16 % vs the Brater-King broad-crest band 1.44-1.49
SI)** and **EWD-05 (0.15 m freeboard = half the NRCS-378 minimum)**, EWD-32
(legacy diversion R, +25 % Q), the berm capacity/fill split (+14 %), and the CN
Good-condition assumption. Resolved in the code's favour: IDF-07 (the pipeline is
standard IDF-at-Tc practice), CAT-18 AMC constants, CAT-13/15 Lancaster ranges,
INC-19 basin z convention, INC-01 SCS duplication (numerically identical),
SWL-21/INC-06 (over-counter and unused primitives confirmed dead). The former
top open item EWD-35 is now CLOSED; no CRITICAL-tier unknowns remain in the
production sizing paths.

**Resume note:** re-running the §5 groups (the still-blocked verifier scopes)
completes the audit; the master checklist with per-ID owner assignments is
preserved in the session scratchpad (`audit_master_checklist.md`) and the
TR-55/HEC-15 PDFs are downloaded there. The code-only §4 candidates (INC-09/14/15/
17/18/20 in particular) need no web access and are the cheapest next step.

---

## §9 Remediation register (2026-07-31)

All four tiers of the triage were implemented in one pass. Full suite green:
**1520 passed, 95.30 % coverage** on `modules/` (gate 95 %), ruff clean, CI
deprecated-API grep gate clean.

### §9.0 One finding WITHDRAWN on re-check

| ID | Original claim | Why it was wrong |
|----|----------------|------------------|
| **SWL-17** (§1 #27) | "Sole production caller passes `freeboard=1.0` while the sizing paths use 0.8 → overtopping under-warned" | `ew.capacity_m3` comes from `calculate_capacity()`, which **already** multiplies by the 0.8 freeboard (swale branch). Passing 1.0 at `earthworks.py:1755` prevents a *double* discount, exactly as its own comment states. No change made. The confirmed-WRONG count is **8, not 9**. |

### §9.1 Tier 1 — the confirmed bugs

| ID | Fix |
|----|-----|
| RPT-07 | Reads `total_inflow_m3` (what `water_balance` actually emits) via `.get`, ending the KeyError on every live render with flow data. The reporting test fixtures had been written to match the *reader*, not the producer, so they agreed with each other and with nothing shipped — replaced with a test that drives `run_water_balance` end-to-end. |
| SIM-26 | Tests `area <= 0` instead of `hasattr(geom, "area")`. Every shapely geometry has `.area`; a LineString's is 0.0, so the `length × width` branch was unreachable and swales / diversions / berms all got a zero wetted footprint. |
| SIM-27 | New `EarthworkStore.elevation_known` flag, set only when the DEM yields a finite non-nodata value. Unknown-elevation stores are excluded from the receiver candidates, from the elevation heuristic, and from the SIM-17 upstream subtraction; new `_top_down_key` sorts them last. An explicit user link is still honoured unless both elevations are known and contradict it. |
| CAT-27 | New `_looks_cumulative()`. A running total must be non-decreasing **and** gain more than half its final value across the record. `5,5,5,5` now reads as 20 mm, not 5 mm; genuine cumulative traces, including ones starting mid-storm, still read correctly. Ambiguity resolves toward incremental — the over-sizing direction. |
| KPA-21 | `continue` on the west-bound test, `break` retained on the east (dc ascending). Pond-site recommendations no longer collapse to the raster's left edge. |
| KPA-33 | `cx, cy = dz_dr, dz_dc` — the map-space contour perpendicular. The old `(−dz_dc, dz_dr)` was −∇z, the fall line, 90° out. |
| KPA-14 | New `_thin_to_centreline()` using `skimage.morphology.skeletonize`, with the historical erosion kept only as a no-skimage fallback. Ridges ≤6 cells wide survive; this also repairs KPA-15, since a skeleton's pixel count really is its length in cells. |
| DEM-05 | `compute_slope_raster` writes its declared `−9999` wherever the slope is non-finite, so a clipped DEM's masked region no longer renders as valid slope. |

### §9.2 Tier 2 — nodata artefacts

| ID | Fix |
|----|-----|
| DEM-04 | `slope_degrees` adopts the ESRI convention: a NoData *neighbour* takes the centre cell's value (window reads flat) instead of 0.0 m (window reads a ~2000 m cliff); a NoData *centre* returns NaN. Also a per-cell divisor — edge replication halves the sampled separation, so border rows/columns now divide by 4Δ, not 8Δ. A 45° scarp at the DEM edge previously read ~27°. |
| KPA-03 | `_rc_to_xy` uses `(row + 0.5) * transform.e`. The old `+ cell_h/2` moved *against* the negative `e`, putting every keypoint, ridgeline and pond site one full cell north — and disagreeing with the sibling conversion (KPA-24). Closes UNI-15's two-versions row. |
| KPA-12 | NaN-aware neighbourhood mean (`uniform_filter` of values ÷ `uniform_filter` of the validity mask). Filling with the whole-DEM mean fabricated a ridge line around every data boundary. The resolution-dependence of the cell-count window is now documented rather than silent. |
| NEW-W8-01 | Thalweg profile drops nodata samples and bridges them by interpolation instead of substituting 0.0 m. One nodata cell was a full-terrain-height cliff, and the keypoint is the argmax of the profile's *second* derivative. |
| NEW-W8-03 | The `np.nanmin` calls are gone with the gradient fills (see FLL-04). |
| CTA-07 / KPA-32 | Both marching-squares paths pass NaN straight to `find_contours`, which excludes it natively and leaves boundary contours open. The `elev−1` / `elev−1e6` pre-fills had walled every hole so the contours closed along the data edge — artefacts indistinguishable from real contours in the picker. KPA-32's nearest-component test now measures in metres, not mixed row/col cell units. |
| CTA-08 / NEW-W8-05 | Contour row/col → map uses the signed transform on both axes. |
| CTA-13 | A contour with no valid slope sample anywhere returns NaN, not 0°. Reporting 0° made a line lying entirely over nodata the *flattest* on the site, sailing through a filter built to reject unsuitable ground; NaN compares False, so unknown is now rejected. |
| CTA-10 | The old-NumPy fallback uses NaN propagation instead of `nan_to_num(…, 0.0)`, which had put the border spike straight back. Stale comment about `slope_degrees` corrected. |
| FLL-04/08/09/15 | New shared `_terrain_gradient()` passes cell size to `np.gradient` (its default is unit spacing → m-per-CELL) and leaves nodata as NaN. `min_grad` is now a true dimensionless grade rather than resolution-dependent, slope is `arctan(mag)` with no anisotropic `/cell_size`, and the DEM-minimum fill that dragged flow lines into the data boundary is gone. Both consumers guard on `isfinite`. |
| KPA-36 / INC-16 | The latent hazard is closed: dinf flow direction is kept as float32 with a `dinf_routing` flag instead of being cast to int32, which collapsed continuous radians to 0–6 and produced a `RuntimeWarning` on every call. |

### §9.3 Tier 3 — design constants (these change sizing output)

| ID | Change | Effect on existing designs |
|----|--------|---------------------------|
| **EWD-35** | `BROAD_CRESTED_WEIR_C = 1.45` (was 1.7), named and sourced to Brater & King's 2.60–2.70 English at 0.20–0.50 m head, crest breadth ≥0.6 m | **Spillways ~17 % wider.** 1.7 is the sharp-crested/ideal ceiling and does not apply to a wide earthen crest. Help text updated. |
| **EWD-05** | `SPILLWAY_MIN_FREEBOARD_M = 0.30` (was 0.15) — the NRCS CPS-378 minimum of 1.0 ft | Highest acceptable crest drops 0.15 m; some existing dams will now report insufficient freeboard. That is the honest reading. |
| **EWD-32** | Legacy diversion wetted perimeter comes from the section's own batter (z = 2 → √5·d slant), not the mismatched √2 | **Reported Q drops ~22 %** on the legacy path. The section was always z = 2; only the perimeter disagreed. |
| **NEW-W5-01** | `Earthwork.summary()` passes `bottom_width_m`, as the dialog does | Feature list and properties dialog now quote the same discharge (they differed ~45 %). |
| **EWD-19 / ALG-06** | Companion-berm capacity credit is `h²` — the berm's own 1:1 triangle, the same section `calculate_fill_volume` builds — not `h × T/2` | Claimed swale capacity with a companion berm falls ~12.5 %, to what the berm has material to hold back. |
| **TOC-09** | `MAX_VALID_TC_HR = 10.0` enforced alongside the 0.1 h floor, with its own warning | Only bites on flow paths over 10 h, where a long Tc was reading a low intensity — the under-sizing direction. |
| **SWL-03** | New `_channel_section()` caps the effective depth at `T/(2z)` when the walls converge | Storage in the converged corner falls to the true `T²/(4z)` maximum. No effect on ordinary sections. |
| **CAT-19 / ADV-03** | Provenance documented (TR-55 Table 2-2, pasture, **Good** condition); `SCSRunoff.soil_reference_cn(soil, condition)` exposes the Fair and Poor rows, and a **Ground condition** picker in the Baseline section now drives the CN auto-fill (§9.7) | **No numeric change by default** — Good is preselected and reproduces every previous curve number. Degraded ground can now be chosen deliberately: Poor is up to 29 CN higher. |

### §9.4 Documentation only

- **NEW-W5-02 / TOC-01:** the 100 ft sheet cap is attributed to NEH-630 Ch. 15 (2010); TR-55 (1986) itself says 300 ft. Behaviour unchanged — 100 ft is current guidance and the conservative choice.
- **IDF-07 / PKF-05:** `rational_peak_flow`'s docstring now states that intensity is the IDF value **at duration = Tc** (standard practice, and what the pipeline supplies), with the whole-storm-average warning kept as the thing to avoid. The old wording contradicted the correct pipeline.
- **INC-22:** `catchment.fast_contributing_area` and `YeomansKeylineAnalysis` no longer claim D8 where the code requests dinf; the `fdir_path` parameter documents the radian encoding; KPA-19's crest note says *candidate cell* elevation +2 m, which is what the code does; SIM-15's comment says domain-wide max, not boundary cell.
- **ADV-03:** the duplicated CN table carries a mirror-me warning and a note that the condition assumption is harmless there (nearest-match lookup, not a runoff figure).

### §9.5 Raised during remediation, not in the original audit

1. **`_rc_to_xy` half-cell sign error** (KPA-03) — the audit had it as "pending confirm"; confirmed and fixed. Everything the keypoint module drew was one cell north.
2. **CTA-13 polarity** — was UNVER-B; an all-nodata contour scoring 0° passed the slope filter. Fixed.
3. **Reporting test fixtures encoded the bug** — four `per_feature` fixtures used the non-existent `inflow_m3` key, which is why RPT-07's KeyError survived a 95 %-covered test suite. Worth a sweep for other fixtures written against a consumer rather than a producer.

### §9.6 Still open

- Every remaining **UNVER-B** row in §5 (pysheds/Tarboton semantics, Yeomans/FAO texts, HIRDS corroboration, registry defaults, shrink/swell factors).
- **UNVER** design constants (§5) — unchanged, and still project decisions rather than defects.
- **CAT-17 / PRM-19 / ALG-11** drain-down constant-area simplification — left as the documented conservative simplification it is.
- **SWL-04 / SWL-06** infiltration top-width and the 0.8 area-fraction freeboard — nonstandard form, conservative direction, unchanged.
- ~~CAT-19 panel wiring for pasture condition~~ — **done**, see §9.7.
- **EWD-26** diversion cut-volume comment says "1:1" for a z=2 section: the *maths* is now consistent with the section throughout, but the bed-vs-top width convention clash (REG-06) is still deferred, per the scope agreed for this pass.

### §9.7 CAT-19 follow-up — ground condition picker (2026-07-31)

The condition the CN table had always assumed is now asked for rather than assumed.

| Layer | File | Change |
|---|---|---|
| Modules | `modules/catchment.py` | Public `GROUND_CONDITIONS` (key → label + what it means on the ground) and `DEFAULT_GROUND_CONDITION = "good"`, so the picker does not invent its own strings |
| UI | `panel.py` | **Ground condition** combo in the Baseline grid, directly under Soil Type; `ground_condition` property; `_refresh_cn_from_soil()` refills the CN spinner when either input changes |
| UI help | `qgis/help_text.py` | New `GROUND_CONDITION`; `CURVE_NUMBER` now shows the full 5×3 table instead of the Good column alone; `SOIL_TYPE` notes it no longer decides CN by itself |
| Persistence | `modules/project_io.py` | `ground_condition` field, registered in `_ENUM_FIELDS` — a bogus value falls back to Good rather than selecting a table at random |
| Controllers | `qgis/controllers/simulation.py` | See below |

**Default is Good, so nothing renumbers.** A design written before the field existed
reloads as Good and reproduces the curve numbers it was sized against
(regression-tested). The condition drives the CN spinner exactly as soil texture
already did — visible in the UI, still overridable, and a typed CN survives a save /
reload because `apply_inputs` restores with signals blocked.

**Two defects found and fixed while wiring it:**

1. **A fourth copy of the CN table.** `panel.py` carried its own hard-coded
   `_SOIL_CN = {"Sand": 39, …}` for the soil→CN auto-fill. The audit's INC-07 counted
   three copies (catchment / advisories / swale alias) and missed this one — the one
   that actually feeds the UI. It would have gone on answering "Loam is 61" after the
   module learned that Loam is 61 *only on well-covered ground*. Replaced with a call
   to `SCSRunoff.soil_reference_cn`.
2. **The simulation ran a different storm from the baseline.**
   `controllers/simulation.py` derived its CN as `SOIL_REFERENCE.get(soil_name, 70)`
   while every other controller uses `self._panel.cn`. That discarded any typed CN
   override — and would have discarded the ground condition too — so the simulation
   could score a design against a storm the baseline never used. Now reads
   `self._panel.cn` like its siblings.

A cross-file test asserts the three places that name these conditions (picker table,
CN tables, design-file enum) stay in step, since drifting apart is silent: an extra
picker entry would quietly select Good, and a missing enum value would refuse to
reload a saved design.

`panel.py` is coverage-omitted and needs a QGIS runtime, so the widget itself is
unverified by pytest — **it wants a manual smoke test in QGIS**: check the combo
appears under Soil Type, that changing it moves the CN spinner (Loam: 61 → 69 → 79),
and that a save / reopen restores the choice.
