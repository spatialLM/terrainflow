# Field-test log

Findings from driving the plugin on a real site, and what came of each. Kept so a
later session does not re-diagnose something already settled, and does not assume a
still-open question was closed.

---

## Round 1 — 2026-08-04

| Finding | Outcome |
|---|---|
| Compare crashed: `UnboundLocalError: 'chosen'` | **Fixed.** `chosen` / `by_id` were scoped inside a guard and read outside it, so Compare died whenever no baseline had run. |
| Dam rows showed Δ +1712% / +1553% | **Fixed.** `calculate_capacity` returns 0 for dams — they impound against terrain, not a drawn section — so Δ measured against a bogus rasterised trench. Dams are now `barrier_impounded`: Geometric/At-grid collapse to one figure and Δ compares the burn against the flooded-volume calculation. Those dams were verifying at ~0%. |
| Selecting a feature in Live Assessment did not show which one | **Fixed.** `NetworkView.selection_changed` was connected to nothing; now relayed to a canvas rubber band. |
| Spillway invisible; connections ran centroid-to-centroid | **Fixed.** Split into outflow + inflow spillways (`ew.spillway` / `ew.inflow_spillway`, with `outflow_spillway` as an alias so old projects load). Connections anchor outflow → inflow where both are placed. |
| Flow chart should arrange by elevation | **Built.** Toggle is now List / Flow / Elevation, with a metre scale down the left. |

## Round 2 — 2026-08-04

| Finding | Outcome |
|---|---|
| `QgsGraduatedSymbolRenderer.createRenderer` crash in `_display_inflow_gradient` | **Fixed.** Cannot classify a degenerate range. Now: bail on zero stretches, cap classes at the distinct-value count, fall back to a single symbol for one value. Also clamped `inflow` before `log10`. |
| Compare showed `0.00 ha` and `0.0 L/s` on every row | **Fixed.** No earthworks meant zero catchment area. Falls back to `baseline_result["domain_area_m2"]` labelled *(whole site)*, and says so plainly when there is no baseline either. |
| "Can't draw swale segments" after running analysis | **Message improved, cause unconfirmed.** Contour Analysis is a separate button from Baseline; the old text did not distinguish "never run" from "layer deleted". Both cases now name the actual precondition. |

---

## Round 3 — 2026-08-05

| Finding | Outcome |
|---|---|
| Swales reading +50% in Design vs Measured (open since round 1) | **Fixed.** Cause below. The companion-berm hypothesis was wrong. |

**Swales reading +50% — settled.** The round-2 design (`05.08.26 Test 1.tfd`, 8 features)
gave a clean split:

| | depth | top width | Measured ÷ At-grid |
|---|---|---|---|
| Swale 1, 2 | 0.50 m | 2.00 m | 1.000, 1.000 |
| Swale 3, 4, 6, 7, 8 | 1.00 m | 3.00 m | 1.501, 1.499, 1.454, 1.502, 1.627 |

**The berm hypothesis is dead** — *every* swale in that design has
`companion_berm = True`, including the two that verify at 0%. Berms cannot explain the
split, and the log should not have leaned on them.

The real cause was a contract mismatch between the burner and the figure it is checked
against. `DEMBurner._burn_storage` steps its walls (`battered_invert`) only when
`batter_run_m > 0`; otherwise it levels a flat floor at full depth (`level_invert`),
which is a **rectangle** however wide the footprint. A drawn swale carries
`batter_run_m = 0`. But `rasterisable_capacity` never consulted the batter run — it
chose on width alone, and any footprint ≥ 3 cells across was discounted by
`mean_width/top_width`. For a 3 m/1 m section that is 2/3, and `1 ÷ (2/3) = 1.5`.

The knife-edge is worth remembering: the guard was `top_width < 3.0 * cell_size`, so a
**3.00 m swale on a 1.00 m DEM** — an entirely ordinary combination — fell on the wrong
side by nothing at all, while the 2.00 m swales fell on the right side and verified
clean. `At grid` < `Geometric` on exactly the anomalous rows was the tell (round 1 read
this correctly as a symptom, then attributed it to the wrong cause).

**Fix:** `rasterisable_capacity` now takes `batter_run` and returns the rectangular
volume unless a batter was actually cut. Report-only — it has a single call site inside
`capacity_breakdown`, and plays no part in the burn, the ponding or the flow
re-analysis, so no hydrology moved. Regression test:
`test_an_unbattered_feature_stays_rectangular_however_wide`.

Residuals after the fix are real, not bugs: Swale 8 stays ≈ +9% because `level_invert`
references a flat floor to the pour point, so the uphill end of a long swale (463 m) is
cut deeper than the design depth; Swale 6 sits ≈ −3% because `np.minimum` never raises
ground already below the floor.

**Still open (parked, not urgent):** these swales have `top_width 3.0 / bottom_width 1.0
/ depth 1.0`, which *implies* a 1:1 batter, yet `batter_run_m = 0` — so the burner never
batters a drawn swale at all. Deriving the batter from the widths would make the burn cut
the true section, but it changes the burned DEM and therefore every downstream number.
Deliberately deferred.

---

## Round 4 — 2026-08-06 (spillway design review)

Driven from `Spillway_Design.pdf` — questions about what the dialog's spillway fields
mean. Two of the five turned out to be defects rather than documentation gaps.

| Finding | Outcome |
|---|---|
| "Is Rim the top lip of the swale? Does it include the berm?" | **Answered, label fixed.** Rim is the lowest cell of the ring *outside* the footprint, from the pre-earthwork DEM — where water would go if you dug the hole and built nothing. Not the dig point; for a dam it is the wall crest. It excludes the companion berm, which the burn *does* include — see STRETCH_GOALS §10(a) for why adding it would be the unsafe direction. Relabelled "Rim (natural spill level)" with the caveat stated on bermed swales. |
| "Is Crest elevation just the spill trigger, and is it used that way?" | **It is used in no calculation at all.** Read only by the dialog, the map label and `summary()`; routing comes from `overflow_target_id` + the D8 walk, and capacity from the drawn section. **Four** places claimed otherwise — `SPILLWAY_GROUP`, `SPILLWAY_CREST`, `test_spillway.py`'s module docstring, and `Spillway`'s own docstring about `auto`. All corrected; the dead, stale `SPILLWAY_DESIGN_FLOW` deleted. |
| "Should Head be an input at all? Peak flow determines it." | **Both framings are right; which applies depends on what is free.** Designing, the width is free so the head is chosen; once a width is committed the head is the consequence. `width_auto` already encoded that, so it now drives it: head is a *target* with the width solved, or the width is committed and the achieved head shown instead. Never both. |
| **Freeboard was checked against the typed head** | **Fixed.** With a committed narrow width the tool reported the width shortfall but reported freeboard as *fine*, because it checked the head the user asked for rather than the one the water will stand at. |
| **`width_auto` never actually tracked** | **Fixed — the significant one.** `Spillway.width_m` was written in exactly one place: inside `get_spillway()`, in the modal dialog. No controller wrote it, so a stored auto width went stale the moment the dialog closed, while `_check_spillway_capacity` skipped auto rows on the grounds that they "track by definition". True of the intent, false of the object. |
| "Is 0.3 m freeboard a hard requirement?" (user: *"realistically not for swales"*) | **Right, and it was breaking the default swale.** NRCS CPS-378's 0.30 m is a *pond embankment* figure. Applied to the registry's 0.5 m default swale it needs 0.60 m of a 0.50 m dig, so `spillway_datum` returned an inverted band and the most ordinary feature in the plugin reported *"the feature would hold nothing"*. Freeboard, default head and the typical-head band are now per-type in the registry (swale 0.15/0.15/0.10–0.30; dam and basin keep 0.30/0.30/0.20–0.50). Freeboard is editable, and 0 is allowed with an error rather than blocked. |
| "Lay the earthworks out, then review the spillways" | **Built.** New collapsed **Spillways** section in the Design stage. Every water-holding feature gets a row *whether or not a spillway is designed*, so the auto-sizing is visible before the user commits; the ▽/▲ markers site the outflow and inlet from the row. The header carries its state while collapsed. |

**Two things worth not re-deriving.**

*Interposing a feature does not shrink the sill below it.* `cascade_peak_flows` routes on
"a full feature passes its whole peak on", so a swale drawn directly above another moves
flow from the lower feature's *own* catchment to its *upstream* contribution and leaves
the total unchanged. What genuinely re-sizes a downstream spillway is flow redirected
*away* — routed elsewhere or off site. Pinned by
`check_a_feature_upslope_re_splits_the_downstream_flow`.

*Never invert a rounded width.* `calculate_spillway_width` rounds to a centimetre, so
`head_for_width` on its output returns a head 1.5–6 mm off — enough to push
`rim − crest` past `head + freeboard` and fire a freeboard warning whose own quoted
numbers satisfy it, which is the regression `_ELEV_EPS` exists to prevent. `effective_head_m`
therefore inverts **only** when the built width is genuinely short of the requirement;
where it is adequate the design head is correct by construction anyway.

## Round 5 — 2026-08-12 (Design vs Measured reads as a capacity)

| Finding | Outcome |
|---|---|
| Swale 33: Design 291, Geometric 364, At grid 453, Measured 453, **Δ +0% in green** | **Fixed, presentationally.** Δ compares Measured against At-grid, so it was green about a comparison the reader was not making. The real gap — Geometric 364 → At-grid 453, +24% — sat in a different pair of columns with nothing marking it. |

**The gap is not resolution, and the old copy said it was.** Four places claimed the cell
size caused it: `help_text.VERIFICATION_TABLE`, the widget footer, `build_verification`'s
caveat, and `verification_sentence`. `_burn_storage` steps its walls (`battered_invert`)
only when `batter_run_m > 0`, and `batter_run_m` is a **basin-only** field — so every
drawn linear feature is levelled to a flat floor at full depth across the whole
`all_touched` footprint. `rasterisable_capacity` correctly models that as
`n_cells × cell_area × depth`, *independent of cell size*. Halving the DEM resolution
would not move the number by a cubic metre. Telling a user to find a finer DEM would have
sent them after a gap it cannot close. All four now describe the flat-floored trench.

**Why the flag is not a width test.** The obvious rule — `top_width < 3 × cell_size`,
mirroring the guard already inside `rasterisable_capacity` — is wrong twice over here. It
misses a wide drawn swale, which is misrepresented exactly as much as a narrow one (with
no batter run the burn lays a rectangle at any width), and it fires on a genuinely
vertical-walled channel whose raster does match its design. The trigger is the **measured**
gap instead: `|resolution_penalty_m3| / geometric > 0.10`, the predicate
`_RESOLUTION_CAVEAT_THRESHOLD` already gated the caveats with, now also carried per row as
`section_overstated` so the panel and the report style from one decision.

**What changed on screen.** A one-line subhead above the table naming Geometric as the
capacity; per-column header tooltips; on a flagged row the Δ keeps its number but loses
the green — informational blue plus a `†`. Only the *pass* colour is withheld: amber and
red survive, because the flag is about which volume was compared, not how badly the burn
missed it. The report says the same in words, since print has no hover.

**Also fixed, found in the screenshot.** `panel.set_verification` gated the "re-analyse
to compare" placeholder on `_verification_table.isVisible()`, which is False for any
widget on a stage page that is not the active one — and verification is computed from the
Design stage. The placeholder sat above four populated rows until something else redrew
the section. Now `isHidden()`.

**Still open (parked, not urgent):** the burn and the verification measurement rasterise
different geometry. The burn buffers a line by exactly `top_width/2`
(`_burn_swale`); `_compute_verification` buffers by `max(width/2, cell_size)`. For any
swale narrower than `2 × cell_size` the measuring mask is wider than what was cut, which
inflates `n_cells`, inflates the At-grid reference, and biases Δ negative. Real, but
correcting it moves published numbers, so it was left out of a display-only change.

Round 3's parked item — *"the burner never batters a drawn swale at all"* — is unchanged
and is what this round makes **visible** rather than resolves.

## Round 6 — 2026-08-12 (Stage 1: the measurement, not the burn)

Reported: on the Quail Island design (42 features, 1 m DEM), At grid and Measured run
above Geometric almost everywhere, with three features far out — Swale 27 +94%, Basin 39
+211%, and a suspicion that a 1027×858 / 2157×1319 raster mismatch invalidated every Δ.

**The run was reproduced from its own artefacts** rather than reasoned about:
`%TEMP%\tfa_3isxmku2\*.tif`, the `.tfd`, and the source `BX24.tif` were all still on
disk, and re-deriving the table from them returns the reported figures exactly
(Swale 22 = 419/524/628/670; Swale 27 = 145/181/232/451; Basin 39 Δ +211%). Every number
below is measured from that run. **Do this first next time** — it took four checks and
settled in minutes what the code alone could only support guesses about.

| Finding | Outcome |
|---|---|
| Basin 39 Δ **+211%** | **Fixed.** Not a burn error and not isolated. Region 44 is one 3,593 m³ pool touched by Basin 39 (766 cells) **and Dam 40** (70) — the dam wall sits on the basin's downhill lip; they are one structure drawn twice. `attribute_ponding_volume` awarded the whole pool to the larger overlap, so Basin 39 read +211% and **Dam 40 read −100%** for the same water. Two more pairs did the same (Basin 14 / Swale 13 at 1,612 m³; Basin 2 / Dam 1 at 1,424 m³), and all three −100% rows were invisible in the report because nobody looks at the feature holding *too little*. Now: features sharing a pool are joined into a set — transitively, and carrying their solo pools with them — and the set gets one row. Basin 39 + Dam 40 → **+55%**, Basin 14 + Swale 13 → **+44%**, Basin 2 + Dam 1 → **+17%**, Dam 3 + Swale 6 → **+0.4%**. Members' Measured and Δ read "shared", the way a sub-cell feature reads "sub-cell". |
| The 1027×858 vs 2157×1319 mismatch "invalidates every Δ" | **Half right, and the smaller half.** Real bug, found: `design_file._apply_dem` set `dem_path` to the design's embedded **clip** but never rebuilt `state.burner`, which is written in exactly one place — `baseline.on_dem_changed`. Timestamps prove the split: `slope.tif` (2157×1319) and `restored_dem.tif` (1027×858) are both 14:32; baseline ran 14:44 on the clip, the burn 14:45 on the stale parent tile. **But** total baseline ponding is **592 m³ against 27,402 — 2.2%.** It moved exactly one row (Dam 15, +26.5% → **+0.0%**) and left Swale 22, Swale 27 and Basin 39 bit-identical. A cause worth fixing; not the cause of the complaint. |
| The two grids "need resampling to a common extent" | **No — and never.** They are the same 1 m cell, same CRS, an exact 66-row / 287-column sub-window. `dem_loader.align_to_grid` places one on the other by integer offset, or refuses. Interpolating a ponding *depth* field would invent water: a one-cell pool becomes two cells of half the depth, in the wrong place. |
| Spillway crest elevations missing from the pipeline | **Confirmed absent, and irrelevant here.** Round 4 already established the crest feeds no calculation. Swale 22, Swale 27 and Basin 39 all carry `crest_elevation: None`, and both Geometric and At grid are full-depth figures, so the crest cannot push one above the other. Not pursued. |
| Δ +94% / +211% are "isolated burn errors needing individual diagnosis" | **Wrong on both counts.** Basin 39 is the systemic attribution defect above. Swale 27 is a systemic *datum* defect — see Stage 2, next round. |
| The headline "57% held on site, 9,975 m³" overstates capture because it comes from Measured | **It does not come from Measured.** `capture_pct` ← `run_water_balance` ← `EarthworkStore.capacity_m3 = ew.capacity_m3` = **Design**. Nothing in the raster tier reaches it. It also checks out: 292,288 domain cells × 60 mm = 17,537 m³; 9,975 ÷ 17,537 = **56.9%**. Σ Design capacity is 17,171 m³, so the features are 58% full — capture here is **interception-limited, not capacity-limited**, and cutting design capacity would barely move it. |
| "Site is 284.5 ha" | **That is the DEM tile.** The analysis domain is **29.23 ha**, identical in both runs. The panel label read "Area: 284.5 ha" and was reasonably taken for the site. Now "DEM covers 284.5 ha", with Baseline printing "Analysed: 29.2 ha" once it knows. `_natural_ponding_m3` had the matching bug — it summed the whole raster while the score covered the domain — and now reuses the worker's already-domain-clipped `ponded_volume_m3`. |

**After Stage 1, on the real design:** no feature reads −100%; the site Δ is +12.1%;
32 of 40 features carry an independent Δ and the other 8 are named in four group rows.
Attribution is also ~30× faster (`np.bincount` over the label array instead of a
full-grid `logical_and` per region × feature — 669 × 40 × 2.8 M cells before).

**Not fixed by Stage 1, and expected to survive it:** Swale 27 (+94%) and Swale 22
(+6.6%) are unchanged, because their cause is the invert datum, not the measurement.
Stage 2 takes that.

**Guarded going forward:** `run_with_earthworks` now refuses to burn when the burner's
grid disagrees with the session DEM, `on_dem_changed` invalidates results when the grid
actually moves (not when the same raster is re-selected), and
`checks_design_file.check_opening_an_embedded_design_moves_the_whole_session_to_the_clip`
opens an embedded design with a larger DEM still selected and asserts burner, `dem_info`
and picker all agree.

## Round 7 — 2026-08-12 (Stage 2: the invert datum)

Round 6 left Swale 27 at Δ +94% and said the cause was the datum. It was.

**`pour_point` was read off the accumulating DEM.** `_burn_swale` and `_burn_basin` passed
the running array — the one every earlier feature has already been cut into — so a
feature whose rim clipped a neighbour's trench was floored to that trench. The datum is a
single minimum over a one-cell-wide rim, which makes it as sensitive as a measurement can
be: **Swale 27 shares exactly one rim cell with Swale 25**, cut to 71.78 m against a
natural 72.44 m. That one cell dropped its floor 0.66 m, and because the neighbouring
trench fills too, it ponded 1.94 m for a 1.00 m design. Reversing the feature list would
have changed the published number.

Fixed by taking the datum from `DEMBurner._datum_surface` — the original ground plus
**this feature's own** companion berm, never other features' cuts. The cut still lands on
the running array, so features stack as before; only the datum is isolated. The berm is
derived once (`_companion_berm` returns mask + height) and shared between the datum pass
and the burn, so this costs nothing.

| | before | after |
|---|---|---|
| **Swale 27** mean cut / design 1.00 m | 2.23 m | **1.28 m** |
| **Swale 27** Δ | **+94.4%** | **+0.0%** |
| **Basin 14 + Swale 13** Δ | +43.9% | **−6.4%** |
| Basin 39, Swale 22, Swale 25, Basin 2 | — | unchanged; their datums were already clean |
| worst solo Δ on the site | +94.4% | **+7.6%** (Swale 8) |
| solo Δ median / mean / max | — | **0.0% / 0.7% / 7.6%** |
| site Δ | +12.1% | **+6.4%** |
| burned cut | 29,290 m³ | 28,540 m³ |

**`steep_ground_warning` was gated on the wrong quantity** and this is why the over-cut
was never reported: `relief > depth` asks whether the footprint is steep *across itself*,
but the datum is the lowest cell of the **rim**, so a perfectly flat strip beside a dip is
floored to the dip. Swale 27 had 0.67 m of relief and was cut 2.23 m. Now gated on the
measured `cut_m3 / storage_m3 > 1.3`, with relief kept as a fallback for the case where
no cut has been measured yet. On this design the warning count went **6 → 18**, which is
the honest number.

**What the datum fix did *not* do.** Site cut fell only 2.6% (29,290 → 28,540 m³), so the
gap against the reported ~12,000 m³ is **not** contamination — it is the level invert
itself. A floor referenced to the lowest rim cell is deeper than the design depth almost
everywhere on real ground, and `calculate_cut_volume` assumes it is not. That is Stage 5,
and it is now measurable rather than guessed at.

**Basin 39 + Dam 40 is still +55%,** and is now the only large residual. It is not the
datum: the pair's cut is unchanged. Basin 39 has **2.65 m of relief across a 1.50 m
design**, so the level floor genuinely over-excavates it, and the pair impounds more than
`footprint × depth` plus the dam's flooded-volume figure. The new warning fires on it.
Left for Stage 3/4 rather than patched here.

Regression tests: `test_burn_order_does_not_change_the_result` (two swales sharing a rim
cell burn identically either way round) and
`test_over_excavation_is_warned_even_on_flat_ground` (one deep rim cell beside a flat
strip — the Swale 27 shape).

## Round 8 — 2026-08-12 (Stage 3: cut what was drawn)

The remaining gap, and the one the whole investigation was about: **At grid sat 2.10×
the drawn section.** Swale 22 is drawn 3.0 / 1.0 / 1.0 m over 149.8 m — a 2.0 m²/m
trapezoid, 299.6 m³ — and the terrain model reported 628 m³ for it. Three causes,
multiplying:

| | factor | why |
|---|---|---|
| Rectangle, not trapezoid | ×1.50 | `batter_run_m` is a **basin-only** field, so `level_invert` squared every drawn channel off to a full-depth rectangle whatever cross-section it carried |
| `all_touched` footprint | ×1.40 | the burn claimed every cell the buffer brushed — 4.19 m of trench for a 3.00 m swale — and levelled all of it to full depth |
| Berm folded into Geometric | ÷1.75 | 225 m³ of Swale 22's 524 m³ "Geometric" is the companion berm's own section, which the trench columns contain none of |

The third was hiding the first two. Against the trench alone the over-count is **+110%**;
against a Geometric carrying a berm it reads a mild +20%, and two large errors in
opposite directions cancelled into something that looked like rounding.

**What changed.**

1. **Volumetric burns rasterise on cell centres** (`DEMBurner._rasterize(all_touched=)`).
   `all_touched` answers *"did we lose the feature?"* and is kept for barriers, routing
   and placement; it is the wrong answer to *"how much earth came out?"*. Connectivity
   was checked before switching — every centre-based footprint tested, 1–3 m wide at
   0/17/45/63°, is still a single 8-connected component. On a grid axis it costs almost
   nothing (4.07 m vs 4.04 m); off-axis, which is what a contour swale is, it claims a
   whole extra cell.
2. **A drawn channel is cut as the section it was drawn as.** `channel_batter_run`
   derives the batter from the two widths the feature already carries — one derivation,
   used by both the burn and the reference, because the last time those two disagreed
   about whether a batter had been cut, every swale on the site read +50% (Round 3).
3. **The taper is a distance transform, not a staircase** (`burn_strategy.tapered_invert`).
   The nested-erosion `battered_invert` was tried first and is **16.7% under** the
   trapezoid it approximates at the shipped 3 steps — its volume is
   `depth/n × Σ Aᵢ = depth × L × [T − (T−b)(n+1)/2n]` — converging only as `1/n`, while
   the erosions themselves vanish once `batter_run/n` drops below a cell. Measured on
   the design it produced a −12% bias. The distance transform is exact, one O(N) pass,
   and degrades honestly: a footprint two cells across has no cell more than half a cell
   from its own edge, so it comes out shallower, which is the truth about that grid.
4. **At grid is now measured, not modelled.** `DEMBurner.burned_storage` is what the cut
   hole holds filled to its own spill level; `capacity_breakdown` prefers it and falls
   back to `rasterisable_capacity` only before a burn. A model standing beside a burn is
   exactly the arrangement that failed in Round 3 and would have failed again here.
5. **Geometric is split** into `section_m3` + `berm_credit_m3`, and the
   `resolution_penalty` is measured against the **section**, so a berm is never reported
   as a grid effect.
6. **The verification measures the burn's own mask.** It used to re-derive footprints by
   buffering `max(width/2, cell_size)` with `all_touched` while the burn buffered
   `top_width/2` — Round 5's parked item, now closed by `DEMBurner.burned_masks`.

**A barrier's mask is not its water.** Recording burn masks exposed this immediately:
`_downstream_footprint` puts a dam's wall on the far side of the drawn line (the drawn
line is the wet face), so the wall band and the pool it impounds share no cell at all.
Attributing by it gave **every dam −100%** — Dam 15 and Dam 40 measuring 0 m³ against
four-figure capacities, with their pools going to whichever basin overlapped them.
Barriers now record a `_contact_mask`: a symmetric band on the drawn line, `all_touched`,
because contact is a different question from volume.

**Measured on the Quail Island design, all three stages in:**

| | before | after |
|---|---|---|
| **At grid vs the drawn section, sitewide** | +101% | **+2.1%** (15,475 against 15,152 m³) |
| Swale 22 — section / at grid / measured | 300 / 628 / 670 | **300 / 308 / 322** (Δ +4.3%) |
| Swale 27 | 104 / 232 / 451 | **104 / 108 / 108** (Δ +0.0%) |
| Swale 25 | 262 / 569 / 537 | **262 / 263 / 263** (Δ −0.0%) |
| Basin 32 | 1,781 / 1,912 / 1,982 | **1,781 / 1,780 / 1,846** (Δ +3.7%) |
| solo Δ — median / mean / max | +94% max | **0.0% / 0.5% / 4.3%** |
| features reading −100% | 3 | **0** |
| rows flagged † | ~all bermed swales | **3 of 40** |
| burned cut / fill | 29,290 / 19,222 m³ | **17,535 / 12,722 m³** |
| site ponding | 27,402 m³ | 17,976 m³ |

Two group rows remain: **Basin 39 + Dam 40 at +45.6%** and **Basin 2 + Dam 1 at +17.3%**.
Basin 39 has 2.65 m of relief across a 1.50 m design, so its level floor genuinely
over-excavates; `steep_ground_warning` now fires on it. That is a design property, not a
measurement defect.

**A fixture bug found on the way, worth remembering:** `MagicMock` answers `float()` with
**1.0**, so every mock earthwork in `tests/test_dem_burner.py` had been silently claiming
a 1 m batter run and taking burn branches the real `Earthwork` (which defaults it to 0.0)
never would. Now set explicitly.

**Prose corrected in the same change,** because all of it described the pre-taper burn:
`STRETCH_GOALS.md` §1 (its "+33% because it cannot hold its sloping walls" example was
measuring a burn choice, not a cell size — the conclusion stands, the reasoning did not),
`rasterisable_capacity`'s docstring, `burn_strategy`'s module docstring, and the four
copy sites Round 5 rewrote, which told the user a drawn swale "is cut as a rectangular
trench at ANY cell size, so a finer DEM would not close that gap." True then; false now.

## Round 9 — 2026-08-12 (Stage 4: the companion berm, keyed and level)

Requested: *"add 'key into banks' as a selectable option, in a way that the edges of the
swales have the soil added to them also so that it pools all the water. Also the soil
from digging the trench should be put in a berm so that the top of the accompanying berm
is level. Also therefore the elevation of top of the berm can also be given."*

All three, and each turned out to be a defect as well as a feature.

**The berm was raised, not levelled.** `dem[berm_mask] += raise_height` put the crest on
the same slope as the ground under it, so the bank was tallest where the water was
shallowest and the pool ran out at its low end. Measured on the Quail Island design the
raised bank stood **1.44–3.53 m against a declared 1.22 m** and still impounded almost
nothing. Now `core/sizing/primitives.level_crest_from_spoil` solves for the level at which
the spoil is exactly consumed — sort-and-accumulate, one pass, exact to floating point
(the crest reproduces the spoil volume to ~1e-12 on random ground), no iteration or
tolerance — and `np.maximum` keeps it a fill.

**`key_into_banks` was dam-only, and the dam's version does not work for a swale.** The
plan was to reuse `extend_to_abutments` — walk out along the alignment's own bearing
until the ground rises to the crest. That is right for a wall thrown across a valley and
useless for a swale, which is laid *along* a contour: the ground off each end is at the
same elevation as the ground under the line, so the walk runs its full 250 m and keys
into nothing. **Measured: it changed the ponded volume by 0 m³.** What works is a
**return at each end** — a cap of bank around each endpoint, reaching from the downhill
berm across the end of the trench to its uphill lip. On planar 5% ground that took the
pond from **782 → 1,014 m³**.

**The berm was being sized from earth the trench no longer produces.** Stage 3 made the
cut a trapezoid; the spoil calculation still assumed a full-depth rectangle, so the bank
was handed about half again the material the excavation yields. `taper_reach` is now
shared between `tapered_invert` and the spoil sum, and a regression test asserts
`fill == 0.75 × cut`.

**And the cut was chasing its own berm.** The floor was referenced to the pour point of
the *bermed* surface, while the spoil was computed from the *bare* one — circular (the
berm is built from the spoil the trench yields) and a licence to over-excavate, since the
trench deepened as its own bank grew. The cut is now referenced to the **natural** pour
point, which makes *depth* mean depth below the ground you are standing on, and removes
the circularity. `_datum_surface` is gone.

**At grid had to move with it.** A level berm raises the level the feature spills at, so
what it holds back stands *inside* the excavation — above the trench's own brim, below
the original ground. It is neither "the trench" nor water held above ground, and a
first attempt to split it out as backwater (`split_backwater`, measured against original
ground) returned ~1 m³ per feature: it was asking the wrong question. The right answer is
that At-grid should measure the finished feature — the storage over its footprint filled
to its **post-berm** spill, taken on its own ground so it stays order-independent. Read
against the bare-trench level instead, a berm doing exactly its job reported as **+80%**.

| Quail Island, all four stages in | before Stage 4 | after |
|---|---|---|
| solo Δ — median / mean / max | 0.0% / 0.5% / 4.3% | **0.0% / 0.4% / 3.7%** |
| Swale 25 — geometric / at grid / measured | 458 / 263 / 263 | **458 / 474 / 474** |
| Swale 22 | 524 / 308 / 322 | **524 / 376 / 389** |
| Swale 27 | 181 / 108 / 108 | **181 / 135 / 135** |
| burned cut / fill | 17,535 / 12,722 m³ | **20,096 / 10,226 m³** |

The berms now do work: Swale 25's bank adds **211 m³** against a credited section of
196 m³, so its design capacity is earned. Swale 22's adds 76 m³ against 225 credited, and
Swale 27's 27 against 78 — which is the honest and useful reading, and is what the †
flag now means. `resolution_penalty` is measured against **geometric** (trench + berm)
rather than the trench, because both grid columns now contain the berm; against the
trench alone it flagged 31 of 40 rows for berms working correctly.

**22 of 40 rows are flagged**, and that is the substantive finding to act on: on this
terrain many companion berms do not deliver the section they are credited with. The
number to look at is Geometric against At grid, per row.

**Still open.** `Basin 39 + Dam 40` at **+45.6%** and `Basin 2 + Dam 1` at **+17.3%** —
Basin 39 has 2.65 m of relief across a 1.50 m design, so its level floor genuinely
over-excavates and `steep_ground_warning` fires. That is a design property, not a
measurement defect. Burned cut is **20,096 m³** against the analytic ~11,948 m³ the
report prints; the datum fix accounts for some of the old gap but not all of it, and
whether to report the burn's own quantities is still the open call from Round 8.

### Round 9a — the berm readout, made to match the berm

Asked: *does the dialog tell you the expected berm height from the excavated material?*
It did — and it was quoting a bank the burner does not build.

`berm_height_estimate` returned `√(0.75 × section)`: the height of a **1:1 triangular
ridge** of the same volume. Round 9 changed the burn to spread that spoil across a band
as wide as the swale, to a level crest. Same earth, different shape, different height —
**1.22 m against 0.50 m** for a 3 / 1 / 1 m swale, a 2.4× overstatement on the readout
that appears while you are drawing.

Only the *height* was wrong. The capacity credit and the fill volume are both
`0.75 × section` — volumes, and so shape-independent — which is why they stayed correct
throughout. That is now stated once, in `berm_spoil_per_metre`, and the three call sites
derive from it instead of each re-deriving a height they do not need.

Also fixed: the dialog called `berm_height_estimate(depth, width)` with **no
`bottom_width`**, so every swale was costed at 1:1 side slopes whatever section it
carried — one of the "four different berm-height derivations" MATHS_AUDIT flagged.

And the berm band itself was still rasterised `all_touched`, the one volumetric burn
Stage 3 missed. It let the two offset bands overlap each other and the trench, so the
side that lost the tie was trimmed and the same spoil went up over a narrower strip. The
crest came out ~50% high **on flat ground**, where terrain explains none of it.

Verified against the burn across four grid alignments:

| centreline | swale mask | cut | berm band | built | predicted |
|---|---|---|---|---|---|
| y = 70.00 | 4.01 c/m | 3.01 m³/m | 3.01 m | 0.75 m | 0.50 m |
| y = 70.25 / 70.50 / 70.75 | 3.02 c/m | 2.01 m³/m | 3.02 m | **0.50 m** | 0.50 m |

Three of four land **exactly** on the prediction. The fourth is the degenerate case where
the buffer edges fall on cell centres, so a 3 m swale claims four cells and cuts half
again the spoil — a real ±half-cell grid effect that no drawn-dimension estimate can
predict, and precisely what the At-grid column exists to expose. The dialog says "level
ground" rather than pretending otherwise.

**On screen now:** under Calculated Capacity, *"Berm ≈ 0.28 m high × 2.0 m wide,
level-topped — 0.56 m³ of spoil per metre"*; beside the key-in checkbox that changes it,
*"Berm crest: 79.40 m — 0.28 m above the ground it sits on (last analysis)"*. Stated as a
height as well as a datum, so the prediction and the built bank can be read against each
other. Both blank when there is no berm.

Pinned by `test_the_dialog_height_is_the_bank_the_burn_builds` (burns a swale and asserts
the crest matches the dialog's figure) and
`checks_visual.check_companion_berm_readout_renders`, whose fixture crest is deliberately
the same number the prediction produces — if they ever diverge, the check fails.

### Round 9b — the berm height is a range, not a number

Asked: *the crest is one elevation, but the ground under it isn't — so isn't the berm's
height different along its length, and doesn't reaching that level cost more soil where
the ground is lower?* Yes to both, and the readout was hiding it.

`berm_ground_elevation` was the **mean** ground under the band, so the dialog reported
`crest − mean` as though the bank had one height. Measured on the Quail Island berms it
does not:

| | crest | min | mean | max | spread |
|---|---|---|---|---|---|
| Swale 29 | 66.35 | **0.60** | 0.98 | **1.73** | 1.14 m |
| Swale 25 | 73.50 | 0.67 | 1.04 | 1.70 | 1.03 m |
| Swale 30 | 62.44 | 0.69 | 1.05 | 1.70 | 1.00 m |
| Swale 22 | 63.57 | 0.62 | 0.75 | 1.12 | 0.50 m |

Most of that is **along** the run, not across the band — at these gradients a 3 m band
only accounts for 0.15–0.3 m of it. The rest is the alignment not sitting on contour
while the crest, by definition, does.

The burn now records `berm_height_m` as `(min, mean, max)` over the cells where a bank
was actually built — excluding any where the ground already stands above the crest, since
folding those zeros in would understate the structure that exists. The panel prints the
range, collapsing to a single figure only when there is genuinely nothing to spread.

**On the threshold for the new warning, and why it sits at the tail.** `max ÷ min` across
the 32 berms runs 1.36 to 2.90 with a **median of 2.06** — one tight unimodal band with
no gap in it. On an off-contour design the typical berm is *already* twice as tall at one
end as the other, so any loose threshold fires on nearly everything: `> 1.5 ×` catches
94%, `> 2.0 ×` catches 56%. That is the noise the Verify dagger had to be rescued from in
Round 8. `berm_variation_warning` therefore uses the physical line
`steep_ground_warning` uses — the ground falls further than the structure is tall — which
names one feature here (Swale 29) and stays quiet on the rest.

The load-bearing half is the range, printed on every berm unconditionally. A warning is
for *go and do something*; the range is for *know what you are building*. Worth saying
plainly: on this design the warning is not a reliable way to find the uneven berms,
because nearly all of them are uneven. The table is.

## Round 10 — 2026-08-13 (Stage 5: two kinds of number, and saying which is which)

Reported: *the report doesn't distinguish analytical figures from measured ones, and
that is causing confusion.* It was — and the cut/fill gap left open at Round 8 is the
sharpest instance of it.

**Both quantities are now printed.** `earthwork_design.burn_quantities` differences the
two elevation surfaces and gives what the terrain model actually moved;
`_page_build_schedule` prints it beside the drawn figure with the ratio spelled out. On
the Quail Island design: **11,948 m³ calculated against 20,096 m³ measured (1.68×)** for
cut, and 5,546 against 10,230 (1.84×) for fill. A landowner pricing an excavator off the
calculated column is 68% short.

The gap is not an error in either. `calculate_cut_volume` is `section × length` — the
earth that comes out if the ground under the feature is flat. The burn cuts a **level**
invert and builds a **level** crest, which is exactly what makes a swale hold its design
volume on a slope instead of a wedge, and on real ground that means cutting deeper at the
high end and building higher at the low end. Site totals only: banks lie outside their own
footprints and cuts overlap, so a per-feature split would report the splitting rule.

**And the distinction is now stated up front rather than left to be inferred.** A new
page 2, *"How to read the numbers in this report"*, defines the two:

- **Calculated** — from the dimensions you drew, using geometry and standard hydraulic
  formulae. Never looks at the ground; checkable with a tape measure and a calculator.
- **Measured** — read off the elevation model, either as the ground is or with the design
  cut into it and flooded. Specific to this block and this survey.

It says which to quote for what (calculated for "is it big enough", measured for pricing),
names the one deliberate hybrid — the headline capture % is **calculated storage against
measured inflow**, and neither half answers it alone — and fences the word *measured*:
measured from the elevation model, **not surveyed on the ground**, on a grid whose cell
size is printed so the limit is concrete.

Every table carrying figures now leads its note with its kind (`Calculated · `,
`Measured · `, `Calculated storage against measured inflow · `), from two constants so the
words cannot drift between sections.

**Two things the render caught that the tests could not.** The definitions first went in
as a `KeyValueTable` and printed **cut off mid-sentence at the page edge** — the
documented layout-table trap (silently overflows rather than wrapping, CLAUDE.md). They
are `Callout`s now. And the page-shot fixture had no burn, so the new table would never
have been shot; its synthetic quantities are scaled to the fixture's single swale, because
the first attempt rendered "334× the drawn figure" and read as a fault in the code rather
than the layout it exists to show.

Also fixed: `test_trivial_natural_ponding_is_not_mentioned` asserted the word "hollows"
appeared nowhere in the document, as a proxy for one sentence. The reading guide has to
describe what a terrain model can see, so it collided. Now matched on the sentence
(`"already collects in hollows"`), like its positive counterpart already was.

## Round 11 — 2026-08-13 (what the ground holds, and sizing against it)

Reported, in two parts. First: *"now that I've added Key Berm into banks they are more
accurately showing the actual water that is ponding — how can I get At grid and Measured
to align?"* Swale 5 read `Design 443 · Geometric 554 · At grid 439 · Measured 1,095 ·
Δ +150% †`, with 22 of 40 rows daggered. Then the sharper one: **the live assessment
sizes on the drawn section**, so a keyed swale reports *full at this storm* while most of
its pond is empty — and the designer enlarges a feature that needed nothing.

### One wrong question, asked in two places

`At grid` was `Σ (held_spill − burned floor)` over the **trench footprint**, with
`held_spill` from `pour_point` — the minimum of the footprint's one-cell outer ring.
Measured on Swale 5 from `%TEMP%\tfa_s0ajaqzl`:

| | |
|---|---|
| pond level in the burned DEM | **69.60 m** (dead flat — the berm works) |
| footprint ring minimum | **68.88 m** — 0.72 m *below* the water |
| Σ (ring min − floor) over the footprint | **439 m³** ← the reported At grid, exactly |
| Σ (pond level − floor) over the footprint | **761 m³** |
| ponding outside the footprint (backwater) | **334 m³** |
| | **1,095 m³** ← the reported Measured, exactly |

**A one-cell ring cannot tell an outlet from a lip with rising ground behind it.** On a
contour swale with a downhill berm the ring minimum is the *uphill* lip, and water
crossing it runs into a hillside and cannot leave. Sitewide across 38 cut features:
**80% of the gap was the low datum, 20% was pond extent beyond the footprint.** Both are
the same mistake — measuring a *pond* by integrating a *footprint* to a *ring minimum*.

### The fix: flood each feature alone

`DEMBurner.feature_storage` generalises `dam_stage_storage`, which had been doing exactly
this for dams since the reshape release — burn one feature, flood a padded window,
subtract natural ponding, escalate the pad while the pond touches the edge. Attributed
**by region, not by window**: a natural hollow sharing the crop is not this feature's
water. Simulated over all 42 features against the real rasters before writing any code:

| | before | after |
|---|---|---|
| Δ median / mean | +52% / +64% | **0.0% / 1.0%** |
| rows within ±10% | 6 of 40 | **38 of 40** |
| sitewide Δ | +98% | **+2%** |
| Swale 5 (At grid → Measured) | 439 → 1,095 | **1,095 → 1,095** |
| Basin 39 | 1,122 → 3,206 | **3,206 → 3,206** |
| dams (no At-grid figure at all) | `impounded` | Dam 15 **2,099** against a Design of 2,099 |

The only non-zero deltas are **Dam 1 (+24%)** and **Dam 40 (+15%)** — the two merged
pools, where a dam alone holds less than the pair holds together. Δ now measures
*interaction between features* and nothing else. Dam 15 landing on its Design exactly is
the strongest check available: a dam's Design capacity already came from
`dam_stage_storage`, so the two agree because they have become the same measurement.

### Sizing against it, which is the part that mattered

`ew.terrain_capacity_m3` is cached beside `capacity_m3` — never in place of it — and
computed on the draw / dialog-OK / vertex-release tier, never on the drag tier, following
`_compute_dam_capacity`'s precedent exactly. `simulation.py:774` is the **one** place the
basis is chosen, so every downstream consumer switches together. Swale 5's bar now reads
**51% of 1,095 m³** instead of **100% of 440 m³**, with the drawn figure printed beside
it on the node and in the footer.

Per the brief's principle — *an honest reflection of what will happen on site, not
something restricted by hidden numbers* — **no freeboard is applied to the measured
figure**. It is the pond filled to the level it actually spills at, and that level is the
real 100%. The spillway threshold that will stop % full ever reaching it is recorded in
STRETCH_GOALS §6, not built here.

### Two findings worth keeping

**Storage built goes 17,200 → ~31,000 m³**, and **the score does not move.** Reconstructing
the cascade from the 12 Aug report's per-feature water table reproduces the published
figures exactly (9,970 m³, 57%) and returns the same 9,970 m³ and the same 57% with every
store raised to its terrain capacity. Of 17,537 m³ of runoff, **9,970 m³ reaches a feature
and 7,567 m³ never does**; eight features fill and spill but every spill is caught
downstream. Capacity is not the constraint here — interception is. The scorecard now says
so in one adaptive line, and prints the second figure instead where the two do differ.

**Several swales have become small dams.** Retained depth across the 36 storage features
runs 0.00–3.13 m, median 0.77 m, with a median 28% of each pond standing above natural
ground. `impoundment_warning` fires past **1.0 m of head or 500 m³ above ground** — 5 of
36 here, against 31 of 36 at a 0.5 m threshold, which would have been noise. It names the
volume that would run downhill if the bank gave way and asks for a designed spillway.

### The daggers

`resolution_penalty_m3` is re-pointed from `rasterisable − geometric` to
`cut_m3 − section_m3`, and `cut_m3` is the new record of the trench brim-full before any
berm. Left where it was it would have inverted: with At grid now the pond, a berm doing
exactly its job would report as a resolution failure of +80% and upward. The key keeps its
name because it now means what that name always claimed. `impoundment_m3` is the new,
separate figure for what bank and hillside add.

**Verified:** 1,978 pure tests, 95.29% coverage, ruff clean, 130 QGIS harness checks, four
screenshots changed and accepted (both verification panels, both report pages).

---

## Round 12 — 2026-08-14 (the pond on the map is not the pond in the panel)

Reported from the Quail Island design: *"why does Dam 1 show water pooling over the top
even though the live assessment says it's only 57% full?"* Then, on the diagnosis: *"does
the dam not actually key into the contour as per the live assessment as it appears it is?
Or is water flowing from the natural ponding area below straight past the dam wall?"* and
*"in the DEM the dam wall is being rendered as 4 m even though it should be 2 m."* All
three were right.

### The layer was never showing a fill

`AnalysisWorker` computes ponding as `get_ponding_layer(burner.original)` on the burned
DEM — `fill_depressions(dem) − dem`, every hollow to its spill point. It never sees the
storm. The panel's 57% is `stored_m3 / capacity_m3` from `run_water_balance`. Two
different quantities, and nothing in the Design or Verify stage drew the second: a dam at
5% and a dam at 100% rendered identically. The layer was called **Water Captured**, which
claimed the fill it was not showing.

### Two real bugs behind the overtopping

| Finding | Outcome |
|---|---|
| `key_into_banks` never reached the burn | **Fixed.** The flag was read in exactly three places: `_burn_swale`, `dam_stage_storage` → `_keyed_dam_dem`, and report label text. `_burn_dam` never looked at it, so a keyed dam's *capacity* was measured against a wall reaching its abutments while every raster — the ponding layer, the streams, the verification burn — got the short one as drawn. The pond filled to the natural saddle and ran round the wall's ends. `_burn_dam` now calls `_key_dam_ends` and raises the same abutment advisory. |
| A 2.0 m dam wall burned ~4.2 m thick | **Fixed.** `_downstream_footprint` builds a correct 2 m band; `_burn_dam` then rasterised it with the `all_touched=True` default. That is a *connectivity* answer, not a volumetric one — `_rasterize`'s own docstring says it adds a near-constant 1.3 m and that "the volumetric burns pass `False` for exactly that reason". `_burn_swale`, `_burn_basin` and the berm zones all do; `_burn_dam` and `_keyed_dam_dem` did not. Worst on a diagonal alignment, which Dam 15 is. |
| Network rows label a dam's *ground* as its crest | **Fixed.** `node["elevation"]` is a DEM sample at the geometry centroid. Dam 15 read `crest 54 m` against a dialog crest of 56.12 m. The node now carries `crest_elevation`; a dam with no crest set falls back to the ground, unlabelled. |

**Do not simply drop `all_touched` on a barrier.** It was also what guaranteed the wall
was continuous, and a band under about one and a half cells wide rasterises on centres to
cells joined only at their corners — which D8 walks straight through. Fixing the width
alone would have swapped one leak for another. `_downstream_footprint` now returns
`(band, centreline)` and both dam burns always burn the centreline's cell path as a seal,
instead of using it only when the band claimed nothing. Note `line_cells` is Bresenham and
therefore 8-connected, so the seal is a continuity guarantee of the same class the burner
already relies on for sub-cell barriers — not a 4-connected one.

### Then: show the water that actually arrives

Requested next, and built. `Water Captured` → **`Pond Capacity (full)`**, with two new
layers beside it on the design side:

- **`Pond Capacity (event)`** — each pool re-filled with only the volume the balance
  routes into it. `level_for_volume` inverts the stage-storage curve exactly (sort the bed
  elevations; the volume held at the k-th is `cell_area × Σ_{i<k}(g_k − g_i)`, monotone and
  piecewise linear, so one `searchsorted` brackets it). A part-full pond therefore comes
  out **smaller and shallower**, standing in the bottom of its basin. The simulation's
  frame layer scales the full pond's depth by a fill fraction instead, which keeps the full
  footprint and paints water up banks it never reaches; that layer is untouched here.
- **`Event Water Line`** — the same extent as cell-edge rings, amber, added at the top of
  the group. A fill under a fill shows nothing; the edge is what reads over a dark pond.

`event_pond_depth` reuses `attribute_ponding_volume`'s labelling and transitive joining,
because the two have to agree about what a pool is. Three judgement calls, all
presentational and none of them measurements: where a joined set owns several pools the
volume is split between them in proportion to capacity (there is no defensible answer);
water is never drawn above the pool's spill level whatever the balance says, because the
surplus is overflow; and the baseline pond's own volume is added back per pool, since
`stored_m3` is what the feature holds *over and above* natural ponding and a dam in a wet
hollow would otherwise draw emptier than the ground is.

Both event layers are built in `_on_earthworks_complete` from the arrays
`_compute_verification` already read (handed over on `_state.pond_context` rather than
re-read and re-aligned — the baseline alignment has three failure modes and a fallback to
zeros). So they carry the same staleness as the capacity layer they qualify, which is the
honest arrangement: both move on re-analyse, neither claims to be current before then.

### Still open

The rename does not reach `PondingQueryTool`'s behaviour — Query Depression / Ponding
still samples the full-capacity raster and so still reports a capacity. It says so now,
but clicking the event layer is not wired up.

---

## Round 13 — 2026-08-14 (the dam that looked like it was spilling, and wasn't)

Reported: *"in the case of Dam 15 it is still spilling at 64%"*, then three Identify
readings — 55.469, **56.120**, 55.417 — and *"I don't understand how the water is
technically flowing from 55.46 to 56.12."*

### It was not spilling, and nothing flowed uphill

Reproduced from `12.08.2026/Quail_Island TerrainFlow Project.tfd` (42 features) burned
with the current code. Every claim below is measured, not inferred:

| | |
|---|---|
| wall thickness | **1.87 m** over the drawn 34.7 m (drawn 2.0) — the round-12 fix holding |
| keyed returns | **0 cells** — both abutments already stand at or above 56.12 |
| pool cells above the crest | **0** |
| pool cells downstream of the wall | **0** — all 1,651 upstream |
| pool surface when full | 56.12 m, exactly the crest |
| pool surface at this storm | **55.57 m** — 0.55 m *below* the crest |

The middle Identify reading was the dam. "Earthworks — Hillshade" is a second raster over
`modified_dem.tif`, so Band 1 is burned **ground**; 56.12 is the crest to the centimetre,
and natural ground under those wall cells is 53.93–54.54. The 55.469 cell is pond bed with
0.65 m of water standing on it — its *surface* is 56.12 too. The pond's surface is flat.

**Why the stream crosses the wall.** Flow direction is computed on a depression-**filled**
surface (`flow_analysis.py:82`), so the Streams layer always draws the spill path out of
every pond whether or not the event fills it. It is a drainage-network map, not an event
map. That is worth knowing before diagnosing any "water goes through my earthwork" report.

### What was actually wrong: nothing takes the overflow

Walking the pool's rim: its eight lowest cells are all the dam's own crest, and the lowest
rim point that is not the dam is *also* 56.12. When that pool fills, the only way out is
over the wall — and **Dam 15 has no spillway sited**.

Built in response, `reporting.overtopping_spill` + `burn_strategy.overtopping_warning` +
an `Earthworks — Overtopping` band layer. On this design:

| | pool | pours at | over | spillway |
|---|---|---|---|---|
| Dam 15 | 2,688 m³ | 56.12 m | **24.0 m of crest** | none |
| Dam 3 | 175 m³ | 69.04 m | **30.0 m of crest** | none |

Dam 1 and Dam 40 are not overtopped, so it is not firing on everything.

**The length is the point, and it answers a second question the user asked:** in reality
water goes over the whole level crest at once, but D8 forces the entire overflow through
one cell, so the stream layer draws a single thread and invites the reading that the water
is picking a spot. The band is the run of crest standing at the pour level, and it is what
discharge per metre — hence whether the face erodes — must be figured against. Changing the
routing to represent this properly would rewrite the hydrology chain; measuring and drawing
the length does not.

### Two defects the real design caught that the synthetic fixture did not

1. **The "natural saddle" was the dam's own keyed returns.** They are raised ground outside
   the feature's recorded mask, so they came back as the alternative way out at the dam's
   own level. `overtopping_spill` now takes `built` — every cell the burn raised anywhere;
   another feature's embankment is not a natural saddle either.
2. **It printed "Raising the crest 0.00 m…"** — a float-epsilon rise. Now needs a
   centimetre, and below that says the thing that is actually true (below).

**Both dams' "next way out" is exactly their own crest.** Not coincidence: the crest is
seeded from the highest ground the drawn line touches, so it lands level with the bank. The
pond reaches crest and abutment at the same instant — **there is no freeboard at the
abutment**, and raising the crest alone moves the failure to the end of the wall, where
there is no structure at all. The advisory says so.

### The caveat the advisory must carry

**The burn is spillway-blind.** No `_burn_*` method reads a spillway; nothing cuts a notch.
The analysis therefore routes overflow over the crest *whether or not* one is designed, and
a sized, sited spillway changes no raster, no routing and no pond. With one sited the
message says this is a limit of the model rather than a fault in the design; without one it
is the design. Burning the notch is not done here.

### Known gap

Only the four dams register as barriers. A swale's recorded mask is its trench and its
companion berm sits beside it, so `mask & raised` is empty and keyed swale berms are never
checked — `impoundment_warning` covers those from the other direction, but this does not.

**Verified:** 2,086 pure tests (13 new), 95.28% coverage, ruff clean, 133 QGIS harness
checks, every screenshot pixel-identical. The new harness check drives a real dam through
the real controller and asserts the band is longer than one cell and no longer than the
wall — the D8 artefact this layer exists to contradict, in both directions.

## Round 14 — 2026-08-14 (the crest that spills at one point, and why)

Reported, from the Quail Island map: *"there is water running out of every point of the dam
wall — however why isn't it more even across the top of the wall, as the heaviest flow is on
the right hand side, which doesn't match reality."*

Both halves of that observation are correct, and **both of the explanations already on this
log are wrong.** Correcting them is the point of this entry.

### Reproduced from the artefacts first (Round 6 method)

`Quail_Island TerrainFlow Project.tfd` (42 features) burned headless under plain CPython —
no QGIS needed. Dam 15 comes back at **1,651 pool cells / 2,688 m³**, pours **56.12 m**,
crest run **24.0 m**, drawn wall 34.7 m, no spillway; Dam 3 at 175 m³ / 69.04 m / 30.0 m;
Dam 1 and Dam 40 not overtopped. Session inputs: routing **`dinf`**, `stream_threshold_ha`
**0.5**, coefficient basis C0.50, 120 mm / 24 h. Every figure below is measured off that run.

### Correction 1 — "D8 forces the entire overflow through one cell" is false

Round 13 said so, and `reporting.overtopping_spill` and `burn_strategy.overtopping_warning`
both shipped it in prose. Measured:

| | |
|---|---|
| distinct cells carrying flow out of the pool | **23** — not one |
| **share through the busiest crest cell** | **87.1%** — **20.9×** an even share |
| crest cells carrying < 1% of even | **16 of 24**; **median crest flux 0** |
| cells clearing the 0.5 ha stream threshold | **1** |
| load that is through-flow from upstream | **96.1%** |

So the flux *is* concentrated, but not because routing sends it through one cell. **The
single thread on the map is a rendering threshold**: Streams draws `accumulation >
stream_threshold`, and no one cell of a shared crest reaches a contributing-area threshold
the whole catchment only just exceeds. That is why Surface Runoff — a log ramp with no
threshold — shows water at every point of the wall, exactly as reported.

### Correction 2 — "`resolve_flats` drains a flat to its single outlet" is false

Verified against the installed pysheds 0.5, not assumed: `resolve_flats` is Barnes et al.
(2015), `_par_get_low_edge_cells_numba` marks **every** qualifying same-elevation neighbour,
and `_grad_towards_lower_numba` seeds its BFS from all of them at once. It is already
multi-outlet. **Do not write a multi-outlet flat resolution.**

### What is actually wrong

`resolve_flats` gives each flat cell a route to its **nearest** way out. A pool is therefore
partitioned among its outlets by geodesic distance — and **a reservoir is a mixing node
while a distance transform is a routing node.** Dam 15 receives essentially its whole
catchment as one channel, and nearest-outlet routing hands that channel to whichever crest
cell is closest to where it arrives. Hence one cell at 87% while sixteen carry nothing.

**The crest is not the limit.** The pour-level band 8-connected to the crest is **65 cells,
one component, 49 of which have a drop to below the pour level** — so the raster can carry
an even spread. It is a routing choice, not a raster limitation.

**But 49 is not a length, and `overtopping_spill` is right to report 24.** The band is the
wall's length × *thickness* — ~25 cells long, 2–3 thick, both faces draining. Using it would
inflate `length_m`, shrink the `q = Q/L` that erosion is judged by, and do so in the unsafe
direction. The pool-facing row is the crest run, and the cap at the drawn length absorbs the
√2 a diagonal cell path over-counts by. Noted because the first draft of this entry called
24 an undercount, which is the wrong way round.

### The fix is not shipped, and here is exactly why

Two approaches were built and measured against the real design. **Neither is in this
change.**

1. **Re-partition the flat, balanced on load instead of distance** (parent pointers +
   re-ramped conditioned surface). It evened the pool-facing row — worst case 20.9× → **0.5×**
   even, starved cells 16 of 24 → **2 of 24** — and still left **84%** of the flux crossing at
   one place. The reason is structural: **flow routing carries one pointer per cell, so a
   cell's whole load goes to exactly one destination.** Dam 15's inflow arrives as a single
   channel of ~36,000 of the pool's 42,000 units, and no partition of *cells* can divide the
   load of *one* cell. **One pointer per cell and a weir are incompatible.**
2. **Split the flux in the accumulation raster instead**, which carries no such constraint.
   This works on the crest — min = median = **863**, **0 of 49** cells starved, busiest
   86.9% → 12.8% — but the site's busiest cell moved **−6% to −10%** and **64 cells went
   negative**. Cause: the "cells leaving the pond" set is not a clean cut, because sheet flow
   reconverges along the wall face, so shares get double-counted. Fixable; not verifiable in
   the time available.

Shipping a change that silently moves the site's peak accumulation by 10% is the failure
mode this log exists to prevent. It is handed back with the measurements above rather than
half-landed.

3. **Contract each pond to a single node** — everything arriving is pooled and leaves divided
   equally between the cells that discharge, applied by two accumulation passes (absorb, then
   re-emit) so the caller's own routing engine does the work and D-infinity keeps splitting
   fractionally everywhere outside a pond. **This is the right algorithm and it produces the
   right crest**: Dam 15's level band goes from `min 1 / median 1 / max 36,782` to
   `min 1 / median 830 / max 830` — **a ratio of exactly 1.0x, a true weir** — with **zero
   negative cells**, against approach 2's 12.8% best case. Still not shipped: **the site's
   busiest cell drops 15.0%**, so mass is not conserving.

   The loss is localised, not mysterious. It is **not** the base weighting (matching pysheds'
   own unweighted array exactly reproduced 15.023%) and **not** pond overlap (4 gated ponds
   and 195 ungated ones both lose). It is the **chained-pond hand-off**: pond 4 sends 15 of
   its 65 exits into another pond, and 401 cells outside ponds drain into pond cells, so the
   topological pass that carries an upstream pond's total down to the next one is where the
   water goes missing. That function — `_pond_order` and the two-loop total propagation in
   `plan_crest_split` — is the whole of what is left to fix.

   Work parked, not lost: the module is in the session scratchpad as `crest_routing_WIP.py`.

**This is a property of ponds, not of dams.** The gate on built ground is a *scope* choice,
not physics: a keyed swale's companion berm impounds a dead-flat pool exactly as a dam does
(Round 9: Swale 5 stands at 69.60 m across its whole pond), a basin fills against its rim, and
a natural hollow spills over its saddle by the same rule. On this DEM there are **195 ponds**
against 4 built ones. Worth deciding deliberately when this lands: correcting only the design
tier would leave baseline and design on different hydrology, so a before/after comparison
would partly measure the change in method. Whatever is chosen, **flow directions are never
touched**, so `feature_inflow_m3`, the catchment labelling and capture % are unaffected either
way.

### What did ship

| | |
|---|---|
| **Unrouted flow is counted** | A cell pysheds marks `flat`/`pit` is rewritten to a **self-loop** before accumulation, so it absorbs its whole upstream and reports none of it downstream — silently. Measured here: **36 cells holding 4,152 cell-units.** `FlowAnalysis.unrouted_flow()` counts it and the run warns past 0.5% of domain runoff. It does **not** re-route them: a cell with no lower neighbour genuinely has nowhere to send water. |

**And they are not ponds — the first cut of that warning would have cried wolf.** Asked
whether these behave as small ponding areas that spill on once full: no.
`fill_depressions` has already raised every hollow to its spill level, which *is* the
fill-then-overflow mechanism, so a real pond never appears here. Classifying the 36:
**0 have standing water, 0 are on the grid border, and 35 are adjacent to nodata.** Quail
Island is an island — those are the **coastline**, where the only downhill neighbour is sea
carrying no elevation. That water has left the analysed area rather than gone missing.
Counting it would have fired the warning on every clipped or coastal DEM for a benign
reason. `unrouted_flow` now makes the split the design tier already makes — `LABEL_EXIT` at
the edge of the data against `LABEL_SINK` in an interior pit — and reports only the second:
**1 cell on this design, not 36**, so the warning stays quiet, which is the honest outcome.
| **Keyed swale berms register as barriers** | `crest = mask & raised` finds a dam's wall and **nothing** for a swale, whose mask is its trench while the berm sits beside it — so keyed swales were never overtopping-checked. New `DEMBurner.burned_raised`, kept **separate** from `burned_masks` (widening that would move pool attribution and every bermed swale's Δ), following the precedent `_contact_mask` already sets. |
| **The runoff basis reaches the earthworks re-run** | `run_with_earthworks` passed neither `sizing_basis` nor `runoff_coefficient`, so the re-run fell back to the worker's constructor defaults — coefficient at C=0.50. A rainfall-basis or SCS-CN session had its earthworks Surface Runoff and exit L/s scaled by a **different depth from the baseline it is compared against**. Now forwarded. Moves those figures for any non-default session. |
| **The tie-break is pinned** | Nothing tested `_OFFSETS` order or `>` vs `>=`, so every level-crest routing decision could have flipped with the suite green. Note the two tiers **disagree**: pysheds scans N, NE, E, SE, S, SW, W, NW and `flow_graph` scans NW, N, NE, W, E, SW, S, SE, both first-max-wins. |
| **Prose corrected** | Both wrong explanations, in `reporting.overtopping_spill`, `burn_strategy.overtopping_warning` (docstring **and** the user-facing advisory), and `flow_graph`'s conditioning list. |

### Three hazards found on the way, worth not re-deriving

- **`fill_depressions` mutates its input in place.** `np.shares_memory(filled, pit_filled)`
  is **True** and `filled - pit_filled` after the call is identically **0.0**. Any ponding
  measured that way silently returns nothing; take the copy *before* the call.
- **`max_iter` is not a problem on this design, and must not be raised casually.** Largest
  flat 145,952 cells, but max BFS depth reached is **265** against `max_iter = 1000`, with
  **0** cells unreached. Raising it would change **every flat in the DEM, natural ones
  included**. It also can never exceed 32,768 — the `uint16` gradient wraps at
  `2·grad_towards_lower + grad_from_higher`.
- **The crest is exactly level.** All 24 spill cells hold **one** distinct float64 value,
  `56.119998932`. It is not bit-identical to the Python float `56.12` only because the burned
  DEM round-trips through **float32** on save, which moves all 24 identically.

### Still open

- ~~The even-crest fix itself (both routes measured above).~~ — landed in Round 15 below.
  Approach 3 was right; the diagnosis of why it leaked was not.
- **The burn is still spillway-blind** (Round 13) — Dam 15 reads 2,688 m³ to the crest against
  1,832 m³ to a sill at 55.52, ~30% overstated. **Now the priority item after the demo video**
  (user, 2026-08-14); scoped in STRETCH_GOALS §5a, and it needs §6's capacity/threshold split
  first.

### Closed in this round, from the same investigation

**The conditioned raster is saved float64 now.** `resolve_flats` inflates a flat by integer
multiples of `eps = 1e-5` m, and float32's spacing reaches ~6.1e-5 m at 1000 m elevation — so
above roughly 600 m the entire flat gradient was quantised away on the way to disk, and
`d8_from_dem`, which needs a *strictly* positive drop, read a genuine flat and called every
cell of it a `LABEL_SINK`. Both ends downcast (`save_result`, and `_ensure_flow_graph` reading
it back with `.astype("float32")`); both are float64. Quail Island at ~56 m could never show
this, so `TestConditionedSurfacePrecision` builds a **1000 m plateau** and asserts the failure
before asserting the fix — otherwise the test proves nothing about the bug it exists for.

## Round 15 — 2026-08-15 (the crest spreads, and the water is all still there)

Round 14's approach 3 — contract each pond to a single node — was the right algorithm and
produced the right crest. It was parked because the site's busiest cell dropped 15.0%. **The
loss is fixed and the feature is shipped.** Round 14's account of *where* the loss was is
corrected below; the algorithm's description stands.

### Reproduced from the artefacts first (Round 6 method)

`burned.tif` / `original.npy` / `own.npy` from the Round 14 scratchpad, 858×1027 at 1 m,
routing `dinf`, under plain CPython. Baseline `acc.max() = 101,654` and the WIP reproduces
**−15.023%** to three decimals, so this is the same run.

### Correction — the busiest cell was never the right test

The invariant is: **the sum of accumulation over every true terminal cell — one pysheds
routes into itself — equals the domain weight.** Baseline: **877,400.0 against W = 877,400,
exact.** Measured that way the WIP delivers **861,393.1: it loses 16,006.9 cell-units,
1.824% of the site.** That is the defect, stated in units that mean something. A single cell
moving is not, and it turns out the fix moves it anyway (see below).

### Correction — the loss is 100% the *indirect* hand-off, 0% the direct one

Round 14 pinned it on `_pond_order` and the two-loop propagation in `plan_crest_split`.
Wrong function. Accumulation is linear in its weights, so the injection can be propagated on
its own and every unit traced to whatever swallows it:

| built-gated, 4 ponds | cell-units |
|---|---|
| injection re-absorbed by a pond | 19,763.4 |
| — landed **directly** on a pond cell (all `_pond_order` can see) | 3,756.5 |
| — landed on open ground and **then flowed into** a pond | **16,006.9** |
| net loss | **16,006.9 — identical** |

The direct hand-off is **mass-neutral by identity**: the unit that dies on the downstream
pond's cell is exactly cancelled by the `share` added to that pond's total. So `_pond_order`
was never the bug — **the graph it ran on was**. `downstream[i] = rid[tgt[cells]]` records an
edge only where an exit's *immediate* D8 neighbour is a pond cell. Measured against the real
transfer operator (one unit emitted per pond, read back off the dinf graph):

```
        -> pond1   pond2   pond3   leaves
pond 3  0.6111  0.3889  0.0000   0.0000    <- 100% into other ponds; the WIP saw no edge
pond 4  0.0308  0.0462  0.8462   0.0769    <- 92.3%;  the WIP saw one, at 0.2308
```

**1 of 5 real edges gated; 12 of 221 ungated — 95% invisible.** Round 14's "401 non-pond
cells drain into pond cells" is the fingerprint of exactly this: the multi-hop entry points a
one-cell graph cannot see. Compactly: **the WIP is the Neumann series truncated after one
term, computed on the wrong operator.** Ungated, 122,603 of 205,169 injected units (59.8%)
are swallowed again, of which the direct case is 4.2%.

Two suspects Round 14 left open, both killed by measurement and needing no code: **0 ponds
with no exit** on this design, and **0 cells claimed by two pond regions**.

### The fix: measure the transfer instead of modelling it

No pond graph is built at all. Emit `T/n` at the cells each pond discharges into, accumulate
that injection **alone**, add the field to the running total, and read off how much of it
landed in a pond again — that is what those ponds have now received, and it goes out on the
next pass. Every transfer path is therefore accounted whether it is one cell long or two
hundred, and a pond that receives its own emission back needs no special case.

The identity `flux to a real terminal + residual == W` holds **at every truncation point**,
not only at convergence, because every quantity in the loop is non-negative — so the loop is
**monotone from below**: it can only under-emit, never over-emit, never drive a cell
negative. Stopping early is one-signed and bounded by a number the function returns.

Measured, all 195 hollows above the size floor:

| | |
|---|---|
| reaching a real terminal | **877,400.0000 — exactly W**, against the WIP's 861,393.1 |
| residual / stranded | **0.00 / 0.00** |
| negative cells | **0** (`acc.min() = 0.000000`) |
| Dam 15's 65-cell level band | `min 1 / median 1 / max 36,782` → **962.1 uniform, ratio 1.00×** |
| flow directions | **bit-identical** — `feature_inflow_m3`, catchment labelling and capture % cannot move |
| accumulation passes | 9 (1 absorb + 8 emission), **2.3 s → 5.7 s** |

### Two things that fell out of building it

- **A `d8` cascade probe under-reads a `dinf` cascade.** Depth reads 6; eight emission rounds
  are needed, because D-infinity fans into ponds a single-pointer chain never enters. The
  pass budget doubles the probe and floors at 16, which costs nothing — the loop stops on an
  empty residual, not on the budget.
- **Residual decay on a chain is linear, not geometric** — each pass moves water one link. A
  synthetic 20-dam terrace (a keyline sequence, i.e. this plugin's own use case) still holds
  **61.1%** after six passes. A small fixed cap would have shipped a quiet 60% loss. A **D8
  pre-solve** collapses that chain to one pass but **over-emits by 0.44–5.29% on Quail
  Island — it invents water** — so it is rejected: losing water is reported, inventing it is
  not.

### The busiest cell still moves 5.7%, and that is the fix working

101,654 → 95,899.7. Not a leak — **relocation, and it cancels: +6,113.7 gained across 82
terminals, −6,113.7 lost across 91, net 1.6e-10.** Per pond: **Dam 15, the feature that was
reported, moves the site peak by +0.000%** — it is in another catchment. The movement is Dam 3
(−4.05%) and the 70.02 m pond (−6.11%), whose level crests straddle a divide, so spilling
uniformly along them genuinely sends part of the discharge to the other coast. That is what a
level crest does. `max_upstream_area_m2` is the figure that moves; it is already documented
in `analysis_worker` as a display statistic and never a capture-% denominator.

Streams over 0.5 ha go **3,850 → 3,286 (−14.6%)**, and the network does **not** fragment: 80
connected components before and after, 78% of the lost cells inside or within 3 cells of a
pond, and below Dam 15 the channel re-forms after one D8 step. A channel does not survive
crossing a reservoir; what is gone is the thread that used to be drawn *through the pool*.

### Scope — all ponds, floor of 9 pool cells

Gating on built ground would leave baseline and design on different hydrology, so a
before/after comparison would partly measure the change in method. It also needs no new
plumbing: `AnalysisWorker` only ever receives the burned DEM, never the pre-burn one.

But the **median pool here is 3 cells holding ~17 mm** and 125 of 195 are 4 cells or fewer —
`fill_depressions` noise, not reservoirs, and contracting one divides a channel across a
puddle's rim. `MIN_POND_CELLS = 9` keeps 53 ponds. Dropping the floor costs about **five extra
accumulation passes** (12–14 against 9) and gains **0.03 percentage points** at the busiest
cell, which is the whole case for it. Raising it to 50 is the documented runtime lever. The
smallest built pond on this design is 561 cells, so nothing designed is near the floor.

**Correction to this entry, measured after it was first written:** the floor was also claimed
to hold down the number of cells reading `acc <= 2`. **It does not.** The pool cells that go
quiet are the *large* ponds' interiors, which no floor removes — and that reading turned out
not to matter anyway; see "Ponds are not through-flow" below.

### What shipped

| | |
|---|---|
| **`modules/crest_routing.py`** | New pure module. `find_impoundments` gains the size floor; `_level_rim` keeps its output bit-identical but works inside a padded bounding box (the whole-grid dilation loop cost **12–34 s on 195 ponds**, more than every accumulation pass combined; **1.24 s** now). `plan_crest_absorption` replaces the first half of `plan_crest_split`; `spread_crests` replaces the second. **`_pond_order` is deleted** — with the transfer measured, no pond graph exists to order. |
| **`FlowAnalysis._spread_crests`** | The pysheds-facing driver. `self.fdir` is never written — the absorbing map is a local copy, or `unrouted_flow()` would report every pond cell as water the routing could not place. The absorbing directions are wrapped with the **direction** raster's viewfinder, not the DEM's: for dinf its nodata is NaN while the DEM's is a sentinel, and the wrong one hands the nodata border a weight of 1. |
| **`run(crest_split=True)`** | New keys `crest_ponds`, `crest_cells`, `crest_passes`, `crest_residual`, `crest_skipped`. `crest_split=False` restores the old pipeline bit-identically, which is what makes "no ponds means nothing moved" assertable. **`runoff_accumulation` is spread too** when CN zones supply weights, or `throughflow_*.tif` and the exit volumes would disagree with the accumulation beside them by the whole correction. |
| **A pond with no way out keeps the default routing** | Its level band reaches every cell around it, so nothing discharges. Contracting it would swallow the domain and hand back water nobody could place; left alone it stays with the mechanism that already exists for water with nowhere to go, and the reason is recorded in `crest_skipped`. |
| **`crest_spread_warning`** | Same shape and threshold as `unrouted_flow_warning`, for a pond chain deeper than the pass budget. Says which way the error runs: under-reported, never over. Silent on this design. |
| **Prose corrected** | `reporting.overtopping_spill` and `burn_strategy.overtopping_warning` both said the stream layer draws one thread partly because the flux concentrates. It no longer concentrates; only the threshold reason is left. |

`CLudeDocs/wip/crest_routing_WIP.py` is superseded and kept only as the record;
`CLudeDocs/wip/measure_crest_split.py` is now the acceptance harness and prints the table
above.

### Ponds are not through-flow, and one predicted consequence of that was wrong

A contracted pond holds its inflow instead of threading a channel through itself. That is the
truthful picture and is why the crest evened out, but it means **the accumulation inside a
pool stops meaning contributing area**. Measured over the 16,771 pool cells of the 53
contracted ponds:

| | before | after |
|---|---|---|
| median accumulation in a pool | 22.3 | **1.0** |
| pool cells over the 0.5 ha stream threshold | 540 | **14** |
| pool cells reading `acc <= 2` | 1,975 | **11,295** |

**A prediction made from that third row was wrong, and this is the correction.** It was
written here that `keypoint_analysis` would start reporting reservoir floors as ridgelines,
since it tests `acc <= 2` and 9,325 pool cells newly satisfy it. It does not. The ridge test
is `tpi > min_tpi_m` **and** `acc <= 2`, and **a pool is a hollow, so its TPI is negative and
excludes it whatever the accumulation says.** Measured both ways on both DEMs: the ridge set
moves by **0 cells inside a pond** (7,212 → 7,176 site-wide on the burned DEM, none of it in
a pool; 7,016 → 7,016 on the source DEM). Keypoints likewise: **12 either way.**

The two conditions are not independent, and reading one of them alone predicted a fault that
does not exist. Pinned in `TestPondThroughflow` so it is not re-derived from half the test.

**What shipped anyway, because the *reading* is still wrong even where the output is not:**

| | |
|---|---|
| **`ponds_{label}.tif`** | Each pool painted with its pond's whole throughput, in the same cell-units as the accumulation. A **lookup, not a distributable quantity** — every cell of a pool carries the same figure, so summing it over an area multiplies by the cell count. |
| **`DrainageLineAnalysis(dem, acc, pond_path)`** | Substitutes that figure inside pools, so all four uses of `self.acc` — the ridge test, a keypoint's catchment, and the two in the pond-site search — read contributing area again. One substitution rather than a mask at each site. |
| **`— Ponds (routed)` layer** | The reservoirs the flow model routes as ponds, drawn as water. A channel entering a pond genuinely stops there now, so without this the map has a gap where the water is. Drawn from the mask rather than by lowering the stream threshold: **folding the pools into Streams took that layer from 3,286 cells to 10,955**, nearly three times its own baseline, which is a flood of ink rather than a reservoir. |

The network was never the problem: **80 connected stream components before and after**, and
the channel re-forms one step below Dam 15.

### "Does not terminate" was a modal dialog

Checked because the crest split touches `keypoint_analysis`, and the one check that
exercises it densely — `recommend_ponds` — had been quarantined since it "ran >12 min with
no result and no error", cause unresolved between *pathological smooth terrain* and *a
non-terminating loop*. **Neither.** The check emitted the pond recommendation without
running the keypoint pass first, so the controller's guard put up

> `Run 'Find Keypoints + Ridgelines' first.`

and waited for a click that offscreen never comes. Stubbing `QMessageBox` made it return in
**0.00 s**. `recommend_pond_sites` itself runs in **0.05 s** on the harness DEM — 120 calls
to `_valley_cross_width`, not the ~44,000 the nested loops allow, because the
`kp_acc < cell_acc <= 4x kp_acc` filter is very restrictive. The crest split neither caused
this nor changes it: with the pond raster, pool cells fail the `> 4x` ceiling, so there are
*fewer* candidates, and the timings are identical to three decimal places.

The general hazard was worse than the one check: the harness never neutralised modal
dialogs, and both keypoint controllers end their `except` blocks with
`QMessageBox.critical` — so **a genuine exception there presented as a timeout rather than
an error.** `RecordingDialogs` now captures them and `assert_no_errors` fails on any
warning or error dialog. Proof it works: the old quarantined check, left in place, went from
burning a 300 s timeout to failing in seconds with `1 modal dialog(s) the run would have
stopped on` and the text attached.

`checks_slow.py` is deleted, `OPT_IN_MODULES` is empty (the mechanism is kept), and the
check now lives in `checks_contour` beside the rest of the keypoint path — split in two, one
asserting the working order and one asserting the guard fires without it.

**And `_valley_cross_width` is numpy now.** It walked 401 cells in Python per call and was
the entire cost of the function (**0.047 s of 0.054 s profiled**); as one `count_nonzero`
over a row slice it is **0.001 s**, ~47x, with `TestValleyCrossWidthEquivalence` asserting
132 cases against the loop it replaced — edges, nodata stripe and flat row included, since
NaN compares False either way.

## Round 16 — 2026-08-17 (the swale that read 46% full and drew a full stream leaving it)

| Finding | Outcome |
|---|---|
| Swale 17 read **97 m³ · 46% full** in the panel while Surface Runoff drew a channel leaving its west end | **Fixed.** The raster had no storage term at all. It now nets what every hollow holds. |

Reported as "it shows it is overtopping". It was not: the panel and the raster agreed on
volume and disagreed on **retention**. Identify at the pour point read **100.8 m³** against
the feature's **96.9 m³** event inflow from 0.2 ha — so the raster was shedding essentially
the swale's own catchment and keeping none of it, against a measured pond of 210 m³.

**Why the picture was structurally incapable of being right.** Three things compound, and
none of them is a bug in isolation:

1. Flow directions come off the depression-**filled** surface (`fill_pits` →
   `fill_depressions` → `resolve_flats`), so the trench the burn cut is level-full before
   any routing happens. Every drop that arrives leaves.
2. `crest_routing` then contracts the pond and re-emits `held[i]` — *everything* it
   received — at whatever cells discharge from it. A cut swale's lip is level at one place,
   so the exits are a handful of cells and the departure is a single concentrated thread.
3. `grep -nE "capacity|stored_m3|infiltrat" flow_analysis.py crest_routing.py` returned
   **nothing**. The two files that build that raster had no notion of storage whatsoever, so
   the picture was identical at 5% full and at 500%.

**The user's own challenge was the key to the fix.** Asked how the re-analysis could
possibly know a swale from a wheel rut — it cannot. `find_impoundments` is called with
`built=None` and finds hollows as `(filled − ground) > 1e-3`, with no footprint, type or id
anywhere near it. Earthwork identity lives in a different pipeline entirely
(`label_direct_catchments`, fed an `interceptor_labels` raster the controller paints), and
the two never speak.

It turns out **no identity is needed**. A hollow's capacity is Σ(filled − ground) over its
pool — the same measurement `burner.feature_storage` makes for `terrain_capacity_m3` and the
same number the panel prints as *At grid*. Swale 17's 210 m³ is that figure. So one code
path serves the bare DEM and the burned one, a natural hollow and a designed swale, with no
special cases: **a pond fills before it spills**, keeping `min(arriving, headroom)` and
shedding only the excess.

Headroom is drawn down pass by pass rather than resolved up front, because a pond on a chain
receives over several passes and one that filled on pass 2 must pass on what reaches it on
pass 3.

**Kept out of it, deliberately.** Retention applies **only** to the runoff-weighted field.
`plan.capacities` is in m³ and the engine's default accumulation is a cell count, so capping
one with the other is a category error — `spread_crests(capacities=)` defaults to off and
`_spread_crests(retain=)` is set on exactly one call. Streams in cell-count mode, keypoint
catchment sizing and the time-of-concentration path all read the unweighted field and are
untouched, which is right: contributing area does not shrink because something upslope holds
water. `pond_flow` likewise still reports what *arrived*, not the surplus, because
`keypoint_analysis` reads it as a catchment-size proxy.

The conservation identity gained a fourth bucket and did not lose one:
`terminal flux + retained + residual + stranded == total`, asserted at every truncation
point. `retained` is a terminal; `residual` is still only water the loop ran out of passes
to place.

**What moved as a consequence, on purpose.** Boundary exit volumes and exit L/s read the
same net field, so they now agree with the panel's *leaves site* instead of being computed
on the opposite assumption. Volume-mode stream delineation thins below a feature that
captures its catchment. A uniform-depth site now goes through the weighted accumulation
rather than reconstructing `acc × runoff_m × cell_area` afterwards — arithmetically the same
field, but the only one a pond can be held against, and no-CN-zones is the common case.
The baseline DEM gets all of this too; correcting only the design tier would have left
before/after partly measuring the change in method.

Measured on the staircase fixture (pools of 27 m³ and 18 m³ on a 1 m grid): at 1 m³/cell the
site generates less than its storage and **nothing** from above the lower pond reaches the
outlet; at 10 and at 20 and at 100 m³/cell the ponds keep 45.0 m³ and not a litre more; and
what the outlet loses is exactly what the ponds kept, against the same run with the split
off. Twice the rain overflows by more than twice as much — the field is deliberately no
longer linear in its weights, and `test_the_weighted_field_is_spread_the_same_way` was
rewritten because it asserted the very linearity this removes.

**Still true, and still worth reading.** 46% is a *lumped* total-vs-total verdict. Where a
drainage line crosses an alignment at one point a swale with adequate total capacity can
still go over the side there — that is `inflow_profile` / `overtopping_station` and the
Stress points layer, and it is a different question from the one fixed here. The burn also
remains spillway-blind: the pour point this exposed is where a designed notch belongs.

## Round 17 — 2026-08-17 (nine findings from a run-through, and four of them were not what they said)

| Finding | Outcome |
|---|---|
| 1. "516 cells … hold ~105.0% of runoff" | **Ratio fixed; the cells are still open.** The percentage was three measurement defects; the 516 are real and now instrumented. |
| 2. No per-crossing exit summary | **Shipped.** `qgis/widgets/exit_table.py`, under the baseline tool. Nothing computed — the data was already there. |
| 3. "Ponds (routed)" unreadable / redundant | **Layer removed, raster kept.** |
| 4. Swale settings are only depth and width | **Already trapezoidal.** The batter was the missing control, not the maths. |
| 5. Segment overlay is zoom-variant | **Now metres-in-map-units**, data-defined off `width_m`. |
| 6. Surface Runoff fades out at the low end | **Opaque stops + layer opacity.** The palette module already forbade what it was doing. |
| 7. Spillway group opens expanded | **Collapses with its tick.** |
| 8. Runoff "stops at the Pond Capacity extent" | **No clip exists.** Mostly item 6; the rest is stated, not drawn. |
| 9. Freeboard-inclusive "Design" column | **Dropped**, applying the decision the report already made. |

**The pattern worth keeping.** Four of the nine described a cause rather than a symptom,
and the cause was wrong in each. The symptoms were all real. Diagnosing before agreeing
is what made items 4 and 8 an afternoon instead of a fortnight.

**Item 1 — most of 105% was the instrument.** Three compounding defects, all in the
ratio and none in the terrain:

1. `unrouted_flow()` masked by `~_at_data_edge()` and **never by the domain**, while the
   caller divided by the *site's* cell count. On a 2.8 M-cell tile against a smaller
   drawn site, a pit in the buffer contributed its whole upstream to a ratio it was not
   part of.
2. `acc` under crest contraction is a **sum over passes** and a pond re-emits at its
   exits, so the same water can be counted at a second stuck cell.
3. The numerator was a **cell count** and the sentence called it runoff.

Fixed: `unrouted_flow(domain=, field=)`, and the worker re-measures after the mask exists
and against `runoff_accumulation`, so it is m³ over m³ over the same ground. An impossible
share is **printed and flagged, never clamped** — `min(share, 1.0)` would have turned an
obviously broken instrument into a plausible reading, which is the worst available
outcome. `unrouted_diagnostics()` writes `unrouted_{label}.txt` beside the rasters:
flats vs pits, in/out of domain, distance to nodata, connected components, and the
steepest available drop on each conditioning surface. **The 516 are not yet explained —
read that file after a field run before designing a fix.**

Also found: the conservation ledger CLAUDE.md quotes names `stranded`, and grep finds the
word only in a `crest_routing` docstring. Nothing computes it.

**Item 8 — there is no clip, and coupling to the event balance would have been a
regression wearing a fix.** Surface Runoff and Pond Capacity come from one worker run over
one grid; their extents are identical by construction. The cut-off is a data effect —
retention caps every pond at full geometric capacity, so a pond under its capacity emits
zero — compounded by a ramp anchored on the band max whose `log` stops open at 1e-4 and
whose second stop was drawn at **alpha 40**. Below a pond that takes its catchment, the
runoff was drawn all along and could not be seen. Feeding *event* capacities into
`spread_crests` would make the map draw water the design does not release; the honest
lever is the pour level, which is the spillway notch (STRETCH_GOALS §5a). What shipped is
the ramp fix plus `pond_retained_m3` surfaced in the panel — computed since Round 16 and
read by nothing.

**The real-QGIS suite rejected a change, correctly.** Drawn swales were wired to seed from
the panel's cross-section boxes; `check_companion_berm_readout_renders` failed on a
predicted 0.13 m berm against a built 0.28 m. Those boxes are the *Find Best Swale
Segments criteria*, and at their defaults they describe a 0.6 m drainage swale — sub-cell
on a 1 m DEM, so unburnable and unverifiable. It replaced the registry's representable
2.0 m with a width the terrain model cannot hold. Reverted; the reason lives at the
construction site so it is not tried again.

**Two divergences closed rather than created.** Dropping the panel's Design column left
the same freeboarded figure as the *headline* of the properties dialog, unlabelled — so
that is now "Usable volume (m³)" with the allowance stated. And naming the panel's new
exit column honestly ("Average rate over the event") would have left the report calling
the same number "Peak flow", which is the name of a different quantity the same document
reports on the spillway page; the report was renamed to match.

**Item 4, on the user's correction: the derivation was the wrong way round.** The criteria
box first grew a *side slope* input with the floor derived from it, and the shipped
defaults then read 0.6 m top over 0.3 m deep at 1:1 — batters meeting exactly at the
drawn depth, floor **0.00 m**, a V-drain of 0.09 m². The user's answer was one sentence:
*"I do want a floored trench, that is how swales are actually built."*

Two things were wrong and only one of them was the numbers:

- **A swale is set out with three tape measurements — top, bottom, depth — and the
  batter is what they come out as.** That is how the properties dialog has always taken
  it (`Depth` / `Width` / `Bottom width`, with `Side slope: 45.0° (1:1)` read-only
  beside them), and it is what the run-through log asked for in as many words. Entering
  the batter instead makes the floor a *consequence* of three other numbers, so it can
  reach zero without anyone choosing that — which is exactly what the shipped default
  did. Inverted: `swale_bottom_width_m` is now an input, `swale_side_slope` is a derived
  property and a read-only label, and it is **not persisted** — storing a derived value
  beside its causes is how a reloaded file describes a section that never existed.
- **The defaults now sit on `core/registry`'s swale**: 2.0 m top, 1.0 m floor, 0.5 m
  deep, 1:1, 0.75 m² of section. The criteria box and a drawn swale finally describe the
  same swale — which is also the divergence that made wiring one to seed the other look
  sensible.

**This was not cosmetic.** At 0.072 m³ per metre the sizing asked for **33.6 km** of
swale on a 7.4 ha catchment in a 65 mm storm, so *every* recommendation returned capped
as "contour too short" and the ranking carried no information at all. At 0.60 m³/m the
same catchment asks for 4.0 km. The screenshots in the run-through show the capped list;
nobody had read it as a defect because a swale recommendation being too long for its
contour is a plausible thing for a tool to say once.

Pinned by `test_the_default_swale_has_a_floor`,
`test_the_batter_is_derived_and_therefore_not_stored` and
`test_the_default_section_is_the_registrys_swale`, which ties the criteria defaults to
the registry so the two cannot drift apart again.

**Left standing, deliberately.** The `taper_reach` **+33% over-cut** on a default swale
(`burn_strategy.py:204-207`) — documented, one-signed, and it moves every capacity figure
on the site, so it gets its own commit. Note it is now reached with a *different* batter
than before, so measure it against the defaults that ship.

---

## Round 18 — 2026-08-18 (four map-reading findings, and the one that was a real question)

From a run-through of the polished demo. Three were legibility; the fourth asked which
reference state a layer is computed from, and the answer was worth the asking.

| Finding | Outcome |
|---|---|
| 1. Swale segment overlay is hard to see against the green core | **New ramp.** Violet→purple, a quadrant clear of *both* core colours. |
| 2. Overtopping flagged while the pond is below the crest | **Not a bug — the wrong question was being answered without saying so.** Split into `(full)` and `(event)` layers. |
| 3. Surface Runoff is a translucent wash and colours every cell | **Light cyan→dark blue from 2 m³ up, fading to nothing below it.** |
| 4. Baseline and Earthworks pond layers use different gradients | **They used the same stops on different scales.** Matching pairs now share one ramp top. |

**Item 1 — the ramp was right for the ground it was designed for, and this is not that
ground.** `INFLOW_RAMP_HEX` is cyan→navy under an explicit rule (*no green, no brown —
those are the imagery's own colours*), and it is read over aerial imagery in two of its
three uses. The third, the peak-inflow overlay, is drawn **inside the recommended swale's
own green core** — and cyan sits ~46° from that green on the wheel, so the thin low bands
were competing with the band they are drawn inside rather than with the photo.

`SEGMENT_INFLOW_RAMP_HEX` is a second constant rather than a change to the first, because
the two are read against different backgrounds. Hue ~290: ≥ 90° from the green core **and**
from the amber one a capped segment turns — the second constraint is what ruled out plain
magenta (69° from amber), which would have failed on exactly the segments worth looking
at. Same four classes, same breaks, same widths, same scale control, so it is one scheme
read on two grounds. Pinned by `TestSegmentInflowRamp`, which asserts the hue gap as a
number against both core colours.

**Item 2 — the layer was answering "filled, does this pool leave over its own wall".**
`_build_overtopping_layer` passes `ctx["full"]` — the depression-fill, every hollow at its
spill point — so `overtopping_spill` measures a **capacity** state. That is the right
reference for the fault it exists to catch: a wall with no freeboard is a fault whether or
not this particular storm finds it, and it is what sizes a spillway. But it was drawn one
row from "Event Water Line", which is the *event* state, and a solid red band beside a
water line a metre below the crest reads as a claim about the storm just routed.

So: not a wrong reference state, and not merely a label. Both states are now measured.
`event_pond_depth` already runs immediately before, so its raster is handed forward on
`pond_context["event"]`; `overtopping_spill(event_depth=)` returns `event_level_m` and
`SpillOver.overtops_this_event`. **None, not False, when no event pond was available** —
a barrier nobody measured is not a barrier cleared, and that distinction is what stops a
grid mismatch from silently clearing every dam on the site.

Carried through to every place the fact appears: the label says "spills over 30 m **when
full**" or "**this event**", and the advisory drops to `pushInfo` when the event falls
short, so it stops competing with warnings that *are* about the storm just routed.

**And then split into two layers**, on the user's call, after seeing the styled version:
`Earthworks — Overtopping (full)` and `Earthworks — Overtopping (event)`, named to match
the pond pair beside them and nested the same way — (event) is a subset of (full), drawn
over it. One layer styled two ways was the wrong shape for the question: the two are asked
at different moments, and a layer cannot be half-ticked-off however it is symbolised.
Judging a design against the storm, the capacity bands are noise and go off; asking about
freeboard, they are the whole answer. Same red and edge in both — one fault on one crest —
with the capacity bands hatched and at higher alpha so they stay legible underneath.

The `(event)` layer is **absent rather than empty** when nothing overtops this event: an
empty layer asserts a question was asked and answered no, which is right when the event
was measured and wrong when there was no event pond to measure against. The `state`
attribute survives on both, because "(full)" holds two different things — `capacity` (the
event was measured and fell short) and `unknown` (nothing was measured) — that one layer
name cannot separate.

**Item 3 — the wash was never an alpha problem.** Round 17 replaced per-stop alpha with
layer opacity and was right that alpha re-orders where opacity dilutes. It still left a
55% wash, because the actual cause is that **every cell on the site has runoff**: the rain
that landed on it has to go somewhere, so the layer legitimately covers the whole map with
cells whose only message is "it rained here".

The first fix was a hard floor at that self-contribution (`runoff depth × cell area`, the
same `vol_per_cell` the volume-mode stream threshold uses). It worked, and the user
replaced it the same day with something better: **fade in over the first 2 m³** — nothing
at 0, half at 1, solid at 2 — with the colour ramp starting at 2 m³ and light cyan as its
bottom.

The floor was right about the cause and wrong about the instrument, for a reason this
layer has now produced twice: **a threshold drawn as an edge reads as water stopping
there.** Round 16 and round 17 item 8 are both that complaint. The fade reaches the same
end — a 2 m grid cell in a 65 mm storm carries 0.26 m³ and draws at 13% — without an edge
to misread, and it means the quiet cells are still *there* to be identified rather than
absent.

`SURFACE_RUNOFF_FADE_TOP_M3 = 2.0` is the one absolute number in a ramp that is otherwise
all fractions of the band maximum, and deliberately so: this is the bottom sliver of a
heavily skewed field, where a fraction of the maximum means nothing to a reader and one
cubic metre of water means something to everybody. `apply_raster_ramp(min_value=)` lays
the stops over `[2.0, top]` and prepends a transparent stop at 0 in the same colour; the
interpolated shader produces the half at 1 m³, which is asserted against the real shader
rather than against the palette.

**This is the one place alpha carries a quantity**, and the bound is what makes it safe:
the fade is *one colour*, so there is nothing below the bottom stop for it to re-order
against — which is the actual defect the "alpha is for absence" rule exists to prevent.
Every stop above the fade top stays opaque, and `test_only_absence_is_transparent` now
names the exception rather than silently excluding it.

The low end is light cyan, and the same cyan `WATER_CAPTURED` opens on, so the shallowest
water on one map and the faintest flow on the next are one colour. It was briefly white,
which reads its distance from the background fast but has nowhere to go underneath it — a
fade needs a colour to fade, and white fading out over an aerial is white fading out over
paper.

**Item 4 — same stops, different scales, and that is a difference the design did not
make.** Every raster overlay called `apply_raster_ramp` with its own `band_max`. A
Baseline layer and its Earthworks counterpart therefore stretched identical stops over
different ranges, so the same two metres of water was mid-blue before the design and navy
after it because the deepest pond on the site had moved. The pair exists to be compared,
and the visible difference was partly the ramp rescaling itself.

`_symbols.apply_shared_ramp` gives each comparable family (`ponding`, `surface_runoff`,
`streams`) one top: the largest maximum any **live** member has claimed, with earlier
members repainted when a new one raises it — a shared scale only the last layer drawn is
on is not shared. Members are held by id and pruned when they stop resolving, so a cleared
stage group releases its claim instead of propping the scale up with rasters nobody can
see. The event pond joins the same family, which generalises the argument its own comment
already made for it.

Verified in real QGIS by `check_matching_before_and_after_layers_share_one_ramp` (reads
the renderers, not the state dict — a shared top is worth nothing if it did not reach the
pixels) and `check_surface_runoff_fades_in_over_the_first_cubic_metres`.

**Two of these four went round twice, and both times the second pass came from the user
looking at the first.** Item 2 shipped as one layer styled two ways and came back as two
layers; item 3 shipped as a hard floor and came back as a fade. Neither first attempt was
wrong about the cause — they were wrong about the shape of the answer, which is the part
that only becomes obvious once it is on the screen. Worth remembering before the next
"the diagnosis is settled, so the fix is settled".

## Round 19 — 2026-08-18 (the 516 cells, explained: the conditioning made them)

The open item from Round 17. It was carried on the promise that
`unrouted_diagnostics()` would say which of several mechanisms it was rather than let
someone pick the most plausible; the field run wrote the file, and it did.

| Finding | Outcome |
|---|---|
| 1. "516 cells … hold about 13.4% of its runoff" | **`resolve_flats` creates them.** Fixed by deriving the inflation step from the surface instead of taking pysheds' fixed default. 516 → **0**. |
| 2. The diagnostic's own m³ ratio read 0.000596 against a warning saying 13.4% | **Wrong denominator.** It divided by the sum of the throughflow field. |
| 3. "The site" was the whole 2.85 km² tile, 70% of it harbour | **Advisory added.** No boundary was drawn and nothing said so. |

**Item 1 — it was never terrain, and it was never the ratio either.** The diagnostic
narrows it to one line:

```
neighbour_drop:
  conditioned:  has_lower: 0     exactly_level: 516
  filled:       has_lower: 516   exactly_level: 0
nodata_distance: no nodata in the DEM
location: at_data_edge: 0   outside_domain: 0   interior_in_domain: 516
```

Every one of the 516 had a strictly lower neighbour on the depression-filled surface and
none after flat resolution. `resolve_flats` returns `filled + eps × drainage_gradient`
(pysheds `sgrid.py`, and nothing else happens to `eps`), the gradient is an **integer BFS
distance to the flat's outlet**, and `eps` is a **fixed 1e-5 m**. So the inflation is not
a constant: it grows with the size of the flat. On this tile **70.4% of the grid
(2,002,831 cells) is one connected plane at ≈ −0.10 m** — Lyttelton Harbour, encoded as
valid ground — the gradient reaches **1842**, and the plane is lifted by up to **1.84 cm**.
Every low bump standing inside it, all of them under a centimetre proud, was lifted under
its own neighbours and became a pit: a cell with a real way downhill that the conditioning
took away, absorbing its whole upstream into a self-loop.

Three independent confirmations, none of them argued:

1. On the saved conditioned raster, the amount by which each stuck cell's lowest
   neighbour now *exceeds* it is an exact integer multiple of 1e-5 — 97% within 1e-3 of
   an integer multiple, spanning 2 to 1528 units. That is `eps`'s signature and nothing
   else's.
2. Re-running the pipeline reproduces `(516, 60 flat, 456 pit)` exactly, and
   `ponded + 1e-5 × gradient == inflated` bit for bit.
3. `d8_from_dem` over the same surface finds the identical 516 interior sinks and
   **356,171 cells draining into them — exactly the `sink=356171` in `basin_diag.txt`.**
   One cause, three symptoms, two tiers.

**The fix is a bound, not a smaller magic number.** For neighbours A and B with
`z[A] > z[B]`, the inflated surface keeps that order only while
`eps × (g[B] − g[A]) < z[A] − z[B]`. `safe_flat_epsilon` minimises that over the eight
offsets, halves it, and clamps: never above pysheds' own default, so it can only ever be
gentler than what it replaced, and never below ~1024 float64 spacings at the surface's own
elevation, so the synthetic gradient still routes. `resolve_flats_safely` gets the integer
gradient for free by calling `resolve_flats(filled, eps=1.0)` and rebuilds the surface
itself — **one flood, not two.** On this DEM the bound lands at **3.554e-07**: the default
was **14× too large**, and the tightest drop it had to respect was 7.108e-07 m per
gradient unit.

| | pysheds stuck cells | `d8_from_dem` interior sinks | cells captured | peak acc |
|---|---|---|---|---|
| `eps = 1e-5` | 516 (60 flat, 456 pit) | 516 | 356,171 | 105,981 |
| derived `eps` | **0** | **0** | **0** | 138,906 |

Afterwards the only sinks left are the 6,948 grid-border cells — `2×(1319+2157) − 4`, the
edge of the tile, which is an exit by definition. Peak accumulation rises 31%, which is
the Surface Runoff ramp top, so the map moves visibly and correctly.

The same one-line call was in `catchment.py` and `keypoint_analysis.py`. Both now use the
helper: a keypoint is *derived from* a flat-resolved surface, so a drowned drop moves one.

**The bound moves the step everywhere and the answers nowhere.** On the 400 m test fixture
it lands at 5.99e-06 rather than the 1e-5 ceiling — so the conditioned surface there *did*
change — and `check_real_terrain_numbers_have_not_moved` reports **0.00% on all nine
figures**. That is the result to want: the numbers were never sensitive to the size of a
synthetic gradient, only to whether it buried a real one. A DEM with no flat big enough to
reach the bound keeps the ceiling and is untouched.

**Item 2 — the file written to check the warning understated it 225-fold.**
`unrouted_diagnostics` divided the held volume by `nansum(field[domain])` — the sum of
throughflow over every cell, which is each cubic metre counted once per cell it passes and
is not a volume of water at all. It printed `0.000596` beside a warning correctly saying
13.4%. Anyone reading the two together would have concluded the warning was broken and
gone looking in the wrong place — the exact failure Round 17 built this file to prevent.
It now takes `runoff_volume_m3` (the worker already had it) and the field's own sum is
still reported, under `field_ink.total_throughflow_m3`, a name that says what it is.

**Item 3 — nobody had said where the site was, and nothing said so.** This DEM declares
`nodata = -9999` and contains **zero** nodata cells, so `_at_data_edge()` matched **0
cells** and the mechanism that keeps a coastal DEM quiet was inert. With no boundary and
no analysis area, `footprint.domain_mask` fell through to "every usable DEM cell" — all
2,845,083 of them, 285 ha, 200 ha of it harbour. That mask is also `runoff_volume_m3`, the
capture %, and every "% of the site" in the report. `domain_mask(with_source=True)` now
says which of its three branches answered and the baseline run pushes one advisory naming
the area in hectares. **No sea detection** — the remedy is the user drawing a boundary,
and the plugin's job is to say the boundary is missing, not to guess where it is.

**What Round 17 got right.** It refused to fix the 516 on the most plausible story and
instrumented them instead. Every candidate cause it listed — a nodata hole, an unreached
plateau, a pit that survived the fill — was wrong, and the file said so in one line each
(`no nodata in the DEM`, 479 singletons, `filled.has_lower: 516`). A day's instrumentation
bought a diagnosis that no amount of staring at the terrain would have produced.

## Round 20 — 2026-08-19 (Stage B: the spillway becomes terrain)

`SPILLWAY_NOTCH_PLAN.md` Stage B. Stage A gave the crest an honest datum; this cuts it
into the burned DEM, so a designed spillway finally moves a raster.

### The B0 gate — measured before and after, on the Quail Island design

`Quail_Island TerrainFlow Project.tfd` (42 features, 1 m clip, 858 x 1027) burned headless
under plain CPython, Round 14's method. The design carries four spillways, two of them
sited; **Dam 15 was given one** for this table, because it is the case STRETCH_GOALS §5a
is written about. Every sill is set to `spillway_datum`'s ceiling under its type's policy.

Reproduction check first: with notches off, Dam 15 comes back at **2,688.4 m³ pooling to
56.12 m** — Round 14's figure to the decimal, so the harness is measuring the same run.

| Feature | Sill | Terrain capacity, no notch | with notch | Measured spill level, no notch | with notch |
|---|---|---|---|---|---|
| Swale 4 | 69.19 | 300 m³ | **103 m³** | 69.85 | **69.19** |
| Swale 8 | 69.33 | 284 m³ | **111 m³** | 69.89 | **69.33** |
| Swale 13 | 57.25 | 247 m³ | **129 m³** | 57.70 | **57.25** |
| **Dam 15** | **55.52** | **2,133 m³** | **1,277 m³** | 56.12 | **55.52** |
| Swale 26 | 74.11 | 134 m³ | **25 m³** | 74.87 | **74.11** |

Site totals: cut 20,096 → **20,119 m³**, fill 9,888 → **9,863 m³**, Σ terrain capacity
17,813 → **16,360 m³**, total ponding 19,594 → **18,137 m³**.

**The shape is right, and it is the shape the plan asked for**: every notched feature's
*measured* spill level lands exactly on its sill, capacity falls to what the sill holds,
and Dam 15 drops out of the overtopping list entirely — it no longer leaves over its own
crest, because it leaves through its spillway.

### The 1,832 m³ figure — found, and it is a different quantity

§5a predicts "~1,832 m³ to a sill at 55.52" and the plan flags that it has no recorded
derivation. It does now. Two figures were being compared that are not the same thing:

| | no notch | with notch |
|---|---|---|
| **pool volume** — all water standing behind the wall, natural ponding included (what `overtopping_spill` reports, and where the 2,688 came from) | 2,688.4 m³ @ 56.12 | **1,832.1 m³ @ 55.52** |
| **`dam_storage`** — the pond the dam *adds*, natural ponding subtracted (what `capacity_m3` is) | 2,132.8 m³ | 1,276.6 m³ |

**1,832.1 against §5a's 1,832.** The prediction was right and it was about the pool, not
about the dam's storage. Both figures are now in the table above so the next reader does
not have to re-derive which is which.

### One decision in the plan had to change, and Stage B is what forced it

The plan has the per-feature isolated floods cut the notch too. Doing that collapses the
containment datum onto the crest: with the notch cut, the pond lets go **at the sill**, so
`FeatureStorage.level_m` comes back as the sill, `_spillway_datums` prefers the measured
level, and the crest band becomes `sill − head − freeboard`. Three consequences, all
silent: every re-open ratchets the crest down by 0.60 m, `spillway_validity` fails every
spillwayed feature for having no freeboard, and A6's "what this sill gives up" readout goes
to zero because the curve now tops out at the sill.

So the isolated flood stays **brim-full** — a container's capacity is a fact about the
container; the spillway is a control on top of it — and the volume held to the sill is read
off the stage–storage curve at the crest. Measured across all five notched features, the
curve answers the notched flood exactly:

| | crest | off the brim-full curve | notched flood | difference |
|---|---|---|---|---|
| Swale 4 | 69.19 | 103.3 | 103.3 | 0.0 |
| Swale 8 | 69.33 | 110.7 | 110.7 | 0.0 |
| Swale 13 | 57.25 | 128.9 | 128.9 | −0.0 |
| Dam 15 | 55.52 | 1,276.6 | 1,276.6 | −0.0 |
| Swale 26 | 74.11 | 24.9 | 24.9 | −0.1 (−0.2%) |

One flood, both numbers, and the datum stays a datum. The **site** burn still cuts the
notch, which is where the rasters, the routing, the ponding layers and the overtopping
check see it. `burn_earthworks(..., sills=)` and `_keyed_dam_dem(..., sills=)` keep the
notched measurement available and tested; nothing in the controller asks for it.

### What else landed

| Change | Why it is not cosmetic |
|---|---|
| **The crest bar runs *along* the alignment** (`plan_geometry.crest_bar`) | A weir's crest is the line the flow crosses; across the bank is the flow direction. Arguable as a map symbol, wrong as the cut — a notch lowered across the alignment runs down the flow path instead of through the bank. `perpendicular_sill` is kept as the breach axis. `tests/test_plan_geometry.py` and `checks_symbology.check_spillway_sill_is_drawn_at_the_built_width` both asserted the old orientation and were rewritten, not accepted. |
| **The notch is a post-pass**, after the type dispatch | Fills are `np.maximum` and `_burn_berm` is additive, so a notch cut inside a `_burn_*` is plugged by a later feature. Pinned by a test that burns a berm straight over a notched dam. |
| **`_keyed_dam_dem` cuts it too** | It bypasses `burn_earthworks` entirely and Dam 15 is keyed, so without this the change is invisible on dams. Both paths call one `_cut_spillway`. |
| **Four guards, each with its own message** | No daylight (the march hits its cap with the bank still above the crest); discharges back into its own *enclosed* pond (the keyed-berm geometry, which the daylight test cannot see); a crest at or below the burned floor; an orphaned sill the controller could not snap. A notch that quietly does nothing looks exactly like a working spillway in every figure the design tier prints. |
| **"Enclosed" is load-bearing in the pool test** | "Below the crest and touching the footprint" describes the whole hillside under the sill, and using it refuses every spillway on falling ground — the ordinary case. Components reaching the window edge are discarded as open ground. |
| **Overtopping subtracts the notch from the barrier crest** | Without it a correctly spillwayed dam still reports "leaves over its own crest" — at its own spillway. `overtopping_warning`'s `has_spillway` caveat ("not cut into the terrain model, so the analysis cannot route water through it") is **deleted**; reaching that function with a spillway now means the sill is not taking the water, and it says so. |
| **`fill_pct` divides by the brim volume**, not the sill volume | Otherwise a working spillway pins its feature at 100% exactly when it starts doing its job. `EarthworkStore.lip_capacity_m3` defaults to 0.0 and every reader falls back to `capacity_m3`, so nothing without a measurement changed. `cascade_overflow` keeps thresholding on `capacity_m3`, which with the notch is correctly the sill volume. |
| **An auto width is no longer persisted** (STRETCH_GOALS §10(d)) | `width_auto` means the width tracks a requirement the design file does not pin down, so opening an old project already rewrote it silently on the first live recompute. Rounding to whole cells would have made that rewrite a *visible* unexplained change to a saved design. The flag is stored; the number is derived, and the restore path recomputes it before the list, the map label or the sill bar render. |
| **The burned width is whole DEM cells** | A sill narrower than a cell cannot be cut as one. Nothing in the raster tier meters flow *rate*, so this cannot change a total — it changes `cells.size` at the exit, and with it the `q = Q/L` the erosion advisory is judged by. The note says exactly that, and deliberately not "the extra width lowers the head": true of the weir equation, and it reads as though widening moved the water level in the feature, which it does not. |
| **Three elevations per spillway**, all measured | Designed sill, sill as burned, and where the pond was actually found to let go. They are allowed to disagree and each disagreement names a different fault. On the Spillways review as a `Sill` column plus the tooltip, and on the report's *Overflow safety* page as its own table. |
| **A `Spillways (burned)` layer under Verify** | "Did the model cut my spillway, and where?" had no answer on the map. The design-stage bar is what was asked for; this is what happened. |

### Not done, and why

**The measured spill level for B6 comes off the site burn's ponding**, per feature, rather
than from a second isolated flood — one labelling pass over a raster the verification has
already read. `_record_spillway_levels` runs beside `_compute_verification` for that
reason.

### Verification

`python -m pytest tests/` — **2,590 passed**, coverage **95.34%** against the 95% gate.
`.\run_qgis_tests.ps1` bare — **200 passed, 0 failed**, every screenshot pixel-identical.
`python -m ruff check` clean.

New cases in `tests/test_spillway_notch.py` (55) covering the pure grid functions, every
refusal, the keyed-dam path and the isolated-burn snapshot; `tests/test_report_model.py`
gains the gauge and as-burned tables; `tests/test_spillway.py` gains the auto-width
persistence rule. `tests/test_plan_geometry.py`'s perpendicularity cases were rewritten to
assert the crest runs along the alignment, with the breach axis kept as its own class.
`checks_earthworks` gains `check_a_placed_spillway_is_cut_into_the_burned_dem`, which is
the whole change end to end in real QGIS: place a sill, re-analyse, and assert the burn
recorded a notch, the burned surface is down to the designed crest along it, the as-burned
sill was measured, and the `Spillways (burned)` layer landed under **Verify** through
`_groups` rather than loose at the top of the legend.

**One screenshot moved and was accepted**: `panel_spillway_review.png` gains the **Sill**
column. A second, `report_page8.png`, moved and **should not have** — and accepting it
first is how the defect behind it was found. It showed *Width designed 2.00 m* beside a
status of *No spillway designed*, which is a width for a structure that does not exist.
Cause: `width_auto` reads True for a feature with no `Spillway` object at all (there is
none to ask), so the row derived a built width for every undesigned feature. The panel hid
it — `SpillwayTable._width_cell` branches on `designed` first and prints "0.6 m needed" —
while the report printed `built_width_m` straight into its own column. What an undesigned
feature has is a *requirement*, and `required_width_m` already carries it. Guarded on
`spillway is not None`; the page is back to an em dash.

Worth keeping the number that made it visible: on the suite's **2 m** fixture a 0.62 m weir
rounds up to **one cell, 2.00 m** — not two cells. On a 1 m DEM the same weir comes out at
1.00 m. The jump looks large only because the grid is coarse relative to the sill, and that
is the honest reading: on a 2 m model the smallest weir that can be cut at all is 2 m wide.

**Two assertions were rewritten rather than accepted**, as the plan required:
`tests/test_plan_geometry.py`'s perpendicularity cases and
`checks_symbology.check_spillway_sill_is_drawn_at_the_built_width`. The second is an
assertion on rendered geometry, not a screenshot, so `-Accept` could never have absorbed
it — it now asserts the bar runs *along* the feature and fails if it is square across.

**Two things the run itself caught, worth recording because neither was visible from the
code.** The new QGIS check first failed with *no crest to cut to*: `_on_spillway_placed`
only seeds a crest when it is handed an elevation, and passing `None` sites a point with
no sill. It then failed with *the as-burned sill was never recorded* — that figure was
gated behind `state.pond_context`, which only exists once the verification pass has run,
and the verification skips any feature with no analytic capacity. It comes off the burner
and needs no pond, so `_record_spillway_levels` now records it first and unconditionally;
only the measured spill level waits for the pond. A feature drawn but not yet sized would
otherwise have had a notch in the terrain and nothing on the review saying so.

## Never run

**Manual QGIS smoke tests.** Everything above is verified by `pytest`, `ruff`, the CI
grep-gate, and a static pass over signal/method/kwarg wiring. No agent session has ever
executed QGIS, so nothing painted has been seen rendering:

- Flow and Elevation charts past ~10 features (chip crowding, sideways-nudge overflow)
- ~~`verification_table` column widths against real feature names~~ — now shot by
  `checks_visual.check_verification_table_renders` (four row states, and an assertion
  that the columns fit the dock viewport) and `checks_report`'s verification page
- The inflow sparkline on a genuinely uneven catchment
- ~~Save → restart → reopen for spillways (both kinds)~~ — now covered by
  `checks_earthworks.check_spillway_survives_a_project_roundtrip` (both kinds, plus a
  freeboard override and a committed width). The HIRDS table is still unverified.
- The Spillways review against a design large enough to scroll (it caps at 8 visible rows)
