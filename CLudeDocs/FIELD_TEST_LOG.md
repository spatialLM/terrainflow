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
