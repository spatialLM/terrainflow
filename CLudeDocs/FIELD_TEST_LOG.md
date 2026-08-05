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

## Never run

**Manual QGIS smoke tests.** Everything above is verified by `pytest`, `ruff`, the CI
grep-gate, and a static pass over signal/method/kwarg wiring. No agent session has ever
executed QGIS, so nothing painted has been seen rendering:

- Flow and Elevation charts past ~10 features (chip crowding, sideways-nudge overflow)
- `verification_table` column widths against real feature names
- The inflow sparkline on a genuinely uneven catchment
- Save → restart → reopen for spillways (both kinds) and the HIRDS table
