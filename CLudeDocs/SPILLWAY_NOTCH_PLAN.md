# Cut the spillway into the terrain

## Context

`spillway-implementation-spec_1.md` asks for one thing and three that hang off it: the burn
is **spillway-blind**, so a designed spillway changes no raster, no routing and no pond, and
the reanalysis cannot reproduce the overflow the design intends. Around that sit a
depth-reference rule per structure type, a width-vs-DEM-resolution problem, and diversion
drains that take their start level from a spillway instead of by guesswork.

This is the project's own next-up item (`CLudeDocs/STRETCH_GOALS.md` §5a, flagged 2026-08-14).
Dam 15 on the Quail Island design impounds a **2,688 m³ pool to its wall crest against ~1,832 m³
to a sill at 55.52**, and every overtopping advisory ships a caveat saying the model cannot see
the structure the user designed.

Intended outcome: a placed spillway becomes a cut in the burned DEM, so pond level, capacity,
routing, verification and the report describe the structure that was designed, and the caveat
gets deleted rather than reworded.

**Decisions taken with the user:**
- Round the burned width to whole **DEM cells**, not a literal 1 m (identical on today's 1 m
  DEM; correct on the drone data STRETCH_GOALS §1 names as the fidelity path).
- **Berms stay out of scope** — spillways remain on swale / dam / basin.
- **Skip STRETCH_GOALS §6's capacity/threshold split** (see the rationale under B6 — it is not
  the one CLAUDE.md's "optional enrichment" line would suggest).
- **Keep the measured before/after** as the gate on Stage B.
- **Reference the crest to the swale floor**, not the rim (user's proposal — see A1).
- **Three stages, each shippable.**

## What the code already gives us

| Already exists | Where |
|---|---|
| `Spillway` with crest, head, freeboard, width, `width_auto`, `point_wkt`, all persisted | [earthwork_design.py:284](terrainflow_assessment/modules/earthwork_design.py#L284) |
| A design invert of `pour_point − depth`, returned by `_spillway_datums` and cut by the burn | [burn_strategy.py:137](terrainflow_assessment/modules/burn_strategy.py#L137), [earthworks.py:2508](terrainflow_assessment/qgis/controllers/earthworks.py#L2508) |
| Two-way crest binding that clamps *before* deriving its partner | `bind_crest`, [earthwork_design.py:381](terrainflow_assessment/modules/earthwork_design.py#L381) |
| Measured berm crest, per feature | `_record_berm` → `ew.berm_crest_elevation`, [earthwork_design.py:1608](terrainflow_assessment/modules/earthwork_design.py#L1608) |
| The pond's actual spill level, computed on draw / dialog-OK / vertex-release and **discarded** | `FeatureStorage.level_m`, [earthwork_design.py:2360](terrainflow_assessment/modules/earthwork_design.py#L2360) |
| Three mask records held apart on purpose, so a fourth is the established pattern | `burned_masks` / `burned_cut` / `burned_raised`, [earthwork_design.py:1663](terrainflow_assessment/modules/earthwork_design.py#L1663) |
| The "which side is downhill" test | `_ground_mean`, used by `_downstream_footprint` and `_companion_berm` |
| The snap from a clicked point to a segment on the feature | `_spillway_sill`, [earthworks.py:287](terrainflow_assessment/qgis/controllers/earthworks.py#L287) |
| Dangling-id tolerance resolved at read time | `overflow_target_id` → `resolve_targets`, [simulation.py:177](terrainflow_assessment/modules/simulation.py#L177) |

**The fact that makes most of this cheap:** `find_impoundments` is called with `built=None` and
measures a hollow's storage as `Σ(filled − ground)` with no idea what made it
([crest_routing.py:309](terrainflow_assessment/modules/crest_routing.py#L309)). Lower the pour
level and retention, exit cells, Surface Runoff, exit volumes and the ponding layers follow with
**no changes to `crest_routing` or `flow_analysis`**. For cut features `feature_storage` floods
`burn_earthworks([ew])`, so the isolated measurement picks the notch up too.

**The exception that must be built explicitly — and it is the headline case.** A *keyed* dam
never goes through `burn_earthworks`: `dam_stage_storage` passes
`isolated_dem=_keyed_dam_dem(dam)` ([earthwork_design.py:2500](terrainflow_assessment/modules/earthwork_design.py#L2500)),
which constructs its surface by hand, and `_refresh_terrain_capacity` returns early for dams
([earthworks.py:1083](terrainflow_assessment/qgis/controllers/earthworks.py#L1083)). **Dam 15 is
keyed** (`key_into_banks: True`). A post-pass alone would move no dam capacity, no dam Verify
reference, no `EarthworkStore.capacity_m3` and no dam figure in the report.

**And the fact that frames the width question:** nothing in the raster tier meters flow *rate*.
`cascade_overflow` transfers surplus unconditionally in one step; `spread_crests` divides a
pond's surplus **equally** over its exit cells; `peak_flow` is imported by no module. Burned
width therefore cannot change a **total** — but it does change `cells.size`
([crest_routing.py:598](terrainflow_assessment/modules/crest_routing.py#L598)), so the *per-cell*
runoff at the exits, and the `q = Q/L` the erosion advisory is judged by, move with it. Rounding
up is about **rasterisability**, not conservatism, and the copy should say exactly that.

---

## Stage A — the datum

*This stage moves saved crests and published spillway figures. It is not raster-neutral — see A6.*

### A1. Reference the crest to the floor

The design invert is single-valued: `pour_point(original, mask) − depth`, which is what
`_spillway_datums` returns and what `_storage_invert` cuts to at full depth. The rim is a
**global** ring minimum over the whole footprint (`footprint.py:262`, `argmin` over
`outer_ring`) — and `help_text` admits "for a contour swale the lowest surrounding ground is
usually at one of the ends" — so a mid-run notch referenced to it is cut at the wrong level on
any swale that falls along its run.

- Add `Spillway.height_above_floor_m` to `_SERIAL_FIELDS`; `from_dict` probes the data dict per
  field, so no version bump. (Note the corollary: an **older** build re-saving the design drops
  the field silently, because `to_dict` iterates its own shorter tuple and `SCHEMA_VERSION` stays
  1. Bump it so `is_from_newer_build()` warns.)
- Extend `bind_crest` to a **three-way** binding — `crest_elevation` (absolute, authoritative,
  the thing that gets burned) ↔ `drop_below_rim_m` ↔ `height_above_floor_m`. It already clamps
  before deriving partners, which is what stops hand-written bindings creeping apart.
- Make height-above-floor the primary control in the dialog's spillway group; keep the absolute
  crest visible. It is the only one of the three a builder can set out with a staff from the
  trench bottom.
- **Precision, not overclaim:** the *burned* floor is only level where `taper_reach == 1`.
  `_storage_invert` routes to `tapered_invert` whenever `channel_batter_run(ew) > 0`, which for a
  swale is normally non-zero, so a narrow footprint never reaches the design invert. The datum is
  the **design** invert; B3's guard tests the **burned** floor.
- **Dam is unchanged** — spec §2 fixes it to the crest wall, which `_spillway_datums` already
  does. Its `invert` is the lowest ground under the wall, not a cut floor, so height-above-floor
  is offered for cut features only.

### A2. Containment becomes a reported clearance, not the datum

`_spillway_datums` returns `(rim, invert)`; make it `(lip, invert, containment)` and **say which
goes where**: `containment` feeds `spillway_datum`'s ceiling and `spillway_validity`; `lip` is
reported beside it; `_spillway_row["rim_elevation"]` carries `containment`, with the lip added as
a new key so the table and report can show both.

- `containment` = `ew.berm_crest_elevation` where a companion berm was built, else the lip.
  Measured, never a berm *height estimate* — which is what STRETCH_GOALS §10(a) rules out.
- Once a burn exists, prefer the measured `FeatureStorage.level_m`, retained as
  `ew.terrain_spill_level_m` (**not** serialised — same rule as `terrain_capacity_m3`). This
  closes §10(a) and answers its open question: *before a burn, the analytic level, said so;
  after one, the measured level.*
- **`spillway_validity` must learn the difference.** Its first check refuses a crest "above the
  lowest containing ground" ([earthwork_design.py:444](terrainflow_assessment/modules/earthwork_design.py#L444))
  and any non-empty `problems` sets `state = "fail"`. A crest above the *lip* but below the
  *berm crest* is now legitimate; the check must test `containment`, and the lip clearance
  becomes a note, not a problem.

This is what stops the storage cliff: `_record_mask`'s own docstring records Swale 5 ponding to
**69.60 m against a ring minimum of 68.88 m — 1,095 m³ against 439 m³**.

### A3. Take the rim locally

Once a spillway is sited, compute `pour_point` over the ring within ±`max(width_m, cell)` of the
sill station rather than the whole footprint. That makes "sill = lip of excavation" true *where
the user clicked* — the spec's §6 sanity check — and is what makes a partially-cut swale
answerable at all.

### A4. Put the datum and the burn on one surface

Two mismatches, not one:

- `_spillway_datums` samples `state.flow_dem`, the **conditioned** DEM (depression-filled and
  flat-inflated), while the burn's datum is `pour_point(self.original, …)`. Move it to the raw
  DEM. `PlacePointTool` already samples `state.dem_path`, the file `DEMBurner.original` reads,
  so it needs no change.
- `_footprint_mask` takes `rasterize_footprint`'s default `all_touched=True` while the volumetric
  burns pass `False`. Match the burn.

### A5. Clamp the map-placed crest (STRETCH_GOALS §10b — "fix both or neither")

`_on_spillway_placed` calls `bind_crest` **without** `band=`
([earthworks.py:386](terrainflow_assessment/qgis/controllers/earthworks.py#L386)), so a clicked
crest is never held inside its band. Once it is burned, that is a notch cut at whatever elevation
was under the cursor. Land it with A4, as §10(b) requires.

### A6. Show what the sill costs, live, as the user sets it

The crest control should say what it is giving up, in m³ and as a percentage, while it moves.
That is the decision the user is actually making, and today nothing in the UI answers it.

- **`feature_storage` returns a stage–storage lookup** alongside `volume_m3` — sorted bed
  elevations over the pond region plus a cumulative volume, enough to answer `volume_at(z)` by
  interpolation. It must be built **inside** `feature_storage`, because `region` is in the flood
  window's frame and a caller cannot integrate it. Mirror `reporting.level_for_volume`
  ([reporting.py:829](terrainflow_assessment/modules/reporting.py#L829)), which already solves
  the inverse by the same sorted-array method.
- **No second flood.** This is extra return from the pass `_refresh_all_terrain_capacities`
  already makes — one sort and a cumulative sum over the pond. Retain it as `ew.stage_storage`,
  derived and not serialised, same rule as `terrain_capacity_m3`.
- **The denominator is the containment level from A2** — what the feature would hold with no
  spillway. So the readout is honest about what "100%" means: *"holds 1,832 m³ of 2,688 at this
  sill — giving up 856 m³ (32%)"*.
- **Before any burn there is no curve.** Say so rather than printing a zero, the way
  `_update_berm_crest` already does in this same dialog ("run Re-analyse with Earthworks").
- Carry the same figure per row in the Spillways review, so it can be read across the whole
  design without opening each dialog, and on the report's *Overflow safety* page — both
  renderers, parity test as the gate.
- **Check the dam path separately.** `dam_stage_storage`
  ([earthwork_design.py:2479](terrainflow_assessment/modules/earthwork_design.py#L2479)) reaches
  `feature_storage` with its own `isolated_dem`, and despite the name may return a single volume
  rather than a curve. It is the natural home for the dam's lookup either way.

This also delivers STRETCH_GOALS §6's stated goal — *"% full should never reach 100%"* — as a
**readout**, without the `simulation.py` threshold change B7 argues against.

### A7. Unticking the spillway group must not silently discard it

`ew.spillway = dlg.get_spillway()` runs unconditionally on OK, and `get_spillway()` returns
`None` when the group is unticked — taking `point_wkt`, the crest and the width with it
(STRETCH_GOALS §10(c)). Harmless while a spillway changed no terrain; from Stage B onward one
stray untick silently un-cuts a hole in a dam and moves its capacity by ~30% with nothing said.

Confirm before clearing, naming what is lost ("this spillway is sited at 56.12 m — remove it?"),
and only then assign `None`. Land it in Stage A so the protection is in place *before* the notch
gives it teeth.

### A8. What Stage A moves, and its migration

`drop_below_rim_m` is serialised and is re-based the moment `rim` changes meaning, so **every
saved design's stored drop becomes wrong against its own crest.** `crest_elevation` is the
authoritative absolute, so migrate on load: recompute `drop_below_rim_m` and
`height_above_floor_m` from the stored crest and the new datums, and leave the crest untouched.

Also moving, all before any notch exists: `row["rim_elevation"]`, `row["freeboard_m"]`,
`row["problems"]`/`row["state"]`, the `SpillwayTable`, `report_model._page_spillways`, and the
`checks_visual.check_spillway_review_renders` screenshot. Stage A's gate is *those*, not rasters.

---

## Stage B — the notch

### B0. Measure first

Burn the Quail Island design headless under plain CPython (Round 14's method; the `.tfd` and
rasters are on disk) with notches off and on. Table per feature: the **pool volume** from
`overtopping_spill` (Dam 15 = 2,688 m³ today — this is the pool, *not* `terrain_capacity_m3`,
which reads 2,098.87), `terrain_capacity_m3`, site capture % and cut/fill.

**Acceptance is the shape, not a single number.** Dam 15's pool must truncate at its sill —
55.52 is `56.12 − 0.30 − 0.30`, exactly `spillway_datum`'s ceiling under the dam's registry
policy — and land in the region of the 1,832 m³ STRETCH_GOALS §5a predicts. That figure has no
recorded derivation in the repo, so treat it as an expectation to explain a deviation from, not
a threshold to pass. The table itself goes into `FIELD_TEST_LOG.md`.

### B1. The crest bar runs *along* the alignment

`perpendicular_sill` returns a bar square across the alignment and its docstring calls that "the
actual crest" ([plan_geometry.py:59](terrainflow_assessment/modules/plan_geometry.py#L59)). A
weir's crest lies **along** the bank the flow crosses; across is the flow direction. That is the
argument — on hydraulics. (It is *not* that `spillway_validity` contradicts it: that check
compares the required width against `feature_length_m`, which for a basin is the perimeter, and
its own comment calls it a deliberately generous sanity bound. And the perpendicular bar is
defensible *as a cartographic gate symbol*, which is how `_symbols.spillway_symbol` describes
it. What is wrong is using it as the burned geometry, and the docstring calling it the crest.)

- New `plan_geometry.crest_bar(points, seg_index, centre, width_m)` — tangent to the local
  alignment, length = burned width. The breach axis stays perpendicular; `perpendicular_sill`'s
  existing maths becomes that.
- The map symbol moves with it, so the drawn crest and the burned crest are one geometry.
- **Two existing tests assert the perpendicularity and must be rewritten, not accepted:**
  `tests/test_plan_geometry.py:62-106`, and
  `tests_qgis/checks_symbology.check_spillway_sill_is_drawn_at_the_built_width`, which is an
  assertion (`off > radians(5)`), not a screenshot — `-Accept` cannot absorb it.

### B2. Cut it

**The snap stays in the controller.** `plan_geometry`'s docstring forbids a second
implementation of "nearest point on this alignment, and which segment", and `burn_earthworks`
receives only `Earthwork` objects. So `_spillway_sill` resolves the crest bar as it does today
and the controller passes `burn_earthworks(earthworks, sills={ew_id: crest_bar_wkt})`. The burner
stays QGIS-free; the snap stays in one place; a sill the controller could not resolve simply is
not in the dict, and B3 reports it.

- `burn_strategy.spillway_notch(...)` and `daylight_reach(...)` — pure grid functions, testable
  under the coverage gate.
- `DEMBurner._cut_spillway(dem, ew, bar)`: crest bar at
  `burn_width = ceil(width_m / cell) * cell`, marched outward on the lower side (`_ground_mean`)
  until burned ground sits at or below the crest, capped; rasterised `all_touched=True` with the
  `line_cells` fallback, because this is `_rasterize`'s own "did we lose the feature?" question.
- `dem[notch] = np.minimum(dem[notch], spillway.crest_elevation)`. Cut only, absolute, taken from
  the stored design and never re-derived at burn time.
- **Outflow only.** Every earthwork also carries an `inflow_spillway` with its own crest; an
  inlet is not a weir, and notching it would drain the pond at the inlet. `_refresh_auto_spillway_widths`
  already draws this line and says why ([earthworks.py:1993](terrainflow_assessment/qgis/controllers/earthworks.py#L1993)).
- Record `burner.burned_notches[id]` as a **fourth** mask record — widening `burned_masks` would
  move pool attribution and every bermed swale's Δ. **Add it to `_isolated_burn`'s
  snapshot/restore list** ([earthwork_design.py:2347](terrainflow_assessment/modules/earthwork_design.py#L2347)),
  which today preserves only four records; without that, the per-feature isolated burns that run
  after the site burn leave the dict holding the last feature's notch — which is exactly what B5
  reads.

**Run it as a second pass in `burn_earthworks`, after the type dispatch.** Fills are `np.maximum`
and `_burn_berm` is additive (`dem[mask] += ew.depth`), so a notch cut inside a `_burn_*` is
plugged by that feature's own berm or by a later feature. A post-pass is the only place where
"the notch is the last thing to touch these cells" is true by construction, and it stays
order-independent because the level is an absolute carried on the design.

### B3. The keyed-dam path, and the guards

- **`_keyed_dam_dem` must cut the notch too**, and `_refresh_terrain_capacity`'s dam branch must
  stop returning early once a dam carries a sited spillway. Without this the whole change is
  invisible on dams — the case §5a is written about. Factor the notch cut so both the post-pass
  and `_keyed_dam_dem` call one function.
- **Daylight.** If the outward march hits its cap with every cell still above the crest, the
  notch discharges into rising ground and changes nothing. Cut nothing; warn with the reach
  tried, through `DEMBurner.warnings` → the message bar.
- **Discharges into its own pond.** A berm keyed at both ends wraps around them
  (`_key_berm_into_banks`), so the march can find ground below the crest that is still *inside*
  the pool — the daylight test passes and the notch still does nothing. Walk `walk_downslope`
  from the outermost notch cell and require it to leave the pool. This is the one failure the
  daylight check cannot see, and it is exactly the geometry a keyed swale presents.
- **Below the floor.** Refuse a crest at or under the **burned** floor (not the analytic
  `rim − depth`) — that catches a `tapered_invert` that could not reach full depth.
- **Orphaned sill.** `_spillway_sill` returns `None` past its 5 m tolerance. Today that skips a
  map symbol; now it would skip a burn. Warn loudly.
- **Re-measure on placement.** `_on_spillway_placed` never calls `_refresh_terrain_capacity`
  ([earthworks.py:390](terrainflow_assessment/qgis/controllers/earthworks.py#L390)) — correct
  while a spillway changed no terrain, wrong the moment it does. Add it.

### B4. Width rounding, and where the note goes

- `burn_width_m = ceil(width_m / cell_size) * cell_size`. State the axis: use
  `max(cell_h, cell_w)` unless the crest axis is known, matching the caution `taper_reach`
  already carries about non-square grids.
- **Stop persisting an auto width** (STRETCH_GOALS §10(d)) — this is what makes the rounding
  question clean rather than worked around. `_refresh_auto_spillway_widths` writes `width_m` on
  every `_recompute_live_assessment`, which also fires on storm and rainfall changes and runs
  during restore, without marking a design edit — so today opening an old project already
  rewrites widths the user never chose, and rounding would make that a *visible* unexplained
  change to a saved design. Exclude `width_m` from `to_dict` when `width_auto` is set; it stays
  on the in-memory object and is recomputed by the restore path's own
  `_recompute_live_assessment` ([earthworks.py:1368](terrainflow_assessment/qgis/controllers/earthworks.py#L1368)).
  Verify that ordering — the map label, the sill bar and the review row all read `width_m`, and
  they must not render a 0.0 between load and the first recompute.
- With that, the rule is simple: a **committed** width is stored rounded (the fixed §5 decision),
  and an **auto** width is derived, rounded on the way to the display and the burn.
- **The real un-rounding site** is `_update_spillway_sizing`, which overwrites `spin_built_width`
  with the raw `calculate_spillway_width` result on every repaint while `width_auto`
  ([earthwork_properties_dialog.py:1259](terrainflow_assessment/earthwork_properties_dialog.py#L1259)).
  `get_spillway` is a straight widget read. The dialog also has no DEM cell size today — add it
  as a constructor argument.
- Add `Spillway.width_required_m` (it does not exist; the requirement is recomputed per row and
  per repaint) so the note can quote both figures, and so the manual path's adequacy checks keep
  testing the requirement. Note the rationale is narrower than it looks: `effective_head_m`
  short-circuits on `width_auto`, and `spillway_validity`'s width check can only become *less*
  true when rounding up — so this matters on the **committed-width** path only.
- `MIN_SILL_M = 0.5` becomes unreachable on any cell ≥ 0.5 m. Retire it or scope it to drawing.
- **The note must NOT go through `spillway_validity.problems`.** `_spillway_row` sets
  `state = "fail"` on any non-empty `problems`
  ([earthworks.py:2110](terrainflow_assessment/qgis/controllers/earthworks.py#L2110)), which
  would mark every auto-sized spillway failed. Add a parallel `notes` list, rendered muted in
  `SpillwayTable` — one producer, both surfaces.
- One sentence, generated in the pure layer: *"2.0 m built — rounded up from the 1.4 m the flow
  needs, so the 1 m terrain model can cut it. Water still leaves at the same level, so this
  changes nothing the feature holds; a wider sill only runs the overflow shallower."*
  **Not "the extra width lowers the head"** — true of the weir equation, but it reads as though
  widening moved the water level in the feature, which it does not. Width is horizontal; the
  overspill point is vertical and unchanged.
- **While here, fix the same confusion in the help text.** `H` in `Q = C·L·H^1.5` is the depth of
  flow *over* the sill at the peak, not a dimension of the structure — a wider sill passes the
  same flow shallower (1.4 → 2.0 m takes a 0.30 m nappe to ≈ 0.24 m, so the peak water surface
  sits ~6 cm lower and freeboard gains that much, while stored volume is untouched). Round 4
  settled the design logic here (head is the target while width is free; the consequence once
  width is committed) but the copy never says what head physically *is*. Audit `help_text`'s
  spillway constants and the dialog's "Design head" / actual-head row labels against that
  sentence.
- **The note reaches the report, so CLAUDE.md's parity rule applies**: extend
  `report_model._page_spillways`, then **both** `report_html.SECTION_HANDLERS` and
  `layout_pdf.SECTION_HANDLERS`, with `tests/test_report_renderer_parity.py` as the gate.

### B5. Fix overtopping in the same change

`overtopping_spill` selects `own = rim & crest & (bed <= pour + 1e-6)`
([reporting.py:800](terrainflow_assessment/modules/reporting.py#L800)) and the barrier crest is
built as `mask & raised`. A notch cut into raised ground stays in `raised`, so a correctly
spillwayed dam would still report "leaves over its own crest" — at its own spillway. Subtract
`burned_notches[id]` from the barrier crest in the controller block, and report the notch
separately as discharging through its designed spillway. Then delete `overtopping_warning`'s
`has_spillway` caveat ([burn_strategy.py:542](terrainflow_assessment/modules/burn_strategy.py#L542))
and the matching prose in the module docstring.

### B6. Record the elevation water leaves at — three numbers, told apart

The level water leaves at is the single most important output of this feature, and it should be
reported the way the plugin already reports every other quantity that exists in a designed and a
measured form (Round 10's CALCULATED / MEASURED discipline):

| Number | Where it comes from | Persisted? |
|---|---|---|
| **Designed sill** | `Spillway.crest_elevation` — already absolute, already persisted, and the value B2 burns | yes |
| **As-burned sill** | new `ew.burned_sill_elevation_m` — `min(modified_dem[burned_notches[id]])`, measured off the cut | no (derived, cleared with the DEM) |
| **Actual spill level** | `ew.terrain_spill_level_m` from `FeatureStorage.level_m` (A2) — where the finished pond really lets go | no |

They can legitimately disagree, and each disagreement names a different fault:

- **as-burned > designed** — the notch could not be cut that deep, or the cell it landed in was
  already lower elsewhere.
- **actual spill > as-burned** — the notch did not daylight, so the pond still leaves somewhere
  else. This is B3's daylight guard, stated as a number rather than only as a warning.
- **actual spill < designed** — a saddle elsewhere on the rim is lower than the sill; the
  spillway is not the control.

Surface all three in the Spillways review row and on the report's *Overflow safety* page (both
renderers, parity test as the gate). This is what turns "the model can now see my spillway" from
a claim into something the user can check.

### B7. "% full" becomes a three-state gauge, not a number that pins at 100

STRETCH_GOALS §6 is **delivered**, not skipped — but as a readout off A6's stage–storage curve
rather than a `simulation.py` redesign. (The reasoning I first gave for skipping it was wrong:
`cascade_overflow` is reached from `run_water_balance`, which *is* the design tier the report is
built from. CLAUDE.md's "optional enrichment" line is about `state.comparison`, the fill
simulation, not `simulation.py` the module.)

Three volumes off one curve, and the gauge reads all three:

| | Volume | Meaning |
|---|---|---|
| **Max storage** | at the **sill** — `capacity_m3` once the notch is burned | what the feature is designed to hold. The spillway is the control, so this is the working ceiling. |
| **100%** | at the **containment level** (A2's lip / berm crest) | the brim. Above it, water leaves over the structure rather than through the spillway. |
| **Surcharge** | at `sill + effective_head_m` | where the water actually stands while the design storm passes the weir. |

- **`fill_pct` divides by the lip volume, not the sill volume.** Give `EarthworkStore` a
  `lip_capacity_m3` defaulting to `capacity_m3`, so every existing caller is unchanged where it
  is absent. `cascade_overflow` keeps thresholding on `capacity_m3` — which, with the notch cut,
  is correctly the sill volume — so the cascade needs no change at all.
- **The band between the max-storage mark and 100% is the spillway doing its job.** It is reached
  through the *rate* tier, not the volume tier: `effective_head_m` already returns the head the
  water will actually stand at given the built width and the peak flow, and the curve converts
  that to a volume.
- **Touching 100% means the spillway is not sufficient** — `sill + effective_head ≥ containment`.
  That is the same condition `spillway_validity`'s freeboard check already fires on, so the gauge
  and the warning cannot disagree about the same feature.

Everything this needs exists: the curve from A6, `effective_head_m`
([earthwork_design.py:1352](terrainflow_assessment/modules/earthwork_design.py#L1352)), the peak
flow from `_state.peak_flows`, and the containment level from A2. Surface it on the scorecard and
in the Spillways review; the report's storage table gains the max-storage mark alongside the
figures it already prints.

### B8. Make the result visible on the map and in the review

- **Draw what was actually cut.** A `Spillways (burned)` layer from `burner.burned_notches`,
  placed via `self.place(layer, G.VERIFY)` through `_groups.py` — never `addMapLayer`. "Did the
  model cut my spillway, and where?" is unanswerable from the map today; the mask is already in
  hand, so this is close to free, and it is the visual partner to B6's three elevations. A second,
  dimmer bar at the burned width beside the design bar carries the same information on the Design
  stage, before a burn exists.
- **Flag spillways that pass nothing this event.** Where the sill sits above the measured event
  water level, the spillway does nothing in the design storm — which distinguishes *sized
  correctly* from *never tested*, and is not otherwise visible. Both figures are already computed
  by `_build_event_pond_layers`; this is a column in the Spillways review and a line on the
  report page.

---

## Stage C — the spillway-linked diversion drain

- `Earthwork.spillway_link_id = "<ew_id>:outflow|inflow"` or `None`, added to `_SERIAL_FIELDS`.
  An id, not a boolean — a flag cannot say *which* spillway. Confirms the spec's §5 lean.
- **Set an absolute start invert, not a depth.** `depth` on a diversion feeds
  `calculate_cut_volume`, `calculate_diversion_discharge`, `taper_reach` and the sub-cell check
  `max(0.05, width − 2·depth)` — overwriting it corrupts the section and the Manning discharge.
  Add `Earthwork.invert_start_m` (absolute, **derived, not serialised**, recomputed on restore);
  `_burn_diversion` prefers it over its ground sample
  ([earthwork_design.py:2079](terrainflow_assessment/modules/earthwork_design.py#L2079)).
- **Live link, resolved at read time**, mirroring `overflow_target_id`: dangling → fall back to
  the ground sample and say so in the Spillways list. Do not eagerly clear links on delete — that
  is a second, silently different failure mode. Confirms the spec's §5 lean; its undo/redo
  concern is empty, because there is no undo stack anywhere in the plugin.
- **Refuse cycles rather than assuming them impossible.** `SPILLWAY_TYPES` is a controller
  constant and is not enforced on the model, so the bipartite argument is not load-bearing. Build
  `{drain_id: source_id}` and run `topological_order`, reusing `on_connection_made`'s existing
  refusal message shape. Also refuse self-links.
- **Link as an action, not a draw button** — draw a diversion as now, then "snap this end to a
  spillway", the shape `activate_connect_earthworks` already uses.
- Sort the burn so a source precedes its linked drains — `_burn_diversion` reads the **running**
  array, so this also removes a latent draw-order dependency that exists today.

---

## Approaches considered

| Approach | Verdict | Why |
|---|---|---|
| **Burn a level notch as a post-pass, at the stored absolute crest** | **Selected** | The only mechanism that moves the *rasters* — and `find_impoundments`/`feature_storage` propagate it with no new plumbing. Post-pass because fills are `np.maximum` and `_burn_berm` is additive. |
| **Plus an explicit keyed-dam path** | **Selected** | `_keyed_dam_dem` bypasses `burn_earthworks` entirely, and Dam 15 is keyed. Without it the change is invisible on the case the spec is about. |
| **Crest referenced to the swale floor, containment reported as clearance** | **Selected** | The design invert is single-valued; the rim is a global ring minimum. Removes the berm-vs-lip question from the datum without crediting a berm estimate. |
| **Round the burned width to whole DEM cells** | **Selected** | Width cannot change a total here, so rounding is about rasterisability. Cells rather than literal metres keeps it honest on finer native data. |
| **A live "this sill gives up N m³ (X%)" readout, from a stage–storage lookup** | **Selected** | The decision the user is actually making, and nothing answers it today. Extra return from a flood already happening — no second pass. Also delivers §6's "% full never reaches 100%" as a readout. |
| Deduct from `capacity_m3`; leave the DEM alone | Rejected | Fixes the panel and nothing else — map, streams, exit volumes and overtopping keep drawing water over the wall. Round 12's complaint verbatim, and Rounds 3 and 8 record what happens when a model stands beside a burn. |
| Do STRETCH_GOALS §6's capacity/threshold split first | Rejected | Once the notch is cut, capacity and threshold coincide correctly. The lip-volume denominator is a cheap follow-on, not a gate. |
| Make the crest the level-invert datum (§6's original framing) | Rejected | Couples the burn to the spillway, leaves the level floor unusable until one exists, and moves the invert whenever head changes. |
| Feed the weir equation into the routing | Rejected | There is no rate dimension to attach it to. Round 14 measured that one pointer per cell and a weir are incompatible. That is STRETCH_GOALS §3's HEC-RAS export. |
| Refine the DEM locally so a 0.4 m notch resolves (Strategy B) | Rejected | Interpolation invents elevation, ~16× cost against `_MAX_PONDING_CELLS`, no mixed-resolution routing in pysheds — and it would buy the one thing that does not matter. |
| Reuse `perpendicular_sill` as the burn geometry | Rejected | It runs across the alignment, i.e. along the flow. Fine as a map symbol; wrong as a cut. |
| Cut the notch inside each `_burn_*` | Rejected | A later fill plugs it, and `_burn_swale` raises its own berm before cutting. |
| A separate `spillway_diversion` earthwork type | Rejected | Identical geometry, burn, sizing, style and policy; a whole registry entry for one datum. |
| Overwrite the drain's `depth` from the spillway | Rejected | `depth` is a section property feeding cut volume, Manning discharge, taper and the sub-cell check. |

**Why this is the right shape.** The burn is the only lever that reaches the rasters, and the
codebase has already built nearly everything a notch needs — a level invert, a measured pond, a
mask-record convention, a downhill-side test, an advisory channel. Most of the value arrives
through machinery that needs no modification at all, because `find_impoundments` was deliberately
built to know nothing about earthworks. What remains is a datum that can be stated honestly, a
geometry that points the right way, one path (the keyed dam) that has to be wired by hand, and
guards that say when the notch did not do what it claimed.

---

## Verification

- **Stage A** — `python -m pytest tests/test_spillway.py tests/test_earthwork_design.py`. New
  cases: the three-way binding round-trips without drift; a bermed swale's ceiling uses the berm
  crest and a crest above the lip is a *note*, not a fail; a falling swale takes its rim locally;
  a map-placed crest is clamped; a saved design migrates `drop_below_rim_m` without moving
  `crest_elevation`; the stage–storage lookup integrates to the same figure `feature_storage`
  already reports at the spill level, and answers a level below it with the volume that level
  holds. Then `.\run_qgis_tests.ps1 spillway`, plus a check that the give-up figure tracks the
  crest control and degrades to "run Re-analyse" with no burn, and a `checks_earthworks` case
  that unticking the spillway group prompts and keeps the sited spillway when declined. The gate
  is the **Spillways review** — figures and the `checks_visual` screenshot — not the rasters.
- **Stage B0** — the headless Quail Island burn, notches off/on, per the table in B0.
- **Stage B** — `python -m pytest tests/` (95% gate on `modules/`). New cases in
  `test_burn_strategy.py` (cuts to the crest and no further; a notch that cannot daylight cuts
  nothing and warns; a notch that daylights back inside its own pool is refused; a crest below
  the burned floor is refused; an inflow spillway is never cut)
  and `test_dem_burner.py` (the notch survives a later feature's berm; `burned_notches` survives
  the isolated burns; a keyed dam's `dam_stage_storage` sees the notch; the as-burned sill equals
  the designed sill on a clean cut and exceeds it on a footprint too narrow to reach depth; a
  non-daylighting notch leaves actual spill level above the as-burned sill — B6's three
  elevations, each pinned by the disagreement it is meant to expose). In
  `test_project_io.py`: an auto width is not written to the design file, and a restored project
  recomputes it before anything renders. Rewrite `test_plan_geometry.py`'s perpendicularity
  assertions. Then `.\run_qgis_tests.ps1` in full (~8 min, background it), fix
  `checks_symbology`'s sill-orientation assertion, add a `checks_earthworks` case that the
  `Spillways (burned)` layer lands under **Verify** via `_groups` rather than loose at the top of
  the legend, and open the changed screenshots.
- **Stage C** — `tests/test_project_io.py` for link round-trip and dangling tolerance;
  `tests/test_dem_burner.py` for a linked drain grading from the crest and for source-before-drain
  burn order; a cycle refusal test.

**End to end, in real QGIS:** `.\deploy.ps1`, re-enable the plugin, open the Quail Island design,
place a spillway on Dam 15, then Re-analyse with Earthworks. Expect: the pond truncates at the
crest, Surface Runoff leaves through the notch rather than over the wall, Dam 15 drops out of
*Overtopping (full)*, and its capacity falls (which only happens if B3's keyed-dam path landed).
Then `.\run_qgis_gui_shot.ps1` to judge type rendering.

**What moves in Verify, per type** — worth knowing before the numbers land. Δ compares Measured
against `rasterisable` = `terrain_capacity_m3`, both floods, so for **swale, bermed swale and
basin** both sides fall together and Δ stays near zero, while Geometric (an analytic prism) is
unchanged — so the *Geometric → At-grid* gap widens, and `impoundment_m3` can go negative on a
bermed swale. For a **dam**, At-grid renders `—` and the reference is `dam_stage_storage`; if B3's
keyed path were skipped, Δ would swing by the whole notch.

**Docs to update in the same change:** `CLudeDocs/FIELD_TEST_LOG.md` (a new round, with the B0
table), `STRETCH_GOALS.md` §5a (done), §6 (reduced to a presentation follow-on), §10(a)/(b)
(closed), and the `CLAUDE.md` hydrology-chain bullet on overtopping.

**Found while exploring, not fixed here** — each moves published numbers and deserves its own
change: `_burn_berm` and `_burn_diversion` still rasterise `all_touched=True`, which the
`_rasterize` docstring measures at a near-constant +1.3 m wider than drawn, while swales, basins
and dams moved to cell centres in Round 8. And `burn_strategy.py:15`, `:161` and `:300` point at
`DEMBurner.burned_storage` and `DEMBurner._datum_surface`, neither of which exists.
