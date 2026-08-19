# TerrainFlow — Stretch Goals (deferred backlog)

Running markdown companion to `TerrainFlow_Next_Steps.docx` (the `.docx` is
Word/binary and not diff-friendly). Deferred items land here so they stay tracked
and reviewable instead of being scattered across code comments and commit messages.

Each entry records **what** was deferred, **why**, and the **revisit trigger** —
the concrete condition that should pull it back into scope.

---

## 1. Strategy B — interpolative whole-DEM refinement

**What.** Resample a coarse DEM to a finer grid (e.g. 16×) so sub-cell earthworks
become directly resolvable on the raster, instead of representing them by their
hydraulic *effect* on the native grid (Strategy C, which shipped).

**Why deferred.**
- **No terrain information gained.** Interpolating a coarse DEM invents elevations
  between real samples; it does not add real sub-cell relief. The apparent detail is
  smoothing, not data.
- **~16× cost.** A 4× linear refinement is 16× the cells — memory and runtime — and
  trips the `_MAX_PONDING_CELLS` guard, forcing a downsample that undoes the refine.
- **Obviated by the intended high-res path.** The planned source of genuine sub-cell
  elevation is **user-captured drone photogrammetry** (real cm-scale native data),
  which resolves most earthworks directly. That is the correct way to get "finer" —
  finer *native* data, not interpolation.
- **pysheds has no mixed-resolution routing**, so a cheaper local-patch refine (fine
  grid only under the earthwork) is not viable either.

Strategy C already keeps one consistent grid, preserves before/after integrity, and
matches how `simulation.py` reads storage (analytic `capacity_m3` cascade, not raster
walls), so the analytical sizing engine is resolution-independent regardless.

**Reaffirmed 2026-07-28** during the Design-tab correctness work. The resolution
penalty is no longer merely asserted — it is **measured** per feature. Capacity is
reported as four numbers (`earthwork_design.capacity_breakdown`): design (with
freeboard), geometric (the drawn shape), **rasterisable** (what this cell size can
represent), and terrain (what the burn produced), with verification Δ comparing terrain
against rasterisable so it isolates burn error alone.

**Corrected 2026-08-12 (Round 8).** The 2026-07-28 paragraph above went on to claim that
the geometric→rasterisable gap *was* the resolution penalty, and offered "a 2.0 m swale
with 1:1 batters on a 1 m grid reports +33%, because it cannot hold its sloping walls."
**That example was measuring a burn choice, not a cell size.** `batter_run_m` was a
basin-only field, so `level_invert` squared every drawn channel off to a full-depth
rectangle whatever cross-section it carried — at any resolution. On the Quail Island
design that put At-grid at **2.10×** the drawn section (Swale 22: 628 m³ against a
299.6 m³ trapezoid), of which none would have been recovered by a finer DEM.

The burn now cuts the section it was given (`burn_strategy.tapered_invert`, a distance
transform rather than the old nested-erosion staircase) and At-grid is measured off that
cut rather than modelled beside it. Sitewide, At-grid now tracks the drawn trench to
**+2.1%**. What remains *is* a genuine cell-size limit, and it is the honest version of
what this section always meant: a footprint two cells across has no cell more than half a
cell from its own edge, so it cannot reach the depth it was drawn at, and it comes out
shallower. That residual is per-feature, flagged, and small.

**None of which changes the conclusion.** Interpolating still invents elevation, the
cost is still ~16× against `_MAX_PONDING_CELLS`, pysheds still has no mixed-resolution
routing, and the route to finer is still finer *native* data. The correction is to the
reasoning, not the decision — and it removes the one argument in this section that a
reader could have acted on by going looking for a finer DEM.

**Revisit trigger.** A concrete need appears that drone-photogrammetry native
resolution cannot meet — e.g. a user with only a coarse regional DEM and a hard
requirement to resolve sub-cell features on the raster itself, where the effective-
parameter representation (Strategy C) proves insufficient for their decision.

---

## 2. Capacity-first / inverse sizing mode

**What.** Solve a feature's free dimension *from* a design storm rather than the
forward direction (dims → capacity). Given a target design-storm runoff volume (via
the existing SCS-CN model) and a stage-storage curve, back out the required depth /
length / footprint so the feature just contains the storm.

**Why deferred.** Sequenced *after* the shared sizing engine lands — the inverse
solver reuses the same primitives (`trapezoid_section`, `pond_volume_frustum`,
`prismatic_volume`) plus a stage-storage curve, so it is cheapest to build once the
forward engine is the single source of truth. Building it first would duplicate the
maths that the engine now centralises.

**Revisit trigger.** The forward engine (this build) is in use and users ask to size
to a storm target directly. Implementation: add an inverse mode that iterates /
inverts the forward primitives against the SCS-CN design-storm volume and the
stage-storage curve; surface the solved dimension in the properties dialog.

---

## 3. Step-3 export to professional 1D/2D hydraulic software

**What.** Export the design for high-fidelity modelling in free professional tools,
recovering the true sub-metre fidelity that Strategy C approximates at 1-cell width.

**Export payload (decided).** The **bare-earth (unburned) DEM + the earthwork
vectors carrying their sizing attributes** — `T`/`b`/`d`, batter, grade, crest,
`H_eff`, soil — **not** the burned raster. The burned raster would inherit Strategy
C's 1-cell approximation and cannot be combined with the vectors without double-
counting the earthwork. Step 1's structured per-feature sizing output (the
`min_dimension`-bearing results from `core/sizing`) *is* this export payload.

**Target software.**
- **HEC-RAS** (default) or **Iber** (GUI) for 1D/2D hydraulics.
- **EPA SWMM** for drain/storage networks.

**Formats.** GeoTIFF (DEM) + GeoPackage / shapefile (vectors with sizing attributes).

**Why deferred.** Depends on the shared engine's structured output existing first
(now it does) and is a later phase than the burn/report stages. Keeps the two/three-
step verification method non-circular: step 3 recovers fidelity that steps 1–2
intentionally approximate.

**Revisit trigger.** Burn stage (Strategy C) and non-circular reporting are landed
and users need to hand a design to a pro modeller.

---

## 4. Further earthwork feature types (fold into Section 4)

Named-but-unbuilt features the user intends to fold into the Earthwork Design
sphere once the current UI settles. The registry redesign (2026-07) made the
panel/dialog/map-layers fully registry-driven, so each of these is now mostly a
`register_type()` + module wiring job rather than a UI rebuild.

**a) Terrace.** The registry docstring's canonical `register_type()` example; the
sizing engine's `contour_spacing` primitive (terrace HI = VI / slope) is built and
tested with zero callers. Needs: registered type, burn method (berm-like bench
cut/fill), capacity/cut maths, and a spacing-advisor UI using `contour_spacing`.

**b) Pond-site → dam creation shortcut.** `recommend_pond_sites`
(keypoint_analysis) already finds optimal impoundment sites and renders markers —
but there is no "start a dam here" action. Needs: click a recommended marker →
seed a dam line across the valley at that point → normal dam dialog/stage-storage
flow. (`pond_volume_frustum` also exists for a stand-alone excavated-pond type if
wanted later.)

**c) Keyline → swale conversion.** ✅ DONE (2026-07). `YeomansKeylineAnalysis` is now
wired into the Analysis section: "Generate Keylines" finds the keypoint and draws the
on-contour keyline + parallel cultivation guides (drift emerges from the parallel
offsets, per Yeomans — `get_cultivation_runs` rebuilt from straight-line to
traced-contour + `offset_curve`). "Draw Keyline" lets a user input a plough guide by
hand (live slope read-out), and "Convert Keyline → Swale" routes the master keyline
through the contour-swale path (`_on_contour_selected_for_swale` → `source_contour_coords`)
so the swale stays reshape-locked. NOTE: keyline was deliberately kept as a cultivation-
guide **layer**, not registered as a hydraulic earthwork type — it has no storage/burn
semantics, so forcing it through the earthwork properties/capacity/burn pipeline would
be wrong and risky. A future registry entry could add it purely for styling metadata.

**d) Sketch/draft design layer.** Rough drag-to-place earthwork placement explored
freely, then committed/refined into real contour-fitted features (user idea,
2026-07-23 — explicitly not now).

**Why deferred.** UI redesign phase capped its scope at exposing already-modelled
abilities; these add new feature semantics.

**Revisit trigger.** The user finishes polishing the current Section-4 UI and asks
to fold further earthworks in (their stated plan).

---

---

## Reference: runoff basis and infiltration policy (2026-07-29)

Not a deferral — a decision worth recording, because it changes every feature size.

**The problem.** Sizing ran off the SCS-CN surface-runoff depth. At the defaults
(CN 61, normal antecedent moisture) a 120 mm storm yielded 31 mm — 26%. The same
ground already wet yields 64 mm (53%); at CN 80 wet, 92 mm (77%). A 3–4× spread on
inputs nobody verifies in the field, with the error running toward **under-sizing**,
where the failure mode is a breached earthwork rather than wasted excavation.

**What we compared against.** Brad Lancaster, *Rainwater Harvesting for Drylands and
Beyond* — the standard water-harvesting reference — uses the rational method:

    runoff volume = catchment area × rainfall depth × runoff coefficient

with one empirical coefficient per surface (grass/lawn 0.05–0.35, typical 0.10–0.25;
bare earth 0.20–0.75, typical 0.35–0.55; healthy indigenous landscape 0.20–0.70,
typical 0.30–0.50; concrete/asphalt 0.80–0.95; metal roof 0.95).
<https://www.harvestingrainwater.com/resource/water-harvesting-calculations/>

Notably, **our SCS figure of 26% already sits above Lancaster's typical grass range**,
so the original model was not absurd for average conditions — the case for
conservatism rests on the spread and on design storms arriving on wet ground, not on
the central estimate being wrong. Equally, sizing on **total rainfall (100%) exceeds
even his metal-roof coefficient** and is not something any established method does for
a landscape catchment.

**What shipped.** Three explicit bases on the Baseline stage, governing the analysis
rasters and earthwork sizing alike:

| Basis | Depth | Notes |
|---|---|---|
| **Runoff coefficient (Lancaster)** — default | `P × C`, C = 0.50 | Lancaster's table as presets, plus a "pasture, wet ground" design default. Lands within ~15% of SCS-CN at wet antecedent moisture. |
| Total rainfall | `P` | Most conservative; above any published coefficient. |
| Surface runoff (SCS-CN) | `Q(P, CN, AMC)` | Physically the right answer once CN and AMC are known for the site. |

**Infiltration.** Credited to capture only when explicitly enabled (default off); the
soakage is still measured and reported as a buffer. Lancaster sizes off an on-site
**percolation test**, whereas we derive a rate from a soil-texture lookup — and a
percolation rate can overestimate true infiltration by a factor of ten. Excluding it
from capture by default is the honest response to a term we cannot verify.

**Drain-down time** is now reported per feature (`catchment.drain_down_hours`), against
Lancaster's third design constraint — earthworks should "work, don't flood, **and don't
puddle**" — flagged above the conventional 24 h / 48 h drawdown thresholds.

**Still not modelled** (candidates, not commitments): a measured-percolation input in
place of the texture lookup; sizing storage on a design event while sizing overflow on
a separate extreme event, as Lancaster does.

---

## 5. Level-pool swale segmentation (per-run inverts + burned sills)

**What.** Split a long contour swale into runs that each fall no more than its design
depth, give each run its own level invert, and burn a **sill** (an un-excavated
cross-bank) at each boundary. Each run then fills and spills over its sill into the
next run down — how contour swales and keyline systems are actually built, with
check-banks.

**Why deferred.** The 2026-07-28 burn rework gave swales a single **global** level
invert referenced to their natural pour point, which is the step change that matters:
burned swale storage went from ≈ 0 (the old `enforce_monotonic_path` breach guaranteed
an outlet) to the correct order of magnitude. Segmentation refines **cost**, not
correctness — a single invert over-excavates the high end when the alignment falls
further than the depth, which is now flagged by `burn_strategy.steep_ground_warning`
with the real cut and storage volumes.

It needs three things this build does not otherwise have: burning the sills into the
DEM, a UI for sill count/spacing, and a capacity model that sums per-run volumes
rather than `section × total length`. The data for auto-placement already exists —
`swale_design.inflow_profile` returns sill-candidate stations at the inflow peaks,
which is where check-banks belong.

**Revisit trigger.** A user objects to over-excavation at one end of a long contour
swale, or the steep-ground warning starts firing routinely on real designs.

---

## 5a. Cut the spillway into the terrain — DONE (2026-08-19)

**The approved plan is [SPILLWAY_NOTCH_PLAN.md](SPILLWAY_NOTCH_PLAN.md), and it supersedes
this section.** Read it rather than the paragraphs below, which are kept only as the record
of what was known before it was written. Three stages, each shippable: **A** the datum
(reference the crest to the swale floor, containment as a reported clearance, the live
give-up readout, the migration), **B** the notch itself (the cut, the keyed-dam path, width
rounding, the three elevations, overtopping), **C** the spillway-linked diversion drain.

**All three stages landed 2026-08-19.** A and B are Round 20 of `FIELD_TEST_LOG.md`,
which carries the measured before/after and settles two things this section got wrong;
Stage C — the spillway-linked diversion drain — is Round 21. Nothing here is open.

**The 1,832 m³ below is right, and it is the pool, not the dam's storage.** Measured: the
pool behind Dam 15 truncates from 2,688.4 m³ at 56.12 m to **1,832.1 m³ at 55.52 m** once
the notch is cut. Its `capacity_m3` — the pond the dam *adds*, natural ponding subtracted —
goes 2,132.8 → 1,276.6 m³. The two were being compared as if they were one figure.

**One decision in the plan changed under Stage B.** The per-feature isolated floods stay
**brim-full**; the volume held to the sill is read off the stage–storage curve at the crest
(measured equal to a notched flood on all five sited spillways). Cutting the notch into
that flood collapses the containment datum onto the crest, which ratchets the crest down by
`head + freeboard` on every re-open, fails `spillway_validity` on every spillwayed feature,
and zeroes the give-up readout. The **site** burn still cuts the notch. See Round 20.

The plan also reduces §6 below to a presentation follow-on rather than a prerequisite — see
its B7, **delivered in Stage B** — and closes §10(a)/(b)/(c)/(d) along the way.

**What.** The burn is spillway-blind: no `_burn_*` method reads a spillway and nothing cuts
a notch. A sized, sited spillway therefore changes no raster, no routing and no pond, and
reported capacity is measured to the crest. Dam 15 reads **2,688 m³ to the crest against
1,832 m³ to a sill at 55.52 — about 30% overstated**. Every overtopping advisory currently
has to carry a caveat saying the model cannot see the structure the user designed.

**Why it is not a small change.** Cutting the notch moves the burned DEM, and with it pond
volumes, capacity, verification Δ, the ponding and event-pond layers, and the report. It
also collides with §6: `simulation.py:308` uses `capacity_m3` for both *how much it holds*
and *when it spills*, so a crest cannot drive an overflow threshold until those are split.

**Do first (done).** This section said: a measured before/after plan over the Quail
Island design, and the capacity/threshold split from §6 — do **not** attempt the notch
directly. The plan kept the measured before/after (its B0, the gate on Stage B, now Round
20) and dropped the §6 split as a prerequisite: once the notch is cut, capacity and
threshold coincide correctly, so the lip-volume denominator was a cheap follow-on rather
than a gate. It shipped with Stage B as `EarthworkStore.lip_capacity_m3`.

**Stage C, for the record.** A diversion drain can now take its start level from another
feature's spillway instead of from the ground under its own first vertex, which was a
guess about where the water it carries arrives. `Earthwork.spillway_link_id` stores it as
`"<source id>:<kind>:<end>"` and is serialised (a decision, recoverable from nothing —
`SCHEMA_VERSION` 2 → 3); `invert_start_m` is the level it resolves to and is derived,
never serialised. The plan left one question open and Round 21 answers it: the **end** of
the drain travels with the link rather than the alignment being reversed on attachment.
No raster behaviour and no new measurement — the burn change is one datum in
`_burn_diversion` — so the risk was all in the link's lifecycle: dangling on delete,
cycles, and which end grades.

---

## 6. Spillway crest as the level-bottom datum — the "% full" half is DONE (2026-08-19)

**The "% full should never reach 100%" goal shipped with Stage B of §5a**, and it shipped
as a readout off the stage–storage curve rather than as the `simulation.py` redesign this
section proposed. `EarthworkStore.lip_capacity_m3` carries the brim volume, `fill_pct()`
divides by it, and `capacity_m3` — which `cascade_overflow` still thresholds on — is
correctly the volume to the sill once the notch is cut. So the two-change list at the
bottom of this section is answered: (1) is `FeatureStorage.stage_storage`, and (2) turned
out not to need a separate threshold field at all.

**What is left here is only the datum question**, which is what the section is titled for
and which Stage B did not touch.

**What.** Let a user-placed spillway's crest elevation set the invert datum for its
feature, instead of the automatically-detected pour point.

**Why deferred.** The invert datum is currently the natural pour point — the lowest
rim cell — which needs no user input and makes analytic and burned volumes agree
immediately. Making the crest the datum couples the burn to the spillway feature and
leaves the level floor unusable until a spillway exists. `Spillway.auto` is carried on
the model specifically so this becomes a small follow-on rather than a rework.

**Revisit trigger.** Users place spillways routinely and want the floor referenced to
the crest they chose rather than to the natural rim.

**Status (2026-07-30).** Block C shipped, so the precondition now exists: `Spillway`
carries `crest_elevation`, `drop_below_rim_m`, `head_m`, `width_m`, `point_wkt` and
`auto`, all persisted with the design. The burn still takes its datum from
`footprint.pour_point`, unchanged. Two things surfaced while building it that this
follow-on will have to answer:

- The crest ceiling is `rim − head − freeboard`, not the rim, so a crest-derived
  invert would move whenever the design head changes. The burn would need to pin the
  head at burn time rather than read it live.
- For a dam the containing rim is the wall crest, not the natural pour point
  (`_spillway_datums` already makes that distinction). A crest-as-datum rule has to
  respect it or dams will burn against the valley floor they are impounding.

**Status (2026-08-13).** A second, larger use for the crest arrived with Round 11, and the
machinery it needs is now built. Storage is measured by flooding each feature alone
(`DEMBurner.feature_storage`), which returns the pond's **level and region** — so
"the volume at any elevation below the spill level" is one integration over arrays already
in hand. The stated goal is that **"% full" should never reach 100%**: a spillway lets
water go before the lip, so the working threshold is the crest volume while the lip volume
stays the true 100%.

That makes this two changes, not one:

1. `feature_storage` returns a stage–storage lookup instead of a single figure, and the
   capacity a store is built with becomes the volume at `min(spillway crest, spill level)`.
2. `EarthworkStore` needs the overflow threshold **separate** from the capacity —
   `simulation.py:308` currently uses `capacity_m3` for both "how much it holds" and "when
   it spills", which is exactly the conflation that stops a feature reading 85% and
   spilling. Deliberately not added speculatively in Round 11; it is a five-line change
   once a crest actually feeds something.

The datum question above is unaffected — this is about the *water level*, not the floor.

---

## 7. Migrating the time-stepped simulation to topological cascade order

**What.** Have `simulation._run_simulation` process stores in flow-path topological
order, as the design-tier balance now does, instead of sorting by centroid elevation.

**Why deferred.** `cascade_overflow(routing=None)` deliberately preserves the legacy
elevation sort, so the Verify-stage simulation is byte-for-byte unaffected by the
Design-tab rework. Migrating it changes simulation output and deserves its own
before/after comparison rather than riding along with an unrelated change.

**Revisit trigger.** The live assessment and the simulation visibly disagree about
where water goes on a real design.

---

## 8. Full node-and-arrow network diagram on its own tab

**What.** A dedicated canvas for the earthwork flow network, with room for a proper
node-and-arrow diagram rather than an indented tree in the dock panel.

**Why deferred.** The chosen design is a List / Flow toggle inside the existing Live
Assessment section, which needs no new panel real estate. The layout maths a larger
canvas would consume (rank and order per node) is the reusable part; a bigger view is
a rendering job on top of it.

**Revisit trigger.** The flow chart outgrows the dock panel's width — realistically
past ~15 features.

---

## 9. Regional CN selector — and the per-cell summing it requires

**What.** Let the user paint curve-number zones over the site (soil group / land cover
varying across a farm) instead of one site-wide CN. Planned feature, not deferred
indefinitely.

**Why it is not a switch.** The plumbing exists — `AnalysisWorker` accepts
`cn_zones_data` and `SCSRunoff.build_cn_raster` builds the raster — but nothing in
`panel.py` or the controllers ever passes it, so CN is uniform today. Turning it on
without the work below would silently corrupt per-feature inflow.

**The blocker.** `EarthworksController.feature_inflow_m3` multiplies a **single
site-wide depth** by the feature's catchment cell count:

```python
counts[ew.id] * meta["cell_area_m2"] * runoff_mm / 1000.0
```

Exactly right while runoff is spatially uniform. With CN zones it is not: a feature
whose catchment sits mostly on high-CN ground would be charged the site average and
under-read, while one on free-draining ground would over-read.

**What has to change — sum per cell, not multiply by a scalar.**

1. Have the worker emit the per-cell runoff-depth raster it already computes
   internally (it currently derives the weighted array and discards it).
2. Replace the scalar multiply with a masked sum over the cached label raster —
   `runoff_depth[labels == i].sum() * cell_area / 1000` — giving each feature the
   runoff its *own* catchment actually generates. The label raster is already in
   memory, so this needs no re-run and no extra pass over the DEM; it is a
   `np.bincount` weighted by runoff depth in place of a multiply.
3. `_current_runoff_mm()` keeps returning a site-wide figure for headline text only,
   and must stop being the per-feature source.

**Consequence for the runoff-basis comparison.** Under uniform runoff the three
bases are exact scalar rescales of one another, which is what lets the spillway
intensity chooser show all three from a single baseline run. With CN zones the SCS-CN
basis produces a different spatial *pattern*, not merely a different magnitude, so
that inter-convertibility ends: the chooser's non-current rows would need their own
weighted sums, and the SCS peak-runoff fraction becomes a per-feature area-weighted
mean rather than one site-wide number.

**Revisit trigger.** Committed as a planned feature (user, 2026-07-30). Do step 1–2
before exposing any zone-painting UI, or the per-feature numbers go quietly wrong.

---

## 10. Spillway follow-ons deferred from the review pass (2026-08-06)

Four things surfaced while answering the spillway review questions and building the
Design-stage Spillways list. Each is real; none belonged in that pass.

**Status (2026-08-19): all four are closed.** (a), (b) and (c) by Stage A of
[SPILLWAY_NOTCH_PLAN.md](SPILLWAY_NOTCH_PLAN.md); (d) by its Stage B, which is where the
width rounding it is bound up with landed. The original text of all four is kept below,
because each records the reasoning that made it a deferral rather than a bug, and (a) in
particular asked a question the fix had to answer rather than dodge.

**(a) Sample the rim from the burned DEM.** `_spillway_datums` takes `pour_point` on the
*pre-earthwork* conditioned DEM, so a swale's companion berm is invisible to it — while
`_burn_swale` raises the berm *before* taking its own pour point, and `calculate_capacity`
credits it as extra section. The rim is therefore the odd one out, and understates the
containing level for a bermed swale.

Deliberately not "add the berm height to the rim": the berm is built on the downhill side
only, so where the natural low point is at an *end* it raises nothing, and crediting it
would claim headroom the ground may not have — the unsafe direction. There are also four
different berm-height derivations in the codebase (declared section, excavated volume ÷
berm cells, the capacity credit, and the dialog's call without `bottom_width`), so any
estimate-based fix would put the crest band on a fifth basis. The honest fix is to sample
the rim from `state.modified_dem_path` once a burn exists, which needs a story for what
the rim means before the first burn.

*Revisit trigger:* users report crest bands that feel too tight on bermed swales, or the
burned/analytic freeboard figures are seen to disagree.

**Closed 2026-08-19.** `_spillway_datums` now returns *lip*, *invert*, *containment* and
which of the three the containment came from. The story the note asked for — what the rim
means before the first burn — is: **before any measurement, the analytic level, said so;
after one, the measured level.** In preference order the containment is
`ew.terrain_spill_level_m` (the pond's own measured spill level, retained off
`FeatureStorage.level_m` and never serialised), then `ew.berm_crest_elevation` (the bank
as built), then the wall crest for a dam, then the lip. Every one of those is a
measurement or a stated design value, so the note's own ruling — *not* a berm height
estimate, of which there are four incompatible derivations — is respected. The lip is
kept and reported beside it, and a crest standing above it is a **note** rather than a
problem, because `_spillway_row` fails a row on any problem at all.

**(b) Clamp a map-placed crest into its band.** `_on_spillway_placed` calls `bind_crest`
without `band=`, so unlike the dialog path a clicked point is not held inside
`rim − head − freeboard`. Not a safe one-liner: on a **dam** the rim is
`ew.crest_elevation` (the wall) while the clicked elevation is sampled from ground under
the wall, so clamping could move a crest by metres. The placement tool also samples the
**raw** DEM (`state.dem_path`) while the rim comes from the **conditioned** one, so the
two are not on the same surface to begin with. Fix both together or neither.

**Closed 2026-08-19, both together as the note required.** `_spillway_datums` moved off
`state.flow_dem` onto the burner's `original` — the raw file `PlacePointTool` was already
sampling — so the click and the band are on one surface, and `_on_spillway_placed` now
binds through `_crest_band_for`, which resolves the same band the dialog does. A clamped
click says so in the message bar rather than moving the crest silently. The dam concern
stands and is handled by the containment rule above: a dam's containment is its wall
crest, which is what the band is computed against.

**(c) Unticking the spillway group discards a placed location.**
`ew.spillway = dlg.get_spillway()` runs unconditionally, and `get_spillway()` returns
`None` when the group is unticked — taking `point_wkt` with it. Pre-existing and
previously invisible; the Spillways list now makes a sited row blank out, so it will get
reported as new. Wants a confirm-before-clear, or to preserve the location separately.

**Closed 2026-08-19 as a confirm-before-clear.** `_apply_spillway_from_dialog` asks before
assigning `None` over a **sited** spillway, and the question names the crest and the width
so it can be answered without reopening anything. An unsited one is still cleared without
asking: it carries nothing the user cannot retype, and a confirmation on an ordinary edit
is a confirmation nobody reads. Landed in Stage A deliberately, ahead of the notch — the
protection wanted to be in place *before* the untick gained the power to un-cut a hole in
a dam.

**(d) An auto width is derived state and probably should not be persisted.** It now
genuinely tracks (`_refresh_auto_spillway_widths`), but it is written during
`_recompute_live_assessment`, which restore also runs — so opening an old project rewrites
every auto width without `_mark_design_edit`, and the next save stores numbers the user
never chose. Self-healing, but silent. Not persisting it at all would dissolve that, at
the cost of a schema change and a new source for the map label's width.

*Revisit trigger:* any report of a design file changing on open, or (d) alongside the next
`Spillway` schema change.

**Closed 2026-08-19 (Stage B).** `Spillway.to_dict` drops `width_m` whenever
`width_auto` is set, so the flag travels and the number is derived — which is what the
flag always claimed. Its own restore path recomputes it (`_refresh_auto_spillway_widths`,
now run before the feature list, the map label and the sill bar are drawn, all three of
which read `width_m`), and `_spillway_row` derives it rather than trusting the stored
figure, so a restored design cannot print a 0.0 in the window between the two.

No schema bump was needed in either direction. Reading a document an older build wrote,
the stored auto width is loaded and then immediately recomputed; reading one this build
wrote, an older build starts from `width_m = 0.0` with `width_auto` set and its own
`_refresh_auto_spillway_widths` fills it in. Nothing a user chose is lost either way,
which is the whole distinction: a **committed** width (`width_auto` False) is a decision
and is still stored, rounded.

What made this worth closing now rather than leaving self-healing: the rewrite on open was
silent while the figure was unrounded, and rounding to whole DEM cells would have made it a
*visible* unexplained change to a saved design.

---

_Last updated alongside the Design-tab correctness rework (2026-07-28): direct-catchment
water balance, level-bottom burn, verification split into design / geometric /
rasterisable / measured. Spillways and peak-flow sizing added 2026-07-30; per-type
spillway policy, the two-mode head/width model and the Design-stage Spillways review
added 2026-08-06. Stage A of the spillway-notch work landed 2026-08-19: the crest is bound
three ways, the ceiling is the level water is held to rather than the lowest bare ground,
the lip is taken locally under the sill, and every feature carries a measured
stage–storage curve. **Stage B landed the same day**: the crest bar runs along the
alignment and is cut into the burned DEM as a post-pass, the keyed-dam path cuts it too,
the burned width is whole DEM cells, an auto width is no longer persisted, "% full" is
measured against the brim rather than the sill, and the overtopping check subtracts the
notch from the barrier crest — so the caveat about the model not seeing the designed
spillway is deleted rather than reworded. See
[SPILLWAY_NOTCH_PLAN.md](SPILLWAY_NOTCH_PLAN.md) and Round 20 of
[FIELD_TEST_LOG.md](FIELD_TEST_LOG.md)._
