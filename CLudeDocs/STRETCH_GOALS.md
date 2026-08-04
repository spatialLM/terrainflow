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
penalty is no longer merely asserted — it is **measured** per feature. Capacity is now
reported as four numbers (`earthwork_design.capacity_breakdown`): design (with
freeboard), geometric (the drawn shape), **rasterisable** (what this cell size can
represent), and terrain (what the burn produced). The gap between geometric and
rasterisable *is* the resolution penalty, shown per feature in the Verify stage, and
verification Δ now compares terrain against rasterisable so it isolates burn error
alone. A 2.0 m swale with 1:1 batters on a 1 m grid, for instance, is reported as
+33% — it cannot hold its sloping walls, so it burns as a rectangular trench.

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

## 6. Spillway crest as the level-bottom datum

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

_Last updated alongside the Design-tab correctness rework (2026-07-28): direct-catchment
water balance, level-bottom burn, verification split into design / geometric /
rasterisable / measured. Spillways and peak-flow sizing added 2026-07-30._
