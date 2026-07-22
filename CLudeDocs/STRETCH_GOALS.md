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

_Last updated alongside the shared sizing engine build (build-sequence steps 1–4)._
