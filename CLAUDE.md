# TerrainFlow — project guide

QGIS plugin for NZ land managers / permaculturists: drainage analysis + earthwork planning
(swales, dams, basins, diversion drains) with before/after flow comparison.

## Where the code lives

- **`terrainflow_assessment/` is the active codebase.** All new work goes here.
- **The retired original was archived outside the repo on 2026-09-12** (`F:\Terrain Flow Design\_archive\plugin_legacy\`). It was a dangling gitlink with no `.gitmodules`, so a fresh clone got an empty `plugin/` directory; nothing in the package, either suite, `deploy.ps1`, `deploy.sh` or CI ever read it.
- Never create a nested `terrainflow_assessment/terrainflow_assessment/` folder (packaging accident — guarded in `.gitignore`).

## Architecture — a layered split (respect it)

The plugin was deliberately refactored out of a 2,159-line god-class into focused layers so
edits stay small and low-risk. Flow is top-down: QGIS entry point → thin dispatcher →
controllers (QGIS glue) → modules (pure logic).

| Layer | Path | Holds |
|---|---|---|
| Entry point | `__init__.py` → `qgis/plugin.py` (`TerrainFlowAssessmentPlugin`) | Boots plugin, wires panel signals → controllers. No logic. |
| UI | `panel.py`, `earthwork_properties_dialog.py`, `design_intensity_dialog.py`, `rainfall_data_dialog.py` | All buttons/inputs/tooltips/tables + the signals they emit |
| Widgets | `qgis/widgets/` (`network_view`, `verification_table`, `scorecard`, `stepper`, `tool_menu`, `run_button`) | Custom-painted panel components |
| Controllers | `qgis/controllers/` (`baseline`, `contour`, `earthworks`, `simulation`, `reporting`, `_state`, `_layers`, `_groups`) | Per-feature QGIS glue: run analysis, render layers, styling |
| Workers | `qgis/workers/` | Background threads. `analysis_worker` / `simulation_worker` are bespoke; `task_worker.TaskWorker` runs any single callable off the GUI thread (contours, segments, keypoints, keyline, the burn, terrain capacities) |
| Adapters | `qgis/adapters/` (`project`, `geom`, `map_image`, `layout_pdf`) | Thin `QgsProject`/geometry wrappers, plus offscreen map rendering and the PDF print-layout builder |
| Modules | `modules/` | **Pure analysis logic, no QGIS UI** — the testable core |
| Map tools | `map_tools/` | Interactive canvas clicking (draw, select, query, connect) |
| Core | `core/registry/`, `core/sizing/` | Pure config (earthwork types) + geometry/hydraulics primitives |

Tooltip copy is centralised in `qgis/help_text.py` as UPPER_SNAKE constants, applied with
`w.setToolTip(H.NAME)`. Never store a raw layer object on `_state` — store `layer.id()`
and resolve through `qgis/controllers/_layers.py`.

Never call `addMapLayer()` from a controller. Every created layer goes through
`qgis/controllers/_groups.py` — `self.place(layer, G.ANALYSIS)` via `LayerTreeMixin` —
which files it under the panel's Site Name in stage order (`Baseline` / `Analysis` /
`Design` / `Verify`) and adds it collapsed. A raw `addMapLayer` drops the layer loose
at the top of the legend and expanded, which is what the grouping exists to prevent.
Stage groups are found by a custom property, not by name, because their name carries
the run tag (`_state.run_tag`, frozen when Baseline runs).

Raster overlay colours live in `core/registry/map_palette.py` — one set of stops per
quantity, read by the renderers, the panel key and the report legend alike. Two rules
it holds: **darker means more water**, and **alpha is for absence, not magnitude** —
a ramp has at most one transparent stop, it sits at the absence end (zero runoff,
zero erosion, driest, planar), and every other stop is fully opaque.
`tests/test_map_palette.py` asserts that over every ramp in the file. Surface runoff
takes the one bounded exception: below `SURFACE_RUNOFF_FADE_TOP_M3` (2 m³) the ramp
is a single colour fading to nothing, so no two stops can re-order against each
other. Whole-layer opacity is not used anywhere.

A Baseline layer and its Earthworks counterpart must share one ramp top — go through
`_symbols.apply_shared_ramp(state, project, family, layer, stops)`, never
`apply_raster_ramp` directly. A pair drawn on two scales shows a difference that
belongs to the ramp and not to the design.

## The hydrology chain (read before touching any sizing number)

Two different questions, deliberately answered by different code. Conflating them is
how a spillway ends up an order of magnitude too small.

**Volume — how much rain falls.** Sizes storage.
`flow_graph.py` builds D8 pointers over the conditioned DEM and labels every cell with
the earthwork that *first* intercepts it (mutually exclusive catchments). `water_balance.py`
routes the event through those catchments; `area_subtotals()` partitions the same
labelling by sub-catchment. Depth comes from `catchment.py` via the panel's **Runoff
Calculation Method** — one of three bases (`coefficient` / `rainfall` / `runoff`), and
whichever is chosen governs the *whole* assessment.

**Rate — how fast it falls.** Sizes spillways.
`peak_flow.py` does `Q = C × i × A` with the cascade of upstream overflow.
`time_of_concentration.py` is TR-55 segmental travel time over the longest flow path
(`flow_graph.longest_flow_path(trace=True)`), and `rainfall_idf.py` holds the user's
HIRDS table so intensity is read at Tc rather than guessed.

Non-obvious invariants:
- `peak_runoff_fraction()` for the SCS basis is the **marginal** `dQ/dP`, not `Q/P`.
  The event average understates the peak by ~2× and would halve a spillway.
- `feature_inflow_m3` multiplies a **single site-wide depth** by cell count. That is
  correct only while runoff is spatially uniform — see STRETCH_GOALS §9 before wiring
  up CN zones.
- Verification compares **measured vs rasterisable**, not vs design. Dams are
  `barrier_impounded` and skip that path entirely (no drawn section to rasterise).
- **Two accumulation fields, and only one of them retains.** `runoff_accumulation`
  (m³ — the Surface Runoff layer, exit volumes, volume-mode streams) has every hollow
  hold back what it can store, so it is the water that *actually* gets downstream:
  `spread_crests(capacities=plan.capacities)`, measured off the DEM as Σ(filled − ground)
  and needing no idea what an earthwork is. `FlowAnalysis.acc` is a **cell count** —
  contributing area, feeding streams, keypoints and Tc — and must never be retained
  against, because capping a count with a volume is a category error. `retain=` is off by
  default for exactly that reason. Conservation is
  `terminal flux + retained + residual + stranded == total`; `retained` is a terminal,
  `residual` is only water the loop ran out of passes to place. **`stranded` is a
  docstring term, not a computed one** — grep finds it only in `crest_routing`'s header,
  so this identity is asserted by prose and not by code. Compute it or drop it.
- **The flat-inflation step is derived, never pysheds' default.** `resolve_flats` returns
  `filled + eps × drainage_gradient`, and that gradient is an integer BFS distance to the
  flat's outlet — so the inflation grows with the *size* of the flat while `eps` stays a
  fixed 1e-5 m. On a tile 70% covered by one harbour plane the gradient reached 1842 and
  lifted it 1.84 cm, burying every low bump standing under a centimetre proud: **516 cells
  that had a way downhill lost it, and swallowed 12.5% of the tile's drainage into
  self-loops.** Go through `resolve_flats_safely()`, which bounds the step by
  `eps × (g[B] − g[A]) < z[A] − z[B]` over every neighbour pair (`safe_flat_epsilon`) —
  gentler than the default by construction, floored so the gradient still routes in
  float64, and reporting what it could not save rather than hiding it. The conditioned
  DEM is saved **float64** because that step can be small.
- **A share of runoff must be m³ over m³, over the same ground.** `unrouted_flow()` and
  the warnings take `domain=` and `field=` because they once took neither: the numerator
  was summed over the whole tile while the denominator counted only the site, off a cell
  count the message called runoff. That is how a field run reported **105% of runoff**
  unrouted. An impossible share is printed and flagged, never clamped, and
  `unrouted_diagnostics()` writes `unrouted_{label}.txt` beside the rasters when anything
  is stuck — pass it `runoff_volume_m3`, or its m³ ratios divide by the sum of the
  throughflow field, which is each cubic metre counted once per cell it passes.
- **"The site" can be a guess, and a run that guessed says so.** `footprint.domain_mask`
  falls through drawn area → boundary → every usable DEM cell → the whole grid, and
  `with_source=True` returns which. A DEM can declare a nodata sentinel and contain none
  of it, so "usable cells" is not "land" — on the Quail Island tile it was 285 ha, 200 of
  them harbour, and that mask is the denominator of every percentage the report prints.
- **Overtopping is measured on the *full* pond, not the event.** `overtopping_spill`
  takes the brim-full ponding raster, so it answers *filled, does this pool leave over
  its own crest* — a freeboard fact about the structure. Pass `event_depth=` and each
  spill also carries `event_level_m` / `overtops_this_event`, where `None` means "not
  asked" and is not `False`. The map draws the two as separate `Overtopping (full)` and
  `Overtopping (event)` layers. **The designed spillway is subtracted from the barrier
  crest** (`burner.burned_notches[id]`, in `_build_overtopping_layer`): a notch cut into
  raised ground is still raised, so without it a correctly spillwayed dam reports leaving
  over its own crest *at its own spillway*. Reaching `overtopping_warning` with a spillway
  designed therefore now means something specific — the sill is not taking the water — and
  the old caveat about the model not seeing the structure is **deleted**, not reworded.
- **A spillway crest sits under the level water is *held* to, which is not the lowest
  ground.** `_spillway_datums` returns `(lip, invert, containment, source)` and the three
  levels do different jobs: `containment` is the ceiling `spillway_datum` and
  `spillway_validity` work against — the measured `ew.terrain_spill_level_m`, else the
  companion berm's crest as built, else a dam's wall, else the lip — and `lip` is the bare
  ring minimum, **reported** beside it. Never a berm *height estimate*: there are four
  incompatible derivations of that in the tree. A crest above the lip is a `spillway_notes`
  note, never a `spillway_validity` problem, because `_spillway_row` fails a row on any
  problem at all. Once a sill is sited the lip is taken **locally** (`pour_point_near`,
  within a sill width), because a contour swale's global ring minimum is at one of its
  ends. The datums read the **raw** DEM through `_burn_surface`, not `state.flow_dem`, so
  the crest and the burn share a surface — and the footprint mask passes
  `all_touched=False` there and only there.
- **A designed spillway is cut into the burned DEM, as a post-pass.** `burn_earthworks`
  takes `sills={ew_id: crest-bar WKT}` and runs `_cut_spillway` **after** the type
  dispatch, because fills are `np.maximum` and `_burn_berm` is additive — a notch cut
  inside a `_burn_*` is plugged by a later feature. The cut is `np.minimum(dem, crest)`,
  absolute and taken from the stored design, so it is order-independent. The snap stays in
  the controller (`_spillway_sills` → `_spillway_sill`), because `plan_geometry` forbids a
  second implementation of *nearest point on this alignment*; the burner receives only
  `Earthwork` objects. **Outflows only** — an inlet is a protected entry, and notching one
  drains the pond through its own inlet. `burned_notches` is a **fourth** mask record and is
  in `_isolated_burn`'s snapshot list; widening `burned_masks` instead would move pool
  attribution and every bermed swale's Δ. **`_keyed_dam_dem` calls the same
  `_cut_spillway`** — it bypasses `burn_earthworks` entirely, and a keyed dam is the case
  the whole change is about.
- **The crest bar runs *along* the alignment; the perpendicular is the breach axis.**
  `plan_geometry.crest_bar` is what gets cut — a weir's crest is the line the flow crosses,
  so it lies along the bank. `perpendicular_sill` is kept for the map symbol and for
  "which way does the water go". Cutting across the alignment would run the notch down the
  flow path instead of through the bank.
- **A notch refuses rather than approximates, and each refusal is a different fault.**
  No daylight (the outward march hit its cap with the bank still above the crest);
  discharges back into its own **enclosed** pond (`notch_pool` — "below the crest and
  touching the footprint" describes the whole hillside, so components reaching the window
  edge are discarded as open ground, or every spillway on falling ground is refused); a
  crest at or below the **burned** floor (`_burned_floor` takes the lower of the burned
  surface and the original, so a barrier's raised line does not read as its bed); an
  orphaned sill the controller could not snap. Every one warns through `DEMBurner.warnings`
  → the message bar, because a notch that quietly does nothing looks like a working
  spillway in every figure the design tier prints.
- **The per-feature isolated flood stays brim-full — the notch is not cut into it.**
  Measured with the notch, the pond lets go *at the sill*, so `FeatureStorage.level_m`
  comes back as the crest, `_spillway_datums` prefers it as the containment, and the crest
  band becomes `sill − head − freeboard`: every re-open ratchets the crest down by that
  much, `spillway_validity` fails every spillwayed feature, and the give-up readout goes to
  zero. So the flood measures the container and `_sill_limited_capacity` reads the volume
  held to the sill off `stage_storage` at the crest — measured equal to a notched flood on
  all five sited spillways of the Quail Island design (Round 20). The **site** burn still
  cuts the notch, which is where the rasters, the routing and the overtopping check see it.
- **The burned width is whole DEM cells** (`spillway_burn_width`, coarser axis). Nothing in
  the raster tier meters flow *rate*, so this cannot change a total — it changes
  `cells.size` at the exit, and with it the `q = Q/L` the erosion advisory is judged by.
  The note that says so is `spillway_notes`', and it must **not** say "the extra width
  lowers the head": true of the weir equation, and it reads as though widening moved the
  water level in the feature, which it does not.
- **An auto spillway width is derived, never serialised.** `Spillway.to_dict` drops
  `width_m` when `width_auto` is set: it tracks a requirement the design file does not pin
  down, so storing it meant opening an old project silently rewrote a figure the user never
  chose. `width_required_m` carries the un-rounded requirement beside the built width.
  The restore path runs `_refresh_auto_spillway_widths` **before** the feature list, the map
  label and the sill bar — all three read `width_m` — and `_spillway_row` derives it rather
  than trusting the stored value.
- **Three elevations describe one spillway, and the disagreements are the point.**
  Designed (`Spillway.crest_elevation`), as burned (`ew.burned_sill_elevation_m`, the
  highest level water crossing the notch must clear) and actual (`ew.actual_spill_level_m`,
  where the finished pond was measured letting go). Both measured ones come from the site
  burn via `_record_spillway_levels` and are cleared with the DEM. As-burned above designed
  = the notch was refused; actual above as-burned = it did not daylight; actual below
  designed = a lower saddle is the control.
- **"% full" divides by the brim volume, not the sill volume.** `EarthworkStore.
  lip_capacity_m3` (from `ew.containment_capacity_m3`) defaults to 0.0 and every reader
  falls back to `capacity_m3`, so nothing unmeasured changed. Without it a working spillway
  pins its feature at 100% exactly when it starts doing its job. `cascade_overflow` keeps
  thresholding on `capacity_m3`, which with the notch cut is correctly the sill volume.
- **`crest_elevation` is the authoritative crest; the other two are views of it.**
  `bind_crest` binds all three (↔ `drop_below_rim_m` ↔ `height_above_floor_m`) and clamps
  into the band *before* deriving partners — one function, because three call sites is how
  a binding creeps apart. Both relatives are re-derived from the crest on restore
  (`rebase_spillway`, run by `_rebase_restored_spillways`): a serialised drop measured
  against a datum that has since changed meaning still looks like a setting-out figure.
- **A pond's stage–storage curve is free, and is never serialised.**
  `FeatureStorage.stage_storage` comes off arrays `feature_storage` already holds — one
  sort and a running sum — and is retained on `ew.stage_storage` under the same rule as
  `terrain_capacity_m3`. It is sampled, not one point per cell, and its datum is the pond
  bed **net of natural ponding**, so `volume_at(level_m)` reproduces `volume_m3` exactly.
  A dam's curve comes from `DEMBurner.dam_storage`, because `dam_stage_storage` returns
  only the volume.
- **A diversion drain can start at a spillway crest, and the link carries which end.**
  `Earthwork.spillway_link_id` is `"<source id>:<kind>:<end>"` — an id because a flag
  cannot say *which* spillway and a feature carries two, and **serialised** because it is
  a decision recoverable from nothing (`SCHEMA_VERSION` 3). `invert_start_m` is the level
  it resolves to; it is derived and never serialised, and it replaces `_burn_diversion`'s
  **ground sample**, so the bed comes out one depth below it and it is not the invert its
  name suggests. Bed-*at*-crest was considered and refused: the notch is cut to the crest
  too, so the two would meet at one level and leave a flat for the conditioning to
  resolve, and under `np.minimum` a drain raised by one depth cuts nothing wherever the
  ground is already below the crest. It must be an absolute off the design: the notch is a post-pass, so when
  `_burn_diversion` runs the source's spillway is not in the array yet, and reading the
  surface would reintroduce the order dependence the link removes. **The end travels with
  the link rather than the alignment being reversed on attachment** — reversing mutates a
  geometry the user drew, desynchronises `source_contour_coords`, and survives an unlink;
  `_burn_diversion` reverses its own working copy instead. Links resolve at **read time**
  (`resolve_spillway_links`, mirroring `overflow_target_id` → `resolve_targets`): a
  deleted, disabled or de-spillwayed source leaves the link dangling and the drain grades
  from its own ground, reported at burn time — clearing links eagerly on delete is a
  second and silently different failure. Cycles are refused via `topological_order`, and
  self-links separately, because that function skips an edge to its own node. Only an
  **outflow** is honoured: an inlet is where water arrives, so a drain on one is
  delivering and its grade would run backwards, which under `np.minimum` cuts nothing at
  all. `burn_order` puts a source ahead of its drains — stable and minimal, so a design
  with no links burns in exactly the order it was given.

## The report (`Site Water Plan`, PDF)

**A baseline is the only precondition.** The report used to gate on `state.comparison`,
written only at the end of a fill simulation, so no document existed until one had run —
and the simulation's per-feature figures were wrong (every feature 0.0 m³ inflow against a
claimed 100% capture). It is now built from the **design tier**: `run_water_balance()` →
`BalanceResult`, which is recomputed on every design edit and is retained on `_state`
(`balance`, `balance_stores`, `spillway_rows`, `spillway_context`) instead of falling out
of scope. A simulation, where one has run, is optional enrichment.

**PDF and HTML are one document with two renderers.** There is no separate HTML report —
`export_html()` and its 300-line f-string are gone. Both formats consume the same
`Report`, the same chart PNGs and the same maps, so they can differ only in paper size.

| File | Layer | Holds |
|---|---|---|
| `modules/report_model.py` | pure | `build_report(data) -> [Section]` — what the document says, and what each section says when its data is missing. Sections degrade with a stated reason; never silently dropped. |
| `modules/report_charts.py` | pure | The flow-network diagram; wrappers over the existing hydrograph/fill-timeline builders. Geometry is **millimetres at printed size** with `set_aspect("equal")`, so point sizes are the sizes on paper. |
| `modules/report_html.py` | pure | → HTML, self-contained (inline CSS, images as data URIs) |
| `qgis/adapters/layout_pdf.py` | QGIS | → PDF. The only file that knows QGIS layouts exist. |

**Adding or changing a section must be done in both renderers, in the same change.** Each
keeps a `SECTION_HANDLERS` table mapping section type → handler, and
`tests/test_report_renderer_parity.py` asserts the two cover exactly the same types and
that every model figure reaches the HTML. Add a type to one renderer only and the suite
fails. Extend the model → add a handler on both sides → the parity test is the gate.

Timing claims live **only** in `_page_simulation`, which appears when `state.comparison`
exists and says on the page that it came from the simulation. Nothing else in the report
may imply *when* water arrives — the rest is an event-total balance.

Three presentation rules the tests enforce: the three storage figures appear in derivation
order with **Δ only ever against the grid**, the two exit volumes are never in one table,
and anything unreliable is **suppressed rather than printed as zero** (`drain_hours=None`
is "does not empty by soaking", not `0`; a NaN volume is an em dash, not `nan` — see
`round_volume`).

There were four storage figures until "Design storage" was dropped: it was `geometric`
less a blanket freeboard fraction, and freeboard on a real feature is set by its
spillway, which is sized on its own page from a peak flow that column knew nothing
about. A figure from a rule of thumb, sitting first in a run of columns meant to be
compared against each other, invited exactly the comparison it could not support.

Two kinds of figure, named the same way everywhere: **geometric calculated** (from the
drawn dimensions, assumes flat ground) and **measured** (off the elevation model). The
words are the `CALCULATED` / `MEASURED` constants in `report_model.py` and every table
carries one via `_tag()`.

Non-obvious layout facts, all measured rather than assumed:
- `QgsLayoutItemTextTable.totalSize()` returns the **frame** height, not the content height,
  and the table **silently overflows** its frame rather than wrapping. Row height is linear
  in font size (4.730 mm at 7 pt) — see `table_content_height()`.
- A scale bar must be **added to the layout before** being configured, or it cannot resolve
  its linked map's scale and renders a fifth of the right width. `applyDefaultSettings()`
  alone leaves 0 m per segment.
- `attemptMove(..., page=N)` takes a **page-relative** point, not a document offset.
- Every glyph must exist in DejaVu Sans or it prints as a box (U+2312 ARC does not).

## The key rule for new features

**Before writing code for a non-trivial feature, propose the file split and get sign-off.**
A feature almost always spans layers (e.g. a new analysis = new `modules/*.py` + extend a
controller + extend `panel.py`). Guidance:
- New *concept/algorithm* → new `modules/*.py`. A *variant* of existing logic → new function
  in the existing module (don't create a tiny file per feature).
- New *feature area* with its own buttons → maybe a new `qgis/controllers/*.py`; otherwise
  extend the fitting existing controller.
- Almost everything also touches `panel.py`.

Only add/modify what's asked — no drive-by refactors of working code.

## Dev workflow

- **Deploy to QGIS:** `deploy.ps1` (copies `terrainflow_assessment/` into the QGIS profile;
  then disable + re-enable the plugin in QGIS Plugin Manager to reload). No zip needed.
  It **prunes build artefacts** on the way in (`__pycache__`, `.pytest_cache`, `*.pyc`,
  `*.pyo`, `*.aux.xml`, `symbology-style.db`) and refuses to report success if any
  bytecode survives. That matters because `run_qgis_tests.ps1` runs QGIS's own Python over
  the source tree, so the repo fills with `cpython-312` `.pyc` — the interpreter QGIS uses —
  and the copy preserves mtimes, so Python would trust that bytecode over the sources
  beside it. Note it deploys the working tree, **not** `git ls-files`: work in progress is
  untracked by definition. Every `.ps1` is ASCII-only and BOM-less, because PowerShell 5.1
  reads a `.ps1` as ANSI and a stray em dash is a parse error — enforced by
  `tests/test_architecture.py`, which also guards the layering rules below.
- **Tests:** `python -m pytest tests/` (target Python 3.9) — **run the whole suite; do not
  scope it.** It takes under a minute, and the profile is flat (one test over
  2 s), so there is no slow tail to skip. Scoping saves under a minute and costs
  correctness: `earthwork_design.py` fans out to 11 test files and `catchment.py` to 6,
  and there is no `modules/earthwork.py` or `modules/dem_burner.py` despite tests named
  for them. A mapping that lossy sells false confidence for a few seconds.
- **Coverage is asked for, never assumed.** `pyproject.toml` gates pure-Python `modules/`
  at 95% but keeps `--cov` **out of `addopts`**, because a forced gate made every scoped
  run print `43 passed` *and* `FAIL Required test coverage`, exiting 1 — an exit code that
  says failure on a green run teaches you to stop reading exit codes. Before a commit run
  `python -m pytest tests/ --cov=terrainflow_assessment --cov-report=term-missing`; that
  is what CI runs, and the `qgis/*` Qt/QGIS layer is omitted from the gate because the
  real-QGIS harness below covers it instead.
- **Lint:** `python -m ruff check terrainflow_assessment/` — bare `ruff` is not on PATH
  here. CI (`.github/workflows/ci.yml`) runs ruff + pytest + a grep-gate against
  deprecated QGIS APIs on every push.

## Real-QGIS testing (`tests_qgis/`) — run this after touching `qgis/`, `panel.py` or `map_tools/`

`pytest tests/` replaces `qgis.core`/`qgis.gui` with mocks, so it never executes the
controllers, renderers, workers or map tools. `tests_qgis/` boots a genuine QgsApplication
(offscreen), builds the real panel and controllers, and drives them through real panel
signals and real mouse events. It lives **outside** `terrainflow_assessment/` on purpose —
only that folder is deployed or zipped, so none of it can reach a shipped build.

```powershell
.\run_qgis_tests.ps1              # the full suite, headless, 8-10 min. Exit code gates.
#                                   longer than a 10-min tool timeout — background it.
.\run_qgis_tests.ps1 checks_baseline   # one module, ~30-40 s. This is the iteration loop.
.\run_qgis_tests.ps1 --skip=checks_report   # everything else
.\run_qgis_tests.ps1 -Prompt      # run, then ASK whether to accept changed screenshots
.\run_qgis_tests.ps1 -Accept      # accept the screenshots on disk (instant, no re-run)
.\run_qgis_gui_shot.ps1           # load in real QGIS, screenshot the window, quit
```

**Scope this suite while iterating; run it whole before you commit.** It is the expensive
one — 8-10 min against ~70 s for `pytest tests/` — and each module is its own QGIS
subprocess costing ~9-10 s to boot, so ~160 s of a full run is boot alone and there is a
~30 s floor under any run at all. One named module is ~30-40 s. Eight minutes every edit
is the kind of honest-but-unaffordable habit that quietly decays into running nothing.

**Pass the full `checks_*` module name, not a bare word.** Patterns are matched against
check *names* as well as module names — which is how a single check gets run by name, and
also how `.\run_qgis_tests.ps1 report` silently selected `checks_crs`, `checks_simulation`
and `checks_threading` (they hold `check_..._reported` / `..._reports_...`) and paid three
extra QGIS boots for three checks nobody wanted. `checks_report` matches only itself. Every
run now prints which modules it chose and what matched them, so check that line.

| Touching | Run |
|---|---|
| `controllers/baseline.py`, `workers/analysis_worker.py` | `checks_baseline` |
| `controllers/contour.py`, `modules/contour_analysis.py`, `modules/keypoint_analysis.py` | `checks_contour` |
| `controllers/terrain.py`, `modules/terrain_indices.py` | `checks_terrain` |
| `controllers/earthworks.py` | `checks_earthworks` |
| `controllers/simulation.py`, `workers/simulation_worker.py` | `checks_simulation` |
| `controllers/reporting.py`, `modules/report_*.py`, `adapters/layout_pdf.py` | `checks_report` |
| `adapters/map_image.py`, anything CRS-shaped | `checks_crs checks_report` |
| `controllers/design_file.py`, `modules/project_io.py` | `checks_design_file` |
| `controllers/_symbols.py`, `core/registry/map_palette.py` | `checks_symbology checks_visual` |
| `qgis/widgets/*`, the dialogs, `help_text.py` | `checks_visual` |
| `map_tools/*` | `checks_maptools checks_visual` |
| `qgis/workers/*` | `checks_threading checks_threading_restart checks_threading_tasks` |
| `qgis/plugin.py`, teardown paths | `checks_lifecycle` |
| sizing or hydrology numbers | `checks_fixture_regression` |
| pit filling, nodata handling, anything reading raw DEM cells | `checks_robustness` |
| `controllers/_groups.py`, `_layers.py` | `checks_layer_tree`, then the full suite |

**Some files have no scope.** `panel.py`, `_state.py`, `_groups.py`, `_layers.py`,
`_symbols.py`, `core/registry/*`, `qgis/plugin.py` and `tests_qgis/_harness.py` reach every
module — touching one means the full run, no shortcut. `tests/test_architecture.py` asserts
this table names every `checks_*.py` on disk and nothing that isn't, so a new module cannot
be added without a row here.

**Say which suites ran.** A scoped pass is not "tests pass". Report it as *"pytest tests/:
all passed. QGIS: `checks_earthworks` only — full suite not yet run."* Before any commit
touching `qgis/`, `panel.py` or `map_tools/`, run `.\run_qgis_tests.ps1` bare and say so.

- Needs QGIS's own Python; the scripts find it. **Not in CI** (no QGIS on the runner) —
  these are local, pre-deploy commands.
- **A modal dialog is recorded, not shown.** Offscreen a `QMessageBox` has nobody to click
  OK, so it blocks until the module's timeout — which is how `recommend_ponds` spent 300 s
  and was quarantined as "does not terminate" when it was really a one-line guard saying
  *run keypoints first*. `RecordingDialogs` captures them; `assert_no_errors` fails on any
  warning or error dialog, so a controller's `except QMessageBox.critical` reports itself
  instead of stalling. Assert on them with `h.dialogs.of("warning")`.
- Nothing is quarantined at present. `run_all.OPT_IN_MODULES` still exists for it —
  a module named there runs only when asked for by name.
- **A module whose process dies is a failure, and prints why.** Every worker enables
  `faulthandler`, so a PyQGIS access violation dumps the Python stack of each thread on
  the way down, and the orchestrator folds the last 40 lines of the module's output into
  the summary. This is not decoration: the runner used to score a dead subprocess `(0, 0)`
  — no pass, no fail, no message — so a run could lose thirty checks and eleven screenshots
  while announcing "0 failed". Two real crashes were diagnosed off these stacks within a
  day of adding them (a panel still wired to `QgsProject` after unload, and a boundary
  seed indexing outside a numba kernel in pysheds).
- **Workers run inline** (`make_workers_synchronous` aliases `start = run`), which is what
  makes the suite deterministic — and what makes every concurrency fault invisible:
  `isRunning()` is never True, so a double-start or a teardown-during-run cannot be
  constructed. A module opts out with a top-level `REAL_THREADS = True`
  (`checks_threading.py` for the analysis/simulation workers,
  `checks_threading_tasks.py` for the `TaskWorker` ones). Don't reach for it elsewhere. Inside such a module the
  harness's own helpers stop being synchronous — `h.run_baseline()` returns while the
  thread runs, so touching `state.analysis_worker` afterwards commits the very bug under
  test; use `run_baseline_and_wait`. Determinism comes from parking a worker on a
  `threading.Event`, never from racing real work.
- Renders the real panel/dialogs/canvas to PNGs in `tests_qgis/_shots/` — **open them**;
  that is how the UI gets verified. Rendering is deterministic, so a plain run reports
  exactly which images a change moved. A changed image is information, not a failure.
- **Accept a deliberate UI change or the diff rots into noise.** Look at the image, then
  `-Accept` it. Unaccepted differences nag with an escalating reminder that counts the
  consecutive runs, because a stale baseline hides the next real regression.
- `tests_qgis/README.md` explains the fixture, the visual-diff workflow, and several
  non-obvious traps (`QTest.mouseMove`/`mouseDClick` do not behave like real input;
  `saveAsImage` omits rubber bands). Read it before extending.
- Offscreen Qt **does** render text — `_harness.py` points `QT_QPA_FONTDIR` at the system
  fonts, which restores ~233 families. Labels are therefore verifiable headlessly; glyph
  shapes still differ from live QGIS, so judge layout/colour/content here and *type* from
  `run_qgis_gui_shot.ps1`. (Qt registers zero families *without* that env var — which is
  what the old "offscreen Qt loads no fonts" note meant.)

## Docs

`CLudeDocs/` holds the Plugin State and Next Steps (roadmap) documents.

Two of its documents are registers rather than prose, and are the authority on what is
known to be wrong:

- `MATHS_AUDIT.md` — the maths audit. §9 is the authority on current state; §1 and §2 are
  left as the record of what was found. §10 maps §2.20's line citations onto the rebuilt
  `keypoint_analysis.py`.
- `ANALYSIS_DEFECTS.md` — 31 findings against the analysis tier as rebuilt 2026-09-10,
  measured on the real fixture. Reproduced by the probes in `tests_qgis/probes/`, which are
  measurement scripts, never auto-run, and deliberately outside the `checks_*.py` glob.
