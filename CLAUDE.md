# TerrainFlow — project guide

QGIS plugin for NZ land managers / permaculturists: drainage analysis + earthwork planning
(swales, dams, basins, diversion drains) with before/after flow comparison.

## Where the code lives

- **`terrainflow_assessment/` is the active codebase.** All new work goes here.
- **`plugin/` is the retired original** (bloated monolith). Do not develop in it; reference only.
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
| Workers | `qgis/workers/` | Background threads (analysis, simulation) |
| Adapters | `qgis/adapters/` | Thin `QgsProject`/geometry wrappers |
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
- **Tests:** `python -m pytest tests/` (target Python 3.9). `pyproject.toml` sets a 95%
  coverage gate on pure-Python `modules/`; the `qgis/*` Qt/QGIS layer is omitted from that
  gate — it is covered by the real-QGIS harness below instead.
- **Lint:** `ruff check terrainflow_assessment/`. CI (`.github/workflows/ci.yml`) runs ruff +
  pytest + a grep-gate against deprecated QGIS APIs on every push.

## Real-QGIS testing (`tests_qgis/`) — run this after touching `qgis/`, `panel.py` or `map_tools/`

`pytest tests/` replaces `qgis.core`/`qgis.gui` with mocks, so it never executes the
controllers, renderers, workers or map tools. `tests_qgis/` boots a genuine QgsApplication
(offscreen), builds the real panel and controllers, and drives them through real panel
signals and real mouse events. It lives **outside** `terrainflow_assessment/` on purpose —
only that folder is deployed or zipped, so none of it can reach a shipped build.

```powershell
.\run_qgis_tests.ps1              # 44 checks, headless, ~2-4 min. Exit code gates.
.\run_qgis_tests.ps1 baseline     # only checks matching "baseline"
.\run_qgis_tests.ps1 -Prompt      # run, then ASK whether to accept changed screenshots
.\run_qgis_tests.ps1 -Accept      # accept the screenshots on disk (instant, no re-run)
.\run_qgis_gui_shot.ps1           # load in real QGIS, screenshot the window, quit
```

- Needs QGIS's own Python; the scripts find it. **Not in CI** (no QGIS on the runner) —
  these are local, pre-deploy commands.
- `checks_slow.py` is quarantined (its `recommend_ponds` check does not terminate) and runs
  only when named: `.\run_qgis_tests.ps1 slow`.
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
