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
| UI | `panel.py`, `earthwork_properties_dialog.py` | All buttons/inputs/tooltips/tables + the signals they emit |
| Controllers | `qgis/controllers/` (`baseline`, `contour`, `earthworks`, `simulation`, `reporting`, `_state`) | Per-feature QGIS glue: run analysis, render layers, styling |
| Workers | `qgis/workers/` | Background threads (analysis, simulation) |
| Adapters | `qgis/adapters/` | Thin `QgsProject`/geometry wrappers |
| Modules | `modules/` | **Pure analysis logic, no QGIS UI** — the testable core |
| Map tools | `map_tools/` | Interactive canvas clicking (draw, select, query) |
| Core | `core/registry/` | Pure config (earthwork type definitions) |

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
  coverage gate on pure-Python `modules/`; the `qgis/*` Qt/QGIS layer is omitted (untestable
  outside a live QGIS runtime — verify it with a manual smoke test in QGIS instead).
- **Lint:** `ruff check terrainflow_assessment/`. CI (`.github/workflows/ci.yml`) runs ruff +
  pytest + a grep-gate against deprecated QGIS APIs on every push.

## Docs

`CLudeDocs/` holds the Plugin State and Next Steps (roadmap) documents.
