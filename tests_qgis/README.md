# tests_qgis — real-QGIS smoke checks

Headless checks that run the plugin against a **genuine QgsApplication**: real
`QgsProject`, real panel widgets, real controllers, real Processing algorithms.

**Nothing here ships.** Only `terrainflow_assessment/` is deployed (`deploy.ps1`)
or zipped for release, and this directory sits outside it. No file under
`terrainflow_assessment/` is modified or imported-into by this harness.

## Why it exists

`tests/conftest.py` replaces `qgis.core` / `qgis.gui` with `MagicMock`s so the pure
`modules/` logic is testable without QGIS. The trade-off is that `pytest tests/`
never executes the `qgis/` layer at all — controllers, adapters, renderers, map
tools, workers. Those were only ever verified by deploying and clicking. These
checks cover that gap.

The two suites are complementary, not competing:

| | `pytest tests/` | `run_qgis_tests.ps1` | `run_qgis_gui_shot.ps1` |
|---|---|---|---|
| QGIS | mocked | real, headless (offscreen Qt) | real, on screen |
| `iface` | mocked | stub (4 methods) | genuine `QgisInterface` |
| Target | `modules/` algorithms, 95 % gate | `qgis/` glue, end-to-end wiring | how it actually looks |
| Speed | seconds | ~2 min | ~20 s |
| Reliability | high | high | depends on GUI startup timing |
| Catches | wrong maths | wrong API calls, dead layer refs, broken signals, styling crashes | theme/font/layout problems, real-`iface` divergence |

## Running

```powershell
.\run_qgis_tests.ps1                    # everything, ~8 min
.\run_qgis_tests.ps1 checks_baseline    # one module, ~30-40 s
.\run_qgis_tests.ps1 --skip=checks_report   # everything else
.\run_qgis_tests.ps1 --timeout=600      # raise the per-module limit (default 300 s)
```

**Which module covers what you changed** is the `Touching -> Run` table in the repo root
`CLAUDE.md`, under "Real-QGIS testing". It lives there rather than here because that is
the file read at the start of every session, and a second copy would be one more thing to
drift; `tests/test_architecture.py` asserts the table names every `checks_*.py` on disk.

Prefer the full `checks_*` module name over a bare word: patterns match check *names* too,
so `report` also selects `checks_crs`, `checks_simulation` and `checks_threading` and pays
a ~10 s QGIS boot for each. Every run prints what it selected and why.

It finds the newest QGIS install automatically; override with
`$env:TERRAINFLOW_QGIS_PYTHON` pointing at `bin\python-qgis-ltr.bat`.

Requires **no extra packages** — deliberately no pytest, because pytest is not in
QGIS's bundled Python and installing into a working QGIS install to run tests is
a bad trade. `run_all.py` is a ~200-line runner. (It also means the coverage gate
in `pyproject.toml` can't collide with these checks.)

Exit code is 0 only when every check passes.

**These do not run in CI today.** `.github/workflows/ci.yml` runs ruff + pytest +
the deprecated-API grep on a runner with no QGIS installed, and both scripts here
need QGIS's own Python. Gating on them would mean provisioning QGIS on the runner.
Until then treat them as local commands: run them before a deploy, and after any
change to the `qgis/` layer.

## Layout

- `_harness.py` — QgsApplication bootstrap, Processing init, stub `iface`,
  synthetic DEM, `PluginHarness`
- `checks_*.py` — the checks; any module-level `check_*(dem_path)` function is
  discovered automatically
- (`checks_slow.py` is gone: it held only `recommend_ponds`, whose "does not
  terminate" was a modal dialog waiting offscreen for a click. The harness records
  those now, so it runs in seconds and lives in `checks_contour`. `OPT_IN_MODULES`
  is kept as a mechanism — quarantining beats deleting a check that found
  something real.)
  routine run by naming the other modules explicitly, or run with a big
  `--timeout` when you want to chase them
- `run_all.py` — discovery, per-module process isolation, timeouts, reporting

Each `checks_*.py` module runs in its **own QGIS subprocess** with a time limit.
That costs ~5 s of boot per module and buys two things worth more: a hung check
or a PyQGIS segfault reports as one failed module instead of wedging the run, and
no module leaks state into the next.

## Writing a check

```python
from _harness import PluginHarness

def check_something(dem_path):
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.some_requested.emit()       # drive the real panel signal
        h.assert_no_errors("some feature")  # nothing pushed to the message bar
        assert h.state.something is not None
```

Two conventions matter:

**Drive panel signals, not controller methods.** `h.panel.run_baseline_requested.emit()`
goes through the same wiring a button click does, so a signal renamed in
`panel.py` but not in `plugin.py` fails the check.

**`assert_no_errors` is the workhorse.** The controllers catch their own
exceptions and report them via `iface.messageBar().pushCritical(...)` rather than
raising. The stub message bar records every push, so asserting no criticals is
how a swallowed exception becomes a test failure. Warnings are allowed — they are
often correct behaviour (missing prerequisite, sub-cell feature).

## Visual checks (screenshots)

`checks_visual.py` renders the real panel, the real dialogs and the real map canvas
to PNGs in `_shots/`, then asserts something was actually *drawn*. This catches
what state assertions cannot: a renderer that builds cleanly but paints nothing, a
layer added beneath an opaque raster, a dock that collapses to zero width.

`_shots.py` provides `save_widget` / `save_canvas`, plus `describe()` and
`assert_rendered()`. Two details there are load-bearing:

- **`QT_QPA_FONTDIR` is mandatory.** Qt's offscreen platform on Windows registers
  **zero** font families, so every `drawText()` is a silent no-op — screenshots
  come out with layout and colour intact but not one glyph, and QGIS map labelling
  produces nothing at all. `_harness._configure_qt_platform()` points Qt at
  `C:\Windows\Fonts` (324 families). Without it, visual checks look like they pass
  while proving nothing about anything textual.
- **`describe()` counts every pixel**, via numpy, rather than sampling a grid. A
  1-cell-wide stream is a few pixels across; a grid coarse enough to be fast in
  pure Python steps straight over it and reports a layer that clearly drew as
  blank. That exact false positive happened during development.

Deliberately **no golden-image assertions**. The UI is being actively restyled, so
a check that *fails* on a pixel change would fail on every legitimate change and get
ignored within a week. The assertions are "it drew, and not as one flat wash"; a
human (or Claude) reads the PNGs for whether it drew *well*.

`_shots/` is gitignored — every run overwrites it, so it is always the latest run.

### Visual diff: which screenshots did my change move?

Rendering is **fully deterministic** — two runs of unchanged code produce
byte-identical images — so a pixel diff carries no antialiasing noise and can be
trusted.

```powershell
.\run_qgis_tests.ps1               # 1. what moved since the last accepted baseline?
#                                    2. open the changed PNGs; is that what you meant?
.\run_qgis_tests.ps1 -Accept       # 3. yes -> make them the new reference (instant)
```

Or let it ask: `-Prompt` runs, lists any changed images and asks whether to accept them.
`-Snapshot` is the same acceptance but bundled into a run. `-Accept` re-uses the
screenshots already on disk, so it takes no time at all.

**Unaccepted changes escalate.** `_shots_baseline/diff_state.json` records which images
differ and for how many consecutive runs; from the second run on, the summary nags with
the streak count. This exists because the real failure mode is not a wrong baseline, it
is an *ignored* one — the same names scroll past every run, you learn to skip them, and
the next genuine regression hides in the list. Reverting the change clears the state on
its own; nothing to tidy up.

Output looks like:

```
Visual diff vs baseline: 6 changed, 0 new, 8 unchanged
  CHANGED   canvas_baseline.png  0.45% of pixels, max channel delta 72
  CHANGED   canvas_dem.png       0.46% of pixels, max channel delta 130
```

A changed image is **information, not a failure** — it never affects the exit code,
because a moved pixel is usually the change you intended. It turns "I think that
looks different" into "these three images moved and these eleven did not."

Notes on the mechanism:
- Only images rewritten *during this run* are compared, so running a subset
  (`checks_visual`) does not report every other screenshot as unchanged.
- Differing image dimensions report as `RESIZED` rather than a pixel percentage,
  since the two can't be compared directly.
- `pixels()` reads PNGs through rasterio/GDAL rather than QImage, so the
  comparison runs in the `run_all.py` orchestrator without booting QGIS — only
  the worker subprocesses have a Qt application.
- `_shots_baseline/` is gitignored too. There is no committed golden set to
  maintain; the baseline is whatever you last snapshotted.

## Map tools: real synthetic mouse input

`checks_maptools.py` clicks on the canvas. `_mouse.py` converts map coordinates to
canvas pixels and delivers real Qt events to `canvas.viewport()`, so input travels
the genuine path — QgsMapCanvas builds the `QgsMapMouseEvent`, the tool converts it
back with `toMapCoordinates()`. A tool that mis-reads a click position fails here
and nowhere else.

```python
h.prepare_canvas_for_input()                       # sized, visible, extent set
h.plugin._earthworks.activate_draw_earthwork("diversion")
click_map(h.canvas, x1, y1)                        # coordinates in metres
click_map(h.canvas, x2, y2)
click_map(h.canvas, x3, y3, button=RIGHT)          # right-click finishes
```

Four things had to be solved, all of them silent failures rather than errors:

- **The tools need the global `qgis.utils.iface`.** Four map-tool modules do
  `from qgis.utils import iface`, and `DrawLineTool.__init__` calls
  `iface.mainWindow().statusBar()`, so headless it raises `AttributeError` before a
  single click. `PluginHarness._install_global_iface()` sets both the
  `qgis.utils.iface` attribute (for modules imported later — several are imported
  lazily inside functions) **and** sweeps `sys.modules` to rebind modules that
  already captured `None`. `from x import y` copies the value, so setting the
  attribute alone does not reach an already-imported module.
- **`QTest.mouseMove` does nothing offscreen.** It warps the real cursor, so
  `canvasMoveEvent` never fires and the rubber-band preview segment silently never
  appears. `move_map()` posts a `QMouseEvent` directly instead.
- **`QTest.mouseDClick` is not a real double click.** It sends a bare
  `MouseButtonDblClick` with no preceding press. A real double click is press,
  release, DblClick — the press lands a vertex which `canvasDoubleClickEvent` then
  pops. Using QTest's version made a 3-vertex line come out with 2. `dclick_map()`
  composes the faithful sequence.
- **`canvas.saveAsImage()` does not capture rubber bands.** It renders map *layers*
  only; a `QgsRubberBand` is a `QgsMapCanvasItem` in the QGraphicsView scene. Use
  `save_widget(canvas, ...)` for anything mid-gesture.

Assert on the tool's own state (`tool.points`, `tool.rubber_band.numberOfVertices()`)
with exact counts, not `>=`. A `>= 2` assertion passed while the cursor move was
never being delivered at all.

What this does **not** cover: whether the interaction *feels* right — snap radius
forgiveness, whether the rubber band reads clearly while dragging. That stays a
human judgement.

### The real-GUI run

`run_qgis_gui_shot.ps1` launches genuine QGIS with `--code launch_in_qgis.py`,
loads the plugin against the true `iface`, runs a baseline, screenshots the whole
main window, and quits. Use it when the question is "does this look right in real
QGIS" — real theme, real UI font, real dock geometry, real layer tree. Add `-Keep`
to leave QGIS open and poke at it by hand.

It waits for readiness rather than sleeping on a guess: a `QTimer` polls until
`qgis.utils.iface` exists and its main window is visible, then settles briefly for
providers to finish loading, and gives up after 120 s. (`qgis.utils.iface` is
populated *during* startup, so it must be read fresh each poll — a module-level
`from qgis.utils import iface` binds `None` forever.) The launcher wraps the run in
a `-TimeoutSec` watchdog (default 300) that kills the whole process tree, since
killing the `.bat` alone would orphan `qgis-ltr-bin.exe`. `-Keep` skips the
watchdog, because there the window is meant to stay open.

Three things it must do, each learned the hard way:

- `--profile TerrainFlowTest` — never touch the working profile's settings.
- `--noplugins` — otherwise the *deployed* copy of TerrainFlow loads as well and
  you get two panels fighting over one project.
- **Unload the plugin and clear the project before quitting.** Calling `quit()`
  with the plugin still loaded reliably crashes QGIS on shutdown
  (`exitQgis` → `QgsProject::clear` → `removeAllMapLayers` → layer-tree teardown →
  access violation), because the plugin still holds its dock, its cached layer ids
  and its earthworks layer-tree group. Live QGIS unloads plugins first; this script
  has to do it itself. See `_teardown_then_quit()`.

It also writes `_shots/gui_report.txt`, so the run is read afterwards rather than
watched, and the script writes a breadcrumb line the moment it loads — if the
report contains only that line, `--code` reached the file but `drive()` never ran.
(QGIS may `exec` `--code` without defining `__file__`; the resulting `NameError` is
swallowed and the script silently never runs, which is why `TFA_TESTS_DIR` exists
as a fallback.)

## Deliberate limitations

- **Modal dialogs** block a headless run. `check_properties_dialog_path` stubs
  `EarthworkPropertiesDialog.exec` to "accepted" and exercises everything else.
  Other dialogs (rainfall table, design intensity) are not covered.
- **Typography in offscreen screenshots is not trustworthy.** With `QT_QPA_FONTDIR`
  set, text renders — but Qt picks from the raw font directory rather than the
  system UI font stack, so glyph shapes and metrics differ from live QGIS. Judge
  layout, colour, spacing and content from `checks_visual.py` output; judge *type*
  from `run_qgis_gui_shot.ps1` output, which uses the real theme.
- **Mouse input is simulated for `DrawLineTool` only** (see `checks_maptools.py`).
  The other seven tools in `map_tools/` are not yet driven; elsewhere geometry is
  handed to the controllers directly. The `_mouse.py` helpers should carry over —
  the tools are all plain `QgsMapTool` subclasses with no dependency on QGIS's
  snapping subsystem — but each has its own completion gesture to work out.
- **Synthetic terrain** is smooth and noise-free — kinder than a real LiDAR DEM
  in some ways, and pathological in others (perfectly flat contours, no pits).
  It is also sized and shaped against the panel's *defaults*: 36 ha because the
  default stream threshold is 5 ha and D-infinity routing on this terrain only ever
  concentrates ~35 % of the site into one cell, and incised on the centreline
  because without a channel the smooth valley disperses flow so widely that the
  stream layer is legitimately empty and its renderer never runs. Changing those
  panel defaults may require re-tuning the fixture.
