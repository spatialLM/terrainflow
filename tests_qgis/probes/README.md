# `tests_qgis/probes/` — measurement, not assertion

A **probe** answers a question about what the analysis tier actually does. A
`checks_*.py` module asserts that an answer has not moved. The two are different jobs and
they live apart on purpose.

* Probes are named `p_*.py`. `run_all.py:281` and `tests/test_architecture.py:298` both
  glob `checks_*.py` **non-recursively in `tests_qgis/`**, so nothing under this directory
  is ever auto-run, and adding a probe does not oblige a row in CLAUDE.md's
  "Touching → Run" table.
* A probe never fails the build. It measures, prints, and writes its numbers to
  `evidence/<name>.json`. A surprise is *recorded*, not raised — the whole point is to see
  what is there.
* Every evidence file names the finding IDs it supports and the date it was written, so a
  register entry in `CLudeDocs/ANALYSIS_DEFECTS.md` can cite it and a reader can re-run it.

## Running one

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
& 'F:\bin\python-qgis-ltr.bat' tests_qgis\probes\p_smoke.py
```

Probes take the DEM to work on from `--fixture=PATH`, falling back to
`tests/fixtures/quail_island_catchment.tif`. They import `_harness` by putting
`tests_qgis/` on `sys.path` themselves, so they can be run from any working directory.

**Never edit source while a QGIS run is in flight.** Each module imports the tree as it
starts; a mid-run edit makes later modules boot against a half-applied change.

## What is here

| Probe | Question | Findings |
|---|---|---|
| `p_smoke.py` | Does the environment do what the campaign assumes? | — (gate) |
| `p_flow_graph.py` | Conditioned vs raw surface; order-1 link population | KPA-38, KPA-48, FLG-18, FLG-19 |
| `p_keypoints.py` | Which guard refuses each valley, and how far guides drift | KPA-39/40/41/42/43/44/46 |
| `p_impoundment.py` | Transect length vs map-space length, per ranked site | IMP-01 |
| `p_controllers.py` | Which panel knobs change no output at all | SWL-22, CTL-01, KPA-48, MHL-01, IMP-02 |
