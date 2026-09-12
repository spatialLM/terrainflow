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
| `p_invariance.py` | Z+600 m and mirror — two exact oracles; and `routing='d8'` | Step C: FLA-26, KPA-53 |
| `p_battery.py` | Signed witnesses, dimensional identities, conservation | Step D: KPA-54, TIX-02 |
| `p_coverage.py` | What neither suite calls, and whether tooltips match their constants | Step E — §4 rows |
| `p_crosscheck.py` | Four quantities computed twice; mask routing vs pointer routing | Step F: KPA-52 |
| `p_topographic_valleys.py` | Can a keypoint be found with no routing at all? | KPA-52 (§9.7) |
| `p_perf_gate.py` | Is an optimisation worth its 10%? The harness the four below use | M-6 |
| `p_gate_reads.py` | What does reading a whole band actually cost? | M-10, Q-11, G-9 |
| `p_gate_burn.py` | `_cut_spillway` and `overtopping_spill`, against the burn and a Verify | M-12, R-5 |
| `p_gate_ui.py` | Five Batch 4 items against the UI operations they sit in | M-11, Q-11, G-9, G-8, M-10 |
| `p_gate_m10.py` | M-10's *real* denominator, and how many reads a draw makes | M-10 |

`p_coverage.py` is the one probe that needs **no QGIS at all** — it reads source. Plain
`python p_coverage.py` is quicker and does the same thing.

The `p_gate_*` probes load the owner's real `.tfd` — 35 features on 1139x1016 — because
the committed fixture cannot price performance work: it is a 400x400 clip with two ponds
and no spillways at all, so `_cut_spillway` is never even reached on it. Their verdicts,
before and after, are in `evidence/p_gate_batch4.json`. `p_gate_reads.py` is the cheap
one — it opens no project and needs no design, and it exists because if a whole-band read
had turned out to cost 2 ms, three of the items would have been settled in a minute.

Steps C, D and F also left five permanent assertions behind, in
`checks_fixture_regression.py`. They pin *properties* rather than measurements — a
relation that holds on any terrain at any elevation — so unlike the recorded constants
above them they never need re-recording, and a failure is always a defect.

## Do not write line numbers down in a probe

`p_keypoints.py` carried a hand-written map of the five `return None` sites in
`keypoint_on_path`. The ridgeline fix (`ac2966b`) moved every one of them down 20 lines,
and the probe went on counting correctly while attributing all 126 refusals to
"UNKNOWN LINE" — which silently inverted the two figures derived from that map, so the
evidence file was wrong about the finding the probe exists to support. `guard_lines()`
now reads the positions off the AST on every run. Anything a probe asserts about source
should be derived from source. (Since KPA-52 closed the guards live in
`keypoint_on_path_with_reason` and return `None, <reason>`; `guard_lines()` counts those
too, and the probe checks the traced line against the returned reason.)
