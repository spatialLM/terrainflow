"""p_smoke — does the environment do what the rest of the campaign assumes?

Step A of the defect-documentation campaign. Nothing here is a finding. Every stage
checks one assumption the later probes are built on, so that a wrong number in
`p_keypoints` is a fact about the code rather than a fact about this machine.

The assumptions, and why each is worth a stage:

1. `FlowAnalysis.run(crest_split=False)` completes on the fixture and returns a
   `conditioned_dem` key. KPA-48's fix note says the conditioned surface is "one dict key
   away" from the caller that recomputes it; if the key is absent that note is wrong
   before any measurement starts.
2. `save_result(..., nodata=-9999)` round-trips with its nodata tag, and `d8_from_dem`
   re-reading the file finds the same sinks as the in-memory array. The float32 default
   quantises a resolved flat away above ~600 m elevation (`save_result`'s own docstring),
   which is the failure this campaign has to be able to distinguish from a real sink.
3. `gdal_contour` is resolvable **from inside the interpreter**, not merely present on
   disk — `F:\\bin\\python-qgis-ltr.bat` is what puts `F:\\bin` on PATH.
4. `PluginHarness` + `make_workers_synchronous()` gets as far as a populated
   `state.baseline_result`. `__enter__` does not call it; `run_all.py:127` does, so a
   probe run outside the runner has to do it itself.
5. `build_synthetic_dem(rough=True, pits=6, voids=2)` against the seed-specific comment
   at `_harness.py:193` ("18 sinks, 180 nodata cells"). That comment is asserted nowhere.
   A mismatch here is a stale comment, **not** an environment failure, so it is logged.

The pure-suite count and wall time are recorded by `--pure-suite`, which shells out to the
standalone interpreter. CLAUDE.md says ~2,590 tests in ~60 s and the project memory says
~2,890 in ~35 s; they cannot both be current.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_smoke.py
"""

import shutil
import subprocess
import sys
import time

import _probe

import numpy as np
import rasterio


def stage_flow_analysis(ev, dem):
    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

    with ev.stage("flow_analysis_run") as rec:
        fa = FlowAnalysis()
        fa.load_dem(str(dem))
        t0 = time.perf_counter()
        result = fa.run(crest_split=False)
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        rec["keys"] = sorted(result.keys())
        rec["has_conditioned_dem"] = "conditioned_dem" in result
        rec["routing"] = fa.routing
        rec["flat_eps"] = None if fa.flat_eps is None else float(fa.flat_eps)
        rec["flat_inversions"] = (
            None if fa.flat_inversions is None else int(fa.flat_inversions))

        if not rec["has_conditioned_dem"]:
            ev.note(
                "run(crest_split=False) returns no 'conditioned_dem' key — KPA-48's "
                f"fix note is unfounded as written. Keys: {rec['keys']}")
        else:
            cond = np.asarray(result["conditioned_dem"], dtype="float64")
            rec["conditioned_dtype"] = str(np.asarray(result["conditioned_dem"]).dtype)
            finite = np.isfinite(cond)
            rec["conditioned_z_min"] = float(cond[finite].min())
            rec["conditioned_z_max"] = float(cond[finite].max())
            ev.data["_conditioned"] = None  # not serialised; the array stays in p_flow_graph
        return result if rec["has_conditioned_dem"] else None


def stage_save_result_roundtrip(ev, dem, result):
    """float32 vs float64 for the conditioned surface, measured rather than argued.

    The claim in `save_result`'s docstring is that float32 quantises the resolved-flat
    gradient away and `d8_from_dem` then reads a genuine flat, turning every cell of it
    into a sink. On the Quail Island clip (low elevation) the docstring predicts the fault
    does **not** appear; recording both dtypes here is what makes that a measurement.
    """
    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
    from terrainflow_assessment.modules.flow_graph import d8_from_dem

    with ev.stage("save_result_roundtrip") as rec:
        if result is None:
            raise RuntimeError("no run() result to save — earlier stage did not complete")

        fa = FlowAnalysis()
        fa.load_dem(str(dem))
        work = _probe.workdir("smoke")
        cond = np.asarray(result["conditioned_dem"], dtype="float64")

        _n, sink_mem = d8_from_dem(cond, fa.transform.a, abs(fa.transform.e),
                                   nodata=-9999.0)
        rec["sinks_in_memory"] = int(sink_mem.sum())

        for dtype in ("float32", "float64"):
            out = work / f"conditioned_{dtype}.tif"
            fa.save_result(cond, str(out), band_description="conditioned",
                           dtype=dtype, nodata=-9999.0)
            with rasterio.open(str(out)) as src:
                back = src.read(1).astype("float64")
                tag = src.nodata
            _n2, sink_disk = d8_from_dem(back, fa.transform.a, abs(fa.transform.e),
                                         nodata=-9999.0)
            rec[dtype] = {
                "nodata_tag_present": tag is not None,
                "nodata_tag": None if tag is None else float(tag),
                "sinks_after_roundtrip": int(sink_disk.sum()),
                "matches_in_memory": int(sink_disk.sum()) == int(sink_mem.sum()),
                "max_abs_delta_m": float(
                    np.nanmax(np.abs(back - cond)[np.isfinite(cond) & np.isfinite(back)])),
            }

        if not rec["float64"]["matches_in_memory"]:
            ev.note(
                "float64 round-trip does NOT preserve the sink count "
                f"({rec['float64']['sinks_after_roundtrip']} on disk vs "
                f"{rec['sinks_in_memory']} in memory) — every later sink figure that "
                "passes through a file is suspect.")
        if rec["float32"]["matches_in_memory"]:
            ev.note(
                "float32 round-trip preserved the sink count on this fixture "
                f"(z_max {rec.get('z_max', 'see dem stats')}) — save_result's warning is "
                "about high-elevation sites and does not bite here. It is still the "
                "wrong dtype for the conditioned surface.")
        else:
            ev.note(
                "float32 round-trip changed the sink count "
                f"({rec['float32']['sinks_after_roundtrip']} vs {rec['sinks_in_memory']}) "
                "— save_result's docstring reproduced on this fixture.")


def stage_gdal_contour(ev):
    with ev.stage("gdal_contour_on_path") as rec:
        found = shutil.which("gdal_contour") or shutil.which("gdal_contour.exe")
        rec["resolved"] = found
        rec["on_path"] = found is not None
        if not found:
            ev.note(
                "gdal_contour is NOT resolvable from inside this interpreter — the "
                "contour tier cannot be probed from here. Launch via "
                "F:\\bin\\python-qgis-ltr.bat, which puts F:\\bin on PATH.")


def stage_harness(ev, dem):
    with ev.stage("plugin_harness_baseline") as rec:
        _harness = _probe.start_qgis()
        _harness.make_workers_synchronous()
        t0 = time.perf_counter()
        with _harness.PluginHarness(str(dem)) as h:
            baseline = h.run_baseline()
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            rec["baseline_is_none"] = baseline is None
            rec["state_baseline_result_set"] = h.state.baseline_result is not None
            rec["baseline_keys"] = (
                sorted(baseline.keys()) if isinstance(baseline, dict) else None)
            rec["layer_names"] = h.layer_names()
            rec["message_bar"] = {
                "criticals": len(h.bar.criticals),
                "warnings": len(h.bar.warnings),
            }
            try:
                h.assert_no_errors("smoke baseline")
                rec["assert_no_errors"] = "clean"
            except AssertionError as exc:
                rec["assert_no_errors"] = str(exc)
                ev.note(f"baseline on the fixture is not error-free: {exc}")

        if not rec["state_baseline_result_set"]:
            ev.note(
                "run_baseline() left state.baseline_result unset — every probe that "
                "drives the harness (p_impoundment, p_controllers) is blocked.")


def stage_synthetic_dem(ev):
    """The `_harness.py:193` comment, checked. A mismatch is a stale comment."""
    from terrainflow_assessment.modules.flow_graph import d8_from_dem

    with ev.stage("synthetic_nasty_dem") as rec:
        _harness = _probe.start_qgis()
        work = _probe.workdir("smoke")
        path = _harness.build_synthetic_dem(
            work / "nasty.tif", rough=True, pits=6, voids=2)
        stats = _probe.dem_stats(path)
        rec["dem"] = stats

        with rasterio.open(str(path)) as src:
            a = src.read(1).astype("float64")
            nodata = src.nodata
        z = np.where(a == nodata, np.nan, a) if nodata is not None else a
        # Both returns of `d8_from_dem` are **flat** (`flow_graph.py:146-147`), so the
        # finite mask has to be ravelled to meet them. A self-pointing nodata cell counts
        # as a sink by construction; "18 sinks" in the comment can only mean the real ones.
        _n, is_sink = d8_from_dem(z, stats["cell_w_m"], stats["cell_h_m"])
        finite_flat = np.isfinite(z).ravel()
        nodata_cells = int(np.count_nonzero(~finite_flat))
        real_sinks = int(np.count_nonzero(is_sink & finite_flat))

        rec["nodata_cells"] = nodata_cells
        rec["sinks_excluding_nodata"] = real_sinks
        rec["sinks_including_nodata"] = int(is_sink.sum())
        rec["comment_claims"] = {"sinks": 18, "nodata_cells": 180,
                                 "at": "_harness.py:193"}
        rec["matches_comment"] = (real_sinks == 18 and nodata_cells == 180)

        if not rec["matches_comment"]:
            ev.note(
                f"_harness.py:193 says '18 sinks, 180 nodata cells'; this seed gives "
                f"{real_sinks} sinks and {nodata_cells} nodata cells. The comment is "
                "asserted nowhere, so this is a stale comment, not an environment "
                "failure — but the register must not quote the comment as a measurement.")

        # Also record the *default* surface, which is what 259 of the checks run over.
        smooth = _harness.build_synthetic_dem(work / "smooth.tif")
        s_stats = _probe.dem_stats(smooth)
        with rasterio.open(str(smooth)) as src:
            sa = src.read(1).astype("float64")
        _n2, s_sink = d8_from_dem(sa, s_stats["cell_w_m"], s_stats["cell_h_m"])
        rec["default_surface"] = {
            "dem": s_stats,
            "sinks": int(s_sink.sum()),
        }


def stage_pure_suite(ev):
    """Count and time the pure suite once. Opt-in: it costs ~35-60 s."""
    with ev.stage("pure_suite") as rec:
        python = (r"C:\Users\Liam\AppData\Local\Programs\Python\Python312"
                  r"\python.exe")
        rec["interpreter"] = python
        t0 = time.perf_counter()
        proc = subprocess.run(
            [python, "-m", "pytest", "tests/", "-q"],
            cwd=str(_probe.REPO), capture_output=True, text=True, timeout=900)
        rec["wall_s"] = round(time.perf_counter() - t0, 1)
        rec["returncode"] = proc.returncode
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-3:]
        rec["tail"] = tail
        ev.note("pure suite: " + " | ".join(tail))
        ev.note(
            f"CLAUDE.md says ~2,590 tests in ~60 s; project memory says ~2,890 in "
            f"~35 s. Measured {rec['wall_s']} s on this machine — see 'tail'.")


def main():
    dem = _probe.fixture_path()
    _probe.banner("p_smoke — environment gate for the analysis-tier defect sweep", dem)

    ev = _probe.Evidence("p_smoke", findings=[], dem=dem)
    ev["dem_stats"] = _probe.dem_stats(dem)
    print(f"  cell {ev['dem_stats']['cell_w_m']} m; default stream threshold "
          f"{ev['dem_stats']['default_stream_threshold_cells']} cells = "
          f"{ev['dem_stats']['default_stream_threshold_ha']:.2f} ha", flush=True)

    result = stage_flow_analysis(ev, dem)
    stage_save_result_roundtrip(ev, dem, result)
    stage_gdal_contour(ev)
    stage_harness(ev, dem)
    stage_synthetic_dem(ev)
    if "--pure-suite" in sys.argv[1:]:
        stage_pure_suite(ev)
    else:
        ev.note("pure suite not measured this run (pass --pure-suite)")

    ev.data.pop("_conditioned", None)
    ev.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
