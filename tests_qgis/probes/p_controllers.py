"""p_controllers — which knobs and arguments change nothing at all.

Step A1 of the defect-documentation campaign. Supports **KPA-48, SWL-22, CTL-01, MHL-01,
IMP-02**, and mechanises the class all five belong to.

Every wiring finding in this campaign was found by a human reading a call site. That is
the one defect class a machine can look for cheaply, and this probe does it two ways.

**Targeted.** Each claim is reproduced directly, because a specific measurement beats a
hash:

* **KPA-48** — `contour.py:1390` constructs `YeomansKeylineAnalysis(dem_path,
  acc_path=acc_path)`, but `_ensure_flow_data` gates on `self._fdir_path and
  self._acc_path` (`:1117`). With only one of the two supplied, the `and` is False and a
  full pysheds run is recomputed. Measured as wall time and as an array comparison
  against the accumulation raster that was handed in and ignored.
* **SWL-22** — `suggest_spacing` calls `capacity_per_metre` without
  `infiltration_mm_hr` (`contour.py:175-180`) while `find_swale_segments` passes a real
  soil rate (`:806`). Swept over soil and duration so the size of the omission is a table,
  not an adjective. Omitting infiltration *lowers* capacity, which *tightens* the advised
  interval, which **over**-sizes — direction `O`, not `U`.
* **CTL-01** — the same function's docstring says it reads slope "over the usable area";
  the call passes no mask though `slope_statistics` takes one (`terrain_indices.py:279`).
  Measured as the difference between masked and unmasked quartiles on the fixture.
* **MHL-01** — `haul_regions` computes `block` from `block_m` and hands it to
  `_regions_from`, which never reads it. Swept over a 2,000x range and hashed.
* **IMP-02** — `DEFAULT_MAX_WALL_M`'s docstring says "either side" and `transect_cells`
  searches +/-200 m, but the refusal compares the **total** run. Reproduced as two numbers.

**The sweep.** For each numeric panel input, set it to 0.1x / 1x / 10x, run the keyline
and keypoint tools, and hash `_state`. Bit-identical output across a 100x range flags an
inert knob. Serial, because `PluginHarness` clears `QgsProject.instance()` on entry — two
at once would tear down each other's project.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_controllers.py
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_controllers.py --sweep
"""

import hashlib
import json
import re
import sys
import time

import _probe

import numpy as np
import rasterio

#: Knobs the analysis tier under review actually reads. The full `INPUT_FIELDS` table
#: includes report and storm fields whose tools this probe does not run; sweeping those
#: would report "inert" about a tool that was never invoked, which is worse than silence.
SWEEP_FIELDS = (
    "keypoint_count",
    "keyline_runs",
    "keyline_spacing_m",
    "keyline_max_grade_n",
    "keyline_max_valleys",
    "stream_threshold_ha",
    "min_catchment_ha",
    "max_slope_deg",
    "min_contour_length_m",
)


# ------------------------------------------------------------------- targeted


def stage_kpa48_inert_acc_path(ev, dem):
    """Is the accumulation raster handed to YeomansKeylineAnalysis ever read?"""
    with ev.stage("kpa48_acc_path_is_inert") as rec:
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis
        from terrainflow_assessment.modules.keypoint_analysis import (
            YeomansKeylineAnalysis,
        )

        work = _probe.workdir("controllers")
        fa = FlowAnalysis()
        fa.load_dem(str(dem))
        baseline = fa.run(crest_split=True)
        acc_path = work / "baseline_acc.tif"
        fa.save_result(np.asarray(baseline["flow_accumulation"], dtype="float64"),
                       str(acc_path), band_description="accumulation",
                       dtype="float64", nodata=None)
        rec["acc_raster_written"] = str(acc_path)
        rec["baseline_was_crest_split"] = True

        with rasterio.open(str(acc_path)) as src:
            supplied = src.read(1).astype("float64")

        # Exactly what contour.py:1390 does: acc_path only, no fdir_path.
        ya = YeomansKeylineAnalysis(str(dem), acc_path=str(acc_path))
        t0 = time.perf_counter()
        _fdir, acc_used = ya._ensure_flow_data()
        rec["wall_s_with_acc_path"] = round(time.perf_counter() - t0, 3)

        acc_used = np.asarray(acc_used, dtype="float64")
        rec["supplied_equals_used"] = bool(
            acc_used.shape == supplied.shape
            and np.allclose(acc_used, supplied, equal_nan=True))
        finite = np.isfinite(acc_used) & np.isfinite(supplied)
        rec["max_abs_difference"] = float(
            np.abs(acc_used[finite] - supplied[finite]).max()) if finite.any() else None
        rec["supplied_max"] = float(np.nanmax(supplied))
        rec["used_max"] = float(np.nanmax(acc_used))

        # And with both paths, which is the branch the gate was written for.
        fdir_path = work / "baseline_fdir.tif"
        fa.save_result(np.asarray(baseline["flow_direction"], dtype="float64"),
                       str(fdir_path), band_description="fdir",
                       dtype="float64", nodata=None)
        ya_both = YeomansKeylineAnalysis(
            str(dem), fdir_path=str(fdir_path), acc_path=str(acc_path))
        t0 = time.perf_counter()
        _f2, acc2 = ya_both._ensure_flow_data()
        rec["wall_s_with_both_paths"] = round(time.perf_counter() - t0, 3)
        rec["both_paths_reads_supplied"] = bool(
            np.allclose(np.asarray(acc2, dtype="float64"), supplied, equal_nan=True))

        ev.note(
            "KPA-48, measured: passing acc_path alone (which is what contour.py:1390 "
            f"does) takes {rec['wall_s_with_acc_path']} s and returns an accumulation "
            f"field that is {'the same as' if rec['supplied_equals_used'] else 'NOT'} "
            f"the raster supplied (max difference "
            f"{rec['max_abs_difference']:.1f} cells; supplied max "
            f"{rec['supplied_max']:.0f}, recomputed max {rec['used_max']:.0f}). "
            f"Passing both paths takes {rec['wall_s_with_both_paths']} s and does read "
            f"the supplied raster ({rec['both_paths_reads_supplied']}). The baseline "
            "accumulation is crest-split and the recompute is not, so the keyline runs "
            "on a pond-uncorrected field.")


def stage_swl22_infiltration(ev):
    """The size of the omitted argument, as a table."""
    with ev.stage("swl22_omitted_infiltration") as rec:
        from terrainflow_assessment.modules.swale_design import (
            capacity_per_metre,
            get_infiltration_rate,
        )

        depth, width, side = 0.6, 2.0, 1.0
        rec["section"] = {"depth_m": depth, "width_m": width, "side_slope": side}
        rec["rows"] = []
        for soil in ("Sand", "Loam", "Clay"):
            rate = get_infiltration_rate(soil)
            for hours in (1.0, 6.0, 24.0):
                without = capacity_per_metre(
                    depth, width, side_slope=side, duration_hr=hours)
                with_it = capacity_per_metre(
                    depth, width, side_slope=side,
                    infiltration_mm_hr=rate, duration_hr=hours)
                rec["rows"].append({
                    "soil": soil,
                    "infiltration_mm_hr": rate,
                    "duration_hr": hours,
                    "capacity_as_advised_m3_per_m": without,
                    "capacity_with_infiltration_m3_per_m": with_it,
                    "shortfall_fraction": (
                        (with_it - without) / without if without else None),
                })
        worst = max(rec["rows"], key=lambda r: r["shortfall_fraction"] or 0.0)
        rec["worst"] = worst
        ev.note(
            "SWL-22, measured: suggest_spacing omits infiltration_mm_hr, so the capacity "
            "it hands capture_spacing is short by up to "
            f"{worst['shortfall_fraction']:.1%} ({worst['soil']} / "
            f"{worst['duration_hr']:.0f} h). A lower capacity gives a TIGHTER advised "
            "interval, so the direction is O (over-sizes), not U.")


def stage_ctl01_unmasked_slope(ev, dem):
    """Site-wide quartiles against usable-area quartiles, on the same raster."""
    with ev.stage("ctl01_unmasked_slope") as rec:
        from terrainflow_assessment.modules.dem_loader import slope_degrees
        from terrainflow_assessment.modules.terrain_indices import slope_statistics

        with rasterio.open(str(dem)) as src:
            z = src.read(1).astype("float64")
            nodata = src.nodata
            cw, ch = abs(src.transform.a), abs(src.transform.e)
        if nodata is not None:
            z[z == nodata] = np.nan
        slope = slope_degrees(z, cw, ch)

        site_wide = slope_statistics(slope)
        # A plausible "usable area": the middle half of the raster, standing in for a
        # property boundary. The point is the size of the difference, not this polygon.
        mask = np.zeros(slope.shape, dtype=bool)
        r0, c0 = slope.shape[0] // 4, slope.shape[1] // 4
        mask[r0:3 * r0, c0:3 * c0] = True
        masked = slope_statistics(slope, mask=mask)

        rec["site_wide"] = site_wide
        rec["masked_middle_half"] = masked
        rec["mask_cells"] = int(mask.sum())
        if site_wide and masked:
            rec["p50_difference_deg"] = masked["p50"] - site_wide["p50"]
            rec["p50_relative_change"] = (
                (masked["p50"] - site_wide["p50"]) / site_wide["p50"]
                if site_wide["p50"] else None)
            ev.note(
                "CTL-01, measured: suggest_spacing calls slope_statistics with no mask "
                "(contour.py:168) though the parameter exists and is unit-tested. On "
                f"this fixture the site-wide median slope is {site_wide['p50']:.2f} deg "
                f"and the middle-half median is {masked['p50']:.2f} deg, a change of "
                f"{rec['p50_relative_change']:.1%}. The docstring at contour.py:137 "
                "says the advice is read 'over the usable area'; it is read site-wide.")


def stage_mhl01_inert_block(ev):
    """`block_m` over a 2,000x range, hashed."""
    with ev.stage("mhl01_block_m_is_inert") as rec:
        from terrainflow_assessment.modules.mass_haul import haul_regions

        rng = np.random.default_rng(7)
        original = rng.normal(50.0, 2.0, (120, 120))
        burned = original.copy()
        burned[20:40, 20:40] -= 1.5      # a cut
        burned[70:95, 60:90] += 2.0      # a fill

        from rasterio.transform import from_origin
        transform = from_origin(0.0, 1200.0, 2.0, 2.0)

        rec["block_m_values"] = []
        for block_m in (0.01, 1.0, 10.0, 100.0, 1000.0, 20.0):
            cuts, fills = haul_regions(original, burned, transform, 4.0,
                                       block_m=block_m)
            payload = json.dumps([cuts, fills], sort_keys=True, default=float)
            rec["block_m_values"].append({
                "block_m": block_m,
                "cuts": len(cuts),
                "fills": len(fills),
                "sha256": hashlib.sha256(payload.encode()).hexdigest()[:16],
            })
        hashes = {v["sha256"] for v in rec["block_m_values"]}
        rec["distinct_outputs"] = len(hashes)
        rec["range_swept"] = 1000.0 / 0.01
        ev.note(
            "MHL-01, measured: haul_regions over block_m 0.01 -> 1000 m (a 100,000x "
            f"range) produces {len(hashes)} distinct output(s). `block` is computed at "
            "mass_haul.py:124 and passed to _regions_from, which never reads it. "
            "earthworks.py, the only caller, never passes block_m at all, so the "
            "parameter is inert in two independent ways.")


def stage_imp02_wall_limit(ev):
    """"Either side" against a total-run comparison."""
    with ev.stage("imp02_wall_limit") as rec:
        from terrainflow_assessment.modules.impoundment_sites import (
            DEFAULT_MAX_WALL_M,
            transect_cells,
        )

        cells, step = transect_cells((400, 400), 200, 200, (0.0, 1.0), 1.0, 1.0,
                                     DEFAULT_MAX_WALL_M)
        rec["default_max_wall_m"] = DEFAULT_MAX_WALL_M
        rec["transect_cells_returned"] = len(cells)
        rec["transect_reach_each_side_m"] = DEFAULT_MAX_WALL_M
        rec["transect_total_span_m"] = len(cells) * step
        rec["refusal_compares"] = "len(run) * step > max_wall_m  (impoundment_sites.py:335)"
        rec["longest_wall_the_search_can_offer_m"] = len(cells) * step
        rec["longest_wall_the_refusal_allows_m"] = DEFAULT_MAX_WALL_M
        ev.note(
            "IMP-02, measured: transect_cells searches +/-"
            f"{DEFAULT_MAX_WALL_M:.0f} m and returns a crest line spanning "
            f"{rec['transect_total_span_m']:.0f} m, while the refusal compares the "
            f"TOTAL run against {DEFAULT_MAX_WALL_M:.0f} m. Code and message agree with "
            "each other; the docstring's 'either side' describes a limit twice as "
            "permissive as the one enforced.")


# ---------------------------------------------------------------------- sweep


#: Three things are new on every run whatever the inputs were, and each of them would
#: make the sweep report "changed" for every knob:
#:
#: * ``<... object at 0x00000224...>`` — a repr carrying a memory address.
#: * ``Keyline_Design_520b08ef_82af_...`` — a QGIS layer id, which is name + a fresh UUID.
#: * ``F:/Temp/tfa_cbx_yrn5/slope.tif`` — the per-run output directory.
#:
#: Scrubbed rather than dropped, because the *shape* of each still carries signal: a run
#: that produced no keyline layer at all differs from one that produced a differently-named
#: one, and dropping the field entirely would hide that.
_SCRUB = (
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"tfa_[a-z0-9_]{6,}"), "tfa_TMP"),
    (re.compile(r"_[0-9a-f]{8}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{4}_[0-9a-f]{12}"),
     "_UUID"),
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"),
     "UUID"),
)


def scrub(text):
    for pattern, repl in _SCRUB:
        text = pattern.sub(repl, text)
    return text


def state_fingerprint(state):
    """A stable hash of everything the analysis tools write to `_state`.

    Arrays are hashed by bytes, QGIS geometry by WKT, and anything else by its repr with
    the run-scoped parts scrubbed. The control run in `stage_knob_sweep` is what proves
    this worked; without it the sweep can only ever say "changed".
    """
    def reduce(value):
        if isinstance(value, str):
            return scrub(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {scrub(str(k)): reduce(v)
                    for k, v in sorted(value.items(), key=str)}
        if isinstance(value, (list, tuple)):
            return [reduce(v) for v in value]
        if isinstance(value, np.ndarray):
            return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        wkt = getattr(value, "asWkt", None)
        if callable(wkt):
            try:
                return scrub(wkt())
            except Exception:                               # noqa: BLE001
                pass
        return scrub(repr(value))

    interesting = {
        name: reduce(getattr(state, name, None))
        for name in sorted(vars(state))
        if not name.startswith("__") and not _run_scoped(name)
    }
    # `_state` is not where most of the keyline output goes. The guides are features in a
    # map layer, so a run that drew three of them and a run that drew sixty have the same
    # `_state`. Without the layer census below, this function would report
    # `keyline_runs` and `keyline_spacing_m` as inert — which is a fact about the
    # fingerprint, not about the code.
    interesting["_layers"] = layer_census()
    blob = json.dumps(interesting, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16], len(blob)


def layer_census():
    """Name and feature count for every layer in the project, ids scrubbed."""
    from qgis.core import QgsProject

    out = {}
    for layer in QgsProject.instance().mapLayers().values():
        name = scrub(layer.name())
        count = None
        counter = getattr(layer, "featureCount", None)
        if callable(counter):
            try:
                count = int(counter())
            except Exception:                                   # noqa: BLE001
                count = "unavailable"
        extent = layer.extent()
        out[name] = {
            "features": count,
            "extent": [round(extent.xMinimum(), 3), round(extent.yMinimum(), 3),
                       round(extent.xMaximum(), 3), round(extent.yMaximum(), 3)],
        }
    return out


#: Fields whose value is new on every run whatever the inputs were — QGIS layer ids and
#: the paths of per-run temporary rasters. Leaving them in makes every run differ from
#: every other, which turns the sweep into a machine that can only ever say "changed".
_RUN_SCOPED_SUFFIXES = ("_layer_id", "_layer_ids", "_path", "_paths", "_dir")


def _run_scoped(name):
    return name.endswith(_RUN_SCOPED_SUFFIXES)


def run_one(dem_path, overrides):
    """One full harness run with *overrides* applied, returning a state fingerprint."""
    _harness = _probe.start_qgis()
    _harness.make_workers_synchronous()

    with _harness.PluginHarness(str(dem_path)) as h:
        inputs = h.panel.collect_inputs()
        inputs.update(overrides)
        h.panel.apply_inputs(inputs)
        landed = {k: h.panel.collect_inputs().get(k) for k in overrides}

        h.run_baseline()
        h.panel.run_keypoint_analysis_requested.emit()
        h.panel.run_keyline_requested.emit()

        digest, size = state_fingerprint(h.state)
        criticals = len(h.bar.criticals)
    return {"fingerprint": digest, "state_bytes": size,
            "landed": landed, "criticals": criticals}


def stage_knob_sweep(ev, dem):
    with ev.stage("knob_sweep") as rec:
        rec["fields"] = list(SWEEP_FIELDS)
        rec["results"] = {}

        # Control first. A fingerprint that does not repeat across two identical runs
        # measures nondeterminism — a temp path, a timestamp — and every "this knob
        # changed the output" below would be noise. Without this the sweep's conclusion
        # is unsound in the direction that flatters it.
        base = run_one(dem, {})
        control = run_one(dem, {})
        rec["baseline_fingerprint"] = base["fingerprint"]
        rec["control_fingerprint"] = control["fingerprint"]
        rec["fingerprint_is_deterministic"] = (
            base["fingerprint"] == control["fingerprint"])
        print(f"    baseline {base['fingerprint']}  control {control['fingerprint']}  "
              f"deterministic={rec['fingerprint_is_deterministic']}", flush=True)
        if not rec["fingerprint_is_deterministic"]:
            ev.note(
                "The state fingerprint does NOT repeat across two identical runs "
                f"({base['fingerprint']} vs {control['fingerprint']}), so it carries "
                "something run-scoped (a temp path or a layer id). Every "
                "'this knob changed the state' result below is therefore uninformative, "
                "and the sweep can only report a knob INERT, never report one live. "
                "Treat the inert list as sound and the rest as not measured.")

        for name in SWEEP_FIELDS:
            entry = {"runs": []}
            for factor in (0.1, 10.0):
                _harness = _probe.start_qgis()
                with _harness.PluginHarness(str(dem)) as probe_h:
                    default = probe_h.panel.collect_inputs()[name]
                value = (type(default)(max(1, round(default * factor)))
                         if isinstance(default, int)
                         else type(default)(default * factor))
                got = run_one(dem, {name: value})
                entry["runs"].append({
                    "factor": factor,
                    "requested": value,
                    "landed": got["landed"].get(name),
                    "fingerprint": got["fingerprint"],
                    "criticals": got["criticals"],
                })
            fps = {r["fingerprint"] for r in entry["runs"]} | {base["fingerprint"]}
            landed = {r["landed"] for r in entry["runs"]}
            entry["distinct_fingerprints"] = len(fps)
            entry["value_actually_changed"] = len(landed) > 1
            entry["inert"] = len(fps) == 1 and entry["value_actually_changed"]
            # Only meaningful when the control repeated: otherwise "the state changed"
            # is the null result, not evidence the knob reached anything.
            entry["live"] = (
                rec["fingerprint_is_deterministic"]
                and len(fps) > 1 and entry["value_actually_changed"])
            rec["results"][name] = entry
            print(f"    {name:24s} distinct={entry['distinct_fingerprints']} "
                  f"landed={sorted(landed)} "
                  f"{'INERT' if entry['inert'] else ''}", flush=True)

        inert = [k for k, v in rec["results"].items() if v["inert"]]
        live = [k for k, v in rec["results"].items() if v["live"]]
        clamped = [k for k, v in rec["results"].items()
                   if not v["value_actually_changed"]]
        rec["inert_knobs"] = inert
        rec["live_knobs"] = live
        rec["knobs_the_widget_clamped"] = clamped
        ev.note(
            f"Knob sweep over {len(SWEEP_FIELDS)} inputs at 0.1x and 10x, running "
            "baseline + find-keypoints + keyline each time. "
            + (f"Inert (the value changed and the state did not): {', '.join(inert)}. "
               if inert else "No knob was inert. ")
            + (f"Demonstrably live: {', '.join(live)}. " if live
               else "No knob could be shown live, because the control run did not "
                    "reproduce its own fingerprint. ")
            + (f"Clamped by their own widget so nothing was tested: "
               f"{', '.join(clamped)}. " if clamped else "")
            + "A knob that is inert here is either unwired or reaching a tool whose "
              "output it cannot move on this fixture — the register must say which.")


def main():
    dem = _probe.fixture_path()
    _probe.start_qgis()
    _probe.banner("p_controllers — inert arguments and inert knobs", dem)

    ev = _probe.Evidence(
        "p_controllers",
        findings=["KPA-48", "SWL-22", "CTL-01", "MHL-01", "IMP-02"], dem=dem)
    ev["dem_stats"] = _probe.dem_stats(dem)

    stage_kpa48_inert_acc_path(ev, dem)
    stage_swl22_infiltration(ev)
    stage_ctl01_unmasked_slope(ev, dem)
    stage_mhl01_inert_block(ev)
    stage_imp02_wall_limit(ev)

    if "--sweep" in sys.argv[1:]:
        stage_knob_sweep(ev, dem)
    else:
        ev.note("knob sweep not run (pass --sweep; it is ~20 harness runs)")

    ev.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
