"""p_gate_burn — price M-12 and R-5 against the operations they sit in.

Both items are inside the earthworks burn on a real design, so both need the real
design to be priced: the committed fixture is a 400x400 clip with no spillways at
all, and `_cut_spillway` is never reached on it.

    M-12  `_cut_spillway` rasterises the crest bar twice (`_bar_cells` at :2607 and
          again inside `_downhill_step`) and allocates two full-grid side masks.
          Enclosing operation: the burn (`burn_earthworks`) and the per-feature
          measurement pass (`feature_storage` x N), which is where `_isolated_burn`
          re-cuts every notch.

    R-5   `overtopping_spill` re-labels and re-dilates the same pool once per
          barrier. Enclosing operation: a Verify run's report build.

Gate rule 1: the denominator is the enclosing operation. If the *whole* of
`_cut_spillway` is under 10% of the burn, no rewrite of it can clear the bar and
the item is settled without being coded.

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_gate_burn.py
"""
import faulthandler
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

faulthandler.enable()

REAL_TFD = (r"F:\Terrain Flow Design\QGIS Working Files"
            r"\Quail_Island 03.09.2026.tfd")


def _counted(obj, name):
    """Count + time calls to a bound attribute of *obj*'s class. Returns (log, undo)."""
    real = getattr(obj, name)
    log = {"n": 0, "secs": 0.0}

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return real(*args, **kwargs)
        finally:
            log["n"] += 1
            log["secs"] += time.perf_counter() - t0

    setattr(obj, name, wrapper)
    return log, lambda: setattr(obj, name, real)


def main():
    import _probe

    _probe.start_qgis()

    from _harness import PluginHarness, make_workers_synchronous

    make_workers_synchronous()

    from p_perf_gate import gate, real_dem

    dem = real_dem()
    print(f"DEM {dem}")

    with PluginHarness(dem, load_boundary=False) as h:
        t0 = time.perf_counter()
        h.plugin._design_file._restore_design(REAL_TFD)
        print(f"restore design            {(time.perf_counter() - t0) * 1000:9.1f} ms")

        ews = h.state.earthwork_manager.get_all()
        sited = [e for e in ews
                 if getattr(getattr(e, "spillway", None), "point_wkt", None)]
        print(f"  earthworks {len(ews)}, outflow spillways sited {len(sited)}")
        if not ews:
            raise SystemExit("the design did not restore — nothing to measure")

        t0 = time.perf_counter()
        result = h.run_baseline()
        print(f"baseline                  {(time.perf_counter() - t0) * 1000:9.1f} ms "
              f"(result {result is not None})")

        controller = h.plugin._earthworks
        burner = h.state.burner
        if burner is None:
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            burner = DEMBurner(h.state.dem_path)
            h.state.burner = burner
        print(f"  burner shape {burner.shape}")

        enabled = [e for e in ews if e.enabled]
        sills = controller._spillway_sills(enabled)
        print(f"  sills resolved {len(sills)}")

        # ---------------------------------------------------------- M-12, the burn
        cut_log, undo_cut = _counted(burner, "_cut_spillway")
        bar_log, undo_bar = _counted(burner, "_bar_cells")
        step_log, undo_step = _counted(burner, "_downhill_step")
        try:
            t0 = time.perf_counter()
            modified = burner.burn_earthworks(enabled, sills=sills)
            burn_secs = time.perf_counter() - t0
        finally:
            undo_cut(); undo_bar(); undo_step()

        print("")
        print(f"burn_earthworks           {burn_secs * 1000:9.1f} ms")
        print(f"  _cut_spillway   n={cut_log['n']:4d}  {cut_log['secs'] * 1000:9.1f} ms")
        print(f"  _bar_cells      n={bar_log['n']:4d}  {bar_log['secs'] * 1000:9.1f} ms")
        print(f"  _downhill_step  n={step_log['n']:4d}  {step_log['secs'] * 1000:9.1f} ms")
        gate("M-12 ceiling: all of _cut_spillway removed from the burn",
             cut_log["secs"], burn_secs)

        # --------------------------------------- M-12, the per-feature measurement
        baseline_ponding = controller._cached_baseline_ponding(
            burner.shape, burner.transform)
        cut_log2, undo_cut = _counted(burner, "_cut_spillway")
        bar_log2, undo_bar = _counted(burner, "_bar_cells")
        step_log2, undo_step = _counted(burner, "_downhill_step")
        try:
            t0 = time.perf_counter()
            for ew in enabled:
                try:
                    if ew.type == "dam":
                        if getattr(ew, "spillway", None) is None:
                            continue
                        burner.dam_storage(
                            ew, baseline_ponding=baseline_ponding,
                            key_into_banks=bool(getattr(ew, "key_into_banks", False)))
                    else:
                        burner.feature_storage(
                            ew, baseline_ponding=baseline_ponding)
                except Exception as exc:
                    print(f"    (measure {ew.name} failed: {exc})")
            measure_secs = time.perf_counter() - t0
        finally:
            undo_cut(); undo_bar(); undo_step()

        print("")
        print(f"terrain capacities pass   {measure_secs * 1000:9.1f} ms "
              f"({len(enabled)} features)")
        print(f"  _cut_spillway   n={cut_log2['n']:4d}  "
              f"{cut_log2['secs'] * 1000:9.1f} ms")
        print(f"  _bar_cells      n={bar_log2['n']:4d}  "
              f"{bar_log2['secs'] * 1000:9.1f} ms")
        print(f"  _downhill_step  n={step_log2['n']:4d}  "
              f"{step_log2['secs'] * 1000:9.1f} ms")
        gate("M-12 ceiling: all of _cut_spillway removed from the capacity pass",
             cut_log2["secs"], measure_secs)

        total_cut = cut_log["secs"] + cut_log2["secs"]
        gate("M-12 ceiling: both passes together",
             total_cut, burn_secs + measure_secs)


        # ------------------------------------------------------------------- R-5
        # Priced where it actually runs: `_build_overtopping_layer`, inside the
        # Verify run's completion handler. The barriers, the pools and the ponding
        # surface are then the real ones rather than anything this probe invented.
        from terrainflow_assessment.modules import reporting

        real_spill = reporting.overtopping_spill
        spill_log = {"n": 0, "secs": 0.0, "barriers": 0, "pairs": 0, "pools": 0}

        def counting_spill(ponding, ground, cell_size_m, barriers, **kw):
            t0 = time.perf_counter()
            try:
                return real_spill(ponding, ground, cell_size_m, barriers, **kw)
            finally:
                spill_log["n"] += 1
                spill_log["secs"] += time.perf_counter() - t0
                spill_log["barriers"] = len(barriers)
                pairs = _count_pairs(ponding, barriers, kw.get("min_depth", 0.001))
                spill_log["pairs"] = pairs["pairs"]
                spill_log["pools"] = pairs["pools"]

        reporting.overtopping_spill = counting_spill
        try:
            t0 = time.perf_counter()
            controller.run_with_earthworks()
            verify_secs = time.perf_counter() - t0
        finally:
            reporting.overtopping_spill = real_spill

        print("")
        print(f"Verify run (run_earthworks) {verify_secs * 1000:9.1f} ms")
        print(f"  overtopping_spill n={spill_log['n']:4d}  "
              f"{spill_log['secs'] * 1000:9.1f} ms")
        print(f"  barriers {spill_log['barriers']}, (barrier, pool) pairs "
              f"{spill_log['pairs']}, distinct pools {spill_log['pools']}")
        if spill_log["pairs"]:
            share = 1.0 - spill_log["pools"] / spill_log["pairs"]
            print(f"  memoising by pool would skip {share * 100:.0f}% of the "
                  f"per-pool work")
        gate("R-5 ceiling: all of overtopping_spill removed from a Verify run",
             spill_log["secs"], verify_secs)


def _count_pairs(ponding, barriers, min_depth):
    """How many (barrier, pool) pairs the loop walks, and over how many pools.

    R-5 memoises the per-pool work, so the saving is bounded by the repeats:
    pairs - pools. One pool per barrier and there is nothing to memoise.
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, label

    labels, _n = label(np.asarray(ponding, dtype="float64") >= min_depth)
    pairs, pools = 0, set()
    for _name, crest, _len in barriers:
        crest = np.asarray(crest, dtype=bool)
        if crest.shape != labels.shape or not crest.any():
            continue
        touching = set(np.unique(labels[binary_dilation(crest)])) - {0}
        pairs += len(touching)
        pools |= touching
    return {"pairs": pairs, "pools": len(pools)}


if __name__ == "__main__":
    main()
