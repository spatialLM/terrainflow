"""p_perf_gate — price one optimisation against the operation it sits in.

The remediation plan's Batch 4 rule: *each item is timed before and after; if the
gain is under 10% of the operation it sits in (or under 50 ms per drag frame) the
item is **dropped, not merged**.* This is the harness for that, so the procedure
stops being re-improvised per item — and so the denominator is chosen on purpose
rather than by whatever was convenient to measure.

Four things it exists to get right, each of which cost a wrong answer first:

**1. The denominator is the enclosing operation, not the function.** "This function
is slow" is not the question; "would removing all of it matter" is. `gate()` takes
both and prints the ratio, and it will tell you an item is unwinnable *before* you
implement it: `find_impoundments` is 4.5% of a baseline run, so no version of M-8
could ever clear a 10% bar.

**2. Measure on the real design, not the committed fixture.**
`tests/fixtures/quail_island_catchment.tif` is a 400x400 clip with 2 ponds; the
owner's `.tfd` carries 35 features on 1139x1016 with 27. Ratios are **not**
scale-invariant between them — M-6 measured 5.9% on the clip and 8.0% on the real
design, because `run()` and `d8_from_dem` do not scale together. `real_dem()`
unpacks the `.tfd` to scratch; it is read-only and never writes to the repo.

**3. Count the calls as well as the clock.** A timing says how much; a count says
whether the mechanism is what you think. M-6's register entry said "up to four
times" — counting said exactly four, and counting again after said one.

**4. Check the precondition the optimisation assumes.** Caching is only sound if
the callers share an input. `same_input()` records what each call actually saw, and
it is the difference between an optimisation and a correctness bug: M-6's cache
looked safe on identity until a test showed pysheds returns the *same* Raster
object run to run and conditions it in place.

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_perf_gate.py
"""
import contextlib
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

#: The owner's real design. Outside the repo on purpose — it is a client working
#: file, 3.5 MB, and is unpacked to scratch read-only.
REAL_TFD = (r"F:\Terrain Flow Design\QGIS Working Files"
            r"\Quail_Island 03.09.2026.tfd")

DRAG_BUDGET_MS = 50.0      # the plan's per-frame bar
OPERATION_BAR = 0.10       # the plan's share-of-operation bar


def real_dem(scratch=None):
    """The DEM embedded in the real design, unpacked to scratch. Read-only."""
    scratch = scratch or os.path.join(
        os.environ.get("TEMP", "."), "tfa_perf_gate")
    os.makedirs(scratch, exist_ok=True)
    dem = os.path.join(scratch, "dem.tif")
    if not os.path.exists(dem):
        if not os.path.exists(REAL_TFD):
            raise SystemExit(
                f"the real design is not at {REAL_TFD}. Measure on it, not on the "
                f"400x400 committed fixture — see the module docstring.")
        with zipfile.ZipFile(REAL_TFD) as z:
            z.extractall(scratch)
    return dem


def bench(fn, repeat=3):
    """Best-of-*repeat* seconds. Best, not mean: noise only ever adds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@contextlib.contextmanager
def counted(module, name):
    """Count and time calls to ``module.name``, and record each call's first arg.

    Yields a dict with ``n``, ``secs`` and ``inputs``. ``inputs`` is what makes the
    precondition checkable — see point 4 in the module docstring.
    """
    real = getattr(module, name)
    log = {"n": 0, "secs": 0.0, "inputs": []}

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = real(*args, **kwargs)
        log["n"] += 1
        log["secs"] += time.perf_counter() - t0
        if args:
            log["inputs"].append(args[0])
        return out

    setattr(module, name, wrapper)
    try:
        yield log
    finally:
        setattr(module, name, real)


def same_input(log):
    """Did every counted call see the same array? The cacheability precondition."""
    import numpy as np

    sums = set()
    for arr in log["inputs"]:
        try:
            sums.add(round(float(np.nansum(np.asarray(arr, dtype="float64"))), 6))
        except Exception:
            return None
    return len(sums) == 1 if sums else None


def gate(item, gain_secs, operation_secs, per_frame=False):
    """Print the verdict, and return True when the item may be merged.

    *gain_secs* is what the fix **removes**, not what the function costs. If you
    only have the function's cost, that is its ceiling: say so, because a ceiling
    under the bar settles the item without implementing it.
    """
    share = gain_secs / operation_secs if operation_secs else 0.0
    print(f"  {item}")
    print(f"    gain      {gain_secs * 1000:9.1f} ms")
    print(f"    operation {operation_secs * 1000:9.1f} ms")
    if per_frame:
        ok = gain_secs * 1000 >= DRAG_BUDGET_MS
        print(f"    -> {gain_secs * 1000:.1f} ms/frame against a "
              f"{DRAG_BUDGET_MS:.0f} ms bar: {'MERGE' if ok else 'DROP'}")
    else:
        ok = share >= OPERATION_BAR
        print(f"    -> {share * 100:.1f}% of the operation against a "
              f"{OPERATION_BAR * 100:.0f}% bar: {'MERGE' if ok else 'DROP'}")
    return ok


def main():
    """Worked example: M-6, the item the gate was hardest on.

    Kept as the template — copy this shape for a new item rather than writing a
    one-off script, so the denominator and the call count are always both stated.
    """
    import _probe

    _probe.start_qgis()

    from terrainflow_assessment.modules import flow_graph
    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

    dem = real_dem()
    print(f"measuring on {dem}")

    fa = FlowAnalysis()
    fa.load_dem(dem)

    with counted(flow_graph, "d8_from_dem") as log:
        operation = bench(lambda: fa.run(), repeat=1)

    print(f"  d8_from_dem calls during run(): {log['n']}")
    print(f"  every call saw one surface:     {same_input(log)}")
    if log["n"] > 1:
        wasted = log["secs"] * (log["n"] - 1) / log["n"]
        gate("M-6 cache the D8 pointers", wasted, operation)
    else:
        print("  nothing repeated — the fix is already in, or the path changed.")


if __name__ == "__main__":
    main()
