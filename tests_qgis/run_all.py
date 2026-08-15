"""
run_all.py — discover and run the real-QGIS smoke checks.

Must run under QGIS's own Python. Use run_qgis_tests.ps1 from the repo root:

    .\run_qgis_tests.ps1                    # everything except OPT_IN_MODULES
    .\run_qgis_tests.ps1 baseline           # only checks matching "baseline"
    .\run_qgis_tests.ps1 slow               # run a quarantined module explicitly
    .\run_qgis_tests.ps1 --timeout=600      # raise the per-module time limit

Each checks_*.py module runs in its own QGIS subprocess with a time limit. That
costs ~5 s of boot per module and buys two things worth more than the seconds: a
hung check (or a PyQGIS segfault) reports as one failed module instead of
wedging the whole run, and no module can leak state into the next.

Exit code is 0 when every check passes, 1 otherwise — so CI can gate on it.
"""

from __future__ import annotations

import importlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_TIMEOUT_S = 300

# Modules skipped unless named explicitly on the command line. Empty since Round 15:
# `checks_slow` held only `recommend_ponds`, whose "does not terminate" was a modal
# QMessageBox waiting for a click that never comes offscreen — the harness records those
# now (see RecordingDialogs), so it runs in seconds and lives in checks_contour with the
# rest of the keypoint path. Kept as a mechanism, because quarantining beats deleting a
# check that has found something real.
OPT_IN_MODULES = set()


# ---------------------------------------------------------------------------
# Worker: runs one module's checks inside a fresh QGIS process
# ---------------------------------------------------------------------------

def collect(module_name, patterns):
    """Return [(check_name, callable)] in definition order, honouring patterns."""
    module = importlib.import_module(module_name)
    checks = [
        (name, obj)
        for name, obj in vars(module).items()
        if name.startswith("check_") and callable(obj)
    ]
    checks.sort(key=lambda item: item[1].__code__.co_firstlineno)

    if not patterns:
        return checks
    return [
        (name, fn)
        for name, fn in checks
        if any(p in name or p in module_name for p in patterns)
    ]


def run_worker(module_name, patterns):
    try:
        import qgis.core  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "\nThese checks need QGIS's bundled Python (PyQGIS is not importable).\n"
            "Run them via run_qgis_tests.ps1 from the repo root instead.\n\n"
        )
        return 2

    import _harness

    checks = collect(module_name, patterns)
    if not checks:
        print(f"RESULT {module_name} 0 0")
        return 0

    _harness.qgis_app()
    # Workers run inline unless a module says otherwise. A module opts out by
    # declaring `REAL_THREADS = True`, which is only right for checks *about*
    # concurrency: with `start = run` a worker has always finished by the time
    # anyone can ask, so `isRunning()` is never True and the whole class of
    # double-start and teardown-during-run faults is unobservable.
    #
    # Safe to do per module because each one already gets its own QGIS
    # subprocess with a time limit, so a deadlocked thread reports as one failed
    # module rather than wedging the run.
    if not getattr(importlib.import_module(module_name), "REAL_THREADS", False):
        _harness.make_workers_synchronous()

    dem_path = os.environ.get("TFA_CHECK_DEM")
    if not dem_path or not os.path.exists(dem_path):
        workdir = Path(tempfile.mkdtemp(prefix="tfa_qgis_checks_"))
        dem_path = _harness.build_synthetic_dem(workdir / "synthetic_dem.tif")

    passed = failed = 0
    for name, fn in checks:
        # The orchestrator reads this line to name the culprit if we time out, so
        # it must be flushed before the check starts.
        print(f"  {name} ... ", end="", flush=True)
        try:
            fn(dem_path)
        except Exception:
            print("FAIL", flush=True)
            print(f"\n--- {module_name}.{name}\n", flush=True)
            print(traceback.format_exc().rstrip(), flush=True)
            failed += 1
        else:
            print("ok", flush=True)
            passed += 1

    print(f"RESULT {module_name} {passed} {failed}", flush=True)
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _pump(stream, sink):
    for line in iter(stream.readline, ""):
        sink.put(line)
    sink.put(None)


def run_module_isolated(module_name, patterns, timeout_s, dem_path):
    """Spawn one QGIS process for a module; stream its output, enforce a deadline."""
    cmd = [sys.executable, str(HERE / "run_all.py"), "--worker", module_name, *patterns]
    env = dict(os.environ, TFA_CHECK_DEM=dem_path, PYTHONUNBUFFERED="1")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    lines = queue.Queue()
    threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()

    passed = failed = 0
    last_started = None
    body = []

    deadline = time.monotonic() + timeout_s

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            proc.wait()
            culprit = last_started or "(before the first check)"
            print(f"TIMEOUT after {timeout_s}s during {culprit}")
            body.append(
                f"\n--- {module_name}.{culprit}\n\n"
                f"    Timed out after {timeout_s}s — no result. Either the check is\n"
                f"    genuinely slow on this DEM or the code under it does not "
                f"terminate."
            )
            return passed, failed + 1, body

        try:
            line = lines.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue

        if line is None:
            break

        stripped = line.rstrip("\n")
        if stripped.startswith("RESULT "):
            _, _mod, p, f = stripped.split()
            passed, failed = int(p), int(f)
            continue
        if stripped.startswith("--- ") or (body and not stripped.startswith("  ")):
            body.append(stripped)
            continue

        if stripped.strip().endswith("..."):
            last_started = stripped.strip()[:-4].strip()
        print(stripped, flush=True)

    proc.wait()
    return passed, failed, body


def main(argv):
    if "--worker" in argv:
        i = argv.index("--worker")
        module_name = argv[i + 1]
        patterns = [a for a in argv[i + 2:] if not a.startswith("-")]
        return run_worker(module_name, patterns)

    timeout_s = DEFAULT_TIMEOUT_S
    snapshot = False
    patterns = []
    for arg in argv:
        if arg.startswith("--timeout="):
            timeout_s = int(arg.split("=", 1)[1])
        elif arg == "--snapshot":
            snapshot = True
        elif arg == "--accept":
            # Accept the screenshots already on disk, without re-running anything.
            from _shots import BASELINE_DIR, snapshot_baseline

            names = snapshot_baseline()
            print(f"Baseline updated: {len(names)} image(s) -> {BASELINE_DIR}")
            return 0
        elif not arg.startswith("-"):
            patterns.append(arg)

    modules = [p.stem for p in sorted(HERE.glob("checks_*.py"))]
    if patterns:
        modules = [
            m for m in modules
            if any(p in m for p in patterns) or collect(m, patterns)
        ]
    else:
        # Quarantined modules run only when named. checks_slow holds checks that do
        # not terminate on this fixture (recommend_ponds), so including it by default
        # would make the plain `run_qgis_tests.ps1` always burn a full timeout and
        # exit non-zero — a default command that always fails is one nobody runs.
        modules = [m for m in modules if m not in OPT_IN_MODULES]
    if not modules:
        print("No checks matched.")
        return 1

    # One DEM for the whole run, shared with every worker.
    import _harness

    workdir = Path(tempfile.mkdtemp(prefix="tfa_qgis_checks_"))
    dem_path = _harness.build_synthetic_dem(workdir / "synthetic_dem.tif")
    print(f"Synthetic DEM: {dem_path}")
    print(f"Per-module timeout: {timeout_s}s")

    total_passed = total_failed = 0
    failures = []
    run_started = time.time()

    for module_name in modules:
        print(f"\n{module_name}")
        passed, failed, body = run_module_isolated(
            module_name, patterns, timeout_s, dem_path
        )
        total_passed += passed
        total_failed += failed
        if body:
            failures.extend(body)

    print("\n" + "=" * 72)
    for line in failures:
        print(line)
    if failures:
        print("\n" + "=" * 72)
    print(f"{total_passed} passed, {total_failed} failed")

    _report_visual_diff(run_started, snapshot)

    return 1 if total_failed else 0


def _report_visual_diff(run_started, snapshot):
    """Compare this run's screenshots to the baseline, or become the new baseline.

    A changed image is information, not a failure: it may be exactly the change that
    was intended. So this never affects the exit code.
    """
    from _shots import (
        BASELINE_DIR,
        compare_to_baseline,
        print_comparison,
        snapshot_baseline,
    )

    if snapshot:
        names = snapshot_baseline()
        print(f"\nBaseline updated: {len(names)} image(s) -> {BASELINE_DIR}")
        print("Make your change, then re-run without --snapshot to see what moved.")
        return

    if not BASELINE_DIR.exists() or not any(BASELINE_DIR.glob("*.png")):
        print("\nNo visual baseline stored. Run with --snapshot to capture one,")
        print("then re-run after a change to see which images moved.")
        return

    print_comparison(compare_to_baseline(since=run_started))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
