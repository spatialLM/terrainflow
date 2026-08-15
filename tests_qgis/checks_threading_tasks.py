"""The heavy analyses, against threads that are genuinely running.

`REAL_THREADS = True` opts this module out of `make_workers_synchronous()`. With
`start` aliased to `run` a `TaskWorker` has always finished by the time anyone can
ask, so none of the faults below can be constructed at all: not a second click
landing on a running analysis, not a burn racing the terrain measurement over one
`DEMBurner`, not `unload` walking away from a live thread.

Its own module for the same reason `checks_threading` is: `run_all.py` gives each
one a separate QGIS subprocess with a time limit, so a thread that deadlocks
reports as one failed module rather than wedging the run.

Nothing here races real analysis. A `TaskWorker` is parked on a `threading.Event`
at a known point, so "a task is running right now" is a fact the check establishes
rather than a window it hopes to hit. Every check releases the gate in a `finally`.
"""

import threading
import time
from contextlib import contextmanager

REAL_THREADS = True

JOIN_MS = 5_000


def _pump(seconds=0.2):
    """Deliver queued cross-thread signals — nothing else runs an event loop."""
    from qgis.PyQt.QtCore import QCoreApplication

    deadline = time.time() + seconds
    while time.time() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)


@contextmanager
def harness(dem_path):
    """`PluginHarness`, with the event loop drained after teardown.

    `unload` ends in `panel.deleteLater()`, a *deferred* delete Qt performs at the
    next event loop. Undrained, check A's panel is deleted inside whichever later
    check first pumps, and takes QGIS with it. See `checks_threading`.
    """
    from _harness import PluginHarness

    try:
        with PluginHarness(dem_path) as h:
            yield h
    finally:
        # In a finally, not after the `with`: a check that fails inside would
        # otherwise skip the drain, and its panel's deferred delete then lands in
        # whichever later check first pumps — taking QGIS down there instead.
        _pump(0.3)


@contextmanager
def gated_task(state, slot="contour_worker"):
    """A `TaskWorker` parked mid-run and filed in *slot*, released on the way out."""
    from terrainflow_assessment.qgis.workers.task_worker import TaskWorker

    entered, release = threading.Event(), threading.Event()

    def work(report):
        entered.set()
        while not release.is_set():
            # Reporting is the abort checkpoint, so a parked task is still
            # cancellable — which is what teardown depends on.
            report(50, "parked")
            time.sleep(0.01)
        return "released"

    worker = TaskWorker(work, label="gated")
    setattr(state, slot, worker)
    worker.start()
    try:
        assert entered.wait(5.0), "the task never entered its work"
        assert worker.isRunning(), "the task entered its work but reports not running"
        yield worker
    finally:
        release.set()
        worker.wait(JOIN_MS)


def check_a_task_worker_actually_threads(dem_path):
    """If this fails every other check in the module is vacuous."""
    with harness(dem_path) as h:
        with gated_task(h.state) as worker:
            assert worker.isRunning()
        assert not worker.isRunning()


def check_a_second_contour_run_is_refused_while_one_is_live(dem_path):
    """The four contour analyses share one slot, and the second assignment would
    drop the only reference keeping the first thread alive."""
    with harness(dem_path) as h:
        # Far enough past the earlier guards to reach the one under test. A real
        # baseline would cost a minute here and prove nothing extra: the guard
        # only asks whether the file exists.
        h.state.baseline_result = {"flow_accumulation": dem_path}
        with gated_task(h.state) as worker:
            h.bar.messages.clear()
            h.panel.run_contour_analysis_requested.emit()

            assert h.state.contour_worker is worker, (
                "the running task was replaced — its thread is now unreferenced")
            assert any("still running" in text.lower()
                       for _, _, text in h.bar.messages), (
                f"the refusal was silent: {h.bar.messages}")


def check_a_second_design_run_is_refused_while_the_burn_is_live(dem_path):
    """Re-analyse is two stages now — a burn, then the analysis it feeds. A click
    during the first would burn against state the first run is still using."""
    from _harness import line_across_valley

    with harness(dem_path) as h:
        ew = h.add_earthwork("swale", line_across_valley())
        ew.capacity_m3 = 50.0

        # No baseline: the busy guard is the first thing run_with_earthworks
        # checks, which is the point — it has to fire before anything touches the
        # burner, not after.
        with gated_task(h.state, slot="design_worker") as worker:
            h.bar.messages.clear()
            h.panel.run_earthworks_requested.emit()

            assert h.state.design_worker is worker, (
                "the running burn was replaced mid-flight")
            assert any("already running" in text.lower()
                       for _, _, text in h.bar.messages), (
                f"the refusal was silent: {h.bar.messages}")


def check_the_terrain_sweep_does_not_flood_while_the_burn_holds_the_burner(dem_path):
    """Both hold `state.burner` and both flood it. Two at once would interleave
    their warnings and share whatever scratch the burner keeps."""
    from _harness import line_across_valley

    with harness(dem_path) as h:
        ew = h.add_earthwork("swale", line_across_valley())
        ew.capacity_m3 = 50.0
        ew.terrain_capacity_m3 = None

        with gated_task(h.state, slot="design_worker"):
            started = h.plugin._earthworks._refresh_all_terrain_capacities()
            assert started is False, (
                "the sweep started a second flood over the burner the burn is using")
            h.plugin._earthworks._refresh_terrain_capacity(ew)
            assert ew.terrain_capacity_m3 is None, (
                "a single-feature refresh flooded the burner mid-burn")


def check_join_workers_waits_for_a_task(dem_path):
    """`unload` deletes `output_dir`; a task still writing into it must be joined
    first, and a slot missing from WORKER_SLOTS is a slot nothing waits for."""
    from terrainflow_assessment.qgis.workers._lifecycle import WORKER_SLOTS, join_workers

    assert "contour_worker" in WORKER_SLOTS and "design_worker" in WORKER_SLOTS, (
        f"a worker slot is not joined on unload: {WORKER_SLOTS}")

    with harness(dem_path) as h:
        with gated_task(h.state) as worker:
            stragglers = join_workers(h.state, timeout_ms=JOIN_MS)
            assert not stragglers, f"the task would not stop: {stragglers}"
            assert not worker.isRunning(), "join_workers returned with it still live"
            assert h.state.contour_worker is None, "the slot was not cleared"


def check_unload_does_not_abandon_a_running_task(dem_path):
    """A QThread outliving the plugin is the "Destroyed while thread is still
    running" abort — QGIS goes down with it, taking the user's project."""
    from _harness import PluginHarness

    with PluginHarness(dem_path) as h:
        worker = None
        try:
            with gated_task(h.state) as w:
                worker = w
                h.plugin.unload()
                assert not worker.isRunning(), (
                    "unload returned while the task was still running")
        finally:
            h.plugin.unload = lambda: None      # the context manager unloads again
    _pump(0.3)
    assert worker is not None and not worker.isRunning()
