"""Worker lifecycle, against threads that are genuinely running.

`REAL_THREADS = True` opts this module out of `make_workers_synchronous()`. That
patch (`start = run`) is right for every other module — it makes results
deterministic with no event loop to spin — but it makes *this* class of fault
invisible by construction: with `start` aliased to `run`, a worker has always
finished by the time anyone can ask, `isRunning()` is never True, and neither a
double-start nor a teardown-during-run can be constructed at all.

The determinism problem that patch was solving is handled differently here.
Nothing races real analysis: a worker is parked on a `threading.Event` at a known
point, so "a thread is running right now" is a fact the check establishes rather
than a window it hopes to hit. Every check releases the gate in a `finally`.

Safe as its own module because `run_all.py` gives each one a separate QGIS
subprocess with a time limit, so a thread that deadlocks reports as one failed
module instead of wedging the run.
"""

import threading
import time
from contextlib import contextmanager

REAL_THREADS = True

# Long enough that a slow machine does not fail spuriously, short enough that a
# genuine deadlock reports well inside the runner's per-module limit.
JOIN_MS = 5_000
SETTLE_S = 2.0


class GatedWorker:
    """A worker parked mid-run until released — a deterministic "still running".

    Subclasses the real `AnalysisWorker` so the lifecycle code under test sees the
    genuine article: a `QThread` whose `run()` has been entered and not returned.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("use GatedWorker.build()")

    @staticmethod
    def build():
        from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker

        class _Gated(AnalysisWorker):
            def __init__(self):
                super().__init__(dem_path="", output_dir="", stream_threshold=1,
                                 cn=70, moisture="normal", rainfall_mm=50,
                                 duration_hours=24)
                self.entered = threading.Event()
                self.release = threading.Event()
                self.finished_cleanly = False

            def run(self):
                self.entered.set()
                # Poll rather than block outright, so an abort is still observable
                # while parked — which is exactly what teardown depends on.
                while not self.release.is_set():
                    if self.aborted:
                        return
                    time.sleep(0.01)
                self.finished_cleanly = True

        return _Gated()


def _started(worker):
    worker.start()
    assert worker.entered.wait(5.0), "worker never entered run()"
    assert worker.isRunning(), "worker entered run() but reports not running"
    return worker


def _pump(seconds=0.2):
    """Deliver queued cross-thread signals.

    A worker's `progress`/`completed` are emitted from the worker thread and
    queued for the GUI thread. Nothing runs an event loop here, so without this
    the completion handlers never fire and the panel never updates.
    """
    from qgis.PyQt.QtCore import QCoreApplication

    deadline = time.time() + seconds
    while time.time() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)


def run_baseline_and_wait(h, timeout_s=180):
    """`h.run_baseline()` is asynchronous *in this module only*.

    Everywhere else the harness patches `start = run`, so the helper returns with
    the analysis already complete and its completion handler already run. Here it
    returns the moment the thread starts. Any check that then touches
    `state.analysis_worker` — which is most of them — would be dropping the only
    reference to a live QThread, which is the very fault under test.
    """
    from qgis.PyQt.QtCore import QCoreApplication

    h.run_baseline()
    worker = h.state.analysis_worker
    if worker is not None:
        deadline = time.time() + timeout_s
        while worker.isRunning() and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        assert not worker.isRunning(), "baseline did not finish within the timeout"
    _pump(0.4)
    h.state.analysis_worker = None


@contextmanager
def harness(dem_path):
    """`PluginHarness`, with the event loop drained after teardown.

    `unload` ends with `panel.deleteLater()`, which is a *deferred* delete: Qt
    performs it the next time an event loop runs. No check runs one, so the
    deletion of check A's panel lands inside whichever later check first pumps —
    and it takes QGIS down with it. Draining here keeps each check's debris inside
    the check that made it.

    Found the hard way: every check in this module passes alone, and the module
    died at the third harness when run in sequence.
    """
    from _harness import PluginHarness

    with PluginHarness(dem_path) as h:
        yield h
    _pump(0.3)


# ---------------------------------------------------------------- the premise

def check_the_harness_is_actually_threading_here(dem_path):
    """Guard the guard: if the sync patch leaks in, every check below is theatre."""
    from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker

    assert AnalysisWorker.start is not AnalysisWorker.run, (
        "workers are still synchronous — REAL_THREADS did not take effect, and "
        "nothing in this module is testing what it claims to")

    worker = GatedWorker.build()
    try:
        _started(worker)
    finally:
        worker.release.set()
        worker.wait(JOIN_MS)


# ---------------------------------------------------------------- cancellation

def check_quit_alone_does_not_stop_an_overridden_run(dem_path):
    """The trap this design exists to avoid, pinned so nobody "simplifies" it back.

    `QThread.quit()` asks the thread's event loop to exit. These workers override
    `run()` and never call `exec()`, so there is no event loop and `quit()` is a
    no-op. A teardown built on `quit()` + `wait()` alone would look correct, pass
    review, and hang.
    """
    worker = GatedWorker.build()
    try:
        _started(worker)
        worker.quit()
        assert not worker.wait(300), (
            "quit() stopped a run() with no event loop — if PyQt has changed this, "
            "the abort flag may no longer be load-bearing")
        assert worker.isRunning()
    finally:
        worker.release.set()
        worker.wait(JOIN_MS)


def check_abort_flag_stops_a_running_worker(dem_path):
    """What actually stops one: the flag it checks, then wait()."""
    worker = GatedWorker.build()
    try:
        _started(worker)
        worker.request_abort()
        assert worker.wait(JOIN_MS), "worker ignored the abort and never returned"
        assert not worker.isRunning()
        assert not worker.finished_cleanly, "worker ran to completion despite abort"
    finally:
        worker.release.set()
        worker.wait(JOIN_MS)


def check_join_workers_clears_the_state_slots(dem_path):
    from terrainflow_assessment.qgis.controllers._state import PluginState
    from terrainflow_assessment.qgis.workers._lifecycle import join_workers

    state = PluginState()
    worker = GatedWorker.build()
    try:
        state.analysis_worker = _started(worker)
        stragglers = join_workers(state, timeout_ms=JOIN_MS)

        assert stragglers == [], f"join_workers gave up on {stragglers}"
        assert state.analysis_worker is None, (
            "the slot must be cleared, or the next unload tries to join a dead thread")
        assert not worker.isRunning()
    finally:
        worker.release.set()
        worker.wait(JOIN_MS)


def check_join_workers_reports_one_that_will_not_stop(dem_path):
    """A worker that ignores the abort is named, not silently assumed stopped.

    `unload` keys the scratch-directory removal off this list: something still
    writing means the directory stays.
    """
    from terrainflow_assessment.qgis.controllers._state import PluginState
    from terrainflow_assessment.qgis.workers._lifecycle import join_workers

    state = PluginState()
    worker = GatedWorker.build()
    worker.request_abort = lambda: None          # deaf to cancellation
    try:
        state.analysis_worker = _started(worker)
        stragglers = join_workers(state, timeout_ms=300)

        assert stragglers == [worker], (
            f"expected the stuck worker to be reported, got {stragglers}")
    finally:
        worker.release.set()
        worker.wait(JOIN_MS)


# ---------------------------------------------------------------- double start

def check_earthworks_refuses_before_it_burns(dem_path):
    """The guard has to precede the synchronous burn, not just the start() call.

    Burning a real DEM takes seconds; a guard at `start()` leaves that whole window
    open, and the second click is already partway through the burn when the first
    thread is dropped.
    """
    from _harness import line_across_valley

    worker = GatedWorker.build()
    with harness(dem_path) as h:
        run_baseline_and_wait(h)
        h.add_earthwork("swale", line_across_valley())
        try:
            h.state.analysis_worker = _started(worker)
            before = h.state.modified_dem_path

            h.panel.run_earthworks_requested.emit()

            assert h.state.analysis_worker is worker
            assert h.state.modified_dem_path == before, (
                "the burn ran anyway — the guard is downstream of it")
        finally:
            worker.release.set()
            worker.wait(JOIN_MS)
            h.state.analysis_worker = None


# ---------------------------------------------------------------- teardown

def check_unload_joins_before_removing_the_output_dir(dem_path):
    """Q-C1: `rmtree` used to run while a worker was still writing into it."""
    import os

    worker = GatedWorker.build()
    with harness(dem_path) as h:
        output_dir = h.state.output_dir
        os.makedirs(output_dir, exist_ok=True)
        h.state.analysis_worker = _started(worker)
        plugin = h.plugin

    # The harness exit calls unload(). The worker honours the abort, so the join
    # succeeds and the directory goes.
    assert not worker.isRunning(), "unload returned with a worker still running"
    assert not os.path.exists(output_dir), (
        "output_dir survived an unload that had nothing left to wait for")
    del plugin


def check_unload_keeps_the_output_dir_when_a_worker_will_not_stop(dem_path):
    """Deleting files under a live writer is worse than leaking a temp directory."""
    import os

    worker = GatedWorker.build()
    worker.request_abort = lambda: None          # deaf to cancellation
    output_dir = None
    try:
        with harness(dem_path) as h:
            output_dir = h.state.output_dir
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "in_progress.tif"), "wb") as fh:
                fh.write(b"0")
            h.state.analysis_worker = _started(worker)

        assert os.path.exists(output_dir), (
            "unload deleted the scratch directory while a worker was still writing")
    finally:
        worker.release.set()
        worker.wait(JOIN_MS)
        if output_dir and os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)


def check_unload_unsets_the_plugins_map_tool(dem_path):
    """A tool left armed feeds clicks into a controller whose world is gone."""
    with harness(dem_path) as h:
        run_baseline_and_wait(h)
        h.plugin._earthworks.activate_draw_line("swale")
        canvas = h.plugin._canvas
        tool = canvas.mapTool()
        assert tool is not None, "no tool was activated — check the fixture"

    assert canvas.mapTool() is not tool, (
        "unload left the plugin's draw tool on the canvas")


def check_a_second_boot_works_in_one_process(dem_path):
    """Reload is load, unload, load. The second one has to be clean."""
    with harness(dem_path) as h:
        first_panel = h.panel
        run_baseline_and_wait(h)
        assert h.panel._stepper.state("baseline") == "done"

    with harness(dem_path) as h:
        assert h.panel is not first_panel, "the second boot reused the first panel"
        assert len(h.iface.dock_widgets) == 1, (
            f"{len(h.iface.dock_widgets)} panels docked after a reload — the first "
            f"was never removed")
        run_baseline_and_wait(h)
        h.assert_no_errors("baseline after reload")
        assert h.panel._stepper.state("baseline") == "done"


def check_canvas_scale_signal_is_disconnected_on_unload(dem_path):
    """Q-C2: `scaleChanged` outlives the plugin and fires on every zoom.

    The canvas belongs to QGIS, so nothing else will ever take the connection down.
    """
    with harness(dem_path) as h:
        run_baseline_and_wait(h)
        canvas = h.plugin._canvas
        baseline = h.plugin._baseline
        calls = []
        baseline.on_map_scale_changed = lambda *a: calls.append(a)

    canvas.scaleChanged.emit(12345.0)
    time.sleep(0.05)
    assert calls == [], (
        f"the dead controller was still called on zoom: {calls}")
