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


def check_join_workers_waits_for_the_terrain_worker(dem_path):
    """A fifth slot existed that `unload` never looked at.

    `terrain_worker` is declared on `PluginState` and assigned a live `TaskWorker`
    by `terrain.py`, but it was absent from `WORKER_SLOTS` — so a reload during
    "Compute Terrain Indices" got an empty straggler list, `unload` rmtree'd
    `output_dir` under a thread mid-write, and the QThread lost its last
    reference. Deliberately its own check rather than a line in the one above: a
    slot nothing joins is invisible in a test that only walks the slots.
    """
    from terrainflow_assessment.qgis.workers._lifecycle import WORKER_SLOTS, join_workers

    assert "terrain_worker" in WORKER_SLOTS, (
        f"the terrain indices thread is not joined on unload: {WORKER_SLOTS}")

    with harness(dem_path) as h:
        with gated_task(h.state, slot="terrain_worker") as worker:
            stragglers = join_workers(h.state, timeout_ms=JOIN_MS)
            assert not stragglers, f"the terrain task would not stop: {stragglers}"
            assert not worker.isRunning(), "join_workers returned with it still live"
            assert h.state.terrain_worker is None, "the slot was not cleared"


def check_a_dam_capacity_defers_while_the_burn_holds_the_burner(dem_path):
    """The fourth flood over `state.burner`, and the one that had no guard.

    `_compute_dam_capacity` is reached from `_on_geometry_drawn` and
    `_on_vertex_edit_finished`, so drawing or reshaping a dam during "Re-analyse
    with Earthworks" ran `dam_storage` on the GUI thread against the burner the
    burn worker was using. Two silent consequences, and this pins both ends of
    the second: `dam_storage` opens with `self.warnings = []`, discarding the
    honesty warnings the worker had accumulated for `_on_burn_complete` to push;
    and the same call's `_isolated_burn` snapshot/restore puts back a
    `burned_masks` taken mid-population, so features burned after it lose their
    mask and `_compute_verification` falls through to the re-derived footprint.
    The mask half is a genuine race and cannot be timed deterministically — but
    it needs the call to happen at all, which the cached-return assertion denies.
    """
    from _harness import line_across_valley

    with harness(dem_path) as h:
        controller = h.plugin._earthworks
        dam = h.add_earthwork("dam", geometry=line_across_valley(row=86))
        dam.crest_elevation = controller._default_crest_elevation(dam.geometry)
        assert dam.crest_elevation is not None, (
            "no crest was derived, so the guard under test is never reached")
        dam.capacity_m3 = 1234.0

        burner = h.state.burner
        assert burner is not None, "no burner on state — nothing for the two to contend for"
        worker_warning = "a sub-cell advisory the burn worker already raised"
        burner.warnings = [worker_warning]

        with gated_task(h.state, slot="design_worker"):
            got = controller._compute_dam_capacity(dam)

        assert got == 1234.0, (
            f"the dam was re-flooded over the burner the burn is using and returned "
            f"{got}; the cached 1234.0 is what a deferred measurement must give back")
        assert burner.warnings == [worker_warning], (
            f"the mid-burn flood cleared the burn worker's warnings: {burner.warnings}")


def check_a_dam_drawn_mid_burn_is_measured_when_the_run_finishes(dem_path):
    """Deferring the flood is only half a fix if nothing ever comes back for it.

    A dam drawn *during* a run has no cached capacity to return, so it defers to
    0 — and `_on_burn_complete` does not measure dams, so without the sweep at the
    end of `_on_earthworks_complete` the feature would keep a 0 m³ capacity and a
    "% full" bar reading full for the rest of the session.
    """
    from _harness import line_across_valley

    with harness(dem_path) as h:
        controller = h.plugin._earthworks
        dam = h.add_earthwork("dam", geometry=line_across_valley(row=86))
        dam.crest_elevation = controller._default_crest_elevation(dam.geometry)
        dam.capacity_m3 = 0.0

        with gated_task(h.state, slot="design_worker"):
            assert controller._compute_dam_capacity(dam) == 0.0, (
                "a dam with no cache did not defer to 0 while the burn held the burner")

        controller._measure_deferred_dams()
        assert dam.capacity_m3 > 0.0, (
            "the deferred dam was never re-measured once the burner was free — its "
            "capacity stays 0 and the live readout reads full")
        assert dam.capacity_l == dam.capacity_m3 * 1000.0, (
            "capacity_l was not kept in step with the re-measured capacity_m3")
