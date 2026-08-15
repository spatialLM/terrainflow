"""The baseline double-start refusal — alone in its module, deliberately.

Same subject as `checks_threading.py`, and it would read better beside its
siblings. It is here because it cannot share a process with them: it passes
cleanly on its own and then kills the *next* harness-based check in the same
QGIS subprocess, somewhere after its own assertions have all succeeded.

Ruled out, so nobody repeats the bisection: it is not `panel.deleteLater()`
(disabling it does not help), not the controller `teardown()` loop (same), not
garbage collection of the finished QThread (pinning the workers does not help),
and not undelivered queued events (draining the loop before teardown does not
help). Every other harness-based check in this module's sibling — including one
that boots two plugins with two real analyses — is fine in sequence.

The fault is in the scaffolding rather than the plugin: nothing here fails, and
what dies is the process afterwards. `run_all.py` already gives each module its
own QGIS subprocess, so one check per module is the isolation that mechanism was
built to provide.
"""


REAL_THREADS = True

JOIN_MS = 5_000


from checks_threading import GatedWorker, _pump, _started, harness  # noqa: E402,F401


def check_a_second_run_is_refused_while_one_is_live(dem_path):
    """The fault: reassigning the slot drops the only reference to a live QThread."""
    from terrainflow_assessment.qgis.workers._lifecycle import worker_is_running

    worker = GatedWorker.build()
    with harness(dem_path) as h:
        try:
            h.state.analysis_worker = _started(worker)
            assert worker_is_running(h.state, "analysis_worker")

            h.panel.run_baseline_requested.emit()

            assert h.state.analysis_worker is worker, (
                "a second run replaced the live worker — the first thread is now "
                "unreferenced and will be collected mid-run")
            assert worker.isRunning(), "the live worker was torn down by a second run"
            assert any("already running" in text
                       for _, _, text in h.bar.messages), (
                f"the refusal was silent: {h.bar.messages}")
        finally:
            worker.release.set()
            worker.wait(JOIN_MS)
            h.state.analysis_worker = None
            _pump(0.3)
