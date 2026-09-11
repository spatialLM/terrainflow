"""_lifecycle.py — starting a worker once, and getting it to stop.

Two failures this exists to prevent, both of which end in a QGIS crash rather than
an error message.

**Orphaned threads.** ``PluginState`` holds the only Python reference to a running
``QThread``. Reassigning ``state.analysis_worker`` drops it, and Python collects the
wrapper while C++ is still inside ``run()`` — "QThread: Destroyed while thread is
still running", then abort. Baseline and earthworks *share* that one attribute, so
clicking Re-analyse while Baseline is running is enough to do it.

**Teardown racing a writer.** ``unload()`` removes the session's scratch directory.
A worker still running is still writing rasters into it.

A note on ``quit()``, because it is the obvious thing to reach for and it does
nothing here. ``QThread.quit()`` asks the thread's *event loop* to exit, and these
workers override ``run()`` without ever calling ``exec()`` — there is no event loop
to ask. The only thing that actually stops one is the abort flag it checks between
stages, and the only thing that confirms it stopped is ``wait()``.
"""


class AbortMixin:
    """Cooperative cancellation for a ``QThread`` that overrides ``run()``.

    The worker calls :meth:`raise_if_aborted` between stages; the GUI thread calls
    :meth:`request_abort`. A plain attribute is enough for the handoff — one writer,
    one reader, and a stale read costs at most one more stage of work.
    """

    _abort = False

    def request_abort(self):
        self._abort = True

    @property
    def aborted(self):
        return bool(getattr(self, "_abort", False))

    def raise_if_aborted(self):
        """Bail out of ``run()`` at the next checkpoint.

        Raises :class:`WorkerAborted`, which the worker's own ``except`` turns into
        a silent return rather than an error banner — the user asked for this.
        """
        if self.aborted:
            raise WorkerAborted()


class WorkerAborted(Exception):
    """Raised inside a worker's ``run()`` when cancellation was requested."""


#: Every slot on ``PluginState`` that can hold a live QThread. A worker missing
#: from here is one ``unload`` will not wait for, which is how a thread outlives
#: the plugin and QGIS aborts on "Destroyed while thread is still running".
#: ``tests/test_architecture.py`` asserts this tuple against every ``PluginState``
#: field ending ``_worker``, because ``terrain_worker`` was missing from here for
#: as long as it had existed and nothing said so.
WORKER_SLOTS = ("analysis_worker", "sim_worker", "contour_worker", "design_worker",
                "terrain_worker")


def join_workers(state, timeout_ms=10_000):
    """Ask every live worker on *state* to stop, and wait until it has.

    Returns the list of workers that did **not** stop within the timeout, so the
    caller can decide whether it is safe to delete their working directory. An
    empty list means nothing is still writing.
    """
    stragglers = []
    for attr in WORKER_SLOTS:
        worker = getattr(state, attr, None)
        if worker is None:
            continue
        try:
            if not worker.isRunning():
                setattr(state, attr, None)
                continue
            if hasattr(worker, "request_abort"):
                worker.request_abort()
            # quit() before wait() is the documented idiom and is harmless, but it
            # is wait() doing the work here — see the module docstring.
            worker.quit()
            if worker.wait(timeout_ms):
                setattr(state, attr, None)
            else:
                stragglers.append(worker)
        except RuntimeError:
            # Already destroyed on the C++ side.
            setattr(state, attr, None)
    return stragglers


def worker_is_running(state, attr):
    """True when *attr* on *state* holds a thread that has not finished.

    Tolerates the attribute being absent, None, or a wrapper whose C++ object has
    already gone — all of which mean "nothing running".
    """
    worker = getattr(state, attr, None)
    if worker is None:
        return False
    try:
        return bool(worker.isRunning())
    except RuntimeError:
        return False
