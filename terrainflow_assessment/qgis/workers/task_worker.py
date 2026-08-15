"""
task_worker.py — one QThread for any single piece of heavy analysis.

The contour, segment, keypoint, keyline, burn and terrain-capacity passes all had
the same shape: read some panel settings, call one pure function in ``modules/``
for several seconds, then build layers from what came back. They ran on the GUI
thread, so Qt could not repaint and Windows painted the window "Not Responding"
over a plugin that was working perfectly well — and every progress callback they
passed had nowhere to draw.

Rather than six worker classes, one parameterised by the work: the caller hands
over a callable that takes a progress reporter and returns whatever the completion
handler needs.

    def work(report):
        return analyse_contours(dem_path=dem, interval_m=interval,
                                progress_callback=report)

    worker = TaskWorker(work, label="contours")

**Everything the work needs must be read before the worker starts.** The callable
runs on the worker thread, so a closure that reaches back into ``self._panel`` or
``self._state`` while it runs is reading Qt state off the GUI thread — the class
this exists to avoid. Capture plain values in locals at the call site; the reviews
below all do.

**No layer creation inside the work.** ``QgsVectorLayer`` and everything that
touches the project registry belong to the GUI thread. The completion handler runs
there (a queued signal), which is where the layers get built.

Cancellation comes from ``AbortMixin``: the reporter is the abort checkpoint, so a
task that reports progress at all can be stopped between reports. See
``_lifecycle`` for why ``quit()`` alone cannot do it.
"""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal

from terrainflow_assessment.qgis.workers._lifecycle import AbortMixin, WorkerAborted


class TaskWorker(AbortMixin, QThread):
    """Run one callable off the GUI thread.

    Signals
    -------
    progress(int, str) — forwarded from the reporter handed to the work
    completed(object)  — whatever the work returned
    error(str)         — the formatted traceback

    ``completed`` rather than ``finished``: ``QThread`` already has a ``finished``
    signal and shadowing it blocks the standard teardown idiom.
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, work, label=""):
        super().__init__()
        self._work = work
        #: For messages and for whoever is reading a stack of running workers.
        self.label = label

    def _report(self, pct, message=""):
        """The progress reporter, doubling as the abort checkpoint.

        Raising out of the work is what stops it: a flag the work has to remember
        to poll is a flag some branch will forget.
        """
        self.raise_if_aborted()
        self.progress.emit(int(pct), str(message))

    def run(self):
        try:
            result = self._work(self._report)
        except WorkerAborted:
            return
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        # Outside the try: a slot that raises is the handler's bug, not the work's,
        # and reporting it as a failed analysis would be a lie about what happened.
        self.completed.emit(result)
