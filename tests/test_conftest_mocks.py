"""The QGIS stand-ins in ``conftest.py``, tested where tests rely on their behaviour.

``conftest.py`` puts fakes into ``sys.modules`` for ``qgis`` and its submodules so the
pure suite runs with no QGIS at all. A fake that is merely *importable* is enough for
most of the suite and not enough for the parts that exercise signals: a stub which
looks right and behaves differently from PyQt moves the failure from the mock (where
it is obvious) into whichever test happens to use it next (where it is not).

Only the behaviours something depends on are pinned here. This is not a PyQt
reimplementation and must not grow into one.
"""

from qgis.PyQt.QtCore import pyqtSignal


class _Worker:
    """Shaped like the real workers: signals declared as class attributes."""

    done = pyqtSignal(object)
    progress = pyqtSignal(int, str)

    def __init__(self, name):
        self.name = name


class TestPyqtSignalStub:
    def test_two_instances_do_not_share_a_signal(self):
        """T-2. ``done = pyqtSignal(object)`` evaluates once, at class-definition
        time, so a stub that returns a plain object makes it a **class attribute** —
        one signal for every worker ever constructed.

        Nothing passed because of it: only four ``.connect()`` calls exist in the
        pure suite. But it is the sharpest edge left in the mock, and the failure it
        produces is the worst kind — a slot on the worker you are testing fires for
        an unrelated worker's result, with nothing pointing back at the stub.
        """
        a, b = _Worker("a"), _Worker("b")
        assert a.done is not b.done, (
            "both workers hold the same signal object, so emitting on one fires the "
            "other's slots"
        )

    def test_emitting_on_one_instance_does_not_reach_the_other(self):
        """The consequence, stated as behaviour rather than as identity."""
        a, b = _Worker("a"), _Worker("b")
        seen_a, seen_b = [], []
        a.done.connect(seen_a.append)
        b.done.connect(seen_b.append)

        a.done.emit("from a")

        assert seen_a == ["from a"]
        assert seen_b == [], f"b's slot fired on a's emit: {seen_b}"

    def test_a_signal_still_delivers_its_arguments(self):
        """The stub is not so isolated that it stops working."""
        w = _Worker("w")
        seen = []
        w.progress.connect(lambda pct, msg: seen.append((pct, msg)))
        w.progress.emit(40, "halfway")
        assert seen == [(40, "halfway")]

    def test_the_same_instance_returns_the_same_signal_each_time(self):
        """``w.done.connect(...)`` then ``w.done.emit(...)`` has to reach one object.

        A descriptor that built a fresh signal per attribute access would pass the
        isolation test above and silently drop every connection.
        """
        w = _Worker("w")
        assert w.done is w.done
        seen = []
        w.done.connect(seen.append)
        w.done.emit("x")
        assert seen == ["x"]

    def test_two_signals_on_one_instance_stay_apart(self):
        w = _Worker("w")
        seen = []
        w.done.connect(lambda *a: seen.append(("done", a)))
        w.progress.connect(lambda *a: seen.append(("progress", a)))
        w.progress.emit(1, "go")
        assert seen == [("progress", (1, "go"))]
