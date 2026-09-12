"""``tests_qgis/run_all.py``'s orchestrator half, tested without QGIS.

The orchestrator spawns one QGIS subprocess per checks module and reads its output.
None of that reading needs QGIS — it is line parsing over a pipe — and the part of
it that matters most is the part that only runs when something has gone wrong, which
is exactly the part a passing suite never exercises.

``run_all.py`` imports nothing but the standard library at module scope (``_harness``
is imported inside ``run_worker``), so it can be imported here directly.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TESTS_QGIS = REPO / "tests_qgis"
if str(TESTS_QGIS) not in sys.path:
    sys.path.insert(0, str(TESTS_QGIS))

run_all = pytest.importorskip("run_all")


class _FakeStdout:
    """A pipe that yields *lines*, then blocks for ever — a hung check."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._blocked = threading.Event()

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._blocked.wait()        # the check that never returns
        return ""

    def release(self):
        self._blocked.set()


class _FakeProc:
    def __init__(self, lines):
        self.stdout = _FakeStdout(lines)
        self.returncode = None
        self.killed = False

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stdout.release()

    def wait(self, timeout=None):
        return self.returncode


class TestTimeoutNamesTheCheck:
    """T-4. Every TIMEOUT used to report "(before the first check)".

    The worker prints ``"  name ... "`` with no newline, so ``readline`` is still
    blocked on it; when the line finally does arrive it ends in ``ok`` or ``FAIL``,
    never in ``...``. The test the orchestrator applied could therefore never match,
    and the name of the hung check — the whole point of the line, per the comment
    above it — was never available. A 300 s quarantine was carried for a check
    nobody could name.
    """

    def _run(self, monkeypatch, lines, timeout_s=1.0):
        proc = _FakeProc(lines)
        monkeypatch.setattr(run_all.subprocess, "Popen",
                            lambda *a, **kw: proc)
        started = time.monotonic()
        passed, failed, body = run_all.run_module_isolated(
            "checks_stub", [], timeout_s, "unused-dem-path")
        proc.stdout.release()
        return passed, failed, "\n".join(body), time.monotonic() - started

    def test_a_hung_check_is_named(self, monkeypatch):
        lines = [
            run_all.started_line("check_one") + "\n",
            "  check_one ... ok\n",
            run_all.started_line("check_two") + "\n",
            "  check_two ... ok\n",
            run_all.started_line("check_three") + "\n",
            # ... and then nothing: check_three never returns.
        ]
        passed, failed, body, elapsed = self._run(monkeypatch, lines)

        assert "check_three" in body, (
            f"the timeout did not name the check that hung:\n{body}")
        assert "(before the first check)" not in body
        assert failed == 1
        assert elapsed < 20, "the orchestrator did not honour its own deadline"

    def test_a_timeout_before_any_check_still_says_so(self, monkeypatch):
        """The honest form of the old message, kept for the case it describes."""
        _passed, failed, body, _elapsed = self._run(monkeypatch, [])
        assert "(before the first check)" in body
        assert failed == 1

    def test_the_marker_is_not_echoed_to_the_reader(self, monkeypatch, capsys):
        """Protocol, not output: the run must look exactly as it did before."""
        lines = [
            run_all.started_line("check_one") + "\n",
            "  check_one ... ok\n",
            run_all.started_line("check_two") + "\n",
        ]
        self._run(monkeypatch, lines)
        printed = capsys.readouterr().out
        assert run_all.started_line("check_one") not in printed
        assert "  check_one ... ok" in printed


class TestMarkerPair:
    """The two halves of the protocol have to agree, and they live together."""

    def test_round_trip(self):
        assert run_all.check_started(run_all.started_line("check_x")) == "check_x"

    @pytest.mark.parametrize("line", [
        "  check_x ... ok",
        "  check_x ... FAIL",
        "  check_x ... ",
        "RESULT checks_x 3 0",
        "--- checks_x.check_y",
        "",
    ])
    def test_ordinary_output_is_not_a_marker(self, line):
        assert run_all.check_started(line) is None


def test_the_worker_announces_every_check_before_running_it():
    """The emitting half, read off the source.

    The orchestrator test above proves the parser; this proves the worker actually
    sends what the parser expects. Both halves failing together is precisely the
    shape T-4 had — a rule that looked enforced from either end alone.
    """
    import ast

    src = (TESTS_QGIS / "run_all.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    worker = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "run_worker")
    loop = next(n for n in ast.walk(worker)
                if isinstance(n, ast.For)
                and isinstance(n.target, ast.Tuple)
                and [e.id for e in n.target.elts] == ["name", "fn"])

    calls = [n for n in ast.walk(loop)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    names = [c.func.id for c in calls]
    assert "started_line" in names, (
        "run_worker's check loop never calls started_line(), so a timeout can "
        "still not name the check that hung"
    )
