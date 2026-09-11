"""Shared plumbing for the `p_*.py` probes: path setup, fixture choice, evidence files.

Kept separate from `_harness.py` because it is not test infrastructure — nothing here is
imported by a `checks_*` module, and `_harness.py` has no scope (touching it means the
full QGIS run). The leading underscore also keeps it out of `run_all.py`'s
`checks_*.py` glob, though this whole directory is already invisible to it.

A probe **records**; it does not assert. `Evidence.note` takes a surprise and writes it
down. Nothing here raises on a wrong number — only on a broken environment, and only from
`stage()`, which names the stage that failed and keeps going.
"""

import datetime
import faulthandler
import json
import os
import sys
import traceback
from pathlib import Path

# A PyQGIS access violation kills the process with an exit code and nothing else —
# `run_all.py:107` installs this for the same reason. Without it, a probe that touches
# Qt before `qgis_app()` exists simply stops mid-line.
faulthandler.enable()

HERE = Path(__file__).resolve().parent
TESTS_QGIS = HERE.parent
REPO = TESTS_QGIS.parent
EVIDENCE = HERE / "evidence"

# `_harness` lives one directory up and is imported by module name, exactly as every
# `checks_*` module does it. Probes are run directly (`python p_smoke.py`), so the
# interpreter puts `probes/` on the path, not `tests_qgis/`.
if str(TESTS_QGIS) not in sys.path:
    sys.path.insert(0, str(TESTS_QGIS))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_FIXTURE = REPO / "tests" / "fixtures" / "quail_island_catchment.tif"


def fixture_path(argv=None):
    """The DEM to probe: `--fixture=PATH`, else the committed Quail Island clip.

    A *missing* explicit path is a hard error rather than a silent fall back to the
    synthetic surface — the same rule `run_qgis_tests.ps1` adopted on 2026-09-10, and for
    the same reason: a probe that quietly measured the smooth synthetic DEM while its
    output said "fixture" would poison every number downstream of it.
    """
    argv = sys.argv[1:] if argv is None else argv
    for arg in argv:
        if arg.startswith("--fixture="):
            p = Path(arg.split("=", 1)[1]).expanduser()
            if not p.exists():
                raise SystemExit(f"--fixture path does not exist: {p}")
            return p
    if not DEFAULT_FIXTURE.exists():
        raise SystemExit(f"default fixture is missing: {DEFAULT_FIXTURE}")
    return DEFAULT_FIXTURE


def today():
    return datetime.date.today().isoformat()


class Evidence:
    """A probe's result file, and its running commentary on stdout.

    `data` is an ordinary dict written verbatim to `evidence/<name>.json`. The envelope
    (`probe`, `date`, `findings`, `dem`, `notes`, `stages`) is fixed so a register entry
    can cite a file and a reader knows where to look without opening five formats.
    """

    def __init__(self, name, findings, dem=None):
        self.name = name
        self.data = {
            "probe": name,
            "date": today(),
            "findings": list(findings),
            "dem": str(dem) if dem else None,
            "notes": [],
            "stages": {},
        }

    # -------------------------------------------------------------- recording

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def note(self, text):
        """Something worth saying in the register. Printed and kept."""
        print(f"    NOTE {text}", flush=True)
        self.data["notes"].append(text)

    def stage(self, label):
        """Context manager: run a measurement, record whether it stood up.

        A raising stage is recorded with its traceback and the probe continues, because
        the stages after it usually still answer their own questions. A probe that died
        on stage two and wrote nothing tells you less than one that wrote six of eight.
        """
        return _Stage(self, label)

    # ---------------------------------------------------------------- writing

    def write(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        path = EVIDENCE / f"{self.name}.json"
        path.write_text(
            json.dumps(self.data, indent=2, sort_keys=False, default=_plain) + "\n",
            encoding="utf-8",
        )
        failed = [k for k, v in self.data["stages"].items() if v.get("status") != "ok"]
        print(f"\n  evidence -> {path}")
        if failed:
            print(f"  STAGES THAT DID NOT COMPLETE: {', '.join(failed)}")
        return path


class _Stage:
    def __init__(self, ev, label):
        self.ev = ev
        self.label = label

    def __enter__(self):
        print(f"\n  [{self.label}]", flush=True)
        self.ev.data["stages"][self.label] = {"status": "running"}
        return self.ev.data["stages"][self.label]

    def __exit__(self, exc_type, exc, tb):
        rec = self.ev.data["stages"][self.label]
        if exc is None:
            rec["status"] = "ok"
            return False
        rec["status"] = "error"
        rec["error"] = f"{exc_type.__name__}: {exc}"
        rec["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))
        print(f"    FAILED {rec['error']}", flush=True)
        return True  # swallowed: the remaining stages still have questions to answer


def _plain(obj):
    """JSON fallback for numpy scalars and Paths, which json.dumps will not take."""
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def dem_stats(path):
    """Cell size, shape and nodata for *path*, read with rasterio.

    Reported by every probe so an evidence file is self-describing: "0.2 ha" means a
    different number of cells on the 1 m fixture and the 2 m synthetic DEM, and that
    distinction is the whole of KPA-46.
    """
    import numpy as np
    import rasterio

    with rasterio.open(str(path)) as src:
        a = src.read(1, masked=False).astype("float64")
        nodata = src.nodata
        cell_w = abs(src.transform.a)
        cell_h = abs(src.transform.e)
        finite = np.isfinite(a)
        if nodata is not None:
            finite &= a != nodata
        return {
            "path": str(path),
            "shape": [int(src.height), int(src.width)],
            "cell_w_m": cell_w,
            "cell_h_m": cell_h,
            "cell_area_m2": cell_w * cell_h,
            "crs": str(src.crs),
            "nodata": None if nodata is None else float(nodata),
            "nodata_cells": int((~finite).sum()),
            "z_min": float(a[finite].min()) if finite.any() else None,
            "z_max": float(a[finite].max()) if finite.any() else None,
            "default_stream_threshold_cells": max(
                20, int(round(2_000.0 / (cell_w * cell_h)))),
            "default_stream_threshold_ha": (
                max(20, int(round(2_000.0 / (cell_w * cell_h))))
                * cell_w * cell_h / 10_000.0),
        }


def start_qgis():
    """Boot QGIS before anything constructs a Qt object, and return `_harness`.

    `run_all.py` calls `_harness.qgis_app()` before it imports a check module
    (`:115`); a probe run directly has no runner to do it. Skipping it does not raise —
    building a `QgsProject` with no `QgsApplication` takes the whole process down with a
    bare exit code, which is exactly the silent-subprocess death this suite has been bitten
    by before. Call this in any probe that touches `PluginHarness` or `qgis.core`.
    """
    import _harness

    _harness.qgis_app()
    return _harness


def workdir(name):
    """A scratch directory for a probe's intermediate rasters.

    Under `probes/evidence/_work/` rather than the system temp dir, because CTA-30 is a
    finding about temp directories nobody deletes and it would be poor manners to add to
    the pile while documenting it. Contents are overwritten on each run.
    """
    d = EVIDENCE / "_work" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def banner(title, dem):
    print("=" * 78)
    print(f"  {title}")
    print(f"  DEM: {dem}")
    print(f"  {today()}  qgis python: {os.path.basename(sys.executable)}")
    print("=" * 78, flush=True)
