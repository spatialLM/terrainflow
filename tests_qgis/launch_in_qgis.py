"""
launch_in_qgis.py — drive the plugin inside GENUINE QGIS and screenshot it.

Run via run_qgis_gui_shot.ps1, which invokes:

    qgis-ltr.bat --profile TerrainFlowTest --noplugins --nologo --code launch_in_qgis.py

This is the fidelity the offscreen harness cannot give: a real QgisInterface, the
real dock layout inside the real main window, the real QGIS theme and UI font, and
the real map canvas bridged to the layer tree. What it gives up is reliability —
it depends on GUI startup timing — so it is a deliberate second opinion on top of
`run_qgis_tests.ps1`, not a replacement for it.

Two isolation choices matter:
  --profile TerrainFlowTest  keeps every setting out of the user's working profile
  --noplugins                stops the *deployed* copy of TerrainFlow loading too,
                             which would otherwise run a second panel alongside
                             the one this script builds from the repo

Results go to tests_qgis/_shots/gui_report.txt so the run can be read afterwards
rather than watched. QGIS quits itself unless TFA_GUI_KEEP=1.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

# QGIS may exec --code without defining __file__, in which case touching it raises
# a NameError that QGIS swallows: the script simply never runs and nothing says so.
# run_qgis_gui_shot.ps1 passes the directory explicitly as a fallback.
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path(os.environ["TFA_TESTS_DIR"]).resolve()

for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SHOTS = HERE / "_shots"
REPORT = SHOTS / "gui_report.txt"
KEEP_OPEN = os.environ.get("TFA_GUI_KEEP") == "1"

# Breadcrumb: proves --code reached this file at all. If the run ends with only
# this line on disk, the failure is between here and drive() firing.
SHOTS.mkdir(parents=True, exist_ok=True)
REPORT.write_text("status: script loaded, drive() not yet run\n", encoding="utf-8")

_lines = []
_plugin = None   # kept alive so teardown can unload it before QGIS exits


def log(message):
    _lines.append(message)
    print(f"[terrainflow-gui] {message}")


def write_report(status):
    SHOTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"status: {status}\n\n" + "\n".join(_lines) + "\n", encoding="utf-8"
    )


def drive():
    """Everything happens here, after QGIS has finished starting up."""
    status = "ok"
    try:
        import _harness
        from qgis.core import (
            Qgis,
            QgsCoordinateReferenceSystem,
            QgsProject,
            QgsRasterLayer,
        )
        from qgis.PyQt.QtCore import QCoreApplication
        from qgis.utils import iface

        log(f"QGIS version: {Qgis.QGIS_VERSION}")
        log(f"font families: {_font_count()}")

        _harness.make_workers_synchronous()

        dem_path = _harness.build_synthetic_dem(
            Path(tempfile.mkdtemp(prefix="tfa_gui_")) / "synthetic_dem.tif"
        )
        log(f"DEM: {dem_path}")

        project = QgsProject.instance()
        project.setCrs(QgsCoordinateReferenceSystem("EPSG:2193"))

        dem_layer = QgsRasterLayer(dem_path, "Test DEM")
        assert dem_layer.isValid(), "synthetic DEM failed to load"
        project.addMapLayer(dem_layer)

        boundary = _harness.site_boundary_layer()
        project.addMapLayer(boundary)

        # The genuine article: the real QgisInterface, not a stub.
        from terrainflow_assessment.qgis.plugin import TerrainFlowAssessmentPlugin

        global _plugin
        plugin = TerrainFlowAssessmentPlugin(iface)
        plugin.initGui()
        _plugin = plugin
        panel = plugin.panel
        log("plugin loaded against the real iface; panel docked")

        panel.dem_changed.emit(dem_layer)
        panel.boundary_changed.emit(boundary)
        panel.run_baseline_requested.emit()

        result = plugin._state.baseline_result
        if result is None:
            status = "baseline produced no result"
            log(f"FAIL: {status}")
        else:
            log(
                "baseline ok - runoff "
                f"{result.get('runoff_mm', 0):.1f} mm, "
                f"{len(result.get('exit_points', []))} exit point(s), "
                f"{len(plugin._state.baseline_layer_ids)} result layer(s)"
            )

        # A fresh profile has no saved window geometry, so QGIS opens small and the
        # layer tree / panel get cropped out of the screenshot.
        window = iface.mainWindow()
        window.resize(1920, 1200)
        for _ in range(3):
            QCoreApplication.processEvents()

        # Renderer class per result layer: the offscreen harness asserts these are
        # pseudocolour ramps, and a real QGIS profile can apply its own default
        # raster styling. Logging it makes any divergence visible rather than a
        # judgement call from a screenshot.
        for lid in plugin._state.baseline_layer_ids:
            layer = project.mapLayer(lid)
            if layer is not None:
                log(f"  renderer: {layer.name()} -> {type(layer.renderer()).__name__}")

        canvas = iface.mapCanvas()
        canvas.setExtent(dem_layer.extent())
        canvas.refresh()
        canvas.waitWhileRendering()
        for _ in range(5):
            QCoreApplication.processEvents()

        SHOTS.mkdir(parents=True, exist_ok=True)
        for name, widget in (("gui_main_window", window), ("gui_panel", panel)):
            path = SHOTS / f"{name}.png"
            if widget.grab().save(str(path)):
                log(f"shot: {path.name}")
            else:
                log(f"FAIL: could not write {path.name}")
                status = f"screenshot failed: {name}"

        canvas_path = SHOTS / "gui_canvas.png"
        canvas.saveAsImage(str(canvas_path))
        log(f"shot: {canvas_path.name}" if canvas_path.exists()
            else "FAIL: canvas produced no image")

    except Exception:
        status = "exception"
        log("EXCEPTION:\n" + traceback.format_exc())

    write_report(status)

    if not KEEP_OPEN:
        _teardown_then_quit(status)


def _teardown_then_quit(status):
    """Unload the plugin and empty the project BEFORE asking QGIS to exit.

    Quitting with the plugin still loaded segfaults on shutdown: QGIS tears the
    project down (QgsProject::clear -> removeAllMapLayers -> layer-tree teardown)
    while the plugin still holds its dock widget, its cached layer ids and its
    earthworks layer-tree group, and the teardown walks into freed objects. Live
    QGIS never hits this because it unloads plugins first; this script has to do
    that itself.
    """
    from qgis.core import QgsApplication, QgsProject
    from qgis.PyQt.QtCore import QTimer

    global _plugin
    try:
        if _plugin is not None:
            _plugin.unload()
            _plugin = None
            log("plugin unloaded")
        QgsProject.instance().clear()
        log("project cleared")
    except Exception:
        log("teardown problem:\n" + traceback.format_exc())
        status = "teardown failed"

    log("quitting QGIS")
    write_report(status)
    QTimer.singleShot(300, QgsApplication.instance().quit)


def _font_count():
    try:
        from qgis.PyQt.QtGui import QFontDatabase

        try:
            return len(QFontDatabase.families())
        except TypeError:
            return len(QFontDatabase().families())
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Startup: wait until QGIS is actually ready, rather than guessing a delay
# ---------------------------------------------------------------------------

READY_TIMEOUT_S = 120     # give up rather than block a run forever
POLL_MS = 250
SETTLE_MS = 1500          # after readiness, let providers finish loading

# The window reports isVisible() almost immediately, so readiness alone would fire
# barely later than --code itself. Hold a floor as well: the fixed 5 s delay this
# replaced did survive a cold first-profile start, and the point of the poll is to
# cope with machines that need *longer*, not to fire sooner.
MIN_WAIT_S = 4.0

_poll_timer = None
_waited_s = 0.0


def _qgis_ready():
    """True once qgis.utils.iface exists and its main window is up.

    qgis.utils.iface is populated during startup, so it must be read fresh on
    every poll -- a module-level `from qgis.utils import iface` would bind None
    forever.
    """
    try:
        import qgis.utils

        iface = getattr(qgis.utils, "iface", None)
        if iface is None:
            return False
        window = iface.mainWindow()
        if window is None or not window.isVisible():
            return False
        # A laid-out window, not merely a shown one, and a canvas to draw on.
        if window.width() < 400 or window.height() < 300:
            return False
        return iface.mapCanvas() is not None
    except Exception:
        return False


def _poll_until_ready():
    global _waited_s

    if _waited_s >= MIN_WAIT_S and _qgis_ready():
        _poll_timer.stop()
        log(f"QGIS ready after {_waited_s:.1f}s; settling {SETTLE_MS} ms")
        QTimer.singleShot(SETTLE_MS, drive)
        return

    _waited_s += POLL_MS / 1000.0
    if _waited_s >= READY_TIMEOUT_S:
        _poll_timer.stop()
        log(f"FAIL: QGIS never became ready within {READY_TIMEOUT_S}s")
        _teardown_then_quit("qgis never became ready")


# QGIS runs --code during startup, before the main window is built and laid out.
# Grabbing it then yields a half-constructed window, so poll for readiness instead
# of sleeping on a guess.
from qgis.PyQt.QtCore import QTimer  # noqa: E402

_poll_timer = QTimer()
_poll_timer.setInterval(POLL_MS)
_poll_timer.timeout.connect(_poll_until_ready)
_poll_timer.start()
