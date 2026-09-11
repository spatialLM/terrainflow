"""
_harness.py — real-QGIS smoke-test harness (headless).

Lives OUTSIDE terrainflow_assessment/ on purpose: only that folder is deployed or
zipped, so nothing here can reach a shipped build.

Why this exists: tests/conftest.py replaces qgis.core / qgis.gui with MagicMocks so
the pure modules/ logic can be tested without QGIS. That means the whole
qgis/ layer — controllers, adapters, renderers, workers — is never executed by
`pytest tests/`. This harness boots a genuine QgsApplication, builds the real
panel and controllers, and drives them through the real panel signals.

Run it with QGIS's own Python (see run_qgis_tests.ps1); it will not work under a
plain interpreter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must be set before QgsApplication is constructed. Widgets are created for real
# (the panel is a QDockWidget) but render to an offscreen surface, so the run is
# headless without giving up widget fidelity.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _configure_qt_platform():
    """Choose the Qt platform + fonts. Must run before any QApplication exists.

    Called from qgis_app() rather than at import time so that launch_in_qgis.py can
    reuse this module inside a *real* QGIS process, where the application already
    exists and forcing offscreen would be wrong.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Qt's offscreen platform on Windows registers ZERO font families, so every
    # drawText() is a silent no-op: widget screenshots come out with layout and
    # colours intact but not one glyph, and QGIS map labelling renders nothing.
    # Pointing Qt at the system font directory restores it (233 families here).
    if os.environ["QT_QPA_PLATFORM"] == "offscreen" and "QT_QPA_FONTDIR" not in os.environ:
        for fontdir in (r"C:\Windows\Fonts", "/usr/share/fonts"):
            if os.path.isdir(fontdir):
                os.environ["QT_QPA_FONTDIR"] = fontdir
                break


# ---------------------------------------------------------------------------
# QGIS application bootstrap
# ---------------------------------------------------------------------------

_APP = None


def qgis_app():
    """Boot (once) and return the QgsApplication. Idempotent."""
    global _APP
    if _APP is not None:
        return _APP

    _configure_qt_platform()

    from qgis.core import QgsApplication

    prefix = os.environ.get("QGIS_PREFIX_PATH")
    if prefix:
        QgsApplication.setPrefixPath(prefix, True)

    app = QgsApplication([], True)
    # Send any QSettings this run touches to a throwaway org/app name rather than
    # the user's real QGIS configuration.
    app.setOrganizationName("TerrainFlowTests")
    app.setApplicationName("TerrainFlowTests")
    app.initQgis()
    _init_processing()

    _APP = app
    return app


def _init_processing():
    """Register the Processing providers — contour.py calls gdal:contour."""
    from qgis.core import QgsApplication

    plugins_dir = os.path.join(QgsApplication.prefixPath(), "python", "plugins")
    if os.path.isdir(plugins_dir) and plugins_dir not in sys.path:
        sys.path.append(plugins_dir)

    from processing.core.Processing import Processing

    Processing.initialize()

    # Processing.initialize() normally registers the native provider itself; add it
    # only if it is genuinely absent, since adding twice logs spurious warnings.
    if QgsApplication.processingRegistry().providerById("native") is None:
        from qgis.analysis import QgsNativeAlgorithms

        QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())


def make_workers_synchronous():
    """Run the QThread workers inline so checks finish deterministically.

    The controllers connect progress/finished/error *before* calling start(), so
    replacing start() with run() delivers exactly the same signals on the main
    thread — no event loop to spin and no timeout to tune.
    """
    from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker
    from terrainflow_assessment.qgis.workers.simulation_worker import SimulationWorker
    from terrainflow_assessment.qgis.workers.task_worker import TaskWorker

    for cls in (AnalysisWorker, SimulationWorker, TaskWorker):
        if getattr(cls, "_tf_sync_patched", False):
            continue
        cls.start = cls.run
        cls._tf_sync_patched = True


# ---------------------------------------------------------------------------
# Synthetic DEM
# ---------------------------------------------------------------------------

CELL_M = 2.0
# 300x300 @ 2 m = 36 ha. Sized deliberately: the panel's default stream threshold
# is 5 ha and, under D-infinity routing on this terrain, the most-accumulated cell
# only ever gathers ~35 % of the site. A smaller site therefore yields a
# legitimately empty stream layer and the stream renderer is never exercised.
NROWS = 300
NCOLS = 300
ORIGIN_X = 1_750_000.0   # NZTM2000 — the plugin's target projection
ORIGIN_Y = 5_900_000.0


NODATA = -9999.0


def _box_blur(arr, passes):
    """Cheap separable 3x3 mean filter, `passes` times. No scipy needed.

    Turns white noise into *spatially correlated* roughness, and that
    distinction is the whole point. White noise at a realistic LiDAR vertical
    accuracy (~0.1 m) on a 2 m cell produces local gradients around 0.05-0.10 —
    the same order as this fixture's 15 % valley slope. Flow routing then
    dissolves into thousands of one-cell sinks and every check fails for a
    reason that has nothing to do with the plugin. Real terrain is rough but
    correlated between neighbours; blurring reproduces that.
    """
    import numpy as np

    out = arr
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:]
            + padded[1:-1, 1:-1]
        ) / 5.0
    return out


def build_synthetic_dem(path, pond=False, rough=False, pits=0, voids=0,
                        seed=42, roughness_m=0.10):
    """Write a deterministic 300x300 @ 2 m DEM (36 ha) in EPSG:2193.

    Shape: a valley draining south, with a concave long profile (~15 % at the top
    easing to ~3 % at the outlet) so keypoint analysis has a real slope break to
    find, and a parabolic cross-section so flow concentrates on the centreline
    and accumulation/exit-point detection has something to detect.

    Both coefficients are constrained, not arbitrary: the long-profile quadratic
    must keep its slope positive across the full 400 m (otherwise the bottom of
    the site tilts back uphill and becomes a false depression), and the
    cross-section rise must stay well under the total longitudinal drop (or the
    valley walls close off basins along the edges).

    ``pond=True`` cuts a basin into the channel so one forms anyway. Everything above is
    written to be depression-free, which means the default surface exercises **nothing** of
    the crest split — no pond, no contraction, nothing for Pond Capacity to draw — and a
    green run over it says nothing about them. The basin is 30 rows x 13 columns at 4 m deep, well
    over ``crest_routing.MIN_POND_CELLS``, and it fills and spills over its downstream lip.

    **The nasty variant.** ``rough``/``pits``/``voids`` deliberately spoil the surface.
    The default is pathologically *kind* — smooth, depression-free and hole-free — so
    pit filling, nodata propagation and anything that has to cope with a jagged contour
    are never exercised by the 259 checks that run over it. Those are exactly the
    failure modes real farm LiDAR would find::

        build_synthetic_dem(p)                              # smooth: 0 sinks
        build_synthetic_dem(p, rough=True, pits=6, voids=2) # 18 sinks, 180 nodata cells

    Synthesised rather than a committed real DEM so ground truth survives: you know
    how many pits you punched and where. Everything random comes from
    ``np.random.default_rng(seed)`` with a fixed default, so two builds are
    byte-identical and the screenshot baseline still holds.

    **The default is unchanged on purpose.** Every existing check and all 49 baseline
    images are calibrated against the smooth surface; these are opt-in and off unless
    asked for. See ``checks_robustness.py``.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    rows = np.arange(NROWS, dtype="float64")[:, None]
    cols = np.arange(NCOLS, dtype="float64")[None, :]

    down = rows * CELL_M                                # m downslope from top edge
    across = (cols - (NCOLS - 1) / 2.0) * CELL_M        # m from the centreline

    z = 100.0 - (0.15 * down - 0.00010 * down**2)       # concave long profile
    z = z + 0.0003 * across**2                          # parabolic cross-section

    # Incised channel on the centreline. Without it the valley is so smooth that
    # D-infinity routing (the panel's default) disperses flow laterally and no cell
    # ever accumulates enough upstream area to register as a stream — the stream
    # renderer then has nothing to draw. Real terrain has a channel; so does this.
    z = z - 4.0 * np.exp(-((across / 5.0) ** 2))

    # Materialise once: every modifier below writes into z in place. .copy()
    # rather than np.ascontiguousarray() — the latter returns the input untouched
    # when it is already contiguous, so the read-only broadcast view stays
    # read-only and the first `z +=` raises "output array is read-only".
    z = np.broadcast_to(z, (NROWS, NCOLS)).copy()

    if pond:
        basin = ((rows >= 120) & (rows < 150)
                 & (np.abs(cols - (NCOLS - 1) / 2.0) <= 6))
        z[np.broadcast_to(basin, (NROWS, NCOLS))] -= 4.0

    rng = np.random.default_rng(seed)
    nodata_mask = np.zeros((NROWS, NCOLS), dtype=bool)

    if rough:
        # Normalise AFTER blurring: each pass shrinks the standard deviation, so
        # scaling first would leave the amplitude at the mercy of the pass count.
        noise = _box_blur(rng.normal(0.0, 1.0, (NROWS, NCOLS)), passes=4)
        noise *= roughness_m / noise.std()
        z += noise

    if pits:
        # Genuine closed depressions, which the smooth surface has none of. Kept
        # off the bottom rows so the outlet stays open, and off the very edge so
        # each is fully enclosed rather than draining off-grid.
        for _ in range(int(pits)):
            r = int(rng.integers(20, NROWS - 60))
            c = int(rng.integers(20, NCOLS - 20))
            radius = float(rng.uniform(3.0, 6.0))          # cells
            depth = float(rng.uniform(1.0, 2.5))           # metres
            z -= depth * np.exp(
                -(((rows - r) ** 2 + (cols - c) ** 2) / (2.0 * radius**2)))

    if voids:
        # Nodata holes, as left by water bodies or removed structures. Kept off
        # the centreline: a void straight through the channel severs the flow
        # path, so every downstream result would be legitimately empty and the
        # fixture would test nothing rather than test robustness.
        for _ in range(int(voids)):
            r = int(rng.integers(20, NROWS - 40))
            c = int(rng.integers(15, NCOLS - 15))
            if abs(c - (NCOLS - 1) / 2.0) < 15:
                c = int((NCOLS - 1) / 2.0) + int(rng.choice([-1, 1])) * 30
            half = int(rng.integers(3, 7))
            nodata_mask[max(0, r - half):r + half,
                        max(0, c - half):c + half] = True

    grid = z.astype("float32")
    grid[nodata_mask] = NODATA

    transform = from_origin(ORIGIN_X, ORIGIN_Y, CELL_M, CELL_M)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=NROWS, width=NCOLS, count=1, dtype="float32",
        crs="EPSG:2193", transform=transform, nodata=NODATA,
    ) as dst:
        dst.write(grid, 1)

    return str(path)


def cropped_dem(src_path, dst_path, top=20, left=20, rows=200, cols=200):
    """A window of *src_path* written to *dst_path* — a genuinely different grid.

    `BaselineController._grid_moved` compares shape **and** origin, so two DEMs of
    the same 300x300 extent are the same grid to it however much their values
    differ, and swapping one for the other invalidates nothing. A crop moves both at
    once, which is what a user actually does: clip a survey to the block being
    designed and point the picker at the clip.

    Deliberately a crop of the fixture rather than a second synthetic surface. The
    ground under the overlap is *identical*, so a measurement that survives the swap
    and still looks plausible is surviving because nothing cleared it — not because
    the two terrains happen to agree.
    """
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src_path) as src:
        window = Window(left, top, cols, rows)
        data = src.read(1, window=window)
        profile = src.profile.copy()
        profile.update(height=rows, width=cols,
                       transform=src.window_transform(window))

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)
    return str(dst_path)


def centreline_x():
    return ORIGIN_X + ((NCOLS - 1) / 2.0) * CELL_M


def site_boundary_layer(inset_m=10.0):
    """An in-memory site boundary inset from the DEM edge.

    Baseline exit points are only detected where flow crosses a boundary polygon
    (see AnalysisWorker: exit_points stays empty without boundary_path), so the
    boundary is what makes outflow assertions meaningful. Deliberately a memory
    layer: that also exercises the controller's own _layer_to_path() conversion.
    """
    from qgis.core import QgsFeature, QgsGeometry, QgsRectangle, QgsVectorLayer

    layer = QgsVectorLayer("Polygon?crs=EPSG:2193", "Site Boundary", "memory")
    rect = QgsRectangle(
        ORIGIN_X + inset_m,
        ORIGIN_Y - NROWS * CELL_M + inset_m,
        ORIGIN_X + NCOLS * CELL_M - inset_m,
        ORIGIN_Y - inset_m,
    )
    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromRect(rect))
    layer.dataProvider().addFeatures([feat])
    layer.updateExtents()

    # QGIS assigns a new vector layer a random pastel fill, which washes the whole
    # site in an arbitrary colour and makes canvas screenshots unreadable (and
    # non-deterministic between runs). Outline only.
    from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer

    symbol = QgsFillSymbol.createSimple({
        "style": "no",
        "outline_color": "70,70,70,255",
        "outline_width": "0.3",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))
    return layer


def line_across_valley(row=60, half_width_m=40.0):
    """A QgsGeometry line crossing the valley — a plausible swale alignment."""
    from qgis.core import QgsGeometry, QgsPointXY

    cx = centreline_x()
    y = ORIGIN_Y - row * CELL_M
    return QgsGeometry.fromPolylineXY([
        QgsPointXY(cx - half_width_m, y),
        QgsPointXY(cx, y),
        QgsPointXY(cx + half_width_m, y),
    ])


# ---------------------------------------------------------------------------
# iface stub — the plugin uses only messageBar / mainWindow / mapCanvas
# plus the toolbar/menu/dock registration pairs.
# ---------------------------------------------------------------------------

class RecordingMessageBar:
    """Captures pushed messages so a check can assert nothing errored."""

    def __init__(self):
        self.messages = []   # list of (level, title, text)

    def _push(self, level, title, text):
        self.messages.append((level, str(title), str(text)))

    def pushInfo(self, title, text):
        self._push("info", title, text)

    def pushSuccess(self, title, text):
        self._push("success", title, text)

    def pushWarning(self, title, text):
        self._push("warning", title, text)

    def pushCritical(self, title, text):
        self._push("critical", title, text)

    def pushMessage(self, *args, **kwargs):
        title = args[0] if args else ""
        text = args[1] if len(args) > 1 else ""
        self._push("message", title, text)

    def clearWidgets(self):
        pass

    def of(self, level):
        return [m for m in self.messages if m[0] == level]

    @property
    def criticals(self):
        return self.of("critical")

    @property
    def warnings(self):
        return self.of("warning")

    def render(self):
        if not self.messages:
            return "    (message bar empty)"
        return "\n".join(f"    [{lvl}] {t}: {m}" for lvl, t, m in self.messages)


class RecordingDialogs:
    """Captures modal message boxes instead of showing them.

    A ``QMessageBox`` shown offscreen has nobody to click OK, so it blocks until the
    module's timeout. That is the whole of why ``check_recommend_ponds`` was quarantined
    as "does not finish, ran >12 min, no result and no error": it emitted the pond
    recommendation without running the keypoint pass first, and the controller's guard
    put up *"Run 'Find Keypoints + Ridgelines' first."* and waited. Stubbed here it
    returns in 0.00 s.

    The general hazard is worse than the one check. Both keypoint controllers end their
    ``except`` blocks with ``QMessageBox.critical``, so a genuine exception in them
    presented as a **timeout rather than an error** — the failure mode that hides its own
    cause. Recorded, it becomes a failed assertion with the traceback attached.

    Nothing that passes today can regress: a check that showed a dialog could not
    previously have finished at all.
    """

    _KINDS = ("information", "warning", "critical", "question", "about")

    def __init__(self):
        self.all = []          # list of (kind, title, text)
        self.answer = None     # what question() should return, if a check needs a say
        self._saved = {}

    def install(self):
        from qgis.PyQt.QtWidgets import QMessageBox

        for kind in self._KINDS:
            self._saved[kind] = getattr(QMessageBox, kind)
            setattr(QMessageBox, kind, staticmethod(self._recorder(kind)))

    def restore(self):
        from qgis.PyQt.QtWidgets import QMessageBox

        for kind, original in self._saved.items():
            setattr(QMessageBox, kind, original)
        self._saved.clear()

    def _recorder(self, kind):
        def _record(parent=None, title="", text="", *args, **kwargs):
            from qgis.PyQt.QtWidgets import QMessageBox

            self.all.append((kind, str(title), str(text)))
            if kind == "about":
                return None
            if kind == "question":
                # Default No: nothing a check has not asked for should proceed.
                return self.answer if self.answer is not None else QMessageBox.No
            return QMessageBox.Ok

        return _record

    def of(self, kind):
        return [d for d in self.all if d[0] == kind]

    @property
    def blocking(self):
        """The ones that mean something went wrong, rather than merely informing."""
        return [d for d in self.all if d[0] in ("warning", "critical")]

    def render(self):
        if not self.all:
            return "    (no dialogs)"
        return "\n".join(f"    [{k}] {t}: {m}" for k, t, m in self.all)


class StubIface:
    """Minimal QgisInterface stand-in covering every iface call the plugin makes."""

    def __init__(self, main_window, canvas):
        self._main_window = main_window
        self._canvas = canvas
        self._message_bar = RecordingMessageBar()
        self.toolbar_actions = []
        self.menu_items = []       # (menu_name, action)
        self.dock_widgets = []

    def mainWindow(self):
        return self._main_window

    def mapCanvas(self):
        return self._canvas

    def messageBar(self):
        return self._message_bar

    def addToolBarIcon(self, action):
        self.toolbar_actions.append(action)

    def removeToolBarIcon(self, action):
        if action in self.toolbar_actions:
            self.toolbar_actions.remove(action)

    def addPluginToMenu(self, menu, action):
        self.menu_items.append((menu, action))

    def removePluginMenu(self, menu, action):
        if (menu, action) in self.menu_items:
            self.menu_items.remove((menu, action))

    def addDockWidget(self, area, widget):
        # Actually dock it, not just record it: the panel needs a real parent and a
        # real layout pass before it can be rendered to an image (see shots.py).
        self._main_window.addDockWidget(area, widget)
        self.dock_widgets.append(widget)

    def removeDockWidget(self, widget):
        self._main_window.removeDockWidget(widget)
        if widget in self.dock_widgets:
            self.dock_widgets.remove(widget)


# ---------------------------------------------------------------------------
# Plugin harness
# ---------------------------------------------------------------------------

class PluginHarness:
    """Builds the real plugin against a stub iface, on a real QgsProject.

    Used as a context manager so every check starts from an empty project and
    unloads cleanly::

        with PluginHarness(dem_path) as h:
            h.run_baseline()
            h.assert_no_errors("baseline")
    """

    #: What the DEM fixture is written in. The project is pinned to the same thing
    #: by default, which is realistic and is also a blind spot: any layer created
    #: in the project's CRS instead of the DEM's draws in exactly the right place,
    #: and any scale bar measured off a project extent reads in metres. Pass
    #: ``project_crs`` to break that coincidence — see `checks_crs.py`.
    DEM_CRS = "EPSG:2193"

    def __init__(self, dem_path=None, load_dem=True, load_boundary=True,
                 project_crs=None):
        self.dem_path = dem_path
        self._load_dem = load_dem and dem_path is not None
        self._load_boundary = load_boundary and self._load_dem
        self.project_crs = project_crs or self.DEM_CRS
        self.dem_layer = None
        self.boundary_layer = None

    def __enter__(self):
        from qgis.core import QgsCoordinateReferenceSystem, QgsProject
        from qgis.gui import QgsMapCanvas
        from qgis.PyQt.QtGui import QColor
        from qgis.PyQt.QtWidgets import QMainWindow

        from terrainflow_assessment.qgis.plugin import TerrainFlowAssessmentPlugin

        project = QgsProject.instance()
        project.clear()
        project.setCrs(QgsCoordinateReferenceSystem(self.project_crs))

        self.main_window = QMainWindow()
        # Generous: the panel is a tall dock and a screenshot of it is clipped to
        # whatever the main window allows. No real screen is involved.
        self.main_window.resize(1800, 1600)
        self.canvas = QgsMapCanvas(self.main_window)
        self.main_window.setCentralWidget(self.canvas)
        self.canvas.setCanvasColor(QColor(255, 255, 255))
        self.canvas.setDestinationCrs(
            QgsCoordinateReferenceSystem(self.project_crs))
        self.iface = StubIface(self.main_window, self.canvas)

        self.plugin = TerrainFlowAssessmentPlugin(self.iface)
        self._install_global_iface()
        self.plugin.initGui()

        self.panel = self.plugin.panel
        self.state = self.plugin._state
        self.bar = self.iface.messageBar()
        self.project = project

        self.dialogs = RecordingDialogs()
        self.dialogs.install()

        if self._load_dem:
            self.dem_layer = self.add_dem()
        if self._load_boundary:
            self.boundary_layer = self.add_boundary()

        return self

    def __exit__(self, *_exc):
        from qgis.core import QgsProject

        try:
            self.plugin.unload()
        finally:
            QgsProject.instance().clear()
            self._restore_global_iface()
            self.dialogs.restore()
        return False

    # ------------------------------------------------------------------ actions

    def add_dem(self):
        from qgis.core import QgsProject, QgsRasterLayer

        layer = QgsRasterLayer(self.dem_path, "Test DEM")
        if not layer.isValid():
            raise AssertionError(f"synthetic DEM failed to load: {self.dem_path}")
        QgsProject.instance().addMapLayer(layer)
        self.panel.dem_changed.emit(layer)
        return layer

    def add_boundary(self):
        from qgis.core import QgsProject

        layer = site_boundary_layer()
        QgsProject.instance().addMapLayer(layer)
        self.panel.boundary_changed.emit(layer)
        return layer

    def run_baseline(self):
        self.panel.run_baseline_requested.emit()
        return self.state.baseline_result

    def add_earthwork(self, ew_type="swale", geometry=None, name=None):
        """Add an earthwork straight to the manager, bypassing the modal dialog."""
        from terrainflow_assessment.modules.earthwork_design import Earthwork

        geom = geometry if geometry is not None else line_across_valley()
        n = len(self.state.earthwork_manager) + 1
        ew = Earthwork(ew_type, geom, name or f"{ew_type.capitalize()} {n}")
        self.state.earthwork_manager.add(ew)
        return ew

    def sync_canvas(self, extent_layer=None):
        """Put the project's *checked* layers on the canvas, in layer-tree order.

        In live QGIS a QgsLayerTreeMapCanvasBridge does this; the harness builds a
        bare canvas, so rendering a screenshot means wiring it up explicitly.

        Unchecked layers are excluded, exactly as the bridge excludes them. Taking
        layerOrder() alone renders layers the user has switched off — and since
        Throughflow ships unchecked precisely because it covers the whole map, that
        put an opaque wash over every canvas screenshot and hid what was under it.
        """
        from qgis.core import QgsProject

        root = QgsProject.instance().layerTreeRoot()
        checked = set(root.checkedLayers())
        layers = [lyr for lyr in root.layerOrder() if lyr is not None and lyr in checked]
        self.canvas.setLayers(layers)

        target = extent_layer if extent_layer is not None else self.dem_layer
        if target is not None:
            extent = target.extent()
            extent.scale(1.05)
            self.canvas.setExtent(extent)
        else:
            self.canvas.zoomToFullExtent()

        self.canvas.refresh()
        self.canvas.waitWhileRendering()
        return layers

    def set_scale(self, denominator, centre=None, size=(1200, 900)):
        """Pin the canvas to a named map scale, e.g. 1:2500.

        sync_canvas() frames a layer's extent, which makes the resulting scale a
        side effect of the fixture's size. Anything asserting *scale-dependent*
        behaviour — a millimetre signature staying constant, a metres band
        halving — has to pin the scale explicitly instead.

        The canvas is resized first, and that is not optional: an unshown canvas
        sits at roughly 100x30 px, so a pinned 1:2500 would frame about 60 m and
        almost nothing would be in view. Same default size as save_canvas(), so a
        pinned scale and a screenshot of it agree.
        """
        from qgis.PyQt.QtCore import QCoreApplication

        if size:
            self.canvas.resize(*size)
            self.canvas.window().show()
            self.canvas.show()
            for _ in range(3):
                QCoreApplication.processEvents()
        if centre is not None:
            self.canvas.setCenter(centre)
        self.canvas.zoomScale(float(denominator))
        self.canvas.refresh()
        self.canvas.waitWhileRendering()
        return self.canvas.scale()

    def labels_drawn(self):
        """{layer_id: sorted set of label texts} actually placed on the canvas.

        Asserting on the labelling engine's own output rather than on pixels: it
        says *which* label was placed and where, and does not move when
        anti-aliasing or a font substitution does.

        Texts are de-duplicated because curved placement reports one position per
        character group — a single "Swale 1" comes back seven times.
        """
        results = self.canvas.labelingResults()
        if results is None:
            return {}
        try:
            positions = results.labelsWithinRect(self.canvas.extent())
        except Exception:
            return {}
        out = {}
        for pos in positions:
            out.setdefault(pos.layerID, set()).add(pos.labelText)
        return {k: sorted(v) for k, v in out.items()}

    def _install_global_iface(self):
        """Point the global qgis.utils.iface — and its existing captures — at the stub.

        Four map-tool modules do `from qgis.utils import iface` at import time, and
        DrawLineTool calls iface.mainWindow().statusBar() from its __init__. Headless
        that global is None, so constructing any draw tool raises AttributeError
        before a single click is delivered.

        Setting qgis.utils.iface alone is not enough: `from x import y` copies the
        value into the importing module's namespace, so a module imported before this
        runs still holds None. Both are needed — the attribute for modules imported
        later (several are imported lazily inside functions), and the sweep for those
        already in sys.modules.
        """
        import qgis.utils

        self._prev_global_iface = getattr(qgis.utils, "iface", None)
        qgis.utils.iface = self.iface

        self._patched_iface_modules = []
        for name, module in list(sys.modules.items()):
            if not name.startswith("terrainflow_assessment") or module is None:
                continue
            if hasattr(module, "iface"):
                self._patched_iface_modules.append((module, module.iface))
                module.iface = self.iface

    def _restore_global_iface(self):
        import qgis.utils

        for module, previous in getattr(self, "_patched_iface_modules", []):
            module.iface = previous
        self._patched_iface_modules = []
        qgis.utils.iface = getattr(self, "_prev_global_iface", None)

    def prepare_canvas_for_input(self, size=(1200, 900)):
        """Make the canvas a real, sized, visible widget so mouse input maps correctly.

        Synthetic clicks are delivered at *pixel* positions and the tools convert
        them back with toMapCoordinates(), so the canvas must have a genuine size and
        a settled extent or every click lands somewhere else.
        """
        from qgis.PyQt.QtCore import QCoreApplication

        self.canvas.resize(*size)
        self.canvas.window().show()
        self.canvas.show()
        self.sync_canvas()
        for _ in range(3):
            QCoreApplication.processEvents()
        return self.canvas

    # ------------------------------------------------------------------ asserts

    def assert_no_errors(self, context=""):
        criticals = self.bar.criticals
        if criticals:
            raise AssertionError(
                f"{context}: {len(criticals)} error(s) pushed to the message bar\n"
                + self.bar.render()
            )
        # A modal warning or error is still an error — it is just one that used to stop
        # the run dead instead of reporting itself. See RecordingDialogs.
        blocking = self.dialogs.blocking
        if blocking:
            raise AssertionError(
                f"{context}: {len(blocking)} modal dialog(s) the run would have "
                f"stopped on\n" + self.dialogs.render()
            )

    def layer_names(self):
        from qgis.core import QgsProject

        return sorted(lyr.name() for lyr in QgsProject.instance().mapLayers().values())

