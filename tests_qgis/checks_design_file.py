"""
Design file (`.tfd`) save / reopen round trips.

`modules/project_io.py` is pure and already under the pytest coverage gate, so the
*schema* is covered there. What is not covered anywhere else is
`qgis/controllers/design_file.py` — 500 lines of QGIS glue: fingerprinting, DEM
clipping, zip writing, layer round-tripping and DEM resolution. `pytest tests/`
mocks all of that away.

The fixture is built programmatically and saved during the check, rather than a
committed `.tfd`: a fixture must not depend on the feature it is validating, and
generating it keeps the suite self-contained and deterministic with no binary in
git.

Modal dialogs are scripted. The two pure prompts (`_ask_embed_dem`,
`_offer_baseline_rerun`) are replaced outright since they only gather an answer;
everything they gate — fingerprinting, clipping, archiving, resolution — runs for
real.
"""

import os
import shutil
import zipfile
from pathlib import Path

from _harness import PluginHarness, line_across_valley, site_boundary_layer

MANIFEST_MEMBER = "design.json"
DEM_MEMBER = "dem.tif"


class ScriptedDialogs:
    """Answer design_file's file/message dialogs without a user.

    Patches the names inside the controller's module, not the Qt classes, so nothing
    global is mutated and the restore is exact.
    """

    def __init__(self, save_path=None, open_path=None, locate_path=None):
        self.save_path = save_path
        self.open_path = open_path
        self.locate_path = locate_path
        self._open_calls = 0

    def __enter__(self):
        from terrainflow_assessment.qgis.controllers import design_file

        self._module = design_file
        self._prev_file_dialog = design_file.QFileDialog
        self._prev_message_box = design_file.QMessageBox

        scripted = self

        class _FileDialog:
            @staticmethod
            def getSaveFileName(*_a, **_k):
                return (scripted.save_path or "", "")

            @staticmethod
            def getOpenFileName(*_a, **_k):
                # First call is "open which design"; a second is "locate the DEM".
                scripted._open_calls += 1
                if scripted._open_calls == 1:
                    return (scripted.open_path or "", "")
                return (scripted.locate_path or "", "")

        class _MessageBox:
            @staticmethod
            def information(*_a, **_k):
                return None

        design_file.QFileDialog = _FileDialog
        design_file.QMessageBox = _MessageBox
        return self

    def __exit__(self, *_exc):
        self._module.QFileDialog = self._prev_file_dialog
        self._module.QMessageBox = self._prev_message_box
        return False


def _script_prompts(controller, embed=False):
    """Replace the two pure prompts. Returns a restore callable."""
    original_embed = controller._ask_embed_dem
    original_rerun = controller._offer_baseline_rerun
    controller._ask_embed_dem = lambda: embed
    controller._offer_baseline_rerun = lambda: None

    def restore():
        controller._ask_embed_dem = original_embed
        controller._offer_baseline_rerun = original_rerun

    return restore


def _design_summary(harness):
    """The parts of the session a round trip must preserve."""
    manager = harness.state.earthwork_manager
    return {
        "inputs": dict(harness.panel.collect_inputs()),
        "earthworks": [(ew.type, ew.name) for ew in manager.get_all()],
    }


def _build_a_design(harness, boundary=True):
    """A session worth saving: DEM, boundary, storm inputs, two earthworks."""
    if boundary:
        from qgis.core import QgsProject

        layer = site_boundary_layer()
        QgsProject.instance().addMapLayer(layer)
        # Setting the combo is the real path — it fires layerChanged, which the
        # panel forwards as boundary_changed to the controller.
        harness.panel._boundary_combo.setLayer(layer)

    harness.add_earthwork("swale", geometry=line_across_valley(row=60))
    harness.add_earthwork("dam", geometry=line_across_valley(row=90))


def check_save_writes_a_readable_archive(dem_path, tmp_dir=None):
    """Saving produces a zip with a manifest — and no DEM when referencing only."""
    with PluginHarness(dem_path) as h:
        _build_a_design(h)
        controller = h.plugin._design_file
        target = os.path.join(h.state.output_dir, "referenced.tfd")

        restore = _script_prompts(controller, embed=False)
        try:
            with ScriptedDialogs(save_path=target):
                controller.save_design()
        finally:
            restore()

        h.assert_no_errors("save design (referenced)")
        assert os.path.exists(target), "no .tfd was written"
        assert zipfile.is_zipfile(target), "the .tfd is not a zip archive"

        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
        assert MANIFEST_MEMBER in names, f"manifest missing from archive: {names}"
        assert DEM_MEMBER not in names, (
            "a referenced save embedded the DEM anyway — the file will be huge"
        )


def check_round_trip_preserves_the_design(dem_path):
    """Save, wipe the session, reopen: inputs, areas and earthworks must return."""
    with PluginHarness(dem_path) as h:
        _build_a_design(h)
        controller = h.plugin._design_file
        target = os.path.join(h.state.output_dir, "roundtrip.tfd")

        before = _design_summary(h)
        assert before["earthworks"], "no earthworks to round trip"

        restore = _script_prompts(controller, embed=False)
        try:
            with ScriptedDialogs(save_path=target):
                controller.save_design()
            h.assert_no_errors("save design")

            # Wipe what the file should bring back. The DEM layer stays loaded, so
            # resolution goes through _find_loaded_dem rather than prompting.
            h.state.earthwork_manager.clear()
            assert len(h.state.earthwork_manager) == 0

            with ScriptedDialogs(open_path=target):
                controller.open_design()
        finally:
            restore()

        h.assert_no_errors("open design")

        after = _design_summary(h)
        assert after["earthworks"] == before["earthworks"], (
            f"earthworks changed across the round trip: "
            f"{before['earthworks']} -> {after['earthworks']}"
        )

        differing = {
            key: (before["inputs"][key], after["inputs"].get(key))
            for key in before["inputs"]
            if after["inputs"].get(key) != before["inputs"][key]
        }
        assert not differing, f"inputs changed across the round trip: {differing}"


def check_embedded_dem_opens_without_the_original(dem_path):
    """The portability promise: an embedded design opens after the DEM is gone.

    This is the case that matters when a file moves between machines, and the only
    way to test it honestly is to actually delete the DEM before reopening. So the
    check works on its own copy, never the shared fixture.
    """
    from qgis.core import QgsProject, QgsRasterLayer

    with PluginHarness(dem_path, load_dem=False) as h:
        # Own copy of the DEM, so deleting it cannot affect any other check.
        private_dem = os.path.join(h.state.output_dir, "private_dem.tif")
        shutil.copy2(dem_path, private_dem)

        layer = QgsRasterLayer(private_dem, "Private DEM")
        assert layer.isValid(), "private DEM copy did not load"
        QgsProject.instance().addMapLayer(layer)
        h.dem_layer = layer
        h.panel.dem_changed.emit(layer)

        _build_a_design(h)
        controller = h.plugin._design_file
        target = os.path.join(h.state.output_dir, "embedded.tfd")

        restore = _script_prompts(controller, embed=True)
        try:
            with ScriptedDialogs(save_path=target):
                controller.save_design()
            h.assert_no_errors("save design (embedded)")

            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
            assert DEM_MEMBER in names, (
                f"embedded save did not carry the DEM: {names}"
            )

            # Remove the layer first: Windows will not delete a raster a provider
            # still holds open.
            QgsProject.instance().removeMapLayer(layer.id())
            h.dem_layer = None
            h.state.dem_path = None
            os.remove(private_dem)
            assert not os.path.exists(private_dem)

            h.state.earthwork_manager.clear()

            with ScriptedDialogs(open_path=target):
                controller.open_design()
        finally:
            restore()

        h.assert_no_errors("open embedded design with the original DEM deleted")
        assert h.state.dem_path and os.path.exists(h.state.dem_path), (
            f"no usable DEM after opening an embedded design: {h.state.dem_path!r}"
        )
        assert len(h.state.earthwork_manager) == 2, (
            "earthworks did not come back from the embedded design"
        )


def check_opening_an_embedded_design_moves_the_whole_session_to_the_clip(dem_path):
    """One DEM per session — burner, dem_info and picker all on the restored clip.

    An embedded design carries a *clip* of its DEM, so opening one repoints the session
    at a smaller raster. ``state.burner`` is written in exactly one place
    (``BaselineController.on_dem_changed``), and the Open path used to set ``dem_path``
    and ``dem_info`` by hand without going near it. The session then straddled two
    grids: Baseline analysed the clip while the burn ran on the parent tile — 1027x858
    against 2157x1319 on the Quail Island design — and the verification, unable to
    subtract two different extents, silently dropped the correction and reported every
    measured volume with the site's natural ponding still inside it.

    The larger DEM is deliberately still selected in the picker when the design opens,
    because that is the situation that produced the bug.
    """
    with PluginHarness(dem_path) as h:
        _build_a_design(h)
        controller = h.plugin._design_file
        target = os.path.join(h.state.output_dir, "clipped.tfd")

        restore = _script_prompts(controller, embed=True)
        try:
            with ScriptedDialogs(save_path=target):
                controller.save_design()
            h.assert_no_errors("save design (embedded clip)")

            with ScriptedDialogs(open_path=target):
                controller.open_design()
        finally:
            restore()

        h.assert_no_errors("open embedded design")

        burner = h.state.burner
        info = h.state.dem_info
        assert burner is not None, "no burner after opening a design"
        assert info is not None, "no dem_info after opening a design"
        assert burner.dem_path == h.state.dem_path, (
            f"burner is built from {burner.dem_path!r} but the session points at "
            f"{h.state.dem_path!r} — the two would produce rasters on different grids"
        )
        assert tuple(burner.shape) == (info.height, info.width), (
            f"burner grid {burner.shape[1]}x{burner.shape[0]} does not match the "
            f"session DEM {info.width}x{info.height}"
        )

        picked = h.panel.dem_layer
        assert picked is not None, "the restored DEM was not selected in the picker"
        assert picked.source().split("|")[0] == h.state.dem_path, (
            "the DEM picker shows a different raster from the one being analysed"
        )


def check_a_non_archive_is_refused(dem_path):
    """Opening something that is not a .tfd reports an error and changes nothing."""
    with PluginHarness(dem_path) as h:
        _build_a_design(h)
        controller = h.plugin._design_file

        bogus = Path(h.state.output_dir) / "not_really.tfd"
        bogus.write_text("this is not a zip archive", encoding="utf-8")

        before = len(h.state.earthwork_manager)
        restore = _script_prompts(controller)
        try:
            with ScriptedDialogs(open_path=str(bogus)):
                controller.open_design()
        finally:
            restore()

        assert h.bar.criticals, "opening a non-archive reported no error"
        assert len(h.state.earthwork_manager) == before, (
            "a failed open modified the session anyway"
        )


def check_cancelling_the_dem_prompt_aborts_cleanly(dem_path):
    """A referenced design whose DEM is gone must abort, not half-restore.

    DEM resolution happens before anything is written into the session precisely so
    that a failure here leaves the current work intact.
    """
    from qgis.core import QgsProject, QgsRasterLayer

    with PluginHarness(dem_path, load_dem=False) as h:
        private_dem = os.path.join(h.state.output_dir, "vanishing_dem.tif")
        shutil.copy2(dem_path, private_dem)

        layer = QgsRasterLayer(private_dem, "Vanishing DEM")
        QgsProject.instance().addMapLayer(layer)
        h.dem_layer = layer
        h.panel.dem_changed.emit(layer)

        _build_a_design(h)
        controller = h.plugin._design_file
        target = os.path.join(h.state.output_dir, "referenced_only.tfd")

        restore = _script_prompts(controller, embed=False)
        try:
            with ScriptedDialogs(save_path=target):
                controller.save_design()
            h.assert_no_errors("save design (referenced)")

            QgsProject.instance().removeMapLayer(layer.id())
            h.dem_layer = None
            os.remove(private_dem)

            surviving_dem = h.state.dem_path
            surviving_count = len(h.state.earthwork_manager)

            # locate_path "" == the user cancelled the file chooser.
            with ScriptedDialogs(open_path=target, locate_path=""):
                controller.open_design()
        finally:
            restore()

        assert h.bar.criticals, "cancelling the DEM prompt reported no error"
        assert h.state.dem_path == surviving_dem, (
            "an aborted open changed the session's DEM"
        )
        assert len(h.state.earthwork_manager) == surviving_count, (
            "an aborted open modified the earthworks"
        )


def _decoy_dem(source, target):
    """A DEM with the same grid as *source* but different elevations.

    Same cell size, CRS, extent and dimensions — everything except the pixels. This is
    the adversarial case the content hash exists for: a fingerprint built from grid
    properties alone would accept this file as the original.
    """
    import rasterio

    with rasterio.open(source) as src:
        profile = src.profile
        data = src.read(1) + 17.0
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data, 1)
    return target


def check_locating_the_wrong_dem_is_refused(dem_path):
    """The worst possible failure: silently re-scoring a design against other terrain.

    A design's capacities, catchments and capture percentage are all statements about
    one specific DEM. Accepting a different one would produce a plausible, fully
    populated, entirely wrong design — no error, no warning, nothing to notice. So a
    fingerprint mismatch must refuse the open outright rather than proceed.
    """
    from qgis.core import QgsProject, QgsRasterLayer

    with PluginHarness(dem_path, load_dem=False) as h:
        private_dem = os.path.join(h.state.output_dir, "original_dem.tif")
        shutil.copy2(dem_path, private_dem)

        layer = QgsRasterLayer(private_dem, "Original DEM")
        assert layer.isValid(), "private DEM copy did not load"
        QgsProject.instance().addMapLayer(layer)
        h.dem_layer = layer
        h.panel.dem_changed.emit(layer)

        _build_a_design(h)
        controller = h.plugin._design_file
        target = os.path.join(h.state.output_dir, "referenced_design.tfd")

        restore = _script_prompts(controller, embed=False)
        try:
            with ScriptedDialogs(save_path=target):
                controller.save_design()
            h.assert_no_errors("save design (referenced)")

            # Same grid, different elevations — written before the original is removed
            # so it inherits an identical profile.
            decoy = _decoy_dem(private_dem,
                               os.path.join(h.state.output_dir, "decoy_dem.tif"))

            QgsProject.instance().removeMapLayer(layer.id())
            h.dem_layer = None
            os.remove(private_dem)

            surviving_dem = h.state.dem_path
            surviving_count = len(h.state.earthwork_manager)

            with ScriptedDialogs(open_path=target, locate_path=decoy):
                controller.open_design()
        finally:
            restore()

        assert h.bar.criticals, (
            "a mismatched DEM was accepted silently — the design would have been "
            "re-scored against terrain it was never designed on"
        )
        assert h.state.dem_path == surviving_dem, (
            "a refused open pointed the session at the wrong DEM anyway"
        )
        assert len(h.state.earthwork_manager) == surviving_count, (
            "a refused open modified the earthworks"
        )


def check_panel_defaults_match_the_persisted_defaults(dem_path):
    """A fresh panel and ``project_io.INPUT_FIELDS`` must agree, key for key.

    ``normalise_inputs`` fills every absent key from ``INPUT_FIELDS``, so any default
    there that disagrees with the spin box it restores silently re-answers the analysis
    when an older design file is reopened. That has now been found twice — the swale
    trio (comment at ``project_io.py:100``), then ``simple_contour_interval_m`` (5.0 vs
    1.0) and ``max_slope_deg`` (15.0 vs 18.0). Two instances of one bug class is the
    point at which the class gets a test rather than the instances getting a patch.

    This has to live here rather than in ``pytest tests/``: the panel needs a real Qt
    runtime to build its widgets, and the whole point is to read what the user actually
    sees on a freshly opened plugin.
    """
    from terrainflow_assessment.modules import project_io

    # Exempt, and each for a stated reason — not a list to grow when something fails.
    #
    # The swale trio: `seed_swale_criteria` starts those boxes at the user's *saved
    # standard* where one exists (precedence is document > standard > shipped), so on a
    # machine that has one they are supposed to differ from the shipped default.
    #
    # The sentinel group: their persisted default is deliberately "unset" — an empty
    # string or a zero — rather than a duplicate of the panel's opening value. A file
    # that carries no soil name should come back as *no soil name*, not as whichever
    # soil the combo happens to open on. `exit_flow_ls` qualifies because its spin box
    # has range (0.0, 10000): the sentinel is inside the widget's range and so it
    # round-trips exactly.
    #
    # `peak_intensity_mm_hr` was here on the same reasoning and did not qualify. Its
    # spin box has range (0.1, 500), so `apply_inputs` clamped the 0.0 sentinel to
    # **0.1** against a panel default of 40.0 — every peak flow off an older or
    # hand-edited .tfd 400x too small, and the "starting value" warning suppressed
    # exactly when it was needed. A sentinel the widget cannot hold is not a sentinel;
    # the persisted default is now `DEFAULT_PEAK_INTENSITY_MM_HR`. Check the widget's
    # *range* before granting this exemption to anything numeric.
    exempt = {
        "swale_depth_m", "swale_width_m", "swale_bottom_width_m",
        "soil_name", "earthwork_soil_name", "moisture",
        "exit_flow_ls",
    }

    # Empty, and meant to stay that way. The two this check found and once held open —
    # rainfall_mm (persisted 120.0 against the panel's 65.0, the design storm itself)
    # and routing (persisted 'd8' against the panel's 'dinf') — were taken as their own
    # decision and fixed at the table in ``project_io``, which is where the fix belongs:
    # exempting them would have left every older file reopening on a storm and a flow
    # algorithm nobody chose.
    #
    # Asserted as an *exact* set, so a new divergence fails here and so does fixing one
    # without removing it from the list. An allowlist that only ever grows is how a
    # characterisation test stops characterising anything.
    known_divergences = set()

    with PluginHarness(dem_path) as h:
        defaults = project_io.default_inputs()
        actual = h.panel.collect_inputs()

        diverged = set()
        detail = []
        for key, expected in sorted(defaults.items()):
            if key in exempt:
                continue
            got = actual.get(key)
            if isinstance(expected, float) and isinstance(got, (int, float)):
                same = abs(float(got) - expected) <= 1e-9
            else:
                same = got == expected
            if not same:
                diverged.add(key)
                detail.append(f"{key}: panel {got!r} vs persisted {expected!r}")

        assert diverged == known_divergences, (
            "the panel and the design-file defaults disagree in a way this check did "
            "not already know about, so reopening a file saved before those keys "
            "existed would answer a different question than the one on screen.\n"
            f"  expected divergences: {sorted(known_divergences)}\n"
            f"  actual divergences:   {sorted(diverged)}\n"
            "  detail:\n    " + "\n    ".join(detail)
        )
