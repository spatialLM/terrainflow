"""Terrain indices: compute, toggle, place, and the two invariants that matter.

``pytest tests/`` covers the maths on analytic surfaces. What it cannot cover is the
controller: which raster it reads, where the layers land, and whether a toggle can ask
for something that does not exist yet.
"""

import os

from _harness import PluginHarness
from _shots import assert_rendered, save_widget

from terrainflow_assessment.qgis.controllers.terrain import INDEX_SPECS


def check_terrain_indices_need_a_baseline(dem_path):
    """The button is gated, and pressing it early says so rather than failing."""
    with PluginHarness(dem_path) as h:
        h.panel.run_terrain_indices_requested.emit()
        assert h.bar.warnings, (
            "computing indices with no baseline should warn — they are built on its "
            "flow accumulation and there is nothing to build on yet"
        )
        assert not h.state.terrain_index_paths


def check_terrain_indices_compute_and_write_every_raster(dem_path):
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices")

        paths = h.state.terrain_index_paths
        assert set(paths) == set(INDEX_SPECS), (
            f"expected every index to be written, got {sorted(paths)}"
        )
        for key, path in paths.items():
            assert os.path.exists(path), f"{key} was reported but not written"
            assert os.path.getsize(path) > 0, f"{key} was written empty"

        # The signed indices carry a display bound taken while the values were in
        # memory. Without it the ramp anchors on a cliff-edge cell and the whole site
        # paints as planar.
        for key in ("plan_curvature", "profile_curvature"):
            assert h.state.terrain_index_bounds.get(key), (
                f"{key} has no symmetric display bound"
            )


def check_terrain_indices_read_the_cell_count_not_the_volume(dem_path):
    """The invariant: contributing area is a CELL COUNT.

    ``FlowAnalysis.acc`` counts cells; ``runoff_accumulation`` is m³ and has already had
    water held back from it by every hollow upstream. A wetness index built on the
    volume field would have a retained numerator over an unretained denominator — and it
    would move with the design storm, which a terrain index must not.

    Asserted by re-running the indices against a different storm and requiring the
    wetness raster to come back byte-identical.
    """
    import hashlib

    def _digest(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")
        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices, first storm")
        first = _digest(h.state.terrain_index_paths["twi"])

        # Double the storm and re-run everything.
        h.panel._rainfall_spin.setValue(h.panel.rainfall_mm * 2.0)
        h.run_baseline()
        h.assert_no_errors("baseline run, second storm")
        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices, second storm")
        second = _digest(h.state.terrain_index_paths["twi"])

        assert first == second, (
            "the wetness index moved when the design storm changed, so it is being "
            "built on a runoff volume rather than on the contributing cell count"
        )


def check_terrain_index_layers_are_placed_not_loose(dem_path):
    """Every layer goes through ``_groups.place()``, so none of them sits at the root.

    Asserted by name against the root's own children rather than by walking a node's
    ``parent()`` — a layer-tree node's Python wrapper is a borrowed pointer and holding
    one across a comparison is a reliable way to take the process down.
    """
    from qgis.core import QgsLayerTree, QgsProject

    def _groups_under(node):
        return [c for c in node.children() if QgsLayerTree.isGroup(c)]

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices")

        h.panel.terrain_index_toggled.emit("twi", True)
        h.assert_no_errors("show wetness index")

        layer_id = h.state.terrain_index_layer_ids.get("twi")
        assert layer_id, "the wetness toggle created no layer"

        root = QgsProject.instance().layerTreeRoot()
        node = root.findLayer(layer_id)
        assert node is not None, "the layer is not in the tree"
        name = node.name()

        # Walk **groups only**, and read node names rather than layer ids. A layer-tree
        # node whose layer has been removed is a dangling pointer, earlier checks in
        # this process leave some behind, and asking one of those for its layerId takes
        # the whole subprocess down with no traceback.
        placed = []
        for site in _groups_under(root):
            for stage in _groups_under(site):
                placed.extend(n.name() for n in stage.findLayers())
        assert name in placed, (
            f"{name!r} is not inside a stage group — it was added loose at the top of "
            "the legend, and every layer goes through _groups.place()"
        )

        # Toggling off hides rather than deletes, so the raster is not re-read.
        h.panel.terrain_index_toggled.emit("twi", False)
        h.assert_no_errors("hide wetness index")
        assert h.state.terrain_index_layer_ids.get("twi") == layer_id
        assert not root.findLayer(layer_id).itemVisibilityChecked()


def check_every_terrain_index_renders(dem_path):
    """Each of the six paints something — a ramp that resolves to one flat colour is
    indistinguishable from a broken layer until someone opens the map."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices")

        for key in INDEX_SPECS:
            h.panel.terrain_index_toggled.emit(key, True)
            h.assert_no_errors(f"show {key}")
            assert h.state.terrain_index_layer_ids.get(key), f"{key} built no layer"
            h.panel.terrain_index_toggled.emit(key, False)


def check_a_toggle_before_computing_is_refused(dem_path):
    """Asking for a raster that has not been computed warns rather than half-working."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.terrain_index_toggled.emit("twi", True)
        assert h.bar.warnings, "a toggle with nothing computed should say so"
        assert not h.state.terrain_index_layer_ids


def check_terrain_panel_renders(dem_path):
    """The Analysis stage with the indices computed and the toggles live.

    ``_show_stage`` matters: a baseline leaves the panel on Design, so a plain grab of
    the whole panel photographs the earthwork tool menu and says nothing at all about
    the controls this check is named for.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices")

        h.panel._show_stage("analysis")
        path = save_widget(h.panel, "terrain_panel", size=(460, 1700))
        assert_rendered(path, "terrain index panel", min_colours=6)
