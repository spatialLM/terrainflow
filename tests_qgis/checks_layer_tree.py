"""Output organisation: the site layer group, its stage subgroups, and collapsing.

Every layer the plugin creates is supposed to land in one tree under the panel's
Site Name, arranged in stage order and collapsed. None of that is visible to
`pytest tests/` — the layer tree only exists inside a real QgsProject — so it is
asserted here, against the same controllers a user drives.
"""

from _harness import PluginHarness, line_across_valley
from _shots import assert_rendered, save_widget

PANEL_SIZE = (460, 1400)


# ---------------------------------------------------------------- helpers

def _root_groups():
    from qgis.core import QgsLayerTree, QgsProject

    return [
        child for child in QgsProject.instance().layerTreeRoot().children()
        if QgsLayerTree.isGroup(child)
    ]


def _find_site(name):
    for grp in _root_groups():
        if grp.name() == name:
            return grp
    raise AssertionError(
        f"no site group named {name!r}; root groups are "
        f"{[g.name() for g in _root_groups()]}"
    )


def _subgroups(group):
    from qgis.core import QgsLayerTree

    return [c for c in group.children() if QgsLayerTree.isGroup(c)]


def _named(group, prefix):
    """The one direct subgroup of *group* whose name starts with *prefix*."""
    for child in _subgroups(group):
        if child.name().startswith(prefix):
            return child
    raise AssertionError(
        f"{group.name()!r} has no subgroup starting {prefix!r}; it holds "
        f"{[c.name() for c in _subgroups(group)]}"
    )


def _set_site(panel, name):
    panel._site_name_edit.setText(name)


# ---------------------------------------------------------------- structure

def check_baseline_lands_under_site_name(dem_path):
    """Baseline rasters go into "<Site Name> › Baseline · <tag>", not the flat legend."""
    with PluginHarness(dem_path) as h:
        _set_site(h.panel, "Quail Island")
        h.run_baseline()
        h.assert_no_errors("baseline run")

        site = _find_site("Quail Island")
        baseline = _named(site, "Baseline")

        assert baseline.name().startswith("Baseline · "), (
            f"the baseline group carries no run tag: {baseline.name()!r}"
        )
        names = [n.name() for n in baseline.findLayers()]
        assert any("Streams" in n for n in names), (
            f"streams raster is not in the Baseline group: {names}"
        )

        # Nothing the plugin made may sit loose at the root.
        from qgis.core import QgsLayerTree, QgsProject

        loose = [
            c.name() for c in QgsProject.instance().layerTreeRoot().children()
            if not QgsLayerTree.isGroup(c) and c.name().startswith("Baseline")
        ]
        assert not loose, f"plugin layers left at the tree root: {loose}"


def check_layers_are_added_collapsed(dem_path):
    """Each result layer's legend starts collapsed — an expanded raster ramp is huge."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        from qgis.core import QgsProject

        root = QgsProject.instance().layerTreeRoot()
        expanded = []
        for lid in h.state.baseline_layer_ids:
            node = root.findLayer(lid)
            if node is not None and node.isExpanded():
                expanded.append(node.name())
        assert not expanded, f"these layers were added expanded: {expanded}"


def check_stages_group_in_panel_order(dem_path):
    """Baseline / Analysis / Design appear in the order the panel produces them."""
    with PluginHarness(dem_path) as h:
        _set_site(h.panel, "Quail Island")
        h.run_baseline()
        h.panel.toggle_slope_class_requested.emit(True)
        h.panel.run_contour_analysis_requested.emit()
        h.add_earthwork("swale", geometry=line_across_valley(row=60))
        h.plugin._earthworks._refresh_ew_layer()
        h.assert_no_errors("full run-through")

        site = _find_site("Quail Island")
        order = [g.name().split(" · ")[0] for g in _subgroups(site)]
        for stage in ("Baseline", "Analysis", "Design"):
            assert stage in order, f"{stage} group missing; site holds {order}"
        assert order.index("Baseline") < order.index("Analysis") < order.index("Design"), (
            f"stage groups are out of pipeline order: {order}"
        )

        # Contour outputs nest one level deeper, under the Analysis stage.
        contour = _named(_named(site, "Analysis"), "Contour Analysis")
        assert contour.findLayers(), "the Contour Analysis group is empty"

        drawn = _named(_named(site, "Design"), "Drawn Earthworks")
        drawn_names = [n.name() for n in drawn.findLayers()]
        assert "Swales" in drawn_names, (
            f"per-type earthwork layers are not under Drawn Earthworks: {drawn_names}"
        )


def check_second_site_name_starts_a_second_tree(dem_path):
    """Renaming the site and re-running gives a separate tree, not a merged one.

    This is how several analyses share one project, so it must not quietly write
    the second run's layers into the first run's groups.
    """
    with PluginHarness(dem_path) as h:
        _set_site(h.panel, "Quail Island")
        h.run_baseline()

        _set_site(h.panel, "Quail Island — 200yr")
        h.panel._rainfall_spin.setValue(250)
        h.run_baseline()
        h.assert_no_errors("second baseline run")

        first = _named(_find_site("Quail Island"), "Baseline")
        second = _named(_find_site("Quail Island — 200yr"), "Baseline")
        assert first.name() != second.name(), (
            f"both runs produced the same group name: {first.name()!r}"
        )
        assert first.findLayers() and second.findLayers(), (
            "one of the two site trees came out empty"
        )


def check_unnamed_run_adopts_a_name_typed_later(dem_path):
    """A name entered after the first run renames the auto-named group.

    Otherwise the run made before the name was typed is orphaned under "Unnamed
    Site" while everything after it goes somewhere else.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")
        _find_site("Unnamed Site")          # raises if the default group is absent

        _set_site(h.panel, "Quail Island")
        h.panel.site_name_changed.emit("Quail Island")

        site = _find_site("Quail Island")
        assert _named(site, "Baseline").findLayers(), (
            "the renamed group lost its layers"
        )
        assert all(g.name() != "Unnamed Site" for g in _root_groups()), (
            "the default group survived alongside the named one"
        )


def check_run_tag_follows_the_runoff_method(dem_path):
    """The tag quotes C under Lancaster and CN under SCS-CN — never a hidden field."""
    with PluginHarness(dem_path) as h:
        _set_site(h.panel, "Tag Test")
        h.panel._sizing_basis_combo.setCurrentIndex(0)      # Lancaster
        h.run_baseline()
        lancaster = _named(_find_site("Tag Test"), "Baseline").name()
        assert "·C0." in lancaster.replace(" ", ""), (
            f"Lancaster run is not tagged with its coefficient: {lancaster!r}"
        )
        assert "CN" not in lancaster, (
            f"Lancaster run quotes a curve number it never read: {lancaster!r}"
        )

        h.panel._sizing_basis_combo.setCurrentIndex(2)      # SCS-CN
        h.run_baseline()
        scs = _named(_find_site("Tag Test"), "Baseline").name()
        assert "CN" in scs, f"SCS-CN run is not tagged with its curve number: {scs!r}"


# ---------------------------------------------------------------- panel visibility

def check_runoff_method_hides_the_other_method_inputs(dem_path):
    """Only the chosen method's inputs are on screen — labels included."""
    with PluginHarness(dem_path) as h:
        p = h.panel
        # The panel must be realised before isVisible() means anything: an unshown
        # widget reports False regardless of what setVisible() was told.
        p.window().show()
        p.show()
        p._show_stage("baseline")

        lancaster = [p._runoff_surface_combo, p._runoff_coeff_spin]
        scs = [p._soil_combo, p._ground_condition_combo, p._cn_spin, p._moisture_combo]

        p._sizing_basis_combo.setCurrentIndex(0)           # Lancaster
        assert all(w.isVisible() for w in lancaster), "Lancaster inputs are hidden"
        assert not any(w.isVisible() for w in scs), "SCS-CN inputs shown under Lancaster"

        p._sizing_basis_combo.setCurrentIndex(2)           # SCS-CN
        assert all(w.isVisible() for w in scs), "SCS-CN inputs are hidden"
        assert not any(w.isVisible() for w in lancaster), "Lancaster inputs shown under SCS-CN"

        p._sizing_basis_combo.setCurrentIndex(1)           # Total rainfall
        assert not any(w.isVisible() for w in lancaster + scs), (
            "Total rainfall asks for calibration inputs it does not read"
        )

        # A label left behind captions the row beneath it.
        for key, rows in p._basis_rows.items():
            for label, widget in rows:
                assert label.isVisible() == widget.isVisible(), (
                    f"{key}: label and field disagree on visibility "
                    f"({label.text()!r})"
                )


def _grid_holding(widget):
    """The QGridLayout that *widget* sits in, found by walking nested layouts.

    ``widget.parentWidget().layout()`` returns the section's outer box layout, not
    the grid — the grid is a sub-layout of it, and Qt exposes no upward link. The
    walk starts at the parent *widget* because a QDockWidget's own layout does not
    contain its content as an ordinary layout item.
    """
    from qgis.PyQt.QtWidgets import QGridLayout

    stack = [widget.parentWidget().layout()]
    while stack:
        layout = stack.pop()
        if layout is None:
            continue
        if isinstance(layout, QGridLayout) and layout.indexOf(widget) >= 0:
            return layout
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.layout() is not None:
                stack.append(item.layout())
            elif item.widget() is not None and item.widget().layout() is not None:
                stack.append(item.widget().layout())
    raise AssertionError(f"{widget} is not inside any QGridLayout")


def check_runoff_method_sits_directly_under_duration(dem_path):
    """The method is asked before the inputs it governs, not buried below them."""
    with PluginHarness(dem_path) as h:
        p = h.panel
        grid = _grid_holding(p._sizing_basis_combo)
        rows = {}
        for widget in (p._duration_spin, p._sizing_basis_combo,
                       p._runoff_coeff_spin, p._cn_spin):
            idx = grid.indexOf(widget)
            assert idx >= 0, (
                f"{widget} is not in the same grid as the method selector"
            )
            rows[widget] = grid.getItemPosition(idx)[0]

        assert rows[p._sizing_basis_combo] == rows[p._duration_spin] + 1, (
            "Runoff Calculation Method is not the row after Duration"
        )
        assert rows[p._sizing_basis_combo] < rows[p._runoff_coeff_spin], (
            "the method is asked after the coefficient it selects"
        )
        assert rows[p._sizing_basis_combo] < rows[p._cn_spin], (
            "the method is asked after the curve number it selects"
        )


def check_baseline_stage_renders_per_method(dem_path):
    """Screenshot the Baseline stage under each method — this is the change to look at."""
    with PluginHarness(dem_path) as h:
        h.panel._show_stage("baseline")
        for index, name in ((0, "lancaster"), (1, "total_rainfall"), (2, "scs_cn")):
            h.panel._sizing_basis_combo.setCurrentIndex(index)
            path = save_widget(h.panel, f"panel_baseline_{name}", size=PANEL_SIZE)
            assert_rendered(path, f"baseline stage ({name})", min_colours=12)


# ---------------------------------------------------------------- backdrop leak

def _design_backdrops():
    """Every "Earthworks — …" backdrop layer currently in the project, by name."""
    from qgis.core import QgsProject

    names = []
    for layer in QgsProject.instance().mapLayers().values():
        if layer.name().startswith("Earthworks — "):
            names.append(layer.name())
    return sorted(names)


def check_a_second_verify_replaces_the_backdrops_it_made(dem_path):
    """Q-13. `_load_burned_dem_layer` places "Earthworks — Hillshade" and
    "Earthworks — Burned DEM" under Design on every Verify run. Nothing removed the
    previous pair: `clear_group` only ever clears Rerun and Baseline, and
    `earthworks_layer_ids` is reassigned in `_load_result_layers` *before* this runs,
    so the old pair's ids were dropped on the floor.

    Three Verify runs left six ticked-on backdrops stacked on one path — and because
    `toggle_before_after` walks `earthworks_layer_ids`, it only ever reached the
    newest pair, so 'Show: with earthworks' stopped hiding the design.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", geometry=line_across_valley())

        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("first verify")
        first = _design_backdrops()
        assert first, "no burned backdrops were placed at all"

        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("second verify")
        second = _design_backdrops()

        assert second == first, (
            f"a second Verify stacked another set of backdrops on the same path: "
            f"{first} -> {second}"
        )


def check_the_before_after_toggle_still_hides_a_re_verified_backdrop(dem_path):
    """Q-13, the half a user notices. The toggle walks `earthworks_layer_ids`, so a
    backdrop whose id fell out of that list stays ticked on whatever the toggle says
    — the design stays painted over the baseline it is supposed to be compared with.
    """
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.add_earthwork("swale", geometry=line_across_valley())
        h.panel.run_earthworks_requested.emit()
        h.panel.run_earthworks_requested.emit()
        h.assert_no_errors("two verify runs")

        h.plugin._baseline.toggle_before_after(False)

        root = QgsProject.instance().layerTreeRoot()
        still_on = []
        for layer in QgsProject.instance().mapLayers().values():
            if not layer.name().startswith("Earthworks — "):
                continue
            node = root.findLayer(layer.id())
            if node is not None and node.itemVisibilityChecked():
                still_on.append(layer.name())

        assert not still_on, (
            f"'Show: baseline' left these earthworks backdrops ticked on: "
            f"{still_on} — their ids are not in earthworks_layer_ids"
        )
