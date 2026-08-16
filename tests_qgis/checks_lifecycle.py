"""Plugin boot / teardown against a real QgsApplication."""

from _harness import PluginHarness


def check_plugin_loads_and_unloads(dem_path):
    """initGui() builds the panel, wires every signal, and unload() reverses it."""
    with PluginHarness(dem_path, load_dem=False) as h:
        assert h.panel is not None, "initGui() did not create the panel"
        assert h.panel in h.iface.dock_widgets, "panel was not docked"
        assert len(h.iface.toolbar_actions) == 1, "toolbar action not registered"
        assert len(h.iface.menu_items) == 1, "plugin menu entry not registered"
        h.assert_no_errors("plugin load")
        iface = h.iface

    assert iface.dock_widgets == [], "unload() left the panel docked"
    assert iface.toolbar_actions == [], "unload() left the toolbar icon behind"
    assert iface.menu_items == [], "unload() left the menu entry behind"


def check_dem_load_populates_state(dem_path):
    """dem_changed → load_dem() → state carries DEM info and a burner."""
    with PluginHarness(dem_path) as h:
        assert h.state.dem_path == dem_path, "dem_path not recorded on state"
        assert h.state.dem_info is not None, "dem_info not populated"
        assert abs(h.state.dem_info.cell_size_m - 2.0) < 1e-6, (
            f"cell size read as {h.state.dem_info.cell_size_m}, expected 2.0"
        )
        assert h.state.burner is not None, "DEMBurner not constructed"
        h.assert_no_errors("DEM load")


def check_panel_inputs_readable(dem_path):
    """Every panel property the controllers read must resolve to a usable value.

    These are the values AnalysisWorker is constructed from; a renamed widget or
    a broken property shows up here rather than as a mid-analysis TypeError.
    """
    with PluginHarness(dem_path) as h:
        numeric = {
            "rainfall_mm": h.panel.rainfall_mm,
            "duration_hr": h.panel.duration_hr,
            "cn": h.panel.cn,
            "stream_threshold_ha": h.panel.stream_threshold_ha,
            "exit_flow_ls": h.panel.exit_flow_ls,
            "runoff_coefficient": h.panel.runoff_coefficient,
            "contour_interval_m": h.panel.contour_interval_m,
            "max_slope_deg": h.panel.max_slope_deg,
        }
        for name, value in numeric.items():
            assert isinstance(value, (int, float)), f"panel.{name} is {value!r}, not numeric"

        for name in ("site_name", "moisture", "routing", "sizing_basis"):
            assert getattr(h.panel, name) is not None, f"panel.{name} is None"

        h.assert_no_errors("panel inputs")


def check_a_second_plugin_boots_cleanly_over_the_first(dem_path):
    """Reload: unload, build another against the same QGIS, and run something.

    This is what the Plugin Manager does, and what `unload()`'s docstring is
    entirely about — anything left attached to an object QGIS owns rather than
    the plugin (the canvas, the layer tree, the project singleton) outlives the
    object it calls back into. Booting once per check, as every other check
    does, never exercises it: the failure needs a *second* plugin alive against
    connections the first left behind.

    A stale connection shows up here as the dead controller raising when the
    signal reaches it, which the harness records as an error rather than a
    traceback nobody sees.
    """
    from terrainflow_assessment.qgis.plugin import TerrainFlowAssessmentPlugin

    with PluginHarness(dem_path) as h:
        h.plugin.unload()

        second = TerrainFlowAssessmentPlugin(h.iface)
        second.initGui()
        # Hand it to the harness now, so its __exit__ tears this one down however
        # the assertions below go — and tears it down exactly once.
        h.plugin = second
        h.panel = second.panel
        try:
            assert len(h.iface.dock_widgets) == 1, (
                f"{len(h.iface.dock_widgets)} panels docked after a reload — "
                "the first one was not removed")
            assert len(h.iface.toolbar_actions) == 1, "two toolbar icons after reload"
            assert len(h.iface.menu_items) == 1, "two menu entries after reload"

            # Reaches the new controllers, and must not reach the old ones.
            second.panel.dem_changed.emit(h.dem_layer)
            second.panel.analysis_inputs_changed.emit()
            h.assert_no_errors("after reload")

            assert second._state.dem_path == dem_path, (
                "the second plugin did not take the DEM — its wiring is not live")
        finally:
            h.state = second._state
