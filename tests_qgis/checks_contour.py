"""Contour analysis, Processing integration, and the keypoint/keyline paths."""

from _harness import PluginHarness


def check_contour_analysis_requires_baseline(dem_path):
    """Without a baseline the controller warns rather than throwing."""
    with PluginHarness(dem_path) as h:
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour without baseline")
        assert h.bar.warnings, "expected a warning when baseline has not been run"


def check_contour_analysis_runs(dem_path):
    """analyse_contours over the real accumulation raster, then layer creation."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        assert isinstance(h.state.contour_features, list), (
            "contour_features should be a list"
        )
        assert h.state.contour_features, (
            "no candidate contours found on a 120x120 valley DEM — check the panel's "
            "interval / max-slope / min-length defaults against the synthetic terrain"
        )


def check_simple_contours_via_processing(dem_path):
    """gdal:contour through QGIS Processing — the one processing.run() call site."""
    with PluginHarness(dem_path) as h:
        h.panel.generate_simple_contours_requested.emit()
        h.assert_no_errors("gdal:contour")

        assert h.state.simple_contour_layer_id, (
            "no contour layer registered — gdal:contour produced nothing"
        )

        from qgis.core import QgsProject

        layer = QgsProject.instance().mapLayer(h.state.simple_contour_layer_id)
        assert layer is not None and layer.isValid(), "contour layer is invalid"
        assert layer.featureCount() > 0, "contour layer has no features"


def check_select_top5_and_clear(dem_path):
    """Ranking/selection styling, then the clear-analysis teardown."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        h.panel.select_top5_contours_requested.emit()
        h.assert_no_errors("select top 5")

        h.panel.show_inflow_bands_requested.emit(True)
        h.panel.show_inflow_bands_requested.emit(False)
        h.assert_no_errors("inflow bands")

        h.panel.clear_analysis_requested.emit()
        h.assert_no_errors("clear analysis")
        assert not h.state.contour_features, "clear_analysis left contour features behind"


def check_segment_analysis(dem_path):
    """Swale-segment finding over the analysed contours."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.panel.find_segments_requested.emit()
        h.assert_no_errors("segment analysis")


def check_keypoint_analysis(dem_path):
    """The concave long profile should give keypoint analysis a slope break."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_keypoint_analysis_requested.emit()
        h.assert_no_errors("keypoint analysis")


def check_keyline_analysis(dem_path):
    """Yeomans keyline generation off the keypoint result."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_keypoint_analysis_requested.emit()
        h.panel.run_keyline_requested.emit()
        h.assert_no_errors("keyline analysis")
