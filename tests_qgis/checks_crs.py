"""Project CRS != DEM CRS — the coincidence the rest of the suite runs on.

Everything else here pins both to EPSG:2193, so a layer created in the project's
CRS instead of the DEM's still draws in exactly the right place, and a scale bar
measured off a project extent still reads in metres. Neither holds for an operator
whose project is in WGS 84 — which is the QGIS default for a new project, and
therefore the common case, not an exotic one.

The DEM stays projected throughout: this is not about analysing geographic
terrain, which the plugin does not attempt. It is about a *display* CRS being
allowed to change what the plugin measures, which it must not.
"""

import os

from _harness import PluginHarness, line_across_valley

GEOGRAPHIC = "EPSG:4326"


def _drawn_layers(h):
    """Every layer the plugin created, by name."""
    from qgis.core import QgsProject

    return {lyr.name(): lyr for lyr in QgsProject.instance().mapLayers().values()}


def check_a_geographic_project_does_not_move_the_analysis(dem_path):
    """The baseline is arithmetic on the DEM's grid and cannot care what the
    operator's project is set to. If it does, everything after it is wrong."""
    with PluginHarness(dem_path) as h_metric:
        h_metric.run_baseline()
        metric = h_metric.state.baseline_report.catchment_area_ha

    with PluginHarness(dem_path, project_crs=GEOGRAPHIC) as h:
        assert h.project.crs().isGeographic(), "the fixture did not diverge"
        h.run_baseline()
        h.assert_no_errors("baseline under a geographic project CRS")
        assert abs(h.state.baseline_report.catchment_area_ha - metric) < 1e-6, (
            "the catchment area changed with the display CRS")


def check_keypoint_layers_land_on_the_dem_not_off_africa(dem_path):
    """Keypoints, ridgelines, pond sites and the keyline all carry DEM grid
    coordinates. Declared in a geographic project CRS, those metres are read as
    degrees and the layer lands a few hundred kilometres off Ghana."""
    with PluginHarness(dem_path, project_crs=GEOGRAPHIC) as h:
        h.run_baseline()
        h.panel.run_keypoint_analysis_requested.emit()
        h.panel.run_keyline_requested.emit()
        h.assert_no_errors("keypoint analysis under a divergent project CRS")

        dem_extent = h.dem_layer.extent()
        drawn = _drawn_layers(h)
        family = {name: layer for name, layer in drawn.items()
                  if any(word in name for word in
                         ("Keypoint", "Ridge", "Pond Site", "Keyline"))}
        assert family, (
            f"no keypoint-family layer was produced to check; got {sorted(drawn)}")

        for name, layer in family.items():
            assert layer.crs() == h.dem_layer.crs(), (
                f"'{name}' is declared in {layer.crs().authid()}, not the DEM's "
                f"{h.dem_layer.crs().authid()}")
            extent = layer.extent()
            if extent.isEmpty():
                continue
            assert extent.intersects(dem_extent), (
                f"'{name}' sits at {extent.toString(1)}, nowhere near the DEM at "
                f"{dem_extent.toString(1)}")


def check_earthwork_layers_take_the_dems_crs(dem_path):
    """The design layers keep whatever CRS they were created with, and nothing
    clears them — so a feature drawn before a DEM loaded froze the wrong one in."""
    with PluginHarness(dem_path, project_crs=GEOGRAPHIC) as h:
        h.run_baseline()
        h.add_earthwork("swale", line_across_valley())
        h.plugin._earthworks._refresh_ew_layer()
        h.assert_no_errors("earthwork layers under a divergent project CRS")

        from terrainflow_assessment.qgis.controllers._layers import resolve_layer

        found = 0
        for layer_id in (h.state.ew_layer_ids or {}).values():
            layer = resolve_layer(h.project, layer_id)
            if layer is None:
                continue
            found += 1
            assert layer.crs() == h.dem_layer.crs(), (
                f"'{layer.name()}' is in {layer.crs().authid()}, "
                f"not the DEM's {h.dem_layer.crs().authid()}")
        assert found, "no earthwork layer was created"


def check_the_report_map_is_drawn_in_the_dems_crs(dem_path):
    """Not the project's. Every metre in the document is measured on the DEM's
    grid, and the scale bar divides the rendered extent to get metres per pixel —
    in degrees that is out by about five orders of magnitude."""
    with PluginHarness(dem_path, project_crs=GEOGRAPHIC) as h:
        h.run_baseline()
        controller = h.plugin._reporting
        data = controller._collect(controller._site_name())
        specs = controller._map_specs(data)
        assert specs, "no report maps were specified"
        for key, spec in specs.items():
            assert not spec.crs.isGeographic(), (
                f"the '{key}' map would be rendered in {spec.crs.authid()}")
            assert spec.crs == h.dem_layer.crs(), (
                f"the '{key}' map is in {spec.crs.authid()}, not the DEM's")


def check_a_geographic_map_gets_no_scale_bar_rather_than_a_wrong_one(dem_path):
    """A missing scale bar is a visible absence. A plausible wrong one is not,
    and it is the more dangerous of the two on a drawing somebody digs from.

    Painted onto flat mid-grey rather than a real map, so "was anything drawn
    here" is a question about the decoration and not about the terrain under it —
    a reprojected DEM leaves white in the corners, which is indistinguishable
    from the plate.
    """
    from qgis.core import QgsRectangle
    from qgis.PyQt.QtGui import QColor, QImage

    from terrainflow_assessment.qgis.adapters.map_image import draw_decorations

    def painted(projected):
        image = QImage(600, 400, QImage.Format_ARGB32)
        image.fill(QColor(90, 90, 90))
        draw_decorations(image, QgsRectangle(0, 0, 600, 400), projected=projected)
        return image

    with_bar, without = painted(True), painted(False)

    # The plate covers about half the sampled corner and its label is dark ink on
    # top, so "some of it" is the assertion; the absent case is flatly zero.
    assert _lightened(with_bar, "bottom-left") > 0.2, (
        "the projected map lost its scale bar")
    assert _lightened(without, "bottom-left") < 0.01, (
        "a geographic map was stamped with a metre scale bar")
    # The north arrow still holds either way — the map is north-up regardless.
    assert _lightened(without, "top-right") > 0.2, (
        "the north arrow went with it; it did not have to")


def _lightened(image, corner):
    """Fraction of that corner the decoration plate has lifted off the grey."""
    w, h = image.width(), image.height()
    if corner == "bottom-left":
        xs, ys = range(5, min(80, w)), range(max(0, h - 45), h - 5)
    else:
        # The arrow's plate is narrow — roughly 18 px wide — so the window is
        # sized to it rather than to a generic corner.
        xs, ys = range(max(0, w - 30), w - 5), range(5, min(40, h))
    lit = total = 0
    for y in ys:
        for x in xs:
            total += 1
            if image.pixelColor(x, y).red() > 140:
                lit += 1
    return lit / total if total else 0.0


def check_the_pdf_omits_the_bar_it_cannot_measure(dem_path):
    """Same rule on the other renderer, which draws its bar as a layout item."""
    from qgis.core import QgsCoordinateReferenceSystem, QgsLayoutItemScaleBar

    from terrainflow_assessment.modules.report_model import build_report
    from terrainflow_assessment.qgis.adapters.layout_pdf import MapSpec, build_layout

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        controller = h.plugin._reporting
        data = controller._collect(controller._site_name())
        report = build_report(data)
        specs = controller._map_specs(data)
        assert specs, "no report maps were specified"

        def bars(map_specs):
            layout = build_layout(h.project.instance(), report, maps=map_specs)
            return [i for i in layout.items()
                    if isinstance(i, QgsLayoutItemScaleBar)]

        assert bars(specs), "the projected report lost its scale bars"

        geographic = QgsCoordinateReferenceSystem(GEOGRAPHIC)
        degraded = {k: MapSpec(s.layers, s.extent, geographic, s.height_mm)
                    for k, s in specs.items()}
        assert not bars(degraded), (
            "the PDF stamped a metre scale bar on a map measured in degrees")


def check_an_area_layer_still_round_trips_under_a_geographic_project(dem_path):
    """The boundary is written to disk for the analysis to read; it must keep the
    coordinates it was drawn with whatever the project is displaying in."""
    with PluginHarness(dem_path, project_crs=GEOGRAPHIC) as h:
        path = h.plugin._baseline._layer_to_path(h.boundary_layer)
        assert path and os.path.exists(path)

        from qgis.core import QgsVectorLayer

        written = QgsVectorLayer(path, "written", "ogr")
        assert written.isValid()
        assert written.extent().intersects(h.dem_layer.extent()), (
            "the written boundary does not overlap the DEM it bounds")
