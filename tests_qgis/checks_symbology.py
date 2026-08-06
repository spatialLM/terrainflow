"""
Symbology checks — does the map actually communicate what the design says?

These exist because "the layer was created and the renderer built without error"
is not the same claim as "the user can see it". Every defect this module guards
produced a clean Tier 1 pass and a broken map:

  * a label placement left at its point-layer default, so no line feature was
    ever named;
  * a QgsTextFormat sized through setFont(), which silently discards the point
    size;
  * arrowheads set through methods that do not exist on the class, hasattr-guarded
    into a no-op;
  * annotation layers re-appended below the bands they annotate on every refresh.

None of those raise. All of them are visible the moment you look, which is why
the assertions here are about placement, order and rendered geometry rather than
about API calls having been made.
"""

from _harness import PluginHarness, line_across_valley

EW_TYPES = ("swale", "berm", "basin", "dam", "diversion")


def _basin_polygon(h, east_offset=0, north_offset=0, side=40):
    """A square basin footprint near the middle of the synthetic DEM."""
    from qgis.core import QgsGeometry, QgsPointXY

    ext = h.dem_layer.extent()
    cx = ext.xMinimum() + ext.width() / 2 + east_offset
    cy = ext.yMinimum() + ext.height() / 2 + north_offset
    half = side / 2.0
    return QgsGeometry.fromPolygonXY([[
        QgsPointXY(cx - half, cy - half), QgsPointXY(cx + half, cy - half),
        QgsPointXY(cx + half, cy + half), QgsPointXY(cx - half, cy + half),
        QgsPointXY(cx - half, cy - half),
    ]])


def _design_of_every_type(h):
    """One earthwork of each registered type, spread so they do not overlap."""
    for i, ew_type in enumerate(EW_TYPES):
        if ew_type == "basin":
            geom = _basin_polygon(h, north_offset=-120)
        else:
            geom = line_across_valley(row=30 + i * 45)
        h.add_earthwork(ew_type, geometry=geom)
    h.plugin._earthworks._refresh_ew_layer()


# ---------------------------------------------------------------- labels

def check_all_labelled_layers_have_placement(dem_path):
    """No labelled layer may be left on the default AroundPoint placement.

    AroundPoint is a *point* arrangement. On a line or polygon layer it yields no
    label at all — which is why no swale, berm, dam or diversion drain has ever
    carried a name on the map, while the basin (a polygon, and so tolerant of the
    default) labelled fine and made the bug look like a collision problem.

    Cheap on purpose: pure settings inspection, no rendering, so it runs in
    milliseconds and covers every layer the plugin will ever add.
    """
    from qgis.core import QgsProject, QgsVectorLayer, QgsWkbTypes

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _design_of_every_type(h)
        h.assert_no_errors("design of every type")

        offenders = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer) or not layer.labelsEnabled():
                continue
            labeling = layer.labeling()
            if labeling is None:
                continue
            settings = labeling.settings()
            geom_type = layer.geometryType()
            if geom_type == QgsWkbTypes.PointGeometry:
                continue        # AroundPoint is correct for a point layer
            if int(settings.placement) == 0:
                offenders.append(f"{layer.name()} ({QgsWkbTypes.geometryDisplayString(geom_type)})")

        if offenders:
            raise AssertionError(
                "labelled non-point layers left on the default AroundPoint "
                "placement, so they will render no labels at all:\n  "
                + "\n  ".join(offenders)
            )


def check_every_earthwork_type_labels(dem_path):
    """Every one of the five types puts its name on the canvas, at three scales.

    Asserted through the labelling engine rather than by counting pixels, so a
    font substitution cannot turn a real regression into a green run — or a green
    run into a phantom failure.

    Only features actually inside the current extent are required to label: at a
    tight scale most of the design is off-screen, and demanding a label for a
    feature nobody can see would be asserting the wrong thing.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _design_of_every_type(h)
        h.assert_no_errors("design of every type")
        h.sync_canvas()

        # Centre on the design, not on the DEM, so tight scales frame earthworks
        # rather than empty hillside.
        all_ew = h.state.earthwork_manager.get_all()
        bbox = all_ew[0].geometry.boundingBox()
        for ew in all_ew[1:]:
            bbox.combineExtentWith(ew.geometry.boundingBox())
        centre = bbox.center()

        missing = []
        seen_at_least_one = False
        for scale in (1000, 2500, 10000):
            h.set_scale(scale, centre=centre)
            extent = h.canvas.extent()
            placed = {t for texts in h.labels_drawn().values() for t in texts}
            for ew in all_ew:
                if not ew.geometry.boundingBox().intersects(extent):
                    continue
                seen_at_least_one = True
                if not any(ew.name in text for text in placed):
                    missing.append(f"1:{scale} — {ew.name} ({ew.type})")

        if not seen_at_least_one:
            raise AssertionError("no earthwork was ever in view — the check framed nothing")
        if missing:
            raise AssertionError(
                "earthworks in view whose name never reached the canvas:\n  "
                + "\n  ".join(missing)
            )


def check_label_text_size_is_honoured(dem_path):
    """The size asked for is the size set.

    QgsTextFormat.setFont() discards a QFont's point size — documented at
    baseline.py:472-474, and repeated in the earthwork labelling, where an
    intended 8 pt rendered at the 10 pt default. Routing every caller through
    _symbols.label_format() is the fix; this is the guard on it.
    """
    from terrainflow_assessment.qgis.controllers import _symbols as S

    with PluginHarness(dem_path):
        for asked in (8.0, 8.5, 9.0, 12.0):
            fmt = S.label_format("#00BCD4", size_pt=asked)
            if abs(fmt.size() - asked) > 1e-6:
                raise AssertionError(
                    f"label_format(size_pt={asked}) produced size {fmt.size()} — "
                    "the QFont point size is being discarded again"
                )


def check_point_labels_have_a_halo(dem_path):
    """Spillway and stress-point text must carry a buffer.

    Unbuffered near-black text over aerial imagery is invisible; that is exactly
    what the field screenshots showed. baseline.py has always buffered its exit
    labels — these two layers were the ones that never did.
    """
    from qgis.core import QgsProject, QgsVectorLayer

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", geometry=line_across_valley(row=60))
        h.plugin._earthworks._refresh_ew_layer()

        pt = ew.geometry.interpolate(ew.geometry.length() / 2).asPoint()
        h.plugin._earthworks._on_spillway_placed(ew.id, pt, 12.0, kind="outflow")
        h.assert_no_errors("spillway placement")

        naked = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer) or not layer.labelsEnabled():
                continue
            if layer.name() not in ("Spillways", "Stress points"):
                continue
            buf = layer.labeling().settings().format().buffer()
            if not buf.enabled() or buf.size() <= 0:
                naked.append(layer.name())

        if naked:
            raise AssertionError(
                f"point label layers with no halo: {', '.join(naked)} — "
                "they will disappear over imagery"
            )
