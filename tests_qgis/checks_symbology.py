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
from _shots import assert_rendered, save_canvas

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


# ---------------------------------------------------------------- band + signature

def _render(h):
    """Render the canvas to an image we can measure pixel-by-pixel."""
    from qgis.core import QgsMapRendererCustomPainterJob, QgsMapSettings
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QImage, QPainter

    settings = QgsMapSettings(h.canvas.mapSettings())
    img = QImage(settings.outputSize(), QImage.Format_ARGB32)
    img.fill(Qt.black)
    painter = QPainter(img)
    job = QgsMapRendererCustomPainterJob(settings, painter)
    job.start()
    job.waitForFinished()
    painter.end()
    return img


def _scanline(h, name, rgb, tol=60, x_frac=0.5, img=None):
    """Count pixels of a colour down one column of the rendered canvas."""
    img = _render(h) if img is None else img
    x = int(img.width() * x_frac)
    hits = 0
    for y in range(img.height()):
        c = img.pixelColor(x, y)
        if (abs(c.red() - rgb[0]) <= tol and abs(c.green() - rgb[1]) <= tol
                and abs(c.blue() - rgb[2]) <= tol):
            hits += 1
    return hits, img


def _band_thickness(h, rgb, tol=60):
    """Thickest run of the band colour found across several columns.

    Sampling one column is not enough: the feature's curved label runs along the
    band and punches a hole through whichever column it happens to cross, which
    reads as a band 30% thinner than it is. Taking the widest run over several
    columns measures the band where nothing is drawn on top of it.
    """
    img = _render(h)
    best = 0
    for frac in (0.12, 0.22, 0.32, 0.68, 0.78, 0.88):
        x = int(img.width() * frac)
        run = 0
        for y in range(img.height()):
            c = img.pixelColor(x, y)
            if (abs(c.red() - rgb[0]) <= tol and abs(c.green() - rgb[1]) <= tol
                    and abs(c.blue() - rgb[2]) <= tol):
                run += 1
            else:
                best = max(best, run)
                run = 0
        best = max(best, run)
    return best


SWALE_RGB = (0, 188, 212)
WHITE_RGB = (255, 255, 255)


def _one_swale(h, width_m=4.0):
    from qgis.core import QgsGeometry, QgsPointXY

    ext = h.dem_layer.extent()
    y = ext.yMinimum() + ext.height() / 2
    geom = QgsGeometry.fromPolylineXY([
        QgsPointXY(ext.xMinimum() - ext.width(), y),
        QgsPointXY(ext.xMaximum() + ext.width(), y),
    ])
    ew = h.add_earthwork("swale", geometry=geom)
    ew.top_width_m = width_m
    h.plugin._earthworks._refresh_ew_layer()
    return ew


def check_band_is_drawn_at_true_ground_width(dem_path):
    """The cyan band measures top_width_m on the ground, and halving the scale
    doubles it in pixels.

    This is the only check that actually demonstrates criterion 2. A width in
    metres that silently degrades to millimetres still builds a valid layer and
    passes every state assertion; here it is off by orders of magnitude.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = _one_swale(h, width_m=4.0)
        h.assert_no_errors("swale for width measurement")
        h.sync_canvas()

        centre = ew.geometry.boundingBox().center()
        problems, measured_px = [], {}
        # Both scales chosen so the band is comfortably above the minSizeMM floor
        # and thick enough that a pixel or two of antialiasing is not the story.
        for scale in (200, 400):
            h.set_scale(scale, centre=centre)
            mupp = h.canvas.mapSettings().mapUnitsPerPixel()
            expected = 4.0 / mupp
            measured = _band_thickness(h, SWALE_RGB)
            measured_px[scale] = (measured, expected)
            if abs(measured - expected) / expected > 0.20:
                problems.append(
                    f"1:{scale}: band measured {measured} px, expected ~{expected:.0f} px "
                    f"for 4.0 m at {mupp:.3f} m/px"
                )

        # Halving the scale must double the band: that is what distinguishes a
        # true ground width from a fixed size that merely happens to look right.
        fine, coarse = measured_px[200][0], measured_px[400][0]
        if coarse > 0 and not (1.6 <= fine / coarse <= 2.4):
            problems.append(
                f"band did not scale with zoom: {fine} px at 1:200 vs {coarse} px "
                f"at 1:400 (ratio {fine / coarse:.2f}, expected ~2)"
            )
        if problems:
            raise AssertionError("band width does not track true ground width:\n  "
                                 + "\n  ".join(problems))


def check_band_and_casing_survive_zoom_out(dem_path):
    """Zoomed right out, neither the band nor its white casing may vanish.

    A metres-in-map-units width goes sub-pixel and renders *nothing* — measured,
    a 2 m band is 0 px at roughly 1:15,000. The old workaround was a fixed 0.5 mm
    centreline drawn over the band, which left a hairline running proud down the
    middle at high zoom. QgsMapUnitScale.minSizeMM is the mechanism that was
    wanted; this asserts it is doing the job.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = _one_swale(h, width_m=2.0)
        h.assert_no_errors("swale for zoom-out")
        h.sync_canvas()
        centre = ew.geometry.boundingBox().center()

        problems = []
        for scale in (2000, 10000, 25000):
            h.set_scale(scale, centre=centre)
            band, _ = _scanline(h, "Swales", SWALE_RGB)
            casing, _ = _scanline(h, "Swales", WHITE_RGB, tol=40)
            if band <= 0:
                problems.append(f"1:{scale}: band disappeared entirely")
            if casing <= 0:
                problems.append(f"1:{scale}: white casing disappeared entirely")
        if problems:
            raise AssertionError("\n  ".join(["symbol vanishes when zoomed out:"] + problems))


def check_direction_marks_hold_across_scales(dem_path):
    """A diversion drain shows flow direction at every working scale.

    The old ribbon never rendered as designed: setArrowHeadLength /
    setArrowHeadThickness do not exist on QgsArrowSymbolLayer, and being
    hasattr-guarded they no-opped silently, leaving every head at the 1.5 mm
    default with the ribbon bowed off the line by isCurved. Millimetre-interval
    marker lines make cadence a function of on-screen length, so this is the
    check that keeps that property honest.
    """
    from qgis.core import QgsGeometry, QgsPointXY

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ext = h.dem_layer.extent()
        y = ext.yMinimum() + ext.height() / 2
        # Many vertices on purpose: the old mechanism drew one arrow per segment.
        n = 200
        pts = [
            QgsPointXY(ext.xMinimum() + ext.width() * i / (n - 1), y)
            for i in range(n)
        ]
        ew = h.add_earthwork("diversion", geometry=QgsGeometry.fromPolylineXY(pts))
        ew.top_width_m = 1.0
        h.plugin._earthworks._refresh_ew_layer()
        h.assert_no_errors("diversion for direction marks")
        h.sync_canvas()

        centre = ew.geometry.boundingBox().center()
        counts = {}
        for scale in (1000, 4000, 12000):
            h.set_scale(scale, centre=centre)
            # Arrowheads are white-filled with a coloured edge, so a horizontal
            # scan across the middle of the band crosses each one.
            _, img = _scanline(h, "Diversion Drains", WHITE_RGB)
            mid = img.height() // 2
            runs, inside = 0, False
            for x in range(img.width()):
                c = img.pixelColor(x, mid)
                white = c.red() > 200 and c.green() > 200 and c.blue() > 200
                if white and not inside:
                    runs += 1
                inside = white
            counts[scale] = runs

        if any(v < 2 for v in counts.values()):
            raise AssertionError(
                "flow-direction marks missing at some scale — "
                f"arrow runs per scale: {counts}"
            )


def check_drawn_group_order_is_stable(dem_path):
    """Annotation stays above the bands, and stays there after a refresh.

    Spillways, stress points and overflow connections are destroyed and rebuilt
    every refresh, and addLayer appends — so without an explicit restack a
    correctly-placed spillway ends up painted underneath the swale it sits on.
    Asserting it twice is the point: the bug is in the refresh cycle, not in the
    initial build.
    """
    from terrainflow_assessment.qgis.controllers import _groups as G

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = _one_swale(h, width_m=2.0)
        pt = ew.geometry.interpolate(ew.geometry.length() / 2).asPoint()
        h.plugin._earthworks._on_spillway_placed(ew.id, pt, 12.0, kind="outflow")
        h.assert_no_errors("spillway placement")

        def order():
            grp = h.plugin._earthworks.group_for(G.DRAWN)
            return [n.layer().name() for n in grp.findLayers() if n.layer()]

        first = order()
        h.plugin._earthworks._refresh_spillway_layer()
        second = order()

        for label, names in (("after first build", first), ("after refresh", second)):
            if "Spillways" not in names:
                raise AssertionError(f"{label}: no Spillways layer in the Drawn group")
            spill = names.index("Spillways")
            bands = [i for i, n in enumerate(names) if n.endswith("s") and n != "Spillways"
                     and n != "Stress points"]
            if bands and spill > min(bands):
                raise AssertionError(
                    f"{label}: Spillways sits below an earthwork band and will render "
                    f"underneath it — order was {names}"
                )


def check_restack_does_not_duplicate_layers(dem_path):
    """Repeated refreshes must not grow the layer tree.

    Reordering a group means rebuilding its child nodes, and a reorder that
    removes the wrong node leaves the replacement behind as a duplicate. That
    multiplies on every refresh — the tree still *looks* right, every state
    assertion still passes, and the canvas quietly gets slower until it stops
    finishing. Cheap to assert, nearly invisible otherwise.
    """
    from terrainflow_assessment.qgis.controllers import _groups as G

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _design_of_every_type(h)
        h.assert_no_errors("design of every type")

        def census():
            grp = h.plugin._earthworks.group_for(G.DRAWN)
            names = [n.layer().name() for n in grp.findLayers() if n.layer()]
            return len(names), sorted(names)

        before = census()
        for _ in range(3):
            h.plugin._earthworks._refresh_ew_layer()
        after = census()

        if after != before:
            raise AssertionError(
                "the Drawn group changed across repeated refreshes — layers are "
                f"being duplicated or lost.\n  before: {before}\n  after:  {after}"
            )


def check_types_read_apart_at_three_scales(dem_path):
    """Screenshots for the eye, at scales a land manager actually works at.

    The assertions above prove the mechanics — width in metres, marks in
    millimetres, order stable. They cannot say whether a dam reads as a dam. That
    judgement needs the images, so this leaves three of them side by side:
    1:1000 (walking a paddock), 1:4000 (a block), 1:12000 (the whole property).

    What to look for: at every scale each type is told apart by its *signature*,
    not merely by hue — ticks across a dam, arrows along a diversion drain,
    chevrons on a berm, plain band for a swale. Hue alone fails for anyone with a
    colour vision deficiency, and washes out over aerial imagery.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _design_of_every_type(h)
        h.assert_no_errors("design of every type")
        h.sync_canvas()

        all_ew = h.state.earthwork_manager.get_all()
        bbox = all_ew[0].geometry.boundingBox()
        for ew in all_ew[1:]:
            bbox.combineExtentWith(ew.geometry.boundingBox())
        centre = bbox.center()

        for scale in (1000, 4000, 12000):
            h.set_scale(scale, centre=centre)
            path = save_canvas(h.canvas, f"canvas_ew_1_{scale}")
            assert_rendered(path, f"earthwork types at 1:{scale}")


def check_drawn_group_sits_above_loose_design_layers(dem_path):
    """The drawn design must never end up under the catchment wash.

    The catchment raster is a loose Design-stage layer at alpha 150. Group
    placement otherwise tucks a subgroup beneath its parent's plain layers —
    correct for supplementary output like Contour Analysis, wrong here, because it
    buries every swale, dam and basin the user drew under a translucent overlay.
    Group position is decided once, when the group is created, so this exercises
    that moment directly: put a loose layer in Design *before* the Drawn group
    exists, then create it. Driving it through the panel does not reach this —
    the assessment recompute already creates the Drawn group for the overflow
    connections, so ordering is settled before any catchment layer appears.
    """
    from qgis.core import QgsLayerTreeLayer, QgsProject, QgsVectorLayer

    from terrainflow_assessment.qgis.controllers import _groups as G

    with PluginHarness(dem_path):
        project = QgsProject
        site, tag = "Ordering Site", "t1"

        loose = QgsVectorLayer("Point?crs=EPSG:2193", "Design — Catchments", "memory")
        G.add_layer(project, loose, G.DESIGN, site_name=site, tag=tag)

        drawn = G.group(project, G.DRAWN, site_name=site, tag=tag)
        design = G.group(project, G.DESIGN, site_name=site, tag=tag)

        children = design.children()
        names = [c.name() for c in children]
        drawn_at = children.index(drawn) if drawn in children else None
        if drawn_at is None:
            raise AssertionError(f"no Drawn Earthworks group under Design — got {names}")

        loose_at = [i for i, c in enumerate(children) if isinstance(c, QgsLayerTreeLayer)]
        if loose_at and drawn_at > min(loose_at):
            raise AssertionError(
                "Drawn Earthworks was created below a loose Design layer, so every "
                f"drawn earthwork renders under it — Design order: {names}"
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
