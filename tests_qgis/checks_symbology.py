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


# ---------------------------------------------------------------- spillways

def _swale_and_tool(h, kind="outflow"):
    """A swale, and the constrained place-point tool armed for it."""
    ew = _one_swale(h, width_m=2.0)
    h.plugin._earthworks.activate_place_spillway(kind=kind, index=0)
    return ew, h.canvas.mapTool()


def check_spillway_off_feature_is_refused(dem_path):
    """A click nowhere near the feature must be refused, and say so.

    This is not a tidiness rule. _on_spillway_placed samples the DEM at the point
    it is given and binds the crest to it, so an off-feature click sizes the weir
    from unrelated ground — the freeboard, head and required width all descend
    from a hillside the water never reaches. The field report that opened this
    work shows exactly that: an outflow for Swale 1 sitting out on its own.
    """
    from qgis.core import QgsPointXY

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.prepare_canvas_for_input()
        ew, tool = _swale_and_tool(h)
        assert tool is not None, "no map tool was armed"

        far = ew.geometry.interpolate(ew.geometry.length() / 2).asPoint()
        off = QgsPointXY(far.x(), far.y() + 120.0)      # 120 m off the alignment
        tool.canvasPressEvent(_press_at(h, off))

        if getattr(ew, "spillway", None) is not None:
            raise AssertionError("an off-feature click was accepted")
        if not any("must be placed on the feature" in str(w) for w in h.bar.warnings):
            raise AssertionError(
                f"no warning naming the rule — warnings were {h.bar.warnings}")
        if h.canvas.mapTool() is not tool:
            raise AssertionError(
                "the tool disarmed after a mis-click; a near miss should cost one "
                "more click, not a trip back through the menu")


def check_spillway_near_feature_snaps_onto_it(dem_path):
    """A near click is snapped onto the alignment, and the crest is read there.

    Sampling at the snapped point rather than the raw click is the correctness
    half of the fix — otherwise the recorded location and the elevation behind
    it describe two different places.
    """
    from qgis.core import QgsGeometry, QgsPointXY

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.prepare_canvas_for_input()
        ew, tool = _swale_and_tool(h)

        on = ew.geometry.interpolate(ew.geometry.length() / 2).asPoint()
        near = QgsPointXY(on.x(), on.y() + h.canvas.mapUnitsPerPixel() * 3)
        tool.canvasPressEvent(_press_at(h, near))

        sp = getattr(ew, "spillway", None)
        if sp is None or not sp.point_wkt:
            raise AssertionError("a click 3 px off the feature was refused")

        placed = QgsGeometry.fromWkt(sp.point_wkt)
        if ew.geometry.distance(placed) > 0.01:
            raise AssertionError(
                f"the recorded point is {ew.geometry.distance(placed):.3f} m off the "
                "feature — it was stored unsnapped"
            )


def check_spillway_sill_is_drawn_at_the_built_width(dem_path):
    """The crest is drawn as a bar of the built width, square across the feature.

    Before this, Spillway.width_m existed only as label text: a 0.5 m sill and a
    6 m emergency weir drew as the same 4 mm triangle at every scale.
    """
    import math

    from qgis.core import QgsProject, QgsVectorLayer

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = _one_swale(h, width_m=2.0)
        mid = ew.geometry.interpolate(ew.geometry.length() / 2).asPoint()

        for width in (0.8, 6.0):
            ew.spillway = None
            h.plugin._earthworks._on_spillway_placed(ew.id, mid, 12.0, kind="outflow")
            ew.spillway.width_m = width
            h.plugin._earthworks._refresh_spillway_layer()
            h.assert_no_errors(f"spillway at {width} m")

            layer = next(
                (lyr for lyr in QgsProject.instance().mapLayers().values()
                 if isinstance(lyr, QgsVectorLayer) and lyr.name() == "Spillways"), None)
            assert layer is not None, "no Spillways layer"
            feat = next(layer.getFeatures(), None)
            assert feat is not None, "the Spillways layer is empty"

            geom = feat.geometry()
            if geom.type() != 1:                     # LineGeometry
                raise AssertionError("the spillway is not drawn as a line")
            if abs(geom.length() - width) > width * 0.02:
                raise AssertionError(
                    f"sill is {geom.length():.2f} m for a {width} m weir")

            pts = geom.asPolyline()
            sill = math.atan2(pts[1].y() - pts[0].y(), pts[1].x() - pts[0].x())
            ew_pts = ew.geometry.asPolyline()
            align = math.atan2(ew_pts[1].y() - ew_pts[0].y(),
                               ew_pts[1].x() - ew_pts[0].x())
            off = abs((sill - align) % math.pi - math.pi / 2)
            if off > math.radians(5):
                raise AssertionError(
                    f"sill is {math.degrees(off):.1f}° off square to the feature")


def check_basin_spillway_must_sit_on_the_rim(dem_path):
    """A spillway inside a basin's footprint is refused.

    A spillway is a notch in the rim water leaves over. A point in the middle of
    the polygon is not somewhere you can build one, so the constraint is the
    boundary rather than the fill.
    """
    from qgis.core import QgsPointXY

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.prepare_canvas_for_input()
        ew = h.add_earthwork("basin", geometry=_basin_polygon(h, side=80))
        h.plugin._earthworks._refresh_ew_layer()
        h.plugin._earthworks.activate_place_spillway(kind="outflow", index=0)
        tool = h.canvas.mapTool()
        assert tool is not None, "no map tool was armed for the basin"

        centre = ew.geometry.centroid().asPoint()
        tool.canvasPressEvent(_press_at(h, QgsPointXY(centre)))

        if getattr(ew, "spillway", None) is not None:
            raise AssertionError(
                "a spillway was accepted in the middle of the basin floor")


def check_reshape_keeps_the_spillway_on_its_feature(dem_path):
    """Dragging a vertex must not leave the spillway floating.

    The alignment moves out from under the spillway, and for a diversion drain
    _orient_downhill can reverse the vertex order — flipping the sill's
    perpendicular and the chevron's direction. Nothing about that is visible
    until someone inspects a design they have already signed off.
    """
    from qgis.core import QgsGeometry, QgsPointXY

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = _one_swale(h, width_m=2.0)
        mid = ew.geometry.interpolate(ew.geometry.length() / 2).asPoint()
        h.plugin._earthworks._on_spillway_placed(ew.id, mid, 12.0, kind="outflow")
        assert ew.spillway is not None and ew.spillway.point_wkt

        pts = ew.geometry.asPolyline()
        moved = QgsGeometry.fromPolylineXY(
            [QgsPointXY(p.x(), p.y() + 60.0) for p in pts])
        h.plugin._earthworks._on_vertex_edit_finished(0, moved)
        h.assert_no_errors("reshape with a spillway attached")

        placed = QgsGeometry.fromWkt(ew.spillway.point_wkt)
        gap = ew.geometry.distance(placed)
        if gap > 0.01:
            raise AssertionError(
                f"after reshaping, the spillway sits {gap:.1f} m off its feature")


# ---------------------------------------------------------------- connections

def _connections_layer():
    from qgis.core import QgsProject, QgsVectorLayer
    return next(
        (lyr for lyr in QgsProject.instance().mapLayers().values()
         if isinstance(lyr, QgsVectorLayer) and lyr.name() == "Overflow connections"),
        None,
    )


def check_user_link_draws_even_when_it_carries_no_volume(dem_path):
    """A link the user drew is part of the design and must appear.

    The field log reads "Swale 1 now overflows into Swale 2" — the link
    registered — and then nothing appeared on the map, which read as Route
    Overflow being broken. The cause was a render-time skip of any edge carrying
    zero volume for the current storm.

    The count is asserted exactly, not as "at least one". resolve_targets writes
    an edge for EVERY store, so a rule that simply drew them all would put a line
    on every earthwork on the site — passing a >=1 assertion while being wrong.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        a = h.add_earthwork("swale", geometry=line_across_valley(row=40))
        b = h.add_earthwork("swale", geometry=line_across_valley(row=70))
        c = h.add_earthwork("swale", geometry=line_across_valley(row=100))
        # Give the source enough storage that this storm never fills it, so its
        # link carries zero volume. That is the case the old rule dropped, and
        # the case the field report hit.
        a.capacity_m3 = 5_000_000.0
        a.capacity_l = a.capacity_m3 * 1000.0
        h.plugin._earthworks._refresh_ew_layer()

        # One deliberate link; c is left unlinked.
        h.plugin._earthworks.on_connection_made(a.id, b.id)
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("user link")

        layer = _connections_layer()
        if layer is None:
            raise AssertionError("a user-drawn link produced no connections layer")

        drawn = [(f["from_name"], f["to_name"], f["is_user_link"], f["overflow_m3"])
                 for f in layer.getFeatures()]
        if not any(d[0] == a.name and d[2] == 1 for d in drawn):
            raise AssertionError(
                f"the user's link from {a.name} was not drawn — got {drawn}")

        # The other half of the rule, stated directly rather than as a count:
        # an inferred link only earns a line by carrying water. resolve_targets
        # writes an edge for every store, so without this a line appears on every
        # earthwork on the site.
        idle = [d for d in drawn if d[2] == 0 and (d[3] or 0) <= 0]
        if idle:
            raise AssertionError(
                f"inferred links with no flow were drawn anyway — {idle}")
        assert c is not None


def check_connections_do_not_outweigh_the_earthworks(dem_path):
    """A link annotates the design; it must not be the heaviest ink on the map.

    These were up to 3 mm of saturated blue spanning the canvas, thicker than the
    structures they describe, with volume driving the width. Fixed weight now,
    and the check is on the symbol rather than on pixels because the failure mode
    is "someone re-adds a data-defined width".
    """
    from qgis.core import QgsSimpleLineSymbolLayer, QgsSymbolLayer

    from terrainflow_assessment.qgis.controllers import _symbols as S

    with PluginHarness(dem_path):
        symbol = S.connection_symbol()
        widths = []
        for i in range(symbol.symbolLayerCount()):
            sl = symbol.symbolLayer(i)
            if not isinstance(sl, QgsSimpleLineSymbolLayer):
                continue        # arrowhead markers are sized, not stroked
            widths.append(sl.width())
            prop = sl.dataDefinedProperties().property(
                QgsSymbolLayer.PropertyStrokeWidth)
            if prop is not None and prop.isActive():
                raise AssertionError(
                    "connection stroke width is data-defined again — volume belongs "
                    "in the Live Assessment panel, not in line thickness"
                )
        if not widths:
            raise AssertionError("the connection symbol has no stroked line layer")
        if max(widths) > 2.0:
            raise AssertionError(
                f"connection line is {max(widths)} mm — heavier than the design it "
                "annotates"
            )


def _press_at(h, point):
    """A synthetic left-click at a map coordinate."""
    from qgis.PyQt.QtCore import QEvent, QPoint, Qt
    from qgis.PyQt.QtGui import QMouseEvent

    pixel = h.canvas.getCoordinateTransform().transform(point)
    return QMouseEvent(
        QEvent.MouseButtonPress,
        QPoint(int(pixel.x()), int(pixel.y())),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
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
