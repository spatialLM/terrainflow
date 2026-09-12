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
from _shots import assert_rendered, save_canvas, save_qimage

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


def _registry_rgb(ew_type):
    """The type's colour, read from the registry rather than transcribed.

    These two checks measure a band by hunting for its colour in the rendered
    pixels, so a hard-coded hex turns any deliberate palette change into a
    failure that reads like a broken symbol.
    """
    from qgis.PyQt.QtGui import QColor

    from terrainflow_assessment.core.registry.earthwork_types import get_type

    c = QColor(get_type(ew_type).style[1])
    return (c.red(), c.green(), c.blue())


SWALE_RGB = _registry_rgb("swale")
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

        # Band layers named from the registry, not guessed from the name. Every
        # earthwork layer ends in "s" — but so does "Overflow connections", and
        # treating that as a band made this check fail on a correct tree.
        from terrainflow_assessment.core.registry.earthwork_types import all_types
        band_names = {f"{cfg.label}s" for cfg in all_types().values()}

        for label, names in (("after first build", first), ("after refresh", second)):
            if "Spillways" not in names:
                raise AssertionError(f"{label}: no Spillways layer in the Drawn group")
            spill = names.index("Spillways")
            bands = [i for i, n in enumerate(names) if n in band_names]
            if bands and spill > min(bands):
                raise AssertionError(
                    f"{label}: Spillways sits below an earthwork band and will render "
                    f"underneath it — order was {names}"
                )


def check_refreshing_annotations_keeps_every_band_layer(dem_path):
    """Refreshing a spillway must not take the earthwork layers with it.

    The Drawn group's order used to be fixed by re-sorting it: drop every child
    node, re-add in the wanted order. In QGIS Desktop that is fatal — the
    layer-tree registry bridge reads "node removed from tree" as "the user
    deleted this layer" and removes it from the project. Every earthwork blinked
    into the panel and vanished, and the next draw dereferenced a dead wrapper.
    Disabling the bridge did not prevent it.

    Ordering is now decided when a layer is inserted (`at_top=True`) and nothing
    re-sorts anything. This asserts the invariant that broke: after the
    annotation layers rebuild, every band layer is still there.

    Headless there is no bridge, so this passes either way — the real proof is
    run_qgis_gui_shot.ps1, which drives the same sequence against QGIS Desktop.
    Kept here because it is the cheap guard against a re-sort creeping back in.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _design_of_every_type(h)
        h.assert_no_errors("initial design")

        before = dict(h.state.ew_layer_ids)
        if not before:
            raise AssertionError("no earthwork layers were registered")

        swale = next(e for e in h.state.earthwork_manager.get_all()
                     if e.type == "swale")
        pt = swale.geometry.interpolate(swale.geometry.length() / 2).asPoint()
        h.plugin._earthworks._on_spillway_placed(swale.id, pt, 12.0, kind="outflow")
        h.plugin._earthworks._refresh_spillway_layer()

        missing = [k for k, lid in before.items()
                   if h.plugin._earthworks._project.instance().mapLayer(lid) is None]
        if missing:
            raise AssertionError(
                f"refreshing the spillway layer removed these from the project: {missing}")

        # The crash itself: drawing again afterwards.
        h.add_earthwork("swale", geometry=line_across_valley(row=120))
        h.plugin._earthworks._refresh_ew_layer()
        h.assert_no_errors("drawing after an annotation refresh")


def check_earthwork_layers_are_held_by_id(dem_path):
    """_state.ew_layer_ids must hold ids, not layer objects.

    CLAUDE.md's rule, and this is the crash it exists to prevent: a stored wrapper
    outlives the C++ object behind it, so the next attribute access raises rather
    than the layer being quietly rebuilt.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _design_of_every_type(h)
        for key, value in h.state.ew_layer_ids.items():
            if not isinstance(value, str):
                raise AssertionError(
                    f"ew_layer_ids[{key!r}] holds {type(value).__name__}, not a layer id"
                )


def check_repeated_refreshes_do_not_grow_the_tree(dem_path):
    """Repeated refreshes must not add or lose layers.

    Annotation layers are destroyed and rebuilt on every refresh. A rebuild that
    fails to remove the old node leaves a duplicate, and one that removes too
    much loses the layer — either way the tree still *looks* plausible, every
    state assertion still passes, and the canvas quietly degrades. Cheap to
    assert, nearly invisible otherwise.
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
    """The crest is drawn as a bar of the built width, running ALONG the feature.

    Two things at once, and the second one changed. Spillway.width_m used to exist only
    as label text — a 0.5 m sill and a 6 m emergency weir drew as the same 4 mm triangle
    at every scale — so the bar has to be at the built width.

    It also has to point the right way, and this check used to assert the opposite. A
    weir's crest is the line the flow **crosses** on its way out, so it lies along the
    bank; across the bank is the direction the water travels. That was arguable while
    the bar was only a cartographic gate symbol, and it stopped being arguable when the
    bar became the geometry the burn cuts: a notch lowered across the alignment runs
    down the flow path instead of through the bank. The perpendicular is kept — it is
    the breach axis, and ``plan_geometry.perpendicular_sill`` still returns it — but it
    is not the crest.
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
            off = (sill - align) % math.pi
            off = min(off, math.pi - off)
            if off > math.radians(5):
                raise AssertionError(
                    f"crest bar is {math.degrees(off):.1f}° off the feature's own "
                    f"alignment — it must run along the bank, not across it")


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


# ---------------------------------------------------------------------------
# The shared raster ramps, as pixels
# ---------------------------------------------------------------------------
#
# `tests/test_map_palette.py` asserts the stop *table* and `checks_terrain`
# photographs the terrain **panel**. Between the two sits everything that turns a
# table into a map — `apply_raster_ramp`'s fractional scaling, its `absolute=`
# path, the symmetric anchoring curvature needs, the band maximum it scales by and
# the renderer's own interpolation — and none of it was watched by anything. H-10
# moved CURVATURE's two flanking stops from alpha 200 to 255, which is instantly
# visible on the map, and all 50 screenshot baselines stayed byte-identical.
#
# So: render every ramp over one fixed gradient, photograph the sheet, and assert
# each stop's colour actually reaches the pixels. The expectation is computed
# *from the palette table*, so this does not freeze the palette — an intentional
# recolour moves both sides and only the baseline image needs accepting. What it
# catches is the table and the pixels disagreeing, which is the whole gap.
#
# The background is **white and opaque** on purpose. Alpha is only visible against
# something, and over a transparent background an alpha regression is a no-op in
# the PNG — which is exactly how H-10 went unseen.

#: One output pixel per raster cell throughout, so no resampling stands between a
#: raster value and the pixel its colour is read from.
STRIP_W, STRIP_H = 361, 40
#: The staircase: one block this wide per stop, on a strip this tall.
STEP_W, STEP_H = 36, 22
LABEL_W = 150


def _ramp_strips():
    """``(name, stops, lo, hi, kwargs)`` — a ramp and the span it is painted over.

    Each ramp gets the span its own stops describe. A 0-1 fraction ramp rendered
    over [-1, 360] is 99% clamp and says nothing about the ramp.
    """
    from terrainflow_assessment.core.registry import map_palette as P

    return (
        ("streams", P.STREAMS, 0.0, 1.0, {}),
        ("water_captured", P.WATER_CAPTURED, 0.0, 1.0, {}),
        ("surface_runoff", P.surface_runoff_ramp(), 0.0, 1.0, {}),
        ("wetness_index", P.WETNESS_INDEX, 0.0, 1.0, {}),
        ("erosive_power", P.EROSIVE_POWER, 0.0, 1.0, {}),
        # Diverging, anchored symmetrically the way `terrain._apply_index_ramp`
        # does it. The bound is fixed at 1.0 rather than taken from a percentile,
        # so the baseline does not move with whatever fixture computed it.
        ("curvature", P.CURVATURE, -1.0, 1.0,
         {"max_value": 1.0, "min_value": -1.0}),
        # Absolute — compass degrees, not fractions. Sending this one through the
        # fractional path is what drew the whole aspect map as a single flat wash,
        # measured at 0.32% of the ramp occupied.
        ("aspect_classes", P.ASPECT_CLASSES, -1.0, 360.0, {"absolute": True}),
    )


def _write_strip(path, row, height):
    """One float32 GeoTIFF, *height* identical rows of *row*."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    data = np.repeat(np.asarray(row, dtype="float32")[None, :], height, axis=0)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=len(row), count=1,
        dtype="float32", crs="EPSG:2193",
        transform=from_origin(0.0, float(height), 1.0, 1.0),
    ) as dst:
        dst.write(data, 1)
    return path


def _gradient_raster(path, lo, hi):
    """A STRIP_W-wide float32 GeoTIFF ramping linearly lo -> hi across.

    This one is for the **picture**: it visits every stop and every interpolated
    span between them, which is what makes an alpha regression across a whole
    span obvious in the baseline diff.
    """
    import numpy as np

    return _write_strip(path, np.linspace(lo, hi, STRIP_W), STRIP_H)


def _steps_raster(path, values):
    """A staircase: one equal block of columns per value, filled with that value.

    This one is for the **assertion**, and the gradient cannot do its job. STREAMS
    puts stops at 0.0 and 0.001 and WATER_CAPTURED does the same — 0.36 of a column
    apart on a 361-wide strip, so on a gradient the two are simply not separable
    and a sample taken at either lands on an interpolated blend of both. A block
    per stop holds each value flat across ~36 px whatever its neighbours are, so
    the colour read back is the renderer's answer for *that* value and nothing else.
    """
    import numpy as np

    n = len(values)
    width = n * STEP_W
    row = np.empty(width, dtype="float32")
    for i, value in enumerate(values):
        row[i * STEP_W:(i + 1) * STEP_W] = float(value)
    return _write_strip(path, row, STEP_H)


def _render_strip(layer, width, height):
    """Render one raster layer 1:1 over opaque white, with no antialiasing."""
    from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QColor

    ms = QgsMapSettings()
    ms.setLayers([layer])
    ms.setDestinationCrs(layer.crs())
    ms.setOutputSize(QSize(width, height))
    ms.setExtent(layer.extent())
    ms.setBackgroundColor(QColor(255, 255, 255))
    # Off: this is a 1:1 render and the point is to read exact colours.
    # Antialiasing would smear every stop into its neighbour.
    ms.setFlag(QgsMapSettings.Antialiasing, False)
    job = QgsMapRendererParallelJob(ms)
    job.start()
    job.waitForFinished()
    return job.renderedImage()


def _over_white(rgba):
    """``(r, g, b, a)`` composited onto opaque white — what the pixel must be."""
    r, g, b, a = rgba
    f = a / 255.0
    return tuple(int(round(c * f + 255 * (1.0 - f))) for c in (r, g, b))


def _stop_value(stop_value, hi, kwargs):
    """The band value `apply_raster_ramp` puts this stop at.

    Mirrors the function rather than assuming: an absolute stop is laid down
    untouched, a fractional one is multiplied by the ramp's top, which is
    ``max_value`` when given and the band maximum — here ``hi`` — when not. The
    `min_value` the curvature call passes is deliberately not in this: a negative
    floor is reset to zero inside `apply_raster_ramp`, which is why the diverging
    ramp's -1.0 lands on -top rather than on the floor.
    """
    if kwargs.get("absolute"):
        return float(stop_value)
    top = kwargs.get("max_value")
    return float(stop_value) * float(hi if top is None else top)


def check_every_shared_ramp_reaches_the_pixels(dem_path):
    """Each ramp rendered twice — as a gradient, and as a block per stop.

    The gradient is the picture the baseline diff watches; the staircase is what
    the assertion reads, because two of these ramps put stops a thousandth apart
    and a gradient cannot separate those at any width.

    `apply_raster_ramp` is called directly and not through `apply_shared_ramp`,
    which CLAUDE.md otherwise forbids. The family-top logic `apply_shared_ramp`
    adds is already covered by
    `check_matching_before_and_after_layers_share_one_ramp`; what was uncovered is
    the primitive underneath it, and a family of one has no top to share.

    The DEM argument is unused — the input is synthetic, so this costs two small
    renders per ramp and says the same thing on any machine and any fixture.
    """
    import tempfile
    from pathlib import Path

    from qgis.core import QgsRasterLayer
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QFont, QImage, QPainter

    from terrainflow_assessment.qgis.controllers._symbols import apply_raster_ramp

    strips = _ramp_strips()
    gap = 6
    block = STRIP_H + STEP_H + gap
    sheet = QImage(LABEL_W + STRIP_W, len(strips) * block + gap,
                   QImage.Format_ARGB32)
    sheet.fill(Qt.white)
    painter = QPainter(sheet)
    painter.setFont(QFont("Arial", 9))

    failures = []
    tmp = Path(tempfile.mkdtemp(prefix="tfa_ramps_"))
    try:
        for i, (name, stops, lo, hi, kwargs) in enumerate(strips):
            values = [_stop_value(v, hi, kwargs) for v, _rgba, _l in stops]
            top = gap + i * block

            layer = QgsRasterLayer(
                _gradient_raster(str(tmp / f"{name}_grad.tif"), lo, hi),
                name, "gdal")
            steps = QgsRasterLayer(
                _steps_raster(str(tmp / f"{name}_step.tif"), values),
                f"{name}_steps", "gdal")
            if not layer.isValid() or not steps.isValid():
                failures.append(f"{name}: raster layer did not load")
                continue
            # Both layers take the ramp built for *this* ramp's own span. The
            # staircase's band maximum is the largest stop value, which for every
            # ramp here is the same number the gradient's `hi` is — but the
            # fractional path scales by the band maximum, so it is passed
            # explicitly rather than left to coincide.
            explicit = dict(kwargs)
            if not explicit.get("absolute") and explicit.get("max_value") is None:
                explicit["max_value"] = float(hi)
            apply_raster_ramp(layer, stops, **explicit)
            apply_raster_ramp(steps, stops, **explicit)

            image = _render_strip(layer, STRIP_W, STRIP_H)
            step_img = _render_strip(steps, len(values) * STEP_W, STEP_H)
            if (image is None or image.isNull()
                    or step_img is None or step_img.isNull()):
                failures.append(f"{name}: rendered nothing")
                continue

            painter.drawImage(LABEL_W, top, image)
            painter.drawImage(LABEL_W, top + STRIP_H, step_img)
            painter.setPen(Qt.black)
            painter.drawText(6, top + STRIP_H // 2 + 4, name)
            failures.extend(_stops_that_did_not_render(name, stops, step_img))
    finally:
        painter.end()

    path = save_qimage(sheet, "ramp_renders")
    assert_rendered(path, "shared raster ramps", min_colours=64)
    assert not failures, (
        "the palette table and the rendered pixels disagree:\n      "
        + "\n      ".join(failures)
        + f"\n    Look at {path}. A stop colour changed on purpose moves both "
          f"sides of this and only the baseline image needs accepting; a failure "
          f"means apply_raster_ramp is not laying the stops where the table puts "
          f"them."
    )


def _stops_that_did_not_render(name, stops, step_img):
    """Every stop's colour must come back from the block filled with its value."""
    out = []
    for i, (stop_value, rgba, label) in enumerate(stops):
        got = step_img.pixelColor(i * STEP_W + STEP_W // 2, STEP_H // 2)
        want = _over_white(rgba)
        off = max(abs(got.red() - want[0]), abs(got.green() - want[1]),
                  abs(got.blue() - want[2]))
        if off > 4:
            out.append(
                f"{name} stop {label!r} ({stop_value}): rendered "
                f"({got.red()},{got.green()},{got.blue()}) but the palette says "
                f"{tuple(rgba)}, which is {want} over white — off by {off}"
            )
    return out
