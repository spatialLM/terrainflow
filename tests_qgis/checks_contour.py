"""Contour analysis, Processing integration, and the keypoint/keyline paths."""

from _harness import PluginHarness
from qgis.PyQt.QtCore import Qt


def check_recommend_ponds(dem_path):
    """Pond-site recommendation, end to end: keypoints first, then dam sites.

    Quarantined in ``checks_slow`` until Round 15 as "does not finish — ran >12 min on a
    120x120 DEM with no result and no error", with the cause down as either pathological
    smooth terrain or a non-terminating loop. **It was neither.** The check emitted the
    recommendation without running the keypoint pass, so the controller put up a modal
    *"Run 'Find Keypoints + Ridgelines' first."* and waited for a click that never comes
    offscreen. ``recommend_pond_sites`` itself runs in ~0.05 s on this DEM.

    Ordering is the whole point of the check now: the two panel buttons have a
    prerequisite between them, and nothing else asserts it.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        h.panel.run_keypoint_analysis_requested.emit()
        h.assert_no_errors("keypoint analysis")
        assert h.state.found_keypoints, (
            "no keypoints found — there is nothing to site a pond against, and the "
            "recommendation below would be asserting on an empty list"
        )

        h.panel.recommend_ponds_requested.emit()
        h.assert_no_errors("recommend ponds")

        from qgis.core import QgsProject

        sites = QgsProject.instance().mapLayersByName("Recommended Pond Sites")
        assert sites, "no 'Recommended Pond Sites' layer created"
        assert sites[0].featureCount() > 0, "pond sites layer is empty"


def check_recommend_ponds_without_keypoints_warns(dem_path):
    """The guard that used to hang the suite, asserted rather than tripped over.

    It is a modal dialog, so offscreen it blocks until the module's timeout unless the
    harness records it — which is what ``RecordingDialogs`` is for.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.recommend_ponds_requested.emit()

        warnings = h.dialogs.of("warning")
        assert warnings, "expected a modal warning when no keypoints have been found"
        assert "Find Keypoints" in warnings[0][2], h.dialogs.render()


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


def check_contour_tick_hides_it_on_the_map(dem_path):
    """Unticking a row must actually take that contour off the canvas.

    The checkbox was decorative — nothing connected it to the layer — so this
    asserts the rendered feature count, not just that a signal fired.
    """
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        layer = QgsProject.instance().mapLayer(h.state.contour_layer_id)
        assert layer is not None, "no contour layer to filter"
        before = layer.featureCount()
        assert before > 1, "need more than one contour to test hiding one"

        rows = h.panel._contour_list
        assert rows.count() > 1, "contour list did not populate"
        rows.item(0).setCheckState(Qt.Unchecked)
        h.assert_no_errors("untick a contour")

        assert layer.featureCount() == before - 1, (
            f"unticking a row left {layer.featureCount()} of {before} contours "
            "drawn — the checkbox is not reaching the layer"
        )
        assert not h.state.contour_features[0].selected, (
            "the tick did not reach the ContourFeature, so it will not scope "
            "Select Top Swales or the gradient either"
        )

        rows.item(0).setCheckState(Qt.Checked)
        assert layer.featureCount() == before, "re-ticking did not restore the contour"


def check_top_swales_respects_unticked_contours(dem_path):
    """"Best N" means best N of what is still ticked, not of everything."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        h.panel._top_n_spin.setValue(3)
        h.panel.select_top5_contours_requested.emit()
        h.assert_no_errors("select top swales")
        full = [f.rank for f in h.state.top_contour_features]
        assert len(full) == 3, f"expected 3 top swales, got {len(full)}"

        # Drop the current best one and re-select: it must fall out of the answer.
        best = h.state.contour_features[0]
        h.panel._contour_list.item(0).setCheckState(Qt.Unchecked)
        h.panel.select_top5_contours_requested.emit()
        h.assert_no_errors("select top swales after untick")

        picked = h.state.top_contour_features
        assert len(picked) == 3, f"expected 3 top swales, got {len(picked)}"
        assert all(f is not best for f in picked), (
            "an unticked contour was still chosen as a top swale"
        )


def check_inflow_gradient_scopes_to_top_swales(dem_path):
    """The gradient grades the top-N pick once one exists, not the whole site."""
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        h.panel.show_inflow_bands_requested.emit(True)
        h.assert_no_errors("inflow gradient, unscoped")
        unscoped = QgsProject.instance().mapLayer(h.state.inflow_bands_layer_id)
        assert unscoped is not None, "no inflow gradient layer"
        n_unscoped = unscoped.featureCount()
        h.panel.show_inflow_bands_requested.emit(False)

        h.panel._top_n_spin.setValue(2)
        h.panel.select_top5_contours_requested.emit()
        h.panel.show_inflow_bands_requested.emit(True)
        h.assert_no_errors("inflow gradient, scoped to top swales")

        scoped = QgsProject.instance().mapLayer(h.state.inflow_bands_layer_id)
        assert scoped is not None, "no scoped inflow gradient layer"
        assert scoped.featureCount() < n_unscoped, (
            f"scoping to the top 2 swales still graded {scoped.featureCount()} of "
            f"{n_unscoped} stretches — the scope is not being applied"
        )


def check_inflow_bands_are_legible_over_imagery(dem_path):
    """Every inflow view must vary line *width*, opaquely, not just colour.

    This is the whole legibility fix: over aerial imagery a colour step can be
    wiped out by sunlit grass or tree shadow, so width carries the ordering and a
    white halo separates the line from whatever is under it. A future tidy-up that
    collapses the bands back to one width would silently undo it.
    """
    from qgis.core import QgsGraduatedSymbolRenderer, QgsProject

    def _core(symbol):
        """The coloured core line — last symbol layer, since a halo is inserted
        beneath at index 0. symbol.width() would report the halo instead."""
        return symbol.symbolLayer(symbol.symbolLayerCount() - 1)

    def _band_widths(renderer, what):
        assert isinstance(renderer, QgsGraduatedSymbolRenderer), (
            f"{what}: expected banded rendering, got {type(renderer).__name__}"
        )
        widths, seen_halo = [], []
        for rng in renderer.ranges():
            sym = rng.symbol()
            widths.append(_core(sym).width())
            seen_halo.append(sym.symbolLayerCount() > 1)
            assert _core(sym).color().alpha() == 255, (
                f"{what}: a translucent band — that is what was illegible over "
                "imagery; quiet bands recede by being thin, not faint"
            )
        return widths, seen_halo

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.panel.show_inflow_bands_requested.emit(True)
        h.assert_no_errors("inflow gradient")

        grad = QgsProject.instance().mapLayer(h.state.inflow_bands_layer_id)
        widths, halos = _band_widths(grad.renderer(), "inflow gradient")
        assert widths == sorted(widths), f"band widths are not ascending: {widths}"
        assert max(widths) >= 3 * min(widths), (
            f"band widths {widths} are too close to read apart over imagery"
        )
        assert all(halos), "inflow gradient bands have no white halo beneath them"

        # The candidate contours draw the same lines — two renderings of one line
        # is what made the gradient unreadable, so they step aside while it is up.
        node = QgsProject.instance().layerTreeRoot().findLayer(h.state.contour_layer_id)
        assert node is not None and not node.itemVisibilityChecked(), (
            "the candidate contour layer is still drawn under the gradient"
        )
        h.panel.show_inflow_bands_requested.emit(False)
        assert node.itemVisibilityChecked(), (
            "turning the gradient off did not bring the candidate contours back"
        )

        # Same contract for the peak-inflow overlay inside the swale segments,
        # minus the halo — the green core is its backdrop there.
        h.panel.find_segments_requested.emit()
        h.panel._segment_gradient_check.setChecked(True)
        h.assert_no_errors("segment inflow overlay")
        seg_grad = QgsProject.instance().mapLayer(h.state.segment_gradient_layer_id)
        seg_widths, _ = _band_widths(seg_grad.renderer(), "segment overlay")
        assert seg_widths == sorted(seg_widths), (
            f"segment band widths are not ascending: {seg_widths}"
        )
        # The two are measured in different units and must stay that way, so there is no
        # width comparison to make between them any more. This used to assert the widest
        # overlay band fitted inside the green core — true only while both were in
        # millimetres, and silently meaningless the moment the core became a real width.
        from qgis.core import QgsSymbolLayer, QgsUnitTypes

        for rng in seg_grad.renderer().ranges():
            assert _core(rng.symbol()).widthUnit() == QgsUnitTypes.RenderMillimeters, (
                "the inflow overlay ranks where water arrives — a ranking is drawn in "
                "millimetres, not as a ground width"
            )

        segs = QgsProject.instance().mapLayer(h.state.segment_layer_id)
        core = _core(segs.renderer().symbol())
        assert core.widthUnit() == QgsUnitTypes.RenderMetersInMapUnits, (
            "the recommended segment is drawn in millimetres again, so its apparent "
            "width changes with zoom and says nothing about the swale's footprint"
        )
        prop = core.dataDefinedProperties().property(
            QgsSymbolLayer.PropertyStrokeWidth)
        assert prop.isActive() and "width_m" in prop.expressionString(), (
            f"the segment core is not driven by width_m: {prop.expressionString()!r}"
        )


def check_inflow_scale_modes_all_band(dem_path):
    """Every scale option produces a usable banding, including on real terrain."""
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        combo = h.panel._inflow_scale_combo
        assert combo.itemText(0) == "Natural", (
            "Natural should lead the scale list so it is the default the list and "
            f"the map agree on, got {combo.itemText(0)!r}"
        )
        h.panel.show_inflow_bands_requested.emit(True)
        for i in range(combo.count()):
            combo.setCurrentIndex(i)
            h.assert_no_errors(f"inflow scale {combo.itemText(i)}")
            grad = QgsProject.instance().mapLayer(h.state.inflow_bands_layer_id)
            assert grad is not None, f"{combo.itemText(i)}: no gradient layer"
            assert grad.featureCount() > 0, f"{combo.itemText(i)}: empty gradient"
            for rng in getattr(grad.renderer(), "ranges", list)():
                assert rng.upperValue() >= rng.lowerValue(), (
                    f"{combo.itemText(i)}: inverted band "
                    f"{rng.lowerValue()}–{rng.upperValue()}"
                )


def check_contour_row_selection_reaches_the_map(dem_path):
    """Clicking a row selects that contour on the map, and the reverse."""
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        layer = QgsProject.instance().mapLayer(h.state.contour_layer_id)
        assert layer is not None, "no contour layer"

        h.panel._contour_list.item(1).setSelected(True)
        h.assert_no_errors("select a contour row")
        selected = layer.selectedFeatures()
        assert len(selected) == 1, (
            f"selecting one row selected {len(selected)} map features"
        )
        assert int(selected[0]["cid"]) == 1, "the wrong contour was selected"

        # Map → table: pick a different feature the way the Select Features tool
        # would, and the matching row should highlight.
        target = next(f for f in layer.getFeatures() if int(f["cid"]) == 3)
        layer.selectByIds([target.id()])
        rows = [item.data(Qt.UserRole)
                for item in h.panel._contour_list.selectedItems()]
        assert rows == [3], f"map selection highlighted rows {rows}, expected [3]"


def check_segment_inflow_gradient(dem_path):
    """The peak-inflow overlay draws inside the segments, keeping the outline."""
    from qgis.core import QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.panel.find_segments_requested.emit()
        h.assert_no_errors("segment analysis")
        assert h.state.segment_features, "no segments to grade"

        h.panel._segment_gradient_check.setChecked(True)
        h.assert_no_errors("segment inflow gradient on")

        grad = QgsProject.instance().mapLayer(h.state.segment_gradient_layer_id)
        assert grad is not None, "no segment gradient layer created"
        assert grad.featureCount() > 0, "segment gradient layer is empty"

        # The green/amber outline stays — the gradient goes inside it, not over it.
        segs = QgsProject.instance().mapLayer(h.state.segment_layer_id)
        assert segs is not None, "the recommended segments layer was replaced"
        assert segs.featureCount() == len(h.state.segment_features)

        h.panel._segment_gradient_check.setChecked(False)
        h.assert_no_errors("segment inflow gradient off")
        assert h.state.segment_gradient_layer_id is None, (
            "turning the overlay off left its layer behind"
        )


def check_contour_bands_are_value_based(dem_path):
    """Candidate contours are banded by inflow value, not by rank position."""
    from qgis.core import QgsGraduatedSymbolRenderer, QgsProject

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis")

        breaks = h.state.contour_breaks
        assert len(breaks) >= 3, f"expected natural-breaks boundaries, got {breaks}"
        assert breaks == sorted(breaks), f"breaks are not ascending: {breaks}"

        layer = QgsProject.instance().mapLayer(h.state.contour_layer_id)
        renderer = layer.renderer()
        assert isinstance(renderer, QgsGraduatedSymbolRenderer), (
            f"expected a value-graduated renderer, got {type(renderer).__name__}"
        )
        assert renderer.classAttribute() == "inflow_m3", (
            f"contours are banded on {renderer.classAttribute()!r}, not the inflow value"
        )
        assert len(renderer.ranges()) == len(breaks) - 1

        # Every contour has to fall in a band; a feature outside every range would
        # simply not draw.
        for feat in layer.getFeatures():
            value = feat["inflow_m3"]
            assert any(r.lowerValue() <= value <= r.upperValue()
                       for r in renderer.ranges()), (
                f"contour with inflow {value} falls outside every band {breaks}"
            )


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
