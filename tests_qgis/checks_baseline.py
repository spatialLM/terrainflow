"""Baseline analysis end to end: AnalysisWorker → result layers → styling."""

import os

from _harness import PluginHarness


def check_baseline_runs(dem_path):
    """run_baseline_requested produces a real result dict and loads real layers."""
    with PluginHarness(dem_path) as h:
        result = h.run_baseline()
        h.assert_no_errors("baseline run")

        assert result is not None, "baseline produced no result"
        acc = result.get("flow_accumulation")
        assert acc and os.path.exists(acc), f"flow_accumulation missing on disk: {acc!r}"
        assert result.get("runoff_mm", 0) > 0, (
            f"runoff_mm was {result.get('runoff_mm')!r} — expected positive runoff"
        )
        assert result.get("catchment_area_m2", 0) > 0, "catchment area came back zero"
        assert h.state.baseline_layer_ids, "no baseline layers registered on state"
        assert h.state.baseline_report is not None, "BaselineReport not built"


def check_baseline_drains_downhill(dem_path):
    """The synthetic valley drains south — exit points must be found on that edge."""
    with PluginHarness(dem_path) as h:
        result = h.run_baseline()
        h.assert_no_errors("baseline run")

        exits = result.get("exit_points", [])
        assert exits, "no exit points detected on a DEM with a clear outlet"
        volumes = [ep.get("volume_m3", 0) for ep in exits]
        assert max(volumes) > 0, f"all exit points carry zero volume: {volumes}"


def check_result_layers_render(dem_path):
    """Styling code paths run against real layers — renderers, ramps, labels."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        from qgis.core import QgsProject

        for lid in h.state.baseline_layer_ids:
            layer = QgsProject.instance().mapLayer(lid)
            assert layer is not None, f"baseline layer {lid} is not in the project"
            assert layer.isValid(), f"baseline layer {layer.name()!r} is invalid"
            assert layer.renderer() is not None, (
                f"baseline layer {layer.name()!r} has no renderer"
            )


def check_before_after_toggle(dem_path):
    """Toggling before/after visibility must not touch a stale layer reference."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.before_after_toggled.emit(True)
        h.panel.before_after_toggled.emit(False)
        h.assert_no_errors("before/after toggle")


def check_overlay_toggles(dem_path):
    """Slope / throughflow / catchment overlays each build their own layer + ramp.

    Warnings are acceptable (an overlay may legitimately need more state); an
    exception reaching the message bar as a critical is not.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.assert_no_errors("baseline run")

        h.panel.toggle_slope_class_requested.emit(True)
        h.assert_no_errors("slope class on")

        h.panel.toggle_slope_vectors_requested.emit(True)
        h.assert_no_errors("slope vectors on")

        h.panel.toggle_throughflow_requested.emit(True)
        h.panel.throughflow_scale_changed.emit("linear")
        h.panel.throughflow_scale_changed.emit("log")
        h.assert_no_errors("throughflow")

        # ...and back off again, which is where dead-reference bugs surface.
        h.panel.toggle_slope_class_requested.emit(False)
        h.panel.toggle_slope_vectors_requested.emit(False)
        h.panel.toggle_throughflow_requested.emit(False)
        h.assert_no_errors("overlays off")


def check_changing_the_storm_marks_the_baseline_stale(dem_path):
    """Anything in `run_tag` invalidates the run that was tagged with it.

    Only the method combo used to mark stale. Change the rainfall depth or the
    duration and Baseline stayed green over rasters routed for a different storm —
    while the legend group and the report still carried the old tag, which is the
    value the report re-derives at export time.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        assert h.panel._stepper.state("baseline") == "done"

        h.panel._rainfall_spin.setValue(h.panel._rainfall_spin.value() + 25)

        assert h.panel._stepper.state("baseline") == "stale", (
            "a different storm depth left the baseline reading as current")
        assert h.panel._stepper.state("analysis") == "stale"


def check_every_run_tag_input_marks_it_stale(dem_path):
    """Each input `param_tag` reads, one at a time — none may be missed."""
    with PluginHarness(dem_path) as h:
        moves = [
            ("rainfall", lambda p: p._rainfall_spin.setValue(p._rainfall_spin.value() + 5)),
            ("duration", lambda p: p._duration_spin.setValue(p._duration_spin.value() + 2)),
            ("threshold", lambda p: p._threshold_spin.setValue(
                p._threshold_spin.value() + 1)),
            ("cn", lambda p: p._cn_spin.setValue(p._cn_spin.value() + 3)),
        ]
        for label, move in moves:
            h.run_baseline()
            assert h.panel._stepper.state("baseline") == "done", (
                f"{label}: fixture did not get back to a clean baseline")
            move(h.panel)
            assert h.panel._stepper.state("baseline") == "stale", (
                f"changing {label} left the baseline reading as current")


def check_a_rerun_clears_the_previous_verification(dem_path):
    """A verification is measured against a baseline, so a new one retires it.

    The scorecard went on showing "Verified · Δ" from the previous burn after the
    baseline underneath it had been replaced.
    """
    from _harness import line_across_valley

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", line_across_valley())
        ew.capacity_m3 = 50.0
        h.panel.run_earthworks_requested.emit()
        assert h.state.verification is not None, "no verification to invalidate"

        h.panel._rainfall_spin.setValue(h.panel._rainfall_spin.value() + 25)
        h.run_baseline()

        assert h.state.verification is None, (
            "the previous burn's verification survived a new baseline")
        assert h.state.verified_delta_pct is None
        assert h.panel._stepper.state("verify") != "done", (
            "Verify still reads green against a baseline it never measured")
