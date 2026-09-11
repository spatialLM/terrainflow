"""Fill-simulation playback — the path `tests/` cannot reach.

`pytest tests/` covers `_run_simulation` itself, which is pure. What it never
touches is what the controller does with the result: writing a frame raster,
swapping the layer that displays it, and advancing through the timeline. That path
had never been driven by any check, which is how playback came to sit on frame 0
without anyone noticing.
"""

import os

from _harness import (
    PluginHarness,
    line_across_valley,
)


def _simulated(h):
    """A baseline, one earthwork, and a completed simulation."""
    h.run_baseline()
    ew = h.add_earthwork("swale", line_across_valley())
    ew.capacity_m3 = 50.0
    h.panel.run_earthworks_requested.emit()
    h.panel.run_simulation_requested.emit()
    h.assert_no_errors("simulation")
    return h.state.sim_result


def check_simulation_produces_frames(dem_path):
    with PluginHarness(dem_path) as h:
        result = _simulated(h)
        assert result, "the simulation produced no result"
        frames = result.get("frames") or []
        assert len(frames) >= 2, f"only {len(frames)} frame(s) to play back"


def check_playback_advances_past_frame_zero(dem_path):
    """The regression: the frame raster is rewritten under the layer holding it open.

    On Windows GDAL keeps that file locked, so the write raised, a bare `except:
    return` swallowed it, and every frame after the first silently did nothing —
    indistinguishable from a slow simulation.
    """
    with PluginHarness(dem_path) as h:
        result = _simulated(h)
        frames = result.get("frames") or []
        assert len(frames) >= 3, "need a few frames to advance through"

        sim = h.plugin._simulation
        seen = []
        for idx in range(min(4, len(frames))):
            sim.show_sim_frame(idx)
            h.assert_no_errors(f"frame {idx}")
            seen.append(h.state.sim_frame_layer_id)

        assert all(seen), f"a frame produced no layer: {seen}"
        assert len(set(seen)) > 1, (
            "every frame reused one layer id — the display never actually changed")


def check_the_ponding_frame_alternates_files(dem_path):
    """Two filenames, so a write never lands on the file the live layer has open."""
    with PluginHarness(dem_path) as h:
        _simulated(h)
        sim = h.plugin._simulation
        if h.state.sim_ponding_capacity is None:
            return          # no pond on this design; nothing to alternate

        paths = []
        for idx in range(3):
            sim.show_sim_frame(idx)
            layer_id = h.state.sim_ponding_frame_layer_id
            if layer_id:
                from qgis.core import QgsProject
                lyr = QgsProject.instance().mapLayer(layer_id)
                if lyr is not None:
                    paths.append(lyr.source())

        h.assert_no_errors("ponding frame playback")
        if len(paths) >= 2:
            assert paths[0] != paths[1], (
                "consecutive ponding frames wrote to the same path — the live layer "
                "still has it open")
            for p in paths:
                assert os.path.exists(p)


def check_a_failed_frame_write_is_reported(dem_path):
    """A frozen playback with no message reads as a slow simulation."""
    with PluginHarness(dem_path) as h:
        _simulated(h)
        sim = h.plugin._simulation
        if h.state.sim_ponding_capacity is None:
            return

        # Point the scratch directory at something that cannot be written into.
        original = h.state.output_dir
        h.state.output_dir = os.path.join(original, "does", "not", "exist")
        try:
            sim.show_sim_frame(1)
        finally:
            h.state.output_dir = original

        assert any("ponding frame" in text.lower() or "playback" in text.lower()
                   for _, _, text in h.bar.messages), (
            f"the failed write was silent: {h.bar.messages}")


def check_the_fill_layer_labels_features_by_name(dem_path):
    """Keyed by id internally; the map still shows the name the user gave it."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        ew = h.add_earthwork("swale", line_across_valley(), name="North Swale")
        ew.capacity_m3 = 50.0
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        centroids = h.state.sim_ew_centroids
        assert centroids, "no centroids were captured"
        assert ew.id in centroids, "centroids are not keyed by feature id"
        _, label = centroids[ew.id]
        assert label == "North Swale", f"the display label was lost: {label!r}"


def check_two_features_sharing_a_name_keep_separate_fill_state(dem_path):
    """Under name keying one store's fill silently overwrote the other's, every frame."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        a = h.add_earthwork("swale", line_across_valley(row=60), name="Swale 1")
        b = h.add_earthwork("swale", line_across_valley(row=120), name="Swale 1")
        a.capacity_m3, b.capacity_m3 = 40.0, 90.0
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation with a duplicated name")

        frames = (h.state.sim_result or {}).get("frames") or []
        assert frames, "no frames"
        fills = frames[-1]["fills"]
        assert a.id in fills and b.id in fills, (
            f"both features must have their own fill state, got {list(fills)}")
        assert all(fd.get("name") == "Swale 1" for fd in fills.values()), (
            "the display name should still travel with the value")

        # The same defect, thirty lines above `frame_fills` in the same file:
        # `timestep_table` was keyed by `store.name`, so the second Swale 1
        # overwrote the first's column in every row. The report's fill-timeline
        # chart and the panel's simulation table both read it, so one feature got
        # no series at all and the other's was drawn from the wrong data.
        rows = (h.state.sim_result or {}).get("timestep_table") or []
        assert rows, "no timestep table"
        keys = [k for k in rows[0] if k.endswith("_fill_pct")]
        assert len(keys) == 2, (
            f"two features must have two fill columns, got {keys}")
        assert f"{a.id}_fill_pct" in rows[0] and f"{b.id}_fill_pct" in rows[0], (
            f"the columns are not keyed by feature id: {keys}")

        summary = (h.state.sim_result or {}).get("earthwork_summary") or []
        assert [s.get("id") for s in summary] == [a.id, b.id], (
            "the summary carries no ids, so no reader can join to those columns")


# ---------------------------------------------------------------------------
# The simulation and the design tier answer with one network
# ---------------------------------------------------------------------------

def _big(ew, capacity=1_000_000.0):
    """Make a feature large enough that it cannot overflow during the event."""
    ew.capacity_m3 = capacity
    ew.terrain_capacity_m3 = capacity
    return ew


def check_feature_inflow_follows_its_own_catchment(dem_path):
    """Each feature gets the runoff from the cells it is FIRST to intercept.

    The model this replaced sampled the *cumulative* accumulation raster at the
    feature's centroid, so a downstream feature was credited with every upstream
    catchment as well, and the surplus was unpicked by an elevation ranking that had
    nothing to do with where water goes. With mutually exclusive catchments, inflow is
    simply proportional to catchment size — which is the assertion below.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        a = _big(h.add_earthwork("swale", line_across_valley(row=60), name="Upper"))
        b = _big(h.add_earthwork("swale", line_across_valley(row=120), name="Lower"))
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        counts = h.state.catchment_counts or {}
        assert counts.get(a.id) and counts.get(b.id), (
            f"both features need a catchment to compare: {counts}")

        summary = {row["id"]: row for row in h.state.sim_result["earthwork_summary"]}
        inflow_a = summary[a.id]["total_inflow_m3"]
        inflow_b = summary[b.id]["total_inflow_m3"]
        assert inflow_a > 0 and inflow_b > 0, f"no inflow at all: {inflow_a}, {inflow_b}"

        # Neither can overflow, so total inflow is the direct catchment share alone.
        assert not summary[a.id]["overflowed"] and not summary[b.id]["overflowed"], (
            "a feature overflowed — the ratio below would include cascaded water")
        want = counts[a.id] / counts[b.id]
        got = inflow_a / inflow_b
        assert abs(got - want) / want < 0.02, (
            f"inflow ratio {got:.3f} does not match the catchment ratio {want:.3f} — "
            "runoff is not being split by direct catchment")


def check_the_simulation_holds_no_more_water_than_fell(dem_path):
    """The double-count was visible as capture exceeding the storm itself."""
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _big(h.add_earthwork("swale", line_across_valley(row=60), name="Upper"))
        _big(h.add_earthwork("swale", line_across_valley(row=120), name="Lower"))
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        table = h.state.sim_result["timestep_table"]
        site_runoff = sum(row["runoff_m3"] for row in table)
        summary = h.state.sim_result["earthwork_summary"]
        held = sum(row["stored_m3"] + row["total_infiltration_m3"] for row in summary)
        assert held <= site_runoff * 1.01, (
            f"features hold {held:,.0f} m3 of a {site_runoff:,.0f} m3 storm")


def check_the_simulation_cascades_along_the_design_network(dem_path):
    """Overflow follows the links the design tier resolved and the report draws.

    Routed separately, the two tiers produced two different networks — the elevation
    heuristic links features across a ridge that no water crosses — and the
    comparative report presented them as one.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        upper = h.add_earthwork("swale", line_across_valley(row=60), name="Upper")
        lower = _big(h.add_earthwork("swale", line_across_valley(row=120),
                                     name="Lower"))
        upper.capacity_m3 = 1.0          # certain to overflow
        upper.terrain_capacity_m3 = 1.0
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        routing = h.state.balance_routing
        assert routing is not None, "the design tier's routing was not retained"

        summary = {row["id"]: row for row in h.state.sim_result["earthwork_summary"]}
        assert summary[upper.id]["overflowed"], "the upstream feature never filled"

        target = routing.edges.get(upper.id)
        counts = h.state.catchment_counts or {}
        if target == lower.id:
            # Its total inflow must exceed its own catchment's share by the overflow.
            share_ratio = counts.get(lower.id, 0) / max(counts.get(upper.id, 1), 1)
            assert summary[lower.id]["total_inflow_m3"] > (
                summary[upper.id]["total_inflow_m3"] * share_ratio), (
                "the routed overflow never arrived downstream")
        else:
            # Routing says it leaves the site; nothing may have been handed downstream.
            assert summary[lower.id]["total_inflow_m3"] <= (
                counts.get(lower.id, 0) * h.state.flow_grid_meta["cell_area_m2"]
                * 1.0), (
                "water arrived at a feature the flow paths do not connect to")


def check_an_unassessed_design_is_assessed_rather_than_refused(dem_path):
    """A design loaded from file and never edited has had no live assessment.

    Refusing there would break a working path, so the simulation asks the design tier
    for the labelling it needs instead of guessing or giving up.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _big(h.add_earthwork("swale", line_across_valley()))
        h.panel.run_earthworks_requested.emit()

        h.state.catchment_labels = None
        h.state.balance_routing = None
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation after an unassessed design")

        assert h.state.catchment_labels is not None, "the labelling was not rebuilt"
        assert h.state.sim_result, "the simulation did not run"


def check_simulating_with_no_labelling_available_is_refused(dem_path):
    """The last resort, when the design tier cannot supply one either.

    Guessing the split is what the previous model did — cumulative accumulation at
    each centroid, unpicked by elevation — and it double-counted every upstream
    catchment. A refusal the user can act on beats a number they cannot check.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        _big(h.add_earthwork("swale", line_across_valley()))
        h.panel.run_earthworks_requested.emit()

        h.plugin._simulation.design_tier = None
        h.state.catchment_labels = None
        h.bar.messages.clear()
        h.panel.run_simulation_requested.emit()

        assert h.state.sim_worker is None or not h.state.sim_worker.isRunning()
        assert any("catchment" in text.lower() or "design analysis" in text.lower()
                   for _, _, text in h.bar.messages), (
            f"the refusal was silent: {h.bar.messages}")


def check_the_frame_shoreline_agrees_with_the_verification_layer(dem_path):
    """Both drawings of one part-full pond sit in the same Verify group.

    The frame layer scaled the full pond's depth by the fill fraction — the exact
    method `event_pond_depth`'s docstring names as wrong, because it keeps the full
    pond's footprint and paints water up banks the water never reaches. So the two
    layers disagreed about where the shoreline was. The frame now solves for the
    level that holds the delivered volume, as the verification layer does.
    """
    import numpy as np

    with PluginHarness(dem_path) as h:
        h.run_baseline()
        # A dam, not a basin: a burned basin has a flat floor, and on a flat floor
        # half the volume still covers the whole footprint at half the depth — the
        # two methods agree there and the check would prove nothing. A dam impounds
        # the valley's own bathymetry, which is where they part company.
        geom = line_across_valley(row=90)
        h.add_earthwork("dam", geometry=geom)
        # add_earthwork bypasses the dialog, which is what sizes a feature; settling
        # the geometry is the controller's own exact tier and fills both capacities.
        h.plugin._earthworks._on_vertex_edit_finished(0, geom)
        h.panel.run_earthworks_requested.emit()
        h.panel.run_simulation_requested.emit()
        h.assert_no_errors("simulation")

        sim = h.plugin._simulation
        capacity = h.state.sim_ponding_capacity
        ground = h.state.sim_ponding_ground
        assert capacity is not None and ground is not None, (
            "the dam impounded nothing — this check needs a pool")
        masks = h.state.sim_ponding_masks
        key = next((k for k, m in masks.items() if (m & (capacity > 0.001)).any()),
                   None)
        assert key is not None, "no feature has a pond to draw"

        pool = masks[key] & (capacity > 0.001)
        bed = ground[pool]
        relief = float(np.nanmax(bed) - np.nanmin(bed))
        assert relief > 0.1, (
            f"the pool bed is flat ({relief:.3f} m). The two methods agree on a flat "
            "floor, so this fixture would prove nothing")

        half = {key: {"fill_pct": 50.0, "overflowed": False}}
        depth = sim._frame_depth(capacity, masks, half)

        wet_full = int(pool.sum())
        wet_half = int((depth > 0.001)[pool].sum())
        assert wet_half > 0, "a half-full pond drew no water at all"
        assert wet_half < wet_full, (
            f"a half-full pond covers {wet_half} of {wet_full} cells — it still has "
            "the full pond's footprint, which is the fraction-scaling it replaced")

        # And it stands in the bottom of the basin, not up the banks.
        assert float(np.max(depth - capacity)) <= 1e-6, (
            "the event pond is deeper than the pond that contains it")
        wet = depth > 0.001
        assert float(np.nanmax(ground[wet])) < float(np.nanmax(bed)) + 1e-6, (
            "the water reaches ground the full pond's own shoreline does not")


def check_the_playback_frame_is_cheap_enough_to_play(dem_path):
    """It runs twice a second. The pool grouping depends on geometry alone, so it
    is computed once per simulation rather than per frame."""
    import time

    with PluginHarness(dem_path) as h:
        _simulated(h)
        sim = h.plugin._simulation
        if h.state.sim_ponding_capacity is None:
            return

        sim.show_sim_frame(0)           # warm: builds the grouping
        start = time.perf_counter()
        for idx in range(1, 4):
            sim.show_sim_frame(idx)
        per_frame = (time.perf_counter() - start) / 3.0
        h.assert_no_errors("timed playback")
        assert per_frame < 0.5, f"{per_frame * 1000:.0f} ms per frame is not playback"


def check_the_ponding_frame_does_not_rescale_itself_each_frame(dem_path):
    """`_apply_ponding_ramp` called `apply_raster_ramp` directly — the rule
    CLAUDE.md states as "never" — and with no `max_value`, so every frame
    stretched its stops over its own band maximum. The same depth therefore
    changed colour as the pools filled, in the same Verify group as layers
    already on the shared `ponding` family. The method's own docstring claimed
    the mismatch had been fixed.

    Asserted on the renderer and against `state.ramp_scales`, so it fails both
    if the frame rescales and if it rescales *consistently* on a scale of its own.
    """
    from qgis.core import QgsProject

    def top_stop(layer):
        shader = layer.renderer().shader().rasterShaderFunction()
        items = shader.colorRampItemList()
        return items[-1].value if items else None

    with PluginHarness(dem_path) as h:
        _simulated(h)
        sim = h.plugin._simulation
        if h.state.sim_ponding_capacity is None:
            return          # no pond on this design; nothing to scale

        tops = []
        for idx in (0, len((h.state.sim_result or {}).get("frames") or []) - 1):
            sim.show_sim_frame(max(idx, 0))
            layer_id = h.state.sim_ponding_frame_layer_id
            if not layer_id:
                continue
            layer = QgsProject.instance().mapLayer(layer_id)
            if layer is not None:
                tops.append(top_stop(layer))
        h.assert_no_errors("ponding frame playback")

        if len(tops) < 2:
            return
        assert tops[0] == tops[1], (
            f"the frame ramp restretched between an early and a late frame "
            f"({tops[0]} then {tops[1]}) — the same depth changes colour as the "
            f"pools fill")
        family = (h.state.ramp_scales or {}).get("ponding") or {}
        assert family.get("top"), (
            "the frame is not on the shared `ponding` family at all, so it is "
            "drawn against a scale no other captured-water layer shares")
        assert tops[0] == family["top"], (
            f"the frame's top stop is {tops[0]} against the family's "
            f"{family['top']}")
