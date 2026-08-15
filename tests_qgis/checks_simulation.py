"""Fill-simulation playback — the path `tests/` cannot reach.

`pytest tests/` covers `_run_simulation` itself, which is pure. What it never
touches is what the controller does with the result: writing a frame raster,
swapping the layer that displays it, and advancing through the timeline. That path
had never been driven by any check, which is how playback came to sit on frame 0
without anyone noticing.
"""

import os

from _harness import PluginHarness, line_across_valley


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
