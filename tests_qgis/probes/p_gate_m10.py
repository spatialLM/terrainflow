"""p_gate_m10 — M-10's real denominator: the operation, not `_orient_downhill`.

`p_gate_ui` priced M-10 against `_orient_downhill` and got 83.5%, which is gate
rule 1's own trap: `_orient_downhill` is *made of* the two samples, so that ratio
says nothing. The enclosing operations are the two handlers that call it —

    `_on_geometry_drawn`        draw a feature, wait for the properties dialog
    `_on_vertex_edit_finished`  release a dragged vertex

— and the first of them turns out to sample the DEM far more than twice:
`_overflow_options` calls `_feature_elevation` once per *other* earthwork, so a
35-feature design reads the whole band 34 more times before the dialog appears.
Counting the opens is what shows that (rule 3); the register only named the two.

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_gate_m10.py
"""
import faulthandler
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

faulthandler.enable()

REAL_TFD = (r"F:\Terrain Flow Design\QGIS Working Files"
            r"\Quail_Island 03.09.2026.tfd")


def main():
    import _probe

    _probe.start_qgis()

    from _harness import PluginHarness, make_workers_synchronous

    make_workers_synchronous()

    import rasterio

    from p_perf_gate import gate, real_dem

    dem = real_dem()

    with PluginHarness(dem, load_boundary=False) as h:
        h.plugin._design_file._restore_design(REAL_TFD)
        h.run_baseline()
        controller = h.plugin._earthworks
        ews = h.state.earthwork_manager.get_all()
        print(f"design restored: {len(ews)} features")

        div = [e for e in ews if e.type == "diversion"]
        ew = div[0] if div else ews[0]

        # A cancelled dialog: everything up to `dlg.exec()` is what the user waits
        # on before the dialog appears, and that is the operation M-10 sits in.
        from terrainflow_assessment import earthwork_properties_dialog as epd

        real_exec = epd.EarthworkPropertiesDialog.exec
        epd.EarthworkPropertiesDialog.exec = lambda self: 0

        opens = {"n": 0}
        real_open = rasterio.open

        def counting_open(*a, **kw):
            opens["n"] += 1
            return real_open(*a, **kw)

        try:
            # Warm: first call pays imports the later ones do not.
            controller._on_geometry_drawn("diversion", ew.geometry)

            rasterio.open = counting_open
            opens["n"] = 0
            t0 = time.perf_counter()
            controller._on_geometry_drawn("diversion", ew.geometry)
            drawn = time.perf_counter() - t0
            drawn_opens = opens["n"]

            opens["n"] = 0
            t0 = time.perf_counter()
            controller._on_vertex_edit_finished(0, ews[0].geometry)
            settled = time.perf_counter() - t0
            settled_opens = opens["n"]
        finally:
            rasterio.open = real_open
            epd.EarthworkPropertiesDialog.exec = real_exec

        print("")
        print(f"_on_geometry_drawn (to the dialog)  {drawn * 1000:9.1f} ms, "
              f"{drawn_opens} rasterio.open")
        print(f"_on_vertex_edit_finished            {settled * 1000:9.1f} ms, "
              f"{settled_opens} rasterio.open")

        saved_per_call = _saved_per_call(dem)
        print(f"per full-band read saved            {saved_per_call * 1000:9.1f} ms")

        # How many of those opens are `snap_point_to_contour_elevation`?
        snaps = _count_snaps(controller, ew, ews)
        print(f"snap_point_to_contour_elevation calls per draw: {snaps}")
        gate("M-10 windowed read (draw a feature, wait for the dialog)",
             snaps * saved_per_call, drawn)


def _saved_per_call(dem):
    """Full band read minus windowed read, both including the open."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    from p_perf_gate import bench

    with rasterio.open(dem) as src:
        row, col = src.height // 2, src.width // 2

    def full():
        with rasterio.open(dem) as src:
            return float(src.read(1)[row, col])

    def win():
        with rasterio.open(dem) as src:
            return float(src.read(1, window=Window(col, row, 1, 1))[0, 0])

    del np
    return bench(full, repeat=5) - bench(win, repeat=5)


def _count_snaps(controller, ew, ews):
    """Count the snap calls one `_on_geometry_drawn` makes, without the dialog."""
    from terrainflow_assessment.modules import swale_design

    real = swale_design.snap_point_to_contour_elevation
    n = {"n": 0}

    def counting(*a, **kw):
        n["n"] += 1
        return real(*a, **kw)

    from terrainflow_assessment import earthwork_properties_dialog as epd

    real_exec = epd.EarthworkPropertiesDialog.exec
    epd.EarthworkPropertiesDialog.exec = lambda self: 0
    swale_design.snap_point_to_contour_elevation = counting
    try:
        controller._on_geometry_drawn("diversion", ew.geometry)
    finally:
        swale_design.snap_point_to_contour_elevation = real
        epd.EarthworkPropertiesDialog.exec = real_exec
    return n["n"]


if __name__ == "__main__":
    main()
