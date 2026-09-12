"""p_drag_cost — what a vertex-drag frame actually costs, per Batch 4's gate.

Three Tier B items sit on this one path and the plan prices them against it:

    M-5a / G-7  the Live Assessment rebuild above the `geometry_settled` guard
    Q-12        `flow_domain_mask.sum()` per frame, against the cache added to stop it
    M-9         `rasterio.open` per feature inside `build_stores_from_earthworks`

The gate is "under 50 ms per drag frame and the item is dropped", so the number
that decides them is the frame, not the primitive. This measures the frame, at a
feature count a real design reaches.

Outside `run_all.py`'s `checks_*` glob on purpose, like every other probe: it
takes a design up to a real analysis and then drags, which is slow, and it is
evidence rather than a check.

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_drag_cost.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

FIXTURE_DEM = os.path.join(
    os.path.dirname(os.path.dirname(HERE)),
    "tests", "fixtures", "quail_island_catchment.tif",
)

N_FEATURES = 12
N_FRAMES = 10


def _line(row, x0, y0, half=40.0):
    from qgis.core import QgsGeometry, QgsPointXY

    cx = x0 + 200.0
    y = y0 - row
    return QgsGeometry.fromPolylineXY(
        [QgsPointXY(cx - half, y), QgsPointXY(cx + half, y)])


def main():
    import _probe

    _probe.start_qgis()

    import rasterio
    from _harness import PluginHarness, make_workers_synchronous

    # `run_all.py` does this for every checks module without REAL_THREADS; a probe
    # run standalone has no runner to do it, so `worker.start()` really threads and
    # `run_baseline()` returns before the result exists.
    make_workers_synchronous()

    with rasterio.open(FIXTURE_DEM) as src:
        x0, y0 = src.transform.c, src.transform.f

    # The fixture needs its own boundary: the harness's is shaped around the
    # synthetic DEM, and without one the baseline finds no perimeter and returns
    # nothing — which would leave `flow_domain_mask` None and quietly take Q-12's
    # whole code path out of the measurement.
    from qgis.core import QgsFeature, QgsGeometry, QgsProject, QgsVectorLayer

    X0, Y0, SIZE, INSET = 1574561.0, 5169232.0, 400.0, 20.0

    with PluginHarness(FIXTURE_DEM, load_boundary=False) as h:
        boundary = QgsVectorLayer("Polygon?crs=EPSG:2193", "Fixture boundary", "memory")
        feat = QgsFeature()
        feat.setGeometry(QgsGeometry.fromWkt(
            "POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))".format(
                x0=X0 + INSET, y0=Y0 + INSET,
                x1=X0 + SIZE - INSET, y1=Y0 + SIZE - INSET)))
        boundary.dataProvider().addFeatures([feat])
        boundary.updateExtents()
        QgsProject.instance().addMapLayer(boundary)
        h.panel.boundary_changed.emit(boundary)

        t0 = time.perf_counter()
        result = h.run_baseline()
        print("baseline                  {:8.1f} ms  (result: {})".format(
            (time.perf_counter() - t0) * 1000, result is not None))
        print("domain mask present:      {}".format(
            h.state.flow_domain_mask is not None))

        for i in range(N_FEATURES):
            h.add_earthwork("swale", geometry=_line(30 + i * 25, x0, y0))
        h.panel.analysis_inputs_changed.emit()

        controller = h.plugin._earthworks

        counters = {"rio_open": 0}
        real_open = rasterio.open

        def counting_open(*a, **kw):
            counters["rio_open"] += 1
            return real_open(*a, **kw)

        rasterio.open = counting_open
        try:
            # One frame first, so import and JIT costs stay out of the mean.
            controller._recompute_live_assessment(geometry_settled=False)
            counters["rio_open"] = 0

            t0 = time.perf_counter()
            for _ in range(N_FRAMES):
                controller._recompute_live_assessment(geometry_settled=False)
            drag = (time.perf_counter() - t0) / N_FRAMES
        finally:
            rasterio.open = real_open

        t0 = time.perf_counter()
        controller._recompute_live_assessment(geometry_settled=True)
        settled = time.perf_counter() - t0

        print("")
        print("features                  {:8d}".format(N_FEATURES))
        print("drag frame (mean)         {:8.1f} ms   <- the 50 ms gate".format(
            drag * 1000))
        print("settled edit              {:8.1f} ms".format(settled * 1000))
        print("rasterio.open() per frame {:8.2f}".format(
            counters["rio_open"] / N_FRAMES))
        print("throttle is 12.5 Hz, so a frame has 80 ms of budget")


if __name__ == "__main__":
    main()
