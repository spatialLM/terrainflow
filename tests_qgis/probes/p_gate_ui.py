"""p_gate_ui — price the remaining Batch 4 items against their enclosing operations.

Five items, each priced on the real design (35 features, 1139x1016) rather than on
the committed 400x400 clip, per gate rule 2:

    M-11  `find_ridgelines` does a full-raster compare per skeleton component.
          Operation: the Keypoint analysis run (`contour.py:1233-1245`'s `work()`).
    Q-11  `_calc_dam_wall_metrics` reads the whole band per spin-box tick.
          Operation: `_update_capacity()`, the dialog's whole response to a tick.
    G-9   the draw tools read the whole slope band in `__init__`.
          Operation: `activate_draw_line()`, arming the tool after a button press.
    G-8   the ponding query polygonises the entire background per click.
          Operation: `PondingQueryTool.canvasPressEvent`'s body.
    M-10  `snap_point_to_contour_elevation` reads the whole band for one cell.
          Operation: the spillway-placement click, and `_orient_downhill`.

Rule 3 throughout: the call **count** is printed beside the clock, because a count
is what says whether the mechanism is the one the register described.

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_gate_ui.py
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

    from p_perf_gate import real_dem

    dem = real_dem()

    with PluginHarness(dem, load_boundary=False) as h:
        t0 = time.perf_counter()
        h.plugin._design_file._restore_design(REAL_TFD)
        print(f"restore design          {(time.perf_counter() - t0) * 1000:9.1f} ms")
        t0 = time.perf_counter()
        h.run_baseline()
        print(f"baseline                {(time.perf_counter() - t0) * 1000:9.1f} ms")

        for fn in (_m11, _q11, _g9, _g8, _m10):
            print("")
            try:
                fn(h)
            except Exception as exc:
                import traceback
                print(f"  !! {fn.__name__} failed: {exc}")
                traceback.print_exc()


# --------------------------------------------------------------------------- M-11

def _m11(h):
    """`find_ridgelines` against the Keypoint analysis run it sits in."""
    import numpy as np

    from p_perf_gate import gate
    from terrainflow_assessment.modules.keypoint_analysis import DrainageLineAnalysis

    result = h.state.baseline_result or {}
    acc = result.get("flow_accumulation")
    pond = result.get("pond_flow")
    if not acc:
        print("M-11: no flow accumulation from the baseline — cannot price it")
        return

    contour = h.plugin._contour
    dem_path = h.state.dem_path
    mask = contour._get_keypoint_boundary_mask(dem_path)
    cell_m2 = h.state.dem_info.cell_area_m2 if h.state.dem_info else 100.0
    min_acc = max(50, int(1.0 * 10_000 / cell_m2))

    t0 = time.perf_counter()
    ka = DrainageLineAnalysis(dem_path, acc, pond)
    t_ctor = time.perf_counter() - t0

    t0 = time.perf_counter()
    ka.find_keypoints(min_acc_cells=min_acc, n_keypoints=h.panel.keypoint_count,
                      boundary_mask=mask)
    t_keypoints = time.perf_counter() - t0

    t0 = time.perf_counter()
    lines = ka.find_ridgelines(boundary_mask=mask)
    t_ridges = time.perf_counter() - t0

    operation = t_ctor + t_keypoints + t_ridges
    print("M-11 find_ridgelines")
    print(f"  DrainageLineAnalysis()  {t_ctor * 1000:9.1f} ms")
    print(f"  find_keypoints          {t_keypoints * 1000:9.1f} ms")
    print(f"  find_ridgelines         {t_ridges * 1000:9.1f} ms -> {len(lines)} lines")

    # Rule 3: the mechanism. How many components does the loop walk, and how many
    # of them does `min_cells` throw away after paying for a full-raster compare?
    counts = _ridgeline_components(ka, mask)
    if counts:
        print(f"  skeleton components     {counts['n']}, of which "
              f"{counts['small']} are under min_cells ({counts['min_cells']})")
        print(f"  per-component full-raster compares: {counts['n']} "
              f"({counts['small']} of them wasted)")
        print(f"  loop cost (compare+argwhere for every component) "
              f"{counts['loop'] * 1000:9.1f} ms")
        print(f"  bincount+find_objects equivalent               "
              f"{counts['fast'] * 1000:9.1f} ms")
        gate("M-11 skip small components before any raster work",
             counts["loop"] - counts["fast"], operation)
    else:
        gate("M-11 ceiling: all of find_ridgelines removed", t_ridges, operation)
    del np


def _ridgeline_components(ka, boundary_mask):
    """Re-derive the skeleton the loop walks, and time the loop both ways.

    Reproduces `find_ridgelines`' own preamble so the component count is the real
    one — an estimate here would be exactly the kind of guess rule 3 exists to
    replace.
    """
    import numpy as np
    from scipy.ndimage import find_objects
    from scipy.ndimage import label as nd_label

    from terrainflow_assessment.modules.keypoint_analysis import _thin_to_centreline
    from terrainflow_assessment.modules.terrain_indices import landform_classes, landform_tpi

    try:
        valid = np.isfinite(ka.dem)
        tpi = landform_tpi(ka.dem, ka.cell_w, ka.cell_h, window_m=15.0)
        with np.errstate(invalid="ignore"):
            above = landform_classes(tpi, sd=1.0, mask=boundary_mask) == 1
            ridge_raw = above & (ka.acc <= 2) & valid
        ridge_raw[[0, -1], :] = False
        ridge_raw[:, [0, -1]] = False
        if boundary_mask is not None:
            ridge_raw &= boundary_mask
        if not ridge_raw.any():
            return None
        skeleton = _thin_to_centreline(ridge_raw)
        if not skeleton.any():
            skeleton = ridge_raw
        labeled, n = nd_label(skeleton)
        min_cells = max(3, int(50.0 / ka.cell_size))
    except Exception as exc:
        print(f"  (could not rebuild the skeleton: {exc})")
        return None

    t0 = time.perf_counter()
    kept = 0
    for rid in range(1, n + 1):
        rc = np.argwhere(labeled == rid)
        if len(rc) >= min_cells:
            kept += 1
    loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    slices = find_objects(labeled)
    kept2 = 0
    for rid in range(1, n + 1):
        if sizes[rid] < min_cells:
            continue
        sl = slices[rid - 1]
        rc = np.argwhere(labeled[sl] == rid)
        rc += np.array([[sl[0].start, sl[1].start]])
        if len(rc) >= min_cells:
            kept2 += 1
    fast = time.perf_counter() - t0

    if kept != kept2:
        print(f"  !! the two loops disagree on kept components: {kept} vs {kept2}")
    return {"n": n, "small": int((sizes[1:] < min_cells).sum()) if n else 0,
            "min_cells": min_cells, "loop": loop, "fast": fast}


# --------------------------------------------------------------------------- Q-11

def _q11(h):
    """`_calc_dam_wall_metrics` against `_update_capacity`, the tick response."""
    from p_perf_gate import gate
    from terrainflow_assessment.earthwork_properties_dialog import (
        EarthworkPropertiesDialog,
    )

    dams = [e for e in h.state.earthwork_manager.get_all() if e.type == "dam"]
    if not dams:
        print("Q-11: the design has no dam — cannot price it")
        return
    ew = dams[0]
    dlg = EarthworkPropertiesDialog(
        ew_type="dam", geometry=ew.geometry, parent=h.main_window,
        earthwork=ew, dem_path=h.state.dem_path,
        cell_size_m=abs(h.state.burner.transform.a) if h.state.burner else 1.0,
    )
    try:
        crest = dlg.spin_crest_elev.value()
        width = dlg.spin_width.value()

        # One call first so the import and the GDAL open are warm, exactly as they
        # are by the time a user is holding the arrow key down.
        dlg._calc_dam_wall_metrics(crest, width)

        n = 10
        t0 = time.perf_counter()
        for i in range(n):
            dlg._calc_dam_wall_metrics(crest + i * 0.01, width)
        per_call = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for i in range(n):
            dlg._update_capacity()
        per_tick = (time.perf_counter() - t0) / n

        print(f"Q-11 dam dialog, {ew.name}")
        print(f"  _calc_dam_wall_metrics  {per_call * 1000:9.1f} ms per call")
        print(f"  _update_capacity        {per_tick * 1000:9.1f} ms per tick "
              f"<- the operation")
        saved = _band_read_cost(h.state.dem_path)
        print(f"  of which the band read  {saved * 1000:9.1f} ms")
        gate("Q-11 cache the band on the dialog", saved, per_tick)
        gate("Q-11 cache the band on the dialog (per-tick bar)", saved, per_tick,
             per_frame=True)
    finally:
        dlg.deleteLater()


def _band_read_cost(dem_path):
    """What `src.read(1).astype("float32")` costs beyond opening the file."""
    import rasterio

    from p_perf_gate import bench

    def full():
        with rasterio.open(dem_path) as src:
            return src.read(1).astype("float32")

    def open_only():
        with rasterio.open(dem_path) as src:
            return src.transform

    return bench(full, repeat=5) - bench(open_only, repeat=5)


# ---------------------------------------------------------------------------- G-9

def _g9(h):
    """`DrawLineTool.__init__`'s slope read against `activate_draw_line`."""
    from p_perf_gate import gate

    controller = h.plugin._earthworks
    slope = getattr(h.state, "slope_raster_path", None)
    print(f"G-9 draw tools; slope raster: {slope}")
    if not slope or not os.path.exists(slope):
        print("  no slope raster on state — the read never happens; nothing to price")
        return

    n = 5
    controller.activate_draw_line("swale")      # warm
    t0 = time.perf_counter()
    for _ in range(n):
        controller.activate_draw_line("swale")
    per_activation = (time.perf_counter() - t0) / n

    saved = _band_read_cost(slope)
    print(f"  activate_draw_line      {per_activation * 1000:9.1f} ms "
          f"<- the operation")
    print(f"  of which the slope read {saved * 1000:9.1f} ms")
    gate("G-9 hold the slope band on _state", saved, per_activation)


# ---------------------------------------------------------------------------- G-8

def _g8(h):
    """`_mask_to_qgs_geometry` against the ponding-query click it sits in."""
    import numpy as np

    from p_perf_gate import gate
    from terrainflow_assessment.map_tools.ponding_query_tool import PondingQueryTool

    result = h.state.baseline_result or {}
    ponding = result.get("ponding") or getattr(h.state, "ponding_raster_path", None)
    if not ponding or not os.path.exists(ponding):
        print("G-8: no ponding raster from the baseline — cannot price it")
        return

    tool = PondingQueryTool(h.canvas, ponding)
    arr = tool.ponding_array
    wet = np.argwhere(arr >= tool.MIN_DEPTH)
    print(f"G-8 ponding query; raster {arr.shape}, {len(wet)} wet cells")
    if not len(wet):
        print("  no wet cells — the click path never runs")
        return

    # The biggest pond, so the number is the one a user actually waits on.
    from scipy.ndimage import label as nd_label
    labels, n = nd_label(arr >= tool.MIN_DEPTH)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    pid = int(sizes.argmax())
    rc = np.argwhere(labels == pid)[0]
    row, col = int(rc[0]), int(rc[1])
    print(f"  {n} ponds; clicking the largest ({sizes[pid]} cells)")

    t0 = time.perf_counter()
    _vol, _cells, visited = tool._flood_fill(row, col)
    t_fill = time.perf_counter() - t0

    t0 = time.perf_counter()
    geom_now = tool._mask_to_qgs_geometry(visited)
    t_poly = time.perf_counter() - t0

    t0 = time.perf_counter()
    geom_fixed = _masked_polygonise(visited, tool.transform)
    t_poly_fixed = time.perf_counter() - t0

    operation = t_fill + t_poly
    print(f"  _flood_fill             {t_fill * 1000:9.1f} ms")
    print(f"  _mask_to_qgs_geometry   {t_poly * 1000:9.1f} ms")
    print(f"  with mask=              {t_poly_fixed * 1000:9.1f} ms")
    print(f"  click operation         {operation * 1000:9.1f} ms")
    same = (geom_now is not None and geom_fixed is not None
            and geom_now.equals(geom_fixed))
    print(f"  identical geometry:     {same}")
    gate("G-8 pass mask= to rasterio shapes", t_poly - t_poly_fixed, operation)


def _masked_polygonise(mask, transform):
    """What G-8's fix does: trace only the True region."""
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    from qgis.core import QgsGeometry

    uint_mask = mask.astype("uint8")
    polys = [shapely_shape(geom)
             for geom, val in rasterio_shapes(uint_mask, mask=mask,
                                              transform=transform)
             if val == 1]
    if not polys:
        return None
    return QgsGeometry.fromWkt(unary_union(polys).wkt)


# --------------------------------------------------------------------------- M-10

def _m10(h):
    """`snap_point_to_contour_elevation` against the two clicks that call it."""
    import numpy as np
    from rasterio.windows import Window

    from p_perf_gate import bench, gate
    from terrainflow_assessment.modules.swale_design import (
        snap_point_to_contour_elevation,
    )

    controller = h.plugin._earthworks
    dem_path = h.state.dem_path
    ews = [e for e in h.state.earthwork_manager.get_all()
           if e.type in ("swale", "diversion")]
    if not ews:
        print("M-10: no linear feature to place on")
        return
    ew = ews[0]
    pt = ew.geometry.centroid().asPoint()

    t_full = bench(lambda: snap_point_to_contour_elevation((pt.x(), pt.y()), dem_path),
                   repeat=5)
    t_win = bench(lambda: _windowed_sample(dem_path, pt.x(), pt.y(), Window), repeat=5)
    print("M-10 snap_point_to_contour_elevation")
    print(f"  as written (full band)  {t_full * 1000:9.1f} ms")
    print(f"  windowed 1x1            {t_win * 1000:9.1f} ms")
    print(f"  same value:             "
          f"{_same_value(dem_path, pt, Window, snap_point_to_contour_elevation)}")

    # Operation 1: the spillway-placement click.
    from terrainflow_assessment.map_tools.place_point_tool import PlacePointTool

    tool = PlacePointTool(h.canvas, snap_raster_path=dem_path,
                         constrain_to=controller._spillway_constraint(ew))
    t0 = time.perf_counter()
    elev = tool._elevation_at(pt)
    t_elev = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        controller._on_spillway_placed(ew.id, pt, elev, kind="outflow")
    except Exception as exc:
        print(f"  (_on_spillway_placed raised: {exc})")
    t_placed = time.perf_counter() - t0
    click = t_elev + t_placed
    print(f"  spillway click: _elevation_at {t_elev * 1000:.1f} ms + "
          f"_on_spillway_placed {t_placed * 1000:.1f} ms = {click * 1000:.1f} ms")
    gate("M-10 windowed read (spillway-placement click)", t_full - t_win, click)

    # Operation 2: `_orient_downhill`, two samples, inside the drawn-geometry path.
    t_orient = bench(lambda: controller._orient_downhill(ew.geometry), repeat=3)
    print(f"  _orient_downhill        {t_orient * 1000:9.1f} ms (2 samples)")
    gate("M-10 windowed read (_orient_downhill, 2 calls)",
         2 * (t_full - t_win), t_orient)
    del np


def _windowed_sample(dem_path, x, y, Window):
    import numpy as np
    import rasterio

    from terrainflow_assessment.modules.swale_design import xy_to_rc

    with rasterio.open(dem_path) as src:
        row, col = xy_to_rc(src.transform, x, y)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        val = src.read(1, window=Window(col, row, 1, 1))[0, 0]
        nodata = src.nodata
        if nodata is None or not np.isclose(val, nodata):
            return float(val)
    return None


def _same_value(dem_path, pt, Window, snap):
    a = snap((pt.x(), pt.y()), dem_path)
    b = _windowed_sample(dem_path, pt.x(), pt.y(), Window)
    return a == b


if __name__ == "__main__":
    main()
