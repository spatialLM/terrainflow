"""p_interaction_freeze — where the 4.6 s and 5.2 s after a click actually go.

Two controller handlers on the real design freeze the UI for seconds:

    `_on_spillway_placed`        4.6 s   (earthworks.py:609)
    `_on_vertex_edit_finished`   5.2 s   (earthworks.py:1560)

**Neither gate bar applies, and that is a gap in the gate rather than a reason to
ignore them.** Both are click-then-freeze, not drag frames, so there is no 50 ms
frame budget; and there is no *enclosing* operation either — the handler **is**
the operation, so gate rule 1's denominator would be the thing itself and every
item would read 100%. The bar for this probe is stated in the brief instead: the
interaction is done when it is either **under ~200 ms**, or asynchronous with
visible feedback.

So this probe does not call `gate()`. It answers the one question that decides
what happens next:

    Is this work that has to happen — a genuine burn-and-reflood — or is it a
    full re-run where an incremental one would do?

Those want opposite answers. Work that has to happen is a threading change (a
progress cursor, or getting it off the paint thread), which is the owner's call
on a QGIS controller. A full re-run where an incremental one would do is an
optimisation, and it can be done here.

Rule 3 throughout: **count** the calls as well as timing them. A count is what
says whether the mechanism is the one the register described — and for these two
the specific suspicion is a per-*feature* cost being paid for all 35 features
when one moved.

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_interaction_freeze.py

Needs the real design: the committed 400x400 clip carries two ponds and no
spillways, so neither handler does on it what it does on a design of 35.

What it found, 2026-09-12
-------------------------
**Both, and in that order.** The measurement reproduces the register's figures
(4,750 ms and 5,448 ms here) and attributes them to one place: over 80% of each
handler is `_recompute_live_assessment`, and within it `refresh_stress_points_layer`
is **74.6%** of the whole vertex-edit handler, spent entirely in
`feature_inflow_profile` called **30 times** — once per linear feature, when one
feature moved.

The scaling stage settles the fork: 35 features enabled cost 5,477 ms and 2 cost
1,207 ms, so **4,269 ms of the freeze is the other features**. That is a full
re-run where an incremental one would do.

Splitting `feature_inflow_profile` into its three parts named the term:

    labels == id, then nonzero      122 ms   (30 full-raster passes)
    line.project per cell         3,236 ms   <- the freeze
    shapely.line_locate_point       622 ms   (same answer, 0.0 m disagreement)
    inflow_profile (the bincount)    18 ms

So `[line.project(Point(x, y)) for ...]` — one Python-level shapely call per
catchment cell, capped at 20,000 per feature, for every feature — was the single
largest thing either handler did. It is now
`swale_design.line_stations`, the same GEOS routine over an array.

    _on_spillway_placed        4,750 -> 2,055 ms   (2.3x)
    _on_vertex_edit_finished   5,448 -> 2,718 ms   (2.0x)

**Still a freeze, and that is the part this probe cannot close.** 2.1 s and 2.7 s
are 10x and 14x the 200 ms bar. What is left is not one hotspot: after the swap
`refresh_stress_points_layer` is down to 28% and the rest is spread across
`_build_spillway_rows` (35 rows, 31 `_spillway_datums`, each sampling the DEM),
`recompute_catchments` and `_refresh_terrain_capacity`. And it is **still** mostly
the other features — 1,819 ms of the remaining 2,718 ms. Closing the gap means
either making the whole live assessment incremental, which needs a soundness
argument this probe has not got (`recompute_catchments` runs first and can change
every feature's catchment, so "only the edited one" is not obviously true), or
moving the work off the paint thread with visible feedback. Both are threading or
architecture decisions on a QGIS controller, and both are the owner's call.
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

#: The brief's bar for this item. Not a percentage — there is nothing to be a
#: percentage of.
RESPONSIVE_MS = 200.0

#: Measured on this machine, on the 03.09.2026 design, **before** the projection
#: in `feature_inflow_profile` was vectorised (2026-09-12). Kept here so a re-run
#: prints the delta rather than a bare number a reader has to find the other half
#: of. Machine-specific in absolute terms; the ratio is the durable part.
BEFORE_MS = {
    "_on_spillway_placed": 4750.2,
    "_on_vertex_edit_finished": 5447.7,
}

#: The methods each handler leans on, timed and counted individually. Named
#: rather than discovered so the printout is stable and a new one shows up as an
#: unexplained remainder rather than being silently absorbed.
WATCHED = (
    "_refresh_terrain_capacity",
    "_refresh_dam_stage_storage",
    "_compute_dam_capacity",
    "_refresh_spillway_layer",
    "_refresh_ew_layer",
    "_recompute_live_assessment",
    "recompute_catchments",
    "_resnap_spillways",
    "_spillway_datums",
    "_orient_downhill",
    "_mark_design_edit",
    "_refresh_spillway_link_inverts",
)


def _counted(obj, name):
    """Count + time calls to a bound attribute of *obj*. Returns (log, undo).

    Copied from `p_gate_burn` deliberately rather than imported: a probe that
    reaches into another probe for its instrumentation makes the other one
    unsafe to change.
    """
    real = getattr(obj, name, None)
    if real is None:
        return None, (lambda: None)
    log = {"n": 0, "secs": 0.0, "name": name}

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return real(*args, **kwargs)
        finally:
            log["n"] += 1
            log["secs"] += time.perf_counter() - t0

    setattr(obj, name, wrapper)
    return log, lambda: setattr(obj, name, real)


class Watch:
    """Instrument every method in WATCHED on one controller, then report."""

    def __init__(self, controller, names=WATCHED):
        self.controller = controller
        self.names = names
        self.logs = []
        self.undos = []

    def __enter__(self):
        for name in self.names:
            log, undo = _counted(self.controller, name)
            if log is not None:
                self.logs.append(log)
            self.undos.append(undo)
        return self

    def __exit__(self, *exc):
        for undo in self.undos:
            undo()
        return False

    def report(self, total_secs):
        """Print each watched method's share, and what is left unattributed.

        The remainder matters as much as the rows: if the named methods account
        for 20% of the freeze, the mechanism is somewhere this probe is not
        looking and no amount of optimising them will be felt.
        """
        rows = sorted((ln for ln in self.logs if ln["n"]),
                      key=lambda ln: ln["secs"], reverse=True)
        print(f"    {'method':<34} {'calls':>6} {'total ms':>10} "
              f"{'ms/call':>9} {'share':>7}")
        # Nested calls double-count against the total (a handler calls
        # `_refresh_terrain_capacity`, which calls `_refresh_dam_stage_storage`),
        # so the shares do not sum to 1 and are not meant to. Each row is "what
        # would disappear if this method were free", which is the question worth
        # asking of each one separately.
        for ln in rows:
            share = ln["secs"] / total_secs if total_secs else 0.0
            print(f"    {ln['name']:<34} {ln['n']:>6} "
                  f"{ln['secs'] * 1000:>10.1f} "
                  f"{ln['secs'] / ln['n'] * 1000:>9.1f} {share:>6.1%}")
        top = max((ln["secs"] for ln in rows), default=0.0)
        print(f"    {'-' * 34} {'-' * 6} {'-' * 10} {'-' * 9} {'-' * 7}")
        print(f"    {'the handler itself':<34} {1:>6} "
              f"{total_secs * 1000:>10.1f} {total_secs * 1000:>9.1f} "
              f"{1.0:>6.1%}")
        print(f"    not inside the largest watched call: "
              f"{(total_secs - top) * 1000:.1f} ms")
        return [dict(ln) for ln in rows]


def verdict(label, secs, features):
    """State the bar's answer. No percentage — there is nothing to divide by."""
    ms = secs * 1000
    ok = ms <= RESPONSIVE_MS
    was = BEFORE_MS.get(label)
    delta = "" if was is None else f" (was {was:.0f} ms, {was / ms:.1f}x)"
    print(f"\n  {label}: {ms:.0f} ms on a design of {features} features "
          f"against a {RESPONSIVE_MS:.0f} ms bar{delta} -> "
          f"{'RESPONSIVE' if ok else 'A FREEZE'}")
    if not ok:
        print(f"    {ms / RESPONSIVE_MS:.0f}x the bar. Either it comes down, or the "
              f"interaction becomes asynchronous with visible feedback.")
    return ok


def main():
    import _probe

    _probe.start_qgis()

    from _harness import PluginHarness, make_workers_synchronous

    make_workers_synchronous()

    from p_perf_gate import real_dem

    dem = real_dem()
    print(f"DEM {dem}")
    if not os.path.exists(REAL_TFD):
        raise SystemExit(
            f"the real design is not at {REAL_TFD}. Both handlers are priced by "
            f"how many features the design carries, so the 400x400 clip — two "
            f"ponds, no spillways — would answer a different question.")

    ev = _probe.Evidence("p_interaction_freeze",
                         ["spillway-placed-4.6s", "vertex-edit-5.2s"], dem=dem)

    with PluginHarness(dem, load_boundary=False) as h:
        t0 = time.perf_counter()
        h.plugin._design_file._restore_design(REAL_TFD)
        print(f"restore design  {(time.perf_counter() - t0) * 1000:9.1f} ms")
        t0 = time.perf_counter()
        h.run_baseline()
        print(f"baseline        {(time.perf_counter() - t0) * 1000:9.1f} ms")

        ews = h.state.earthwork_manager.get_all()
        ev["features"] = len(ews)
        ev["by_type"] = {t: sum(1 for e in ews if e.type == t)
                         for t in sorted({e.type for e in ews})}
        print(f"features        {len(ews)}  {ev['by_type']}")

        with ev.stage("spillway_placed"):
            ev["spillway_placed"] = _profile_spillway(h, ev)
        with ev.stage("vertex_edit_finished"):
            ev["vertex_edit_finished"] = _profile_vertex(h, ev)
        with ev.stage("per_feature_scaling"):
            ev["per_feature_scaling"] = _scaling(h, ev)
        with ev.stage("inside_live_assessment"):
            ev["inside_live_assessment"] = _inside(h, ev)
        with ev.stage("inside_inflow_profile"):
            ev["inside_inflow_profile"] = _inside_profile(h, ev)

    ev.write()


# ------------------------------------------------------------ _on_spillway_placed

def _profile_spillway(h, ev):
    """Place an outflow spillway on a real dam, timed and counted."""
    controller = h.plugin._earthworks

    dams = [e for e in h.state.earthwork_manager.get_all() if e.type == "dam"]
    if not dams:
        print("  no dam in the design — nothing to place a spillway on")
        return None
    ew = dams[0]
    pt = ew.geometry.centroid().asPoint()

    from terrainflow_assessment.modules.swale_design import (
        snap_point_to_contour_elevation,
    )
    elev = snap_point_to_contour_elevation((pt.x(), pt.y()), h.state.dem_path)

    # A crest this design already carries sends the handler down the short path —
    # `spillway.crest_elevation is None` is what gates `_spillway_datums`. Clear
    # it so the *expensive* path is the one being measured, which is the one a
    # user gets the first time they place a sill.
    existing = getattr(ew, "spillway", None)
    had_crest = getattr(existing, "crest_elevation", None) if existing else None
    if existing is not None:
        existing.crest_elevation = None

    print(f"\n  _on_spillway_placed on {ew.name} "
          f"(crest was {had_crest}, cleared for the cold path)")
    with Watch(controller) as w:
        t0 = time.perf_counter()
        controller._on_spillway_placed(ew.id, pt, elev, kind="outflow")
        total = time.perf_counter() - t0
        rows = w.report(total)

    ok = verdict("_on_spillway_placed", total, len(h.state.earthwork_manager))
    return {"total_ms": total * 1000, "before_ms": BEFORE_MS["_on_spillway_placed"],
            "responsive": ok, "feature": ew.name, "rows": rows}


# ------------------------------------------------------- _on_vertex_edit_finished

def _profile_vertex(h, ev):
    """Finish a vertex edit on a real linear feature, timed and counted."""
    from qgis.core import QgsGeometry, QgsPointXY

    controller = h.plugin._earthworks
    manager = h.state.earthwork_manager
    ews = manager.get_all()

    idx = next((i for i, e in enumerate(ews)
                if e.type in ("swale", "diversion")), None)
    if idx is None:
        print("  no linear feature to reshape")
        return None
    ew = manager.get(idx)

    # Move the last vertex a metre. A real edit, and small enough that nothing
    # downstream refuses it for being off the DEM.
    pts = [QgsPointXY(p) for p in ew.geometry.asPolyline()]
    if len(pts) < 2:
        print("  the feature has no polyline to reshape")
        return None
    pts[-1] = QgsPointXY(pts[-1].x() + 1.0, pts[-1].y() + 1.0)
    moved = QgsGeometry.fromPolylineXY(pts)

    print(f"\n  _on_vertex_edit_finished on {ew.name} (index {idx}, "
          f"{len(pts)} vertices)")
    with Watch(controller) as w:
        t0 = time.perf_counter()
        controller._on_vertex_edit_finished(idx, moved)
        total = time.perf_counter() - t0
        rows = w.report(total)

    ok = verdict("_on_vertex_edit_finished", total, len(manager))
    return {"total_ms": total * 1000,
            "before_ms": BEFORE_MS["_on_vertex_edit_finished"],
            "responsive": ok, "feature": ew.name,
            "vertices": len(pts), "rows": rows}


# ------------------------------------------------------------- the scaling question

def _scaling(h, ev):
    """Does the cost scale with the design, or with the one feature that moved?

    This is the fork. A handler whose cost is a constant per *edited* feature is
    doing work that has to happen and the answer is threading. One that costs
    N x per-feature for a design of N when a single feature moved is a full
    re-run where an incremental one would do, and that is an optimisation.

    Measured by disabling features rather than by reasoning about the call graph:
    `_recompute_live_assessment` and `recompute_catchments` both loop over
    `earthwork_manager.get_all()`, and whether that loop is the cost is a
    question about what is *inside* it.
    """
    from qgis.core import QgsGeometry, QgsPointXY

    controller = h.plugin._earthworks
    manager = h.state.earthwork_manager
    ews = manager.get_all()

    idx = next((i for i, e in enumerate(ews)
                if e.type in ("swale", "diversion")), None)
    if idx is None:
        return None
    ew = manager.get(idx)
    pts = [QgsPointXY(p) for p in ew.geometry.asPolyline()]
    if len(pts) < 2:
        return None

    rows = []
    original = [e.enabled for e in ews]
    try:
        for keep in (len(ews), 18, 9, 4, 1):
            # Leave the edited feature enabled whatever else is off; disabling it
            # would measure a different edit rather than a smaller design.
            for i, e in enumerate(ews):
                e.enabled = (i == idx) or (i < keep)
            live = sum(1 for e in ews if e.enabled)
            pts[-1] = QgsPointXY(pts[-1].x() + 0.5, pts[-1].y() + 0.5)
            moved = QgsGeometry.fromPolylineXY(pts)
            t0 = time.perf_counter()
            controller._on_vertex_edit_finished(idx, moved)
            secs = time.perf_counter() - t0
            rows.append({"enabled_features": live, "ms": secs * 1000})
            print(f"    {live:>3} features enabled -> {secs * 1000:8.1f} ms")
    finally:
        for e, was in zip(ews, original):
            e.enabled = was

    if len(rows) >= 2:
        big, small = rows[0], rows[-1]
        span = big["ms"] - small["ms"]
        print(f"    {big['enabled_features']} features {big['ms']:.0f} ms vs "
              f"{small['enabled_features']} {small['ms']:.0f} ms — "
              f"{span:.0f} ms of the freeze is the *other* features")
        print("    -> " + (
            "the cost is the design, not the edit: a full re-run where an "
            "incremental one would do"
            if span > 0.5 * big["ms"] else
            "the cost follows the edited feature: work that has to happen"))
    return rows


# ------------------------------------------------------- inside the one that dominates

def _inside(h, ev):
    """`cProfile` the whole handler, because the named methods miss most of it.

    The watched list accounts for `_spillway_datums` and little else inside
    `_recompute_live_assessment` — about 800 ms of 4,500 — so most of the freeze
    is in code this probe has no name for. Naming it by hand would be guessing
    twice; `cProfile` names it once.

    Cumulative time, not total: the question is which *call* owns the freeze, and
    a leaf reached from three places tells you less than the branch above it.
    Deterministic profiling inflates everything, so read the ordering and the
    ratios here; the absolute milliseconds come from the unprofiled runs above.
    """
    import cProfile
    import pstats

    from qgis.core import QgsGeometry, QgsPointXY

    controller = h.plugin._earthworks
    manager = h.state.earthwork_manager
    ews = manager.get_all()

    # Profile the **handler**, not `_recompute_live_assessment` on its own. A
    # second bare call to the latter runs in a quarter of the time the handler
    # pays for it — something in there is warm once nothing has been edited — so
    # profiling it directly would attribute the freeze against a run that is not
    # the one a user waits on. This is gate rule 1 wearing a different hat: the
    # measurement has to be of the operation, not of a convenient proxy for it.
    idx = next((i for i, e in enumerate(ews)
                if e.type in ("swale", "diversion")), None)
    if idx is None:
        return None
    ew = manager.get(idx)
    pts = [QgsPointXY(p) for p in ew.geometry.asPolyline()]
    if len(pts) < 2:
        return None
    pts[-1] = QgsPointXY(pts[-1].x() + 0.25, pts[-1].y() + 0.25)

    prof = cProfile.Profile()
    prof.enable()
    controller._on_vertex_edit_finished(idx, QgsGeometry.fromPolylineXY(pts))
    prof.disable()

    stats = pstats.Stats(prof)
    total = stats.total_tt
    rows = []
    for (fname, lineno, func), (_cc, nc, _tt, ct, _cal) in stats.stats.items():
        if "terrainflow_assessment" not in fname and "rasterio" not in fname \
                and "pysheds" not in fname:
            continue
        rows.append({"func": f"{os.path.basename(fname)}:{lineno} {func}",
                     "calls": nc, "cumulative_ms": ct * 1000,
                     "share": ct / total if total else 0.0})
    rows.sort(key=lambda r: r["cumulative_ms"], reverse=True)

    print(f"    _on_vertex_edit_finished under cProfile: {total * 1000:.0f} ms "
          f"(inflated by profiling; read the ratios, not the milliseconds)")
    print(f"    {'function':<62} {'calls':>6} {'cum ms':>9} {'share':>7}")
    for r in rows[:22]:
        print(f"    {r['func']:<62} {r['calls']:>6} "
              f"{r['cumulative_ms']:>9.1f} {r['share']:>6.1%}")
    return {"total_ms": total * 1000, "rows": rows[:40]}


# ---------------------------------------------- the three parts of one profile

def _inside_profile(h, ev):
    """Split `feature_inflow_profile` into its three parts, for every feature.

    Three candidates and they want different fixes, so they are timed apart:

    1. ``labels == index`` then ``np.nonzero`` — a full-raster pass **per
       feature**, which is the shape M-11 already found once in this codebase.
    2. ``[line.project(Point(x, y)) for ...]`` — one shapely call per catchment
       cell, capped at ``_PROFILE_MAX_CELLS`` (20,000) but uncapped in Python
       overhead below that.
    3. ``inflow_profile`` itself, which its own docstring calls "a bincount
       rather than new analysis".

    The vectorised form of (2) is timed beside it and its answer compared
    element-by-element, because a faster projection that moves a station is not a
    speedup, it is a defect.
    """
    import numpy as np
    from shapely.geometry import Point

    controller = h.plugin._earthworks
    labels = h.state.catchment_labels
    meta = h.state.flow_grid_meta
    label_ids = list(h.state.catchment_label_ids or [])
    if labels is None or meta is None or not label_ids:
        print("    no catchment labels on state — nothing to split")
        return None

    try:
        from shapely import line_locate_point
    except ImportError:
        line_locate_point = None
        print("    shapely.line_locate_point unavailable — (2) has no vectorised form")

    rows_out = []
    t_mask = t_project = t_profile = t_vector = 0.0
    worst = 0
    mismatch = 0.0
    for ew in h.state.earthwork_manager.get_all():
        if not ew.enabled or ew.type not in ("swale", "diversion", "berm"):
            continue
        if ew.id not in label_ids:
            continue
        line = controller._shapely_of(ew)
        if line is None or line.length <= 0:
            continue

        t0 = time.perf_counter()
        mask = labels == label_ids.index(ew.id)
        rs, cs = np.nonzero(mask)
        t_mask += time.perf_counter() - t0
        if rs.size == 0:
            continue
        scale = 1.0
        if rs.size > controller._PROFILE_MAX_CELLS:
            stride = int(np.ceil(rs.size / controller._PROFILE_MAX_CELLS))
            rs, cs = rs[::stride], cs[::stride]
            scale = float(stride)
        worst = max(worst, rs.size)

        tr = meta["transform"]
        xs = tr.c + (cs + 0.5) * tr.a
        ys = tr.f + (rs + 0.5) * tr.e

        t0 = time.perf_counter()
        distances = [line.project(Point(float(x), float(y)))
                     for x, y in zip(xs, ys)]
        t_project += time.perf_counter() - t0

        if line_locate_point is not None:
            from shapely import points as shapely_points
            t0 = time.perf_counter()
            fast = line_locate_point(line, shapely_points(xs, ys))
            t_vector += time.perf_counter() - t0
            mismatch = max(mismatch,
                           float(np.abs(np.asarray(distances) - fast).max()))

        from terrainflow_assessment.modules.swale_design import inflow_profile
        per_cell = meta["cell_area_m2"] * 60.0 / 1000.0 * scale
        t0 = time.perf_counter()
        inflow_profile(distances, [per_cell] * len(distances), line.length)
        t_profile += time.perf_counter() - t0
        rows_out.append({"feature": ew.name, "cells": int(rs.size)})

    n = len(rows_out)
    print(f"    {n} linear features profiled, largest {worst:,} cells "
          f"(cap {controller._PROFILE_MAX_CELLS:,})")
    print(f"    labels == id, then nonzero   {t_mask * 1000:9.1f} ms  "
          f"({n} full-raster passes)")
    print(f"    line.project per cell        {t_project * 1000:9.1f} ms")
    print(f"    shapely.line_locate_point    {t_vector * 1000:9.1f} ms"
          if line_locate_point is not None else
          "    shapely.line_locate_point          n/a")
    print(f"    inflow_profile (the bincount){t_profile * 1000:9.1f} ms")
    if line_locate_point is not None:
        print(f"    largest disagreement between the two projections: "
              f"{mismatch:.3e} m")
    return {
        "features": n,
        "largest_cells": worst,
        "mask_ms": t_mask * 1000,
        "project_ms": t_project * 1000,
        "vector_ms": t_vector * 1000,
        "profile_ms": t_profile * 1000,
        "max_projection_disagreement_m": mismatch,
        "rows": rows_out,
    }


if __name__ == "__main__":
    main()
