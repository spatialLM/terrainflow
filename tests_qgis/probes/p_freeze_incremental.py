"""When one vertex moves, how much of the design actually changes?

`66101e6` took the vertex-edit freeze from 5,448 ms to 2,718 ms by vectorising the
projection. What is left is not one slow call: it is **thirty features being recomputed
because one of them moved**. Of the ~2,700 ms remaining, roughly 1,819 ms is spent on
features the user did not touch.

Whether that work is *wasted* is the whole question, and it is not obvious. Moving a
vertex re-routes water, and re-routed water can change which feature a cell drains to
anywhere downslope — so "only the edited feature changed" is a hypothesis about this
terrain, not a fact about the algorithm. If most features' catchments are untouched by a
typical edit, a cache keyed on each feature's own catchment cells skips most of the work
and is exactly correct. If they all shift, no cache is sound and the answer is
asynchrony, which is the owner's call.

This measures which. It does not change anything.

Run: & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_freeze_incremental.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _probe  # noqa: E402

#: The owner's design. Outside the repo — the terrain is open data and committed, the
#: design is client work and is not. Every stage skips cleanly when it is absent.
REAL_TFD = (r"F:\Terrain Flow Design\QGIS Working Files"
            r"\Quail_Island 03.09.2026.tfd")

#: How far a probed vertex is nudged. Small on purpose: a realistic tweak, not a
#: redesign, because the question is what an ordinary edit disturbs.
NUDGE_M = 0.25

#: A larger move, to check the answer is not an artefact of the small one.
BIG_NUDGE_M = 25.0


def catchment_fingerprints(state):
    """`{feature id: (cell count, digest of its cells)}` from the current labels.

    The digest is over the flat indices, so two different sets of the same size are
    different — a count alone would call a catchment that moved sideways unchanged,
    which is the quantity this probe exists to measure not moving.
    """
    labels = state.catchment_labels
    ids = list(state.catchment_label_ids or [])
    if labels is None or not ids:
        return {}
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    # One pass: the run of each label in the sorted array gives that label's cells.
    bounds = np.searchsorted(sorted_flat, np.arange(len(ids) + 1))
    out = {}
    for i, fid in enumerate(ids):
        cells = np.sort(order[bounds[i]:bounds[i + 1]])
        out[fid] = (int(cells.size),
                    hashlib.sha256(cells.tobytes()).hexdigest()[:16])
    return out


def compare(before, after):
    ids = sorted(set(before) | set(after))
    changed, same, appeared, vanished = [], [], [], []
    for fid in ids:
        b, a = before.get(fid), after.get(fid)
        if b is None:
            appeared.append(fid)
        elif a is None:
            vanished.append(fid)
        elif b == a:
            same.append(fid)
        else:
            changed.append((fid, b[0], a[0]))
    return changed, same, appeared, vanished


def nudge(h, controller, idx, metres):
    """Move the last vertex of feature *idx* and run the real settled handler."""
    from qgis.core import QgsGeometry, QgsPointXY

    ew = h.state.earthwork_manager.get(idx)
    pts = [QgsPointXY(p) for p in ew.geometry.asPolyline()]
    if len(pts) < 2:
        return None
    pts[-1] = QgsPointXY(pts[-1].x() + metres, pts[-1].y() + metres)

    t0 = time.perf_counter()
    controller._on_vertex_edit_finished(idx, QgsGeometry.fromPolylineXY(pts))
    return (time.perf_counter() - t0) * 1000.0


def main():
    _probe.start_qgis()
    ev = {"probe": "p_freeze_incremental", "stages": {}}

    if not os.path.exists(REAL_TFD):
        print(f"SKIP: the design is not on this machine ({REAL_TFD})")
        ev["stages"]["all"] = {"status": "skipped", "reason": "design absent"}
        return _write(ev)

    from _harness import PluginHarness, make_workers_synchronous

    # Without this `run_baseline()` hands off to a worker thread and returns in 0 ms,
    # and every measurement below would be of an empty state.
    make_workers_synchronous()

    from p_perf_gate import real_dem

    dem = real_dem()
    print(f"DEM: {dem}")
    ev["dem"] = dem

    with PluginHarness(dem, load_boundary=False) as h:
        h.plugin._design_file._restore_design(REAL_TFD)
        controller = h.plugin._earthworks
        manager = h.state.earthwork_manager
        ews = manager.get_all()
        print(f"design: {len(ews)} features")

        # A baseline has to exist before catchments mean anything.
        t0 = time.perf_counter()
        h.run_baseline()
        print(f"baseline: {(time.perf_counter() - t0) * 1000:.0f} ms")
        controller.recompute_catchments()

        rows = []
        for label, metres in (("small nudge", NUDGE_M), ("25 m move", BIG_NUDGE_M)):
            before = catchment_fingerprints(h.state)
            if not before:
                print("  no catchment labels — cannot answer the question")
                break

            idx = next((i for i, e in enumerate(manager.get_all())
                        if e.type in ("swale", "diversion")), None)
            if idx is None:
                break
            moved = manager.get(idx)
            ms = nudge(h, controller, idx, metres)
            after = catchment_fingerprints(h.state)

            changed, same, appeared, vanished = compare(before, after)
            moved_changed = any(fid == moved.id for fid, _b, _a in changed)

            total = len(before)
            others_changed = len([1 for fid, _b, _a in changed if fid != moved.id])
            row = {
                "case": label,
                "nudge_m": metres,
                "handler_ms": round(ms, 1) if ms else None,
                "feature": moved.name,
                "features_with_catchments": total,
                "unchanged": len(same),
                "changed": len(changed),
                "moved_feature_changed": moved_changed,
                "other_features_changed": others_changed,
                "appeared": len(appeared),
                "vanished": len(vanished),
            }
            rows.append(row)

            print("")
            print(f"  --- {label} ({metres} m) on {moved.name}, "
                  f"handler {row['handler_ms']} ms ---")
            print(f"    features with a catchment      {total}")
            print(f"    unchanged                      {len(same)}")
            print(f"    changed                        {len(changed)}"
                  f"   (the moved one: {moved_changed})")
            print(f"    other features changed         {others_changed}"
                  f"   <- what a cache could skip: {total - others_changed - 1} of {total}")
            if changed[:8]:
                print("    biggest movers (id, cells before -> after):")
                for fid, b, a in sorted(changed, key=lambda c: -abs(c[2] - c[1]))[:8]:
                    print(f"      {fid[:12]:12s} {b:8,d} -> {a:8,d}  ({a - b:+,d})")

        ev["stages"]["catchment_churn"] = {"status": "ok", "rows": rows}

        if rows:
            worst = max(r["other_features_changed"] for r in rows)
            total = rows[0]["features_with_catchments"]
            print("")
            print(f"  VERDICT: at worst {worst} of {total} untouched features had their "
                  f"catchment change.")
            if worst == 0:
                print("    A per-feature cache keyed on the catchment cells is exactly "
                      "correct and skips all of it.")
            elif worst < total / 4:
                print("    A cache keyed on the catchment cells is sound and skips most "
                      "of the work; the changed ones still recompute.")
            else:
                print("    Too much moves for a cache to help. The answer is asynchrony, "
                      "which is the owner's call, not an optimisation.")

    return _write(ev)


def _write(ev):
    dest = os.path.join(HERE, "evidence", "p_freeze_incremental.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(ev, fh, indent=2)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
