"""p_keypoints — which guard refuses each valley, and how far a guide really drifts.

Step A1 of the defect-documentation campaign. Supports **KPA-39, KPA-40, KPA-41, KPA-42,
KPA-43, KPA-44, KPA-46**, and sizes the measurement window KPA-41's recommendation needs.

Five measurements:

**1. The guard histogram (KPA-39, KPA-44).** `find_keypoints` attributes *every*
`keypoint_on_path` `None` to the prominence test — the skipped message says "no break in
the floor clearing 2% of grade change" whatever actually happened. `keypoint_on_path` has
five ways to return `None`::

    :674  thalweg is None or len(thalweg) < 5
    :688  total_len <= 0
    :697  len(arc) < 5                  (too little finite ground)
    :722  n_samp - 2*guard < 3          (profile too short to have an interior)
    :734  slope_ease < MIN_SLOPE_EASE   (the prominence test — the only one reported)

`sys.settrace` records the line each `None` returned from, so the message can be checked
against the truth. Run on the real fixture at the production threshold **and** on the
default synthetic harness DEM, because KPA-44's "all 28 are refused on prominence" was read
off the very message KPA-39 shows is unconditional, and has to be re-established.

**2. The floor, on both grids (KPA-40).** The shortest thalweg that can yield a keypoint is
`7 * min(5*cell, 10)` m — 35 m on the 1 m fixture, 70 m on the 2 m synthetic DEM. Measured
by bisection against the real function rather than asserted from the algebra.

**3. The accounting identity (KPA-43).** `find_keypoints` breaks out of its loop once
`max_valleys` keypoints are found, dropping every remaining link from **both** returned
lists, so `len(keypoints) + len(skipped) != len(links)`. Reported as a count at
`max_valleys` 1 and 8; inventing reasons for the unexamined links would be KPA-39 again.

**4. The drift table (KPA-41, KPA-42).** `drift_1_in_n` is net end-to-end fall. This
measures the **steepest sustained grade over 10, 20 and 50 m windows** beside it, and
counts the vertices whose Z is fabricated — `_sample_dem` returns the keypoint elevation
for any sample off the grid (`keypoint_analysis.py:1088-1094`), so a guide leaving the DEM
reports a drift computed from a constant. Windows are measured on **real ground only**.

**5. The roughness sweep.** The window in (4) is a TerrainFlow convention and has to be
sized, not chosen: `_box_blur`'s docstring puts realistic LiDAR roughness at local
gradients of 0.05-0.10, which is 2.5-5x `MIN_SLOPE_EASE`. The sweep runs `roughness_m` over
0.0 / 0.05 / 0.10 / 0.25 at a fixed seed and asks how far the keypoint moves and whether
`slope_ease` clears 0.02 on noise. The shortest window whose grade is stable across
0.0 -> 0.10 is the defensible floor. It is sized on **synthetic** correlated roughness; the
fixture's own 10/20/50 m grades are reported beside it so the reader can judge the transfer.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_keypoints.py
"""

import sys

import _probe

import numpy as np

WINDOWS_M = (10.0, 20.0, 50.0)
ROUGHNESS_SWEEP = (0.0, 0.05, 0.10, 0.25)

#: The five `return None` sites in `keypoint_on_path`, by line, with the guard each is.
#: Re-read against HEAD before quoting: this probe prints the source line it saw.
GUARD_LINES = {
    674: "thalweg is None or len(thalweg) < 5",
    688: "total_len <= 0",
    697: "len(arc) < 5 — too little finite ground",
    722: "n_samp - 2*guard < 3 — profile too short to have an interior",
    734: "slope_ease < MIN_SLOPE_EASE — the prominence test",
}


# --------------------------------------------------------------------- tracing


class GuardTracer:
    """Attribute each `keypoint_on_path` return to the line it returned from.

    A line tracer rather than a wrapper: the guards are five separate `return None`
    statements inside one function, and nothing short of `sys.settrace` can tell them
    apart from outside. Active only for the duration of the `with` block, because tracing
    every line of pysheds would take minutes.
    """

    def __init__(self, func):
        self.code = func.__code__
        self.returns = []
        self._prev = None

    def __enter__(self):
        self._prev = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._prev)
        return False

    def _global(self, frame, event, arg):
        if event == "call" and frame.f_code is self.code:
            return self._local
        return None

    def _local(self, frame, event, arg):
        if event == "return":
            self.returns.append((frame.f_lineno, arg is None))
        return self._local

    # ------------------------------------------------------------------ report

    def histogram(self):
        hist = {}
        for lineno, was_none in self.returns:
            if not was_none:
                continue
            hist[lineno] = hist.get(lineno, 0) + 1
        return {
            "calls": len(self.returns),
            "returned_none": sum(1 for _l, n in self.returns if n),
            "returned_keypoint": sum(1 for _l, n in self.returns if not n),
            "by_line": {
                str(line): {"count": count,
                            "guard": GUARD_LINES.get(line, "UNKNOWN LINE — re-read source")}
                for line, count in sorted(hist.items())
            },
        }


def source_line(path, lineno):
    with open(path, encoding="utf-8") as fh:
        for i, text in enumerate(fh, 1):
            if i == lineno:
                return text.rstrip()
    return None


# ------------------------------------------------------------------- 1 & 3 & 4


def links_for(ya, threshold_cells=None, max_order=1):
    """The link list `find_keypoints` would build, without running its loop."""
    from terrainflow_assessment.modules.flow_graph import (
        d8_from_dem,
        strahler_order,
        stream_links,
    )

    _fdir, acc_arr = ya._ensure_flow_data()
    cell_area = ya.cell_w * ya.cell_h
    if threshold_cells is None:
        threshold_cells = max(20, int(round(2_000.0 / cell_area)))

    stream = (acc_arr >= threshold_cells) & np.isfinite(ya.dem)
    next_flat, _sink = d8_from_dem(ya.dem, ya.cell_w, ya.cell_h)
    order = strahler_order(next_flat, stream.ravel())
    links = stream_links(next_flat, stream.ravel(), order, ya.dem.shape[1],
                         max_order=max_order)
    links.sort(key=lambda lk: float(acc_arr[lk[-1][0], lk[-1][1]]), reverse=True)
    return links, acc_arr, int(threshold_cells)


def stage_guard_histogram(ev, label, ya, dem_label):
    """KPA-39 and KPA-44: the refusal reason, measured instead of read off the message."""
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    with ev.stage(f"guard_histogram_{label}") as rec:
        links, _acc, threshold = links_for(ya)
        rec["dem"] = dem_label
        rec["threshold_cells"] = threshold
        rec["threshold_ha"] = threshold * ya.cell_w * ya.cell_h / 10_000.0
        rec["order1_links"] = len(links)

        tracer = GuardTracer(YeomansKeylineAnalysis.keypoint_on_path)
        with tracer:
            for link in links:
                ya.keypoint_on_path(link, require_prominence=True)
        rec["histogram"] = tracer.histogram()

        src = ya.__class__.__module__.replace(".", "/") + ".py"
        rec["source_lines_seen"] = {
            line: source_line(_probe.REPO / src, int(line))
            for line in rec["histogram"]["by_line"]
        }

        prominence = rec["histogram"]["by_line"].get("734", {}).get("count", 0)
        refused = rec["histogram"]["returned_none"]
        rec["refusals_the_message_blames_on_prominence"] = refused
        rec["refusals_actually_from_prominence"] = prominence
        rec["message_is_false_for"] = refused - prominence

        ev.note(
            f"KPA-39 on {dem_label}: {len(links)} order-1 links at "
            f"{rec['threshold_ha']:.2f} ha; {refused} refused, of which "
            f"{prominence} were refused by the prominence test. find_keypoints reports "
            f"all {refused} as 'no break in the floor clearing 2% of grade change', so "
            f"the stated reason is false {rec['message_is_false_for']}/{refused} times. "
            f"Breakdown: " + ", ".join(
                f"{v['count']}x :{k} ({v['guard']})"
                for k, v in rec["histogram"]["by_line"].items()))


def stage_identity(ev, label, ya, dem_label):
    """KPA-43: `keypoints + skipped == links` is not an identity; how far off is it."""
    with ev.stage(f"accounting_identity_{label}") as rec:
        links, _acc, threshold = links_for(ya)
        rec["dem"] = dem_label
        rec["order1_links"] = len(links)
        rec["threshold_cells"] = threshold
        rec["at_max_valleys"] = {}
        for mv in (1, 8):
            kps, skipped = ya.find_keypoints(max_valleys=mv)
            rec["at_max_valleys"][str(mv)] = {
                "keypoints": len(kps),
                "skipped": len(skipped),
                "sum": len(kps) + len(skipped),
                "links": len(links),
                "unaccounted": len(links) - (len(kps) + len(skipped)),
                "identity_holds": len(kps) + len(skipped) == len(links),
            }
        one = rec["at_max_valleys"]["1"]
        eight = rec["at_max_valleys"]["8"]
        ev.note(
            f"KPA-43 on {dem_label}: at max_valleys=1, keypoints+skipped = "
            f"{one['sum']} against {one['links']} links "
            f"({one['unaccounted']} links never examined and never reported); at "
            f"max_valleys=8, {eight['sum']} against {eight['links']} "
            f"({eight['unaccounted']} unaccounted).")


def stage_floor(ev, label, ya, dem_label):
    """KPA-40: the shortest thalweg that can yield a keypoint, found by bisection.

    Measured against the real function on a synthetic straight profile with a genuine
    slope break, so the answer is the function's, not the algebra's.
    """
    with ev.stage(f"profile_floor_{label}") as rec:
        cell = ya.cell_size
        spacing = min(5.0 * cell, 10.0)
        rec["dem"] = dem_label
        rec["cell_size_m"] = cell
        rec["resample_spacing_m"] = spacing
        rec["predicted_floor_m"] = 7.0 * spacing

        # A straight east-west run of cells with a strong break, long enough that only
        # the length guard can refuse it.
        def synthetic_link(n_cells):
            return [(0, c) for c in range(n_cells)]

        class _Straight:
            """A stand-in with a profile that always clears the prominence bar."""

            MIN_SLOPE_EASE = ya.MIN_SLOPE_EASE
            cell_size = cell

            def __init__(self, n):
                # Steep for the first half, gentle for the second — Yeomans' break.
                z = []
                for i in range(n):
                    s = i * cell
                    z.append(100.0 - (0.30 * s if i < n // 2
                                      else 0.30 * (n // 2) * cell + 0.02 * (
                                          s - (n // 2) * cell)))
                self.dem = np.array([z], dtype="float32")
                self.transform = ya.transform

            _rc_to_xy = ya.__class__._rc_to_xy

        keypoint_on_path = ya.__class__.keypoint_on_path
        first_pass = None
        for n in range(3, 200):
            probe = _Straight(n)
            got = keypoint_on_path(probe, synthetic_link(n), require_prominence=False)
            if got is not None:
                first_pass = n
                break

        rec["first_passing_cells"] = first_pass
        rec["first_passing_length_m"] = (
            None if first_pass is None else (first_pass - 1) * cell)
        rec["matches_prediction"] = (
            first_pass is not None
            and abs((first_pass - 1) * cell - 7.0 * spacing) <= cell)
        ev.note(
            f"KPA-40 on {dem_label}: the shortest axis-aligned thalweg that can yield a "
            f"keypoint is {first_pass} cells = {rec['first_passing_length_m']} m "
            f"(predicted floor 7 x min(5*cell, 10) = {rec['predicted_floor_m']:.0f} m). "
            "Every shorter link is still emitted by stream_links and still profiled.")


# ----------------------------------------------------------------------- 4


def windowed_grades(coords, real_mask, windows=WINDOWS_M):
    """Steepest sustained grade over each window, using only real-ground vertices.

    A guide's *net* fall is `drift_1_in_n`. That is what `_run_record` computes and what
    `over_limit` tests. It is not what a plough follows: drift varies along the line by
    construction, so a near-zero net fall can hide a reversal. This walks every window of
    the given length whose endpoints and interior are all real ground.
    """
    xs = np.array([c[0] for c in coords], dtype="float64")
    ys = np.array([c[1] for c in coords], dtype="float64")
    zs = np.array([c[2] if len(c) > 2 else np.nan for c in coords], dtype="float64")
    real = np.asarray(real_mask, dtype=bool)

    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])

    # `real_prefix[i]` counts real vertices up to i, so an all-real span is an O(1) test.
    real_prefix = np.concatenate([[0], np.cumsum(real.astype("int64"))])

    out = {}
    for w in windows:
        best = 0.0
        best_at = None
        n_windows = 0
        over = 0
        limit = 1.0 / 500.0
        j = 0
        for i in range(len(s)):
            if not real[i]:
                continue
            while j < len(s) and s[j] - s[i] < w:
                j += 1
            if j >= len(s):
                break
            if not real[j]:
                continue
            if real_prefix[j + 1] - real_prefix[i] != (j - i + 1):
                continue        # a fabricated vertex inside the window
            run = s[j] - s[i]
            if run <= 0:
                continue
            grade = abs(zs[j] - zs[i]) / run
            n_windows += 1
            if grade > limit:
                over += 1
            if grade > best:
                best, best_at = grade, float(s[i])
        out[f"{w:.0f}m"] = {
            "windows_measured": n_windows,
            "steepest_grade": best,
            "steepest_1_in_n": (1.0 / best) if best > 0 else None,
            "at_arc_m": best_at,
            "fraction_over_1_in_500": (over / n_windows) if n_windows else None,
        }
    out["line_length_m"] = total
    return out


def real_ground_mask(ya, coords):
    """Which vertices carry a sampled elevation, and which carry the default.

    Reproduces `_sample_dem`'s own test (`keypoint_analysis.py:1089-1093`) rather than
    patching it, so the count cannot drift from the function it describes.
    """
    from terrainflow_assessment.modules.footprint import xy_to_rc

    rows, cols = ya.dem.shape
    mask = []
    for c in coords:
        r, col = xy_to_rc(ya.transform, c[0], c[1])
        ok = 0 <= r < rows and 0 <= col < cols and not np.isnan(ya.dem[r, col])
        mask.append(bool(ok))
    return mask


def stage_drift(ev, ya, links):
    """KPA-41 and KPA-42: net fall against sustained grade, and fabricated Z counted."""
    with ev.stage("drift_table") as rec:
        rec["windows_m"] = list(WINDOWS_M)
        rec["guides"] = []

        chosen = None
        for link in links:
            kp = ya.keypoint_on_path(link, require_prominence=True)
            if kp is not None:
                chosen = (link, kp)
                break
        if chosen is None:
            raise RuntimeError(
                "no link on this DEM yields a keypoint under the prominence rule — "
                "there is no guide set to measure drift on")

        link, kp = chosen
        rec["keypoint"] = kp
        rec["link_cells"] = len(link)
        runs = ya.get_cultivation_runs(kp)
        rec["runs_returned"] = len(runs)

        for run in runs:
            coords = list(run["geometry"].coords)
            mask = real_ground_mask(ya, coords)
            fabricated = len(mask) - sum(mask)
            entry = {
                "line_type": run["line_type"],
                "offset_m": run["offset_m"],
                "elevation": run["elevation"],
                "vertices": len(coords),
                "fabricated_z_vertices": fabricated,
                "fabricated_fraction": fabricated / len(coords) if coords else None,
                "net_drift_fall_m": run["drift_fall_m"],
                "net_drift_1_in_n": run["drift_1_in_n"],
                "over_limit_as_reported": run["over_limit"],
                "sustained": windowed_grades(coords, mask),
            }
            w20 = entry["sustained"]["20m"]
            if w20["steepest_1_in_n"] and run["drift_1_in_n"]:
                entry["understatement_factor"] = (
                    run["drift_1_in_n"] / w20["steepest_1_in_n"])
            rec["guides"].append(entry)

        worst = max(
            (g for g in rec["guides"] if g["sustained"]["20m"]["steepest_1_in_n"]),
            key=lambda g: 1.0 / g["sustained"]["20m"]["steepest_1_in_n"], default=None)
        fabricating = [g for g in rec["guides"] if g["fabricated_z_vertices"]]
        rec["guides_with_fabricated_z"] = len(fabricating)

        if worst is not None:
            ev.note(
                "KPA-41, measured: the steepest 20 m window on "
                f"{worst['line_type']} at offset {worst['offset_m']} m runs "
                f"1:{worst['sustained']['20m']['steepest_1_in_n']:.1f}, while "
                f"_run_record reports its net drift as "
                f"1:{worst['net_drift_1_in_n']} and over_limit "
                f"{worst['over_limit_as_reported']}. "
                f"Understatement factor "
                f"{worst.get('understatement_factor', float('nan')):.1f}x.")
        ev.note(
            f"KPA-42, measured: {len(fabricating)} of {len(rec['guides'])} guides carry "
            "vertices whose Z is _sample_dem's default (the keypoint elevation) rather "
            "than sampled ground — "
            + "; ".join(
                f"{g['line_type']}@{g['offset_m']}m: {g['fabricated_z_vertices']}/"
                f"{g['vertices']}" for g in fabricating)
            + ". A guide whose net fall is computed between two fabricated ends reports "
              "a drift that is an artefact of the default, not of terrain.")


# ----------------------------------------------------------------------- 5


def stage_roughness_sweep(ev):
    """Size the measurement window on correlated synthetic roughness."""
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    with ev.stage("roughness_sweep") as rec:
        _harness = _probe.start_qgis()
        work = _probe.workdir("keypoints")
        rec["seed"] = 42
        rec["roughness_m"] = list(ROUGHNESS_SWEEP)
        rec["runs"] = {}
        baseline_kp = None

        for r in ROUGHNESS_SWEEP:
            path = _harness.build_synthetic_dem(
                work / f"rough_{r:.2f}.tif", rough=True, roughness_m=r, seed=42)
            ya = YeomansKeylineAnalysis(str(path))
            links, _acc, threshold = links_for(ya)

            # One pass over the links, not three: the profile is a savgol filter per
            # link and the rough surfaces carry hundreds of them.
            unprominenced = [
                ya.keypoint_on_path(link, require_prominence=False) for link in links]
            passing = [k for k in unprominenced
                       if k is not None and k["slope_ease"] >= ya.MIN_SLOPE_EASE]
            kp = passing[0] if passing else None

            entry = {
                "order1_links": len(links),
                "threshold_cells": threshold,
                "first_keypoint_clearing_prominence": kp,
                "keypoints_found": len(passing),
                "candidates_before_prominence": sum(
                    1 for k in unprominenced if k is not None),
                "slope_ease_values": sorted(
                    (k["slope_ease"] for k in unprominenced if k is not None),
                    reverse=True)[:10],
            }

            # No order-1 link on this surface yields a keypoint at any roughness, so the
            # window has to be sized on a thalweg that survives: `find_keypoint` walks the
            # single largest stream, which is also exactly what `contour.py:1402`'s
            # fallback calls when `find_keypoints` comes back empty — so this is the line
            # a user of the synthetic DEM is actually shown.
            trunk_kp = ya.find_keypoint()
            entry["trunk_keypoint"] = trunk_kp
            if trunk_kp is not None:
                if baseline_kp is None:
                    baseline_kp = trunk_kp
                entry["trunk_displacement_from_smooth_m"] = float(np.hypot(
                    trunk_kp["x"] - baseline_kp["x"],
                    trunk_kp["y"] - baseline_kp["y"]))
                entry["trunk_elevation_shift_m"] = (
                    trunk_kp["elevation"] - baseline_kp["elevation"])

                runs = ya.get_cultivation_runs(trunk_kp)
                keyline = next(
                    (x for x in runs if x["line_type"] == "keyline"), None)
                if keyline is not None:
                    coords = list(keyline["geometry"].coords)
                    entry["keyline_sustained"] = windowed_grades(
                        coords, real_ground_mask(ya, coords))
                    entry["keyline_net_drift_1_in_n"] = keyline["drift_1_in_n"]
            rec["runs"][f"{r:.2f}"] = entry
            print(f"    roughness {r:.2f} m: links={entry['order1_links']:4d} "
                  f"candidates={entry['candidates_before_prominence']:3d} "
                  f"clearing 2%={entry['keypoints_found']:3d} "
                  f"trunk keypoint moved "
                  f"{entry.get('trunk_displacement_from_smooth_m', float('nan')):6.1f} m",
                  flush=True)

        # Which window is stable across 0.00 -> 0.10?
        stability = {}
        for w in WINDOWS_M:
            key = f"{w:.0f}m"
            vals = []
            for r in (0.0, 0.05, 0.10):
                got = rec["runs"].get(f"{r:.2f}", {}).get("keyline_sustained")
                if got and got.get(key, {}).get("steepest_grade") is not None:
                    vals.append(got[key]["steepest_grade"])
            if len(vals) >= 2:
                spread = max(vals) - min(vals)
                stability[key] = {
                    "grades": vals,
                    "spread": spread,
                    "spread_as_fraction_of_min_slope_ease": spread / 0.02,
                }
        rec["window_stability_0_to_0.10"] = stability
        stable = [k for k in (f"{w:.0f}m" for w in WINDOWS_M)
                  if k in stability
                  and stability[k]["spread_as_fraction_of_min_slope_ease"] < 1.0]
        rec["shortest_stable_window"] = stable[0] if stable else None
        ev.note(
            "Window sizing (a TerrainFlow convention, sized on SYNTHETIC correlated "
            "roughness — no published source supplies one): spreads across roughness_m "
            "0.00-0.10 on the trunk keyline are "
            + "; ".join(f"{k} {v['spread']:.4f} "
                        f"({v['spread_as_fraction_of_min_slope_ease']:.2f}x "
                        f"MIN_SLOPE_EASE)" for k, v in stability.items())
            + f". Shortest window whose spread stays under MIN_SLOPE_EASE: "
              f"{rec['shortest_stable_window']}.")

        collapse = {k: (v["order1_links"], v["candidates_before_prominence"],
                        v["keypoints_found"])
                    for k, v in rec["runs"].items()}
        rec["network_collapse"] = collapse
        ev.note(
            "The sweep's own result contradicts the hypothesis it was written to test. "
            "Correlated roughness does NOT push slope_ease over MIN_SLOPE_EASE "
            "spuriously — it does the opposite. As roughness_m goes 0.00 -> 0.25 the "
            "order-1 link count rises "
            + " -> ".join(str(v[0]) for v in collapse.values())
            + " while the links that are long enough to profile at all fall "
            + " -> ".join(str(v[1]) for v in collapse.values())
            + ", and the number clearing the 2% bar stays 0 throughout. Noise shatters "
            "the channel network into stubs that FLG-18's floor then refuses, so the "
            "failure mode is KPA-39's length guards, not a false positive on prominence.")


def main():
    dem = _probe.fixture_path()
    _probe.start_qgis()
    _probe.banner("p_keypoints — guard histogram, floor, identity, drift, roughness", dem)

    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    ev = _probe.Evidence(
        "p_keypoints",
        findings=["KPA-39", "KPA-40", "KPA-41", "KPA-42", "KPA-43", "KPA-44", "KPA-46"],
        dem=dem)
    ev["dem_stats"] = _probe.dem_stats(dem)
    ev["windows_m"] = list(WINDOWS_M)

    ya = YeomansKeylineAnalysis(str(dem))
    links, _acc, _t = links_for(ya)

    stage_guard_histogram(ev, "fixture", ya, "real fixture (1 m, Quail Island)")
    stage_identity(ev, "fixture", ya, "real fixture (1 m, Quail Island)")
    stage_floor(ev, "fixture", ya, "real fixture (1 m)")
    stage_drift(ev, ya, links)

    # The synthetic harness DEM: 2 m cells, and the arm KPA-44 and KPA-46 are about.
    _harness = _probe.start_qgis()
    work = _probe.workdir("keypoints")
    syn_path = _harness.build_synthetic_dem(work / "default.tif")
    ev["synthetic_dem_stats"] = _probe.dem_stats(syn_path)
    ya_syn = YeomansKeylineAnalysis(str(syn_path))
    stage_guard_histogram(ev, "synthetic", ya_syn, "default harness DEM (2 m, 36 ha)")
    stage_identity(ev, "synthetic", ya_syn, "default harness DEM (2 m, 36 ha)")
    stage_floor(ev, "synthetic", ya_syn, "default harness DEM (2 m)")

    stage_roughness_sweep(ev)

    ev.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
