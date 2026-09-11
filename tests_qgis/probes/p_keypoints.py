"""p_keypoints — which guard refuses each valley, and how far a guide really drifts.

Step A1 of the defect-documentation campaign. Supports **KPA-39, KPA-40, KPA-41, KPA-42,
KPA-43, KPA-44, KPA-46**, and sizes the measurement window KPA-41's recommendation needs.

Five measurements:

**1. The guard histogram (KPA-39, KPA-44).** `find_keypoints` used to attribute *every*
`keypoint_on_path` `None` to the prominence test — the skipped message said "no break in
the floor clearing 2% of grade change" whatever actually happened. Since KPA-52 closed the
criterion is `keypoint_on_path_with_reason`, which returns ``(None, reason)`` from four
guards::

    thalweg is None or len(thalweg) < 2 * reach + 1          (too few cells to fit)
    arc_all[-1] <= 0                                          (zero-length path)
    len(arc) < 2 * reach + 1                                  (too little finite ground)
    require_prominence and slope_ease < self.MIN_SLOPE_EASE  (the prominence test)

`sys.settrace` records the line each refusal returned from, so the *reason string* can be
checked against the *line* — two independent routes to the same histogram. Those line
numbers are **derived from the source at run time** and are deliberately not written down
here — see `guard_lines()`. Run on the real fixture at the production threshold **and** on
the default synthetic harness DEM.

**2. The floor, on both grids (KPA-40).** There is no longer a metric floor: the two-slope
fit needs `2 * MIN_REACH_CELLS + 1` cells and nothing else. Measured by bisection against
the real function rather than asserted from the constant, so a floor creeping back in
would show here first.

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

import ast
import inspect
import sys

import _probe

import numpy as np

WINDOWS_M = (10.0, 20.0, 50.0)
ROUGHNESS_SWEEP = (0.0, 0.05, 0.10, 0.25)

#: A readable gloss per guard, keyed by the test expression exactly as `ast.unparse`
#: renders it. A guard whose condition is rewritten loses its gloss and falls back to the
#: source text, which is honest; a guard that *moves* keeps it, which is the point.
GUARD_GLOSS = {
    "thalweg is None or len(thalweg) < 2 * reach + 1": "no valley, or too few cells to fit",
    "arc_all[-1] <= 0": "zero-length path",
    "len(arc) < 2 * reach + 1": "too little finite ground",
    "require_prominence and slope_ease < self.MIN_SLOPE_EASE": "the prominence test",
}

#: The one guard `find_keypoints` used to claim for every refusal. KPA-39 was the gap
#: between the count at this line and the total, so it is named by its *condition*, not
#: its line.
PROMINENCE_TEST = "require_prominence and slope_ease < self.MIN_SLOPE_EASE"


# --------------------------------------------------------------------- tracing


def guard_lines(func):
    """`{lineno: test_source}` for every `return None` in *func*, read off the AST.

    This used to be a hand-written dict of line numbers and it silently rotted. When
    `find_ridgelines` was rewritten on 2026-09-11 (`ac2966b`) every guard in
    `keypoint_on_path` shifted down 20 lines; the tracer still counted 78 / 48 / 26
    correctly but attributed all of them to "UNKNOWN LINE", and the two derived figures —
    `refusals_actually_from_prominence` and `message_is_false_for` — inverted, reporting
    that the message was false 28 times out of 28 when it is false twice. A probe whose
    own evidence file can be wrong about the finding it exists to support is worse than no
    probe, so the mapping is now computed from the source every run.

    Nested `if`s are walked so the innermost enclosing test is the one reported. A
    `return None` with no enclosing `if` (there is none today) is labelled as such rather
    than dropped. Since KPA-52 closed the guards return ``None, <reason>`` — a tuple whose
    first element is ``None`` — and those count as refusals too.
    """
    src, start = inspect.getsourcelines(func)
    # `getsourcelines` hands back the method still indented inside its class, which
    # `ast.parse` rejects. Dedent by the `def`'s own indent; `start` puts the line numbers
    # back into module space afterwards.
    indent = len(src[0]) - len(src[0].lstrip())
    tree = ast.parse("".join(line[indent:] if line.strip() else line for line in src))
    found = {}

    def _is_none(node):
        return isinstance(node, ast.Constant) and node.value is None

    def walk(node, enclosing):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, child)
            elif isinstance(child, ast.Return):
                value = child.value
                refusal = (value is None or _is_none(value)
                           or (isinstance(value, ast.Tuple) and value.elts
                               and _is_none(value.elts[0])))
                if refusal:
                    test = (ast.unparse(enclosing.test) if enclosing is not None
                            else "<unconditional>")
                    found[child.lineno + start - 1] = test
            else:
                walk(child, enclosing)

    walk(tree, None)
    return found


class GuardTracer:
    """Attribute each `keypoint_on_path` return to the line it returned from.

    A line tracer rather than a wrapper: the guards are five separate `return None`
    statements inside one function, and nothing short of `sys.settrace` can tell them
    apart from outside. Active only for the duration of the `with` block, because tracing
    every line of pysheds would take minutes.
    """

    def __init__(self, func):
        self.code = func.__code__
        self.guards = guard_lines(func)
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
            refused = arg is None or (isinstance(arg, tuple) and arg[0] is None)
            self.returns.append((frame.f_lineno, refused))
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
                str(line): {
                    "count": count,
                    "test": self.guards.get(line, "NOT A GUARD — re-read source"),
                    "guard": GUARD_GLOSS.get(
                        self.guards.get(line), self.guards.get(line, "unknown")),
                }
                for line, count in sorted(hist.items())
            },
        }

    def count_at(self, test):
        """How many `None`s came from the guard whose condition is *test*.

        By condition rather than by line, so the answer survives the file moving under it.
        """
        lines = [line for line, src in self.guards.items() if src == test]
        return sum(count for line, count in
                   ((line, sum(1 for lo, n in self.returns if n and lo == line))
                    for line in lines))


def source_line(path, lineno):
    with open(path, encoding="utf-8") as fh:
        for i, text in enumerate(fh, 1):
            if i == lineno:
                return text.rstrip()
    return None


# ------------------------------------------------------------------- 1 & 3 & 4


def links_for(ya, threshold_cells=None, max_order=1):
    """The valley list `find_keypoints` walks — from `primary_valleys` itself, so this
    cannot drift from production the way a hand-built copy of the construction did."""
    _fdir, acc_arr = ya._ensure_flow_data()
    cell_area = ya.cell_w * ya.cell_h
    if threshold_cells is None:
        threshold_cells = max(20, int(round(2_000.0 / cell_area)))
    valleys = ya.primary_valleys(stream_threshold_cells=threshold_cells,
                                 max_order=max_order)
    return [v["cells"] for v in valleys], acc_arr, int(threshold_cells)


def stage_guard_histogram(ev, label, ya, dem_label):
    """KPA-39 and KPA-44: the refusal reason, measured instead of read off the message.

    Two routes to one histogram: the line each refusal returned from (settrace) and the
    reason string it returned. Since KPA-52 closed they must agree; the note says whether
    they do.
    """
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    with ev.stage(f"guard_histogram_{label}") as rec:
        links, _acc, threshold = links_for(ya)
        rec["dem"] = dem_label
        rec["threshold_cells"] = threshold
        rec["threshold_ha"] = threshold * ya.cell_w * ya.cell_h / 10_000.0
        rec["order1_links"] = len(links)

        reasons = {}
        tracer = GuardTracer(YeomansKeylineAnalysis.keypoint_on_path_with_reason)
        with tracer:
            for link in links:
                kp, reason = ya.keypoint_on_path_with_reason(link, require_prominence=True)
                if kp is None:
                    key = reason.split(" (")[0]
                    reasons[key] = reasons.get(key, 0) + 1
        rec["histogram"] = tracer.histogram()
        rec["reasons_returned"] = reasons

        src = ya.__class__.__module__.replace(".", "/") + ".py"
        rec["source_lines_seen"] = {
            line: source_line(_probe.REPO / src, int(line))
            for line in rec["histogram"]["by_line"]
        }

        rec["guard_lines_derived"] = {str(k): v for k, v in
                                      sorted(tracer.guards.items())}
        prominence = tracer.count_at(PROMINENCE_TEST)
        refused = rec["histogram"]["returned_none"]
        by_reason = sum(n for k, n in reasons.items() if "grade change" in k)
        rec["refusals_from_prominence_by_line"] = prominence
        rec["refusals_from_prominence_by_reason_string"] = by_reason
        rec["line_and_reason_agree"] = prominence == by_reason and sum(
            reasons.values()) == refused

        ev.note(
            f"KPA-39 on {dem_label}: {len(links)} primary valleys at "
            f"{rec['threshold_ha']:.2f} ha; {refused} refused, of which "
            f"{prominence} by the prominence test (by line) and {by_reason} (by the "
            f"reason string) — {'agree' if rec['line_and_reason_agree'] else 'DISAGREE'}. "
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
            f"{one['sum']} against {one['links']} valleys "
            f"({one['unaccounted']} unaccounted — since KPA-52 closed, a valley the cap "
            "stops short of is reported as 'not examined', so this should be 0); at "
            f"max_valleys=8, {eight['sum']} against {eight['links']} "
            f"({eight['unaccounted']} unaccounted).")


def stage_floor(ev, label, ya, dem_label):
    """KPA-40: the shortest valley that can yield a keypoint, found by bisection.

    Measured against the real function on a synthetic straight profile with a genuine
    slope break, so the answer is the function's, not the constant's. Since KPA-52
    closed the prediction is `2 * MIN_REACH_CELLS + 1` cells, on any grid.
    """
    with ev.stage(f"profile_floor_{label}") as rec:
        cell = ya.cell_size
        rec["dem"] = dem_label
        rec["cell_size_m"] = cell
        rec["predicted_floor_cells"] = 2 * ya.MIN_REACH_CELLS + 1
        rec["predicted_floor_m"] = (rec["predicted_floor_cells"] - 1) * cell

        # A straight east-west run of cells with a strong break, long enough that only
        # the length guard can refuse it.
        def synthetic_link(n_cells):
            return [(0, c) for c in range(n_cells)]

        class _Straight:
            """A stand-in with a profile that always clears the prominence bar."""

            MIN_SLOPE_EASE = ya.MIN_SLOPE_EASE
            MIN_REACH_CELLS = ya.MIN_REACH_CELLS
            REFUSED_TOO_FEW_CELLS = ya.REFUSED_TOO_FEW_CELLS
            REFUSED_ZERO_LENGTH = ya.REFUSED_ZERO_LENGTH
            REFUSED_TOO_LITTLE_GROUND = ya.REFUSED_TOO_LITTLE_GROUND
            REFUSED_NO_BREAK = ya.REFUSED_NO_BREAK
            keypoint_on_path_with_reason = ya.__class__.keypoint_on_path_with_reason
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
        rec["matches_prediction"] = first_pass == rec["predicted_floor_cells"]
        ev.note(
            f"KPA-40 on {dem_label}: the shortest axis-aligned valley that can yield a "
            f"keypoint is {first_pass} cells = {rec['first_passing_length_m']} m "
            f"(predicted 2 * MIN_REACH_CELLS + 1 = {rec['predicted_floor_cells']} cells). "
            "There is no metric floor any more: once a valley starts at its divide it "
            "is tens of cells long before it is a channel at all.")


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

            # One pass over the valleys, not three: each is a two-slope fit and the
            # rough surfaces carry hundreds of them.
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
            "Roughness sweep, roughness_m 0.00 -> 0.25: primary valleys "
            + " -> ".join(str(v[0]) for v in collapse.values())
            + "; valleys the fit could be run on "
            + " -> ".join(str(v[1]) for v in collapse.values())
            + "; valleys clearing the 2% bar "
            + " -> ".join(str(v[2]) for v in collapse.values())
            + ". (Before KPA-52 closed, noise shattered the channel network into stubs "
            "that the 35 m floor refused, so nothing cleared the bar at any roughness; "
            "the floor is gone and every valley is fitted, so the third series is now "
            "the one that says whether noise fabricates keypoints.)")


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
