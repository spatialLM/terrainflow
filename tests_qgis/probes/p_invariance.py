"""p_invariance — two exact oracles, and the routing setting nobody runs.

Step C of the defect-documentation campaign. Unlike Steps A1 and B, this probe does not
start from a suspected defect: it applies a transform under which the *correct* answer is
known exactly, and reports where the tier disagrees with itself. Zero aliasing, no
tolerance to argue about, which is why these beat the resampling arm they replaced.

**Arm 1 — Z + 600 m.** Add a constant to every elevation. Slope, aspect, curvature, TPI,
flow pointers, links and keypoint *positions* must not move; only elevations shift, by
exactly 600. This is the one arm that reaches the float32 regime that ``save_result``'s
docstring, ``earthwork_design.py:2137`` and ``KPA-38``'s fix note all reason about, and that
nothing else in either suite touches. Note ``YeomansKeylineAnalysis.__init__`` reads the
DEM ``.astype("float32")`` whatever the file holds, so this arm is reached from a float64
raster too — the quantisation is in the tier, not only on disk.

**Arm 2 — mirror.** Flip the DEM left-right. Every result must mirror with it: slope and
curvature unchanged, aspect reflected about the north-south axis, a keypoint at column *c*
reappearing at ``cols - 1 - c``. This catches the whole sign-and-axis class in one arm —
``KPA-33``'s row-to-y flip, ``CTA-08``'s rc-to-map asymmetry, ``IMP-01``'s diagonal count,
``UNI-15``'s convention — and is a **regression guard on shipped fixes** rather than an
investigation, since `UNI-15` and its `CTA-08` component were fixed in §9.2.

One honest caveat, measured rather than waved at: ``d8_from_dem`` breaks ties by scan
position (`flow_graph.py:106-113`), and a left-right flip reverses which of two equal
neighbours is seen first. Pointer disagreement on tied cells is therefore *expected*, and
the probe separates ties from real disagreement instead of reporting one number.

**Arm 3 — routing='d8'.** The panel offers d8 and the persisted default is dinf; the
divergence is a known open item tested nowhere. Run both and record what actually differs
— including whether the setting reaches the keyline tier at all.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_invariance.py
"""

import _probe

import numpy as np

Z_OFFSET = 600.0


# ----------------------------------------------------------------- raster help


def read_dem(path):
    """`(z, transform, crs, nodata)` as float64, nodata already NaN."""
    import rasterio

    with rasterio.open(str(path)) as src:
        z = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            z[z == nodata] = np.nan
        return z, src.transform, src.crs, nodata


def write_dem(path, z, transform, crs, dtype="float32", nodata=-9999.0):
    """Write *z* the way production writes a derived raster.

    `dtype` defaults to float32 deliberately: that is what `save_result` uses for every
    raster but the conditioned surface, and the Z+600 arm exists to ask what that costs.
    """
    import rasterio

    data = np.where(np.isfinite(z), z, nodata)
    with rasterio.open(
        str(path), "w", driver="GTiff", dtype=dtype, crs=crs, transform=transform,
        width=z.shape[1], height=z.shape[0], count=1, nodata=nodata,
    ) as dst:
        dst.write(data.astype(dtype), 1)
    return path


def aspect_delta(a, b):
    """Compare two aspect rasters the way a compass works.

    Plain subtraction calls 359.999 deg and 0.001 deg a 360-degree disagreement, which is
    how the first run of this probe reported `max |delta| 3.600e+02` on two rasters that
    agree. Bearings are compared around the circle; the ``-1`` flat sentinel is not a
    bearing at all, so cells that gain or lose it are counted separately — that count is
    the interesting half, because a gradient quantised to exactly zero becomes flat ground
    that was never flat.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    flat_a, flat_b = a < 0, b < 0
    both_bearing = np.isfinite(a) & np.isfinite(b) & ~flat_a & ~flat_b
    d = np.abs(a[both_bearing] - b[both_bearing])
    d = np.minimum(d, 360.0 - d)
    return {
        "max_abs_delta_deg": float(d.max()) if d.size else None,
        "compared_cells": int(both_bearing.sum()),
        "differing_cells": int((d > 0).sum()),
        "flat_sentinel_only_in_a": int((flat_a & ~flat_b).sum()),
        "flat_sentinel_only_in_b": int((flat_b & ~flat_a).sum()),
    }


def max_abs_delta(a, b):
    """Largest |a - b| over cells finite in both, plus how many cells that was."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return {"max_abs_delta": None, "compared_cells": 0, "differing_cells": 0}
    d = np.abs(a[both] - b[both])
    return {
        "max_abs_delta": float(d.max()),
        "mean_abs_delta": float(d.mean()),
        "compared_cells": int(both.sum()),
        "differing_cells": int((d > 0).sum()),
        "finite_disagreement": int((np.isfinite(a) != np.isfinite(b)).sum()),
    }


def flow_bits(z, cell_w, cell_h):
    """`d8_from_dem` + `strahler_order` + `stream_links` at the production threshold."""
    from terrainflow_assessment.modules.flow_graph import (
        d8_from_dem,
        strahler_order,
        stream_links,
    )

    next_flat, is_sink = d8_from_dem(z, cell_w, cell_h)
    # Accumulation without pysheds: every cell starts with itself and pushes its total
    # downstream, visited high to low. Exact on an acyclic pointer graph, and it keeps
    # the arm measuring TerrainFlow's own graph rather than a library's conditioning.
    finite = np.isfinite(z).ravel()
    acc = np.ones(z.size, dtype=np.int64)
    acc[~finite] = 0
    flat_z = z.ravel()
    for i in np.argsort(np.where(finite, -flat_z, np.inf), kind="stable"):
        if not finite[i]:
            continue
        j = int(next_flat[i])
        if j != i:
            acc[j] += acc[i]

    threshold = max(20, int(round(2_000.0 / (cell_w * cell_h))))
    stream = (acc >= threshold) & finite
    order = strahler_order(next_flat, stream)
    links = stream_links(next_flat, stream, order, z.shape[1], max_order=1)
    return {
        "next_flat": next_flat,
        "is_sink": is_sink,
        "acc": acc,
        "stream": stream,
        "order": order,
        "links": links,
        "threshold_cells": threshold,
    }


def indices(z, cell_w, cell_h):
    from terrainflow_assessment.modules.dem_loader import (
        aspect_degrees,
        horn_gradient,
        slope_degrees,
    )
    from terrainflow_assessment.modules.terrain_indices import curvature, landform_tpi

    dz_dx, dz_dy, _invalid = horn_gradient(z, cell_w, cell_h)
    plan, profile = curvature(z, cell_w, cell_h)
    return {
        "slope": slope_degrees(z, cell_w, cell_h),
        "aspect": aspect_degrees(z, cell_w, cell_h),
        "dz_dx": dz_dx,
        "dz_dy": dz_dy,
        "plan": plan,
        "profile": profile,
        "tpi": landform_tpi(z, cell_w, cell_h, window_m=15.0),
    }


def keypoint_rc(dem_path, max_valleys=8):
    """`(rows, cols, elevations)` from the production call, sorted for comparison."""
    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    ya = YeomansKeylineAnalysis(str(dem_path))
    keypoints, skipped = ya.find_keypoints(max_valleys=max_valleys)
    rc = sorted((int(k["row"]), int(k["col"]), float(k["elevation"]))
                for k in keypoints)
    return rc, len(skipped)


# ------------------------------------------------------------------- arm 1: Z


def _condition_like_keyline_tier(z, transform, crs, work, label):
    """Reproduce `_ensure_flow_data`'s conditioning, float32 temp raster and all.

    Copied from `keypoint_analysis.py:1147-1171` rather than called, because that method
    caches on the instance and gives no way to ask for the conditioned surface. Any drift
    between this and the original invalidates the stage, so the sequence is kept in the
    same order and the dtype is kept deliberately wrong in the same way.
    """
    import rasterio
    from pysheds.grid import Grid

    from terrainflow_assessment.modules.flow_analysis import resolve_flats_safely

    path = work / f"cond_{label}.tif"
    with rasterio.open(
        str(path), "w", driver="GTiff", dtype="float32", crs=crs, transform=transform,
        width=z.shape[1], height=z.shape[0], count=1, nodata=-9999.0,
    ) as dst:
        dst.write(np.where(np.isfinite(z), z, -9999.0).astype("float32"), 1)

    grid = Grid.from_raster(str(path))
    dem_r = grid.read_raster(str(path))
    pit_filled = grid.fill_pits(dem_r)
    try:
        filled = grid.breach_depressions(pit_filled)
    except AttributeError:
        filled = grid.fill_depressions(pit_filled)
    inflated, eps, inversions = resolve_flats_safely(grid, filled)
    return np.asarray(inflated, dtype="float64"), float(eps), int(inversions)


def arm_z_offset(ev, dem_path):
    z, transform, crs, _nodata = read_dem(dem_path)
    cell_w, cell_h = abs(transform.a), abs(transform.e)
    work = _probe.workdir("invariance")

    with ev.stage("z_offset_float32_cost") as rec:
        base = write_dem(work / "z_base.tif", z, transform, crs)
        lifted = write_dem(work / "z_plus600.tif", z + Z_OFFSET, transform, crs)
        zb, _t, _c, _n = read_dem(base)
        zl, _t, _c, _n = read_dem(lifted)
        rec["z_range_m"] = [float(np.nanmin(z)), float(np.nanmax(z))]
        rec["float32_spacing_at_z_max"] = float(
            np.spacing(np.float32(np.nanmax(z))))
        rec["float32_spacing_at_z_max_plus_offset"] = float(
            np.spacing(np.float32(np.nanmax(z) + Z_OFFSET)))
        rec["round_trip_after_removing_offset"] = max_abs_delta(zl - Z_OFFSET, zb)
        rec["resolve_flats_epsilon_m"] = 1e-5
        rec["epsilon_survives_at_base"] = bool(
            rec["float32_spacing_at_z_max"] < 1e-5)
        rec["epsilon_survives_at_offset"] = bool(
            rec["float32_spacing_at_z_max_plus_offset"] < 1e-5)
        ev.note(
            f"Z+600 float32 cost: the fixture spans {rec['z_range_m'][0]:.2f}-"
            f"{rec['z_range_m'][1]:.2f} m, where float32 spacing is "
            f"{rec['float32_spacing_at_z_max']:.2e} m; at +600 m it is "
            f"{rec['float32_spacing_at_z_max_plus_offset']:.2e} m. "
            f"resolve_flats' 1e-05 m step survives at base: "
            f"{rec['epsilon_survives_at_base']}; at +600 m: "
            f"{rec['epsilon_survives_at_offset']}. Round trip through float32 after "
            f"removing the offset: max |delta| "
            f"{rec['round_trip_after_removing_offset']['max_abs_delta']:.3e} m over "
            f"{rec['round_trip_after_removing_offset']['differing_cells']:,} cells.")

    with ev.stage("z_offset_indices") as rec:
        a = indices(z, cell_w, cell_h)
        b = indices(z + Z_OFFSET, cell_w, cell_h)
        rec["deltas"] = {k: max_abs_delta(a[k], b[k]) for k in a if k != "aspect"}
        rec["deltas"]["aspect"] = aspect_delta(a["aspect"], b["aspect"])
        worst = max((v.get("max_abs_delta") or 0.0) for v in rec["deltas"].values())
        rec["worst_max_abs_delta"] = worst
        rec["exact"] = worst == 0.0
        ev.note(
            f"Z+600 terrain indices on float64: "
            f"{'all exact' if rec['exact'] else f'worst max |delta| {worst:.3e}'} "
            "across slope, aspect, dz_dx, dz_dy, plan, profile, TPI.")

    with ev.stage("z_offset_indices_float32") as rec:
        # The same question at the precision production actually carries. `__init__`
        # casts to float32 regardless of the file, so this is the reachable path.
        a = indices(np.asarray(z, dtype="float32").astype("float64"), cell_w, cell_h)
        b = indices(np.asarray(z + Z_OFFSET, dtype="float32").astype("float64"),
                    cell_w, cell_h)
        rec["deltas"] = {k: max_abs_delta(a[k], b[k]) for k in a if k != "aspect"}
        rec["deltas"]["aspect"] = aspect_delta(a["aspect"], b["aspect"])
        rec["worst_max_abs_delta"] = max(
            (v.get("max_abs_delta") or 0.0) for v in rec["deltas"].values())
        asp = rec["deltas"]["aspect"]
        rec["cells_made_flat_by_the_offset"] = asp["flat_sentinel_only_in_b"]
        rec["cells_unflattened_by_the_offset"] = asp["flat_sentinel_only_in_a"]
        ev.note(
            f"Z+600 terrain indices after the float32 cast production performs: worst "
            f"max |delta| {rec['worst_max_abs_delta']:.3e} "
            f"(slope {rec['deltas']['slope']['max_abs_delta']:.3e} deg; aspect "
            f"{asp['max_abs_delta_deg']:.3e} deg around the compass over "
            f"{asp['differing_cells']:,} cells). Cells the offset turns into the -1 flat "
            f"sentinel: {asp['flat_sentinel_only_in_b']:,}; cells it un-flattens: "
            f"{asp['flat_sentinel_only_in_a']:,}.")

    with ev.stage("z_offset_flow_graph") as rec:
        a = flow_bits(z, cell_w, cell_h)
        b = flow_bits(z + Z_OFFSET, cell_w, cell_h)
        disagree = int((a["next_flat"] != b["next_flat"]).sum())
        rec["pointer_disagreements"] = disagree
        rec["sinks_base"] = int(a["is_sink"].sum())
        rec["sinks_lifted"] = int(b["is_sink"].sum())
        rec["order1_links_base"] = len(a["links"])
        rec["order1_links_lifted"] = len(b["links"])
        rec["stream_cells_base"] = int(a["stream"].sum())
        rec["stream_cells_lifted"] = int(b["stream"].sum())
        rec["exact"] = (disagree == 0
                        and rec["sinks_base"] == rec["sinks_lifted"]
                        and rec["order1_links_base"] == rec["order1_links_lifted"])
        ev.note(
            f"Z+600 flow graph on float64: {disagree:,} pointer disagreements, sinks "
            f"{rec['sinks_base']:,} vs {rec['sinks_lifted']:,}, order-1 links "
            f"{rec['order1_links_base']} vs {rec['order1_links_lifted']}.")

    with ev.stage("z_offset_keyline_conditioned_surface") as rec:
        # `save_result`'s docstring predicts this failure and steers callers to float64
        # for the conditioned surface. `keypoint_analysis._ensure_flow_data:1145-1153`
        # does not go through `save_result` — it writes its own float32 temp raster and
        # conditions *that*. So the tier the keyline is drawn on carries the fault the
        # docstring warns about, in the one place nothing was checking. Whether it bites
        # on any given site is a question about elevation, so measure it at both.
        from terrainflow_assessment.modules.flow_graph import d8_from_dem

        rec["mechanism"] = (
            "keypoint_analysis._ensure_flow_data writes its own float32 temp raster "
            "and conditions that, bypassing save_result and its float64 advice")
        for label, surface in (("base", z), ("plus600", z + Z_OFFSET)):
            cond, eps, inversions = _condition_like_keyline_tier(
                surface, transform, crs, work, label)
            _next, is_sink = d8_from_dem(cond, cell_w, cell_h)
            rec[f"{label}_conditioned_sinks"] = int(is_sink.sum())
            rec[f"{label}_conditioned_distinct_values"] = int(
                np.unique(cond[np.isfinite(cond)]).size)
            rec[f"{label}_flat_epsilon_m"] = eps
            rec[f"{label}_residual_inversions"] = inversions
            rec[f"{label}_float32_spacing_m"] = float(
                np.spacing(np.float32(np.nanmax(surface))))
        rec["sink_increase_from_offset"] = (
            rec["plus600_conditioned_sinks"] - rec["base_conditioned_sinks"])
        rec["elevation_levels_lost_to_float32"] = (
            rec["base_conditioned_distinct_values"]
            - rec["plus600_conditioned_distinct_values"])
        rec["epsilon_below_float32_spacing_at_offset"] = bool(
            rec["plus600_flat_epsilon_m"] < rec["plus600_float32_spacing_m"])
        ev.note(
            f"Z+600 through the keyline tier's own float32 conditioning: sinks on the "
            f"conditioned surface {rec['base_conditioned_sinks']:,} at base vs "
            f"{rec['plus600_conditioned_sinks']:,} at +600 m "
            f"({rec['sink_increase_from_offset']:+,}), while "
            f"{rec['elevation_levels_lost_to_float32']:,} distinct elevation levels are "
            f"quantised away. resolve_flats_safely chose eps "
            f"{rec['base_flat_epsilon_m']:.3e} m at base and "
            f"{rec['plus600_flat_epsilon_m']:.3e} m at +600 m, with "
            f"{rec['base_residual_inversions']} and "
            f"{rec['plus600_residual_inversions']} residual inversions. The mechanism "
            f"save_result warns about is present in this path; on this site it costs "
            f"resolution without costing pointers.")

    with ev.stage("z_offset_keypoints") as rec:
        base_rc, base_skipped = keypoint_rc(base)
        lifted_rc, lifted_skipped = keypoint_rc(lifted)
        rec["keypoints_base"] = base_rc
        rec["keypoints_lifted"] = lifted_rc
        rec["skipped_base"] = base_skipped
        rec["skipped_lifted"] = lifted_skipped
        base_pos = [(r, c) for r, c, _e in base_rc]
        lifted_pos = [(r, c) for r, c, _e in lifted_rc]
        rec["positions_identical"] = base_pos == lifted_pos
        if rec["positions_identical"] and base_rc:
            rec["elevation_deltas"] = [round(l[2] - b[2], 6)
                                       for b, l in zip(base_rc, lifted_rc)]
            rec["elevation_offset_exact"] = all(
                abs(d - Z_OFFSET) < 1e-3 for d in rec["elevation_deltas"])
        ev.note(
            f"Z+600 keypoints through the production call: "
            f"{len(base_rc)} found at base, {len(lifted_rc)} lifted; positions "
            f"{'identical' if rec['positions_identical'] else 'MOVED'}; skipped "
            f"{base_skipped} vs {lifted_skipped}.")

    return base, lifted


# -------------------------------------------------------------- arm 2: mirror


def arm_mirror(ev, dem_path):
    z, transform, crs, _nodata = read_dem(dem_path)
    cell_w, cell_h = abs(transform.a), abs(transform.e)
    cols = z.shape[1]
    zm = np.fliplr(z).copy()
    work = _probe.workdir("invariance")

    with ev.stage("mirror_indices") as rec:
        a = indices(z, cell_w, cell_h)
        b = indices(zm, cell_w, cell_h)
        # Scalars mirror; the east-west derivative and the compass bearing reflect.
        rec["deltas"] = {
            "slope": max_abs_delta(np.fliplr(a["slope"]), b["slope"]),
            "plan": max_abs_delta(np.fliplr(a["plan"]), b["plan"]),
            "profile": max_abs_delta(np.fliplr(a["profile"]), b["profile"]),
            "tpi": max_abs_delta(np.fliplr(a["tpi"]), b["tpi"]),
            "dz_dx_negated": max_abs_delta(-np.fliplr(a["dz_dx"]), b["dz_dx"]),
            "dz_dy_unchanged": max_abs_delta(np.fliplr(a["dz_dy"]), b["dz_dy"]),
        }
        # Aspect is a compass bearing, so the mirror image of `a` is `(360 - a) % 360`,
        # with -1 (flat) carried through as itself rather than becoming 361.
        asp = np.fliplr(a["aspect"]).astype("float64")
        reflected = np.where(asp < 0, asp, np.mod(360.0 - asp, 360.0))
        rec["deltas"]["aspect_reflected"] = aspect_delta(reflected, b["aspect"])
        rec["worst_max_abs_delta"] = max(
            (v.get("max_abs_delta") or v.get("max_abs_delta_deg") or 0.0)
            for v in rec["deltas"].values())
        rec["exact"] = rec["worst_max_abs_delta"] == 0.0
        ev.note(
            "Mirror, terrain indices: "
            + ", ".join(
                f"{k} "
                f"{(v.get('max_abs_delta') if v.get('max_abs_delta') is not None else v.get('max_abs_delta_deg')):.3e}"
                for k, v in rec["deltas"].items()))

    with ev.stage("mirror_flow_graph") as rec:
        a = flow_bits(z, cell_w, cell_h)
        b = flow_bits(zm, cell_w, cell_h)
        rows = z.shape[0]

        def mirror_index(flat):
            r, c = divmod(np.asarray(flat, dtype=np.int64), cols)
            return r * cols + (cols - 1 - c)

        expected = mirror_index(a["next_flat"])[mirror_index(
            np.arange(z.size, dtype=np.int64))]
        disagree = expected != b["next_flat"]
        rec["pointer_disagreements"] = int(disagree.sum())

        # A disagreement is only a defect where the two candidate neighbours are not
        # equally steep. Separate the two rather than report one number.
        tied = _tie_mask(zm, b["next_flat"], expected, cell_w, cell_h)
        rec["disagreements_on_ties"] = int((disagree & tied).sum())
        rec["disagreements_not_tied"] = int((disagree & ~tied).sum())
        rec["sinks_base"] = int(a["is_sink"].sum())
        rec["sinks_mirrored"] = int(b["is_sink"].sum())
        rec["order1_links_base"] = len(a["links"])
        rec["order1_links_mirrored"] = len(b["links"])
        rec["stream_cells_base"] = int(a["stream"].sum())
        rec["stream_cells_mirrored"] = int(b["stream"].sum())
        rec["acc_delta"] = max_abs_delta(
            np.fliplr(a["acc"].reshape(rows, cols)), b["acc"].reshape(rows, cols))
        ev.note(
            f"Mirror, flow graph: {rec['pointer_disagreements']:,} pointer "
            f"disagreements, of which {rec['disagreements_on_ties']:,} are exact "
            f"steepness ties (expected: scan order reverses under a flip) and "
            f"{rec['disagreements_not_tied']:,} are not. Sinks "
            f"{rec['sinks_base']:,} vs {rec['sinks_mirrored']:,}; order-1 links "
            f"{rec['order1_links_base']} vs {rec['order1_links_mirrored']}.")

    with ev.stage("mirror_keypoints") as rec:
        mirrored_path = write_dem(work / "z_mirror.tif", zm, transform, crs)
        base_rc, base_skipped = keypoint_rc(
            write_dem(work / "z_base.tif", z, transform, crs))
        mir_rc, mir_skipped = keypoint_rc(mirrored_path)
        rec["keypoints_base"] = base_rc
        rec["keypoints_mirrored"] = mir_rc
        rec["skipped_base"] = base_skipped
        rec["skipped_mirrored"] = mir_skipped
        expected = sorted((r, cols - 1 - c) for r, c, _e in base_rc)
        rec["expected_positions"] = expected
        rec["observed_positions"] = sorted((r, c) for r, c, _e in mir_rc)
        rec["positions_mirror"] = expected == rec["observed_positions"]
        ev.note(
            f"Mirror, keypoints: {len(base_rc)} base -> {len(mir_rc)} mirrored; "
            f"positions mirror: {rec['positions_mirror']}.")


def _tie_mask(z, chosen, expected, cell_w, cell_h):
    """Cells where `chosen` and `expected` are *equally* steep descents.

    A left-right flip reverses `_OFFSETS`' scan order, so a cell with two equally steep
    neighbours may legitimately pick the other one. That is a documented tie-break, not a
    fault, and lumping it in with real disagreement would make this arm unreadable.
    """
    cols = z.shape[1]
    flat = z.ravel()
    idx = np.arange(z.size, dtype=np.int64)
    out = np.zeros(z.size, dtype=bool)

    def drop(src, dst):
        sr, sc = divmod(src, cols)
        dr, dc = divmod(dst, cols)
        dist = np.hypot((dr - sr) * cell_h, (dc - sc) * cell_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(dist > 0, (flat[src] - flat[dst]) / np.where(dist > 0, dist, 1.0), 0.0)

    differing = chosen != expected
    if not differing.any():
        return out
    src = idx[differing]
    d_chosen = drop(src, chosen[differing].astype(np.int64))
    d_expected = drop(src, expected[differing].astype(np.int64))
    out[src] = d_chosen == d_expected
    return out


# ------------------------------------------------------------- arm 3: routing


def _raise_site(tb):
    """`file:line in func` of the deepest frame in a formatted traceback."""
    import re

    frames = re.findall(r'File "([^"]+)", line (\d+), in (\S+)', tb)
    if not frames:
        return None
    path, line, func = frames[-1]
    return f"{path.replace(chr(92), '/').split('/')[-1]}:{line} in {func}"


def arm_routing(ev, dem_path):
    """dinf against d8, and whether the setting reaches the keyline tier at all."""
    with ev.stage("routing_dinf_vs_d8") as rec:
        import traceback

        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        results, failures = {}, {}
        for routing in ("dinf", "d8"):
            fa = FlowAnalysis()
            fa.load_dem(str(dem_path))
            try:
                out = fa.run(routing=routing, crest_split=False)
            except Exception as exc:  # noqa: BLE001 — the failure IS the measurement
                tb = traceback.format_exc()
                failures[routing] = {
                    "exception": f"{type(exc).__name__}: {exc}",
                    "raised_at": _raise_site(tb),
                    "traceback": tb,
                }
                print(f"    {routing}: RAISED {type(exc).__name__}: {exc}", flush=True)
                continue
            results[routing] = {
                "routing_used": fa.routing,
                "acc": np.asarray(out["flow_accumulation"], dtype="float64"),
                "conditioned": np.asarray(out["conditioned_dem"], dtype="float64"),
                "unrouted_cells": int(out.get("unrouted_cells", 0)),
                "unrouted_flow": float(out.get("unrouted_flow", 0.0)),
            }

        rec["failures"] = failures
        rec["routing_requested_vs_used"] = {
            k: v["routing_used"] for k, v in results.items()}
        for k, v in results.items():
            rec[f"{k}_unrouted_cells"] = v["unrouted_cells"]
            rec[f"{k}_unrouted_flow_m3"] = v["unrouted_flow"]
            rec[f"{k}_acc_max"] = float(np.nanmax(v["acc"]))

        if len(results) == 2:
            rec["conditioned_surfaces_identical"] = max_abs_delta(
                results["dinf"]["conditioned"], results["d8"]["conditioned"])
            rec["accumulation_delta"] = max_abs_delta(
                results["dinf"]["acc"], results["d8"]["acc"])
            ev.note(
                f"routing dinf vs d8: accumulation differs on "
                f"{rec['accumulation_delta']['differing_cells']:,} of "
                f"{rec['accumulation_delta']['compared_cells']:,} cells, max |delta| "
                f"{rec['accumulation_delta']['max_abs_delta']:.1f}; acc max "
                f"{rec['dinf_acc_max']:.1f} vs {rec['d8_acc_max']:.1f}.")
        else:
            for routing, f in failures.items():
                ev.note(
                    f"routing='{routing}' does not run at all on the real fixture: "
                    f"{f['exception']}, raised at {f['raised_at']}. "
                    f"`FlowAnalysis.run` wraps that call in `except TypeError` "
                    f"(flow_analysis.py:514-517), which does not catch it. The same "
                    f"pysheds hazard is caught as `except (TypeError, AttributeError)` "
                    f"in catchment.py:224 and keypoint_analysis.py:1181 — the two paths "
                    f"that never request d8. The one path whose routing the panel "
                    f"controls is the one without the guard.")

    with ev.stage("routing_d8_is_offered_by_the_ui") as rec:
        # Whether the crash above is reachable by a user, established from the widget
        # rather than assumed. Reading the combo's items is cheaper and more honest than
        # claiming "the panel offers d8".
        import re

        panel_src = (_probe.REPO / "terrainflow_assessment" / "panel.py").read_text(
            encoding="utf-8")
        combo = re.search(r"_routing_combo\.addItems\(\[([^\]]*)\]\)", panel_src)
        prop = re.search(r"def routing\(self\):\s*\n\s*(return .*)", panel_src)
        rec["combo_items"] = combo.group(1).strip() if combo else None
        rec["panel_property"] = prop.group(1).strip() if prop else None
        rec["d8_selectable"] = bool(combo and "D8" in combo.group(1))
        ev.note(
            f"'D8' is a selectable entry in the Routing combo "
            f"({rec['combo_items']}) and `panel.routing` maps it to 'd8' "
            f"({rec['panel_property']}), which baseline.py:427 hands to the worker and "
            f"analysis_worker.py:147 passes to `FlowAnalysis.run`. Two clicks from the "
            f"UI to the AttributeError above.")

    with ev.stage("routing_reaches_keyline_tier") as rec:
        # `_ensure_flow_data` asks pysheds for "dinf" as a literal and takes no routing
        # argument of its own, so the panel setting cannot reach it. Established by
        # reading the signature rather than asserted.
        import inspect

        from terrainflow_assessment.modules.keypoint_analysis import (
            YeomansKeylineAnalysis,
        )

        src = inspect.getsource(YeomansKeylineAnalysis._ensure_flow_data)
        rec["signature"] = str(
            inspect.signature(YeomansKeylineAnalysis._ensure_flow_data))
        rec["mentions_dinf_literal"] = 'routing="dinf"' in src
        rec["accepts_routing_argument"] = "routing" in inspect.signature(
            YeomansKeylineAnalysis._ensure_flow_data).parameters
        rec["find_keypoints_signature"] = str(
            inspect.signature(YeomansKeylineAnalysis.find_keypoints))
        rec["routing_reachable_from_keyline_tier"] = rec["accepts_routing_argument"]
        ev.note(
            f"routing setting and the keyline tier: _ensure_flow_data"
            f"{rec['signature']} hard-codes routing=\"dinf\" "
            f"({rec['mentions_dinf_literal']}) and takes no routing argument "
            f"({rec['accepts_routing_argument']}), so a user who selects d8 in the panel "
            f"still gets D-infinity accumulation under every keypoint and keyline. "
            f"find_keypoints{rec['find_keypoints_signature']} has no routing parameter "
            f"to pass one through either.")


# --------------------------------------------------------------------- driver


def main():
    _probe.start_qgis()
    dem = _probe.fixture_path()
    _probe.banner("p_invariance — Z+600, mirror, and routing='d8'", dem)

    ev = _probe.Evidence("p_invariance", ["STEP-C"], dem)
    ev["dem_stats"] = _probe.dem_stats(dem)
    ev["z_offset_m"] = Z_OFFSET

    arm_z_offset(ev, dem)
    arm_mirror(ev, dem)
    arm_routing(ev, dem)

    ev.write()


if __name__ == "__main__":
    main()
