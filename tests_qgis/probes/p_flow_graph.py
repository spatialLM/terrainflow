"""p_flow_graph — the conditioned surface, and the population of order-1 links.

Step A1 of the defect-documentation campaign. Supports **KPA-38, KPA-48, FLG-18, FLG-19**
and supplies the production-path numbers that `checks_fixture_regression` pins in Step G.

Four questions, in order:

**1. Is `keypoint_analysis._ensure_flow_data` recomputing a surface it could have been
handed?** (KPA-38, and KPA-48's fix note that it is "one dict key away".) The probe
rebuilds the surface by copying `_ensure_flow_data`'s own sequence — `fill_pits`, then
`breach_depressions` if it exists and `fill_depressions` when it does not, then
`resolve_flats_safely` — and measures max |Δ| against
`FlowAnalysis.run(crest_split=False)["conditioned_dem"]`. A third surface repeats the
sequence **without** `_ensure_flow_data`'s float32 GeoTIFF round-trip, which separates
"the two pipelines differ" from "float32 threw away 3 µm".

**2. Which surface does the production path actually walk?** `find_keypoints` builds its
stream mask from `acc_arr` — D-infinity accumulation over the *conditioned* surface — and
then calls `d8_from_dem(self.dem, ...)` on the **raw** DEM
(`keypoint_analysis.py:787`). The pointer graph and the mask therefore come from
different surfaces. FLG-19 is the consequence, measured here as sinks and pointers that
leave the finite domain, raw versus conditioned.

**3. Do the link filter and the profile floor agree?** (FLG-18.) `stream_links` drops
links under `min_cells=3`. `keypoint_on_path` needs `n_samp - 2*guard >= 3`
(`keypoint_analysis.py:721-722`), and with `spacing = min(5*cell, 10)` and
`win >= 5` that resolves to **`n_samp >= 7`**, i.e. an arc length of at least
`7 * min(5*cell, 10)` m. On a 1 m grid that is 35 m; on the 2 m synthetic DEM, 70 m. Every
link between those two floors is emitted, profiled, and refused.

**4. Which valley is rank 1, and where does its keyline go?** The controller ranks by
contributing area and keeps `max_valleys` of them, so rank 1 is the one a user sees first.

Thresholds. **0.2 ha is the production path** — `contour.py:1397` passes `max_valleys`
alone, so `stream_threshold_cells` takes its `max(20, round(2000 / cell_area))` default
(`keypoint_analysis.py:781`), which is 0.2 ha on the 1 m fixture *and* on the 2 m
synthetic DEM. 0.5 ha and 1.0 ha are run as **sensitivity** and are labelled so in the
evidence file; only the 0.2 ha row may be pinned or quoted as production behaviour.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_flow_graph.py
"""

import os
import sys
import tempfile

import _probe

import numpy as np
import rasterio

# ``spacing = min(5 * cell, 10)`` and ``win >= 5`` force ``guard >= 2``, so the interior
# test ``n_samp - 2 * guard >= 3`` first passes at ``n_samp == 7``. Derived here rather
# than hard-coded so the arithmetic is visible next to the number it produces.
MIN_SAMPLES_FOR_A_KEYPOINT = 7


def profile_floor_m(cell_size):
    """Shortest thalweg, in metres, that `keypoint_on_path` can return a keypoint on."""
    return MIN_SAMPLES_FOR_A_KEYPOINT * min(5.0 * cell_size, 10.0)


def arc_length_m(link, cell_size):
    total = 0.0
    for i in range(1, len(link)):
        dr = link[i][0] - link[i - 1][0]
        dc = link[i][1] - link[i - 1][1]
        total += (dr * dr + dc * dc) ** 0.5 * cell_size
    return total


def describe(values):
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype="float64")
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "total": float(a.sum()),
    }


# --------------------------------------------------------------------------- 1


def rebuild_like_ensure_flow_data(dem_float32, crs, transform, via_float32_file):
    """Reproduce `_ensure_flow_data`'s conditioning, optionally without its file hop.

    `keypoint_analysis.py:1127-1151` writes the DEM to a **float32** temporary GeoTIFF
    with NaN mapped to -9999, reads it back through pysheds, and conditions that. Passing
    ``via_float32_file=False`` runs the same three pysheds steps on the array as held,
    which is the only way to tell a pipeline difference from a dtype difference.
    """
    from pysheds.grid import Grid

    from terrainflow_assessment.modules.flow_analysis import resolve_flats_safely

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)
    try:
        dtype = "float32" if via_float32_file else "float64"
        with rasterio.open(
            tmp_path, "w", driver="GTiff", dtype=dtype,
            crs=crs, transform=transform,
            width=dem_float32.shape[1], height=dem_float32.shape[0],
            count=1, nodata=-9999.0,
        ) as dst:
            data = np.where(np.isnan(dem_float32), -9999.0, dem_float32)
            dst.write(data.astype(dtype), 1)

        grid = Grid.from_raster(tmp_path)
        dem_r = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_r)
        try:
            filled = grid.breach_depressions(pit_filled)
            used = "breach_depressions"
        except AttributeError:
            filled = grid.fill_depressions(pit_filled)
            used = "fill_depressions"
        inflated, eps, inversions = resolve_flats_safely(grid, filled)
        return np.asarray(inflated, dtype="float64"), {
            "depression_step": used,
            "flat_eps": float(eps),
            "flat_inversions": int(inversions),
            "temp_raster_dtype": dtype,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def stage_conditioned_surfaces(ev, dem, ya):
    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

    with ev.stage("conditioned_surfaces") as rec:
        fa = FlowAnalysis()
        fa.load_dem(str(dem))
        result = fa.run(crest_split=False)
        run_surface = np.asarray(result["conditioned_dem"], dtype="float64")
        rec["run_crest_split_false"] = {
            "flat_eps": float(fa.flat_eps),
            "flat_inversions": int(fa.flat_inversions),
            "has_conditioned_dem_key": True,
            "dtype_in_result": str(np.asarray(result["conditioned_dem"]).dtype),
        }

        as_ensure, meta32 = rebuild_like_ensure_flow_data(
            ya.dem, ya.crs, ya.transform, via_float32_file=True)
        as_ensure64, meta64 = rebuild_like_ensure_flow_data(
            ya.dem, ya.crs, ya.transform, via_float32_file=False)
        rec["ensure_flow_data_replica"] = meta32
        rec["ensure_flow_data_replica_float64"] = meta64

        def delta(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a[m] - b[m])
            return {
                "cells_compared": int(m.sum()),
                "max_abs_delta_m": float(d.max()) if d.size else None,
                "mean_abs_delta_m": float(d.mean()) if d.size else None,
                "cells_differing": int((d > 0).sum()),
                "cells_differing_over_1mm": int((d > 1e-3).sum()),
            }

        rec["replica_vs_run"] = delta(as_ensure, run_surface)
        rec["replica_float64_vs_run"] = delta(as_ensure64, run_surface)
        rec["replica_vs_replica_float64"] = delta(as_ensure, as_ensure64)

        work = _probe.workdir("flow_graph")
        np.save(work / "conditioned_run.npy", run_surface)
        np.save(work / "conditioned_ensure_replica.npy", as_ensure)
        rec["cached"] = [str(work / "conditioned_run.npy"),
                         str(work / "conditioned_ensure_replica.npy")]

        ev.note(
            "KPA-48 fix note, measured: run(crest_split=False) does return a "
            "'conditioned_dem' key, and the surface _ensure_flow_data rebuilds differs "
            f"from it by at most {rec['replica_vs_run']['max_abs_delta_m']:.3e} m "
            f"({rec['replica_vs_run']['cells_differing']} of "
            f"{rec['replica_vs_run']['cells_compared']} cells differ at all, "
            f"{rec['replica_vs_run']['cells_differing_over_1mm']} by more than 1 mm). "
            "Dropping the float32 file hop leaves max |Δ| "
            f"{rec['replica_float64_vs_run']['max_abs_delta_m']:.3e} m.")
        ev.note(
            f"The depression step that actually runs is '{meta32['depression_step']}' — "
            "pysheds 0.5 has no breach_depressions, so the except branch is the live one "
            "in both flow_analysis.run and _ensure_flow_data.")

        return run_surface, as_ensure


# --------------------------------------------------------------------------- 2, 3


def pointer_health(next_flat, is_sink, finite_flat, stream_flat):
    """Sinks and pointers that leave the domain, restricted to the channel network.

    A stream cell whose pointer is a sink stops `stream_links`' walk early; a pointer
    into a non-finite cell stops it too. Counting them on the raw surface and again on
    the conditioned one is FLG-19 stated as a number.
    """
    nxt = np.asarray(next_flat, dtype=np.int64)
    sink = np.asarray(is_sink, dtype=bool)
    stream = np.asarray(stream_flat, dtype=bool)
    finite = np.asarray(finite_flat, dtype=bool)

    points_at_nonfinite = ~finite[nxt]
    return {
        "sinks_total": int(sink.sum()),
        "sinks_on_finite_ground": int((sink & finite).sum()),
        "stream_cells": int(stream.sum()),
        "stream_cells_that_are_sinks": int((stream & sink).sum()),
        "stream_cells_pointing_at_nonfinite": int(
            (stream & points_at_nonfinite & ~sink).sum()),
        "stream_cells_pointing_out_of_stream": int(
            (stream & ~sink & ~stream[nxt]).sum()),
    }


def link_population(ya, acc_arr, surface, threshold_cells, cell_size, cols):
    """Everything `find_keypoints` computes about links, for one surface + threshold."""
    from terrainflow_assessment.modules.flow_graph import (
        d8_from_dem,
        strahler_order,
        stream_links,
    )

    stream = (acc_arr >= threshold_cells) & np.isfinite(ya.dem)
    next_flat, is_sink = d8_from_dem(surface, ya.cell_w, ya.cell_h)
    order = strahler_order(next_flat, stream.ravel())
    links = stream_links(next_flat, stream.ravel(), order, cols, max_order=1)

    lengths = [arc_length_m(lk, cell_size) for lk in links]
    floor = profile_floor_m(cell_size)
    clears = [lg for lg in lengths if lg >= floor]

    def catchment(link):
        r, c = link[-1]
        return float(acc_arr[r, c])

    ranked = sorted(links, key=catchment, reverse=True)

    return {
        "threshold_cells": int(threshold_cells),
        "threshold_ha": threshold_cells * ya.cell_w * ya.cell_h / 10_000.0,
        "stream_cells": int(stream.sum()),
        "order1_links": len(links),
        "link_cells": describe([len(lk) for lk in links]),
        "link_length_m": describe(lengths),
        "profile_floor_m": floor,
        "links_clearing_floor": len(clears),
        "links_below_floor": len(links) - len(clears),
        "links_shorter_than_5_cells": sum(1 for lk in links if len(lk) < 5),
        "pointers": pointer_health(
            next_flat, is_sink, np.isfinite(ya.dem).ravel(), stream.ravel()),
        "rank1": ({
            "cells": len(ranked[0]),
            "length_m": arc_length_m(ranked[0], cell_size),
            "catchment_cells": catchment(ranked[0]),
            "catchment_ha": catchment(ranked[0]) * ya.cell_w * ya.cell_h / 10_000.0,
            "head_rc": list(ranked[0][0]),
            "outlet_rc": list(ranked[0][-1]),
            "clears_floor": arc_length_m(ranked[0], cell_size) >= floor,
        } if ranked else None),
    }, ranked


def stage_link_populations(ev, ya, conditioned):
    with ev.stage("link_populations") as rec:
        _fdir, acc_arr = ya._ensure_flow_data()
        cols = ya.dem.shape[1]
        cell_area = ya.cell_w * ya.cell_h
        production_cells = max(20, int(round(2_000.0 / cell_area)))

        rec["production_threshold_cells"] = production_cells
        rec["production_threshold_ha"] = production_cells * cell_area / 10_000.0
        rec["surfaces"] = {}
        rank1_by_surface = {}

        thresholds = [
            ("production_0.2ha", production_cells, "production"),
            ("sensitivity_0.5ha", int(round(5_000.0 / cell_area)), "sensitivity"),
            ("sensitivity_1.0ha", int(round(10_000.0 / cell_area)), "sensitivity"),
        ]

        for surface_name, surface in (("raw", np.asarray(ya.dem, dtype="float64")),
                                      ("conditioned", conditioned)):
            rec["surfaces"][surface_name] = {}
            for label, cells, kind in thresholds:
                stats, ranked = link_population(
                    ya, acc_arr, surface, cells, ya.cell_size, cols)
                stats["role"] = kind
                rec["surfaces"][surface_name][label] = stats
                if label.startswith("production"):
                    rank1_by_surface[surface_name] = ranked
                print(f"    {surface_name:12s} {label:18s} "
                      f"links={stats['order1_links']:4d} "
                      f"clearing {stats['profile_floor_m']:.0f} m="
                      f"{stats['links_clearing_floor']:4d} "
                      f"sinks={stats['pointers']['sinks_on_finite_ground']:5d}",
                      flush=True)

        # The rows above are the *before* picture — a D-infinity mask walked by D8
        # pointers on two surfaces. Production since KPA-52 closed: one graph on the
        # conditioned surface, each valley walked back to its divide and cut at the
        # data edge. Recorded beside them so the change is one file apart.
        valleys = ya.primary_valleys()
        all_kps, all_skipped = ya.find_keypoints(max_valleys=len(valleys) + 1)
        rec["production_after_fix"] = {
            "conditioned_source": ya.conditioned_source,
            "order1_links": len(valleys),
            "link_length_m": describe([v["length_m"] for v in valleys]),
            "extension_cells": describe([v["extension_cells"] for v in valleys]),
            "links_cut_at_the_data_edge": sum(
                1 for v in valleys if v["runs_off_dem_m"] > 0),
            "links_with_divide_on_the_data_edge": sum(
                1 for v in valleys if v["head_on_boundary"]),
            "keypoints": len(all_kps),
            "refused": len(all_skipped),
            "refusal_reasons": sorted({s.split("): ", 1)[-1].split(" (")[0]
                                       for s in all_skipped}),
        }
        after = rec["production_after_fix"]
        print(f"    {'after fix':12s} {'production_0.2ha':18s} "
              f"links={after['order1_links']:4d} keypoints={after['keypoints']:4d} "
              f"cut at edge={after['links_cut_at_the_data_edge']:3d}", flush=True)

        prod = rec["surfaces"]["raw"]["production_0.2ha"]
        cond = rec["surfaces"]["conditioned"]["production_0.2ha"]
        ev.note(
            f"After the KPA-52 fix: {after['order1_links']} primary valleys of median "
            f"{after['link_length_m']['median']:.0f} m, divide to foot, "
            f"{after['keypoints']} keypoints and {after['refused']} refusals "
            f"({', '.join(after['refusal_reasons'])}). The FLG-18/19 figures below are "
            "the before picture and are kept as such.")
        ev.note(
            "FLG-19, measured on the production path BEFORE the fix: the stream mask "
            "came from "
            "D-infinity accumulation over the conditioned surface, but the pointer graph "
            "comes from d8_from_dem over the RAW DEM. Raw gives "
            f"{prod['pointers']['sinks_on_finite_ground']} sinks on finite ground and "
            f"{prod['pointers']['stream_cells_that_are_sinks']} stream cells that are "
            f"themselves sinks; the conditioned surface gives "
            f"{cond['pointers']['sinks_on_finite_ground']} and "
            f"{cond['pointers']['stream_cells_that_are_sinks']}. Order-1 links: "
            f"{prod['order1_links']} raw vs {cond['order1_links']} conditioned.")
        ev.note(
            "FLG-18, measured: stream_links emits at min_cells=3, but keypoint_on_path "
            f"cannot return a keypoint below {prod['profile_floor_m']:.0f} m of thalweg "
            f"(n_samp >= {MIN_SAMPLES_FOR_A_KEYPOINT}). At 0.2 ha on the raw surface "
            f"{prod['order1_links']} links are emitted and only "
            f"{prod['links_clearing_floor']} can clear that floor — "
            f"{prod['links_below_floor']} are profiled and refused by construction.")

        return rank1_by_surface


# --------------------------------------------------------------------------- 4


def stage_rank1_keyline(ev, ya, rank1_by_surface):
    """Rank 1 is the valley a user sees first: it is drawn, and it becomes the master."""
    with ev.stage("rank1_keyline") as rec:
        for surface_name, ranked in rank1_by_surface.items():
            entry = {"links_ranked": len(ranked)}
            if not ranked:
                rec[surface_name] = entry
                continue
            link = ranked[0]
            entry["link"] = {
                "cells": len(link),
                "length_m": arc_length_m(link, ya.cell_size),
                "head_rc": list(link[0]),
                "outlet_rc": list(link[-1]),
            }
            kp = ya.keypoint_on_path(link, require_prominence=True)
            entry["keypoint_with_prominence"] = kp
            entry["keypoint_without_prominence"] = ya.keypoint_on_path(
                link, require_prominence=False)

            if kp is None:
                entry["runs"] = None
                entry["note"] = (
                    "rank-1 valley yields no keypoint under the prominence rule, so no "
                    "keyline_master_geom is ever set from it")
            else:
                runs = ya.get_cultivation_runs(kp)
                entry["runs"] = [{
                    "line_type": r["line_type"],
                    "offset_m": r.get("offset_m"),
                    "elevation": r.get("elevation"),
                    "drift_fall_m": r.get("drift_fall_m"),
                    "drift_1_in_n": r.get("drift_1_in_n"),
                    "over_limit": r.get("over_limit"),
                    "vertices": len(list(r["geometry"].coords)),
                    "length_m": float(r["geometry"].length),
                } for r in runs]
                master = next(
                    (r for r in entry["runs"] if r["line_type"] == "keyline"), None)
                entry["keyline_master"] = master
                if master is None:
                    ev.note(
                        f"[{surface_name}] get_cultivation_runs returned no run of "
                        "line_type 'keyline' — contour.py:1600-1605 would leave "
                        "keyline_master_geom at its previous value.")
            rec[surface_name] = entry


def main():
    dem = _probe.fixture_path()
    _probe.start_qgis()
    _probe.banner("p_flow_graph — conditioned surface and order-1 link population", dem)

    from terrainflow_assessment.modules.keypoint_analysis import YeomansKeylineAnalysis

    ev = _probe.Evidence(
        "p_flow_graph", findings=["KPA-38", "KPA-48", "FLG-18", "FLG-19"], dem=dem)
    ev["dem_stats"] = _probe.dem_stats(dem)
    ev["profile_floor_m"] = profile_floor_m(
        (ev["dem_stats"]["cell_w_m"] + ev["dem_stats"]["cell_h_m"]) / 2.0)

    ya = YeomansKeylineAnalysis(str(dem))
    ev["yeomans_dem_dtype"] = str(ya.dem.dtype)

    _run_surface, replica = stage_conditioned_surfaces(ev, dem, ya)
    rank1 = stage_link_populations(ev, ya, replica)
    if rank1:
        stage_rank1_keyline(ev, ya, rank1)

    ev.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
