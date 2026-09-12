"""p_ridgelines — why the Ridgelines button returns nothing, in a distribution.

The open finding: on the owner's 1139x1016 design every one of **1,462** skeleton
components is shorter than ``min_cells`` (50 at 1 m), so ``find_ridgelines`` draws
nothing. M-11 made the loop fast (5.8 s -> 85 ms) without making it useful.

**Probe before patching.** 1,462 components none reaching 50 m reads two ways and
the fix is different for each:

* the longest is ~40 m with a tail behind it -> the **bar** is too high for this
  terrain, and the answer is a lower ``min_length_m`` justified from the numbers
  below;
* the longest is ~8 m and there are 1,462 of them -> the **skeleton is
  fragmenting**, and lowering the bar would draw hundreds of confetti fragments,
  which is worse than the current honest nothing. The fix would be joining
  collinear neighbours before filtering, which is a separate decision.

So this probe prints the distribution and renders the raw skeleton, and asserts
nothing. Run it against the committed full tile — as of `215363d`
``tests/fixtures/quail_island_full.tif`` **is** the DEM embedded in the real
``.tfd``, bit-identical, so this finding needs no client file::

    $env:PYTHONPATH="F:\\Terrain Flow Design\\TerrainFlow"
    & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_ridgelines.py

**No design, and no `PluginHarness`.** The finding is pure terrain: the DEM, the
TPI, the skeleton and the filter are all library code, and the only reason this
needs the QGIS interpreter at all is `flow_analysis`'s backward-compatibility
re-export of `AnalysisWorker` at its foot, which imports `qgis.PyQt`. Both
interpreters carry the same scikit-image, so the skeleton is the same one
production thins — worth checking, because `_thin_to_centreline` falls back to
three erosions without it and that is a different mask, not a slower one.

The 400x400 clip is available with ``--fixture=`` for contrast, but every number
that matters is the full tile's: the clip is 1/7th of the cells and has no nodata
at all, and the sea is 39% of the real tile.

Three distinct measurements of "length", because the filter and the output do not
use the same one:

1. ``sizes[region_id]`` — the **pixel count** per component. This is what
   ``min_cells`` is compared against, so it decides what survives.
2. ``LineString(...).length`` — the **map length** of the ordered polyline, which
   is what the user reads off the label and what ``length_m`` reports. A diagonal
   run of *n* pixels is ``1.41 * (n - 1)`` m, so this is the larger number, and
   the two are not interchangeable.
3. the **bounding-box diagonal** — how far a component reaches end to end. A
   component whose pixel count far exceeds its extent is a tangle, not a line.

And the fragmentation question gets its own direct test: dilate the skeleton by
one cell and re-label. If 1,462 components collapse into a handful, they are
pieces of the same ridges separated by one-cell gaps, and that is answer two.

What it found, 2026-09-12
-------------------------
**Answer two, and with a named mechanism.** Median component: **1 cell**. 880 of
the 1,462 are single cells and the longest is 27, against a bar of 50. Nothing in
the distribution is near 40 m, so no defensible lowering of ``min_length_m``
reaches a useful answer — at 25 m it draws one 4.4 m fragment.

Three things had to be separated to see it, and each moved the numbers:

1. **The mask sets the bar.** ``min_tpi_sd`` is a cut in standard deviations *of
   the TPI inside the mask*, and this tile is 39% sea with a ~15 m coastal scarp
   round the land. With no mask the sd is **0.732 m** and the ridge cells land
   almost entirely on the shoreline break — see ``overview.png``, which is the
   coast traced in full and the farm left blank. With the design's own earthworks
   area the sd is **0.190 m** and the detections move inland where the ridges are.
   The probe reproduces the register's **1,462** exactly in that configuration, so
   the numbers below are production's and not an approximation of them.

2. **`nd_label(skeleton)` is 4-connected and `_order_pixels` walks 8**, twenty
   lines apart in the same function. scipy's default structure is the cross, so a
   skeleton running diagonally — which is what `skeletonize` produces wherever a
   ridge is not aligned to the grid — is labelled as *one component per cell*.
   ``stage connectivity_oracle`` demonstrates it on a 10-cell diagonal: **10
   components** as production labels it, 1 when labelled the way the walker
   traverses. On the real skeleton it is 1,462 against 816.

3. **It is a ceiling, not a contribution.** ``stage acc_bar`` sweeps the other
   term from ``acc <= 1`` to dropping it altogether (TPI alone, 38,911 ridge cells
   against production's 4,816). Under 4-connected labelling the count of
   components reaching 50 cells is **zero at every one of those bars**. The button
   cannot draw a ridgeline at any setting of the knobs it exposes. Labelled
   8-connected the same sweep gives 1 component at ``acc <= 2``, 14 at ``<= 10``
   and 19 at ``<= 20``.

So the connectivity is a defect to fix on its own terms, and it is *not*
sufficient: at production's ``acc <= 2`` it yields one ridgeline. Making the
button useful also means a view on what ``acc <= 2`` is for, and that is a
decision about what counts as a ridge rather than a bug.

What was done about it
----------------------
Both, on the owner's call. The labelling is 8-connected, and the accumulation
term is now ``max_catchment_m2`` — a catchment **area**, defaulting to 20 m².
The bar was chosen off ``stage candidate_bars``, which prices each candidate by
what it actually draws: 5 m² gives 7 lines / 150 m, 10 m² gives 14 / 338 m, 20 m²
gives 19 / 586 m, and beyond it 30 m² and 50 m² add only fragments (22 / 695 m
and 25 / 689 m) while the *median* line falls from 26.6 m to 17.7 m. 20 m² is
where the longest ridge reaches its full 157 m and the median peaks.
``min_length_m`` stays at 50: the bar was never the problem.

Real design 0 -> 19 ridgelines; committed 400x400 clip 1 -> 7.

**Still open, and deliberately.** What counts as a ridge here is "convex ground
that sheds nearly all its own water" — a proxy chosen for being computable off a
DEM, not for being right, and 20 m² is a threshold on that proxy rather than a
definition. A divide traced from the flow field, or a multi-scale TPI, would be a
different and probably better answer. The owner has this down for a later pass;
this probe is the evidence it should start from.

Re-running this probe after that pass is the point of it: every stage below is
production's own preamble, so it follows the source rather than describing a
version of it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import _probe  # noqa: E402  (path setup above must run first)

#: The full LINZ tile, committed at `215363d`. Bit-identical to the DEM inside
#: `Quail_Island 03.09.2026.tfd`, so the finding reproduces without the design.
FULL_TILE = _probe.REPO / "tests" / "fixtures" / "quail_island_full.tif"

#: The owner's design, for its **area polygon only**. `find_ridgelines` is handed
#: one by `contour.py:1226`, and `min_tpi_sd` is a cut in standard deviations *of
#: the TPI inside it*, so a run with no mask is not a weaker version of the
#: production run — it is a different question. Outside the repo, read-only, and
#: every stage that needs it skips cleanly when it is absent.
REAL_TFD = (r"F:\Terrain Flow Design\QGIS Working Files"
            r"\Quail_Island 03.09.2026.tfd")

#: `find_ridgelines`' production defaults, restated so the probe cannot drift from
#: them silently. Checked against the signature in `stage_defaults`.
TPI_WINDOW_M = 15.0
MIN_TPI_SD = 1.0
MIN_LENGTH_M = 50.0
MAX_CATCHMENT_M2 = 20.0

#: What the probe sweeps `min_length_m` over when asking "what would a lower bar
#: draw?". Deliberately reaches well below anything defensible, so the shape of
#: the curve is visible rather than just two points on it.
SWEEP_M = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]


def fixture(argv=None):
    """The DEM to probe: `--fixture=PATH`, else the committed **full** tile.

    Note the default differs from `_probe.fixture_path`, which falls back to the
    400x400 clip. This finding is about a 1139x1016 site; measuring it on the clip
    would answer a different question and look like it had answered this one.
    """
    argv = sys.argv[1:] if argv is None else argv
    for arg in argv:
        if arg.startswith("--fixture="):
            from pathlib import Path
            p = Path(arg.split("=", 1)[1]).expanduser()
            if not p.exists():
                raise SystemExit(f"--fixture path does not exist: {p}")
            return p
    if not FULL_TILE.exists():
        raise SystemExit(
            f"the committed full tile is missing: {FULL_TILE}\n"
            f"It is tracked as of 215363d; `git checkout` it before probing.")
    return FULL_TILE


def accumulation(dem_path, work):
    """Flow accumulation and pond throughflow for *dem_path*, cached on disk.

    `DrainageLineAnalysis` reads both from files, and the controller hands it the
    baseline's rasters. A full `FlowAnalysis.run()` on the tile is ~5.4 s, which is
    worth caching across the probe's own re-runs but not worth skipping: the ridge
    test is a bar on the accumulation, so measuring against one this DEM did not
    produce would answer nothing.
    """
    import numpy as np

    from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

    acc_path = work / "acc.tif"
    pond_path = work / "pond_flow.tif"
    stamp = work / "source.txt"
    want = f"{dem_path}|{os.path.getmtime(dem_path)}|{os.path.getsize(dem_path)}"
    # Keyed on the source's size and mtime, not its path — the same precondition
    # rule G-9's slope cache needed, and for the same reason: a probe rewriting its
    # scratch tile to the same name would otherwise be served yesterday's flow.
    if (acc_path.exists() and stamp.exists()
            and stamp.read_text(encoding="utf-8") == want):
        print(f"    accumulation: cached at {acc_path}")
        return str(acc_path), (str(pond_path) if pond_path.exists() else None)

    fa = FlowAnalysis()
    fa.load_dem(str(dem_path))
    result = fa.run()
    fa.save_result(np.asarray(fa.acc, dtype="float32"), str(acc_path),
                   band_description="flow accumulation (cells)", nodata=float("nan"))
    pond = result.get("pond_flow")
    out_pond = None
    if pond is not None:
        fa.save_result(np.asarray(pond, dtype="float32"), str(pond_path),
                       band_description="pond throughflow (cells)",
                       nodata=float("nan"))
        out_pond = str(pond_path)
    stamp.write_text(want, encoding="utf-8")
    print(f"    accumulation: computed, {result.get('crest_ponds')} ponds contracted")
    return str(acc_path), out_pond


def build_skeleton(ka, boundary_mask=None, acc_bar=None):
    """Re-derive exactly what `find_ridgelines`' filter loop walks.

    Every line here is copied from `find_ridgelines`' preamble rather than
    approximated, because an approximation is precisely the guess this probe exists
    to replace — `p_gate_ui._ridgeline_components` takes the same approach and for
    the same reason. Returns the intermediates the distribution is computed from.

    ``acc_bar`` is in **cells**, not m², because that is the units of the array it
    is compared against. ``None`` means production's own bar and is what every
    stage but `acc_bar` uses; ``float("inf")`` drops the term.
    """
    import numpy as np
    from scipy.ndimage import label as nd_label

    from terrainflow_assessment.modules.keypoint_analysis import _thin_to_centreline
    from terrainflow_assessment.modules.terrain_indices import (
        landform_classes,
        landform_tpi,
    )

    valid = np.isfinite(ka.dem)
    tpi = landform_tpi(ka.dem, ka.cell_w, ka.cell_h, window_m=TPI_WINDOW_M)
    with np.errstate(invalid="ignore"):
        above = landform_classes(tpi, sd=MIN_TPI_SD, mask=boundary_mask) == 1
        # Production's own bar unless a caller is sweeping it. Swept in
        # `stage acc_bar`, because "which of the two terms breaks the line" is not
        # answerable from the product of them.
        if acc_bar is None:
            acc_bar = max(1.0, MAX_CATCHMENT_M2 / (ka.cell_w * ka.cell_h))
        low_acc = ka.acc <= acc_bar
        ridge_raw = above & low_acc & valid
    ridge_raw[[0, -1], :] = False
    ridge_raw[:, [0, -1]] = False
    if boundary_mask is not None:
        ridge_raw &= boundary_mask

    skeleton = _thin_to_centreline(ridge_raw)
    if not skeleton.any():
        skeleton = ridge_raw
    # Production's own call. It was `nd_label(skeleton)` with no `structure=` —
    # scipy's 4-connected default — until this probe found that that is why the
    # button returned nothing; the historical labelling is still measured beside
    # it in `stage connectivity` rather than dropped.
    labeled, n = nd_label(skeleton, structure=np.ones((3, 3), dtype=int))
    labeled4, n4 = nd_label(skeleton)
    return {
        "tpi": tpi,
        "valid": valid,
        "above": above,
        "low_acc": low_acc,
        "ridge_raw": ridge_raw,
        "skeleton": skeleton,
        "labeled": labeled,
        "n": int(n),
        "labeled4": labeled4,
        "n4": int(n4),
        "acc_bar_cells": float(acc_bar),
        "min_cells": max(3, int(MIN_LENGTH_M / ka.cell_size)),
    }


def component_table(ka, labeled, n):
    """Per-component pixel count, map length and extent — the whole distribution.

    Map length is the **production** number: `_order_pixels` then `LineString`,
    the same two calls `find_ridgelines` makes, so a figure here is a figure a user
    would have read off the label had the component survived.
    """
    import numpy as np
    from scipy.ndimage import find_objects
    from shapely.geometry import LineString

    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    boxes = find_objects(labeled)
    rows = []
    for rid in range(1, n + 1):
        box = boxes[rid - 1]
        if box is None:
            continue
        rc = np.argwhere(labeled[box] == rid)
        rc += (box[0].start, box[1].start)
        ordered = ka._order_pixels(rc.tolist())
        length_m = 0.0
        if len(ordered) >= 2:
            xy = [ka._rc_to_xy(r, c) for r, c in ordered]
            try:
                length_m = float(LineString(xy).length)
            except Exception:
                length_m = 0.0
        h = box[0].stop - box[0].start
        w = box[1].stop - box[1].start
        rows.append({
            "id": rid,
            "cells": int(sizes[rid]),
            "length_m": length_m,
            "extent_m": float(np.hypot((h - 1) * ka.cell_h, (w - 1) * ka.cell_w)),
            "box": [int(box[0].start), int(box[1].start), int(h), int(w)],
        })
    return rows


def summarise(ka, mask, label, acc_bar=None):
    """One row of the mask-sensitivity table: what this mask makes of the terrain.

    `min_tpi_sd` is a cut in standard deviations **of the TPI over the masked
    area**, so the mask does not merely restrict where ridges may be found — it
    sets the bar. That is the whole point of `landform_classes`' ``mask``
    parameter ("classify a site against its own relief, not against a tile that is
    mostly harbour") and it means a run with no mask and a run with the design's
    analysis area are asking two different questions of the same terrain.
    """
    import numpy as np
    from shapely.geometry import LineString

    sk = build_skeleton(ka, boundary_mask=mask, acc_bar=acc_bar)
    finite = np.isfinite(sk["tpi"])
    sample = finite if mask is None else (finite & mask)
    lab8, n8 = sk["labeled"], sk["n"]
    sizes8 = np.bincount(lab8.ravel(), minlength=n8 + 1)[1:]

    longest_m = 0.0
    if n8:
        rid = int(sizes8.argmax()) + 1
        rc = np.argwhere(lab8 == rid)
        ordered = ka._order_pixels(rc.tolist())
        if len(ordered) >= 2:
            try:
                longest_m = float(LineString(
                    [ka._rc_to_xy(r, c) for r, c in ordered]).length)
            except Exception:
                longest_m = 0.0

    min_cells = sk["min_cells"]
    return {
        "mask": label,
        "acc_bar": acc_bar,
        "mask_cells": int(sample.sum()),
        "tpi_sd_m": float(np.std(sk["tpi"][sample])) if sample.any() else 0.0,
        "cut_m": (float(MIN_TPI_SD * np.std(sk["tpi"][sample]))
                  if sample.any() else 0.0),
        "ridge_raw_cells": int(sk["ridge_raw"].sum()),
        "skeleton_cells": int(sk["skeleton"].sum()),
        "components_4": sk["n4"],
        "components_8": int(n8),
        "largest_cells_8": int(sizes8.max()) if n8 else 0,
        "largest_length_m_8": longest_m,
        # The number the button's output actually turns on: how many components
        # would clear `min_cells` at all. Zero here is zero ridgelines drawn.
        "over_min_cells_4": int((np.bincount(
            sk["labeled4"].ravel(), minlength=sk["n4"] + 1)[1:] >= min_cells).sum())
            if sk["n4"] else 0,
        "over_min_cells_8": int((sizes8 >= min_cells).sum()) if n8 else 0,
        "_sk": sk,
    }


def print_summary_header():
    print("    mask                 mask_cells   sd_m   cut_m  ridge_raw  "
          "comp4  comp8  big8  big8_m  >=50c4  >=50c8")


def print_summary(row):
    print(f"    {row['mask']:<20} {row['mask_cells']:>10,} "
          f"{row['tpi_sd_m']:>6.3f} {row['cut_m']:>7.3f} "
          f"{row['ridge_raw_cells']:>10,} {row['components_4']:>6,} "
          f"{row['components_8']:>6,} {row['largest_cells_8']:>5} "
          f"{row['largest_length_m_8']:>7.1f} {row['over_min_cells_4']:>7} "
          f"{row['over_min_cells_8']:>7}")


def design_mask(ka, dem_path):
    """The mask `contour.py` really passes: the design's own area polygon.

    `_get_keypoint_boundary_mask` takes the **earthworks** area layer if there is
    one and the **analysis** area otherwise, reprojects it to the DEM's CRS and
    rasterises with ``all_touched=True``. Read straight out of the `.tfd`'s
    `design.json` here — `shapely` and `rasterio.features` do the whole job, so
    this needs neither QGIS nor `PluginHarness`, and the `.tfd` is opened
    read-only and never unpacked.

    Returns ``(mask, label)`` or ``(None, reason)``. The design is a client file
    outside the repo: its absence is a skipped stage, not a failure.
    """
    import json
    import zipfile

    import rasterio
    from rasterio.features import rasterize
    from shapely import wkt as shapely_wkt

    if not os.path.exists(REAL_TFD):
        return None, f"not present: {REAL_TFD}"
    with zipfile.ZipFile(REAL_TFD) as z:
        design = json.loads(z.read("design.json").decode("utf-8"))
    areas = design.get("areas") or {}
    for key in ("earthworks", "analysis"):
        area = areas.get(key)
        if area and area.get("wkt"):
            geom = shapely_wkt.loads(area["wkt"])
            with rasterio.open(str(dem_path)) as src:
                if src.crs and area.get("crs") and \
                        src.crs.to_string() != area["crs"]:
                    return None, (f"the {key} area is {area['crs']} and the DEM is "
                                  f"{src.crs.to_string()} — not reprojecting here")
                mask = rasterize([(geom, 1)], out_shape=(src.height, src.width),
                                 transform=src.transform, fill=0,
                                 all_touched=True, dtype="uint8").astype(bool)
            if mask.any():
                return mask, f"design {key} area"
    return None, "the design carries no earthworks or analysis area"


def simulate_fixed(ka, mask, max_catchment_m2, min_length_m=MIN_LENGTH_M):
    """What `find_ridgelines` draws at a given catchment bar.

    This used to reimplement the fix so the candidate bars could be priced before
    the source was touched. The fix has landed, so it calls the real function —
    a reimplementation kept beside the thing it imitates is a second copy waiting
    to drift, and this probe exists partly because an earlier copy of this very
    preamble did exactly that.
    """
    return ka.find_ridgelines(
        tpi_window_m=TPI_WINDOW_M, min_tpi_sd=MIN_TPI_SD,
        min_length_m=min_length_m, max_catchment_m2=max_catchment_m2,
        boundary_mask=mask)


def render_lines(ka, mask, bars, work):
    """Draw what each candidate bar would put on the map, over the terrain.

    The count and the quantiles cannot answer "are these ridges?". This can.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from terrainflow_assessment.modules.footprint import xy_to_rc

    fig, axes = plt.subplots(1, len(bars), figsize=(7 * len(bars), 7))
    axes = np.atleast_1d(axes)
    hill = np.where(np.isfinite(ka.dem), ka.dem, np.nan)
    for ax, m2 in zip(axes, bars):
        lines = simulate_fixed(ka, mask, m2)
        ax.imshow(hill, cmap="terrain")
        for ln in lines:
            rc = [xy_to_rc(ka.transform, x, y) for x, y in ln["geometry"].coords]
            ax.plot([c for _r, c in rc], [r for r, _c in rc],
                    color="magenta", linewidth=1.4)
        ax.set_title(f"max_catchment {m2} m² — {len(lines)} lines, "
                     f"{sum(ln['length_m'] for ln in lines):.0f} m total")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    p = work / "candidate_bars.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return [p]


def describe(rows, label):
    """Print and return the three distributions for one labelling of the skeleton."""
    import numpy as np

    cells = [r["cells"] for r in rows]
    lengths = [r["length_m"] for r in rows]
    extents = [r["extent_m"] for r in rows]
    out = {
        "labelling": label,
        "n": len(rows),
        "cells": quantiles(cells),
        "length_m": quantiles(lengths),
        "extent_m": quantiles(extents),
        "cells_hist": histogram(cells, [1, 2, 3, 5, 8, 12, 20, 30, 50, 100]),
        "length_hist": histogram(lengths, [0, 2, 5, 10, 15, 20, 30, 40, 50, 100]),
        "total_skeleton_cells": int(sum(cells)),
        "mean_cells": float(np.mean(cells)) if cells else 0.0,
        "top20": sorted(rows, key=lambda r: r["cells"], reverse=True)[:20],
    }
    scale = max(1, len(rows) // 60 or 1)
    print(f"    {label}: {len(rows):,} components, {sum(cells):,} skeleton cells, "
          f"mean {out['mean_cells']:.1f} cells each")
    print("    pixel count   " + _q(out["cells"]))
    print("    map length m  " + _q(out["length_m"]))
    print("    bbox diag m   " + _q(out["extent_m"]))
    for title, key in (("pixel-count histogram", "cells_hist"),
                       ("map-length histogram (m)", "length_hist")):
        print(f"\n    {title}")
        for b in out[key]:
            hi = f"<{b['hi']}" if b["hi"] else "+"
            print(f"      {b['lo']:>4} {hi:<6} {b['n']:>6}  "
                  + "#" * min(60, b["n"] // scale))
    print("\n    the 20 largest components")
    print("      rank  cells  length_m  extent_m  box(row,col,h,w)")
    for i, r in enumerate(out["top20"], 1):
        print(f"      {i:>4}  {r['cells']:>5}  {r['length_m']:>8.1f}  "
              f"{r['extent_m']:>8.1f}  {r['box']}")
    return out


def sweep_bar(rows, ka, label):
    """What each candidate ``min_length_m`` would actually draw.

    `find_ridgelines` caps its output at the 30 longest, so both numbers matter:
    how many clear the bar, and what the 30 that get drawn look like. A bar that
    lets 400 through and draws the 30 longest is not "30 ridges" — it is a lawn
    with 30 blades picked off it, and the shortest drawn length says which.
    """
    out = []
    for m in SWEEP_M:
        mc = max(3, int(m / ka.cell_size))
        survivors = sorted((r for r in rows if r["cells"] >= mc),
                           key=lambda r: r["length_m"], reverse=True)
        drawn = survivors[:30]
        out.append({
            "min_length_m": m,
            "min_cells": mc,
            "survivors": len(survivors),
            "drawn": len(drawn),
            "shortest_drawn_m": drawn[-1]["length_m"] if drawn else None,
            "longest_drawn_m": drawn[0]["length_m"] if drawn else None,
        })
    print(f"    {label}")
    print("    min_length_m  min_cells  survive  drawn  longest_m  shortest_drawn_m")
    for s in out:
        lo = "-" if s["shortest_drawn_m"] is None else f"{s['shortest_drawn_m']:.1f}"
        hi = "-" if s["longest_drawn_m"] is None else f"{s['longest_drawn_m']:.1f}"
        print(f"      {s['min_length_m']:>8}  {s['min_cells']:>9}  "
              f"{s['survivors']:>7}  {s['drawn']:>5}  {hi:>9}  {lo:>16}")
    return out


def quantiles(values):
    import numpy as np

    if not len(values):
        return {}
    a = np.asarray(values, dtype="float64")
    qs = [0, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def histogram(values, edges):
    import numpy as np

    a = np.asarray(values, dtype="float64")
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append({"lo": lo, "hi": hi, "n": int(((a >= lo) & (a < hi)).sum())})
    out.append({"lo": edges[-1], "hi": None,
                "n": int((a >= edges[-1]).sum())})
    return out


def main():
    import numpy as np

    # Nothing here builds a Qt object, but `flow_analysis` imports `qgis.PyQt` on
    # the way in and the documented trap is that a Qt import without a
    # `QgsApplication` can take the process down with a bare exit code and no
    # traceback. Two lines is cheaper than diagnosing that a second time.
    _probe.start_qgis()

    dem_path = fixture()
    ev = _probe.Evidence("p_ridgelines",
                         ["ridgelines-returns-nothing"], dem=dem_path)
    _probe.banner("p_ridgelines — the component length distribution", dem_path)

    work = _probe.workdir("ridgelines")

    with ev.stage("dem"):
        info = _probe.dem_stats(dem_path)
        ev["dem_stats"] = info
        cells = info["shape"][0] * info["shape"][1]
        print(f"    {info['shape'][0]}x{info['shape'][1]} = {cells:,} cells, "
              f"{cells - info['nodata_cells']:,} valid "
              f"({info['nodata_cells'] / cells * 100:.0f}% nodata)")
        print(f"    cell {info['cell_w_m']} x {info['cell_h_m']} m, "
              f"z {info['z_min']:.1f}..{info['z_max']:.1f} m")

    with ev.stage("defaults"):
        # Rule: the probe must be measuring the *production* bar. Read it off the
        # signature rather than trusting the constants at the top of this file.
        import inspect

        from terrainflow_assessment.modules.keypoint_analysis import (
            DrainageLineAnalysis,
        )
        sig = inspect.signature(DrainageLineAnalysis.find_ridgelines)
        live = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}
        ev["production_defaults"] = {k: v for k, v in live.items()
                                     if not hasattr(v, "shape")}
        print(f"    find_ridgelines defaults: {ev['production_defaults']}")
        for name, mine in (("tpi_window_m", TPI_WINDOW_M),
                           ("min_tpi_sd", MIN_TPI_SD),
                           ("min_length_m", MIN_LENGTH_M)):
            if live.get(name) != mine:
                ev.note(f"probe uses {name}={mine} but production default is "
                        f"{live.get(name)} — the numbers below are NOT production's")

    with ev.stage("accumulation"):
        acc_path, pond_path = accumulation(dem_path, work)
        ev["acc_path"] = acc_path
        ev["pond_path"] = pond_path

    from terrainflow_assessment.modules.keypoint_analysis import DrainageLineAnalysis
    ka = DrainageLineAnalysis(str(dem_path), acc_path, pond_path)

    sk = None
    with ev.stage("skeleton"):
        sk = build_skeleton(ka)
        tpi_finite = sk["tpi"][np.isfinite(sk["tpi"])]
        ev["tpi"] = {
            "sd_m": float(np.std(tpi_finite)),
            "max_m": float(tpi_finite.max()),
            "min_m": float(tpi_finite.min()),
            "cut_m": float(MIN_TPI_SD * np.std(tpi_finite)),
        }
        ev["masks"] = {
            "ridge_raw_cells": int(sk["ridge_raw"].sum()),
            "skeleton_cells": int(sk["skeleton"].sum()),
            "components": sk["n"],
            "min_cells": sk["min_cells"],
        }
        print(f"    TPI sd {ev['tpi']['sd_m']:.3f} m, range "
              f"{ev['tpi']['min_m']:.2f}..{ev['tpi']['max_m']:.2f} m; "
              f"ridge cut at {ev['tpi']['cut_m']:.3f} m")
        print(f"    ridge_raw {ev['masks']['ridge_raw_cells']:,} cells -> "
              f"skeleton {ev['masks']['skeleton_cells']:,} cells -> "
              f"{sk['n']:,} components")
        print(f"    min_cells = {sk['min_cells']} "
              f"(min_length_m {MIN_LENGTH_M} / cell {ka.cell_size} m)")

    with ev.stage("mask_decomposition"):
        # `ridge_raw = above & (acc <= 2) & valid` is a three-way AND and the
        # component count is downstream of all of it. Which term is doing the
        # cutting is not guessable from the product: 3,013 cells could be a strict
        # TPI cut over a broad low-accumulation crest, or a generous TPI cut over a
        # hairline of `acc <= 2`. Those want different fixes, so count them apart.
        v = sk["valid"]
        terms = {
            "valid": int(v.sum()),
            "above_tpi": int((sk["above"] & v).sum()),
            "acc_le_2": int((sk["low_acc"] & v).sum()),
            "above_and_acc": int(sk["ridge_raw"].sum()),
        }
        terms["above_pct_of_valid"] = terms["above_tpi"] / terms["valid"] * 100
        terms["acc_pct_of_valid"] = terms["acc_le_2"] / terms["valid"] * 100
        # If the two were independent, the product of the shares would predict the
        # intersection. Ridges and low accumulation are positively correlated, so
        # the real intersection should be *larger* than independence predicts; if it
        # is not, the two terms are fighting each other.
        terms["independent_prediction"] = (
            terms["above_tpi"] * terms["acc_le_2"] / terms["valid"])
        ev["mask_decomposition"] = terms
        print(f"    valid cells                 {terms['valid']:>9,}")
        print(f"    TPI > {MIN_TPI_SD} sd                 {terms['above_tpi']:>9,}  "
              f"({terms['above_pct_of_valid']:.1f}% of valid)")
        print(f"    acc <= 2                    {terms['acc_le_2']:>9,}  "
              f"({terms['acc_pct_of_valid']:.1f}% of valid)")
        print(f"    both (= ridge_raw)          {terms['above_and_acc']:>9,}")
        print(f"    if the two were independent {terms['independent_prediction']:>9,.0f}")
        for acc_bar in (1, 2, 3, 5, 10, 20):
            n_acc = int((sk["above"] & (ka.acc <= acc_bar) & v).sum())
            print(f"      ridge_raw at acc <= {acc_bar:<3}     {n_acc:>9,}")

    with ev.stage("connectivity"):
        # The measurement the gap histogram demanded, kept after the fix because it
        # is what says the fix is still in. A skeletonised line moves diagonally
        # wherever the ridge does not run along a grid axis, and `nd_label`'s
        # **default** structure is the 4-connected cross — so a diagonal step was a
        # break, while `_order_pixels` twenty lines further down walked all eight
        # neighbours. The components being filtered were not the components the
        # walker would trace. Production now labels 8-connected; the historical
        # labelling is measured beside it so the gap stays visible.
        sizes8 = np.bincount(sk["labeled"].ravel(), minlength=sk["n"] + 1)[1:]
        sizes4 = np.bincount(sk["labeled4"].ravel(), minlength=sk["n4"] + 1)[1:]
        conn = {
            "eight_connected (production)": {
                "components": sk["n"],
                "largest_cells": int(sizes8.max()) if sk["n"] else 0,
                "mean_cells": float(sizes8.mean()) if sk["n"] else 0.0,
                "singletons": int((sizes8 == 1).sum()),
            },
            "four_connected (was)": {
                "components": sk["n4"],
                "largest_cells": int(sizes4.max()) if sk["n4"] else 0,
                "mean_cells": float(sizes4.mean()) if sk["n4"] else 0.0,
                "singletons": int((sizes4 == 1).sum()),
            },
        }
        ev["connectivity"] = conn
        for key, c in conn.items():
            print(f"    {key:<28}  {c['components']:>5} components, "
                  f"largest {c['largest_cells']:>4} cells, "
                  f"mean {c['mean_cells']:.1f}, "
                  f"{c['singletons']} single-cell")
        if sk["n"] and sk["n4"]:
            ev.note(f"4-connected labelling splits this skeleton into "
                    f"{sk['n4']} components where production's 8-connected gives "
                    f"{sk['n']}; `_order_pixels` walks 8 neighbours")

    rows = []
    with ev.stage("distribution"):
        rows = component_table(ka, sk["labeled"], sk["n"])
        ev["distribution"] = describe(rows, "8-connected (production)")

    with ev.stage("sweep"):
        ev["sweep"] = sweep_bar(rows, ka, "8-connected (production)")

    rows4 = []
    with ev.stage("distribution_4connected"):
        # The same three measurements over the same skeleton, labelled the way it
        # used to be. Everything else is held fixed — same DEM, same TPI, same cut,
        # same skeleton — so any difference here is the `structure=` argument and
        # nothing else.
        rows4 = component_table(ka, sk["labeled4"], sk["n4"])
        ev["distribution_4connected"] = describe(rows4, "4-connected (was)")

    with ev.stage("sweep_4connected"):
        ev["sweep_4connected"] = sweep_bar(rows4, ka, "4-connected (was)")

    with ev.stage("fragmentation"):
        # The direct test of answer two. If one cell of dilation collapses 1,462
        # components into a handful, they are pieces of the same ridges with
        # one-cell gaps between them and the bar is not the problem.
        from scipy.ndimage import binary_dilation
        from scipy.ndimage import label as nd_label

        frag = {"as_is": sk["n"]}
        for radius in (1, 2, 3):
            size = 2 * radius + 1
            grown = binary_dilation(sk["skeleton"], structure=np.ones((size, size)))
            _lab, m = nd_label(grown)
            gsizes = np.bincount(_lab.ravel(), minlength=m + 1)[1:]
            frag[f"dilate_{radius}"] = {
                "components": int(m),
                "largest_cells": int(gsizes.max()) if m else 0,
                # The dilated blob is fatter than the skeleton it came from, so its
                # own cell count says nothing about ridge length. What it *can* say
                # is how many original components fell into one group.
                "skeleton_cells_in_largest": int(
                    sk["skeleton"][_lab == (int(gsizes.argmax()) + 1)].sum()
                ) if m else 0,
                "original_components_in_largest": int(len(np.unique(
                    sk["labeled"][(_lab == (int(gsizes.argmax()) + 1))
                                  & sk["skeleton"]]
                ))) if m else 0,
            }
        ev["fragmentation"] = frag
        print(f"    as labelled                 {frag['as_is']:,} components")
        for radius in (1, 2, 3):
            f = frag[f"dilate_{radius}"]
            print(f"    dilated by {radius} cell(s)       {f['components']:,} groups; "
                  f"largest holds {f['original_components_in_largest']:,} of them "
                  f"({f['skeleton_cells_in_largest']:,} skeleton cells)")

    with ev.stage("endpoint_gaps"):
        # How far is each component's nearest *other* component? A fragmenting
        # skeleton has a mode at 1-2 cells; genuinely separate ridges do not.
        from scipy.ndimage import distance_transform_edt

        gaps = []
        skel = sk["skeleton"]
        labeled = sk["labeled"]
        # One EDT over the whole skeleton gives, for every cell, the distance to
        # the nearest skeleton cell — which is 0 on the skeleton itself. So the
        # per-component question needs the EDT of "everything except me", and doing
        # that 1,462 times is a raster pass each. Sample instead: the 200 largest
        # components answer the shape question and cost 200 passes, not 1,462.
        order = sorted(rows, key=lambda r: r["cells"], reverse=True)[:200]
        for r in order:
            mine = labeled == r["id"]
            others = skel & ~mine
            if not others.any():
                continue
            d = distance_transform_edt(~others, sampling=(ka.cell_h, ka.cell_w))
            gaps.append(float(d[mine].min()))
        ev["endpoint_gaps"] = {
            "sampled": len(gaps),
            "quantiles_m": quantiles(gaps),
            "hist_m": histogram(gaps, [0, 1.5, 2.5, 3.5, 5, 8, 12, 20, 40]),
        }
        print(f"    nearest-other-component gap, {len(gaps)} largest components")
        print("    " + _q(ev["endpoint_gaps"]["quantiles_m"]))
        for b in ev["endpoint_gaps"]["hist_m"]:
            hi = f"<{b['hi']}" if b["hi"] else "+"
            print(f"      {b['lo']:>4} {hi:<6} {b['n']:>5}  " + "#" * min(60, b["n"]))

    with ev.stage("mask_sensitivity"):
        # Entirely from the committed tile. `min_tpi_sd` is a cut in standard
        # deviations of the TPI **over the masked area**, and this tile is 39% sea
        # with a 15 m coastal scarp around the land. Eroding the land boundary
        # inward drops that scarp out of the sample, so the sd falls, so the cut
        # falls — and the table says by how much, without needing the design.
        from scipy.ndimage import binary_erosion

        land = np.isfinite(ka.dem)
        sens = []
        print_summary_header()
        row = summarise(ka, None, "none (production)")
        sens.append(row)
        print_summary(row)
        for metres in (25, 50, 100, 150):
            cells = int(round(metres / ka.cell_size))
            inner = binary_erosion(land, structure=np.ones((3, 3), dtype=bool),
                                   iterations=cells)
            if not inner.any():
                print(f"    land eroded {metres} m      (nothing left)")
                continue
            row = summarise(ka, inner, f"land eroded {metres} m")
            sens.append(row)
            print_summary(row)
        for r in sens:
            r.pop("_sk", None)
        ev["mask_sensitivity"] = sens

    with ev.stage("production_mask"):
        # The configuration the finding was actually reported against. The number
        # in the register — 1,462 components — came from a run with the design's
        # area polygon, and this probe's 857 came from a run without one. Those are
        # not the same measurement, and reporting a diagnosis off the wrong one
        # would be the "measure on the real design" rule failing in a new costume.
        mask, label = design_mask(ka, dem_path)
        ev["production_mask"] = {"label": label, "available": mask is not None}
        if mask is None:
            print(f"    skipped — {label}")
            ev.note(f"production mask unavailable ({label}); every number above is "
                    f"the no-mask configuration, which is NOT what the button runs")
        else:
            row = summarise(ka, mask, label)
            sk_p = row.pop("_sk")
            ev["production_mask"].update(row)
            print_summary_header()
            print_summary(row)
            rows_p = component_table(ka, sk_p["labeled"], sk_p["n"])
            ev["production_mask"]["distribution"] = describe(
                rows_p, f"{label}, 8-connected (production)")
            rows_p4 = component_table(ka, sk_p["labeled4"], sk_p["n4"])
            ev["production_mask"]["distribution_4connected"] = describe(
                rows_p4, f"{label}, 4-connected (was)")
            ev["production_mask"]["sweep"] = sweep_bar(
                rows_p, ka, f"{label}, 8-connected (production)")
            ev["production_mask"]["sweep_4connected"] = sweep_bar(
                rows_p4, ka, f"{label}, 4-connected (was)")
            ev["production_mask"]["renders"] = [
                str(p) for p in render(ka, sk_p, rows_p, work, prefix="design_")]
            lines = ka.find_ridgelines(boundary_mask=mask)
            ev["production_mask"]["production_lines"] = len(lines)
            print(f"    find_ridgelines(boundary_mask=design) -> {len(lines)} lines")

    with ev.stage("connectivity_oracle"):
        # Two synthetic shapes with a known answer, so the claim about `nd_label`
        # rests on a demonstration rather than on a reading of scipy's docs. A
        # diagonal ridge and a staircase are both **one** line by any definition,
        # and both are what `skeletonize` produces wherever a ridge does not run
        # along a grid axis.
        from scipy.ndimage import generate_binary_structure
        from scipy.ndimage import label as nd_label

        diag = np.zeros((12, 12), dtype=bool)
        for i in range(10):
            diag[i + 1, i + 1] = True
        stair = np.zeros((12, 20), dtype=bool)
        r, c = 2, 1
        for _ in range(6):
            stair[r, c] = stair[r, c + 1] = True
            c += 2
            r += 1
        eight = np.ones((3, 3), dtype=int)
        oracle = {
            "default_structure": generate_binary_structure(2, 1).astype(int).tolist(),
            "diagonal_10_cells": {"as_production": nd_label(diag)[1],
                                  "eight_connected": nd_label(diag, eight)[1]},
            "staircase_6_runs": {"as_production": nd_label(stair)[1],
                                 "eight_connected": nd_label(stair, eight)[1]},
        }
        ev["connectivity_oracle"] = oracle
        print(f"    scipy's default structure is the 4-connected cross: "
              f"{oracle['default_structure']}")
        for name, o in (("a 10-cell diagonal ridge", oracle["diagonal_10_cells"]),
                        ("a 6-step staircase", oracle["staircase_6_runs"])):
            print(f"    {name:<26} -> {o['as_production']} components as "
                  f"production labels it, {o['eight_connected']} 8-connected")
        if oracle["diagonal_10_cells"]["as_production"] > 1:
            ev.note("a diagonal ridge of N cells becomes N single-cell components "
                    "under `nd_label(skeleton)`; `_order_pixels` would have walked "
                    "it as one line")

    with ev.stage("acc_bar"):
        # Which of the two terms breaks the line? `ridge_raw` is
        # ``above & (acc <= 2) & valid``, and the crop render shows a scatter of
        # clumps rather than a line with gaps in it — so the question is whether
        # the TPI cut is picking a dotted set, or whether ``acc <= 2`` is cutting a
        # continuous crest into the cells that happen to be flow origins. A crest
        # is not level: water runs *along* it, so accumulation climbs as you walk
        # it, and a hard bar of 2 keeps only the first cell or two of each run.
        # ``acc_bar=None`` drops the term entirely and is the ceiling.
        mask_p = None
        label_p = "none"
        try:
            mask_p, label_p = design_mask(ka, dem_path)
        except Exception as exc:
            print(f"    (no design mask: {exc})")
        rows_acc = []
        print_summary_header()
        for bar in (1, 2, 3, 5, 10, 20, 50, 100, float("inf")):
            name = "acc<=inf (TPI only)" if bar == float("inf") else f"acc<={bar}"
            r = summarise(ka, mask_p, name, acc_bar=bar)
            r.pop("_sk", None)
            rows_acc.append(r)
            print_summary(r)
        ev["acc_bar"] = {"mask": label_p, "rows": rows_acc}

    with ev.stage("candidate_bars"):
        # The owner's decision is connectivity **and** a relaxed accumulation bar.
        # This stage simulates the fixed function — 8-connected labelling, the
        # accumulation term as a catchment **area** — and reports the lines it
        # would actually draw, at the unchanged `min_length_m=50`. The number to
        # choose is not "which bar draws most" but which draws ridges: a bar so
        # loose that the TPI class is all that is left stops selecting divides.
        mask_c, label_c = design_mask(ka, dem_path)
        if mask_c is None:
            print(f"    skipped — {label_c}")
        else:
            cand = []
            for m2 in (2, 5, 10, 20, 30, 50):
                lines = simulate_fixed(ka, mask_c, m2)
                lens = sorted((ln["length_m"] for ln in lines), reverse=True)
                cand.append({
                    "max_catchment_m2": m2,
                    "acc_cells_equivalent": m2 / (ka.cell_w * ka.cell_h),
                    "lines": len(lines),
                    "longest_m": lens[0] if lens else 0.0,
                    "median_m": lens[len(lens) // 2] if lens else 0.0,
                    "shortest_m": lens[-1] if lens else 0.0,
                    "total_m": float(sum(lens)),
                })
            ev["candidate_bars"] = cand
            print(f"    {label_c}, 8-connected, min_length_m={MIN_LENGTH_M}")
            print("    max_catchment_m2  acc<=  lines  longest_m  median_m  "
                  "shortest_m  total_m")
            for c in cand:
                print(f"      {c['max_catchment_m2']:>14}  "
                      f"{c['acc_cells_equivalent']:>5.0f}  {c['lines']:>5}  "
                      f"{c['longest_m']:>9.1f}  {c['median_m']:>8.1f}  "
                      f"{c['shortest_m']:>10.1f}  {c['total_m']:>7.0f}")
            ev["candidate_renders"] = [
                str(p) for p in render_lines(ka, mask_c, (5, 10, 20), work)]
            for p in ev["candidate_renders"]:
                print(f"    {p}")

    with ev.stage("render"):
        pngs = render(ka, sk, rows, work)
        ev["renders"] = [str(p) for p in pngs]
        for p in pngs:
            print(f"    {p}")

    with ev.stage("production_call"):
        # And the honest end-to-end: what the button returns today.
        lines = ka.find_ridgelines()
        ev["production_lines"] = len(lines)
        print(f"    find_ridgelines() with production defaults -> {len(lines)} lines")

    ev.write()


def _q(d):
    return "  ".join(f"{k}={v:.1f}" for k, v in d.items())


def render(ka, sk, rows, work, prefix=""):
    """Photograph the raw mask, the skeleton, and a crop around the biggest piece.

    The picture is half the evidence here. "1,462 components" is compatible with a
    lawn of 8-cell specks and with 30 ridges broken into 50 pieces each, and those
    two want opposite fixes; an image separates them in a glance where a quantile
    table has to be argued over.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = []

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    hill = np.where(np.isfinite(ka.dem), ka.dem, np.nan)
    axes[0].imshow(hill, cmap="terrain")
    axes[0].set_title(f"DEM {ka.dem.shape[0]}x{ka.dem.shape[1]}")
    axes[1].imshow(hill, cmap="gray", alpha=0.6)
    axes[1].imshow(np.where(sk["ridge_raw"], 1.0, np.nan), cmap="autumn",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"ridge_raw — {int(sk['ridge_raw'].sum()):,} cells")
    axes[2].imshow(hill, cmap="gray", alpha=0.6)
    axes[2].imshow(np.where(sk["skeleton"], 1.0, np.nan), cmap="cool",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"skeleton — {sk['n']:,} components")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    p = work / f"{prefix}overview.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    out.append(p)

    # The skeleton on its own, full size and unsmoothed. Downsampling a 1-px-wide
    # skeleton erases it, so this is written at one image pixel per DEM cell.
    h, w = sk["skeleton"].shape
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(sk["skeleton"], cmap="binary", interpolation="nearest")
    ax.set_axis_off()
    p = work / f"{prefix}skeleton_full.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    out.append(p)

    # A 200x200 crop centred on the largest component, at 4x, so individual cells
    # and the gaps between components are separately visible.
    if rows:
        big = max(rows, key=lambda r: r["cells"])
        r0, c0, bh, bw = big["box"]
        cr = int(r0 + bh / 2)
        cc = int(c0 + bw / 2)
        half = 100
        rs = slice(max(0, cr - half), min(h, cr + half))
        cs = slice(max(0, cc - half), min(w, cc + half))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(hill[rs, cs], cmap="terrain", interpolation="nearest")
        axes[0].imshow(np.where(sk["ridge_raw"][rs, cs], 1.0, np.nan),
                       cmap="autumn", vmin=0, vmax=1, interpolation="nearest",
                       alpha=0.8)
        axes[0].set_title(f"ridge_raw, crop around component {big['id']}")
        axes[1].imshow(hill[rs, cs], cmap="terrain", interpolation="nearest")
        axes[1].imshow(np.where(sk["skeleton"][rs, cs], 1.0, np.nan),
                       cmap="cool", vmin=0, vmax=1, interpolation="nearest")
        axes[1].set_title(f"skeleton, same crop — biggest is "
                          f"{big['cells']} cells / {big['length_m']:.0f} m")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        p = work / f"{prefix}crop_largest.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(p)

    return out


if __name__ == "__main__":
    main()
