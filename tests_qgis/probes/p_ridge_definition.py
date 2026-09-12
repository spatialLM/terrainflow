"""What *is* a ridge? Three definitions, priced against each other on real ground.

`bfc1a3b` fixed the Ridgelines button by repairing a connectivity bug and replacing a
cell-count threshold with an area. It changed **thresholds**, and the owner deferred the
larger question at the time: "the definition of the ridge needs to be revisited at a
later date". This is that revisit, and it is deliberately a *comparison*, not a change —
what counts as a ridge is a decision about the domain, and the useful thing an engineer
can do is make it decidable.

`find_ridgelines` currently defines a ridge as **convex ground that sheds nearly all its
own water**: `landform_classes(TPI) == 1` AND a contributing area under
`max_catchment_m2`. That is a proxy chosen for being computable off a DEM, and it has
two known weaknesses, both visible in the numbers below:

* **One scale.** A 15 m window cannot tell a 15 m bump from a catchment divide, so the
  cut has to lean on the accumulation term to throw the bumps away.
* **It is a statement about a cell, not about a line.** Nothing in the definition knows
  that a divide is *continuous*, which is why the output arrives as fragments and needs
  a length filter to become usable.

The two candidates the docstring names:

**A — the divide from the flow field.** A ridge is where water parts. Negate the
terrain, fill its (now inverted) pits and run drainage on it: the streams of the
upside-down surface are the divides of the real one. This is a statement about
connectivity by construction — a drainage network is continuous — so it should not
fragment.

**B — multi-scale TPI.** Keep convexity, ask it at several window sizes, and require the
cell to be convex at all of them. A 15 m bump fails the 60 m question; a catchment
divide passes every one.

Every definition is pushed through **the same tail as production** — thin, label
8-connected, drop components under `min_length_m`, order, vectorise — so the comparison
is of definitions and not of pipelines.

Run: & F:\\bin\\python-qgis-ltr.bat tests_qgis\\probes\\p_ridge_definition.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _probe  # noqa: E402,F401  (sets sys.path for the two below)

import p_ridgelines as PR  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from terrainflow_assessment.modules.keypoint_analysis import (  # noqa: E402
    DrainageLineAnalysis,
    _thin_to_centreline,
)
from terrainflow_assessment.modules.terrain_indices import (  # noqa: E402
    landform_classes,
    landform_tpi,
)

#: Production defaults, restated from `find_ridgelines`' signature.
TPI_WINDOW_M = PR.TPI_WINDOW_M
MIN_TPI_SD = PR.MIN_TPI_SD
MIN_LENGTH_M = PR.MIN_LENGTH_M
MAX_CATCHMENT_M2 = 20.0

#: Candidate B's window ladder. 15 m is production's; 30 and 60 are the scales at which
#: a farm-sized divide is still convex and a bank or a terrace tread is not.
MULTISCALE_WINDOWS_M = (15.0, 30.0, 60.0)

#: Candidate A's bar, as contributing area on the INVERTED surface — i.e. how much
#: ground has to drain *away* from a line before it counts as a divide. Swept, because
#: nobody knows the right value and the shape of the curve is the finding.
INVERTED_BARS_M2 = (200.0, 500.0, 1000.0, 2000.0, 5000.0)


# ---------------------------------------------------------------------------
# The shared tail — production's, so the comparison is of definitions only
# ---------------------------------------------------------------------------

def lines_from_mask(ka, mask, min_length_m=MIN_LENGTH_M):
    """`find_ridgelines`' tail, from `ridge_raw` onward. Mirrors the source."""
    from scipy.ndimage import find_objects
    from scipy.ndimage import label as nd_label

    mask = np.asarray(mask, dtype=bool).copy()
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    if not mask.any():
        return []

    skeleton = _thin_to_centreline(mask)
    if not skeleton.any():
        skeleton = mask

    labeled, n = nd_label(skeleton, structure=np.ones((3, 3), dtype=int))
    min_cells = max(3, int(min_length_m / ka.cell_size))
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    boxes = find_objects(labeled)

    out = []
    for rid in range(1, n + 1):
        if sizes[rid] < min_cells:
            continue
        box = boxes[rid - 1]
        if box is None:
            continue
        rc = np.argwhere(labeled[box] == rid)
        rc += (box[0].start, box[1].start)
        ordered = ka._order_pixels(rc.tolist())
        if len(ordered) < 2:
            continue
        try:
            geom = LineString([ka._rc_to_xy(r, c) for r, c in ordered])
        except Exception:
            continue
        out.append({"geometry": geom, "length_m": float(geom.length)})

    out.sort(key=lambda ln: ln["length_m"], reverse=True)
    return out[:30]


def describe(label, mask, lines, total_cells):
    lengths = [ln["length_m"] for ln in lines]
    return {
        "definition": label,
        "ridge_cells": int(np.count_nonzero(mask)),
        "ridge_cells_pct": round(100.0 * np.count_nonzero(mask) / total_cells, 3),
        "lines": len(lines),
        "total_length_m": round(sum(lengths), 1),
        "median_length_m": round(float(np.median(lengths)), 1) if lengths else 0.0,
        "max_length_m": round(max(lengths), 1) if lengths else 0.0,
    }


# ---------------------------------------------------------------------------
# The three definitions
# ---------------------------------------------------------------------------

def mask_current(ka, boundary_mask=None):
    """Production: convex at one scale AND shedding nearly all its own water."""
    valid = np.isfinite(ka.dem)
    tpi = landform_tpi(ka.dem, ka.cell_w, ka.cell_h, window_m=TPI_WINDOW_M)
    with np.errstate(invalid="ignore"):
        above = landform_classes(tpi, sd=MIN_TPI_SD, mask=boundary_mask) == 1
        acc_bar = max(1.0, MAX_CATCHMENT_M2 / (ka.cell_w * ka.cell_h))
        mask = above & (ka.acc <= acc_bar) & valid
    if boundary_mask is not None:
        mask &= boundary_mask
    return mask


def mask_multiscale(ka, boundary_mask=None, windows=MULTISCALE_WINDOWS_M):
    """Candidate B: convex at EVERY window. No accumulation term at all.

    Dropping the accumulation term is the point — if convexity at three scales is a
    good definition it should not need a second rule to rescue it, and keeping one
    would make this a variation on production rather than an alternative to it.
    """
    valid = np.isfinite(ka.dem)
    mask = valid.copy()
    per_scale = {}
    for w in windows:
        tpi = landform_tpi(ka.dem, ka.cell_w, ka.cell_h, window_m=w)
        with np.errstate(invalid="ignore"):
            this = landform_classes(tpi, sd=MIN_TPI_SD, mask=boundary_mask) == 1
        per_scale[w] = int(np.count_nonzero(this & valid))
        mask &= this
    if boundary_mask is not None:
        mask &= boundary_mask
    return mask, per_scale


def mask_inverted_drainage(ka, dem_path, work, bar_m2, boundary_mask=None):
    """Candidate A: the streams of the upside-down terrain.

    Water on an inverted surface runs along the real surface's divides, so a drainage
    network computed there *is* a divide network — continuous by construction, which is
    the property the TPI definition cannot state.
    """
    import rasterio

    dem = ka.dem
    finite = np.isfinite(dem)
    if not finite.any():
        return None, None

    # Negate about the site's own maximum so the result stays positive and the nodata
    # sentinel cannot be mistaken for terrain.
    top = float(np.nanmax(dem[finite]))
    inverted = np.where(finite, top - dem, np.nan).astype("float32")

    path = os.path.join(str(work), f"inverted_{int(bar_m2)}.tif")
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.where(finite, inverted, -9999.0).astype("float32"), 1)

    acc_path, _pond = PR.accumulation(path, work)
    if acc_path is None:
        return None, None
    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float32")

    bar_cells = max(1.0, bar_m2 / (ka.cell_w * ka.cell_h))
    mask = (acc >= bar_cells) & finite
    if boundary_mask is not None:
        mask &= boundary_mask
    return mask, acc


# ---------------------------------------------------------------------------

def render(ka, panels, work, name):
    """Draw each definition over the terrain.

    Counts and quantiles cannot answer "are these ridges?" — they can say one set of
    lines is longer and better connected than another, which is what the table below
    does, but not whether the lines are in the right places. That is a judgement about
    the ground, and it needs a picture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from terrainflow_assessment.modules.footprint import xy_to_rc

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 7))
    axes = np.atleast_1d(axes)
    hill = np.where(np.isfinite(ka.dem), ka.dem, np.nan)
    for ax, (label, lines) in zip(axes, panels):
        ax.imshow(hill, cmap="terrain")
        for ln in lines:
            rc = [xy_to_rc(ka.transform, x, y) for x, y in ln["geometry"].coords]
            ax.plot([c for _r, c in rc], [r for r, _c in rc],
                    color="magenta", linewidth=1.4)
        ax.set_title(f"{label}\n{len(lines)} lines, "
                     f"{sum(ln['length_m'] for ln in lines):,.0f} m total")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(str(work), name)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"    rendered {out}")
    return out


def agreement(a, b):
    """Jaccard of two ridge masks — how much the two definitions pick the same ground."""
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return round(100.0 * inter / union, 1) if union else 0.0


def main():
    _probe.start_qgis()
    work = _probe.workdir("ridge_definition")
    dem_path = str(PR.FULL_TILE)
    print(f"DEM: {dem_path}")

    acc_path, pond_path = PR.accumulation(dem_path, work)
    ka = DrainageLineAnalysis(dem_path, acc_path, pond_path)
    total = ka.dem.size
    print(f"grid {ka.dem.shape[0]} x {ka.dem.shape[1]} = {total:,} cells, "
          f"cell {ka.cell_w} x {ka.cell_h} m")

    # BOTH runs. Unmasked is a different question, not a weaker version of the
    # production one: `min_tpi_sd` cuts in standard deviations of the TPI *inside the
    # mask*, and on this tile the unmasked sd is 0.732 m — which puts the ridge cells
    # on the coastal scarp and leaves the farm blank. Production is handed the design's
    # own area polygon by `contour.py`, so the masked run is the one to read.
    design, why = PR.design_mask(ka, dem_path)
    scopes = [("whole tile (unmasked)", None)]
    if design is not None:
        scopes.append((f"design area ({why})", design))
    else:
        print(f"  design mask unavailable: {why}")

    everything = {}
    for scope_label, boundary in scopes:
        rows = []
        masks = {}
        in_scope = total if boundary is None else int(np.count_nonzero(boundary))

        drawn = {}

        m = mask_current(ka, boundary)
        masks["current"] = m
        drawn["current"] = lines_from_mask(ka, m)
        rows.append(describe("current (TPI@15m + acc<=20m2)", m,
                             drawn["current"], in_scope))

        mb, per_scale = mask_multiscale(ka, boundary)
        masks["multiscale"] = mb
        drawn["multiscale"] = lines_from_mask(ka, mb)
        row = describe(f"multi-scale TPI {MULTISCALE_WINDOWS_M}", mb,
                       drawn["multiscale"], in_scope)
        row["cells_per_scale"] = {str(k): v for k, v in per_scale.items()}
        rows.append(row)

        for bar in INVERTED_BARS_M2:
            ma, _acc = mask_inverted_drainage(ka, dem_path, work, bar, boundary)
            if ma is None:
                print("  inverted drainage unavailable (no accumulation backend)")
                break
            masks[f"inverted@{int(bar)}"] = ma
            drawn[f"inverted@{int(bar)}"] = lines_from_mask(ka, ma)
            rows.append(describe(f"inverted drainage, bar {bar:,.0f} m2", ma,
                                 drawn[f"inverted@{int(bar)}"], in_scope))

        print("")
        print(f"================ {scope_label} — {in_scope:,} cells in scope")
        hdr = (f"{'definition':40s} {'cells':>9s} {'%scope':>7s} {'lines':>6s} "
               f"{'total m':>9s} {'median':>8s} {'max':>8s}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['definition']:40s} {r['ridge_cells']:9,d} "
                  f"{r['ridge_cells_pct']:7.3f} {r['lines']:6d} "
                  f"{r['total_length_m']:9,.0f} {r['median_length_m']:8.1f} "
                  f"{r['max_length_m']:8.1f}")

        print("")
        print("  agreement (Jaccard %, shared ridge cells):")
        names = list(masks)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"    {a:22s} vs {b:22s} {agreement(masks[a], masks[b]):5.1f}%")
        try:
            panels = [("current: TPI@15 m + acc<=20 m2", drawn["current"]),
                      ("multi-scale TPI 15/30/60 m", drawn["multiscale"])]
            if "inverted@500" in drawn:
                panels.append(("inverted drainage, 500 m2", drawn["inverted@500"]))
            slug = "masked" if boundary is not None else "whole_tile"
            render(ka, panels, work, f"ridge_definitions_{slug}.png")
        except Exception as exc:
            print(f"    render skipped: {exc}")

        everything[scope_label] = {
            "cells_in_scope": in_scope,
            "rows": rows,
            "agreement": {f"{a}|{b}": agreement(masks[a], masks[b])
                          for i, a in enumerate(names) for b in names[i + 1:]},
        }

    out = {
        "probe": "p_ridge_definition",
        "dem": dem_path,
        "grid": list(ka.dem.shape),
        "scopes": everything,
    }
    dest = os.path.join(HERE, "evidence", "p_ridge_definition.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
