"""
burn_strategy.py — pure Strategy-C DEM-burning algorithms.

Strategy C represents a sub-cell earthwork by its hydraulic *effect* on the native
grid, never by resampling the DEM finer (see CLudeDocs/STRETCH_GOALS.md for why
interpolative refinement — Strategy B — is deferred). These helpers are the pure,
grid-only building blocks that ``DEMBurner`` (in ``earthwork_design.py``) orchestrates:

    bresenham            — integer cells along a straight segment
    line_cells           — connected, in-bounds cell path along a polyline
    enforce_monotonic_path — breach a strictly-downhill invert along a carved path
    level_invert         — excavate a footprint to a flat floor below its spill level
    battered_invert      — stepped approximation of battered walls
    rasterisable_capacity — storage the grid can actually represent (resolution penalty)
    steep_ground_warning — advisory when a level floor over-excavates one end
    sub_cell_warning     — advisory when a feature is narrower than one cell
    ponding_resolution_warning — advisory when a DEM exceeds the ponding memory cap

Design rules baked in (spec §3):
* **Storage features get a level floor** (``level_invert``), referenced to their
  natural pour point, so a basin or swale holds its design volume on sloping ground
  instead of the wedge a constant-depth translation leaves behind.
* Conveyances — and only conveyances — get a **connected, monotonic downhill 1-cell
  path** by breaching (``enforce_monotonic_path``), so depression-filling can't erase
  the drain and flow actually reaches the outlet. Applying this to a swale guaranteed
  an outlet and therefore near-zero ponding, which is why swales no longer use it.
* The empty-mask no-op (a sub-cell buffer rasterises to nothing) is fixed by a
  **nearest-cell snap**: ``line_cells`` always yields the in-bounds cells the line
  passes through, which the burner incises directly when the area mask is empty.
* Never downsample below native resolution — ``ponding_resolution_warning`` surfaces
  the memory-cap degrade instead of letting it happen silently.

All functions are pure (grid indices + numpy arrays in, values out) and unit-agnostic.
"""

from __future__ import annotations


def bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer (row, col) cells along the segment (r0, c0) → (r1, c1), endpoints included."""
    cells: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr
    return cells


def line_cells(coords, transform, shape) -> list[tuple[int, int]]:
    """Connected, in-bounds cell path along a polyline.

    ``coords`` is an ordered list of (x, y) map coordinates; ``transform`` is the DEM
    affine (``.a``/``.e``/``.c``/``.f``, no rotation, as elsewhere in the burner);
    ``shape`` is ``(rows, cols)``. Consecutive vertices are joined with Bresenham so
    the path is 8-connected, then clipped to the raster and de-duplicated (order
    preserved). Vertices outside the extent are dropped — a feature entirely off the
    DEM yields an empty path (no spurious edge-cell burn); a within-extent sub-cell
    feature still yields ≥ 1 cell (the nearest-cell snap).
    """
    rows, cols = shape
    a, e, c0, f = transform.a, transform.e, transform.c, transform.f

    idx: list[tuple[int, int]] = []
    for x, y in coords:
        col = int((x - c0) / a)
        row = int((y - f) / e)
        idx.append((row, col))

    raw: list[tuple[int, int]] = []
    if len(idx) == 1:
        raw = [idx[0]]
    else:
        for k in range(len(idx) - 1):
            for cell in bresenham(idx[k][0], idx[k][1], idx[k + 1][0], idx[k + 1][1]):
                if raw and raw[-1] == cell:
                    continue
                raw.append(cell)

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in raw:
        if 0 <= r < rows and 0 <= c < cols and (r, c) not in seen:
            seen.add((r, c))
            out.append((r, c))
    return out


def enforce_monotonic_path(dem, path_cells, min_drop: float = 1e-3):
    """Breach a strictly-downhill invert along *path_cells* (Lindsay-style).

    Returns a copy of *dem* in which the carved path is guaranteed to descend
    monotonically from its higher end to its lower end, cutting only the humps that
    would otherwise pond the drain (never raising a cell, never a blanket deep cut).
    A path of fewer than two cells is returned unchanged (nothing to connect).

    ``min_drop`` is the minimum fall (in DEM units) enforced per step — a small
    positive value so the invert strictly descends even across originally-flat ground.
    """
    out = dem.copy()
    if len(path_cells) < 2:
        return out

    # Orient the walk so we start at the higher end and descend.
    if out[path_cells[0]] >= out[path_cells[-1]]:
        cells = list(path_cells)
    else:
        cells = list(reversed(path_cells))

    running = out[cells[0]]
    for rc in cells[1:]:
        ceiling = running - min_drop
        target = out[rc] if out[rc] <= ceiling else ceiling
        out[rc] = target
        running = target
    return out


def level_invert(dem, mask, depth: float, spill_elev: float):
    """Excavate *mask* to a **flat floor** at ``spill_elev − depth``.

    Returns a copy of *dem* in which every masked cell sits at or below the floor —
    ``np.minimum``, so ground already lower than the floor is never raised and the
    burn can only ever cut.

    This replaces a constant-depth offset (``dem[mask] -= depth``), which translated
    the existing terrain downward and so preserved its slope. A translated basin only
    holds water up to its lowest rim, and on sloping ground that is far less than its
    design volume — measured on a 400 m² × 1.5 m basin (600 m³ nominal): 600 m³ held
    on flat ground, 390 m³ at 5% slope, 210 m³ at 10%. A level floor holds 600 m³ at
    every slope, which is what the analytic prism claims, so the two tiers agree.

    Referencing the floor to *spill_elev* (the natural pour point — the lowest rim
    cell) rather than to the ground directly above it is what makes the pond fill
    evenly to the level at which it would actually overflow.
    """
    import numpy as np

    out = dem.copy()
    if mask is None or not mask.any():
        return out
    floor = spill_elev - depth
    out[mask] = np.minimum(out[mask], floor)
    return out


def battered_invert(dem, step_masks, spill_elev: float):
    """Excavate nested *step_masks* to a stepped approximation of battered walls.

    ``step_masks`` is an ordered list of ``(mask, depth)`` pairs, outermost/shallowest
    first, as produced by successively shrinking the footprint. Each step is levelled
    to ``spill_elev − depth`` deepest-last, so the deeper inner steps win. This is how
    a trapezoidal or battered section is represented on a grid — the walls become
    stairs, and how well they approximate the true batter is exactly what
    :func:`rasterisable_capacity` reports.
    """
    out = dem.copy()
    for mask, depth in sorted(step_masks, key=lambda pair: pair[1]):
        out = level_invert(out, mask, depth, spill_elev)
    return out


def rasterisable_capacity(n_cells: int, cell_area: float, depth: float,
                          top_width: float, bottom_width: float, cell_size: float):
    """Storage the burned raster can actually represent, in m³.

    The design geometry and the grid rarely agree, and the difference is not an
    error — it is the resolution penalty, and it is usually **positive**. A 2.0 m
    swale with 1:1 batters has a 0.75 m²/m trapezoidal section, but on a 1 m grid the
    sloping walls cannot be represented: it burns as a 2-cell rectangular trench at
    1.0 m²/m, a third more. A sub-metre feature is widened to one whole cell, likewise.

    Reporting this separately is what lets the Verify stage compare like with like:
    measured ponding against *this* number isolates burn correctness, while the gap
    between this and the true geometric volume is the honest cost of the cell size.

    A footprint wide enough to hold the batter (roughly three cells) keeps its
    trapezoidal section; anything narrower collapses to the rectangle the grid can
    hold.
    """
    if n_cells <= 0 or depth <= 0 or cell_area <= 0:
        return 0.0
    if cell_size > 0 and top_width < 3.0 * cell_size:
        # Too narrow for the walls to be resolved — a flat-bottomed trench.
        return n_cells * cell_area * depth
    mean_width = (top_width + max(0.0, bottom_width)) / 2.0
    if top_width <= 0:
        return n_cells * cell_area * depth
    return n_cells * cell_area * depth * (mean_width / top_width)


def steep_ground_warning(name: str, relief: float, depth: float,
                         cut_m3=None, storage_m3=None):
    """Advisory when a level floor means cutting deeper than the design depth.

    A level floor is hydraulically right at any slope — it fills evenly to its spill
    level and holds its design volume. The cost is excavation: if the footprint falls
    further across than the feature is deep, the uphill end is cut deeper than asked
    for, and on a 10% slope that can be ~70% more earth moved for the same storage.
    The user should be told in cubic metres, not warned off in the abstract.
    """
    if relief is None or depth <= 0 or relief <= depth:
        return None
    msg = (
        f"'{name}': the footprint falls {relief:.1f} m across, more than its "
        f"{depth:.1f} m design depth. A level floor means cutting up to {relief:.1f} m "
        f"at the uphill end"
    )
    if cut_m3 and storage_m3:
        msg += f" — {cut_m3:,.0f} m³ of excavation for {storage_m3:,.0f} m³ of storage"
    msg += (
        ". Consider a smaller footprint, terracing into two features, or a dam wall "
        "on the low side."
    )
    return msg


def sub_cell_warning(name: str, min_dimension, cell_size: float):
    """Advisory string when a feature is narrower than one DEM cell, else ``None``.

    Below the native cell size a feature can only be represented at 1-cell width — a
    routing effect, not true hydraulics (spec §3 UI-honesty). ``min_dimension`` comes
    from the sizing result (``core.sizing``); ``None`` skips the check.
    """
    if min_dimension is not None and cell_size > 0 and min_dimension < cell_size:
        return (
            f"'{name}': narrowest dimension {min_dimension:.2f} m is below the DEM cell "
            f"size {cell_size:.2f} m — represented at 1-cell width (routing effect only, "
            f"not true hydraulics)."
        )
    return None


def ponding_resolution_warning(n_cells: int, max_cells: int):
    """Advisory string when a DEM exceeds the ponding memory cap, else ``None``.

    The cap is a memory guard, not a resolution choice: when it trips, ponding is
    computed on a coarsened copy. Surface that rather than degrading silently — and
    never upsample below native resolution (that is deferred Strategy B).
    """
    if n_cells > max_cells:
        return (
            f"DEM has {n_cells:,} cells (over the {max_cells:,}-cell ponding cap); ponding "
            f"is computed at reduced resolution and is approximate. Features near the cap "
            f"may be under-resolved (the DEM is never upsampled below native resolution)."
        )
    return None
