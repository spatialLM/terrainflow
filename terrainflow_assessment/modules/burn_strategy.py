"""
burn_strategy.py — pure Strategy-C DEM-burning algorithms.

Strategy C represents a sub-cell earthwork by its hydraulic *effect* on the native
grid, never by resampling the DEM finer (see CLudeDocs/STRETCH_GOALS.md for why
interpolative refinement — Strategy B — is deferred). These helpers are the pure,
grid-only building blocks that ``DEMBurner`` (in ``earthwork_design.py``) orchestrates:

    bresenham            — integer cells along a straight segment
    line_cells           — connected, in-bounds cell path along a polyline
    enforce_monotonic_path — breach a strictly-downhill invert along a carved path
    sub_cell_warning     — advisory when a feature is narrower than one cell
    ponding_resolution_warning — advisory when a DEM exceeds the ponding memory cap

Design rules baked in (spec §3):
* Conveyances get a **connected, monotonic downhill 1-cell path** by breaching
  (``enforce_monotonic_path``), not naive deep incision — so depression-filling
  can't erase the drain and flow actually reaches the outlet.
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
