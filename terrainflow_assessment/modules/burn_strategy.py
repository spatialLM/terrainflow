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
    spillway_notch       — cut a designed sill through a bank, or refuse and say why
    daylight_reach       — march a crest run outward until it finds ground below it
    notch_pool           — the pond a notch must NOT discharge back into
    spillway_burn_width  — the built width rounded up to whole DEM cells
    tapered_invert       — excavate a footprint as its true battered section
    rasterisable_capacity — storage the grid can represent, modelled (see its docstring:
                           prefer ``DEMBurner.feature_storage_m3``, which measures it)
    steep_ground_warning — advisory when a level floor over-excavates one end
    sub_cell_warning     — advisory when a feature is narrower than one cell
    ponding_resolution_warning — advisory when a DEM exceeds the ponding memory cap

Design rules baked in (spec §3):
* **Storage features get a level floor** (``level_invert``), referenced to their
  natural pour point, so a basin or swale holds its design volume on sloping ground
  instead of the wedge a constant-depth translation leaves behind. Level along the
  feature — but across it the floor follows the **drawn section** (``tapered_invert``),
  so a trapezoidal channel is cut as a trapezoid rather than squared off to a
  full-depth rectangle a third to a half larger than it was specified.
* The spill datum is read from the feature's **own** ground, never from a DEM other
  features have already been burned into: it is a single minimum over a one-cell rim,
  so one neighbouring cell that has been cut takes the whole floor down with it.
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

from typing import NamedTuple

from terrainflow_assessment.modules.footprint import xy_to_rc


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

    idx: list[tuple[int, int]] = [
        xy_to_rc(transform, x, y) for x, y in coords
    ]

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

    *spill_elev* must be read from the feature's **own** ground — the original terrain
    plus whatever this feature builds on it — and never from a DEM other features have
    already been burned into. The datum is a single minimum over a one-cell-wide rim, so
    one neighbouring cell that has been cut lowers the whole floor by that cut's depth:
    Swale 27 on the Quail Island design shares exactly one rim cell with Swale 25, was
    floored 0.66 m too low because of it, and ponded 1.94 m for a 1.00 m design. The
    caller owns that choice; see ``DEMBurner._storage_invert``, whose ``spill`` is taken
    with ``pour_point`` on ``self.original`` rather than on the running array.
    """
    import numpy as np

    out = dem.copy()
    if mask is None or not mask.any():
        return out
    floor = spill_elev - depth
    out[mask] = np.minimum(out[mask], floor)
    return out


def _axis_spacing(cell_size):
    """``cell_size`` as ``(row spacing, column spacing)`` in metres.

    Accepts a scalar for the square case every caller used to assume, or a pair
    for a grid whose cells are not square.
    """
    try:
        cell_h, cell_w = cell_size
    except TypeError:
        cell_h = cell_w = cell_size
    return abs(float(cell_h)), abs(float(cell_w))


def taper_reach(mask, batter_run: float, cell_size=1.0):
    """Per-cell fraction of full depth for a battered section, over *mask*.

    ``0`` at the footprint edge, ``1`` once a cell is ``batter_run`` inside it, linear
    between — so ``depth × reach`` is the trapezoid's depth at that offset. Returns
    ``None`` when there is no batter to apply or scipy is unavailable, meaning "cut it
    flat".

    Shared by :func:`tapered_invert` and the companion-berm spoil calculation, because
    the bank is built from the earth the trench produced and those two must agree about
    how much that is. They did not for one revision: the berm was sized from a
    full-depth rectangular cut while the trench was tapered, so it was handed roughly
    half again the spoil the excavation actually yields.

    On an axis-aligned footprint the result integrates to the drawn section exactly —
    it is a midpoint-rule quadrature of a piecewise-linear profile — whenever
    ``batter_run`` is a whole multiple of the crossing axis's spacing. Off that, the
    kink at full depth falls inside a cell rather than on its edge and the quadrature
    over-reads: about +7% at ``batter_run`` 1.5 m on metre cells, and +33% for the
    registry's default swale, whose 0.5 m run is half a cell and leaves no taper at all.
    That is a separate error from the one this function guards against, it is always an
    over-cut, and it is not fixed here.
    """
    import numpy as np

    inside = np.asarray(mask, dtype=bool)
    if not inside.any() or not batter_run or batter_run <= 0:
        return None
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
        return None

    # Sampled per axis. `cell_size` is one number for a square grid and the row and
    # column spacings for any other, so a 2 m x 5 m cell tapers over five metres
    # north-south and two east-west, as the ground does.
    cell_h, cell_w = _axis_spacing(cell_size)
    # The transform measures cell centre to nearest *outside cell centre*, but the taper
    # starts at the mask edge, which is half a cell further out. So half a cell comes
    # off — and the half has to be of the axis the distance was actually measured along.
    if cell_h == cell_w:
        # Square cells: both axes agree, so there is one answer and no need to ask
        # which way the nearest outside cell lies. This is also the whole of the old
        # behaviour, kept bit-for-bit, because it was only ever wrong when the axes
        # disagreed.
        dist = distance_transform_edt(inside, sampling=(cell_h, cell_w))
        return np.clip((dist - cell_h / 2.0) / float(batter_run), 0.0, 1.0)

    # Rectangular cells, and `min()` used to answer here — the *finest* axis, whichever
    # way the boundary actually lay. On a 1 m x 2 m grid that under-subtracts for a
    # feature crossed row-wise, and since the shortfall goes straight into `reach` it is
    # always an over-cut: an east-west swale on those cells collapsed to `reach == 1` and
    # was burned as a full-depth rectangle, +60% against the same swale drawn north-south.
    #
    # Where the nearest outside cell is straight across an axis, the mask edge is
    # unambiguously half *that* axis away, and this is then exact. Where it lies
    # diagonally the cell is at a corner of the mask with no single edge direction to
    # measure from, and `min()` is kept: projecting a half-cell along the diagonal was
    # measurably worse against the true geometry, and worse on square grids too.
    dist, idx = distance_transform_edt(
        inside, sampling=(cell_h, cell_w), return_indices=True)
    rows, cols = np.indices(inside.shape)
    d_row = rows - idx[0]
    d_col = cols - idx[1]
    inset = np.where(d_col == 0, cell_h / 2.0,
                     np.where(d_row == 0, cell_w / 2.0,
                              min(cell_h, cell_w) / 2.0))
    return np.clip((dist - inset) / float(batter_run), 0.0, 1.0)


def tapered_invert(dem, mask, depth: float, batter_run: float, spill_elev: float,
                   cell_size: float = 1.0):
    """Excavate *mask* as a **true battered section**, deepening away from the edge.

    Every cell is floored at ``spill_elev − depth × min(1, x / batter_run)``, where ``x``
    is its distance to the nearest cell outside the footprint. That is the trapezoid,
    exactly: at ``x = 0`` the cut is nothing, at ``x ≥ batter_run`` it is full depth, and
    in between it follows the batter. Integrated over a strip of top width ``T`` and
    bottom width ``b`` it gives ``(T + b) / 2 × depth`` per metre — the drawn section.

    This replaces the nested-erosion staircase this module used to carry, which was
    wrong in
    two ways that a grid makes worse. Its volume was ``depth/n × Σ Aᵢ`` over ``n`` eroded
    footprints, which for a strip works out to ``depth × L × [T − (T−b)(n+1)/2n]`` — with
    the shipped ``n = 3`` that is **16.7% under** the trapezoid it is approximating, and
    it converges only as ``1/n``. And the erosions themselves vanish once
    ``batter_run / n`` drops below a cell, so raising ``n`` to close the gap is exactly
    what stops the steps existing. A distance transform has neither problem: it is one
    O(N) pass, it is as accurate as the cell size allows at any batter run, and it
    degrades gracefully — a footprint too narrow to hold the batter simply comes out
    shallower, which is the truth about that grid.

    Falls back to a level floor when scipy is unavailable or the batter is not positive.
    """
    import numpy as np

    if mask is None or not mask.any():
        return dem.copy()
    reach = taper_reach(mask, batter_run, cell_size)
    if reach is None:
        return level_invert(dem, mask, depth, spill_elev)

    inside = np.asarray(mask, dtype=bool)
    out = dem.copy()
    floors = spill_elev - depth * reach[inside]
    out[inside] = np.minimum(out[inside], floors)
    return out


#: How far a notch may be marched outward before it is called undaylighted, in
#: metres. A spillway is a cut *through* a bank, so the distance that matters is a
#: wall thickness or a keyed berm's width — both a few metres on any design this
#: plugin sizes. Marching much further stops being a weir and starts being a
#: diversion drain, which is a feature the user draws rather than one the burn
#: invents. Generous enough to cross a keyed dam wall; short enough that a notch
#: pointed at rising ground is reported rather than tunnelled to the horizon.
SPILLWAY_MAX_REACH_M = 20.0


class NotchCut(NamedTuple):
    """What :func:`spillway_notch` did, and — when it did nothing — why.

    ``dem`` is the surface after the cut, or the input unchanged when the notch was
    refused. ``cells`` is ``(rows, cols)`` of what was lowered, empty on a refusal, so
    the caller can record the mask without re-deriving it.

    ``sill_elev`` is the **as-burned sill**: the highest level water has to clear on its
    way through, measured on the surface as it stands afterwards. It is reported whether
    or not anything was cut, and the disagreement is the information — equal to the
    designed crest on a clean cut, *below* it where the whole path was already lower
    than the sill, and *above* it wherever the notch was refused and the bank is still
    standing at its own level.
    """

    dem: object
    cells: tuple
    reach_m: float
    daylit: bool
    into_pool: bool
    sill_elev: float = None


def notch_pool(dem, footprint, crest_elev, window=None, tolerance=1e-6):
    """The water a notch at *crest_elev* would be cutting into — its own **enclosed** pond.

    A cut whose outer end lands in here has not daylighted: the ground it found is below
    the crest and still inside the pool, so the notch drains one part of the pond into
    another and the water leaves wherever it was leaving before. That is the one failure
    the plain daylight test cannot see, and it is exactly the geometry a keyed berm
    presents — ``_key_berm_into_banks`` wraps the bank round both ends of the alignment,
    so the march can walk downhill and still be inside.

    **Enclosed is the whole of it.** "Below the crest and touching the footprint" is not
    a pond: on any hillside the entire slope below the sill answers that description, and
    using it would refuse every spillway on falling ground — the ordinary case. So the
    below-crest region is labelled and any component reaching the edge of *window* is
    thrown away as open ground with a way out. What is left is water the crest is
    actually holding in, and only that counts as "still inside".

    *window* is an optional ``(rows_slice, cols_slice)`` bounding the flood; the whole
    grid otherwise. The window edge is the escape test, so it must be padded well past
    the march's reach — see ``DEMBurner._notch_window``. Returns a bool array the same
    shape as *dem*.
    """
    import numpy as np
    from scipy.ndimage import label as _label

    dem = np.asarray(dem)
    footprint = np.asarray(footprint, dtype=bool)
    pool = np.zeros(dem.shape, dtype=bool)
    if not footprint.any():
        return pool

    win = window if window is not None else (slice(None), slice(None))
    sub = dem[win]
    under = np.zeros(sub.shape, dtype=bool)
    finite = np.isfinite(sub)
    under[finite] = sub[finite] <= float(crest_elev) + tolerance
    if not under.any():
        return pool
    # 8-connected, matching every other pond question in this codebase: a pool that
    # touches its neighbour only at a corner is one pool, not two.
    labels, _n = _label(under, structure=np.ones((3, 3), dtype=int))
    escapes = set(np.unique(np.concatenate([
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])))
    own = set(np.unique(labels[footprint[win] & under])) - {0} - escapes
    if own:
        pool[win] = np.isin(labels, list(own))
    return pool


def daylight_reach(dem, crest_cells, crest_elev, step_rc, max_reach_m,
                   cell_size=1.0, pool=None, tolerance=1e-6):
    """March the crest run outward until it reaches ground already below the crest.

    *crest_cells* is ``(rows, cols)`` of the cells the crest bar claims; *step_rc* is a
    ``(drow, dcol)`` unit step pointing to the **lower** side of the bank (the caller
    chooses it — see ``DEMBurner._ground_mean``). The bar is translated one step at a
    time and every cell it passes over joins the notch, so what comes back is a channel
    of the crest's width running down through the bank.

    The march stops as soon as **every** cell of the translated bar sits at or below the
    crest: that is the outside face, and cutting past it would take ground the water
    does not have to cross. It stops for the other reason at *max_reach_m*, and that is
    the reported failure — a notch pointed at rising ground discharges nowhere, so
    cutting it would move terrain and change nothing anyone can see.

    *pool* is the feature's own pond (:func:`notch_pool`). Ground found inside it does
    not count as daylight, however far down it is.

    Returns ``(rows, cols, reach_m, daylit, into_pool, control)``. ``rows``/``cols``
    always include the crest run itself, so a refused notch still knows where it was.
    ``control`` is the highest level water crossing this channel has to clear on the
    surface **as given** — the greatest, over the march, of the lowest cell across the
    bar. That is the pour point along the channel, and it is what the as-burned sill is
    measured from; cutting to *crest_elev* simply takes the lower of the two.
    """
    import numpy as np

    dem = np.asarray(dem)
    rows0 = np.asarray(crest_cells[0], dtype=int)
    cols0 = np.asarray(crest_cells[1], dtype=int)
    if rows0.size == 0:
        return rows0, cols0, 0.0, False, False, None

    dr, dc = float(step_rc[0]), float(step_rc[1])
    norm = (dr * dr + dc * dc) ** 0.5
    if norm == 0:
        return rows0, cols0, 0.0, False, False, None
    dr, dc = dr / norm, dc / norm

    cell = float(cell_size) if cell_size else 1.0
    max_steps = max(1, int(round(float(max_reach_m) / cell)))
    n_rows, n_cols = dem.shape
    claimed = {(int(r), int(c)) for r, c in zip(rows0, cols0)}

    def _bar_low(rr, cc):
        """The lowest cell across the bar at this step — holes ignored."""
        z = dem[rr, cc]
        z = z[np.isfinite(z)]
        return float(z.min()) if z.size else None

    control = _bar_low(rows0, cols0)
    daylit = into_pool = False
    reach_m = 0.0
    for k in range(1, max_steps + 1):
        rr = rows0 + int(round(k * dr))
        cc = cols0 + int(round(k * dc))
        inside = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
        if not inside.any():
            break                      # walked off the grid: nothing left to daylight to
        rr, cc = rr[inside], cc[inside]
        reach_m = k * cell
        low = _bar_low(rr, cc)
        if low is not None:
            control = low if control is None else max(control, low)
        z = dem[rr, cc]
        # NaN is not "below the crest". A hole in the DEM is ground we cannot see, and
        # calling it daylight would let a notch discharge into unmapped terrain.
        below = np.isfinite(z) & (z <= float(crest_elev) + tolerance)
        if below.all():
            if pool is not None and bool(np.asarray(pool)[rr, cc].all()):
                into_pool = True
                # Keep marching: the pool may be crossed and real ground found beyond
                # it, which is what a berm keyed round one end looks like.
            else:
                daylit = True
                into_pool = False
                break
        claimed.update((int(r), int(c)) for r, c in zip(rr, cc))

    rows = np.fromiter((rc[0] for rc in claimed), dtype=int, count=len(claimed))
    cols = np.fromiter((rc[1] for rc in claimed), dtype=int, count=len(claimed))
    return rows, cols, reach_m, daylit, into_pool, control


def spillway_notch(dem, crest_cells, crest_elev, step_rc,
                   max_reach_m=SPILLWAY_MAX_REACH_M, cell_size=1.0, pool=None,
                   floor_elev=None):
    """Cut a **level** notch at *crest_elev*, from the crest run out through the bank.

    The one thing that makes a designed spillway visible to everything downstream. The
    cut is ``np.minimum(dem, crest_elev)`` over the marched channel — absolute, taken
    from the stored design, never re-derived from the terrain at burn time, so it is
    order-independent and a second burn of the same design produces the same surface.

    It refuses rather than approximates, and each refusal names a different fault:

    * **no daylight** — the march hit its cap with the bank still standing above the
      crest, so the notch would discharge into rising ground.
    * **into its own pool** — it found ground below the crest that is still inside the
      pond (:func:`notch_pool`), so the water would leave where it already leaves.
    * **below the floor** — a crest at or under *floor_elev*, the **burned** floor of
      the feature, empties it. Checked against what was cut rather than against the
      analytic ``rim − depth``, because a footprint too narrow for its batter never
      reaches full depth and the analytic figure would refuse a legitimate sill.

    Returns a :class:`NotchCut`. On a refusal ``dem`` is the input untouched and
    ``cells`` is empty, so a caller can record "nothing cut" without inspecting flags —
    and ``sill_elev`` still reports the level the bank is standing at, which is the
    figure that says *how far* the refusal is from working.
    """
    import numpy as np

    dem = np.asarray(dem)
    empty = (np.empty(0, dtype=int), np.empty(0, dtype=int))
    if crest_elev is None:
        return NotchCut(dem, empty, 0.0, False, False, None)
    if floor_elev is not None and float(crest_elev) <= float(floor_elev):
        return NotchCut(dem, empty, 0.0, False, False, None)

    rows, cols, reach_m, daylit, into_pool, control = daylight_reach(
        dem, crest_cells, crest_elev, step_rc, max_reach_m,
        cell_size=cell_size, pool=pool)
    if not daylit:
        return NotchCut(dem, empty, reach_m, False, into_pool, control)

    out = dem.copy()
    out[rows, cols] = np.minimum(out[rows, cols], float(crest_elev))
    # Cutting to an absolute level lowers every step of the march that stood above it,
    # so the channel's control afterwards is simply the lower of the two.
    burned = (None if control is None
              else min(float(control), float(crest_elev)))
    return NotchCut(out, (rows, cols), reach_m, True, False, burned)


def spillway_burn_width(width_m, cell_size):
    """The built width rounded **up to a whole number of DEM cells**.

    A sill narrower than a cell cannot be cut as one, and rounding down would burn a
    weir narrower than the design flow needs. Rounding up is about *rasterisability*,
    not conservatism: nothing in the raster tier meters flow rate, so the burned width
    cannot change a total. What it does change is ``cells.size`` at the exit, and with
    it the per-cell runoff and the ``q = Q/L`` the erosion advisory is judged by.

    *cell_size* may be a scalar or a ``(cell_h, cell_w)`` pair; the **larger** axis is
    used, matching the caution ``taper_reach`` already carries about non-square grids —
    the crest axis is not known here, and the coarser axis is the one that can fail to
    resolve the sill.
    """
    import math

    cell_h, cell_w = _axis_spacing(cell_size)
    cell = max(float(cell_h), float(cell_w))
    if cell <= 0:
        return max(0.0, float(width_m or 0.0))
    wanted = max(0.0, float(width_m or 0.0))
    if wanted <= 0:
        return cell
    return math.ceil(wanted / cell - 1e-9) * cell


def rasterisable_capacity(n_cells: int, cell_area: float, depth: float,
                          top_width: float, bottom_width: float, cell_size: float,
                          batter_run: float = 0.0):
    """Storage the burned raster can represent, in m³ — **the estimate, not the answer.**

    Prefer ``DEMBurner.feature_storage_m3``, which is this same quantity measured off the
    hole that was actually cut. This function is the answer before any burn has run, and
    a model standing beside a burn is precisely the arrangement that has now gone wrong
    twice: Round 3, when it discounted for a batter the burner never cut and every swale
    on the site read ``Measured`` a flat +50% above ``At grid``; and again when the burn
    learned to taper, at which point the smooth trapezoid below became the over-estimate
    for any footprint too narrow to hold the taper. Both times the ground was right and
    the yardstick was wrong. ``capacity_breakdown`` now takes the measurement when it has
    one and falls back here when it does not.

    The model: a trapezoidal section survives to the grid when the feature carries a
    batter run and its footprint is wide enough (roughly three cells) to hold it;
    otherwise the grid can only hold a flat-bottomed trench, and this returns the
    rectangle. ``batter_run`` for a channel is derived from its own two widths — see
    ``earthwork_design.channel_batter_run`` — rather than read from the basin-only
    ``batter_run_m`` field, which is what used to make every drawn channel look
    vertical-walled to this function *and* to the burner.
    """
    if n_cells <= 0 or depth <= 0 or cell_area <= 0:
        return 0.0
    rectangular = n_cells * cell_area * depth
    if not batter_run or batter_run <= 0:
        # No batter was cut — a flat floor at full depth is a rectangular trench.
        return rectangular
    if cell_size > 0 and top_width < 3.0 * cell_size:
        # Too narrow for the walls to be resolved — a flat-bottomed trench.
        return rectangular
    if top_width <= 0:
        return rectangular
    mean_width = (top_width + max(0.0, bottom_width)) / 2.0
    return rectangular * (mean_width / top_width)


# How much more earth than storage counts as worth saying out loud. A level floor always
# costs a little more than its nominal volume — the ground is never flat — so a bare
# ratio > 1 would fire on everything. Past ~1.3 the extra excavation is a real cost the
# user is paying without having asked for it.
_OVER_EXCAVATION_RATIO = 1.3


def steep_ground_warning(name: str, relief: float, depth: float,
                         cut_m3=None, storage_m3=None):
    """Advisory when a level floor means cutting deeper than the design depth.

    A level floor is hydraulically right at any slope — it fills evenly to its spill
    level and holds its design volume. The cost is excavation: if the footprint falls
    further across than the feature is deep, the uphill end is cut deeper than asked
    for, and on a 10% slope that can be ~70% more earth moved for the same storage.
    The user should be told in cubic metres, not warned off in the abstract.

    **Gated on the measured cut, with relief as a fallback.** ``relief > depth`` is a
    proxy, and it misses the case that matters most: the datum is the lowest cell of the
    rim, so a footprint that is *internally* flat but sits beside a dip is floored to the
    dip and cut deep throughout, with no relief across it to notice. Swale 27 on the
    Quail Island design was cut 2.23 m mean for a 1.00 m design — 517 m³ of excavation
    for 232 m³ of storage — with 0.67 m of relief, and this stayed silent. Where the cut
    has actually been measured, ask it directly.
    """
    over_cut = (cut_m3 and storage_m3
                and cut_m3 > storage_m3 * _OVER_EXCAVATION_RATIO)
    steep = relief is not None and depth > 0 and relief > depth
    if not (over_cut or steep):
        return None

    if steep:
        msg = (
            f"'{name}': the footprint falls {relief:.1f} m across, more than its "
            f"{depth:.1f} m design depth. A level floor means cutting up to "
            f"{relief:.1f} m at the uphill end"
        )
    else:
        msg = (
            f"'{name}': a level floor referenced to the lowest point of its rim cuts "
            f"this well below its {depth:.1f} m design depth"
        )
    if cut_m3 and storage_m3:
        msg += f" — {cut_m3:,.0f} m³ of excavation for {storage_m3:,.0f} m³ of storage"
    msg += (
        ". Consider a smaller footprint, terracing into two features, or a dam wall "
        "on the low side."
    )
    return msg


def berm_variation_warning(name: str, min_h: float, max_h: float, mean_h: float):
    """Advisory when a level crest means a berm of very uneven height, else ``None``.

    A companion berm is built to one elevation — that is what makes it hold water — but
    the ground under it is not one elevation, so its *height* is ``crest − ground`` and
    changes along the run. On an alignment that is off contour, that variation is large:
    Swale 29 on the Quail Island design stands **0.60 m at one end and 1.73 m at the
    other**, and its mean of 0.98 m describes neither.

    Fires on the same principle as :func:`steep_ground_warning` — the ground has fallen
    further than the structure is tall, so one end is being built double while the other
    barely needs a bank.

    **Deliberately set at the tail, and the population is why.** Measured over the 32
    companion berms of the Quail Island design, ``max ÷ min`` runs 1.36 to 2.90 with a
    **median of 2.06**: on an off-contour design the *typical* berm is already twice as
    tall at one end as the other, and the distribution is one tight unimodal band with no
    gap in it to put a threshold. Anything looser fires on nearly everything —
    ``> 1.5 ×`` catches 94%, ``> 2.0 ×`` catches 56% — which is the noise the Verify
    table's dagger already had to be rescued from. So this names the genuine outlier
    only (3% here), and the *range* is reported unconditionally on every berm, which is
    what actually keeps the reader informed. A warning is for "go and do something"; the
    range is for "know what you are building".

    The fix, where it does fire, is to split the run into segments each at its own level
    — which is what the trench wants for the same reason (STRETCH_GOALS §5).
    """
    if min_h is None or max_h is None or not mean_h or mean_h <= 0:
        return None
    spread = max_h - min_h
    if spread <= mean_h:
        return None
    return (
        f"'{name}': the ground under its companion berm falls {spread:.1f} m, more than "
        f"the {mean_h:.1f} m the bank averages — so a level crest means building it "
        f"{max_h:.1f} m tall at one end and {min_h:.1f} m at the other. Consider "
        f"splitting the swale into segments, each at its own level."
    )


# When a bank stops being a bank. Both measured over the 36 storage features of the Quail
# Island design, whose ponds were flooded individually: retained depth runs 0.00–3.13 m
# with a **median of 0.77 m**, and 28% of a typical pond stands above natural ground. So
# retaining *something* is what a companion berm is for, and a threshold anywhere near the
# median fires on everything — 0.5 m catches 31 of 36. Past a metre of head the population
# thins sharply (4 of 36) and the consequence changes: that is a structure whose failure
# releases a wave rather than a puddle. The volume arm catches the other shape of the same
# risk — a shallower bank holding a great deal, like Swale 5 at 0.91 m and 603 m³.
_RETAINED_DEPTH_M = 1.0
_RETAINED_VOLUME_M3 = 500.0


def impoundment_warning(name: str, retained_depth_m=None, above_ground_m3=None,
                        has_spillway=False):
    """Advisory when a feature has stopped being a swale and become a small dam.

    Sizing against measured terrain storage is honest, and it has a consequence worth
    saying out loud: much of that storage is water held **above natural ground** by a bank
    of spoil. A trench that fails spills into itself. A metre of head behind a companion
    berm fails downhill, all at once, and the volume it releases is
    ``above_ground_m3`` — not the pond total, because the part below natural ground stays
    in the hole.

    Fires on either arm because the risk has two shapes: deep head behind a short bank,
    or modest head behind a long one holding a great deal. Neither is a reason not to
    build it — keyed berms are ordinary practice and they are why the design holds what it
    holds — but both are a reason to give it a designed overflow instead of letting it
    choose its own low point, and to build the bank properly rather than tip spoil.

    ``has_spillway`` softens the wording where the user has already designed one; the
    structural point still stands, so it is said either way.
    """
    depth = retained_depth_m or 0.0
    volume = above_ground_m3 or 0.0
    if depth < _RETAINED_DEPTH_M and volume < _RETAINED_VOLUME_M3:
        return None

    msg = (
        f"'{name}' is retaining water, not just holding it: {depth:.1f} m stands above "
        f"natural ground at the bank, impounding {volume:,.0f} m³ that would run "
        f"downhill if the bank gave way."
    )
    if has_spillway:
        msg += (
            " Check the spillway is sized for the peak inflow and that the bank is built "
            "and compacted as a water-retaining structure, not tipped spoil."
        )
    else:
        msg += (
            " Give it a designed spillway rather than letting it overtop at its own "
            "lowest point, and build the bank as a water-retaining structure. Storage "
            "this size may be consented work — check with your regional council."
        )
    return msg


def overtopping_warning(name: str, length_m: float, pour_level_m: float,
                        alt_saddle_m=None, has_spillway: bool = False,
                        reaches_crest=None, event_level_m=None):
    """Advisory when a pool's only way out is over the structure holding it.

    Measured, not assumed: the pool's rim was walked and its lowest point is this
    feature's own crest (see :func:`~terrainflow_assessment.modules.reporting.overtopping_spill`).
    Water leaving over an earth embankment is how one fails, and it is the thing a
    spillway exists to prevent — so the fact is worth saying whether or not the crest
    was deliberate.

    The length is stated because it is the number that is not obvious. A level crest spills
    along its whole length at once, and the discharge per metre — the thing that decides
    whether the face erodes — follows from that length, not from wherever the flow map
    happens to draw a channel.

    The stream layer used to show one thread over the wall for two reasons, and Round 14
    measured both. Flow really does leave the pool at many cells (23 for Dam 15), but flat
    resolution routed each cell to its *nearest* exit, so 87% of the flux went through one
    of them; and the Streams layer only draws cells over a contributing-area threshold,
    which a shared crest never reaches. Neither was "D8 forcing the overflow through one
    cell", which is what this used to say.

    Round 15 fixed the first: the flux is now spread evenly along the crest, which is what
    the length below has always described. Only the threshold is left, so a wall with no
    channel drawn on it means the layer's cut-off and nothing more.

    ``has_spillway`` used to change what the message could honestly claim, because the
    burn did not cut a spillway notch and the flow analysis routed overflow over the
    crest whether or not one was designed — so a sited spillway made this a limit of the
    model rather than a fault in the design, and the message said so at length.

    **The burn now cuts the notch** (``spillway_notch``), and the overtopping check
    subtracts it from the barrier's crest, so a spillway that works no longer reaches
    this function at all. Reaching it *with* a spillway therefore means something
    specific and worth saying plainly: the pool is leaving over the bank as well as, or
    instead of, through the structure — the sill is too high, too narrow, or it did not
    daylight. No caveat, because there is nothing left to caveat.

    ``reaches_crest`` says which storm this is about. The pour level is measured on the
    **full** pond — the pool filled to its spill point — so the sentence holds whatever
    the event does, and that is the right reference state for a freeboard fault. But
    said without qualification beside an event water line drawn well below the crest, it
    read as a claim about the modelled storm. ``None`` (nothing was asked) keeps the
    unqualified wording; ``True``/``False`` add the one sentence that separates *this
    event goes over* from *filling it would*, with ``event_level_m`` for the figure.
    """
    if not length_m or length_m <= 0:
        return None

    where = f"'{name}' fills to {pour_level_m:.2f} m and leaves over its own crest"
    span = (f", along about {length_m:.0f} m of it at once — not at the one place the "
            f"stream layer draws a channel, which is where the flow map concentrates it "
            f"rather than where the water goes")

    event_note = ""
    if reaches_crest is True:
        event_note = (" The modelled event fills it to that level, so the water goes "
                      "over in this run and not only in a larger one.")
    elif reaches_crest is False:
        where = (f"'{name}' has no freeboard — filled to {pour_level_m:.2f} m it leaves "
                 f"over its own crest")
        stands = ("" if event_level_m is None
                  else f", standing at {event_level_m:.2f} m")
        event_note = (f" The modelled event does not fill it that far{stands}, so this "
                      f"is the structure's freeboard rather than something this storm "
                      f"does.")

    if has_spillway:
        return (
            f"{where}{span}. The designed spillway is cut into the terrain model and "
            f"this water is going over the bank anyway — so the sill is not taking it. "
            f"Check that the crest is below this level, that the weir is wide enough, "
            f"and that the notch daylights onto falling ground.{event_note}"
        )

    tail = ""
    if alt_saddle_m is not None and alt_saddle_m != float("inf"):
        rise = alt_saddle_m - pour_level_m
        # A centimetre, not a float epsilon: below that the two ways out are the same
        # level, and "raise the crest 0.00 m" is worse than saying nothing.
        if rise >= 0.01:
            tail = (f" Raising the crest {rise:.2f} m would send it to the natural "
                    f"saddle at {alt_saddle_m:.2f} m instead.")
        else:
            # A crest seeded from the highest ground the line touches lands exactly on
            # the bank, so the pond reaches the crest and the abutment at the same
            # moment. Raising one without the other just moves the failure to the end
            # of the wall, where there is no structure at all.
            tail = (" The bank beside it stands at the same level, so there is no "
                    "freeboard at the abutment: raising the crest alone would send the "
                    "water round the end of the wall rather than over it.")
    return (
        f"{where}{span}. Nothing is designed to take it — give it a spillway, or the "
        f"overflow chooses its own place to cut and takes the bank with it.{tail}"
        f"{event_note}"
    )


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
