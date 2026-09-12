"""
footprint.py — pure earthwork-footprint raster geometry.

One home for "turn this drawn feature into cells, and ask the terrain about them".
Previously each caller rasterised footprints on its own terms — the burner without
``all_touched``, the catchment and contour code with it — so a feature covered a
different set of cells depending on which stage was asking. Everything shares these:

    rasterize_footprint — shapely geometry → boolean cell mask
    xy_to_rc_array      — array form of xy_to_rc, same floor/no-clamp semantics
    line_points         — (x, y) at given chainages along a polyline
    sample_along_line   — raster values along a polyline, in one vectorised pass
    outer_ring          — the cells immediately surrounding a mask (its rim)
    pour_point          — lowest rim cell: the elevation the feature spills at
    outlet_cell         — lowest cell *inside* the mask: where its overflow starts
    internal_relief     — elevation range across the footprint
    min_dimension       — narrowest dimension, for the sub-cell honesty checks

``pour_point`` is the datum for level-bottom burning: a basin floored at
``pour_point − depth`` fills evenly to the level at which it would naturally spill, so
its analytic prism and its burned pond agree. ``outlet_cell`` is where overflow leaves,
and so is the start of the downslope walk that resolves a feature's routing target.

Pure numpy + shapely + rasterio.features: no QGIS.
"""

from __future__ import annotations

import math

import numpy as np
from rasterio.features import rasterize

# Cells whose *centre* falls inside the polygon are not the same as cells the polygon
# touches. Footprints use all_touched so a narrow or diagonal feature is never lost —
# a 0.8 m swale on a 1 m grid still claims the cells it crosses.
DEFAULT_ALL_TOUCHED = True


def xy_to_rc(transform, x, y):
    """Map coordinate → the ``(row, col)`` of the cell it falls in.

    Floor, not ``int()``. ``int()`` truncates toward zero, so a point in the band
    immediately *north* or *west* of the grid divides to something like -0.4 and
    truncates to 0 — passing a ``0 <= row < rows`` bounds check as though it were
    inside. Anything drawn just off the top or left edge was therefore burned into
    row 0 or column 0 instead of being rejected as outside the DEM, and the dam
    abutment walk marched along column 0 looking for ground.

    Returns out-of-range indices rather than clamping or raising, on purpose.
    "Which cell is this in" and "is that cell on the grid" are different questions,
    and the second one has a different answer at each call site — skip the feature,
    stop the walk, warn the user. Clamping here would silently pick one of them.
    """
    col = math.floor((x - transform.c) / transform.a)
    row = math.floor((y - transform.f) / transform.e)
    return int(row), int(col)


def xy_to_rc_array(transform, xs, ys):
    """Array form of :func:`xy_to_rc` — same cell for every point, in one pass.

    Identical semantics, deliberately: ``np.floor`` rather than a cast (a cast
    truncates toward zero, which is how points just north or west of the grid used to
    land in row 0), and out-of-range indices are returned rather than clamped, because
    "which cell is this in" and "is that cell on the grid" are still different
    questions. Callers mask; this does not.

    Returns ``(rows, cols)`` as int64 arrays shaped like the inputs.
    """
    xs = np.asarray(xs, dtype="float64")
    ys = np.asarray(ys, dtype="float64")
    cols = np.floor((xs - transform.c) / transform.a).astype("int64")
    rows = np.floor((ys - transform.f) / transform.e).astype("int64")
    return rows, cols


def line_points(coords, distances):
    """``(xs, ys)`` at *distances* along the polyline *coords*.

    The vectorised stand-in for ``geom.interpolate(d)`` in a loop. For a polyline the
    two agree exactly — both walk the same cumulative chainage and interpolate linearly
    within the segment the distance lands in — but this does the whole set in two
    ``np.interp`` calls instead of one Python-level shapely call per point.

    Deliberately numpy rather than ``shapely.line_interpolate_point``: that is a
    shapely 2.0 API, ``metadata.txt`` declares ``qgisMinimumVersion=3.22``, and
    ``keypoint_analysis.offset_parts`` already hedges ``offset_curve`` against shapely
    1.x. Reaching for the vectorised shapely call would quietly raise the plugin's
    floor to buy nothing this does not already do.

    Distances outside ``[0, length]`` clamp to the ends, as ``interpolate`` does.
    """
    pts = np.asarray(coords, dtype="float64")
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("line_points needs a sequence of (x, y) coordinates")
    pts = pts[:, :2]
    d = np.asarray(distances, dtype="float64")
    if pts.shape[0] == 1:
        return np.full(d.shape, pts[0, 0]), np.full(d.shape, pts[0, 1])

    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    # Duplicate vertices give zero-length segments, and ``np.interp`` needs an
    # increasing xp. Drop them: the first vertex always stays, and every later one
    # survives only if it actually advanced the chainage.
    keep = np.concatenate(([True], seg_len > 0.0))
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))[keep]
    kept = pts[keep]
    if kept.shape[0] == 1:
        return np.full(d.shape, kept[0, 0]), np.full(d.shape, kept[0, 1])

    return np.interp(d, cum, kept[:, 0]), np.interp(d, cum, kept[:, 1])


def sample_along_line(coords, transform, array, distances,
                      nodata=None, fill=np.nan):
    """Values of *array* at *distances* along the polyline *coords*.

    One home for "walk this line and ask the raster", which five call sites were doing
    a point at a time — ``geom.interpolate`` plus a single-element index, per
    cell-width, per contour. On a 1 m DEM with two hundred 2 km contours that is of the
    order of 400,000 Python-level shapely calls per pass, and two passes run.

    ``fill`` is what an off-grid point contributes, and it is **not** always NaN: a
    profile indexed by position (``find_swale_segments``) needs a fixed-length result
    with off-grid reading as zero accumulation, while a mean over valid ground
    (``filter_by_slope``) needs NaN so it can be excluded. Pass what the caller means.

    ``nodata`` maps the raster's declared sentinel to NaN. Slope rasters declare
    ``-9999`` and write it, so a caller that omits this averages a hole toward minus
    ten thousand degrees and calls it the flattest ground on the site.

    Always returns a float64 array as long as *distances*.
    """
    xs, ys = line_points(coords, distances)
    rows, cols = xy_to_rc_array(transform, xs, ys)

    h, w = array.shape
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)

    out = np.full(rows.shape, float(fill), dtype="float64")
    if inside.any():
        vals = array[rows[inside], cols[inside]].astype("float64")
        if nodata is not None and np.isfinite(nodata):
            vals = np.where(vals == nodata, np.nan, vals)
        out[inside] = vals
    return out


def rasterize_footprint(shapely_geom, shape, transform,
                        all_touched: bool = DEFAULT_ALL_TOUCHED):
    """Boolean cell mask for *shapely_geom* over a raster of *shape* / *transform*.

    Returns an all-False mask for an empty or missing geometry rather than raising —
    callers decide whether that is a sub-cell fallback case or a genuine no-op.
    """
    if shapely_geom is None or getattr(shapely_geom, "is_empty", False):
        return np.zeros(shape, dtype=bool)
    return rasterize(
        [(shapely_geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype(bool)


# Which branch of :func:`domain_mask` produced the mask. Not decoration: only the first
# means the user said where the site is, and the other two are the plugin guessing.
DOMAIN_FROM_POLYGON = "polygon"
DOMAIN_FROM_VALID_DEM = "valid_dem"
DOMAIN_FROM_WHOLE_GRID = "whole_grid"


def domain_mask(shape, transform, polygons=None, valid=None,
                all_touched: bool = DEFAULT_ALL_TOUCHED, with_source: bool = False):
    """Boolean mask of "the site" — the denominator of the water balance.

    *polygons* is an ordered list of shapely-polygon lists, most-specific first (e.g.
    ``[analysis_area, boundary]``); the first non-empty one wins. When none rasterises
    to anything, falls back to *valid* (the usable DEM cells), and failing that to the
    whole grid.

    This exists because the capture denominator was previously
    ``flow_accumulation.max() × cell_area`` — the area draining to the single busiest
    cell, which is one outlet's catchment, not the site. On a multi-outlet DEM that is
    a small fraction of the real area, which is half of why "100% of the storm held on
    site" was being reported.

    ``with_source=True`` also returns which branch answered, as one of the
    ``DOMAIN_FROM_*`` constants. The fallbacks are not equivalent to a drawn boundary
    and the difference is not visible in the mask: on the Quail Island tile the DEM
    declares a nodata sentinel and contains **no** nodata cells, so "every usable DEM
    cell" was all 2.85 km² — 70% of it harbour, and rain on the harbour went into every
    "% of the site" the report prints. A caller that can say so should.
    """
    source = DOMAIN_FROM_POLYGON
    for group in (polygons or []):
        if not group:
            continue
        geoms = [g for g in group if g is not None and not getattr(g, "is_empty", False)]
        if not geoms:
            continue
        mask = rasterize(
            [(g, 1) for g in geoms],
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ).astype(bool)
        if valid is not None:
            mask &= np.asarray(valid, dtype=bool)
        if mask.any():
            return (mask, source) if with_source else mask

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.any():
            mask = valid.copy()
            return (mask, DOMAIN_FROM_VALID_DEM) if with_source else mask
    mask = np.ones(shape, dtype=bool)
    return (mask, DOMAIN_FROM_WHOLE_GRID) if with_source else mask


def clip_raster_to_polygons(src_path, geoms, out_path,
                            all_touched: bool = DEFAULT_ALL_TOUCHED):
    """Write a copy of *src_path* with everything outside *geoms* set to nodata.

    For the report's flow map. The Surface Runoff raster covers the whole DEM
    tile, and on a coastal tile most of that tile is not the block — so the
    figure captioned "where the water goes" came out with the site sitting in a
    fan of blue streaks running off every edge, none of which is anywhere the
    owner can do anything about. Masking to the drawn boundary leaves the same
    ramp saying the same thing about the same ground.

    A copy, not a change to the source. The raster on the canvas is the analysis
    output and the numbers behind it are measured over the whole grid; clipping
    the original would quietly move what the layer means everywhere else it is
    read.

    Returns *out_path*, or ``None`` when there is nothing to clip to — an empty
    geometry list, or a mask that misses the raster entirely. Both mean "print
    the unclipped layer", which is what the caller does with ``None``.
    """
    import rasterio

    geoms = [g for g in (geoms or [])
             if g is not None and not getattr(g, "is_empty", False)]
    if not geoms:
        return None

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        mask = rasterize(
            [(g, 1) for g in geoms],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ).astype(bool)
        if not mask.any():
            return None
        data = src.read(1)
        nodata = src.nodata

    if nodata is None:
        # No sentinel declared, so one has to be chosen. NaN for a float band
        # reads as "no data" to every renderer without colliding with a real
        # value; an integer band has no such value, so it keeps its zeros and
        # is masked to the band minimum instead.
        if np.issubdtype(data.dtype, np.floating):
            nodata = float("nan")
        else:
            nodata = int(np.min(data))
        profile.update(nodata=nodata)

    out = np.where(mask, data, np.asarray(nodata).astype(data.dtype))
    profile.update(count=1, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out.astype(data.dtype), 1)
    return out_path


def domain_fallback_warning(source, domain_cells, cell_area_m2):
    """Advisory when "the site" was guessed rather than drawn. ``None`` when it was drawn.

    Deliberately not a percentage of anything: the point is that the denominator every
    other percentage is built on has not been set, so quoting one here would be circular.
    Says the area instead, which the reader can recognise as their farm or not.
    """
    if source == DOMAIN_FROM_POLYGON:
        return None
    area_ha = (domain_cells * float(cell_area_m2)) / 10_000.0
    where = ("the whole DEM" if source == DOMAIN_FROM_WHOLE_GRID
             else "every cell the DEM has an elevation for")
    return (
        f"No site boundary or analysis area is set, so the site is {where} — "
        f"{area_ha:,.0f} ha. Runoff totals, capture % and every other share of the site "
        f"are measured over that, including any water body or neighbouring land the DEM "
        f"happens to cover. Draw a Site Boundary to measure them over your ground."
    )


def _dilate8(mask):
    """8-connected dilation of *mask* (the mask itself included)."""
    rows, cols = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.zeros_like(mask)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out |= padded[dr:dr + rows, dc:dc + cols]
    return out


def outer_ring(mask):
    """The ring of cells immediately outside *mask* — its rim.

    Empty when the mask is empty, or when it covers the whole grid (nothing surrounds
    it); callers treat that as "no rim to spill over" and fall back accordingly.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    return _dilate8(mask) & ~mask


def _valid(dem, mask, nodata=None):
    """Cells of *mask* carrying usable elevations."""
    ok = mask & np.isfinite(dem)
    if nodata is not None:
        ok &= dem != nodata
    return ok


def pour_point(dem, mask, nodata=None):
    """Lowest rim cell of *mask* — the level at which the footprint spills.

    Returns ``(elevation, (row, col))``, or ``(None, None)`` when there is nothing to
    measure. This is the natural pour point: fill the footprint and this is where water
    escapes first, so it is the correct datum for a level invert.

    Falls back to the *highest* cell inside the footprint when the mask has no usable
    rim (it reaches the grid edge, or covers everything) — that is the most water the
    footprint could hold before spilling, which keeps the caller conservative rather
    than handing back a datum that over-excavates.
    """
    dem = np.asarray(dem, dtype="float64")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None, None

    rim = _valid(dem, outer_ring(mask), nodata)
    if rim.any():
        flat = int(np.argmin(np.where(rim, dem, np.inf)))
        return float(dem.ravel()[flat]), np.unravel_index(flat, dem.shape)

    inside = _valid(dem, mask, nodata)
    if not inside.any():
        return None, None
    flat = int(np.argmax(np.where(inside, dem, -np.inf)))
    return float(dem.ravel()[flat]), np.unravel_index(flat, dem.shape)


def pour_point_near(dem, mask, centre_rc, radius_cells, nodata=None):
    """Lowest rim cell of *mask* **within *radius_cells* of *centre_rc***.

    The same measurement as :func:`pour_point`, restricted to the part of the ring that
    is near a given place. That distinction only starts to matter once something is sited
    on a feature rather than describing the whole of it: a contour swale's global ring
    minimum is usually at one of its ends, so a sill placed half way along and referenced
    to it is referenced to ground it does not share and, on a swale that falls along its
    run, to an elevation it never reaches.

    *centre_rc* is a ``(row, col)``; *radius_cells* is a Chebyshev radius, which is the
    right shape for a square window over a grid and costs nothing to build. Falls back
    to the global :func:`pour_point` whenever the local window catches no usable rim —
    off the grid, or a footprint that reaches the edge — because a datum from further
    away beats no datum at all, and the caller cannot tell the difference in the answer
    it gets either way.
    """
    dem = np.asarray(dem, dtype="float64")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or centre_rc is None:
        return pour_point(dem, mask, nodata)

    rim = _valid(dem, outer_ring(mask), nodata)
    if not rim.any():
        return pour_point(dem, mask, nodata)

    row, col = int(centre_rc[0]), int(centre_rc[1])
    reach = max(1, int(radius_cells))
    rows, cols = dem.shape
    r0, r1 = max(0, row - reach), min(rows, row + reach + 1)
    c0, c1 = max(0, col - reach), min(cols, col + reach + 1)
    if r0 >= r1 or c0 >= c1:
        return pour_point(dem, mask, nodata)

    local = np.zeros_like(rim)
    local[r0:r1, c0:c1] = rim[r0:r1, c0:c1]
    if not local.any():
        return pour_point(dem, mask, nodata)

    flat = int(np.argmin(np.where(local, dem, np.inf)))
    return float(dem.ravel()[flat]), np.unravel_index(flat, dem.shape)


def outlet_cell(dem, mask, nodata=None):
    """Lowest cell *inside* *mask* — where its overflow leaves from.

    Returns ``(row, col)`` or ``None``. Used as the start of the downslope walk that
    resolves a feature's routing target, and as the point a zero-capacity feature
    (berm, diversion drain) re-emits the water it intercepts.
    """
    dem = np.asarray(dem, dtype="float64")
    inside = _valid(dem, np.asarray(mask, dtype=bool), nodata)
    if not inside.any():
        return None
    flat = int(np.argmin(np.where(inside, dem, np.inf)))
    return np.unravel_index(flat, dem.shape)


def internal_relief(dem, mask, nodata=None):
    """Elevation range across *mask* (max − min), or ``0.0`` when unmeasurable.

    Compared against the design depth this is what says whether a level floor is a
    reasonable excavation or a cliff: relief greater than the depth means the uphill
    end is cut deeper than designed.
    """
    dem = np.asarray(dem, dtype="float64")
    inside = _valid(dem, np.asarray(mask, dtype=bool), nodata)
    if not inside.any():
        return 0.0
    vals = dem[inside]
    return float(vals.max() - vals.min())


def min_dimension(shapely_geom, bottom_width=None):
    """Narrowest dimension of a footprint, for the sub-cell resolution checks.

    Lines carry their width as an attribute, so *bottom_width* is returned directly for
    them. Polygons use ``2 × area / perimeter`` — the width of the equivalent strip,
    which tends to the true width for a long thin shape and to the inradius for a
    compact one. Returns ``None`` when there is nothing meaningful to measure.
    """
    if bottom_width is not None:
        return float(bottom_width)
    if shapely_geom is None or getattr(shapely_geom, "is_empty", False):
        return None

    area = float(getattr(shapely_geom, "area", 0.0) or 0.0)
    perimeter = float(getattr(shapely_geom, "length", 0.0) or 0.0)
    if area <= 0 or perimeter <= 0:
        return None
    return 2.0 * area / perimeter
