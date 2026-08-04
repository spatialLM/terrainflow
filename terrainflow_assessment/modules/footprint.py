"""
footprint.py — pure earthwork-footprint raster geometry.

One home for "turn this drawn feature into cells, and ask the terrain about them".
Previously each caller rasterised footprints on its own terms — the burner without
``all_touched``, the catchment and contour code with it — so a feature covered a
different set of cells depending on which stage was asking. Everything shares these:

    rasterize_footprint — shapely geometry → boolean cell mask
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

import numpy as np
from rasterio.features import rasterize

# Cells whose *centre* falls inside the polygon are not the same as cells the polygon
# touches. Footprints use all_touched so a narrow or diagonal feature is never lost —
# a 0.8 m swale on a 1 m grid still claims the cells it crosses.
DEFAULT_ALL_TOUCHED = True


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


def domain_mask(shape, transform, polygons=None, valid=None,
                all_touched: bool = DEFAULT_ALL_TOUCHED):
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
    """
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
            return mask

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.any():
            return valid.copy()
    return np.ones(shape, dtype=bool)


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
