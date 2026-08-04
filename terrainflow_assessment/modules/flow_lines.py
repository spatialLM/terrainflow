"""
flow_lines.py — downslope flow streamlines traced from a DEM.

Replaces the fixed grid of slope-direction arrows with smooth curved flowlines:
seed a regular grid of start points and, from each, step in the steepest-descent
direction (from the DEM gradient) cell by cell until the path leaves the grid,
reaches flat/pit ground, or hits a step cap. The result reads as the path water
would take across the slope.

Routing-agnostic — it follows the DEM surface gradient directly, so it does not
depend on the pysheds D8/D-infinity flow-direction encoding.

All geometry returned as shapely LineStrings in the DEM's map CRS.
"""

import numpy as np
import rasterio
from shapely.geometry import LineString


def _terrain_gradient(dem, cell_w, cell_h):
    """``(dz/d_row, dz/d_col)`` as true dimensionless gradients (m per m).

    The cell size has to be handed to ``np.gradient``: its documented default is
    unit spacing, which yields metres-per-CELL. On a 0.5 m LiDAR grid that made the
    reported slope half the real one and turned the ``min_grad`` stop threshold into a
    different physical grade on every DEM.

    Nodata stays NaN rather than being filled with the DEM minimum. That fill built a
    cliff around every hole, and its gradient dragged flow lines straight into the data
    boundary — exactly where a user who clipped to their property line is looking.
    Callers must therefore treat a non-finite gradient as "no information here".
    """
    gy, gx = np.gradient(np.asarray(dem, dtype="float64"), cell_h, cell_w)
    return gy, gx


def trace_flow_lines(dem_path, seed_spacing_m=60.0, max_steps=400,
                     min_grad=1e-4, min_length_m=None, return_slope=False):
    """
    Trace downslope flowlines across a DEM.

    Parameters
    ----------
    dem_path : str — projected DEM GeoTIFF
    seed_spacing_m : float — spacing between seed start points (m)
    max_steps : int — maximum cells a single line may traverse
    min_grad : float — stop when the local gradient magnitude (m/m, dimensionless)
        falls below this (flat ground / pit)
    min_length_m : float or None — drop lines shorter than this (default:
        1.5 × seed_spacing_m, so only lines that actually travel are kept)
    return_slope : bool — when True, return dicts carrying the mean ground slope
        (degrees) along each line so the line can be coloured by steepness.

    Returns
    -------
    return_slope False → list of shapely LineString (map coordinates)
    return_slope True  → list of {"geometry": LineString, "mean_slope_deg": float}
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan

    rows, cols = dem.shape
    cell_w = abs(transform.a)
    cell_h = abs(transform.e)
    cell_size = (cell_w + cell_h) / 2.0
    if min_length_m is None:
        min_length_m = 1.5 * seed_spacing_m

    # Ground gradient (m per m) — direction and steepness.
    gy, gx = _terrain_gradient(dem, cell_w, cell_h)

    seed_step = max(1, int(round(seed_spacing_m / cell_size)))

    def _to_xy(col, row):
        x = transform.c + (col + 0.5) * transform.a
        y = transform.f + (row + 0.5) * transform.e
        return float(x), float(y)

    out = []
    for r0 in range(seed_step // 2, rows, seed_step):
        for c0 in range(seed_step // 2, cols, seed_step):
            if np.isnan(dem[r0, c0]):
                continue
            r, c = float(r0), float(c0)
            pts = [_to_xy(c, r)]
            slopes = []
            for _ in range(max_steps):
                ri, ci = int(round(r)), int(round(c))
                if not (0 <= ri < rows and 0 <= ci < cols):
                    break
                if np.isnan(dem[ri, ci]):
                    break
                # Steepest descent = negative gradient (row, col components).
                d_row = -gy[ri, ci]
                d_col = -gx[ri, ci]
                mag = (d_row ** 2 + d_col ** 2) ** 0.5
                if not np.isfinite(mag) or mag < min_grad:
                    break  # flat ground, a pit, or the edge of the data — flow stops
                # mag is a dimensionless ground gradient → slope is arctan(mag).
                slopes.append(np.degrees(np.arctan(mag)))
                r += d_row / mag
                c += d_col / mag
                if not (0 <= r < rows and 0 <= c < cols):
                    break
                pts.append(_to_xy(c, r))

            if len(pts) < 2:
                continue
            line = LineString(pts)
            if line.length < min_length_m:
                continue
            if return_slope:
                mean_slope = float(np.mean(slopes)) if slopes else 0.0
                out.append({"geometry": line, "mean_slope_deg": round(mean_slope, 1)})
            else:
                out.append(line)

    return out


def slope_vectors(dem_path, spacing_m=50.0, min_slope_deg=0.5):
    """
    Sample a regular grid of downslope slope vectors from a DEM.

    Each vector carries the downslope compass bearing (for arrow rotation) and
    the ground slope in degrees (for colour/size), giving a true slope field that
    shows both direction *and* steepness — the complement to the flow lines.

    Parameters
    ----------
    dem_path : str
    spacing_m : float — grid spacing between arrows (m)
    min_slope_deg : float — skip near-flat cells below this slope

    Returns
    -------
    list of {"x", "y", "angle_deg", "slope_deg"} (angle: compass bearing of the
    downslope direction, 0 = north, clockwise)
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan

    rows, cols = dem.shape
    cell_w = abs(transform.a)
    cell_h = abs(transform.e)
    cell_size = (cell_w + cell_h) / 2.0
    gy, gx = _terrain_gradient(dem, cell_w, cell_h)

    step = max(1, int(round(spacing_m / cell_size)))
    out = []
    for r in range(step // 2, rows, step):
        for c in range(step // 2, cols, step):
            if np.isnan(dem[r, c]):
                continue
            mag = (gx[r, c] ** 2 + gy[r, c] ** 2) ** 0.5
            if not np.isfinite(mag):
                continue          # beside nodata — no gradient to draw an arrow from
            slope_deg = float(np.degrees(np.arctan(mag)))
            if slope_deg < min_slope_deg:
                continue
            # Downslope direction in map space: east = -gx, north = gy
            # (row increases south). Compass bearing = atan2(east, north).
            bearing = float(np.degrees(np.arctan2(-gx[r, c], gy[r, c]))) % 360.0
            x = transform.c + (c + 0.5) * transform.a
            y = transform.f + (r + 0.5) * transform.e
            out.append({
                "x": float(x), "y": float(y),
                "angle_deg": round(bearing, 1),
                "slope_deg": round(slope_deg, 1),
            })
    return out
