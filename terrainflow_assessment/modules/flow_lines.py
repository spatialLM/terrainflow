"""
flow_lines.py — the downslope slope field sampled from a DEM.

Samples a regular ground-spaced grid and reports, at each sample, the direction
of steepest descent and how steep the ground is there. Rendered as hachures —
short tapered strokes pointing downhill — this reads as the direction water runs
and how hard it runs, without covering the map the way a full overlay does.

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


def hachure_segments(dem_path, spacing_m=30.0, min_slope_deg=0.5,
                     length_fraction=0.7):
    """
    Sample the slope field as short downhill line segments, ready to draw as hachures.

    A hachure is drawn, not coloured — the stroke runs downhill and thickens with
    steepness, so a renderer needs a *line* to taper along rather than the point a
    rotated arrow marker sits on. Each segment starts at the sample point and runs
    ``length_fraction × spacing_m`` in the downslope direction, so neighbouring
    strokes stay clear of one another at any spacing.

    Parameters
    ----------
    dem_path : str
    spacing_m : float — ground spacing between hachures (m). Fixed distance, not a
        cell count, so the field reads the same on a 0.5 m LiDAR grid and an 8 m DEM.
    min_slope_deg : float — skip near-flat ground below this slope. Flat ground
        having no hachures is the convention, not a gap in the data.
    length_fraction : float — segment length as a fraction of ``spacing_m``.

    Returns
    -------
    list of {"geometry": LineString, "slope_deg": float} — the LineString running
    from the sample point downhill, in the DEM's map CRS.
    """
    seg_len = spacing_m * length_fraction
    out = []
    for v in slope_vectors(dem_path, spacing_m=spacing_m,
                           min_slope_deg=min_slope_deg):
        # Bearing is a compass angle (0 = north, clockwise); map dx/dy invert that.
        theta = np.radians(v["angle_deg"])
        dx = seg_len * np.sin(theta)
        dy = seg_len * np.cos(theta)
        out.append({
            "geometry": LineString([
                (v["x"], v["y"]),
                (v["x"] + float(dx), v["y"] + float(dy)),
            ]),
            "slope_deg": v["slope_deg"],
        })
    return out
