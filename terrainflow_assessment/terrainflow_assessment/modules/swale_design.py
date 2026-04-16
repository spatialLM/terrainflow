"""
swale_design.py — Swale design helpers and soil reference data.

Provides:
  SOIL_REFERENCE            — soil type → CN mapping (from SCS model)
  recommend_swale_length()  — size a swale to intercept a given inflow volume
  snap_geometry_to_contour() — snap a drawn line to the nearest contour elevation
  contour_to_swale_geometry() — convert a full contour line to a swale QgsGeometry
"""

from .catchment import SCSRunoff


# ---------------------------------------------------------------------------
# Soil reference (shared with SCS runoff model)
# ---------------------------------------------------------------------------

# CN values by general soil texture (USDA-NRCS, normal moisture AMC II).
# Displayed in the soil type dropdown; CN is passed to the SCS model.
SOIL_REFERENCE = SCSRunoff.SOIL_REFERENCE  # {name: CN}

# Approximate steady-state infiltration rates by soil texture (mm/hr).
# Used in the fill simulation to compute losses from swales/ponds over time.
INFILTRATION_RATE_MM_HR = {
    "Sand":       15.0,
    "Sandy loam":  8.0,
    "Loam":        4.0,
    "Clay loam":   2.5,
    "Clay":        1.5,
}


def get_infiltration_rate(soil_name):
    """Return infiltration rate (mm/hr) for a soil type, defaulting to Loam."""
    return INFILTRATION_RATE_MM_HR.get(soil_name, INFILTRATION_RATE_MM_HR["Loam"])


# ---------------------------------------------------------------------------
# Swale sizing
# ---------------------------------------------------------------------------

def recommend_swale_length(peak_inflow_m3, depth, width):
    """
    Estimate recommended swale length to intercept a given inflow volume.

    Uses a rectangular cross-section approximation (conservative — actual
    trapezoidal capacity is larger for the same depth/width).

    Parameters
    ----------
    peak_inflow_m3 : float — design-storm inflow volume (m³)
    depth : float — swale depth (m)
    width : float — swale top width (m)

    Returns
    -------
    float — recommended swale length (m), or 0.0 if inputs are invalid.
    """
    if depth <= 0 or width <= 0 or peak_inflow_m3 <= 0:
        return 0.0

    # Rectangular cross-section as lower bound
    cross_section = depth * width  # m²
    # Apply 0.8 freeboard factor consistent with capacity calculations
    length = peak_inflow_m3 / (cross_section * 0.8)
    return round(length, 1)


# ---------------------------------------------------------------------------
# Contour-based swale drawing helpers
# ---------------------------------------------------------------------------

def contour_to_swale_geometry(contour_qgs_geom):
    """
    Convert a clicked contour QgsGeometry directly to a swale geometry.

    The contour line already follows the terrain elevation, making it an ideal
    swale alignment.  This function returns the geometry unchanged — the swale
    will be placed exactly on the contour.

    Parameters
    ----------
    contour_qgs_geom : QgsGeometry (polyline)

    Returns
    -------
    QgsGeometry — the same polyline, ready to use as a swale geometry.
    """
    return contour_qgs_geom


def snap_point_to_contour_elevation(point_xy, dem_path):
    """
    Given a map point (x, y), return the DEM elevation at that location.

    Used in the freehand contour-snap drawing mode: each vertex click
    queries the DEM to find the elevation, ensuring the drawn swale
    follows the terrain grade.

    Parameters
    ----------
    point_xy : tuple (x, y) in the DEM's CRS
    dem_path : str

    Returns
    -------
    float — elevation in metres, or None if outside the raster extent.
    """
    import rasterio
    import numpy as np

    x, y = point_xy
    try:
        with rasterio.open(dem_path) as src:
            transform = src.transform
            col = int((x - transform.c) / transform.a)
            row = int((y - transform.f) / transform.e)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1)[row, col]
                nodata = src.nodata
                if nodata is None or not np.isclose(val, nodata):
                    return float(val)
    except Exception:
        pass
    return None


def sample_peak_inflow(qgs_geom, acc_path, n_samples=30):
    """
    Sample the peak flow accumulation value along a geometry (swale or contour).

    Parameters
    ----------
    qgs_geom : QgsGeometry (polyline or polygon)
    acc_path : str — path to flow accumulation GeoTIFF
    n_samples : int

    Returns
    -------
    float — peak accumulation cell count crossing this geometry.
    """
    import rasterio, json
    import numpy as np
    from shapely.geometry import shape as shapely_shape

    try:
        shapely_geom = shapely_shape(json.loads(qgs_geom.asJson()))
    except Exception:
        return 0.0

    try:
        with rasterio.open(acc_path) as src:
            acc = src.read(1).astype("float32")
            transform = src.transform
            nodata = src.nodata

        if nodata is not None:
            acc = np.where(acc == nodata, 0.0, acc)

        total_len = shapely_geom.length
        if total_len == 0:
            return 0.0

        steps = np.linspace(0, total_len, n_samples)
        peak = 0.0
        for dist in steps:
            pt = shapely_geom.interpolate(dist)
            col = int((pt.x - transform.c) / transform.a)
            row = int((pt.y - transform.f) / transform.e)
            if 0 <= row < acc.shape[0] and 0 <= col < acc.shape[1]:
                v = float(acc[row, col])
                if v > peak:
                    peak = v
        return peak
    except Exception:
        return 0.0
