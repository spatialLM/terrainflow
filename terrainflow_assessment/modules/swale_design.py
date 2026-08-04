"""
swale_design.py — Swale design helpers and soil reference data.

Provides:
  SOIL_REFERENCE            — soil type → CN mapping (from SCS model)
  recommend_swale_length()  — size a swale to intercept a given inflow volume
  required_storage_at_length() — does the swale **as drawn** hold its event?
  inflow_profile()          — where along the alignment the water actually arrives
  overtopping_station()     — first point where concentrated inflow outruns storage
  snap_geometry_to_contour() — snap a drawn line to the nearest contour elevation
  contour_to_swale_geometry() — convert a full contour line to a swale QgsGeometry
  sample_total_inflow()     — total intercepted flow accumulation along a feature
"""

from dataclasses import dataclass

from ..core.sizing.primitives import trapezoid_section
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

def _channel_section(top_width, depth, side_slope):
    """Trapezoid actually achievable at *top_width* / *depth* with *side_slope* batter.

    When 2·z·d exceeds the top width the battered walls meet before the drawn depth is
    reached: the deepest section that fits under that top width is the triangle of
    height T/(2z), area T²/(4z). Clamping only the bottom width to zero while keeping
    the full depth built a taller triangle than the batter permits and over-stated
    storage — in the narrow-and-deep corner, which is where a swale is tightest.
    """
    effective_depth = depth
    if side_slope > 0:
        effective_depth = min(depth, top_width / (2.0 * side_slope))
    bottom_width = max(0.0, top_width - 2.0 * side_slope * effective_depth)
    return trapezoid_section(top_width, bottom_width, effective_depth)


def recommend_swale_length(peak_inflow_m3, depth, width, *,
                           side_slope=1.0, freeboard=0.8,
                           infiltration_mm_hr=0.0, duration_hr=0.0):
    """
    Estimate the swale length needed to manage a design-storm inflow volume.

    A swale is an infiltration/detention feature, **not** a full-storm reservoir,
    so it is sized by balancing inflow against the two ways a swale actually sheds
    water over the event: live storage in its (trapezoidal) trench, and infiltration
    through its wetted footprint.

    Model
    -----
    Over a storm of ``duration_hr`` hours, per metre of swale::

        capacity_per_m = A_x · freeboard          (trench storage, m³/m)
                       + (f / 1000) · duration · T (infiltration, m³/m)

    where ``A_x`` is the trapezoidal cross-section area for ``depth`` / top width
    ``T`` at ``side_slope`` (H:V run-per-rise), and ``f`` is the soil infiltration
    rate (mm/hr). The required length is then::

        L = peak_inflow_m3 / capacity_per_m

    With no infiltration data (``infiltration_mm_hr`` or ``duration_hr`` == 0) this
    reduces to pure trapezoidal storage sizing — still a larger capacity (shorter
    length) than the old rectangular ``depth·width`` approximation, because the
    battered walls add cross-section.

    Parameters
    ----------
    peak_inflow_m3 : float — design-storm inflow volume (m³)
    depth : float — swale depth (m)
    width : float — swale **top** width (m)
    side_slope : float — wall batter, H:V run-per-rise (default 1.0 = 1:1)
    freeboard : float — usable fraction of the trench cross-section (default 0.8)
    infiltration_mm_hr : float — soil steady-state infiltration rate (mm/hr)
    duration_hr : float — storm duration (hr); infiltration only counts over the event

    Returns
    -------
    float — recommended swale length (m), or 0.0 if inputs are invalid.
    """
    if depth <= 0 or width <= 0 or peak_inflow_m3 <= 0:
        return 0.0

    # Trapezoidal cross-section (bottom width narrows with battered walls).
    section = _channel_section(width, depth, side_slope)

    storage_per_m = section.area * freeboard
    infil_per_m = (max(0.0, infiltration_mm_hr) / 1000.0) * max(0.0, duration_hr) * width
    capacity_per_m = storage_per_m + infil_per_m
    if capacity_per_m <= 0:
        return 0.0

    return round(peak_inflow_m3 / capacity_per_m, 1)


@dataclass
class StorageCheck:
    """Whether a swale as drawn holds its event, and what would fix it if not."""
    available_m3: float = 0.0        # storage + infiltration at the drawn length
    storage_m3: float = 0.0          # of which: live storage in the trench
    infiltration_m3: float = 0.0     # of which: soakage over the event
    inflow_m3: float = 0.0
    deficit_m3: float = 0.0          # 0 when it holds
    required_section_m2: float = 0.0  # cross-section that would hold the event
    required_depth_m: float = 0.0    # depth at the drawn top width that would hold it
    depth_reachable: bool = True     # False → no depth at this top width can hold it
    recommended_length_m: float = 0.0  # length that would hold it at these dimensions
    holds: bool = False


def required_storage_at_length(inflow_m3, length_m, depth, width, *,
                               side_slope=1.0, freeboard=0.8,
                               infiltration_mm_hr=0.0, duration_hr=0.0):
    """Does this swale, **as drawn**, hold its event — and if not, by how much?

    This is the readout the properties dialog leads with, in place of "recommended
    length". Recommended length is a poor primary framing for a contour swale because
    the inflow scales with the length: a longer swale intercepts proportionally more
    hillside, so the ratio ``required_length / drawn_length`` is very nearly constant
    and the user can never clear a red flag by extending. (The previous implementation
    made that worse by dividing by a rectangular ``depth × width``, ignoring the
    battered walls, the freeboard allowance and infiltration entirely — so it also
    disagreed with :func:`recommend_swale_length`, which the dialog never called.)

    Deficit at the drawn length is always actionable: deepen, widen, or route the
    surplus to a downstream feature, and each responds immediately.

    ``required_depth_m`` inverts the trapezoid at the drawn top width ``T``. With
    ``A = d·(T + b)/2`` and ``b = T − 2·z·d`` this is ``A = d·T − z·d²``, so::

        d = (T − √(T² − 4·z·A)) / (2·z)          for z > 0
        d = A / T                                 for vertical walls

    taking the smaller root (the shallower of the two solutions). When the discriminant
    is negative no depth at this top width can reach the required section — the walls
    close in first — and the drawn depth is returned unchanged so the caller can say
    "widen it" rather than print an imaginary number.

    Returns a :class:`StorageCheck`. Invalid inputs give an all-zero result with
    ``holds`` False.
    """
    result = StorageCheck(inflow_m3=max(0.0, float(inflow_m3 or 0.0)))
    if depth <= 0 or width <= 0 or length_m <= 0:
        return result

    section = _channel_section(width, depth, side_slope)

    result.storage_m3 = section.area * freeboard * length_m
    result.infiltration_m3 = (
        (max(0.0, infiltration_mm_hr) / 1000.0) * max(0.0, duration_hr)
        * width * length_m
    )
    result.available_m3 = result.storage_m3 + result.infiltration_m3
    result.deficit_m3 = max(0.0, result.inflow_m3 - result.available_m3)
    result.holds = result.deficit_m3 <= 0

    result.recommended_length_m = recommend_swale_length(
        result.inflow_m3, depth, width, side_slope=side_slope, freeboard=freeboard,
        infiltration_mm_hr=infiltration_mm_hr, duration_hr=duration_hr,
    )

    # The cross-section that would close the gap, and the depth that delivers it.
    infil_per_m = (max(0.0, infiltration_mm_hr) / 1000.0) * max(0.0, duration_hr) * width
    needed_per_m = max(0.0, result.inflow_m3 / length_m - infil_per_m)
    result.required_section_m2 = needed_per_m / freeboard if freeboard > 0 else 0.0

    target = result.required_section_m2
    if target <= 0:
        result.required_depth_m = 0.0
    elif side_slope <= 0:
        result.required_depth_m = target / width
    else:
        disc = width * width - 4.0 * side_slope * target
        if disc < 0:
            # The battered walls meet before the section is reached — the widest
            # possible section at this top width is T²/(4z), and we need more.
            result.required_depth_m = depth
            result.depth_reachable = False
        else:
            result.required_depth_m = (width - disc ** 0.5) / (2.0 * side_slope)

    return result


def inflow_profile(distances_m, volumes_m3, length_m, n_stations=24):
    """Distribute a feature's catchment along its alignment, station by station.

    Real inflow is concentrated where drainage lines cross a swale, not spread evenly
    along it — so a swale whose *total* capacity is adequate can still overtop locally.
    ``distances_m`` is each contributing cell's distance along the centreline (from
    projecting the cell onto the alignment) and ``volumes_m3`` its runoff volume; both
    come from the catchment label raster the caller already has, so this is a bincount
    rather than new analysis.

    Returns a dict with:

    ``stations``      station midpoints (m along the alignment)
    ``inflow_m3``     runoff arriving in each station's reach
    ``cumulative_m3`` running total from the start of the alignment
    ``peak_station``  station carrying the most inflow — where a check-bank belongs
    ``uniformity``    1.0 when inflow is perfectly even, → 0 when it all arrives at
                      one point. Below ~0.5 the lumped total-vs-total comparison is
                      optimistic and the local check below matters.

    An empty or zero-length input gives an all-zero profile rather than raising.
    """
    import numpy as np

    n = max(1, int(n_stations))
    step = float(length_m) / n if length_m and length_m > 0 else 0.0
    stations = [(i + 0.5) * step for i in range(n)]
    empty = {
        "stations": stations,
        "inflow_m3": [0.0] * n,
        "cumulative_m3": [0.0] * n,
        "peak_station": 0.0,
        "uniformity": 1.0,
        "station_length_m": step,
    }
    if step <= 0:
        return empty

    d = np.asarray(list(distances_m), dtype="float64")
    v = np.asarray(list(volumes_m3), dtype="float64")
    if d.size == 0 or v.size == 0 or d.size != v.size:
        return empty

    idx = np.clip((d / step).astype(int), 0, n - 1)
    binned = np.bincount(idx, weights=v, minlength=n)
    total = float(binned.sum())
    if total <= 0:
        return empty

    # Uniformity: mean/max, i.e. how far the worst station is above an even share.
    uniformity = float(binned.mean() / binned.max()) if binned.max() > 0 else 1.0

    return {
        "stations": stations,
        "inflow_m3": [float(x) for x in binned],
        "cumulative_m3": [float(x) for x in np.cumsum(binned)],
        "peak_station": stations[int(np.argmax(binned))],
        "uniformity": round(uniformity, 3),
        "station_length_m": step,
    }


def overtopping_station(profile, capacity_per_m, freeboard=1.0):
    """First point along the alignment where arriving water outruns storage upstream.

    Walks the profile from the start comparing cumulative inflow against the storage
    available in the reach traversed so far. A level swale shares water along its whole
    length, so this only bites when inflow is concentrated — which is exactly when the
    lumped "total capacity vs total inflow" verdict is too generous.

    Returns ``(station_m, surplus_m3)``, or ``(None, 0.0)`` when it holds all the way
    along. ``station_m`` is a distance along the drawn alignment, so the caller can
    place a marker with ``geometry.interpolate(station_m)``.
    """
    if not profile or capacity_per_m <= 0:
        return None, 0.0
    stations = profile.get("stations") or []
    cumulative = profile.get("cumulative_m3") or []
    step = profile.get("station_length_m") or 0.0
    if not stations or step <= 0:
        return None, 0.0

    usable_per_m = capacity_per_m * freeboard
    for i, station in enumerate(stations):
        reach_m = (i + 1) * step
        available = usable_per_m * reach_m
        if cumulative[i] > available:
            return station, round(cumulative[i] - available, 1)
    return None, 0.0


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


def contour_section(contour_coords, end_a_xy, end_b_xy):
    """
    Extract the section of a contour polyline between two endpoint positions.

    Used by the reshape tool for contour-locked swales: each endpoint is
    projected onto the contour and the polyline between the two projections is
    returned, so a slid endpoint always re-follows the contour (same substring
    behaviour as the original Pick Segment draw).

    Parameters
    ----------
    contour_coords : list of (x, y) — the full contour polyline
    end_a_xy, end_b_xy : tuple (x, y) — the two endpoints (any order)

    Returns
    -------
    list of (x, y) for the section, or None if it can't be built
    (degenerate contour or zero-length section).
    """
    if not contour_coords or len(contour_coords) < 2:
        return None
    try:
        from shapely.geometry import LineString, Point
        from shapely.ops import substring

        line = LineString(contour_coords)
        d0 = line.project(Point(end_a_xy))
        d1 = line.project(Point(end_b_xy))
        if d0 > d1:
            d0, d1 = d1, d0
        if d1 - d0 <= 0:
            return None
        section = substring(line, d0, d1)
        if section.is_empty or section.geom_type != "LineString":
            return None
        return [(float(x), float(y)) for x, y in section.coords]
    except Exception:
        return None


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
    import numpy as np
    import rasterio

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
    import json

    import numpy as np
    import rasterio
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


def sample_total_inflow(qgs_geom, acc_path):
    """
    Total flow accumulation (cell count) intercepted along a geometry.

    Walks the line at half-cell steps and sums the accumulation of every
    **unique** raster cell it passes through, giving a cheap relative measure of how
    much flow a line intercepts — useful for *ranking* candidate alignments against
    each other, which is what the contour analysis uses it for.

    .. warning::
       **Not a catchment area.** Flow accumulation is *cumulative*, so two adjacent
       cells on a contour each carry nearly the whole hillside above them; summing
       across the line therefore over-counts the shared catchment by roughly
       (line length ÷ cell size). Using this as a per-feature inflow was half the
       reason the water balance reported "100% of the storm held on site". For a real
       contributing area use
       :func:`~terrainflow_assessment.modules.flow_graph.label_direct_catchments`,
       whose per-feature catchments are mutually exclusive and exhaustive.

    Polygons (basins) are sampled along their exterior boundary. Returns 0.0 on
    any failure (missing raster, degenerate geometry).

    Parameters
    ----------
    qgs_geom : QgsGeometry (polyline or polygon)
    acc_path : str — path to flow accumulation GeoTIFF

    Returns
    -------
    float — total intercepted accumulation cell count.
    """
    import json

    import numpy as np
    import rasterio
    from shapely.geometry import shape as shapely_shape

    try:
        shapely_geom = shapely_shape(json.loads(qgs_geom.asJson()))
        if hasattr(shapely_geom, "exterior"):   # Polygon → sample its boundary
            shapely_geom = shapely_geom.exterior
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

        step = max(abs(transform.a) / 2.0, 1e-6)   # half-cell keeps every crossing
        seen = set()
        total = 0.0
        for dist in np.arange(0.0, total_len + step, step):
            pt = shapely_geom.interpolate(min(dist, total_len))
            col = int((pt.x - transform.c) / transform.a)
            row = int((pt.y - transform.f) / transform.e)
            if not (0 <= row < acc.shape[0] and 0 <= col < acc.shape[1]):
                continue
            if (row, col) in seen:
                continue
            seen.add((row, col))
            total += max(0.0, float(acc[row, col]))
        return total
    except Exception:
        return 0.0
