"""
contour_analysis.py — Contour extraction, slope filtering, and flow-crossing ranking.

Provides:
  extract_contours()        — extract contours from DEM via GDAL
  filter_by_slope()         — reject contours where mean slope > max_slope_deg
  rank_by_flow_crossing()   — rank contours by peak flow accumulation crossing them
  clip_to_usable_area()     — clip contours to a user-defined usable polygon
  ContourFeature            — lightweight container for a ranked contour
"""

import logging
import numpy as np

_log = logging.getLogger(__name__)


class ContourFeature:
    """A single contour line with its analysis results."""

    def __init__(self, geometry, elevation, rank=None, peak_acc=0.0,
                 mean_slope_deg=0.0, length_m=0.0,
                 cell_area_m2=None, runoff_mm=None):
        """
        Parameters
        ----------
        geometry : shapely LineString
        elevation : float — contour elevation (m)
        rank : int or None — 1 = highest flow crossing
        peak_acc : float — peak flow accumulation crossing this contour
        mean_slope_deg : float — mean terrain slope along the contour
        length_m : float — contour length (m)
        cell_area_m2 : float or None — DEM cell area (m²), used for ha / m³ labels
        runoff_mm : float or None — event runoff depth (mm), used for m³ inflow label
        """
        self.geometry = geometry
        self.elevation = elevation
        self.rank = rank
        self.peak_acc = peak_acc
        self.mean_slope_deg = mean_slope_deg
        self.length_m = length_m
        self.cell_area_m2 = cell_area_m2
        self.runoff_mm = runoff_mm
        self.selected = True   # user can deselect individual contours

    @property
    def label(self):
        rank_str = f"#{self.rank} " if self.rank else ""
        if self.cell_area_m2:
            area_ha = self.peak_acc * self.cell_area_m2 / 10_000
            area_str = f"{area_ha:,.1f} ha upslope"
            if self.runoff_mm:
                inflow_m3 = self.peak_acc * self.cell_area_m2 * self.runoff_mm / 1000.0
                inflow_str = f" — {inflow_m3:,.0f} m³ inflow"
            else:
                inflow_str = ""
            return (
                f"{rank_str}Elev {self.elevation:.1f} m — "
                f"{area_str}{inflow_str}"
            )
        # Fallback when cell size is not known
        return (
            f"{rank_str}Elev {self.elevation:.1f} m — "
            f"peak acc {self.peak_acc:,.0f} cells"
        )


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contours(dem_path, interval_m=1.0, output_path=None):
    """
    Extract contour lines from a DEM at fixed elevation intervals via GDAL.

    Parameters
    ----------
    dem_path : str — path to DEM GeoTIFF
    interval_m : float — contour interval in metres (default 1.0)
    output_path : str or None — if given, write contours to a GeoPackage

    Returns
    -------
    list of ContourFeature (geometry in DEM CRS, elevation from GDAL output)
    """
    import tempfile, os, subprocess
    import rasterio
    import geopandas as gpd

    if output_path is None:
        output_path = tempfile.mktemp(suffix="_contours.gpkg")

    # Run GDAL contour
    cmd = [
        "gdal_contour",
        "-a", "ELEV",
        "-i", str(interval_m),
        dem_path,
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"gdal_contour failed: {result.stderr}")
    except FileNotFoundError:
        # gdal_contour not on PATH — fall back to rasterio + shapely marching squares
        return _extract_contours_scipy(dem_path, interval_m)

    try:
        gdf = gpd.read_file(output_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read contour output: {exc}")

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        elev = float(row.get("ELEV", 0.0) or 0.0)
        length_m = geom.length
        features.append(ContourFeature(
            geometry=geom,
            elevation=elev,
            length_m=length_m,
        ))

    return features


def _extract_contours_scipy(dem_path, interval_m):
    """Fallback contour extraction using scipy.ndimage (no external GDAL)."""
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    from shapely.geometry import LineString, MultiLineString
    from skimage import measure

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    valid = dem[np.isfinite(dem)]
    if len(valid) == 0:
        return []

    elev_min = float(np.nanmin(valid))
    elev_max = float(np.nanmax(valid))
    levels = np.arange(
        np.ceil(elev_min / interval_m) * interval_m,
        elev_max,
        interval_m,
    )

    cell_w = abs(transform.a)
    cell_h = abs(transform.e)
    features = []

    for elev in levels:
        contours_rc = measure.find_contours(np.nan_to_num(dem, nan=elev - 1), elev)
        for rc_path in contours_rc:
            if len(rc_path) < 2:
                continue
            # Convert row/col → map coordinates
            xs = transform.c + rc_path[:, 1] * cell_w + cell_w / 2
            ys = transform.f + rc_path[:, 0] * transform.e + transform.e / 2
            coords = list(zip(xs, ys))
            if len(coords) < 2:
                continue
            geom = LineString(coords)
            features.append(ContourFeature(
                geometry=geom,
                elevation=float(elev),
                length_m=geom.length,
            ))

    return features


# ---------------------------------------------------------------------------
# Slope filter
# ---------------------------------------------------------------------------

def filter_by_slope(contours, dem_path, max_slope_deg=18.0, n_samples=20):
    """
    Remove contours where the mean terrain slope exceeds max_slope_deg.

    Samples N evenly-spaced points along each contour, queries the DEM slope
    at those points, and rejects the contour if the mean exceeds the limit.

    Parameters
    ----------
    contours : list of ContourFeature
    dem_path : str — path to DEM GeoTIFF (slope computed on the fly)
    max_slope_deg : float — default 18° (approx 1:3 grade)
    n_samples : int — number of sample points per contour

    Returns
    -------
    list of ContourFeature — only contours with mean slope ≤ max_slope_deg,
    with ``mean_slope_deg`` populated on each feature.
    """
    import rasterio
    import scipy.ndimage as ndi

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        nodata = src.nodata
        cell_w = abs(transform.a)
        cell_h = abs(transform.e)

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    dz_dy, dz_dx = np.gradient(np.nan_to_num(dem, nan=0.0), cell_h, cell_w)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype("float32")

    def _sample_slope(geom):
        """Sample slope at N evenly-spaced points along a line."""
        total_len = geom.length
        if total_len == 0:
            return 0.0
        steps = np.linspace(0, total_len, min(n_samples, max(2, int(total_len / cell_w))))
        values = []
        for dist in steps:
            pt = geom.interpolate(dist)
            col = int((pt.x - transform.c) / transform.a)
            row = int((pt.y - transform.f) / transform.e)
            if 0 <= row < slope_deg.shape[0] and 0 <= col < slope_deg.shape[1]:
                values.append(float(slope_deg[row, col]))
        return float(np.mean(values)) if values else 0.0

    valid = []
    for feat in contours:
        mean_slope = _sample_slope(feat.geometry)
        feat.mean_slope_deg = mean_slope
        if mean_slope <= max_slope_deg:
            valid.append(feat)

    return valid


# ---------------------------------------------------------------------------
# Flow-crossing ranking
# ---------------------------------------------------------------------------

def rank_by_flow_crossing(contours, acc_path, n_samples=50):
    """
    Rank contours by peak flow accumulation crossing them.

    Samples the flow accumulation raster at N points along each contour and
    records the maximum (peak) value.  Contours are sorted descending by
    peak accumulation and assigned a rank starting at 1.

    Parameters
    ----------
    contours : list of ContourFeature
    acc_path : str — path to flow accumulation GeoTIFF
    n_samples : int — sample points per contour

    Returns
    -------
    list of ContourFeature sorted by peak_acc descending, with rank set.
    """
    import rasterio

    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float32")
        transform = src.transform
        cell_w = abs(transform.a)
        nodata = src.nodata

    if nodata is not None:
        acc = np.where(acc == nodata, 0.0, acc)

    for feat in contours:
        geom = feat.geometry
        total_len = geom.length
        if total_len == 0:
            feat.peak_acc = 0.0
            continue

        steps = np.linspace(0, total_len, min(n_samples, max(2, int(total_len / cell_w))))
        peak = 0.0
        for dist in steps:
            pt = geom.interpolate(dist)
            col = int((pt.x - transform.c) / transform.a)
            row = int((pt.y - transform.f) / transform.e)
            if 0 <= row < acc.shape[0] and 0 <= col < acc.shape[1]:
                v = float(acc[row, col])
                if v > peak:
                    peak = v
        feat.peak_acc = peak

    contours.sort(key=lambda f: f.peak_acc, reverse=True)
    for i, feat in enumerate(contours):
        feat.rank = i + 1

    return contours


# ---------------------------------------------------------------------------
# Usable area clipping
# ---------------------------------------------------------------------------

def clip_to_usable_area(contours, usable_polygon):
    """
    Clip contour lines to a user-defined usable area polygon.

    Contours entirely outside the polygon are dropped.  Contours that cross
    the boundary are trimmed to the interior portion(s).

    Parameters
    ----------
    contours : list of ContourFeature
    usable_polygon : shapely Polygon or MultiPolygon

    Returns
    -------
    list of ContourFeature — trimmed, geometry replaced with clipped version.
    Rank order is preserved.
    """
    from shapely.geometry import MultiLineString, LineString

    clipped = []
    for feat in contours:
        try:
            intersection = feat.geometry.intersection(usable_polygon)
        except Exception:
            continue

        if intersection.is_empty:
            continue

        # Normalise to a single LineString (longest segment if MultiLineString)
        if isinstance(intersection, LineString):
            clipped_geom = intersection
        elif isinstance(intersection, MultiLineString):
            geoms = list(intersection.geoms)
            clipped_geom = max(geoms, key=lambda g: g.length)
        else:
            # GeometryCollection or other — skip
            continue

        if clipped_geom.length < 0.5:  # ignore tiny slivers
            continue

        new_feat = ContourFeature(
            geometry=clipped_geom,
            elevation=feat.elevation,
            rank=feat.rank,
            peak_acc=feat.peak_acc,
            mean_slope_deg=feat.mean_slope_deg,
            length_m=clipped_geom.length,
        )
        clipped.append(new_feat)

    return clipped


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------

def analyse_contours(dem_path, acc_path, interval_m=1.0, max_slope_deg=18.0,
                     usable_polygon=None, progress_callback=None,
                     cell_area_m2=None, runoff_mm=None, min_length_m=0.0):
    """
    Run the full contour analysis pipeline:
      1. Extract contours from DEM
      2. Filter by slope (< max_slope_deg)
      3. Rank by flow crossing
      4. (Optional) clip to usable area

    Parameters
    ----------
    dem_path : str
    acc_path : str — flow accumulation raster path
    interval_m : float — contour interval (m)
    max_slope_deg : float — slope cutoff (degrees)
    usable_polygon : shapely geometry or None
    progress_callback : callable(int, str) or None
    cell_area_m2 : float or None — DEM cell area for ha/m³ labels
    runoff_mm : float or None — event runoff depth for m³ inflow labels

    Returns
    -------
    list of ContourFeature ranked by flow crossing
    """
    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _p(5, "Extracting contours...")
    contours = extract_contours(dem_path, interval_m)
    _log.info(f"Extracted {len(contours)} contours at {interval_m} m interval.")

    _p(30, f"Filtering by slope (< {max_slope_deg}°)...")
    contours = filter_by_slope(contours, dem_path, max_slope_deg)
    _log.info(f"{len(contours)} contours remain after slope filter.")

    if min_length_m > 0:
        before = len(contours)
        contours = [f for f in contours if f.geometry.length >= min_length_m]
        _log.info(f"{len(contours)} contours remain after min length filter "
                  f"({before - len(contours)} removed, min={min_length_m} m).")

    _p(60, "Ranking by flow crossing...")
    contours = rank_by_flow_crossing(contours, acc_path)

    if usable_polygon is not None:
        _p(85, "Clipping to usable area...")
        contours = clip_to_usable_area(contours, usable_polygon)
        _log.info(f"{len(contours)} contours after usable area clip.")

    # Stamp cell size / runoff onto every feature so labels can show ha and m³
    if cell_area_m2 is not None or runoff_mm is not None:
        for feat in contours:
            feat.cell_area_m2 = cell_area_m2
            feat.runoff_mm = runoff_mm

    _p(100, "Contour analysis complete.")
    return contours


# ---------------------------------------------------------------------------
# Sub-contour segment analysis
# ---------------------------------------------------------------------------

class SwaleSegment:
    """
    A recommended swale placement derived from a natural flow-crossing on a
    ranked contour.

    The segment geometry is sized to store the full inflow volume:
        required_length = inflow_m3 / (swale_depth_m × swale_width_m)
    and is centered on the peak accumulation point on the contour.

    If no runoff depth is available the natural landscape crossing extent
    (walk until acc drops to drop_fraction × peak) is used instead.
    """

    def __init__(self, geometry, elevation, peak_acc, contributing_ha,
                 inflow_m3, contour_rank, segment_rank, length_m,
                 required_length_m=None):
        self.geometry = geometry
        self.elevation = elevation
        self.peak_acc = peak_acc
        self.contributing_ha = contributing_ha
        self.inflow_m3 = inflow_m3
        self.contour_rank = contour_rank
        self.segment_rank = segment_rank
        self.length_m = length_m
        # Length required to store inflow (may be capped by contour length)
        self.required_length_m = required_length_m if required_length_m is not None else length_m

    @property
    def label(self):
        parts = [
            f"#{self.segment_rank}  Elev {self.elevation:.1f} m",
            f"{self.contributing_ha:,.1f} ha upslope",
        ]
        if self.inflow_m3:
            parts.append(f"{self.inflow_m3:,.0f} m³ inflow")
            parts.append(f"{self.required_length_m:.0f} m swale required")
        else:
            parts.append(f"{self.length_m:.0f} m long")
        return " — ".join(parts)


def find_swale_segments(contours, acc_path,
                        cell_area_m2, runoff_mm=None,
                        min_acc_ha=0.5, drop_fraction=0.25,
                        swale_depth_m=0.3, swale_width_m=0.6,
                        max_segments_per_contour=3,
                        progress_callback=None):
    """
    For each ranked contour, locate natural flow-crossing zones and size the
    swale segment to capture the full incoming runoff volume.

    Algorithm
    ---------
    1. Sample the flow accumulation raster at every cell-width along the contour.
    2. Find local peaks with contributing area ≥ *min_acc_ha* — these are where
       drainage lines cross the contour.
    3. Calculate inflow volume:  inflow_m3 = peak_acc × cell_area × runoff_mm
    4. Calculate required swale length to store that volume:
           required_length = inflow_m3 / (swale_depth_m × swale_width_m)
       This is based on a rectangular cross-section — conservative but standard.
    5. Extract a segment of that length centered on the peak point.
       If runoff_mm is unavailable, fall back to the landscape-walk extent
       (walk outward until acc drops below drop_fraction × peak).

    Parameters
    ----------
    contours         : list of ContourFeature (already ranked)
    acc_path         : str — flow accumulation GeoTIFF
    cell_area_m2     : float — DEM cell area (m²)
    runoff_mm        : float or None — event runoff depth (mm)
    min_acc_ha       : float — minimum contributing area (ha) for a crossing to
                       qualify.  Default 0.5 ha.
    drop_fraction    : float — fallback landscape walk: ends where acc < fraction × peak
    swale_depth_m    : float — swale design depth (m), default 0.3 m
    swale_width_m    : float — swale base width (m), default 0.6 m
    max_segments_per_contour : int — max crossings extracted per contour
    progress_callback : callable(int, str) or None

    Returns
    -------
    list of SwaleSegment, ranked globally by inflow_m3 descending
    (or contributing_ha if runoff_mm is not provided).
    """
    import rasterio
    from shapely.ops import substring

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    min_acc_cells = max(1, int(min_acc_ha * 10_000 / cell_area_m2))

    _p(5, "Loading accumulation raster…")
    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float32")
        transform = src.transform
        cell_w = abs(transform.a)
        nodata = src.nodata
    if nodata is not None:
        acc = np.where(acc == nodata, 0.0, acc)

    all_segments = []
    n = len(contours)

    for ci, feat in enumerate(contours):
        _p(5 + int(90 * ci / max(n, 1)), f"Analysing contour {ci + 1}/{n}…")
        geom = feat.geometry
        total_len = geom.length
        if total_len < 1.0:
            continue

        # Sample every cell_w along the contour
        step = max(cell_w, 1.0)
        n_steps = max(2, int(total_len / step))
        dists = np.linspace(0, total_len, n_steps)

        profile = []  # (distance_along_contour, acc_value)
        for d in dists:
            pt = geom.interpolate(d)
            col = int((pt.x - transform.c) / transform.a)
            row = int((pt.y - transform.f) / transform.e)
            v = float(acc[row, col]) if (
                0 <= row < acc.shape[0] and 0 <= col < acc.shape[1]
            ) else 0.0
            profile.append((float(d), v))

        if not profile:
            continue

        # --- Find local peaks (stream crossings) ---
        found_peaks = []  # (index_in_profile, peak_acc)
        i = 0
        while i < len(profile):
            d_i, v_i = profile[i]
            if v_i < min_acc_cells:
                i += 1
                continue

            # Is this a local maximum? Check neighbours within ±5 samples
            lo = max(0, i - 5)
            hi = min(len(profile), i + 6)
            neighbourhood = [v for _, v in profile[lo:hi]]
            if v_i < max(neighbourhood):
                i += 1
                continue

            found_peaks.append((i, v_i))

            # Advance past this peak by enough to avoid duplicates (10 samples)
            i += 10

        if not found_peaks:
            continue

        found_peaks.sort(key=lambda x: x[1], reverse=True)
        found_peaks = found_peaks[:max_segments_per_contour]

        for peak_idx, peak_acc_val in found_peaks:
            peak_dist = profile[peak_idx][0]
            contrib_ha = peak_acc_val * cell_area_m2 / 10_000
            inflow_m3 = (peak_acc_val * cell_area_m2 * runoff_mm / 1000.0
                         if runoff_mm else 0.0)

            if runoff_mm and swale_depth_m > 0 and swale_width_m > 0:
                # Size segment to store the full inflow volume
                # required_length = V / (depth × width)  [rectangular cross-section]
                cross_section_m2 = swale_depth_m * swale_width_m
                required_length = inflow_m3 / cross_section_m2
                required_length = max(required_length, 1.0)  # at least 1 m

                # Center the segment on the peak, capped to contour extent
                half = required_length / 2.0
                seg_start = max(0.0, peak_dist - half)
                seg_end = min(total_len, peak_dist + half)

                # If capped at one end, extend the other to preserve total length
                actual_len = seg_end - seg_start
                if actual_len < required_length:
                    shortfall = required_length - actual_len
                    if seg_start == 0.0:
                        seg_end = min(total_len, seg_end + shortfall)
                    else:
                        seg_start = max(0.0, seg_start - shortfall)

                capped = seg_end - seg_start < required_length - 0.5
            else:
                # Fallback: landscape-walk extent
                threshold = peak_acc_val * drop_fraction
                left = peak_idx
                while left > 0 and profile[left - 1][1] >= threshold:
                    left -= 1
                right = peak_idx
                while right < len(profile) - 1 and profile[right + 1][1] >= threshold:
                    right += 1
                seg_start = profile[left][0]
                seg_end = profile[right][0]
                required_length = seg_end - seg_start
                capped = False

            if seg_end - seg_start < 1.0:
                continue  # degenerate — skip

            try:
                seg_geom = substring(geom, seg_start, seg_end)
            except Exception:
                continue
            if seg_geom is None or seg_geom.is_empty:
                continue

            seg_len = seg_geom.length

            all_segments.append(SwaleSegment(
                geometry=seg_geom,
                elevation=feat.elevation,
                peak_acc=peak_acc_val,
                contributing_ha=round(contrib_ha, 1),
                inflow_m3=round(inflow_m3, 0),
                contour_rank=feat.rank or 0,
                segment_rank=0,
                length_m=round(seg_len, 0),
                required_length_m=round(required_length, 0),
            ))

    all_segments.sort(
        key=lambda s: s.inflow_m3 if runoff_mm else s.contributing_ha,
        reverse=True,
    )
    for i, seg in enumerate(all_segments):
        seg.segment_rank = i + 1

    _p(100, f"Found {len(all_segments)} candidate swale segment(s).")
    return all_segments
