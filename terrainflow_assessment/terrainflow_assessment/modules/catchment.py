"""
catchment.py — Contributing catchment delineation and SCS runoff model.

Combines logic from the base plugin's contributing_area.py and scs_runoff.py
into a single module.  Exposes:

  fast_contributing_area()   — uphill BFS from site boundary
  clip_dem_to_polygon()      — clip DEM to catchment + buffer
  SCSRunoff                  — Curve Number rainfall-runoff model
"""

import logging

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize, shapes as rasterio_shapes
from scipy.ndimage import zoom as _zoom

_log = logging.getLogger(__name__)

_MAX_PREVIEW_CELLS = 500_000   # ~700 × 700 — keep preview under ~5 s


# ---------------------------------------------------------------------------
# Contributing area (uphill BFS)
# ---------------------------------------------------------------------------

def fast_contributing_area(dem_path, boundary_path, progress_callback=None):
    """
    Delineate the contributing catchment from a site boundary using an uphill
    BFS flood-fill.  Any cell connected to the boundary via a monotonically
    non-decreasing elevation path contributes surface runoff to the site.

    Parameters
    ----------
    dem_path : str
        Path to the full-resolution DEM GeoTIFF.
    boundary_path : str
        Path to a polygon vector file (site boundary, in any CRS).
    progress_callback : callable(int, str) or None

    Returns
    -------
    dict
        catchment_polygon : shapely Polygon
        clip_polygon      : shapely Polygon  (catchment bbox + 20 % buffer)
        area_ha           : float
        dem_area_ha       : float
        coverage_pct      : float
        scale             : float
    """
    import heapq
    import geopandas as gpd
    from shapely.geometry import shape as shapely_shape, box
    from shapely.ops import unary_union

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _p(5, "Reading DEM...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    rows, cols = dem.shape
    n_cells = rows * cols
    cell_w = abs(transform.a)
    cell_h = abs(transform.e)
    dem_area_ha = (rows * cell_h) * (cols * cell_w) / 10_000

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    scale = min(1.0, (_MAX_PREVIEW_CELLS / n_cells) ** 0.5)
    if scale < 1.0:
        _p(10, f"Downsampling DEM to {scale * 100:.0f}% for preview...")
        work_dem = _zoom(dem, scale, order=1).astype("float32")
        work_transform = Affine(
            transform.a / scale, transform.b, transform.c,
            transform.d, transform.e / scale, transform.f,
        )
    else:
        work_dem = dem
        work_transform = transform

    nrows, ncols = work_dem.shape

    _p(20, "Rasterizing site boundary...")
    gdf = gpd.read_file(boundary_path)
    gdf = gdf.to_crs(crs.to_wkt())
    gdf_dissolved = gdf.dissolve()

    inside_mask = rasterize(
        [(geom, 1) for geom in gdf_dissolved.geometry],
        out_shape=work_dem.shape,
        transform=work_transform,
        fill=0, all_touched=True, dtype="uint8",
    ).astype(bool)

    if not inside_mask.any():
        raise RuntimeError(
            "Site boundary does not intersect the DEM extent. "
            "Check that both layers use the same CRS."
        )

    boundary_lines = [
        geom.exterior
        for geom in gdf_dissolved.geometry
        if geom is not None and hasattr(geom, "exterior")
    ]
    if not boundary_lines:
        raise RuntimeError("Site boundary has no valid polygon geometry.")

    boundary_line_mask = rasterize(
        [(line, 1) for line in boundary_lines],
        out_shape=work_dem.shape,
        transform=work_transform,
        fill=0, dtype="uint8",
    ).astype(bool)

    seed_rows, seed_cols = np.where(boundary_line_mask)
    if seed_rows.size == 0:
        raise RuntimeError(
            "Site boundary line did not intersect the DEM. "
            "Check that both layers use the same CRS."
        )

    _p(35, "Tracing contributing catchment from slope...")
    NBRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    visited = inside_mask & ~boundary_line_mask
    contributing = inside_mask.copy()

    heap = []
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist()):
        if not visited[r, c]:
            elev = float(work_dem[r, c])
            if np.isfinite(elev):
                visited[r, c] = True
                heapq.heappush(heap, (elev, r, c))

    _p(50, "Expanding uphill watershed...")
    while heap:
        elev, r, c = heapq.heappop(heap)
        contributing[r, c] = True
        for dr, dc in NBRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols and not visited[nr, nc]:
                next_elev = float(work_dem[nr, nc])
                if np.isfinite(next_elev) and next_elev >= elev:
                    visited[nr, nc] = True
                    heapq.heappush(heap, (next_elev, nr, nc))

    _p(85, "Building catchment polygon...")
    polys = [
        shapely_shape(geom)
        for geom, val in rasterio_shapes(
            contributing.astype("uint8"), transform=work_transform
        )
        if val == 1
    ]
    if not polys:
        raise RuntimeError("Catchment raster produced no polygon geometry.")

    catchment_poly = unary_union(polys)
    area_ha = catchment_poly.area / 10_000

    minx, miny, maxx, maxy = catchment_poly.bounds
    buf_x = (maxx - minx) * 0.20
    buf_y = (maxy - miny) * 0.20
    clip_poly = box(minx - buf_x, miny - buf_y, maxx + buf_x, maxy + buf_y)

    coverage_pct = (area_ha / dem_area_ha * 100.0) if dem_area_ha > 0 else 0.0

    _p(100, "Done.")
    return {
        "catchment_polygon": catchment_poly,
        "clip_polygon": clip_poly,
        "area_ha": round(area_ha, 1),
        "dem_area_ha": round(dem_area_ha, 1),
        "coverage_pct": round(coverage_pct, 1),
        "scale": scale,
    }


def clip_dem_to_polygon(dem_path, clip_polygon, output_path):
    """Clip DEM to a shapely polygon (in the DEM's CRS) and save."""
    import rasterio.mask
    from shapely.geometry import mapping

    with rasterio.open(dem_path) as src:
        out_image, out_transform = rasterio.mask.mask(
            src, [mapping(clip_polygon)], crop=True,
            nodata=src.nodata if src.nodata is not None else -9999,
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "nodata": src.nodata if src.nodata is not None else -9999,
        })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    return output_path


# ---------------------------------------------------------------------------
# SCS Curve Number rainfall-runoff model
# ---------------------------------------------------------------------------

class SCSRunoff:
    """
    SCS Curve Number rainfall-runoff model.

    Calculates the depth of surface runoff from rainfall based on the soil's
    Curve Number (CN) and antecedent moisture condition (AMC).

    Reference: USDA-NRCS National Engineering Handbook, Part 630.
    """

    _AMC = {
        "dry":    lambda cn: 4.2 * cn / (10 - 0.058 * cn),
        "normal": lambda cn: cn,
        "wet":    lambda cn: 23 * cn / (10 + 0.13 * cn),
    }

    # Typical CN values by soil type for the soil selector dropdown
    SOIL_REFERENCE = {
        "Sand":        39,
        "Sandy loam":  49,
        "Loam":        61,
        "Clay loam":   74,
        "Clay":        80,
    }

    STORM_PRESETS = {
        "Custom":                (None, None),
        "Light (~25 mm/hr)":     (25, 1),
        "Moderate (~40 mm/hr)":  (40, 1),
        "Heavy (~65 mm/hr)":     (65, 1),
        "Extreme (~80 mm/hr)":   (80, 1),
    }

    def adjust_cn(self, cn, moisture_condition):
        """Adjust CN for antecedent moisture condition. Returns CN in [1, 100]."""
        adjusted = self._AMC[moisture_condition](cn)
        return max(1.0, min(100.0, adjusted))

    def runoff_depth(self, rainfall_mm, cn):
        """
        Calculate runoff depth (Q) from rainfall (P) and curve number (CN).

        SCS formula:
            S  = (25400 / CN) - 254
            Ia = 0.2 * S
            Q  = (P - Ia)^2 / (P - Ia + S)  if P > Ia, else 0

        Returns runoff in mm (>= 0).
        """
        if cn <= 0:
            return 0.0
        S = (25400 / cn) - 254
        Ia = 0.2 * S
        if rainfall_mm <= Ia:
            return 0.0
        return max(0.0, (rainfall_mm - Ia) ** 2 / (rainfall_mm - Ia + S))

    def runoff_ratio(self, rainfall_mm, cn):
        """Return fraction of rainfall that becomes runoff (0–1)."""
        if rainfall_mm <= 0:
            return 0.0
        return self.runoff_depth(rainfall_mm, cn) / rainfall_mm

    def catchment_volume(self, runoff_mm, catchment_area_m2):
        """Total runoff volume (m³) for a catchment."""
        return (runoff_mm / 1000.0) * catchment_area_m2

    def build_cn_raster(self, shape, transform, zone_geoms_cn, default_cn, moisture):
        """
        Build a moisture-adjusted CN raster from user-drawn zone polygons.

        Parameters
        ----------
        shape : tuple (rows, cols)
        transform : rasterio.Affine
        zone_geoms_cn : list of (shapely_geometry, int)
        default_cn : int
        moisture : str — 'dry', 'normal', or 'wet'

        Returns float32 array of adjusted CN values, same shape as DEM.
        """
        from rasterio.features import rasterize as _rasterize

        cn_array = np.full(shape, float(default_cn), dtype="float32")
        for geom, cn_val in zone_geoms_cn:
            if geom is None or geom.is_empty:
                continue
            burned = _rasterize(
                [(geom, float(cn_val))],
                out_shape=shape, transform=transform,
                fill=0.0, dtype="float32",
            )
            cn_array = np.where(burned > 0, burned, cn_array)

        adjusted = np.vectorize(
            lambda cn: self.adjust_cn(float(cn), moisture)
        )(cn_array)
        return adjusted.astype("float32")

    def build_runoff_raster(self, cn_array, rainfall_mm):
        """Per-cell runoff depth (mm) from a moisture-adjusted CN raster."""
        def _q(cn):
            if cn <= 0:
                return 0.0
            S = (25400.0 / cn) - 254.0
            Ia = 0.2 * S
            if rainfall_mm <= Ia:
                return 0.0
            return max(0.0, (rainfall_mm - Ia) ** 2 / (rainfall_mm - Ia + S))
        return np.vectorize(_q)(cn_array).astype("float32")

    @staticmethod
    def parse_hyetograph_csv(path):
        """
        Parse a rainfall time-series CSV into cumulative rainfall points.

        Expected columns: ``time_min``, ``rainfall_mm`` (header required).
        Auto-detects cumulative vs per-interval format.
        Always prepends (0, 0.0) baseline.

        Returns list of (time_min: int, cum_rainfall_mm: float).
        """
        import csv

        rows = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty.")
            headers = [h.strip().lower() for h in reader.fieldnames]
            if "time_min" not in headers or "rainfall_mm" not in headers:
                raise ValueError(
                    f"CSV must have columns 'time_min' and 'rainfall_mm'. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                try:
                    rows.append((float(row["time_min"]), float(row["rainfall_mm"])))
                except (KeyError, ValueError):
                    continue

        if not rows:
            raise ValueError("CSV contains no valid data rows.")

        rows.sort(key=lambda x: x[0])
        rain_vals = [r for _, r in rows]
        is_cumulative = all(
            rain_vals[i] >= rain_vals[i - 1] - 1e-6 for i in range(1, len(rain_vals))
        )

        if is_cumulative:
            cum_pairs = [(int(round(t)), float(r)) for t, r in rows]
        else:
            cum = 0.0
            cum_pairs = []
            for t, r in rows:
                cum += max(0.0, r)
                cum_pairs.append((int(round(t)), cum))

        if cum_pairs[0][0] != 0:
            cum_pairs.insert(0, (0, 0.0))

        return cum_pairs
