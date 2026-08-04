"""
dem_loader.py — DEM loading and metadata extraction.

Reads a GeoTIFF DEM and exposes cell size, CRS, extent, and a path suitable
for passing into other analysis modules.
"""

import hashlib
import os

import numpy as np
import rasterio


class DEMValidationError(ValueError):
    """Raised when a loaded DEM fails a validation check (e.g. geographic CRS)."""
    pass


class DEMInfo:
    """Container for DEM spatial metadata."""

    def __init__(self, path):
        self.path = path
        self.cell_size_m = None       # metres (assumes projected CRS)
        self.cell_area_m2 = None
        self.crs = None
        self.crs_wkt = None
        self.transform = None
        self.width = None
        self.height = None
        self.nodata = None
        self.bounds = None            # (left, bottom, right, top)
        self.area_ha = None

    def __repr__(self):
        return (
            f"<DEMInfo {self.width}×{self.height} | "
            f"cell={self.cell_size_m:.2f} m | "
            f"area={self.area_ha:.1f} ha | "
            f"CRS={self.crs}>"
        )


def load_dem(dem_path):
    """
    Load a GeoTIFF DEM and return a :class:`DEMInfo` with its metadata.

    Parameters
    ----------
    dem_path : str
        Absolute path to the DEM GeoTIFF.

    Returns
    -------
    DEMInfo

    Raises
    ------
    RuntimeError
        If the file cannot be opened or is not a valid raster.
    """
    try:
        with rasterio.open(dem_path) as src:
            info = DEMInfo(dem_path)
            info.crs = src.crs
            info.crs_wkt = src.crs.to_wkt() if src.crs else None

            # Require a projected CRS so distances and areas are in metres.
            if not src.crs or not src.crs.is_projected:
                raise DEMValidationError(
                    f"DEM '{dem_path}' has a geographic (non-projected) CRS "
                    f"({src.crs}).  TerrainFlow requires a projected CRS such as "
                    "NZTM2000 (EPSG:2193) so that all distance and area calculations "
                    "are in metres.  Reproject the DEM before loading."
                )

            info.transform = src.transform
            info.width = src.width
            info.height = src.height
            info.nodata = src.nodata
            info.bounds = src.bounds

            cell_w = abs(src.transform.a)
            cell_h = abs(src.transform.e)
            info.cell_size_m = (cell_w + cell_h) / 2.0
            info.cell_area_m2 = cell_w * cell_h
            info.area_ha = (src.width * cell_w * src.height * cell_h) / 10_000.0

        return info

    except DEMValidationError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Cannot load DEM '{dem_path}': {exc}") from exc


# Beyond this, hash a deterministic sample instead of the whole file: a full read of a
# multi-gigabyte regional DEM would stall the UI on every save and open. The threshold is
# a function of file size alone, so the same file always takes the same branch — a saved
# digest and a later re-read agree.
_FULL_HASH_MAX_BYTES = 256 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


def dem_content_digest(dem_path):
    """A digest identifying this DEM's *contents*, tagged with how it was computed.

    Grid properties alone cannot identify a DEM — two different rasters covering the same
    site share cell size, CRS and extent — so a content digest is what makes "is this the
    same DEM?" answerable when a design file is opened on another machine.

    The returned string carries its algorithm as a prefix so a sampled digest can never
    compare equal to a full one, which would otherwise silently accept the wrong raster.
    """
    size = os.path.getsize(dem_path)
    digest = hashlib.sha256()
    # Size participates in the hash so two files sharing sampled regions still differ.
    digest.update(str(size).encode("ascii"))

    with open(dem_path, "rb") as handle:
        if size <= _FULL_HASH_MAX_BYTES:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
            return f"sha256:{digest.hexdigest()}"

        for offset in (0, max(0, size // 2 - _HASH_CHUNK_BYTES // 2),
                       max(0, size - _HASH_CHUNK_BYTES)):
            handle.seek(offset)
            digest.update(handle.read(_HASH_CHUNK_BYTES))
    return f"sha256-sampled:{digest.hexdigest()}"


def fingerprint_dem(dem_path, info=None):
    """Identity of the DEM at *dem_path*, as plain data for a design file.

    Returns the keys :class:`~terrainflow_assessment.modules.project_io.DemReference`
    consumes. Returning a dict rather than that class keeps the dependency one-way — this
    module knows nothing about the design-file schema, and the schema knows nothing about
    rasterio.

    Pass *info* when a :class:`DEMInfo` has already been loaded (the plugin holds one on
    its state) to avoid re-reading the header.
    """
    info = info or load_dem(dem_path)
    crs = None
    if info.crs is not None:
        try:
            crs = info.crs.to_string()
        except Exception:
            crs = info.crs_wkt
    return {
        "fingerprint": dem_content_digest(dem_path),
        "original_path": dem_path,
        "cell_size_m": info.cell_size_m,
        "crs": crs,
        "extent": list(info.bounds) if info.bounds is not None else None,
        "width": info.width,
        "height": info.height,
    }


def clip_dem_to_polygon(dem_path, clip_polygon, output_path):
    """
    Clip a DEM to a shapely polygon (in the DEM's CRS) and save to output_path.

    Parameters
    ----------
    dem_path : str
    clip_polygon : shapely geometry
    output_path : str

    Returns
    -------
    str — output_path
    """
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


# Nodata value carried by every slope raster this module writes.
_SLOPE_NODATA = -9999.0


def slope_degrees(dem, cell_w, cell_h):
    """Slope in degrees via Horn's (1981) 8-neighbour method — the GDAL/QGIS
    standard, less noisy than a 2-cell central difference.

    Shared single implementation so all callers agree numerically.

    Nodata follows the ESRI convention: a NoData *neighbour* takes the centre cell's
    own value, so the window simply reads flat in that direction. Filling nodata with
    0.0 m instead made every cell beside a hole read as a ~2000 m drop, ringing the
    whole data boundary in near-vertical slope that is indistinguishable from real
    terrain — and clipping a DEM to a property boundary creates exactly that boundary.
    A cell that is itself nodata returns NaN, as GDAL and ESRI both emit.

    Border cells use edge replication, with the divisor halved to match: replicating
    the edge row puts the "outer" sample one cell away rather than two, so dividing by
    the full 8Δ under-read every border slope by half.

    Callers that must strictly mask nodata so a NaN border cannot fabricate an
    interior gradient (e.g. :func:`contour_analysis.filter_by_slope`) keep their
    own masked-array computation on purpose — see the note there.

    Parameters
    ----------
    dem : 2-D array of elevations (m); NaN allowed for invalid cells
    cell_w, cell_h : float — cell size (m) in x and y

    Returns
    -------
    float32 array of slope in degrees, same shape as *dem*; NaN where *dem* is NaN.
    """
    z = np.asarray(dem, dtype="float64")
    invalid = ~np.isfinite(z)
    centre = np.where(invalid, 0.0, z)

    zp = np.pad(np.where(invalid, np.nan, z), 1, mode="edge")

    def _nb(block):
        """A nodata neighbour reads as the centre cell — ESRI's convention."""
        return np.where(np.isnan(block), centre, block)

    a, b, c = _nb(zp[:-2, :-2]), _nb(zp[:-2, 1:-1]), _nb(zp[:-2, 2:])
    d,    f = _nb(zp[1:-1, :-2]),                    _nb(zp[1:-1, 2:])
    g, h, i = _nb(zp[2:, :-2]),  _nb(zp[2:, 1:-1]),  _nb(zp[2:, 2:])

    # Edge replication halves the sampled separation on the first/last row and column.
    rows, cols = z.shape
    span_x = np.full(cols, 8.0 * cell_w)
    span_y = np.full(rows, 8.0 * cell_h)
    if cols > 1:
        span_x[0] = span_x[-1] = 4.0 * cell_w
    if rows > 1:
        span_y[0] = span_y[-1] = 4.0 * cell_h

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / span_x[np.newaxis, :]
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / span_y[:, np.newaxis]
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    slope[invalid] = np.nan
    return slope.astype("float32")


def compute_slope_raster(dem_path, output_path):
    """
    Compute a slope raster (degrees) from a DEM and save it.

    Uses Horn's 8-neighbour method (:func:`slope_degrees`) — the GDAL/QGIS
    standard. Sufficient for slope class display and the <18° contour filter.

    Nodata input is written back out as the declared −9999 nodata. Previously the
    raster declared −9999 and never wrote it, so a clipped DEM's masked-out region
    rendered as perfectly valid slope — flat inside, a near-vertical ring at the edge —
    with nothing to distinguish it from real ground.

    Parameters
    ----------
    dem_path : str
    output_path : str

    Returns
    -------
    str — output_path
    """

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        cell_w = abs(transform.a)
        cell_h = abs(transform.e)

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    slope_deg = slope_degrees(dem, cell_w, cell_h)
    slope_deg = np.where(np.isfinite(slope_deg), slope_deg,
                         _SLOPE_NODATA).astype("float32")

    with rasterio.open(
        output_path, "w",
        driver="GTiff", dtype="float32",
        crs=crs, transform=transform,
        width=dem.shape[1], height=dem.shape[0],
        count=1, compress="lzw",
        nodata=_SLOPE_NODATA,
    ) as dst:
        dst.write(slope_deg, 1)

    return output_path
