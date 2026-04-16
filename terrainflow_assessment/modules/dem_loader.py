"""
dem_loader.py — DEM loading and metadata extraction.

Reads a GeoTIFF DEM and exposes cell size, CRS, extent, and a path suitable
for passing into other analysis modules.
"""

import rasterio
import numpy as np


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

    except Exception as exc:
        raise RuntimeError(f"Cannot load DEM '{dem_path}': {exc}") from exc


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


def compute_slope_raster(dem_path, output_path):
    """
    Compute a slope raster (degrees) from a DEM and save it.

    Uses a simple 3×3 Sobel-like gradient for speed.
    Sufficient for slope class display and the <18° contour filter.

    Parameters
    ----------
    dem_path : str
    output_path : str

    Returns
    -------
    str — output_path
    """
    import scipy.ndimage as ndi

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        cell_w = abs(transform.a)
        cell_h = abs(transform.e)

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # Compute gradient in x and y
    dz_dy, dz_dx = np.gradient(np.nan_to_num(dem, nan=0.0), cell_h, cell_w)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype("float32")

    with rasterio.open(
        output_path, "w",
        driver="GTiff", dtype="float32",
        crs=crs, transform=transform,
        width=dem.shape[1], height=dem.shape[0],
        count=1, compress="lzw",
        nodata=-9999.0,
    ) as dst:
        dst.write(slope_deg, 1)

    return output_path
