"""
earthwork_design.py — Earthwork data model, DEM burning, and capacity calculations.

Combines the base plugin's earthwork.py and dem_burner.py into a single module.

Exports
-------
Earthwork            — data class for a single earthwork feature
EarthworkManager     — manages the list of earthworks
DEMBurner            — burns earthworks into DEM, computes ponding
calculate_capacity   — storage capacity from geometry + dimensions
calculate_cut_volume — excavation volume (for reporting)
calculate_fill_volume— material placed (for reporting)
calculate_diversion_discharge — Manning's discharge for diversion drains
calculate_spillway_width      — broad-crested weir sizing
berm_height_estimate          — companion berm height from swale volume
"""

import json
import logging

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape as shapely_shape

_log = logging.getLogger(__name__)

_MAX_PONDING_CELLS = 4_000_000  # ~2000 × 2000


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Earthwork:
    """Represents a single earthwork feature."""

    TYPE_SWALE = "swale"
    TYPE_BERM = "berm"
    TYPE_BASIN = "basin"
    TYPE_DAM = "dam"
    TYPE_DIVERSION = "diversion"

    def __init__(self, ew_type, geometry, name):
        self.type = ew_type          # 'swale' | 'berm' | 'basin' | 'dam' | 'diversion'
        self.geometry = geometry     # QgsGeometry
        self.name = name
        self.depth = 0.5             # metres cut/raised (not used for dam)
        self.width = 2.0             # metres cross-section
        self.companion_berm = False  # swales only
        self.crest_elevation = None  # dam only: absolute crest elevation (m)
        self.gradient_pct = 1.0      # diversion only: channel gradient (%)
        self.spillway_point = None   # QgsPointXY or None
        self.spillway_elevation = None
        self.enabled = True
        self.capacity_m3 = 0.0
        self.capacity_l = 0.0

    def type_label(self):
        labels = {"diversion": "Diversion Drain"}
        return labels.get(self.type, self.type.capitalize())

    def summary(self):
        status = "" if self.enabled else " [OFF]"
        if self.type == "dam":
            elev_str = f"{self.crest_elevation:.1f} m" if self.crest_elevation is not None else "?"
            return f"{self.name} (Dam) — crest {elev_str}{status}"
        if self.type == "diversion":
            q = calculate_diversion_discharge(self.depth, self.width, self.gradient_pct)
            return f"{self.name} (Diversion) — {self.gradient_pct:.1f}% | Q={q:.3f} m³/s{status}"
        return f"{self.name} ({self.type_label()}) — {self.capacity_m3:.1f} m³{status}"


class EarthworkManager:
    """Manages the list of earthworks for the current session."""

    def __init__(self):
        self._earthworks = []

    def add(self, earthwork):
        self._earthworks.append(earthwork)

    def remove(self, index):
        if 0 <= index < len(self._earthworks):
            self._earthworks.pop(index)

    def toggle(self, index):
        if 0 <= index < len(self._earthworks):
            self._earthworks[index].enabled = not self._earthworks[index].enabled

    def get(self, index):
        return self._earthworks[index]

    def get_all(self):
        return list(self._earthworks)

    def get_enabled(self):
        return [e for e in self._earthworks if e.enabled]

    def clear(self):
        self._earthworks.clear()

    def __len__(self):
        return len(self._earthworks)


# ---------------------------------------------------------------------------
# Capacity and hydraulic calculations
# ---------------------------------------------------------------------------

def calculate_capacity(ew_type, geometry, depth, width, companion_berm=False):
    """
    Calculate storage capacity of an earthwork.

    Swale — trapezoidal cross-section (1:1 side slopes) × length × 0.8 freeboard.
    Basin — polygon area × depth × 0.8 freeboard.
    Berm / Dam / Diversion — no storage, returns (0.0, 0.0).

    Returns (volume_m3, volume_l).
    """
    if ew_type in ("berm", "dam", "diversion"):
        return 0.0, 0.0

    if ew_type == "swale":
        length = geometry.length()
        bottom_width = max(0.1, width - 2 * depth)
        top_width = width + 2 * depth
        cross_section = ((bottom_width + top_width) / 2) * depth

        if companion_berm and cross_section > 0:
            berm_height = (cross_section * 0.75) ** 0.5
            additional_cs = berm_height * top_width / 2
            cross_section += additional_cs

        volume_m3 = cross_section * length * 0.8

    elif ew_type == "basin":
        area_m2 = geometry.area()
        volume_m3 = area_m2 * depth * 0.8
    else:
        return 0.0, 0.0

    return round(volume_m3, 2), round(volume_m3 * 1000, 1)


def calculate_cut_volume(ew_type, geometry, depth, width):
    """
    Calculate the volume of soil excavated (cut) by an earthwork.

    Swale — trapezoidal cross-section (1:1 slopes, no freeboard) × length.
    Basin / Diversion — area × depth.
    Berm / Dam — 0 (these place material, not remove it).

    Returns cut volume in m³.
    """
    if ew_type in ("berm", "dam"):
        return 0.0

    if ew_type == "swale":
        length = geometry.length()
        bottom_width = max(0.1, width - 2 * depth)
        top_width = width + 2 * depth
        cross_section = ((bottom_width + top_width) / 2) * depth
        return round(cross_section * length, 2)

    if ew_type == "basin":
        area_m2 = geometry.area()
        return round(area_m2 * depth, 2)

    if ew_type == "diversion":
        length = geometry.length()
        bottom_width = max(0.05, width - 2 * depth)
        top_width = width + 2 * depth
        cross_section = ((bottom_width + top_width) / 2) * depth
        return round(cross_section * length, 2)

    return 0.0


def calculate_fill_volume(ew_type, geometry, depth, width, companion_berm=False):
    """
    Calculate the volume of material placed (fill) by an earthwork.

    Berm — triangular cross-section (1:1 slopes) × length.
    Swale + companion berm — companion berm fill (from volume conservation).
    Dam — wall footprint × depth (approximate).
    Others — 0.

    Returns fill volume in m³.
    """
    if ew_type == "berm":
        length = geometry.length()
        cross_section = depth * depth  # triangular: base=2*depth, height=depth → area=depth²
        return round(cross_section * length, 2)

    if ew_type == "swale" and companion_berm:
        length = geometry.length()
        bottom_width = max(0.1, width - 2 * depth)
        top_width = width + 2 * depth
        swale_cs = ((bottom_width + top_width) / 2) * depth
        berm_height = (swale_cs * 0.75) ** 0.5
        berm_cs = berm_height * berm_height  # triangular
        return round(berm_cs * length, 2)

    if ew_type == "dam":
        length = geometry.length()
        return round(width * depth * length, 2) if depth and width else 0.0

    return 0.0


def berm_height_estimate(depth, width):
    """Estimate companion berm height from swale excavation (75% compaction)."""
    bottom_width = max(0.1, width - 2 * depth)
    top_width = width + 2 * depth
    cross_section = ((bottom_width + top_width) / 2) * depth
    return round((cross_section * 0.75) ** 0.5, 2)


def calculate_diversion_discharge(depth, width, gradient_pct):
    """
    Peak discharge capacity of a diversion drain (Manning's equation).

    Trapezoidal cross-section, 1:1 side slopes, n=0.025 (compacted earthen).
    Returns Q in m³/s.
    """
    import math
    n = 0.025
    s = gradient_pct / 100.0
    if s <= 0 or depth <= 0 or width <= 0:
        return 0.0
    bottom_width = max(0.05, width - 2 * depth)
    top_width = width + 2 * depth
    area = ((bottom_width + top_width) / 2) * depth
    slant_side = math.sqrt(2) * depth  # 1:1 slope
    wetted_perimeter = bottom_width + 2 * slant_side
    if wetted_perimeter <= 0:
        return 0.0
    r = area / wetted_perimeter
    q = (1.0 / n) * area * (r ** (2.0 / 3.0)) * (s ** 0.5)
    return round(q, 4)


def calculate_spillway_width(peak_flow_m3s, head_m, weir_coeff=1.7):
    """
    Minimum spillway width — broad-crested weir formula.

    L = Q / (C × H^1.5)

    Returns spillway width in metres.
    """
    if head_m <= 0 or peak_flow_m3s <= 0:
        return 0.0
    return round(peak_flow_m3s / (weir_coeff * head_m ** 1.5), 2)


# ---------------------------------------------------------------------------
# DEM burning
# ---------------------------------------------------------------------------

class DEMBurner:
    """
    Burns earthwork geometries into a DEM copy and computes ponding.

    Each earthwork type modifies the DEM differently:
      Swale      → lower cells within the swale footprint
      Berm       → raise cells within the berm footprint
      Basin      → lower cells within the basin polygon
      Dam        → set cells to crest elevation
      Diversion  → graded channel (grade-controlled depth)
    """

    def __init__(self, dem_path):
        with rasterio.open(dem_path) as src:
            self.original = src.read(1).astype("float32")
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.shape = self.original.shape
        self.cell_size = abs(self.transform.a)

    def burn_earthworks(self, earthworks):
        """
        Apply all enabled earthworks to a copy of the original DEM.
        Returns modified DEM as float32 numpy array.
        """
        modified = self.original.copy()
        for ew in earthworks:
            if not ew.enabled:
                continue
            shapely_geom = self._to_shapely(ew.geometry)
            if shapely_geom is None:
                continue
            if ew.type == "swale":
                modified = self._burn_swale(modified, shapely_geom, ew)
            elif ew.type == "berm":
                modified = self._burn_berm(modified, shapely_geom, ew)
            elif ew.type == "basin":
                modified = self._burn_basin(modified, shapely_geom, ew)
            elif ew.type == "dam":
                modified = self._burn_dam(modified, shapely_geom, ew)
            elif ew.type == "diversion":
                modified = self._burn_diversion(modified, shapely_geom, ew)
        return modified

    def save(self, array, output_path):
        """Save a DEM array to GeoTIFF with LZW compression."""
        with rasterio.open(
            output_path, "w",
            driver="GTiff", dtype="float32",
            crs=self.crs, transform=self.transform,
            width=self.shape[1], height=self.shape[0],
            count=1, nodata=self.nodata, compress="lzw",
        ) as dst:
            dst.write(array, 1)

    # ------------------------------------------------------------------ helpers

    def _to_shapely(self, qgs_geometry):
        try:
            return shapely_shape(json.loads(qgs_geometry.asJson()))
        except Exception:
            return None

    def _rasterize(self, shapely_geom):
        return rasterize(
            [(shapely_geom, 1)],
            out_shape=self.shape,
            transform=self.transform,
            fill=0, dtype="uint8",
        ).astype(bool)

    # ---------------------------------------------------------------- earthwork types

    def _burn_swale(self, dem, line, ew):
        footprint = line.buffer(ew.width / 2)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        dem[mask] -= ew.depth
        if ew.companion_berm:
            dem = self._add_companion_berm(dem, line, mask, ew)
        return dem

    def _add_companion_berm(self, dem, line, swale_mask, ew):
        berm_width = ew.width
        try:
            left_line = line.parallel_offset(ew.width / 2 + berm_width / 2, "left")
            right_line = line.parallel_offset(ew.width / 2 + berm_width / 2, "right")
            left_zone = left_line.buffer(berm_width / 2)
            right_zone = right_line.buffer(berm_width / 2)
        except Exception:
            return dem

        left_mask = self._rasterize(left_zone)
        right_mask = self._rasterize(right_zone)

        if left_mask.any() and right_mask.any():
            left_mean = float(np.mean(self.original[left_mask]))
            right_mean = float(np.mean(self.original[right_mask]))
            berm_mask = left_mask if left_mean < right_mean else right_mask
        elif left_mask.any():
            berm_mask = left_mask
        elif right_mask.any():
            berm_mask = right_mask
        else:
            return dem

        n_swale = int(np.sum(swale_mask))
        n_berm = int(np.sum(berm_mask))
        raise_height = (n_swale / n_berm) * ew.depth if n_berm > 0 else ew.depth

        dem = dem.copy()
        dem[berm_mask] += raise_height
        return dem

    def _burn_berm(self, dem, line, ew):
        footprint = line.buffer(ew.width / 2)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        dem[mask] += ew.depth
        return dem

    def _burn_basin(self, dem, polygon, ew):
        mask = self._rasterize(polygon)
        dem = dem.copy()
        dem[mask] -= ew.depth
        return dem

    def _burn_dam(self, dem, line, ew):
        if ew.crest_elevation is None:
            return self._burn_berm(dem, line, ew)
        footprint = line.buffer(ew.width / 2)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        dem[mask] = np.maximum(dem[mask], ew.crest_elevation)
        return dem

    def _burn_diversion(self, dem, line, ew):
        dem = dem.copy()
        coords = list(line.coords)
        if len(coords) < 2:
            return dem

        x0, y0 = coords[0]
        col0 = int((x0 - self.transform.c) / self.transform.a)
        row0 = int((y0 - self.transform.f) / self.transform.e)
        row0 = max(0, min(self.shape[0] - 1, row0))
        col0 = max(0, min(self.shape[1] - 1, col0))
        start_elev = float(dem[row0, col0])

        cum_dist = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            cum_dist.append(cum_dist[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        total_length = cum_dist[-1]
        if total_length == 0:
            return dem

        gradient_frac = ew.gradient_pct / 100.0

        for seg_i in range(len(coords) - 1):
            x1, y1 = coords[seg_i]
            x2, y2 = coords[seg_i + 1]
            seg_dist = cum_dist[seg_i + 1] - cum_dist[seg_i]
            if seg_dist == 0:
                continue
            n_steps = max(2, int(seg_dist / self.cell_size * 3))
            for step in range(n_steps + 1):
                t = step / n_steps
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                dist_along = cum_dist[seg_i] + t * seg_dist
                target_floor = start_elev - dist_along * gradient_frac

                from shapely.geometry import Point
                pt_geom = Point(x, y).buffer(ew.width / 2)
                cell_mask = self._rasterize(pt_geom)
                if not cell_mask.any():
                    continue
                burn_elev = target_floor - ew.depth
                dem[cell_mask] = np.minimum(dem[cell_mask], burn_elev)

        return dem

    def get_ponding_layer(self, modified_dem):
        """
        Calculate ponding depth where water pools in the modified DEM.

        Compares modified DEM against a depression-filled version.
        Returns float32 array of ponding depth in metres (0 = no ponding).
        Auto-downsamples large DEMs to stay within memory limits.
        """
        from pysheds.grid import Grid
        from rasterio.transform import Affine
        import tempfile, os

        rows, cols = modified_dem.shape
        n_cells = rows * cols
        scale = 1.0

        if n_cells > _MAX_PONDING_CELLS:
            from scipy.ndimage import zoom as _zoom
            scale = (_MAX_PONDING_CELLS / n_cells) ** 0.5
            work_dem = _zoom(modified_dem, scale, order=1).astype("float32")
        else:
            work_dem = modified_dem

        scaled_transform = Affine(
            self.transform.a / scale, self.transform.b, self.transform.c,
            self.transform.d, self.transform.e / scale, self.transform.f,
        )

        tmp = tempfile.mktemp(suffix=".tif")
        try:
            with rasterio.open(
                tmp, "w", driver="GTiff", dtype="float32",
                crs=self.crs, transform=scaled_transform,
                width=work_dem.shape[1], height=work_dem.shape[0],
                count=1, nodata=self.nodata,
            ) as dst:
                dst.write(work_dem, 1)

            grid = Grid.from_raster(tmp)
            dem_raster = grid.read_raster(tmp)

            try:
                dem_raster = grid.fill_pits(dem_raster)
            except MemoryError:
                pass

            try:
                depression_filled = grid.fill_depressions(dem_raster)
            except MemoryError:
                _log.error("Ponding analysis failed (MemoryError) — returning empty layer.")
                return np.zeros_like(modified_dem)

            ponding = np.array(depression_filled, dtype="float32") - work_dem
            ponding = np.clip(ponding, 0, None)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        if scale < 1.0:
            from scipy.ndimage import zoom as _zoom
            ponding = _zoom(
                ponding,
                (rows / ponding.shape[0], cols / ponding.shape[1]),
                order=1,
            )
            ponding = np.clip(ponding.astype("float32"), 0, None)

        return ponding
