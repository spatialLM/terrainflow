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
import uuid

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape as shapely_shape

from terrainflow_assessment.core.registry.earthwork_types import get_type
from terrainflow_assessment.core.sizing import manning_flow, trapezoid_section
from terrainflow_assessment.modules.burn_strategy import (
    enforce_monotonic_path,
    line_cells,
    ponding_resolution_warning,
    sub_cell_warning,
)
from terrainflow_assessment.qgis.adapters.geom import shapely_area, shapely_length

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
        self.id = uuid.uuid4().hex   # stable identity (for overflow linkage; survives reorder)
        self.depth = 0.5             # metres cut/raised (not used for dam)
        self.top_width_m = 2.0       # declared top width of cross-section (metres)
        # Bottom width is the canonical stored cross-section field; side_slope is derived
        # from it (see the side_slope property). Seeded from the type's default batter so a
        # fresh feature reproduces its historical slope (channels default 1:1 → bottom = 1.0 m).
        try:
            default_slope = get_type(ew_type).default_side_slope
        except KeyError:
            default_slope = 1.0
        self.bottom_width_m = max(0.1, self.top_width_m - 2 * default_slope * self.depth)
        self.batter_run_m = 0.0      # basin only: horizontal inset to full depth (0 = vertical)
        self.companion_berm = False  # swales only
        self.crest_elevation = None  # dam only: absolute crest elevation (m)
        self.gradient_pct = 1.0      # diversion only: channel gradient (%)
        self.overflow_target_id = None  # user-intended overflow recipient (None = analytics decide)
        self.enabled = True
        self.capacity_m3 = 0.0
        self.capacity_l = 0.0

    # ------------------------------------------------------------------
    # Derived geometry fields
    # ------------------------------------------------------------------

    @property
    def side_slope(self):
        """Channel side slope as an H:V ratio, derived from the stored widths.

        side_slope = (top_width_m − bottom_width_m) / (2 × depth). 1.0 == 1:1.
        Returns 0.0 for zero depth (avoids divide-by-zero).
        """
        if self.depth <= 0:
            return 0.0
        return (self.top_width_m - self.bottom_width_m) / (2.0 * self.depth)

    @side_slope.setter
    def side_slope(self, value):
        """Set the slope by back-solving the canonical bottom_width_m at the current depth."""
        self.bottom_width_m = max(0.1, self.top_width_m - 2.0 * value * self.depth)

    @property
    def wall_slope(self):
        """Basin wall batter as an H:V ratio, derived from batter_run_m / depth.

        Returns 0.0 (vertical) for zero depth.
        """
        if self.depth <= 0:
            return 0.0
        return self.batter_run_m / self.depth

    @wall_slope.setter
    def wall_slope(self, value):
        """Set the wall batter by back-solving the canonical batter_run_m at the current depth."""
        self.batter_run_m = max(0.0, value * self.depth)

    @property
    def length_m(self):
        """Feature length in metres, derived from the geometry (single source of truth)."""
        return shapely_length(self.geometry)

    @property
    def buffer_radius_m(self):
        """Raster-burn buffer radius = top_width_m / 2."""
        return self.top_width_m / 2.0

    # backward-compat alias so existing code using ew.width keeps working
    @property
    def width(self):
        return self.top_width_m

    @width.setter
    def width(self, value):
        self.top_width_m = value

    def type_label(self):
        try:
            return get_type(self.type).label
        except KeyError:
            return self.type.capitalize()

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

def _resolve_bottom_width(bottom_width, top_width, depth):
    """Return the trapezoid bottom width, defaulting to the 1:1 derivation when unset.

    ``bottom_width is None`` reproduces the historical ``top_width − 2×depth`` (1:1 side
    slopes); an explicit value honours the feature's stored side slope. Clamped to ≥ 0.1 m.
    """
    if bottom_width is None:
        return max(0.1, top_width - 2 * depth)
    return max(0.1, bottom_width)


def calculate_capacity(ew_type, geometry, depth, width, companion_berm=False,
                       bottom_width=None):
    """
    Calculate storage capacity of an earthwork.

    Swale — trapezoidal cross-section × length × 0.8 freeboard.
    Basin — polygon area × depth × 0.8 freeboard.
    Berm / Dam / Diversion — no storage, returns (0.0, 0.0).

    ``bottom_width`` is the trapezoid's bottom width (m). When ``None`` it is derived
    from the declared top width assuming 1:1 side slopes (``top_width − 2×depth``) —
    preserving historical behaviour. Pass ``Earthwork.bottom_width_m`` to honour the
    feature's stored side slope.

    Returns (volume_m3, volume_l).
    """
    try:
        if not get_type(ew_type).has_capacity:
            return 0.0, 0.0
    except KeyError:
        return 0.0, 0.0

    if ew_type == "swale":
        length = shapely_length(geometry)
        top_width = width
        bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
        cross_section = trapezoid_section(top_width, bottom_width, depth).area

        if companion_berm and cross_section > 0:
            berm_height = (cross_section * 0.75) ** 0.5
            additional_cs = berm_height * top_width / 2
            cross_section += additional_cs

        volume_m3 = cross_section * length * 0.8

    elif ew_type == "basin":
        # TODO(feature-list): honour Earthwork.batter_run_m for sloped basin walls;
        # currently vertical (prismatic) — value change, deferred out of the shape pass.
        area_m2 = shapely_area(geometry)
        volume_m3 = area_m2 * depth * 0.8
    else:
        return 0.0, 0.0

    return round(volume_m3, 2), round(volume_m3 * 1000, 1)


def calculate_cut_volume(ew_type, geometry, depth, width, bottom_width=None):
    """
    Calculate the volume of soil excavated (cut) by an earthwork.

    Swale — trapezoidal cross-section (no freeboard) × length.
    Basin / Diversion — area × depth.
    Berm / Dam — 0 (these place material, not remove it).

    ``bottom_width`` — trapezoid bottom width (m); ``None`` derives it from 1:1 side slopes
    (historical behaviour). Applies to the swale branch. See ``calculate_capacity``.

    Returns cut volume in m³.
    """
    try:
        if not get_type(ew_type).has_cut:
            return 0.0
    except KeyError:
        return 0.0

    if ew_type == "swale":
        length = shapely_length(geometry)
        top_width = width
        bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
        cross_section = trapezoid_section(top_width, bottom_width, depth).area
        return round(cross_section * length, 2)

    if ew_type == "basin":
        area_m2 = shapely_area(geometry)
        return round(area_m2 * depth, 2)

    if ew_type == "diversion":
        # Diversion 'width' is the *bed* width here (a different convention from the swale
        # top width — reconciled in the calc pass). The section is now computed via the
        # shared trapezoid_section primitive so it can no longer diverge from the
        # capacity/discharge copies: a symmetric ±2×depth (1:1) section around the bed.
        length = shapely_length(geometry)
        bottom_width = max(0.05, width - 2 * depth)
        top_width = width + 2 * depth
        cross_section = trapezoid_section(top_width, bottom_width, depth).area
        return round(cross_section * length, 2)

    return 0.0


def calculate_fill_volume(ew_type, geometry, depth, width, companion_berm=False,
                          bottom_width=None):
    """
    Calculate the volume of material placed (fill) by an earthwork.

    Berm — triangular cross-section (1:1 slopes) × length.
    Swale + companion berm — companion berm fill (from volume conservation).
    Dam — wall footprint × depth (approximate).
    Others — 0.

    ``bottom_width`` — swale trapezoid bottom width (m) for the companion-berm branch;
    ``None`` derives it from 1:1 side slopes (historical behaviour).

    Returns fill volume in m³.
    """
    if ew_type == "berm":
        # TODO(feature-list): honour the stored side slope for a battered berm; currently
        # a fixed 1:1 triangle (base=2*depth, height=depth → area=depth²).
        length = shapely_length(geometry)
        cross_section = depth * depth
        return round(cross_section * length, 2)

    if ew_type == "swale" and companion_berm:
        length = shapely_length(geometry)
        top_width = width
        bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
        swale_cs = trapezoid_section(top_width, bottom_width, depth).area
        berm_height = (swale_cs * 0.75) ** 0.5
        berm_cs = berm_height * berm_height  # triangular
        return round(berm_cs * length, 2)

    if ew_type == "dam":
        length = shapely_length(geometry)
        return round(width * depth * length, 2) if depth and width else 0.0

    return 0.0


def berm_height_estimate(depth, width, bottom_width=None):
    """Estimate companion berm height from swale excavation (75% compaction).

    ``width`` is the declared top width of the swale (metres). ``bottom_width`` is the
    trapezoid bottom width; ``None`` derives it from 1:1 side slopes (historical behaviour).
    """
    top_width = width
    bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
    cross_section = trapezoid_section(top_width, bottom_width, depth).area
    return round((cross_section * 0.75) ** 0.5, 2)


def calculate_diversion_discharge(depth, width, gradient_pct, bottom_width=None):
    """
    Peak discharge capacity of a diversion drain (Manning's equation).

    Trapezoidal cross-section, n=0.025 (compacted earthen). Returns Q in m³/s.

    ``bottom_width`` — trapezoid bottom width (m):
      * ``None`` (default): legacy behaviour — ``width`` is the *bed* width, section is
        symmetric (bottom = width−2×depth, top = width+2×depth) with 1:1 side slopes.
        TODO(feature-list): this bed-width convention differs from the swale top-width
        convention; reconcile in the calc pass.
      * explicit value: ``width`` is the *top* width and ``bottom_width`` the bottom, so the
        side slope (and hence the slant term ``sqrt(1+s²)×depth``) follow the stored geometry.
    """
    import math
    n = 0.025
    s = gradient_pct / 100.0
    if s <= 0 or depth <= 0 or width <= 0:
        return 0.0
    if bottom_width is None:
        # Legacy bed-width convention — preserved byte-for-byte. Symmetric ±2×depth
        # section around the bed, with the historical 1:1 slant term (geometrically
        # inconsistent with the ±2×depth widths; the fix is deferred with the
        # width-convention reconciliation). Area comes from the shared primitive; the
        # wetted perimeter keeps the legacy √2 slant so the number is unchanged.
        bottom = max(0.05, width - 2 * depth)
        top = width + 2 * depth
        area = trapezoid_section(top, bottom, depth).area
        # bottom ≥ 0.05 and depth > 0 (guarded above) → wetted perimeter is always > 0.
        wetted_perimeter = bottom + 2 * (math.sqrt(2) * depth)
        r = area / wetted_perimeter
    else:
        # Stored-geometry path — width is the top width; the section (and hence the
        # wetted perimeter / hydraulic radius) follow the stored widths exactly.
        sec = trapezoid_section(width, max(0.05, bottom_width), depth)
        area = sec.area
        r = sec.hydraulic_radius
    q = manning_flow(area, r, s, n).discharge
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
        # Non-fatal advisories raised during the last burn / ponding pass (sub-cell
        # features, resolution-cap degrade). The controller surfaces these to the
        # QGIS message bar — Strategy C is honest about what it approximates.
        self.warnings = []

    def burn_earthworks(self, earthworks):
        """
        Apply all enabled earthworks to a copy of the original DEM.
        Returns modified DEM as float32 numpy array.
        """
        modified = self.original.copy()
        self.warnings = []
        _dispatch = {
            "swale":     self._burn_swale,
            "berm":      self._burn_berm,
            "basin":     self._burn_basin,
            "dam":       self._burn_dam,
            "diversion": self._burn_diversion,
        }
        for ew in earthworks:
            if not ew.enabled:
                continue
            shapely_geom = self._to_shapely(ew.geometry)
            if shapely_geom is None:
                continue
            burn_fn = _dispatch.get(ew.type)
            if burn_fn is None:
                continue
            modified = burn_fn(modified, shapely_geom, ew)
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

    def _line_path_cells(self, line):
        """Connected in-bounds cell path along *line* (nearest-cell snap fallback)."""
        try:
            coords = list(line.coords)
        except (NotImplementedError, AttributeError):
            return []
        return line_cells(coords, self.transform, self.shape)

    def _warn_sub_cell(self, name, min_dimension):
        """Record a sub-cell advisory if *min_dimension* is narrower than one cell."""
        w = sub_cell_warning(name, min_dimension, self.cell_size)
        if w:
            self.warnings.append(w)

    def _downstream_footprint(self, line, thickness):
        """Wall footprint offset to the downstream (lower) side of *line* (§7).

        The drawn dam line is the inner (wet-side) wall; all wall thickness is added
        downstream. Offsets the line by ``thickness/2`` to each side, keeps the side
        whose ground is lower, and buffers it into a wall band. Falls back to a
        centred buffer if the offset can't be built (short/degenerate lines).
        """
        half = thickness / 2.0
        try:
            left = line.parallel_offset(half, "left")
            right = line.parallel_offset(half, "right")
            left_mask = self._rasterize(left.buffer(half))
            right_mask = self._rasterize(right.buffer(half))
            left_mean = float(np.mean(self.original[left_mask])) if left_mask.any() else np.inf
            right_mean = (
                float(np.mean(self.original[right_mask])) if right_mask.any() else np.inf
            )
            if np.isinf(left_mean) and np.isinf(right_mean):
                return line.buffer(half)
            chosen = left if left_mean <= right_mean else right
            return chosen.buffer(half)
        except Exception:
            return line.buffer(half)

    # ---------------------------------------------------------------- earthwork types

    def _burn_swale(self, dem, line, ew):
        # Strategy C: incise the true footprint where the swale is cell-resolvable;
        # fall back to the nearest-cell path when the buffer rasterises empty (the
        # sub-cell no-op fix), then breach a monotonic downhill invert so the drain
        # stays connected through depression-filling.
        footprint = line.buffer(ew.buffer_radius_m)  # radius = top_width_m / 2
        mask = self._rasterize(footprint)
        path_cells = self._line_path_cells(line)
        dem = dem.copy()
        if mask.any():
            dem[mask] -= ew.depth
        else:
            for rc in path_cells:
                dem[rc] -= ew.depth
        dem = enforce_monotonic_path(dem, path_cells)
        self._warn_sub_cell(ew.name, ew.bottom_width_m)
        if ew.companion_berm:
            dem = self._add_companion_berm(dem, line, mask, ew)
        return dem

    def _add_companion_berm(self, dem, line, swale_mask, ew):
        berm_width = ew.top_width_m
        try:
            left_line = line.parallel_offset(ew.buffer_radius_m + berm_width / 2, "left")
            right_line = line.parallel_offset(ew.buffer_radius_m + berm_width / 2, "right")
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
        # Barrier: raise a flow-blocking ridge (never a cut). Incise-free — the
        # footprint band where resolvable, the nearest-cell path when sub-cell.
        footprint = line.buffer(ew.width / 2)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        if mask.any():
            dem[mask] += ew.depth
        else:
            for rc in self._line_path_cells(line):
                dem[rc] += ew.depth
        self._warn_sub_cell(ew.name, ew.width)
        return dem

    def _burn_basin(self, dem, polygon, ew):
        mask = self._rasterize(polygon)
        dem = dem.copy()
        dem[mask] -= ew.depth
        return dem

    def _burn_dam(self, dem, line, ew):
        if ew.crest_elevation is None:
            return self._burn_berm(dem, line, ew)
        # Inner-wall convention (§7): wall thickness sits downstream of the drawn
        # line; raise the wall band (or the nearest-cell path) up to the crest.
        footprint = self._downstream_footprint(line, ew.width)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        if mask.any():
            dem[mask] = np.maximum(dem[mask], ew.crest_elevation)
        else:
            for rc in self._line_path_cells(line):
                dem[rc] = max(dem[rc], ew.crest_elevation)
        self._warn_sub_cell(ew.name, ew.width)
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
                burn_elev = target_floor - ew.depth
                if cell_mask.any():
                    dem[cell_mask] = np.minimum(dem[cell_mask], burn_elev)
                else:
                    # Sub-cell channel: the buffer rasterised empty. Snap to the
                    # nearest cell so the graded invert still carves ≥ 1 cell — but
                    # only when the sample lies within the DEM (an off-extent point
                    # stays a no-op, never a spurious edge-cell burn).
                    col = int((x - self.transform.c) / self.transform.a)
                    row = int((y - self.transform.f) / self.transform.e)
                    if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                        dem[row, col] = min(float(dem[row, col]), burn_elev)

        # Bed (bottom) width of the trapezoidal channel drives the sub-cell check.
        self._warn_sub_cell(ew.name, max(0.05, ew.width - 2 * ew.depth))
        return dem

    def get_ponding_layer(self, modified_dem):
        """
        Calculate ponding depth where water pools in the modified DEM.

        Compares modified DEM against a depression-filled version.
        Returns float32 array of ponding depth in metres (0 = no ponding).
        Auto-downsamples large DEMs to stay within memory limits.
        """
        import os
        import tempfile

        from pysheds.grid import Grid
        from rasterio.transform import Affine

        rows, cols = modified_dem.shape
        n_cells = rows * cols
        scale = 1.0

        # Resolution-aware cap: the cell cap is a memory guard, not a resolution
        # choice. When it trips we coarsen a *copy* to stay within memory, but the
        # DEM is never upsampled below native res (deferred Strategy B) — surface the
        # degrade as a warning rather than letting it happen silently (§3).
        cap_warning = ponding_resolution_warning(n_cells, _MAX_PONDING_CELLS)
        if cap_warning:
            self.warnings.append(cap_warning)

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
