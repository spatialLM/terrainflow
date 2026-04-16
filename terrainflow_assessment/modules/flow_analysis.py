"""
flow_analysis.py — Pysheds-based flow analysis for TerrainFlow Assessment.

Adapts the base plugin's FlowAnalysis class with one addition:
  exit point dicts now also include ``volume_m3`` (total runoff volume that
  passed through the exit over the analysis period) rather than flow velocity.

Key classes / functions
-----------------------
FlowAnalysis          — flow direction, accumulation, stream network, catchments
AnalysisWorker        — QThread wrapper for background analysis
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pysheds.grid import Grid

from qgis.PyQt.QtCore import QThread, pyqtSignal


# ---------------------------------------------------------------------------
# Core flow analysis
# ---------------------------------------------------------------------------

class FlowAnalysis:
    """
    Pysheds-based flow direction, accumulation, and catchment delineation.

    Usage::

        fa = FlowAnalysis()
        fa.load_dem(dem_path)
        result = fa.run(routing='dinf')
        exit_pts = fa.get_boundary_exit_points(boundary_path, threshold, runoff_mm, hours)
    """

    def __init__(self):
        self.grid = None
        self.dem = None
        self.fdir = None
        self.acc = None
        self.crs = None
        self.transform = None
        self.nodata = None
        self.routing = 'dinf'

    def load_dem(self, dem_path):
        """Load DEM from GeoTIFF and initialise pysheds grid."""
        self.grid = Grid.from_raster(dem_path)
        self.dem = self.grid.read_raster(dem_path)
        with rasterio.open(dem_path) as src:
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata
        return True

    def run(self, routing='dinf', runoff_weights=None):
        """
        Run the full flow analysis pipeline:
          1. Fill pits
          2. Breach depressions
          3. Resolve flats
          4. Compute flow direction (D-infinity or D8)
          5. Compute flow accumulation (optionally weighted by per-cell runoff)

        Parameters
        ----------
        routing : str — 'dinf' or 'd8'
        runoff_weights : numpy array or None
            Per-cell runoff volume (m³) for weighted accumulation.

        Returns dict with 'flow_direction', 'flow_accumulation',
        and optionally 'runoff_accumulation'.
        """
        if self.grid is None:
            raise RuntimeError("DEM not loaded. Call load_dem() first.")

        self.routing = routing

        pit_filled = self.grid.fill_pits(self.dem)
        try:
            breached = self.grid.breach_depressions(pit_filled)
        except AttributeError:
            breached = self.grid.fill_depressions(pit_filled)

        inflated = self.grid.resolve_flats(breached)

        try:
            self.fdir = self.grid.flowdir(inflated, routing=routing)
        except TypeError:
            self.routing = 'd8'
            self.fdir = self.grid.flowdir(inflated)

        try:
            self.acc = self.grid.accumulation(self.fdir, routing=self.routing)
        except TypeError:
            self.acc = self.grid.accumulation(self.fdir)

        result = {
            "flow_direction": self.fdir,
            "flow_accumulation": self.acc,
        }

        if runoff_weights is not None:
            try:
                from pysheds.sview import Raster as _PR
                weights_raster = _PR(
                    runoff_weights.astype("float64"),
                    viewfinder=self.fdir.viewfinder,
                )
            except Exception:
                weights_raster = runoff_weights
            try:
                weighted = self.grid.accumulation(
                    self.fdir, weights=weights_raster, routing=self.routing
                )
            except TypeError:
                weighted = self.grid.accumulation(self.fdir, weights=weights_raster)
            result["runoff_accumulation"] = weighted

        return result

    def delineate_catchment(self, x, y):
        """Return boolean mask for the catchment draining to (x, y)."""
        if self.fdir is None:
            raise RuntimeError("Run flow analysis first.")
        try:
            catch = self.grid.catchment(
                x=x, y=y, fdir=self.fdir, xytype="coordinate", routing=self.routing
            )
        except TypeError:
            catch = self.grid.catchment(x=x, y=y, fdir=self.fdir, xytype="coordinate")
        return catch

    def get_stream_network(self, accumulation_threshold=1000):
        """Boolean mask where accumulation > threshold."""
        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")
        return self.acc > accumulation_threshold

    def get_runoff_volume_raster(self, runoff_mm, cell_area_m2):
        """
        Runoff volume raster (m³): each cell = upstream_cells × runoff_m × cell_area.
        """
        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")
        runoff_m = runoff_mm / 1000.0
        return np.array(self.acc, dtype="float32") * runoff_m * cell_area_m2

    def get_boundary_exit_points(self, boundary_path, accumulation_threshold,
                                 runoff_mm, duration_hours):
        """
        Find significant flow exit points on the site boundary.

        Each exit point dict contains:
          x, y, accumulation, volume_m3, flow_ls, label

        ``volume_m3`` is the total runoff volume that passed through the exit
        over the analysis period (acc_cells × cell_area × runoff_m).

        Parameters
        ----------
        boundary_path : str
        accumulation_threshold : int
        runoff_mm : float
        duration_hours : float

        Returns list of dicts.
        """
        import geopandas as gpd
        from rasterio.features import rasterize

        if self.acc is None:
            raise RuntimeError("Run flow analysis first.")

        gdf = gpd.read_file(boundary_path)
        gdf = gdf.to_crs(self.crs.to_wkt())

        boundary_lines = [
            geom.exterior for geom in gdf.geometry
            if geom is not None and hasattr(geom, "exterior")
        ]
        if not boundary_lines:
            return []

        boundary_mask = rasterize(
            [(line, 1) for line in boundary_lines],
            out_shape=self.grid.shape,
            transform=self.transform,
            fill=0, dtype="uint8",
        ).astype(bool)

        acc_array = np.array(self.acc)
        exit_mask = boundary_mask & (acc_array > accumulation_threshold)
        rows, cols = np.where(exit_mask)
        if len(rows) == 0:
            return []

        cell_w = abs(self.transform.a)
        cell_h = abs(self.transform.e)
        xs = self.transform.c + cols * self.transform.a + cell_w / 2
        ys = self.transform.f + rows * self.transform.e + cell_h / 2
        accs = acc_array[rows, cols]

        # Cluster — keep highest-acc cell per 20-cell radius
        min_dist = cell_w * 20
        candidates = sorted(zip(accs, xs, ys), reverse=True)
        kept = []
        for acc, x, y in candidates:
            too_close = any(
                ((x - kx) ** 2 + (y - ky) ** 2) < min_dist ** 2
                for _, kx, ky in kept
            )
            if not too_close:
                kept.append((acc, x, y))

        cell_area_m2 = cell_w * cell_h
        runoff_m = runoff_mm / 1000.0
        duration_s = duration_hours * 3600.0

        results = []
        for i, (acc, x, y) in enumerate(kept):
            volume_m3 = acc * cell_area_m2 * runoff_m
            flow_ls = (volume_m3 * 1000.0 / duration_s) if duration_s > 0 else 0
            results.append({
                "x": float(x),
                "y": float(y),
                "accumulation": int(acc),
                "volume_m3": round(volume_m3, 1),
                "flow_ls": round(flow_ls, 1),
                "label": f"Exit {i + 1}",
            })

        results.sort(key=lambda r: r["volume_m3"], reverse=True)
        for i, r in enumerate(results):
            r["label"] = (
                f"Exit {i + 1}: {r['volume_m3']:,.0f} m³ over event "
                f"| avg {r['flow_ls']:,.1f} L/s"
            )

        return results

    def get_catchment_polygons(self, outlet_points=None, stream_threshold=500):
        """
        Delineate non-overlapping sub-catchment polygons.

        Parameters
        ----------
        outlet_points : list of (x, y) or None (auto-detected from grid edge)
        stream_threshold : int

        Returns list of dicts: {id, geometry, area_m2, area_ha, flow_bearing, label}
        """
        from rasterio.features import shapes as rasterio_shapes
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union

        if self.fdir is None or self.acc is None:
            raise RuntimeError("Run flow analysis first.")

        cell_w = abs(self.transform.a)
        cell_h = abs(self.transform.e)
        acc_array = np.array(self.acc)

        if not outlet_points:
            outlet_points = self._find_boundary_outlets(acc_array, stream_threshold)
            if not outlet_points:
                max_idx = np.unravel_index(np.argmax(acc_array), acc_array.shape)
                row, col = max_idx
                x = self.transform.c + col * self.transform.a + cell_w / 2
                y = self.transform.f + row * self.transform.e + cell_h / 2
                outlet_points = [(x, y)]

        def _acc_at(x, y):
            col = int((x - self.transform.c) / self.transform.a)
            row = int((y - self.transform.f) / self.transform.e)
            row = max(0, min(acc_array.shape[0] - 1, row))
            col = max(0, min(acc_array.shape[1] - 1, col))
            return int(acc_array[row, col])

        outlet_points_sorted = sorted(outlet_points, key=lambda pt: _acc_at(*pt))
        fdir_array = np.array(self.fdir)
        claimed = np.zeros(acc_array.shape, dtype=bool)

        results = []
        for i, (x, y) in enumerate(outlet_points_sorted):
            try:
                try:
                    catch_mask = self.grid.catchment(
                        x=x, y=y, fdir=self.fdir, xytype="coordinate",
                        routing=self.routing
                    )
                except TypeError:
                    catch_mask = self.grid.catchment(
                        x=x, y=y, fdir=self.fdir, xytype="coordinate"
                    )
            except Exception:
                continue

            catch_bool = np.array(catch_mask).astype(bool)
            local_catch = catch_bool & ~claimed
            claimed |= catch_bool

            if not local_catch.any():
                continue

            local_uint8 = np.where(local_catch, 1, 0).astype("uint8")
            polys = [
                shapely_shape(geom)
                for geom, val in rasterio_shapes(local_uint8, transform=self.transform)
                if val == 1
            ]
            if not polys:
                continue

            catchment_poly = unary_union(polys)
            area_m2 = catchment_poly.area
            area_ha = area_m2 / 10000

            dominant_bearing = 0.0
            try:
                if self.routing == 'dinf':
                    safe_fdir = np.nan_to_num(fdir_array, nan=0.0)
                    bearings = (90.0 - np.degrees(safe_fdir)) % 360.0
                else:
                    d8_map = {64: 0, 128: 45, 1: 90, 2: 135,
                              4: 180, 8: 225, 16: 270, 32: 315}
                    bearings = np.vectorize(
                        lambda v: d8_map.get(int(v), 0)
                    )(fdir_array)
                weights = acc_array * local_catch
                total_w = weights.sum()
                if total_w > 0:
                    sin_sum = np.sum(np.sin(np.radians(bearings)) * weights)
                    cos_sum = np.sum(np.cos(np.radians(bearings)) * weights)
                    dominant_bearing = float(np.degrees(np.arctan2(sin_sum, cos_sum)) % 360)
            except Exception:
                pass

            results.append({
                "id": i + 1,
                "geometry": catchment_poly,
                "area_m2": round(area_m2, 1),
                "area_ha": round(area_ha, 2),
                "flow_bearing": round(dominant_bearing, 1),
                "label": f"Catchment {i + 1}: {area_ha:.1f} ha",
                "_outlet_acc": _acc_at(x, y),
            })

        results.sort(key=lambda r: r["_outlet_acc"], reverse=True)
        for idx, r in enumerate(results):
            r["id"] = idx + 1
            r["label"] = f"Catchment {idx + 1}: {r['area_ha']:.1f} ha ({r['area_m2']:,.0f} m²)"

        return results

    def _find_boundary_outlets(self, acc_array, min_acc):
        """Find significant outlet points on grid boundary edge."""
        rows, cols = acc_array.shape
        cell_w = abs(self.transform.a)

        boundary_cells = set()
        for c in range(cols):
            boundary_cells.add((0, c))
            boundary_cells.add((rows - 1, c))
        for r in range(rows):
            boundary_cells.add((r, 0))
            boundary_cells.add((r, cols - 1))

        candidates = [
            (int(acc_array[r, c]), r, c)
            for r, c in boundary_cells if acc_array[r, c] >= min_acc
        ]
        candidates.sort(reverse=True)

        min_dist = cell_w * 20
        kept = []
        for acc, r, c in candidates:
            x = self.transform.c + c * self.transform.a + cell_w / 2
            y = self.transform.f + r * self.transform.e + abs(self.transform.e) / 2
            too_close = any(
                (x - kx) ** 2 + (y - ky) ** 2 < min_dist ** 2
                for _, kx, ky in kept
            )
            if not too_close:
                kept.append((acc, x, y))

        return [(x, y) for _, x, y in kept]

    def get_fdir_description(self):
        """Return band description for the flow direction raster."""
        if self.routing == 'dinf':
            return "D-infinity flow direction (angle in radians, CCW from east)"
        return "D8 flow direction (ESRI codes: 1=E 2=SE 4=S 8=SW 16=W 32=NW 64=N 128=NE)"

    def get_profile(self):
        """rasterio write profile for result GeoTIFFs."""
        return {
            "driver": "GTiff", "dtype": "float32",
            "crs": self.crs, "transform": self.transform,
            "width": self.grid.shape[1], "height": self.grid.shape[0],
            "count": 1, "compress": "lzw",
        }

    def save_result(self, array, output_path, band_description=""):
        """Save a result array to GeoTIFF."""
        profile = self.get_profile()
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype("float32"), 1)
            if band_description:
                dst.update_tags(1, description=band_description)


# ---------------------------------------------------------------------------
# QThread worker
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    """
    Background worker that runs FlowAnalysis and emits result paths.

    Signals
    -------
    progress(int pct, str message)
    finished(dict result_paths)
    error(str traceback)
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dem_path, output_dir, stream_threshold, cn, moisture,
                 rainfall_mm, duration_hours, boundary_path=None, label="",
                 run_catchments=False, threshold_mode="cells", volume_threshold=50.0,
                 routing='dinf', cn_zones_data=None):
        super().__init__()
        self.dem_path = dem_path
        self.output_dir = output_dir
        self.stream_threshold = stream_threshold
        self.cn = cn
        self.moisture = moisture
        self.rainfall_mm = rainfall_mm
        self.duration_hours = duration_hours
        self.boundary_path = boundary_path
        self.label = label
        self.run_catchments = run_catchments
        self.threshold_mode = threshold_mode
        self.volume_threshold = volume_threshold
        self.routing = routing
        self.cn_zones_data = cn_zones_data or []

    def run(self):
        import traceback, os
        try:
            self._do_analysis()
        except Exception:
            self.error.emit(traceback.format_exc())

    def _do_analysis(self):
        import os
        from .catchment import SCSRunoff

        self.progress.emit(5, "Loading DEM...")
        fa = FlowAnalysis()
        fa.load_dem(self.dem_path)

        with rasterio.open(self.dem_path) as src:
            cell_w = abs(src.transform.a)
            cell_h = abs(src.transform.e)
            cell_area_m2 = cell_w * cell_h
            shape = (src.height, src.width)
            transform = src.transform

        scs = SCSRunoff()

        self.progress.emit(10, "Building runoff model...")
        runoff_weights = None
        if self.cn_zones_data:
            from shapely.wkt import loads as wkt_loads
            zone_geoms_cn = []
            for z in self.cn_zones_data:
                try:
                    zone_geoms_cn.append((wkt_loads(z["wkt"]), z["cn"]))
                except Exception:
                    pass
            cn_raster = scs.build_cn_raster(
                shape, transform, zone_geoms_cn, self.cn, self.moisture
            )
            runoff_raster = scs.build_runoff_raster(cn_raster, self.rainfall_mm)
            runoff_weights = (runoff_raster / 1000.0) * cell_area_m2
            eff_cn = float(np.nanmean(cn_raster))
            runoff_mm = float(np.nanmean(runoff_raster))
        else:
            eff_cn = scs.adjust_cn(self.cn, self.moisture)
            runoff_mm = scs.runoff_depth(self.rainfall_mm, eff_cn)

        self.progress.emit(20, "Running flow analysis...")
        result = fa.run(routing=self.routing, runoff_weights=runoff_weights)

        acc_array = np.array(fa.acc)
        fdir_array = np.array(fa.fdir)

        # Stream threshold
        if self.threshold_mode == "volume":
            if runoff_weights is not None and "runoff_accumulation" in result:
                stream_mask = np.array(result["runoff_accumulation"]) > self.volume_threshold
            else:
                vol_per_cell = (runoff_mm / 1000.0) * cell_area_m2
                stream_mask = acc_array * vol_per_cell > self.volume_threshold
        else:
            stream_mask = acc_array > self.stream_threshold

        # Build stream accumulation raster (acc value at stream cells, 0 elsewhere)
        stream_acc = np.where(stream_mask, acc_array, 0).astype("float32")
        stream_acc_max = float(stream_acc.max()) if stream_acc.max() > 0 else 1.0

        self.progress.emit(50, "Saving rasters...")
        os.makedirs(self.output_dir, exist_ok=True)

        acc_path = os.path.join(self.output_dir, f"flow_accumulation_{self.label}.tif")
        fdir_path = os.path.join(self.output_dir, f"flow_direction_{self.label}.tif")
        stream_path = os.path.join(self.output_dir, f"streams_{self.label}.tif")
        runoff_path = os.path.join(self.output_dir, f"runoff_volume_{self.label}.tif")

        fa.save_result(acc_array, acc_path, "flow accumulation (cell count)")
        fa.save_result(fdir_array, fdir_path, fa.get_fdir_description())
        fa.save_result(stream_acc, stream_path, "stream accumulation")

        runoff_vol = fa.get_runoff_volume_raster(runoff_mm, cell_area_m2)
        fa.save_result(runoff_vol, runoff_path, "runoff volume (m³)")

        self.progress.emit(65, "Detecting exit points...")
        exit_points = []
        if self.boundary_path:
            try:
                exit_points = fa.get_boundary_exit_points(
                    self.boundary_path, self.stream_threshold, runoff_mm, self.duration_hours
                )
            except Exception:
                pass

        self.progress.emit(75, "Delineating catchments...")
        catchments = []
        if self.run_catchments:
            try:
                catchments = fa.get_catchment_polygons(
                    stream_threshold=self.stream_threshold
                )
            except Exception:
                pass

        self.progress.emit(90, "Computing ponding...")
        # Ponding is computed by DEMBurner on the modified DEM (earthworks run)
        # For baseline, we detect natural depressions
        ponding_path = None
        try:
            from .earthwork_design import DEMBurner
            burner = DEMBurner(self.dem_path)
            ponding = burner.get_ponding_layer(burner.original)
            ponding_path = os.path.join(self.output_dir, f"ponding_{self.label}.tif")
            burner.save(ponding, ponding_path)
        except Exception:
            pass

        catchment_area_m2 = float(acc_array.max()) * cell_area_m2

        self.progress.emit(100, "Analysis complete.")
        self.finished.emit({
            "label": self.label,
            "flow_accumulation": acc_path,
            "flow_direction": fdir_path,
            "stream_network": stream_path,
            "stream_acc_max": stream_acc_max,
            "stream_threshold": self.stream_threshold,
            "runoff_volume": runoff_path,
            "ponding": ponding_path,
            "effective_cn": eff_cn,
            "runoff_mm": runoff_mm,
            "cell_area_m2": cell_area_m2,
            "catchment_area_m2": catchment_area_m2,
            "runoff_volume_m3": (runoff_mm / 1000.0) * catchment_area_m2,
            "exit_points": exit_points,
            "catchments": catchments,
        })

    def _flow_dir_description(self):
        if self.routing == 'dinf':
            return "D-infinity flow direction (angle in radians, CCW from east)"
        return "D8 flow direction (ESRI codes)"
