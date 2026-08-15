"""
analysis_worker.py — QThread wrapper for background flow analysis.

Extracted from modules/flow_analysis.py so the pure FlowAnalysis logic
can be imported and tested without a live QGIS runtime.
"""

import numpy as np
import rasterio
from qgis.PyQt.QtCore import QThread, pyqtSignal

from terrainflow_assessment.qgis.workers._lifecycle import AbortMixin, WorkerAborted


class AnalysisWorker(AbortMixin, QThread):
    """
    Background worker that runs FlowAnalysis and emits result paths.

    Signals
    -------
    progress(int pct, str message)
    completed(dict result_paths)
    error(str traceback)

    ``completed`` rather than ``finished``: ``QThread`` already defines a
    ``finished()`` signal, emitted by Qt when ``run()`` returns, and declaring one
    of our own shadowed it. That made the standard ``finished -> deleteLater``
    teardown unreachable — the only thing a connection could ever see was our own
    dict, emitted from inside ``run()`` while the thread is still very much alive.
    """
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dem_path, output_dir, stream_threshold, cn, moisture,
                 rainfall_mm, duration_hours, boundary_path=None, label="",
                 run_catchments=False, threshold_mode="cells", volume_threshold=50.0,
                 routing='dinf', cn_zones_data=None, exit_flow_ls=0.5,
                 analysis_area_path=None, earthworks_area_path=None,
                 sizing_basis="coefficient", runoff_coefficient=0.5):
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
        self.exit_flow_ls = exit_flow_ls  # min L/s for a boundary exit to show
        self.analysis_area_path = analysis_area_path
        self.earthworks_area_path = earthworks_area_path
        # 'rainfall' — every mm that falls is routed; 'runoff' — only the SCS-CN
        # surface runoff is. Set once on the Baseline stage so the analysis rasters
        # and earthwork sizing describe the same storm.
        self.sizing_basis = sizing_basis
        self.runoff_coefficient = runoff_coefficient

    def _stage(self, pct, message):
        """Report progress, and stop here if cancellation has been requested.

        Every stage boundary already emits progress, so making that the abort
        checkpoint costs nothing and puts one wherever the work can be interrupted
        without leaving a half-written raster behind.
        """
        self.raise_if_aborted()
        self.progress.emit(pct, message)

    def run(self):
        import traceback
        try:
            self._do_analysis()
        except WorkerAborted:
            return          # asked for, so not an error and not worth a banner
        except Exception:
            self.error.emit(traceback.format_exc())

    def _do_analysis(self):
        import os

        from terrainflow_assessment.modules.catchment import SCSRunoff
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

        self._stage(5, "Loading DEM...")
        fa = FlowAnalysis()
        fa.load_dem(self.dem_path)

        with rasterio.open(self.dem_path) as src:
            cell_w = abs(src.transform.a)
            cell_h = abs(src.transform.e)
            cell_area_m2 = cell_w * cell_h
            shape = (src.height, src.width)
            transform = src.transform

        scs = SCSRunoff()

        self._stage(10, "Building runoff model...")
        runoff_weights = None
        if self.sizing_basis in ("rainfall", "coefficient"):
            # A uniform depth across the site, so CN zoning has nothing to vary —
            # the curve number is still reported for context.
            from terrainflow_assessment.modules.catchment import coefficient_runoff_depth
            eff_cn = scs.adjust_cn(self.cn, self.moisture)
            runoff_mm = (
                float(self.rainfall_mm) if self.sizing_basis == "rainfall"
                else coefficient_runoff_depth(self.rainfall_mm, self.runoff_coefficient)
            )
        elif self.cn_zones_data:
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

        self._stage(20, "Running flow analysis...")
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

        self._stage(50, "Saving rasters...")
        os.makedirs(self.output_dir, exist_ok=True)

        acc_path = os.path.join(self.output_dir, f"flow_accumulation_{self.label}.tif")
        fdir_path = os.path.join(self.output_dir, f"flow_direction_{self.label}.tif")
        stream_path = os.path.join(self.output_dir, f"streams_{self.label}.tif")
        runoff_path = os.path.join(self.output_dir, f"runoff_volume_{self.label}.tif")
        cond_path = os.path.join(self.output_dir, f"conditioned_dem_{self.label}.tif")
        domain_path = os.path.join(self.output_dir, f"domain_{self.label}.tif")

        # Every raster declares its own nodata. An untagged GeoTIFF is read back by
        # pysheds as nodata=0, which for the D-infinity direction grid means "due
        # east" — so each east-flowing cell would return as a routing self-loop and
        # the simulation would quietly disagree with the flow map it was built from.
        fa.save_result(acc_array, acc_path, "flow accumulation (cell count)",
                       nodata=np.nan)
        fa.save_result(fdir_array, fdir_path, fa.get_fdir_description(),
                       nodata=fa.get_fdir_nodata())
        fa.save_result(stream_acc, stream_path, "stream accumulation", nodata=np.nan)

        runoff_vol = fa.get_runoff_volume_raster(runoff_mm, cell_area_m2)
        fa.save_result(runoff_vol, runoff_path, "runoff volume (m³)", nodata=np.nan)

        # Standing water, valued by what the pond passes. Written because the accumulation
        # beside it stops meaning contributing area inside a pool — a contracted pond holds
        # its inflow rather than threading a channel through itself — so a reader that wants
        # catchment size there has to be given it rather than left to infer it.
        pond_path = None
        pond_flow = result.get("pond_flow")
        if pond_flow is not None and np.any(pond_flow):
            pond_path = os.path.join(self.output_dir, f"ponds_{self.label}.tif")
            fa.save_result(pond_flow, pond_path, "pond throughflow (cell count)",
                           nodata=np.nan)

        # The conditioned surface drives the design-tier catchment labelling
        # (flow_graph.d8_from_dem). Saved rather than kept in memory because the
        # FlowAnalysis instance dies with this thread.
        #
        # float64, unlike every other raster here: this one carries resolve_flats' synthetic
        # flat gradient, which is integer multiples of 1e-5 m. float32's spacing passes that
        # around 600 m elevation, so on a hill country site the gradient would be rounded
        # out of existence on the way to disk and every flat cell would come back a sink.
        #
        # It inherits the source DEM's sentinel, not NaN: this is the raster
        # `d8_from_dem` conditions its routing on, and it has to be able to tell an
        # interior hole from ground. Untagged, the hole reads as an elevation of
        # -9999 — a pit ten kilometres deep that captures the catchment around it.
        conditioned = np.array(result.get("conditioned_dem", fa.dem), dtype="float64")
        fa.save_result(conditioned, cond_path, "hydrologically conditioned DEM",
                       dtype="float64", nodata=fa.nodata)

        # The site itself — the capture-% denominator. Most specific area wins.
        # No nodata: this is a mask, and its 0 means "outside the site", not "unknown".
        domain = self._build_domain_mask(fa, shape, transform)
        fa.save_result(domain.astype("float32"), domain_path, "analysis domain (1 = site)")
        domain_cells = int(domain.sum())
        domain_area_m2 = domain_cells * cell_area_m2

        # "Total water through this cell" for the throughflow gradient. The weighted
        # accumulation is truthful under spatially varying CN; the uniform volume
        # raster is exact when there are no CN zones and identical in that case.
        if runoff_weights is not None and "runoff_accumulation" in result:
            throughflow_path = os.path.join(
                self.output_dir, f"throughflow_{self.label}.tif")
            fa.save_result(np.array(result["runoff_accumulation"]), throughflow_path,
                           "event throughflow (m³ per cell, CN-weighted)",
                           nodata=np.nan)
        else:
            throughflow_path = runoff_path

        self._stage(65, "Detecting exit points...")
        # Prefer the runoff-weighted accumulation (m³ per cell) for a truthful
        # per-cell flow, falling back to the uniform-storm volume raster.
        if runoff_weights is not None and "runoff_accumulation" in result:
            exit_vol = np.array(result["runoff_accumulation"])
        else:
            exit_vol = runoff_vol

        exit_points = []
        if self.boundary_path:
            try:
                exit_points = fa.get_boundary_exit_points(
                    self.boundary_path, self.exit_flow_ls, runoff_mm,
                    self.duration_hours, volume_raster=exit_vol,
                )
            except Exception:
                pass

        # Per-area outflow, reported as two distinct figures because they answer two
        # different questions. ``volume_m3``/``flow_ls`` sum only the crossings drawn on
        # the map, so they move when the user changes "Show exits above (L/s)";
        # ``total_volume_m3``/``total_flow_ls`` measure the whole flux across the polygon
        # and are independent of that threshold. Conflating them made the readout look
        # like a site total when it was only ever the sum of the visible markers.
        area_outflow = {}
        for key, path in (("site", self.boundary_path),
                          ("analysis", self.analysis_area_path),
                          ("earthworks", self.earthworks_area_path)):
            if not path:
                continue
            try:
                pts = fa.get_boundary_exit_points(
                    path, self.exit_flow_ls, runoff_mm, self.duration_hours,
                    volume_raster=exit_vol,
                )
                entry = {
                    "volume_m3": round(sum(p["volume_m3"] for p in pts), 1),
                    "flow_ls": round(sum(p["flow_ls"] for p in pts), 2),
                    "n_exits": len(pts),
                }
            except Exception:
                continue
            try:
                total = fa.boundary_outflow_total(
                    path, runoff_mm, self.duration_hours, volume_raster=exit_vol,
                )
                entry["total_volume_m3"] = total["volume_m3"]
                entry["total_flow_ls"] = total["flow_ls"]
            except Exception:
                pass
            area_outflow[key] = entry

        self._stage(75, "Delineating catchments...")
        catchments = []
        if self.run_catchments:
            try:
                catchments = fa.get_catchment_polygons(
                    stream_threshold=self.stream_threshold
                )
            except Exception:
                pass

        self._stage(90, "Computing ponding...")
        ponding_path = None
        ponded_volume_m3 = None
        try:
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            from terrainflow_assessment.modules.reporting import raster_ponding_volume
            burner = DEMBurner(self.dem_path)
            ponding = burner.get_ponding_layer(burner.original)
            ponding_path = os.path.join(self.output_dir, f"ponding_{self.label}.tif")
            burner.save(ponding, ponding_path)
            # Clipped to the domain so the figure describes the site, not whatever else
            # the DEM happens to cover. get_ponding_layer resamples back to native shape,
            # so cell_area_m2 is the right multiplier even when it downsampled to compute.
            ponded_volume_m3 = round(
                raster_ponding_volume(np.where(domain, ponding, 0.0), cell_area_m2), 1)
        except Exception:
            pass

        # The area draining to the single busiest cell — one outlet's catchment. Kept
        # under its own name because it is a genuine statistic, but it is NOT the site
        # area and must never again be used as the capture-% denominator.
        max_upstream_area_m2 = float(acc_array.max()) * cell_area_m2

        # Water the routing could not place anywhere. Measured against the domain rather
        # than the whole tile, because a nodata margin outside the site is not a loss.
        from terrainflow_assessment.modules.flow_analysis import (
            crest_spread_warning,
            unrouted_flow_warning,
        )
        unrouted_cells = int(result.get("unrouted_cells", 0))
        unrouted_flow = float(result.get("unrouted_flow", 0.0))
        unrouted_warning = unrouted_flow_warning(
            unrouted_cells, unrouted_flow, domain_cells)

        # A pond spills along its whole level crest at once, so each pond is contracted to a
        # mixing node and sheds its inflow evenly over the cells that discharge from it.
        # Anything it could not pass on is reported here rather than left to read as a leak.
        crest_unplaced = float(result.get("crest_residual", 0.0))
        crest_warning = crest_spread_warning(crest_unplaced, domain_cells)

        self._stage(100, "Analysis complete.")
        self.completed.emit({
            "label": self.label,
            "flow_accumulation": acc_path,
            "flow_direction": fdir_path,
            "conditioned_dem": cond_path,
            "domain_mask": domain_path,
            "stream_network": stream_path,
            "stream_acc_max": stream_acc_max,
            "stream_threshold": self.stream_threshold,
            "runoff_volume": runoff_path,
            "throughflow": throughflow_path,
            "ponding": ponding_path,
            "effective_cn": eff_cn,
            "runoff_mm": runoff_mm,
            "sizing_basis": self.sizing_basis,
            "runoff_coefficient": self.runoff_coefficient,
            "rainfall_mm": self.rainfall_mm,
            "cell_area_m2": cell_area_m2,
            "domain_cells": domain_cells,
            "domain_area_m2": domain_area_m2,
            "catchment_area_m2": domain_area_m2,
            "max_upstream_area_m2": max_upstream_area_m2,
            "unrouted_cells": unrouted_cells,
            "unrouted_flow_m3": unrouted_flow * cell_area_m2 * runoff_mm / 1000.0,
            "unrouted_warning": unrouted_warning,
            "crest_ponds": int(result.get("crest_ponds", 0)),
            "crest_cells": int(result.get("crest_cells", 0)),
            "crest_passes": int(result.get("crest_passes", 0)),
            "pond_flow": pond_path,
            "crest_unplaced_m3": crest_unplaced * cell_area_m2 * runoff_mm / 1000.0,
            "crest_warning": crest_warning,
            "runoff_volume_m3": (runoff_mm / 1000.0) * domain_area_m2,
            "exit_points": exit_points,
            "area_outflow": area_outflow,
            "ponded_volume_m3": ponded_volume_m3,
            "catchments": catchments,
        })

    def _build_domain_mask(self, fa, shape, transform):
        """Boolean mask of the site, most-specific area first.

        Analysis area → site boundary → every usable DEM cell. Read here (file I/O)
        and decided in ``modules.footprint.domain_mask`` (pure, tested).
        """
        from terrainflow_assessment.modules.footprint import domain_mask

        groups = []
        for path in (self.analysis_area_path, self.boundary_path):
            if not path:
                continue
            try:
                import geopandas as gpd
                gdf = gpd.read_file(path).to_crs(fa.crs.to_wkt())
                groups.append([g for g in gdf.geometry if g is not None])
            except Exception:
                continue

        try:
            dem_arr = np.array(fa.dem, dtype="float64")
            valid = np.isfinite(dem_arr)
            if fa.nodata is not None:
                valid &= dem_arr != fa.nodata
        except Exception:
            valid = None

        return domain_mask(shape, transform, polygons=groups, valid=valid)

    def _flow_dir_description(self):
        if self.routing == 'dinf':
            return "D-infinity flow direction (angle in radians, CCW from east)"
        return "D8 flow direction (ESRI codes)"
