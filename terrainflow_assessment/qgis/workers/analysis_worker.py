"""
analysis_worker.py — QThread wrapper for background flow analysis.

Extracted from modules/flow_analysis.py so the pure FlowAnalysis logic
can be imported and tested without a live QGIS runtime.
"""

import numpy as np
import rasterio

from qgis.PyQt.QtCore import QThread, pyqtSignal


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
        import traceback
        try:
            self._do_analysis()
        except Exception:
            self.error.emit(traceback.format_exc())

    def _do_analysis(self):
        import os
        from terrainflow_assessment.modules.catchment import SCSRunoff
        from terrainflow_assessment.modules.flow_analysis import FlowAnalysis

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
        ponding_path = None
        try:
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
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
