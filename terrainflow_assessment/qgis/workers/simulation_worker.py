"""
simulation_worker.py — QThread wrapper for background fill simulation.

Extracted from modules/simulation.py so the pure simulation logic
(_run_simulation, EarthworkStore, cascade_overflow, etc.) can be imported
and tested without a live QGIS runtime.
"""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal

from terrainflow_assessment.qgis.workers._lifecycle import AbortMixin, WorkerAborted


class SimulationWorker(AbortMixin, QThread):
    """
    Background worker for time-stepped fill simulation.

    Parameters
    ----------
    dem_path : str
    fdir_path : str — pre-computed flow direction raster
    output_dir : str
    cn : int — default Curve Number
    moisture : str — 'dry', 'normal', or 'wet'
    rainfall_data : list of (time_min, cum_rainfall_mm)
    routing : str — 'dinf' or 'd8'
    cn_zones_data : list of dict (optional)
    earthwork_stores : list of EarthworkStore (optional)
    soil_name : str — default soil type for infiltration rate lookup

    Signals
    -------
    progress(int, str)
    completed(dict)
    error(str)

    ``completed`` rather than ``finished`` — see AnalysisWorker: QThread already
    has a ``finished()`` signal and ours was shadowing it.
    """
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dem_path, fdir_path, output_dir, cn, moisture,
                 rainfall_data, routing='dinf', cn_zones_data=None,
                 earthwork_stores=None, soil_name="Loam"):
        super().__init__()
        self.dem_path = dem_path
        self.fdir_path = fdir_path
        self.output_dir = output_dir
        self.cn = cn
        self.moisture = moisture
        self.rainfall_data = rainfall_data
        self.routing = routing
        self.cn_zones_data = cn_zones_data or []
        self.earthwork_stores = earthwork_stores or []
        self.soil_name = soil_name

    def _stage(self, pct, message):
        """Progress callback that doubles as the abort checkpoint.

        `_run_simulation` reports once per timestep, so cancellation lands within
        one step rather than at the end of the event.
        """
        self.raise_if_aborted()
        self.progress.emit(pct, message)

    def run(self):
        try:
            from terrainflow_assessment.modules.simulation import _run_simulation
            result = _run_simulation(
                dem_path=self.dem_path,
                fdir_path=self.fdir_path,
                output_dir=self.output_dir,
                cn=self.cn,
                moisture=self.moisture,
                rainfall_data=self.rainfall_data,
                routing=self.routing,
                cn_zones_data=self.cn_zones_data,
                earthwork_stores=self.earthwork_stores,
                progress_callback=self._stage,
            )
            self.completed.emit(result)
        except WorkerAborted:
            return
        except Exception:
            self.error.emit(traceback.format_exc())
