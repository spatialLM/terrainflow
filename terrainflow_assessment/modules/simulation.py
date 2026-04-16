"""
simulation.py — Time-stepped fill simulation with cascading overflow.

Adapts the base plugin's SimulationWorker and adds:
  - Per-earthwork fill tracking (EarthworkStore)
  - Cascading overflow: when an earthwork overflows, surplus is routed to
    the next downslope earthwork (or exits the site)
  - Infiltration losses per timestep (USCS-derived rate × earthwork area)
  - Output: timestep table, earthwork summary, total site outflow

Key exports
-----------
EarthworkStore        — tracks fill state for one earthwork
SimulationWorker      — QThread worker
run_simulation()      — standalone function (for testing / non-GUI use)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import rasterio

from qgis.PyQt.QtCore import QThread, pyqtSignal

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Earthwork fill state
# ---------------------------------------------------------------------------

@dataclass
class EarthworkStore:
    """Tracks the fill state of one earthwork over the simulation."""
    name: str
    ew_type: str
    capacity_m3: float
    area_m2: float              # footprint area for infiltration
    infiltration_rate_mm_hr: float = 4.0  # default Loam
    elevation: float = 0.0     # approximate centroid elevation (for cascade ordering)

    # Per-timestep accumulators (set externally each step)
    inflow_m3: float = 0.0
    stored_m3: float = 0.0
    overflowed: bool = False
    first_overflow_hr: Optional[float] = None
    total_overflow_m3: float = 0.0
    total_inflow_m3: float = 0.0
    total_infiltration_m3: float = 0.0
    peak_fill_pct: float = 0.0

    # Cut/fill volumes (populated from earthwork geometry, set before simulation)
    cut_vol_m3: float = 0.0
    fill_vol_m3: float = 0.0

    # Raster coordinates of the earthwork centroid (for inflow sampling)
    centroid_row: Optional[int] = None
    centroid_col: Optional[int] = None

    def step_infiltration(self, dt_hr):
        """Infiltration loss this timestep (m³)."""
        rate_m_hr = self.infiltration_rate_mm_hr / 1000.0
        return rate_m_hr * self.area_m2 * dt_hr


# ---------------------------------------------------------------------------
# Cascading overflow routing
# ---------------------------------------------------------------------------

def _find_downslope_store(store, all_stores):
    """
    Return the nearest lower-elevation EarthworkStore to route overflow into.

    Simple heuristic: find the store with the highest elevation that is still
    below this store (i.e. directly downslope).  Returns None if no lower
    store exists (overflow exits the site).
    """
    candidates = [
        s for s in all_stores
        if s is not store and s.elevation < store.elevation
    ]
    if not candidates:
        return None
    # Nearest in elevation = most direct downslope receiver
    return max(candidates, key=lambda s: s.elevation)


def cascade_overflow(stores: List[EarthworkStore], time_hr: float,
                     dt_hr: float) -> float:
    """
    Process one simulation timestep for all earthwork stores.

    Processes stores from highest to lowest elevation:
    1. Add inflow and subtract infiltration losses.
    2. If stored > capacity: overflow to next downslope store or exit.
    3. Update peak fill % and first overflow time.

    Parameters
    ----------
    stores : list of EarthworkStore, processed highest elevation first
    time_hr : float — current simulation time (hours from start)
    dt_hr   : float — timestep duration (hours)

    Returns
    -------
    float — total volume that exited the site this timestep (m³)
    """
    sorted_stores = sorted(stores, key=lambda s: s.elevation, reverse=True)
    site_exit_m3 = 0.0

    for store in sorted_stores:
        infiltration = store.step_infiltration(dt_hr)
        infiltration = min(infiltration, store.stored_m3 + store.inflow_m3)

        store.stored_m3 += store.inflow_m3 - infiltration
        store.stored_m3 = max(0.0, store.stored_m3)
        store.total_inflow_m3 += store.inflow_m3
        store.total_infiltration_m3 += infiltration
        store.inflow_m3 = 0.0  # reset for next step

        # Check overflow
        if store.stored_m3 > store.capacity_m3:
            overflow = store.stored_m3 - store.capacity_m3
            store.stored_m3 = store.capacity_m3
            store.total_overflow_m3 += overflow
            if not store.overflowed:
                store.overflowed = True
                store.first_overflow_hr = time_hr

            # Route overflow to next downslope store or exit
            downstream = _find_downslope_store(store, stores)
            if downstream is not None:
                downstream.inflow_m3 += overflow
            else:
                site_exit_m3 += overflow

        fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
        if fill_pct > store.peak_fill_pct:
            store.peak_fill_pct = fill_pct

    return site_exit_m3


# ---------------------------------------------------------------------------
# QThread worker
# ---------------------------------------------------------------------------

class SimulationWorker(QThread):
    """
    Background worker for time-stepped fill simulation.

    Extends the base plugin's SimulationWorker with per-earthwork fill
    tracking and cascading overflow logic.

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
        Pre-configured fill stores for cascade simulation.
        If None, only the raster-based flow simulation is run.
    soil_name : str — default soil type for infiltration rate lookup

    Signals
    -------
    progress(int, str)
    finished(dict)
    error(str)
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
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

    def run(self):
        import traceback
        try:
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
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


# ---------------------------------------------------------------------------
# Core simulation logic
# ---------------------------------------------------------------------------

def _run_simulation(dem_path, fdir_path, output_dir, cn, moisture,
                    rainfall_data, routing='dinf', cn_zones_data=None,
                    earthwork_stores=None, progress_callback=None):
    """
    Run the time-stepped simulation and return results.

    Parameters and return value match SimulationWorker.finished signal.
    Can also be called directly (without QThread) for testing.
    """
    from pysheds.grid import Grid
    from .catchment import SCSRunoff

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    earthwork_stores = earthwork_stores or []
    scs = SCSRunoff()

    _p(2, "Loading DEM for simulation...")
    with rasterio.open(dem_path) as src:
        cell_w = abs(src.transform.a)
        cell_h = abs(src.transform.e)
        cell_area_m2 = cell_w * cell_h
        shape = (src.height, src.width)
        out_meta = src.meta.copy()
        out_meta.update(dtype="float32", count=1, nodata=-9999.0)

    _p(5, "Loading flow direction...")
    grid = Grid.from_raster(dem_path)
    fdir = grid.read_raster(fdir_path)

    _p(8, "Building CN raster...")
    if cn_zones_data:
        from shapely.wkt import loads as wkt_loads
        with rasterio.open(dem_path) as src:
            transform = src.transform
        zone_geoms_cn = []
        for z in cn_zones_data:
            try:
                zone_geoms_cn.append((wkt_loads(z["wkt"]), z["cn"]))
            except Exception:
                pass
        cn_array = scs.build_cn_raster(
            shape, transform, zone_geoms_cn, cn, moisture
        )
    else:
        eff_cn = scs.adjust_cn(cn, moisture)
        cn_array = np.full(shape, eff_cn, dtype="float32")

    sim_dir = os.path.join(output_dir, "simulation")
    os.makedirs(sim_dir, exist_ok=True)
    for f in os.listdir(sim_dir):
        try:
            os.remove(os.path.join(sim_dir, f))
        except OSError:
            pass

    n_steps = len(rainfall_data) - 1
    if n_steps < 1:
        raise ValueError("rainfall_data must have at least 2 entries.")

    frames = []
    cum_acc = np.zeros(shape, dtype="float64")
    prev_q_mm = np.zeros(shape, dtype="float64")

    # Timestep table: one row per timestep
    timestep_table = []
    total_site_outflow_m3 = 0.0
    peak_outflow_ls = 0.0
    peak_outflow_time_hr = 0.0

    # Track outflow at the main boundary exit (approximated from raster)
    # We'll estimate it from the incremental accumulation at the highest-acc
    # boundary cell if no earthwork stores are provided.

    for i in range(n_steps):
        time_min = rainfall_data[i + 1][0]
        time_hr = time_min / 60.0
        p_cum = float(rainfall_data[i + 1][1])

        pct = int(10 + 80 * i / n_steps)
        _p(pct, f"Timestep {i + 1}/{n_steps} — T = {time_min} min...")

        # SCS incremental runoff this step
        q_cum_now = np.vectorize(
            lambda c: scs.runoff_depth(p_cum, float(c)) if c > 0 else 0.0
        )(cn_array).astype("float64")
        dq_mm = np.maximum(0.0, q_cum_now - prev_q_mm)
        prev_q_mm = q_cum_now

        dq_m3 = (dq_mm / 1000.0) * cell_area_m2

        try:
            from pysheds.sview import Raster as _PR
            dq_raster = _PR(dq_m3.astype("float64"), viewfinder=fdir.viewfinder)
        except Exception:
            dq_raster = dq_m3.astype("float64")

        try:
            inc_acc = grid.accumulation(fdir, weights=dq_raster, routing=routing)
        except TypeError:
            inc_acc = grid.accumulation(fdir, weights=dq_raster)
        inc_acc = np.array(inc_acc, dtype="float32")

        cum_acc += inc_acc.astype("float64")
        cum_arr = cum_acc.astype("float32")

        # Estimate site-level outflow this step from raster max boundary cell
        # (rough proxy — detailed per-earthwork cascade handles the rest)
        inc_max = float(inc_acc.max())
        dt_hr = (time_hr - (rainfall_data[i][0] / 60.0)) if i > 0 else time_hr
        dt_s = max(dt_hr * 3600.0, 1.0)
        outflow_ls_raster = inc_max * 1000.0 / dt_s  # L/s proxy

        # Per-earthwork cascade
        # Inject catchment runoff into stores based on their catchment fraction
        total_runoff_m3_step = float(np.sum(dq_m3))

        if earthwork_stores:
            # Sample the weighted accumulation raster at each earthwork's
            # centroid to get the actual m³ of runoff flowing to that location.
            # inc_acc is already in m³ (weighted by dq_m3 per cell), so the
            # value at an earthwork's position is the total upstream inflow.
            for store in earthwork_stores:
                r, c = store.centroid_row, store.centroid_col
                if r is not None and c is not None:
                    r = max(0, min(r, inc_acc.shape[0] - 1))
                    c = max(0, min(c, inc_acc.shape[1] - 1))
                    inflow_m3 = float(inc_acc[r, c])
                    store.inflow_m3 += max(0.0, inflow_m3)
                # else: no raster position — store gets no inflow (centroid outside DEM)

            step_exit = cascade_overflow(
                stores=earthwork_stores,
                time_hr=time_hr,
                dt_hr=dt_hr,
            )
            total_site_outflow_m3 += step_exit
            outflow_ls = step_exit * 1000.0 / dt_s
        else:
            outflow_ls = outflow_ls_raster
            total_site_outflow_m3 += inc_max  # approximate

        if outflow_ls > peak_outflow_ls:
            peak_outflow_ls = outflow_ls
            peak_outflow_time_hr = time_hr

        # Baseline (no-storage) outflow: all runoff exits immediately this step
        outflow_ls_baseline = total_runoff_m3_step * 1000.0 / dt_s if dt_s > 0 else 0.0

        # Build timestep row
        row = {
            "time_min": time_min,
            "time_hr": round(time_hr, 2),
            "rainfall_cum_mm": round(p_cum, 1),
            "runoff_m3": round(total_runoff_m3_step, 1),
            "outflow_ls": round(outflow_ls, 1),
            "outflow_ls_baseline": round(outflow_ls_baseline, 1),
        }
        for store in earthwork_stores:
            fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
            row[f"{store.name}_fill_pct"] = round(fill_pct, 1)
            row[f"{store.name}_overflow"] = store.overflowed
        timestep_table.append(row)

        # Save rasters
        inc_path = os.path.join(sim_dir, f"inc_{i:03d}.tif")
        cum_path = os.path.join(sim_dir, f"cum_{i:03d}.tif")
        with rasterio.open(inc_path, "w", **out_meta) as dst:
            data = inc_acc.copy()
            data[data < 0] = out_meta["nodata"]
            dst.write(data, 1)
        with rasterio.open(cum_path, "w", **out_meta) as dst:
            data = cum_arr.copy()
            data[data < 0] = out_meta["nodata"]
            dst.write(data, 1)

        # Snapshot fill state for each store this frame
        frame_fills = {}
        for store in earthwork_stores:
            fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
            frame_fills[store.name] = {
                "fill_pct": round(min(fill_pct, 100.0), 1),
                "overflowed": store.overflowed,
                "first_overflow_this_step": (
                    store.overflowed and
                    store.first_overflow_hr is not None and
                    abs(store.first_overflow_hr - time_hr) < (dt_hr + 1e-6)
                ),
            }
        frames.append({"time_min": time_min, "inc": inc_path, "cum": cum_path,
                        "fills": frame_fills})

    # Peak flow raster
    _p(96, "Computing peak flow raster...")
    peak_arr = np.zeros(shape, dtype="float32")
    for frame in frames:
        with rasterio.open(frame["inc"]) as src:
            data = src.read(1).astype("float32")
            data[data == -9999.0] = 0.0
            peak_arr = np.maximum(peak_arr, data)

    peak_path = os.path.join(sim_dir, "peak_flow.tif")
    with rasterio.open(peak_path, "w", **out_meta) as dst:
        peak_arr[peak_arr < 0] = out_meta["nodata"]
        dst.write(peak_arr, 1)

    # Build earthwork summary
    earthwork_summary = []
    for store in earthwork_stores:
        fill_pct = (store.stored_m3 / store.capacity_m3 * 100.0) if store.capacity_m3 > 0 else 0.0
        earthwork_summary.append({
            "name": store.name,
            "type": store.ew_type,
            "capacity_m3": round(store.capacity_m3, 1),
            "stored_m3": round(store.stored_m3, 1),
            "peak_fill_pct": round(store.peak_fill_pct, 1),
            "final_fill_pct": round(fill_pct, 1),
            "overflowed": store.overflowed,
            "first_overflow_hr": round(store.first_overflow_hr, 2) if store.first_overflow_hr else None,
            "total_overflow_m3": round(store.total_overflow_m3, 1),
            "total_inflow_m3": round(store.total_inflow_m3, 1),
            "total_infiltration_m3": round(store.total_infiltration_m3, 1),
            "cut_vol_m3": round(store.cut_vol_m3, 1),
            "fill_vol_m3": round(store.fill_vol_m3, 1),
        })

    # Time labels for UI slider
    time_labels = [f"{row['time_min']} min" for row in timestep_table]

    _p(100, "Simulation complete.")
    return {
        "frames": frames,
        "peak_path": peak_path,
        "timestep_table": timestep_table,
        "time_labels": time_labels,
        "earthwork_summary": earthwork_summary,
        "total_outflow_m3": round(total_site_outflow_m3, 1),
        "peak_outflow_ls": round(peak_outflow_ls, 1),
        "peak_outflow_time_hr": round(peak_outflow_time_hr, 2),
    }


# ---------------------------------------------------------------------------
# Helper: build EarthworkStore list from plugin earthworks
# ---------------------------------------------------------------------------

def build_stores_from_earthworks(earthworks, soil_name="Loam", dem_path=None):
    """
    Build a list of EarthworkStore objects from the plugin's Earthwork list.

    Parameters
    ----------
    earthworks : list of Earthwork (from earthwork_design module)
    soil_name : str — global soil type for infiltration rates
    dem_path : str or None — used to look up centroid elevation

    Returns
    -------
    list of EarthworkStore
    """
    import json
    from shapely.geometry import shape as shapely_shape
    from .swale_design import get_infiltration_rate
    from .earthwork_design import calculate_cut_volume, calculate_fill_volume

    infil_rate = get_infiltration_rate(soil_name)
    stores = []

    for ew in earthworks:
        if not ew.enabled or ew.capacity_m3 <= 0:
            continue

        # Get footprint area from shapely geometry
        try:
            shapely_geom = shapely_shape(json.loads(ew.geometry.asJson()))
            if hasattr(shapely_geom, "area"):
                area_m2 = shapely_geom.area
            else:
                # LineString: approximate as length × width
                area_m2 = shapely_geom.length * ew.width
        except Exception:
            area_m2 = 100.0  # fallback

        # Centroid elevation + raster coordinates from DEM
        elevation = 0.0
        centroid_row = None
        centroid_col = None
        if dem_path:
            try:
                centroid = shapely_geom.centroid
                import rasterio
                with rasterio.open(dem_path) as src:
                    t = src.transform
                    col = int((centroid.x - t.c) / t.a)
                    row = int((centroid.y - t.f) / t.e)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        elevation = float(src.read(1)[row, col])
                        centroid_row = row
                        centroid_col = col
            except Exception:
                pass

        cut_vol = calculate_cut_volume(ew.type, ew.geometry, ew.depth, ew.width)
        fill_vol = calculate_fill_volume(ew.type, ew.geometry, ew.depth, ew.width,
                                         ew.companion_berm)

        store = EarthworkStore(
            name=ew.name,
            ew_type=ew.type,
            capacity_m3=ew.capacity_m3,
            area_m2=area_m2,
            infiltration_rate_mm_hr=infil_rate,
            elevation=elevation,
            cut_vol_m3=cut_vol,
            fill_vol_m3=fill_vol,
            centroid_row=centroid_row,
            centroid_col=centroid_col,
        )
        stores.append(store)

    return stores
