"""
terrain.py — TerrainController

The terrain-index layers: wetness, stream power, sediment transport, plan and profile
curvature, aspect. One button computes them, one toggle each shows them.

**Its own controller, and only new work in it.** ``contour.py`` is already 1,600 lines
carrying five features and ``earthworks.py`` is past 6,000; a new feature area with its
own buttons is what CLAUDE.md's rule about new controllers is for. The existing slope
tools stay where they are — moving them would be the drive-by the same rule forbids.

**Computed on demand, one index per toggle, never as part of the baseline run.** Each is
a full-size float32 raster (~11 MB on the fixture grid, more on a real 1 m tile) and four
of the five will never be opened in a given session. Making them a side effect of
Baseline would double its memory and its disk for output nobody asked for.
"""

from __future__ import annotations

import os

from qgis.core import QgsRasterLayer

from terrainflow_assessment.core.registry import map_palette as P
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers._layers import remove_layer, resolve_layer
from terrainflow_assessment.qgis.controllers._symbols import apply_raster_ramp
from terrainflow_assessment.qgis.workers._lifecycle import worker_is_running
from terrainflow_assessment.qgis.workers.task_worker import TaskWorker

#: key → (layer title, palette stops, whether the ramp is anchored symmetrically).
#:
#: The symmetric flag is not decoration: curvature is signed and its zero is a real
#: boundary, so anchoring on the raw min and max would let one tail's spike flatten the
#: other and the eye would read an asymmetry the terrain does not have. See the
#: signed-quantity rule in ``map_palette``.
INDEX_SPECS = {
    "twi": ("Wetness Index (TWI)", P.WETNESS_INDEX, False),
    "spi": ("Stream Power Index", P.EROSIVE_POWER, False),
    "sti": ("Sediment Transport Index", P.EROSIVE_POWER, False),
    "plan_curvature": ("Plan Curvature", P.CURVATURE, True),
    "profile_curvature": ("Profile Curvature", P.CURVATURE, True),
    "aspect": ("Aspect", P.ASPECT_CLASSES, False),
}


class TerrainController(G.LayerTreeMixin):
    def __init__(self, state, panel, project, iface, canvas=None):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    def teardown(self):
        for layer_id in list(self._state.terrain_index_layer_ids.values()):
            remove_layer(self._project, layer_id)
        self._state.terrain_index_layer_ids = {}

    # ------------------------------------------------------------------ compute

    def run_terrain_indices(self):
        """Compute every index once, off the GUI thread, and write them beside the DEM.

        Slope is taken from the **raw** DEM, not the conditioned one. ``resolve_flats``
        lifts every flat by an integer multiple of 1e-5 m so the router has somewhere to
        send water; that is a routing device, not terrain, and computing tan β on it
        would put a fabricated slope under every flat and make the wetness index look
        measured exactly where it is invented.

        Contributing area is ``flow_accumulation`` — the **cell count**. Never
        ``throughflow`` or any runoff volume: those have already had water held back
        from them by every hollow upstream, so a wetness index built on one would have a
        retained numerator over an unretained denominator.
        """
        if not self._state.dem_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM first.")
            return

        acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
        if not acc_path or not os.path.exists(acc_path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run baseline analysis first — the indices need its flow accumulation.")
            return

        if worker_is_running(self._state, "terrain_worker"):
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Terrain indices are waiting — another run is still going.")
            return

        dem_path = self._state.dem_path
        out_dir = self._state.output_dir

        def work(report):
            return _compute_indices(dem_path, acc_path, out_dir, report)

        worker = TaskWorker(work, label="terrain indices")
        worker.progress.connect(self._panel.set_terrain_progress)
        worker.completed.connect(self._on_indices_ready)

        def _failed(tb):
            self._panel.set_terrain_complete("")
            print("TerrainFlow Assessment — terrain indices failed:")
            print(tb)
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment",
                "Terrain indices failed — see the Python console.")

        worker.error.connect(_failed)
        self._state.terrain_worker = worker
        self._panel.set_terrain_progress(1, "Starting…")
        worker.start()

    def _on_indices_ready(self, result):
        """Back on the GUI thread."""
        self._state.terrain_index_paths = result["paths"]
        self._state.terrain_index_bounds = result["bounds"]
        floored = result["floored_cells"]
        total = result["domain_cells"]

        self._panel.set_terrain_complete(
            f"{len(result['paths'])} terrain indices computed.")
        self._panel.set_terrain_available(sorted(result["paths"]))

        # The floored share is reported, not swallowed. On a tile that is mostly flat —
        # a harbour plane, a terrace, a lake — the wetness index over that ground is
        # computed against an assumed minimum slope rather than a measured one, and a
        # map that does not say so is a fiction. Same discipline as unrouted_flow's
        # impossible-share report.
        if floored and total:
            share = 100.0 * floored / total
            if share >= 1.0:
                self._iface.messageBar().pushInfo(
                    "TerrainFlow Assessment",
                    f"{share:.0f}% of the site is too flat to have a wetness index — "
                    f"those cells use the {result['min_tan_beta']:.3f} m/m floor.")

    # ------------------------------------------------------------------ display

    def toggle_terrain_index(self, key, checked):
        """Show or hide one index layer, building it the first time it is asked for."""
        if key not in INDEX_SPECS:
            return

        existing = self._state.terrain_index_layer_ids.get(key)
        layer = resolve_layer(self._project, existing)
        if layer is not None:
            # ``layer_tree_root``, not ``layerTreeRoot``: ``self._project`` is the
            # ProjectAdapter, and its snake_case surface is the point of it. Reaching
            # for the Qt spelling raises AttributeError *inside a Qt slot*, which PyQt
            # turns into a process abort — no traceback, no failed check, just a dead
            # subprocess and thirty other checks never counted.
            node = self._project.layer_tree_root().findLayer(layer.id())
            if node is not None:
                node.setItemVisibilityChecked(bool(checked))
            return

        if not checked:
            return  # nothing built yet, so nothing to hide

        path = (self._state.terrain_index_paths or {}).get(key)
        if not path or not os.path.exists(path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Compute the terrain indices first.")
            return

        title, stops, symmetric = INDEX_SPECS[key]
        layer = QgsRasterLayer(path, title)
        if not layer.isValid():
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not load {title}.")
            return

        self._apply_index_ramp(key, layer, stops, symmetric)
        self.place(layer, G.ANALYSIS)
        self._state.terrain_index_layer_ids[key] = layer.id()

    def _apply_index_ramp(self, key, layer, stops, symmetric):
        """Paint the layer, anchoring a signed index symmetrically about zero.

        The bound is the 95th percentile of the absolute value, computed when the index
        was — **not** the band maximum. Curvature's extremes are a handful of cells on a
        cliff edge or a building footprint; letting them set the scale paints the entire
        rest of the map as "planar" and the layer says nothing at all.
        """
        if not symmetric:
            apply_raster_ramp(layer, stops)
            return

        bound = (self._state.terrain_index_bounds or {}).get(key) or 1.0
        apply_raster_ramp(layer, stops, max_value=bound, min_value=-bound)


def _compute_indices(dem_path, acc_path, out_dir, report):
    """Pure-ish worker body: read, compute, write. No Qt beyond the progress callable."""
    import numpy as np
    import rasterio

    from terrainflow_assessment.modules.dem_loader import (
        aspect_degrees,
        slope_degrees,
    )
    from terrainflow_assessment.modules.terrain_indices import (
        DEFAULT_MIN_TAN_BETA,
        curvature,
        sediment_transport_index,
        stream_power_index,
        topographic_wetness_index,
    )

    report(5, "Reading terrain…")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        cell_w = abs(transform.a)
        cell_h = abs(transform.e)
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float64")
        acc_nodata = src.nodata
    if acc_nodata is not None:
        acc = np.where(acc == acc_nodata, 0.0, acc)

    report(20, "Slope and aspect…")
    slope = slope_degrees(dem, cell_w, cell_h)
    aspect = aspect_degrees(dem, cell_w, cell_h)

    report(40, "Wetness index…")
    twi, floored = topographic_wetness_index(acc, slope, cell_w, cell_h)

    report(55, "Stream power…")
    spi = stream_power_index(acc, slope, cell_w, cell_h)

    report(70, "Sediment transport…")
    sti = sediment_transport_index(acc, slope, cell_w, cell_h)

    report(85, "Curvature…")
    plan, profile = curvature(dem, cell_w, cell_h)

    report(95, "Writing rasters…")
    bands = {
        "twi": twi, "spi": spi, "sti": sti,
        "plan_curvature": plan, "profile_curvature": profile,
        "aspect": aspect,
    }
    paths = {}
    for key, band in bands.items():
        path = os.path.join(out_dir, f"terrain_{key}.tif")
        _write_like(path, band, transform, crs)
        paths[key] = path

    # The display bound for each signed index, taken here where the values are still in
    # memory. p95 of |value|, not the extreme: curvature's largest magnitudes are a
    # handful of cells on a cliff edge, and scaling to them paints everything else flat.
    bounds = {}
    for key in ("plan_curvature", "profile_curvature"):
        finite = bands[key][np.isfinite(bands[key])]
        bounds[key] = float(np.percentile(np.abs(finite), 95)) if finite.size else 1.0

    report(100, "Terrain indices complete.")
    return {
        "paths": paths,
        "bounds": bounds,
        "floored_cells": floored,
        "domain_cells": int(np.isfinite(dem).sum()),
        "min_tan_beta": DEFAULT_MIN_TAN_BETA,
    }


#: Every raster this plugin writes declares its own nodata. An untagged band is read by
#: pysheds as 0, which for a D-infinity direction means "due east".
_INDEX_NODATA = -9999.0


def _write_like(path, band, transform, crs):
    import numpy as np
    import rasterio

    out = np.where(np.isfinite(band), band, _INDEX_NODATA).astype("float32")
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="float32",
        crs=crs, transform=transform,
        width=out.shape[1], height=out.shape[0],
        count=1, compress="lzw", nodata=_INDEX_NODATA,
    ) as dst:
        dst.write(out, 1)
    return path
