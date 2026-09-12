"""
simulation.py — SimulationController

Handles: simulation run, per-frame display, ponding visualisation, fill-status
layer, play/pause timer, and post-intervention report building.
"""

from __future__ import annotations

import os

from qgis.core import (
    QgsColorRampShader,
    QgsFeature,
    QgsField,
    QgsGeometry,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProperty,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsSymbolLayer,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QMetaType, QTimer
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.modules.catchment import SCSRunoff
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers import _symbols as S
from terrainflow_assessment.qgis.controllers._layers import (
    dem_crs,
    remove_layer,
    resolve_layer,
)
from terrainflow_assessment.qgis.workers._lifecycle import worker_is_running
from terrainflow_assessment.qgis.workers.simulation_worker import SimulationWorker


class SimulationController(G.LayerTreeMixin):
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

        self._sim_timer = QTimer()
        self._sim_timer.timeout.connect(self._advance_sim_frame)

        # Set by the plugin: the EarthworksController, which owns the catchment
        # labelling and the overflow routing this simulation must share.
        self.design_tier = None

    def teardown(self):
        """Stop the playback timer before the panel it drives is deleted.

        `plugin.py`'s unload loop skips a controller with no `teardown` attribute,
        and this controller had none — so a plugin reload with Play still toggled
        left a 500 ms `QTimer` running against a dismantled panel. Two frames
        later `_advance_sim_frame` reads `self._panel.sim_frame()` on a deleted
        widget, inside a Qt slot, which PyQt turns into a process abort with no
        traceback: the failure mode `terrain.py:157-161` describes.

        Disconnect as well as stop. A stopped timer is still connected, and the
        one Python reference keeping this controller alive is the plugin's, which
        unload is in the middle of dropping.
        """
        try:
            self._sim_timer.stop()
            self._sim_timer.timeout.disconnect(self._advance_sim_frame)
        except (TypeError, RuntimeError):
            # Never connected, or the C++ timer is already gone.
            pass

    # ---------------------------------------------------------------- Run

    def run_simulation(self):
        """Start the fill simulation, re-arming the button on any path that does not.

        The button is disabled on click, so every early return here — no DEM, no
        stores, a refusal because one is already running — has to put it back or the
        stage is dead until the panel is rebuilt.
        """
        started = False
        try:
            started = self._start_simulation()
        finally:
            if not started:
                self._panel.set_simulation_idle()

    def _start_simulation(self):
        """Returns True once a worker is actually running."""
        if worker_is_running(self._state, "sim_worker"):
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "A simulation is already running — wait for it to finish.")
            return

        # Use earthworks DEM + flow direction when available, fall back to baseline
        if self._state.modified_dem_path and self._state.earthworks_result:
            dem_path = self._state.modified_dem_path
            fdir_path = self._state.earthworks_result.get("flow_direction")
        else:
            dem_path = self._state.dem_path
            fdir_path = (self._state.baseline_result or {}).get("flow_direction")

        if not dem_path or not fdir_path or not os.path.exists(fdir_path):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run baseline analysis first to generate flow direction. "
                "For earthworks simulation, run 'Re-analyse with Earthworks' first."
            )
            return

        # Build rainfall data
        if self._panel.sim_mode == "uniform":
            rain_mm = self._panel.sim_rainfall_mm
            dur_hr = self._panel.sim_duration_hr
            step_min = self._panel.sim_timestep_min
            n_steps = max(1, int(dur_hr * 60 / step_min))
            rainfall_data = [(0, 0.0)]
            for i in range(1, n_steps + 1):
                t_min = i * step_min
                cum_rain = rain_mm * (t_min / (dur_hr * 60))
                rainfall_data.append((t_min, cum_rain))
        else:
            csv_path = self._panel.sim_csv_path
            if not csv_path or not os.path.exists(csv_path):
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", "Select a valid CSV file."
                )
                return
            try:
                rainfall_data = SCSRunoff.parse_hyetograph_csv(csv_path)
            except Exception as exc:
                self._iface.messageBar().pushCritical(
                    "TerrainFlow Assessment", f"CSV parse error: {exc}"
                )
                return

        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        stores = build_stores_from_earthworks(
            self._state.earthwork_manager.get_enabled(),
            soil_name=self._panel.earthwork_soil_name,
            dem_path=dem_path,
        )

        # The simulation splits runoff by the same direct-catchment labelling the
        # design tier uses, and cascades along the same overflow network. Ask for them
        # rather than guessing: a design loaded from file and never edited has had no
        # live assessment, and there is no honest way to apportion runoff without one.
        if stores and self.design_tier is not None:
            self.design_tier.ensure_design_tier()
        if stores and self._state.catchment_labels is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run the design analysis before simulating — the simulation needs "
                "each feature's catchment to know where the water goes.",
            )
            return

        self._state.sim_worker = SimulationWorker(
            dem_path=dem_path,
            fdir_path=fdir_path,
            output_dir=self._state.output_dir,
            # The panel's CN, as every other controller uses. Re-deriving it from the
            # soil table here discarded both a typed override and the ground condition,
            # so the simulation could run the same site against a different storm than
            # the baseline it is being compared with.
            cn=self._panel.cn,
            moisture=self._panel.moisture,
            rainfall_data=rainfall_data,
            routing=self._panel.routing,
            earthwork_stores=stores,
            soil_name=self._panel.earthwork_soil_name,
            catchment_labels=self._state.catchment_labels,
            catchment_label_ids=self._state.catchment_label_ids,
            store_routing=self._state.balance_routing,
        )
        self._state.sim_worker.progress.connect(self._panel.set_simulation_progress)
        self._state.sim_worker.completed.connect(self._on_simulation_complete)
        self._state.sim_worker.error.connect(self._on_simulation_error)
        self._state.sim_worker.start()
        return True

    # ---------------------------------------------------------------- Completion

    def _on_simulation_error(self, tb):
        self._panel.set_simulation_idle()
        # The message says to look at the console, so put something there. This was
        # bound and discarded, which made the instruction actively misleading —
        # baseline and earthworks both print theirs.
        print("TerrainFlow Assessment — simulation error:")
        print(tb)
        self._iface.messageBar().pushCritical(
            "TerrainFlow Assessment", "Simulation failed — see Python console."
        )

    def _on_simulation_complete(self, result):
        self._state.sim_result = result
        self._panel.set_simulation_ready(result)

        # Pre-compute global colour-scale maxima
        self._state.sim_global_max_inc = 1.0
        self._state.sim_global_max_cum = 1.0
        try:
            import rasterio
            peak_path = result.get("peak_path")
            if peak_path and os.path.exists(peak_path):
                with rasterio.open(peak_path) as src:
                    data = src.read(1).astype("float64")
                    data[data == src.nodata] = 0.0
                    v = float(data.max())
                    self._state.sim_global_max_inc = max(v, 1.0)
            frames = result.get("frames", [])
            if frames:
                last_cum = frames[-1].get("cum")
                if last_cum and os.path.exists(last_cum):
                    with rasterio.open(last_cum) as src:
                        data = src.read(1).astype("float64")
                        data[data == src.nodata] = 0.0
                        v = float(data.max())
                        self._state.sim_global_max_cum = max(v, 1.0)
        except Exception:
            pass

        # Build earthwork centroid map for the fill-status layer
        self._state.sim_ew_centroids = {}
        try:
            import json

            from shapely.geometry import shape as _shp
            for ew in self._state.earthwork_manager.get_enabled():
                shp = _shp(json.loads(ew.geometry.asJson()))
                c = shp.centroid
                self._state.sim_ew_centroids[ew.id] = (QgsPointXY(c.x, c.y), ew.name)
        except Exception:
            pass

        self._create_sim_fill_layer()
        self._setup_sim_ponding()

        # Backfill baseline hydrograph
        sim_ts = result.get("timestep_table", [])
        if sim_ts and self._state.baseline_report:
            self._state.baseline_report.timestep_table = [
                {"time_hr": r["time_hr"],
                 "outflow_ls": r.get("outflow_ls_baseline", 0.0)}
                for r in sim_ts
            ]
            peak_bl = max(
                (r["outflow_ls"] for r in self._state.baseline_report.timestep_table),
                default=0.0,
            )
            if not self._state.baseline_report.peak_outflow_ls:
                self._state.baseline_report.peak_outflow_ls = peak_bl

        from terrainflow_assessment.modules.reporting import PostInterventionReport, compare

        earthwork_summary = result.get("earthwork_summary", [])
        self._merge_verification_into_summary(earthwork_summary)

        self._state.post_report = PostInterventionReport(
            exit_volume_m3=result.get("total_outflow_m3", 0),
            peak_outflow_ls=result.get("peak_outflow_ls", 0),
            peak_outflow_time_hr=result.get("peak_outflow_time_hr", 0),
            earthwork_summary=earthwork_summary,
            timestep_table=result.get("timestep_table", []),
            exit_points=(self._state.earthworks_result or {}).get("exit_points", []),
        )

        if self._state.baseline_report and self._state.post_report:
            comparison = compare(self._state.baseline_report, self._state.post_report)
            comparison.verification = self._state.verification  # terrain-vs-analytic (§4)
            self._state.comparison = comparison
            # Enrichment, not the source: the summary is the design tier's, and a
            # simulation adds the two timing lines nothing else can support.
            self._panel.set_report_summary(
                self._state.balance, comparison=comparison,
                burn=self._state.burn_quantities)

    def _merge_verification_into_summary(self, summary):
        """Copy per-feature terrain ponding + Δ from state.verification onto summary rows."""
        v = self._state.verification
        if not v:
            return
        # Joined on id. `per_feature` rows carry display names, which two features
        # can share, so matching on those merged one feature's terrain ponding onto
        # the other's summary row.
        by_id = {f.get("id") or f["name"]: f for f in v.per_feature}
        for row in summary:
            feat = by_id.get(row.get("id") or row.get("name"))
            if feat:
                row["terrain_ponding_m3"] = feat["terrain_m3"]
                row["capacity_delta_pct"] = feat["delta_pct"]
                row["routing_only"] = feat["routing_only"]

    # ---------------------------------------------------------------- Ponding

    def _setup_sim_ponding(self):
        import json

        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rasterize
        from shapely.geometry import shape as _shp

        ponding_path = self._state.ponding_raster_path
        if not ponding_path or not os.path.exists(ponding_path):
            return

        try:
            with rasterio.open(ponding_path) as src:
                capacity = src.read(1).astype("float32")
                # `is not None`, not `or`: a raster that legitimately declares
                # nodata=0.0 is falsy, and `or` would silently swap it for
                # -9999 and leave every real hole unmasked.
                nodata = src.nodata if src.nodata is not None else -9999.0
                capacity[capacity == nodata] = 0.0
                self._state.sim_ponding_capacity = capacity
                self._state.sim_ponding_meta = dict(src.meta)
                transform = src.transform
                shape = capacity.shape
        except Exception:
            return

        # The bed under the water. Each frame solves for the level that holds the
        # volume delivered so far, which needs the ground, so it is read once here
        # rather than twice a second during playback.
        self._state.sim_ponding_ground = None
        dem_path = self._state.modified_dem_path or self._state.dem_path
        try:
            with rasterio.open(dem_path) as src:
                ground = src.read(1).astype("float64")
                if src.nodata is not None:
                    ground[ground == src.nodata] = np.nan
                if ground.shape == shape:
                    self._state.sim_ponding_ground = ground
        except Exception:
            pass

        self._sim_pools = None
        self._state.sim_ponding_masks = {}
        for ew in self._state.earthwork_manager.get_enabled():
            try:
                shp = _shp(json.loads(ew.geometry.asJson()))
                buf = getattr(ew, "width", 2.0) or 2.0
                shp_buf = shp.buffer(buf * 5)
                mask = _rasterize(
                    [(shp_buf, 1)],
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                )
                self._state.sim_ponding_masks[ew.id] = (mask == 1) & (capacity > 0.001)
            except Exception:
                pass

        # Remove any previous frame/outline layers
        for attr in ("sim_ponding_frame_layer_id", "sim_ponding_outline_layer_id"):
            remove_layer(self._project, getattr(self._state, attr, None))
            setattr(self._state, attr, None)

        # Create static full-capacity outline layer
        try:
            outline_path = os.path.join(self._state.output_dir, "ponding_outline.tif")
            out_meta = dict(self._state.sim_ponding_meta)
            out_meta.update(dtype="float32", nodata=-9999.0, count=1)
            import numpy as np
            outline_data = np.where(capacity > 0.001, 1.0, -9999.0).astype("float32")
            with rasterio.open(outline_path, "w", **out_meta) as dst:
                dst.write(outline_data, 1)

            outline_layer = QgsRasterLayer(outline_path, "Ponding Capacity Outline")
            if outline_layer.isValid():
                shader = QgsRasterShader()
                cr = QgsColorRampShader()
                cr.setColorRampType(QgsColorRampShader.Exact)
                cr.setColorRampItemList([
                    QgsColorRampShader.ColorRampItem(1.0, QColor(0, 230, 200, 120), "Max capacity"),
                ])
                shader.setRasterShaderFunction(cr)
                renderer = QgsSingleBandPseudoColorRenderer(
                    outline_layer.dataProvider(), 1, shader)
                outline_layer.setRenderer(renderer)
                self.place(outline_layer, G.VERIFY)
                self._state.sim_ponding_outline_layer_id = outline_layer.id()
        except Exception:
            pass

        self._update_sim_ponding_frame({})

    def _frame_depth(self, capacity, masks, fills):
        """Where the water stands this frame, as a depth raster.

        Solved the way the verification layer solves it — ``event_pond_depth`` fills
        each pool from the bottom to the level that holds the volume delivered. The
        frame used to scale the full pond's depth by the fill fraction instead, which
        is the exact method that function's docstring names as wrong: it keeps the
        full pond's footprint and paints water up banks it never reaches. Both layers
        sit in the same Verify group, so the two drawings of one part-full pond
        disagreed about where its shoreline was.

        Falls back to the fraction scaling only when there is no ground surface to
        solve against — an empty map is worse than an approximate one, but the
        approximation is never preferred.
        """
        import numpy as np

        from terrainflow_assessment.modules.reporting import (
            event_pond_depth,
            group_pools,
        )

        ground = getattr(self._state, "sim_ponding_ground", None)
        meta = self._state.sim_ponding_meta or {}
        transform = meta.get("transform")
        cell_area = (abs(transform.a * transform.e) if transform is not None else 0.0)

        footprints = [(key, mask) for key, mask in masks.items() if np.any(mask)]
        if ground is None or not footprints or cell_area <= 0:
            partial = np.zeros_like(capacity, dtype="float32")
            for key, mask in footprints:
                fd = fills.get(key)
                fraction = min((fd["fill_pct"] / 100.0), 1.0) if fd else 0.0
                partial[mask] = capacity[mask] * fraction
            return partial

        # Geometry alone, so it holds for every frame of the playback.
        if getattr(self, "_sim_pools", None) is None:
            self._sim_pools = group_pools(capacity, footprints)
        pools = self._sim_pools

        # Each feature's share in cubic metres, measured off the pond the burn
        # actually made — the same raster the shoreline is drawn from, so the two
        # cannot disagree about how much water a full pond is.
        stored = {}
        for key, mask in footprints:
            fd = fills.get(key)
            fraction = min((fd["fill_pct"] / 100.0), 1.0) if fd else 0.0
            if fraction <= 0:
                continue
            stored[key] = float(capacity[mask].sum()) * cell_area * fraction

        depth = event_pond_depth(capacity, ground, cell_area, footprints, stored,
                                 pools=pools)
        return depth.astype("float32")

    def _update_sim_ponding_frame(self, fills):
        import numpy as np
        import rasterio

        capacity = self._state.sim_ponding_capacity
        meta = self._state.sim_ponding_meta
        masks = self._state.sim_ponding_masks
        if capacity is None or meta is None:
            return

        partial = self._frame_depth(capacity, masks, fills)

        # Two filenames, alternated. The previous frame's layer still has its file
        # open, and on Windows GDAL holds that lock — so rewriting the same path threw,
        # the bare `except: return` below swallowed it, and playback sat on frame 0
        # looking like a slow simulation rather than a failure. Removing the layer
        # first is not enough on its own either: the provider closes lazily, so the
        # lock can outlive the call. Writing to the file the live layer is *not* using
        # sidesteps the question entirely.
        self._sim_frame_slot = 1 - getattr(self, "_sim_frame_slot", 1)
        frame_path = os.path.join(self._state.output_dir,
                                  f"sim_ponding_frame_{self._sim_frame_slot}.tif")
        out_meta = dict(meta)
        out_meta.update(dtype="float32", nodata=-9999.0, count=1)
        data = np.where(partial > 0.001, partial, -9999.0).astype("float32")
        try:
            with rasterio.open(frame_path, "w", **out_meta) as dst:
                dst.write(data, 1)
        except Exception as exc:
            # Said out loud. A frozen playback with no message is indistinguishable
            # from a simulation that is simply still thinking.
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"Could not write the ponding frame — playback will not advance ({exc})")
            return

        remove_layer(self._project, self._state.sim_ponding_frame_layer_id)
        self._state.sim_ponding_frame_layer_id = None

        layer = QgsRasterLayer(frame_path, "Ponding Fill")
        if layer.isValid():
            self._apply_ponding_ramp(layer)
            self.place(layer, G.VERIFY)
            self._state.sim_ponding_frame_layer_id = layer.id()

    def _apply_ponding_ramp(self, layer):
        """The same ramp Baseline paints its captured water with — and the same
        *scale*, which is the half this method was missing.

        It called `apply_raster_ramp` directly, which CLAUDE.md states as "never",
        and with no `max_value` — so every frame stretched the stops over its own
        band maximum and the same depth changed colour as the pools filled, in the
        same Verify group as layers already on the shared `ponding` family. This
        method's docstring already claimed the mismatch was fixed.

        Scaled to the **capacity** raster, not to this frame: that is what
        `_build_event_pond_layers` does for the event pond against the full pond,
        and the argument is the same one — a frame is a stage of the pond, so it
        has to be read against the pond.
        """
        import numpy as np

        from terrainflow_assessment.core.registry.map_palette import (
            WATER_CAPTURED,
        )

        capacity = self._state.sim_ponding_capacity
        top = None
        if capacity is not None:
            finite = np.asarray(capacity, dtype="float64")
            if np.isfinite(finite).any():
                top = float(np.nanmax(finite))
        S.apply_shared_ramp(self._state, self._project, "ponding", layer,
                            WATER_CAPTURED, top)

    # ---------------------------------------------------------------- Fill layer

    def _create_sim_fill_layer(self):
        remove_layer(self._project, self._state.sim_fill_layer_id)
        self._state.sim_fill_layer_id = None

        if not self._state.sim_ew_centroids:
            return

        crs_str = dem_crs(self._state)
        layer = QgsVectorLayer(f"Point?crs={crs_str}", "Earthwork Fill Status", "memory")
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("name",       QMetaType.QString),
            QgsField("fill_pct",   QMetaType.Double),
            QgsField("overflowed", QMetaType.Int),
        ])
        layer.updateFields()

        feats = []
        for pt, label in self._state.sim_ew_centroids.values():
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(pt))
            f.setAttributes([label, 0.0, 0])
            feats.append(f)
        pr.addFeatures(feats)

        sym = QgsMarkerSymbol.createSimple({
            "name": "circle", "size": "7", "color": "0,180,0,220",
            "outline_color": "50,50,50,200", "outline_width": "0.5",
        })
        fill_color_expr = (
            "color_hsv("
            "  scale_linear(\"fill_pct\", 0, 100, 120, 0),"
            "  90, 85"
            ")"
        )
        outline_expr = (
            "CASE WHEN \"overflowed\" = 1 "
            "THEN color_rgb(220,0,0) "
            "ELSE color_rgb(50,50,50) END"
        )
        outline_width_expr = (
            "CASE WHEN \"overflowed\" = 1 THEN 2.5 ELSE 0.5 END"
        )
        sym.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyFillColor, QgsProperty.fromExpression(fill_color_expr))
        sym.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(outline_expr))
        sym.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth, QgsProperty.fromExpression(outline_width_expr))
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

        from qgis.core import QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont
        text_fmt = QgsTextFormat()
        font = QFont()
        font.setBold(True)
        font.setPointSize(8)
        text_fmt.setFont(font)
        text_fmt.setColor(QColor(20, 20, 20))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor(255, 255, 255))
        buf.setSize(1.2)
        text_fmt.setBuffer(buf)
        lbl = QgsPalLayerSettings()
        lbl.fieldName = "concat(\"name\", '\\n', round(\"fill_pct\", 0), '%')"
        lbl.isExpression = True
        lbl.enabled = True
        lbl.setFormat(text_fmt)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
        layer.setLabelsEnabled(True)

        self.place(layer, G.VERIFY)
        self._state.sim_fill_layer_id = layer.id()

    # ---------------------------------------------------------------- Frame display

    def show_sim_frame(self, idx):
        if not self._state.sim_result:
            return
        frames = self._state.sim_result.get("frames", [])
        if not frames or idx >= len(frames):
            return
        frame = frames[idx]
        mode = self._panel.sim_display_mode

        path = frame.get("inc" if mode == "inc" else "cum")
        if path and os.path.exists(path):
            # By id, like every other layer the plugin owns. The name scan this
            # replaces would have removed any layer of the user's called
            # "Simulation Frame" along with ours.
            remove_layer(self._project, self._state.sim_frame_layer_id)
            self._state.sim_frame_layer_id = None
            layer = QgsRasterLayer(path, "Simulation Frame")
            if layer.isValid():
                global_max = (
                    self._state.sim_global_max_inc
                    if mode == "inc"
                    else self._state.sim_global_max_cum
                )
                self._apply_stream_ramp(layer, global_max)
                self.place(layer, G.VERIFY)
                self._state.sim_frame_layer_id = layer.id()

        time_labels = self._state.sim_result.get("time_labels", [])
        time_str = time_labels[idx] if idx < len(time_labels) else "—"

        fills = frame.get("fills", {})
        new_overflows = [fd.get("name", key) for key, fd in fills.items()
                         if fd.get("first_overflow_this_step")]
        overflow_str = ("  ⚠ OVERFLOW: " + ", ".join(new_overflows)) if new_overflows else ""
        self._panel.set_sim_time_label(time_str + overflow_str)

        fill_layer = resolve_layer(self._project, self._state.sim_fill_layer_id)
        centroids = self._state.sim_ew_centroids
        if fill_layer is not None and centroids:
            pr = fill_layer.dataProvider()
            pr.truncate()
            feats = []
            for key, (pt, label) in centroids.items():
                fd = fills.get(key, {"fill_pct": 0.0, "overflowed": False})
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPointXY(pt))
                f.setAttributes([label, fd["fill_pct"], 1 if fd["overflowed"] else 0])
                feats.append(f)
            pr.addFeatures(feats)
            fill_layer.triggerRepaint()

        if self._state.sim_ponding_capacity is not None:
            self._update_sim_ponding_frame(fills)

    def _apply_stream_ramp(self, layer, max_acc=None):
        """The same channel ramp Baseline uses, on the same shared scale.

        Decision #3: through the `"streams"` family, as `baseline.py` paints its own
        streams, so the Simulation Frame reads against Baseline's streams on one
        scale rather than on a global maximum of its own. It does not rescale per
        frame — `max_acc` is already a whole-run maximum — so this was a decision
        about comparability, not a defect; but it was also the tree's second direct
        `apply_raster_ramp` call, and the shared family is where the rule puts it.
        """
        from terrainflow_assessment.core.registry.map_palette import STREAMS

        S.apply_shared_ramp(self._state, self._project, "streams",
                            layer, STREAMS, max_acc)

    # ---------------------------------------------------------------- Timer

    def on_sim_play_toggled(self, playing):
        if playing:
            self._sim_timer.start(500)
        else:
            self._sim_timer.stop()

    def _advance_sim_frame(self):
        if not self._state.sim_result:
            return
        n = len(self._state.sim_result.get("frames", []))
        next_idx = self._panel.sim_frame() + 1
        if next_idx >= n:
            next_idx = 0
        self._panel.set_sim_frame(next_idx)
