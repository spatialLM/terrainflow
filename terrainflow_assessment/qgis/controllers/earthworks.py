"""
earthworks.py — EarthworksController

Handles: earthwork drawing tools, properties dialog, layer management,
DEM re-analysis with earthworks, ponding query, slope visualisation.
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
from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.core.registry.earthwork_types import all_types, get_type
from terrainflow_assessment.map_tools.contour_segment_tool import ContourSegmentTool
from terrainflow_assessment.map_tools.draw_line_tool import DrawLineTool
from terrainflow_assessment.map_tools.draw_polygon_tool import DrawPolygonTool
from terrainflow_assessment.map_tools.edit_earthwork_tool import EditEarthworkTool
from terrainflow_assessment.map_tools.ponding_query_tool import PondingQueryTool
from terrainflow_assessment.map_tools.select_contour_tool import SelectContourTool
from terrainflow_assessment.modules.earthwork_design import (
    Earthwork,
    calculate_capacity,
)
from terrainflow_assessment.modules.swale_design import (
    contour_to_swale_geometry,
    sample_total_inflow,
)
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker


class EarthworksController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    # ---------------------------------------------------------------- Drawing tools

    def activate_draw_swale(self, mode):
        if mode == "contour":
            if not self._state.contour_layer:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "Run contour analysis first, then pick a segment on a contour.",
                )
                return
            tool = ContourSegmentTool(self._canvas, self._state.contour_layer)
            tool.segment_selected.connect(
                lambda geom, elev, coords: self._on_contour_selected_for_swale(
                    geom, elev, coords
                )
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)
        elif mode == "full_contour":
            if not self._state.contour_layer:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment",
                    "Run contour analysis first, then click a contour.",
                )
                return
            tool = SelectContourTool(self._canvas, self._state.contour_layer)
            tool.contour_selected.connect(
                lambda geom, elev, coords: self._on_contour_selected_for_swale(
                    geom, elev, coords
                )
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)
        else:
            tool = DrawLineTool(self._canvas,
                                slope_raster_path=self._state.slope_raster_path,
                                tool_label="swale")
            tool.line_drawn.connect(lambda geom: self._on_geometry_drawn("swale", geom))
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)

    def _on_contour_selected_for_swale(self, geom, elevation, contour_coords=None):
        swale_geom = contour_to_swale_geometry(geom)
        self._on_geometry_drawn("swale", swale_geom, source_contour=contour_coords)

    def activate_draw_earthwork(self, key):
        """Registry-driven draw dispatch: geometry type decides the map tool."""
        try:
            cfg = get_type(key)
        except KeyError:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Unknown earthwork type: {key}"
            )
            return
        if cfg.geom_type == "Polygon":
            self.activate_draw_polygon(key)
        else:
            self.activate_draw_line(key)

    def activate_draw_line(self, ew_type):
        tool = DrawLineTool(self._canvas,
                            slope_raster_path=self._state.slope_raster_path,
                            tool_label=ew_type)
        tool.line_drawn.connect(lambda geom: self._on_geometry_drawn(ew_type, geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)

    def activate_draw_polygon(self, ew_type):
        tool = DrawPolygonTool(self._canvas,
                               slope_raster_path=self._state.slope_raster_path,
                               tool_label=ew_type)
        tool.polygon_drawn.connect(lambda geom: self._on_geometry_drawn(ew_type, geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)

    def on_usable_area_source_changed(self, source):
        import json

        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union

        if source == "none":
            self._state.usable_polygon = None
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment", "Usable area cleared — full DEM will be used."
            )
            return

        layer = (
            self._panel.analysis_area_layer if source == "analysis"
            else self._panel.earthworks_area_layer
        )
        if layer is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "No layer selected for that area — set it in the Data section first.",
            )
            return

        try:
            polys = []
            for feat in layer.getFeatures():
                geom = feat.geometry()
                if geom and not geom.isEmpty():
                    polys.append(shapely_shape(json.loads(geom.asJson())))
            if not polys:
                raise ValueError("Layer has no valid polygon features.")
            self._state.usable_polygon = unary_union(polys)
            label = "Analysis Area" if source == "analysis" else "Earthworks Area"
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"Usable area set from '{label}'. Run contour analysis to apply.",
            )
        except Exception as exc:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not read usable area layer: {exc}"
            )

    def _on_geometry_drawn(self, ew_type, geometry, source_contour=None):
        from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog

        # Diversions run (and grade) downhill — orient the geometry high→low so the
        # flow arrow and the channel invert both follow the actual slope.
        if ew_type == "diversion":
            geometry = self._orient_downhill(geometry)

        peak_inflow = 0.0
        if self._state.baseline_result:
            acc_path = self._state.baseline_result.get("flow_accumulation")
            if acc_path:
                # Total intercepted accumulation (every crossing), not one peak cell.
                acc_cells = sample_total_inflow(geometry, acc_path)
                cell_area = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
                runoff_mm = self._state.baseline_result.get("runoff_mm", 0)
                peak_inflow = acc_cells * cell_area * runoff_mm / 1000.0

        crest_elev = None
        if ew_type == "dam" and self._state.dem_path:
            try:
                import json

                import rasterio
                from shapely.geometry import shape as shapely_shape
                shp = shapely_shape(json.loads(geometry.asJson()))
                centroid = shp.centroid
                with rasterio.open(self._state.dem_path) as src:
                    t = src.transform
                    col = int((centroid.x - t.c) / t.a)
                    row = int((centroid.y - t.f) / t.e)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        crest_elev = float(src.read(1)[row, col]) + 2.0
            except Exception:
                pass

        n = len(self._state.earthwork_manager) + 1
        ew_name = f"{ew_type.capitalize()} {n}"
        ew = Earthwork(ew_type, geometry, ew_name)
        ew.source_contour_coords = source_contour  # reshape stays contour-locked

        dlg = EarthworkPropertiesDialog(
            ew_type=ew_type,
            geometry=geometry,
            parent=self._iface.mainWindow(),
            earthwork=ew,
            peak_inflow_m3=peak_inflow,
            crest_elevation=crest_elev,
            duration_hours=self._panel.duration_hr,
            dem_path=self._state.dem_path if ew_type == "dam" else None,
            soil_name=self._panel.earthwork_soil_name,
            cn=self._panel.cn,
            overflow_options=self._overflow_options(exclude_id=ew.id),
            own_elevation=self._feature_elevation(geometry),
        )

        if dlg.exec():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            ew.width = getattr(dlg, "get_width", lambda: ew.width)()
            if ew_type == "dam":
                ew.crest_elevation = dlg.get_crest_elevation()
                ew.key_into_banks = getattr(dlg, "get_key_into_banks", lambda: False)()
            elif ew_type == "swale":
                ew.companion_berm = getattr(dlg, "get_companion_berm", lambda: False)()
            elif ew_type == "diversion":
                # Was silently dropped on create (only the edit path read it).
                ew.gradient_pct = getattr(dlg, "get_gradient_pct", lambda: ew.gradient_pct)()
            if ew_type == "basin":
                # After depth — the wall_slope setter back-solves batter_run from it.
                ew.wall_slope = getattr(dlg, "get_wall_slope", lambda: 0.0)()
            ew.overflow_target_id = getattr(dlg, "get_overflow_target_id", lambda: None)()
            # Apply the bottom width (channels only; None otherwise) — the canonical
            # cross-section field that drives capacity and the burn footprint.
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw

            if ew_type == "dam":
                ew.capacity_m3 = self._compute_dam_capacity(ew)
                ew.capacity_l = ew.capacity_m3 * 1000.0
            else:
                ew.capacity_m3, ew.capacity_l = calculate_capacity(
                    ew_type, geometry, ew.depth, ew.width,
                    getattr(ew, "companion_berm", False),
                    bottom_width=getattr(ew, "bottom_width_m", None),
                    batter_run=getattr(ew, "batter_run_m", None),
                )
            self._state.earthwork_manager.add(ew)
            self._panel.add_earthwork_to_list(
                len(self._state.earthwork_manager) - 1, ew.summary()
            )
            self._refresh_ew_layer()
            self._recompute_live_assessment()
            self._mark_design_edit()
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def _on_draw_cancelled(self):
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def edit_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        ew = self._state.earthwork_manager.get(idx)
        from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog
        dlg = EarthworkPropertiesDialog(
            ew_type=ew.type,
            geometry=ew.geometry,
            parent=self._iface.mainWindow(),
            earthwork=ew,
            duration_hours=self._panel.duration_hr,
            dem_path=self._state.dem_path if ew.type == "dam" else None,
            soil_name=self._panel.earthwork_soil_name,
            cn=self._panel.cn,
            overflow_options=self._overflow_options(exclude_id=ew.id),
            own_elevation=self._feature_elevation(ew.geometry),
        )
        if dlg.exec():
            ew.name = dlg.get_name()
            ew.depth = dlg.get_depth()
            # Re-read every edited dimension (the edit path previously dropped
            # width / companion / gradient / side-slope changes silently).
            ew.width = getattr(dlg, "get_width", lambda: ew.width)()
            if ew.type == "dam":
                ew.crest_elevation = dlg.get_crest_elevation()
                ew.key_into_banks = getattr(dlg, "get_key_into_banks", lambda: False)()
            elif ew.type == "swale":
                ew.companion_berm = getattr(dlg, "get_companion_berm", lambda: False)()
            elif ew.type == "diversion":
                ew.gradient_pct = getattr(dlg, "get_gradient_pct", lambda: ew.gradient_pct)()
            if ew.type == "basin":
                # After depth — the wall_slope setter back-solves batter_run from it.
                ew.wall_slope = getattr(dlg, "get_wall_slope", lambda: 0.0)()
            ew.overflow_target_id = getattr(dlg, "get_overflow_target_id", lambda: None)()
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw
            if ew.type == "dam":
                ew.capacity_m3 = self._compute_dam_capacity(ew)
                ew.capacity_l = ew.capacity_m3 * 1000.0
            else:
                ew.capacity_m3, ew.capacity_l = calculate_capacity(
                    ew.type, ew.geometry, ew.depth, ew.width,
                    getattr(ew, "companion_berm", False),
                    bottom_width=getattr(ew, "bottom_width_m", None),
                    batter_run=getattr(ew, "batter_run_m", None),
                )
            self._panel.update_earthwork_in_list(idx, ew.summary())
            self._refresh_ew_layer()
            self._recompute_live_assessment()
            self._mark_design_edit()

    def _overflow_options(self, exclude_id=None):
        """(id, name, elevation) of every other earthwork — the dialog's overflow
        targets. Elevation (DEM at centroid, same sampling as the water balance's
        stores) lets the dialog warn when a chosen target sits uphill; None when
        no DEM is loaded."""
        return [
            (e.id, e.name, self._feature_elevation(e.geometry))
            for e in self._state.earthwork_manager.get_all()
            if e.id != exclude_id
        ]

    def _orient_downhill(self, geometry):
        """Reverse a line so its first vertex is the higher end (diversions).

        The diversion arrow points along the drawn line AND the burn grades the
        invert from the first vertex, so both must run high→low. Sample the DEM at
        each endpoint; reverse when the line was drawn uphill. No-op without a DEM
        or on any sampling failure (geometry returned unchanged)."""
        if not self._state.dem_path:
            return geometry
        try:
            import json

            from shapely.geometry import LineString
            from shapely.geometry import shape as shapely_shape

            from terrainflow_assessment.modules.swale_design import (
                snap_point_to_contour_elevation,
            )
            coords = list(shapely_shape(json.loads(geometry.asJson())).coords)
            if len(coords) < 2:
                return geometry
            z0 = snap_point_to_contour_elevation(coords[0], self._state.dem_path)
            z1 = snap_point_to_contour_elevation(coords[-1], self._state.dem_path)
            if z0 is not None and z1 is not None and z1 > z0:
                return QgsGeometry.fromWkt(LineString(coords[::-1]).wkt)
        except Exception:
            pass
        return geometry

    def _feature_elevation(self, geometry):
        """DEM elevation at the geometry centroid (matches store elevation), or None."""
        if not self._state.dem_path:
            return None
        try:
            import json

            from shapely.geometry import shape as shapely_shape

            from terrainflow_assessment.modules.swale_design import (
                snap_point_to_contour_elevation,
            )
            centroid = shapely_shape(json.loads(geometry.asJson())).centroid
            return snap_point_to_contour_elevation(
                (centroid.x, centroid.y), self._state.dem_path
            )
        except Exception:
            return None

    def delete_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.remove(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())
        self._refresh_ew_layer()
        self._recompute_live_assessment()
        self._mark_design_edit()

    def toggle_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.toggle(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())
        self._recompute_live_assessment()
        self._mark_design_edit()

    # ---------------------------------------------------------------- Vertex reshaping (live)

    def activate_edit_earthwork_vertices(self):
        """Start the reshape tool: drag vertices with a live analytical readout."""
        earthworks = self._state.earthwork_manager.get_all()
        if not earthworks:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "No earthworks to reshape — draw one first."
            )
            return
        tool = EditEarthworkTool(
            self._canvas,
            [
                (i, ew.geometry, ew.type, getattr(ew, "source_contour_coords", None))
                for i, ew in enumerate(earthworks)
            ],
        )
        tool.geometry_edited.connect(self._on_vertex_drag)
        tool.edit_finished.connect(self._on_vertex_edit_finished)
        tool.session_ended.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)

    def _on_vertex_drag(self, idx, geometry):
        """Throttled live tier during a drag: cheap geometry metrics + water balance.

        No DEM flood here — a dam keeps its cached capacity until release, so the
        readout stays instant (design record: throttle only the flow-dependent part,
        never run the heavy sub-calcs per mouse event).
        """
        manager = self._state.earthwork_manager
        if not (0 <= idx < len(manager)):
            return  # stale tool index (feature deleted mid-session)
        ew = manager.get(idx)
        ew.geometry = geometry
        if ew.type != "dam":
            ew.capacity_m3, ew.capacity_l = calculate_capacity(
                ew.type, geometry, ew.depth, ew.width,
                getattr(ew, "companion_berm", False),
                bottom_width=getattr(ew, "bottom_width_m", None),
                batter_run=getattr(ew, "batter_run_m", None),
            )
        self._recompute_live_assessment()

    def _on_vertex_edit_finished(self, idx, geometry):
        """Exact tier on release / insert / delete: full capacity + layer + list."""
        manager = self._state.earthwork_manager
        if not (0 <= idx < len(manager)):
            return  # stale tool index (feature deleted mid-session)
        ew = manager.get(idx)
        if ew.type == "diversion":
            geometry = self._orient_downhill(geometry)
        ew.geometry = geometry
        if ew.type == "dam":
            ew.capacity_m3 = self._compute_dam_capacity(ew)
            ew.capacity_l = ew.capacity_m3 * 1000.0
        else:
            ew.capacity_m3, ew.capacity_l = calculate_capacity(
                ew.type, geometry, ew.depth, ew.width,
                getattr(ew, "companion_berm", False),
                bottom_width=getattr(ew, "bottom_width_m", None),
                batter_run=getattr(ew, "batter_run_m", None),
            )
        self._panel.update_earthwork_in_list(idx, ew.summary())
        self._refresh_ew_layer()
        self._recompute_live_assessment()
        self._mark_design_edit()

    # ---------------------------------------------------------------- Verified-vs-design tracking

    def _mark_design_edit(self):
        """A discrete design change happened — the last burn verification is now
        one edit more stale (once a verification exists)."""
        if self._state.edits_since_verify is not None:
            self._state.edits_since_verify += 1
            self._panel.mark_stage("verify", "stale")
        self._update_verified_chip()

    def _update_verified_chip(self):
        n = self._state.edits_since_verify
        if n is None:
            self._panel.set_verified_chip("", False)   # never burned → hidden
        elif n == 0:
            d = self._state.verified_delta_pct
            extra = f" · Δ {d:+.0f}%" if d is not None else ""
            self._panel.set_verified_chip(f"Verified{extra}", True)
        else:
            self._panel.set_verified_chip(
                f"{n} edit{'s' if n != 1 else ''} since verify", False
            )

    # ---------------------------------------------------------------- Dam analytical capacity

    def _compute_dam_capacity(self, ew):
        """One-time DEM flood → dam impounded volume (m³), cached on ``ew.capacity_m3``.

        Heavy (a depression-fill), so run only at dam draw/edit — the live readout reads
        the cached value. Returns 0 without a DEM or a crest; failures degrade to 0.
        """
        if not self._state.dem_path or getattr(ew, "crest_elevation", None) is None:
            return 0.0
        try:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment", "Computing dam storage…"
            )
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            burner = self._state.burner or DEMBurner(self._state.dem_path)
            baseline_ponding = self._cached_baseline_ponding(burner.shape)
            volume = round(
                burner.dam_stage_storage(
                    ew, baseline_ponding=baseline_ponding,
                    key_into_banks=getattr(ew, "key_into_banks", False),
                ),
                2,
            )
            for msg in getattr(burner, "warnings", []):
                self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)
            return volume
        except Exception as exc:
            print(f"TerrainFlow Assessment — dam storage error: {exc}")
            return 0.0

    def _cached_baseline_ponding(self, shape):
        """Baseline ponding array (saves a flood pass), or None if unavailable/mismatched."""
        path = (self._state.baseline_result or {}).get("ponding")
        if not path or not os.path.exists(path):
            return None
        try:
            import numpy as np
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                nodata = src.nodata
            if arr.shape != shape:
                return None
            if nodata is not None:
                arr[arr == nodata] = 0.0
            return np.clip(arr, 0.0, None)
        except Exception:
            return None

    # ---------------------------------------------------------------- Live analytical assessment

    def _recompute_live_assessment(self):
        """Design-tier: recompute the live analytical water balance → panel readout.

        Fast, no burn. Geometry metrics (capacity/cut/fill) always; storm capture %
        once a baseline (flow accumulation) exists. Runoff is recomputed live from the
        current storm/soil inputs, so the score reacts without re-running baseline.
        Fires on every earthwork edit and storm/soil change; failures are swallowed so
        the readout never breaks the edit flow.
        """
        try:
            from terrainflow_assessment.modules.catchment import SCSRunoff
            from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
            from terrainflow_assessment.modules.water_balance import run_water_balance

            all_ews = self._state.earthwork_manager.get_all()
            if not all_ews:
                self._panel.set_network([], {}, 0.0)
                self._panel.set_live_assessment("")
                self._panel.scorecard_empty()
                return

            enabled = [
                ew for ew in all_ews
                if ew.enabled and getattr(ew, "capacity_m3", 0.0) > 0
            ]

            scs = SCSRunoff()
            runoff_mm = scs.runoff_depth(
                self._panel.rainfall_mm,
                scs.adjust_cn(self._panel.cn, self._panel.moisture),
            )
            duration_hr = self._panel.duration_hr or 1.0
            soil = self._panel.earthwork_soil_name

            baseline = self._state.baseline_result or {}
            acc_path = baseline.get("flow_accumulation")
            have_flow = bool(acc_path)
            cell_area = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
            total_runoff_m3 = (
                runoff_mm / 1000.0 * baseline.get("catchment_area_m2", 0.0)
                if have_flow else 0.0
            )

            stores = build_stores_from_earthworks(
                enabled, soil_name=soil, dem_path=self._state.dem_path
            )
            if have_flow:
                by_name = {s.name: s for s in stores}
                for ew in enabled:
                    store = by_name.get(ew.name)
                    if store is None:
                        continue
                    try:
                        # Total intercepted accumulation along the feature — a long
                        # contour swale crosses many drainage paths; a single peak
                        # sample under-read it by orders of magnitude.
                        acc_cells = sample_total_inflow(ew.geometry, acc_path)
                        store.inflow_m3 = acc_cells * cell_area * runoff_mm / 1000.0
                    except Exception:
                        store.inflow_m3 = 0.0

            result = run_water_balance(stores, duration_hr, total_runoff_m3) if stores else None

            # Flow network (Live Assessment) — every earthwork, ordered high→low.
            from terrainflow_assessment.modules.simulation import overflow_graph
            nodes = self._build_network_nodes(all_ews, stores, result)
            edges = overflow_graph(stores)
            exit_m3 = result.site_exit_m3 if result is not None else 0.0
            self._panel.set_network(nodes, edges, exit_m3)
            self._panel.set_live_assessment(self._network_footer(result))

            # Persistent scorecard (Workbench header) — blue means actual water.
            if result is not None and have_flow:
                stored = max(0.0, result.total_captured_m3 - result.total_infiltration_m3)
                self._panel.update_scorecard(
                    result.capture_pct, stored,
                    result.total_infiltration_m3, result.site_exit_m3,
                )
            elif result is not None:
                self._panel.scorecard_no_flow(result.total_capacity_m3)
            else:
                self._panel.scorecard_empty(
                    "No storage yet — draw a swale, basin or dam."
                )
        except Exception as exc:  # never let the readout break the edit flow
            print(f"TerrainFlow Assessment — live assessment error: {exc}")

    def _build_network_nodes(self, all_ews, stores, result):
        """Node dicts for the flow network — one per earthwork, water from the balance."""
        from terrainflow_assessment.core.registry.earthwork_types import get_type

        store_elev = {s.name: s.elevation for s in stores}
        per = {f["name"]: f for f in result.per_feature} if result is not None else {}
        nodes = []
        for i, ew in enumerate(all_ews):
            try:
                colour = get_type(ew.type).style[1]
            except KeyError:
                colour = "#888888"
            elev = store_elev.get(ew.name)
            if elev is None:
                elev = self._feature_elevation(ew.geometry) or 0.0
            f = per.get(ew.name)
            nodes.append({
                "index": i,
                "id": ew.id,
                "name": ew.name,
                "ew_type": ew.type,
                "colour": colour,
                "elevation": elev,
                "capacity_m3": getattr(ew, "capacity_m3", 0.0) or 0.0,
                "stored_m3": f["stored_m3"] if f else 0.0,
                "fill_pct": f["fill_pct"] if f else 0.0,
                "overflowed": bool(f["overflowed"]) if f else False,
                "enabled": bool(ew.enabled),
                "has_water": f is not None,
                "summary": ew.summary(),
            })
        return nodes

    def _network_footer(self, result):
        """Totals + disclaimer line beneath the network."""
        if result is None:
            return ""
        return (
            f"Capacity {result.total_capacity_m3:,.0f} m³ · "
            f"Cut {result.total_cut_m3:,.0f} · Fill {result.total_fill_m3:,.0f} m³"
            "  ·  Analytical estimate — verify with Re-analyse."
        )

    # ---------------------------------------------------------------- Earthwork layers

    def _ensure_ew_layers(self):
        from qgis.core import QgsRuleBasedRenderer, QgsTextBufferSettings
        from qgis.PyQt.QtGui import QFont

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        root = self._project.instance().layerTreeRoot()
        self._state.ew_group = (
            root.findGroup("Earthworks") or root.insertGroup(0, "Earthworks")
        )

        # Registry-driven: the type registry is the single source of layer styling
        # (matching the panel's draw-button colours); a future register_type() gets
        # its map layer automatically.
        for ew_type, cfg in all_types().items():
            geom_type = cfg.geom_type
            display_name = f"{cfg.label}s"
            color_hex = cfg.style[1]
            existing = self._state.ew_layers.get(ew_type)
            if existing and self._project.instance().mapLayer(existing.id()):
                continue

            layer = QgsVectorLayer(f"{geom_type}?crs={crs_str}", display_name, "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name",        QMetaType.QString),
                QgsField("type",        QMetaType.QString),
                QgsField("capacity_m3", QMetaType.Double),
                QgsField("enabled",     QMetaType.Int),
            ])
            layer.updateFields()

            # Rule-based: enabled features get the rich casing+signature symbol;
            # disabled ones render greyed + dashed so a toggled-off earthwork reads
            # as inactive at a glance.
            on_sym = self._build_ew_symbol(cfg, enabled=True)
            off_sym = self._build_ew_symbol(cfg, enabled=False)
            root_rule = QgsRuleBasedRenderer.Rule(None)
            root_rule.appendChild(
                QgsRuleBasedRenderer.Rule(on_sym, filterExp='"enabled" = 1', label="Enabled")
            )
            root_rule.appendChild(
                QgsRuleBasedRenderer.Rule(off_sym, filterExp='"enabled" = 0', label="Disabled")
            )
            layer.setRenderer(QgsRuleBasedRenderer(root_rule))

            text_fmt = QgsTextFormat()
            font = QFont()
            font.setBold(True)
            font.setPointSize(8)
            text_fmt.setFont(font)
            text_fmt.setColor(QColor(color_hex))
            buf = QgsTextBufferSettings()
            buf.setEnabled(True)
            buf.setColor(QColor(255, 255, 255))
            buf.setSize(1.2)
            text_fmt.setBuffer(buf)
            lbl = QgsPalLayerSettings()
            # Name + storage metric (e.g. "Swale 1 · 140 m³"). Type is carried by
            # the symbol signature + colour, so it's not repeated in the label.
            lbl.fieldName = (
                "\"name\" || CASE WHEN \"capacity_m3\" > 0 THEN "
                "' · ' || format_number(\"capacity_m3\", 0) || ' m³' ELSE '' END"
            )
            lbl.isExpression = True
            lbl.enabled = True
            lbl.setFormat(text_fmt)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
            layer.setLabelsEnabled(True)

            self._project.instance().addMapLayer(layer, False)
            self._state.ew_group.addLayer(layer)
            self._state.ew_layers[ew_type] = layer

    # ---------------------------------------------------------------- Earthwork symbology

    def _build_ew_symbol(self, cfg, enabled=True):
        """Rich per-type canvas symbol. Casing (white underlay) for legibility on any
        background, a per-type line signature (dam ticks / diversion flow arrows /
        berm + diversion dash patterns), and a greyed dashed variant when disabled.
        Every embellishment is best-effort — on any failure it degrades to a plain
        coloured symbol so layer creation never breaks."""
        from qgis.PyQt.QtGui import QColor

        base = QColor(cfg.style[1])
        try:
            main_w = float(cfg.style[2])
        except (TypeError, ValueError):
            main_w = 2.0
        colour = base if enabled else QColor("#9aa4a2")

        if cfg.geom_type == "Polygon":
            return self._build_ew_fill(colour, enabled, main_w)
        return self._build_ew_line(cfg.key, colour, enabled, main_w)

    def _build_ew_fill(self, colour, enabled, main_w):
        from qgis.core import QgsFillSymbol, QgsSimpleFillSymbolLayer
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QColor
        try:
            rgba = QColor(colour.red(), colour.green(), colour.blue(), 45 if enabled else 22)
            fl = QgsSimpleFillSymbolLayer(rgba)
            fl.setStrokeColor(colour)
            fl.setStrokeWidth(0.7 if enabled else 0.4)
            if not enabled:
                fl.setStrokeStyle(Qt.PenStyle.DashLine)
            return QgsFillSymbol([fl])
        except Exception:
            return QgsFillSymbol.createSimple(
                {"style": "no", "outline_color": colour.name(), "outline_width": str(main_w)}
            )

    def _build_ew_line(self, key, colour, enabled, main_w):
        from qgis.core import QgsLineSymbol, QgsSimpleLineSymbolLayer
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtGui import QColor
        try:
            layers = []
            if enabled:
                casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 235))
                casing.setWidth(main_w + 0.9)
                casing.setPenCapStyle(Qt.PenCapStyle.RoundCap)
                casing.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
                layers.append(casing)

            # Diversion: render the channel itself as a repeated flow-arrow ribbon
            # (QgsArrowSymbolLayer follows the drawn line, so direction is
            # unambiguous — no marker-rotation guessing). Falls back to a dash-dot
            # line if the arrow layer isn't available.
            arrow = self._arrow_line_layer(colour, main_w) if (enabled and key == "diversion") else None
            if arrow is not None:
                layers.append(arrow)
                return QgsLineSymbol(layers)

            main = QgsSimpleLineSymbolLayer(colour)
            main.setWidth(main_w if enabled else max(0.4, main_w * 0.6))
            main.setPenCapStyle(Qt.PenCapStyle.RoundCap)
            main.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
            if not enabled:
                main.setPenStyle(Qt.PenStyle.DashLine)
            elif key == "diversion":
                main.setPenStyle(Qt.PenStyle.DashDotLine)
            elif key == "berm":
                main.setPenStyle(Qt.PenStyle.DashLine)
            layers.append(main)

            if enabled and key == "dam":
                # Embankment/barrier look: short white dashes across the wall
                # (marker-free, so it always renders).
                hatch = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 235))
                hatch.setWidth(max(0.6, main_w * 0.55))
                hatch.setPenCapStyle(Qt.PenCapStyle.FlatCap)
                try:
                    hatch.setUseCustomDashPattern(True)
                    hatch.setCustomDashVector([1.4, 2.6])  # dash, gap (mm)
                except Exception:
                    hatch.setPenStyle(Qt.PenStyle.DotLine)
                layers.append(hatch)

            return QgsLineSymbol(layers)
        except Exception:
            return QgsLineSymbol.createSimple(
                {"color": colour.name(), "width": str(main_w),
                 "capstyle": "round", "joinstyle": "round"}
            )

    def _arrow_line_layer(self, colour, main_w):
        """Flow-direction ribbon along a line — repeated arrowheads pointing in the
        drawn (downhill) direction, via QgsArrowSymbolLayer. Returns None on failure."""
        from qgis.core import QgsArrowSymbolLayer, QgsFillSymbol
        try:
            arrow = QgsArrowSymbolLayer()
            shaft = max(0.7, main_w * 0.55)
            for name, val in (
                ("setArrowWidth", shaft),
                ("setArrowStartWidth", shaft),
                ("setArrowHeadLength", 3.4),
                ("setArrowHeadThickness", 3.4),
            ):
                if hasattr(arrow, name):
                    getattr(arrow, name)(val)
            if hasattr(arrow, "setIsRepeated"):
                arrow.setIsRepeated(True)   # multiple arrowheads down the line
            fill = QgsFillSymbol.createSimple(
                {"color": colour.name(), "outline_style": "no"}
            )
            arrow.setSubSymbol(fill)
            return arrow
        except Exception:
            return None

    def _refresh_ew_layer(self):
        self._ensure_ew_layers()
        for layer in self._state.ew_layers.values():
            if layer and self._project.instance().mapLayer(layer.id()):
                layer.dataProvider().truncate()

        for ew in self._state.earthwork_manager.get_all():
            layer = self._state.ew_layers.get(ew.type)
            if not layer or not self._project.instance().mapLayer(layer.id()):
                continue
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(ew.geometry.asWkt()))
            f.setAttributes([ew.name, ew.type, ew.capacity_m3,
                              1 if getattr(ew, "enabled", True) else 0])
            layer.dataProvider().addFeature(f)

        for layer in self._state.ew_layers.values():
            if layer and self._project.instance().mapLayer(layer.id()):
                layer.triggerRepaint()

    # ---------------------------------------------------------------- Earthworks analysis

    def run_with_earthworks(self):
        if not self._state.dem_path or not self._state.burner:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM and run baseline first."
            )
            return

        enabled = self._state.earthwork_manager.get_enabled()
        if not enabled:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "No enabled earthworks to re-analyse with."
            )
            return

        modified_dem = self._state.burner.burn_earthworks(enabled)
        # Surface Strategy-C honesty warnings (sub-cell features, resolution-cap
        # degrade) so a 1-cell routing approximation is never silent.
        for msg in getattr(self._state.burner, "warnings", []):
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)
        mod_path = os.path.join(self._state.output_dir, "modified_dem.tif")
        self._state.burner.save(modified_dem, mod_path)
        self._state.modified_dem_path = mod_path

        cell_area_m2 = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
        threshold_cells = int(
            self._panel.stream_threshold_ha * 10_000 / cell_area_m2
        ) if cell_area_m2 > 0 else 1000

        self._state.analysis_worker = AnalysisWorker(
            dem_path=mod_path,
            output_dir=self._state.output_dir,
            stream_threshold=threshold_cells,
            cn=self._panel.cn,
            moisture=self._panel.moisture,
            rainfall_mm=self._panel.rainfall_mm,
            duration_hours=self._panel.duration_hr,
            boundary_path=self._state.boundary_path,
            label="earthworks",
            run_catchments=False,
            routing=self._panel.routing,
        )
        self._state.analysis_worker.progress.connect(self._panel.set_earthworks_progress)
        self._state.analysis_worker.finished.connect(self._on_earthworks_complete)
        self._state.analysis_worker.error.connect(self._on_analysis_error)
        self._state.analysis_worker.start()

    def _on_earthworks_complete(self, result):
        self._state.earthworks_result = result
        # Delegate layer loading to baseline controller's helper via the shared project
        from terrainflow_assessment.qgis.controllers.baseline import BaselineController
        bl = BaselineController(self._state, self._panel, self._project,
                                self._iface, self._canvas)
        bl._load_result_layers(result, is_earthworks=True)
        self._load_burned_dem_layer()
        if result.get("ponding"):
            self._state.ponding_raster_path = result["ponding"]

        # Non-circular check: terrain-derived ponding vs analytic capacity (§4).
        self._state.verification = self._compute_verification()
        msg = "Earthworks analysis complete. Toggle 'Show: with earthworks' to compare."
        v = self._state.verification
        if v is not None:
            msg += (
                f"\nTerrain-derived storage {v.terrain_total_m3:,.0f} m³ vs analytic "
                f"{v.analytic_total_m3:,.0f} m³ (Δ {v.delta_pct:+.0f}%)."
            )
        self._panel.set_earthworks_complete(msg)

        # The design is now verified against a burn — reset the drift counter.
        self._state.edits_since_verify = 0
        self._state.verified_delta_pct = v.delta_pct if v is not None else None
        self._update_verified_chip()

    def _compute_verification(self):
        """Reconcile terrain-derived ponding (burned DEM) against analytic capacity (§4).

        Returns a reporting.VerificationResult, or None if the ponding raster or any
        storage feature is unavailable. Pure maths lives in modules/reporting.py; this
        only reads the rasters + builds per-feature footprint masks.
        """
        import json

        import numpy as np
        import rasterio
        from rasterio.features import rasterize as _rasterize
        from shapely.geometry import shape as _shp

        from terrainflow_assessment.modules.reporting import (
            attribute_ponding_volume,
            build_verification,
            raster_ponding_volume,
        )

        ew_result = self._state.earthworks_result or {}
        ew_pond_path = ew_result.get("ponding")
        if not ew_pond_path or not os.path.exists(ew_pond_path):
            return None

        try:
            with rasterio.open(ew_pond_path) as src:
                ew_pond = src.read(1).astype("float64")
                nodata = src.nodata
                transform = src.transform
                shape = ew_pond.shape
            cell_area = abs(transform.a * transform.e)
            cell_size = abs(transform.a)
            if nodata is not None:
                ew_pond[ew_pond == nodata] = 0.0
            ew_pond = np.clip(ew_pond, 0.0, None)
        except Exception:
            return None

        # Baseline ponding (absent → zeros; only used if it aligns to the same grid).
        bl_pond = np.zeros(shape, dtype="float64")
        bl_pond_path = (self._state.baseline_result or {}).get("ponding")
        if bl_pond_path and os.path.exists(bl_pond_path):
            try:
                with rasterio.open(bl_pond_path) as src:
                    arr = src.read(1).astype("float64")
                    bnd = src.nodata
                if arr.shape == shape:
                    if bnd is not None:
                        arr[arr == bnd] = 0.0
                    bl_pond = np.clip(arr, 0.0, None)
            except Exception:
                pass

        diff = np.clip(ew_pond - bl_pond, 0.0, None)

        analytic_by_name = {}
        min_dims = {}
        footprints = []
        for ew in self._state.earthwork_manager.get_enabled():
            if getattr(ew, "capacity_m3", 0.0) <= 0:
                continue
            analytic_by_name[ew.name] = ew.capacity_m3
            # Sub-cell check keys off the channel bottom width; polygons never sub-cell.
            min_dims[ew.name] = (
                getattr(ew, "bottom_width_m", None) if ew.type == "swale" else None
            )
            try:
                geom = _shp(json.loads(ew.geometry.asJson()))
                if geom.geom_type in ("LineString", "MultiLineString"):
                    geom = geom.buffer(max(getattr(ew, "width", 2.0) / 2.0, cell_size))
                mask = _rasterize(
                    [(geom, 1)], out_shape=shape, transform=transform,
                    fill=0, dtype="uint8",
                ).astype(bool)
                footprints.append((ew.name, mask))
            except Exception:
                footprints.append((ew.name, np.zeros(shape, dtype=bool)))

        if not analytic_by_name:
            return None

        terrain_by_name, unattributed = attribute_ponding_volume(diff, cell_area, footprints)
        baseline_total = raster_ponding_volume(bl_pond, cell_area)
        earthworks_total = raster_ponding_volume(ew_pond, cell_area)

        result = build_verification(
            analytic_by_name, terrain_by_name, baseline_total, earthworks_total,
            min_dims, cell_size,
        )
        result.unattributed_m3 = unattributed
        return result

    def _load_burned_dem_layer(self):
        """Add the burned (Strategy-C) DEM to the layer panel so the carve/ridge is visible.

        Placed at the bottom of the layer tree (it is a backdrop, not a result overlay)
        and registered under the earthworks layer group so it shows/hides with the
        'with earthworks' toggle. Silently skips if the burn produced no valid raster.
        """
        path = self._state.modified_dem_path
        if not path or not os.path.exists(path):
            return
        from qgis.core import QgsRasterLayer
        layer = QgsRasterLayer(path, "Earthworks — Burned DEM")
        if not layer.isValid():
            return
        project = self._project.instance()
        project.addMapLayer(layer, False)          # don't auto-add to legend top
        project.layerTreeRoot().addLayer(layer)     # append at the bottom instead
        ids = list(getattr(self._state, "earthworks_layer_ids", None) or [])
        ids.append(layer.id())
        self._state.earthworks_layer_ids = ids

    def _on_analysis_error(self, tb):
        self._panel.set_earthworks_complete("Analysis failed — see Python console for details.")
        print("TerrainFlow Assessment — Analysis error:\n" + tb)
        self._iface.messageBar().pushCritical("TerrainFlow Assessment",
                                               "Analysis failed. See Python console.")

    # ---------------------------------------------------------------- Ponding query

    def activate_ponding_query(self):
        if not self._state.ponding_raster_path:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Run baseline analysis first."
            )
            return
        tool = PondingQueryTool(self._canvas, self._state.ponding_raster_path)
        tool.ponding_selected.connect(self._on_ponding_selected)
        tool.no_ponding.connect(self._on_no_ponding)
        self._canvas.setMapTool(tool)

    def _on_ponding_selected(self, volume_m3, volume_l, cell_count, area_m2,
                              outline_geom, inflow_m3, fill_fraction):
        from qgis.PyQt.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

        dlg = QDialog(self._iface.mainWindow())
        dlg.setWindowTitle("Depression / Ponding Results")
        dlg.setMinimumWidth(340)
        layout = QVBoxLayout(dlg)

        def _row(bold_text, value_text, tooltip=None):
            lbl = QLabel(f"<b>{bold_text}</b>  {value_text}")
            if tooltip:
                lbl.setToolTip(tooltip)
            layout.addWidget(lbl)

        _row("Volume held:",
             f"{volume_m3:,.1f} m³  ({volume_l:,.0f} L)",
             "Total water volume in the connected depression area.")

        area_ha = area_m2 / 10_000
        area_str = f"{area_ha:.2f} ha  ({area_m2:,.0f} m²)" if area_ha >= 0.1 else f"{area_m2:,.0f} m²"
        _row("Surface area:", area_str)

        if area_m2 > 0:
            _row("Mean depth:", f"{volume_m3 / area_m2:.2f} m")

        _row("Cell count:", f"{cell_count:,} cells")

        if fill_fraction >= 0:
            fill_pct = fill_fraction * 100.0
            if fill_fraction <= 1.0:
                fill_str = f"{fill_pct:.0f}% of design-storm inflow captured  ✓"
                colour = "#006600"
            else:
                fill_str = f"{fill_pct:.0f}% of design-storm inflow — depression overflows  ⚠"
                colour = "#cc4400"
            lbl_fill = QLabel(
                f"<b>Storm fill:</b>  <span style=\"color:{colour}\">{fill_str}</span>"
            )
            lbl_fill.setToolTip(
                "Ratio of design-storm inflow volume to depression capacity.\n"
                "Below 100%: depression absorbs the full storm event.\n"
                "Above 100%: overflow will occur — consider enlarging the earthwork."
            )
            layout.addWidget(lbl_fill)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _on_no_ponding(self):
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "No ponding at that location — click a blue zone in the Water Captured layer.",
        )

    # ---------------------------------------------------------------- Slope visualisation

    def toggle_slope_class(self, checked):
        if not self._state.slope_raster_path or not os.path.exists(self._state.slope_raster_path):
            return
        lid = self._state.slope_class_layer_id
        if lid:
            node = self._project.instance().layerTreeRoot().findLayer(lid)
            if node:
                node.setItemVisibilityChecked(checked)
                self._canvas.refresh()
                return
        layer = QgsRasterLayer(self._state.slope_raster_path, "Slope Classification")
        if layer.isValid():
            self._apply_slope_class_ramp(layer)
            layer.setOpacity(0.6)
            self._project.instance().addMapLayer(layer)
            self._state.slope_class_layer_id = layer.id()
            node = self._project.instance().layerTreeRoot().findLayer(layer)
            if node:
                node.setItemVisibilityChecked(checked)
            self._canvas.refresh()

    def _apply_slope_class_ramp(self, layer):
        shader = QgsColorRampShader()
        shader.setColorRampType(QgsColorRampShader.Interpolated)
        shader.setColorRampItemList([
            QgsColorRampShader.ColorRampItem(3,  QColor("#50C850"), "0–3°"),
            QgsColorRampShader.ColorRampItem(8,  QColor("#DCDC1E"), "3–8°"),
            QgsColorRampShader.ColorRampItem(13, QColor("#FFA500"), "8–13°"),
            QgsColorRampShader.ColorRampItem(18, QColor("#FF6600"), "13–18°"),
            QgsColorRampShader.ColorRampItem(25, QColor("#CC2200"), "18–25°"),
            QgsColorRampShader.ColorRampItem(90, QColor("#660000"), ">25°"),
        ])
        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, raster_shader)
        layer.setRenderer(renderer)

    def toggle_slope_arrows(self, checked):
        if checked:
            existing = (self._state.slope_arrows_layer_id and
                        self._project.instance().mapLayer(self._state.slope_arrows_layer_id))
            if existing:
                node = self._project.instance().layerTreeRoot().findLayer(
                    self._state.slope_arrows_layer_id)
                if node:
                    node.setItemVisibilityChecked(True)
            else:
                self._generate_slope_arrows()
        else:
            if self._state.slope_arrows_layer_id:
                node = self._project.instance().layerTreeRoot().findLayer(
                    self._state.slope_arrows_layer_id)
                if node:
                    node.setItemVisibilityChecked(False)
            self._canvas.refresh()

    def _generate_slope_arrows(self):
        if not self._state.dem_path:
            return
        try:
            import processing
            import rasterio

            result = processing.run("gdal:aspect", {
                "INPUT": self._state.dem_path,
                "BAND": 1,
                "TRIG_ANGLE": False,
                "ZERO_FOR_FLAT": True,
                "COMPUTE_EDGES": True,
                "ZEVENBERGEN": False,
                "OUTPUT": "TEMPORARY_OUTPUT",
            })
            aspect_path = result["OUTPUT"]
            if hasattr(aspect_path, "source"):
                aspect_path = aspect_path.source()

            with rasterio.open(self._state.dem_path) as src:
                cell_size = abs(src.transform.a)
                transform = src.transform

            with rasterio.open(aspect_path) as src:
                aspect = src.read(1).astype("float32")
                rows, cols = aspect.shape

            step = max(1, int(50.0 / cell_size))

            layer = QgsVectorLayer("Point", "Slope Direction", "memory")
            layer.setCrs(self._project.instance().crs())
            pr = layer.dataProvider()
            pr.addAttributes([QgsField("angle", QMetaType.Double)])
            layer.updateFields()

            features = []
            for r in range(0, rows, step):
                for c in range(0, cols, step):
                    val = float(aspect[r, c])
                    if val < 0 or val > 360:
                        continue
                    x = transform.c + c * transform.a + cell_size / 2
                    y = transform.f + r * transform.e + cell_size / 2
                    f = QgsFeature()
                    from qgis.core import QgsGeometry, QgsPointXY
                    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
                    f.setAttributes([val])
                    features.append(f)

            pr.addFeatures(features)
            layer.updateExtents()

            symbol = QgsMarkerSymbol.createSimple({
                "name": "arrow", "color": "60,60,200,200",
                "outline_color": "20,20,120,200", "size": "5", "angle": "0",
            })
            symbol.symbolLayer(0).setDataDefinedProperty(
                QgsSymbolLayer.PropertyAngle, QgsProperty.fromField("angle"))
            layer.setRenderer(QgsSingleSymbolRenderer(symbol))

            self._project.instance().addMapLayer(layer)
            self._state.slope_arrows_layer_id = layer.id()
            self._canvas.refresh()

        except Exception as exc:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not generate slope direction arrows: {exc}"
            )
