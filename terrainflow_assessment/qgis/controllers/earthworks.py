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
    QgsPointXY,
    QgsProperty,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsSymbolLayer,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType, Qt
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
from terrainflow_assessment.modules.swale_design import contour_to_swale_geometry
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers import _symbols as S
from terrainflow_assessment.qgis.controllers._layers import resolve_layer
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker


class EarthworksController(G.LayerTreeMixin):
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    # ---------------------------------------------------------------- Drawing tools

    def activate_draw_swale(self, mode):
        from terrainflow_assessment.qgis.controllers._layers import resolve_layer
        contour_layer = resolve_layer(self._project, self._state.contour_layer_id)
        if mode == "contour":
            if contour_layer is None:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", self._no_contour_layer_message("segment"))
                return
            tool = ContourSegmentTool(self._canvas, contour_layer)
            tool.segment_selected.connect(
                lambda geom, elev, coords: self._on_contour_selected_for_swale(
                    geom, elev, coords
                )
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self._canvas.setMapTool(tool)
        elif mode == "full_contour":
            if contour_layer is None:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", self._no_contour_layer_message("contour"))
                return
            tool = SelectContourTool(self._canvas, contour_layer)
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

    def _no_contour_layer_message(self, what):
        """Say which precondition is missing, not just that one is.

        "Run contour analysis first" is unhelpful to someone who has just run the
        *baseline* analysis — they are different buttons on different stages, and a
        deleted layer looks identical from here unless the two cases are separated.
        """
        if getattr(self._state, "contour_layer_id", None):
            return (f"The contour layer has been removed from the project, so there "
                    f"is nothing to pick a {what} from. Re-run Contour Analysis on "
                    f"the Analysis stage.")
        return (f"Run Contour Analysis on the Analysis stage first, then pick a "
                f"{what}. The Baseline run does not generate contours on its own.")

    def _on_contour_selected_for_swale(self, geom, elevation, contour_coords=None):
        swale_geom = contour_to_swale_geometry(geom)
        self._on_geometry_drawn("swale", swale_geom, source_contour=contour_coords)

    def create_swale_from_keyline(self):
        """Convert the current master keyline (generated or drawn) into a swale,
        reusing the contour-swale path so the swale stays reshape-locked to the
        keyline (source_contour provenance)."""
        geom = getattr(self._state, "keyline_master_geom", None)
        coords = getattr(self._state, "keyline_master_coords", None)
        if geom is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Generate or draw a keyline first, then convert it to a swale.",
            )
            return
        self._on_contour_selected_for_swale(geom, None, coords)

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

    # ---------------------------------------------------------------- Spillways

    def activate_place_spillway(self, kind="outflow", index=None):
        """Site an earthwork's spillway by clicking the map.

        The crest is a design decision the dialog owns; *where* it sits is a
        decision about ground, so it is made on the ground. Placing it also pins the
        crest to a sampled elevation rather than a typed one.

        *index* names the feature explicitly, which is how the Spillways list drives
        this — it has its own rows and should not have to reach through the flow
        network's selection to say which one it means. Left None the panel selection is
        used, as the tool-menu rows have always done.
        """
        if index is None:
            index = self._panel.get_selected_earthwork_index()
        idx = index
        if idx is None or not (0 <= idx < len(self._state.earthwork_manager)):
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Select an earthwork in the list first, then place its spillway.",
            )
            return
        ew = self._state.earthwork_manager.get(idx)
        if ew.type not in self.SPILLWAY_TYPES:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{ew.name} does not hold water, so it has nothing to spill.",
            )
            return

        from terrainflow_assessment.map_tools.place_point_tool import PlacePointTool

        kind = "inflow" if kind == "inflow" else "outflow"
        tool = PlacePointTool(
            self._canvas,
            snap_raster_path=self._state.dem_path,
            constrain_to=self._spillway_constraint(ew),
        )
        tool.point_placed.connect(
            lambda pt, elev, ew_id=ew.id, k=kind:
                self._on_spillway_placed(ew_id, pt, elev, kind=k))
        tool.rejected.connect(
            lambda dist, name=ew.name, k=kind: self._on_spillway_rejected(dist, name, k))
        tool.cancelled.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)
        prompt = ("Click where water ENTERS {name} from upslope."
                  if kind == "inflow" else
                  "Click where {name} should OVERFLOW.")
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            prompt.format(name=ew.name) + " Esc to cancel.",
        )

    # Beyond this, a recorded point is not describing a place on the feature —
    # it predates constrained placement, or the feature has since been reshaped
    # somewhere else entirely.
    _SILL_ORPHAN_TOLERANCE_M = 5.0

    def _spillway_sill(self, ew, point_geom, width_m):
        """The crest bar for a spillway: a segment of the built width, lying
        square across the feature at the recorded point.

        Returns ``(QgsGeometry, bearing_rad)``, or ``None`` when the point is not
        on the feature — which is a statement worth making rather than papering
        over, since it means the crest was sized from the wrong ground.
        """
        from terrainflow_assessment.modules.plan_geometry import perpendicular_sill

        target = self._spillway_constraint(ew)
        if target is None:
            return None
        try:
            pt = point_geom.asPoint()
            distance_sq, closest, after, _side = target.closestSegmentWithContext(pt)
            if closest is None:
                return None
            if float(distance_sq) ** 0.5 > self._SILL_ORPHAN_TOLERANCE_M:
                return None

            polyline = target.asPolyline()
            if not polyline:
                parts = target.asMultiPolyline()
                polyline = parts[0] if parts else []
            if len(polyline) < 2:
                return None

            # closestSegmentWithContext reports the vertex *after* the segment.
            result = perpendicular_sill(
                polyline, max(0, int(after) - 1),
                (closest.x(), closest.y()), width_m,
            )
            if result is None:
                return None
            (x1, y1), (x2, y2), bearing = result
            geom = QgsGeometry.fromPolylineXY(
                [QgsPointXY(x1, y1), QgsPointXY(x2, y2)])
            return geom, bearing
        except Exception:
            return None

    @staticmethod
    def _spillway_constraint(ew):
        """The geometry a spillway for *ew* must sit on.

        For a basin that is the rim, not the floor: a spillway is a notch in the
        edge the water leaves over, and a point in the middle of the polygon is
        not a place you can build one.
        """
        geom = getattr(ew, "geometry", None)
        if geom is None or geom.isEmpty():
            return None
        try:
            if geom.type() == QgsWkbTypes.PolygonGeometry:
                boundary = QgsGeometry(geom.constGet().boundary())
                return boundary if not boundary.isEmpty() else geom
        except Exception:
            pass
        return geom

    def _on_spillway_rejected(self, distance_m, name, kind):
        """Refuse an off-feature click, and say so in the words the ask used."""
        label = "Inflow" if kind == "inflow" else "Outflow"
        self._iface.messageBar().pushWarning(
            "TerrainFlow Assessment",
            f"{label} spillway must be placed on the feature selected — {name}. "
            f"That click was {distance_m:.1f} m away. Click on the drawn feature.",
        )

    def place_spillway_for(self, index, kind="outflow"):
        """Slot for the Spillways list, which names the feature by row."""
        self.activate_place_spillway(kind=kind, index=index)

    def _on_spillway_placed(self, ew_id, point, elevation, kind="outflow"):
        """Record the placed location, and seed the crest from the ground there."""
        from qgis.core import QgsGeometry

        from terrainflow_assessment.modules.earthwork_design import Spillway, bind_crest

        ew = next((e for e in self._state.earthwork_manager.get_all()
                   if e.id == ew_id), None)
        self._canvas.unsetMapTool(self._canvas.mapTool())
        if ew is None:
            return

        attr = "inflow_spillway" if kind == "inflow" else "spillway"
        spillway = getattr(ew, attr, None) or Spillway()
        spillway.point_wkt = QgsGeometry.fromPointXY(point).asWkt()

        # A crest already chosen by hand is left alone — placing the point tells us
        # where, not how deep. Only an auto crest follows the ground it landed on.
        if elevation is not None and (spillway.crest_elevation is None or spillway.auto):
            rim, _invert = self._spillway_datums(
                ew.geometry, ew.type,
                top_width_m=getattr(ew, "top_width_m", None),
                depth=getattr(ew, "depth", None),
                crest_elevation=getattr(ew, "crest_elevation", None),
            )
            crest, drop = bind_crest(rim, crest=float(elevation))
            spillway.crest_elevation = crest
            spillway.drop_below_rim_m = drop
        setattr(ew, attr, spillway)

        self._refresh_spillway_layer()
        self._recompute_live_assessment()
        self._mark_design_edit()
        where = ("" if spillway.crest_elevation is None
                 else f" — {spillway.crest_elevation:.2f} m")
        label = "inflow" if kind == "inflow" else "outflow"
        self._iface.messageBar().pushSuccess(
            "TerrainFlow Assessment", f"{ew.name} {label} spillway placed{where}.")

    def _refresh_spillway_layer(self):
        """Point layer of every placed spillway, labelled with crest and width."""
        from terrainflow_assessment.qgis.controllers._layers import remove_layer

        remove_layer(self._project, self._state.spillway_layer_id)
        self._state.spillway_layer_id = None

        placed = []
        for ew in self._state.earthwork_manager.get_all():
            for attr, kind in (("spillway", "outflow"), ("inflow_spillway", "inflow")):
                sp = getattr(ew, attr, None)
                if sp is not None and sp.point_wkt:
                    placed.append((ew, sp, kind))
        if not placed:
            return
        try:
            crs = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
            layer = QgsVectorLayer(f"LineString?crs={crs}", "Spillways", "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name", QMetaType.QString),
                QgsField("label", QMetaType.QString),
                QgsField("crest_m", QMetaType.Double),
                QgsField("width_m", QMetaType.Double),
                QgsField("kind", QMetaType.QString),
                QgsField("bearing", QMetaType.Double),
            ])
            layer.updateFields()

            feats = []
            for ew, sp, kind in placed:
                point = QgsGeometry.fromWkt(sp.point_wkt)
                if point is None or point.isEmpty():
                    continue
                crest = sp.crest_elevation
                width = sp.width_m or 0.0
                sill = self._spillway_sill(ew, point, width)
                if sill is None:
                    # The recorded point is nowhere near its feature — an older
                    # design placed before placement was constrained. Drawing a
                    # crest bar across unrelated ground would assert a structure
                    # that does not exist, so draw nothing and say why.
                    self._iface.messageBar().pushWarning(
                        "TerrainFlow Assessment",
                        f"{ew.name}'s {kind} spillway is not on the feature and was "
                        "not drawn. Place it again to fix it.",
                    )
                    continue
                geom, bearing = sill
                # The parent earthwork now carries its own name label, so repeating
                # it here just stacks two labels on the same spot (three, where an
                # inflow and an outflow sit close together).
                bits = [kind]
                if crest is not None:
                    bits.append(f"{crest:.2f} m")
                if kind == "outflow" and width > 0:
                    bits.append(f"{width:.1f} m")
                f = QgsFeature()
                f.setGeometry(geom)
                f.setAttributes([
                    ew.name, " · ".join(bits),
                    float(crest) if crest is not None else None, float(width), kind,
                    float(bearing),
                ])
                feats.append(f)
            if not feats:
                return
            pr.addFeatures(feats)
            layer.updateExtents()

            layer.renderer().setSymbol(S.spillway_symbol())

            settings = S.point_label_settings(
                "label",
                S.label_format(QColor(30, 60, 90), size_pt=8.5),
                priority=S.PRIORITY_SPILLWAY,
                max_scale=S.MAX_SCALE_POINT_LABEL,
            )
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            layer.setLabelsEnabled(True)

            self.place(layer, G.DRAWN)
            self._state.spillway_layer_id = layer.id()
            self.restack(S.DRAW_ORDER, G.DRAWN)
        except Exception as exc:
            print(f"TerrainFlow Assessment — spillway layer error: {exc}")

    # ---------------------------------------------------------------- Connections

    def activate_connect_earthworks(self):
        """Route one feature's overflow into another by clicking source then target."""
        features = [(ew.id, ew.name, ew.geometry)
                    for ew in self._state.earthwork_manager.get_all() if ew.enabled]
        if len(features) < 2:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Draw at least two earthworks before connecting them.",
            )
            return

        from terrainflow_assessment.map_tools.connect_earthworks_tool import (
            ConnectEarthworksTool,
        )

        tool = ConnectEarthworksTool(self._canvas, features)
        tool.connection_made.connect(self.on_connection_made)
        tool.source_picked.connect(
            lambda name: self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{name} overflows into… click the receiving feature. Esc to undo.",
            )
        )
        tool.cancelled.connect(self._on_draw_cancelled)
        self._canvas.setMapTool(tool)
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "Click the feature that overflows, then the one it flows into.",
        )

    def on_connection_made(self, from_id, to_id):
        """Accept a user overflow link unless it closes a loop.

        A cycle is not merely unsimulatable — it is physically impossible, since
        water cannot overflow back into something already overflowing into it. It is
        refused here rather than silently demoted downstream, so the refusal names
        the loop the user just drew.
        """
        from terrainflow_assessment.modules.flow_graph import topological_order

        by_id = {ew.id: ew for ew in self._state.earthwork_manager.get_all()}
        src, tgt = by_id.get(from_id), by_id.get(to_id)
        if src is None or tgt is None:
            return

        edges = {ew.id: ew.overflow_target_id
                 for ew in self._state.earthwork_manager.get_all()
                 if ew.overflow_target_id}
        edges[from_id] = to_id
        _order, broken = topological_order(edges)
        if broken:
            names = " → ".join(
                by_id[i].name for i in broken if i in by_id) or "these features"
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"{src.name} → {tgt.name} would create a loop ({names}). "
                "Water cannot overflow back into what is feeding it.",
            )
            return

        src.overflow_target_id = to_id
        self._recompute_live_assessment()
        self._mark_design_edit()
        self._iface.messageBar().pushSuccess(
            "TerrainFlow Assessment",
            f"{src.name} now overflows into {tgt.name}.",
        )

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

        # Direct contributing catchment for the not-yet-added feature: label the site
        # as if it were already there, so the dialog opens with real numbers.
        peak_inflow, catchment_m2 = self._provisional_catchment(ew_type, geometry)

        crest_elev = None
        if ew_type == "dam" and self._state.dem_path:
            crest_elev = self._default_crest_elevation(geometry)

        n = len(self._state.earthwork_manager) + 1
        ew_name = f"{ew_type.capitalize()} {n}"
        ew = Earthwork(ew_type, geometry, ew_name)
        ew.source_contour_coords = source_contour  # reshape stays contour-locked

        rim, invert = self._spillway_datums(
            geometry, ew_type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=crest_elev,
        )
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
            catchment_m2=catchment_m2,
            count_infiltration=self._panel.count_infiltration,
            rim_elevation=rim,
            invert_elevation=invert,
            peak_flow_m3s=self._provisional_peak_flow(catchment_m2),
            harvesting_coefficient=self._using_harvesting_coefficient(),
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
            ew.soil_name = getattr(dlg, "get_soil_name", lambda: None)()
            # The dialog is modal and cannot site a spillway, so it round-trips the
            # location the map tool set. Read it back regardless — until now the
            # dialog computed a spillway width, displayed it, and dropped it on OK.
            ew.spillway = getattr(dlg, "get_spillway", lambda: None)()
            # Apply the bottom width (channels only; None otherwise) — the canonical
            # cross-section field that drives capacity and the burn footprint.
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw

            if ew_type == "dam":
                # Key into the banks first: capacity must be flooded against the wall
                # that will actually be built, not the shorter line as drawn.
                if getattr(ew, "key_into_banks", False):
                    self._key_dam_into_banks(ew)
                    geometry = ew.geometry
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
            self._refresh_spillway_layer()
            self.recompute_catchments()
            self._recompute_live_assessment()
            self._mark_design_edit()
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def _on_draw_cancelled(self):
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def edit_earthwork_at(self, index):
        """Open the properties dialog for a feature named by row.

        Used by the Spillways list, which has its own rows and should not have to reach
        through the flow network's selection to say which feature it means.
        """
        self.edit_selected_earthwork(index=index)

    def edit_selected_earthwork(self, *, index=None):
        """Open the properties dialog for the selected feature, or for *index*.

        *index* is keyword-only on purpose: this is wired straight to the Edit button's
        ``clicked`` signal, which emits a ``checked`` bool. Positionally that would bind
        to the index and quietly edit feature 0 instead of the selected one.
        """
        idx = self._panel.get_selected_earthwork_index() if index is None else index
        if idx is None or not (0 <= idx < len(self._state.earthwork_manager)):
            return
        ew = self._state.earthwork_manager.get(idx)
        from terrainflow_assessment.earthwork_properties_dialog import EarthworkPropertiesDialog
        rim, invert = self._spillway_datums(
            ew.geometry, ew.type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=getattr(ew, "crest_elevation", None),
        )
        peak_total, peak_upstream = self._peak_flow_for(ew)
        edit_profile = self.feature_inflow_profile(ew)
        edit_station, edit_surplus = self.feature_overtopping(ew, edit_profile)
        dlg = EarthworkPropertiesDialog(
            ew_type=ew.type,
            geometry=ew.geometry,
            parent=self._iface.mainWindow(),
            earthwork=ew,
            # Was omitted, which hid the sizing block AND the whole Spillway group
            # whenever an existing feature was reopened. Free from the cached counts.
            peak_inflow_m3=self.feature_inflow_m3(ew) or None,
            catchment_m2=self.feature_catchment_m2(ew) or None,
            duration_hours=self._panel.duration_hr,
            dem_path=self._state.dem_path if ew.type == "dam" else None,
            soil_name=self._panel.earthwork_soil_name,
            cn=self._panel.cn,
            overflow_options=self._overflow_options(exclude_id=ew.id),
            own_elevation=self._feature_elevation(ew.geometry),
            count_infiltration=self._panel.count_infiltration,
            rim_elevation=rim,
            invert_elevation=invert,
            peak_flow_m3s=peak_total,
            upstream_flow_m3s=peak_upstream,
            harvesting_coefficient=self._using_harvesting_coefficient(),
            inflow_profile=edit_profile,
            overtop_station=edit_station,
            overtop_surplus=edit_surplus,
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
            ew.soil_name = getattr(dlg, "get_soil_name", lambda: None)()
            # The dialog is modal and cannot site a spillway, so it round-trips the
            # location the map tool set. Read it back regardless — until now the
            # dialog computed a spillway width, displayed it, and dropped it on OK.
            ew.spillway = getattr(dlg, "get_spillway", lambda: None)()
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw
            if ew.type == "dam":
                if getattr(ew, "key_into_banks", False):
                    self._key_dam_into_banks(ew)
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
            self._refresh_spillway_layer()
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
        self._refresh_spillway_layer()   # else the deleted feature's marker lingers
        self.recompute_catchments()
        self._recompute_live_assessment()
        self._mark_design_edit()

    def toggle_selected_earthwork(self):
        idx = self._panel.get_selected_earthwork_index()
        if idx is None:
            return
        self._state.earthwork_manager.toggle(idx)
        self._panel.refresh_earthwork_list(self._state.earthwork_manager.get_all())
        self.recompute_catchments()
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
        self._recompute_live_assessment(geometry_settled=False)

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
        self._resnap_spillways(ew)
        self._panel.update_earthwork_in_list(idx, ew.summary())
        self._refresh_ew_layer()
        self._refresh_spillway_layer()
        self.recompute_catchments()
        self._recompute_live_assessment()
        self._mark_design_edit()

    def _resnap_spillways(self, ew):
        """Keep a spillway on its feature after the feature has been reshaped.

        Dragging a vertex moves the alignment out from under any spillway on it,
        and for a diversion drain _orient_downhill can reverse the vertex order —
        which flips the local tangent, and with it the sill's perpendicular and
        the chevron's direction. Neither shows up until someone looks closely at
        a design they have already signed off.

        Only the point is moved, and only onto the nearest place on the new
        alignment; crest, width and freeboard are design decisions and stay put.
        """
        target = self._spillway_constraint(ew)
        if target is None:
            return
        for attr in ("spillway", "inflow_spillway"):
            sp = getattr(ew, attr, None)
            if sp is None or not sp.point_wkt:
                continue
            try:
                geom = QgsGeometry.fromWkt(sp.point_wkt)
                if geom is None or geom.isEmpty():
                    continue
                _dist_sq, closest, _after, _side = \
                    target.closestSegmentWithContext(geom.asPoint())
                if closest is not None:
                    sp.point_wkt = QgsGeometry.fromPointXY(
                        QgsPointXY(closest)).asWkt()
            except Exception:
                continue

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
            self._panel.set_verified_chip(
                f"Verified{extra}", True,
                tooltip=self.verification_sentence(self._state.verification),
            )
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

    def _natural_ponding_m3(self):
        """Water the bare terrain already holds, from the baseline ponding raster.

        Context for the scorecard, not an input to it: the headline scores the design
        only, so runoff that never reaches an earthwork is counted as leaving the site
        whether or not it would settle in a hollow first. Returns 0.0 when there is no
        baseline raster to read — the caller hides the line rather than guessing.
        """
        path = (self._state.baseline_result or {}).get("ponding")
        if not path or not os.path.exists(path):
            return 0.0
        try:
            import numpy as np
            import rasterio

            from terrainflow_assessment.modules.reporting import raster_ponding_volume
            with rasterio.open(path) as src:
                arr = src.read(1).astype("float64")
                nodata = src.nodata
                cell_area = abs(src.transform.a * src.transform.e)
            if nodata is not None:
                arr[arr == nodata] = 0.0
            return float(raster_ponding_volume(np.clip(arr, 0.0, None), cell_area))
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- Live analytical assessment

    # ---------------------------------------------------------------- Flow-graph cache

    # ---------------------------------------------------------------- Persistence

    _PROJECT_SCOPE = "TerrainFlow"
    _PROJECT_KEY = "earthworks"

    def save_to_project(self):
        """Write the current design into the QGIS project file.

        Without this the whole design was lost on restart: earthworks lived only in
        an in-memory list, and the map layers mirrored 5 of ~16 fields, so even a
        saved project could not reconstruct them.
        """
        try:
            manager = self._state.earthwork_manager
            if manager is None:
                return
            self._project.instance().writeEntry(
                self._PROJECT_SCOPE, self._PROJECT_KEY, manager.to_json())
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not save earthworks: {exc}")

    def load_from_project(self):
        """Restore the design stored by :meth:`save_to_project`."""
        try:
            text, ok = self._project.instance().readEntry(
                self._PROJECT_SCOPE, self._PROJECT_KEY, "")
            if not ok or not text:
                return 0
            return self.restore_earthworks_from_json(text, source="the project")
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not load earthworks: {exc}")
            return 0

    def restore_earthworks_from_json(self, text, source=None):
        """Replace the design with the one in *text* and bring the UI back in step.

        Shared by the QGIS-project hook and the portable design file so both restore
        paths refresh exactly the same things — a design that renders but never
        re-labels, or re-labels but never re-scores, is the failure mode this avoids.

        *source* names where the design came from, for the confirmation message; pass
        ``None`` when the caller reports success itself.
        """
        try:
            manager = self._state.earthwork_manager
            if manager is None:
                return 0

            n = manager.from_json(text)
            if not n:
                return 0

            # Each refresh stands alone. Run as one block, a failure in the table refresh
            # skipped the map layers entirely — the design came back listed and scored but
            # invisible on the canvas, which reads as "the earthworks did not load" even
            # though they had. Partial UI beats a design that is present but undrawable.
            #
            # recompute_catchments and the live assessment are both safe before a baseline
            # exists: labelling needs the flow graph and quietly does nothing without it,
            # and the assessment has a no-flow branch that reports geometry only.
            for step in (
                lambda: self._panel.refresh_earthwork_list(manager.get_all()),
                self._refresh_ew_layer,
                self._refresh_spillway_layer,
                self.recompute_catchments,
                self._recompute_live_assessment,
            ):
                try:
                    step()
                except Exception as exc:
                    print(f"TerrainFlow Assessment — restore step failed: {exc}")

            if source:
                self._iface.messageBar().pushInfo(
                    "TerrainFlow Assessment",
                    f"Restored {n} earthwork{'s' if n != 1 else ''} from {source}.",
                )
            return n
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not restore earthworks: {exc}")
            return 0

    def _ensure_flow_graph(self):
        """Build (once per DEM) the steepest-descent pointers over the conditioned DEM.

        Cached because it costs ~0.4 s on a 285 ha 1 m site and depends only on the
        terrain — not on the storm, and not on the earthworks.
        """
        if self._state.flow_next is not None:
            return True
        baseline = self._state.baseline_result or {}
        cond_path = baseline.get("conditioned_dem")
        if not cond_path or not os.path.exists(cond_path):
            return False

        try:
            import numpy as np
            import rasterio

            from terrainflow_assessment.modules.flow_graph import d8_from_dem

            with rasterio.open(cond_path) as src:
                dem = src.read(1).astype("float32")
                transform = src.transform
                nodata = src.nodata
            cell_w, cell_h = abs(transform.a), abs(transform.e)

            next_flat, is_sink = d8_from_dem(dem, cell_w, cell_h, nodata=nodata)

            domain = None
            dom_path = baseline.get("domain_mask")
            if dom_path and os.path.exists(dom_path):
                with rasterio.open(dom_path) as src:
                    domain = src.read(1) > 0.5
            if domain is None or domain.shape != dem.shape:
                domain = np.ones(dem.shape, dtype=bool)

            self._state.flow_next = next_flat
            self._state.flow_sink = is_sink
            self._state.flow_dem = dem
            self._state.flow_domain_mask = domain
            self._state.flow_grid_meta = {
                "shape": dem.shape,
                "transform": transform,
                "cell_area_m2": cell_w * cell_h,
                "cell_size_m": (cell_w + cell_h) / 2.0,
                "nodata": nodata,
            }
            return True
        except Exception as exc:
            print(f"TerrainFlow Assessment — flow graph error: {exc}")
            return False

    def recompute_catchments(self):
        """Label every site cell with the earthwork that first intercepts its runoff.

        Storm-independent, so this only runs when geometry changes — a storm-slider
        change re-scores from the cached cell counts in milliseconds. Measured at
        ~0.3 s for 2.9 M cells, which is why it runs inline rather than on a thread.
        """
        self._state.invalidate_catchment_cache()
        if not self._ensure_flow_graph():
            return False

        try:
            import numpy as np

            from terrainflow_assessment.modules.flow_graph import (
                LABEL_NONE,
                label_direct_catchments,
            )
            from terrainflow_assessment.modules.footprint import (
                outlet_cell,
                rasterize_footprint,
            )

            meta = self._state.flow_grid_meta
            shape, transform = meta["shape"], meta["transform"]
            dem = self._state.flow_dem

            interceptors = np.full(shape, LABEL_NONE, dtype=np.int32)
            label_ids, outlets = [], {}

            for ew in self._state.earthwork_manager.get_all():
                if not ew.enabled:
                    continue
                shp = self._shapely_of(ew)
                if shp is None:
                    continue
                # Lines are conveyances with real width — buffer to their footprint so
                # a swale intercepts the cells it actually crosses, not a 1-cell thread.
                if shp.geom_type in ("LineString", "MultiLineString"):
                    shp = shp.buffer(max(getattr(ew, "top_width_m", 1.0), 0.1) / 2.0)
                mask = rasterize_footprint(shp, shape, transform)
                if not mask.any():
                    continue
                label = len(label_ids)
                interceptors[mask] = label
                label_ids.append(ew.id)
                rc = outlet_cell(dem, mask, nodata=meta.get("nodata"))
                if rc is not None:
                    outlets[ew.id] = int(rc[0]) * shape[1] + int(rc[1])

            res = label_direct_catchments(
                self._state.flow_next, interceptors, self._state.flow_domain_mask,
                is_sink=self._state.flow_sink,
            )

            self._state.catchment_labels = res.labels
            self._state.catchment_label_ids = label_ids
            self._state.catchment_counts = {
                ew_id: int(res.counts[i]) if i < len(res.counts) else 0
                for i, ew_id in enumerate(label_ids)
            }
            self._state.catchment_exit_cells = res.exit_cells
            self._state.catchment_sink_cells = res.sink_cells
            self._state.catchment_outlets = outlets
            if not res.is_exhaustive():
                print("TerrainFlow Assessment — catchment labelling did not close: "
                      f"{res.accounted} of {res.domain_cells} cells accounted for.")
            return True
        except Exception as exc:
            print(f"TerrainFlow Assessment — catchment labelling error: {exc}")
            self._state.invalidate_catchment_cache()
            return False

    # ---------------------------------------------------------------- Dam geometry

    _DAM_MAX_EXTEND_M = 250.0

    def _elevation_sampler(self):
        """Return ``f(x, y) -> elevation or None`` over the source DEM, or None."""
        if not self._state.dem_path:
            return None
        try:
            import rasterio
            src = rasterio.open(self._state.dem_path)
        except Exception:
            return None

        band = src.read(1)
        t, nodata = src.transform, src.nodata
        rows, cols = band.shape
        src.close()

        def _at(x, y):
            col = int((x - t.c) / t.a)
            row = int((y - t.f) / t.e)
            if not (0 <= row < rows and 0 <= col < cols):
                return None
            v = float(band[row, col])
            if nodata is not None and v == nodata:
                return None
            return v if v == v else None       # NaN → unknown

        return _at

    def _default_crest_elevation(self, geometry):
        """Crest for a freshly drawn dam: the **highest ground the line touches**.

        Previously this sampled the line's centroid and added 2 m. A dam's centroid
        sits in the middle of the valley — its lowest point — so the default crest was
        the valley floor plus 2 m: a low bump that water simply flowed around, however
        the wall was drawn. Using the high point instead means the wall rises from the
        valley floor to the level of its higher abutment, which is what "lock the
        height at the highest point" should mean.
        """
        sample = self._elevation_sampler()
        shp = None
        try:
            import json

            from shapely.geometry import shape as shapely_shape
            shp = shapely_shape(json.loads(geometry.asJson()))
        except Exception:
            return None
        if sample is None or shp is None:
            return None

        try:
            step = self._state.dem_info.cell_size_m if self._state.dem_info else 1.0
            step = max(step, 0.5)
            n = max(2, int(shp.length / step) + 1)
            elevs = []
            for i in range(n + 1):
                pt = shp.interpolate(shp.length * i / n)
                e = sample(pt.x, pt.y)
                if e is not None:
                    elevs.append(e)
            return max(elevs) if elevs else None
        except Exception:
            return None

    def _key_dam_into_banks(self, ew):
        """Extend a dam wall at each end until the ground reaches its crest.

        A wall that stops short of high ground does not impound — water goes round the
        end. This walks outward from both endpoints until the terrain rises to the
        crest, replacing the drawn geometry with the wall that would actually have to
        be built. Any end that finds no high ground within
        ``_DAM_MAX_EXTEND_M`` is reported, because the design does not work as drawn.
        """
        from qgis.core import QgsGeometry, QgsPointXY

        from terrainflow_assessment.modules.earthwork_design import (
            abutment_warning,
            extend_to_abutments,
        )

        if ew.type != "dam" or ew.crest_elevation is None:
            return False
        sample = self._elevation_sampler()
        if sample is None:
            return False

        shp = self._shapely_of(ew)
        if shp is None or shp.geom_type != "LineString":
            return False

        step = self._state.dem_info.cell_size_m if self._state.dem_info else 1.0
        coords, info = extend_to_abutments(
            list(shp.coords), sample, ew.crest_elevation,
            max_extend_m=self._DAM_MAX_EXTEND_M, step_m=max(step, 0.5),
        )
        added = info["start_m"] + info["end_m"]
        if added > 0:
            ew.geometry = QgsGeometry.fromPolylineXY(
                [QgsPointXY(x, y) for x, y in coords])
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{ew.name}: wall extended {info['start_m']:.0f} m and "
                f"{info['end_m']:.0f} m at its ends to key into the banks at "
                f"{ew.crest_elevation:.2f} m.",
            )
        warn = abutment_warning(ew.name, info, ew.crest_elevation,
                                self._DAM_MAX_EXTEND_M)
        if warn:
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", warn)
        return added > 0

    # ---------------------------------------------------------------- Peak flows

    def _peak_runoff_fraction(self):
        """Instantaneous runoff fraction for the panel's basis — the rational ``C``."""
        from terrainflow_assessment.modules.catchment import SCSRunoff
        from terrainflow_assessment.modules.peak_flow import peak_runoff_fraction

        scs = SCSRunoff()
        return peak_runoff_fraction(
            self._panel.sizing_basis,
            rainfall_mm=self._panel.rainfall_mm,
            coefficient=self._panel.runoff_coefficient,
            cn=scs.adjust_cn(self._panel.cn, self._panel.moisture),
        )

    def recompute_peak_flows(self, routing=None):
        """Per-feature peak design flow (m³/s), with upstream overflow cascaded in.

        Cached on the state as ``{id: (total, from_upstream)}`` so the properties
        dialog and the "now too small" check read the same numbers. Cheap — the
        catchment cell counts already exist, and the basis enters as one scalar.
        """
        from terrainflow_assessment.modules.flow_graph import topological_order
        from terrainflow_assessment.modules.peak_flow import (
            rational_peak_flow,
            upstream_contributions,
        )

        self._state.peak_flows = {}
        counts = self._state.catchment_counts or {}
        meta = self._state.flow_grid_meta
        if not counts or meta is None:
            return {}

        fraction = self._peak_runoff_fraction()
        intensity = self._panel.peak_intensity_mm_hr
        cell_area = meta["cell_area_m2"]

        direct = {
            ew_id: rational_peak_flow(fraction, intensity, n * cell_area)
            for ew_id, n in counts.items()
        }
        if routing is not None and getattr(routing, "edges", None):
            edges = dict(routing.edges)
        else:
            edges = {ew.id: ew.overflow_target_id
                     for ew in self._state.earthwork_manager.get_all()
                     if ew.overflow_target_id}
        for ew_id in direct:
            edges.setdefault(ew_id, None)

        order, _broken = topological_order(edges)
        self._state.peak_flows = upstream_contributions(direct, edges, order)
        return self._state.peak_flows

    def feature_time_of_concentration(self, ew, channel_length_m=None,
                                      hydraulic_radius_m=None):
        """TR-55 time of concentration for *ew*'s catchment, or None.

        Computed lazily rather than on every live re-assessment: the longest-path
        traversal costs ~45 ms on a 2.8 ha catchment and ~570 ms on 50 ha, which is
        fine on demand and would be felt during a vertex drag.

        Both slope and channel length come from the *traced* path rather than from
        catchment-wide averages. Each leg takes its slope from the actual elevation
        profile, because hillslopes are concave and a single average is far too gentle
        where sheet flow happens — worth about 25% of Tc, in the direction that
        undersizes the overflow. Channel length is likewise measured, at the point
        where contributing area first exceeds the channel threshold.
        """
        from terrainflow_assessment.modules.flow_graph import longest_flow_path
        from terrainflow_assessment.modules.time_of_concentration import (
            channel_length_from_area,
            profile_leg_slopes,
            split_flow_path,
            time_of_concentration,
        )

        labels = self._state.catchment_labels
        meta = self._state.flow_grid_meta
        label_ids = self._state.catchment_label_ids
        if labels is None or meta is None or ew.id not in (label_ids or []):
            return None
        try:
            import numpy as np

            index = list(label_ids).index(ew.id)
            mask = labels == index
            # Travel time is time to *reach* the feature, so the path must stop at its
            # edge. The catchment deliberately includes the footprint (rain landing in
            # a swale does drain into it, which is right for volume), but counting
            # travel *within* it is wrong — a level pool shares water along its whole
            # length the moment any arrives. Measured at 5% of Tc on a 30 m basin, and
            # in the direction that undersizes the overflow.
            own = self._footprint_mask(ew.geometry, getattr(ew, "top_width_m", None))
            if own is not None:
                trimmed = mask & ~own
                if trimmed.any():
                    mask = trimmed
            if not mask.any():
                return None

            cols = meta["shape"][1]
            cell_w = cell_h = meta["cell_size_m"]
            transform = meta.get("transform")
            if transform is not None:
                cell_w, cell_h = abs(transform.a), abs(transform.e)

            length_m, _visited, path = longest_flow_path(
                self._state.flow_next, mask, cols, cell_w, cell_h, trace=True)
            if length_m <= 0 or len(path) < 2:
                return None

            distances, elevations = self._path_profile(path, cols, cell_w, cell_h)

            if channel_length_m is None:
                channel_length_m = channel_length_from_area(
                    distances, self._path_upstream_areas(path, meta))

            slopes = profile_leg_slopes(distances, elevations,
                                        channel_length_m=channel_length_m)
            sheet, shallow, channel = split_flow_path(
                length_m, slopes, channel_length_m=channel_length_m)

            # The channel leg may run through the user's own drains, whose sections
            # are known exactly, as well as natural ground where they are not. Build
            # it as sub-segments rather than averaging that away.
            channel_hours, runs, radius = self._channel_legs(
                path, distances, elevations, channel_length_m,
                override_radius_m=hydraulic_radius_m)

            tc = time_of_concentration(
                sheet, shallow, channel, p2_mm=self._sheet_flow_p2_mm(),
                channel_hours=channel_hours,
            )
            tc.flow_path_m = length_m
            tc.slope = (float(np.mean([s for s in slopes if s > 0]))
                        if any(s > 0 for s in slopes) else 0.0)
            tc.leg_slopes = slopes
            tc.channel_length_m = channel_length_m
            tc.hydraulic_radius_m = radius
            tc.channel_runs = runs
            return tc
        except Exception as exc:
            print(f"TerrainFlow Assessment — time of concentration error: {exc}")
            return None

    def _channel_legs(self, path, distances, elevations, channel_length_m,
                      override_radius_m=None):
        """Channel travel time over sub-segments — ``(hours, runs, representative R)``.

        Where the path crosses a swale or diversion the radius is computed from that
        feature's entered dimensions; elsewhere it is the default farm-drain preset.
        An explicit *override_radius_m* replaces the lot, for a user who knows better
        than either.
        """
        from terrainflow_assessment.modules.time_of_concentration import (
            DEFAULT_CHANNEL_RADIUS_M,
            DEFAULT_CHANNEL_ROUGHNESS,
            mixed_channel_hours,
        )

        if channel_length_m <= 0 or len(path) < 2:
            return None, [], (override_radius_m or DEFAULT_CHANNEL_RADIUS_M)

        total = distances[-1]
        start = next((i for i, d in enumerate(distances)
                      if d >= total - channel_length_m), 0)
        leg_path = path[start:]
        leg_d = [d - distances[start] for d in distances[start:]]
        leg_z = elevations[start:]
        if len(leg_path) < 2:
            return None, [], (override_radius_m or DEFAULT_CHANNEL_RADIUS_M)

        if override_radius_m is not None:
            radii = [float(override_radius_m)] * len(leg_path)
            roughs = [DEFAULT_CHANNEL_ROUGHNESS] * len(leg_path)
        else:
            sections = self._channel_sections_along(leg_path)
            radii = [s[0] if s else DEFAULT_CHANNEL_RADIUS_M for s in sections]
            roughs = [s[1] if s else DEFAULT_CHANNEL_ROUGHNESS for s in sections]

        hours, runs = mixed_channel_hours(leg_d, radii, elevations_m=leg_z,
                                          roughnesses=roughs)
        # Report the section carrying the most length, as the single headline figure.
        representative = DEFAULT_CHANNEL_RADIUS_M
        if runs:
            representative = max(runs, key=lambda r: r[0])[2]
        return hours, runs, representative

    def _channel_sections_along(self, leg_path):
        """``(radius, roughness)`` per cell, from any swale/diversion it falls inside.

        Only those two registry types carry a cross-section — a berm or dam is not
        conveyance, and a basin is not a channel. ``None`` means natural ground.
        """
        from terrainflow_assessment.modules.time_of_concentration import (
            EARTHWORK_CHANNEL_ROUGHNESS,
            section_hydraulic_radius,
        )

        sections = []
        for ew in self._state.earthwork_manager.get_all():
            if not ew.enabled or ew.type not in ("swale", "diversion"):
                continue
            radius = section_hydraulic_radius(
                getattr(ew, "top_width_m", None),
                getattr(ew, "bottom_width_m", None),
                getattr(ew, "depth", None))
            if radius is None:
                continue
            mask = self._footprint_mask(ew.geometry, getattr(ew, "top_width_m", None))
            if mask is None:
                continue
            sections.append((mask.ravel(), radius))

        if not sections:
            return [None] * len(leg_path)

        out = []
        for cell in leg_path:
            hit = next((r for m, r in sections if m[cell]), None)
            out.append((hit, EARTHWORK_CHANNEL_ROUGHNESS) if hit else None)
        return out

    def _path_profile(self, path, cols, cell_w, cell_h):
        """``(cumulative distances, elevations)`` along a traced flow path."""
        import math

        dem_flat = self._state.flow_dem.ravel()
        distances, elevations = [0.0], [float(dem_flat[path[0]])]
        diag = math.hypot(cell_w, cell_h)
        for prev, cur in zip(path, path[1:]):
            dr = abs(cur // cols - prev // cols)
            dc = abs(cur % cols - prev % cols)
            step = diag if (dr == 1 and dc == 1) else (cell_h if dr == 1 else cell_w)
            distances.append(distances[-1] + step)
            elevations.append(float(dem_flat[cur]))
        return distances, elevations

    def _path_upstream_areas(self, path, meta):
        """Contributing area (m²) at each cell of the path, or None without one.

        Read from the baseline's flow accumulation, which is already on disk — the
        channel transition is then measured rather than assumed.
        """
        try:
            import rasterio

            acc_path = (self._state.baseline_result or {}).get("flow_accumulation")
            if not acc_path or not os.path.exists(acc_path):
                return None
            with rasterio.open(acc_path) as src:
                acc = src.read(1)
            if acc.shape != tuple(meta["shape"]):
                return None
            flat = acc.ravel()
            cell_area = meta["cell_area_m2"]
            return [float(flat[i]) * cell_area for i in path]
        except Exception:
            return None

    def _sheet_flow_p2_mm(self):
        """2-year 24-hour depth from the entered rainfall data, or None."""
        table = getattr(self._state, "idf_table", None)
        return table.sheet_flow_p2_mm() if table is not None else None

    def edit_rainfall_data(self):
        """Open the depth-duration-frequency entry dialog and store the result."""
        from terrainflow_assessment.modules.rainfall_idf import IDFTable
        from terrainflow_assessment.rainfall_data_dialog import RainfallDataDialog

        current = getattr(self._state, "idf_table", None) or IDFTable()
        dlg = RainfallDataDialog(parent=self._iface.mainWindow(), table=current)
        if not dlg.exec():
            return
        self._state.idf_table = dlg.get_table()
        self.save_rainfall_data()
        if self._state.idf_table.has_data():
            aris = ", ".join(f"{a} yr" for a in self._state.idf_table.available_aris())
            self._iface.messageBar().pushSuccess(
                "TerrainFlow Assessment",
                f"Rainfall data stored ({aris}). Use Compare on the Baseline tab to "
                f"read an intensity at your catchment's response time.",
            )

    _RAINFALL_KEY = "idf_table"

    def save_rainfall_data(self):
        try:
            table = getattr(self._state, "idf_table", None)
            self._project.instance().writeEntry(
                self._PROJECT_SCOPE, self._RAINFALL_KEY,
                table.to_json() if table is not None else "")
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not save rainfall data: {exc}")

    def load_rainfall_data(self):
        try:
            text, ok = self._project.instance().readEntry(
                self._PROJECT_SCOPE, self._RAINFALL_KEY, "")
            self.restore_rainfall_from_json(text if ok else "")
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not load rainfall data: {exc}")

    def restore_rainfall_from_json(self, text):
        """Set the IDF table from *text*, or to an empty table when there is none.

        Shared by the project hook and the design file. An empty table rather than None
        because the depth lookups treat "no data entered" as a real state and handle it —
        leaving None would make them fail instead.
        """
        from terrainflow_assessment.modules.rainfall_idf import IDFTable
        try:
            self._state.idf_table = IDFTable.from_json(text) if text else IDFTable()
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not restore rainfall data: {exc}")
            self._state.idf_table = IDFTable()

    def _provisional_peak_flow(self, catchment_m2):
        """Peak flow for a feature being drawn but not yet in the network.

        Own catchment only — it has no routing yet, so nothing spills into it. Returns
        None when there is no catchment to work from, which leaves the dialog showing
        "set a peak intensity" rather than a confident zero.
        """
        from terrainflow_assessment.modules.peak_flow import rational_peak_flow

        if not catchment_m2:
            return None
        return rational_peak_flow(
            self._peak_runoff_fraction(),
            self._panel.peak_intensity_mm_hr,
            catchment_m2,
        )

    def _peak_flow_for(self, ew):
        """``(total_m3s, upstream_m3s)`` for one feature, or ``(None, 0.0)``."""
        flows = getattr(self._state, "peak_flows", None) or {}
        if ew.id not in flows:
            return (None, 0.0)
        own, upstream = flows[ew.id]
        return (own + upstream, upstream)

    def _using_harvesting_coefficient(self):
        from terrainflow_assessment.modules.peak_flow import (
            coefficient_is_harvesting_grade,
        )
        return coefficient_is_harvesting_grade(
            self._panel.sizing_basis, self._panel.runoff_coefficient)

    SPILLWAY_TYPES = ("swale", "dam", "basin")

    def _refresh_auto_spillway_widths(self):
        """Keep every auto-tracking spillway width on its requirement.

        ``width_auto`` promised this and never delivered it: the width was recomputed
        only inside the modal properties dialog, so the stored figure — the one on the
        map label, in the feature summary and in any review of the design — went stale
        the moment the dialog closed and another feature was drawn. Worse,
        :meth:`_check_spillway_capacity` skips auto widths on the grounds that they
        track by definition, which was true of the intention and false of the object.

        Outflows only. An inlet is a protected entry, not a weir; giving it a width from
        the overflow formula would invent a design procedure that does not exist.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            calculate_spillway_width,
            spillway_policy,
        )

        for ew in self._state.earthwork_manager.get_all():
            spillway = getattr(ew, "spillway", None)
            if spillway is None or not spillway.width_auto:
                continue
            total, _upstream = self._peak_flow_for(ew)
            if not total or total <= 0:
                continue
            head = spillway.head_m or spillway_policy(ew.type)[1]
            required = calculate_spillway_width(total, head)
            # 0.0 means "these inputs say nothing" — no intensity, no catchment — not a
            # spillway zero metres wide. Writing it through would persist that fiction.
            if required > 0:
                spillway.width_m = required

    def _spillway_row(self, ew):
        """One review row for *ew*, or None if this type has nothing to spill.

        Built for every water-holding feature, **including those with no ``Spillway``
        object yet**: the required width is a function of the terrain and the storm, not
        of whether the user has ticked a box, and showing it before they commit is the
        whole of "features auto-size their spillways as you add them".
        """
        from terrainflow_assessment.modules.earthwork_design import (
            calculate_spillway_width,
            effective_freeboard_m,
            effective_head_m,
            spillway_policy,
            spillway_validity,
        )

        if ew.type not in self.SPILLWAY_TYPES:
            return None

        spillway = getattr(ew, "spillway", None)
        inlet = getattr(ew, "inflow_spillway", None)
        policy_freeboard, policy_head, head_band = spillway_policy(ew.type)

        freeboard = effective_freeboard_m(spillway, ew.type)
        target_head = (spillway.head_m if spillway is not None and spillway.head_m
                       else policy_head)
        width_auto = True if spillway is None else bool(spillway.width_auto)

        row = {
            "index": None,                    # filled by the caller; see _build_spillway_rows
            "id": ew.id,
            "name": ew.name,
            "ew_type": ew.type,
            "enabled": bool(ew.enabled),
            "designed": spillway is not None,
            "sited": bool(spillway is not None and spillway.point_wkt),
            "inlet_sited": bool(inlet is not None and inlet.point_wkt),
            "width_auto": width_auto,
            "target_head_m": target_head,
            "freeboard_min_m": freeboard,
            "standard_freeboard_m": policy_freeboard,
            "peak_flow_m3s": None,
            "upstream_m3s": 0.0,
            "required_width_m": None,
            "built_width_m": None if spillway is None else (spillway.width_m or 0.0),
            "actual_head_m": None,
            "freeboard_m": None,
            "crest_elevation": None if spillway is None else spillway.crest_elevation,
            "rim_elevation": None,
            "problems": [],
            "state": "ok",
        }

        # A disabled feature is out of the catchment labelling entirely, so it has no
        # flow and no meaningful sizing — say so rather than render a row of zeros that
        # reads as a failure.
        if not ew.enabled:
            row["state"] = "disabled"
            return row

        total, upstream = self._peak_flow_for(ew)
        row["peak_flow_m3s"], row["upstream_m3s"] = total, upstream
        if not total or total <= 0:
            row["state"] = "no_flow"
            return row

        row["required_width_m"] = calculate_spillway_width(total, target_head) or None
        built = row["built_width_m"] if spillway is not None else row["required_width_m"]
        row["actual_head_m"] = effective_head_m(
            target_head, peak_flow_m3s=total, width_m=built, width_auto=width_auto)

        rim, invert = self._spillway_datums(
            ew.geometry, ew.type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=getattr(ew, "crest_elevation", None),
        )
        row["rim_elevation"] = rim
        if rim is not None and row["crest_elevation"] is not None:
            row["freeboard_m"] = rim - row["crest_elevation"] - (row["actual_head_m"] or 0.0)

        row["problems"] = spillway_validity(
            row["crest_elevation"], rim, invert_elevation=invert,
            head_m=row["actual_head_m"] or target_head,
            min_freeboard_m=freeboard,
            width_m=built, required_width_m=row["required_width_m"],
            standard_freeboard_m=policy_freeboard,
            typical_head_m=head_band,
            feature_length_m=getattr(ew, "length_m", None),
        )
        if rim is None:
            row["state"] = "no_datum"
        elif row["problems"]:
            row["state"] = "fail"
        elif not row["designed"]:
            row["state"] = "undesigned"
        elif not row["sited"]:
            row["state"] = "unsited"
        return row

    def _build_spillway_rows(self):
        """Refresh the Design-stage spillway review.

        Deliberately **not** called from :meth:`_recompute_live_assessment`: that runs on
        every frame of a vertex drag, and each row samples the DEM through
        ``_spillway_datums`` (a footprint rasterisation plus a rim scan). The same
        reasoning already keeps time-of-concentration off that path. Geometry-dependent
        work belongs on the discrete edits, where the geometry has actually settled.
        """
        try:
            rows = []
            for i, ew in enumerate(self._state.earthwork_manager.get_all()):
                row = self._spillway_row(ew)
                if row is not None:
                    row["index"] = i
                    rows.append(row)
            self._panel.set_spillway_review(rows, self._spillway_context())
        except Exception as exc:
            import traceback
            print(f"TerrainFlow Assessment — spillway review error: {exc}")
            traceback.print_exc()

    def _spillway_context(self):
        """Facts the review footer needs that are not per-row."""
        from terrainflow_assessment.modules.peak_flow import (
            DEFAULT_PEAK_INTENSITY_MM_HR,
        )

        idf = getattr(self._state, "idf_table", None)
        return {
            "intensity_mm_hr": self._panel.peak_intensity_mm_hr,
            "intensity_is_default": (
                abs(self._panel.peak_intensity_mm_hr - DEFAULT_PEAK_INTENSITY_MM_HR)
                < 1e-9),
            "has_idf": bool(idf is not None and getattr(idf, "depths", None)),
            "harvesting_coefficient": self._using_harvesting_coefficient(),
        }

    def _check_spillway_capacity(self):
        """Warn where a *built* spillway no longer passes its design flow.

        Fires after geometry or routing changes, because the usual way a sized
        spillway becomes undersized is that someone added a feature upslope and
        routed more through it — a change made somewhere else entirely, which the
        user has no reason to connect to a structure they finished last week.

        The Spillways list carries the same fact persistently; this stays because a list
        is something you have to look at. This is the channel that reaches someone who
        is not looking — so it only fires on the discrete edits, never per drag frame.
        Auto widths are skipped, and now genuinely do track (see
        :meth:`_refresh_auto_spillway_widths`), so the skip is finally true.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            calculate_spillway_width,
            head_for_width,
            spillway_policy,
        )

        short = []
        for ew in self._state.earthwork_manager.get_all():
            spillway = getattr(ew, "spillway", None)
            if spillway is None or spillway.width_auto:
                continue          # an auto width tracks the requirement by definition
            total, upstream = self._peak_flow_for(ew)
            if total is None or total <= 0:
                continue
            head = spillway.head_m or spillway_policy(ew.type)[1]
            required = calculate_spillway_width(total, head)
            built = spillway.width_m or 0.0
            if required > built + 0.01:
                short.append((ew, built, required, head,
                              head_for_width(total, built), upstream))

        if not short:
            return
        parts = []
        for ew, built, required, head, actual, upstream in short[:3]:
            # Lead with what the water does, not with the shortfall in metres: "two
            # metres short" is hard to act on, "it will run 18 cm deeper than you
            # designed for" is the same fact in the units that decide the outcome.
            if actual is not None:
                note = (f"{ew.name}: at {built:.2f} m the water runs {actual:.2f} m "
                        f"deep, not {head:.2f} m — needs {required:.2f} m")
            else:
                note = f"{ew.name}: built {built:.2f} m, now needs {required:.2f} m"
            if upstream > 0:
                note += f" ({upstream * 1000:,.0f} L/s of that arrives from upslope)"
            parts.append(note)
        more = "" if len(short) <= 3 else f" (+{len(short) - 3} more)"
        self._iface.messageBar().pushWarning(
            "TerrainFlow Assessment",
            "Spillway now undersized — " + "; ".join(parts) + more,
        )

    def choose_design_intensity(self):
        """Open the peak-intensity comparison, costed against a real catchment here.

        Uses the selected feature's catchment when there is one, else the largest on
        site, so the widths quoted are ones the user will actually meet rather than a
        worked example from a manual.
        """
        from terrainflow_assessment.design_intensity_dialog import DesignIntensityDialog
        from terrainflow_assessment.modules.catchment import SCSRunoff

        area_m2, label = 0.0, ""
        # Initialised out here on purpose: both are read below, and scoping them
        # inside the guard crashed the dialog with an UnboundLocalError whenever the
        # baseline had not been run — which is exactly when a user reaches for it.
        chosen = None
        by_id = {}
        counts = self._state.catchment_counts or {}
        meta = self._state.flow_grid_meta
        if counts and meta is not None:
            by_id = {ew.id: ew for ew in self._state.earthwork_manager.get_all()}
            idx = self._panel.get_selected_earthwork_index()
            if idx is not None:
                candidate = self._state.earthwork_manager.get(idx)
                if candidate.id in counts:
                    chosen = candidate.id
            if chosen is None:
                chosen = max(counts, key=counts.get)
            area_m2 = counts[chosen] * meta["cell_area_m2"]
            label = by_id[chosen].name if chosen in by_id else ""

        if area_m2 <= 0:
            # No earthworks yet — which is exactly when someone is choosing a design
            # intensity, since the number feeds the sizing they are about to do.
            # Costing against the whole site keeps every row a real figure instead of
            # a column of 0.0 L/s and 0.00 m that answers nothing.
            baseline = self._state.baseline_result or {}
            area_m2 = float(baseline.get("domain_area_m2") or 0.0)
            label = "whole site" if area_m2 > 0 else ""

        # Time of concentration for the same catchment the widths are costed against,
        # so the dialog can offer the duration the rational method actually asks for
        # rather than a list of guesses.
        tc = None
        if chosen is not None and chosen in by_id:
            tc = self.feature_time_of_concentration(by_id[chosen])

        scs = SCSRunoff()
        dlg = DesignIntensityDialog(
            parent=self._iface.mainWindow(),
            area_m2=area_m2,
            area_label=label,
            rainfall_mm=self._panel.rainfall_mm,
            duration_hr=self._panel.duration_hr,
            basis=self._panel.sizing_basis,
            coefficient=self._panel.runoff_coefficient,
            cn=scs.adjust_cn(self._panel.cn, self._panel.moisture),
            current_intensity=self._panel.peak_intensity_mm_hr,
            travel_time=tc,
            idf_table=getattr(self._state, "idf_table", None),
        )
        if dlg.exec():
            self._panel.set_peak_intensity(dlg.get_intensity())

    def compute_area_subtotals(self):
        """Break the live balance down by sub-catchment — your "smaller areas".

        Partitions the *same* cell-exact labelling the headline uses. Deliberately
        not built from ``area_outflow``: that is a 3x3 max-pooled boundary-crossing
        sum which cannot reconcile with a cell-exact balance, and two disagreeing
        "water leaving" numbers in one panel would be worse than one.
        """
        from terrainflow_assessment.modules.water_balance import area_subtotals

        labels = self._state.catchment_labels
        meta = self._state.flow_grid_meta
        domain = self._state.flow_domain_mask
        if labels is None or meta is None or domain is None:
            return []
        try:
            polygons = (self._state.baseline_result or {}).get("catchments")
            if not polygons:
                return []
            masks = {}
            for i, poly in enumerate(polygons):
                shp = self._shapely_of_polygon(poly)
                if shp is None:
                    continue
                mask = self._footprint_mask_from_shapely(shp)
                if mask is None or not mask.any():
                    continue
                label = (poly.get("label") if isinstance(poly, dict) else None)
                masks[label or f"Catchment {i + 1}"] = mask
            if not masks:
                return []
            return area_subtotals(labels, domain, masks,
                                  meta["cell_area_m2"], self._current_runoff_mm())
        except Exception as exc:
            print(f"TerrainFlow Assessment — area subtotals error: {exc}")
            return []

    def _shapely_of_polygon(self, poly):
        """Shapely geometry from a sub-catchment entry (WKT, mapping, or geometry)."""
        try:
            from shapely.geometry import shape as shapely_shape
            if hasattr(poly, "asJson"):
                import json
                return shapely_shape(json.loads(poly.asJson()))
            if isinstance(poly, dict):
                return shapely_shape(poly.get("geometry", poly))
            if isinstance(poly, str):
                from shapely import wkt
                return wkt.loads(poly)
            return poly if hasattr(poly, "geom_type") else None
        except Exception:
            return None

    def _footprint_mask_from_shapely(self, shp):
        meta = self._state.flow_grid_meta
        if meta is None or shp is None:
            return None
        try:
            from terrainflow_assessment.modules.footprint import rasterize_footprint
            return rasterize_footprint(shp, meta["shape"], meta["transform"])
        except Exception:
            return None

    # ---------------------------------------------------------------- Inflow profile

    _PROFILE_MAX_CELLS = 20_000

    def feature_inflow_profile(self, ew, runoff_mm=None):
        """Where along a linear feature its catchment actually arrives.

        The lumped "total capacity vs total inflow" verdict is only safe because a
        level swale shares water along its whole length. Where inflow is concentrated
        — a drainage line crossing the alignment — a swale with adequate *total*
        capacity can still overtop locally, and the headline number will not say so.

        Each catchment cell is attributed to its nearest point along the centreline.
        That is an approximation of where its flow path actually crosses: exact for a
        contour swale, where runoff runs perpendicular to the alignment, and looser as
        the line departs from the contour. Tracing every cell's path instead would be
        another full traversal for a refinement the profile's shape does not need.

        Returns the profile dict from :func:`swale_design.inflow_profile`, or None.
        """
        from terrainflow_assessment.modules.swale_design import inflow_profile

        labels = self._state.catchment_labels
        meta = self._state.flow_grid_meta
        label_ids = self._state.catchment_label_ids or []
        if labels is None or meta is None or ew.id not in label_ids:
            return None
        if ew.type not in ("swale", "diversion", "berm"):
            return None                      # a polygon has no alignment to profile
        try:
            import numpy as np

            line = self._shapely_of(ew)
            if line is None or line.length <= 0:
                return None

            mask = labels == list(label_ids).index(ew.id)
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                return None

            # Subsample very large catchments: the profile's shape is what matters,
            # and every retained cell is scaled up so the total stays exact.
            scale = 1.0
            if rows.size > self._PROFILE_MAX_CELLS:
                stride = int(np.ceil(rows.size / self._PROFILE_MAX_CELLS))
                rows, cols = rows[::stride], cols[::stride]
                scale = float(stride)

            transform = meta["transform"]
            xs = transform.c + (cols + 0.5) * transform.a
            ys = transform.f + (rows + 0.5) * transform.e

            from shapely.geometry import Point
            distances = [line.project(Point(float(x), float(y)))
                         for x, y in zip(xs, ys)]

            if runoff_mm is None:
                runoff_mm = self._current_runoff_mm()
            per_cell = meta["cell_area_m2"] * runoff_mm / 1000.0 * scale
            volumes = [per_cell] * len(distances)

            return inflow_profile(distances, volumes, line.length)
        except Exception as exc:
            print(f"TerrainFlow Assessment — inflow profile error: {exc}")
            return None

    def feature_overtopping(self, ew, profile=None):
        """``(station_m, surplus_m3)`` where a linear feature overtops, or (None, 0)."""
        from terrainflow_assessment.modules.swale_design import overtopping_station

        if profile is None:
            profile = self.feature_inflow_profile(ew)
        if not profile:
            return None, 0.0
        length = getattr(ew, "length_m", 0.0) or 0.0
        capacity = float(getattr(ew, "capacity_m3", 0.0) or 0.0)
        if length <= 0 or capacity <= 0:
            return None, 0.0
        # capacity_m3 already carries the freeboard allowance, so pass 1.0 rather than
        # discounting it twice.
        return overtopping_station(profile, capacity / length, freeboard=1.0)

    def refresh_stress_points_layer(self):
        """Mark where features overtop, on the map beside the problem.

        A station along an existing alignment, so this costs one interpolate per
        feature. It is the visible counterpart to the lumped headline: "adequate
        overall, but it goes over the side 40 m from the east end".
        """
        from terrainflow_assessment.qgis.controllers._layers import remove_layer

        remove_layer(self._project, self._state.stress_points_layer_id)
        self._state.stress_points_layer_id = None

        feats = []
        try:

            for ew in self._state.earthwork_manager.get_all():
                if not ew.enabled or ew.type not in ("swale", "diversion"):
                    continue
                profile = self.feature_inflow_profile(ew)
                station, surplus = self.feature_overtopping(ew, profile)
                if station is None:
                    continue
                point = ew.geometry.interpolate(float(station))
                if point is None or point.isEmpty():
                    continue
                f = QgsFeature()
                f.setGeometry(point)
                f.setAttributes([
                    ew.name, round(float(station), 1), round(float(surplus), 1),
                    f"{ew.name} · overtops at {station:,.0f} m · +{surplus:,.0f} m³",
                ])
                feats.append(f)

            if not feats:
                return

            crs = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
            layer = QgsVectorLayer(f"Point?crs={crs}", "Stress points", "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name", QMetaType.QString),
                QgsField("station_m", QMetaType.Double),
                QgsField("surplus_m3", QMetaType.Double),
                QgsField("label", QMetaType.QString),
            ])
            layer.updateFields()
            pr.addFeatures(feats)
            layer.updateExtents()

            layer.renderer().setSymbol(S.stress_symbol())

            settings = S.point_label_settings(
                "label",
                S.label_format(QColor(120, 70, 10), size_pt=8.5),
                priority=S.PRIORITY_STRESS,
                max_scale=S.MAX_SCALE_POINT_LABEL,
            )
            layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
            layer.setLabelsEnabled(True)

            self.place(layer, G.DRAWN)
            self._state.stress_points_layer_id = layer.id()
            self.restack(S.DRAW_ORDER, G.DRAWN)
        except Exception as exc:
            print(f"TerrainFlow Assessment — stress points layer error: {exc}")

    def _footprint_mask(self, geometry, top_width_m=None):
        """Rasterise a geometry onto the flow grid, buffering lines to their width."""
        meta = self._state.flow_grid_meta
        if meta is None:
            return None
        try:
            import json

            from shapely.geometry import shape as shapely_shape

            from terrainflow_assessment.modules.footprint import rasterize_footprint

            shp = shapely_shape(json.loads(geometry.asJson()))
            if shp.geom_type in ("LineString", "MultiLineString"):
                shp = shp.buffer(max(top_width_m or 1.0, 0.1) / 2.0)
            mask = rasterize_footprint(shp, meta["shape"], meta["transform"])
            return mask if mask.any() else None
        except Exception:
            return None

    def _spillway_datums(self, geometry, ew_type, top_width_m=None, depth=None,
                         crest_elevation=None):
        """``(rim, invert)`` for the spillway controls — the two levels a crest sits between.

        *rim* is the lowest containing ground: where the feature would spill if
        nothing were built. For a dam that is the wall crest, because the wall
        **is** the containment — using the natural pour point there would sample the
        valley floor the dam is holding back and give a rim below the design water
        level.

        *invert* is the floor. For a cut feature that is the level-invert datum the
        burn uses (pour point − depth), so the dialog and the DEM agree; for a dam
        it is the lowest ground the wall touches.
        """
        if not self._ensure_flow_graph():
            return (None, None)
        dem = self._state.flow_dem
        mask = self._footprint_mask(geometry, top_width_m)
        if dem is None or mask is None:
            return (None, None)
        try:
            from terrainflow_assessment.modules.footprint import pour_point

            natural_rim, _cell = pour_point(
                dem, mask, nodata=self._state.flow_grid_meta.get("nodata"))
            if natural_rim is None:
                return (None, None)
            natural_rim = float(natural_rim)

            if ew_type == "dam":
                floor = float(dem[mask].min())
                rim = float(crest_elevation) if crest_elevation is not None else natural_rim
                return (rim, floor)

            drop = float(depth) if depth else 0.0
            return (natural_rim, natural_rim - drop)
        except Exception:
            return (None, None)

    def _shapely_of(self, ew):
        """Shapely geometry for an earthwork, or None."""
        try:
            import json

            from shapely.geometry import shape as shapely_shape
            return shapely_shape(json.loads(ew.geometry.asJson()))
        except Exception:
            return None

    def _make_walker(self):
        """Return ``walker(store) -> target_id`` following the real flow path.

        Walks downslope from the feature's outlet cell until it enters another
        earthwork's footprint or leaves the site — replacing the old rule of "the
        highest feature below this one", which linked features across ridges.
        """
        labels = self._state.catchment_labels
        outlets = self._state.catchment_outlets
        label_ids = self._state.catchment_label_ids
        if labels is None or not label_ids:
            return None

        import numpy as np

        from terrainflow_assessment.modules.flow_graph import LABEL_NONE, walk_downslope

        # The catchment raster says who *receives* a cell; for the walk we need who
        # *occupies* it, so rebuild the interceptor footprints from the label ids.
        shape = self._state.flow_grid_meta["shape"]
        interceptors = np.full(shape, LABEL_NONE, dtype=np.int32)
        from terrainflow_assessment.modules.footprint import rasterize_footprint
        transform = self._state.flow_grid_meta["transform"]
        index_of = {}
        for ew in self._state.earthwork_manager.get_all():
            if ew.id not in outlets and ew.id not in label_ids:
                continue
            if not ew.enabled:
                continue
            shp = self._shapely_of(ew)
            if shp is None:
                continue
            if shp.geom_type in ("LineString", "MultiLineString"):
                shp = shp.buffer(max(getattr(ew, "top_width_m", 1.0), 0.1) / 2.0)
            mask = rasterize_footprint(shp, shape, transform)
            if not mask.any():
                continue
            idx = len(index_of)
            interceptors[mask] = idx
            index_of[ew.id] = idx
        by_index = {v: k for k, v in index_of.items()}
        inter_flat = interceptors.ravel()
        next_flat = self._state.flow_next

        def walker(store):
            start = outlets.get(store.id)
            if start is None:
                return None
            label, _ = walk_downslope(
                next_flat, start, inter_flat,
                skip_label=index_of.get(store.id, LABEL_NONE),
            )
            return by_index.get(label) if label is not None else None

        return walker

    def feature_inflow_m3(self, ew, runoff_mm=None):
        """Event runoff arriving at *ew* from its own direct catchment (m³).

        Read straight from the cached cell counts — no raster sampling — so the
        properties dialog can show it on both the create and the edit path.
        """
        counts = self._state.catchment_counts or {}
        meta = self._state.flow_grid_meta
        if not counts or meta is None or ew.id not in counts:
            return 0.0
        if runoff_mm is None:
            runoff_mm = self._current_runoff_mm()
        return counts[ew.id] * meta["cell_area_m2"] * runoff_mm / 1000.0

    def _provisional_catchment(self, ew_type, geometry, top_width_m=None):
        """Direct catchment of a feature that is being drawn but not yet added.

        Labels the site with the existing earthworks *plus* this candidate, so the
        properties dialog opens with the inflow the feature will actually receive —
        already net of whatever upslope features intercept first. Returns
        ``(inflow_m3, catchment_m2)``, or ``(0.0, 0.0)`` without a baseline.
        """
        if not self._ensure_flow_graph():
            return 0.0, 0.0
        try:
            import json

            import numpy as np
            from shapely.geometry import shape as shapely_shape

            from terrainflow_assessment.modules.flow_graph import (
                LABEL_NONE,
                label_direct_catchments,
            )
            from terrainflow_assessment.modules.footprint import rasterize_footprint

            meta = self._state.flow_grid_meta
            shape, transform = meta["shape"], meta["transform"]
            interceptors = np.full(shape, LABEL_NONE, dtype=np.int32)

            def _foot(shp, width):
                if shp.geom_type in ("LineString", "MultiLineString"):
                    shp = shp.buffer(max(width or 1.0, 0.1) / 2.0)
                return rasterize_footprint(shp, shape, transform)

            label = 0
            for ew in self._state.earthwork_manager.get_all():
                if not ew.enabled:
                    continue
                shp = self._shapely_of(ew)
                if shp is None:
                    continue
                mask = _foot(shp, getattr(ew, "top_width_m", 1.0))
                if mask.any():
                    interceptors[mask] = label
                    label += 1

            candidate = shapely_shape(json.loads(geometry.asJson()))
            if top_width_m is None:
                try:
                    top_width_m = get_type(ew_type).default_top_width
                except KeyError:
                    top_width_m = 1.0
            cand_mask = _foot(candidate, top_width_m)
            if not cand_mask.any():
                return 0.0, 0.0
            interceptors[cand_mask] = label

            res = label_direct_catchments(
                self._state.flow_next, interceptors, self._state.flow_domain_mask,
                is_sink=self._state.flow_sink,
            )
            cells = int(res.counts[label]) if label < len(res.counts) else 0
            area = cells * meta["cell_area_m2"]
            return area * self._current_runoff_mm() / 1000.0, area
        except Exception as exc:
            print(f"TerrainFlow Assessment — provisional catchment error: {exc}")
            return 0.0, 0.0

    def feature_catchment_m2(self, ew):
        """Direct contributing area of *ew* (m²), or 0 when unlabelled."""
        counts = self._state.catchment_counts or {}
        meta = self._state.flow_grid_meta
        if not counts or meta is None or ew.id not in counts:
            return 0.0
        return counts[ew.id] * meta["cell_area_m2"]

    def _current_runoff_mm(self):
        """Depth of water to size against, per the panel's sizing basis.

        'rainfall' treats every millimetre falling on a feature's catchment as
        arriving at it. That is not what happens hydrologically — much of it soaks in
        where it lands — but it is how earthworks are sized in the field, and it is
        the safe side of a very sensitive assumption: at CN 61 in normal conditions
        SCS-CN passes only 26% of a 120 mm storm, while the same ground on wet
        antecedent conditions passes 53%, and at CN 80 wet, 77%. Sizing on the low
        figure and being wrong by that margin means a breached swale.

        'runoff' uses the SCS-CN surface runoff depth — physically the right answer
        for what reaches a feature, and appropriate once the curve number and
        antecedent condition are known for the site rather than assumed.
        """
        from terrainflow_assessment.modules.catchment import (
            SCSRunoff,
            coefficient_runoff_depth,
        )
        basis = self._panel.sizing_basis
        if basis == "rainfall":
            return self._panel.rainfall_mm
        if basis == "coefficient":
            return coefficient_runoff_depth(
                self._panel.rainfall_mm, self._panel.runoff_coefficient)
        scs = SCSRunoff()
        return scs.runoff_depth(
            self._panel.rainfall_mm,
            scs.adjust_cn(self._panel.cn, self._panel.moisture),
        )

    def _runoff_basis_note(self):
        """One line naming the sizing basis and what it costs or buys."""
        from terrainflow_assessment.modules.catchment import SCSRunoff
        scs = SCSRunoff()
        rain = self._panel.rainfall_mm or 0.0
        if rain <= 0:
            return ""
        cn = scs.adjust_cn(self._panel.cn, self._panel.moisture)
        runoff = scs.runoff_depth(rain, cn)
        coeff = runoff / rain if rain else 0.0
        basis = self._panel.sizing_basis
        if basis == "coefficient":
            c = self._panel.runoff_coefficient
            return (
                f"Working from {rain * c:.0f} mm — {c:.2f} of the {rain:.0f} mm storm "
                f"(rational method, Lancaster). SCS-CN at CN {cn:.0f} "
                f"({self._panel.moisture}) would give {runoff:.0f} mm ({coeff:.0%}); "
                f"the full rainfall would give {rain:.0f} mm."
            )
        if basis == "rainfall":
            return (
                f"Working from the full {rain:.0f} mm of rainfall — every millimetre "
                f"assumed to run off. That is above Lancaster's metal-roof coefficient "
                f"(0.95) and no established method sizes a landscape catchment this "
                f"way; SCS-CN at CN {cn:.0f} ({self._panel.moisture}) would give "
                f"{runoff:.0f} mm ({coeff:.0%})."
            )
        return (
            f"<span style='color:#b9770e;'>Working from {runoff:.0f} mm of surface "
            f"runoff — {coeff:.0%} of the {rain:.0f} mm storm. Curve number and "
            f"antecedent moisture swing this by 3–4×; wetter ground or a higher CN "
            f"would demand far larger features.</span>"
        )

    # ---------------------------------------------------------------- Live assessment

    def _recompute_live_assessment(self, geometry_settled=True):
        """Design-tier: recompute the live analytical water balance → panel readout.

        Fast, no burn. Geometry metrics (capacity/cut/fill) always; storm capture %
        once a baseline exists. Everything storm-dependent is arithmetic over the
        cached catchment cell counts, so this stays responsive while dragging.
        Fires on every earthwork edit and storm/soil change; failures are swallowed so
        the readout never breaks the edit flow.

        *geometry_settled* is False only on the throttled vertex-drag tier. Two things
        wait for the release: the spillway review, because each of its rows samples the
        DEM and doing that per feature at 12.5 Hz is exactly the cost this method's
        docstring promises to avoid; and the undersized-spillway warning, because a
        message bar that repaints on every mouse move is not a warning, it is a
        flicker. Both are correct to defer — mid-drag geometry is not a design.
        """
        try:
            from terrainflow_assessment.modules.simulation import (
                build_stores_from_earthworks,
                resolve_targets,
            )
            from terrainflow_assessment.modules.water_balance import run_water_balance

            all_ews = self._state.earthwork_manager.get_all()
            if not all_ews:
                self._panel.set_network([], {}, 0.0)
                self._panel.set_live_assessment("")
                self._panel.scorecard_empty()
                return

            enabled = [ew for ew in all_ews if ew.enabled]
            runoff_mm = self._current_runoff_mm()
            duration_hr = self._panel.duration_hr or 1.0
            soil = self._panel.earthwork_soil_name

            # The catchment labels are the source of truth for who gets what water.
            if self._state.catchment_labels is None:
                self.recompute_catchments()
            counts = self._state.catchment_counts or {}
            meta = self._state.flow_grid_meta
            have_flow = bool(counts) and meta is not None

            cell_area = meta["cell_area_m2"] if have_flow else 0.0
            runoff_m = runoff_mm / 1000.0
            if have_flow:
                domain_cells = int(self._state.flow_domain_mask.sum())
                total_runoff_m3 = domain_cells * cell_area * runoff_m
                uncaptured_m3 = (
                    (self._state.catchment_exit_cells + self._state.catchment_sink_cells)
                    * cell_area * runoff_m
                )
            else:
                total_runoff_m3 = uncaptured_m3 = 0.0

            stores = build_stores_from_earthworks(
                enabled, soil_name=soil, dem_path=self._state.dem_path
            )
            for store in stores:
                cells = counts.get(store.id, 0)
                store.direct_catchment_m2 = cells * cell_area
                store.inflow_m3 = cells * cell_area * runoff_m
                store.outlet_flat = self._state.catchment_outlets.get(store.id)

            routing = resolve_targets(stores, walker=self._make_walker())
            # Peak rates follow the same routing as the volumes, so a link the user
            # drew moves both the water and the spillway it has to pass.
            self.recompute_peak_flows(routing=routing)
            # Auto widths follow the peak rates. Arithmetic over figures just computed,
            # so it is safe on the drag path — unlike the review rows, which sample the
            # DEM and are built on the discrete edits instead. Kept in its own guard:
            # the outer handler would swallow a failure here along with the network
            # repaint, the scorecard and everything else below it.
            try:
                self._refresh_auto_spillway_widths()
            except Exception as exc:
                print(f"TerrainFlow Assessment — auto spillway width error: {exc}")
            result = run_water_balance(
                stores, duration_hr, total_runoff_m3,
                uncaptured_m3=uncaptured_m3, routing=routing,
                count_infiltration=self._panel.count_infiltration,
            ) if stores else None

            # Flow network (Live Assessment) — every earthwork, ordered high→low.
            nodes = self._build_network_nodes(all_ews, stores, result)
            edges = {sid: (tgt, routing.is_user.get(sid, False))
                     for sid, tgt in routing.edges.items()}
            exit_m3 = result.site_exit_m3 if result is not None else 0.0
            self._panel.set_network(nodes, edges, exit_m3)
            self._panel.set_live_assessment(self._network_footer(result))
            self._refresh_connections_layer(result, routing)
            self._panel.set_area_subtotals(self.compute_area_subtotals())
            self.refresh_stress_points_layer()
            if geometry_settled:
                self._build_spillway_rows()
                self._check_spillway_capacity()

            # Persistent scorecard (Workbench header) — blue means actual water.
            if result is not None and have_flow:
                stored = max(0.0, result.total_captured_m3 - result.total_infiltration_m3)
                self._panel.update_scorecard(
                    result.capture_pct, stored,
                    result.total_infiltration_m3, result.site_exit_m3,
                    natural_ponding_m3=self._natural_ponding_m3(),
                )
            elif result is not None:
                self._panel.scorecard_no_flow(result.total_capacity_m3)
            else:
                self._panel.scorecard_empty(
                    "No storage yet — draw a swale, basin or dam."
                )
        except Exception as exc:  # never let the readout break the edit flow
            import traceback
            print(f"TerrainFlow Assessment — live assessment error: {exc}")
            traceback.print_exc()

    def _build_network_nodes(self, all_ews, stores, result):
        """Node dicts for the flow network — one per earthwork, water from the balance.

        Keyed by ``ew.id``, not name: the default name is
        ``f"{type} {len(manager)+1}"``, counting *all* earthworks, so deleting one and
        drawing another reproduces an existing name. Under name keying the first of the
        pair became unreachable in the lookup dicts (its water silently read as 0) and
        both cards rendered the same row.
        """
        from terrainflow_assessment.core.registry.earthwork_types import get_type

        store_elev = {s.id: s.elevation for s in stores}
        per = {f["id"]: f for f in result.per_feature} if result is not None else {}
        nodes = []
        for i, ew in enumerate(all_ews):
            try:
                colour = get_type(ew.type).style[1]
            except KeyError:
                colour = "#888888"
            elev = store_elev.get(ew.id)
            if elev is None:
                elev = self._feature_elevation(ew.geometry) or 0.0
            f = per.get(ew.id)
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
                "overflow_m3": f["overflow_m3"] if f else 0.0,
                "soaked_m3": f["infiltration_m3"] if f else 0.0,
                "drain_hours": f.get("drain_hours") if f else None,
                "catchment_m2": f["direct_catchment_m2"] if f else 0.0,
                "is_terminal": bool(f["is_terminal"]) if f else True,
                "enabled": bool(ew.enabled),
                "has_water": f is not None,
                "summary": ew.summary(),
            })
        return nodes

    def _network_footer(self, result):
        """Totals + disclaimer line beneath the network."""
        if result is None:
            return ""
        parts = [
            f"Capacity {result.total_capacity_m3:,.0f} m³ · "
            f"Cut {result.total_cut_m3:,.0f} · Fill {result.total_fill_m3:,.0f} m³"
        ]
        if result.terminal_deficit_m3 > 0:
            parts.append(
                f"<span style='color:#b9770e;'>{result.terminal_deficit_m3:,.0f} m³ "
                f"overflows past the last feature — that much more storage is needed "
                f"upslope to hold this storm.</span>"
            )
        if result.uncaptured_m3 > 0:
            parts.append(
                f"{result.uncaptured_m3:,.0f} m³ of the site drains to no feature at all."
            )
        if not result.counts_infiltration and result.infiltration_buffer_m3 > 1:
            parts.append(
                f"Sized on held volume only — a further "
                f"{result.infiltration_buffer_m3:,.0f} m³ would soak away over the "
                f"event, as spare capacity rather than relied-upon storage."
            )
        if not result.mass_balance_ok:
            parts.append(
                "<span style='color:#c0392b;'>Water balance does not close — the "
                "capture figure above is unreliable. Re-run Baseline.</span>"
            )
        for warn in result.routing_warnings[:2]:
            parts.append(f"<span style='color:#b9770e;'>{warn}</span>")
        note = self._runoff_basis_note()
        if note:
            parts.append(note)
        parts.append("Analytical estimate — verify with Re-analyse.")
        return "  ·  ".join(parts)

    @staticmethod
    def _point_on(ew):
        """A point that is actually *on* the feature, for anchoring a link.

        The centroid of a curved swale is off in the field beside it, so a link
        falling back to a centroid starts in mid-air — one of the "lines at odd
        angles" complaints. Midpoint along the line, or a guaranteed-interior
        point for a polygon.
        """
        geom = ew.geometry
        try:
            if geom.type() == QgsWkbTypes.PolygonGeometry:
                return geom.pointOnSurface().asPoint()
            return geom.interpolate(geom.length() / 2.0).asPoint()
        except Exception:
            return geom.centroid().asPoint()

    def _refresh_connections_layer(self, result, routing):
        """Draw the resolved overflow links on the map as arrows between features.

        One line per edge, from the source's outlet toward the target, styled with the
        existing :meth:`_arrow_line_layer` ribbon so direction is unambiguous. Solid
        for a user-set link, dashed for one resolved from the flow path; thickness
        scales with the volume actually routed, so a heavily-loaded link reads heavier.
        """
        from terrainflow_assessment.qgis.controllers._layers import remove_layer

        if result is None or not routing.edges:
            remove_layer(self._project, self._state.connections_layer_id)
            self._state.connections_layer_id = None
            return
        try:
            from qgis.core import QgsPointXY

            by_id = {ew.id: ew for ew in self._state.earthwork_manager.get_all()}
            per = {f["id"]: f for f in result.per_feature}

            feats = []
            for src_id, tgt_id in routing.edges.items():
                src = by_id.get(src_id)
                tgt = by_id.get(tgt_id) if tgt_id else None
                row = per.get(src_id, {})
                volume = float(row.get("overflow_m3", 0.0) or 0.0)
                is_user = bool(routing.is_user.get(src_id))
                if src is None:
                    continue
                # A link the user drew is part of the design and must be visible
                # whether or not this particular storm fills it — "Swale 1 now
                # overflows into Swale 2" succeeded in the field log, and then
                # nothing appeared on the map, which read as the tool not working.
                #
                # Auto-resolved links still need volume: resolve_targets writes an
                # edge for EVERY store, so drawing them all unconditionally would
                # put a line on every earthwork on the site.
                if not is_user and volume <= 0:
                    continue
                # Anchor to the real structures where they exist. A centroid-to-
                # centroid line says two features are linked; an outflow-to-inflow
                # line says *where* the water crosses, which is the thing you go and
                # build. Falls back to centroids so an unplaced pair still draws.
                start = (self._spillway_point(src, "spillway")
                         or self._point_on(src))
                if tgt is not None:
                    end = (self._spillway_point(tgt, "inflow_spillway")
                           or self._point_on(tgt))
                else:
                    end = self._downslope_exit_point(src)
                    if end is None:
                        continue
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPolylineXY(
                    [QgsPointXY(start), QgsPointXY(end)]))
                f.setAttributes([
                    src.name,
                    tgt.name if tgt is not None else "leaves site",
                    1 if routing.is_user.get(src_id) else 0,
                    round(volume, 1),
                ])
                feats.append(f)

            remove_layer(self._project, self._state.connections_layer_id)
            self._state.connections_layer_id = None
            if not feats:
                return

            crs = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
            layer = QgsVectorLayer(f"LineString?crs={crs}", "Overflow connections", "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("from_name", QMetaType.QString),
                QgsField("to_name", QMetaType.QString),
                QgsField("is_user_link", QMetaType.Int),
                QgsField("overflow_m3", QMetaType.Double),
            ])
            layer.updateFields()
            pr.addFeatures(feats)
            layer.updateExtents()

            layer.setRenderer(QgsSingleSymbolRenderer(S.connection_symbol()))

            self.place(layer, G.DRAWN)
            self._state.connections_layer_id = layer.id()
            self.restack(S.DRAW_ORDER, G.DRAWN)
        except Exception as exc:
            print(f"TerrainFlow Assessment — connections layer error: {exc}")

    def refresh_catchment_layer(self, visible=True):
        """Render "which earthwork catches what" — one colour per feature.

        The label raster is already in memory from the balance, so this costs one
        write and a paletted renderer. It is the clearest single answer to "why is
        my capture only 19%?": grey is ground that reaches nothing.
        """
        from terrainflow_assessment.qgis.controllers._layers import remove_layer

        remove_layer(self._project, self._state.catchment_labels_layer_id)
        self._state.catchment_labels_layer_id = None

        labels = self._state.catchment_labels
        meta = self._state.flow_grid_meta
        if labels is None or meta is None or not visible:
            return
        try:
            import rasterio
            from qgis.core import QgsPalettedRasterRenderer

            from terrainflow_assessment.modules.flow_graph import LABEL_EXIT

            path = os.path.join(self._state.output_dir, "catchment_labels.tif")
            with rasterio.open(
                path, "w", driver="GTiff", dtype="int16", nodata=-1,
                crs=self._state.dem_info.crs if self._state.dem_info else None,
                transform=meta["transform"],
                width=meta["shape"][1], height=meta["shape"][0],
                count=1, compress="lzw",
            ) as dst:
                dst.write(labels.astype("int16"), 1)

            layer = QgsRasterLayer(path, "Catchment by earthwork")
            if not layer.isValid():
                return

            by_id = {ew.id: ew for ew in self._state.earthwork_manager.get_all()}
            classes = []
            for i, ew_id in enumerate(self._state.catchment_label_ids):
                ew = by_id.get(ew_id)
                if ew is None:
                    continue
                try:
                    colour = QColor(get_type(ew.type).style[1])
                except KeyError:
                    colour = QColor("#888888")
                colour.setAlpha(150)
                classes.append(QgsPalettedRasterRenderer.Class(i, colour, ew.name))
            classes.append(QgsPalettedRasterRenderer.Class(
                LABEL_EXIT, QColor(150, 150, 150, 110), "leaves site"))

            if not classes:
                return
            layer.setRenderer(
                QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
            self.place(layer, G.DESIGN)
            self._state.catchment_labels_layer_id = layer.id()
            self._canvas.refresh()
        except Exception as exc:
            print(f"TerrainFlow Assessment — catchment layer error: {exc}")

    def toggle_catchment_layer(self, visible):
        """Show/hide the direct-catchment layer from the panel."""
        self.refresh_catchment_layer(visible=visible)

    def highlight_selected_earthwork(self, index):
        """Outline the selected feature on the canvas.

        With five swales in the list there was no way to tell which row referred to
        which line on the map. A rubber band rather than layer selection: it survives
        the layer being rebuilt on every edit, and does not disturb whatever the user
        has selected for their own purposes.
        """
        from qgis.core import QgsWkbTypes
        from qgis.gui import QgsRubberBand

        band = getattr(self, "_selection_band", None)
        if band is None:
            band = QgsRubberBand(self._canvas, QgsWkbTypes.LineGeometry)
            band.setColor(QColor(46, 125, 85, 220))
            band.setWidth(4)
            self._selection_band = band

        # Reset to the type the band was built with. `band.geometryType()` does not exist
        # on QgsRubberBand and raised AttributeError on every selection change — polygon
        # footprints are highlighted via their boundary, so this is a line band throughout.
        band.reset(QgsWkbTypes.LineGeometry)
        if index is None:
            self._canvas.refresh()
            return
        try:
            ew = self._state.earthwork_manager.get(index)
        except (IndexError, AttributeError):
            return
        try:
            # A polygon footprint is outlined rather than filled, so the highlight
            # never hides the ponding raster underneath it.
            band.setToGeometry(ew.geometry.constGet().boundary()
                               if ew.geometry.type() == 2 else ew.geometry, None)
        except Exception:
            try:
                band.setToGeometry(ew.geometry, None)
            except Exception:
                return
        self._canvas.refresh()

    def _spillway_point(self, ew, attr):
        """The placed point of one of *ew*'s spillways, or None if not sited."""
        spillway = getattr(ew, attr, None)
        if spillway is None or not spillway.point_wkt:
            return None
        try:
            geom = QgsGeometry.fromWkt(spillway.point_wkt)
            if geom is None or geom.isEmpty():
                return None
            return geom.asPoint()
        except Exception:
            return None

    # How far a "leaves site" stub runs, in ground metres. Previously this was a
    # fixed 40 D8 cells, which is 40 m on a 1 m DEM and 200 m on a 5 m one — the
    # same code drew a tick on one survey and a line across the paddock on
    # another. Length should mean a distance, not a cell count.
    _EXIT_STUB_M = 40.0

    def _downslope_exit_point(self, ew):
        """Where a terminal feature's overflow heads — a short stub down the flow path."""
        try:
            meta = self._state.flow_grid_meta
            start = self._state.catchment_outlets.get(ew.id)
            if meta is None or start is None:
                return None
            next_flat = self._state.flow_next
            cols = meta["shape"][1]
            transform = meta["transform"]
            cell_m = abs(float(transform.a)) or 1.0
            steps = max(4, int(round(self._EXIT_STUB_M / cell_m)))
            cur = int(start)
            for _ in range(steps):        # a short, legible stub, not the whole path
                nxt = int(next_flat[cur])
                if nxt == cur:
                    break
                cur = nxt
            row, col = divmod(cur, cols)
            from qgis.core import QgsPointXY
            return QgsPointXY(
                transform.c + (col + 0.5) * transform.a,
                transform.f + (row + 0.5) * transform.e,
            )
        except Exception:
            return None

    # ---------------------------------------------------------------- Earthwork layers

    def _ensure_ew_layers(self):
        from qgis.core import QgsRuleBasedRenderer

        crs_str = self._state.dem_info.crs_wkt if self._state.dem_info else "EPSG:4326"
        self._state.ew_group = self.group_for(G.DRAWN)

        # Registry-driven: the type registry is the single source of layer styling
        # (matching the panel's draw-button colours); a future register_type() gets
        # its map layer automatically.
        for ew_type, cfg in all_types().items():
            geom_type = cfg.geom_type
            display_name = f"{cfg.label}s"
            color_hex = cfg.style[1]
            if resolve_layer(self._project, self._state.ew_layers.get(ew_type)):
                continue

            layer = QgsVectorLayer(f"{geom_type}?crs={crs_str}", display_name, "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name",        QMetaType.QString),
                QgsField("type",        QMetaType.QString),
                QgsField("capacity_m3", QMetaType.Double),
                QgsField("enabled",     QMetaType.Int),
                QgsField("width_m",     QMetaType.Double),
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

            lbl = S.earthwork_label_settings(cfg, color_hex)
            layer.setLabeling(QgsVectorLayerSimpleLabeling(lbl))
            layer.setLabelsEnabled(True)

            # Through place(), not addMapLayer + addLayer by hand — CLAUDE.md's
            # rule, and this was the one spot in the controller sidestepping it.
            # Same resulting tree position; the restack below owns ordering.
            self.place(layer, G.DRAWN)
            # The id, never the layer. A stored wrapper outlives the C++ object it
            # points at — delete the layer in the Layers panel, or drop it while
            # re-stacking the group, and the next access raises "wrapped C/C++
            # object has been deleted" instead of quietly rebuilding.
            self._state.ew_layers[ew_type] = layer.id()

        self.restack(S.DRAW_ORDER, G.DRAWN)

    # ---------------------------------------------------------------- Earthwork symbology

    def _build_ew_symbol(self, cfg, enabled=True):
        """Per-type canvas symbol. Lives in ``_symbols.py`` — see that module for the
        metres-vs-millimetres rule the whole visual grammar hangs off."""
        return S.earthwork_symbol(cfg, enabled)

    def _refresh_ew_layer(self):
        self._ensure_ew_layers()
        for layer_id in self._state.ew_layers.values():
            layer = resolve_layer(self._project, layer_id)
            if layer is not None:
                layer.dataProvider().truncate()

        for ew in self._state.earthwork_manager.get_all():
            layer = resolve_layer(self._project, self._state.ew_layers.get(ew.type))
            if layer is None:
                continue
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromWkt(ew.geometry.asWkt()))
            # width_m drives the data-defined, map-units symbol width so each
            # earthwork renders at its real ground width. Basin (polygon) writes
            # 0.0 harmlessly — its fill symbol never reads the field.
            w = float(getattr(ew, "top_width_m", 0.0) or 0.0)
            f.setAttributes([ew.name, ew.type, ew.capacity_m3,
                              1 if getattr(ew, "enabled", True) else 0, w])
            layer.dataProvider().addFeature(f)

        for layer_id in self._state.ew_layers.values():
            layer = resolve_layer(self._project, layer_id)
            if layer is not None:
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
            msg += "\n" + self.verification_sentence(v)
        self._panel.set_earthworks_complete(msg)

        # A skipped baseline subtraction inflates every measured figure by whatever
        # ponded there naturally, so it cannot be left to the table alone.
        if v is not None and getattr(v, "baseline_uncorrected", None):
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"Measured storage is not corrected for natural ponding: "
                f"{v.baseline_uncorrected}."
            )

        # Per-feature breakdown in the Verify stage. The chip carries one site-wide
        # delta, which cannot distinguish a burn error from the freeboard allowance.
        cell = self._state.dem_info.cell_size_m if self._state.dem_info else 1.0
        self._panel.set_verification(v, cell_size_m=cell)

        # The design is now verified against a burn — reset the drift counter.
        self._state.edits_since_verify = 0
        self._state.verified_delta_pct = v.delta_pct if v is not None else None
        self._update_verified_chip()

    def verification_sentence(self, v):
        """Explain the verification delta in words.

        "Verified · Δ −38%" said nothing about what was being compared or what the
        user should do. The delta now measures **only** the burn — measured ponding
        against what the grid can represent — with freeboard and resolution reported
        as separate, expected differences rather than folded into the same number.
        """
        if v is None:
            return ""
        freeboard = sum(f.get("freeboard_m3", 0.0) for f in v.per_feature)
        penalty = sum(f.get("resolution_penalty_m3", 0.0) for f in v.per_feature)
        reference = sum(f.get("rasterisable_m3", 0.0) for f in v.per_feature)

        parts = [
            f"Δ {v.delta_pct:+.0f}% — measured storage {v.terrain_total_m3:,.0f} m³ "
            f"against the {reference:,.0f} m³ this grid can represent."
        ]
        if abs(freeboard) > 1:
            parts.append(f"Freeboard accounts for a further {freeboard:,.0f} m³ "
                         f"deliberately kept empty.")
        if abs(penalty) > 1:
            parts.append(f"Grid resolution shifts the drawn shape by "
                         f"{penalty:+,.0f} m³.")
        return " ".join(parts)

    def _compute_verification(self):
        """Reconcile terrain-derived ponding (burned DEM) against analytic capacity (§4).

        Returns a reporting.VerificationResult, or None if the ponding raster or any
        storage feature is unavailable. Pure maths lives in modules/reporting.py; this
        only reads the rasters + builds per-feature footprint masks.
        """
        import json

        import numpy as np
        import rasterio
        from shapely.geometry import shape as _shp

        from terrainflow_assessment.modules.earthwork_design import capacity_breakdown
        from terrainflow_assessment.modules.footprint import (
            min_dimension,
            rasterize_footprint,
        )
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
        #
        # This fallback used to be silent on all three of its paths, and the cost was
        # a wrong number presented as a verified one: without the subtraction, water
        # that ponded naturally is counted as earthwork storage. A dam sitting in a
        # hollow that already held ~555 m³ reported +77% against its design, with
        # nothing on screen to say the correction had been skipped — and the swales,
        # cut into slopes that pond nothing, all verified clean and made it look like
        # a dam-specific fault. Name the reason and let the caller surface it.
        bl_pond = np.zeros(shape, dtype="float64")
        bl_uncorrected = None
        bl_pond_path = (self._state.baseline_result or {}).get("ponding")
        if not bl_pond_path or not os.path.exists(bl_pond_path):
            bl_uncorrected = ("no baseline ponding raster is available. Run the "
                              "Baseline stage first")
        else:
            try:
                with rasterio.open(bl_pond_path) as src:
                    arr = src.read(1).astype("float64")
                    bnd = src.nodata
                if arr.shape != shape:
                    bl_uncorrected = (
                        f"the baseline ponding raster is {arr.shape[1]}×{arr.shape[0]} "
                        f"but the earthworks raster is {shape[1]}×{shape[0]}. Re-run "
                        f"the Baseline stage at the current extent"
                    )
                else:
                    if bnd is not None:
                        arr[arr == bnd] = 0.0
                    bl_pond = np.clip(arr, 0.0, None)
            except Exception as exc:
                bl_uncorrected = (f"the baseline ponding raster could not be read "
                                  f"({exc})")

        diff = np.clip(ew_pond - bl_pond, 0.0, None)

        analytic_by_name = {}
        min_dims = {}
        breakdowns = {}
        footprints = []
        for ew in self._state.earthwork_manager.get_enabled():
            if getattr(ew, "capacity_m3", 0.0) <= 0:
                continue
            analytic_by_name[ew.name] = ew.capacity_m3
            # Sub-cell check: channels key off the bottom width, polygons off their
            # equivalent strip width. Basins previously passed None, so a footprint
            # smaller than a cell still claimed its full analytic volume unflagged.
            try:
                geom = _shp(json.loads(ew.geometry.asJson()))
            except Exception:
                geom = None
            if ew.type == "swale":
                min_dims[ew.name] = getattr(ew, "bottom_width_m", None)
            else:
                min_dims[ew.name] = min_dimension(geom) if geom is not None else None

            mask = np.zeros(shape, dtype=bool)
            if geom is not None:
                try:
                    foot = geom
                    if foot.geom_type in ("LineString", "MultiLineString"):
                        foot = foot.buffer(
                            max(getattr(ew, "width", 2.0) / 2.0, cell_size))
                    mask = rasterize_footprint(foot, shape, transform,
                                               all_touched=True)
                except Exception:
                    mask = np.zeros(shape, dtype=bool)
            footprints.append((ew.name, mask))

            # What this grid can actually represent — the reference the delta is
            # measured against, so the headline isolates burn error from cell size.
            try:
                breakdowns[ew.name] = capacity_breakdown(
                    ew, cell_size=cell_size, n_cells=int(mask.sum()))
            except Exception:
                pass

        if not analytic_by_name:
            return None

        terrain_by_name, unattributed = attribute_ponding_volume(diff, cell_area, footprints)
        # Water already standing here before any earthwork, over the same footprints —
        # so a feature built in a hollow can report what it adds, what was already
        # there, and the pool that ends up on the ground. One extra region-labelling
        # pass; the flood it depends on has already run.
        existing_by_name, _ = attribute_ponding_volume(bl_pond, cell_area, footprints)
        baseline_total = raster_ponding_volume(bl_pond, cell_area)
        earthworks_total = raster_ponding_volume(ew_pond, cell_area)

        result = build_verification(
            analytic_by_name, terrain_by_name, baseline_total, earthworks_total,
            min_dims, cell_size, breakdowns=breakdowns,
            existing_by_name=existing_by_name,
        )
        result.unattributed_m3 = unattributed
        # None when the subtraction was applied; a reason string when every measured
        # figure still carries whatever ponded there naturally.
        result.baseline_uncorrected = bl_uncorrected
        return result

    def _load_burned_dem_layer(self):
        """Add the burned (Strategy-C) DEM to the layer panel so the carve/ridge is visible.

        Placed at the bottom of the Design group (it is a backdrop, not a result
        overlay) and registered under the earthworks layer ids so it shows/hides with
        the 'with earthworks' toggle. Silently skips if the burn produced no valid raster.
        """
        path = self._state.modified_dem_path
        if not path or not os.path.exists(path):
            return
        from qgis.core import QgsRasterLayer
        layer = QgsRasterLayer(path, "Earthworks — Burned DEM")
        if not layer.isValid():
            return
        self.place(layer, G.DESIGN)
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
            self.place(layer, G.ANALYSIS, visible=checked)
            self._state.slope_class_layer_id = layer.id()
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
            QgsColorRampShader.ColorRampItem(50, QColor("#660000"), "25–50°"),
            QgsColorRampShader.ColorRampItem(90, QColor("#3A0000"), "≥50°"),
        ])
        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, raster_shader)
        layer.setRenderer(renderer)

    def toggle_slope_vectors(self, checked):
        if checked:
            existing = (self._state.slope_vectors_layer_id and
                        self._project.instance().mapLayer(self._state.slope_vectors_layer_id))
            if existing:
                node = self._project.instance().layerTreeRoot().findLayer(
                    self._state.slope_vectors_layer_id)
                if node:
                    node.setItemVisibilityChecked(True)
            else:
                self._generate_slope_vectors()
        else:
            if self._state.slope_vectors_layer_id:
                node = self._project.instance().layerTreeRoot().findLayer(
                    self._state.slope_vectors_layer_id)
                if node:
                    node.setItemVisibilityChecked(False)
            self._canvas.refresh()

    # Hachures are drawn in one neutral dark tone, not on the slope-class ramp.
    # They are usually read *over* the slope raster, and a red-family stroke
    # vanishes into the red end of that ramp exactly where the ground is steepest
    # and the reading matters most. Steepness is carried by stroke width instead.
    _HACHURE_COLOR = "51,56,59,217"
    # Taper: 0.3 mm on flat ground widening to 1.8 mm at 45° and above. Capping the
    # slope term stops a single cliff cell from drawing a stroke the width of a road.
    _HACHURE_WIDTH_EXPR = '0.3 + min("slope_deg", 45) / 45.0 * 1.5'

    def _generate_slope_vectors(self):
        if not self._state.dem_path:
            return
        try:
            from qgis.core import QgsGeometry

            from terrainflow_assessment.modules.flow_lines import hachure_segments

            segs = hachure_segments(self._state.dem_path, spacing_m=30.0)
            crs = self._state.dem_info.crs_wkt if self._state.dem_info else None
            uri = f"LineString?crs={crs}" if crs else "LineString"
            layer = QgsVectorLayer(uri, "Slope Vectors", "memory")
            if crs is None:
                layer.setCrs(self._project.instance().crs())
            pr = layer.dataProvider()
            pr.addAttributes([QgsField("slope_deg", QMetaType.Double)])
            layer.updateFields()

            feats = []
            for seg in segs:
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromWkt(seg["geometry"].wkt))
                f.setAttributes([seg["slope_deg"]])
                feats.append(f)
            pr.addFeatures(feats)
            layer.updateExtents()

            layer.setRenderer(QgsSingleSymbolRenderer(self._hachure_symbol()))
            self.place(layer, G.ANALYSIS)
            self._state.slope_vectors_layer_id = layer.id()
            self._canvas.refresh()

        except Exception as exc:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", f"Could not generate slope vectors: {exc}"
            )

    def _hachure_symbol(self):
        """A tapered downhill stroke: broad at the uphill end, drawn to a point
        downhill, width set by steepness.

        Built on the arrow symbol layer with the head suppressed, which is the only
        stock symbol that varies width along a line. Falls back to a plain
        constant-taper stroke if that API is unavailable, so the layer still draws.
        """
        from qgis.core import QgsLineSymbol
        try:
            from qgis.core import QgsArrowSymbolLayer

            arrow = QgsArrowSymbolLayer()
            arrow.setArrowStartWidth(1.8)   # uphill end — overridden per feature below
            arrow.setArrowWidth(0.1)        # downhill end — tapers to a point
            arrow.setHeadLength(0.0)        # no arrowhead; a hachure is a wedge
            arrow.setHeadThickness(0.0)
            arrow.setDataDefinedProperty(
                QgsSymbolLayer.PropertyArrowStartWidth,
                QgsProperty.fromExpression(self._HACHURE_WIDTH_EXPR))
            sub = arrow.subSymbol()
            if sub is not None:
                sub.setColor(QColor(51, 56, 59, 217))
                for i in range(sub.symbolLayerCount()):
                    try:
                        sub.symbolLayer(i).setStrokeStyle(Qt.NoPen)
                    except Exception:
                        pass
            sym = QgsLineSymbol()
            sym.changeSymbolLayer(0, arrow)
            return sym
        except Exception:
            sym = QgsLineSymbol.createSimple({
                "width": "0.5", "capstyle": "round",
                "color": self._HACHURE_COLOR,
            })
            sym.symbolLayer(0).setDataDefinedProperty(
                QgsSymbolLayer.PropertyStrokeWidth,
                QgsProperty.fromExpression(self._HACHURE_WIDTH_EXPR))
            return sym
