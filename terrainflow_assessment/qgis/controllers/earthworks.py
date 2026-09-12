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
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProperty,
    QgsRasterLayer,
    QgsRasterShader,
    QgsSettings,
    QgsSingleBandPseudoColorRenderer,
    QgsSingleSymbolRenderer,
    QgsSymbolLayer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QMetaType, Qt
from qgis.PyQt.QtGui import QColor

from terrainflow_assessment.core.registry.earthwork_defaults import (
    decode as decode_earthwork_defaults,
)
from terrainflow_assessment.core.registry.earthwork_defaults import (
    encode as encode_earthwork_defaults,
)
from terrainflow_assessment.core.registry.earthwork_defaults import (
    resolve_dimensions,
    shipped_dims,
)
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
from terrainflow_assessment.qgis import help_text as H
from terrainflow_assessment.qgis.controllers import _groups as G
from terrainflow_assessment.qgis.controllers import _symbols as S
from terrainflow_assessment.qgis.controllers._layers import (
    crs_object,
    dem_crs,
    resolve_layer,
)
from terrainflow_assessment.qgis.controllers._tools import MapToolMixin
from terrainflow_assessment.qgis.workers._lifecycle import worker_is_running
from terrainflow_assessment.qgis.workers.analysis_worker import AnalysisWorker
from terrainflow_assessment.qgis.workers.task_worker import TaskWorker


class EarthworksController(G.LayerTreeMixin, MapToolMixin):
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas
        # Layer nodes whose visibility we are already listening to, so a rebuilt
        # layer does not accumulate connections.
        self._watched_layer_ids = set()
        # Set by the plugin: the one BaselineController. The re-analysis renders its
        # results through the same helper the baseline does, and the exit-marker ids
        # have to land on the controller that owns the `scaleChanged` connection.
        self.baseline = None
        # Which features `_check_spillway_capacity` has already named. It runs on every
        # settled recompute, so without this the same warning is re-pushed on every
        # subsequent edit and stops reading as news. None until the first check, which is
        # distinct from "checked, and nothing was short".
        self._short_spillways_reported = None
        # The user's standard earthwork dimensions, per type. Per user rather than per
        # project: "the trough on my tractor" follows the person to every site, and a
        # project-scoped standard would be empty in a project not yet saved — which is
        # exactly when the first swale gets drawn. Loaded once here; refreshed when the
        # properties dialog saves one.
        self._earthwork_defaults = {}
        self.load_earthwork_defaults()

    def _watch_visibility(self, layer_id):
        """Clear the highlight when this layer is unticked.

        Unticking an earthwork layer is the obvious way to make its highlight go
        away, and it did not: the band is a canvas item and belongs to no layer
        at all. Connected per node rather than to the tree root — the root's
        ``visibilityChanged`` does not reliably carry a descendant's change.
        """
        if layer_id in self._watched_layer_ids:
            return
        try:
            node = self._project.instance().layerTreeRoot().findLayer(layer_id)
            if node is None:
                return
            node.visibilityChanged.connect(self._on_layer_visibility_changed)
            self._watched_layer_ids.add(layer_id)
        except Exception:
            pass

    def teardown(self):
        """Release the per-node visibility hooks and the active tool.

        `_watched_layer_ids` accumulates one connection per layer the design has
        ever created, all of them on layer-tree nodes owned by the project rather
        than by the plugin. Left connected, each one calls back into a dead
        controller the next time the user ticks a checkbox in the legend.
        """
        root = None
        try:
            root = self._project.instance().layerTreeRoot()
        except (AttributeError, RuntimeError):
            root = None
        for layer_id in list(getattr(self, "_watched_layer_ids", ())):
            try:
                node = root.findLayer(layer_id) if root is not None else None
                if node is not None:
                    node.visibilityChanged.disconnect(
                        self._on_layer_visibility_changed)
            except (TypeError, RuntimeError, AttributeError):
                pass
        self._watched_layer_ids.clear()
        self.release_tool()

    def _on_layer_visibility_changed(self, node):
        try:
            if not node.isVisible():
                self.clear_selection_highlight()
        except Exception:
            pass

    # ---------------------------------------------------------------- Drawing tools

    def _contour_pick_layers(self):
        """Every contour layer a swale may be drawn on, analysed candidates first.

        Both layers are offered because a contour does not have to be a ranked
        candidate to be worth a swale. The analysis only promotes contours that
        pass the slope cutoff, the minimum length and the usable-area clip, and
        for a while those were the only lines the two picking tools could see —
        so a contour the user was looking at, on a layer the plugin had drawn,
        could be un-clickable for reasons nothing on screen explained.

        Candidates come first because ties go to the earlier layer: at a shared
        elevation the two layers draw the same line, and the candidate is the one
        carrying rank and inflow.
        """
        from terrainflow_assessment.qgis.controllers._layers import resolve_layer
        layers = [resolve_layer(self._project, layer_id)
                  for layer_id in (self._state.contour_layer_id,
                                   getattr(self._state, "simple_contour_layer_id", None))]
        return [layer for layer in layers if layer is not None]

    def activate_draw_swale(self, mode):
        contour_layers = self._contour_pick_layers()
        if mode == "contour":
            if not contour_layers:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", self._no_contour_layer_message("segment"))
                return
            tool = ContourSegmentTool(self._canvas, contour_layers)
            tool.segment_selected.connect(
                lambda geom, elev, coords: self._on_contour_selected_for_swale(
                    geom, elev, coords
                )
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self.use_tool(tool)
        elif mode == "full_contour":
            if not contour_layers:
                self._iface.messageBar().pushWarning(
                    "TerrainFlow Assessment", self._no_contour_layer_message("contour"))
                return
            tool = SelectContourTool(self._canvas, contour_layers)
            tool.contour_selected.connect(
                lambda geom, elev, coords: self._on_contour_selected_for_swale(
                    geom, elev, coords
                )
            )
            tool.cancelled.connect(self._on_draw_cancelled)
            self.use_tool(tool)
        else:
            tool = DrawLineTool(self._canvas,
                                slope_band=self._state.slope_band(),
                                tool_label="swale")
            tool.line_drawn.connect(lambda geom: self._on_geometry_drawn("swale", geom))
            tool.cancelled.connect(self._on_draw_cancelled)
            self.use_tool(tool)

    def _no_contour_layer_message(self, what):
        """Say which precondition is missing, not just that one is.

        "Run contour analysis first" is unhelpful to someone who has just run the
        *baseline* analysis — they are different buttons on different stages, and a
        deleted layer looks identical from here unless the two cases are separated.
        """
        if (getattr(self._state, "contour_layer_id", None)
                or getattr(self._state, "simple_contour_layer_id", None)):
            return (f"The contour layers have been removed from the project, so there "
                    f"is nothing to pick a {what} from. Generate Contours, or re-run "
                    f"Contour Analysis, on the Analysis stage.")
        return (f"Generate Contours or run Contour Analysis on the Analysis stage "
                f"first, then pick a {what}. Either layer can be drawn on; the "
                f"Baseline run does not produce contours on its own.")

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
                            slope_band=self._state.slope_band(),
                            tool_label=ew_type)
        tool.line_drawn.connect(lambda geom: self._on_geometry_drawn(ew_type, geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self.use_tool(tool)

    def activate_draw_polygon(self, ew_type):
        tool = DrawPolygonTool(self._canvas,
                               slope_band=self._state.slope_band(),
                               tool_label=ew_type)
        tool.polygon_drawn.connect(lambda geom: self._on_geometry_drawn(ew_type, geom))
        tool.cancelled.connect(self._on_draw_cancelled)
        self.use_tool(tool)

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
        self.use_tool(tool)
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
        **along** the feature at the recorded point.

        Along, not across. A weir's crest is the line the flow crosses on its way
        out, so it lies along the bank; across the bank is the direction the water
        travels. That was the wrong way round while this was only a map symbol, and
        it is the geometry the burn now cuts — see
        :func:`~terrainflow_assessment.modules.plan_geometry.crest_bar`.

        Returns ``(QgsGeometry, bearing_rad)``, or ``None`` when the point is not
        on the feature — which is a statement worth making rather than papering
        over, since it means the crest was sized from the wrong ground.
        """
        from terrainflow_assessment.modules.plan_geometry import crest_bar

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
            result = crest_bar(
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

    def _spillway_sills(self, earthworks=None, warn=False):
        """``{earthwork id: crest-bar WKT}`` for every sited **outflow** spillway.

        What the burn is given so it can cut the notches (``DEMBurner.burn_earthworks``
        takes it as ``sills=``). The snap stays here rather than moving into the burner:
        ``plan_geometry``'s docstring forbids a second implementation of *nearest point
        on this alignment*, QGIS already answers it, and the burner receives only
        ``Earthwork`` objects. So the controller resolves the bar and the burner cuts it.

        Outflows only, and the reason is the same one ``_refresh_auto_spillway_widths``
        gives: an inlet is a protected entry, not a weir, and notching one would open the
        bank at the point water arrives and drain the pond through its own inlet.

        *warn* pushes the orphaned-sill message. A recorded point more than
        ``_SILL_ORPHAN_TOLERANCE_M`` from its feature used to cost a map symbol; now it
        costs the cut, which is a change the user has to be told about rather than left
        to notice in a capacity figure. Off by default because this is called from the
        live tier too, and a message bar that repaints on every edit is not a warning.
        """
        ews = (self._state.earthwork_manager.get_enabled()
               if earthworks is None else earthworks)
        sills, orphans = {}, []
        for ew in ews:
            spillway = getattr(ew, "spillway", None)
            if spillway is None or not spillway.point_wkt:
                continue
            if spillway.crest_elevation is None:
                continue
            point = QgsGeometry.fromWkt(spillway.point_wkt)
            if point is None or point.isEmpty():
                continue
            sill = self._spillway_sill(ew, point, spillway.width_m or 0.0)
            if sill is None:
                orphans.append(ew.name)
                continue
            sills[ew.id] = sill[0].asWkt()
        if warn and orphans:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"{', '.join(orphans)}: the recorded spillway point is not on the "
                f"feature, so no notch was cut into the terrain for it and the model "
                f"still routes the overflow over the bank. Place the spillway again.",
            )
        return sills

    @staticmethod
    def _sill_point(ew, kind="outflow"):
        """Where this feature's spillway is sited, as a ``QgsPointXY``, or ``None``.

        The datums are read locally round this point once it exists (see
        ``_spillway_datums``), so "not sited yet" and "sited over there" have to be
        answerable before any DEM work starts.
        """
        from qgis.core import QgsGeometry

        attr = "inflow_spillway" if kind == "inflow" else "spillway"
        spillway = getattr(ew, attr, None) if ew is not None else None
        wkt = getattr(spillway, "point_wkt", None) if spillway is not None else None
        if not wkt:
            return None
        try:
            geom = QgsGeometry.fromWkt(wkt)
            return None if geom is None or geom.isEmpty() else geom.asPoint()
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

    def set_spillway_depth(self, index, depth_m):
        """Set a feature's sill depth from the review table, designing one if it has none.

        The second write path to a crest, after the properties dialog, and it holds the
        same rules: the depth is bound through :func:`bind_crest` so the stored triple
        cannot describe two containing-ground levels; no band, so the depth typed is the
        depth stored; and ``auto`` is retired, because a level the user set by hand is the
        user's and the placement path is entitled to re-seed one that is still the tool's.

        The datums are measured here rather than read off the cached review row. A
        controller reading its own UI cache to make a design decision is a loop that is
        eventually wrong, and a row can predate a burn; measuring means the display and
        the write share one rule, which is what makes typing back the figure already on
        screen a no-op rather than a nudge.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            Spillway,
            bind_crest,
            spillway_policy,
        )

        earthworks = self._state.earthwork_manager.get_all()
        # Bounds only. Deliberately not a cross-check against
        # `state.spillway_rows[index]`: that list is filtered to the types that can
        # spill and *then* stamped with the manager index, so a berm sitting before a
        # swale makes position and index disagree.
        if not (0 <= index < len(earthworks)):
            return
        ew = earthworks[index]
        if ew.type not in self.SPILLWAY_TYPES:
            return

        spillway = getattr(ew, "spillway", None)
        _lip, invert, containment, _src = self._spillway_datums(
            ew.geometry, ew.type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=getattr(ew, "crest_elevation", None),
            ew=ew, sill_point=self._sill_point(ew),
            sill_width_m=None if spillway is None else spillway.width_m,
        )
        if containment is None:
            # The cell should not have been editable, so this is defence rather than a
            # path. Said out loud all the same: `_build_spillway_rows` swallows its
            # exceptions to a console print, and a typed value that disappears without a
            # word is the worst thing this column can do.
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"No ground level for {ew.name} yet, so there is nothing to measure a "
                f"sill depth against. Run Baseline, or set the depth in the feature's "
                f"properties.",
            )
            return

        created = spillway is None
        if created:
            # The type's head, not the constructor's 0.30 - a swale's is 0.15, and a sill
            # created here carrying twice the head the dialog gives it would be a second
            # design for one gesture. `freeboard_m` stays None *because* the requirement
            # is the type's policy: None is what means "take the type's", and writing
            # today's figure in freezes it into the design, so a later change to the
            # standard would reach new features and silently skip saved ones.
            spillway = Spillway(
                head_m=spillway_policy(ew.type)[1],
                width_m=0.0, width_auto=True,     # _refresh_auto_spillway_widths fills it
                freeboard_m=None,
                point_wkt=None,                   # designing one is not siting one
                auto=False,
            )
            ew.spillway = spillway

        # No band. Clamping would put the column at odds with the number typed into it -
        # type 0.10 and read back 0.60 - which is worse in a cell than in a dialog,
        # because the corrected figure appears where the user's just was. The placement
        # path clamps because a map click lands on arbitrary ground; a typed depth does
        # not, and `spillway_validity` is what says a shallow sill is a bad one.
        #
        # A dam is given no invert: its floor is the ground under the wall rather than a
        # cut, and a height set out from it is a figure the dialog refuses to offer.
        crest, drop, height = bind_crest(
            containment, drop=max(0.0, float(depth_m)),
            invert_elevation=None if ew.type == "dam" else invert)
        # Stored unrounded. Every control shows the crest to the centimetre and the
        # review re-derives the depth from it, so rounding here is what would make the
        # figure on screen drift off the figure typed.
        spillway.crest_elevation = crest
        spillway.drop_below_rim_m = drop
        spillway.height_above_floor_m = None if ew.type == "dam" else height
        spillway.auto = False

        self._reapply_sill_capacity(ew, created=created)
        self._panel.update_earthwork_in_list(index, ew.summary())
        # Recompute before the layer, not after: a spillway created here has no width
        # until `_refresh_auto_spillway_widths` derives one, and the map label draws it.
        self._recompute_live_assessment()
        self._refresh_spillway_layer()
        self._mark_design_edit()

    def _reapply_sill_capacity(self, ew, created=False):
        """Re-read what *ew* holds to its crest, off the curve already measured.

        A crest move cannot change the stage-storage curve, the measured spill level or
        the brim volume: the flood behind all three is deliberately brim-full with the
        notch not cut, for the reason :meth:`_apply_measured_levels` gives. The only
        thing that reads the crest is :meth:`_sill_limited_capacity`, an interpolation on
        the cached curve.

        So this is not an optimisation to be tidied back into a
        :meth:`_refresh_terrain_capacity` call. That would spend a depression fill per
        typed depth to arrive at the same two numbers - and worse, it returns *silently*
        while a burn holds the burner, so mid-burn it would leave the capacity stale with
        nothing said.

        The one case that must flood is a dam getting its first spillway:
        :meth:`_refresh_dam_stage_storage` clears the measured levels and returns for a
        dam that has none, so until this moment there was no curve to read.
        """
        if created and ew.type == "dam":
            self._refresh_terrain_capacity(ew)
            return
        brim = getattr(ew, "containment_capacity_m3", None)
        if brim is None or getattr(ew, "stage_storage", None) is None:
            # Never measured. Nothing to re-read, and flooding here would put a
            # depression fill behind every edit on a design nothing has analysed.
            return
        held = round(self._sill_limited_capacity(ew, brim), 2)
        if ew.type == "dam":
            ew.capacity_m3, ew.capacity_l = held, held * 1000.0
            ew.terrain_capacity_m3 = float(held or 0.0) or None
        else:
            ew.terrain_capacity_m3 = held or None

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

        # A crest this design already carries is left alone — placing the point tells us
        # where, not how deep. Only a spillway that has never had a crest takes one from
        # the ground it landed on, which is the sill created by this click itself.
        #
        # ``spillway.auto`` used to be a second way in here, and it was the hole the
        # dialog's `_note_spillway_edit` could not close. That flag is only retired when
        # the user *moves* one of the three level controls, so a sill configured any other
        # way — opening the group and accepting the depth it seeds, or setting a built
        # width and nothing else — was still marked auto and was overwritten the moment it
        # was placed. The user's crest came back as whatever ground was under the cursor,
        # clamped to the top of the band, and every figure derived from it (depth,
        # freeboard, storage given up) came back with it. A crest that exists is a crest
        # somebody accepted; where it came from does not change that.
        if elevation is not None and spillway.crest_elevation is None:
            # Datums taken round the point that was just placed, so the lip is the lip
            # of the excavation *there* rather than at whichever end of the feature is
            # lowest.
            lip, invert, containment, _src = self._spillway_datums(
                ew.geometry, ew.type,
                top_width_m=getattr(ew, "top_width_m", None),
                depth=getattr(ew, "depth", None),
                crest_elevation=getattr(ew, "crest_elevation", None),
                ew=ew, sill_point=point,
                sill_width_m=getattr(spillway, "width_m", None),
            )
            # Through the band, not raw. Without it a click was recorded at whatever
            # elevation happened to be under the cursor, including above the ground that
            # contains the feature — harmless while a spillway moved no terrain, and a
            # notch cut at the wrong level the moment one does.
            #
            # Outflow only. The band is `containment − head − freeboard`, which is a
            # statement about a weir passing its design nappe; an inlet is a protected
            # entry with no head to pass, and holding one under a weir's ceiling would
            # move the recorded entry point away from the ground the user clicked on.
            band = (self._crest_band_for(ew, spillway, containment, invert)
                    if kind != "inflow" else None)
            crest, drop, height = bind_crest(
                containment, crest=float(elevation), band=band,
                invert_elevation=None if ew.type == "dam" else invert)
            spillway.crest_elevation = crest
            spillway.drop_below_rim_m = drop
            spillway.height_above_floor_m = height
            if crest is not None and abs(crest - float(elevation)) > 0.005:
                self._iface.messageBar().pushInfo(
                    "TerrainFlow Assessment",
                    f"{ew.name}: the ground there is {float(elevation):.2f} m, which "
                    f"leaves no room for the design head and freeboard — the crest was "
                    f"set to {crest:.2f} m, the highest this feature can offer.")
        setattr(ew, attr, spillway)

        # Siting an **outflow** now moves terrain, so what this feature holds has to be
        # measured again. It bought nothing while a spillway changed no raster, which is
        # why this was left out; from the notch onward, skipping it leaves the panel
        # showing a capacity taken against an unbreached bank — 2,133 m³ against 1,277 on
        # Dam 15 of the Quail Island design. An inlet is not cut, so it is not remeasured.
        if kind != "inflow":
            self._refresh_terrain_capacity(ew)
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
            crs = dem_crs(self._state)
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

            # at_top: a spillway sits ON a swale, so it has to paint over that
            # swale's band. Inserted in place rather than re-sorted afterwards —
            # see the note on _groups.add_layer().
            self.place(layer, G.DRAWN, at_top=True)
            self._state.spillway_layer_id = layer.id()
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
        self.use_tool(tool)
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

    # ------------------------------------------------------------------
    # Spillway links — a diversion drain that starts where another feature spills
    # ------------------------------------------------------------------

    def _refresh_spillway_link_inverts(self):
        """Re-derive every linked drain's ``invert_start_m``; return what dangled.

        The link is stored, the level it resolves to is not — same rule
        ``terrain_capacity_m3`` follows, and for a sharper reason: the datum is another
        feature's crest, so a level frozen into a project file outlives the design that
        produced it and would go on grading a drain from a crest that has since moved.

        Returns ``[(drain name, why)]`` for the links that could not be resolved, so a
        caller about to burn can say what fell back to a ground sample. The burner
        cannot say it: a link is resolved out here, and it receives only ``Earthwork``
        objects with one derived number on them.
        """
        manager = self._state.earthwork_manager
        if manager is None:
            return []
        from terrainflow_assessment.modules.earthwork_design import (
            resolve_spillway_links,
        )

        all_ews = manager.get_all()
        inverts, dangling = resolve_spillway_links(all_ews)
        for ew in all_ews:
            ew.invert_start_m = inverts.get(getattr(ew, "id", None))
        return dangling

    def _warn_dangling_spillway_links(self):
        """Refresh the link levels and say which drains fell back to a ground sample.

        Pushed at burn time, beside the orphaned-sill warning and for the same reason: a
        Verify run is the moment the user is asking what the terrain does, and a link
        that has quietly stopped resolving changes the answer without changing anything
        they can see. Not pushed from the live tier — a message bar that repaints on
        every edit is not a warning.
        """
        dangling = self._refresh_spillway_link_inverts()
        if not dangling:
            return
        detail = "; ".join(f"{name} — {why}" for name, why in dangling)
        self._iface.messageBar().pushWarning(
            "TerrainFlow Assessment",
            f"{detail}. These drains were graded from the ground under their own "
            f"alignment instead, which is what they did before they were linked. "
            f"Link them again, or accept the sampled level.",
        )

    def _drains_linked_to(self, ew):
        """Names of the diversion drains taking their start level from *ew*'s spillway.

        Read from the drains rather than kept on the source, because the link lives on
        the drain and one spillway can feed several. Used twice: as a note on the
        Spillways review, and to name what a spillway removal costs before it happens.
        """
        from terrainflow_assessment.modules.earthwork_design import parse_spillway_link

        manager = self._state.earthwork_manager
        key = getattr(ew, "id", None)
        if manager is None or key is None:
            return []
        names = []
        for other in manager.get_all():
            link = parse_spillway_link(getattr(other, "spillway_link_id", None))
            if link is not None and link[0] == key and link[1] == "outflow":
                names.append(other.name)
        return names

    def activate_link_drain_to_spillway(self):
        """Give a diversion drain its start level from another feature's spillway.

        Two clicks, the shape :meth:`activate_connect_earthworks` already uses: the end
        of the drain that attaches, then the feature whose spillway supplies the level.
        Clicking the **end** rather than the drain is what settles which way the drain
        grades — ``_burn_diversion`` runs its grade down from the linked end, and a drain
        graded from the wrong one runs uphill from an entirely plausible-looking level.

        Only sited outflow spillways are offered. An inlet is where water arrives, so a
        drain attached to one is delivering rather than taking and its start level is at
        its far end; and an unsited spillway has no crest to start from.
        """
        from terrainflow_assessment.modules.earthwork_design import parse_spillway_link

        drains, sources = [], []
        for ew in self._state.earthwork_manager.get_all():
            if not ew.enabled:
                continue
            if ew.type == "diversion":
                link = parse_spillway_link(getattr(ew, "spillway_link_id", None))
                drains.append((ew.id, ew.name, ew.geometry, link is not None))
            spillway = getattr(ew, "spillway", None)
            if (spillway is not None and spillway.point_wkt
                    and spillway.crest_elevation is not None):
                # The sill **point**, not the feature it sits on. The two are within a
                # few metres of each other, so hit-testing the feature meant a click
                # aimed at an outflow could land on the feature carrying it and be
                # right by accident — and could not tell an outflow from an inlet on
                # the same bank, which is the pair this link has to distinguish.
                point = self._sill_point(ew)
                if point is not None:
                    sources.append((ew.id, ew.name, point,
                                    spillway.crest_elevation))

        if not drains:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Draw a diversion drain first — this links one to a spillway so it "
                "starts at that crest instead of at the ground under its own line.",
            )
            return
        if not sources:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "No outflow spillway is sited yet. Place one on the map — a drain "
                "takes its start level from the crest, so there has to be a crest "
                "to take.",
            )
            return

        from terrainflow_assessment.map_tools.link_spillway_tool import (
            LinkSpillwayTool,
        )

        tool = LinkSpillwayTool(self._canvas, drains, sources)
        tool.link_made.connect(self.on_spillway_link_made)
        tool.drain_picked.connect(
            lambda name, end: self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{name}'s {end} end starts at… click the outflow spillway that feeds "
                f"it — the sills are marked. Esc to undo.",
            )
        )
        tool.cancelled.connect(self._on_draw_cancelled)
        self.use_tool(tool)
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "Click the end of the drain that attaches — both ends of every drain are "
            "marked — then the outflow spillway it takes its level from.",
        )

    def on_spillway_link_made(self, drain_id, end, source_id):
        """Accept a drain → spillway link unless it is a self-link or closes a loop.

        **Repeating the same link removes it.** There is no undo stack anywhere in this
        plugin, and a mis-clicked link is otherwise unrecoverable without a second piece
        of UI; making the identical gesture the way back keeps the action to one menu row
        and one rule. Linking the same drain to a *different* spillway still just moves
        it, which is the common correction.

        A cycle is refused rather than assumed impossible: nothing on the model stops a
        diversion carrying a spillway of its own, so drain A can be told to start at
        drain B's crest while B starts at A's. The refusal names the loop, in the shape
        :meth:`on_connection_made` already uses.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            format_spillway_link,
            spillway_link_cycle,
        )

        by_id = {ew.id: ew for ew in self._state.earthwork_manager.get_all()}
        drain, source = by_id.get(drain_id), by_id.get(source_id)
        if drain is None or source is None:
            return

        if drain_id == source_id:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"{drain.name} cannot start at its own spillway — a drain has to take "
                f"its level from something upstream of it.",
            )
            return

        # The tool only offers sources with a sited crest, but this is a signal handler
        # and the check is one line: a link to a spillway with nothing to start from
        # would resolve to nothing on every burn and say so every time.
        spillway = getattr(source, "spillway", None)
        crest = None if spillway is None else spillway.crest_elevation
        if crest is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"{source.name} has no spillway crest to start from. Place its "
                f"overflow first, then link the drain to it.",
            )
            return

        wanted = format_spillway_link(source_id, "outflow", end)
        if getattr(drain, "spillway_link_id", None) == wanted:
            drain.spillway_link_id = None
            self._after_spillway_link_change()
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{drain.name} is no longer linked to {source.name} — it goes back to "
                f"grading from the ground under its own alignment.",
            )
            return

        broken = spillway_link_cycle(
            self._state.earthwork_manager.get_all(), extra=(drain_id, source_id))
        if broken:
            names = " → ".join(
                by_id[i].name for i in broken if i in by_id) or "these features"
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                f"{drain.name} → {source.name} would create a loop ({names}). "
                f"A drain cannot start where it ends.",
            )
            return

        drain.spillway_link_id = wanted
        self._after_spillway_link_change()
        self._iface.messageBar().pushSuccess(
            "TerrainFlow Assessment",
            f"{drain.name}'s {end} end now starts at {source.name}'s spillway crest "
            f"({crest:.2f} m). Run this on the same pair again to unlink it.",
        )

    def _after_spillway_link_change(self):
        """One place for what a link change has to bring back into step."""
        self._refresh_spillway_link_inverts()
        self._build_spillway_rows()
        self._recompute_live_assessment()
        self._mark_design_edit()

    def on_usable_area_source_changed(self, source):
        from shapely.ops import unary_union

        from terrainflow_assessment.qgis.adapters.geom import polygons_in_dem_crs

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
            # ``_state.usable_polygon`` is always in the DEM's CRS — see its comment on
            # PluginState. Reading the layer's own coordinates and handing them to the
            # raster tier is how every contour came to be silently discarded.
            polys = polygons_in_dem_crs(layer, dem_crs(self._state))
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

        # The size this feature will be seeded at, resolved once and used twice. The
        # catchment below is labelled against the feature's footprint width, and it runs
        # before the Earthwork exists — so reading the width separately in each place is
        # how the number on screen comes to describe a different swale than the one being
        # built. One resolve makes them agree by construction.
        dims = self._resolved_dims(ew_type)

        # Direct contributing catchment for the not-yet-added feature: label the site
        # as if it were already there, so the dialog opens with real numbers.
        peak_inflow, catchment_m2 = self._provisional_catchment(
            ew_type, geometry, top_width_m=dims.top_width_m)

        crest_elev = None
        if ew_type == "dam" and self._state.dem_path:
            crest_elev = self._default_crest_elevation(geometry)

        n = len(self._state.earthwork_manager) + 1
        ew_name = f"{ew_type.capitalize()} {n}"
        # Seeded from `core/registry` overlaid by the user's own standard, and still
        # NOT wired live to the panel's swale cross-section boxes. Those live in the Find
        # Best Swale Segments criteria and answer a different question — "what size of
        # swale should these contour segments be sized for".
        #
        # The original reason for keeping them apart no longer holds and should not be
        # quoted back: it was that the criteria defaults described a 0.6 m drainage swale,
        # sub-cell on a 1 m DEM and impossible to burn or verify, so wiring them replaced
        # a representable default with one the terrain model cannot hold. Those defaults
        # have since been realigned to the registry's floored trench (see the comment on
        # `_swale_width_spin` in panel.py), so both now describe the same swale. What
        # remains is a question of *meaning*, not of magnitude — a ranking input is not a
        # build dimension — so the criteria grid is seeded from the same standard rather
        # than driven by it, and a user who edits one has not silently edited the other.
        ew = Earthwork(ew_type, geometry, ew_name, dims=dims)
        ew.source_contour_coords = source_contour  # reshape stays contour-locked

        lip, invert, containment, containment_src = self._spillway_datums(
            geometry, ew_type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=crest_elev,
            ew=ew,
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
            containment_elevation=containment,
            lip_elevation=lip,
            containment_source=containment_src,
            invert_elevation=invert,
            # A feature that has not been drawn yet has not been flooded either, so
            # there is no curve and the readout says so rather than printing a zero.
            stage_storage=None,
            peak_flow_m3s=self._provisional_peak_flow(catchment_m2),
            harvesting_coefficient=self._using_harvesting_coefficient(),
            cell_size_m=self._dem_cell_size_m(),
            # Only a feature being drawn may offer to set the standard. Note this
            # cannot be left to the dialog's own `_editing`, which is True here too —
            # the create path passes a constructed Earthwork, so the dialog cannot
            # tell a fresh feature from an old one on its own.
            is_new=True,
            standard_dims=self._earthwork_defaults.get(ew_type),
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
                # A berm open at its ends impounds nothing on ground that falls along
                # the swale, so this decides whether the companion berm is real storage.
                ew.key_into_banks = getattr(
                    dlg, "get_key_into_banks", lambda: True)()
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
            self._apply_spillway_from_dialog(ew, dlg)
            # Apply the bottom width (channels only; None otherwise) — the canonical
            # cross-section field that drives capacity and the burn footprint.
            bw = getattr(dlg, "get_bottom_width", lambda: None)()
            if bw is not None:
                ew.bottom_width_m = bw

            # Every dimension is final by here, so this is the moment the feature can
            # be offered as the standard for its type. Create path only — editing a
            # feature drawn months ago must not rewrite what new ones start at.
            if getattr(dlg, "get_save_as_standard", lambda: False)():
                self._remember_standard_dims(ew)

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
            self._refresh_terrain_capacity(ew)
            self._state.earthwork_manager.add(ew)
            self._panel.add_earthwork_to_list(
                len(self._state.earthwork_manager) - 1, ew.summary()
            )
            self._refresh_ew_layer()
            self.recompute_catchments()
            # Before the sill bar is drawn, not after. An auto width is settled by
            # `_build_spillway_rows`, at the sill depth this recompute measures — so a
            # layer built first carries the width the previous, uncapped pass guessed,
            # and the label disagrees with the Spillways list until some unrelated later
            # edit repaints it.
            self._recompute_live_assessment()
            self._refresh_spillway_layer()
            self._mark_design_edit()
        self._canvas.unsetMapTool(self._canvas.mapTool())

    def _apply_spillway_from_dialog(self, ew, dlg):
        """Take the spillway off the dialog — and ask before throwing one away.

        ``get_spillway()`` returns ``None`` when the group's tick is cleared, and that
        tick is the feature's data rather than a disclosure arrow: clearing it is how a
        user says "no spillway here". The trouble is that it takes the sited location,
        the crest and the width with it, and unticking a group is a very small gesture
        for a very large deletion — one that is silent, and that no undo stack exists
        anywhere in this plugin to recover from.

        So a spillway that has been **sited** is confirmed before it goes, and the
        question names what is lost so it can be answered without reopening anything.
        Declining keeps what was there; nothing else on the dialog is affected either
        way, because everything else has already been read back by the time this runs.
        """
        from qgis.PyQt.QtWidgets import QMessageBox

        new = getattr(dlg, "get_spillway", lambda: None)()
        old = getattr(ew, "spillway", None)
        if new is None and old is not None and getattr(old, "point_wkt", None):
            where = ("" if old.crest_elevation is None
                     else f" sited at {old.crest_elevation:.2f} m")
            # Naming the drains, because removing the spillway does not remove their
            # links — those resolve at read time and simply stop resolving. Without this
            # sentence the user answers a smaller question than the one being asked: the
            # drains go on looking linked and go back to grading from sampled ground.
            drains = self._drains_linked_to(ew)
            cost = ""
            if drains:
                verb = "takes its" if len(drains) == 1 else "take their"
                cost = (f"\n\n{', '.join(drains)} {verb} start level from this crest. "
                        f"Removing it leaves the link dangling, and the drain is cut "
                        f"from the ground under its own line instead.")
            answer = QMessageBox.question(
                self._iface.mainWindow(),
                "Remove this spillway?",
                f"{ew.name} has an overflow{where}, {old.width_m:.1f} m wide, placed on "
                f"the map. Clearing the Spillway tick removes it — the location, the "
                f"crest and the width all go.{cost}\n\nRemove it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        ew.spillway = new
        # A whole new object, so the crest it carries is not the one the cached depth was
        # measured against. Cleared rather than recomputed: the settled recompute that
        # follows this call measures it again, and a stale value in between would cap the
        # auto width against a sill that no longer exists.
        ew.measured_sill_depth_m = None

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
        lip, invert, containment, containment_src = self._spillway_datums(
            ew.geometry, ew.type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=getattr(ew, "crest_elevation", None),
            ew=ew, sill_point=self._sill_point(ew),
            sill_width_m=getattr(getattr(ew, "spillway", None), "width_m", None),
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
            containment_elevation=containment,
            lip_elevation=lip,
            containment_source=containment_src,
            invert_elevation=invert,
            # Measured when the feature was last flooded — the whole reason the crest
            # control can say what it is giving up without re-flooding on every spin.
            stage_storage=getattr(ew, "stage_storage", None),
            peak_flow_m3s=peak_total,
            upstream_flow_m3s=peak_upstream,
            harvesting_coefficient=self._using_harvesting_coefficient(),
            cell_size_m=self._dem_cell_size_m(),
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
                # A berm open at its ends impounds nothing on ground that falls along
                # the swale, so this decides whether the companion berm is real storage.
                ew.key_into_banks = getattr(
                    dlg, "get_key_into_banks", lambda: True)()
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
            self._apply_spillway_from_dialog(ew, dlg)
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
            self._refresh_terrain_capacity(ew)
            self._panel.update_earthwork_in_list(idx, ew.summary())
            self._refresh_ew_layer()
            # See the draw path: the recompute is what settles an auto width against the
            # measured sill depth, so the bar is drawn from it rather than before it.
            self._recompute_live_assessment()
            self._refresh_spillway_layer()
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

            from terrainflow_assessment.modules.swale_design import sample_elevations

            coords = list(shapely_shape(json.loads(geometry.asJson())).coords)
            if len(coords) < 2:
                return geometry
            # Both ends under one open: two questions about the same raster are one
            # visit to it, and this runs on the path between finishing a line and
            # the properties dialog appearing.
            z0, z1 = sample_elevations([coords[0], coords[-1]],
                                       self._state.dem_path)
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
        self.use_tool(tool)

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
        self._refresh_terrain_capacity(ew)
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
        # The same guard the other three floods open with, and the only one of the
        # four that lacked it. The burn holds `state.burner` on a worker thread for
        # several seconds and this floods the same object, so drawing or reshaping a
        # dam during "Re-analyse with Earthworks" ran `dam_storage` on the GUI thread
        # against it. Two consequences, both silent: `dam_storage` opens with
        # `self.warnings = []`, discarding whatever the worker had accumulated; and
        # `_isolated_burn`'s snapshot/restore puts back a `burned_masks` taken
        # mid-population, so every feature burned after the snapshot loses its mask
        # and `_compute_verification` falls through to the re-derived footprint that
        # biases every Δ negative.
        #
        # Returns the cached figure rather than 0.0 — this method's contract is a
        # volume, and 0.0 would read as "measured, and it impounds nothing". A dam
        # drawn mid-burn has no cache yet, so it defers to 0 and is re-measured by
        # `_on_earthworks_complete` when the run finishes.
        if worker_is_running(self._state, "design_worker"):
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "Dam storage will be measured when the current run finishes.")
            return float(getattr(ew, "capacity_m3", 0.0) or 0.0)
        try:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment", "Computing dam storage…"
            )
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            burner = self._state.burner or DEMBurner(self._state.dem_path)
            baseline_ponding = self._cached_baseline_ponding(
                burner.shape, burner.transform)
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

    # ------------------------------------------------- Terrain capacity (the live basis)

    def _refresh_terrain_capacity(self, ew, quiet=False):
        """Flood *ew* alone and cache what it impounds on ``ew.terrain_capacity_m3``.

        Heavy — a depression-fill — so this belongs on the same tier as
        :meth:`_compute_dam_capacity`: draw, dialog OK, and vertex-edit release. Never the
        drag tier, which keeps whatever was last measured, exactly as a dam does.

        This is what stops the design being oversized. Until now the live "% full" bar was
        driven by the drawn cross-section, so a swale with a companion berm keyed into the
        banks reported *full at this storm* while most of its pond was still empty — Swale
        5 of the Quail Island design read 100% of 440 m³ against a pond of 1,095 m³ — and
        the only thing that would have contradicted it was a Verify run, long after the
        sizing decisions were made.

        A dam is skipped because ``capacity_m3`` **is** this measurement already
        (:meth:`_compute_dam_capacity` floods the same way), so re-flooding would buy an
        identical number at the same price. Degrades to None without a DEM, which puts the
        readout back on the drawn figure and says so.

        **Except once a dam carries a sited spillway.** Then ``capacity_m3`` is the wrong
        figure: it is what the wall holds brim-full, and the sill lets water go well below
        that — worth about 30% of Dam 15's pool on the Quail Island design. The flood
        :meth:`_refresh_dam_stage_storage` already runs for the curve is the one that can
        answer it, so its volume is taken as well rather than a second flood being started,
        and :meth:`_sill_limited_capacity` cuts it back to the sill off the same curve.
        """
        if ew.type == "dam":
            # A wall, and its recorded band is the drawn line rather than the wall's own
            # footprint, so there is no excavation this measurement could honestly claim.
            ew.excavation_m3 = None
            storage = self._refresh_dam_stage_storage(ew)
            if storage is not None:
                ew.capacity_m3 = round(
                    self._sill_limited_capacity(ew, storage.volume_m3), 2)
                ew.capacity_l = ew.capacity_m3 * 1000.0
            ew.terrain_capacity_m3 = float(getattr(ew, "capacity_m3", 0.0) or 0.0) or None
            return
        if not self._state.dem_path:
            ew.terrain_capacity_m3 = None
            # Cleared with the capacity it was measured beside: a figure off a DEM that
            # is no longer loaded outlives the terrain that produced it.
            ew.excavation_m3 = None
            self._clear_measured_levels(ew)
            return
        # The burn holds `state.burner` for several seconds on a worker thread, and
        # this floods the same object. Two floods at once would interleave their
        # `warnings` and share whatever scratch the burner keeps, so an edit made
        # mid-burn leaves its capacity for the run that follows rather than racing
        # this one. `_recompute_live_assessment` still redraws off the cached figures.
        if worker_is_running(self._state, "design_worker"):
            return
        try:
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            burner = self._state.burner or DEMBurner(self._state.dem_path)
            storage = burner.feature_storage(
                ew, baseline_ponding=self._cached_baseline_ponding(
                    burner.shape, burner.transform),
            )
            self._apply_measured_levels(ew, storage)
            ew.terrain_capacity_m3 = round(
                self._sill_limited_capacity(ew, storage.volume_m3), 2) or None
            ew.impounded_above_ground_m3 = round(storage.above_ground_m3, 2)
            ew.retained_depth_m = round(storage.retained_depth_m, 2)
            # The earth this takes out, measured on the same isolated burn that measured
            # the pond — free here, and the only per-feature excavation that is
            # order-independent. `or None` would erase a genuine zero, which a feature
            # drawn entirely on flat ground at sub-cell width really can be.
            ew.excavation_m3 = round(storage.excavation_m3, 2)
            if not quiet:
                self._warn_impoundment(ew)
        except Exception as exc:
            print(f"TerrainFlow Assessment — terrain capacity error: {exc}")
            ew.terrain_capacity_m3 = None
            ew.excavation_m3 = None
            self._clear_measured_levels(ew)

    @staticmethod
    def _apply_measured_levels(ew, storage):
        """Carry the measured containment level, curve and brim volume onto *ew*.

        All three are **derived and never serialised**, the same rule
        ``terrain_capacity_m3`` follows: a level measured off one terrain model, saved
        into a design file and opened against another is a stale answer wearing an
        authoritative face.

        ``level_m`` is where the finished pond was found to let go, and it is the honest
        containment datum once anything has been built — the analytic ring minimum is
        what the hillside offers before the spoil goes anywhere.

        **The flood behind it is deliberately brim-full: the spillway is not cut into
        it.** That is a departure from the first draft of the notch plan, and Stage B is
        what forced it. Once the notch is cut, the pond lets go *at the sill*, so a
        measured level fed back in as the containment datum collapses onto the crest —
        the crest band becomes ``sill − head − freeboard`` and every re-open ratchets the
        crest down by that much, ``spillway_validity`` fails every spillwayed feature for
        having no freeboard, and the "what this sill gives up" readout goes to zero
        because the curve now tops out at the sill. A container's capacity is a fact
        about the container; the spillway is a control on top of it.

        Nothing is lost by measuring it this way. The volume held **to the sill** comes
        off the same curve — ``volume_at(crest)`` — and it agrees with a notched flood to
        within the rounding: measured on Dam 15 of the Quail Island design, a notched
        ``dam_storage`` returns 1,276.6 m³ and the brim-full curve answers 1,276.6 m³ at
        the same level. One flood, both numbers, and the datum stays a datum. The site
        burn still cuts the notch, which is where the rasters, the routing and the
        overtopping check see it.
        """
        level = getattr(storage, "level_m", None)
        ew.terrain_spill_level_m = None if level is None else float(level)
        ew.stage_storage = getattr(storage, "stage_storage", None)
        volume = getattr(storage, "volume_m3", None)
        ew.containment_capacity_m3 = None if volume is None else float(volume)

    @staticmethod
    def _sill_limited_capacity(ew, brim_m3):
        """What *ew* holds **to its spillway crest**, off the brim-full curve.

        The figure the balance, the cascade and the report size against, because once a
        spillway is sited the sill is the control and everything above it is freeboard
        the design has deliberately given away. Without a sited spillway, or without a
        curve to read, the brim volume is the answer — which is what it always was.

        ``min`` rather than the curve's word alone: a crest above the brim holds the
        whole pond and nothing more, and the curve extrapolating past its own top would
        be inventing storage.
        """
        brim = float(brim_m3 or 0.0)
        spillway = getattr(ew, "spillway", None)
        crest = None if spillway is None else spillway.crest_elevation
        curve = getattr(ew, "stage_storage", None)
        if crest is None or curve is None or not getattr(spillway, "point_wkt", None):
            return brim
        held = curve.volume_at(crest)
        return brim if held is None else min(brim, float(held))

    @staticmethod
    def _clear_measured_levels(ew):
        """Drop the measured level, curve, brim volume and sill depth, for when there
        is none.

        The containment family only. Its five callers are all "this measurement is
        unavailable" cases — no DEM, a flood that threw, a dam with no spillway — and
        each of them must keep what the *site* burn measured, which is a different
        pass and still valid. For "the terrain itself changed", which invalidates
        both, see ``Earthwork.clear_terrain_measurements``.
        """
        ew.terrain_spill_level_m = None
        ew.stage_storage = None
        ew.containment_capacity_m3 = None
        # A fourth member of the same family: measured off the terrain, never serialised,
        # and read by the two width solves that cannot measure for themselves.
        ew.measured_sill_depth_m = None

    def _refresh_dam_stage_storage(self, ew):
        """Measure a dam's stage–storage curve, which its capacity figure throws away.

        ``_refresh_terrain_capacity`` skips a dam because ``capacity_m3`` **is** the
        flood already — which is true of the volume and not of the curve, and the curve
        is what the crest control reads. So the flood is repeated here, and only where
        it can pay for itself: a dam that carries a spillway. On one without, nothing
        would read the answer.

        The flood is **brim-full** — the notch is deliberately not cut into it, for the
        reason :meth:`_apply_measured_levels` gives — so the curve it returns is the one
        that can be asked what the feature holds *to the sill* without losing its own
        containment datum.

        Returns the :class:`FeatureStorage` where one was measured, and ``None`` where
        none was, because the caller needs the volume as well as the curve:
        :meth:`_sill_limited_capacity` reads the crest off it and that becomes the dam's
        capacity. ``None`` therefore means *keep what you had*, which is the right answer
        both for a dam with no spillway and for one measured while the burn worker holds
        the burner.
        """
        if getattr(ew, "spillway", None) is None or not self._state.dem_path:
            self._clear_measured_levels(ew)
            return None
        if worker_is_running(self._state, "design_worker"):
            # Both this and the burn hold `state.burner`; see _refresh_terrain_capacity.
            return None
        try:
            from terrainflow_assessment.modules.earthwork_design import DEMBurner
            burner = self._state.burner or DEMBurner(self._state.dem_path)
            storage = burner.dam_storage(
                ew, baseline_ponding=self._cached_baseline_ponding(
                    burner.shape, burner.transform),
                key_into_banks=bool(getattr(ew, "key_into_banks", False)),
            )
            self._apply_measured_levels(ew, storage)
            return storage
        except Exception as exc:
            print(f"TerrainFlow Assessment — dam stage storage error: {exc}")
            self._clear_measured_levels(ew)
            return None

    def _refresh_all_terrain_capacities(self):
        """Measure every feature's pond in one pass — on design open, or a DEM change.

        One flood per feature at ~0.1–0.3 s, so a full design is comparable to a Verify
        run and worth a progress notice. Deliberately **not** serialised into the ``.tfd``:
        a terrain number cached in a project file outlives the terrain that produced it,
        and there is no cheap way to tell that it has.
        """
        ews = [e for e in self._state.earthwork_manager.get_all() if e.enabled]
        if not ews or not self._state.dem_path:
            return False
        if worker_is_running(self._state, "design_worker"):
            # Both this and the burn hold `state.burner`; see _refresh_terrain_capacity.
            return False

        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        burner = self._state.burner or DEMBurner(self._state.dem_path)
        baseline_ponding = self._cached_baseline_ponding(
            burner.shape, burner.transform)
        # A flood per feature at ~0.1-0.3 s: on a forty-feature design that is ten
        # seconds of frozen window, which is why it moved off the GUI thread.
        # Measured here, applied to the features in the completion handler — an
        # Earthwork is plain Python, but so is every other rule in this file about
        # what the worker may touch, and one exception is how the rule stops being
        # one. `feature_storage` reads the burner and nothing else.
        def work(report):
            measured = []
            for i, ew in enumerate(ews):
                report(int(5 + 90 * i / len(ews)),
                       f"Measuring {ew.name} ({i + 1} of {len(ews)})…")
                try:
                    if ew.type == "dam":
                        # `capacity_m3` is already this flood, so a dam is skipped —
                        # except for its stage–storage curve, which that figure throws
                        # away and the crest control needs. Measured here rather than in
                        # the completion handler, because the handler is the GUI thread
                        # and a depression fill per dam is exactly what moved this pass
                        # off it in the first place.
                        if getattr(ew, "spillway", None) is None:
                            measured.append((ew, None))
                            continue
                        measured.append((ew, burner.dam_storage(
                            ew, baseline_ponding=baseline_ponding,
                            key_into_banks=bool(getattr(ew, "key_into_banks", False)))))
                        continue
                    measured.append(
                        (ew, burner.feature_storage(
                            ew, baseline_ponding=baseline_ponding)))
                except Exception as exc:
                    print(f"TerrainFlow Assessment — terrain capacity error: {exc}")
                    measured.append((ew, False))
            return measured

        def _failed(tb):
            # Its own handler: `_on_analysis_error` marks the Verify stage failed,
            # and a measurement that could not be taken is not a verification that
            # came back wrong. The readouts fall back to the drawn figures, which
            # is what they show before any measurement anyway.
            self._panel.set_earthworks_idle()
            print("TerrainFlow Assessment — terrain measurement error:\n" + tb)
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Could not measure terrain storage — the readouts show the drawn "
                "figures. See the Python console.")
            self.recompute_catchments()
            self._recompute_live_assessment()

        worker = TaskWorker(work, label="terrain capacities")
        worker.progress.connect(self._panel.set_earthworks_progress)
        worker.completed.connect(self._on_terrain_capacities_ready)
        worker.error.connect(_failed)
        self._state.design_worker = worker
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            f"Measuring what {len(ews)} features hold on this terrain…")
        self._panel.set_earthworks_progress(1, "Measuring terrain storage…")
        worker.start()
        return True

    def _on_terrain_capacities_ready(self, measured):
        for ew, storage in measured:
            if ew.type == "dam":                    # capacity_m3 IS this figure
                ew.excavation_m3 = None             # a wall: see _refresh_terrain_capacity
                if storage in (None, False):        # not asked for, or it failed
                    self._clear_measured_levels(ew)
                else:
                    # …unless a notch has been cut, in which case the figure saved with
                    # the design was taken against an unbreached wall. This flood is the
                    # one that saw the spillway, so it is the capacity.
                    # See _refresh_terrain_capacity's dam branch.
                    self._apply_measured_levels(ew, storage)
                    ew.capacity_m3 = round(
                        self._sill_limited_capacity(ew, storage.volume_m3), 2)
                    ew.capacity_l = ew.capacity_m3 * 1000.0
                ew.terrain_capacity_m3 = (
                    float(getattr(ew, "capacity_m3", 0.0) or 0.0) or None)
                continue
            if storage is False:                    # measurement failed; say nothing new
                ew.terrain_capacity_m3 = None
                ew.excavation_m3 = None
                self._clear_measured_levels(ew)
                continue
            self._apply_measured_levels(ew, storage)
            ew.terrain_capacity_m3 = round(
                self._sill_limited_capacity(ew, storage.volume_m3), 2) or None
            ew.impounded_above_ground_m3 = round(storage.above_ground_m3, 2)
            ew.retained_depth_m = round(storage.retained_depth_m, 2)
            ew.excavation_m3 = round(storage.excavation_m3, 2)
        for ew, _ in measured:
            self._warn_impoundment(ew)
        self._panel.set_earthworks_idle()
        # The scoring follows the measurement, as it did when this was inline. The
        # capacities are the basis the assessment scores on, and painting first
        # against the drawn figures shows every keyed swale full and then corrects
        # itself — which is the flicker the restore path's own comment rules out.
        self.recompute_catchments()
        self._recompute_live_assessment()

    def _warn_impoundment(self, ew):
        """Surface the retaining-structure advisory for *ew*, if it has become one."""
        from terrainflow_assessment.modules.burn_strategy import impoundment_warning

        msg = impoundment_warning(
            ew.name,
            retained_depth_m=getattr(ew, "retained_depth_m", None),
            above_ground_m3=getattr(ew, "impounded_above_ground_m3", None),
            has_spillway=getattr(ew, "spillway", None) is not None,
        )
        if msg:
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)

    def _cached_baseline_ponding(self, shape, transform=None):
        """Baseline ponding array on the burner's grid, or None if it cannot be lined up.

        A shape match alone was the old test, which is two things at once: it rejected a
        baseline that merely sat on a sub-window of the same grid, and it accepted one
        that happened to share dimensions from a different origin — subtracting cell for
        cell while geographically offset. Pass *transform* and both are answered by
        ``align_to_grid``; without it the shape test is kept, since a caller that cannot
        say where its grid is cannot be told whether another one matches.
        """
        path = (self._state.baseline_result or {}).get("ponding")
        if not path or not os.path.exists(path):
            return None
        try:
            import numpy as np
            import rasterio

            from terrainflow_assessment.modules.dem_loader import align_to_grid
            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                nodata = src.nodata
                src_transform = src.transform
            if nodata is not None:
                arr[arr == nodata] = 0.0
            arr = np.clip(arr, 0.0, None)
            if arr.shape == shape and (transform is None or src_transform == transform):
                return arr
            if transform is None:
                return None
            return align_to_grid(arr, src_transform, transform, shape)
        except Exception:
            return None

    def _natural_ponding_m3(self):
        """Water the bare terrain already holds, over the analysed site.

        Context for the scorecard, not an input to it: the headline scores the design
        only, so runoff that never reaches an earthwork is counted as leaving the site
        whether or not it would settle in a hollow first. Returns 0.0 when no baseline
        has run — the caller hides the line rather than guessing.

        Takes the worker's ``ponded_volume_m3``, which is already clipped to the analysis
        domain, rather than re-summing the raster. Summing the raster answered a
        different question from the one beside it: on the Quail Island run the domain is
        29.2 ha of an 88.1 ha raster, so a line printed under a 57% score described three
        times the ground the score did.
        """
        volume = (self._state.baseline_result or {}).get("ponded_volume_m3")
        try:
            return max(0.0, float(volume))
        except (TypeError, ValueError):
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
            #
            # Terrain capacities are measured before the assessment, not after: they are
            # the basis it scores on, and a first paint against the drawn figures would
            # show every keyed swale full and then quietly correct itself.
            # An auto width is derived, not stored (see ``Spillway.to_dict``), so it
            # has to be recomputed *before* the list, the map label and the sill bar are
            # drawn — all three read ``width_m``, and the live assessment that would
            # otherwise supply it runs after them, or on a worker's completion.
            # A drain's start level is derived from the crest its link names, so it has
            # to be resolved before anything reads it — and after
            # `_rebase_restored_spillways`, which is where a restored crest settles.
            for step in (
                self._rebase_restored_spillways,
                self._refresh_auto_spillway_widths,
                self._refresh_spillway_link_inverts,
                lambda: self._panel.refresh_earthwork_list(manager.get_all()),
                self._refresh_ew_layer,
                self._refresh_spillway_layer,
            ):
                try:
                    step()
                except Exception as exc:
                    print(f"TerrainFlow Assessment — restore step failed: {exc}")

            # Measuring is threaded now, so the two steps that consume it are
            # chained to its completion rather than following it in this list. When
            # nothing is measured there is nothing to wait for and they run here.
            try:
                if not self._refresh_all_terrain_capacities():
                    self.recompute_catchments()
                    self._recompute_live_assessment()
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

    def _rebase_restored_spillways(self):
        """Re-derive every restored spillway's relative figures. The crest does not move.

        ``drop_below_rim_m`` is serialised, and its datum has changed meaning: it used to
        be the lowest bare ground on the ring outside the whole footprint, and it is now
        the level the feature is actually held to, taken locally round the sill where one
        is sited. Every design already on disk therefore carries a drop that no longer
        describes its own crest — off by the height of a companion berm on one feature,
        by the fall along the alignment on another, and by nothing at all on a third.
        None of which is visible, because a drop looks like a setting-out figure whether
        or not it is still true.

        ``crest_elevation`` is the absolute the design is really made of, so it is the
        one thing this must not touch. Recompute the pair from it and leave it alone.

        ``height_above_floor_m`` did not exist before this build and is simply filled in.

        Runs on the restore path only. Editing a feature already re-binds through the
        dialog, and a live edit never had a stale datum to correct.
        """
        manager = self._state.earthwork_manager
        if manager is None or not self._state.dem_path:
            return
        from terrainflow_assessment.modules.earthwork_design import rebase_spillway

        for ew in manager.get_all():
            spillway = getattr(ew, "spillway", None)
            if spillway is None or spillway.crest_elevation is None:
                continue
            _lip, invert, containment, _src = self._spillway_datums(
                ew.geometry, ew.type,
                top_width_m=getattr(ew, "top_width_m", None),
                depth=getattr(ew, "depth", None),
                crest_elevation=getattr(ew, "crest_elevation", None),
                ew=ew, sill_point=self._sill_point(ew),
                sill_width_m=spillway.width_m,
            )
            if containment is None:
                continue
            rebase_spillway(spillway, containment,
                            None if ew.type == "dam" else invert)

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
                # float64: this raster carries resolve_flats' flat gradient in multiples of
                # 1e-5 m, and reading it as float32 would quantise that away above ~600 m
                # elevation — leaving d8_from_dem, which needs a strictly positive drop, to
                # read a true flat and call every cell of it a sink.
                dem = src.read(1).astype("float64")
                transform = src.transform
                nodata = src.nodata
            if nodata is None:
                # A conditioned surface written before the analysis worker started
                # tagging its output. Handing None to d8_from_dem leaves an interior
                # hole looking like ground at -9999 — a ten-kilometre pit that
                # captures the catchment around it and labels it as drained. The
                # session DEM's own sentinel is the right value; the conditioned
                # surface is derived from it and inherits its holes.
                info = self._state.dem_info
                nodata = getattr(info, "nodata", None) if info is not None else None
            cell_w, cell_h = abs(transform.a), abs(transform.e)

            next_flat, is_sink = d8_from_dem(dem, cell_w, cell_h, nodata=nodata)

            domain = None
            dom_path = baseline.get("domain_mask")
            if dom_path and os.path.exists(dom_path):
                with rasterio.open(dom_path) as src:
                    domain = src.read(1) > 0.5
            domain_is_fallback = domain is None or domain.shape != dem.shape
            if domain_is_fallback:
                # No usable site mask, so "the site" becomes the whole tile. This was
                # silent, and it is the denominator of every percentage on the panel:
                # on the Quail Island tile it stands 222 ha of DEM in for a 29 ha
                # block. Carried on the meta so a readout can say so out loud.
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
                # Cached rather than re-summed. _recompute_live_assessment runs on the
                # vertex-drag tier, and one reduce over a multi-million-cell bool array
                # per frame was already enough without the coverage readout adding a
                # second. Lives here so invalidate_flow_cache clears it with the rest.
                "domain_cells": int(domain.sum()),
                "domain_is_fallback": domain_is_fallback,
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
        # `with`, and the read inside it: this was the one bare `rasterio.open(` in
        # the tree, with `src.read(1)` outside the `try`, so a truncated file or a
        # MemoryError left the DEM open for the life of the process — and on Windows
        # an open handle blocks the burn from rewriting it. A failure here returns
        # None, the same answer the open failure already gave, and both callers
        # (`_default_crest_elevation`, `_key_dam_into_banks`) test for it.
        try:
            import rasterio
            with rasterio.open(self._state.dem_path) as src:
                band = src.read(1)
                t, nodata = src.transform, src.nodata
        except Exception:
            return None

        rows, cols = band.shape

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

    # ---- The user's standard earthwork dimensions

    # Per user, per QGIS profile — not per project. "The trough on my tractor" follows
    # the person to every site, and QgsProject.readEntry returns nothing for a project
    # that has never been saved, which is precisely when the first swale is drawn.
    _EARTHWORK_DEFAULTS_KEY = "TerrainFlow/earthwork_defaults"

    def _resolved_dims(self, ew_type):
        """The cross-section a fresh feature of *ew_type* should start at."""
        return resolve_dimensions(ew_type, self._earthwork_defaults)

    def _remember_standard_dims(self, ew):
        """Adopt this feature's cross-section as the standard for its type.

        Storage is sparse at the type level: a feature drawn at the shipped size stores
        nothing, so ticking the toggle on one is how a standard is *cleared* rather than
        a way to pin the user to a number a later release may ship differently.
        """
        from terrainflow_assessment.core.registry.earthwork_defaults import (
            DimensionDefaults,
        )
        prefs = dict(self._earthwork_defaults)
        prefs[ew.type] = DimensionDefaults(
            depth=ew.depth,
            top_width_m=ew.top_width_m,
            bottom_width_m=ew.bottom_width_m,
        )
        # decode(encode(...)) rather than assigning straight through: the sanitising
        # gate is what drops a triple that merely matches the shipped one, so the
        # stored state and what a later session reads back are the same object.
        self._earthwork_defaults = decode_earthwork_defaults(
            encode_earthwork_defaults(prefs))
        self.save_earthwork_defaults()

        stored = self._earthwork_defaults.get(ew.type)
        shipped = shipped_dims(ew.type)
        if stored is None:
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                f"{ew.type_label()} dimensions are back to the standard "
                f"{shipped.depth:.2f} m deep by {shipped.top_width_m:.2f} m wide.",
            )
        else:
            self._iface.messageBar().pushSuccess(
                "TerrainFlow Assessment",
                f"Saved {ew.depth:.2f} m deep by {ew.top_width_m:.2f} m wide as your "
                f"standard {ew.type_label().lower()}. Ones already drawn are unchanged.",
            )

    def save_earthwork_defaults(self):
        try:
            QgsSettings().setValue(
                self._EARTHWORK_DEFAULTS_KEY,
                encode_earthwork_defaults(self._earthwork_defaults))
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not save standard dimensions: {exc}")

    def load_earthwork_defaults(self):
        try:
            text = QgsSettings().value(self._EARTHWORK_DEFAULTS_KEY, "")
            self._earthwork_defaults = decode_earthwork_defaults(text)
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not load standard dimensions: {exc}")
            self._earthwork_defaults = {}

    def seed_swale_criteria_from_standard(self):
        """Start the panel's swale-segment criteria at the user's standard section.

        Only when a standard exists: with none set, the criteria keep the shipped
        values they have always had, so a user who never touches this sees no change.
        """
        if self._earthwork_defaults.get("swale") is None:
            return
        dims = self._resolved_dims("swale")
        try:
            self._panel.seed_swale_criteria(
                dims.depth, dims.top_width_m, dims.bottom_width_m)
        except Exception as exc:
            print(f"TerrainFlow Assessment — could not seed swale criteria: {exc}")

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

        The width written is the **buildable** one: the requirement rounded up to whole
        DEM cells, which is what the burn cuts. The raw requirement travels beside it on
        ``width_required_m`` so the rounding can be explained rather than just applied,
        and so the committed-width path's adequacy checks keep testing the requirement
        rather than the rounded figure.

        The requirement is recorded on **every** outflow, auto or not. On the committed
        path it is the number the shortfall check and the rounding note both quote, and
        recomputing it in each of them is how two surfaces come to disagree.

        Outflows only. An inlet is a protected entry, not a weir; giving it a width from
        the overflow formula would invent a design procedure that does not exist.

        The width is solved at the head the **sill** can pass, not at the type's design
        head — see :func:`sill_limited_head_m`, and see :meth:`_spillway_row`, which shows
        the same figure. This tier does no DEM work by design, so it reads the depth off
        ``ew.measured_sill_depth_m``, which the last settled review measured. Where that
        is ``None`` — never measured, no DEM, a feature toggled off — nothing is capped,
        which is exactly what the review row does with the same absent datum. The settled
        tier corrects whatever this leaves behind (:meth:`_build_spillway_rows`), so a
        depth that moved since the last release costs one release, not a wrong figure.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            adoptable_spillway_width,
            calculate_spillway_width,
            sill_limited_head_m,
            spillway_policy,
        )

        cell = self._dem_cell_size_m()
        for ew in self._state.earthwork_manager.get_all():
            spillway = getattr(ew, "spillway", None)
            if spillway is None:
                continue
            total, _upstream = self._peak_flow_for(ew)
            if not total or total <= 0:
                continue
            head = spillway.head_m or spillway_policy(ew.type)[1]
            required = calculate_spillway_width(
                total, sill_limited_head_m(head, getattr(ew, "measured_sill_depth_m", None)))
            # 0.0 means "these inputs say nothing" — no intensity, no catchment, or a sill
            # with no depth to spill through — not a spillway zero metres wide. Writing it
            # through would persist that fiction, so the last good width is kept instead
            # and `spillway_validity` is what says the sill is unusable.
            if required <= 0:
                continue
            spillway.width_required_m = required
            if spillway.width_auto:
                # `None` where the requirement is wider than the feature: keep the last
                # width that could actually be built rather than committing to one the
                # burn would cut through the ground holding the water in.
                built = adoptable_spillway_width(
                    required, cell_size=cell,
                    feature_length_m=getattr(ew, "length_m", None))
                if built is not None:
                    spillway.width_m = built

    def _spillway_row(self, ew):
        """One review row for *ew*, or None if this type has nothing to spill.

        Built for every water-holding feature, **including those with no ``Spillway``
        object yet**: the required width is a function of the terrain and the storm, not
        of whether the user has ticked a box, and showing it before they commit is the
        whole of "features auto-size their spillways as you add them".
        """
        from terrainflow_assessment.modules.earthwork_design import (
            # Private on purpose, and imported on purpose: it is the millimetre
            # `spillway_validity` makes its own elevation comparisons at, and the gauge
            # below has to fire on exactly the condition the freeboard warning does or
            # the two contradict each other about one feature.
            _ELEV_EPS,
            adoptable_spillway_width,
            calculate_spillway_width,
            default_sill_depth_m,
            effective_freeboard_m,
            effective_head_m,
            sill_limited_head_m,
            spillway_notes,
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
            # The head the required width was actually solved at: the design head, or the
            # sill depth where that is shallower. Carried so every surface that shows a
            # width can say which — a capped width beside an uncapped `actual_head_m` does
            # not reconcile through the weir equation, and the two answer different
            # questions ("how wide to pass this through *this* notch" against "how deep
            # the design wants to run"). The properties dialog has always said so inline;
            # the review table and the message bar had no way to.
            "sizing_head_m": None,
            "built_width_m": None if spillway is None else (spillway.width_m or 0.0),
            "actual_head_m": None,
            "freeboard_m": None,
            "crest_elevation": None if spillway is None else spillway.crest_elevation,
            "height_above_floor_m": (
                None if spillway is None else spillway.height_above_floor_m),
            # Kept under its old name because it is the level everything on this row is
            # measured against, which is what it always meant — but it now carries the
            # *containment* level rather than the bare ring minimum. The ring minimum is
            # `lip_elevation` beside it, and the two differ by whatever the feature is
            # holding up.
            "rim_elevation": None,
            "lip_elevation": None,
            "containment_source": None,
            # The sill's drop below containment, measured against the containment level
            # *this build* found — deliberately not `spillway.drop_below_rim_m`. The
            # stored partner is only re-based on restore, while this datum is recomputed
            # every build, so after a re-analysis the two can describe different rims.
            # Anything editable has to render the live one, or typing back the number on
            # screen would move the crest.
            "sill_depth_m": None,
            # What a feature with no spillway would open at, so the review can offer a
            # starting depth without re-deriving policy. Type policy only: it knows
            # nothing about a per-feature freeboard override, which is correct where it
            # is used — a feature that has no spillway has none to carry.
            "seed_depth_m": default_sill_depth_m(ew.type),
            # What this sill holds, what it would hold with no spillway, and the
            # difference. All None until the feature has been flooded.
            "sill_storage_m3": None,
            "containment_storage_m3": None,
            "given_up_m3": None,
            "given_up_pct": None,
            # Where the water stands while the design storm passes the weir: the sill
            # plus the head the built width actually produces. Between the sill volume
            # and the containment volume is the spillway doing its job; touching the
            # containment level means it is not sufficient.
            "surcharge_level_m": None,
            "surcharge_storage_m3": None,
            "spillway_insufficient": None,
            # The other two of B6's three elevations, both measured off the last burn
            # and both None until one has run. Designed is `crest_elevation` above.
            "burned_sill_m": None,
            "actual_spill_level_m": None,
            # Where the event water actually reached, and whether it ever got to the
            # sill. A spillway the design storm never touches is not *sized wrong* — it
            # is untested, and that is a different thing to tell the user.
            "event_level_m": None,
            "passes_this_event": None,
            # Diversion drains that take their start level from this spillway's crest.
            # Filled before the early returns below, because a disabled or flow-less
            # feature can still be the thing a drain was graded from — and that is
            # exactly when the user needs to know.
            "linked_drains": self._drains_linked_to(ew),
            "problems": [],
            # A parallel channel to `problems`, and it has to be parallel: this row goes
            # to "fail" on any problem at all, so a crest standing legitimately above
            # natural ground would mark every bermed swale failed if it went in there.
            # A link is in the same class — true, fine, and not a fault.
            "notes": [],
            "state": "ok",
        }

        # A disabled feature is out of the catchment labelling entirely, so it has no
        # flow and no meaningful sizing — say so rather than render a row of zeros that
        # reads as a failure.
        #
        # Both early returns clear the measured-depth cache on the way past. It is read
        # by the two tiers that cannot measure for themselves, and a feature toggled off
        # would otherwise keep capping them against a depth nothing computes any more.
        if not ew.enabled:
            ew.measured_sill_depth_m = None
            row["state"] = "disabled"
            return row

        total, upstream = self._peak_flow_for(ew)
        row["peak_flow_m3s"], row["upstream_m3s"] = total, upstream
        if not total or total <= 0:
            ew.measured_sill_depth_m = None
            row["state"] = "no_flow"
            return row

        # The datums come first, and that ordering is the whole of this block. The width
        # is solved at the head this sill can actually pass (`sill_limited_head_m`), which
        # is not knowable until the containment level has been measured — and until it
        # was, this row sized its weir at the design head while the properties dialog
        # sized the same weir at the sill depth, and the two disagreed on screen.
        lip, invert, containment, containment_src = self._spillway_datums(
            ew.geometry, ew.type,
            top_width_m=getattr(ew, "top_width_m", None),
            depth=getattr(ew, "depth", None),
            crest_elevation=getattr(ew, "crest_elevation", None),
            ew=ew, sill_point=self._sill_point(ew),
            sill_width_m=None if spillway is None else spillway.width_m,
        )
        row["rim_elevation"] = containment
        row["lip_elevation"] = lip
        row["containment_source"] = containment_src
        if containment is not None and row["crest_elevation"] is not None:
            row["sill_depth_m"] = containment - row["crest_elevation"]
        # Derived and never serialised, exactly as `terrain_spill_level_m` and
        # `stage_storage` are: `_refresh_auto_spillway_widths` runs on the drag tier and
        # `_check_spillway_capacity` runs without a footprint mask, so neither can call
        # `_spillway_datums` for itself, and this is the measurement they read. `None` is
        # a meaningful value here — it means no cap, which is what this row does too.
        ew.measured_sill_depth_m = row["sill_depth_m"]

        row["sizing_head_m"] = sill_limited_head_m(target_head, row["sill_depth_m"])
        row["required_width_m"] = (
            calculate_spillway_width(total, row["sizing_head_m"]) or None)
        # An auto width is derived rather than stored, so the row derives it too — a row
        # that trusted `width_m` would print 0.0 for every restored design between the
        # load and the first live recompute.
        #
        # Only where a spillway actually exists. `width_auto` reads True for a feature
        # that has none at all (there is no object to ask), so without this guard every
        # undesigned feature reports a built width — and the report prints that straight
        # into its "Width designed" column, beside a status of "No spillway designed".
        # What such a feature has is a *requirement*, which `required_width_m` already
        # carries and which is the whole point of giving it a row.
        if spillway is not None and width_auto and row["required_width_m"]:
            row["built_width_m"] = adoptable_spillway_width(
                row["required_width_m"], cell_size=self._dem_cell_size_m(),
                feature_length_m=getattr(ew, "length_m", None),
            ) or row["built_width_m"]
        built = row["built_width_m"] if spillway is not None else row["required_width_m"]
        # Deliberately `target_head`, not the sizing head. This is the "is the committed
        # width adequate at the head this type designs for" test, and answering it at the
        # sill depth instead would improve the freeboard on exactly the sills that are too
        # shallow — redefining a bad design into compliance, which is the one thing the
        # sill cap must not be allowed to do.
        row["actual_head_m"] = effective_head_m(
            target_head, peak_flow_m3s=total, width_m=built, width_auto=width_auto)
        if row["sill_depth_m"] is not None:
            row["freeboard_m"] = (
                row["sill_depth_m"] - (row["actual_head_m"] or 0.0))

        # What the sill costs, per row, so the whole design can be read at once instead
        # of one dialog at a time. Straight off the curve the last flood already
        # measured — nothing is computed here.
        curve = getattr(ew, "stage_storage", None)
        if curve is not None and containment is not None:
            full = curve.volume_at(containment)
            held = (full if row["crest_elevation"] is None
                    else curve.volume_at(row["crest_elevation"]))
            if full is not None and held is not None:
                row["containment_storage_m3"] = round(full, 1)
                row["sill_storage_m3"] = round(held, 1)
                row["given_up_m3"] = round(max(0.0, full - held), 1)
                row["given_up_pct"] = (
                    100.0 * max(0.0, full - held) / full if full > 0 else None)
            # The surcharge mark: where the water stands while the storm passes the
            # weir. `effective_head_m` already answers "what head does the built width
            # actually produce at this peak", and the curve turns that into a volume —
            # so the gauge's three marks all come off one measurement.
            if row["crest_elevation"] is not None and row["actual_head_m"]:
                level = row["crest_elevation"] + row["actual_head_m"]
                row["surcharge_level_m"] = round(level, 2)
                at = curve.volume_at(level)
                if at is not None:
                    row["surcharge_storage_m3"] = round(min(at, full or at), 1)
                # The same condition `spillway_validity`'s freeboard check fires on, so
                # the gauge and the warning cannot disagree about one feature.
                row["spillway_insufficient"] = bool(
                    level >= containment - _ELEV_EPS)

        # Measured off the last burn, not derived from the design — see
        # `_record_spillway_levels` for what each disagreement means.
        row["burned_sill_m"] = getattr(ew, "burned_sill_elevation_m", None)
        row["actual_spill_level_m"] = getattr(ew, "actual_spill_level_m", None)
        event = self._event_water_level(ew)
        row["event_level_m"] = event
        if event is not None and row["crest_elevation"] is not None:
            row["passes_this_event"] = bool(event > row["crest_elevation"] + _ELEV_EPS)

        row["problems"] = spillway_validity(
            row["crest_elevation"], containment, invert_elevation=invert,
            head_m=row["actual_head_m"] or target_head,
            min_freeboard_m=freeboard,
            width_m=built, required_width_m=row["required_width_m"],
            standard_freeboard_m=policy_freeboard,
            typical_head_m=head_band,
            feature_length_m=getattr(ew, "length_m", None),
        )
        row["notes"] = spillway_notes(
            row["crest_elevation"], lip_elevation=lip,
            containment_elevation=containment,
            containment_source=containment_src,
            berm_crest_elevation=getattr(ew, "berm_crest_elevation", None),
            built_width_m=built, required_width_m=row["required_width_m"],
        )
        # Appended after `spillway_notes`, which returns a fresh list. A note and not a
        # column: it says what else moves if this crest moves, which is the one thing
        # about a linked drain that is invisible from the drain's own row.
        linked = row["linked_drains"]
        if linked:
            takes = "takes its" if len(linked) == 1 else "take their"
            them = "it" if len(linked) == 1 else "them"
            row["notes"].append(
                f"{', '.join(linked)} {takes} start level from this crest, so moving "
                f"the crest re-cuts {them} with it."
            )
        if containment is None:
            row["state"] = "no_datum"
        elif row["problems"]:
            row["state"] = "fail"
        elif not row["designed"]:
            row["state"] = "undesigned"
        elif not row["sited"]:
            row["state"] = "unsited"
        return row

    def _event_water_level(self, ew):
        """Where the design storm's water actually stands in *ew*, or ``None``.

        Off the stage–storage curve and the balance, not off a raster: the balance is
        recomputed on every design edit, and the curve inverts to a level exactly, so
        this answers on the live tier rather than waiting for a burn.

        The point of it is to separate *sized wrong* from *never tested*. A sill the
        modelled event never reaches passes nothing in this run, and nothing else on the
        review says so — the freeboard column reads fine, because the water it is
        measuring against is water that is not there.
        """
        curve = getattr(ew, "stage_storage", None)
        balance = self._state.balance
        if curve is None or balance is None:
            return None
        key = getattr(ew, "id", None)
        for f in (getattr(balance, "per_feature", None) or []):
            if (f.get("id") or f.get("name")) == key:
                stored = f.get("stored_m3")
                if stored is None:
                    return None
                level = curve.level_at(float(stored))
                return None if level is None else round(float(level), 2)
        return None

    def _build_spillway_rows(self):
        """Refresh the Design-stage spillway review.

        Deliberately **not** called from :meth:`_recompute_live_assessment`: that runs on
        every frame of a vertex drag, and each row samples the DEM through
        ``_spillway_datums`` (a footprint rasterisation plus a rim scan). The same
        reasoning already keeps time-of-concentration off that path. Geometry-dependent
        work belongs on the discrete edits, where the geometry has actually settled.
        """
        try:
            manager = self._state.earthwork_manager
            rows = []
            for i, ew in enumerate(manager.get_all()):
                row = self._spillway_row(ew)
                if row is not None:
                    row["index"] = i
                    rows.append(row)
            self._adopt_reviewed_widths(manager, rows)
            context = self._spillway_context()
            # Retained for the report, which has no handle on this controller.
            self._state.spillway_rows = rows
            self._state.spillway_context = context
            self._panel.set_spillway_review(rows, context)
        except Exception as exc:
            import traceback
            print(f"TerrainFlow Assessment — spillway review error: {exc}")
            traceback.print_exc()

    def _adopt_reviewed_widths(self, manager, rows):
        """Write the widths this build solved back onto the spillways that own them.

        :meth:`_refresh_auto_spillway_widths` runs earlier in the same recompute and does
        no DEM work, so it caps against whatever depth the *previous* settled build
        measured. This tier has just measured it again. Without this the stored width lags
        the displayed one by a recompute — the Width column right, and the map label, the
        feature summary and the burn wrong — which is the divergence this whole change
        exists to remove, relocated rather than fixed.

        **A second pass, deliberately.** :meth:`_build_spillway_rows` swallows to a console
        print, so a throw inside the row loop would leave the first *k* features carrying
        new widths and the rest their old ones, *and* discard the rows — model and display
        disagreeing in exactly the way this is meant to prevent. Building first and
        writing after means a failure leaves the model untouched.

        Only an **auto** width is written. A committed one is a decision the user made and
        the only one of the two that is serialised (``Spillway.to_dict``), so nothing here
        can change a saved design. A row that returned before it solved a requirement —
        disabled, no flow — writes nothing at all.
        """
        for row in rows:
            index = row.get("index")
            if index is None or not row.get("required_width_m"):
                continue
            ew = manager.get_all()[index]
            spillway = getattr(ew, "spillway", None)
            if spillway is None:
                continue
            spillway.width_required_m = row["required_width_m"]
            if spillway.width_auto and row.get("built_width_m"):
                spillway.width_m = row["built_width_m"]

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

        The requirement is the one the Spillways list shows, which means it is solved at
        the head the **sill** can pass (:func:`sill_limited_head_m`) off the depth the
        last review measured. Sizing it here at the design head instead put "needs 1.40 m"
        in the message bar in the same frame the list said "needs 2.60 m", for the same
        sill — the divergence this channel exists to report, appearing inside it.

        It fires **only when the shortfall is news.** This runs on every settled
        recompute, so re-pushing the same set of features on every subsequent edit turns a
        change-notification into a permanent nag, and a message bar that repaints on every
        edit is not a warning. The set is remembered and a repeat is dropped; a feature
        joining or leaving it is a change, and says so again.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            calculate_spillway_width,
            head_for_width,
            sill_limited_head_m,
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
            sizing = sill_limited_head_m(
                head, getattr(ew, "measured_sill_depth_m", None))
            required = calculate_spillway_width(total, sizing)
            built = spillway.width_m or 0.0
            if required > built + 0.01:
                # The achieved depth is bounded by the sill for the same reason the width
                # is: water standing deeper than the notch is over the bank, not over the
                # weir. Unbounded, the sentence contradicted itself — "the water runs
                # 0.18 m deep" over a sill 0.02 m deep is not a state the design describes.
                actual = head_for_width(total, built)
                if actual is not None and sizing is not None:
                    actual = min(actual, max(0.0, float(sizing)))
                short.append((ew, built, required, head, sizing, actual, upstream))

        seen = frozenset(ew.id for ew, *_rest in short)
        if not short:
            self._short_spillways_reported = seen
            return
        if seen == getattr(self, "_short_spillways_reported", None):
            return          # already said, and nothing about the set has changed
        self._short_spillways_reported = seen
        parts = []
        for ew, built, required, head, sizing, actual, upstream in short[:3]:
            # Lead with what the water does, not with the shortfall in metres: "two
            # metres short" is hard to act on, "it will run 18 cm deeper than you
            # designed for" is the same fact in the units that decide the outcome.
            if actual is not None:
                note = (f"{ew.name}: at {built:.2f} m the water runs {actual:.2f} m "
                        f"deep, not {head:.2f} m — needs {required:.2f} m")
            else:
                note = f"{ew.name}: built {built:.2f} m, now needs {required:.2f} m"
            # Said out loud when the sill rather than the type's design head is what the
            # width was solved against — the same sentence the properties dialog puts
            # under its Min-width row, because the figure is otherwise unreconcilable
            # with the head named beside it.
            if sizing is not None and sizing < head - 0.005:
                note += f" at the {sizing:.2f} m this sill can pass"
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
        from terrainflow_assessment.modules.earthwork_design import spillway_policy

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

        # The head every width in the table is solved at. The dialog defaults it to
        # 0.30 m — the *embankment* figure — and nothing passed one, so a swale (whose
        # registry head is 0.15 m) had every width understated by (0.30/0.15)^1.5, about
        # 2.8x, under a column headed "Spillway @ 0.30 m" stating it as fact. Where a
        # feature is chosen the table is costed against its catchment, so it should be
        # costed at its head too; where none is, the embankment default stands but is
        # now attributed rather than presented as measured.
        head_m, head_note = 0.30, "embankment default — no feature selected"
        if chosen is not None and chosen in by_id:
            head_m = spillway_policy(by_id[chosen].type)[1]
            head_note = f"{by_id[chosen].type_label().lower()} default"

        scs = SCSRunoff()
        dlg = DesignIntensityDialog(
            parent=self._iface.mainWindow(),
            area_m2=area_m2,
            area_label=label,
            head_m=head_m,
            head_note=head_note,
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

    def _site_is_guessed(self):
        """True when "the site" was not drawn, so the denominator is a stand-in.

        Two independent fallbacks, and they fail at different layers: the worker's
        own (no analysis area and no boundary, reported as ``domain_source``) and
        ``_ensure_flow_graph``'s, which quietly substitutes the whole tile when the
        mask raster is missing or the wrong shape.
        """
        from terrainflow_assessment.modules.footprint import DOMAIN_FROM_POLYGON

        meta = self._state.flow_grid_meta or {}
        if meta.get("domain_is_fallback"):
            return True
        baseline = self._state.baseline_result or {}
        source = baseline.get("domain_source")
        return source is not None and source != DOMAIN_FROM_POLYGON

    def compute_catchment_coverage(self):
        """How much of the site drains into an enabled feature — by **area**.

        The number behind the "which earthwork catches what" layer. Read against the
        scorecard's capture percentage, which is a share of storm *volume*: a design
        can score badly because its features are too small or because most of the
        block drains straight past them, and those call for opposite work.

        Deliberately **not** gated on the live assessment's ``have_flow``
        (``bool(counts) and meta is not None``). With every feature disabled the
        counts are empty and that flag goes False — but the honest answer is then
        "0% of the site", not a blank, and disabling a feature to see what it was
        doing is exactly when the figure is wanted.

        The denominator is ``flow_domain_mask``: the same mask the analysis worker
        measured "Analysed: X ha" over, and the same one ``capture_pct`` divides by.
        Never the DEM extent, and never ``baseline_report.catchment_area_ha`` — a
        second site size in one panel is a second answer to one question.
        """
        from terrainflow_assessment.modules.water_balance import catchment_coverage

        meta = self._state.flow_grid_meta
        domain = self._state.flow_domain_mask
        if meta is None or domain is None:
            return None
        try:
            cell_area = meta["cell_area_m2"]
            domain_cells = meta.get("domain_cells")
            if domain_cells is None:  # meta built before the key existed
                domain_cells = int(domain.sum())
            counts = self._state.catchment_counts or {}
            return catchment_coverage(
                sum(counts.values()) * cell_area,
                domain_cells * cell_area,
                exit_m2=self._state.catchment_exit_cells * cell_area,
                sink_m2=self._state.catchment_sink_cells * cell_area,
            )
        except Exception as exc:
            print(f"TerrainFlow Assessment — catchment coverage error: {exc}")
            return None

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
        return overtopping_station(profile, capacity / length)

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

            crs = dem_crs(self._state)
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

            self.place(layer, G.DRAWN, at_top=True)
            self._state.stress_points_layer_id = layer.id()
        except Exception as exc:
            print(f"TerrainFlow Assessment — stress points layer error: {exc}")

    def _footprint_mask(self, geometry, top_width_m=None, meta=None,
                        all_touched=None):
        """Rasterise a geometry onto the flow grid, buffering lines to their width.

        *meta* names the grid and defaults to the flow grid; the spillway datums pass the
        burner's, so the datum and the cut are read off one surface.

        *all_touched* defaults to ``rasterize_footprint``'s own default, which is True —
        a footprint claims every cell it crosses so a narrow or diagonal feature is never
        lost. The spillway datums pass **False**, on purpose and only there: they are
        looking for the ring *outside* the footprint, and a mask one cell wider than the
        one the burn cuts puts that ring one cell further out, on ground the excavation
        never reaches. The two time-of-concentration callers keep the wide mask, because
        for them the question is which cells the water is travelling through rather than
        which cells get dug.

        **Passing False falls back to True when it would lose the feature**, which is the
        same fallback ``DEMBurner._rasterize`` describes and the burns already carry: a
        section narrower than a cell claims no cell centre at all, and a spillway on one
        would then have no datum whatsoever rather than one measured a cell too far out.
        A slightly wide ring beats no ring — the first is an approximation the caller can
        reason about, the second reports the feature as having no containing ground.
        """
        meta = meta if meta is not None else self._state.flow_grid_meta
        if meta is None:
            return None
        try:
            import json

            from shapely.geometry import shape as shapely_shape

            from terrainflow_assessment.modules.footprint import (
                DEFAULT_ALL_TOUCHED,
                rasterize_footprint,
            )

            shp = shapely_shape(json.loads(geometry.asJson()))
            if shp.geom_type in ("LineString", "MultiLineString"):
                shp = shp.buffer(max(top_width_m or 1.0, 0.1) / 2.0)
            wanted = DEFAULT_ALL_TOUCHED if all_touched is None else bool(all_touched)
            mask = rasterize_footprint(shp, meta["shape"], meta["transform"],
                                       all_touched=wanted)
            if not mask.any() and not wanted:
                mask = rasterize_footprint(shp, meta["shape"], meta["transform"],
                                           all_touched=True)
            return mask if mask.any() else None
        except Exception:
            return None

    def _crest_band_for(self, ew, spillway, containment, invert):
        """``(lowest, highest)`` crest elevations this feature can offer right now.

        The same band the dialog computes, resolved here so the map-placement path and
        the dialog cannot disagree about what a feature can accept — which they did,
        because only one of them asked.

        *spillway* is passed in rather than read off *ew*: the placement path resolves
        the band **before** assigning, so reading the feature would use the head and
        freeboard of whatever spillway is being replaced.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            effective_freeboard_m,
            spillway_datum,
            spillway_policy,
        )

        _policy_freeboard, policy_head, _band = spillway_policy(ew.type)
        head = (spillway.head_m if spillway is not None and spillway.head_m
                else policy_head)
        return spillway_datum(containment, invert, head_m=head,
                              min_freeboard_m=effective_freeboard_m(spillway, ew.type))

    def _burn_surface(self):
        """The DEM the burn takes its own datum from, with its grid — or ``None``.

        ``state.flow_dem`` is the **conditioned** surface: depression-filled, and flat-
        inflated by an epsilon that grows with the size of the flat. The burn's datum is
        ``pour_point(self.original, …)`` off the raw file. Reading a crest against one
        and cutting it into the other is a mismatch measured in centimetres on a good
        tile and in centimetres times a flat's drainage gradient on a bad one, and there
        is no reason to carry it: ``PlacePointTool`` already samples ``state.dem_path``,
        the same file ``DEMBurner.original`` reads.

        Returns ``(dem, meta)`` where *meta* has the keys ``_footprint_mask`` wants.
        """
        path = self._state.dem_path
        if not path:
            return None
        burner = self._state.burner
        if burner is None:
            # `state.burner` is written in exactly one place (`on_dem_changed`), so this
            # is the path where a DEM was adopted some other way. Cached against the DEM
            # it was built from, because `_build_spillway_rows` asks once per feature and
            # re-reading the raster forty times would be paid on every design edit.
            cached = getattr(self, "_datum_burner", None)
            if cached is not None and cached.dem_path == path:
                return cached.original, self._burn_surface_meta(cached)
            try:
                from terrainflow_assessment.modules.earthwork_design import DEMBurner
                burner = DEMBurner(path)
            except Exception:
                return None
            self._datum_burner = burner
        return burner.original, self._burn_surface_meta(burner)

    @staticmethod
    def _burn_surface_meta(burner):
        """``_footprint_mask``'s grid keys for *burner*'s raster.

        ``nodata`` is None on purpose: ``DEMBurner.__init__`` has already replaced the
        sentinel with NaN in ``original``, so the sentinel is spent and passing it on
        would have the readers test for a value that is no longer in the array.
        """
        return {
            "shape": burner.shape,
            "transform": burner.transform,
            "nodata": None,
            "cell_size_m": max(burner.cell_size, burner.cell_h),
        }

    def _dem_cell_size_m(self):
        """The DEM cell the spillway width has to be rounded to, or ``None``.

        The **coarser** axis, matching ``spillway_burn_width``: the crest axis is not
        known at this point and the coarser one is what can fail to resolve the sill.
        Off the **transform**, not ``DEMInfo.cell_size_m`` — that field is the mean of
        the two axes, which on a non-square grid is neither of them and rounds a sill to
        a width no cell has. The burner is the fallback so the dialog still answers
        before a baseline has run.
        """
        info = self._state.dem_info
        t = getattr(info, "transform", None) if info is not None else None
        if t is not None:
            try:
                return max(abs(t.a), abs(t.e)) or None
            except (AttributeError, TypeError):
                pass
        burner = self._state.burner or getattr(self, "_datum_burner", None)
        if burner is not None:
            return max(burner.cell_size, burner.cell_h) or None
        return None

    def _spillway_datums(self, geometry, ew_type, top_width_m=None, depth=None,
                         crest_elevation=None, ew=None, sill_point=None,
                         sill_width_m=None):
        """``(lip, invert, containment, source)`` — the levels a crest sits between.

        Three levels, not two, because the old *rim* was doing two jobs that had come
        apart. It was the ceiling the crest was clamped under **and** the figure reported
        beside it, and it was measured as the lowest bare ground on the ring outside the
        footprint. On anything that holds water above natural ground those are different
        elevations, and using the lower one as the ceiling gave the storage away: a
        bermed swale keyed into its banks ponds to 69.60 m against a ring minimum of
        68.88 m, so 1,095 m³ was clamped down to 439 m³ before the user saw either.

        *lip* is that bare ring minimum, kept and reported — it is what says how much of
        the water is standing on built ground rather than in the hillside.

        *containment* is the level the water is **actually** held to, and it is the one
        the band and the validity checks use. In order of preference: the measured spill
        level from the last flood of this feature; the companion berm's crest as built;
        the wall crest for a dam; else the lip. Each is a measurement or a stated design
        value — never a berm *height estimate*, which is the thing that cannot be trusted
        here.

        *source* names which of those it was, so the dialog and the report can say so
        rather than presenting an estimate and a measurement in the same typeface.

        *invert* is the floor. For a cut feature that is the level-invert datum the burn
        uses (pour point − depth), so the dialog and the DEM agree; for a dam it is the
        lowest ground the wall touches.

        Pass *sill_point* (a ``QgsPointXY``) once a spillway is sited and **the lip alone**
        is taken locally, on the ring within a sill width of that point — which is what
        makes "the sill is the lip of the excavation" true where the user clicked instead
        of true at whichever end of a falling swale happens to be lowest.

        *invert* and the fallback *containment* stay **global**, and that asymmetry is
        the point rather than an oversight. The burn cuts one level bottom for the whole
        feature (``_storage_invert`` is ``pour_point(original, mask) − depth`` over the
        entire footprint), so a floor measured under the sill would be a floor the burn
        never cuts, and "height above floor" would be measured from an imaginary one. And
        a feature with nothing built on it spills at its lowest rim cell wherever that
        cell is, which is a fact about the feature and not about where the user clicked.
        """
        from terrainflow_assessment.modules.earthwork_design import (
            CONTAINMENT_BERM,
            CONTAINMENT_LIP,
            CONTAINMENT_MEASURED,
            CONTAINMENT_WALL,
        )

        surface = self._burn_surface()
        if surface is None:
            return (None, None, None, None)
        dem, meta = surface
        # Cell centres, matching the volumetric burns — see _footprint_mask.
        mask = self._footprint_mask(geometry, top_width_m, meta=meta,
                                    all_touched=False)
        if dem is None or mask is None:
            return (None, None, None, None)
        try:
            import numpy as np

            from terrainflow_assessment.modules.footprint import pour_point, pour_point_near

            # The global ring minimum: the burn's own datum, and the level this feature
            # would spill at with nothing built on it.
            natural, _cell = pour_point(dem, mask)
            if natural is None:
                return (None, None, None, None)
            natural = float(natural)

            centre = None
            if sill_point is not None:
                try:
                    from terrainflow_assessment.modules.footprint import xy_to_rc
                    centre = xy_to_rc(meta["transform"], sill_point.x(), sill_point.y())
                except Exception:
                    centre = None

            lip = natural
            if centre is not None:
                cell = float(meta.get("cell_size_m") or 1.0)
                reach_m = max(float(sill_width_m or 0.0), cell)
                local, _c = pour_point_near(
                    dem, mask, centre, max(1, int(round(reach_m / cell))))
                if local is not None:
                    lip = float(local)

            if ew_type == "dam":
                inside = dem[mask]
                inside = inside[np.isfinite(inside)]
                floor = float(inside.min()) if inside.size else None
                if crest_elevation is not None:
                    return (lip, floor, float(crest_elevation), CONTAINMENT_WALL)
                return (lip, floor, natural, CONTAINMENT_LIP)

            drop = float(depth) if depth else 0.0
            # Off the global pour point, not the local lip: this is the level the burn
            # cuts to, and it is one level for the whole footprint.
            invert = natural - drop

            # Measured beats built beats bare, and each of the three is something that
            # was either observed or specified. `terrain_spill_level_m` is where the
            # finished pond was found to let go; `berm_crest_elevation` is where the last
            # burn's spoil bank actually reached.
            measured = getattr(ew, "terrain_spill_level_m", None) if ew is not None else None
            berm = getattr(ew, "berm_crest_elevation", None) if ew is not None else None
            if measured is not None and float(measured) > natural:
                return (lip, invert, float(measured), CONTAINMENT_MEASURED)
            if berm is not None and float(berm) > natural:
                return (lip, invert, float(berm), CONTAINMENT_BERM)
            return (lip, invert, natural, CONTAINMENT_LIP)
        except Exception:
            return (None, None, None, None)

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

    def ensure_design_tier(self):
        """Guarantee the catchment labelling and overflow routing exist.

        Both come out of the live assessment, which runs on every design edit, so in
        normal use they are simply there. They are *not* after a baseline re-run on a
        design that was loaded from file and never touched — and the fill simulation
        needs both: the labelling to split runoff between features, the routing to
        cascade overflow along the same links the design tier and the report use.

        Returns True when a labelling is available. Public because the simulation
        controller asks for this; everything else here reaches it through an edit.
        """
        if self._state.catchment_labels is None or self._state.balance_routing is None:
            self._recompute_live_assessment()
        return self._state.catchment_labels is not None

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
            # A linked drain's start level is derived, never serialised, and it is a
            # function of *another* feature's crest — so it goes stale on edits this
            # drain knows nothing about. Re-derived here because this is the one method
            # that runs after every design edit, and it is a pure loop over the model
            # with no DEM read in it.
            self._refresh_spillway_link_inverts()
            if not all_ews:
                self._panel.set_network([], {}, 0.0)
                self._panel.set_live_assessment("")
                self._panel.set_catchment_coverage(None)
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
                # From the cache added to stop exactly this. `meta["domain_cells"]`
                # carries a comment naming this method as the reason it exists, and
                # `compute_catchment_coverage` already reads it — while this line
                # re-reduced a 2.85 M-element mask on every one of 12.5 frames a
                # second. Same two-line fallback that method uses, for a meta dict
                # written before the key existed.
                domain_cells = meta.get("domain_cells")
                if domain_cells is None:
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
            # Keep them for the report. Without this the balance falls out of scope
            # at the end of this method and the report has no design-tier numbers to
            # print — which is why it used to demand a simulation first.
            self._state.balance = result
            self._state.balance_stores = stores
            # The fill simulation cascades along this same network — see _state.
            self._state.balance_routing = routing

            # Flow network (Live Assessment) — every earthwork, ordered high→low.
            exit_m3 = result.site_exit_m3 if result is not None else 0.0
            # Text stays live through a drag; anything that rebuilds a widget tree
            # or a map layer does not. `set_live_assessment` is a string, and the
            # scorecard below it is strings — a designer dragging a vertex is
            # watching those numbers move, which is the whole point of the live
            # readout, and they are cheap.
            self._panel.set_live_assessment(self._network_footer(result))

            drawn_result = None
            if geometry_settled:
                # Measured on the fixture at 12 features: a frame cost 526 ms
                # against the 80 ms a 12.5 Hz throttle allows, so the readout was
                # ~6.6x over its own budget and every one of these ran per mouse
                # event. `set_network` alone deletes every child widget and
                # reconstructs a `_NodeCard` and a connector per feature — about
                # 190 widget constructions and 220 QSS parses on a 31-feature
                # design — and `_refresh_connections_layer` rebuilds a map layer.
                #
                # The guard's own comment already said "doing that per feature at
                # 12.5 Hz is exactly the cost this method's docstring promises to
                # avoid"; these simply sat above it. Mid-drag geometry is not a
                # design, by that docstring, so a chip layout and a connections
                # layer drawn for it are answers to a question nobody asked yet.
                nodes = self._build_network_nodes(all_ews, stores, result)
                edges = {sid: (tgt, routing.is_user.get(sid, False))
                         for sid, tgt in routing.edges.items()}
                self._panel.set_network(nodes, edges, exit_m3)
                self._refresh_connections_layer(result, routing)
                self._panel.set_area_subtotals(self.compute_area_subtotals())
                self._panel.set_catchment_coverage(
                    self.compute_catchment_coverage(),
                    site_is_guessed=self._site_is_guessed())
                # The Report stage's on-screen headline, from the same
                # BalanceResult the report itself prints. It used to come only
                # from a ComparisonResult, which only a fill simulation produces —
                # so the summary was blank for a document that has needed nothing
                # but a baseline since it was rebuilt on the design tier.
                self._panel.set_report_summary(
                    result, comparison=self._state.comparison,
                    burn=self._state.burn_quantities)
                # A full-raster pass plus up to 20,000 GEOS `project` calls per
                # linear feature (M-5a). `_on_vertex_drag`'s docstring promises
                # never to run the heavy sub-calcs per mouse event; this was the
                # largest one that did.
                self.refresh_stress_points_layer()
                self._build_spillway_rows()
                self._check_spillway_capacity()
                drawn_result = self._drawn_basis_balance(
                    enabled, soil, duration_hr, total_runoff_m3, uncaptured_m3, routing)

            # Persistent scorecard (Workbench header) — blue means actual water.
            if result is not None and have_flow:
                stored = max(0.0, result.total_captured_m3 - result.total_infiltration_m3)
                self._panel.update_scorecard(
                    result.capture_pct, stored,
                    result.total_infiltration_m3, result.site_exit_m3,
                    natural_ponding_m3=self._natural_ponding_m3(),
                    capacity_note=self._capacity_note(result, drawn_result),
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

    def _drawn_basis_balance(self, enabled, soil, duration_hr, total_runoff_m3,
                             uncaptured_m3, routing):
        """The same storm scored against the **drawn** capacities, for comparison.

        The live score is sized on measured terrain storage, which on a keyed design is
        nearly double the drawn figure. That invites an obvious and important question —
        *did all that extra storage buy anything?* — and it is answerable for the price of
        one more balance pass (~30 ms over arithmetic already in hand), so it is answered
        rather than left to be assumed.

        On the Quail Island design the answer is **no**: 57% either way, because 7,567 m³
        of the 17,537 m³ storm never reaches a feature at all and every overflow that does
        occur is caught downstream. Capacity is not the constraint there; interception is.
        Returns None when nothing is measured, so there is nothing to compare.
        """
        from terrainflow_assessment.modules.simulation import build_stores_from_earthworks
        from terrainflow_assessment.modules.water_balance import run_water_balance

        if not any(getattr(e, "terrain_capacity_m3", None) for e in enabled):
            return None
        try:
            drawn_stores = build_stores_from_earthworks(
                enabled, soil_name=soil, dem_path=self._state.dem_path, basis="drawn")
            counts = self._state.catchment_counts or {}
            meta = self._state.flow_grid_meta or {}
            cell_area = meta.get("cell_area_m2", 0.0)
            runoff_m = self._current_runoff_mm() / 1000.0
            for store in drawn_stores:
                cells = counts.get(store.id, 0)
                store.direct_catchment_m2 = cells * cell_area
                store.inflow_m3 = cells * cell_area * runoff_m
                store.outlet_flat = self._state.catchment_outlets.get(store.id)
            return run_water_balance(
                drawn_stores, duration_hr, total_runoff_m3,
                uncaptured_m3=uncaptured_m3, routing=routing,
                count_infiltration=self._panel.count_infiltration,
            ) if drawn_stores else None
        except Exception as exc:
            print(f"TerrainFlow Assessment — drawn-basis balance error: {exc}")
            return None

    def _capacity_note(self, result, drawn_result):
        """One line under the scorecard band: is capacity what limits this design?

        Adaptive rather than decorative. Where the two bases agree, the useful thing to
        say is that storage is not the lever and how much water never arrives; where they
        differ, the useful thing is by how much. Both are one sentence, because the
        scorecard is a headline and this is a footnote to it.
        """
        if result is None or drawn_result is None:
            return None
        built = result.total_capacity_m3
        drawn = drawn_result.total_capacity_m3
        if built <= drawn * 1.02:
            return None
        gain = result.capture_pct - drawn_result.capture_pct
        if abs(gain) < 0.5:
            missed = max(0.0, result.site_exit_m3)
            return (
                f"Measured on the ground your earthworks hold {built:,.0f} m³, not the "
                f"{drawn:,.0f} m³ the drawn sections give — but the score is the same "
                f"either way. Capacity is not what limits this design: {missed:,.0f} m³ "
                f"never reaches a feature. Intercept more, not store more."
            )
        return (
            f"Measured on the ground your earthworks hold {built:,.0f} m³ against the "
            f"{drawn:,.0f} m³ the drawn sections give, and that is worth {gain:+.0f} "
            f"points of capture — water the banks hold above natural ground."
        )

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
                # The ground at the centroid — it orders the network. A dam's *crest*
                # is a separate figure, and the row prints that instead: sampling the
                # ground under a wall and calling it the crest read 54 m for a crest
                # of 56.12 m.
                "elevation": elev,
                "crest_elevation": getattr(ew, "crest_elevation", None),
                # Both, always. The node's bar is filled against ``capacity_m3``, which is
                # the measured pond wherever one exists — and a figure that can be twice
                # the drawn one cannot be shown without the drawn one beside it.
                "capacity_m3": (getattr(ew, "terrain_capacity_m3", None)
                                or getattr(ew, "capacity_m3", 0.0) or 0.0),
                "drawn_capacity_m3": getattr(ew, "capacity_m3", 0.0) or 0.0,
                "capacity_is_measured": getattr(ew, "terrain_capacity_m3", None) is not None,
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
        # "Capacity" here is what the ground holds, which is the basis the bars above are
        # filled against. Where that differs from the drawn sections, both are named —
        # the drawn total is what a contractor prices and what can be checked by hand.
        drawn = sum(getattr(s, "drawn_capacity_m3", 0.0) or 0.0
                    for s in (self._state.balance_stores or []))
        cap = f"Capacity {result.total_capacity_m3:,.0f} m³"
        if drawn and result.total_capacity_m3 > drawn * 1.02:
            cap += f" on the ground ({drawn:,.0f} m³ drawn)"
        parts = [
            f"{cap} · "
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

            crs = dem_crs(self._state)
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

            self.place(layer, G.DRAWN, at_top=True)
            self._state.connections_layer_id = layer.id()
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

    def clear_selection_highlight(self):
        """Take the highlight off the canvas.

        The band is deliberately independent of the earthwork layers so it can
        survive them being rebuilt on every edit — which also means hiding
        those layers does not hide it, and there was no other way to get rid of
        it than selecting something else. Anything that ends the selection, or
        that is about to photograph the canvas, calls this.
        """
        from qgis.core import QgsWkbTypes

        band = getattr(self, "_selection_band", None)
        if band is None:
            return
        band.reset(QgsWkbTypes.LineGeometry)
        try:
            self._canvas.refresh()
        except Exception:
            pass

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

        crs_str = dem_crs(self._state)
        want = crs_object(crs_str)

        # Registry-driven: the type registry is the single source of layer styling
        # (matching the panel's draw-button colours); a future register_type() gets
        # its map layer automatically.
        for ew_type, cfg in all_types().items():
            geom_type = cfg.geom_type
            display_name = f"{cfg.label}s"
            color_hex = cfg.style[1]
            existing = resolve_layer(self._project, self._state.ew_layer_ids.get(ew_type))
            if existing is not None:
                # A layer keeps the CRS it was created with, forever, and nothing
                # clears ew_layer_ids — so a feature drawn before a DEM was loaded
                # left every later one declared in the CRS of that first moment.
                # An empty one is replaced outright; a populated one is re-declared,
                # which is all that is needed because the features carry DEM grid
                # coordinates either way. Nothing is reprojected: the numbers were
                # always right, only the label on them was wrong.
                if existing.crs() != want:
                    existing.setCrs(want)
                    existing.triggerRepaint()
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
            # Appended, so the annotation layers placed with at_top=True stay
            # above the bands without anything having to re-sort the group.
            self.place(layer, G.DRAWN)
            # The id, never the layer. A stored wrapper outlives the C++ object it
            # points at — delete the layer in the Layers panel, or drop it while
            # re-stacking the group, and the next access raises "wrapped C/C++
            # object has been deleted" instead of quietly rebuilding.
            self._state.ew_layer_ids[ew_type] = layer.id()
            self._watch_visibility(layer.id())

    # ---------------------------------------------------------------- Earthwork symbology

    def _build_ew_symbol(self, cfg, enabled=True):
        """Per-type canvas symbol. Lives in ``_symbols.py`` — see that module for the
        metres-vs-millimetres rule the whole visual grammar hangs off."""
        return S.earthwork_symbol(cfg, enabled)

    def _refresh_ew_layer(self):
        self._ensure_ew_layers()
        for layer_id in self._state.ew_layer_ids.values():
            layer = resolve_layer(self._project, layer_id)
            if layer is not None:
                layer.dataProvider().truncate()

        for ew in self._state.earthwork_manager.get_all():
            layer = resolve_layer(self._project, self._state.ew_layer_ids.get(ew.type))
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

        for layer_id in self._state.ew_layer_ids.values():
            layer = resolve_layer(self._project, layer_id)
            if layer is not None:
                layer.triggerRepaint()

    # ---------------------------------------------------------------- Earthworks analysis

    def _burner_grid_mismatch(self):
        """Why the burner cannot be used against the session's DEM, or None.

        The burner is built once, from whichever DEM was selected at the time. Every
        raster it writes inherits that grid, while Baseline analyses ``state.dem_path`` —
        so if the two ever disagree, the before and after rasters describe different
        ground and nothing downstream can reconcile them. On the Quail Island run they
        disagreed by an entire extent (2157x1319 against 1027x858) and the only visible
        symptom was a line of small print under the verification table.
        """
        burner = self._state.burner
        info = self._state.dem_info
        if burner is None or info is None:
            return None
        if getattr(burner, "dem_path", None) == self._state.dem_path:
            return None
        try:
            same_grid = tuple(burner.shape) == (int(info.height), int(info.width))
        except Exception:
            same_grid = False
        if same_grid:
            return None
        return (
            f"Baseline and the earthworks burn would run on different terrain: the "
            f"burn is set up for a {burner.shape[1]}x{burner.shape[0]} grid but the "
            f"session's DEM is {info.width}x{info.height}. Re-select the DEM in the "
            f"panel and run Baseline again."
        )

    def run_with_earthworks(self):
        # Both slots. The run is two stages now — a burn on `design_worker`, then
        # the analysis it feeds on `analysis_worker` — and a second click during
        # either would burn against state the first is still using, then race to
        # assign the slot that is the only reference keeping its thread alive.
        if (worker_is_running(self._state, "analysis_worker")
                or worker_is_running(self._state, "design_worker")):
            self._iface.messageBar().pushInfo(
                "TerrainFlow Assessment",
                "An analysis is already running — wait for it to finish.")
            return
        if not self._state.dem_path or not self._state.burner:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "Load a DEM and run baseline first."
            )
            return
        mismatch = self._burner_grid_mismatch()
        if mismatch:
            # Refuse rather than warn. Burning against a burner built from a different
            # DEM produces a ponding raster on a different grid from the baseline's, and
            # the verification can only respond by skipping the subtraction — so every
            # measured volume comes back carrying the site's natural ponding, with the
            # design's own numbers looking plausible throughout.
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", mismatch)
            return

        enabled = self._state.earthwork_manager.get_enabled()
        if not enabled:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment", "No enabled earthworks to re-analyse with."
            )
            return

        # The burn is seconds of numpy on a real DEM and used to run right here, on
        # the GUI thread, with no repaint possible — which is what put "Not
        # Responding" over a plugin that was working. It goes to a worker; the
        # analysis it feeds is started from the completion handler.
        burner = self._state.burner
        mod_path = os.path.join(self._state.output_dir, "modified_dem.tif")
        cell_area = self._state.dem_info.cell_area_m2 if self._state.dem_info else 1.0
        # Resolved here, on the GUI thread, because the snap goes through QGIS geometry
        # and the worker may not touch it. This is also where the orphaned-sill warning
        # belongs: a Verify run is the moment the user is asking what the terrain does.
        sills = self._spillway_sills(enabled, warn=True)
        # Resolved here for the same reason, and reported here for the same one: a link
        # that has stopped resolving changes what the burn cuts without changing
        # anything the user can see, and this is the moment they are asking what the
        # terrain does. The worker is handed features that already carry the level.
        self._warn_dangling_spillway_links()

        def work(report):
            from terrainflow_assessment.modules.earthwork_design import burn_quantities

            report(5, "Cutting the design into the terrain…")
            modified_dem = burner.burn_earthworks(enabled, sills=sills)
            report(20, "Saving the burned surface…")
            burner.save(modified_dem, mod_path)
            # What the terrain model actually moved, measured here because both
            # surfaces are in hand. The report prints it beside the drawn-section
            # figure, which assumes flat ground and so understates a level cut on
            # any real slope.
            try:
                moved = burn_quantities(burner.original, modified_dem, cell_area)
            except Exception:
                moved = None

            # Where the earth has to go, computed in the same pass because both
            # surfaces are already in frame. Re-reading them elsewhere would be a
            # second alignment path over the same two rasters.
            haul = None
            try:
                report(75, "Matching cut to fill...")
                from terrainflow_assessment.modules.mass_haul import (
                    allocate_haul,
                    haul_regions,
                )
                cuts, fills = haul_regions(
                    burner.original, modified_dem, burner.transform, cell_area)
                haul = allocate_haul(cuts, fills)
            except Exception:
                # A haul plan is enrichment; a burn that produced a surface must not
                # fail because the earth could not be matched.
                haul = None

            return {"path": mod_path, "burn_quantities": moved, "haul_plan": haul,
                    "warnings": list(getattr(burner, "warnings", []))}

        worker = TaskWorker(work, label="burn")
        worker.progress.connect(self._panel.set_earthworks_progress)
        worker.completed.connect(self._on_burn_complete)
        worker.error.connect(self._on_analysis_error)
        self._state.design_worker = worker
        self._panel.set_earthworks_progress(1, "Starting…")
        worker.start()

    def _on_burn_complete(self, burned):
        """Back on the GUI thread: report the burn, then run the analysis on it."""
        for msg in burned["warnings"]:
            # Strategy-C honesty warnings (sub-cell features, resolution-cap
            # degrade) so a 1-cell routing approximation is never silent.
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)
        self._state.modified_dem_path = burned["path"]
        self._state.burn_quantities = burned["burn_quantities"]
        self._state.haul_plan = burned.get("haul_plan")
        mod_path = burned["path"]

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
            # Without these the re-run silently fell back to the worker's constructor
            # defaults — coefficient at C=0.50 — so a rainfall-basis or SCS-CN session got
            # its earthworks Surface Runoff and exit L/s scaled by a *different* depth from
            # the baseline it is meant to be compared against. The chosen basis governs the
            # whole assessment, which has to include this run.
            sizing_basis=self._panel.sizing_basis,
            runoff_coefficient=self._panel.runoff_coefficient,
            # Same argument, three more arguments. `exit_flow_ls` defaults to 0.5, so a
            # session that raised the panel's "Show exits above" threshold got earthworks
            # exit markers drawn through a *different* filter from the baseline's — and the
            # before/after exit comparison, which is the point of running this tier at all,
            # was then partly a comparison of thresholds. The two area paths default to
            # None, which collapses `area_outflow` to the whole site and loses the
            # per-area split the baseline reports.
            exit_flow_ls=self._panel.exit_flow_ls,
            analysis_area_path=self._state.analysis_area_path,
            earthworks_area_path=self._state.earthworks_area_path,
        )
        self._state.analysis_worker.progress.connect(self._panel.set_earthworks_progress)
        self._state.analysis_worker.completed.connect(self._on_earthworks_complete)
        self._state.analysis_worker.error.connect(self._on_analysis_error)
        self._state.analysis_worker.start()

    def _on_earthworks_complete(self, result):
        self._state.earthworks_result = result
        # The plugin's own BaselineController renders these layers, not a throwaway
        # copy. A fresh one connects `scaleChanged` in its constructor and is then
        # dropped, so every re-analysis left another connection to a dead controller —
        # and the exit-marker ids it collected went with it, which is why earthworks
        # exit markers never rescaled on zoom while the baseline's did.
        bl = self.baseline
        bl._load_result_layers(result, is_earthworks=True)
        self._load_burned_dem_layer(bl)
        if result.get("ponding"):
            self._state.ponding_raster_path = result["ponding"]

        # Reported on this tier as well as the baseline, so the two runs are comparable: a
        # burn that creates unrouted cells the bare ground did not have is worth knowing
        # about, and it is exactly the case a design would introduce.
        unrouted = result.get("unrouted_warning")
        if unrouted:
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", unrouted)
        diag_path = result.get("unrouted_diag_path")
        if diag_path:
            print(f"TerrainFlow Assessment — unrouted diagnostics: {diag_path}")

        crest = result.get("crest_warning")
        if crest:
            self._iface.messageBar().pushWarning("TerrainFlow Assessment", crest)

        # Non-circular check: terrain-derived ponding vs analytic capacity (§4).
        self._state.verification = self._compute_verification()
        # All three need what that pass read off disk, so they follow it rather than
        # standing on their own.
        self._record_spillway_levels()
        self._build_event_pond_layers()
        self._build_overtopping_layer()
        self._build_burned_spillway_layer()
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
        # delta, which cannot distinguish a burn error from a terrain effect.
        cell = self._state.dem_info.cell_size_m if self._state.dem_info else 1.0
        self._panel.set_verification(v, cell_size_m=cell)

        # The design is now verified against a burn — reset the drift counter.
        self._state.edits_since_verify = 0
        self._state.verified_delta_pct = v.delta_pct if v is not None else None
        self._update_verified_chip()

        self._measure_deferred_dams()

    def _measure_deferred_dams(self):
        """Flood any dam whose capacity was deferred because the burn held the burner.

        `_compute_dam_capacity` returns the cached figure while `design_worker` is
        running, and a dam drawn *during* the run has no cache — it sits at 0. The
        burn's own completion handler does not measure dams, so without this nothing
        ever would: the feature would keep a 0 m³ capacity and a "% full" bar reading
        full, for the rest of the session.

        The retry is bounded by construction — a dam that measures successfully is no
        longer at 0, and one that fails gets exactly one more flood per Re-analyse
        rather than a loop.
        """
        deferred = [ew for ew in self._state.earthwork_manager.get_all()
                    if ew.type == "dam"
                    and not getattr(ew, "capacity_m3", None)
                    and getattr(ew, "crest_elevation", None) is not None]
        if not deferred:
            return
        for ew in deferred:
            ew.capacity_m3 = self._compute_dam_capacity(ew)
            ew.capacity_l = ew.capacity_m3 * 1000.0
        self._recompute_live_assessment()

    def verification_sentence(self, v):
        """Explain the verification delta in words.

        "Verified · Δ −38%" said nothing about what was being compared or what the user
        should do. The delta now compares two floods — what each feature impounds alone
        against what the finished site ponds there — so it isolates interaction between
        features, with impoundment and the grid's fidelity reported as separate terms
        rather than folded into the same number.
        """
        from terrainflow_assessment.modules.reporting import (
            fmt_volume,
            round_volume,
        )

        if v is None:
            return ""
        penalty = sum(f.get("resolution_penalty_m3", 0.0) for f in v.per_feature)
        impounded = sum(f.get("impoundment_m3", 0.0) for f in v.per_feature)
        reference = sum(f.get("rasterisable_m3", 0.0) for f in v.per_feature)

        # Through `fmt_volume`, like every other volume the document prints. This
        # sentence is quoted verbatim into the report, so a raw `:,.0f` here claims
        # a precision the rest of the page is careful not to — see `round_volume`.
        parts = [
            f"Δ {v.delta_pct:+.0f}% — the finished site ponds "
            f"{fmt_volume(v.terrain_total_m3)} against the {fmt_volume(reference)} "
            f"these features hold on their own."
        ]
        if impounded > 1:
            parts.append(f"Of that, {fmt_volume(impounded)} is water the banks hold "
                         f"above natural ground — beyond any drawn cross-section.")
        if abs(penalty) > 1:
            parts.append(f"The grid cut the drawn trench {round_volume(penalty):+,} m³ "
                         f"differently; where that is large, Geometric is the capacity "
                         f"figure.")
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

        from terrainflow_assessment.modules.dem_loader import align_to_grid
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
                    bl_transform = src.transform
                if bnd is not None:
                    arr[arr == bnd] = 0.0
                arr = np.clip(arr, 0.0, None)
                if arr.shape == shape and bl_transform == transform:
                    bl_pond = arr
                else:
                    # Different shape is not necessarily different ground. A design file
                    # carries a *clip* of its DEM, so the baseline commonly sits on an
                    # exact sub-window of the burn's grid at the same cell size — on the
                    # Quail Island run, 66 rows and 287 columns in. Align it; only give
                    # up when the two grids genuinely do not share cells.
                    aligned = align_to_grid(arr, bl_transform, transform, shape)
                    if aligned is not None:
                        bl_pond = aligned
                    else:
                        bl_uncorrected = (
                            f"the baseline ponding raster ({arr.shape[1]}×{arr.shape[0]}) "
                            f"does not share a grid with the earthworks raster "
                            f"({shape[1]}×{shape[0]}), so it cannot be lined up cell for "
                            f"cell. Re-run the Baseline stage at the current extent"
                        )
            except Exception as exc:
                bl_uncorrected = (f"the baseline ponding raster could not be read "
                                  f"({exc})")

        diff = np.clip(ew_pond - bl_pond, 0.0, None)

        burned_masks = getattr(self._state.burner, "burned_masks", None) or {}
        burned_cut = getattr(self._state.burner, "burned_cut", None) or {}
        # Keyed by `ew.id`, never by name. The default name is
        # f"{type} {len(manager)+1}", counted over *all* earthworks, so deleting one
        # and drawing another reproduces a name that is already in use — the same
        # collision `_build_network_nodes` documents. Under name keying the first of
        # the pair vanished from every one of these dicts, so its water read as zero
        # and the verification scored one feature's pond against the other's
        # capacity. `build_verification` treats the key as opaque; display names are
        # put back below, once, from `name_by_id`.
        name_by_id = {}

        analytic_by_name = {}
        min_dims = {}
        breakdowns = {}
        footprints = []
        for ew in self._state.earthwork_manager.get_enabled():
            if getattr(ew, "capacity_m3", 0.0) <= 0:
                continue
            key = getattr(ew, "id", None) or ew.name
            name_by_id[key] = ew.name
            analytic_by_name[key] = ew.capacity_m3
            # Sub-cell check: channels key off the bottom width, polygons off their
            # equivalent strip width. Basins previously passed None, so a footprint
            # smaller than a cell still claimed its full analytic volume unflagged.
            try:
                geom = _shp(json.loads(ew.geometry.asJson()))
            except Exception:
                geom = None
            if ew.type == "swale":
                min_dims[key] = getattr(ew, "bottom_width_m", None)
            else:
                min_dims[key] = min_dimension(geom) if geom is not None else None

            # NB the terrain capacity cached on the feature by
            # ``_refresh_terrain_capacity`` is deliberately *not* read here. This
            # table's measured column comes from the whole-design flood
            # (``added.per_name``), which is the only figure that can see two
            # features sharing one pool; a per-feature cache cannot.
            #
            # The cells the burn actually claimed, straight from the burner. Re-deriving
            # them here is what let the two drift: this buffered a line by
            # max(width/2, cell_size) with all_touched while _burn_swale buffered by
            # top_width/2, so for anything narrower than two cells the measuring mask was
            # wider than the trench, inflating n_cells, inflating the At-grid reference
            # and biasing every Δ negative. Falls back to re-deriving only when no burn
            # has run on this grid.
            mask = burned_masks.get(getattr(ew, "id", None))
            if mask is None or mask.shape != shape:
                mask = np.zeros(shape, dtype=bool)
                if geom is not None:
                    try:
                        foot = geom
                        if foot.geom_type in ("LineString", "MultiLineString"):
                            foot = foot.buffer(
                                max(getattr(ew, "width", 2.0) / 2.0, cell_size))
                        mask = rasterize_footprint(foot, shape, transform,
                                                   all_touched=False)
                    except Exception:
                        mask = np.zeros(shape, dtype=bool)
            footprints.append((key, mask))

            # What this grid can actually represent — the reference the delta is
            # measured against, so the headline isolates burn error from cell size.
            try:
                breakdowns[key] = capacity_breakdown(
                    ew, cell_size=cell_size, cell_area=cell_area,
                    n_cells=int(mask.sum()),
                    terrain_storage_m3=getattr(ew, "terrain_capacity_m3", None),
                    cut_m3=burned_cut.get(getattr(ew, "id", None)),
                    excavation_m3=getattr(ew, "excavation_m3", None))
            except Exception:
                pass

        if not analytic_by_name:
            return None

        added = attribute_ponding_volume(diff, cell_area, footprints)
        # Water already standing here before any earthwork, over the same footprints —
        # so a feature built in a hollow can report what it adds, what was already
        # there, and the pool that ends up on the ground. One extra region-labelling
        # pass; the flood it depends on has already run.
        existing = attribute_ponding_volume(bl_pond, cell_area, footprints)
        baseline_total = raster_ponding_volume(bl_pond, cell_area)
        earthworks_total = raster_ponding_volume(ew_pond, cell_area)

        result = build_verification(
            analytic_by_name, added.per_name, baseline_total, earthworks_total,
            min_dims, cell_size, breakdowns=breakdowns,
            existing_by_name=existing.per_name,
            merged_groups=added.groups,
            unattributed_m3=added.unattributed_m3,
        )
        # Ids did the arithmetic; names do the reading. Rewritten in one place so a
        # collision can never make two rows indistinguishable — they were computed
        # apart and only the label is shared.
        for row in result.per_feature:
            # `id` stays on the row as well as the label, so anything joining to these
            # later — the simulation summary does — can match on identity rather than
            # on a name two features can share.
            row["id"] = row["name"]
            row["name"] = name_by_id.get(row["name"], row["name"])
        for group in result.merged_groups:
            group["names"] = tuple(name_by_id.get(n, n) for n in group.get("names", ()))

        # None when the subtraction was applied; a reason string when every measured
        # figure still carries whatever ponded there naturally.
        result.baseline_uncorrected = bl_uncorrected

        # Hand the arrays forward rather than reading and re-aligning them a second
        # time for the event pond. Same rasters, same footprints, same grid — which is
        # what stops the two views of one pool from disagreeing about where it is.
        self._state.pond_context = {
            "full": ew_pond, "baseline": bl_pond, "footprints": footprints,
            "transform": transform, "shape": shape, "cell_area_m2": cell_area,
        }
        return result

    def _record_spillway_levels(self):
        """Measure, off the finished burn, the two levels the design cannot state itself.

        Three elevations describe a spillway and they are allowed to disagree — each
        disagreement names a different fault, which is what turns *the model can now see
        my spillway* from a claim into something the user can check:

        * **designed sill** — ``Spillway.crest_elevation``, the absolute the burn cut to.
        * **as-burned sill** (here) — the highest level water crossing the notch has to
          clear, on the surface as it stands. Above the designed sill means the notch was
          refused and the bank is still there; below it means the whole path was already
          lower than the sill, so the cut did nothing.
        * **actual spill level** (here) — where the finished pond, on the whole burned
          site, was found to let go. Above the as-burned sill means the notch did not
          daylight and the water is leaving somewhere else; below the designed sill means
          a saddle elsewhere on the rim is lower and the spillway is not the control.

        Both are derived and cleared with the DEM, the same rule ``terrain_capacity_m3``
        follows. Measured on the **site** burn rather than a per-feature one, because the
        question is what the finished design does, neighbours included.
        """
        import numpy as np
        import rasterio
        from scipy.ndimage import label as _label

        from terrainflow_assessment.modules.earthwork_design import _POND_MIN_DEPTH_M

        for ew in self._state.earthwork_manager.get_all():
            ew.burned_sill_elevation_m = None
            ew.actual_spill_level_m = None

        burner = self._state.burner
        if burner is None:
            return

        # The as-burned sill comes off the burn and needs nothing else, so it is
        # recorded first and unconditionally. Gating it behind the verification pass
        # would lose it on every design the verification skips — and it skips anything
        # with no analytic capacity, which is exactly a feature drawn but not yet sized.
        sills = getattr(burner, "burned_sills", None) or {}
        by_ew = {getattr(ew, "id", None): ew
                 for ew in self._state.earthwork_manager.get_enabled()}
        for key, level in sills.items():
            ew = by_ew.get(key)
            if ew is not None:
                ew.burned_sill_elevation_m = float(level)

        # The measured spill level does need the pond, and the pond comes from the
        # verification pass. No pond, no answer — which is different from "it lets go at
        # the sill" and has to stay so.
        ctx = self._state.pond_context
        dem_path = self._state.modified_dem_path
        if not ctx or not dem_path or not os.path.exists(dem_path):
            return

        try:
            with rasterio.open(dem_path) as src:
                ground = src.read(1).astype("float64")
        except Exception:
            return
        if ground.shape != ctx["shape"]:
            return

        pond = np.asarray(ctx["full"], dtype="float64")
        # The same floor `attribute_ponding_volume` and `feature_storage` use, so all
        # three agree about what counts as a pond rather than each choosing a threshold.
        labels, _n = _label(pond > _POND_MIN_DEPTH_M)
        surface = ground + pond
        by_id = dict(ctx.get("footprints") or [])
        for key, ew in by_ew.items():
            mask = by_id.get(key)
            if mask is None or mask.shape != ground.shape:
                continue
            own = set(np.unique(labels[mask])) - {0}
            if not own:
                continue
            region = np.isin(labels, list(own))
            # A pool is level, so any wet cell answers it; the max is taken only to be
            # immune to the ragged cells at its edge — the same reading
            # ``overtopping_spill`` makes of the same rasters.
            ew.actual_spill_level_m = float(surface[region].max())

    def _build_burned_spillway_layer(self):
        """Draw the notches the burn actually cut, so *did it work* is a map question.

        "Did the model cut my spillway, and where?" has no answer on the map otherwise:
        the sill bar on the Design stage is what was asked for, and the cut is what
        happened. The mask is already in hand from the burn, so this is close to free,
        and it is the visual partner to the three elevations :meth:`_record_spillway_levels`
        records — a feature with a sill bar and no burned band is one whose notch was
        refused.

        Under **Verify**, through ``_groups``: it is a measurement off the burn, like the
        ponding and overtopping layers it sits beside, not part of the drawn design.
        """
        from rasterio.features import shapes as _shapes

        from terrainflow_assessment.qgis.controllers._layers import remove_layer

        remove_layer(self._project, self._state.burned_spillway_layer_id)
        self._state.burned_spillway_layer_id = None

        burner = self._state.burner
        notches = getattr(burner, "burned_notches", None) or {}
        if not notches:
            return
        sills = getattr(burner, "burned_sills", None) or {}
        by_id = {ew.id: ew for ew in self._state.earthwork_manager.get_all()}

        try:
            layer = QgsVectorLayer(f"Polygon?crs={dem_crs(self._state)}",
                                   "Earthworks — Spillways (burned)", "memory")
            pr = layer.dataProvider()
            pr.addAttributes([
                QgsField("name", QMetaType.QString),
                QgsField("label", QMetaType.QString),
                QgsField("sill_m", QMetaType.Double),
                QgsField("cells", QMetaType.Int),
            ])
            layer.updateFields()

            feats = []
            for key, mask in notches.items():
                ew = by_id.get(key)
                if ew is None or mask is None or not mask.any():
                    continue
                sill = sills.get(key)
                bits = [ew.name, "spillway cut"]
                if sill is not None:
                    bits.append(f"{float(sill):.2f} m")
                parts = []
                for geom, _v in _shapes(mask.astype("uint8"), mask=mask,
                                        transform=burner.transform):
                    rings = geom.get("coordinates", [])
                    if not rings:
                        continue
                    poly = QgsGeometry.fromPolygonXY(
                        [[QgsPointXY(float(x), float(y)) for x, y in ring]
                         for ring in rings])
                    if poly is not None and not poly.isEmpty():
                        parts.append(poly)
                if not parts:
                    continue
                merged = parts[0]
                for extra in parts[1:]:
                    merged = merged.combine(extra)
                f = QgsFeature(layer.fields())
                f.setGeometry(merged)
                f.setAttributes([ew.name, " · ".join(bits),
                                 None if sill is None else round(float(sill), 2),
                                 int(mask.sum())])
                feats.append(f)
            if not feats:
                return
            pr.addFeatures(feats)
            layer.updateExtents()
            layer.renderer().setSymbol(S.burned_spillway_symbol())
            self.place(layer, G.VERIFY, at_top=True)
            self._state.burned_spillway_layer_id = layer.id()
        except Exception as exc:
            print(f"TerrainFlow Assessment — burned spillway layer error: {exc}")

    def _build_event_pond_layers(self):
        """Draw where *this event's* water actually stands, over the full pond.

        The ponding raster beside it is a capacity map: every hollow filled to its
        spill point, drawn brim-full whatever the storm delivers, so a dam at 5% and a
        dam at 100% render identically. This is the other half of that picture — each
        pool re-filled with only the volume the balance routes into it, solved for the
        level that holds it, so a part-full pond sits small and shallow in the bottom
        of its basin.

        **Two layers from one raster**, because the comparison is what the user is
        after and a fill under a fill shows nothing: the raster carries the depth, and
        the outline is the water's edge, which is the thing that survives being drawn
        over a dark pond. Both go in at the top of the group, so the line reads over
        the fill and the fill over the capacity beneath it.

        Built here rather than live on the design tier because it needs the burn — the
        pools come from the raster the re-analysis produced. It therefore carries the
        same staleness as the layer it qualifies, which is the honest arrangement:
        both move when you re-analyse, and neither claims to be current before then.
        """
        import numpy as np
        import rasterio
        from rasterio.features import shapes as _shapes

        from terrainflow_assessment.core.registry.map_palette import (
            EVENT_WATER_LINE,
            WATER_CAPTURED,
        )
        from terrainflow_assessment.modules.reporting import event_pond_depth

        ctx = self._state.pond_context
        balance = self._state.balance
        dem_path = self._state.modified_dem_path
        if not ctx or balance is None or not dem_path or not os.path.exists(dem_path):
            return

        # Keyed to match `ctx["footprints"]`, which `_compute_verification` keys by
        # `ew.id`. The balance rows carry both, so this is a lookup rather than a
        # translation — but the two sides have to agree or every footprint misses its
        # volume, `event_pond_depth` finds nothing to place, and the event layer is
        # silently not built at all.
        stored = {(f.get("id") or f["name"]): f.get("stored_m3", 0.0)
                  for f in (getattr(balance, "per_feature", None) or [])}
        if not stored:
            return

        try:
            with rasterio.open(dem_path) as src:
                ground = src.read(1).astype("float64")
                crs = src.crs
        except Exception:
            return
        if ground.shape != ctx["shape"]:
            return

        try:
            depth = event_pond_depth(
                ctx["full"], ground, ctx["cell_area_m2"], ctx["footprints"], stored,
                existing=ctx["baseline"],
            )
        except Exception as exc:
            print(f"TerrainFlow Assessment — event pond error: {exc}")
            return

        # Handed to the overtopping check, which runs next and otherwise has only the
        # full-capacity pond to answer from — the whole of the reason its band read as
        # a claim about this storm when it was a claim about the structure.
        ctx["event"] = depth

        wet = depth > 0.001
        if not wet.any():
            return

        path = os.path.join(self._state.output_dir, "event_pond.tif")
        try:
            with rasterio.open(
                path, "w", driver="GTiff", dtype="float32", count=1,
                height=depth.shape[0], width=depth.shape[1],
                crs=crs, transform=ctx["transform"], nodata=-9999.0,
            ) as dst:
                dst.write(np.where(wet, depth, -9999.0).astype("float32"), 1)
        except Exception:
            return

        layer = QgsRasterLayer(path, "Earthworks — Pond Capacity (event)")
        if layer.isValid():
            # Scaled to the *full* pond's deepest cell, not its own. Left to scale
            # itself, a shallower event pond would stretch the same ramp over a
            # smaller range and the two layers would say "deepest" in the same navy
            # at different depths — which is exactly the comparison being made here.
            #
            # Through the shared ``ponding`` scale, so the same holds against the
            # Baseline capacity layer as well as against the Earthworks one: a
            # before/after pair drawn on two different stretches of one ramp shows a
            # difference that is the ramp's, not the design's.
            S.apply_shared_ramp(self._state, self._project, "ponding", layer,
                                WATER_CAPTURED,
                                float(np.asarray(ctx["full"]).max()))
            self.place(layer, G.RERUN, at_top=True)
            self._state.earthworks_layer_ids.append(layer.id())

        crs_str = dem_crs(self._state)
        line = QgsVectorLayer(f"LineString?crs={crs_str}",
                              "Earthworks — Event Water Line", "memory")
        if not line.isValid():
            return
        feats = []
        for geom, _value in _shapes(wet.astype("uint8"), mask=wet,
                                    transform=ctx["transform"]):
            # Cell-edge rings, deliberately unsmoothed: the water's edge is known to
            # the cell and drawing it as a curve would claim a precision the grid
            # does not have. Holes come through as their own rings — an island in a
            # pond has a shoreline too.
            for ring in geom.get("coordinates", []):
                pts = [QgsPointXY(float(x), float(y)) for x, y in ring]
                if len(pts) < 2:
                    continue
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPolylineXY(pts))
                feats.append(f)
        if not feats:
            return
        line.dataProvider().addFeatures(feats)
        line.updateExtents()
        line.setRenderer(QgsSingleSymbolRenderer(QgsLineSymbol.createSimple({
            "color": ",".join(str(c) for c in EVENT_WATER_LINE),
            "width": "0.5",
        })))
        self.place(line, G.RERUN, at_top=True)
        self._state.earthworks_layer_ids.append(line.id())

    def _build_overtopping_layer(self):
        """Find barriers their own pools pour over, say so, and draw the run of crest.

        Two things the map could not previously tell you, both measured off the burn.

        *That it happens at all*: a pool leaves at the lowest point of its rim, and where
        that point is the structure's own crest the water goes over the wall. Finding it
        by eye meant clicking cells with Identify and comparing them against a crest
        elevation held in a dialog.

        *That it happens along the whole crest*: D8 sends the entire overflow through one
        cell, so the stream layer draws a single thread crossing the wall and invites the
        reading that water is picking a spot. A level crest does not work that way — the
        pool's surface stays flat as it rises, so it goes over every metre standing at
        the pour level simultaneously. The band drawn here is that length, and it is what
        discharge per metre has to be figured against.

        *And which storm it is about.* The measurement is made on the **full** pond, so
        it answers "filled, does this pool leave over its own wall" — a freeboard fact
        about the structure, true whatever the event does, and the right question for a
        layer that exists to catch a dam with no spillway. Drawn without qualification it
        was read as the other question: a solid red band beside an event water line
        sitting a metre below the crest says the modelled storm is going over the top,
        and it is not. So the event level is measured too, and the answer is split into
        the same **(full)** / **(event)** pair the pond layers already use — see
        :meth:`_place_overtopping_layer`.
        """
        import rasterio

        from terrainflow_assessment.core.registry.map_palette import (
            OVERTOPPING_CAPACITY_FILL,
            OVERTOPPING_FILL,
        )
        from terrainflow_assessment.modules.burn_strategy import overtopping_warning
        from terrainflow_assessment.modules.reporting import overtopping_spill

        ctx = self._state.pond_context
        dem_path = self._state.modified_dem_path
        burner = self._state.burner
        if not ctx or burner is None or not dem_path or not os.path.exists(dem_path):
            return

        try:
            with rasterio.open(dem_path) as src:
                ground = src.read(1).astype("float64")
        except Exception:
            return
        if ground.shape != ctx["shape"]:
            return

        original = getattr(burner, "original", None)
        masks = getattr(burner, "burned_masks", None) or {}
        built_by = getattr(burner, "burned_raised", None) or {}
        notched = getattr(burner, "burned_notches", None) or {}
        if original is None or original.shape != ground.shape:
            return
        # The cells this feature *raised* — its crest — not its whole footprint. A cut
        # cannot be overtopped; only built ground can.
        raised = ground > original + 1e-6

        # Keyed by id, and `key` was already being computed two lines down for the
        # mask lookups while the barrier list went on using the display name. The
        # default name counter reproduces a deleted feature's name — documented at
        # `:4465-4469`, `:5311-5318` and `simulation.py:273-276`, each of which keys
        # by id because of it — so with two dams called "Dam 2" the freeboard
        # advisory reported `has_spillway` off whichever one `by_name` kept, and the
        # "(full)" layer's `spillway` attribute was wrong on that row.
        #
        # `overtopping_spill` carries the key through untouched as `spill.name`, so
        # the id travels and `name_by_id` puts the display name back wherever a
        # person reads it: the advisory text and the layer's `feature` attribute.
        barriers, by_id, name_by_id = [], {}, {}
        for ew in self._state.earthwork_manager.get_enabled():
            key = getattr(ew, "id", None)
            # ``burned_raised`` is the bank this feature built; ``burned_masks`` is what it
            # claimed. For a dam the two overlap, but a swale's mask is its trench and its
            # companion berm sits beside it, so ``mask & raised`` is empty and a keyed
            # swale was never checked at all. Prefer the bank, fall back to the mask.
            mask = built_by.get(key)
            if mask is None or mask.shape != ground.shape:
                mask = masks.get(key)
            if mask is None or mask.shape != ground.shape:
                continue
            crest = mask & raised
            # The designed spillway is not part of the crest. A notch cut into raised
            # ground is still raised — it is a lowered piece of a bank that was built —
            # so without this subtraction a correctly spillwayed dam reports "leaves over
            # its own crest" *at its own spillway*, which is the one place it is supposed
            # to leave. Water going through the notch is the design working.
            notch = notched.get(key)
            if notch is not None and notch.shape == ground.shape:
                crest = crest & ~notch
            if not crest.any():
                continue
            try:
                length = float(ew.geometry.length())
            except Exception:
                length = 0.0
            barriers.append((key, crest, length))
            by_id[key] = ew
            name_by_id[key] = ew.name

        if not barriers:
            return
        try:
            t = ctx["transform"]
            spills = overtopping_spill(ctx["full"], ground,
                                       abs(t.a), barriers, built=raised,
                                       cell_area_m2=abs(t.a * t.e),
                                       event_depth=ctx.get("event"))
        except Exception as exc:
            print(f"TerrainFlow Assessment — overtopping check failed: {exc}")
            return
        if not spills:
            return

        for spill in spills:
            # `spill.name` carries the id this method put in; the person reading the
            # advisory needs the label.
            ew = by_id.get(spill.name)
            msg = overtopping_warning(
                name_by_id.get(spill.name, spill.name),
                spill.length_m, spill.pour_level_m,
                alt_saddle_m=spill.alt_saddle_m,
                has_spillway=getattr(ew, "spillway", None) is not None,
                reaches_crest=spill.overtops_this_event,
                event_level_m=spill.event_level_m,
            )
            if not msg:
                continue
            # A crest this event does not reach is a design note, not an alarm. Pushed
            # as a warning it competed with the ones that are about the storm just
            # routed, and a bar full of undifferentiated red is a bar nobody reads.
            if spill.overtops_this_event is False:
                self._iface.messageBar().pushInfo("TerrainFlow Assessment", msg)
            else:
                self._iface.messageBar().pushWarning("TerrainFlow Assessment", msg)

        # Two layers, named to match the pond pair sitting beside them: **(full)** is
        # every barrier that pours over itself once its pool is brim-full, and
        # **(event)** is the subset this storm actually reaches — nested inside the
        # first exactly as "Pond Capacity (event)" nests inside "(full)".
        #
        # A tick per question, rather than one layer styled two ways, because the two
        # are asked at different moments. Judging a design against the storm just
        # routed, the capacity bands are noise and go off; asking whether a wall has
        # freeboard, they are the whole answer. One layer cannot be half-turned-off
        # however it is symbolised, and the styling that stood in for it had to be read
        # off a hatch pattern rather than off a name.
        #
        # (full) goes down first so (event) lands above it: where both apply, the
        # statement about this storm is the one that should be on top.
        self._place_overtopping_layer(
            "Earthworks — Overtopping (full)", spills, by_id, name_by_id, ctx,
            fill=OVERTOPPING_CAPACITY_FILL, hatched=True,
            suffix=" when full", priority=3)
        self._place_overtopping_layer(
            "Earthworks — Overtopping (event)",
            [s for s in spills if s.overtops_this_event], by_id, name_by_id, ctx,
            fill=OVERTOPPING_FILL, hatched=False,
            suffix=" this event", priority=8)

    def _place_overtopping_layer(self, name, spills, by_id, name_by_id, ctx,
                                 fill, hatched, suffix, priority):
        """One of the two overtopping band layers. No layer at all when empty.

        An empty "(event)" layer would be a row in the legend asserting a question was
        asked and answered no — which is right when the event was measured and wrong
        when there was no event pond to measure against. Absent, it says neither, and
        the advisory in the message bar is where "nothing goes over in this run" is
        stated in words.
        """
        from qgis.PyQt.QtCore import Qt as _Qt
        from rasterio.features import shapes as _shapes

        from terrainflow_assessment.core.registry.map_palette import OVERTOPPING_EDGE

        if not spills:
            return

        crs_str = dem_crs(self._state)
        layer = QgsVectorLayer(f"Polygon?crs={crs_str}", name, "memory")
        if not layer.isValid():
            return
        pr = layer.dataProvider()
        pr.addAttributes([
            QgsField("feature", QMetaType.QString),
            QgsField("length_m", QMetaType.Double),
            QgsField("level_m", QMetaType.Double),
            QgsField("spillway", QMetaType.QString),
            # Which storm the band is about. Redundant with the layer name for
            # "event", and not redundant in "(full)": a spill sits there both when
            # the event was measured and fell short ("capacity") and when there was
            # no event pond to ask ("unknown"), and those are different states that
            # one layer name cannot separate.
            QgsField("state", QMetaType.QString),
            QgsField("event_m", QMetaType.Double),
        ])
        layer.updateFields()

        feats = []
        for spill in spills:
            # Id in, label out — see `_build_overtopping_layer`. Keyed by name, two
            # features sharing one put the wrong feature's spillway in this column.
            ew = by_id.get(spill.name)
            label = name_by_id.get(spill.name, spill.name)
            sited = "designed" if getattr(ew, "spillway", None) is not None else "none"
            reaches = spill.overtops_this_event
            state = ("unknown" if reaches is None
                     else "event" if reaches else "capacity")
            parts = []
            for geom, _v in _shapes(spill.mask.astype("uint8"), mask=spill.mask,
                                    transform=ctx["transform"]):
                rings = geom.get("coordinates", [])
                if not rings:
                    continue
                poly = QgsGeometry.fromPolygonXY(
                    [[QgsPointXY(float(x), float(y)) for x, y in ring]
                     for ring in rings])
                if poly is not None and not poly.isEmpty():
                    parts.append(poly)
            if not parts:
                continue
            merged = parts[0]
            for extra in parts[1:]:
                merged = merged.combine(extra)
            f = QgsFeature(layer.fields())
            f.setGeometry(merged)
            f.setAttributes([label, round(spill.length_m, 1),
                             round(spill.pour_level_m, 2), sited, state,
                             None if spill.event_level_m is None
                             else round(spill.event_level_m, 2)])
            feats.append(f)
        if not feats:
            return
        pr.addFeatures(feats)
        layer.updateExtents()

        # Same red and the same edge in both layers: it is one fault on one crest, and
        # a second hue would read as a second kind of thing. The capacity bands are
        # hatched so the two stay distinguishable where they overlap — (event) is drawn
        # over (full) by construction — and carry more alpha, because a diagonal hatch
        # at the solid fill's alpha is barely on the map.
        sym = QgsFillSymbol.createSimple({
            "color": ",".join(str(c) for c in fill),
            "outline_color": ",".join(str(c) for c in OVERTOPPING_EDGE),
            "outline_width": "0.6",
        })
        if hatched:
            try:
                sym.symbolLayer(0).setBrushStyle(_Qt.BrushStyle.BDiagPattern)
            except Exception:
                pass
        layer.setRenderer(QgsSingleSymbolRenderer(sym))

        # Labelled with the length, because that is the whole point of the band — and
        # with the reference state, because "spills over 30 m" beside an event water
        # line a metre below the crest is the sentence that has to be qualified.
        #
        # ``priority`` is what settles the two labels a barrier in both layers would
        # draw on one crest: PAL competes across layers, so the low number is dropped
        # first, and where both apply the event statement is the one to keep.
        settings = QgsPalLayerSettings()
        settings.fieldName = (
            f"concat(\"feature\", ' spills over ', "
            f"format_number(\"length_m\", 0), ' m{suffix}')")
        settings.isExpression = True
        settings.priority = priority
        text = QgsTextFormat()
        text.setSize(9)
        text.setColor(QColor(120, 20, 12))
        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setSize(1.0)
        buf.setColor(QColor(255, 255, 255, 220))
        text.setBuffer(buf)
        settings.setFormat(text)
        settings.placement = QgsPalLayerSettings.Placement.OverPoint
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(True)

        self.place(layer, G.RERUN, at_top=True)
        self._state.earthworks_layer_ids.append(layer.id())

    def _load_burned_dem_layer(self, baseline=None):
        """Add the burned (Strategy-C) DEM, and its shaded relief, to the layer panel.

        Both are backdrops rather than result overlays, so they go at the bottom of the
        Design group and are registered under the earthworks layer ids to show/hide with
        the 'with earthworks' toggle. Silently skips if the burn produced no valid raster.

        **The hillshade is what makes the design visible as earth.** On the elevation
        ramp a 1 m swale inside 60 m of relief is one shade of grey against another and
        cannot be seen at all; shaded, it reads immediately as a cut line with a bank
        beside it, and the keyed returns at each end show as the hooks they are. Ticked
        on arrival, unlike Baseline's copy of the same idea, because looking at it is the
        entire point.

        Order matters and is the reason this is one method rather than two. ``place``
        appends, and appended means bottom of the group means painted **underneath**, so
        the shading has to be added *before* the elevation raster it is shading or it
        renders behind it and nothing changes on screen. *baseline* supplies the renderer
        (:meth:`BaselineController._add_hillshade`) so there is one definition of what a
        TerrainFlow hillshade looks like rather than two that can drift apart.
        """
        path = self._state.modified_dem_path
        if not path or not os.path.exists(path):
            return
        from qgis.core import QgsRasterLayer

        from terrainflow_assessment.qgis.controllers._layers import remove_layer

        # The pair this method placed last time. Nothing else removes them: the only
        # ``clear_group`` call clears Rerun and Baseline, and ``earthworks_layer_ids``
        # has already been reassigned by ``_load_result_layers`` by the time this
        # runs, so the previous ids were simply dropped. Three Verify runs therefore
        # left six ticked-on backdrops stacked on one file — and three live
        # ``QgsRasterLayer``s holding ``modified_dem.tif`` open while the next run
        # rewrites it, which on Windows is the GDAL lock ``simulation.py:447-453``
        # already documents.
        for old_id in list(self._state.burned_backdrop_layer_ids):
            remove_layer(self._project, old_id)
        self._state.burned_backdrop_layer_ids = []

        ids = list(getattr(self._state, "earthworks_layer_ids", None) or [])
        backdrops = []

        def _place(layer, visible=True):
            self.place(layer, G.DESIGN, visible=visible)
            # Both lists. ``toggle_before_after`` walks ``earthworks_layer_ids``, so
            # the ids have to stay there or the toggle stops reaching the backdrops;
            # ``backdrops`` is only what lets the *next* run remove this pair.
            ids.append(layer.id())
            backdrops.append(layer.id())

        if baseline is not None:
            baseline._add_hillshade(_place, path, "Earthworks — Hillshade")

        layer = QgsRasterLayer(path, "Earthworks — Burned DEM")
        if layer.isValid():
            _place(layer)
        self._state.earthworks_layer_ids = ids
        self._state.burned_backdrop_layer_ids = backdrops

    def _on_analysis_error(self, tb):
        self._panel.set_earthworks_failed(
            "Analysis failed — see Python console for details.")
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
        # The storm-fill verdict, wired. The tool has always taken
        # `earthwork_inflows` and documented the shape it wants; the one
        # construction site passed none, so `_find_nearest_inflow` returned -1.0
        # on every click, `fill_fraction` was always -1.0, and the checkmark /
        # warning block in `_on_ponding_selected` never rendered. Silent in both
        # directions: no verdict, and no word that there was not going to be one.
        #
        # The controller already owns the figure — `feature_inflow_m3` is what the
        # spillway review is sized from — so nothing new is computed here.
        inflows = [
            (float(self.feature_inflow_m3(ew) or 0.0), ew.geometry, ew.name)
            for ew in self._state.earthwork_manager.get_enabled()
        ]
        tool = PondingQueryTool(self._canvas, self._state.ponding_raster_path,
                                earthwork_inflows=inflows)
        tool.ponding_selected.connect(self._on_ponding_selected)
        tool.no_ponding.connect(self._on_no_ponding)
        self.use_tool(tool)

    def _on_ponding_selected(self, volume_m3, volume_l, cell_count, area_m2,
                              outline_geom, inflow_m3, fill_fraction,
                              inflow_name=""):
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
             H.PONDING_VOLUME_HELD)

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
            # Named. The verdict is a comparison against one feature's inflow,
            # picked by distance from where the user clicked — a decision they
            # cannot see, so a bare percentage is not checkable.
            against = f" (against {inflow_name})" if inflow_name else ""
            lbl_fill = QLabel(
                f"<b>Storm fill:</b>  <span style=\"color:{colour}\">{fill_str}</span>"
                f"{against}"
            )
            lbl_fill.setToolTip(H.FILL_RATIO)
            layout.addWidget(lbl_fill)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)
        dlg.exec()

    def _on_no_ponding(self):
        self._iface.messageBar().pushInfo(
            "TerrainFlow Assessment",
            "No ponding at that location — click a blue zone in a Pond Capacity layer.",
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
                layer.setCrs(crs_object(dem_crs(self._state)))
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
