from collections import deque

import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.core import QgsWkbTypes, QgsGeometry, QgsPointXY


class PondingQueryTool(QgsMapTool):
    """
    Click on the 'Water Captured' raster to select a connected ponding area.

    Emits ponding_selected(volume_m3, volume_l, cell_count, area_m2,
                           QgsGeometry, inflow_m3, fill_fraction)

    inflow_m3 and fill_fraction are -1.0 when no earthwork inflow data is
    available (e.g. baseline natural-depressions run).
    """
    ponding_selected = pyqtSignal(float, float, int, float, object, float, float)
    no_ponding = pyqtSignal()

    MIN_DEPTH = 0.001  # metres — ignore cells shallower than 1 mm (floating-point noise)

    def __init__(self, canvas, ponding_raster_path, earthwork_inflows=None):
        """
        Parameters
        ----------
        canvas : QgsMapCanvas
        ponding_raster_path : str
        earthwork_inflows : list of (inflow_m3: float, QgsGeometry) or None
            Per-earthwork design-storm inflow volumes.  When provided the tool
            finds the nearest earthwork to the clicked point and reports fill
            fraction and a sizing verdict.
        """
        super().__init__(canvas)
        self.canvas = canvas
        self.ponding_raster_path = ponding_raster_path
        self.earthwork_inflows = earthwork_inflows or []

        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(0, 80, 200, 160))
        self.rubber_band.setFillColor(QColor(100, 180, 255, 80))
        self.rubber_band.setWidth(2)

        # Load raster data once when tool is activated
        with rasterio.open(ponding_raster_path) as src:
            self.ponding_array = src.read(1).astype("float32")
            self.transform = src.transform
            self.cell_size = abs(src.transform.a)
            self.cell_area_m2 = abs(src.transform.a * src.transform.e)
            self.rows, self.cols = self.ponding_array.shape

    def canvasPressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        map_pt = self.toMapCoordinates(event.pos())
        row, col = self._map_to_rowcol(map_pt.x(), map_pt.y())

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self.no_ponding.emit()
            return

        if self.ponding_array[row, col] < self.MIN_DEPTH:
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.no_ponding.emit()
            return

        volume_m3, cell_count, visited_mask = self._flood_fill(row, col)
        area_m2 = cell_count * self.cell_area_m2
        volume_l = volume_m3 * 1000.0

        # Sizing: find the nearest earthwork and compare inflow to capacity
        inflow_m3 = self._find_nearest_inflow(map_pt.x(), map_pt.y())
        if inflow_m3 >= 0 and volume_m3 > 0:
            fill_fraction = inflow_m3 / volume_m3
        else:
            fill_fraction = -1.0

        # Build outline polygon from the visited mask
        outline_geom = self._mask_to_qgs_geometry(visited_mask)
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        if outline_geom:
            self.rubber_band.addGeometry(outline_geom, None)

        self.ponding_selected.emit(
            volume_m3, volume_l, cell_count, area_m2, outline_geom,
            inflow_m3, fill_fraction,
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
            self.canvas.unsetMapTool(self)

    def deactivate(self):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        super().deactivate()

    # ---------------------------------------------------------------- internals

    def _map_to_rowcol(self, x, y):
        col = int((x - self.transform.c) / self.transform.a)
        row = int((y - self.transform.f) / self.transform.e)
        return row, col

    def _flood_fill(self, start_row, start_col):
        """
        8-connected BFS flood fill from the clicked cell.
        Returns (total_volume_m3, cell_count, visited_bool_mask).
        """
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        queue = deque([(start_row, start_col)])
        visited[start_row, start_col] = True
        depth_sum = 0.0

        neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        while queue:
            r, c = queue.popleft()
            depth_sum += float(self.ponding_array[r, c])

            for dr, dc in neighbours:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and not visited[nr, nc]
                        and self.ponding_array[nr, nc] >= self.MIN_DEPTH):
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        volume_m3 = depth_sum * self.cell_area_m2
        cell_count = int(np.sum(visited))
        return volume_m3, cell_count, visited

    def _find_nearest_inflow(self, x, y):
        """
        Return the inflow_m3 of the earthwork nearest to map point (x, y).
        Returns -1.0 if no earthwork inflow data is available.
        """
        if not self.earthwork_inflows:
            return -1.0

        pt_geom = QgsGeometry.fromPointXY(QgsPointXY(x, y))
        min_dist = float("inf")
        nearest = -1.0

        for inflow_m3, geom in self.earthwork_inflows:
            try:
                dist = geom.distance(pt_geom)
                if dist < min_dist:
                    min_dist = dist
                    nearest = inflow_m3
            except Exception:
                continue

        return nearest

    def _mask_to_qgs_geometry(self, mask):
        """Convert a boolean numpy mask to a QgsGeometry polygon."""
        try:
            uint_mask = mask.astype("uint8")
            polys = [
                shapely_shape(geom)
                for geom, val in rasterio_shapes(uint_mask, transform=self.transform)
                if val == 1
            ]
            if not polys:
                return None
            union = unary_union(polys)
            return QgsGeometry.fromWkt(union.wkt)
        except Exception:
            return None
