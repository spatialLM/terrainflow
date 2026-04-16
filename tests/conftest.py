"""
Shared test fixtures and QGIS mock setup.

Must run before any test module is imported. Mocks the QGIS runtime so that
processing/analysis modules that import qgis.PyQt can be imported in a plain
Python environment without a QGIS installation.
"""
import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

# ---------------------------------------------------------------------------
# Mock QGIS runtime — must happen before any module-under-test is imported
# ---------------------------------------------------------------------------

class _QThread:
    """Stub base class so SimulationWorker(QThread) is valid Python inheritance."""
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def quit(self):
        pass
    def isRunning(self):
        return False
    def wait(self):
        pass


class _PyqtSignal:
    """Minimal pyqtSignal descriptor stub."""
    def __init__(self, *args, **kwargs):
        self._callbacks = []

    def connect(self, fn):
        self._callbacks.append(fn)

    def emit(self, *args):
        for cb in self._callbacks:
            cb(*args)

    def __call__(self, *args, **kwargs):
        # Called as a class attribute: `progress = pyqtSignal(int, str)`
        return _PyqtSignal(*args, **kwargs)


_qt_core_mock = MagicMock()
_qt_core_mock.QThread = _QThread
_qt_core_mock.pyqtSignal = _PyqtSignal()

for _mod in [
    "qgis",
    "qgis.core",
    "qgis.gui",
    "qgis.PyQt",
    "qgis.PyQt.QtCore",
    "qgis.PyQt.QtWidgets",
    "qgis.PyQt.QtGui",
]:
    sys.modules.setdefault(_mod, MagicMock())

# Override QtCore explicitly so QThread is a real class
sys.modules["qgis.PyQt.QtCore"] = _qt_core_mock

# ---------------------------------------------------------------------------
# sys.path — expose both source trees
# ---------------------------------------------------------------------------

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _write_dem(path, data, cell_size=1.0, crs="EPSG:32632", nodata=-9999.0):
    """Helper: write a numpy array as a GeoTIFF DEM."""
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


@pytest.fixture
def tmp_dem(tmp_path):
    """
    20×20 DEM sloping uniformly south: elevation 100m at top row, 62m at bottom.
    Cell size = 1m, CRS = EPSG:32632.
    """
    data = np.fromfunction(
        lambda r, c: 100.0 - r * 2.0, (20, 20), dtype=float
    )
    return _write_dem(str(tmp_path / "dem.tif"), data)


@pytest.fixture
def tmp_dem_flat(tmp_path):
    """20×20 flat DEM at 50m elevation."""
    data = np.full((20, 20), 50.0, dtype="float32")
    return _write_dem(str(tmp_path / "dem_flat.tif"), data)


@pytest.fixture
def tmp_dem_bowl(tmp_path):
    """
    20×20 bowl-shaped DEM: edges at 100m, centre at 50m.
    Useful for ponding tests.
    """
    r = np.arange(20)
    c = np.arange(20)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    data = 50.0 + ((rr - 9.5) ** 2 + (cc - 9.5) ** 2) * 0.3
    return _write_dem(str(tmp_path / "dem_bowl.tif"), data)


# ---------------------------------------------------------------------------
# Mock-geometry helpers (simulate QgsGeometry interface)
# ---------------------------------------------------------------------------

def make_mock_line_geom(coords=None):
    """
    Return a MagicMock that behaves like QgsGeometry(polyline).

    asJson()   → valid GeoJSON LineString
    length()   → shapely length of the line
    """
    from shapely.geometry import LineString, mapping

    if coords is None:
        coords = [(5.0, 10.0), (15.0, 10.0)]  # 10m horizontal line
    line = LineString(coords)
    geojson = json.dumps(mapping(line))

    g = MagicMock()
    g.asJson.return_value = geojson
    g.length.return_value = float(line.length)
    return g


def make_mock_polygon_geom(bounds=None):
    """
    Return a MagicMock that behaves like QgsGeometry(polygon).

    asJson()  → valid GeoJSON Polygon
    area()    → shapely area
    """
    from shapely.geometry import box, mapping

    if bounds is None:
        bounds = (2.0, 2.0, 12.0, 12.0)  # 10×10 = 100 m²
    poly = box(*bounds)
    geojson = json.dumps(mapping(poly))

    g = MagicMock()
    g.asJson.return_value = geojson
    g.area.return_value = float(poly.area)
    return g


@pytest.fixture
def mock_line_geom():
    return make_mock_line_geom()


@pytest.fixture
def mock_polygon_geom():
    return make_mock_polygon_geom()
