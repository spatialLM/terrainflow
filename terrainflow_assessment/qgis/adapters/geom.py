"""
geom.py — Geometry conversion utilities between QgsGeometry and shapely.

This module intentionally imports nothing from qgis at module level so it
can be tested without a live QGIS runtime (the qgs_geometry argument is
duck-typed: any object with an .asJson() method works).
"""

from __future__ import annotations

import json

from shapely.geometry import shape as _shapely_shape
from shapely.geometry.base import BaseGeometry


def qgs_to_shapely(qgs_geometry) -> BaseGeometry:
    """Convert a QgsGeometry to a shapely geometry via GeoJSON round-trip."""
    return _shapely_shape(json.loads(qgs_geometry.asJson()))


def shapely_length(geom) -> float:
    """
    Return the length of *geom*, handling both shapely (property) and
    QgsGeometry (callable method) transparently.

    This bridge function lets calculation code work correctly whether
    Earthwork.geometry stores a QgsGeometry (current) or a shapely
    geometry (future, after Step 6 migration).
    """
    length = geom.length
    if callable(length):
        return length()
    return float(length)


def shapely_area(geom) -> float:
    """
    Return the area of *geom*, handling both shapely (property) and
    QgsGeometry (callable method) transparently.
    """
    area = geom.area
    if callable(area):
        return area()
    return float(area)
