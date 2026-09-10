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


def polygons_in_dem_crs(layer, dem_crs_wkt):
    """Every polygon in *layer*, as shapely, reprojected into the DEM's CRS.

    One home for "take this vector layer and give me shapes the raster tier can use".
    There were two implementations and only one of them reprojected: the usable-area
    build read ``feat.geometry().asJson()`` straight into shapely in the *layer's* CRS,
    then handed it to code that intersects it against DEM-CRS contours. On a mismatch
    every intersection came back empty and every contour was dropped without a word —
    NZTM eastings are around 1.5e6 and WGS84 longitudes around 172, so the two never
    touch. The keypoint boundary mask had it right all along, thirty lines away.

    Uses ``QgsCoordinateTransform`` rather than a geopandas ``to_crs``: this moves a
    handful of vertices, and standing up a GeoDataFrame and a pyproj pipeline to do it
    is most of the cost of the call.

    QGIS imports are function-level, so this module still imports nothing from qgis at
    module level and stays testable without a live runtime.

    Returns ``[]`` when the layer has no usable polygons. Raises ``ValueError`` when
    *dem_crs_wkt* is empty — a polygon with no target CRS is the same bug wearing a
    different hat, and guessing would put it straight back.
    """
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsCoordinateTransform,
        QgsGeometry,
        QgsProject,
    )

    if not dem_crs_wkt:
        raise ValueError(
            "Cannot place the area layer without the DEM's CRS — load a DEM first.")

    target = QgsCoordinateReferenceSystem()
    target.createFromWkt(dem_crs_wkt)
    source = layer.crs()

    transform = None
    if source.isValid() and target.isValid() and source != target:
        transform = QgsCoordinateTransform(source, target, QgsProject.instance())

    polys = []
    for feat in layer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        if transform is not None:
            # ``transform`` mutates, and this geometry belongs to the feature. Copy
            # first: reprojecting the user's own layer as a side effect of reading it
            # is a surprise that would be very hard to trace back to here.
            geom = QgsGeometry(geom)
            if geom.transform(transform) != 0:
                # A vertex that will not transform is not a shape we can honestly
                # intersect against the terrain. Skip it rather than pass the
                # untransformed geometry on, which is the failure this exists to stop.
                continue
        polys.append(qgs_to_shapely(geom))
    return polys


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
