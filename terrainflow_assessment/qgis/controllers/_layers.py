"""
_layers.py — safe layer-reference helpers.

Holding a raw QgsVectorLayer/QgsRasterLayer object across runs is fragile: if the
underlying layer is destroyed (the user deletes it in the QGIS panel, or it is
swapped while a map tool still holds it), the Python wrapper becomes a dead
"wrapped C/C++ object has been deleted" reference and any access raises.

Store the layer **id** (a str) instead and resolve it on demand through the
project — ``mapLayer(id)`` returns ``None`` if the layer is gone, so callers can
skip safely rather than dereference a dead wrapper.
"""

from __future__ import annotations


def resolve_layer(project, layer_id):
    """Return the live layer for *layer_id*, or None if unset/deleted."""
    if not layer_id:
        return None
    try:
        return project.instance().mapLayer(layer_id)
    except Exception:
        return None


def remove_layer(project, layer_id):
    """Remove the layer named by *layer_id* if it still exists (no-op otherwise)."""
    layer = resolve_layer(project, layer_id)
    if layer is not None:
        try:
            project.instance().removeMapLayer(layer)
        except Exception:
            pass


class MissingDemCrs(RuntimeError):
    """No DEM is loaded, so there is no coordinate system to place a layer in."""


def dem_crs(state):
    """The session DEM's CRS, as a string a layer URI or ``setCrs`` accepts.

    Every geometry this plugin creates carries DEM grid coordinates, in metres.
    The fallback here used to be ``"EPSG:4326"`` — which declares those metres to
    be degrees, puts the site a few hundred metres off the coast of Ghana, and
    says nothing about it. On a workflow that has no meaning outside a projected
    CRS there is no defensible default, so this raises instead. The callers all
    sit behind a handler that reports; a refusal the operator can act on beats a
    layer that draws in the wrong hemisphere.
    """
    info = getattr(state, "dem_info", None)
    wkt = getattr(info, "crs_wkt", None) if info is not None else None
    if not wkt:
        raise MissingDemCrs(
            "No DEM is loaded, so there is no coordinate system to draw this in. "
            "Load a DEM first.")
    return wkt


def dem_crs_or_project(state, project):
    """:func:`dem_crs`, falling back to the project's CRS rather than raising.

    For layers that are display furniture rather than measurements — where being
    drawn in the operator's CRS is merely unhelpful, not wrong.
    """
    try:
        return dem_crs(state)
    except MissingDemCrs:
        return project.instance().crs().toWkt()


def crs_object(wkt):
    """A ``QgsCoordinateReferenceSystem`` from what :func:`dem_crs` returns."""
    from qgis.core import QgsCoordinateReferenceSystem

    return QgsCoordinateReferenceSystem(wkt)
