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
