"""
project.py — Thin wrapper around QgsProject.instance().

Centralises the 50 QgsProject.instance() call sites in plugin.py so that:
  - Tests can inject a mock project without patching sys.modules.
  - Future refactors have a single seam to target.

Usage in plugin.py::

    self._project = ProjectAdapter()           # live QGIS
    self._project = ProjectAdapter(mock_proj)  # in tests
"""

from __future__ import annotations

from qgis.core import QgsProject


class ProjectAdapter:
    """Wraps QgsProject.instance() with a stable, injectable interface."""

    def __init__(self, project: QgsProject | None = None):
        # None → always call QgsProject.instance() at method-call time (correct
        # for live QGIS where the project singleton may not exist at __init__).
        self._project = project

    def _p(self) -> QgsProject:
        return self._project if self._project is not None else QgsProject.instance()

    # ------------------------------------------------------------------
    # Layer management

    def add_layer(self, layer, add_to_legend: bool = True):
        self._p().addMapLayer(layer, add_to_legend)

    def remove_layer(self, layer_or_id):
        lid = layer_or_id if isinstance(layer_or_id, str) else layer_or_id.id()
        self._p().removeMapLayer(lid)

    def layer_by_id(self, lid: str):
        return self._p().mapLayer(lid)

    def layers_by_name(self, name: str):
        return self._p().mapLayersByName(name)

    def layer_tree_root(self):
        return self._p().layerTreeRoot()

    # ------------------------------------------------------------------
    # Project metadata

    def crs(self):
        return self._p().crs()

    def transform_context(self):
        return self._p().transformContext()

    def instance(self) -> QgsProject:
        """Return the raw QgsProject for call sites not yet migrated."""
        return self._p()
