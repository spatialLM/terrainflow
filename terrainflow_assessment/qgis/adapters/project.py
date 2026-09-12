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
    #
    # Four more methods lived here and none of them had a caller: `add_layer`,
    # `remove_layer`, `layer_by_id` and `layers_by_name`. Two were worse than
    # merely unused. `add_layer` was the only `addMapLayer(` outside `_groups.py`
    # — a ready-made bypass of the rule that every layer is filed under its stage
    # group, and invisible to the gate in `test_architecture.py`, which scans the
    # controllers directory and not this one. `layers_by_name` packaged the
    # by-name lookup `_state.py:157-161` bans, the one that picks the wrong
    # feature as soon as two share a name.

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
