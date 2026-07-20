"""
reporting.py — ReportingController

Handles HTML report export.
"""

from __future__ import annotations

import os

from qgis.PyQt.QtWidgets import QFileDialog


class ReportingController:
    def __init__(self, state, panel, project, iface, canvas):
        self._state = state
        self._panel = panel
        self._project = project
        self._iface = iface
        self._canvas = canvas

    def export_report(self):
        if self._state.comparison is None:
            self._iface.messageBar().pushWarning(
                "TerrainFlow Assessment",
                "Run baseline + simulation first to generate a comparison."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self._iface.mainWindow(),
            "Save Report",
            os.path.expanduser("~/terrainflow_report.html"),
            "HTML Files (*.html)",
        )
        if not path:
            return

        from terrainflow_assessment.modules.reporting import export_html
        try:
            export_html(self._state.comparison, path)
            self._iface.messageBar().pushSuccess(
                "TerrainFlow Assessment",
                f"Report saved to {path}"
            )
            import sys
            if sys.platform == "win32":
                os.startfile(path)
        except Exception as exc:
            self._iface.messageBar().pushCritical(
                "TerrainFlow Assessment", f"Export failed: {exc}"
            )
