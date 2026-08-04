"""
Quarantined checks — known not to complete in reasonable time.

Kept separate so the rest of the suite stays fast: process isolation means this
module burns its timeout alone rather than stalling the module it came from.

    .\run_qgis_tests.ps1 slow --timeout=1800     # run these deliberately
"""

from _harness import PluginHarness


def check_recommend_ponds(dem_path):
    """Pond-site recommendation.

    STATUS: does not finish. Ran >12 min on a 120x120 (5.8 ha) DEM before being
    killed, with no result and no error. Every other analysis on the same DEM
    completes in seconds, so this is not raw DEM size.

    Unresolved: whether the cause is the smooth synthetic terrain (a noise-free
    parabolic valley gives perfectly flat contours and no pits, which could be a
    pathological input for a pond-site search) or a non-terminating loop that
    real terrain also hits. Worth resolving either way — if it is the former,
    the code still hangs rather than degrading on flat ground.
    """
    with PluginHarness(dem_path) as h:
        h.run_baseline()
        h.panel.recommend_ponds_requested.emit()
        h.assert_no_errors("recommend ponds")
