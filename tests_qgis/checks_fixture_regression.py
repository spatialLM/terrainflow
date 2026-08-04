"""Numerical regression against a fixed piece of real terrain.

Every other check asks whether the code *runs*. This one asks whether it still gives the
same **answers**. It runs the pipeline over a committed clip of the Quail Island survey
and compares the results against values recorded when the fixture was made.

Why a real DEM rather than the synthetic one the rest of the suite uses: the synthetic
terrain is smooth and noise-free — no pits, no flats, perfectly regular contours — so it
cannot tell you whether the sizing maths behaves on ground that has all three. The
fixture is 16 ha of actual hillside carrying an 8 ha catchment, which is the scale the
plugin is designed for.

**A moved number here is information, not necessarily a bug.** Level-pool swale
segmentation and the regional-CN work will both change these figures deliberately. The
point is that you find out, and then decide. When a change is intended, re-record the
constants below in the same commit that causes them to move.

Tolerances are proportional and deliberately tight (0.5%). Rounding differences across
platforms land far inside that; a real change to the sizing or routing maths lands well
outside it.
"""

import os

from _harness import PluginHarness
from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

FIXTURE_DEM = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "quail_island_catchment.tif",
)

# Clip bounds (EPSG:2193), from the fixture's own header. Earthworks are placed at fixed
# offsets inside these so the geometry is identical on every run and on every machine.
X0, Y0 = 1574561.0, 5169232.0
SIZE = 400.0

# Storm held constant so a changed panel default cannot silently move these numbers.
STORM = {
    "rainfall_mm": 120.0,
    "duration_hr": 24.0,
    "cn": 61,
    "stream_threshold_ha": 1.0,
    "sizing_basis": "coefficient",
    "runoff_coefficient": 0.50,
}

# Recorded 2026-08-05 against fixture quail_island_catchment.tif.
#
# The three catchment figures and uncaptured_cells come off the conditioned flow graph —
# they are the terrain-dependent ones, and what the regional-CN work will move. The two
# capacities come from the sizing primitives, which is what level-pool segmentation will
# move. runoff_mm and catchment_area_ha are input-derived sanity anchors: if either drifts
# without the storm changing, something upstream of the terrain maths has broken.
EXPECTED = {
    "runoff_mm": 60.0000,
    "catchment_area_ha": 12.9600,
    "exit_points": 4.0000,
    "swale_a_catchment_ha": 1.6551,
    "swale_b_catchment_ha": 2.1829,
    "basin_c_catchment_ha": 0.1884,
    "uncaptured_cells": 89336.0000,
    "swale_capacity_m3": 201.6000,
    "basin_capacity_m3": 936.0000,
}
TOLERANCE = 0.005


def _swale_line(northing_offset):
    """A straight line across the clip at a fixed northing — deterministic by construction."""
    y = Y0 + northing_offset
    return QgsGeometry.fromWkt(
        f"LINESTRING ({X0 + 60} {y}, {X0 + 200} {y}, {X0 + 340} {y})")


def _basin_polygon(east_offset, north_offset, side=30.0):
    x, y = X0 + east_offset, Y0 + north_offset
    return QgsGeometry.fromWkt(
        f"POLYGON (({x} {y}, {x + side} {y}, {x + side} {y + side}, {x} {y + side}, "
        f"{x} {y}))")


def _apply_storm(panel):
    """Pin the storm inputs through the panel's own restore path."""
    inputs = dict(panel.collect_inputs())
    inputs.update(STORM)
    panel.apply_inputs(inputs)


def _boundary_layer(inset_m=20.0):
    """A site boundary inset from the fixture's own bounds.

    The harness's `site_boundary_layer` is shaped around the synthetic DEM, so the
    fixture needs its own. Without a boundary the exit-point search has no perimeter to
    test against and reports none at all.
    """
    x0, y0 = X0 + inset_m, Y0 + inset_m
    x1, y1 = X0 + SIZE - inset_m, Y0 + SIZE - inset_m
    layer = QgsVectorLayer("Polygon?crs=EPSG:2193", "Fixture boundary", "memory")
    feature = QgsFeature()
    feature.setGeometry(QgsGeometry.fromWkt(
        f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"))
    layer.dataProvider().addFeatures([feature])
    layer.updateExtents()
    return layer


def _size(earthwork):
    """Compute and store capacity the way the properties dialog does.

    `add_earthwork` puts a feature straight into the manager, bypassing the dialog — so
    without this every capacity stays at its 0.0 default and the check asserts nothing.
    """
    from terrainflow_assessment.modules.earthwork_design import calculate_capacity

    m3, litres = calculate_capacity(
        earthwork.type,
        earthwork.geometry,
        earthwork.depth,
        earthwork.width,
        companion_berm=getattr(earthwork, "companion_berm", False),
        bottom_width=getattr(earthwork, "bottom_width_m", None),
        batter_run=getattr(earthwork, "batter_run_m", None),
    )
    earthwork.capacity_m3 = m3
    earthwork.capacity_l = litres
    return earthwork


def _build_design(harness):
    """Two swales across the slope and one basin, at fixed coordinates."""
    swale = harness.add_earthwork("swale", geometry=_swale_line(140.0), name="Swale A")
    swale.depth = 0.6
    swale.top_width_m = 2.0
    swale.bottom_width_m = 1.0

    swale_b = harness.add_earthwork("swale", geometry=_swale_line(220.0), name="Swale B")
    swale_b.depth = 0.6
    swale_b.top_width_m = 2.0
    swale_b.bottom_width_m = 1.0

    basin = harness.add_earthwork("basin", geometry=_basin_polygon(170.0, 60.0),
                                  name="Basin C")
    basin.depth = 1.5
    basin.batter_run_m = 2.0

    return [_size(ew) for ew in (swale, swale_b, basin)]


def _relative_gap(actual, expected):
    if expected == 0:
        return abs(actual)
    return abs(actual - expected) / abs(expected)


def _compare(observed, failures):
    if not EXPECTED:
        print("    NO RECORDED VALUES — copy the block below into EXPECTED:")
        print("    EXPECTED = {")
        for key, value in observed.items():
            print(f'        "{key}": {value:.4f},')
        print("    }")
        failures.append(
            "EXPECTED is empty, so nothing was actually asserted. Record the values above."
        )
        return

    for key, expected in EXPECTED.items():
        actual = observed.get(key)
        if actual is None:
            failures.append(f"{key}: not produced by this run")
            continue
        gap = _relative_gap(actual, expected)
        marker = "ok " if gap <= TOLERANCE else "MOVED"
        print(f"    {marker} {key:24} {actual:14.3f}  (recorded {expected:.3f}, "
              f"{gap * 100:.2f}%)")
        if gap > TOLERANCE:
            failures.append(
                f"{key}: {actual:.3f} vs recorded {expected:.3f} "
                f"({gap * 100:.2f}% > {TOLERANCE * 100:.1f}%)"
            )


def check_fixture_dem_is_present_and_sane(dem_path):
    """The fixture must be committed, projected, and actual land rather than sea."""
    import numpy as np
    import rasterio

    assert os.path.exists(FIXTURE_DEM), (
        f"fixture DEM missing: {FIXTURE_DEM} — it is committed to the repo, not generated"
    )
    with rasterio.open(FIXTURE_DEM) as src:
        assert src.crs and src.crs.is_projected, "fixture DEM must be in a projected CRS"
        assert src.width == src.height == 400, f"unexpected grid {src.width}x{src.height}"
        assert abs(abs(src.transform.a) - 1.0) < 1e-6, "expected 1 m cells"
        data = src.read(1)

    # The survey encodes sea as a flat -0.1 m surface rather than nodata, so a fixture cut
    # from it can look valid while being mostly water. It happened on the first attempt.
    land = float((data > 1.0).mean())
    assert land > 0.95, f"fixture is only {land * 100:.0f}% land — re-cut it"
    assert np.ptp(data) > 20.0, "fixture has too little relief to route flow"


def check_real_terrain_numbers_have_not_moved(dem_path):
    """Run the pipeline over the fixture and compare against recorded values."""
    from qgis.core import QgsProject

    with PluginHarness(FIXTURE_DEM, load_boundary=False) as h:
        boundary = _boundary_layer()
        QgsProject.instance().addMapLayer(boundary)
        h.panel.boundary_changed.emit(boundary)

        _apply_storm(h.panel)
        _build_design(h)

        h.run_baseline()
        h.assert_no_errors("baseline over the fixture DEM")

        result = h.state.baseline_result
        assert result, "baseline produced no result over the fixture DEM"

        manager = h.state.earthwork_manager
        by_name = {ew.name: ew for ew in manager.get_all()}

        # Terrain-dependent: which cells drain to which feature. Computed off the flow
        # graph, so this is the figure the regional-CN work will move. It also confirms
        # the baseline_finished -> live-assessment wiring fired, since nothing else here
        # asks for catchments.
        counts = h.state.catchment_counts or {}
        cell_area = (h.state.flow_grid_meta or {}).get("cell_area_m2", 1.0)

        def catchment_ha(name):
            return float(counts.get(by_name[name].id, 0)) * cell_area / 1e4

        observed = {
            "runoff_mm": float(result.get("runoff_mm", 0.0)),
            "catchment_area_ha": float(result.get("catchment_area_m2", 0.0)) / 1e4,
            "exit_points": float(len(result.get("exit_points", []))),
            "swale_a_catchment_ha": catchment_ha("Swale A"),
            "swale_b_catchment_ha": catchment_ha("Swale B"),
            "basin_c_catchment_ha": catchment_ha("Basin C"),
            "uncaptured_cells": float(h.state.catchment_exit_cells
                                      + h.state.catchment_sink_cells),
            "swale_capacity_m3": float(by_name["Swale A"].capacity_m3 or 0.0),
            "basin_capacity_m3": float(by_name["Basin C"].capacity_m3 or 0.0),
        }

        print("\n    --- fixture numbers ---")
        failures = []
        _compare(observed, failures)
        print(f"    (exit points: {len(result.get('exit_points', []))})")

        assert not failures, (
            "numbers moved against the recorded fixture:\n      "
            + "\n      ".join(failures)
            + "\n    If the change was intended, re-record EXPECTED in this file."
        )
