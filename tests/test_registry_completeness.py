"""Every registered earthwork type is wired everywhere a type has to be.

The registry's own docstring says adding a type "requires touching exactly this one
file", and for the map layer and the tool menu that is true — both iterate
``all_types()``. Everything else that knows a type by name is a table somewhere else:
the burn dispatch, the panel glyph, the report glyph, the map-symbol grammar. A type
missing from any of them does not fail. It draws, it lists, it reports a capacity of
0.0, it changes no ground, and nothing says so. This module turns that into a red test.

The burn table is the one that matters most, and it was the one with no guard at all:
``burn_earthworks`` kept a private dict keyed on ``ew.type`` and skipped anything it
did not know with a bare ``continue``.
"""

import numpy as np

from terrainflow_assessment.core.registry.earthwork_types import (
    _REGISTRY,
    EarthworkTypeConfig,
    all_types,
    register_type,
)
from terrainflow_assessment.modules import report_charts
from terrainflow_assessment.modules.earthwork_design import DEMBurner, Earthwork
from terrainflow_assessment.qgis import _theme
from terrainflow_assessment.qgis.controllers import _symbols


def _probe(burn_method="probe"):
    """A type wired to nothing: a burn method nobody implements, in no glyph table."""
    return EarthworkTypeConfig(
        key="unwired_probe",
        label="Unwired probe",
        geom_type="LineString",
        has_storage=True,
        has_capacity=True,
        has_cut=True,
        has_fill=False,
        burn_method=burn_method,
        style=("line", "#000000", "1.0"),
    )


def _burn_is_missing(cfg):
    name = DEMBurner._BURN_DISPATCH.get(cfg.burn_method)
    return name is None or not callable(getattr(DEMBurner, name, None))


def _gaps(cfg):
    """The tables *cfg* is missing from, by name; empty when it is fully wired."""
    gaps = []
    if _burn_is_missing(cfg):
        gaps.append(f"DEMBurner._BURN_DISPATCH[{cfg.burn_method!r}]")
    if cfg.key not in _theme.GLYPH:
        gaps.append("qgis/_theme.GLYPH")
    if cfg.key not in report_charts._GLYPH:
        gaps.append("modules/report_charts._GLYPH")
    if cfg.key not in _symbols._GRAMMAR:
        gaps.append("qgis/controllers/_symbols._GRAMMAR")
    return gaps


def test_every_registered_type_is_wired_everywhere():
    offenders = {key: _gaps(cfg) for key, cfg in all_types().items() if _gaps(cfg)}
    assert not offenders, (
        "registered earthwork types missing from a per-type table — each would draw on "
        "the map and burn nothing, or fall back to a bullet glyph and a bare line:\n  "
        + "\n  ".join(f"{k}: {', '.join(v)}" for k, v in offenders.items())
    )


def test_every_panel_glyph_is_one_bmp_code_point():
    """Qt on a DejaVu host renders one BMP character per chip. A surrogate pair or a
    two-character string shows as a box, or as two."""
    bad = {k: g for k, g in _theme.GLYPH.items() if len(g) != 1 or ord(g) > 0xFFFF}
    assert not bad, f"panel glyphs that are not a single BMP code point: {bad}"


def test_an_unwired_type_is_caught(registry_restored):
    """The negative case. Without it the positive test could pass by checking nothing."""
    register_type(_probe())
    assert _gaps(all_types()["unwired_probe"]) == [
        "DEMBurner._BURN_DISPATCH['probe']",
        "qgis/_theme.GLYPH",
        "modules/report_charts._GLYPH",
        "qgis/controllers/_symbols._GRAMMAR",
    ]


def test_an_unwired_type_burns_nothing_and_says_so(tmp_dem, mock_line_geom,
                                                    registry_restored):
    """The §0 failure, made loud: the DEM is untouched and the burner says why."""
    register_type(_probe())
    burner = DEMBurner(tmp_dem)
    burned = burner.burn_earthworks([Earthwork("unwired_probe", mock_line_geom, "Probe 1")])
    assert np.array_equal(burned, burner.original, equal_nan=True)
    assert any("Probe 1" in w and "'unwired_probe'" in w and "changes no ground" in w
               for w in burner.warnings), burner.warnings


def test_a_registered_type_may_share_a_shipped_burn(tmp_dem, mock_line_geom,
                                                     registry_restored):
    """The dispatch is keyed on ``burn_method``, not on the type key, so a new type can
    name an existing burn and get it — with no warning, because nothing is missing."""
    register_type(_probe(burn_method="berm"))
    burner = DEMBurner(tmp_dem)
    ew = Earthwork("unwired_probe", mock_line_geom, "Probe 2")
    burned = burner.burn_earthworks([ew])
    assert not np.array_equal(burned, burner.original, equal_nan=True)
    assert not any("no burn method" in w for w in burner.warnings), burner.warnings


def test_an_unregistered_type_that_names_a_shipped_burn_still_burns(tmp_dem, mock_line_geom,
                                                                     registry_restored):
    """A design file can name a type the registry does not know. The old table burned
    it whenever the key matched a burn, and that has to stay true: nothing that burned
    before this table existed stops burning because of it."""
    _REGISTRY.pop("swale")
    burner = DEMBurner(tmp_dem)
    ew = Earthwork("swale", mock_line_geom, "Orphan swale")
    burned = burner.burn_earthworks([ew])
    assert not np.array_equal(burned, burner.original, equal_nan=True)
    assert not any("no burn method" in w for w in burner.warnings), burner.warnings
