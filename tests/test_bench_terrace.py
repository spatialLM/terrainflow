"""The bench terrace: FAO's reverse-sloped bench, cut across the slope with no dyke.

A drainage type — it breaks a long slope into short steps and sheds — so it holds no
capacity and offers no spillway. It shares the cutback swale's burn and FAO chain; what
differs is the tilt across the bench, 5 % back into the hill about the drawn line.

The synthetic ground is the cutback tests' 30 m hillside falling south at 10 %, one metre
a cell, with the terrace drawn along row 15 (``y = 14.5``). Rows increase downhill.
"""

import numpy as np
import pytest

from terrainflow_assessment.core.registry.earthwork_types import (
    bench_mode_of,
    get_type,
    is_crest_type,
    is_linear_store,
    offers_spillway,
)
from terrainflow_assessment.modules.earthwork_design import (
    DEMBurner,
    Earthwork,
    calculate_capacity,
    calculate_cut_volume,
    calculate_fill_volume,
)
from tests.conftest import make_mock_line_geom
from tests.test_cutback_swale import _hillside

KEY = "bench_terrace"


def _terrace(coords=((5.0, 14.5), (25.0, 14.5)), width=4.0, **overrides):
    ew = Earthwork(KEY, make_mock_line_geom(list(coords)), "Terrace 1")
    ew.top_width_m = width
    for name, value in overrides.items():
        setattr(ew, name, value)
    return ew


# ---------------------------------------------------------------- registry

class TestRegistry:
    def test_it_is_a_reverse_bench_that_sheds(self):
        cfg = get_type(KEY)
        assert cfg.geom_type == "LineString"
        assert cfg.category == "control"
        assert cfg.burn_method == "bench"
        assert bench_mode_of(KEY) == "reverse"
        assert not cfg.has_storage and not cfg.has_capacity
        assert cfg.has_cut and cfg.has_fill
        assert cfg.dyke_top_width_m == 0.0

    def test_it_takes_a_bench_width_and_no_depth(self):
        cfg = get_type(KEY)
        assert cfg.independent_dims == ("top_width",)
        assert cfg.width_label == "Bench width:"

    def test_it_is_not_a_store_a_wall_or_a_weir(self):
        assert not is_linear_store(KEY)
        assert not is_crest_type(KEY)
        assert not offers_spillway(KEY)


# ---------------------------------------------------------------- the primitive

class TestSignedOffsetFromPath:
    def _offset(self, cell_size=1.0):
        from terrainflow_assessment.modules.burn_strategy import signed_offset_from_path

        path = np.zeros((11, 7), dtype=bool)
        path[5, :] = True
        side = np.zeros_like(path)
        side[6:, :] = True
        return signed_offset_from_path(path, side, cell_size)

    def test_zero_on_the_path_positive_on_the_side_negative_off_it(self):
        off = self._offset()
        assert np.all(off[5] == 0.0)
        assert off[8, 3] == pytest.approx(3.0)
        assert off[2, 3] == pytest.approx(-3.0)

    def test_distances_are_in_metres_per_axis(self):
        off = self._offset(cell_size=(2.0, 5.0))       # 2 m rows, 5 m columns
        assert off[8, 3] == pytest.approx(6.0)

    def test_an_empty_path_means_no_tilt(self):
        from terrainflow_assessment.modules.burn_strategy import signed_offset_from_path

        empty = np.zeros((4, 4), dtype=bool)
        assert signed_offset_from_path(empty, empty) is None


# ---------------------------------------------------------------- analytic

class TestAnalytic:
    def test_it_holds_no_capacity(self):
        ew = _terrace()
        assert calculate_capacity(KEY, ew.geometry, 0.0, 4.0) == (0.0, 0.0)

    def test_cut_and_fill_are_one_figure_by_construction(self):
        """FAO balances the bench about its centreline: what comes off the uphill half
        builds the downhill half. Equal figures are the definition, not a copy-paste."""
        ew = _terrace()
        cut = calculate_cut_volume(KEY, ew.geometry, 0.0, 4.0, ground_slope_pct=20.0)
        fill = calculate_fill_volume(KEY, ew.geometry, 0.0, 4.0, ground_slope_pct=20.0)
        assert cut > 0
        assert cut == fill

    def test_the_rise_across_the_bench_is_in_the_riser(self):
        from terrainflow_assessment.core.sizing import bench_geometry

        b = bench_geometry(4.0, 20.0, riser_slope=1.0, mode="reverse")
        assert b.edge_rise == pytest.approx(0.20)


# ---------------------------------------------------------------- burn

class TestBurn:
    def _burn(self, tmp_path, ew=None):
        ew = ew or _terrace()
        b = DEMBurner(_hillside(tmp_path))
        out = b.burn_earthworks([ew])
        return b, ew, out

    def test_the_bench_tilts_back_into_the_hill(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        mask = b.burned_masks[ew.id]
        rows = sorted(set(np.nonzero(mask)[0].tolist()))
        mid = 15
        inner, outer = min(rows), max(rows)
        col = 15
        # Downhill (outer) edge stands above the uphill (inner) edge: water runs back
        # to the cut face, not over the riser.
        assert out[outer, col] > out[inner, col]
        # 5 % per metre across the drawn line, on both sides of it.
        assert out[mid + 1, col] - out[mid, col] == pytest.approx(0.05, abs=1e-3)
        assert out[mid, col] - out[mid - 1, col] == pytest.approx(0.05, abs=1e-3)

    def test_the_tilt_is_about_the_datum_so_cut_and_fill_still_balance(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        mask = b.burned_masks[ew.id]
        datum = float(np.nanmean(b.original[mask]))
        assert float(np.mean(out[mask])) == pytest.approx(datum, abs=0.02)

    def test_it_is_level_along_its_run(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        assert np.allclose(out[15, 6:24], out[15, 6])

    def test_no_dyke_and_no_bank(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        assert ew.berm_crest_elevation is None
        mask = b.burned_masks[ew.id]
        raised = b.burned_raised.get(ew.id)
        if raised is not None:
            assert not (raised & ~mask).any(), "a terrace raised ground off its bench"

    def test_a_level_cutback_is_unchanged_by_the_tilt(self, tmp_path):
        """The tilt is keyed on the mode, so the cutback's burn is the cutback's."""
        from tests.test_cutback_swale import CREST, DATUM, _cutback

        ew = _cutback()
        b = DEMBurner(_hillside(tmp_path))
        out = b.burn_earthworks([ew])
        levels = sorted(np.unique(np.round(out[b.burned_masks[ew.id]], 3)).tolist())
        assert levels == [pytest.approx(DATUM, abs=1e-3), pytest.approx(CREST, abs=1e-3)]
