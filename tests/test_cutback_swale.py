"""The cutback swale: a level bench with a dyke along its outer edge.

FAO 13/3's level bench terrace with a lip, registered as a storage type that holds a
shallow pond on its platform. These are the build's specification — the registry entry,
the analytic figures and the burn — written before the code, so each fails on the tree
without it.

The burn's synthetic ground is a 30 m square hillside falling south at 10 %, one metre
a cell, with a 20 m cutback drawn along row 15 (``y = 14.5``) and a 3.0 m bench. The
platform therefore spans rows 14, 15 and 16, whose original ground is 98.6, 98.5 and
98.4 m: the datum is 98.5, row 14 is cut, row 16 is filled and, being the downhill edge,
carries the dyke at 98.7.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from terrainflow_assessment.core.registry.earthwork_types import (
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

KEY = "cutback_swale"
DATUM = 98.5
CREST = 98.7


def _dem(tmp_path, data, cell_size=1.0):
    path = str(tmp_path / "dem.tif")
    h, w = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
        crs="EPSG:32632", nodata=-9999.0,
        transform=from_bounds(0, 0, w * cell_size, h * cell_size, w, h),
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _hillside(tmp_path, grade=0.10, size=30, cell_size=1.0):
    """A plane falling south at *grade*: row 0 is the top of the hill."""
    data = np.fromfunction(lambda r, c: 100.0 - r * grade * cell_size, (size, size))
    return _dem(tmp_path, data, cell_size)


def _cutback(coords=((5.0, 14.5), (25.0, 14.5)), **overrides):
    ew = Earthwork(KEY, make_mock_line_geom(list(coords)), "Cutback 1")
    ew.top_width_m = 3.0
    ew.bottom_width_m = 3.0
    ew.depth = 0.20
    for name, value in overrides.items():
        setattr(ew, name, value)
    return ew


# ---------------------------------------------------------------- registry

class TestRegistry:
    def test_it_is_a_level_bench_that_holds_water(self):
        cfg = get_type(KEY)
        assert cfg.bench_mode == "level"
        assert cfg.burn_method == "bench"
        assert cfg.category == "storage"
        assert cfg.has_storage and cfg.has_capacity and cfg.has_cut and cfg.has_fill
        assert cfg.geom_type == "LineString"
        assert cfg.dyke_top_width_m == pytest.approx(0.30)
        assert cfg.riser_slope == pytest.approx(1.0)
        assert cfg.default_side_slope == 0.0
        assert cfg.independent_dims == ("depth", "top_width")
        assert cfg.derived_dims == ()

    def test_the_dialog_rows_say_what_the_aliased_dimensions_hold(self):
        cfg = get_type(KEY)
        assert cfg.depth_label == "Dyke height:"
        assert cfg.width_label == "Bench width:"
        assert cfg.default_depth == pytest.approx(0.20)
        assert cfg.depth_range == (0.10, 0.30)
        assert cfg.default_top_width == pytest.approx(4.0)
        assert cfg.top_width_range == (2.5, 8.0)

    def test_the_predicates_read_it_as_a_swale_not_a_wall(self):
        assert is_linear_store(KEY)
        assert offers_spillway(KEY)
        assert not is_crest_type(KEY)

    def test_its_spillway_policy_is_a_dyke_not_an_embankment(self):
        cfg = get_type(KEY)
        assert cfg.spillway_freeboard_m == pytest.approx(0.05)
        assert cfg.spillway_head_m == pytest.approx(0.10)
        assert cfg.spillway_head_band == (0.05, 0.15)
        # The default dyke keeps a usable crest above head plus freeboard.
        assert cfg.default_depth - cfg.spillway_head_m - cfg.spillway_freeboard_m > 0

    def test_a_fresh_feature_is_seeded_closed_at_both_ends(self):
        ew = Earthwork(KEY, make_mock_line_geom(), "x")
        assert ew.key_into_banks is True
        assert ew.depth == pytest.approx(0.20)
        assert ew.top_width_m == pytest.approx(4.0)
        assert ew.bottom_width_m == pytest.approx(4.0)     # a rectangle: side slope 0
        assert ew.ground_slope_pct is None
        assert "ground_slope_pct" in Earthwork._SERIAL_FIELDS

    def test_the_dyke_has_a_containment_source_of_its_own(self):
        from terrainflow_assessment.modules import earthwork_design as ed

        assert ed.CONTAINMENT_DYKE == "dyke"


# ---------------------------------------------------------------- analytic

class TestAnalytic:
    def test_capacity_is_the_ponded_rectangle_brim_full(self):
        geom = make_mock_line_geom([(0.0, 0.0), (10.0, 0.0)])
        m3, litres = calculate_capacity(KEY, geom, 0.20, 4.0)
        assert m3 == pytest.approx((4.0 - 0.30) * 0.20 * 10.0)    # 7.4; no freeboard off
        assert litres == pytest.approx(7400.0)

    def test_capacity_ignores_the_channel_arguments(self):
        geom = make_mock_line_geom([(0.0, 0.0), (10.0, 0.0)])
        plain = calculate_capacity(KEY, geom, 0.20, 4.0)
        assert calculate_capacity(KEY, geom, 0.20, 4.0, True, bottom_width=2.0) == plain

    def test_cut_and_fill_are_faos_balanced_figure_per_metre(self):
        """4.0 m bench, 25 %, 1:1 riser, 0.20 m dyke: C = 0.77 m²/m (test_sizing)."""
        geom = make_mock_line_geom([(0.0, 0.0), (10.0, 0.0)])
        cut = calculate_cut_volume(KEY, geom, 0.20, 4.0, ground_slope_pct=25.0)
        fill = calculate_fill_volume(KEY, geom, 0.20, 4.0, ground_slope_pct=25.0)
        assert cut == pytest.approx(0.77 * 10.0)
        assert fill == cut

    def test_without_a_ground_slope_the_analytic_earth_is_zero_not_a_guess(self):
        geom = make_mock_line_geom([(0.0, 0.0), (10.0, 0.0)])
        assert calculate_cut_volume(KEY, geom, 0.20, 4.0) == 0.0
        assert calculate_fill_volume(KEY, geom, 0.20, 4.0) == 0.0

    def test_ground_too_steep_for_the_riser_gives_zero_rather_than_raising(self):
        geom = make_mock_line_geom([(0.0, 0.0), (10.0, 0.0)])
        assert calculate_cut_volume(KEY, geom, 0.20, 4.0, ground_slope_pct=150.0) == 0.0

    def test_the_bench_property_reads_the_chain_off_the_stored_slope(self):
        ew = _cutback(top_width_m=4.0, bottom_width_m=4.0)
        assert ew.bench is None
        ew.ground_slope_pct = 25.0
        assert ew.bench.terrace_width == pytest.approx(5.53)
        assert ew.bench.riser_height == pytest.approx(1.53)
        ew.ground_slope_pct = 150.0
        assert ew.bench is None


# ---------------------------------------------------------------- burn

class TestBurn:
    def _burn(self, tmp_path, ew=None, cell_size=1.0, grade=0.10):
        ew = ew or _cutback()
        b = DEMBurner(_hillside(tmp_path, grade=grade, cell_size=cell_size))
        out = b.burn_earthworks([ew])
        return b, ew, out

    def test_the_platform_is_level_at_the_mean_original_ground(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        mask = b.burned_masks[ew.id]
        assert float(np.nanmean(b.original[mask])) == pytest.approx(DATUM, abs=1e-3)
        # Every platform cell sits at the datum except the dyke, which stands DH above.
        # (Compared with a tolerance: the DEM is float32, which cannot hold 98.7.)
        levels = sorted(np.unique(np.round(out[mask], 3)).tolist())
        assert len(levels) == 2, levels
        assert levels[0] == pytest.approx(DATUM, abs=1e-3)
        assert levels[1] == pytest.approx(CREST, abs=1e-3)

    def test_uphill_is_cut_and_downhill_is_filled_to_the_same_level(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        mask = b.burned_masks[ew.id]
        cut = mask & (b.original > DATUM + 1e-6)
        fill = mask & (b.original < DATUM - 1e-6)
        assert cut.any() and fill.any()
        assert np.allclose(out[cut], DATUM)
        assert (out[fill] >= DATUM - 1e-6).all()

    def test_the_dyke_stands_on_the_downhill_edge(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        mask = b.burned_masks[ew.id]
        dyke = mask & np.isclose(out, CREST)
        assert dyke.any()
        rows = np.nonzero(dyke)[0]
        assert rows.min() == rows.max() == 16, "the dyke is not on the lower edge row"
        assert ew.berm_crest_elevation == pytest.approx(CREST)

    def test_every_raised_cell_is_recorded_not_only_the_dyke(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        raised = b.burned_raised[ew.id]
        lifted = out > b.original + 1e-6
        assert raised[lifted].all(), "ground was raised that the overtopping check cannot see"
        dyke_cells = (b.burned_masks[ew.id] & np.isclose(out, CREST)).sum()
        assert raised.sum() > dyke_cells, "only the dyke was recorded; the fill half was not"

    def test_the_dyke_is_one_continuous_bank_with_closed_ends(self, tmp_path):
        from scipy.ndimage import label

        b, ew, out = self._burn(tmp_path)
        bank = np.isclose(out, CREST) & (out > b.original + 1e-6)
        _labels, pieces = label(bank, structure=np.ones((3, 3)))
        assert pieces == 1, f"the bank is in {pieces} pieces"
        cols = np.nonzero(bank)[1]
        assert cols.min() < 5 and cols.max() > 24, "the end caps do not reach past the ends"

    def test_unkeyed_the_bank_stops_at_the_ends(self, tmp_path):
        b, ew, out = self._burn(tmp_path, ew=_cutback(key_into_banks=False))
        bank = np.isclose(out, CREST) & (out > b.original + 1e-6)
        cols = np.nonzero(bank)[1]
        assert cols.min() >= 4 and cols.max() <= 25

    def _pond_m3(self, b, ew, out):
        """What the raster can hold: every platform cell at the datum, DH deep."""
        pond_cells = int((b.burned_masks[ew.id] & np.isclose(out, DATUM)).sum())
        assert pond_cells >= 40, pond_cells         # two full rows of 20 at the least
        return pond_cells * 0.20 * b.cell_area

    def test_it_holds_a_pond_that_the_flood_can_find(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        storage = b.feature_storage(ew)
        assert storage.volume_m3 == pytest.approx(self._pond_m3(b, ew, out), rel=0.05)
        analytic, _ = calculate_capacity(KEY, ew.geometry, ew.depth, ew.width)
        # The raster dyke is a cell wide where the drawn one is 0.30 m, so on a 1 m grid
        # the pond is narrower than the analytic rectangle — never wider.
        assert storage.volume_m3 <= analytic

    def test_the_trench_record_is_the_pond_to_the_dyke_crest(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        assert b.burned_cut[ew.id] == pytest.approx(self._pond_m3(b, ew, out), rel=0.01)

    def test_the_datum_is_the_original_ground_whatever_burned_before(self, tmp_path):
        """A berm raised over the bench's uphill row first: the platform still comes
        out at the mean of the *original* ground, and the bench cuts through the berm
        the way a swale trench cuts through a wall burned before it."""
        berm = Earthwork("berm", make_mock_line_geom([(5.0, 15.5), (25.0, 15.5)]), "Berm")
        berm.depth = 0.5
        b = DEMBurner(_hillside(tmp_path))
        out = b.burn_earthworks([berm, _cutback()])
        assert np.allclose(out[14, 6:24], DATUM), "the platform datum followed the berm"

    def test_the_sub_cell_advisory_is_about_the_pond_not_the_dyke(self, tmp_path):
        b, _ew, _out = self._burn(tmp_path)
        assert not [w for w in b.warnings if "narrowest dimension" in w], b.warnings
        coarse, _ew, _out = self._burn(tmp_path, cell_size=5.0, grade=0.02)
        assert [w for w in coarse.warnings if "narrowest dimension 2.70 m" in w], (
            coarse.warnings)

    def test_no_ground_slope_is_needed_to_burn(self, tmp_path):
        b, ew, out = self._burn(tmp_path)
        assert ew.ground_slope_pct is None
        assert not np.array_equal(out, b.original, equal_nan=True)
