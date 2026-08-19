"""The designed spillway, cut into the terrain.

Until this landed the burn was spillway-blind: a placed spillway changed no raster, no
routing and no pond, so the re-analysis could not reproduce the overflow the design
intended and every overtopping advisory shipped a caveat saying so.

These pin the two halves of the fix. The pure grid functions
(:mod:`~terrainflow_assessment.modules.burn_strategy`) decide *whether* a notch can be
cut and refuse with a reason when it cannot; ``DEMBurner`` decides *where* the crest
lies and makes the cut the last thing to touch those cells. Each refusal has its own
case here, because a notch that quietly does nothing is worse than no notch at all — it
looks like a working spillway in every figure the design tier prints.
"""

import numpy as np
import pytest

from terrainflow_assessment.modules.burn_strategy import (
    daylight_reach,
    notch_pool,
    spillway_burn_width,
    spillway_notch,
)
from terrainflow_assessment.modules.earthwork_design import (
    DEMBurner,
    Spillway,
    StageStorage,
)
from tests.conftest import make_mock_line_geom
from tests.test_dem_burner import _mock_ew, _write_dem

# ---------------------------------------------------------------------------
# spillway_burn_width — rasterisability, not conservatism
# ---------------------------------------------------------------------------

class TestSpillwayBurnWidth:
    @pytest.mark.parametrize("wanted,cell,expected", [
        (1.4, 1.0, 2.0),
        (2.0, 1.0, 2.0),        # already a whole cell: unchanged, not bumped
        (0.2, 1.0, 1.0),        # sub-cell still claims one
        (2.1, 0.5, 2.5),
        (0.2, 0.25, 0.25),      # finer data cuts a finer sill
    ])
    def test_rounds_up_to_whole_cells(self, wanted, cell, expected):
        assert spillway_burn_width(wanted, cell) == pytest.approx(expected)

    def test_uses_the_coarser_axis_on_a_non_square_grid(self):
        """The crest axis is not known here, and the coarser axis is the one that can
        fail to resolve the sill — the same caution ``taper_reach`` carries."""
        assert spillway_burn_width(1.4, (5.0, 2.0)) == pytest.approx(5.0)

    def test_no_width_still_claims_a_cell(self):
        assert spillway_burn_width(0.0, 1.0) == pytest.approx(1.0)
        assert spillway_burn_width(None, 1.0) == pytest.approx(1.0)

    def test_a_degenerate_cell_size_passes_the_width_through(self):
        assert spillway_burn_width(1.4, 0.0) == pytest.approx(1.4)


# ---------------------------------------------------------------------------
# notch_pool — "still inside the pond" has to mean enclosed
# ---------------------------------------------------------------------------

#: Rows 8-12 of every fixture below are a channel running east; everything outside
#: them is a 20 m valley wall, so a pond in the channel is bounded north and south and
#: the questions here stay one-dimensional without also being unbounded.
_CHANNEL = slice(8, 13)


def _channel_dem():
    """A 21x21 valley with a channel falling east from 12.0 m to 2.0 m.

    No structure on it. A dam burned across the channel makes a real pond here, which
    is what the storage measurements need — a hollow that was already there nets out
    against the baseline and measures zero.
    """
    dem = np.full((21, 21), 20.0)
    dem[_CHANNEL, :] = np.linspace(12.0, 2.0, 21)
    return dem


def _bowl_with_bank():
    """The same channel, with a bank across it at column 10 and a pond behind it.

    The bank stands at 10.0 and the pond floor is 6.0 in columns 6-9, so a crest at
    8.0 has the bank above it and the pond below it — which is the geometry every
    refusal below turns on.
    """
    dem = _channel_dem()
    dem[_CHANNEL, 6:10] = 6.0       # the pond floor
    dem[_CHANNEL, 10] = 10.0        # the bank
    return dem


class TestNotchPool:
    def test_the_pond_behind_a_bank_is_enclosed(self):
        dem = _bowl_with_bank()
        foot = np.zeros(dem.shape, dtype=bool)
        foot[_CHANNEL, 6:10] = True
        pool = notch_pool(dem, foot, 8.0)
        assert pool[_CHANNEL, 6:10].all()
        assert not pool[:, 11:].any(), "ground below the bank is not inside the pond"

    def test_open_hillside_below_the_crest_is_not_a_pond(self):
        """The case that would refuse every spillway on falling ground.

        "Below the crest and touching the footprint" describes the whole slope under
        the sill. Only ground the crest is actually holding in counts.
        """
        dem = _channel_dem()
        foot = np.zeros(dem.shape, dtype=bool)
        foot[_CHANNEL, 8:10] = True
        assert not notch_pool(dem, foot, 8.0).any()

    def test_an_empty_footprint_has_no_pool(self):
        dem = _bowl_with_bank()
        assert not notch_pool(dem, np.zeros(dem.shape, dtype=bool), 8.0).any()


# ---------------------------------------------------------------------------
# spillway_notch — cuts to the crest, or refuses and says which fault it is
# ---------------------------------------------------------------------------

def _crest_cells(rows=(9, 10, 11), col=10):
    return (np.array(rows), np.array([col] * len(rows)))


class TestSpillwayNotch:
    def test_cuts_to_the_crest_and_no_further(self):
        dem = _bowl_with_bank()
        cut = spillway_notch(dem, _crest_cells(), 8.0, (0, 1), cell_size=1.0)
        assert cut.daylit
        assert cut.dem[9, 10] == pytest.approx(8.0)
        # Never a raise, and never past the crest: the ground either side is untouched.
        assert cut.dem[9, 9] == pytest.approx(dem[9, 9])
        assert (cut.dem <= dem + 1e-9).all(), "the notch raised ground somewhere"

    def test_the_cut_is_absolute_so_a_second_pass_changes_nothing(self):
        """Order-independence, which is what makes the post-pass safe."""
        dem = _bowl_with_bank()
        once = spillway_notch(dem, _crest_cells(), 8.0, (0, 1), cell_size=1.0).dem
        twice = spillway_notch(once, _crest_cells(), 8.0, (0, 1), cell_size=1.0).dem
        assert np.array_equal(once, twice)

    def test_the_as_burned_sill_equals_the_designed_sill_on_a_clean_cut(self):
        dem = _bowl_with_bank()
        cut = spillway_notch(dem, _crest_cells(), 8.0, (0, 1), cell_size=1.0)
        assert cut.sill_elev == pytest.approx(8.0)

    def test_a_path_already_below_the_sill_reports_a_lower_as_burned_sill(self):
        """Cutting moved nothing, and the figure says so rather than echoing the design."""
        dem = _channel_dem()
        cut = spillway_notch(dem, _crest_cells(col=15), 8.0, (0, 1), cell_size=1.0)
        assert cut.daylit
        assert cut.sill_elev < 8.0

    def test_a_notch_into_rising_ground_cuts_nothing(self):
        """No daylight: the march hits its cap with the bank still above the crest."""
        dem = _channel_dem()[:, ::-1].copy()                 # rises eastward
        cut = spillway_notch(dem, _crest_cells(col=4), 3.0, (0, 1),
                             max_reach_m=5.0, cell_size=1.0)
        assert not cut.daylit
        assert cut.cells[0].size == 0
        assert np.array_equal(cut.dem, dem)
        assert cut.reach_m == pytest.approx(5.0)

    def test_a_notch_that_daylights_inside_its_own_pool_is_refused(self):
        """The one failure the plain daylight test cannot see.

        The bank wraps round, so the march walks downhill and is still in the pond. The
        notch would drain one part of the pool into another and the water would leave
        exactly where it was leaving before.
        """
        dem = _bowl_with_bank()
        foot = np.zeros(dem.shape, dtype=bool)
        foot[_CHANNEL, 6:10] = True
        pool = notch_pool(dem, foot, 8.0)
        # Marching *west*, from the bank back into the pond.
        cut = spillway_notch(dem, _crest_cells(), 8.0, (0, -1),
                             cell_size=1.0, pool=pool)
        assert not cut.daylit
        assert cut.into_pool
        assert np.array_equal(cut.dem, dem)

    def test_a_crest_at_or_below_the_burned_floor_is_refused(self):
        """It would empty the feature. Measured against what was cut, never against
        the analytic ``rim − depth``: a footprint too narrow for its batter never
        reaches full depth, and the analytic figure would refuse a legitimate sill."""
        dem = _bowl_with_bank()
        cut = spillway_notch(dem, _crest_cells(), 6.0, (0, 1),
                             cell_size=1.0, floor_elev=6.0)
        assert not cut.daylit
        assert np.array_equal(cut.dem, dem)

    def test_no_crest_is_a_no_op(self):
        dem = _bowl_with_bank()
        cut = spillway_notch(dem, _crest_cells(), None, (0, 1), cell_size=1.0)
        assert np.array_equal(cut.dem, dem)
        assert cut.sill_elev is None

    def test_a_direction_of_nowhere_is_a_no_op(self):
        dem = _bowl_with_bank()
        cut = spillway_notch(dem, _crest_cells(), 8.0, (0, 0), cell_size=1.0)
        assert np.array_equal(cut.dem, dem)

    def test_an_empty_crest_bar_is_a_no_op(self):
        dem = _bowl_with_bank()
        empty = (np.empty(0, dtype=int), np.empty(0, dtype=int))
        cut = spillway_notch(dem, empty, 8.0, (0, 1), cell_size=1.0)
        assert np.array_equal(cut.dem, dem)

    def test_a_hole_in_the_dem_is_not_daylight(self):
        """You cannot discharge into ground the model does not have."""
        dem = _bowl_with_bank()
        dem[_CHANNEL, 11:] = np.nan
        cut = spillway_notch(dem, _crest_cells(), 8.0, (0, 1),
                             max_reach_m=5.0, cell_size=1.0)
        assert not cut.daylit


class TestDaylightReach:
    def test_the_channel_includes_the_crest_run_even_when_refused(self):
        dem = _channel_dem()[:, ::-1].copy()                 # rises eastward
        rows, cols, _reach, daylit, _pool, _control = daylight_reach(
            dem, _crest_cells(col=4), 3.0, (0, 1), 5.0, cell_size=1.0)
        assert not daylit
        assert rows.size >= 3, "a refused notch still knows where it was"

    def test_the_control_is_the_highest_bar_minimum_along_the_march(self):
        """A channel is only as low as its highest cross-section."""
        dem = _bowl_with_bank()
        _r, _c, _reach, _d, _p, control = daylight_reach(
            dem, _crest_cells(), 8.0, (0, 1), 20.0, cell_size=1.0)
        assert control == pytest.approx(10.0)   # the bank, which is the control


# ---------------------------------------------------------------------------
# DEMBurner — where the crest lies, and the post-pass
# ---------------------------------------------------------------------------

def _channel_on_disk(tmp_path, name="channel.tif"):
    """The bare channel, on disk, north-up at 1 m — the dam makes its own pond here.

    Deliberately not the bowl: a hollow that was already there nets out against the
    baseline ponding and every storage measurement comes back zero, which would make
    the notch look like it changed nothing.

    Column *c* is at x = c + 0.5; row *r* is at y = 21 − r − 0.5.
    """
    return _write_dem(str(tmp_path / name), _channel_dem())


def _sill_along(col=10, r0=8, r1=13):
    """A crest bar WKT running north-south along column *col* — along a bank that
    runs the same way, which is what a crest does."""
    x = col + 0.5
    return f"LINESTRING ({x} {21 - r1 - 0.5}, {x} {21 - r0 - 0.5})"


class TestBurnerCutsTheNotch:
    def test_a_dam_gets_its_notch_cut_through_the_wall(self, tmp_path):
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")

        plain = burner.burn_earthworks([dam])
        notched = burner.burn_earthworks([dam], sills={dam.id: _sill_along()})

        assert burner.burned_notches, "no notch was recorded"
        mask = burner.burned_notches[dam.id]
        assert float(np.nanmax(notched[mask])) <= 9.0 + 1e-6
        assert float(np.nanmax(plain[mask])) > 9.0, "the fixture proves nothing"
        assert burner.burned_sills[dam.id] == pytest.approx(9.0)

    def test_the_notch_survives_a_later_feature_that_fills_over_it(self, tmp_path):
        """Why it is a post-pass and not part of ``_burn_dam``.

        Fills are ``np.maximum`` and a berm is additive, so a notch cut inside a
        ``_burn_*`` is plugged by whatever is burned after it. The post-pass is the only
        place where *the notch is the last thing to touch these cells* is true by
        construction.
        """
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")
        # A berm drawn straight over the dam line, raising the same cells.
        berm = _mock_ew("berm", make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)]),
                        depth=3.0, width=3.0, name="Berm B")

        out = burner.burn_earthworks([dam, berm], sills={dam.id: _sill_along()})
        mask = burner.burned_notches[dam.id]
        assert float(np.nanmax(out[mask])) <= 9.0 + 1e-6, (
            "the berm plugged the notch — the cut is not running last")

    def test_an_inflow_spillway_is_never_cut(self, tmp_path):
        """An inlet is a protected entry, not a weir. Notching one opens the bank at
        the point water arrives and drains the pond through its own inlet."""
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = None
        dam.inflow_spillway = Spillway(crest_elevation=9.0, width_m=1.0,
                                       point_wkt="POINT (10.5 10.5)")

        burner.burn_earthworks([dam], sills={dam.id: _sill_along()})
        assert not burner.burned_notches

    def test_a_feature_with_no_sill_resolved_is_not_cut(self, tmp_path):
        """The controller could not snap the recorded point to the feature, so it is
        absent from the dict — and the burner invents no location for it."""
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")
        burner.burn_earthworks([dam], sills={})
        assert not burner.burned_notches

    def test_a_refused_notch_warns_with_the_reach_it_tried(self, tmp_path):
        """A swale cut into a flat plateau: the ground round it is above the sill in
        every direction, so the notch would discharge into rising ground and nothing is
        cut. Sited on a saddle it would work; sited here it cannot, and the difference
        has to be said rather than left as a spillway that quietly does nothing."""
        burner = DEMBurner(_write_dem(str(tmp_path / "flat.tif"),
                                      np.full((21, 21), 10.0)))
        # One cell wide, so the first step out of the trench is already on the plateau
        # — the refusal under test is the ground *around* the feature, not its own bed.
        line = make_mock_line_geom([(5.5, 10.5), (15.5, 10.5)])
        swale = _mock_ew("swale", line, depth=2.0, width=1.0, name="Swale A")
        swale.spillway = swale.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")

        burner.burn_earthworks([swale],
                               sills={swale.id: "LINESTRING (9.5 10.5, 11.5 10.5)"})
        assert not burner.burned_notches
        assert any("does not daylight" in w for w in burner.warnings), burner.warnings

    def test_the_burned_width_is_whole_cells(self, tmp_path):
        """A 1.4 m weir on a 1 m grid is cut 2 m wide, centred where it was sited."""
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.4, point_wkt="POINT (10.5 10.5)")
        burner.burn_earthworks([dam], sills={dam.id: _sill_along(r0=10, r1=11)})
        mask = burner.burned_notches[dam.id]
        rows = np.unique(np.nonzero(mask)[0])
        assert len(rows) == 2, f"a 1.4 m sill claimed {len(rows)} rows, not 2"

    def test_burned_notches_survive_the_isolated_per_feature_burns(self, tmp_path):
        """Those run *after* the site burn, once per feature, and they reset the
        burner's records. Without the snapshot the dict ends up holding whichever
        feature was measured last — and it is what the overtopping check subtracts
        its barrier crest against."""
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")
        other = _mock_ew("basin",
                         make_mock_line_geom([(2.5, 2.5), (4.5, 2.5)]).__class__ and
                         make_mock_line_geom([(2.5, 2.5), (4.5, 2.5)]),
                         name="Other")

        sills = {dam.id: _sill_along()}
        burner.burn_earthworks([dam, other], sills=sills)
        before = dict(burner.burned_notches)
        burner.feature_storage(other)
        assert set(burner.burned_notches) == set(before)
        assert np.array_equal(burner.burned_notches[dam.id], before[dam.id])


class TestKeyedDamSeesTheNotch:
    """``_keyed_dam_dem`` bypasses ``burn_earthworks`` entirely, and a keyed dam is the
    case the spillway spec is written about. Without an explicit path the whole change
    is invisible on it."""

    def _keyed_dam(self, tmp_path):
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A",
                       key_into_banks=True)
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")
        return burner, dam

    def test_the_keyed_surface_is_notched_when_sills_are_passed(self, tmp_path):
        burner, dam = self._keyed_dam(tmp_path)
        plain = burner._keyed_dam_dem(dam)
        notched = burner._keyed_dam_dem(dam, sills={dam.id: _sill_along()})
        assert (notched <= plain + 1e-9).all()
        lowered = notched < plain - 1e-9
        assert lowered.any(), "the keyed path cut nothing"
        assert float(np.nanmin(plain[lowered])) > 9.0, (
            "the fixture proves nothing — that ground was already under the sill")
        assert float(np.nanmax(notched[lowered])) <= 9.0 + 1e-6

    def test_a_keyed_dams_stage_storage_sees_the_notch(self, tmp_path):
        burner, dam = self._keyed_dam(tmp_path)
        full = burner.dam_stage_storage(dam, key_into_banks=True)
        cut = burner.dam_stage_storage(dam, key_into_banks=True,
                                       sills={dam.id: _sill_along()})
        assert cut < full, f"notched {cut} is not less than unnotched {full}"

    def test_the_keyed_path_leaves_the_site_burns_records_alone(self, tmp_path):
        """It is an idealisation of a wall the user drew shorter, so its masks must not
        replace the ones the map and the verification are measured off."""
        burner, dam = self._keyed_dam(tmp_path)
        burner.burn_earthworks([dam], sills={dam.id: _sill_along()})
        site = dict(burner.burned_notches)
        burner._keyed_dam_dem(dam, sills={dam.id: _sill_along()})
        assert set(burner.burned_notches) == set(site)


# ---------------------------------------------------------------------------
# StageStorage.level_at — the inverse the event readout asks for
# ---------------------------------------------------------------------------

class TestStageStorageLevelAt:
    def _curve(self):
        return StageStorage(levels_m=np.array([10.0, 11.0, 12.0]),
                            volumes_m3=np.array([0.0, 100.0, 300.0]),
                            area_m2=200.0)

    def test_round_trips_against_volume_at(self):
        curve = self._curve()
        for z in (10.0, 10.5, 11.0, 11.7, 12.0):
            assert curve.level_at(curve.volume_at(z)) == pytest.approx(z)

    def test_below_the_bed_sits_on_the_bed(self):
        assert self._curve().level_at(0.0) == pytest.approx(10.0)
        assert self._curve().level_at(-5.0) == pytest.approx(10.0)

    def test_above_the_spill_level_extrapolates_on_the_top_area(self):
        """A feature the balance credits with more than its pond holds is overflowing,
        and a level above the rim is the honest way to print that."""
        assert self._curve().level_at(500.0) == pytest.approx(13.0)

    def test_none_in_none_out(self):
        assert self._curve().level_at(None) is None


# ---------------------------------------------------------------------------
# The guards, at the burner's level — each refusal says which one it was
# ---------------------------------------------------------------------------

class TestBurnerRefusals:
    def test_a_crest_at_the_burned_floor_is_refused_and_named(self, tmp_path):
        burner = DEMBurner(_write_dem(str(tmp_path / "flat.tif"),
                                      np.full((21, 21), 10.0)))
        line = make_mock_line_geom([(5.5, 10.5), (15.5, 10.5)])
        swale = _mock_ew("swale", line, depth=2.0, width=1.0, name="Swale A")
        swale.spillway = swale.outflow_spillway = Spillway(
            crest_elevation=8.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")

        burner.burn_earthworks([swale],
                               sills={swale.id: "LINESTRING (9.5 10.5, 11.5 10.5)"})
        assert not burner.burned_notches
        assert any("at or below the floor" in w for w in burner.warnings), \
            burner.warnings

    def test_a_notch_back_into_its_own_pond_is_refused_and_named(self, tmp_path):
        """The keyed-berm geometry, reduced to its essentials.

        The bank wraps round the end of the feature, so the ground *downhill* of the
        sill is a pocket of the same pond rather than open hillside. The march finds
        ground below the crest, the plain daylight test is satisfied, and the notch
        would still change nothing: the water leaves where it was already leaving.
        """
        dem = np.full((21, 21), 20.0)
        dem[_CHANNEL, 0:10] = 12.0       # high ground west of the bank
        dem[_CHANNEL, 10] = 10.0         # the bank the sill sits on
        dem[_CHANNEL, 11:15] = 4.0       # the pocket, enclosed by the 20 m ground east
        burner = DEMBurner(_write_dem(str(tmp_path / "wrap.tif"), dem))
        line = make_mock_line_geom([(12.5, 8.5), (12.5, 12.5)])
        ew = _mock_ew("basin", line, depth=1.0, width=2.0, name="Pocket A")
        ew.spillway = ew.outflow_spillway = Spillway(
            crest_elevation=8.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")
        burner.burned_masks[ew.id] = np.zeros(dem.shape, dtype=bool)
        burner.burned_masks[ew.id][_CHANNEL, 11:15] = True

        out = burner._cut_spillway(burner.original.copy(), ew, _sill_along())
        assert np.array_equal(out, burner.original)
        assert not burner.burned_notches
        assert any("own pond" in w for w in burner.warnings), burner.warnings

    def test_a_sill_that_is_not_a_line_is_ignored(self, tmp_path):
        """Whatever the controller handed over, the burner does not guess from it."""
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")

        for junk in ("not wkt at all", "POINT (10.5 10.5)"):
            burner.burn_earthworks([dam], sills={dam.id: junk})
            assert not burner.burned_notches, junk

    def test_a_zero_length_sill_has_no_direction_and_is_ignored(self, tmp_path):
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")

        burner.burn_earthworks(
            [dam], sills={dam.id: "LINESTRING (10.5 10.5, 10.5 10.5)"})
        assert not burner.burned_notches

    def test_with_no_recorded_footprint_the_crest_run_stands_in_for_one(self, tmp_path):
        """``_keyed_dam_dem`` supplies its own mask; nothing else does. A caller that
        supplies none still gets a cut rather than a crash — the bar is a footprint of
        last resort, and it is the one place the notch is certainly standing on."""
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=9.0, width_m=1.0, point_wkt="POINT (10.5 10.5)")

        dem = burner.original.copy()
        burner.burned_masks = {}
        out = burner._cut_spillway(dem, dam, _sill_along())
        assert burner.burned_notches, "no notch was cut without a recorded footprint"
        assert float(np.nanmax(out[burner.burned_notches[dam.id]])) <= 9.0 + 1e-6


# ---------------------------------------------------------------------------
# The refusals and degenerate inputs, one case each
#
# Every branch below returns the DEM untouched, and that is the point: a notch that
# quietly does nothing looks exactly like a working spillway in every figure the
# design tier prints, so each way of doing nothing has to be reachable, reached, and
# (where the user could act on it) named in `warnings`.
# ---------------------------------------------------------------------------

class TestNotchRefusals:
    def _dam(self, tmp_path, crest=9.0, width=1.0):
        burner = DEMBurner(_channel_on_disk(tmp_path))
        line = make_mock_line_geom([(10.5, 4.5), (10.5, 16.5)])
        dam = _mock_ew("dam", line, crest_elevation=11.0, width=2.0, name="Dam A")
        dam.spillway = dam.outflow_spillway = Spillway(
            crest_elevation=crest, width_m=width, point_wkt="POINT (10.5 10.5)")
        return burner, dam

    def test_unparseable_sill_geometry_is_a_no_op(self, tmp_path):
        burner, dam = self._dam(tmp_path)
        burner.burn_earthworks([dam], sills={dam.id: "not a linestring"})
        assert not burner.burned_notches
        assert not burner.warnings, "a malformed WKT is a bug, not a design fault"

    def test_a_one_point_sill_is_a_no_op(self, tmp_path):
        burner, dam = self._dam(tmp_path)
        burner.burn_earthworks([dam], sills={dam.id: "LINESTRING (10.5 10.5)"})
        assert not burner.burned_notches

    def test_a_sill_off_the_grid_claims_no_cells(self, tmp_path):
        burner, dam = self._dam(tmp_path)
        burner.burn_earthworks(
            [dam], sills={dam.id: "LINESTRING (500 500, 500 510)"})
        assert not burner.burned_notches

    def test_a_crest_under_the_burned_floor_warns(self, tmp_path):
        burner, dam = self._dam(tmp_path, crest=1.0)
        burner.burn_earthworks([dam], sills={dam.id: _sill_along()})
        assert not burner.burned_notches
        assert any("at or below the floor" in w for w in burner.warnings), \
            burner.warnings


class TestNotchHelpers:
    def _burner(self, tmp_path):
        return DEMBurner(_channel_on_disk(tmp_path))

    def test_a_zero_length_bar_collapses_to_its_own_centre(self, tmp_path):
        bar = self._burner(tmp_path)._scaled_bar([(5.0, 5.0), (5.0, 5.0)], 2.0)
        assert bar.length == pytest.approx(0.0)

    def test_no_width_keeps_the_bar_as_drawn(self, tmp_path):
        bar = self._burner(tmp_path)._scaled_bar([(5.0, 5.0), (5.0, 9.0)], 0.0)
        assert bar.length == pytest.approx(4.0)

    def test_a_sub_cell_bar_still_claims_a_cell(self, tmp_path):
        """``all_touched`` plus the ``line_cells`` fallback, which is ``_rasterize``'s
        own *did we lose the feature?* question asked of the sill."""
        from shapely.geometry import LineString as _LS

        burner = self._burner(tmp_path)
        rows, cols = burner._bar_cells(_LS([(10.5, 10.5), (10.5, 10.5)]))
        assert rows.size >= 1

    def test_a_bar_with_no_direction_has_no_downhill_side(self, tmp_path):
        from shapely.geometry import LineString as _LS

        burner = self._burner(tmp_path)
        assert burner._downhill_step(_LS([(10.5, 10.5), (10.5, 10.5)])) is None

    def test_a_bar_that_is_not_a_line_has_no_downhill_side(self, tmp_path):
        burner = self._burner(tmp_path)
        assert burner._downhill_step(object()) is None

    def test_ground_we_cannot_see_on_either_side_has_no_downhill_side(self, tmp_path):
        """``_ground_mean`` answers ``inf`` for a side that is all hole, and a side we
        cannot see the ground of is never the lower one — so neither is."""
        from shapely.geometry import LineString as _LS

        dem = np.full((21, 21), np.nan)
        burner = DEMBurner(_write_dem(str(tmp_path / "hole.tif"), dem))
        assert burner._downhill_step(_LS([(10.5, 8.5), (10.5, 12.5)])) is None

    def test_no_footprint_has_no_floor(self, tmp_path):
        burner = self._burner(tmp_path)
        assert burner._burned_floor(burner.original, None) is None
        assert burner._burned_floor(
            burner.original, np.zeros(burner.shape, dtype=bool)) is None

    def test_the_floor_of_a_barrier_is_the_ground_under_it_not_the_wall(self, tmp_path):
        """A cut has its bed in the burned array; a barrier's mask is the drawn line,
        which the burn *raises*. Reading only the burned array would return the top of
        the wall and refuse every spillway on every dam."""
        burner = self._burner(tmp_path)
        foot = np.zeros(burner.shape, dtype=bool)
        foot[10, 10] = True
        raised = burner.original.copy()
        raised[10, 10] = 99.0
        assert burner._burned_floor(raised, foot) == pytest.approx(
            float(burner.original[10, 10]))
