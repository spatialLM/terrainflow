"""Tests for modules/impoundment_sites.

The headline test is the rotation one. The proxy this replaces measured valley width by
scanning along the raster row, so an east–west valley had its *length* measured as its
width and scored accordingly. The same valley turned ninety degrees must rank the same.
"""

import math

import numpy as np
import pytest

from terrainflow_assessment.modules.impoundment_sites import (
    embankment_volume,
    flow_bearing,
    impounded_volume,
    rank_impoundment_sites,
    transect_cells,
    wall_run,
)

CELL = 1.0
N = 81
MID = N // 2


def _rc_to_xy(row, col):
    """Cell → map, north-up, origin at the top-left corner."""
    return (col + 0.5) * CELL, (N - row - 0.5) * CELL


def _v_valley(axis="ns", floor_grade=0.10, side_grade=0.30):
    """A straight V-valley with a gently falling floor.

    ``axis="ns"`` runs the valley north–south (down increasing row); ``"ew"`` is the
    same landform rotated a quarter turn, running west–east.

    The floor grade is deliberately steep. A pool behind a 2 m wall reaches
    ``height / floor_grade`` metres upstream, so on a gentle floor it would run off this
    81-cell fixture entirely and be refused as unenclosed — correctly, but for a reason
    that has nothing to do with what these tests are about.
    """
    r, c = np.mgrid[0:N, 0:N].astype("float64")
    if axis == "ns":
        along, across = r, np.abs(c - MID)
    else:
        along, across = c, np.abs(r - MID)
    return 100.0 - floor_grade * along + side_grade * across


def _acc_for(axis="ns"):
    """Accumulation that grows down the valley floor, zero on the sides."""
    r, c = np.mgrid[0:N, 0:N].astype("float64")
    if axis == "ns":
        on_floor = np.abs(c - MID) < 1.5
        along = r
    else:
        on_floor = np.abs(r - MID) < 1.5
        along = c
    return np.where(on_floor, along * 10.0, 0.0)


class TestFlowBearing:
    def test_points_downslope_on_a_plane(self):
        from terrainflow_assessment.modules.dem_loader import horn_gradient

        r, _c = np.mgrid[0:21, 0:21]
        dem = 100.0 - r.astype("float64")        # falls south
        dz_dx, dz_dy, _ = horn_gradient(dem, 1.0, 1.0)
        ex, ny = flow_bearing(dz_dx, dz_dy, 10, 10)
        assert ex == pytest.approx(0.0, abs=1e-9)
        assert ny == pytest.approx(-1.0, abs=1e-9)   # southward

    def test_flat_ground_has_no_direction(self):
        from terrainflow_assessment.modules.dem_loader import horn_gradient

        dem = np.full((9, 9), 12.0)
        dz_dx, dz_dy, _ = horn_gradient(dem, 1.0, 1.0)
        assert flow_bearing(dz_dx, dz_dy, 4, 4) is None


class TestTransect:
    def test_runs_square_across_the_flow(self):
        """A wall crosses the valley; a line along it would dam nothing."""
        cells, _step = transect_cells((N, N), MID, MID, (0.0, -1.0), CELL, CELL, 10.0)
        rows = {r for r, _c in cells}
        assert rows == {MID}, "flow running south should give an east–west crest line"

    def test_rotating_the_flow_rotates_the_crest(self):
        cells, _step = transect_cells((N, N), MID, MID, (1.0, 0.0), CELL, CELL, 10.0)
        cols = {c for _r, c in cells}
        assert cols == {MID}, "flow running east should give a north–south crest line"

    def test_it_passes_through_the_candidate(self):
        cells, _step = transect_cells((N, N), 30, 40, (0.0, -1.0), CELL, CELL, 8.0)
        assert (30, 40) in cells


class TestWallRun:
    def test_stops_at_the_abutments(self):
        """The wall ends where the ground already reaches the crest."""
        dem = _v_valley("ns")
        cells, _step = transect_cells((N, N), MID, MID, (0.0, -1.0), CELL, CELL, 40.0)
        here = cells.index((MID, MID))
        run = wall_run(dem, cells, dem[MID, MID] + 2.0, here)
        # Side grade 0.30 m/m, so a 2 m wall reaches ~6.7 m either side: ~13-15 cells.
        assert 11 <= len(run) <= 19
        assert all(h > 0 for _r, _c, h in run)

    def test_a_candidate_above_the_crest_has_no_wall(self):
        dem = _v_valley("ns")
        cells, _step = transect_cells((N, N), MID, MID, (0.0, -1.0), CELL, CELL, 40.0)
        here = cells.index((MID, MID))
        assert wall_run(dem, cells, dem[MID, MID] - 1.0, here) == []


class TestEmbankmentVolume:
    def test_matches_the_trapezoid_by_hand(self):
        """h·(W + z·h) per station, times the station spacing."""
        run = [(0, 0, 2.0)]
        got = embankment_volume(run, step_m=1.0, crest_width_m=3.0, batter=2.0)
        assert got == pytest.approx(2.0 * (3.0 + 2.0 * 2.0))

    def test_is_not_a_rectangle(self):
        """A rectangular wall under-states a battered one badly, and by more as it grows."""
        run = [(0, 0, 4.0)]
        battered = embankment_volume(run, 1.0, crest_width_m=3.0, batter=2.0)
        rectangular = 4.0 * 3.0
        assert battered > 2.0 * rectangular

    def test_zero_height_stations_contribute_nothing(self):
        assert embankment_volume([(0, 0, 0.0)], 1.0) == 0.0


class TestImpoundedVolume:
    def test_a_pool_that_reaches_the_window_edge_is_refused(self):
        """Not enclosed means the water goes round, which is not an impoundment."""
        dem = _v_valley("ns")
        acc = _acc_for("ns")
        cells, _step = transect_cells((N, N), MID, MID, (0.0, -1.0), CELL, CELL, 40.0)
        run = wall_run(dem, cells, dem[MID, MID] + 2.0, cells.index((MID, MID)))
        _v, _a, reason = impounded_volume(
            dem, acc, MID, MID, dem[MID, MID] + 2.0, run, CELL * CELL,
            window_cells=3)          # deliberately far too small
        assert reason and "not enclosed" in reason

    def test_an_enclosed_pool_reports_a_volume(self):
        dem = _v_valley("ns")
        acc = _acc_for("ns")
        crest = dem[MID, MID] + 2.0
        cells, _step = transect_cells((N, N), MID, MID, (0.0, -1.0), CELL, CELL, 40.0)
        run = wall_run(dem, cells, crest, cells.index((MID, MID)))
        volume, area, reason = impounded_volume(
            dem, acc, MID, MID, crest, run, CELL * CELL, window_cells=30)
        assert reason is None
        assert volume > 0 and area > 0


class TestRanking:
    @staticmethod
    def _rank(axis):
        dem = _v_valley(axis)
        acc = _acc_for(axis)
        return rank_impoundment_sites(
            dem, acc, [(MID, MID)], CELL, CELL, _rc_to_xy,
            crest_heights_m=(1.0, 2.0, 3.0), runoff_mm=25.0)

    def test_the_same_valley_rotated_ranks_the_same(self):
        """The defect this module exists for.

        The proxy measured width along the raster row, so an east–west valley read its
        own length as its width. Storage per cubic metre of fill is a property of the
        landform, not of the survey's orientation.
        """
        ns = self._rank("ns")[0]
        ew = self._rank("ew")[0]

        assert ns["notes"] is None and ew["notes"] is None
        assert ns["storage_ratio"] == pytest.approx(ew["storage_ratio"], rel=0.15)
        assert ns["wall_length_m"] == pytest.approx(ew["wall_length_m"], rel=0.15)
        assert ns["storage_m3"] == pytest.approx(ew["storage_m3"], rel=0.15)

    def test_the_index_is_storage_over_fill(self):
        site = self._rank("ns")[0]
        assert site["storage_ratio"] == pytest.approx(
            site["storage_m3"] / site["fill_m3"])

    def test_the_winning_wall_height_is_reported(self):
        """A ratio with no wall attached cannot be acted on."""
        site = self._rank("ns")[0]
        assert site["wall_height_m"] in (1.0, 2.0, 3.0)

    def test_a_narrow_neck_beats_a_wide_saddle(self):
        """The whole point of the ratio: same water, less wall."""
        r, c = np.mgrid[0:N, 0:N].astype("float64")
        across = np.abs(c - MID)
        # One valley, two cross-sections: steep sides above row 42 (a narrow neck, so a
        # short wall), gentle sides below it (a wide saddle needing a long one). Same
        # floor, so the same water arrives at both.
        side_grade = np.where(r < 42, 0.60, 0.12)
        dem = 100.0 - 0.10 * r + side_grade * across
        acc = np.where(across < 1.5, r * 10.0, 0.0)

        sites = rank_impoundment_sites(
            dem, acc, [(30, MID), (55, MID)], CELL, CELL, _rc_to_xy,
            crest_heights_m=(2.0,), runoff_mm=25.0)
        usable = [s for s in sites if s["notes"] is None]
        assert usable, "neither candidate produced a wall"
        assert usable[0]["row"] == 30, (
            "the narrow neck should outrank the wide saddle on storage per m³ of fill")

    def test_flat_ground_is_refused_with_a_reason_not_dropped(self):
        """A site the user can see was considered beats a silently shorter list."""
        dem = np.full((N, N), 50.0)
        acc = np.zeros((N, N))
        sites = rank_impoundment_sites(dem, acc, [(MID, MID)], CELL, CELL, _rc_to_xy)
        assert len(sites) == 1
        assert sites[0]["storage_ratio"] == 0.0
        assert sites[0]["notes"]

    def test_catchment_yield_is_a_column_not_a_term_in_the_score(self):
        with_storm = self._rank("ns")[0]
        assert with_storm["event_yield_m3"] > 0
        assert with_storm["fills_in_events"] is not None
        # The ratio itself must not move with the storm.
        dem, acc = _v_valley("ns"), _acc_for("ns")
        dry = rank_impoundment_sites(dem, acc, [(MID, MID)], CELL, CELL, _rc_to_xy,
                                     crest_heights_m=(1.0, 2.0, 3.0))[0]
        assert dry["storage_ratio"] == pytest.approx(with_storm["storage_ratio"])

    def test_results_are_ranked_and_numbered(self):
        dem, acc = _v_valley("ns"), _acc_for("ns")
        sites = rank_impoundment_sites(
            dem, acc, [(30, MID), (45, MID), (60, MID)], CELL, CELL, _rc_to_xy,
            crest_heights_m=(2.0,))
        assert [s["rank"] for s in sites] == [1, 2, 3]
        ratios = [s["storage_ratio"] for s in sites]
        assert ratios == sorted(ratios, reverse=True)

    def test_progress_is_reported(self):
        seen = []
        dem, acc = _v_valley("ns"), _acc_for("ns")
        rank_impoundment_sites(dem, acc, [(MID, MID)], CELL, CELL, _rc_to_xy,
                               crest_heights_m=(2.0,),
                               progress=lambda p, m: seen.append(p))
        assert seen and all(0 <= p <= 100 for p in seen)


class TestPrismaticVolumeFinallyHasACaller:
    def test_embankment_volume_goes_through_the_primitive(self):
        """`prismatic_volume` shipped built, tested and unused until this module."""
        from terrainflow_assessment.core.sizing.primitives import (
            prismatic_volume,
            trapezoid_section,
        )

        h, w, z, step = 2.5, 3.0, 2.0, 1.0
        section = trapezoid_section(w + 2 * z * h, w, h)
        expected = prismatic_volume(section.area, step).volume
        assert embankment_volume([(0, 0, h)], step, w, z) == pytest.approx(expected)
        assert not math.isnan(expected)
