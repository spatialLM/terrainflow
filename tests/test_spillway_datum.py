"""test_spillway_datum.py — Stage A of the spillway-notch work: the datum.

Every case here pins one claim the datum change is making, and each of them was false
before it: the crest is bound three ways rather than two, the ceiling is the level water
is *held* to rather than the lowest bare ground, the lip is taken under the sill rather
than at whichever end of a falling swale is lowest, a stored drop is re-derived rather
than trusted, and a pond can be asked what it holds at any level rather than only at the
top.

These are deliberately separate from ``test_spillway.py``, which is the weir-sizing
suite. Sizing asks how wide; this asks how deep, and the two have different failure
modes.
"""

import numpy as np
import pytest

from terrainflow_assessment.modules.earthwork_design import (
    CONTAINMENT_BERM,
    CONTAINMENT_LIP,
    CONTAINMENT_MEASURED,
    CONTAINMENT_WALL,
    Spillway,
    bind_crest,
    build_stage_storage,
    rebase_spillway,
    spillway_datum,
    spillway_notes,
    spillway_validity,
)
from terrainflow_assessment.modules.footprint import pour_point, pour_point_near

# ---------------------------------------------------------------------------
# A1 — the three-way binding
# ---------------------------------------------------------------------------

class TestTheCrestIsBoundThreeWays:
    """Crest, drop below containment and height above floor are one value, three views."""

    RIM, INVERT = 100.0, 97.0

    def test_a_crest_derives_both_partners(self):
        crest, drop, height = bind_crest(
            self.RIM, crest=99.0, invert_elevation=self.INVERT)
        assert (crest, drop, height) == pytest.approx((99.0, 1.0, 2.0))

    def test_a_height_derives_the_crest_and_the_drop(self):
        crest, drop, height = bind_crest(
            self.RIM, height=2.0, invert_elevation=self.INVERT)
        assert (crest, drop, height) == pytest.approx((99.0, 1.0, 2.0))

    def test_a_drop_derives_the_crest_and_the_height(self):
        crest, drop, height = bind_crest(
            self.RIM, drop=1.0, invert_elevation=self.INVERT)
        assert (crest, drop, height) == pytest.approx((99.0, 1.0, 2.0))

    @pytest.mark.parametrize("height", [0.0, 0.05, 0.37, 1.5, 2.99])
    def test_a_round_trip_through_the_height_control_cannot_drift(self, height):
        """The whole point of one binding function rather than three assignments."""
        crest, drop, back = bind_crest(
            self.RIM, height=height, invert_elevation=self.INVERT)
        again, drop2, height2 = bind_crest(
            self.RIM, crest=crest, invert_elevation=self.INVERT)
        third, drop3, height3 = bind_crest(
            self.RIM, drop=drop2, invert_elevation=self.INVERT)
        assert back == pytest.approx(height)
        assert (again, drop2, height2) == pytest.approx((crest, drop, height))
        assert (third, drop3, height3) == pytest.approx((crest, drop, height))

    def test_the_clamp_happens_before_the_partners_are_derived(self):
        """A clamped crest that left a stale height is exactly the drift being avoided."""
        band = spillway_datum(self.RIM, self.INVERT, head_m=0.30, min_freeboard_m=0.30)
        crest, drop, height = bind_crest(
            self.RIM, height=2.9, band=band, invert_elevation=self.INVERT)
        assert crest == pytest.approx(band[1])            # clamped to rim − 0.60
        assert drop == pytest.approx(self.RIM - crest)    # and both partners follow
        assert height == pytest.approx(crest - self.INVERT)
        assert height < 2.9

    def test_a_missing_floor_yields_no_height_rather_than_a_guess(self):
        """A dam has no cut floor, and inventing one would report the wall height."""
        crest, drop, height = bind_crest(self.RIM, crest=99.0)
        assert (crest, drop) == pytest.approx((99.0, 1.0))
        assert height is None

    def test_a_missing_containment_yields_no_drop(self):
        crest, drop, height = bind_crest(None, crest=99.0, invert_elevation=self.INVERT)
        assert (crest, height) == pytest.approx((99.0, 2.0))
        assert drop is None

    def test_the_absolute_crest_wins_over_both_relatives(self):
        crest, _drop, _h = bind_crest(
            self.RIM, crest=99.0, drop=5.0, height=0.1, invert_elevation=self.INVERT)
        assert crest == pytest.approx(99.0)

    def test_a_height_alone_with_no_floor_resolves_nothing(self):
        """Better to hand the inputs back than to anchor a crest to a datum that is absent."""
        assert bind_crest(100.0, height=2.0) == (None, None, 2.0)


class TestTheHeightSurvivesSerialisation:

    def test_the_field_round_trips(self):
        sp = Spillway(crest_elevation=99.0, drop_below_rim_m=1.0,
                      height_above_floor_m=2.0)
        back = Spillway.from_dict(sp.to_dict())
        assert back.height_above_floor_m == pytest.approx(2.0)

    def test_a_document_written_before_the_field_existed_still_opens(self):
        old = {"crest_elevation": 99.0, "drop_below_rim_m": 1.0, "head_m": 0.30,
               "width_m": 2.0, "width_auto": True, "point_wkt": None, "auto": True}
        back = Spillway.from_dict(old)
        assert back.crest_elevation == pytest.approx(99.0)
        assert back.height_above_floor_m is None


# ---------------------------------------------------------------------------
# A2 — containment is the ceiling; the lip clearance is a note
# ---------------------------------------------------------------------------

class TestContainmentIsTheCeilingNotTheLip:
    """A bermed swale holds water above the ring minimum. That is the design, not a fault."""

    LIP, BERM, INVERT = 68.88, 69.60, 68.00

    def test_the_band_opens_up_to_the_berm_crest(self):
        _lo, against_lip = spillway_datum(self.LIP, self.INVERT,
                                          head_m=0.20, min_freeboard_m=0.15)
        _lo, against_berm = spillway_datum(self.BERM, self.INVERT,
                                           head_m=0.20, min_freeboard_m=0.15)
        assert against_berm - against_lip == pytest.approx(self.BERM - self.LIP)

    def test_a_crest_above_the_lip_is_no_longer_a_problem(self):
        """It was, and any problem fails the row — so every bermed swale read broken."""
        crest = 69.15                       # above the lip, below the berm crest
        assert spillway_validity(crest, self.BERM, invert_elevation=self.INVERT,
                                 head_m=0.20, min_freeboard_m=0.15) == []
        against_lip = spillway_validity(crest, self.LIP, invert_elevation=self.INVERT,
                                        head_m=0.20, min_freeboard_m=0.15)
        assert any("above the lowest containing ground" in p for p in against_lip)

    def test_but_it_is_still_said_out_loud_as_a_note(self):
        notes = spillway_notes(69.30, lip_elevation=self.LIP,
                               containment_elevation=self.BERM,
                               containment_source=CONTAINMENT_BERM,
                               berm_crest_elevation=self.BERM)
        assert len(notes) == 1
        assert "0.42 m above natural ground" in notes[0]
        assert "69.60" in notes[0]

    def test_a_measured_level_is_named_as_measured(self):
        notes = spillway_notes(69.30, lip_elevation=self.LIP,
                               containment_elevation=69.55,
                               containment_source=CONTAINMENT_MEASURED)
        assert "measured this pond holding to 69.55" in notes[0]

    def test_with_nothing_measured_the_note_says_so_rather_than_reassuring(self):
        notes = spillway_notes(69.30, lip_elevation=self.LIP,
                               containment_elevation=self.LIP,
                               containment_source=CONTAINMENT_LIP)
        assert "Nothing measured is holding it there yet" in notes[0]

    def test_a_dam_names_the_wall(self):
        notes = spillway_notes(56.12, lip_elevation=54.00,
                               containment_elevation=56.42,
                               containment_source=CONTAINMENT_WALL)
        assert "The wall is the containment here" in notes[0]

    def test_a_crest_below_the_lip_says_nothing(self):
        """The ordinary case. A note that fires on every feature is not a note."""
        assert spillway_notes(68.50, lip_elevation=self.LIP,
                              containment_elevation=self.LIP,
                              containment_source=CONTAINMENT_LIP) == []

    def test_no_crest_says_nothing(self):
        assert spillway_notes(None, lip_elevation=self.LIP) == []


# ---------------------------------------------------------------------------
# A3 — the lip is taken locally, once a sill is sited
# ---------------------------------------------------------------------------

class TestAFallingSwaleTakesItsLipLocally:
    """A contour swale's global ring minimum is at one of its ends, not under the sill."""

    @staticmethod
    def _falling_swale():
        """A 20x20 tile falling 1 m per column, with a swale run across it."""
        dem = np.tile(np.arange(20, dtype="float64"), (20, 1)) + 100.0
        mask = np.zeros((20, 20), dtype=bool)
        mask[9:11, 4:16] = True
        return dem, mask

    def test_the_global_lip_is_at_the_low_end(self):
        dem, mask = self._falling_swale()
        level, _cell = pour_point(dem, mask)
        assert level == pytest.approx(103.0)          # the ring cell at column 3

    def test_a_sill_at_the_high_end_reads_the_ground_it_is_actually_in(self):
        dem, mask = self._falling_swale()
        level, _cell = pour_point_near(dem, mask, (10, 15), 2)
        assert level == pytest.approx(113.0)
        # Ten metres of fall between the two answers, on a swale twelve cells long.
        # A crest referenced to the global figure would be cut ten metres too deep.
        assert level - pour_point(dem, mask)[0] == pytest.approx(10.0)

    def test_a_window_that_catches_no_rim_falls_back_to_the_global_answer(self):
        """A datum from further away beats no datum at all."""
        dem, mask = self._falling_swale()
        level, _cell = pour_point_near(dem, mask, (0, 0), 1)
        assert level == pytest.approx(pour_point(dem, mask)[0])

    def test_a_centre_off_the_grid_falls_back_rather_than_raising(self):
        dem, mask = self._falling_swale()
        assert pour_point_near(dem, mask, (500, 500), 2)[0] == pytest.approx(
            pour_point(dem, mask)[0])

    def test_no_centre_at_all_is_the_global_measurement(self):
        dem, mask = self._falling_swale()
        assert pour_point_near(dem, mask, None, 3) == pour_point(dem, mask)

    def test_an_empty_mask_is_unmeasurable_either_way(self):
        dem, _mask = self._falling_swale()
        assert pour_point_near(dem, np.zeros((20, 20), bool), (5, 5), 2) == (None, None)


# ---------------------------------------------------------------------------
# A6 — the stage–storage curve
# ---------------------------------------------------------------------------

class TestTheStageStorageCurve:
    """What the pond holds at any level, from arrays the flood already produced."""

    @staticmethod
    def _wedge(cells=100, cell_area=1.0):
        """A pond 1.00 m deep at one end and 0.01 m at the other, standing at 10.0 m."""
        depths = np.linspace(0.01, 1.00, cells)
        return build_stage_storage(depths, 10.0, cell_area), depths, cell_area

    def test_it_integrates_to_the_volume_the_flood_reported(self):
        curve, depths, cell_area = self._wedge()
        assert curve.volume_at(10.0) == pytest.approx(
            float(depths.sum()) * cell_area, rel=1e-9)

    def test_it_answers_a_level_below_the_top_with_what_that_level_holds(self):
        curve, depths, cell_area = self._wedge()
        for z in (9.2, 9.5, 9.75, 9.99):
            expected = float(np.clip(z - (10.0 - depths), 0.0, None).sum()) * cell_area
            assert curve.volume_at(z) == pytest.approx(expected, rel=1e-3)

    def test_the_bed_holds_nothing(self):
        curve, depths, _a = self._wedge()
        assert curve.volume_at(10.0 - depths.max()) == pytest.approx(0.0)
        assert curve.volume_at(0.0) == pytest.approx(0.0)

    def test_it_is_sampled_rather_than_one_point_per_cell(self):
        """Tens of megabytes of curve retained per design is the thing being avoided."""
        curve, _d, _a = self._wedge(cells=50_000)
        assert len(curve.levels_m) <= 130
        assert curve.volume_at(10.0) == pytest.approx(
            float(np.linspace(0.01, 1.00, 50_000).sum()), rel=1e-6)

    def test_a_small_pond_keeps_every_cell(self):
        """Twelve bed elevations, plus the spill level pinned exactly on the end."""
        curve, _d, _a = self._wedge(cells=12)
        assert len(curve.levels_m) == 13
        assert curve.levels_m[-1] == pytest.approx(10.0)

    def test_a_non_square_cell_scales_the_whole_curve(self):
        square, depths, _a = self._wedge(cell_area=1.0)
        wide, _d, _a2 = self._wedge(cell_area=2.5)
        assert wide.volume_at(9.6) == pytest.approx(2.5 * square.volume_at(9.6))

    def test_above_the_spill_level_it_extrapolates_on_the_wetted_area(self):
        """Not a prediction — the pond is leaving by then — but a bounded answer."""
        curve, depths, cell_area = self._wedge()
        over = curve.volume_at(10.5) - curve.volume_at(10.0)
        assert over == pytest.approx(0.5 * len(depths) * cell_area)

    def test_an_empty_pond_has_no_curve(self):
        assert build_stage_storage(np.array([]), 10.0, 1.0) is None
        assert build_stage_storage(np.array([0.0, 0.0]), 10.0, 1.0) is None
        assert build_stage_storage(np.array([0.5]), None, 1.0) is None

    def test_a_flat_bottomed_pond_is_linear_in_level(self):
        curve = build_stage_storage(np.full(20, 0.80), 10.0, 1.0)
        assert curve.volume_at(10.0) == pytest.approx(16.0)
        assert curve.volume_at(9.6) == pytest.approx(8.0)
        assert curve.volume_at(9.2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# A8 — the migration
# ---------------------------------------------------------------------------

class TestASavedDesignIsRebasedNotMoved:
    """``crest_elevation`` is the value the design is made of. It must not move."""

    def test_the_crest_is_untouched_and_the_drop_is_recomputed(self):
        # Saved against the old datum: a bare ring minimum of 68.88.
        sp = Spillway(crest_elevation=69.30, drop_below_rim_m=-0.42)
        rebase_spillway(sp, 69.60, 68.00)          # the berm crest, and the floor
        assert sp.crest_elevation == pytest.approx(69.30)
        assert sp.drop_below_rim_m == pytest.approx(0.30)
        assert sp.height_above_floor_m == pytest.approx(1.30)

    def test_the_height_is_filled_in_where_it_never_existed(self):
        sp = Spillway.from_dict({"crest_elevation": 99.0, "drop_below_rim_m": 1.0})
        assert sp.height_above_floor_m is None
        rebase_spillway(sp, 100.0, 97.0)
        assert sp.height_above_floor_m == pytest.approx(2.0)

    def test_a_dam_gets_no_height_rather_than_its_wall_height(self):
        sp = Spillway(crest_elevation=56.12)
        rebase_spillway(sp, 56.42, None)
        assert sp.crest_elevation == pytest.approx(56.12)
        assert sp.drop_below_rim_m == pytest.approx(0.30)
        assert sp.height_above_floor_m is None

    def test_a_spillway_with_no_crest_is_left_entirely_alone(self):
        """Then the stored drop is the only thing the user ever chose."""
        sp = Spillway(crest_elevation=None, drop_below_rim_m=0.45)
        rebase_spillway(sp, 100.0, 97.0)
        assert sp.crest_elevation is None
        assert sp.drop_below_rim_m == pytest.approx(0.45)

    def test_no_spillway_is_not_an_error(self):
        assert rebase_spillway(None, 100.0, 97.0) is None

    def test_rebasing_twice_changes_nothing_the_second_time(self):
        sp = Spillway(crest_elevation=69.30, drop_below_rim_m=-0.42)
        rebase_spillway(sp, 69.60, 68.00)
        first = (sp.crest_elevation, sp.drop_below_rim_m, sp.height_above_floor_m)
        rebase_spillway(sp, 69.60, 68.00)
        assert (sp.crest_elevation, sp.drop_below_rim_m,
                sp.height_above_floor_m) == pytest.approx(first)
