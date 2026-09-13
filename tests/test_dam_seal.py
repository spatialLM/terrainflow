"""A thin dam wall on a diagonal must still hold water.

``_burn_dam`` raises a band of the drawn thickness on cell centres, then seals the band's
centreline with ``line_cells`` — a Bresenham path, 8-connected. A wall about a cell thick
or less claims few centres of its own, so on a diagonal alignment the seal is all there
is, and a Bresenham staircase joins its cells only at their corners. Depression filling
walks corners, so the pond left through the wall.

Measured on the 1 m Quail Island fixture and a 2 m resample (2026-09-13): a keyed 2 m wall
on the 2 m grid at 45 degrees reported 241 m³ where the sealed wall holds 783 m³; at 1 m
thickness on the 1 m grid, 0.4 m³ against 23.7. From about 1.42 cells (the diagonal of a
cell) upward the band is solid at every angle and the seal changes nothing.

The synthetic ground here is a 40 m square, one metre a cell, with a V-shaped valley
along the diagonal ``row = col`` falling to the south-east. The dam is drawn across it
along ``row + col = 40``, the perpendicular diagonal, so its seal is a pure staircase.
"""

import math
import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from terrainflow_assessment.modules.burn_strategy import (
    WALL_SEAL_CELLS,
    thin_wall_warning,
)
from terrainflow_assessment.modules.earthwork_design import DEMBurner, Earthwork
from tests.conftest import make_mock_line_geom

SIZE = 40
CREST = 99.0          # a metre above the valley floor where the dam crosses it


def _diagonal_valley(tmp_path):
    data = np.fromfunction(
        lambda r, c: 100.0 - 0.05 * (r + c) + 0.4 * np.abs(r - c) / math.sqrt(2),
        (SIZE, SIZE))
    path = str(tmp_path / "valley.tif")
    with rasterio.open(
        path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1, dtype="float32",
        crs="EPSG:32632", nodata=-9999.0,
        transform=from_bounds(0, 0, SIZE, SIZE, SIZE, SIZE),
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _dam(thickness, keyed=False, kind="dam"):
    # (x 10.5, y 9.5) is row 30 col 10; (x 30.5, y 29.5) is row 10 col 30.
    ew = Earthwork(kind, make_mock_line_geom([(10.5, 9.5), (30.5, 29.5)]), "Dam 1")
    ew.crest_elevation = CREST
    ew.top_width_m = thickness
    ew.key_into_banks = keyed
    return ew


class TestTheWallHolds:
    @pytest.mark.parametrize("thickness", [0.3, 0.5, 1.0])
    def test_a_thin_wall_across_a_diagonal_valley_holds_water(self, tmp_path, thickness):
        dem = _diagonal_valley(tmp_path)
        thick = DEMBurner(dem).dam_stage_storage(_dam(3.0))
        thin = DEMBurner(dem).dam_stage_storage(_dam(thickness))
        assert thick > 10.0, f"the reference 3 m wall holds only {thick:.1f} m³"
        # Not equal: the thin wall stands a cell further upstream and takes a little
        # pond with it. But of the same order, not zero.
        assert thin > 0.5 * thick, (
            f"a {thickness} m wall held {thin:.1f} m³ against the 3 m wall's "
            f"{thick:.1f} m³ — the pond is leaving through the seal's corners")

    def test_the_keyed_estimate_holds_through_the_same_seal(self, tmp_path):
        dem = _diagonal_valley(tmp_path)
        thick = DEMBurner(dem).dam_stage_storage(_dam(3.0, keyed=True), key_into_banks=True)
        thin = DEMBurner(dem).dam_stage_storage(_dam(0.5, keyed=True), key_into_banks=True)
        assert thin > 0.5 * thick, (thin, thick)

    def test_the_seal_joins_every_wall_cell_by_an_edge(self, tmp_path):
        from scipy.ndimage import label

        b = DEMBurner(_diagonal_valley(tmp_path))
        out = b.burn_earthworks([_dam(0.5)])
        wall = np.isclose(out, CREST) & (out > b.original + 1e-6)
        four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        _labels, pieces = label(wall, structure=four)
        assert pieces == 1, f"the wall is {pieces} pieces joined only at corners"

    @pytest.mark.parametrize("kind", ["detainment_bund", "wascob"])
    def test_every_crest_type_shares_the_sealed_wall(self, tmp_path, kind):
        dem = _diagonal_valley(tmp_path)
        assert DEMBurner(dem).dam_stage_storage(_dam(0.5, kind=kind)) > 10.0


class TestThinWallWarning:
    def test_the_threshold_is_the_diagonal_of_a_cell(self):
        assert WALL_SEAL_CELLS == pytest.approx(math.sqrt(2))

    def test_a_wall_under_the_diagonal_is_told_what_the_model_does(self):
        w = thin_wall_warning("Dam 1", 1.0, 1.0)
        assert w is not None
        assert "1.00 m" in w and "1.41 m" in w
        assert "sealed" in w
        # And not the old wording, which blamed routing and never mentioned storage.
        assert "routing effect" not in w

    def test_at_or_over_the_diagonal_there_is_nothing_to_say(self):
        assert thin_wall_warning("Dam 1", 1.42, 1.0) is None
        assert thin_wall_warning("Dam 1", 3.0, 2.0) is None

    def test_unknown_figures_say_nothing(self):
        assert thin_wall_warning("Dam 1", None, 1.0) is None
        assert thin_wall_warning("Dam 1", 1.0, 0.0) is None

    def test_the_burn_raises_it_and_not_the_generic_sub_cell_one(self, tmp_path):
        b = DEMBurner(_diagonal_valley(tmp_path))
        b.burn_earthworks([_dam(0.5)])
        assert any("sealed" in w for w in b.warnings), b.warnings
        assert not any("routing effect" in w for w in b.warnings), b.warnings

    def test_a_thick_wall_burns_without_it(self, tmp_path):
        b = DEMBurner(_diagonal_valley(tmp_path))
        b.burn_earthworks([_dam(3.0)])
        assert not any("sealed" in w for w in b.warnings), b.warnings


class TestTheKeyedExtensionHolds:
    """`_key_dam_ends` walks outward from each end of the wall in half-cell steps and
    raises what it crosses. Its comment said half a cell kept that path 4-connected. On
    a bearing off the axes a single step can cross a row and a column boundary together,
    and the pond leaves through the corner — the same leak as the seal, in the part of
    the wall that exists only because the banks were too low.

    Measured on the fixture (2026-09-13), a 20 m, 3 m-thick wall keyed in across the
    valley at the keyline keypoint: 0.0-0.7 m³ held at 16-60 degrees where a corner-joined
    extension holds 185-2,010 m³, and a wall dilated a cell all round (which cannot leak
    at a corner) agrees with the corner-joined one to 10-20 %. Synthetic ground is too
    small to show it — its ponds reach the grid edge first — so this runs on the tile.
    """

    FIXTURE = str(pathlib.Path(__file__).parent / "fixtures" / "quail_island_catchment.tif")
    X0, Y0 = 1574561.0, 5169232.0

    def _wall(self, angle_deg, crest):
        a = math.radians(angle_deg)
        dx, dy = math.cos(a) * 10.0, math.sin(a) * 10.0
        cx, cy = self.X0 + 152.8, self.Y0 + 183.8
        ew = Earthwork("dam", make_mock_line_geom([(cx - dx, cy - dy), (cx + dx, cy + dy)]),
                       "Dam 1")
        ew.crest_elevation = crest
        ew.top_width_m = 3.0
        ew.key_into_banks = True
        return ew

    @pytest.mark.parametrize("angle", [16.6, 30.0, 45.0])
    def test_a_wall_keyed_in_on_a_diagonal_bearing_holds_water(self, angle):
        b = DEMBurner(self.FIXTURE)
        held = b.dam_stage_storage(self._wall(angle, 64.5), key_into_banks=True)
        assert any("keyed" in w for w in b.warnings), "the banks did not need keying in"
        assert held > 300.0, (
            f"a wall keyed in at {angle} degrees holds {held:.1f} m³ — the pond is leaving "
            f"through a corner of the keyed extension")

    def test_the_extension_joins_its_cells_by_an_edge(self):
        from scipy.ndimage import label

        b = DEMBurner(self.FIXTURE)
        ew = self._wall(30.0, 64.5)
        line = b._to_shapely(ew.geometry)
        dem = b.original.copy()
        b._key_dam_ends(dem, line, ew.crest_elevation)
        ext = dem > b.original + 1e-9
        four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        eight = np.ones((3, 3), dtype=int)
        assert label(ext, structure=four)[1] == label(ext, structure=eight)[1], (
            "the keyed extension has cells joined only at a corner")
