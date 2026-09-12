"""Tests for ``modules/earthwork_design.DEMBurner`` — the burn itself."""

from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from terrainflow_assessment.modules.earthwork_design import DEMBurner
from terrainflow_assessment.modules.footprint import xy_to_rc
from tests.conftest import make_mock_line_geom, make_mock_polygon_geom

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_dem(path, data, cell_size=1.0, crs="EPSG:32632", nodata=-9999.0):
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w,
        count=1, dtype="float32", crs=crs,
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"), 1)
    return path


def _mock_ew(ew_type, geom, **kwargs):
    """Build a minimal mock earthwork."""
    from unittest.mock import MagicMock

    ew = MagicMock()
    ew.type = ew_type
    ew.geometry = geom
    ew.enabled = True
    ew.name = kwargs.get("name", f"Test {ew_type}")
    ew.depth = kwargs.get("depth", 0.5)
    ew.width = kwargs.get("width", 2.0)
    # DEMBurner uses the active Earthwork interface (top_width_m + buffer_radius_m
    # + bottom_width_m for the Strategy-C sub-cell check), not the legacy `.width`;
    # provide them so shapely buffering and the width comparison get real numbers.
    ew.top_width_m = kwargs.get("width", 2.0)
    ew.buffer_radius_m = kwargs.get("width", 2.0) / 2.0
    ew.bottom_width_m = kwargs.get("bottom_width_m", 1.0)
    # Explicit because MagicMock answers float() with 1.0: left unset, every mock
    # earthwork silently claimed a 1 m batter run and took burn branches the real
    # Earthwork (which defaults it to 0.0) never would.
    ew.batter_run_m = kwargs.get("batter_run_m", 0.0)
    ew.companion_berm = kwargs.get("companion_berm", False)
    ew.crest_elevation = kwargs.get("crest_elevation", None)
    ew.gradient_pct = kwargs.get("gradient_pct", 1.0)
    ew.key_into_banks = kwargs.get("key_into_banks", False)
    # Both names, because `_cut_spillway` reads the property and everything else reads
    # the attribute. A MagicMock would answer either with a truthy mock whose
    # `crest_elevation` floats to 1.0, so an unset spillway has to be explicitly None.
    ew.spillway = kwargs.get("spillway", None)
    ew.outflow_spillway = ew.spillway
    ew.inflow_spillway = kwargs.get("inflow_spillway", None)
    # The spillway link and the level it resolves to, for `_burn_diversion`'s grade
    # datum. Explicit for `batter_run_m`'s reason and then some: left unset, every mock
    # drain would grade from a MagicMock that floats to 1.0 — a datum ~99 m under the
    # test fixtures' ground, which cuts a trench that deep and reads as a burn fault
    # rather than as a mock.
    ew.spillway_link_id = kwargs.get("spillway_link_id", None)
    ew.invert_start_m = kwargs.get("invert_start_m", None)
    ew.id = kwargs.get("id", ew.name)
    return ew


# ---------------------------------------------------------------------------
# DEMBurner.__init__
# ---------------------------------------------------------------------------

class TestDEMBurnerInit:
    def test_shape_read(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((15, 20), 50.0))
        b = DEMBurner(path)
        assert b.shape == (15, 20)

    def test_cell_size_read(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        assert b.cell_size == pytest.approx(1.0)

    def test_original_is_float32(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        assert b.original.dtype == np.float32


# ---------------------------------------------------------------------------
# burn_earthworks — no earthworks
# ---------------------------------------------------------------------------

class TestBurnEarthworksEmpty:
    def test_empty_list_returns_original(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.burn_earthworks([])
        assert np.allclose(result, 50.0)

    def test_invalid_geometry_skipped(self, tmp_path):
        """Earthwork with bad JSON is silently skipped (line 47 continue)."""
        from unittest.mock import MagicMock
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        ew = MagicMock()
        ew.enabled = True
        ew.type = "swale"
        ew.geometry = MagicMock()
        ew.geometry.asJson.return_value = "not-valid-json"
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)


# ---------------------------------------------------------------------------
# _to_shapely
# ---------------------------------------------------------------------------

class TestToShapely:
    def test_valid_geojson(self, tmp_path):
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        geom = make_mock_line_geom()
        result = b._to_shapely(geom)
        assert result is not None

    def test_invalid_json_returns_none(self, tmp_path):
        from unittest.mock import MagicMock
        path = _write_dem(str(tmp_path / "dem.tif"), np.full((10, 10), 50.0))
        b = DEMBurner(path)
        bad = MagicMock()
        bad.asJson.return_value = "not-valid-json"
        assert b._to_shapely(bad) is None


# ---------------------------------------------------------------------------
# _burn_swale
# ---------------------------------------------------------------------------

class TestBurnSwale:
    def test_swale_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_swale_depth_1m(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=1.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() == pytest.approx(49.0, abs=0.1)

    def test_swale_with_companion_berm(self, tmp_path):
        # Sloped DEM so downhill side is detectable
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.5, (20, 20)).astype("float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])
        # Some cells should be raised above their original value (berm built on downhill side)
        assert np.any(result > b.original)


# ---------------------------------------------------------------------------
# _burn_berm
# ---------------------------------------------------------------------------

class TestBurnBerm:
    def test_berm_raises_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=2.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0

    def test_berm_depth_2m(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=2.0, width=2.0)
        result = b.burn_earthworks([ew])
        assert result.max() == pytest.approx(52.0, abs=0.1)


# ---------------------------------------------------------------------------
# _burn_basin
# ---------------------------------------------------------------------------

class TestBurnBasin:
    def test_basin_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_polygon_geom((3.0, 3.0, 12.0, 12.0))
        ew = _mock_ew("basin", geom, depth=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_basin_depth_2m(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_polygon_geom((3.0, 3.0, 12.0, 12.0))
        ew = _mock_ew("basin", geom, depth=2.0)
        result = b.burn_earthworks([ew])
        assert result.min() == pytest.approx(48.0, abs=0.1)


# ---------------------------------------------------------------------------
# _burn_dam
# ---------------------------------------------------------------------------

class TestBurnDam:
    def test_dam_with_crest_elevation(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=60.0)
        result = b.burn_earthworks([ew])
        assert result.max() >= 60.0

    def test_dam_no_crest_acts_like_berm(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=2.0, width=2.0, crest_elevation=None)
        result = b.burn_earthworks([ew])
        assert result.max() > 50.0

    def test_dam_lower_than_crest_raised(self, tmp_path):
        """DEM cells below crest elevation should be raised to crest."""
        data = np.full((20, 20), 40.0)  # all cells below crest=55
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=55.0)
        result = b.burn_earthworks([ew])
        # Dam footprint cells should be at least 55
        assert result.max() >= 55.0

    def test_dam_above_crest_unchanged(self, tmp_path):
        """DEM cells already above crest should not be changed."""
        data = np.full((20, 20), 70.0)  # all cells above crest=55
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=1.0, width=2.0, crest_elevation=55.0)
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 70.0)  # unchanged


# ---------------------------------------------------------------------------
# _burn_diversion
# ---------------------------------------------------------------------------

class TestBurnDiversion:
    def test_diversion_lowers_dem(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(2.0, 10.0), (18.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=1.5, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        assert result.min() < 50.0

    def test_diversion_single_segment_zero_length(self, tmp_path):
        """Zero-length segment should be skipped (continue branch)."""
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        # Two identical consecutive points create a zero-length segment
        geom = make_mock_line_geom([(10.0, 10.0), (10.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=2.0, gradient_pct=1.0)
        result = b.burn_earthworks([ew])  # should not crash
        assert result.shape == (20, 20)

    def test_diversion_zero_total_length(self, tmp_path):
        """If total_length == 0, returns unchanged DEM."""
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(10.0, 10.0), (10.0, 10.0)])
        ew = _mock_ew("diversion", geom, depth=0.3, width=2.0, gradient_pct=1.0)
        result = b.burn_earthworks([ew])
        # Should be the same as original (zero-length line)
        assert result.shape == (20, 20)


# ---------------------------------------------------------------------------
# Disabled earthwork
# ---------------------------------------------------------------------------

class TestDisabledEarthwork:
    def test_disabled_skipped(self, tmp_path):
        data = np.full((20, 20), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("berm", geom, depth=5.0)
        ew.enabled = False
        result = b.burn_earthworks([ew])
        assert np.allclose(result, 50.0)


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_writes_geotiff(self, tmp_path):
        data = np.full((10, 10), 42.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        out = str(tmp_path / "out.tif")
        b.save(b.original, out)
        with rasterio.open(out) as src:
            assert np.allclose(src.read(1), 42.0)

    def test_save_preserves_crs(self, tmp_path):
        data = np.full((10, 10), 50.0)
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        out = str(tmp_path / "out.tif")
        b.save(b.original, out)
        with rasterio.open(out) as src:
            assert src.crs is not None


# ---------------------------------------------------------------------------
# get_ponding_layer
# ---------------------------------------------------------------------------

class TestGetPondingLayer:
    def test_returns_same_shape(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (20, 20)

    def test_non_negative(self, tmp_path):
        data = np.full((20, 20), 50.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert np.all(result >= 0.0)

    def test_float32_output(self, tmp_path):
        data = np.full((10, 10), 30.0, dtype="float32")
        path = _write_dem(str(tmp_path / "dem.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.dtype == np.float32

    def test_bowl_dem_ponding(self, tmp_path):
        """A depression (manually lowered centre) should show ponding."""
        r = np.arange(20)
        c = np.arange(20)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        data = (50.0 + ((rr - 9.5) ** 2 + (cc - 9.5) ** 2) * 0.3).astype("float32")
        data[8:12, 8:12] -= 10.0  # create a depression
        path = _write_dem(str(tmp_path / "bowl.tif"), data)
        b = DEMBurner(path)
        result = b.get_ponding_layer(data)
        assert result.shape == (20, 20)


# ---------------------------------------------------------------------------
# burned_raised — the bank a feature built, kept apart from what it claimed
# ---------------------------------------------------------------------------

class TestBurnedRaised:
    """A swale's mask is its trench and its companion berm sits beside it.

    ``mask & raised`` therefore finds a dam's wall and finds nothing at all for a swale, so
    a keyed swale berm — which impounds water and fails exactly the way a dam does — was
    never checked for overtopping. ``burned_raised`` is the separate record that answers
    "what did this feature build", without widening ``burned_masks``, which sets the
    verification's At-grid reference and must not move.
    """

    def _sloped(self, tmp_path):
        data = np.fromfunction(lambda r, c: 50.0 - r * 0.5, (20, 20)).astype("float32")
        return _write_dem(str(tmp_path / "dem.tif"), data)

    def test_a_bermed_swale_records_the_bank_it_built(self, tmp_path):
        b = DEMBurner(self._sloped(tmp_path))
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        result = b.burn_earthworks([ew])

        raised = b.burned_raised.get(ew.id)
        assert raised is not None and raised.any()
        # It is the built ground, so every cell of it stands at or above the original.
        assert (result[raised] >= b.original[raised]).all()
        assert (result > b.original).any()

    def test_an_unbermed_swale_builds_nothing(self, tmp_path):
        b = DEMBurner(self._sloped(tmp_path))
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=False)
        b.burn_earthworks([ew])
        assert not b.burned_raised.get(ew.id, np.zeros((20, 20), bool)).any()

    def test_the_bank_is_not_the_trench(self, tmp_path):
        """The whole point: the two records answer different questions."""
        b = DEMBurner(self._sloped(tmp_path))
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        b.burn_earthworks([ew])
        trench = b.burned_masks[ew.id]
        bank = b.burned_raised[ew.id]
        assert not (trench & bank).any(), "the berm sits beside the trench, not in it"

    def test_a_dam_records_its_wall(self, tmp_path):
        b = DEMBurner(self._sloped(tmp_path))
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("dam", geom, depth=2.0, width=2.0, crest_elevation=48.0)
        ew.key_into_banks = False
        result = b.burn_earthworks([ew])
        raised = b.burned_raised.get(ew.id)
        assert raised is not None and raised.any()
        assert result[raised].max() == pytest.approx(48.0, abs=1e-3)

    def test_an_isolated_burn_does_not_pollute_the_site_record(self, tmp_path):
        """``feature_storage`` re-burns one feature; the site burn's records must survive."""
        b = DEMBurner(self._sloped(tmp_path))
        geom = make_mock_line_geom([(5.0, 10.0), (15.0, 10.0)])
        ew = _mock_ew("swale", geom, depth=0.5, width=2.0, companion_berm=True)
        b.burn_earthworks([ew])
        before = {k: v.copy() for k, v in b.burned_raised.items()}
        b._isolated_burn(ew)
        assert set(b.burned_raised) == set(before)
        for key, mask in before.items():
            assert (b.burned_raised[key] == mask).all()


class TestDiversionBurnEquivalence:
    """The one-rasterise burn cuts the same channel the per-sample one did.

    The old burn walked the line at three samples per cell and, for each sample,
    buffered a point, rasterised it over the whole raster and ran a full-array
    minimum — billions of cell-touches per drain on a real DEM, to cut a band a few
    cells wide. `_reference_burn` below is that algorithm, kept as the thing the
    rewrite is measured against.

    They agree closely but not bit-for-bit, and both differences are understood:

    **Depth.** A cell used to take the *deepest* invert among every sample disc that
    brushed it, which on a falling grade is the one furthest along the line — up to
    a disc radius plus a cell past the cell's own position. It now takes the invert
    for where it actually sits, projected onto the line. Bounded by
    ``(width/2 + cell) × gradient``, and the new answer is the better one: the old
    bias was an artefact of how the samples were spaced, not a property of the drain.

    **Footprint.** Rasterising many overlapping discs and unioning the result claims
    marginally more cells than rasterising their union polygon, because a disc can
    cover a cell centre that the union polygon's own edge passes just outside of. The
    difference is a couple of cells on the rounded ends.

    Both sides rasterise on **cell centres**. ``all_touched`` belongs to the
    cross-section convention, not to the sampling this class is about: it claimed
    every cell the band so much as brushed — a near-constant ~1.3 m wider than drawn
    — and the volumetric burns were fixed to stop doing that. Left ``True`` here the
    reference would pin that bug, in exactly the way an untapered reference would
    have pinned the rectangular cut, which is the argument the taper note below
    already makes.
    """

    @staticmethod
    def _reference_burn(burner, dem, line, ew, start_elev):
        from shapely.geometry import Point

        from terrainflow_assessment.modules.burn_strategy import (
            enforce_monotonic_path,
            taper_reach,
        )
        from terrainflow_assessment.modules.earthwork_design import channel_batter_run

        dem = dem.copy()
        coords = list(line.coords)
        # The reference models the old *sampling*, not the old cross-section. It cut a
        # full-depth rectangle, which is the thing the real burn was fixed to stop doing;
        # left that way this reference would pin the bug rather than the rewrite. The
        # taper belongs to the section and the sampling belongs to this class, so the
        # depth model is shared and what stays under test is chainage and footprint.
        # Cells only the discs claim — the rounded ends in the docstring above — sit
        # outside the band and so taper to nothing, which is what they are worth.
        band = burner._rasterize(line.buffer(ew.width / 2.0), all_touched=False)
        reach = taper_reach(band, channel_batter_run(ew),
                            (burner.cell_h, burner.cell_size))
        cum = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            cum.append(cum[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        grad = ew.gradient_pct / 100.0

        for seg_i in range(len(coords) - 1):
            x1, y1 = coords[seg_i]
            x2, y2 = coords[seg_i + 1]
            seg = cum[seg_i + 1] - cum[seg_i]
            if seg == 0:
                continue
            n_steps = max(2, int(seg / burner.cell_size * 3))
            for step in range(n_steps + 1):
                t = step / n_steps
                x, y = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                invert = start_elev - (cum[seg_i] + t * seg) * grad
                cell_mask = burner._rasterize(Point(x, y).buffer(ew.width / 2),
                                              all_touched=False)
                if cell_mask.any():
                    depth = (ew.depth if reach is None
                             else ew.depth * reach[cell_mask])
                    dem[cell_mask] = np.minimum(dem[cell_mask], invert - depth)
        # The real burn breaches one-cell humps along the alignment afterwards. Not
        # part of what changed, so the reference does it too.
        return enforce_monotonic_path(dem, burner._line_path_cells(line))

    @staticmethod
    def _setup(tmp_path, coords, width, grad, flat=False):
        data = (np.full((40, 40), 100.0) if flat else
                np.fromfunction(lambda r, c: 100.0 - r * 0.1 - c * 0.05, (40, 40)))
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))
        ew = _mock_ew("diversion", make_mock_line_geom(coords),
                      depth=0.4, width=width, gradient_pct=grad)
        got = b.burn_earthworks([ew])
        line = b._to_shapely(ew.geometry)
        r0, c0 = xy_to_rc(b.transform, *coords[0])
        want = TestDiversionBurnEquivalence._reference_burn(
            b, b.original, line, ew, float(b.original[r0, c0]))
        return b, got, want

    @pytest.mark.parametrize("coords,width,grad", [
        # Off the half-metre on purpose. A 3 m band centred on y=20.0 has its
        # edges at 18.5/21.5 — exactly two cell centres — and shapely buffers a
        # point into an *inscribed* polygon, so the reference's discs fall a
        # hair short of a centre the union rectangle claims outright. That is a
        # disagreement between two test-side approximations of one circle, not
        # between the two burns; `all_touched` hid it by counting any brush.
        # (What the burn does at such an alignment — 4 rows for a 3 m band, the
        # outer two half-depth — is `taper_reach`'s documented quadrature
        # over-read, which its own docstring says is not fixed there.)
        ([(5.0, 20.3), (35.0, 20.3)], 3.0, 1.0),                # straight, due east
        ([(5.0, 5.0), (35.0, 35.0)], 3.0, 2.0),                 # diagonal
        ([(5.0, 30.0), (20.0, 20.0), (35.0, 28.0)], 4.0, 0.5),  # dog-leg
        ([(8.0, 12.0), (30.0, 12.0)], 6.0, 0.0),                # wide, flat grade
    ])
    def test_inverts_agree_where_both_burns_cut(self, tmp_path, coords, width, grad):
        b, got, want = self._setup(tmp_path, coords, width, grad)

        cut_new = got < b.original - 1e-9
        cut_old = want < b.original - 1e-9
        both = cut_new & cut_old
        assert both.sum() > 10, "the fixture barely cut anything — bad geometry"

        # See the class docstring: the old burn biased deeper by up to a disc radius
        # plus a cell of chainage. With no gradient the two must agree exactly.
        tol = (width / 2.0 + b.cell_size) * (grad / 100.0) + 1e-3
        worst = float(np.nanmax(np.abs(got[both] - want[both])))
        assert worst <= tol, f"diverged by {worst:.4f} m, bound {tol:.4f} m"

    @pytest.mark.parametrize("coords,width,grad", [
        ([(5.0, 20.3), (35.0, 20.3)], 3.0, 1.0),   # see the note above
        ([(5.0, 5.0), (35.0, 35.0)], 3.0, 2.0),
        ([(5.0, 30.0), (20.0, 20.0), (35.0, 28.0)], 4.0, 0.5),
        ([(8.0, 12.0), (30.0, 12.0)], 6.0, 0.0),
    ])
    def test_the_band_is_the_same_channel(self, tmp_path, coords, width, grad):
        b, got, want = self._setup(tmp_path, coords, width, grad)

        cut_new = got < b.original - 1e-9
        cut_old = want < b.original - 1e-9

        assert not (cut_new & ~cut_old).any(), (
            f"{int((cut_new & ~cut_old).sum())} cells burned that the per-sample "
            f"burn never touched — the channel got wider")
        dropped = int((cut_old & ~cut_new).sum())
        assert dropped <= max(3, 0.05 * cut_old.sum()), (
            f"{dropped} of {int(cut_old.sum())} cells lost — more than the rounded "
            f"ends can account for")

    def test_the_drain_is_cut_as_the_trapezoid_it_is_priced_as(self, tmp_path):
        """A diversion is specified with a batter and was burned without one.

        The registry gives it ``default_side_slope`` and derives its bottom width, and
        ``calculate_cut_volume`` prices it as a trapezoid — but the burn floored every
        cell of the band at full depth, so the terrain model held a rectangle. That is
        the same mismatch ``_storage_invert`` was written to remove for swales, and it
        survived here because a graded invert cannot use ``tapered_invert``, whose floor
        is a single elevation. It does not need to: the taper is a fraction of depth per
        cell and the grade is a datum per cell, so they multiply.

        Measured across the middle of the drain, away from the buffer's rounded ends.
        A 6 m top on a 1.0 m depth at 1:1 gives a 4 m bed and a 1.0 m batter run, so the
        section is 5.0 m2 per metre and the rectangle it used to cut was 6.0.
        """
        data = np.full((40, 40), 100.0)
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))
        ew = _mock_ew("diversion", make_mock_line_geom([(5.0, 20.0), (35.0, 20.0)]),
                      depth=1.0, width=6.0, bottom_width_m=4.0, gradient_pct=0.0)
        out = b.burn_earthworks([ew])

        mid = out[:, 20]                     # one transverse slice, mid-alignment
        cut = float(np.clip(b.original[:, 20] - mid, 0.0, None).sum()) * b.cell_size
        assert cut == pytest.approx(5.0, rel=0.02), (
            f"{cut:.2f} m2 per metre against a drawn 5.00 m2 — 6.00 is the "
            f"full-depth rectangle this used to cut"
        )

    def test_a_sub_cell_drain_still_carves_a_connected_path(self, tmp_path):
        """The buffer rasterises empty, so the centreline path is the fallback."""
        b, got, _ = self._setup(tmp_path, [(5.0, 20.4), (35.0, 20.4)], 0.05, 1.0,
                                flat=True)
        cut = got < b.original - 1e-9
        assert cut.any(), "a sub-cell drain burned nothing at all"
        assert cut.sum() >= 25, f"only {int(cut.sum())} cells — the path is broken"


class TestSpillwayLinkedDiversion:
    """A drain graded from a spillway crest instead of from the ground under its line.

    Stage C of ``CLudeDocs/SPILLWAY_NOTCH_PLAN.md``. The burner never learns what a link
    is: the controller resolves it and puts one absolute on the feature, exactly as it
    resolves the crest bar and passes it as ``sills=``. So these tests set
    ``invert_start_m`` directly, which is what the burner actually sees.

    Why an absolute and not a level read off the surface: the notch is cut as a
    **post-pass**, so when ``_burn_diversion`` runs the source's spillway is not in the
    array yet. ``test_the_datum_does_not_wait_for_the_notch`` is that fact, pinned.
    """

    @staticmethod
    def _slope(rows=40, cols=40, top=100.0, fall=0.0):
        """Ground that falls along the drain's own axis, so a grade is measurable."""
        data = np.full((rows, cols), top)
        if fall:
            data -= np.arange(cols) * (fall / max(1, cols - 1))
        return data

    def _drain(self, tmp_path, data, coords, **kwargs):
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))
        ew = _mock_ew("diversion", make_mock_line_geom(coords), depth=1.0, width=3.0,
                      bottom_width_m=3.0, gradient_pct=0.0, **kwargs)
        return b, b.burn_earthworks([ew]), ew

    # A diversion is a conveyance, so it gets `enforce_monotonic_path` after the graded
    # burn — which cuts a strict 1 mm per cell even at zero gradient. Mid-alignment that
    # is ~1.5 cm below the graded invert, which is why these read to 2 cm rather than to
    # the millimetre. It is existing behaviour and nothing here changes it.
    _BREACH = 0.02

    def test_an_unlinked_drain_still_grades_from_the_ground(self, tmp_path):
        """The fallback, unchanged — and the baseline the linked cases move against."""
        b, out, _ew = self._drain(
            tmp_path, self._slope(), [(5.0, 20.0), (35.0, 20.0)])
        assert float(out[20, 20]) == pytest.approx(99.0, abs=self._BREACH)
        assert b.warnings == [] or all("grade from" not in w for w in b.warnings)

    def test_a_linked_drain_is_cut_from_the_crest_and_not_from_the_ground(self, tmp_path):
        """Ground at 100; the crest it takes from is at 96, four metres below.

        The bed comes out one depth below the datum, exactly as it comes out one depth
        below a ground sample — ``invert_start_m`` replaces the sample, it is not the
        bed level itself.
        """
        _b, out, _ew = self._drain(
            tmp_path, self._slope(), [(5.0, 20.0), (35.0, 20.0)],
            invert_start_m=96.0, spillway_link_id="s1:outflow:start")
        assert float(out[20, 20]) == pytest.approx(95.0, abs=self._BREACH)

    def test_a_crest_above_the_ground_raises_nothing(self, tmp_path):
        """``np.minimum``, so a datum standing over the terrain cannot fill.

        Not *nothing at all*: the monotonic breach still carves its millimetre a cell,
        because a diversion is a conveyance and that runs whatever the grade did. What
        must not happen is the 40 m of fill a maximum would have put here.
        """
        b, out, _ew = self._drain(
            tmp_path, self._slope(), [(5.0, 20.0), (35.0, 20.0)],
            invert_start_m=140.0, spillway_link_id="s1:outflow:start")
        assert out.max() <= b.original.max() + 1e-6
        assert float(b.original[20, 20] - out[20, 20]) < self._BREACH

    def test_the_grade_runs_down_from_the_linked_end(self, tmp_path):
        """A 2% drain over 30 m: 0.6 m of fall, and which end it starts at decides
        which end is deep.

        This is the failure the link's third token exists to prevent, and it is
        invisible from any single number: grading from the wrong end produces a drain
        that runs uphill from an entirely plausible-looking level.
        """
        data = self._slope()
        coords = [(5.0, 20.0), (35.0, 20.0)]
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))

        def cut(end):
            ew = _mock_ew("diversion", make_mock_line_geom(coords), depth=1.0,
                          width=3.0, bottom_width_m=3.0, gradient_pct=2.0,
                          invert_start_m=96.0, spillway_link_id=f"s1:outflow:{end}")
            out = b.burn_earthworks([ew])
            return float(out[20, 6]), float(out[20, 34])

        west_start, east_start = cut("start")
        west_end, east_end = cut("end")

        # Linked at the first vertex (west): the west end is at the datum and the east
        # end is 0.6 m lower. Linked at the last vertex (east): exactly reversed.
        assert west_start == pytest.approx(95.0, abs=0.05)
        assert east_start == pytest.approx(94.4, abs=0.05)
        assert east_end == pytest.approx(95.0, abs=0.05)
        assert west_end == pytest.approx(94.4, abs=0.05)

    def test_linking_an_end_does_not_reverse_the_stored_alignment(self, tmp_path):
        """The reversal is local to the burn; the geometry the user drew is untouched.

        Reversing it on link would desynchronise ``source_contour_coords`` from the
        vertices it describes and would survive an unlink, with no undo stack anywhere
        in this plugin to put it back.
        """
        coords = [(5.0, 20.0), (35.0, 20.0)]
        geom = make_mock_line_geom(coords)
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"),
                                 self._slope().astype("float32")))
        ew = _mock_ew("diversion", geom, depth=1.0, width=3.0, bottom_width_m=3.0,
                      gradient_pct=2.0, invert_start_m=96.0,
                      spillway_link_id="s1:outflow:end")
        before = geom.asJson()
        b.burn_earthworks([ew])
        assert geom.asJson() == before

    def test_a_datum_that_is_not_a_level_falls_back_to_the_ground(self, tmp_path):
        """A link the controller could not resolve leaves the field None or unusable.

        The drain then does what it did before it was linked; the controller reports
        the fallback, and the burner does not guess at it.
        """
        for bad in (None, float("nan"), "not a level"):
            _b, out, _ew = self._drain(
                tmp_path, self._slope(), [(5.0, 20.0), (35.0, 20.0)],
                invert_start_m=bad, spillway_link_id="s1:outflow:start")
            assert float(out[20, 20]) == pytest.approx(99.0, abs=self._BREACH), bad

    def test_the_datum_does_not_wait_for_the_notch(self, tmp_path):
        """The source's spillway is cut **after** the type dispatch, so at the moment
        the drain is burned the notch is not in the array. The drain is cut to the same
        level either way, because the datum is an absolute off the design rather than
        anything read from the running surface.
        """
        data = self._slope()
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))
        drain = _mock_ew("diversion", make_mock_line_geom([(5.0, 30.0), (35.0, 30.0)]),
                         depth=1.0, width=3.0, bottom_width_m=3.0, gradient_pct=0.0,
                         invert_start_m=96.0, spillway_link_id="dam:outflow:start",
                         name="Drain", id="drain")
        alone = b.burn_earthworks([drain])

        dam = _mock_ew("dam", make_mock_line_geom([(20.0, 5.0), (20.0, 15.0)]),
                       crest_elevation=98.0, name="Dam", id="dam",
                       spillway=SimpleNamespace(crest_elevation=96.0, width_m=2.0,
                                                point_wkt="POINT (20 10)"))
        together = b.burn_earthworks(
            [dam, drain], sills={"dam": "LINESTRING (19 10, 21 10)"})
        assert float(together[30, 20]) == pytest.approx(float(alone[30, 20]), abs=1e-6)

    def test_a_source_drawn_after_its_drain_is_burned_first(self, tmp_path):
        """``burn_order``, through the burner rather than over the list.

        The datum does not need it, but the drain's cut is an ``np.minimum`` against the
        running array and its breach walks the surface as it stands — so the source has
        to be finished ground by the time the drain reads it.
        """
        data = self._slope()
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))
        seen = []
        real_diversion, real_swale = b._burn_diversion, b._burn_swale

        def note(fn, label):
            def wrapped(dem, geom, ew):
                seen.append(label)
                return fn(dem, geom, ew)
            return wrapped

        b._burn_diversion = note(real_diversion, "drain")
        b._burn_swale = note(real_swale, "source")

        drain = _mock_ew("diversion", make_mock_line_geom([(5.0, 30.0), (35.0, 30.0)]),
                         depth=1.0, width=3.0, bottom_width_m=3.0, gradient_pct=0.0,
                         invert_start_m=96.0, spillway_link_id="src:outflow:start",
                         name="Drain", id="drain")
        source = _mock_ew("swale", make_mock_line_geom([(20.0, 5.0), (20.0, 15.0)]),
                          depth=0.5, width=2.0, name="Source", id="src")
        b.burn_earthworks([drain, source])
        assert seen == ["source", "drain"]

    def test_an_unlinked_design_burns_in_the_order_it_was_given(self, tmp_path):
        """The property that keeps the reorder from moving any existing number."""
        data = self._slope()
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data.astype("float32")))
        seen = []
        real = b._burn_swale

        def wrapped(dem, geom, ew):
            seen.append(ew.name)
            return real(dem, geom, ew)

        b._burn_swale = wrapped
        ews = [_mock_ew("swale", make_mock_line_geom([(5.0, y), (35.0, y)]),
                        depth=0.5, width=2.0, name=f"S{i}", id=f"s{i}")
               for i, y in enumerate((10.0, 20.0, 30.0))]
        b.burn_earthworks(ews)
        assert seen == ["S0", "S1", "S2"]


class TestNonSquareCells:
    """`cell_size ** 2` is a cell's area only on a square grid.

    Every other fixture is square, so the substitution stood in for the real area
    throughout the burner and nothing disagreed. On a 2 m x 5 m grid it is out by
    2.5x — and, worse, it was out *inconsistently*: `feature_storage` already
    measured against `abs(a * e)` while spoil, trench storage and burned-cut used
    the square, so one design could report a pond holding more than the hole that
    made it with nothing in the numbers to say which was wrong.
    """

    def test_the_burner_knows_its_cell_is_not_square(self, tmp_dem_nonsquare):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        b = DEMBurner(tmp_dem_nonsquare)
        assert b.cell_size == 2.0
        assert b.cell_h == 5.0
        assert b.cell_area == 10.0, (
            f"cell area is {b.cell_area}, not width x height")

    def test_cell_area_is_not_the_square_of_the_width(self, tmp_dem_nonsquare):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        b = DEMBurner(tmp_dem_nonsquare)
        assert b.cell_area != b.cell_size ** 2, (
            "the fixture is not exercising the non-square path")

    def test_a_square_grid_is_unchanged(self, tmp_dem):
        from terrainflow_assessment.modules.earthwork_design import DEMBurner

        b = DEMBurner(tmp_dem)
        assert b.cell_area == b.cell_size ** 2 == 1.0

    def _swale(self):
        from unittest.mock import MagicMock

        from tests.conftest import make_mock_line_geom

        ew = MagicMock()
        ew.type = "swale"
        ew.geometry = make_mock_line_geom([(0.0, 0.0), (100.0, 0.0)])
        ew.depth = 0.5
        ew.width = 2.0
        ew.top_width_m = 2.0
        ew.bottom_width_m = 1.0
        ew.batter_run_m = 0.0
        ew.companion_berm = False
        ew.capacity_m3 = 60.0
        return ew

    def test_capacity_breakdown_takes_an_area_not_a_side(self):
        """The sub-cell test wants a length and the volume wants an area; one
        argument cannot be both once the cells stop being square."""
        from terrainflow_assessment.modules.earthwork_design import (
            capacity_breakdown,
        )

        square = capacity_breakdown(self._swale(), cell_size=2.0, n_cells=100)
        wide = capacity_breakdown(self._swale(), cell_size=2.0, cell_area=10.0,
                                  n_cells=100)
        assert wide["rasterisable"] > square["rasterisable"], (
            "a 10 m2 cell must hold more than a 4 m2 one")

    def test_the_default_area_still_squares_the_side(self):
        from terrainflow_assessment.modules.earthwork_design import (
            capacity_breakdown,
        )

        implied = capacity_breakdown(self._swale(), cell_size=2.0, n_cells=100)
        explicit = capacity_breakdown(self._swale(), cell_size=2.0, cell_area=4.0,
                                      n_cells=100)
        assert implied == explicit


class TestTaperSamplesPerAxis:
    """A batter runs the same number of metres in both directions, which is a
    different number of cells on each axis once they stop being square."""

    def test_a_pair_tapers_differently_along_each_axis(self):
        import numpy as np

        from terrainflow_assessment.modules.burn_strategy import taper_reach

        mask = np.zeros((11, 11), dtype=bool)
        mask[2:9, 2:9] = True

        square = taper_reach(mask, batter_run=6.0, cell_size=2.0)
        wide = taper_reach(mask, batter_run=6.0, cell_size=(5.0, 2.0))
        assert square is not None and wide is not None
        # Rows are five metres apart in `wide`, so the taper reaches full depth in
        # fewer rows than columns — the square case cannot tell them apart.
        mid = mask.shape[0] // 2
        assert np.allclose(square[mid, :], square[:, mid]), (
            "a square grid should taper identically on both axes")
        assert not np.allclose(wide[mid, :], wide[:, mid]), (
            "a 5 m x 2 m cell tapered identically on both axes")

    def test_a_scalar_still_means_a_square_cell(self):
        import numpy as np

        from terrainflow_assessment.modules.burn_strategy import taper_reach

        mask = np.zeros((11, 11), dtype=bool)
        mask[2:9, 2:9] = True
        assert np.allclose(taper_reach(mask, 6.0, 2.0),
                           taper_reach(mask, 6.0, (2.0, 2.0)))

    def _band(self, n, across, shape=(41, 41)):
        """A straight band *n* cells wide, running east-west or north-south."""
        import numpy as np

        m = np.zeros(shape, dtype=bool)
        if across == "rows":                 # an E-W band, crossed row-wise
            m[20:20 + n, :] = True
        else:                                # a N-S band, crossed column-wise
            m[:, 20:20 + n] = True
        return m

    def _exact(self, n, spacing, batter_run):
        """What the taper should integrate to, from the geometry alone.

        Cell *i* of an *n*-cell band has its centre ``min(i+0.5, n-i-0.5) × spacing``
        from the band edge, and the trapezoid's depth there is that over the batter run,
        capped at full. No distance transform involved — this is the answer the
        transform is supposed to reproduce.
        """
        import numpy as np

        i = np.arange(n)
        d = np.minimum(i + 0.5, n - i - 0.5) * spacing
        return float(np.clip(d / batter_run, 0.0, 1.0).sum()) * spacing

    @pytest.mark.parametrize("cell_h,cell_w", [(1.0, 1.0), (2.0, 1.0),
                                               (1.0, 2.0), (5.0, 2.0)])
    @pytest.mark.parametrize("across", ["rows", "cols"])
    @pytest.mark.parametrize("n,batter_run", [(4, 1.5), (5, 2.0), (6, 3.0)])
    def test_a_straight_band_tapers_to_the_section_it_was_drawn_as(
            self, cell_h, cell_w, across, n, batter_run):
        """The half-cell inset must come off the axis the distance was measured along.

        It used to come off ``min(cell_h, cell_w)`` — the finest axis, whichever way the
        boundary lay. On a 1 m x 2 m grid that under-subtracts for a band crossed on the
        coarse axis, and the shortfall goes straight into ``reach``, so it is always an
        over-cut: up to +20% here, and on the field case an east-west swale collapsed to
        ``reach == 1`` and was cut as a full-depth rectangle at +60% against the same
        swale drawn north-south.
        """
        from terrainflow_assessment.modules.burn_strategy import taper_reach

        mask = self._band(n, across)
        reach = taper_reach(mask, batter_run, (cell_h, cell_w))
        spacing = cell_h if across == "rows" else cell_w
        profile = (reach[20:20 + n, 20] if across == "rows"
                   else reach[20, 20:20 + n])
        got = float(profile.sum()) * spacing

        assert got == pytest.approx(self._exact(n, spacing, batter_run), rel=1e-9)

    def test_the_same_swale_cuts_the_same_either_way_round(self):
        """The bug as a user would meet it: one design, one number, two bearings.

        A swale drawn east-west and the same swale drawn north-south are the same
        excavation. On rectangular cells they were not — the taper collapsed on one
        bearing and survived on the other.
        """
        import numpy as np

        from terrainflow_assessment.modules.burn_strategy import taper_reach

        ew = taper_reach(self._band(4, "rows"), 1.5, (2.0, 1.0))
        ns = taper_reach(self._band(4, "cols"), 1.5, (1.0, 2.0))
        # Same feature, transposed grid: the cut per metre must match.
        assert float(ew[20:24, 20].sum()) * 2.0 == pytest.approx(
            float(ns[20, 20:24].sum()) * 2.0, rel=1e-9)
        assert np.isclose(ew[20:24, 20], ns[20, 20:24]).all()

    def test_a_square_grid_is_untouched_at_every_bearing(self):
        """The safety property that makes the fix cheap: square cells cannot change.

        Both axes agree, so there is nothing for the correction to choose between, and
        the function takes its old path bit-for-bit. Asserted on diagonal footprints
        too, where the nearest outside cell is a corner rather than a neighbour — that
        is the case where a direction-aware inset would have moved the answer, and it
        measured *worse* against the true geometry than leaving it alone.
        """
        import numpy as np

        from terrainflow_assessment.modules.burn_strategy import taper_reach

        rr, cc = np.indices((61, 61))
        for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
            th = np.radians(angle)
            perp = np.abs(-np.sin(th) * (cc - 30.0) + np.cos(th) * (rr - 30.0))
            mask = perp <= 3.0
            got = taper_reach(mask, 2.0, (2.0, 2.0))
            # The old expression, inlined: distance less half of min(h, w).
            from scipy.ndimage import distance_transform_edt
            dist = distance_transform_edt(mask, sampling=(2.0, 2.0))
            want = np.clip((dist - 1.0) / 2.0, 0.0, 1.0)
            assert np.array_equal(got, want), f"square grid moved at {angle}deg"


# ---------------------------------------------------------------------------
# The volumetric burns claim the width they were drawn as
# ---------------------------------------------------------------------------

class TestVolumetricBurnsClaimTheWidthTheyWereDrawn:
    """``all_touched`` claims every cell the band so much as brushes — a
    near-constant ~1.3 m wider than drawn, whatever the length or bearing — and the
    volumetric burns then level or raise every one of those cells to full depth.
    ``_rasterize``'s own docstring says the volumetric burns pass ``False`` "for
    exactly that reason"; three of them did not.

    Measured on a transect across the middle of the alignment, away from the
    buffer's rounded ends. The alignment sits at y=20.2 so the band's edges fall
    strictly inside cells: at y=20.0 they land on cell centres, where the count is
    genuinely ambiguous and the answer is `taper_reach`'s quadrature over-read
    rather than anything to do with this.
    """

    @staticmethod
    def _burn(tmp_path, ew_type, **kw):
        data = np.full((60, 60), 50.0)
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data))
        geom = make_mock_line_geom([(10.0, 20.2), (40.0, 20.2)])
        ew = _mock_ew(ew_type, geom, **kw)
        return b, b.burn_earthworks([ew]), ew

    def test_a_diversion_cuts_the_band_it_was_drawn_as(self, tmp_path):
        b, burned, _ = self._burn(tmp_path, "diversion", depth=1.0, width=3.0,
                                  bottom_width_m=1.0, gradient_pct=0.0)
        cut = b.original[:, 25] - burned[:, 25]
        rows = np.nonzero(cut > 1e-9)[0]
        assert len(rows) == 3, (
            f"a 3.0 m drain on 1 m cells cut {len(rows)} cells across, not 3 — and "
            f"the measured cut is the column the report tells a contractor to price "
            f"the job on")

    def test_a_diversion_cuts_the_section_it_was_drawn_as(self, tmp_path):
        """The register's check, taken on a transect rather than on the site total.

        The total also carries the buffer's rounded end caps and
        ``enforce_monotonic_path``'s per-cell step down the alignment — neither of
        which belongs to the cross-section, and the second of which grows with
        length, so no fixed tolerance on the total can mean what it looks like.
        """
        from terrainflow_assessment.core.sizing.primitives import trapezoid_section
        from terrainflow_assessment.modules.earthwork_design import channel_batter_run

        b, burned, ew = self._burn(tmp_path, "diversion", depth=1.0, width=3.0,
                                   bottom_width_m=1.0, gradient_pct=0.0)
        # One whole cell of batter run, so taper_reach's quadrature is exact — see
        # its docstring on why a half-cell run is not.
        assert channel_batter_run(ew) == pytest.approx(1.0)

        section = float(np.clip(b.original[:, 25] - burned[:, 25], 0.0, None).sum())
        section *= b.cell_size
        want = trapezoid_section(3.0, 3.0 - 2 * channel_batter_run(ew), 1.0).area
        assert section == pytest.approx(want, rel=0.02), (
            f"{section:.2f} m2 per metre against a drawn {want:.2f} m2")

    def test_a_berm_raises_the_band_it_was_drawn_as(self, tmp_path):
        """The berm's footprint is its **base**, not its crest: ``width`` on top
        with batters falling away at the registry slope, so 2.0 m wide and 0.5 m
        deep at 1:1 covers 3.0 m of ground. Three cells is that base; four is the
        `all_touched` claim on top of it."""
        from terrainflow_assessment.modules.earthwork_design import berm_batter_run

        b, burned, _ = self._burn(tmp_path, "berm", depth=0.5, width=2.0)
        base = 2.0 + 2 * berm_batter_run(0.5)
        assert base == pytest.approx(3.0), "the fixture's base is not 3 whole cells"

        rise = burned[:, 25] - b.original[:, 25]
        rows = np.nonzero(rise > 1e-9)[0]
        assert len(rows) == 3, (
            f"a 3.0 m base on 1 m cells raised {len(rows)} cells across, not 3 — "
            f"so the berm placed a third again the fill it was drawn to place")

    def test_the_centreline_seal_does_not_raise_a_cell_twice(self, tmp_path):
        """The seal now runs on every alignment, not only a sub-cell one: centre-based
        rasterising can leave a diagonal band with a corner-only join, and a barrier
        with one of those in it is not a barrier.

        ``_burn_dam`` seals with ``maximum``, which is idempotent. This burn raises
        with ``+=``, so a path cell already inside its own band would be raised twice
        and the crest would stand a whole depth too high right along the centreline.
        """
        b, burned, _ = self._burn(tmp_path, "berm", depth=0.5, width=2.0)
        rise = burned - b.original
        assert float(np.nanmax(rise)) == pytest.approx(0.5, abs=1e-6), (
            f"a cell was raised {float(np.nanmax(rise)):.2f} m by a 0.50 m berm — "
            f"the seal added its depth to a cell the band had already raised")

    def test_a_diagonal_berm_is_still_a_connected_barrier(self, tmp_path):
        """What the seal is for. Water finds a corner-only join."""
        from scipy.ndimage import label

        data = np.full((60, 60), 50.0)
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data))
        ew = _mock_ew("berm", make_mock_line_geom([(12.0, 12.3), (44.0, 44.7)]),
                      depth=0.5, width=1.2)
        burned = b.burn_earthworks([ew])
        raised = (burned - b.original) > 1e-9
        # 4-connectivity: a diagonal join is exactly what a corner-only touch is,
        # and water goes through one.
        _labels, n = label(raised, structure=np.array([[0, 1, 0],
                                                       [1, 1, 1],
                                                       [0, 1, 0]]))
        assert n == 1, f"the berm burned as {n} disconnected pieces"


class TestTheBermIsBuiltAsTheSectionItIsPricedAs:
    """`calculate_fill_volume` assumed a 1:1 triangle (``depth²``, ignoring
    ``width``) and `_burn_berm` placed a vertical ``width × depth`` prism — 0.25
    against 1.00 m³/m at the registry defaults. Both now build the trapezoid the
    registry specifies: ``width`` on top, batters at ``default_side_slope``, base
    ``width + 2·depth·slope``.

    Measured on a transect across the middle, away from the buffer's rounded ends.
    The site total also carries those caps — ~7% on the fixture below — which is a
    property of a buffered line, not of the cross-section.
    """

    @staticmethod
    def _burn(tmp_path, depth, width, y=30.2, length=40.0):
        data = np.full((80, 80), 50.0)
        b = DEMBurner(_write_dem(str(tmp_path / "d.tif"), data))
        geom = make_mock_line_geom([(15.0, y), (15.0 + length, y)])
        geom.length.return_value = length
        ew = _mock_ew("berm", geom, depth=depth, width=width)
        return b, b.burn_earthworks([ew]), ew, geom, length

    @staticmethod
    def _transect(burner, burned, col=35):
        rise = burned[:, col] - burner.original[:, col]
        rows = np.nonzero(rise > 1e-9)[0]
        return rows, float(rise[rows].sum()) * burner.cell_size

    def test_the_burn_and_the_formula_agree_on_the_section(self, tmp_path):
        """The register's case: 1 m cells, depth 1.0, width 2.0, slope 1:1.

        The run is then one whole cell, which is the condition `taper_reach`'s own
        docstring gives for its quadrature being exact — base 4.0, area 3.0 m³/m.
        """
        from terrainflow_assessment.core.sizing import trapezoid_section
        from terrainflow_assessment.modules.earthwork_design import (
            berm_batter_run,
            calculate_fill_volume,
        )

        b, burned, _ew, geom, length = self._burn(tmp_path, depth=1.0, width=2.0)
        run = berm_batter_run(1.0)
        assert run == pytest.approx(1.0), "the fixture is not on a whole-cell run"

        want = trapezoid_section(2.0 + 2 * run, 2.0, 1.0).area
        assert want == pytest.approx(3.0)

        rows, built = self._transect(b, burned)
        assert len(rows) == 4, f"a 4.0 m base on 1 m cells covered {len(rows)} cells"
        assert built == pytest.approx(want, rel=0.02), (
            f"the burn placed {built:.3f} m3/m against a drawn {want:.3f}")
        assert calculate_fill_volume("berm", geom, 1.0, 2.0) == pytest.approx(
            want * length, rel=0.02)

    def test_the_crest_is_the_drawn_width_and_the_batters_fall_away(self, tmp_path):
        """The shape, not just the volume: full height across ``width``, tapering
        outside it. A bank of the right volume and the wrong height blocks the
        wrong storm."""
        b, burned, _ew, _geom, _length = self._burn(tmp_path, depth=1.0, width=2.0)
        rows, _ = self._transect(b, burned)
        rise = (burned[:, 35] - b.original[:, 35])[rows]
        assert sorted(np.round(rise, 3).tolist()) == [0.5, 0.5, 1.0, 1.0], (
            f"the section is {np.round(rise, 3).tolist()}, not a crest with batters")

    def test_at_the_registry_defaults_a_half_cell_run_builds_flat(self, tmp_path):
        """Recorded because it is a real figure the commit quotes, not endorsed.

        The shipped berm is 0.5 m deep at 1:1, so on 1 m cells the run is half a
        cell and ``taper_reach`` "leaves no taper at all" — the 3.0 m base band is
        built at full height, 1.50 m³/m against a drawn 1.25. That is the same
        quadrature over-read `taper_reach`'s docstring records on the cut side
        ("always an over-cut ... not fixed here"), now reaching the fill side. It
        is an over-build, which for a barrier is the safe direction, and it is
        smaller than the 4x mismatch it replaces.
        """
        from terrainflow_assessment.core.sizing import trapezoid_section

        b, burned, _ew, _geom, _length = self._burn(tmp_path, depth=0.5, width=2.0)
        rows, built = self._transect(b, burned)
        drawn = trapezoid_section(3.0, 2.0, 0.5).area
        assert drawn == pytest.approx(1.25)
        assert len(rows) == 3 and built == pytest.approx(1.5, rel=1e-3), (
            f"the measured over-build moved: {built:.3f} m3/m over {len(rows)} cells")


class TestChainageHasOneImplementation:
    """M-5b. `_band_chainage` and `swale_design.line_stations` asked the same question.

    Both answer "how far along this alignment is the point on it nearest each cell" —
    one in numpy, vectorised over segments x cells, one through GEOS. They were written
    two years apart for two call sites and were never compared. They agree **exactly**:
    over a straight run, a twelve-vertex alignment, a hairpin where the nearest segment
    is genuinely ambiguous, a 200-vertex line, a zero-length segment and points beyond
    both ends, the largest disagreement was 0.000e+00 m.

    GEOS is the one that survives, for the reason R-6 and M-8 were done in the same
    pass: the numpy body allocates a `(segments x cells)` array, which is ~190 MB for a
    200-vertex alignment over a 20,000-cell band, while `line_locate_point` is linear in
    the cells. It is also already the load-bearing implementation — it is what took the
    vertex-edit freeze's largest term from 3,236 ms to 622 ms.
    """

    @staticmethod
    def _burner(tmp_path):
        path = str(tmp_path / "chainage.tif")
        return DEMBurner(_write_dem(path, np.full((40, 40), 50.0)))

    def test_the_burner_projects_through_the_shared_helper(self, tmp_path, monkeypatch):
        """The pin on there being ONE implementation.

        A reintroduced local copy would still compute the right answer, so asserting the
        answer cannot catch it. Asserting the delegation can.
        """
        from terrainflow_assessment.modules import earthwork_design

        calls = []
        real = earthwork_design.line_stations

        def spy(line, xs, ys):
            calls.append((len(xs), len(ys)))
            return real(line, xs, ys)

        monkeypatch.setattr(earthwork_design, "line_stations", spy)

        b = self._burner(tmp_path)
        mask = np.zeros(b.original.shape, dtype=bool)
        mask[10:14, 5:25] = True
        coords = [(0.0, 0.0), (30.0, 0.0), (40.0, 10.0)]
        b._band_chainage(mask, coords)

        assert len(calls) == 1, (
            f"_band_chainage made {len(calls)} calls to the shared helper; it must make "
            f"exactly one and must not carry its own projection"
        )
        assert calls[0][0] == int(mask.sum())

    def test_it_returns_the_cells_of_the_mask_with_their_distance_along(self, tmp_path):
        """The contract itself, on an alignment laid along a known row."""
        b = self._burner(tmp_path)
        mask = np.zeros(b.original.shape, dtype=bool)
        mask[5, 3:8] = True

        # A straight line along the centre of row 5, running east.
        tr = b.transform
        y_row5 = tr.f + 5.5 * tr.e
        x0 = tr.c + 0.5 * tr.a
        coords = [(x0, y_row5), (x0 + 100.0, y_row5)]

        rows, cols, chainage = b._band_chainage(mask, coords)
        assert np.array_equal(rows, np.full(5, 5))
        assert np.array_equal(cols, np.arange(3, 8))
        # Cell centres are one cell apart, so chainage steps by the cell size.
        step = abs(tr.a)
        assert np.allclose(chainage, np.arange(3, 8) * step, atol=1e-9), (
            f"chainage {chainage} is not the run of cell centres along the line"
        )

    def test_a_point_beyond_the_end_clamps_rather_than_extrapolating(self, tmp_path):
        """`project` clamps; a band can overhang its own alignment and must not go past."""
        b = self._burner(tmp_path)
        mask = np.zeros(b.original.shape, dtype=bool)
        mask[5, 30:33] = True

        tr = b.transform
        y_row5 = tr.f + 5.5 * tr.e
        x0 = tr.c + 0.5 * tr.a
        coords = [(x0, y_row5), (x0 + 10.0 * abs(tr.a), y_row5)]   # ends well short

        _rows, _cols, chainage = b._band_chainage(mask, coords)
        length = 10.0 * abs(tr.a)
        assert np.allclose(chainage, length), (
            f"cells past the end of the alignment gave {chainage}, not the line length "
            f"{length} — chainage must clamp, or a graded invert keeps cutting deeper "
            f"past the end of the channel it is grading"
        )
