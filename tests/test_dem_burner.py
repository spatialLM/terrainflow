"""Tests for plugin/processing/dem_burner.py"""

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

    **Footprint.** Rasterising many overlapping discs with ``all_touched`` and
    unioning the result claims marginally more cells than rasterising their union
    polygon, because a cell merely brushed by any one disc counts. The difference is
    a couple of cells on the rounded ends.
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
        band = burner._rasterize(line.buffer(ew.width / 2.0))
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
                cell_mask = burner._rasterize(Point(x, y).buffer(ew.width / 2))
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
        ([(5.0, 20.0), (35.0, 20.0)], 3.0, 1.0),                # straight, due east
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
        ([(5.0, 20.0), (35.0, 20.0)], 3.0, 1.0),
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
