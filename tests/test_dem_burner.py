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

        from terrainflow_assessment.modules.burn_strategy import enforce_monotonic_path

        dem = dem.copy()
        coords = list(line.coords)
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
                burn = start_elev - (cum[seg_i] + t * seg) * grad - ew.depth
                cell_mask = burner._rasterize(Point(x, y).buffer(ew.width / 2))
                if cell_mask.any():
                    dem[cell_mask] = np.minimum(dem[cell_mask], burn)
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

    def test_a_sub_cell_drain_still_carves_a_connected_path(self, tmp_path):
        """The buffer rasterises empty, so the centreline path is the fallback."""
        b, got, _ = self._setup(tmp_path, [(5.0, 20.4), (35.0, 20.4)], 0.05, 1.0,
                                flat=True)
        cut = got < b.original - 1e-9
        assert cut.any(), "a sub-cell drain burned nothing at all"
        assert cut.sum() >= 25, f"only {int(cut.sum())} cells — the path is broken"
