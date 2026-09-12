"""The committed DEM fixtures are the terrain we think they are.

Binary fixtures fail quietly. A GeoTIFF that has been LF-normalised on checkout, or
swapped for a re-export with different nodata, does not announce itself — it produces
slightly different numbers in a hundred other tests, and the search starts everywhere
except the fixture. `.gitattributes` marks `*.tif binary` so the normalisation cannot
happen; this asserts the outcome rather than trusting the declaration.

Every pin here is on a quantity that MOVES if the file changes: the sha256 of the pixel
bytes, the count of valid cells, the elevation range. The geometry (size, CRS, cell
size, origin) is pinned too, because a reprojection or a resample would keep the file
openable while making every measured result incomparable to the ones in the register.

See `README.md` in this directory for provenance and for which fixture to use when.
"""

import hashlib
import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

#: Everything checkable about each fixture, measured 2026-09-12 on the files as
#: committed. `sha256` is over `array.tobytes()` — the pixels, not the file — so a
#: lossless recompression is allowed to change the file and forbidden to change this.
SPECS = {
    "quail_island_catchment.tif": {
        "width": 400,
        "height": 400,
        "origin": (1574561.0, 5169632.0),
        "sha256": "c8817c47bc75935ee1310dfe4c6ada2d426eb501978463a4e568364e0694903d",
        "n_valid": 160_000,
        "elev_min": -0.1000,
        "elev_max": 84.7793,
    },
    "quail_island_full.tif": {
        "width": 1139,
        "height": 1016,
        "origin": (1574379.0, 5169967.0),
        "sha256": "8565111982621bfdb1bdc2719cd0bb4f5ff16c731dc01bf85cef5481eec299d8",
        "n_valid": 702_134,
        "elev_min": -0.1000,
        "elev_max": 84.7793,
    },
}

#: Shared by both, because both are extracts of the same LINZ tile.
CRS = "EPSG:2193"
CELL_SIZE_M = 1.0
NODATA = -9999.0


def _read(name):
    """Return (array, dataset profile pieces) for one fixture."""
    with rasterio.open(FIXTURES / name) as src:
        return src.read(1), {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": src.transform,
            "nodata": src.nodata,
        }


@pytest.mark.parametrize("name", sorted(SPECS))
class TestTheFixtureIsWhatItClaims:
    """One class per assertion rather than one big test, so a failure names the axis."""

    def test_it_exists(self, name):
        assert (FIXTURES / name).is_file(), (
            f"{name} is missing. It is committed; a missing file means a bad checkout, "
            f"not a test to skip."
        )

    def test_the_grid_is_unchanged(self, name):
        spec = SPECS[name]
        _a, p = _read(name)
        assert (p["width"], p["height"]) == (spec["width"], spec["height"])
        assert p["crs"] == CRS
        assert p["transform"].a == CELL_SIZE_M
        assert p["transform"].e == -CELL_SIZE_M
        # Tolerance is 1e-6 m: far under a 1 m cell, far over the float noise the
        # original clip left behind (the full tile's easting is 1574379.0000000002).
        assert p["transform"].c == pytest.approx(spec["origin"][0], abs=1e-6), (
            "the origin moved — this fixture has been re-clipped, and every extent, "
            "area and coordinate measured against it is now off by the difference"
        )
        assert p["transform"].f == pytest.approx(spec["origin"][1], abs=1e-6)

    def test_nodata_is_declared(self, name):
        _a, p = _read(name)
        assert p["nodata"] == NODATA, (
            f"{name} declares nodata={p['nodata']}. A DEM whose nodata is None reads "
            f"its voids as real elevations, which is how a sea cell becomes a -9999 m pit"
        )

    def test_the_pixels_are_byte_for_byte_what_was_committed(self, name):
        a, _p = _read(name)
        assert a.dtype == np.float32, f"{name} is {a.dtype}, not float32"
        digest = hashlib.sha256(a.tobytes()).hexdigest()
        assert digest == SPECS[name]["sha256"], (
            f"{name}'s pixels changed. This is the check that catches a corrupted or "
            f"silently replaced fixture; if the replacement was deliberate, re-record "
            f"the spec and say in the commit what changed and why"
        )

    def test_the_valid_cell_count_holds(self, name):
        a, _p = _read(name)
        n_valid = int((a != NODATA).sum())
        assert n_valid == SPECS[name]["n_valid"], (
            f"{name} has {n_valid:,} valid cells, expected "
            f"{SPECS[name]['n_valid']:,} — the nodata mask moved"
        )

    def test_the_elevation_range_holds(self, name):
        a, _p = _read(name)
        valid = a[a != NODATA]
        assert float(valid.min()) == pytest.approx(SPECS[name]["elev_min"], abs=1e-4)
        assert float(valid.max()) == pytest.approx(SPECS[name]["elev_max"], abs=1e-4)


class TestTheTwoFixturesAreTheSameTerrain:
    """The clip is a window of the full tile — the property that makes them comparable.

    Not a resample and not a reprojection: the same cells, at the same coordinates, with
    the same values. This is what lets a number measured on the clip be reasoned about
    against the same number on the full tile, and it is the first thing to break if
    either file is ever re-exported from a different source.
    """

    #: Offset of the clip within the full tile, in cells. Derived from the two origins,
    #: and asserted rather than assumed.
    COL, ROW = 182, 335

    def test_the_offset_is_what_the_origins_imply(self):
        _ca, cp = _read("quail_island_catchment.tif")
        _fa, fp = _read("quail_island_full.tif")
        col = round(cp["transform"].c - fp["transform"].c)
        row = round(fp["transform"].f - cp["transform"].f)
        assert (col, row) == (self.COL, self.ROW)

    def test_the_clip_is_a_bit_exact_window_of_the_full_tile(self):
        clip, cp = _read("quail_island_catchment.tif")
        with rasterio.open(FIXTURES / "quail_island_full.tif") as full:
            sub = full.read(1, window=Window(self.COL, self.ROW, cp["width"], cp["height"]))
        # Compare the raw bits, not the floats: `==` would call two different NaN
        # payloads equal-ish and 0.0 == -0.0, and a re-export can change either.
        assert np.array_equal(sub.view(np.uint32), clip.view(np.uint32)), (
            "the clip is no longer a bit-exact window of the full tile — one of them "
            "has been re-exported, resampled or reprojected, and results measured on "
            "the two are no longer comparable"
        )


class TestTheReadmeMatchesTheFiles:
    """H-3/H-4/H-9 taught this repo that a number in prose goes stale unmeasured.

    The fixtures README quotes dimensions and cell counts. They are cheap to check, so
    they are checked.
    """

    @pytest.mark.parametrize("name", sorted(SPECS))
    def test_the_readme_quotes_the_real_dimensions(self, name):
        readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
        spec = SPECS[name]
        stated = f"{spec['width']} x {spec['height']}"
        assert stated in readme, (
            f"the fixtures README does not state {name} as {stated!r}"
        )

    @pytest.mark.parametrize("name", sorted(SPECS))
    def test_the_readme_quotes_the_real_cell_counts(self, name):
        readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
        spec = SPECS[name]
        total = f"{spec['width'] * spec['height']:,}"
        valid = f"{spec['n_valid']:,}"
        assert total in readme, f"README is missing {name}'s total cell count {total}"
        assert valid in readme, f"README is missing {name}'s valid cell count {valid}"
