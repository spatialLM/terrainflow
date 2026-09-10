"""The analysis over a DEM that is *not* well behaved.

Every other module in this suite runs against the smooth synthetic surface,
which is deliberately kind: no noise, no closed depressions, no nodata. Real
farm LiDAR has all three, and none of the code that has to cope with them is
touched by the other 259 checks.

These build their own fixture via ``build_synthetic_dem(rough=..., pits=...,
voids=...)`` rather than using the shared one, so the smooth default and all
49 baseline screenshots stay exactly as they were.

What counts as a failure here is narrower than elsewhere. Different *numbers*
on rougher terrain are correct — more streams, shorter contours, a different
keypoint. What must not happen is an exception, a swallowed error on the
message bar, a nodata sentinel leaking into a result as if it were an
elevation, or an output so degenerate it could never be drawn.
"""

import os
import tempfile
from pathlib import Path

from _harness import NODATA, PluginHarness, build_synthetic_dem

# One nasty DEM per module run, built once and shared by every check below.
# Module-level so the cost is paid once; each check still gets a fresh project.
_NASTY = None


def _nasty_dem():
    global _NASTY
    if _NASTY is None:
        workdir = Path(tempfile.mkdtemp(prefix="tfa_nasty_"))
        _NASTY = build_synthetic_dem(
            workdir / "nasty_dem.tif", rough=True, pits=6, voids=2)
    return _NASTY


def _band(path):
    """Band 1 as a numpy array, read without needing a QGIS layer."""
    import numpy as np
    import rasterio

    with rasterio.open(str(path)) as src:
        return np.asarray(src.read(1))


# ---------------------------------------------------------------------------
# The fixture itself — if these fail, nothing below means anything
# ---------------------------------------------------------------------------

def check_nasty_fixture_is_actually_nasty(dem_path):
    """rough/pits/voids must really spoil the surface, not merely claim to.

    A robustness fixture that quietly produced the smooth surface would be the
    worst kind of test asset: everything below it passes and none of it touches
    the code it exists to reach.
    """
    import numpy as np

    smooth, nasty = _band(dem_path), _band(_nasty_dem())

    holes = int((nasty == NODATA).sum())
    assert holes > 0, "voids=2 produced no nodata cells"
    assert holes < nasty.size * 0.05, (
        f"{holes} nodata cells is {holes / nasty.size:.1%} of the raster — the "
        "voids are the story rather than an edge case")

    def sinks(arr):
        """Cells strictly below all eight neighbours — genuine closed depressions."""
        arr = np.where(arr == NODATA, np.inf, arr)
        core = arr[1:-1, 1:-1]
        lower = np.ones_like(core, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    lower &= core < arr[1 + dr:arr.shape[0] - 1 + dr,
                                        1 + dc:arr.shape[1] - 1 + dc]
        return int(lower.sum())

    smooth_sinks, nasty_sinks = sinks(smooth), sinks(nasty)
    assert nasty_sinks > smooth_sinks, (
        f"{nasty_sinks} closed depressions vs {smooth_sinks} on the smooth "
        "surface — rough/pits did not bite")

    finite = nasty[nasty != NODATA]
    relief = float(finite.max() - finite.min())
    assert relief > 40.0, (
        f"only {relief:.1f} m of relief left — the noise has swamped the valley "
        "and nothing downstream would be meaningful")
    print(f"\n    {nasty_sinks} sinks (smooth: {smooth_sinks}), {holes} nodata "
          f"cells, {relief:.1f} m relief", end="")


def check_nasty_fixture_is_reproducible(dem_path):
    """Same seed, byte-identical file — or the screenshot baseline stops meaning
    anything.

    Everything random goes through ``np.random.default_rng(seed)``. One
    unseeded call anywhere and two runs of unchanged code produce different
    terrain, every image moves, and the visual diff becomes noise.
    """
    import hashlib

    workdir = Path(tempfile.mkdtemp(prefix="tfa_seed_"))
    opts = dict(rough=True, pits=6, voids=2)

    def digest(name, **kwargs):
        p = build_synthetic_dem(workdir / name, **kwargs)
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    first, second = digest("a.tif", **opts), digest("b.tif", **opts)
    assert first == second, (
        "two builds with the same seed differ — something uses an unseeded RNG "
        "and no baseline over this fixture can ever be stable")
    assert digest("c.tif", seed=7, **opts) != first, (
        "changing the seed produced an identical file — seed is not wired through")


# ---------------------------------------------------------------------------
# The hunt
# ---------------------------------------------------------------------------

def check_baseline_survives_rough_terrain(dem_path):
    """The whole baseline path over noise, 6 pits and 2 nodata holes.

    The broadest net in the file: if anything in AnalysisWorker cannot cope
    with a surface that is not smooth, it surfaces here as a critical on the
    message bar rather than as a wrong number nobody notices.
    """
    with PluginHarness(_nasty_dem()) as h:
        result = h.run_baseline()
        h.assert_no_errors("baseline over rough terrain")

        assert result is not None, "baseline produced no result on rough terrain"
        acc = result.get("flow_accumulation")
        assert acc and os.path.exists(acc), (
            f"flow_accumulation missing on disk: {acc!r}")
        assert result.get("runoff_mm", 0) > 0, (
            f"runoff_mm was {result.get('runoff_mm')!r} — the storm did not run")
        assert result.get("catchment_area_m2", 0) > 0, (
            "catchment area came back zero on a DEM that still has 80 m of relief")


def check_nodata_does_not_leak_into_results(dem_path):
    """-9999 must never survive into a result raster as if it were an elevation.

    This is the specific way nodata handling goes wrong quietly: the sentinel is
    read as a real value, so a hillshade renders a cliff, a slope grid reports
    thousands of degrees, and the statistics are silently poisoned. Nothing
    raises, and no state assertion notices.
    """
    from qgis.core import QgsProject

    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.assert_no_errors("baseline over rough terrain")

        offenders = []
        for lid in h.state.baseline_layer_ids:
            layer = QgsProject.instance().mapLayer(lid)
            if layer is None or not hasattr(layer, "dataProvider"):
                continue
            provider = layer.dataProvider()
            if not hasattr(provider, "bandStatistics"):
                continue
            stats = provider.bandStatistics(1)
            # Only the NEGATIVE side is diagnostic. A surviving -9999 drags the
            # minimum down near it, and nothing these layers legitimately hold —
            # elevation, upstream-cell counts, runoff volume, pond capacity — is
            # ever hugely negative.
            #
            # Deliberately no upper bound. An earlier version of this check also
            # failed anything above +9000 and duly "found" Streams at 52,632 on
            # the rough DEM. That is flow accumulation counting upstream cells:
            # the smooth DEM, which contains no nodata whatsoever, reaches
            # 32,400 on the same layer. Large positives are the normal output,
            # so the rule was the bug, not the plugin.
            if stats.minimumValue <= -9000.0:
                offenders.append(
                    f"{layer.name()}: {stats.minimumValue:.1f}..{stats.maximumValue:.1f}")

        assert not offenders, (
            "the nodata sentinel reached a result layer as a real value:\n    "
            + "\n    ".join(offenders))


def check_streams_do_not_shatter_on_noise(dem_path):
    """Rough terrain must not turn the stream network into confetti.

    More streams on rougher ground is correct. Thousands of one-cell fragments
    is not — it means flow is being routed cell by cell rather than
    concentrated, which usually points at depressions never having been filled.
    A count is a blunt instrument, but the failure it catches is unmistakable.
    """
    from qgis.core import QgsProject, QgsVectorLayer

    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.assert_no_errors("baseline over rough terrain")

        for lid in h.state.baseline_layer_ids:
            layer = QgsProject.instance().mapLayer(lid)
            if isinstance(layer, QgsVectorLayer) and "stream" in layer.name().lower():
                count = layer.featureCount()
                assert count < 2000, (
                    f"{layer.name()} has {count} features on a 36 ha site — the "
                    "network has shattered rather than concentrated")
                break


def check_contours_survive_rough_terrain(dem_path):
    """Contour generation over a noisy surface.

    The smooth fixture produces perfectly clean contours, which is the one thing
    real terrain never does. Noise is what makes a contour algorithm produce
    hundreds of thousands of tiny wiggles, or fail outright.
    """
    from qgis.core import QgsProject, QgsVectorLayer

    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.panel.generate_simple_contours_requested.emit()
        h.assert_no_errors("contours over rough terrain")

        for layer in QgsProject.instance().mapLayers().values():
            if isinstance(layer, QgsVectorLayer) and "contour" in layer.name().lower():
                count = layer.featureCount()
                assert count < 20000, (
                    f"{layer.name()} has {count} features over 36 ha — noise has "
                    "shattered the contours into unusable fragments")
                break


def check_contour_analysis_survives_rough_terrain(dem_path):
    """The segment/inflow pass, which reads geometry the noise has made ragged."""
    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.panel.generate_simple_contours_requested.emit()
        h.panel.run_contour_analysis_requested.emit()
        h.assert_no_errors("contour analysis over rough terrain")


def check_keypoints_survive_rough_terrain(dem_path):
    """Keypoint + ridgeline detection where the slope break is no longer clean.

    On the smooth surface the slope break is analytic and unambiguous. Noise
    puts thousands of local slope reversals in the way, which is exactly the
    condition a real site presents.
    """
    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.panel.run_keypoint_analysis_requested.emit()
        h.assert_no_errors("keypoints over rough terrain")


def check_terrain_indices_survive_rough_terrain(dem_path):
    """Curvature and TPI are second-derivative measures — noise hits them hardest.

    A plan-curvature grid over a smooth parabola is well conditioned. Over a
    rough surface it is the first thing to blow up to absurd magnitudes.
    """
    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.panel.run_terrain_indices_requested.emit()
        h.assert_no_errors("terrain indices over rough terrain")


def check_pits_actually_produce_pond_capacity(dem_path):
    """Six real depressions must give the crest split something to find.

    ``pond=True`` cuts one large clean basin by hand. This is six irregular
    ones of varying depth scattered off the channel — closer to real terrain,
    and the case where crest routing chooses between competing basins rather
    than finding the obvious one.

    The assertion is a measured contrast, not a guess. Pond Capacity over the
    smooth default is 0.00..0.00 everywhere, because that surface is
    depression-free by construction; over this one it reaches ~0.98. If it
    ever comes back flat zero here too, the pits are not reaching the routing
    code and this whole module is only testing that nothing crashed.
    """
    from qgis.core import QgsProject

    with PluginHarness(_nasty_dem()) as h:
        h.run_baseline()
        h.assert_no_errors("baseline over rough terrain")
        h.panel.analysis_inputs_changed.emit()
        h.assert_no_errors("re-analysis with multiple depressions")

        capacity = None
        for lid in h.state.baseline_layer_ids:
            layer = QgsProject.instance().mapLayer(lid)
            if layer is not None and "pond capacity" in layer.name().lower():
                capacity = layer.dataProvider().bandStatistics(1).maximumValue
                break

        assert capacity is not None, "no Pond Capacity layer was produced"
        assert capacity > 0.0, (
            "Pond Capacity is flat zero over a DEM with six genuine "
            "depressions — the pits are not reaching the crest split, so this "
            "module is only proving that nothing crashed")
        print(f"\n    pond capacity peaks at {capacity:.2f} (smooth DEM: 0.00)",
              end="")
