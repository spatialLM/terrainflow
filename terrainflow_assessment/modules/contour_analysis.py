"""
contour_analysis.py — Contour extraction, slope filtering, and flow-crossing ranking.

Provides:
  extract_contours()        — extract contours from DEM via GDAL
  filter_by_slope()         — reject contours where mean slope > max_slope_deg
  rank_by_flow_crossing()   — rank contours by peak flow accumulation crossing them
  clip_to_usable_area()     — clip contours to a user-defined usable polygon
  natural_breaks()          — Jenks class boundaries over a set of inflow values
  ContourFeature            — lightweight container for a ranked contour
"""

import logging
import math
import os
import shutil

import numpy as np

from terrainflow_assessment.modules.footprint import sample_along_line, xy_to_rc

_log = logging.getLogger(__name__)

# Colour grammar shared by every "how much water arrives here" display: the ranked
# candidate contours, the along-contour inflow gradient, and the peak-inflow overlay
# inside the recommended swale segments.
#
# Read over aerial imagery, which is where this is actually used. Three rules come
# out of that, and the first two overturn earlier attempts:
#
#  * **Opaque.** A near-transparent low end was tried for overlay-friendliness and
#    is simply not visible against foliage — you cannot tell line from background.
#    Quiet stretches recede by getting *thin* instead (see the width ramp in
#    controllers/contour.py); a 0.7 mm hairline is unobtrusive and still legible,
#    which a translucent wash is not.
#  * **Chroma, not just lightness.** A single-hue light→dark ramp asks the eye to
#    compare lightness against a background that has its own huge lightness range
#    (sunlit grass to tree shadow), so the pale end vanished on grass and the dark
#    end on shadow. This ramp keeps saturation high the whole way and darkens, so
#    the two channels reinforce each other.
#  * **No green, no brown.** Those are the imagery's own colours. Cyan→navy sits
#    outside the aerial palette at every step, while staying one perceptual family
#    so it still reads as one quantity rising rather than a rainbow's worth of
#    categories.
#
# Lives here, beside the classification it colours, so the panel legend and the
# renderers cannot drift apart — both import it.
INFLOW_RAMP_HEX = ("#6fd8ef", "#1f9ed4", "#1b58b8", "#101a63")

# The same quantity, read against a different background — and that is the whole
# reason it is a second ramp rather than a shared one.
#
# The two ramps above are drawn on *ground*: candidate contours and the along-contour
# gradient lie on aerial imagery, where cyan→navy is right precisely because it is not
# one of the imagery's colours. The segment overlay is drawn **inside the recommended
# swale's own green core**, which is a colour, and blue against green is a hue step of
# well under a quadrant: on a field run-through the bands were hard to pick out of the
# band they sit in, and the palest one — the thinnest, so already the quietest —
# effectively was not there.
#
# Violet-magenta, at a hue a full quadrant clear of *both* colours the core can be:
# green where the segment holds its inflow, amber where it cannot. The overlay
# therefore separates from the core by hue rather than by asking lightness to do the
# work a second time, and it does not stop working on the segments that are flagged —
# which a warm ramp would have, on exactly the segments worth looking at. It stays
# outside the aerial palette too, so the rule the ramp above is built on is not broken
# on the way past.
#
# Same shape as INFLOW_RAMP_HEX — four classes, light to dark with volume — so it is
# the same grammar read on a different ground, not a second scheme to learn.
SEGMENT_INFLOW_RAMP_HEX = ("#f5c2ff", "#d458ed", "#a01ab8", "#5e076b")


def _sample_line(geom, transform, array, distances, fill=np.nan, nodata=None):
    """Raster values along *geom* at *distances* — vectorised where that is safe.

    Five call sites in this module walked a line a point at a time, each doing a
    ``geom.interpolate`` and a single-element raster index per cell-width. On a 1 m DEM
    with two hundred 2 km contours that is of the order of 400,000 Python-level shapely
    calls per pass, and two passes run. A single ``LineString`` now goes through
    ``footprint.sample_along_line`` in one vectorised pass instead.

    A ``MultiLineString`` deliberately keeps the old path. Shapely's chainage runs
    across its parts *without* the gaps between them; concatenating the parts'
    coordinates would turn each gap into a real segment and resample the whole line on a
    different chainage. Rare, and not worth answering a different question quickly.

    ``fill`` is what an off-grid point contributes — zero for an accumulation profile
    indexed by position, NaN for a mean that must exclude it. ``nodata`` maps a declared
    sentinel to NaN.
    """
    distances = np.asarray(distances, dtype="float64")
    if getattr(geom, "geom_type", None) == "LineString":
        return sample_along_line(list(geom.coords), transform, array, distances,
                                 nodata=nodata, fill=fill)

    out = np.full(distances.shape, float(fill), dtype="float64")
    rows, cols = array.shape
    for i, dist in enumerate(distances):
        pt = geom.interpolate(float(dist))
        row, col = xy_to_rc(transform, pt.x, pt.y)
        if 0 <= row < rows and 0 <= col < cols:
            value = float(array[row, col])
            if nodata is not None and np.isfinite(nodata) and value == nodata:
                value = float("nan")
            out[i] = value
    return out


class ContourFeature:
    """A single contour line with its analysis results."""

    def __init__(self, geometry, elevation, rank=None, peak_acc=0.0,
                 mean_slope_deg=0.0, length_m=0.0,
                 cell_area_m2=None, runoff_mm=None):
        """
        Parameters
        ----------
        geometry : shapely LineString
        elevation : float — contour elevation (m)
        rank : int or None — 1 = highest flow crossing
        peak_acc : float — peak flow accumulation crossing this contour
        mean_slope_deg : float — mean terrain slope along the contour
        length_m : float — contour length (m)
        cell_area_m2 : float or None — DEM cell area (m²), used for ha / m³ labels
        runoff_mm : float or None — event runoff depth (mm), used for m³ inflow label
        """
        self.geometry = geometry
        self.elevation = elevation
        self.rank = rank
        self.peak_acc = peak_acc
        self.mean_slope_deg = mean_slope_deg
        self.length_m = length_m
        self.cell_area_m2 = cell_area_m2
        self.runoff_mm = runoff_mm
        self.selected = True   # user can deselect individual contours
        # Position in the candidate list, set by whoever builds that list. The
        # feature's only unique handle: elevation, rank and length can all be
        # shared by two contours. -1 until it is in a list.
        self.index = -1

    @property
    def inflow_m3(self):
        """Event runoff arriving at this contour's peak crossing, or None.

        None rather than 0.0 when the cell size or runoff depth is unknown: those
        are missing inputs, not a contour with no water on it, and a classifier
        told 0.0 would rank an unmeasured contour as the driest on the site.
        """
        if self.cell_area_m2 and self.runoff_mm:
            return self.peak_acc * self.cell_area_m2 * self.runoff_mm / 1000.0
        return None

    @property
    def label(self):
        rank_str = f"#{self.rank} " if self.rank else ""
        if self.cell_area_m2:
            area_ha = self.peak_acc * self.cell_area_m2 / 10_000
            area_str = f"{area_ha:,.1f} ha upslope"
            inflow_m3 = self.inflow_m3
            if inflow_m3 is not None:
                inflow_str = f" — {inflow_m3:,.0f} m³ inflow"
            else:
                inflow_str = ""
            return (
                f"{rank_str}Elev {self.elevation:.1f} m — "
                f"{area_str}{inflow_str}"
            )
        # Fallback when cell size is not known
        return (
            f"{rank_str}Elev {self.elevation:.1f} m — "
            f"peak acc {self.peak_acc:,.0f} cells"
        )


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def extract_contours(dem_path, interval_m=1.0, output_path=None):
    """
    Extract contour lines from a DEM at fixed elevation intervals via GDAL.

    Parameters
    ----------
    dem_path : str — path to DEM GeoTIFF
    interval_m : float — contour interval in metres (default 1.0)
    output_path : str or None — if given, write contours to a GeoPackage

    Returns
    -------
    list of ContourFeature (geometry in DEM CRS, elevation from GDAL output)
    """
    import tempfile

    scratch = None
    if output_path is None:
        # A private directory, not mktemp's bare name: mktemp leaves the window between
        # returning a name and gdal_contour creating it open to anyone. mkstemp is the
        # usual answer and is wrong here — it *creates* the file, and gdal_contour
        # refuses to write to a path that already exists ("A file system object called
        # ... already exists"). Creating the directory instead reserves the name safely
        # and still hands gdal_contour a path it can create.
        #
        # CTA-30: it was also never removed. This runs on every contour run, every
        # keyline run and every segment pick, and one test window left 456 of these
        # directories holding 138.7 MB. ``scratch`` is what records that *we* made
        # it, so the cleanup below can never touch a path the caller named.
        scratch = tempfile.mkdtemp(prefix="tfa_contours_")
        output_path = os.path.join(scratch, "contours.gpkg")

    try:
        return _gdal_contour(dem_path, interval_m, output_path)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)


def _gdal_contour(dem_path, interval_m, output_path):
    """The body of :func:`extract_contours`, with its output path already chosen.

    Split out so the scratch cleanup is one ``finally`` around every exit. There are
    five: the two falls back to the marching-squares path, the two ``RuntimeError``
    raises, and the normal return — and every one of them used to leak.
    """
    import subprocess

    import geopandas as gpd
    import rasterio

    # gdal_contour picks the range of levels it will emit from the band statistics,
    # and it accepts *approximate* ones. QGIS writes exactly those into a PAM
    # sidecar (`<dem>.tif.aux.xml`, STATISTICS_APPROXIMATE=YES) the first time it
    # renders a raster, and an approximate maximum is computed from a decimated
    # sample, so it misses summits: on the 1 m Quail Island DEM it reads 72.4 m
    # against a true 84.8 m, and every contour above 72 m went missing while the
    # ones below it were correct. Turning PAM off makes gdal_contour compute the
    # real min/max from the pixels. The declared nodata is the one thing worth
    # keeping from the sidecar, so read it here — where PAM is still on — and pass
    # it explicitly, or a DEM that declares nodata only there would be contoured
    # straight through its -9999 fill.
    with rasterio.open(dem_path) as src:
        nodata = src.nodata

    cmd = [
        "gdal_contour",
        "--config", "GDAL_PAM_ENABLED", "NO",
        "-a", "ELEV",
        "-i", str(interval_m),
    ]
    if nodata is not None:
        cmd += ["-snodata", repr(float(nodata))]
    cmd += [dem_path, output_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"gdal_contour failed: {result.stderr}")
    except FileNotFoundError:
        # gdal_contour not on PATH — fall back to rasterio + shapely marching squares
        return _extract_contours_scipy(dem_path, interval_m)
    except subprocess.TimeoutExpired:
        # A large DEM at a fine interval can outrun the 120 s budget. That used to
        # propagate as an unhandled exception through TaskWorker and surface as
        # "Contour analysis failed — see the Python console". The marching-squares path
        # is slower but bounded, so take it and say why: the fallback has different
        # edge behaviour (CTA-07) and a silent switch would hide that.
        _log.warning("gdal_contour exceeded its 120 s budget on %s — falling back to "
                     "the marching-squares path, which handles the data boundary "
                     "differently.", dem_path)
        return _extract_contours_scipy(dem_path, interval_m)

    try:
        gdf = gpd.read_file(output_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read contour output: {exc}")

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        elev = float(row.get("ELEV", 0.0) or 0.0)
        length_m = geom.length
        features.append(ContourFeature(
            geometry=geom,
            elevation=elev,
            length_m=length_m,
        ))

    return features


def _extract_contours_scipy(dem_path, interval_m):
    """Fallback contour extraction using scipy.ndimage (no external GDAL)."""
    import rasterio
    from shapely.geometry import LineString
    from skimage import measure

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    valid = dem[np.isfinite(dem)]
    if len(valid) == 0:
        return []

    elev_min = float(np.nanmin(valid))
    elev_max = float(np.nanmax(valid))
    levels = np.arange(
        np.ceil(elev_min / interval_m) * interval_m,
        elev_max,
        interval_m,
    )

    features = []

    for elev in levels:
        # NaN is passed through, not filled. find_contours excludes NaN natively and
        # leaves the contour open where the data stops; substituting elev−1 built a
        # synthetic wall around every hole, so marching squares closed each contour
        # along the data boundary. Those edge-hugging artefacts are indistinguishable
        # from real contours in the picker and are selectable for swale alignment.
        contours_rc = measure.find_contours(dem.astype("float64"), elev)
        for rc_path in contours_rc:
            if len(rc_path) < 2:
                continue
            # Convert row/col → map coordinates (cell centres). Both axes use the
            # signed transform: hard-coding |a| for x only happens to agree with it
            # while a > 0, which is not something the geometry should rely on.
            xs = transform.c + (rc_path[:, 1] + 0.5) * transform.a
            ys = transform.f + (rc_path[:, 0] + 0.5) * transform.e
            coords = list(zip(xs, ys))
            if len(coords) < 2:
                continue
            geom = LineString(coords)
            features.append(ContourFeature(
                geometry=geom,
                elevation=float(elev),
                length_m=geom.length,
            ))

    return features


# ---------------------------------------------------------------------------
# Slope filter
# ---------------------------------------------------------------------------

def filter_by_slope(contours, dem_path, max_slope_deg=18.0, n_samples=20):
    """
    Remove contours where the mean terrain slope exceeds max_slope_deg.

    Samples N evenly-spaced points along each contour, queries the DEM slope
    at those points, and rejects the contour if the mean exceeds the limit.

    Parameters
    ----------
    contours : list of ContourFeature
    dem_path : str — path to DEM GeoTIFF (slope computed on the fly)
    max_slope_deg : float — default 18° (approx 1:3 grade)
    n_samples : int — number of sample points per contour

    Returns
    -------
    list of ContourFeature — only contours with mean slope ≤ max_slope_deg,
    with ``mean_slope_deg`` populated on each feature.
    """
    import rasterio

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        nodata = src.nodata
        cell_w = abs(transform.a)
        cell_h = abs(transform.e)

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # NOTE: this keeps a bespoke masked-array central difference rather than the shared
    # dem_loader.slope_degrees() (Horn's method) on purpose — the two are different
    # published estimators and this one is the stricter of the pair for a filter whose
    # whole job is to reject steep ground. Both now mask nodata rather than filling it.
    dem_masked = np.ma.masked_invalid(dem)
    # np.gradient handles masked arrays from NumPy 1.16+; on anything older, plain NaN
    # propagation gives the same protection (filling with 0.0 would have put the
    # fabricated border spike straight back).
    try:
        dz_dy, dz_dx = np.gradient(dem_masked, cell_h, cell_w)
    except TypeError:
        dz_dy, dz_dx = np.gradient(dem.astype("float64"), cell_h, cell_w)
    # Convert to a plain float32 array with NaN for masked/invalid cells so
    # that indexing never returns a masked scalar (which converts to nan with
    # a warning and then poisons the mean).
    _raw = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    if isinstance(_raw, np.ma.MaskedArray):
        slope_deg = np.ma.filled(_raw, fill_value=np.nan).astype("float32")
    else:
        slope_deg = _raw.astype("float32")

    def _sample_slope(geom):
        """Sample slope at N evenly-spaced points along a line."""
        total_len = geom.length
        if total_len == 0:
            return 0.0
        steps = np.linspace(0, total_len, min(n_samples, max(2, int(total_len / cell_w))))
        values = _sample_line(geom, transform, slope_deg, steps, fill=np.nan)
        values = values[np.isfinite(values)]
        # No valid sample anywhere along the line → NaN, not 0°. Reporting 0° made a
        # contour lying entirely over nodata the *flattest* line on the site and it
        # sailed through a filter whose purpose is to reject unsuitable ground.
        return float(values.mean()) if values.size else float("nan")

    valid = []
    for feat in contours:
        mean_slope = _sample_slope(feat.geometry)
        feat.mean_slope_deg = mean_slope
        if mean_slope <= max_slope_deg:      # NaN compares False — unknown is rejected
            valid.append(feat)

    return valid


# ---------------------------------------------------------------------------
# Flow-crossing ranking
# ---------------------------------------------------------------------------

def rank_by_flow_crossing(contours, acc_path, n_samples=50):
    """
    Rank contours by peak flow accumulation crossing them.

    Samples the flow accumulation raster at N points along each contour and
    records the maximum (peak) value.  Contours are sorted descending by
    peak accumulation and assigned a rank starting at 1.

    Parameters
    ----------
    contours : list of ContourFeature
    acc_path : str — path to flow accumulation GeoTIFF
    n_samples : int — sample points per contour

    Returns
    -------
    list of ContourFeature sorted by peak_acc descending, with rank set.
    """
    import rasterio

    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float32")
        transform = src.transform
        cell_w = abs(transform.a)
        nodata = src.nodata

    if nodata is not None:
        acc = np.where(acc == nodata, 0.0, acc)

    for feat in contours:
        geom = feat.geometry
        total_len = geom.length
        if total_len == 0:
            feat.peak_acc = 0.0
            continue

        steps = np.linspace(0, total_len, min(n_samples, max(2, int(total_len / cell_w))))
        values = _sample_line(geom, transform, acc, steps, fill=np.nan)
        values = values[np.isfinite(values)]
        # Floored at zero because the old running maximum started there: an off-grid or
        # negative sample never lowered the peak, and must not start doing so now.
        feat.peak_acc = max(float(values.max()), 0.0) if values.size else 0.0

    contours.sort(key=lambda f: f.peak_acc, reverse=True)
    for i, feat in enumerate(contours):
        feat.rank = i + 1

    return contours


# ---------------------------------------------------------------------------
# Value-based classification (natural breaks)
# ---------------------------------------------------------------------------

def natural_breaks(values, n_classes=4, max_sample=200):
    """
    Jenks natural-breaks boundaries over *values*.

    Used to band candidate contours (and the inflow gradient) by how much water
    actually arrives on them. Percentile-of-count bands were rejected for this:
    "top 5 / top 10 / rest" says only where a contour sits in a queue, so at the
    handful-of-swales counts this tool produces the bands land in arbitrary places
    and two contours carrying 1,400 m³ and 90 m³ can end up one rank apart looking
    equally good. Natural breaks put the boundaries where the gaps in the values
    are, so a band means "this much water".

    Parameters
    ----------
    values : iterable of float — the quantity to classify (non-finite entries and
        None are dropped)
    n_classes : int — target number of bands (clamped to the number of distinct
        values available)
    max_sample : int — Jenks is O(n²k); above this many values the input is
        evenly subsampled over its sorted order before the boundaries are solved.
        The boundaries move by a hair on huge inputs; the alternative is a
        classification that takes seconds to draw a legend.

    Returns
    -------
    list of float — ``n_classes + 1`` ascending boundaries ``[min, b1, …, max]``,
    i.e. class *i* spans ``breaks[i] … breaks[i + 1]``. Empty list if there is
    nothing finite to classify; ``[v, v]`` for a single distinct value.
    """
    clean = sorted(
        float(v) for v in values
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
    )
    if not clean:
        return []

    distinct = len(set(clean))
    if distinct == 1:
        return [clean[0], clean[0]]

    k = max(1, min(int(n_classes), distinct))
    if k == 1:
        return [clean[0], clean[-1]]

    data = clean
    if len(data) > max_sample:
        last = len(data) - 1
        data = [data[round(i * last / (max_sample - 1))] for i in range(max_sample)]

    n = len(data)
    # mat1[l][j] — first index of the last class in the best j-class split of data[:l]
    # mat2[l][j] — that split's total within-class variance
    mat1 = [[0] * (k + 1) for _ in range(n + 1)]
    mat2 = [[0.0] * (k + 1) for _ in range(n + 1)]
    for j in range(1, k + 1):
        mat1[1][j] = 1
        mat2[1][j] = 0.0
        for i in range(2, n + 1):
            mat2[i][j] = float("inf")

    for l in range(2, n + 1):  # noqa: E741 — l/m/i3/i4 keep the published Jenks names
        s1 = s2 = w = 0.0
        v = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = data[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, k + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v

    breaks = [0.0] * (k + 1)
    breaks[0] = data[0]
    breaks[k] = data[n - 1]
    idx = n
    for count in range(k, 1, -1):
        breaks[count - 1] = data[int(mat1[idx][count]) - 2]
        idx = int(mat1[idx][count]) - 1

    # A degenerate run of identical values can hand back a non-ascending list,
    # which would build a renderer with a negative-width class.
    for i in range(1, k + 1):
        if breaks[i] < breaks[i - 1]:
            breaks[i] = breaks[i - 1]
    return breaks


def class_breaks(values, mode="natural", n_classes=4):
    """Band boundaries for *values* under the user's chosen scale.

    One entry point for every inflow display, so the candidate contours, the
    along-contour gradient and the segment overlay are banded by the same code
    and differ only in where the user asked the boundaries to fall.

    mode:
      "natural"  — Jenks; boundaries land on the gaps in the data (default)
      "log"      — even bands in log10 space; keeps a long tail readable when one
                   crossing dwarfs everything else
      "linear"   — even bands from 0 to the maximum; a true absolute scale, but a
                   single extreme value flattens the rest into one band
      "quantile" — equal count per band; maximum separation, but a band becomes a
                   rank rather than an amount

    Returns the same ``n_classes + 1`` ascending boundary list as
    :func:`natural_breaks`, or ``[]`` when there is nothing finite to band.
    """
    clean = sorted(
        float(v) for v in values
        if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))
    )
    if not clean:
        return []
    if mode == "natural":
        return natural_breaks(clean, n_classes=n_classes)

    lo, hi = clean[0], clean[-1]
    if lo == hi:
        return [lo, lo]
    k = max(1, min(int(n_classes), len(set(clean))))

    if mode == "quantile":
        last = len(clean) - 1
        breaks = [clean[round(i * last / k)] for i in range(k)] + [hi]
    elif mode == "log":
        # +1 keeps a zero-inflow stretch (a real, common value) inside the domain
        # rather than at -inf, which would collapse every other boundary.
        log_lo, log_hi = math.log10(max(lo, 0.0) + 1.0), math.log10(hi + 1.0)
        step = (log_hi - log_lo) / k
        breaks = [10 ** (log_lo + i * step) - 1.0 for i in range(k)] + [hi]
    else:  # linear
        step = (hi - lo) / k
        breaks = [lo + i * step for i in range(k)] + [hi]

    # Equal-count banding repeats a boundary wherever a value is common enough to
    # span one, which would build a zero-width band.
    for i in range(1, len(breaks)):
        if breaks[i] < breaks[i - 1]:
            breaks[i] = breaks[i - 1]
    return breaks


def classify_by_breaks(value, breaks):
    """Index of the band *value* falls in for a :func:`natural_breaks` list.

    Values below the first boundary land in band 0 and values above the last in
    the top band, so a feature classified against breaks computed from a subset
    (or from an earlier run) is still coloured rather than dropped.
    """
    if not breaks or len(breaks) < 2 or value is None:
        return 0
    n_bands = len(breaks) - 1
    for i in range(1, n_bands):
        if value <= breaks[i]:
            return i - 1
    return n_bands - 1


# ---------------------------------------------------------------------------
# Usable area clipping
# ---------------------------------------------------------------------------

class UsableAreaDisjoint(ValueError):
    """The usable area and the contours do not overlap at all.

    Almost always a CRS mismatch rather than a drawing mistake: NZTM eastings are around
    1.5e6 and WGS84 longitudes around 172, so a polygon that never went through the
    reprojection lands an entire hemisphere away. Carries both bounding boxes, because
    the numbers themselves are what say which CRS each side is in.
    """

    def __init__(self, contour_bounds, polygon_bounds):
        self.contour_bounds = contour_bounds
        self.polygon_bounds = polygon_bounds
        super().__init__(
            "The usable area does not overlap the contours at all — check that the "
            "area layer and the DEM share a CRS. "
            f"Contours span {contour_bounds}; the usable area spans {polygon_bounds}."
        )


def clip_to_usable_area(contours, usable_polygon):
    """
    Clip contour lines to a user-defined usable area polygon.

    Contours entirely outside the polygon are dropped.  Contours that cross
    the boundary are trimmed to the interior portion(s) — **all** of them: a contour
    crossing a concave area in two places is two real alignments, and keeping only the
    longer one silently discarded ground the user had selected.

    Parameters
    ----------
    contours : list of ContourFeature
    usable_polygon : shapely Polygon or MultiPolygon

    Returns
    -------
    (list of ContourFeature, dropped_count) — trimmed, geometry replaced with the
    clipped version, rank order preserved. ``dropped_count`` is how many input contours
    fell away entirely, so the caller can say so instead of returning a quiet nothing.

    Raises
    ------
    UsableAreaDisjoint
        When the two do not overlap at all. Every contour clipping away used to be a
        silent ``continue`` per contour, which is correct for one contour outside the
        area and catastrophic for all of them; the distinction is *how many*, so that is
        what is measured.
    """
    from shapely.geometry import LineString, MultiLineString

    if not contours:
        return [], 0

    # Cheap bounds test first: it costs one pass over the features and catches the CRS
    # class outright, before any per-contour intersection work.
    minx = min(f.geometry.bounds[0] for f in contours)
    miny = min(f.geometry.bounds[1] for f in contours)
    maxx = max(f.geometry.bounds[2] for f in contours)
    maxy = max(f.geometry.bounds[3] for f in contours)
    pminx, pminy, pmaxx, pmaxy = usable_polygon.bounds
    if pmaxx < minx or pminx > maxx or pmaxy < miny or pminy > maxy:
        raise UsableAreaDisjoint((minx, miny, maxx, maxy),
                                 (pminx, pminy, pmaxx, pmaxy))

    clipped = []
    dropped = 0
    for feat in contours:
        try:
            intersection = feat.geometry.intersection(usable_polygon)
        except Exception:
            dropped += 1
            continue

        if intersection.is_empty:
            dropped += 1
            continue

        if isinstance(intersection, LineString):
            parts = [intersection]
        elif isinstance(intersection, MultiLineString):
            parts = list(intersection.geoms)
        else:
            # A GeometryCollection can still carry usable lines — a contour that grazes
            # the boundary comes back as lines plus the touching points. Take the lines.
            parts = [g for g in getattr(intersection, "geoms", [])
                     if isinstance(g, LineString)]

        kept_any = False
        for part in parts:
            if part.length < 0.5:  # ignore tiny slivers
                continue
            clipped.append(ContourFeature(
                geometry=part,
                elevation=feat.elevation,
                rank=feat.rank,
                peak_acc=feat.peak_acc,
                mean_slope_deg=feat.mean_slope_deg,
                length_m=part.length,
            ))
            kept_any = True
        if not kept_any:
            dropped += 1

    return clipped, dropped


# ---------------------------------------------------------------------------
# Full pipeline helper
# ---------------------------------------------------------------------------

def analyse_contours(dem_path, acc_path, interval_m=1.0, max_slope_deg=18.0,
                     usable_polygon=None, progress_callback=None,
                     cell_area_m2=None, runoff_mm=None, min_length_m=0.0,
                     on_warning=None):
    """
    Run the full contour analysis pipeline:
      1. Extract contours from DEM
      2. Filter by slope (< max_slope_deg)
      3. Rank by flow crossing
      4. (Optional) clip to usable area

    Parameters
    ----------
    dem_path : str
    acc_path : str — flow accumulation raster path
    interval_m : float — contour interval (m)
    max_slope_deg : float — slope cutoff (degrees)
    usable_polygon : shapely geometry or None
    progress_callback : callable(int, str) or None
    cell_area_m2 : float or None — DEM cell area for ha/m³ labels
    runoff_mm : float or None — event runoff depth for m³ inflow labels
    on_warning : callable(str) or None — non-fatal notices the caller should show.
        The usable-area clip reports through this rather than returning a shorter list
        in silence; a clip that takes everything is a different event from a clip that
        takes some, and both used to look identical from outside.

    Returns
    -------
    list of ContourFeature ranked by flow crossing

    Raises
    ------
    UsableAreaDisjoint
        When a usable area was supplied that does not overlap the contours at all.
    """
    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    def _warn(msg):
        _log.warning(msg)
        if on_warning:
            on_warning(msg)

    _p(5, "Extracting contours...")
    contours = extract_contours(dem_path, interval_m)
    _log.info(f"Extracted {len(contours)} contours at {interval_m} m interval.")

    _p(30, f"Filtering by slope (< {max_slope_deg}°)...")
    contours = filter_by_slope(contours, dem_path, max_slope_deg)
    _log.info(f"{len(contours)} contours remain after slope filter.")

    if min_length_m > 0:
        before = len(contours)
        contours = [f for f in contours if f.geometry.length >= min_length_m]
        _log.info(f"{len(contours)} contours remain after min length filter "
                  f"({before - len(contours)} removed, min={min_length_m} m).")

    _p(60, "Ranking by flow crossing...")
    contours = rank_by_flow_crossing(contours, acc_path)

    if usable_polygon is not None:
        _p(85, "Clipping to usable area...")
        before = len(contours)
        contours, dropped = clip_to_usable_area(contours, usable_polygon)
        _log.info(f"{len(contours)} contours after usable area clip.")
        if before and not contours:
            _warn("The usable area removed every contour. It overlaps the DEM but "
                  "no contour falls inside it — check the area layer covers the "
                  "ground you meant.")
        elif dropped:
            _warn(f"The usable area dropped {dropped} of {before} contours.")

    # Stamp cell size / runoff onto every feature so labels can show ha and m³
    if cell_area_m2 is not None or runoff_mm is not None:
        for feat in contours:
            feat.cell_area_m2 = cell_area_m2
            feat.runoff_mm = runoff_mm

    _p(100, "Contour analysis complete.")
    return contours


# ---------------------------------------------------------------------------
# Sub-contour segment analysis
# ---------------------------------------------------------------------------

class SwaleSegment:
    """
    A recommended swale placement derived from a natural flow-crossing on a
    ranked contour.

    The segment length is sized by :func:`swale_design.recommend_swale_length`,
    which balances the design-storm inflow against trapezoidal trench storage plus
    infiltration over the event — a swale is an infiltration/detention feature, not a
    full-storm reservoir. The segment is centred on the peak accumulation point on the
    contour. If the required length exceeds the available contour, the segment is
    capped to the contour and ``capped`` is set True (the swale alone cannot hold the
    design inflow).

    If no runoff depth is available the natural landscape crossing extent
    (walk until acc drops to drop_fraction × peak) is used instead.
    """

    def __init__(self, geometry, elevation, peak_acc, contributing_ha,
                 inflow_m3, contour_rank, segment_rank, length_m,
                 required_length_m=None, capped=False, segment_slope_deg=None):
        self.geometry = geometry
        self.elevation = elevation
        self.peak_acc = peak_acc
        self.contributing_ha = contributing_ha
        self.inflow_m3 = inflow_m3
        self.contour_rank = contour_rank
        self.segment_rank = segment_rank
        self.length_m = length_m
        # Length required to manage inflow (may exceed the available contour length)
        self.required_length_m = required_length_m if required_length_m is not None else length_m
        # True when the contour was too short to fit required_length_m, False when it
        # fits — and **None** when no required length was ever asked for. The
        # landscape-walk branch (no runoff depth) defines its own extent, so there is
        # nothing for it to fall short of; False there would be a claim about a
        # question never put, the same distinction overtopping draws.
        self.capped = capped
        # Mean ground slope sampled along the segment (deg), or None if not computed.
        self.segment_slope_deg = segment_slope_deg

    @property
    def label(self):
        parts = [
            f"#{self.segment_rank}  Elev {self.elevation:.1f} m",
            f"{self.contributing_ha:,.1f} ha upslope",
        ]
        if self.inflow_m3:
            parts.append(f"{self.inflow_m3:,.0f} m³ inflow")
            if self.capped:
                parts.append(
                    f"⚠ needs {self.required_length_m:.0f} m — contour only "
                    f"{self.length_m:.0f} m"
                )
            else:
                parts.append(f"{self.required_length_m:.0f} m swale required")
        else:
            parts.append(f"{self.length_m:.0f} m long")
        return " — ".join(parts)


def find_swale_segments(contours, acc_path,
                        cell_area_m2, runoff_mm=None,
                        min_acc_ha=0.5, drop_fraction=0.25,
                        # The registry's swale, and the panel's criteria boxes — a
                        # floored trench. These were 0.3/0.6, where 1:1 batters meet at
                        # the drawn depth and the section is a V of 0.09 m².
                        swale_depth_m=0.5, swale_width_m=2.0,
                        max_segments_per_contour=3,
                        side_slope=1.0, infiltration_mm_hr=0.0, duration_hr=0.0,
                        rank_mode="catchment", slope_path=None,
                        seg_max_slope_deg=None, progress_callback=None):
    """
    For each ranked contour, locate natural flow-crossing zones and size a swale
    segment to *manage* the incoming runoff over the design storm.

    Algorithm
    ---------
    1. Sample the flow accumulation raster at every cell-width along the contour.
    2. Find local peaks with contributing area ≥ *min_acc_ha* — these are where
       drainage lines cross the contour.
    3. Calculate inflow volume:  inflow_m3 = peak_acc × cell_area × runoff_mm
    4. Size the required swale length with
       :func:`swale_design.recommend_swale_length` — trapezoidal trench storage plus
       infiltration over the event, not the old "store the whole storm in a
       rectangular trench" model.
    5. Extract a segment of that length centred on the peak point. If the contour is
       shorter than the required length the segment is capped to the contour and
       ``SwaleSegment.capped`` is set True. If runoff_mm is unavailable, fall back to
       the landscape-walk extent (walk outward until acc drops below
       drop_fraction × peak).

    Parameters
    ----------
    contours         : list of ContourFeature (already ranked)
    acc_path         : str — flow accumulation GeoTIFF
    cell_area_m2     : float — DEM cell area (m²)
    runoff_mm        : float or None — event runoff depth (mm)
    min_acc_ha       : float — minimum contributing area (ha) for a crossing to
                       qualify.  Default 0.5 ha.
    drop_fraction    : float — fallback landscape walk: ends where acc < fraction × peak
    swale_depth_m    : float — swale design depth (m), default 0.5 m
    swale_width_m    : float — swale top width (m), default 2.0 m
    side_slope       : float — wall batter H:V run-per-rise (default 1.0). Together
                       the three give a 1.0 m bottom width: a swale is dug with a
                       floor, and a combination that leaves none describes a V-drain
                       holding a fraction of the intended section.
    infiltration_mm_hr : float — soil infiltration rate (mm/hr); 0 → storage only
    duration_hr      : float — storm duration (hr); infiltration counts over the event
    max_segments_per_contour : int — max crossings extracted per contour
    progress_callback : callable(int, str) or None

    Returns
    -------
    list of SwaleSegment, ranked globally by inflow_m3 descending
    (or contributing_ha if runoff_mm is not provided).
    """
    import rasterio
    from shapely.ops import substring

    from .swale_design import recommend_swale_length

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # rank_mode "inflow": relax the catchment filter to a 1-cell floor so small
    # holdings (where no crossing reaches min_acc_ha) still surface their largest
    # inflow points. "catchment": keep the min_acc_ha threshold.
    if rank_mode == "inflow":
        min_acc_cells = 1
    else:
        min_acc_cells = max(1, int(min_acc_ha * 10_000 / cell_area_m2))

    _p(5, "Loading accumulation raster…")
    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float32")
        transform = src.transform
        cell_w = abs(transform.a)
        nodata = src.nodata
    if nodata is not None:
        acc = np.where(acc == nodata, 0.0, acc)

    # Optional per-segment slope filter (F7): sample a slope raster along each
    # candidate segment and drop segments whose mean slope exceeds the limit.
    slope_arr = None
    slope_nodata = None
    if slope_path and seg_max_slope_deg is not None:
        try:
            with rasterio.open(slope_path) as ss:
                slope_arr = ss.read(1).astype("float32")
                slope_nodata = ss.nodata
        except Exception:
            slope_arr = None
            slope_nodata = None

    def _mean_segment_slope(seg_geom):
        """Mean slope under *seg_geom* — NaN where the ground is unknown.

        ``None`` and NaN mean different things here and the filter reads both: ``None``
        is "no slope raster was supplied, do not filter", NaN is "this segment lies over
        ground we have no slope for, reject it".

        The nodata read is the whole point. ``compute_slope_raster`` declares ``-9999``
        *and writes it*, so sampling raw averaged a segment over a hole toward minus ten
        thousand degrees — not merely passing the steepness filter but sorting as the
        flattest ground on the site. This is CTA-13 (``filter_by_slope`` above, which
        returns NaN for exactly this case) re-created five hundred lines further down;
        the two must not diverge a third time.
        """
        if slope_arr is None:
            return None
        n_s = max(2, int(seg_geom.length / max(cell_w, 1.0)))
        dists = np.linspace(0, seg_geom.length, n_s)
        vals = _sample_line(seg_geom, transform, slope_arr, dists,
                            fill=np.nan, nodata=slope_nodata)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    all_segments = []
    n = len(contours)

    for ci, feat in enumerate(contours):
        _p(5 + int(90 * ci / max(n, 1)), f"Analysing contour {ci + 1}/{n}…")
        geom = feat.geometry
        total_len = geom.length
        if total_len < 1.0:
            continue

        # Sample every cell_w along the contour
        step = max(cell_w, 1.0)
        n_steps = max(2, int(total_len / step))
        dists = np.linspace(0, total_len, n_steps)

        # Off-grid reads as zero accumulation, not NaN: this profile is indexed by
        # position, so every sample has to keep its slot.
        values = _sample_line(geom, transform, acc, dists, fill=0.0)
        profile = [(float(d), float(v)) for d, v in zip(dists, values)]

        if not profile:
            continue

        # --- Find local peaks (stream crossings) ---
        found_peaks = []  # (index_in_profile, peak_acc)
        i = 0
        while i < len(profile):
            d_i, v_i = profile[i]
            if v_i < min_acc_cells:
                i += 1
                continue

            # Is this a local maximum? Check neighbours within ±5 samples
            lo = max(0, i - 5)
            hi = min(len(profile), i + 6)
            neighbourhood = [v for _, v in profile[lo:hi]]
            if v_i < max(neighbourhood):
                i += 1
                continue

            found_peaks.append((i, v_i))

            # Advance past this peak by enough to avoid duplicates (10 samples)
            i += 10

        if not found_peaks:
            continue

        found_peaks.sort(key=lambda x: x[1], reverse=True)
        found_peaks = found_peaks[:max_segments_per_contour]

        for peak_idx, peak_acc_val in found_peaks:
            peak_dist = profile[peak_idx][0]
            contrib_ha = peak_acc_val * cell_area_m2 / 10_000
            inflow_m3 = (peak_acc_val * cell_area_m2 * runoff_mm / 1000.0
                         if runoff_mm else 0.0)

            capped = False
            if runoff_mm and swale_depth_m > 0 and swale_width_m > 0:
                # Size to manage the inflow via trapezoidal storage + infiltration
                # (a swale is not a full-storm reservoir).
                required_length = recommend_swale_length(
                    inflow_m3, swale_depth_m, swale_width_m,
                    side_slope=side_slope,
                    infiltration_mm_hr=infiltration_mm_hr,
                    duration_hr=duration_hr,
                )
                required_length = max(required_length, 1.0)  # at least 1 m

                # Center the segment on the peak, capped to contour extent
                half = required_length / 2.0
                seg_start = max(0.0, peak_dist - half)
                seg_end = min(total_len, peak_dist + half)

                # If capped at one end, extend the other to preserve total length
                actual_len = seg_end - seg_start
                if actual_len < required_length:
                    shortfall = required_length - actual_len
                    if seg_start == 0.0:
                        seg_end = min(total_len, seg_end + shortfall)
                    else:
                        seg_start = max(0.0, seg_start - shortfall)

                # Contour too short to fit the required swale length.
                capped = (seg_end - seg_start) < (required_length - 0.5)
            else:
                # Fallback: landscape-walk extent
                threshold = peak_acc_val * drop_fraction
                left = peak_idx
                while left > 0 and profile[left - 1][1] >= threshold:
                    left -= 1
                right = peak_idx
                while right < len(profile) - 1 and profile[right + 1][1] >= threshold:
                    right += 1
                seg_start = profile[left][0]
                seg_end = profile[right][0]
                required_length = seg_end - seg_start
                # "Not capped" would be a claim about a question never asked. This
                # branch has no *required* length to fall short of — the walk defines
                # its own extent, so required_length is the extent by construction.
                # None is "not asked", the same distinction overtopping draws between
                # False and None; the label prints nothing for it.
                capped = None

            if seg_end - seg_start < 1.0:
                continue  # degenerate — skip

            try:
                seg_geom = substring(geom, seg_start, seg_end)
            except Exception:
                continue
            if seg_geom is None or seg_geom.is_empty:
                continue

            # F7 — drop segments crossing ground steeper than the segment limit, and
            # segments over ground whose slope is unknown. The rejection is written
            # out rather than left to comparison polarity: NaN > limit is False, so
            # "unknown" would otherwise sail through a filter built to reject
            # unsuitable ground. Same decision as filter_by_slope's.
            seg_slope = _mean_segment_slope(seg_geom)
            if seg_max_slope_deg is not None and seg_slope is not None:
                if not np.isfinite(seg_slope) or seg_slope > seg_max_slope_deg:
                    continue

            seg_len = seg_geom.length

            all_segments.append(SwaleSegment(
                geometry=seg_geom,
                elevation=feat.elevation,
                peak_acc=peak_acc_val,
                contributing_ha=round(contrib_ha, 1),
                inflow_m3=round(inflow_m3, 0),
                contour_rank=feat.rank or 0,
                segment_rank=0,
                length_m=round(seg_len, 0),
                required_length_m=round(required_length, 0),
                capped=capped,
                segment_slope_deg=round(seg_slope, 1) if seg_slope is not None else None,
            ))

    all_segments.sort(
        key=lambda s: s.inflow_m3 if runoff_mm else s.contributing_ha,
        reverse=True,
    )
    for i, seg in enumerate(all_segments):
        seg.segment_rank = i + 1

    _p(100, f"Found {len(all_segments)} candidate swale segment(s).")
    return all_segments


def classify_contour_inflow(contours, acc_path, cell_area_m2, runoff_mm=None,
                            duration_hr=0.0, window=3, progress_callback=None,
                            source_ids=None):
    """
    Split each contour into short sub-segments carrying the **absolute** inflow
    that drains onto them, for a continuous, map-wide comparable gradient.

    Samples flow accumulation along every contour and emits a coloured stretch
    every ``window`` samples, each stamped with the runoff volume (m³ over the
    event) and rate (L/s) arriving on it. Because every stretch reports an
    absolute value (not a per-contour share), a downstream renderer can colour
    them on one continuous ramp keyed to the global maximum — so a stretch on one
    contour is directly comparable to a stretch on any other.

    Parameters
    ----------
    contours : list — anything carrying ``.geometry`` (a shapely LineString) and
        ``.elevation``; ``.rank`` is used when present. ContourFeature and
        SwaleSegment both qualify, so the recommended segments can be graded with
        the same routine that grades the whole contours.
    acc_path : str — flow accumulation GeoTIFF
    cell_area_m2 : float
    runoff_mm : float or None — event runoff depth; needed to convert accumulation
        to m³/L·s (falls back to raw accumulation cell-count if None)
    duration_hr : float — storm duration (for the L/s rate)
    window : int — samples per emitted stretch (larger = coarser/smoother)
    progress_callback : callable(int, str) or None
    source_ids : list of int or None — an id per input feature, stamped onto every
        stretch it produced as ``source_id``. Lets a caller hide the stretches of a
        contour the user unticked without reclassifying the whole site. Defaults to
        the feature's position in *contours*.

    Returns
    -------
    list of dict: {geometry (sub-LineString), inflow_m3, flow_ls, acc,
    contour_rank, elevation, source_id}. Also carries ``global_max_m3`` on every
    dict so the renderer knows the shared upper bound.
    """
    import rasterio
    from shapely.ops import substring

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _p(5, "Loading accumulation raster…")
    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype("float32")
        transform = src.transform
        cell_w = abs(transform.a)
        nodata = src.nodata
    if nodata is not None:
        acc = np.where(acc == nodata, 0.0, acc)

    runoff_m = (runoff_mm / 1000.0) if runoff_mm else None
    duration_s = duration_hr * 3600.0

    results = []
    n = len(contours)
    for ci, feat in enumerate(contours):
        _p(5 + int(90 * ci / max(n, 1)), f"Classifying contour {ci + 1}/{n}…")
        source_id = source_ids[ci] if source_ids is not None and ci < len(source_ids) else ci
        geom = feat.geometry
        total_len = geom.length
        if total_len < 1.0:
            continue

        step = max(cell_w, 1.0)
        n_steps = max(3, int(total_len / step))
        dists = np.linspace(0.0, total_len, n_steps)

        # Off-grid reads as zero accumulation, not NaN: the window means below are
        # taken by slice, so every sample has to keep its slot.
        vals = _sample_line(geom, transform, acc, dists, fill=0.0)

        if vals.max() <= 0:
            continue

        # Emit a stretch every `window` samples with that window's mean flow.
        for k in range(0, len(dists) - 1, window):
            k2 = min(k + window, len(dists) - 1)
            d0, d1 = float(dists[k]), float(dists[k2])
            if d1 - d0 < step:
                continue
            acc_val = float(np.mean(vals[k:k2 + 1]))
            try:
                sub = substring(geom, d0, d1)
            except Exception:
                sub = None
            if sub is None or sub.is_empty:
                continue
            if runoff_m is not None:
                inflow_m3 = acc_val * cell_area_m2 * runoff_m
                flow_ls = (inflow_m3 * 1000.0 / duration_s) if duration_s > 0 else 0.0
            else:
                inflow_m3 = acc_val  # fall back to raw accumulation
                flow_ls = 0.0
            results.append({
                "geometry": sub,
                "acc": round(acc_val, 1),
                "inflow_m3": round(inflow_m3, 1),
                "flow_ls": round(flow_ls, 2),
                "contour_rank": getattr(feat, "rank", 0) or 0,
                "elevation": feat.elevation,
                "source_id": source_id,
            })

    # Stamp the shared global maximum so the renderer can scale one ramp to it.
    global_max = max((r["inflow_m3"] for r in results), default=0.0)
    for r in results:
        r["global_max_m3"] = global_max

    _p(100, f"Classified {len(results)} contour stretch(es).")
    return results
