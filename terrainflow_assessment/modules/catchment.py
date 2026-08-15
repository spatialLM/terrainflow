"""
catchment.py — Contributing catchment delineation and SCS runoff model.

Combines logic from the base plugin's contributing_area.py and scs_runoff.py
into a single module.  Exposes:

  fast_contributing_area()   — uphill BFS from site boundary
  clip_dem_to_polygon()      — clip DEM to catchment + buffer
  SCSRunoff                  — Curve Number rainfall-runoff model
"""

import logging

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.features import shapes as rasterio_shapes
from rasterio.transform import Affine
from scipy.ndimage import zoom as _zoom

_log = logging.getLogger(__name__)

_MAX_PREVIEW_CELLS = 500_000   # ~700 × 700 — keep preview under ~5 s


# ---------------------------------------------------------------------------
# Contributing area (uphill BFS)
# ---------------------------------------------------------------------------

def fast_contributing_area(dem_path, boundary_path, progress_callback=None):
    """
    Delineate the contributing catchment from a site boundary by reverse traversal
    of the flow-direction network via pysheds.

    Routing is **D-infinity** where pysheds supports it, falling back to its default
    (D8) only when the installed version rejects the keyword — dinf avoids the
    ``np.in1d`` call NumPy 2.x removed. Both are traversed the same way.

    Walks the flow-direction network *backwards* from the highest-accumulation
    pour point on the site boundary — the faithful "find all cells that drain into
    the site" operation.  The previous BFS implementation incorrectly crossed
    saddles and included knolls whose water flows away from the site.

    Parameters
    ----------
    dem_path : str
        Path to the full-resolution DEM GeoTIFF.
    boundary_path : str
        Path to a polygon vector file (site boundary, in any CRS).
    progress_callback : callable(int, str) or None

    Returns
    -------
    dict
        catchment_polygon : shapely Polygon
        clip_polygon      : shapely Polygon  (catchment bbox + 20 % buffer)
        area_ha           : float
        dem_area_ha       : float
        coverage_pct      : float
        scale             : float
    """
    import os
    import tempfile

    import geopandas as gpd
    from pysheds.grid import Grid
    from shapely.geometry import box
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    def _p(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _p(5, "Reading DEM...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    rows, cols = dem.shape
    n_cells = rows * cols
    cell_w = abs(transform.a)
    cell_h = abs(transform.e)
    dem_area_ha = (rows * cell_h) * (cols * cell_w) / 10_000

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    scale = min(1.0, (_MAX_PREVIEW_CELLS / n_cells) ** 0.5)
    if scale < 1.0:
        _p(10, f"Downsampling DEM to {scale * 100:.0f}% for preview...")
        work_dem = _zoom(dem, scale, order=1).astype("float32")
        work_transform = Affine(
            transform.a / scale, transform.b, transform.c,
            transform.d, transform.e / scale, transform.f,
        )
    else:
        work_dem = dem
        work_transform = transform

    nrows, ncols = work_dem.shape

    _p(20, "Rasterizing site boundary...")
    gdf = gpd.read_file(boundary_path)
    gdf = gdf.to_crs(crs.to_wkt())
    gdf_dissolved = gdf.dissolve()

    inside_mask = rasterize(
        [(geom, 1) for geom in gdf_dissolved.geometry],
        out_shape=work_dem.shape,
        transform=work_transform,
        fill=0, all_touched=True, dtype="uint8",
    ).astype(bool)

    if not inside_mask.any():
        raise RuntimeError(
            "Site boundary does not intersect the DEM extent. "
            "Check that both layers use the same CRS."
        )

    boundary_lines = [
        geom.exterior
        for geom in gdf_dissolved.geometry
        if geom is not None and hasattr(geom, "exterior")
    ]
    if not boundary_lines:
        raise RuntimeError("Site boundary has no valid polygon geometry.")

    boundary_line_mask = rasterize(
        [(line, 1) for line in boundary_lines],
        out_shape=work_dem.shape,
        transform=work_transform,
        fill=0, dtype="uint8",
    ).astype(bool)

    seed_rows, seed_cols = np.where(boundary_line_mask)
    if seed_rows.size == 0:
        raise RuntimeError(
            "Site boundary line did not intersect the DEM. "
            "Check that both layers use the same CRS."
        )

    _p(35, "Computing flow direction for catchment delineation...")

    # Write working DEM to a temp file for pysheds
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(tmp_fd)
    try:
        work_nodata = -9999.0
        with rasterio.open(
            tmp_path, "w", driver="GTiff", dtype="float32",
            crs=crs, transform=work_transform,
            width=ncols, height=nrows, count=1, nodata=work_nodata,
        ) as dst:
            out_arr = np.where(np.isnan(work_dem), work_nodata, work_dem)
            dst.write(out_arr.astype("float32"), 1)

        grid = Grid.from_raster(tmp_path)
        dem_r = grid.read_raster(tmp_path)
        pit_filled = grid.fill_pits(dem_r)
        # A fill, not a breach: ``breach_depressions`` does not exist in pysheds 0.5, so
        # the except branch is the one that has always run (Round 14).
        try:
            filled = grid.breach_depressions(pit_filled)
        except AttributeError:
            filled = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(filled)
        # Use dinf routing — avoids np.in1d which was removed in NumPy 2.x
        try:
            fdir = grid.flowdir(inflated, routing="dinf")
            _routing = "dinf"
        except TypeError:
            fdir = grid.flowdir(inflated)
            _routing = None
        try:
            acc = grid.accumulation(fdir, routing=_routing) if _routing else grid.accumulation(fdir)
        except (TypeError, AttributeError):
            acc = grid.accumulation(fdir)

        acc_arr = np.array(acc, dtype="float32")

        _p(50, "Finding pour point on boundary...")
        bnd_rows, bnd_cols = np.where(boundary_line_mask)
        bnd_accs = acc_arr[bnd_rows, bnd_cols]
        best_idx = int(np.argmax(bnd_accs))
        pour_r = int(bnd_rows[best_idx])
        pour_c = int(bnd_cols[best_idx])

        # Convert raster cell centre to map coordinates
        px = work_transform.c + (pour_c + 0.5) * work_transform.a
        py = work_transform.f + (pour_r + 0.5) * work_transform.e  # e is negative

        _p(65, "Delineating catchment via reverse traversal...")
        try:
            catch_mask = grid.catchment(
                x=px, y=py, fdir=fdir,
                xytype="coordinate", routing=_routing,
            ) if _routing else grid.catchment(x=px, y=py, fdir=fdir, xytype="coordinate")
        except (TypeError, AttributeError):
            catch_mask = grid.catchment(x=px, y=py, fdir=fdir, xytype="coordinate")

        contributing = np.array(catch_mask, dtype=bool)
        contributing |= inside_mask  # always include the declared site

    finally:
        os.unlink(tmp_path)

    _p(85, "Building catchment polygon...")
    polys = [
        shapely_shape(geom)
        for geom, val in rasterio_shapes(
            contributing.astype("uint8"), transform=work_transform
        )
        if val == 1
    ]
    if not polys:
        raise RuntimeError("Catchment raster produced no polygon geometry.")

    catchment_poly = unary_union(polys)
    area_ha = catchment_poly.area / 10_000

    minx, miny, maxx, maxy = catchment_poly.bounds
    buf_x = (maxx - minx) * 0.20
    buf_y = (maxy - miny) * 0.20
    clip_poly = box(minx - buf_x, miny - buf_y, maxx + buf_x, maxy + buf_y)

    coverage_pct = (area_ha / dem_area_ha * 100.0) if dem_area_ha > 0 else 0.0

    _p(100, "Done.")
    return {
        "catchment_polygon": catchment_poly,
        "clip_polygon": clip_poly,
        "area_ha": round(area_ha, 1),
        "dem_area_ha": round(dem_area_ha, 1),
        "coverage_pct": round(coverage_pct, 1),
        "scale": scale,
    }


def clip_dem_to_polygon(dem_path, clip_polygon, output_path):
    """Clip DEM to a shapely polygon (in the DEM's CRS) and save."""
    import rasterio.mask
    from shapely.geometry import mapping

    with rasterio.open(dem_path) as src:
        out_image, out_transform = rasterio.mask.mask(
            src, [mapping(clip_polygon)], crop=True,
            nodata=src.nodata if src.nodata is not None else -9999,
        )
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "nodata": src.nodata if src.nodata is not None else -9999,
        })

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    return output_path


# ---------------------------------------------------------------------------
# SCS Curve Number rainfall-runoff model
# ---------------------------------------------------------------------------

# Hydrologic condition of the pasture, as TR-55 Table 2-2 defines it for the
# "Pasture, grassland, or range" row. The condition is a statement about ground cover
# and grazing pressure, and it moves the curve number more than the soil texture does:
# on sand, Good is CN 39 and Poor is 68.
#
# It was previously assumed to be Good and never asked, which is the optimistic end of
# the range and reads *less* runoff than the ground actually sheds — the under-sizing
# direction on hard-grazed country. Ordered least to most runoff, which is the order
# the picker shows.
#
#   key: (label, what it means on the ground)
GROUND_CONDITIONS = {
    "good": ("Good — dense cover, lightly grazed",
             "More than 75% ground cover; grazed lightly or not at all."),
    "fair": ("Fair — patchy cover, moderately grazed",
             "50–75% ground cover; grazed but not hard."),
    "poor": ("Poor — thin cover, heavily grazed",
             "Under 50% ground cover, or heavily grazed, or bare and compacted."),
}

# Good reproduces every curve number this plugin used before the condition was asked,
# so an existing design reopens unchanged.
DEFAULT_GROUND_CONDITION = "good"


# ---------------------------------------------------------------------------
# Rational-method runoff coefficients (Brad Lancaster)
# ---------------------------------------------------------------------------

# Runoff coefficients from Brad Lancaster, *Rainwater Harvesting for Drylands and
# Beyond* — https://www.harvestingrainwater.com/resource/water-harvesting-calculations/
#
#     runoff volume = catchment area × rainfall depth × runoff coefficient
#
# One empirical fraction per surface, in place of the SCS-CN storage model. Simpler,
# far more transparent, and what water-harvesting practitioners actually use in the
# field — which matters, because a designer can sanity-check a coefficient against
# ground they are standing on in a way they cannot check a curve number.
#
# Values are the midpoint of Lancaster's published typical range, with his full range
# kept alongside so the UI can show the spread rather than implying false precision.
# His own ranges span 3–7× for a single surface; that uncertainty is the honest state
# of the art, not a defect of the table.
LANCASTER_COEFFICIENTS = {
    #  label                              (typical, low, high, source)
    "Pasture / grass, wet ground": (0.50, 0.35, 0.70, "design"),
    "Grass / lawn":                (0.18, 0.05, 0.35, "lancaster"),
    "Healthy indigenous landscape": (0.40, 0.20, 0.70, "lancaster"),
    "Bare earth":                  (0.45, 0.20, 0.75, "lancaster"),
    "Concrete / asphalt":          (0.88, 0.80, 0.95, "lancaster"),
    "Metal roof":                  (0.95, 0.95, 0.95, "lancaster"),
}

# The default: pasture on ground already wet when the design storm arrives. Above
# Lancaster's grass figure because the storm that breaks an earthwork is usually the
# second one, and below the "every millimetre runs off" assumption, which exceeds even
# his metal-roof coefficient and no established method uses for a landscape catchment.
DEFAULT_RUNOFF_COEFFICIENT = 0.50


def coefficient_runoff_depth(rainfall_mm, coefficient):
    """Runoff depth (mm) by the rational method: ``P × C`` (Lancaster).

    The coefficient is the fraction of rainfall that leaves the catchment as surface
    flow; everything else is intercepted, ponded in surface hollows or soaked in where
    it fell. Clamped to 0–1 — a catchment cannot shed more than it receives.
    """
    if rainfall_mm is None or rainfall_mm <= 0:
        return 0.0
    return float(rainfall_mm) * max(0.0, min(1.0, float(coefficient)))


def scs_marginal_runoff_fraction(rainfall_mm, cn):
    """Fraction of the *next* millimetre of rain that runs off — ``dQ/dP``.

    Peak flow needs a different quantity from event volume. The rational method's
    coefficient is an **instantaneous** fraction: what proportion of rain falling at
    the moment of peak intensity becomes flow. The SCS curve number instead gives
    *cumulative* runoff from *cumulative* rainfall, and its event-average ratio
    ``Q/P`` badly understates the instantaneous value, because the initial
    abstraction is paid off early and the ground sheds progressively more as the
    storm proceeds.

    Differentiating ``Q = u² / (u + S)`` with ``u = P − Ia`` (and ``Ia = 0.2S``, so
    ``du/dP = 1``) gives

        dQ/dP = u (u + 2S) / (u + S)²

    which is 0 at ``P = Ia``, rises monotonically, approaches 1 for very large
    storms, and always exceeds ``Q/P``. On CN 61 under a 120 mm storm it is 0.578
    against an event average of 0.255 — using the average would undersize a spillway
    by more than half.

    .. warning::
       Evaluated at the rainfall depth passed in. Supplying the full storm total
       assumes the peak burst arrives at the *end* of the event, on the wettest
       ground, which is the conservative corner rather than a neutral one: the same
       CN 61 storm gives 0.269 half way through and 0.578 at the close. Deliberate
       for spillway sizing, where the cost of being low is a breached embankment.
    """
    if cn is None or cn <= 0 or rainfall_mm is None:
        return 0.0
    s = (25400.0 / cn) - 254.0
    u = float(rainfall_mm) - 0.2 * s
    if u <= 0:
        return 0.0
    return max(0.0, min(1.0, u * (u + 2.0 * s) / ((u + s) ** 2)))


def drain_down_hours(stored_m3, area_m2, infiltration_mm_hr):
    """Hours for standing water to soak away, or ``None`` if it never does.

    Lancaster sizes water-harvesting earthworks so they "work, don't flood, and don't
    puddle" — the third constraint being one this model otherwise ignores entirely.
    Water that stands too long breeds mosquitoes, drowns the plantings the earthwork
    exists to support (most species fail after 2–3 days waterlogged), and leaves no
    freeboard for the next storm. Conventional infiltration-basin practice is complete
    drawdown inside 24–48 hours.

    Uses the soil's infiltration rate over the feature's wetted area regardless of
    whether soakage is being credited as capture: that is a sizing policy, whereas this
    is what physically happens once the rain stops.
    """
    if stored_m3 is None or stored_m3 <= 0:
        return 0.0
    rate_m_hr = (infiltration_mm_hr or 0.0) / 1000.0
    if rate_m_hr <= 0 or not area_m2 or area_m2 <= 0:
        return None            # nothing draining it — standing water, indefinitely
    return stored_m3 / (rate_m_hr * area_m2)


def _looks_cumulative(rain_vals):
    """Is this rainfall column a running total, or per-interval depths?

    Both forms are common in the wild, neither is labelled, so the reader has to
    guess. "Non-decreasing ⇒ cumulative" is the obvious rule and it fails on the most
    ordinary file there is: a constant-rate storm (5, 5, 5, 5 mm) is non-decreasing,
    and reading it as a running total throws away 15 of its 20 mm — the simulation
    then runs on a quarter of the design rainfall.

    The test that separates them is whether the trace *accumulates across the record*.
    Read cumulatively, ``5, 5, 5, 5`` says all 5 mm fell before the first reading and
    nothing after — not a storm. So a running total must be non-decreasing **and**
    gain more than half its final value between the first and last reading. That keeps
    genuine cumulative traces, including ones whose record starts a step or two into
    the storm, and rejects the flat and near-flat incremental ones.

    A hyetograph that rises monotonically to its last interval stays ambiguous; it is
    read as incremental, which over-states rainfall rather than under-stating it.
    """
    if len(rain_vals) < 2:
        return False
    non_decreasing = all(
        rain_vals[i] >= rain_vals[i - 1] - 1e-6 for i in range(1, len(rain_vals))
    )
    if not non_decreasing:
        return False
    first, last = rain_vals[0], rain_vals[-1]
    if last <= 0:
        return False
    return (last - first) > 0.5 * last


class SCSRunoff:
    """
    SCS Curve Number rainfall-runoff model.

    Calculates the depth of surface runoff from rainfall based on the soil's
    Curve Number (CN) and antecedent moisture condition (AMC).

    Reference: USDA-NRCS National Engineering Handbook, Part 630.
    """

    _AMC = {
        "dry":    lambda cn: 4.2 * cn / (10 - 0.058 * cn),
        "normal": lambda cn: cn,
        "wet":    lambda cn: 23 * cn / (10 + 0.13 * cn),
    }

    # Typical CN values by soil type for the soil selector dropdown.
    #
    # PROVENANCE: TR-55 Table 2-2, row "Pasture, grassland, or range — continuous
    # forage for grazing", hydrologic condition **Good**, across hydrologic soil
    # groups A–D (39 / 61 / 74 / 80); texture stands in for the soil group, and 49
    # interpolates the Fair/A value for sandy loam.
    #
    # That bakes in an assumption the picker does not show: Good condition means
    # >75% ground cover and light grazing. The same soils in Poor condition (<50%
    # cover, heavy grazing) are CN 68 / 79 / 86 / 89 — so on degraded or hard-grazed
    # ground this table under-reads runoff, by 29 CN at the sandy end. Choose a
    # condition deliberately with :meth:`soil_reference_cn` rather than accepting
    # Good by default on country that has not earned it.
    SOIL_REFERENCE = {
        "Sand":        39,
        "Sandy loam":  49,
        "Loam":        61,
        "Clay loam":   74,
        "Clay":        80,
    }

    # TR-55 Table 2-2, the same pasture row at its other two hydrologic conditions.
    # Fair = 50–75% cover, not heavily grazed; Poor = <50% cover, heavily grazed.
    _SOIL_REFERENCE_BY_CONDITION = {
        "good": SOIL_REFERENCE,
        "fair": {"Sand": 49, "Sandy loam": 59, "Loam": 69,
                 "Clay loam": 79, "Clay": 84},
        "poor": {"Sand": 68, "Sandy loam": 74, "Loam": 79,
                 "Clay loam": 86, "Clay": 89},
    }

    @classmethod
    def soil_reference_cn(cls, soil_name, condition=DEFAULT_GROUND_CONDITION):
        """Reference CN for a soil texture at a stated pasture hydrologic condition.

        ``condition`` is a key of :data:`GROUND_CONDITIONS`. Unknown soils and unknown
        conditions fall back to Loam and Good respectively, so a hand-edited design
        file degrades to the historical answer rather than to nonsense.
        """
        table = cls._SOIL_REFERENCE_BY_CONDITION.get(
            str(condition).lower(), cls.SOIL_REFERENCE)
        return table.get(soil_name, table["Loam"])

    STORM_PRESETS = {
        "Custom":                (None, None),
        "Light (~25 mm/hr)":     (25, 1),
        "Moderate (~40 mm/hr)":  (40, 1),
        "Heavy (~65 mm/hr)":     (65, 1),
        "Extreme (~80 mm/hr)":   (80, 1),
    }

    def adjust_cn(self, cn, moisture_condition):
        """Adjust CN for antecedent moisture condition. Returns CN in [1, 100]."""
        adjusted = self._AMC[moisture_condition](cn)
        return max(1.0, min(100.0, adjusted))

    def adjust_cn_array(self, cn, moisture_condition):
        """:meth:`adjust_cn` over a whole array, in one numpy expression.

        The lambdas in ``_AMC`` are already array-safe — they are arithmetic — so
        the only per-cell work was the clamp, and ``np.vectorize`` is a Python loop
        with a numpy signature on it. Asserted equal to the scalar form by test.
        """
        adjusted = self._AMC[moisture_condition](np.asarray(cn, dtype="float64"))
        return np.clip(adjusted, 1.0, 100.0)

    @staticmethod
    def runoff_depth_array(rainfall_mm, cn):
        """:meth:`runoff_depth` over a whole array, closed form.

        ``S = 25400/CN − 254; Ia = 0.2S; Q = (P−Ia)²/(P−Ia+S)`` where ``P > Ia``,
        and zero elsewhere — including where CN is zero or less, which the scalar
        form short-circuits. Asserted equal to it by test.

        This was ``np.vectorize`` per cell per timestep: about 84 million Python
        calls for a 2.8 M-cell site over 30 steps.
        """
        cn_arr = np.asarray(cn, dtype="float64")
        valid = cn_arr > 0
        # Guard the division rather than the result: 25400/0 warns and yields inf,
        # and inf - 254 then propagates into arithmetic nothing masks afterwards.
        s = np.where(valid, 25400.0 / np.where(valid, cn_arr, 1.0) - 254.0, 0.0)
        ia = 0.2 * s
        excess = np.asarray(rainfall_mm, dtype="float64") - ia
        wet = excess > 0
        # np.where evaluates both branches, so the divisor is guarded too: with
        # CN and rainfall both zero it is 0/0, which is a NaN and a warning even
        # though the branch is never selected.
        denominator = np.where(wet, excess + s, 1.0)
        runoff = np.where(wet, excess ** 2 / denominator, 0.0)
        return np.where(valid, np.maximum(runoff, 0.0), 0.0)

    def runoff_depth(self, rainfall_mm, cn):
        """
        Calculate runoff depth (Q) from rainfall (P) and curve number (CN).

        SCS formula:
            S  = (25400 / CN) - 254
            Ia = 0.2 * S
            Q  = (P - Ia)^2 / (P - Ia + S)  if P > Ia, else 0

        Returns runoff in mm (>= 0).
        """
        if cn <= 0:
            return 0.0
        S = (25400 / cn) - 254
        Ia = 0.2 * S
        if rainfall_mm <= Ia:
            return 0.0
        return max(0.0, (rainfall_mm - Ia) ** 2 / (rainfall_mm - Ia + S))

    def runoff_ratio(self, rainfall_mm, cn):
        """Return fraction of rainfall that becomes runoff (0–1)."""
        if rainfall_mm <= 0:
            return 0.0
        return self.runoff_depth(rainfall_mm, cn) / rainfall_mm

    def catchment_volume(self, runoff_mm, catchment_area_m2):
        """Total runoff volume (m³) for a catchment."""
        return (runoff_mm / 1000.0) * catchment_area_m2

    def build_cn_raster(self, shape, transform, zone_geoms_cn, default_cn, moisture):
        """
        Build a moisture-adjusted CN raster from user-drawn zone polygons.

        Parameters
        ----------
        shape : tuple (rows, cols)
        transform : rasterio.Affine
        zone_geoms_cn : list of (shapely_geometry, int)
        default_cn : int
        moisture : str — 'dry', 'normal', or 'wet'

        Returns float32 array of adjusted CN values, same shape as DEM.
        """
        from rasterio.features import rasterize as _rasterize

        # One pass over the grid for every zone, not one pass per zone. rasterize
        # burns the shapes in order, so later zones win where they overlap —
        # exactly what the per-zone np.where did, at 1/n the cost.
        shapes = [(geom, float(cn_val)) for geom, cn_val in zone_geoms_cn
                  if geom is not None and not geom.is_empty]
        if shapes:
            cn_array = _rasterize(
                shapes, out_shape=shape, transform=transform,
                fill=float(default_cn), dtype="float32",
            )
            # A zone whose CN is 0 would be indistinguishable from unburned fill,
            # so anything non-positive falls back to the default rather than
            # becoming a hole the runoff maths reads as "no runoff here".
            cn_array = np.where(cn_array > 0, cn_array,
                                np.float32(default_cn)).astype("float32")
        else:
            cn_array = np.full(shape, float(default_cn), dtype="float32")

        return self.adjust_cn_array(cn_array, moisture).astype("float32")

    def build_runoff_raster(self, cn_array, rainfall_mm):
        """Per-cell runoff depth (mm) from a moisture-adjusted CN raster."""
        return self.runoff_depth_array(rainfall_mm, cn_array).astype("float32")

    @staticmethod
    def parse_hyetograph_csv(path):
        """
        Parse a rainfall time-series CSV into cumulative rainfall points.

        Expected columns: ``time_min``, ``rainfall_mm`` (header required).
        Auto-detects cumulative vs per-interval format.
        Always prepends (0, 0.0) baseline.

        Returns list of (time_min: int, cum_rainfall_mm: float).
        """
        import csv
        import os

        if not os.path.exists(path):
            raise ValueError(f"Cannot open hyetograph CSV: {path}")

        rows = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file appears to be empty.")
            headers = [h.strip().lower() for h in reader.fieldnames]
            if "time_min" not in headers or "rainfall_mm" not in headers:
                raise ValueError(
                    f"CSV must have columns 'time_min' and 'rainfall_mm'. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                try:
                    rows.append((float(row["time_min"]), float(row["rainfall_mm"])))
                except (KeyError, ValueError):
                    continue

        if not rows:
            raise ValueError("CSV contains no valid data rows.")

        rows.sort(key=lambda x: x[0])
        rain_vals = [r for _, r in rows]
        is_cumulative = _looks_cumulative(rain_vals)

        if is_cumulative:
            cum_pairs = [(int(round(t)), float(r)) for t, r in rows]
        else:
            cum = 0.0
            cum_pairs = []
            for t, r in rows:
                cum += max(0.0, r)
                cum_pairs.append((int(round(t)), cum))

        if cum_pairs[0][0] != 0:
            cum_pairs.insert(0, (0, 0.0))

        return cum_pairs
