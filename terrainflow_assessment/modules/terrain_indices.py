"""
terrain_indices.py — terrain derivatives that rank ground by shape and catchment.

The analysis tier could say *where water goes*; it could not say *where ground is wet,
where flow gathers force, or which way a face looks*. These are the standard published
answers to that, computed off rasters the baseline run already produces:

    specific_catchment_area   — upslope area per unit contour width (m²/m)
    topographic_wetness_index — ln(a / tanβ)                    Beven & Kirkby 1979
    stream_power_index        — a · tanβ                        Moore et al. 1991
    sediment_transport_index  — the unit-stream-power LS form    Moore & Burch 1986
    curvature                 — plan and profile (1/m)          Zevenbergen & Thorne 1987
    landform_tpi              — topographic position index       Weiss 2001 / Jenness 2006
    slope_statistics          — the site's slope distribution

**What these are not.** They rank ground by shape and contributing area *only* — no
soil, no ground cover, no rainfall. They say where water tends to gather, where flow
gathers force, and where soil tends to leave. They do not predict a depth, a rate or a
tonnage, and they look far more quantitative than they are.

Three things about the inputs are load-bearing:

**1. Contributing area is a CELL COUNT.** Every function here takes
``contributing_cells`` and means ``FlowAnalysis.acc`` — never ``runoff_accumulation``,
which is m³ and which every hollow has already held water back from. Feeding the volume
field in would produce a "wetness index" whose numerator has been retained and whose
denominator has not; the parameter is named for its unit so the mistake has to be typed
out to be made. See CLAUDE.md's note on the two accumulation fields.

**2. tan β comes from the RAW DEM, not the conditioned one.** ``resolve_flats`` lifts
every flat by an integer multiple of 1e-5 m so the router has somewhere to send water.
That gradient is a routing device, not terrain. Computing tan β on the conditioned
surface puts a fabricated 1e-5 slope under every flat and makes TWI look *measured*
exactly where it is invented. Pair acc-from-conditioned with slope-from-raw, and floor
the result explicitly.

**3. A flat cell has no wetness index, and saying so is the honest answer.** ``ln(a/0)``
is infinite. The floor is applied and the number of cells that hit it is **returned**
rather than swallowed — on a tile that is most of a harbour plane, the unreported
version is a fiction. Same discipline as ``unrouted_flow``'s impossible-share report.

Pure numpy + scipy: no QGIS, no rasterio.
"""

from __future__ import annotations

import numpy as np

# Below this, ground is level enough that the wetness index would run away. 0.001 m/m
# (about 0.057°) is the conventional floor in the TWI literature (Quinn et al. 1991's
# minimum-slope treatment for flats). It is a *choice*, not a derivation.
DEFAULT_MIN_TAN_BETA = 0.001

# Weiss (2001) classifies landform on TPI expressed in standard deviations, not metres,
# which is what makes one threshold mean the same thing on a scarp and on a floodplain.
DEFAULT_TPI_SD = 1.0


def specific_catchment_area(contributing_cells, cell_w, cell_h):
    """Upslope area per unit contour width, m²/m — the ``a`` of every index below.

    ``a = (acc + 1) · cell_area / contour_width``.

    **The ``+1`` was measured, not assumed.** pysheds' accumulation counts the cells
    *upslope* of each cell and excludes the cell itself: on a uniform plane the ridge
    row reads 0, the row below it 1, the fifth row 5. Without the ``+1`` every ridge
    cell has zero catchment and ``ln(a)`` is ``-inf`` along the top of every hill. With
    it, a ridge cell contributes its own footprint, which is what it does.

    Contour width is taken as the x resolution, the D8 convention. On a non-square grid
    that is axis-dependent and there is no single right answer — the published index
    assumes square cells. Noted rather than hidden; see the cell-size register in the
    maths audit for the other places this bites.
    """
    acc = np.asarray(contributing_cells, dtype="float64")
    cell_area = float(cell_w) * float(cell_h)
    return (acc + 1.0) * cell_area / float(cell_w)


def _tan_beta(slope_deg):
    return np.tan(np.radians(np.asarray(slope_deg, dtype="float64")))


def topographic_wetness_index(contributing_cells, slope_deg, cell_w, cell_h,
                              min_tan_beta=DEFAULT_MIN_TAN_BETA):
    """``ln(a / tanβ)`` — Beven & Kirkby (1979). Returns ``(values, floored_cells)``.

    High where a lot of ground drains to somewhere flat: the places that hold water,
    and so the places a pond or a basin has something to hold.

    ``floored_cells`` is how many cells were too flat to have a defensible value and
    took ``min_tan_beta`` instead. It is returned rather than logged because on a tile
    with a large flat — a harbour plane, a lake, a terrace — it can be most of the map,
    and a wetness index over invented slope should be reported as such.
    """
    a = specific_catchment_area(contributing_cells, cell_w, cell_h)
    tan_beta = _tan_beta(slope_deg)

    invalid = ~np.isfinite(tan_beta)
    floored = np.isfinite(tan_beta) & (tan_beta < min_tan_beta)
    tan_beta = np.where(floored, min_tan_beta, tan_beta)

    with np.errstate(invalid="ignore", divide="ignore"):
        twi = np.log(a / tan_beta)
    twi = np.where(invalid, np.nan, twi)
    return twi.astype("float32"), int(floored.sum())


def stream_power_index(contributing_cells, slope_deg, cell_w, cell_h, log=True):
    """``a · tanβ`` — Moore, Grayson & Ladson (1991). The erosive power of overland flow.

    High where a large catchment meets steep ground: where a drain will scour and where
    a gully is already forming. Nothing is divided, so no floor is needed — a flat cell
    correctly has no stream power.

    ``log=True`` returns ``ln(1 + SPI)``. The raw field spans six decades, so a linear
    ramp over it paints one gully bright and the entire rest of the site black. The
    ``1 +`` keeps zero at zero rather than at minus infinity.
    """
    a = specific_catchment_area(contributing_cells, cell_w, cell_h)
    tan_beta = _tan_beta(slope_deg)
    spi = a * np.where(np.isfinite(tan_beta), np.maximum(tan_beta, 0.0), np.nan)
    if log:
        with np.errstate(invalid="ignore"):
            spi = np.log1p(spi)
    return spi.astype("float32")


def sediment_transport_index(contributing_cells, slope_deg, cell_w, cell_h,
                             m=0.6, n=1.3):
    """``(a/22.13)^m · (sinβ/0.0896)^n`` — Moore & Burch (1986).

    The unit-stream-power analogue of RUSLE's LS factor. The two constants are the
    published normalisers, not tuning.

    **Relative only.** Without soil erodibility and ground cover this is not a sediment
    transport rate and must never be printed as one — rank it, label it a proxy, and say
    what it leaves out. An absolute-looking number here invites exactly the reading it
    cannot support.
    """
    a = specific_catchment_area(contributing_cells, cell_w, cell_h)
    beta = np.radians(np.asarray(slope_deg, dtype="float64"))
    sin_beta = np.sin(beta)
    with np.errstate(invalid="ignore"):
        sti = ((a / 22.13) ** m) * ((np.maximum(sin_beta, 0.0) / 0.0896) ** n)
    return np.where(np.isfinite(sin_beta), sti, np.nan).astype("float32")


def curvature(dem, cell_w, cell_h):
    """Plan and profile curvature in 1/m — Zevenbergen & Thorne (1987).

    The closed-form quadratic-surface fit GRASS's ``r.slope.aspect`` and ArcGIS both
    use. Deliberately a different estimator from ``slope_degrees``' Horn: Horn gives
    first derivatives only, and second derivatives are the whole point here.

    **Sign convention, stated because every published source states a different one and
    getting it wrong silently inverts a landform classification:**

    * ``profile`` — curvature along the direction of steepest descent.
      **Positive is convex** (the slope steepens downhill; flow accelerates; ground
      sheds). Negative is concave (the slope eases; flow decelerates; deposition). The
      Yeomans keypoint is the strongest *negative* profile curvature on a valley floor.
    * ``plan`` — curvature across the slope.
      **Positive is convex in plan** (flow diverges — a nose, a ridge). Negative is
      concave (flow converges — a hollow, a valley).

    So a ridge is positive in plan and a valley is negative in plan, which is the
    distinction Yeomans' two cultivation patterns turn on.

    On a genuinely flat cell both are 0.0 — undefined in the ratio, but flat ground has
    no curvature and zero is the honest reading rather than NaN.

    Returns ``(plan, profile)``, both float32, NaN where the DEM is NaN.
    """
    z = np.asarray(dem, dtype="float64")
    invalid = ~np.isfinite(z)
    zp = np.pad(np.where(invalid, np.nan, z), 1, mode="edge")

    # A nodata neighbour reads as the centre cell, as the Horn stencil does — so the
    # quadric is fitted to a locally flat window rather than to a cliff that is not
    # there. Same reasoning as dem_loader's, and the same boundary it protects.
    centre = np.where(invalid, 0.0, z)

    def _nb(block):
        return np.where(np.isnan(block), centre, block)

    z1, z2, z3 = _nb(zp[:-2, :-2]), _nb(zp[:-2, 1:-1]), _nb(zp[:-2, 2:])
    z4, z5, z6 = _nb(zp[1:-1, :-2]), centre, _nb(zp[1:-1, 2:])
    z7, z8, z9 = _nb(zp[2:, :-2]), _nb(zp[2:, 1:-1]), _nb(zp[2:, 2:])

    lx = float(cell_w)
    ly = float(cell_h)

    d = ((z4 + z6) / 2.0 - z5) / (lx * lx)
    e = ((z2 + z8) / 2.0 - z5) / (ly * ly)
    f = (-z1 + z3 + z7 - z9) / (4.0 * lx * ly)
    g = (-z4 + z6) / (2.0 * lx)
    # Row index increases southward, so the north-up rise is z2 (north) minus z8 (south).
    h = (z2 - z8) / (2.0 * ly)

    p = g * g + h * h
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = np.where(p > 0, -2.0 * (d * g * g + e * h * h + f * g * h) / p, 0.0)
        # Negated against Zevenbergen & Thorne's printed plan term on purpose. Their
        # sign makes a diverging nose negative and a converging hollow positive, which
        # is the opposite of the convention documented above and of TPI's (ridge
        # positive). Two landform signals that disagree about which way is "ridge" is
        # how a cultivation pattern ends up applied to the wrong half of a hillside.
        # The paired cone/bowl test is what pins this.
        plan = np.where(p > 0, -2.0 * (d * h * h + e * g * g - f * g * h) / p, 0.0)

    profile = np.where(invalid, np.nan, profile)
    plan = np.where(invalid, np.nan, plan)
    return plan.astype("float32"), profile.astype("float32")


def landform_tpi(dem, cell_w, cell_h, window_m=15.0):
    """Topographic position index in metres — Weiss (2001), Jenness (2006).

    ``z − mean(z)`` over a neighbourhood: positive on ridges and spurs, negative in
    hollows and valleys, near zero on a planar slope.

    **The window is metres, not cells.** It used to be a fixed cell count, which made
    the landform scale silently resolution-dependent — 15 cells is 15 m on a 1 m grid
    and 75 m on a 5 m grid, and Weiss's whole point is that the answer depends on the
    scale you ask at. Asking in metres means the same question on every DEM.

    The neighbourhood mean is taken over **valid cells only**. Filling nodata with the
    whole-DEM mean dragged the local mean toward it for every cell within half a window
    of a hole, fabricating a ridge line all the way round the data boundary — which is
    the one place users most often clip to.
    """
    from scipy.ndimage import uniform_filter

    z = np.asarray(dem, dtype="float64")
    cell = (float(cell_w) + float(cell_h)) / 2.0
    size = max(3, int(round(float(window_m) / max(cell, 1e-9))))
    if size % 2 == 0:
        size += 1   # an even window has no centre cell to be the position of

    valid = np.isfinite(z)
    filled = np.where(valid, z, 0.0)
    total = uniform_filter(filled, size=size)
    count = uniform_filter(valid.astype("float64"), size=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        neighbourhood_mean = np.where(count > 0, total / count, np.nan)
    return np.where(valid, filled - neighbourhood_mean, np.nan).astype("float32")


def landform_classes(tpi, slope_deg, sd=DEFAULT_TPI_SD, mask=None):
    """TPI → ``{-1 valley, 0 midslope, +1 ridge}``, thresholded in standard deviations.

    Weiss's own formulation: the cut is ±*sd* standard deviations of the TPI over the
    area being classified, so one threshold means the same thing on a scarp and on a
    floodplain. An absolute metre threshold does not.

    ``mask`` restricts which cells the standard deviation is computed over — classify a
    site against its own relief, not against a tile that is mostly harbour.

    Returns int8, with 0 (midslope) where the TPI is NaN: unknown ground is not a ridge
    and is not a valley, and midslope is the class that claims least.
    """
    tpi = np.asarray(tpi, dtype="float64")
    finite = np.isfinite(tpi)
    sample = finite if mask is None else (finite & np.asarray(mask, dtype=bool))
    if not sample.any():
        return np.zeros(tpi.shape, dtype="int8")

    threshold = float(sd) * float(np.std(tpi[sample]))
    out = np.zeros(tpi.shape, dtype="int8")
    if threshold <= 0.0:
        return out
    with np.errstate(invalid="ignore"):
        out[finite & (tpi > threshold)] = 1
        out[finite & (tpi < -threshold)] = -1
    return out


def slope_statistics(slope_deg, mask=None, quantiles=(25, 50, 75)):
    """The site's slope distribution — ``{"p25": …, "p50": …, "p75": …, "n": …}``.

    A farm is not one slope, and a single mean is the answer that hides the paddock the
    advice is wrong for. Percentiles in degrees, plus the same values as decimal grade
    (rise/run) because the spacing rules are written in grade.

    Returns ``None`` when nothing is measurable, rather than a dictionary of NaN.
    """
    s = np.asarray(slope_deg, dtype="float64")
    ok = np.isfinite(s)
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if not ok.any():
        return None

    values = s[ok]
    out = {"n": int(values.size)}
    for q in quantiles:
        deg = float(np.percentile(values, q))
        out[f"p{int(q)}"] = deg
        out[f"p{int(q)}_grade"] = float(np.tan(np.radians(deg)))
    return out
