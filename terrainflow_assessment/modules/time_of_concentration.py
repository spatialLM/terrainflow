"""
time_of_concentration.py — TR-55 segmental travel time.

Time of concentration is how long runoff takes to travel from the hydraulically most
distant point of a catchment to the point of interest. It matters because peak flow
occurs when the *whole* catchment is contributing at once, which happens at ``t = Tc``:
a shorter storm is more intense but engages only part of the catchment, a longer one
engages all of it but is weaker. So Tc is the duration at which to read an IDF curve.

TR-55 (USDA-NRCS, *Urban Hydrology for Small Watersheds*, 2nd ed. 1986, chapter 3)
splits the flow path into three regimes and sums their travel times:

    sheet flow  →  shallow concentrated flow  →  channel flow

All formulae here are the metric forms of TR-55's, with the conversion factors derived
from the published English coefficients in code rather than pasted as rounded
constants — the derivation is then checkable, and a transcription slip in a fourth
decimal place cannot hide.

Pure: no QGIS, no rasters.
"""

import math

# Sheet flow is capped at 100 ft. Beyond that runoff has concentrated into rills and
# the sheet-flow equation no longer describes it.
#
# ATTRIBUTION: 100 ft is the *revised* limit, from NEH-630 Ch. 15 (2010); TR-55 (1986,
# p. 3-3) itself says 300 ft. The 100 ft figure is the current guidance and the
# conservative one — a shorter sheet leg shortens Tc, raising the design intensity —
# so the behaviour stands; only the citation was wrong.
MAX_SHEET_FLOW_M = 100.0 * 0.3048          # 30.48 m

_FT_PER_M = 1.0 / 0.3048
_IN_PER_MM = 1.0 / 25.4

# TR-55 eq. 3-3 (English): Tt = 0.007 (nL)^0.8 / (P2^0.5 s^0.4), L in ft, P2 in in,
# Tt in hours. Converting L to metres and P2 to millimetres:
_SHEET_COEFF = 0.007 * (_FT_PER_M ** 0.8) / (_IN_PER_MM ** 0.5)     # ≈ 0.0913

# TR-55 eq. 3-1 / fig. 3-1 (English): V = 16.1345 √s unpaved, 20.3282 √s paved (ft/s).
_SHALLOW_V_UNPAVED = 16.1345 * 0.3048      # ≈ 4.918 m/s per √(m/m)
_SHALLOW_V_PAVED = 20.3282 * 0.3048        # ≈ 6.196 m/s per √(m/m)

# Manning's n for sheet flow (TR-55 table 3-1). Overland values, much higher than
# channel roughness: they represent raindrop impact and drag through the sward, not
# open-channel friction, and using a channel n here would understate Tc badly.
SHEET_ROUGHNESS = {
    "Smooth surfaces (concrete, asphalt, bare soil)": 0.011,
    "Fallow (no residue)": 0.05,
    "Cultivated, residue cover <= 20%": 0.06,
    "Cultivated, residue cover > 20%": 0.17,
    "Short grass prairie": 0.15,
    "Dense grasses / pasture": 0.24,
    "Bermuda grass": 0.41,
    "Range (natural)": 0.13,
    "Woods, light underbrush": 0.40,
    "Woods, dense underbrush": 0.80,
}
DEFAULT_SHEET_ROUGHNESS = 0.24             # dense grasses — NZ pasture

# Manning's n for the channel leg. Ordinary open-channel values.
DEFAULT_CHANNEL_ROUGHNESS = 0.05           # natural, some weeds and stones

# Representative hydraulic radii, computed from real trapezoidal sections rather than
# asserted (see tests). R = flow area / wetted perimeter; for a wide shallow channel it
# is close to the flow depth. Used only where the channel is natural ground — where it
# is one of the user's own drains, section_hydraulic_radius knows the exact figure.
CHANNEL_RADII = {
    "Shallow grassed swale (1.5 m x 0.15 m)": 0.125,
    "Small farm drain (1.6 m x 0.30 m)": 0.211,
    "Gully / farm stream (3 m x 0.5 m)": 0.366,
    "Incised watercourse (5 m x 1.2 m)": 0.761,
}
DEFAULT_CHANNEL_RADIUS_M = 0.211           # small farm drain

# Manning's n for an excavated, grassed earth channel — a swale or diversion drain.
# Matches the 0.025 `calculate_diversion_discharge` already uses for compacted earth,
# so a drain's travel time and its discharge rest on the same roughness.
EARTHWORK_CHANNEL_ROUGHNESS = 0.025

# Contributing area above which the flow path is treated as a defined watercourse
# rather than open ground. Channel flow is roughly twice the speed of shallow
# concentrated flow, so assuming *no* channel is not the cautious default — it
# lengthens Tc, lowers the design intensity, and undersizes the overflow.
DEFAULT_CHANNEL_THRESHOLD_HA = 1.0

# TR-55 App. F states the validity range of Tc as "minimum, 0.1; maximum, 10.0" hours.
# Outside it the method is beyond its calibration, and the answer is reported as a
# bound rather than quoted as computed. The ceiling matters in the dangerous
# direction: a long Tc reads a low intensity off the IDF curve, which under-sizes.
MIN_VALID_TC_HR = 0.1
MAX_VALID_TC_HR = 10.0


class TravelTime:
    """Travel time broken into its legs, so a surprising Tc can be interrogated."""

    def __init__(self, sheet_hr=0.0, shallow_hr=0.0, channel_hr=0.0, warnings=None):
        self.sheet_hr = float(sheet_hr)
        self.shallow_hr = float(shallow_hr)
        self.channel_hr = float(channel_hr)
        self.warnings = list(warnings or [])

    @property
    def total_hr(self):
        return self.sheet_hr + self.shallow_hr + self.channel_hr

    @property
    def total_min(self):
        return self.total_hr * 60.0

    @property
    def clamped(self):
        """True when the raw total fell outside TR-55's stated validity range."""
        return not (MIN_VALID_TC_HR <= self.total_hr <= MAX_VALID_TC_HR)

    @property
    def design_hr(self):
        """Total, held inside TR-55's validity range. Use this for an IDF lookup.

        The 10-hour ceiling is enforced as well as the 0.1-hour floor: letting a
        computed Tc run past it reads an intensity off the far tail of the IDF curve,
        under-stating the peak the design has to pass.
        """
        return min(max(self.total_hr, MIN_VALID_TC_HR), MAX_VALID_TC_HR)

    @property
    def design_min(self):
        return self.design_hr * 60.0

    def summary(self):
        return (f"{self.design_min:,.0f} min "
                f"(sheet {self.sheet_hr * 60:,.0f} + "
                f"shallow {self.shallow_hr * 60:,.0f} + "
                f"channel {self.channel_hr * 60:,.0f})")


def sheet_flow_hours(length_m, slope, roughness=DEFAULT_SHEET_ROUGHNESS, p2_mm=None):
    """TR-55 sheet flow travel time (hours).

    ``Tt = 0.0913 (nL)^0.8 / (P2^0.5 s^0.4)`` with *length_m* in metres, *p2_mm* the
    2-year 24-hour rainfall depth in millimetres, and *slope* as a fall/run ratio.

    *p2_mm* enters because sheet flow depth — and therefore its velocity — depends on
    how hard it is raining. It comes from the same IDF table the intensity does, so
    it is not an extra question to answer. Returns 0.0 when it is unknown, and the
    caller is expected to say the sheet leg was skipped rather than let it read as
    instantaneous.
    """
    if not length_m or length_m <= 0 or not p2_mm or p2_mm <= 0:
        return 0.0
    length = min(float(length_m), MAX_SHEET_FLOW_M)
    s = max(float(slope or 0.0), 1e-4)     # a truly flat cell would divide by zero
    n = max(float(roughness or DEFAULT_SHEET_ROUGHNESS), 1e-3)
    return _SHEET_COEFF * ((n * length) ** 0.8) / ((float(p2_mm) ** 0.5) * (s ** 0.4))


def shallow_concentrated_hours(length_m, slope, paved=False):
    """TR-55 shallow concentrated flow travel time (hours) — ``V = k √s``."""
    if not length_m or length_m <= 0:
        return 0.0
    s = max(float(slope or 0.0), 1e-4)
    k = _SHALLOW_V_PAVED if paved else _SHALLOW_V_UNPAVED
    velocity = k * math.sqrt(s)
    return float(length_m) / (velocity * 3600.0)


def channel_flow_hours(length_m, slope, hydraulic_radius_m=DEFAULT_CHANNEL_RADIUS_M,
                       roughness=DEFAULT_CHANNEL_ROUGHNESS):
    """Channel travel time (hours) from Manning's ``V = (1/n) R^(2/3) √s``.

    *hydraulic_radius_m* defaults to a small farm drain. Where the channel is one of
    the user's own swales or diversions it should come from
    :func:`section_hydraulic_radius` instead, which knows the entered dimensions
    exactly rather than assuming a shape.
    """
    if not length_m or length_m <= 0:
        return 0.0
    s = max(float(slope or 0.0), 1e-4)
    n = max(float(roughness or DEFAULT_CHANNEL_ROUGHNESS), 1e-3)
    velocity = (1.0 / n) * (float(hydraulic_radius_m) ** (2.0 / 3.0)) * math.sqrt(s)
    if velocity <= 0:
        return 0.0
    return float(length_m) / (velocity * 3600.0)


def time_of_concentration(sheet=None, shallow=None, channel=None, p2_mm=None,
                          channel_hours=None):
    """Sum the three legs into a :class:`TravelTime`.

    Each argument is ``(length_m, slope)``, optionally with extra keys:
    *sheet* accepts ``roughness``, *shallow* accepts ``paved``, *channel* accepts
    ``hydraulic_radius_m`` and ``roughness``. Any leg may be ``None`` — a catchment
    with no defined channel simply has no channel leg.

    *channel_hours* replaces the channel calculation with a value already worked out
    by :func:`mixed_channel_hours`, for a path whose section changes along its length.

    The over-long-sheet-leg warning looks unreachable and is not. Both builders of
    these tuples — :func:`split_flow_path` and :func:`profile_leg_slopes` — clip the
    sheet leg at ``MAX_SHEET_FLOW_M`` and give the remainder to the shallow leg, so
    nothing in the plugin can trigger it. It is for a caller assembling the legs
    itself: :func:`sheet_flow_hours` clips silently, and without the warning that
    caller would get a quietly shortened path and no way to know.
    """
    warnings = []

    sheet_hr = 0.0
    if sheet:
        length, slope = sheet[0], sheet[1]
        extra = sheet[2] if len(sheet) > 2 and isinstance(sheet[2], dict) else {}
        if not p2_mm:
            warnings.append(
                "No 2-year 24-hour rainfall depth, so the sheet-flow leg was skipped. "
                "Time of concentration is under-stated, which over-states peak flow."
            )
        else:
            sheet_hr = sheet_flow_hours(
                length, slope, extra.get("roughness", DEFAULT_SHEET_ROUGHNESS), p2_mm)
            if length and length > MAX_SHEET_FLOW_M:
                warnings.append(
                    f"Sheet flow capped at {MAX_SHEET_FLOW_M:.0f} m per TR-55; the "
                    f"remaining {length - MAX_SHEET_FLOW_M:,.0f} m was treated as "
                    f"shallow concentrated flow."
                )

    shallow_hr = 0.0
    if shallow:
        extra = shallow[2] if len(shallow) > 2 and isinstance(shallow[2], dict) else {}
        shallow_hr = shallow_concentrated_hours(
            shallow[0], shallow[1], paved=extra.get("paved", False))

    channel_hr = 0.0
    if channel_hours is not None:
        channel_hr = max(0.0, float(channel_hours))
    elif channel:
        extra = channel[2] if len(channel) > 2 and isinstance(channel[2], dict) else {}
        channel_hr = channel_flow_hours(
            channel[0], channel[1],
            hydraulic_radius_m=extra.get("hydraulic_radius_m", DEFAULT_CHANNEL_RADIUS_M),
            roughness=extra.get("roughness", DEFAULT_CHANNEL_ROUGHNESS),
        )

    tt = TravelTime(sheet_hr, shallow_hr, channel_hr, warnings)
    if 0 < tt.total_hr < MIN_VALID_TC_HR:
        tt.warnings.append(
            f"Computed {tt.total_min:,.1f} min, below TR-55's {MIN_VALID_TC_HR * 60:.0f} "
            f"min validity floor — held at the floor. Small steep catchments respond "
            f"faster than the method resolves."
        )
    elif tt.total_hr > MAX_VALID_TC_HR:
        tt.warnings.append(
            f"Computed {tt.total_hr:,.1f} hr, above TR-55's {MAX_VALID_TC_HR:.0f} hr "
            f"validity ceiling — held at the ceiling. A flow path this long is beyond "
            f"what the method was calibrated on; check the path before trusting the "
            f"design intensity it produces."
        )
    return tt


def section_hydraulic_radius(top_width_m, bottom_width_m, depth_m):
    """R for a channel whose dimensions the user has entered — ``area / wetted P``.

    Swales and diversions are the only registry types carrying a cross-section, and
    for those nothing needs estimating: the trapezoid is known exactly. Delegates to
    the same :func:`~terrainflow_assessment.core.sizing.trapezoid_section` that sizes
    diversion discharge, so a drain's travel time and its capacity cannot disagree
    about its shape.

    .. note::
       This is the **bankfull** radius — the section running full. Real flow rarely
       fills a channel, and a part-full channel has a smaller R and so a lower
       velocity. Assuming bankfull therefore overstates velocity, shortens the time
       of concentration, raises the design intensity and **oversizes** the overflow.
       That is the safe direction, and it is consistent with how diversion capacity
       is already computed.
    """
    from terrainflow_assessment.core.sizing import trapezoid_section

    try:
        section = trapezoid_section(
            float(top_width_m), max(0.05, float(bottom_width_m)), float(depth_m))
    except (TypeError, ValueError):
        return None
    radius = getattr(section, "hydraulic_radius", 0.0)
    return float(radius) if radius and radius > 0 else None


def mixed_channel_hours(distances_m, radii_m, elevations_m=None,
                        roughnesses=None, slope=None):
    """Channel travel time (hours) where the cross-section changes along the path.

    A real flow path is rarely one channel. It may run down a natural gully, then
    along a diversion drain the user dug, then into a swale — each with its own
    section and roughness, and a diversion is roughly twice the hydraulic radius of
    the gully above it. Treating that as one average section throws away dimensions
    that were entered exactly.

    *distances_m* is cumulative distance along the channel leg; *radii_m* and
    *roughnesses* are the section at each of those points. Consecutive points sharing
    a section form one run. Each run takes its slope end-to-end from *elevations_m*,
    or from *slope* when no profile is supplied.

    Returns ``(hours, runs)`` — *runs* being ``(length_m, slope, radius, roughness)``
    per sub-segment, so the caller can show what it found rather than only the total.
    """
    if not distances_m or not radii_m or len(distances_m) != len(radii_m):
        return 0.0, []
    if roughnesses is None:
        roughnesses = [DEFAULT_CHANNEL_ROUGHNESS] * len(radii_m)

    if len(distances_m) < 2:
        return 0.0, []

    def _slope_between(i, j):
        if slope is not None or elevations_m is None:
            return float(slope or 0.0)
        run = distances_m[j] - distances_m[i]
        if run <= 0:
            return 0.0
        return abs(elevations_m[i] - elevations_m[j]) / run

    def _section(i):
        return (radii_m[i] or DEFAULT_CHANNEL_RADIUS_M,
                roughnesses[i] or DEFAULT_CHANNEL_ROUGHNESS)

    # A section belongs to the *segment* between two points, taken from the
    # downstream one — so a change recorded at point i starts applying on the way
    # into i, not on the way out of it. Indexing by point instead put the whole run
    # on the wrong side of every transition.
    runs, total = [], 0.0
    start, last = 0, len(distances_m) - 1
    for i in range(1, len(distances_m)):
        if i < last and _section(i + 1) == _section(i):
            continue
        length = distances_m[i] - distances_m[start]
        if length > 0:
            seg_slope = _slope_between(start, i)
            radius, n = _section(i)
            total += channel_flow_hours(length, seg_slope, radius, n)
            runs.append((length, seg_slope, radius, n))
        start = i
    return total, runs


def channel_length_from_area(distances_m, upstream_area_m2,
                             threshold_ha=DEFAULT_CHANNEL_THRESHOLD_HA):
    """Length of the flow path that runs in a defined watercourse, in metres.

    Runoff concentrates into a channel once enough ground drains through a point, so
    the transition is measurable rather than a matter of opinion: walk the path from
    the ridge and find where contributing area first crosses *threshold_ha*;
    everything below that is channel.

    Measuring it matters because zero is not the safe default. Channel flow is about
    twice the speed of shallow concentrated flow, so pretending there is no channel
    lengthens Tc, lowers the design intensity, and undersizes the overflow.

    Returns 0.0 when the threshold is never reached — a small paddock catchment with
    no watercourse, where zero is the right answer rather than a fallback.
    """
    if not distances_m or not upstream_area_m2:
        return 0.0
    if len(distances_m) != len(upstream_area_m2):
        return 0.0

    threshold_m2 = max(0.0, float(threshold_ha)) * 10_000.0
    total = float(distances_m[-1])
    for distance, area in zip(distances_m, upstream_area_m2):
        if area is not None and float(area) >= threshold_m2:
            return max(0.0, total - float(distance))
    return 0.0


def profile_leg_slopes(distances_m, elevations_m, channel_length_m=0.0):
    """Per-leg slopes from an elevation profile along the flow path.

    TR-55 wants a slope for each segment, not one figure for the whole path, and the
    difference is not academic. Hillslopes are concave — steep off the ridge,
    flattening into the valley — so a single average is far too gentle where sheet
    flow happens and too steep in the channel. On a typical concave profile it
    lengthens Tc by about 25%, which lowers the design intensity and undersizes the
    overflow.

    *distances_m* is cumulative distance from the most distant point; *elevations_m*
    the matching elevations. Returns ``(sheet_slope, shallow_slope, channel_slope)``,
    each a fall/run ratio over its own leg.

    Slope is taken end-to-end across each leg rather than averaged cell by cell: it is
    the hydraulic grade line the travel-time formulae ask for, and it is unaffected by
    DEM noise between the ends.
    """
    if distances_m is None or elevations_m is None:
        return (0.0, 0.0, 0.0)
    distances = [float(d) for d in distances_m]
    elevations = [float(z) for z in elevations_m]
    if len(distances) < 2 or len(distances) != len(elevations):
        return (0.0, 0.0, 0.0)

    total = distances[-1]
    if total <= 0:
        return (0.0, 0.0, 0.0)

    channel_len = max(0.0, min(float(channel_length_m or 0.0), total))
    overland = total - channel_len
    sheet_end = min(overland, MAX_SHEET_FLOW_M)
    shallow_end = overland

    def _slope(start, end):
        if end - start <= 0:
            return 0.0
        z0 = _interp(distances, elevations, start)
        z1 = _interp(distances, elevations, end)
        return abs(z0 - z1) / (end - start)

    return (_slope(0.0, sheet_end),
            _slope(sheet_end, shallow_end),
            _slope(shallow_end, total))


def _interp(xs, ys, x):
    """Linear interpolation on a monotonically increasing *xs*."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if xs[i] >= x:
            x0, x1 = xs[i - 1], xs[i]
            if x1 == x0:
                return ys[i]
            t = (x - x0) / (x1 - x0)
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def split_flow_path(total_length_m, slope, channel_length_m=0.0):
    """Divide a measured flow path into TR-55's three legs.

    Sheet flow takes the first 30.5 m (TR-55's cap), the channel leg takes whatever
    length was identified as channel, and shallow concentrated flow takes the
    remainder. Returns the three ``(length_m, slope)`` tuples ready for
    :func:`time_of_concentration`.

    *slope* may be a single value applied to every leg, or a ``(sheet, shallow,
    channel)`` triple from :func:`profile_leg_slopes`. Prefer the triple: one average
    over a concave hillslope is far too gentle where sheet flow happens, and errs
    toward undersizing the overflow.
    """
    total = max(0.0, float(total_length_m or 0.0))
    channel_len = max(0.0, min(float(channel_length_m or 0.0), total))
    overland = total - channel_len

    sheet_len = min(overland, MAX_SHEET_FLOW_M)
    shallow_len = overland - sheet_len

    if isinstance(slope, (tuple, list)) and len(slope) == 3:
        sheet_s, shallow_s, channel_s = (float(s or 0.0) for s in slope)
    else:
        sheet_s = shallow_s = channel_s = float(slope or 0.0)

    return (
        (sheet_len, sheet_s),
        (shallow_len, shallow_s),
        (channel_len, channel_s),
    )
