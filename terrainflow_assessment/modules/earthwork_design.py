"""
earthwork_design.py — Earthwork data model, DEM burning, and capacity calculations.

Combines the base plugin's earthwork.py and dem_burner.py into a single module.

Exports
-------
Earthwork            — data class for a single earthwork feature
EarthworkManager     — manages the list of earthworks
DEMBurner            — burns earthworks into DEM, computes ponding
calculate_capacity   — storage capacity from geometry + dimensions
calculate_cut_volume — excavation volume (for reporting)
calculate_fill_volume— material placed (for reporting)
calculate_diversion_discharge — Manning's discharge for diversion drains
calculate_spillway_width      — broad-crested weir sizing (head chosen → width)
head_for_width                — the same weir equation inverted (width built → head)
effective_head_m              — which of those two applies, decided in one place
Spillway             — designed overflow point (crest, head, width, location)
spillway_datum       — crest elevations a feature can physically offer
spillway_policy      — per-type freeboard / head / head band, from the registry
effective_freeboard_m — the freeboard in force: stored override, else type policy
bind_crest           — two-way crest ↔ drop-below-rim binding
spillway_validity    — plain-language problems with a proposed spillway
berm_height_estimate          — companion berm height from swale volume
FeatureStorage       — what one earthwork impounds, measured by flooding it alone
"""

import json
import logging
import uuid
from typing import NamedTuple, Optional

import numpy as np
import rasterio
from shapely.geometry import LineString
from shapely.geometry import shape as shapely_shape

from terrainflow_assessment.core.registry.earthwork_types import get_type
from terrainflow_assessment.core.sizing import (
    basin_volume_battered,
    level_crest_from_spoil,
    manning_flow,
    trapezoid_section,
)
from terrainflow_assessment.modules.burn_strategy import (
    berm_variation_warning,
    enforce_monotonic_path,
    level_invert,
    line_cells,
    ponding_resolution_warning,
    rasterisable_capacity,
    steep_ground_warning,
    sub_cell_warning,
    taper_reach,
    tapered_invert,
)
from terrainflow_assessment.modules.footprint import (
    internal_relief,
    min_dimension,
    pour_point,
    rasterize_footprint,
    xy_to_rc,
)
from terrainflow_assessment.qgis.adapters.geom import shapely_area, shapely_length

_log = logging.getLogger(__name__)

# ~2000 × 2000. A memory guard, deliberately counted in cells rather than derived
# from a ground distance: what it is protecting is the size of the arrays the flood
# allocates, and that is a cell count whatever the cell happens to measure.
_MAX_PONDING_CELLS = 4_000_000
_DAM_WINDOW_PAD_CELLS = 64      # initial crop padding for the windowed per-feature flood
# Below a millimetre a "pond" is flood-fill noise, not water. Matches the default in
# ``reporting.attribute_ponding_volume`` so the two agree about what counts as ponding.
_POND_MIN_DEPTH_M = 0.001


class FeatureStorage(NamedTuple):
    """What one earthwork impounds, measured by flooding it alone on the original ground.

    ``volume_m3``        the pond it creates, over and above what ponds there naturally
    ``level_m``          the water surface it fills to — the level it actually spills at
    ``region``           bool mask of the pond, in the flood window's frame
    ``above_ground_m3``  the part standing proud of natural ground (the bank's work)
    ``retained_depth_m`` how deep the water stands against the bank that holds it

    The last two are what separate a swale from a small dam, and neither is visible in a
    volume alone: a 1,095 m³ pond is unremarkable if it sits in a hollow and is a
    retaining structure if 1.0 m of it stands against a spoil bank.
    """

    volume_m3: float
    level_m: Optional[float]
    region: object
    above_ground_m3: float
    retained_depth_m: float


def extend_to_abutments(coords, elevation_at, crest_elev, max_extend_m=250.0,
                        step_m=1.0):
    """Extend a dam alignment at both ends until the ground reaches the crest.

    A wall only impounds water if it runs into ground at least as high as its crest.
    Stop short of that and the pond simply flows around the end, however tall the wall
    is in the middle — the drawn length, not the wall height, sets what it holds.

    Walks outward from each endpoint along the bearing of its terminal segment,
    sampling the ground every *step_m*, and stops at the first point where the terrain
    has risen to ``crest_elev`` (the natural abutment). An end already at or above the
    crest is left alone.

    Parameters
    ----------
    coords : ordered [(x, y), ...] of the drawn alignment.
    elevation_at : ``f(x, y) -> float or None`` — None means off the DEM, which ends
        that walk at the last point still on it.
    crest_elev : absolute crest elevation (m).
    max_extend_m : how far to search before giving up on an end.
    step_m : sampling interval — the DEM cell size is the sensible choice.

    Returns ``(coords, info)`` where *info* carries ``start_m`` / ``end_m`` (metres
    added) and ``start_keyed`` / ``end_keyed`` (whether the abutment was actually
    reached). An end that returns False did **not** find high ground within
    *max_extend_m*: water will go round it, and the caller should say so rather than
    quietly present a wall that cannot hold its stated volume.
    """
    import math

    info = {"start_m": 0.0, "end_m": 0.0, "start_keyed": False, "end_keyed": False}
    try:
        pts = [(float(c[0]), float(c[1])) for c in coords]
    except (TypeError, ValueError, IndexError):
        return list(coords), info
    if len(pts) < 2 or crest_elev is None or step_m <= 0 or max_extend_m <= 0:
        return pts, info

    def _walk(anchor, inward):
        """March outward from *anchor*, directly away from *inward*."""
        ax, ay = anchor
        dx, dy = ax - inward[0], ay - inward[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return None, 0.0, False

        here = elevation_at(ax, ay)
        if here is not None and here >= crest_elev:
            return None, 0.0, True          # already keyed into high ground

        ux, uy = dx / length, dy / length
        dist = 0.0
        last = None
        while dist < max_extend_m:
            dist = min(dist + step_m, max_extend_m)
            px, py = ax + ux * dist, ay + uy * dist
            elev = elevation_at(px, py)
            if elev is None:
                break                       # ran off the DEM
            last = (px, py)
            if elev >= crest_elev:
                return (px, py), dist, True
        return last, dist, False

    start_pt, start_m, start_keyed = _walk(pts[0], pts[1])
    end_pt, end_m, end_keyed = _walk(pts[-1], pts[-2])

    out = list(pts)
    info["start_keyed"], info["end_keyed"] = start_keyed, end_keyed
    if start_pt is not None:
        out.insert(0, start_pt)
        info["start_m"] = start_m
    if end_pt is not None:
        out.append(end_pt)
        info["end_m"] = end_m
    return out, info


def abutment_warning(name, info, crest_elev, max_extend_m):
    """Advisory when a dam could not be keyed into the banks at its crest, else None."""
    open_ends = [side for side in ("start", "end") if not info.get(f"{side}_keyed")]
    if not open_ends:
        return None
    which = " and ".join("west/start" if s == "start" else "east/end" for s in open_ends)
    return (
        f"'{name}': the ground does not rise to the {crest_elev:.2f} m crest within "
        f"{max_extend_m:.0f} m at the {which} — water will flow around that end rather "
        f"than be held. Lower the crest, or redraw the wall across a narrower section."
    )


def _as_float(value, default=0.0):
    """Coerce an optional numeric attribute, tolerating None and non-numerics."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pond_touches_edge(pond, eps=1e-6):
    """True if ponding reaches the window rim (the pond may be clipped).

    Depression-filling treats the array boundary as the drainage outlet, so the
    boundary cells themselves never pond — a clipped pond shows up in the ring
    one cell in from the edge instead. Tiny windows always report clipped.
    """
    if pond.shape[0] < 4 or pond.shape[1] < 4:
        return True
    return bool(
        (pond[1, :] > eps).any() or (pond[-2, :] > eps).any()
        or (pond[:, 1] > eps).any() or (pond[:, -2] > eps).any()
    )


# ---------------------------------------------------------------------------
# Spillway
# ---------------------------------------------------------------------------

# Clear height required between the design nappe and the lowest containing ground.
# A spillway exists so that overflow leaves at a chosen, armoured place. If the
# water surface at design head reaches the surrounding rim, water escapes there
# too and that choice is lost — so this is the margin that makes the spillway the
# actual control rather than merely the intended one.
#
# 0.30 m is the NRCS Conservation Practice Standard 378 (Pond) minimum: "a minimum of
# 1.0 feet of freeboard between design high-water-flow elevation in the auxiliary
# spillway and the top of the settled embankment".
#
# Read that quote carefully: it is an *embankment* standard. It measures to the top of a
# built wall, and it exists because overtopping a wall breaches it. A cut contour swale
# has no embankment — its containment is undisturbed ground, and its failure mode is
# water leaving at an unarmoured point rather than a wall giving way. Applying this
# figure to a 0.5 m swale spends 0.60 m of a 0.50 m dig before any head is added, which
# reports the most ordinary swale in the plugin as impossible.
#
# So this stays the *embankment* number — the default for dams, basins and any
# unrecognised type — and per-type values live in the registry. See spillway_policy.
SPILLWAY_MIN_FREEBOARD_M = 0.30

# Head range that broad-crested weir practice treats as ordinary. Outside it the
# weir formula still holds; it simply stops being a routine design. Embankment default;
# per-type bands live in the registry (a swale's is lower, or it would warn always).
SPILLWAY_TYPICAL_HEAD_M = (0.20, 0.50)


def spillway_policy(ew_type):
    """Per-type spillway design policy — ``(freeboard_m, head_m, head_band)``.

    Freeboard and head are not one number across types: see the comment on
    :data:`SPILLWAY_MIN_FREEBOARD_M` for why an embankment figure does not describe a
    cut channel. The values live in the type registry, which is the one-file home of
    per-type policy; this resolves them and falls back to the embankment figures for an
    unrecognised type, which is the conservative direction.
    """
    try:
        cfg = get_type(ew_type)
    except KeyError:
        return (SPILLWAY_MIN_FREEBOARD_M, 0.30, SPILLWAY_TYPICAL_HEAD_M)
    return (
        getattr(cfg, "spillway_freeboard_m", SPILLWAY_MIN_FREEBOARD_M),
        getattr(cfg, "spillway_head_m", 0.30),
        getattr(cfg, "spillway_head_band", SPILLWAY_TYPICAL_HEAD_M),
    )

# Elevation comparisons are made to the millimetre. Without this a crest clamped
# to exactly the highest value spillway_datum offers reports as *insufficient*,
# because rim − head − freeboard does not reconstruct head + freeboard in binary
# floating point. Sub-millimetre setting-out is meaningless on a DEM anyway.
_ELEV_EPS = 0.001

# The crest, head and freeboard fields all step in centimetres, and elevations are
# reported to two decimals. A shortfall finer than that is one the user has no way
# to correct — nudging the crest down by one step overshoots — and the warning
# would print "only 0.30 m ... needs 0.30 m", every figure rounding to the same
# number while the message insists they differ.
#
# So the fit test is made at the precision the design is actually expressed in.
# That also keeps the message honest: a warning can now only fire when the two
# figures differ by at least one displayed unit.
_CREST_FIT_EPS = 0.01


class Spillway:
    """Where a feature is *designed* to overflow, and how wide that has to be.

    Two numbers describe one crest, and the dialog binds them both ways.
    ``crest_elevation`` is absolute; ``drop_below_rim_m`` is how far it sits below
    the rim — the lowest containing ground, which is where the feature would spill
    if nothing were built. The rim is the datum because it is the elevation the DEM
    actually supplies: an absolute crest typed without reference to it is
    unanchored, and a drop is meaningless without it.

    ``auto`` means a crest seeded from the ground is still the tool's rather than the
    user's, so placing the spillway on the map may re-read it from the DEM. It does
    **not** re-derive the crest when the DEM or the footprint changes: the only reader
    is ``_on_spillway_placed``, and a saved crest is otherwise re-clamped into the band
    only when the dialog next opens.
    """

    def __init__(self, crest_elevation=None, drop_below_rim_m=None, head_m=0.30,
                 width_m=0.0, point_wkt=None, auto=True, width_auto=True,
                 freeboard_m=None):
        self.crest_elevation = crest_elevation
        self.drop_below_rim_m = drop_below_rim_m
        self.head_m = head_m
        # Clear height demanded below the rim. ``None`` inherits the feature type's
        # policy (see spillway_policy) rather than freezing today's number into the
        # saved design — the same "None means take the site/type default" convention
        # Earthwork.soil_name uses. A stored value is a deliberate user override.
        self.freeboard_m = freeboard_m
        self.width_m = width_m        # width as BUILT (or tracking, while width_auto)
        # Whether the built width tracks the computed requirement. Separate from
        # ``auto`` (which tracks the crest against the rim) because a user who has
        # committed to a dug width has not thereby fixed the crest, or vice versa.
        self.width_auto = width_auto
        self.point_wkt = point_wkt    # placed location, or None for "not sited yet"
        self.auto = auto

    _SERIAL_FIELDS = (
        "crest_elevation", "drop_below_rim_m", "head_m", "width_m",
        "width_auto", "point_wkt", "auto", "freeboard_m",
    )

    def to_dict(self):
        return {f: getattr(self, f, None) for f in self._SERIAL_FIELDS}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return None
        sp = cls()
        for field in cls._SERIAL_FIELDS:
            if field in data:
                setattr(sp, field, data[field])
        return sp

    def summary(self):
        if self.crest_elevation is None:
            return "no crest set"
        sited = "" if self.point_wkt else " · not sited"
        return f"crest {self.crest_elevation:.2f} m · {self.width_m:.1f} m wide{sited}"


def effective_freeboard_m(spillway, ew_type):
    """The freeboard actually in force — a stored override, else the type's policy.

    ``Spillway.freeboard_m`` is ``None`` on anything not deliberately overridden
    (including every design saved before the field existed), so the type policy is the
    normal answer and the stored value is the exception.
    """
    override = getattr(spillway, "freeboard_m", None) if spillway is not None else None
    if override is not None:
        return max(0.0, float(override))
    return spillway_policy(ew_type)[0]


def spillway_datum(rim_elevation, invert_elevation=None, head_m=0.30,
                   min_freeboard_m=SPILLWAY_MIN_FREEBOARD_M):
    """Crest elevations physically available on this feature — ``(lowest, highest)``.

    The ceiling is not the rim itself but ``rim − head − freeboard``: the crest has
    to sit low enough that a full design nappe still clears the containing ground.
    Raising the head therefore lowers the highest usable crest, which is exactly the
    trade-off the dialog needs to show.

    The floor is the feature's invert — a crest there stores nothing, which is
    degenerate rather than invalid, so it is the bound rather than an error.

    Returns ``(None, None)`` when the rim is unknown. ``highest < lowest`` is a
    meaningful answer: the feature is too shallow to pass that head at all.
    """
    if rim_elevation is None:
        return (None, None)
    rim = float(rim_elevation)
    highest = rim - max(0.0, float(head_m)) - max(0.0, float(min_freeboard_m))
    lowest = float(invert_elevation) if invert_elevation is not None else highest
    return (lowest, highest)


def bind_crest(rim_elevation, crest=None, drop=None, band=None):
    """Resolve the crest/drop pair from whichever one the user just changed.

    Give ``crest`` to derive the drop, or ``drop`` to derive the crest; passing both
    lets the absolute crest win. Returns ``(crest, drop)``, exact inverses of each
    other so a round trip through either control cannot drift.

    *band* is an optional ``(lowest, highest)`` from :func:`spillway_datum`. When
    supplied the crest is clamped into it **before** the partner value is computed,
    so the two controls never disagree after a clamp — the failure mode that makes
    hand-written two-way bindings creep apart.
    """
    if rim_elevation is None:
        return (crest, drop)
    rim = float(rim_elevation)

    if crest is not None:
        value = float(crest)
    elif drop is not None:
        value = rim - float(drop)
    else:
        return (None, None)

    if band is not None:
        lo, hi = band
        # An inverted band (too shallow for this head) has no satisfiable value;
        # clamping to either end would fabricate one, so leave the crest alone and
        # let spillway_validity say why.
        if lo is not None and hi is not None and hi >= lo:
            value = max(lo, min(hi, value))

    return (value, rim - value)


def spillway_validity(crest_elevation, rim_elevation, invert_elevation=None,
                      head_m=0.30, min_freeboard_m=SPILLWAY_MIN_FREEBOARD_M,
                      width_m=None, required_width_m=None,
                      standard_freeboard_m=None, typical_head_m=None,
                      feature_length_m=None):
    """Plain-language problems with a proposed spillway; empty list means fine.

    Every message quotes the numbers it is objecting to, because "invalid" on its
    own gives the user nothing to act on.

    *min_freeboard_m* is the margin actually in force — a type default, or a user
    override. *standard_freeboard_m* is what that type's policy asks for, supplied
    separately so a deliberate reduction can be named as a reduction; omit it and no
    such objection is raised. *typical_head_m* is the type's ordinary head band,
    defaulting to the embankment one. *feature_length_m* enables the "this weir does
    not fit on this feature" check; omit it and that check is skipped.

    All three are optional and default to the previous behaviour exactly, so existing
    callers are unaffected.
    """
    problems = []
    if crest_elevation is None or rim_elevation is None:
        return problems

    crest = float(crest_elevation)
    rim = float(rim_elevation)
    head = max(0.0, float(head_m))
    freeboard = max(0.0, float(min_freeboard_m))

    if crest > rim + _ELEV_EPS:
        problems.append(
            f"Crest {crest:.2f} m is above the lowest containing ground "
            f"({rim:.2f} m) — water will escape around the spillway before it "
            f"ever reaches the crest."
        )
    elif rim - crest < head + freeboard - _CREST_FIT_EPS:
        problems.append(
            f"Only {rim - crest:.2f} m between the crest and the rim, but "
            f"{head:.2f} m of head plus {freeboard:.2f} m freeboard needs "
            f"{head + freeboard:.2f} m. Lower the crest or design for less head."
        )

    if invert_elevation is not None and crest <= float(invert_elevation):
        problems.append(
            f"Crest {crest:.2f} m is at or below the floor "
            f"({float(invert_elevation):.2f} m) — the feature would hold nothing."
        )

    # A freeboard the user has cut below what this type asks for. Said plainly and
    # allowed: on stable ground a shallower margin can be a real decision. Zero is
    # not — it means the design nappe reaches the containing ground, so the spillway
    # stops being the control and the choice of where to overflow is given up.
    if standard_freeboard_m is not None:
        standard = max(0.0, float(standard_freeboard_m))
        if freeboard <= 0 < standard:
            problems.append(
                "Freeboard is zero — at design head the water surface reaches the "
                "surrounding ground, so it will leave there too and the spillway stops "
                f"being the control. This type is designed for {standard:.2f} m."
            )
        elif freeboard < standard - _ELEV_EPS:
            problems.append(
                f"Freeboard {freeboard:.2f} m is under the {standard:.2f} m this type "
                f"is designed for. That margin is what keeps the overflow at the place "
                f"you armoured rather than at whichever point of the rim is lowest."
            )

    lo, hi = typical_head_m if typical_head_m is not None else SPILLWAY_TYPICAL_HEAD_M
    if head > 0 and not (lo <= head <= hi):
        problems.append(
            f"Head of {head:.2f} m is outside the usual {lo:.2f}–{hi:.2f} m range. "
            + ("A low head needs a wide weir." if head < lo
               else "A high head cuts into freeboard and speeds up the outflow.")
        )

    if width_m is not None and required_width_m is not None and required_width_m > 0:
        if float(width_m) < float(required_width_m):
            problems.append(
                f"Built width {float(width_m):.1f} m is under the "
                f"{float(required_width_m):.1f} m the design flow needs at this head."
            )

    # Does the weir fit the thing it is cut into? Required width grows without bound
    # with the flow, and the usual way that happens is someone adding features upslope
    # long after this one was sized. Compared against the feature's whole characteristic
    # length — a deliberately generous test, so it catches nonsense rather than nagging
    # a wide basin. A weir as wide as the entire feature is not a spillway.
    if (feature_length_m is not None and required_width_m is not None
            and required_width_m > 0 and float(feature_length_m) > 0):
        if float(required_width_m) > float(feature_length_m):
            problems.append(
                f"This needs a {float(required_width_m):.1f} m weir, but the feature is "
                f"only {float(feature_length_m):.1f} m long — it cannot pass its own "
                f"design flow. Split the catchment upslope, add storage above it, or "
                f"design for more head."
            )

    return problems


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Earthwork:
    """Represents a single earthwork feature."""

    TYPE_SWALE = "swale"
    TYPE_BERM = "berm"
    TYPE_BASIN = "basin"
    TYPE_DAM = "dam"
    TYPE_DIVERSION = "diversion"

    def __init__(self, ew_type, geometry, name):
        self.type = ew_type          # 'swale' | 'berm' | 'basin' | 'dam' | 'diversion'
        self.geometry = geometry     # QgsGeometry
        self.name = name
        self.id = uuid.uuid4().hex   # stable identity (for overflow linkage; survives reorder)
        # Registry-seeded sizing defaults (the registry is the one-file home of
        # per-type policy); unknown types fall back to the historical 0.5 / 2.0.
        # Bottom width is the canonical stored cross-section field; side_slope is derived
        # from it (see the side_slope property). Seeded from the type's default batter so a
        # fresh feature reproduces its historical slope (channels default 1:1 → bottom = 1.0 m).
        try:
            cfg = get_type(ew_type)
            self.depth = cfg.default_depth
            self.top_width_m = cfg.default_top_width
            default_slope = cfg.default_side_slope
        except KeyError:
            self.depth = 0.5         # metres cut/raised (not used for dam)
            self.top_width_m = 2.0   # declared top width of cross-section (metres)
            default_slope = 1.0
        self.bottom_width_m = max(0.1, self.top_width_m - 2 * default_slope * self.depth)
        self.batter_run_m = 0.0      # basin only: horizontal inset to full depth (0 = vertical)
        self.companion_berm = False  # swales only
        self.crest_elevation = None  # dam only: absolute crest elevation (m)
        # Dam: extend the wall into the banks so it holds to its crest. Swale with a
        # companion berm: wrap the berm round both ends of the alignment, for the same
        # reason — a bank open at its ends impounds nothing on ground that falls along
        # it, however tall it is. Off by default for a dam (it is an idealisation of a
        # wall the user drew shorter); on by default for a swale berm, where keying in
        # is ordinary practice and the alternative silently holds no water.
        self.key_into_banks = (ew_type == "swale")
        # The companion berm as built, filled in by the burn. Not inputs: the crest is
        # whatever the spoil from the trench reaches to, so neither can be known until
        # the cut is. ``berm_height_m`` is (min, mean, max) of the bank's height over
        # the ground it stands on — a range because the crest is one elevation and the
        # ground is not, so a single figure describes neither end of a long run.
        self.berm_crest_elevation = None
        self.berm_height_m = None
        # Contour provenance: full contour polyline [(x, y), ...] when this feature was
        # born from a contour (Pick Segment / Full Contour). The reshape tool uses it to
        # slide endpoints ALONG the contour instead of free vertex dragging. None = freehand.
        self.source_contour_coords = None
        self.gradient_pct = 1.0      # diversion only: channel gradient (%)
        self.overflow_target_id = None  # user-intended overflow recipient (None = analytics decide)
        # Soil under THIS feature; None inherits the site-wide soil set on the Design
        # tab. Soil rarely reads uniform across a farm, and infiltration is the term
        # most sensitive to it, so a basin in a clay hollow can be sized honestly
        # without misrepresenting the loam elsewhere.
        self.soil_name = None
        # Designed overflow point — where water LEAVES. None means "not designed
        # yet": the feature still overflows, it just does so wherever the ground
        # happens to be lowest.
        self.spillway = None
        # Where water ENTERS from an upstream feature. A separate structure with a
        # separate job: an outflow is a weir sized to pass a peak, an inlet is a
        # protected entry that stops the incoming jet cutting the bank. Keeping them
        # apart also lets a connection be drawn between the two real points rather
        # than between two centroids.
        self.inflow_spillway = None
        self.enabled = True
        self.capacity_m3 = 0.0
        self.capacity_l = 0.0
        # What this feature impounds on the actual ground, measured by flooding its own
        # burn (``DEMBurner.feature_storage``). Kept *beside* ``capacity_m3`` and never in
        # place of it: the drawn figure stays reproducible by hand from the dimensions
        # above, which is what makes it checkable and what makes it survive a DEM change.
        #
        # This one is the honest one, and it is usually much larger — a companion berm
        # keyed into the banks holds water above natural ground and further up the hill
        # than the trench reaches, which no cross-section can predict. The live readout
        # sizes against it, because sizing against the drawn figure means every keyed
        # swale reports "full" while it still has most of its pond in hand, and the
        # designer enlarges a feature that did not need it.
        #
        # None until a DEM has been flooded for this feature. Not serialised: a terrain
        # number cached in a project file outlives the terrain that produced it.
        self.terrain_capacity_m3 = None
        # The pond's own diagnostics, from the same flood — how much of it stands above
        # natural ground, and how deep it stands against the bank holding it back.
        self.impounded_above_ground_m3 = None
        self.retained_depth_m = None

    # ------------------------------------------------------------------
    # Derived geometry fields
    # ------------------------------------------------------------------

    @property
    def side_slope(self):
        """Channel side slope as an H:V ratio, derived from the stored widths.

        side_slope = (top_width_m − bottom_width_m) / (2 × depth). 1.0 == 1:1.
        Returns 0.0 for zero depth (avoids divide-by-zero).
        """
        if self.depth <= 0:
            return 0.0
        return (self.top_width_m - self.bottom_width_m) / (2.0 * self.depth)

    @side_slope.setter
    def side_slope(self, value):
        """Set the slope by back-solving the canonical bottom_width_m at the current depth."""
        self.bottom_width_m = max(0.1, self.top_width_m - 2.0 * value * self.depth)

    @property
    def wall_slope(self):
        """Basin wall batter as an H:V ratio, derived from batter_run_m / depth.

        Returns 0.0 (vertical) for zero depth.
        """
        if self.depth <= 0:
            return 0.0
        return self.batter_run_m / self.depth

    @wall_slope.setter
    def wall_slope(self, value):
        """Set the wall batter by back-solving the canonical batter_run_m at the current depth."""
        self.batter_run_m = max(0.0, value * self.depth)

    @property
    def length_m(self):
        """Feature length in metres, derived from the geometry (single source of truth)."""
        return shapely_length(self.geometry)

    @property
    def buffer_radius_m(self):
        """Raster-burn buffer radius = top_width_m / 2."""
        return self.top_width_m / 2.0

    # backward-compat alias so existing code using ew.width keeps working
    @property
    def width(self):
        return self.top_width_m

    @width.setter
    def width(self, value):
        self.top_width_m = value

    def type_label(self):
        try:
            return get_type(self.type).label
        except KeyError:
            return self.type.capitalize()

    def summary(self):
        status = "" if self.enabled else " [OFF]"
        if self.type == "dam":
            elev_str = f"{self.crest_elevation:.1f} m" if self.crest_elevation is not None else "?"
            mode = "keyed" if getattr(self, "key_into_banks", False) else "as-drawn"
            cap_str = f" · {self.capacity_m3:,.0f} m³ ({mode})" if self.capacity_m3 else ""
            return f"{self.name} (Dam) — crest {elev_str}{cap_str}{status}"
        if self.type == "diversion":
            # Pass the stored bottom width, as the properties dialog does. Omitting it
            # dropped this label onto the legacy bed-width path while the dialog used
            # the stored geometry, so the feature list and the dialog quoted different
            # discharges — around 45% apart — for one and the same drain.
            q = calculate_diversion_discharge(
                self.depth, self.width, self.gradient_pct,
                bottom_width=self.bottom_width_m,
            )
            return f"{self.name} (Diversion) — {self.gradient_pct:.1f}% | Q={q:.3f} m³/s{status}"
        return f"{self.name} ({self.type_label()}) — {self.capacity_m3:.1f} m³{status}"


    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    # Scalar fields carried verbatim through a round trip. Geometry is handled
    # separately (WKT), and `id` is preserved so overflow links survive.
    _SERIAL_FIELDS = (
        "type", "name", "id", "depth", "top_width_m", "bottom_width_m",
        "batter_run_m", "companion_berm", "crest_elevation", "key_into_banks",
        "source_contour_coords", "gradient_pct", "overflow_target_id",
        "soil_name", "enabled", "capacity_m3", "capacity_l",
    )

    def to_dict(self):
        """Plain-data form of this earthwork, geometry as WKT.

        Earthworks previously lived only in memory: closing QGIS discarded an entire
        design, and the map layers mirrored just 5 of ~16 fields, so nothing could be
        reconstructed from a saved project either.
        """
        data = {f: getattr(self, f, None) for f in self._SERIAL_FIELDS}
        try:
            data["geometry_wkt"] = self.geometry.asWkt()
        except Exception:
            data["geometry_wkt"] = None
        # Nested rather than flat: the spillway is its own object with its own
        # round trip, and flattening it into the earthwork's fields would couple
        # the two schemas together for no gain.
        spillway = getattr(self, "spillway", None)
        data["spillway"] = spillway.to_dict() if spillway is not None else None
        inflow = getattr(self, "inflow_spillway", None)
        data["inflow_spillway"] = inflow.to_dict() if inflow is not None else None
        return data

    @classmethod
    def from_dict(cls, data, geometry_factory=None):
        """Rebuild an earthwork from :meth:`to_dict`.

        *geometry_factory* turns WKT back into a geometry object; defaults to
        ``QgsGeometry.fromWkt`` so the pure module stays importable without QGIS.
        Returns ``None`` when the geometry cannot be rebuilt — a feature without a
        location is not worth resurrecting.
        """
        if geometry_factory is None:
            from qgis.core import QgsGeometry
            geometry_factory = QgsGeometry.fromWkt

        wkt = data.get("geometry_wkt")
        if not wkt:
            return None
        geometry = geometry_factory(wkt)
        if geometry is None:
            return None

        ew = cls(data.get("type", "swale"), geometry, data.get("name", "Earthwork"))
        for field in cls._SERIAL_FIELDS:
            if field in ("type", "name") or field not in data:
                continue
            value = data[field]
            if value is not None:
                setattr(ew, field, value)
        ew.spillway = Spillway.from_dict(data.get("spillway"))
        ew.inflow_spillway = Spillway.from_dict(data.get("inflow_spillway"))
        return ew

    @property
    def outflow_spillway(self):
        """Alias for :attr:`spillway`, once inlets exist and the pair needs naming.

        Kept as an alias rather than a rename so projects saved before inlets existed
        still load: their ``spillway`` key is the outflow, which is what it always was.
        """
        return self.spillway

    @outflow_spillway.setter
    def outflow_spillway(self, value):
        self.spillway = value


class EarthworkManager:
    """Manages the list of earthworks for the current session."""

    def __init__(self):
        self._earthworks = []

    def add(self, earthwork):
        self._earthworks.append(earthwork)

    def remove(self, index):
        if 0 <= index < len(self._earthworks):
            self._earthworks.pop(index)

    def toggle(self, index):
        if 0 <= index < len(self._earthworks):
            self._earthworks[index].enabled = not self._earthworks[index].enabled

    def get(self, index):
        return self._earthworks[index]

    def get_all(self):
        return list(self._earthworks)

    def get_enabled(self):
        return [e for e in self._earthworks if e.enabled]

    def clear(self):
        self._earthworks.clear()

    def __len__(self):
        return len(self._earthworks)

    def to_json(self):
        """Serialise the whole design to a JSON string (for the QGIS project file)."""
        return json.dumps({
            "version": 1,
            "earthworks": [ew.to_dict() for ew in self._earthworks],
        })

    def from_json(self, text, geometry_factory=None):
        """Replace the current design with one restored from :meth:`to_json`.

        Malformed or partial data loads what it can rather than failing outright —
        losing one unreadable feature beats discarding an entire saved design.
        Returns the number of earthworks restored.
        """
        self._earthworks.clear()
        if not text:
            return 0
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return 0
        for item in payload.get("earthworks", []):
            try:
                ew = Earthwork.from_dict(item, geometry_factory=geometry_factory)
            except Exception:
                ew = None
            if ew is not None:
                self._earthworks.append(ew)
        return len(self._earthworks)


# ---------------------------------------------------------------------------
# Capacity and hydraulic calculations
# ---------------------------------------------------------------------------

def _resolve_bottom_width(bottom_width, top_width, depth):
    """Return the trapezoid bottom width, defaulting to the 1:1 derivation when unset.

    ``bottom_width is None`` reproduces the historical ``top_width − 2×depth`` (1:1 side
    slopes); an explicit value honours the feature's stored side slope. Clamped to ≥ 0.1 m.
    """
    if bottom_width is None:
        return max(0.1, top_width - 2 * depth)
    return max(0.1, bottom_width)


def channel_batter_run(ew):
    """Horizontal inset to full depth for *ew*, in metres — one derivation, two users.

    ``batter_run_m`` is a basin-only field, so a drawn channel always reported 0 and was
    burned as a rectangle at full depth however its cross-section was specified. But a
    channel's batter is not missing: it is implied by the two widths it already carries,
    ``(top_width − bottom_width) / 2`` per side. Deriving it here rather than in either
    caller is what keeps the burn and ``rasterisable_capacity`` describing the same hole
    — the last time those two disagreed about whether a batter had been cut, every swale
    on the site reported Measured a flat +50% above At-grid (Round 3).

    A polygon feature keeps its explicit ``batter_run_m``: a basin's walls are a design
    choice, not something to infer from a plan shape.
    """
    if getattr(ew, "type", None) == "basin":
        return _as_float(getattr(ew, "batter_run_m", 0.0))
    explicit = _as_float(getattr(ew, "batter_run_m", 0.0))
    if explicit > 0:
        return explicit
    top = _as_float(getattr(ew, "top_width_m", 0.0))
    bottom = _as_float(getattr(ew, "bottom_width_m", 0.0))
    if top <= 0 or bottom <= 0 or bottom >= top:
        return 0.0
    return (top - bottom) / 2.0


def calculate_capacity(ew_type, geometry, depth, width, companion_berm=False,
                       bottom_width=None, batter_run=None):
    """
    Calculate storage capacity of an earthwork.

    Swale — trapezoidal cross-section × length × 0.8 freeboard.
    Basin — battered-wall inset-prism volume × 0.8 freeboard (vertical when
    ``batter_run`` is None/0 — identical to the historical prism).
    Berm / Dam / Diversion — no storage, returns (0.0, 0.0).

    ``bottom_width`` is the trapezoid's bottom width (m). When ``None`` it is derived
    from the declared top width assuming 1:1 side slopes (``top_width − 2×depth``) —
    preserving historical behaviour. Pass ``Earthwork.bottom_width_m`` to honour the
    feature's stored side slope.

    ``batter_run`` (basins) is the horizontal inset to full depth
    (``Earthwork.batter_run_m``); side slope z = batter_run / depth.

    Returns (volume_m3, volume_l).
    """
    try:
        if not get_type(ew_type).has_capacity:
            return 0.0, 0.0
    except KeyError:
        return 0.0, 0.0

    if ew_type == "swale":
        length = shapely_length(geometry)
        top_width = width
        bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
        cross_section = trapezoid_section(top_width, bottom_width, depth).area

        if companion_berm and cross_section > 0:
            # The credit is the berm's own section — the spoil it is built from, at 75%
            # compaction — and nothing more. The old form credited ``h × top_width / 2``,
            # which exceeded the material available by T/(2h): a third larger at the
            # registry defaults, claiming ~14% more swale capacity than the berm has
            # earth to hold back.
            #
            # A **volume**, so it does not depend on the bank's shape: it is the same
            # whether the burn lays that earth as a triangular ridge or spreads it to a
            # level crest, which is what it now does. Only the *height* differs, and
            # that lives in ``berm_height_estimate``. Deliberately not the wedge the
            # bank impounds on falling ground, which is larger and DEM-dependent — the
            # design figure stays hand-checkable from the drawn dimensions.
            cross_section += berm_spoil_per_metre(depth, top_width, bottom_width)

        volume_m3 = cross_section * length * 0.8

    elif ew_type == "basin":
        # Battered walls via the shared inset-prism primitive (0.8 freeboard stays
        # here — policy lives in modules, geometry in core/sizing). NOTE: the burn
        # tier (_burn_basin) still carves a vertical drop this phase; the analytic/
        # burned divergence for battered basins is surfaced by the verification tier.
        area_m2 = shapely_area(geometry)
        perimeter_m = shapely_length(geometry)
        z = (batter_run / depth) if batter_run and depth > 0 else 0.0
        volume_m3 = basin_volume_battered(area_m2, perimeter_m, depth, z).volume * 0.8
    else:
        return 0.0, 0.0

    return round(volume_m3, 2), round(volume_m3 * 1000, 1)


# Fraction of the cross-section kept free of water as freeboard. A design allowance,
# not a physical limit — which is exactly why it must be reported separately from the
# geometry rather than baked into a single "capacity" the Verify stage then measures
# against and finds wanting.
FREEBOARD = 0.8


def capacity_breakdown(ew, cell_size=1.0, n_cells=None, terrain_storage_m3=None,
                       cut_m3=None, cell_area=None):
    """Split a feature's capacity into the numbers the Verify stage compares.

    A single "capacity" figure conflated three unrelated gaps, which is why a
    verification delta of −38% was uninterpretable — it mixed a burn error, a terrain
    effect and a design allowance into one percentage.

    ``design``      geometric × freeboard — the drawn shape you can check by hand
    ``geometric``   the exact drawn shape: "if I dug precisely this, how big is it?"
    ``rasterisable`` **what this feature impounds on this terrain**, measured by flooding
                    its own burn in isolation (``DEMBurner.feature_storage``)
    ``freeboard_m3``          design ← geometric (an allowance you chose)
    ``cut_m3``                the trench as the burn cut it, brim-full, no berm
    ``resolution_penalty_m3`` cut ← section (did the grid hold the section you drew?)
    ``impoundment_m3``        rasterisable ← section (what berm and hillside add)
    ``section_m3`` / ``berm_credit_m3``   what ``geometric`` is made of

    Verification compares measured ponding against ``rasterisable``. Both are now floods,
    so a non-zero delta means the *finished site* ponds differently from the feature
    alone — interaction with a neighbour, and nothing else.

    **``rasterisable`` used to mean "the drawn section at this cell size"** and was
    computed by integrating the burned footprint to its own rim. That cannot see water a
    companion berm holds above natural ground, which on a keyed swale is most of the
    storage: Swale 5 of the Quail Island design read 439 m³ against a pond of 1,095 m³.
    The name is kept because it is threaded through both renderers and the panel; what it
    measures is now the pond. The old quantity survives as ``cut_m3``, which still answers
    the narrow grid-fidelity question, and ``resolution_penalty_m3`` is measured against it
    — so that key means what it always claimed to.

    **Geometric is two things and used to say so nowhere.** For a swale with a companion
    berm, ``calculate_capacity`` adds the berm's own section — 0.75 × the trench section —
    so 43% of Swale 22's 524 m³ is water held *behind* a berm rather than *in* a trench.
    Splitting them is what lets the columns be read against each other.

    ``terrain_storage_m3`` is the measured pond; ``n_cells`` (the burned footprint's cell
    count) drives the pre-burn model fallback. Without either, nothing is claimed beyond
    the geometry.
    """
    geometric, _ = calculate_capacity(
        ew.type, ew.geometry, ew.depth, ew.width,
        getattr(ew, "companion_berm", False),
        bottom_width=getattr(ew, "bottom_width_m", None),
        batter_run=getattr(ew, "batter_run_m", None),
    )
    geometric = geometric / FREEBOARD if FREEBOARD > 0 else geometric
    design = getattr(ew, "capacity_m3", 0.0) or 0.0

    # The same figure with the berm credit withheld — the trench alone, which is what
    # the grid columns describe.
    section, _ = calculate_capacity(
        ew.type, ew.geometry, ew.depth, ew.width, False,
        bottom_width=getattr(ew, "bottom_width_m", None),
        batter_run=getattr(ew, "batter_run_m", None),
    )
    section = section / FREEBOARD if FREEBOARD > 0 else section

    # Barrier-impounded storage — a dam. There is no drawn cross-section to rasterise:
    # the shape of the water is the shape of the valley, and the analytic capacity is
    # *already* a flooded-volume computation over the same grid the burn uses. Passing
    # a dam through the trapezoid path yielded a geometric of 0 and a rasterisable of
    # whatever a channel of that width would hold, so a dam that verified perfectly
    # (2,148 m³ measured against 2,148 m³ designed) reported Δ +1712%.
    barrier = geometric <= 0 and design > 0
    if barrier:
        # A dam has no drawn cross-section, but it does have a pond, and since that pond
        # is now measured the same way every other feature's is, it can be reported.
        raster = float(terrain_storage_m3) if terrain_storage_m3 is not None else design
        return {
            "design": round(design, 2),
            "geometric": round(design, 2),
            "rasterisable": round(raster, 2),
            "freeboard_m3": 0.0,
            "cut_m3": None,
            "resolution_penalty_m3": 0.0,
            "impoundment_m3": 0.0,
            "section_m3": round(design, 2),
            "berm_credit_m3": 0.0,
            "barrier_impounded": True,
        }

    # The pond, measured. ``rasterisable_capacity`` remains the answer before any burn has
    # run — a model standing beside a burn is the arrangement that has gone wrong twice
    # (Round 3's phantom batter discount, then the narrow-footprint taper), so wherever a
    # measurement exists it wins.
    # A length and an area, separately: `cell_size` decides whether a feature is
    # sub-cell, `cell_area` converts a cell count to square metres. They are the
    # same number only on a square grid, and defaulting one from the other keeps
    # every caller that has only ever seen square cells working unchanged.
    if cell_area is None:
        cell_area = float(cell_size) ** 2

    if terrain_storage_m3 is not None:
        raster = float(terrain_storage_m3)
    elif n_cells:
        raster = rasterisable_capacity(
            n_cells, cell_area, ew.depth,
            ew.width, getattr(ew, "bottom_width_m", ew.width), cell_size,
            batter_run=channel_batter_run(ew),
        )
    else:
        raster = section

    # Against the drawn *trench*, because that is what was cut: this is the grid-fidelity
    # term and nothing else. Measured against the whole design it would carry the berm as
    # well, and a working berm would report as a resolution failure.
    penalty = round(cut_m3 - section, 2) if cut_m3 is not None else 0.0

    return {
        "design": round(design, 2),
        "geometric": round(geometric, 2),
        "rasterisable": round(raster, 2),
        "freeboard_m3": round(geometric - design, 2),
        "cut_m3": round(float(cut_m3), 2) if cut_m3 is not None else None,
        "resolution_penalty_m3": penalty,
        # What the bank and the hillside add beyond the drawn trench. On falling ground a
        # keyed berm holds water above natural ground and further up the slope than the
        # trench reaches, so this is routinely the larger half of the storage — and it is
        # the number a designer needs, because it is the part no cross-section predicts.
        "impoundment_m3": round(raster - section, 2),
        "section_m3": round(section, 2),
        "berm_credit_m3": round(geometric - section, 2),
        "barrier_impounded": False,
    }


def calculate_cut_volume(ew_type, geometry, depth, width, bottom_width=None):
    """
    Calculate the volume of soil excavated (cut) by an earthwork.

    Swale — trapezoidal cross-section (no freeboard) × length.
    Basin / Diversion — area × depth.
    Berm / Dam — 0 (these place material, not remove it).

    ``bottom_width`` — trapezoid bottom width (m); ``None`` derives it from 1:1 side slopes
    (historical behaviour). Applies to the swale branch. See ``calculate_capacity``.

    Returns cut volume in m³.
    """
    try:
        if not get_type(ew_type).has_cut:
            return 0.0
    except KeyError:
        return 0.0

    if ew_type == "swale":
        length = shapely_length(geometry)
        top_width = width
        bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
        cross_section = trapezoid_section(top_width, bottom_width, depth).area
        return round(cross_section * length, 2)

    if ew_type == "basin":
        area_m2 = shapely_area(geometry)
        return round(area_m2 * depth, 2)

    if ew_type == "diversion":
        # Diversion 'width' is the *bed* width here (a different convention from the swale
        # top width — reconciled in the calc pass). The section is now computed via the
        # shared trapezoid_section primitive so it can no longer diverge from the
        # capacity/discharge copies: a symmetric ±2×depth (1:1) section around the bed.
        length = shapely_length(geometry)
        bottom_width = max(0.05, width - 2 * depth)
        top_width = width + 2 * depth
        cross_section = trapezoid_section(top_width, bottom_width, depth).area
        return round(cross_section * length, 2)

    return 0.0


def calculate_fill_volume(ew_type, geometry, depth, width, companion_berm=False,
                          bottom_width=None):
    """
    Calculate the volume of material placed (fill) by an earthwork.

    Berm — triangular cross-section (1:1 slopes) × length.
    Swale + companion berm — companion berm fill (from volume conservation).
    Dam — wall footprint × depth (approximate).
    Others — 0.

    ``bottom_width`` — swale trapezoid bottom width (m) for the companion-berm branch;
    ``None`` derives it from 1:1 side slopes (historical behaviour).

    Returns fill volume in m³.
    """
    if ew_type == "berm":
        # TODO(feature-list): honour the stored side slope for a battered berm; currently
        # a fixed 1:1 triangle (base=2*depth, height=depth → area=depth²).
        length = shapely_length(geometry)
        cross_section = depth * depth
        return round(cross_section * length, 2)

    if ew_type == "swale" and companion_berm:
        # The spoil, placed. A volume, so the bank's shape does not enter into it —
        # see the note in calculate_capacity.
        length = shapely_length(geometry)
        return round(berm_spoil_per_metre(depth, width, bottom_width) * length, 2)

    if ew_type == "dam":
        length = shapely_length(geometry)
        return round(width * depth * length, 2) if depth and width else 0.0

    return 0.0


def burn_quantities(original, burned, cell_area_m2):
    """Cut and fill the burn actually performed, in m³ — ``{"cut_m3", "fill_m3"}``.

    The difference between the two elevation surfaces, summed. Not a model of the
    earthworks: the earthworks as the terrain model now has them.

    This exists because it does not agree with ``calculate_cut_volume``, and the gap is
    not an error in either. The analytic figure is ``section × length`` — what comes out
    if the ground is flat. The burn cuts to a **level** invert referenced to the natural
    pour point and builds to a **level** crest, so on real ground the trench is deeper
    than its design depth nearly everywhere and the bank is taller where the ground is
    lower. That is the point of a level structure — it is why a swale holds its design
    volume on a slope instead of a wedge — and the extra earthmoving is its price.

    On the Quail Island design the drawn sections imply 11,948 m³ of cut and the terrain
    model moves 20,096 m³. Quoting the first to a contractor is a 68% shortfall.

    **Site totals only.** Per feature would need earthmoving attributed between
    overlapping cuts and banks that lie outside their own footprints, and a figure
    invented by an attribution rule is worse than one honestly aggregated.
    """
    orig = np.asarray(original, dtype="float64")
    burn = np.asarray(burned, dtype="float64")
    diff = burn - orig
    return {
        "cut_m3": float(np.clip(-diff, 0.0, None).sum()) * cell_area_m2,
        "fill_m3": float(np.clip(diff, 0.0, None).sum()) * cell_area_m2,
    }


def berm_spoil_per_metre(depth, width, bottom_width=None):
    """Spoil available per metre of swale, in m³/m — trench section at 75% compaction.

    The one quantity every companion-berm figure is derived from. It is shape-independent,
    which is why the capacity credit and the fill volume were always right even while the
    height was wrong: both are volumes.
    """
    bottom_width = _resolve_bottom_width(bottom_width, width, depth)
    return trapezoid_section(width, bottom_width, depth).area * 0.75


def berm_height_estimate(depth, width, bottom_width=None):
    """Height of the companion berm **as the burn builds it**, on level ground.

    The bank is the trench's spoil spread across a band as wide as the swale's top
    width, to a *level* crest — so on flat ground its height is simply
    ``spoil per metre ÷ band width``.

    This used to return ``√(0.75 × section)``, the height of a 1:1 **triangular** ridge
    of the same volume. Both hold the same earth; they are not the same bank, and they
    are not the same height. For a 3.0 / 1.0 / 1.0 m swale the triangle is 1.22 m tall
    and the flat-topped bank the burner actually builds is 0.50 m over a 3 m band — so
    the dialog was quoting a figure the burn never produced, off by 2.4×.

    Level ground is the honest caveat: a real crest is one elevation over ground that
    varies, so its height above that ground varies with it, and keying the berm into
    the banks spreads the same spoil over a longer band and lowers it further. What the
    burn settled on is carried on the feature as ``berm_crest_elevation`` once an
    analysis has run — see ``core.sizing.level_crest_from_spoil``.
    """
    if width <= 0:
        return 0.0
    return round(berm_spoil_per_metre(depth, width, bottom_width) / width, 2)


def calculate_diversion_discharge(depth, width, gradient_pct, bottom_width=None):
    """
    Peak discharge capacity of a diversion drain (Manning's equation).

    Trapezoidal cross-section, n=0.025 (compacted earthen). Returns Q in m³/s.

    ``bottom_width`` — trapezoid bottom width (m):
      * ``None`` (default): legacy behaviour — ``width`` is the *bed* width, section is
        symmetric (bottom = width−2×depth, top = width+2×depth), i.e. z = 2 side slopes.
        TODO(feature-list): this bed-width convention differs from the swale top-width
        convention; reconcile in the calc pass.
      * explicit value: ``width`` is the *top* width and ``bottom_width`` the bottom, so the
        side slope (and hence the slant term ``sqrt(1+s²)×depth``) follow the stored geometry.
    """
    n = 0.025
    s = gradient_pct / 100.0
    if s <= 0 or depth <= 0 or width <= 0:
        return 0.0
    if bottom_width is None:
        # Legacy bed-width convention: a symmetric ±2×depth section around the bed.
        # Widening by 2×depth per side IS a z=2 batter, so each wall's slant length is
        # √(1+2²)·depth = √5·depth. The old code paired that z=2 area with a √2 (z=1)
        # slant, which understated the wetted perimeter and so overstated the hydraulic
        # radius by ~40% and Manning Q by ~25% — a diversion drain reported as able to
        # carry a quarter more than it can. Both terms now come from one section.
        bottom = max(0.05, width - 2 * depth)
        top = width + 2 * depth
        sec = trapezoid_section(top, bottom, depth)
        area = sec.area
        # bottom ≥ 0.05 and depth > 0 (guarded above) → wetted perimeter is always > 0.
        r = sec.hydraulic_radius
    else:
        # Stored-geometry path — width is the top width; the section (and hence the
        # wetted perimeter / hydraulic radius) follow the stored widths exactly.
        sec = trapezoid_section(width, max(0.05, bottom_width), depth)
        area = sec.area
        r = sec.hydraulic_radius
    q = manning_flow(area, r, s, n).discharge
    return round(q, 4)


# Broad-crested weir discharge coefficient, SI (Q = C·L·H^1.5 with Q in m³/s, L and H
# in m). Brater & King (1976), Table of broad-crested weir coefficients: at the
# 0.20–0.50 m head band this module treats as ordinary, and a crest breadth of 0.6 m or
# more — which any earthen dam crest is — C is 2.60–2.70 in English units, i.e.
# 1.44–1.49 SI. 1.45 sits in that band.
#
# The former 1.7 (≈3.08 English) is the sharp-crested/ideal ceiling: it only occurs at
# high head over a *short* crest. Applied to a wide earthen crest it over-states
# discharge and so under-states the width needed by about 1.7/1.45 ≈ 16 % — an
# under-sized spillway, which is the direction that breaches embankments.
BROAD_CRESTED_WEIR_C = 1.45


def calculate_spillway_width(peak_flow_m3s, head_m, weir_coeff=BROAD_CRESTED_WEIR_C):
    """
    Minimum spillway width — broad-crested weir formula.

    L = Q / (C × H^1.5)

    ``weir_coeff`` defaults to :data:`BROAD_CRESTED_WEIR_C` (1.45 SI, Brater & King
    for a wide crest at ordinary head). Returns 0.0 for non-positive head or flow —
    callers must read that as "invalid input", never as a designed width of zero.

    Returns spillway width in metres.
    """
    if head_m <= 0 or peak_flow_m3s <= 0:
        return 0.0
    return round(peak_flow_m3s / (weir_coeff * head_m ** 1.5), 2)


def head_for_width(peak_flow_m3s, width_m, weir_coeff=BROAD_CRESTED_WEIR_C):
    """How deep the water actually runs over a weir of *width_m* — the inverse.

    ``H = (Q / (C·L))^(2/3)``, the same weir equation solved the other way. Which
    direction applies is a question about the design, not the physics: while the width
    is still free the head is chosen and the width follows
    (:func:`calculate_spillway_width`); once a width is committed the head is whatever
    the flow makes it, and that is this.

    It is the number the freeboard has to be checked against in that case. A width
    committed under one flow and then asked to pass a larger one does not fail by
    "being too narrow" in any way the user can act on — it fails by standing the water
    deeper than planned and eating the margin under the rim.

    Returns ``None`` for non-positive input, deliberately *not* the ``0.0`` its sibling
    returns: a head of zero is a physically meaningful reading (nothing is flowing) and
    must not be confused with "these inputs say nothing".

    Note this is not an exact numerical inverse of :func:`calculate_spillway_width`,
    which rounds its answer to a centimetre — round-tripping a rounded width returns a
    head a few millimetres off. Callers must not feed that back into an elevation
    comparison; where the width is auto-tracked the head *is* the design head by
    construction and this function should not be called at all.
    """
    if width_m is None or peak_flow_m3s is None:
        return None
    if width_m <= 0 or peak_flow_m3s <= 0:
        return None
    return (peak_flow_m3s / (weir_coeff * width_m)) ** (2.0 / 3.0)


def effective_head_m(head_m, peak_flow_m3s=None, width_m=None, width_auto=True,
                     weir_coeff=BROAD_CRESTED_WEIR_C):
    """The head this design will actually stand at — the one freeboard is spent on.

    Which of the two weir framings applies is decided here, once, so no caller has to
    get it right twice:

    * **Width still free** (``width_auto``) — the width is solved *from* the head, so
      the head is the design head by construction. Nothing to invert.
    * **Width committed and adequate** — a weir at or wider than the requirement passes
      the flow at or below the design head, so the freeboard budget is already met.
      Returning the design head is both correct and the conservative reading.
    * **Width committed and short** — the water stands deeper than planned. *This* is
      the case worth computing and worth warning about, and it is the only one where
      :func:`head_for_width` is used.

    Restricting the inverse to the deficient case is not merely tidy, it is required.
    :func:`calculate_spillway_width` rounds to a centimetre, so inverting a width that
    exactly matches the requirement returns a head a few millimetres *above* the design
    head — enough to push ``rim − crest`` past ``head + freeboard`` and accuse a
    freshly-seeded crest of a shortfall it does not have. Where the width is genuinely
    short the difference is real (a 2 m weir asked to pass a 4 m flow stands 0.48 m deep
    against a 0.30 m design), far beyond any rounding.

    Returns ``head_m`` unchanged whenever there is not enough information to say more.
    """
    if head_m is None:
        return None
    if width_auto or width_m is None or peak_flow_m3s is None:
        return head_m
    if width_m <= 0 or peak_flow_m3s <= 0:
        return head_m

    required = calculate_spillway_width(peak_flow_m3s, head_m, weir_coeff=weir_coeff)
    if required <= 0 or float(width_m) >= required:
        return head_m
    actual = head_for_width(peak_flow_m3s, width_m, weir_coeff=weir_coeff)
    return head_m if actual is None else actual


# ---------------------------------------------------------------------------
# DEM burning
# ---------------------------------------------------------------------------

class DEMBurner:
    """
    Burns earthwork geometries into a DEM copy and computes ponding.

    Each earthwork type modifies the DEM differently:
      Swale      → lower cells within the swale footprint
      Berm       → raise cells within the berm footprint
      Basin      → lower cells within the basin polygon
      Dam        → set cells to crest elevation
      Diversion  → graded channel (grade-controlled depth)
    """

    #: Holes are NaN in ``self.original`` and in everything derived from it.
    #:
    #: The sentinel is masked once, at the door, rather than guarded for at each of
    #: the twenty-odd places the array is read. A raw -9999 is not an absence, it is
    #: an elevation ten kilometres down, and it behaves like one: ``scipy.zoom`` used
    #: to interpolate it into its neighbours, producing values like -4974.5 that match
    #: no declared nodata, so ``fill_depressions`` filled a kilometre-deep phantom pit
    #: and per-feature storage came back orders of magnitude too large. ``np.mean``
    #: over a footprint clipped by a hole dragged a dam wall onto the upstream side.
    #: ``internal_relief`` reported 10,049 m of fall and warned about steep ground.
    #:
    #: NaN has the property the sentinel lacks: it propagates instead of pretending.
    #: Anything that reduces over the array uses a nan-aware form, and anything that
    #: *writes* one converts back — see :meth:`save`.
    def __init__(self, dem_path):
        with rasterio.open(dem_path) as src:
            self.original = src.read(1).astype("float32")
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.shape = self.original.shape
        if self.nodata is not None and np.isfinite(self.nodata):
            self.original[self.original == self.nodata] = np.nan
        # Which DEM this burner is *of*. Everything it writes inherits this grid, so a
        # caller holding a burner built from one DEM while the session points at
        # another produces rasters that cannot be compared with the baseline's — worth
        # being able to check for rather than discover in a shape mismatch downstream.
        self.dem_path = dem_path
        self.cell_size = abs(self.transform.a)
        self.cell_h = abs(self.transform.e)
        # The real area of one cell, not ``cell_size ** 2``. On a non-square grid the
        # two differ by cell_h/cell_w, and the volumes computed from them were
        # inconsistent with each other: ``feature_storage`` already measured against
        # ``abs(a * e)`` while spoil, trench storage and burned-cut used the square.
        # One design could therefore report a pond that held more than the hole that
        # made it, with nothing in the numbers to say which was wrong.
        self.cell_area = abs(self.transform.a * self.transform.e)
        # Non-fatal advisories raised during the last burn / ponding pass (sub-cell
        # features, resolution-cap degrade). The controller surfaces these to the
        # QGIS message bar — Strategy C is honest about what it approximates.
        self.warnings = []
        # {earthwork id: bool mask} from the last burn — the cells each feature actually
        # claimed, so the verification measures the hole that was cut rather than one it
        # re-derives on slightly different terms — and {earthwork id: m³} for what the
        # bare trench holds, brim-full, before any companion berm.
        self.burned_masks = {}
        self.burned_cut = {}
        self.burned_raised = {}

    def burn_earthworks(self, earthworks):
        """
        Apply all enabled earthworks to a copy of the original DEM.
        Returns modified DEM as float32 numpy array.

        Also records ``self.burned_masks`` — ``{earthwork id: bool mask}``, the cells each
        feature actually claimed. The verification used to re-derive these from the
        geometry on its own terms (buffering by ``max(width/2, cell_size)`` where the burn
        buffers by ``top_width/2``, and with ``all_touched`` where the burn no longer uses
        it), so the mask that set the At-grid reference was not the mask that was cut.
        Measuring the burn against a footprint the burn did not use is a difference that
        can only ever be noise in the answer.
        """
        modified = self.original.copy()
        self.warnings = []
        self.burned_masks = {}
        self.burned_cut = {}
        self.burned_raised = {}
        _dispatch = {
            "swale":     self._burn_swale,
            "berm":      self._burn_berm,
            "basin":     self._burn_basin,
            "dam":       self._burn_dam,
            "diversion": self._burn_diversion,
        }
        for ew in earthworks:
            if not ew.enabled:
                continue
            shapely_geom = self._to_shapely(ew.geometry)
            if shapely_geom is None:
                continue
            burn_fn = _dispatch.get(ew.type)
            if burn_fn is None:
                continue
            modified = burn_fn(modified, shapely_geom, ew)
        return modified

    def save(self, array, output_path):
        """Save a DEM array to GeoTIFF with LZW compression.

        Holes go back out as the sentinel the file declares. Writing NaN under a
        ``nodata=-9999`` tag would leave the two disagreeing, and the next reader to
        mask on the declared value would find nothing to mask.
        """
        out = np.asarray(array, dtype="float32")
        if self.nodata is not None and np.isfinite(self.nodata):
            out = np.where(np.isfinite(out), out, self.nodata).astype("float32")
        with rasterio.open(
            output_path, "w",
            driver="GTiff", dtype="float32",
            crs=self.crs, transform=self.transform,
            width=self.shape[1], height=self.shape[0],
            count=1, nodata=self.nodata, compress="lzw",
        ) as dst:
            dst.write(out, 1)

    # ------------------------------------------------------------------ helpers

    def _ground_mean(self, mask):
        """Mean ground level under *mask*, holes excluded; ``inf`` when it is all hole.

        Used to choose which side of a line is the lower one. ``inf`` rather than NaN
        because that comparison has to be total: NaN makes every ``<`` False, so a
        side that is entirely hole would win or lose by whichever branch the code
        falls through to rather than by ground level. ``inf`` states the intent —
        a side we cannot see the ground of is never the lower one.

        A plain ``np.mean`` over the raw band was how a dam wall ended up on the
        upstream side of boundary-clipped sites: one -9999 cell is worth ten
        kilometres of fall and drags the mean under any real terrain beside it.
        """
        if mask is None or not mask.any():
            return np.inf
        vals = self.original[mask]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else np.inf

    def _to_shapely(self, qgs_geometry):
        try:
            return shapely_shape(json.loads(qgs_geometry.asJson()))
        except Exception:
            return None

    def _rasterize(self, shapely_geom, all_touched=True):
        """Footprint cell mask.

        ``all_touched=True`` claims every cell the geometry so much as brushes, which is
        the right answer to *"did we lose the feature?"* — a narrow or diagonal alignment
        keeps a continuous, connected footprint and goes on routing water. It is the
        wrong answer to *"how much earth came out?"*, and the volumetric burns pass
        ``False`` for exactly that reason.

        Measured on this project's stack, a buffered line rasterised with ``all_touched``
        comes out a near-constant **1.3 m wider than drawn**, whatever its length,
        sinuosity or bearing: a 3.0 m swale claims 4.3 m. On Swale 22 of the Quail Island
        design that is 628 cells over 149.8 m — an effective 4.19 m — and every cell of it
        was then levelled to full depth, so both the excavation and the storage read
        against a trench half again as wide as the one the user drew. Centre-based
        rasterising reproduces the drawn width to within 1%.

        The sub-cell fallbacks the burns already carry (nearest-cell path for a line,
        centroid cell for a polygon) cover what ``all_touched`` was introduced to
        protect: a feature too narrow to claim a cell centre still claims cells.
        """
        return rasterize_footprint(shapely_geom, self.shape, self.transform,
                                   all_touched=all_touched)

    def _band_chainage(self, mask, coords):
        """``(rows, cols, chainage)`` for the cells of *mask*, along the polyline.

        Chainage is the distance from the line's start to the closest point on it —
        what a graded invert needs in order to know how deep to cut each cell.

        Vectorised over segments × cells rather than looped, because it can be: an
        alignment has a handful of vertices and its band a few thousand cells, so the
        whole projection is a couple of small array operations. The obvious
        alternative, one ``line.project(Point(...))`` per cell, puts a shapely call
        back in an inner loop, which is what this was rewritten to remove.
        """
        rows, cols = np.nonzero(mask)
        xs = self.transform.c + (cols + 0.5) * self.transform.a
        ys = self.transform.f + (rows + 0.5) * self.transform.e

        pts = np.asarray(coords, dtype="float64")
        starts, ends = pts[:-1], pts[1:]
        deltas = ends - starts                             # (S, 2)
        seg_len2 = (deltas ** 2).sum(axis=1)               # (S,)
        seg_len = np.sqrt(seg_len2)
        cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))

        # Projection parameter of every cell onto every segment, clamped to the
        # segment so a cell beside a bend projects to the vertex, not past it.
        qx = xs[None, :] - starts[:, 0:1]
        qy = ys[None, :] - starts[:, 1:2]
        safe = np.where(seg_len2 > 0.0, seg_len2, 1.0)[:, None]
        t = np.clip((qx * deltas[:, 0:1] + qy * deltas[:, 1:2]) / safe, 0.0, 1.0)

        gap2 = (qx - t * deltas[:, 0:1]) ** 2 + (qy - t * deltas[:, 1:2]) ** 2
        nearest = np.argmin(gap2, axis=0)
        cell = np.arange(t.shape[1])
        chainage = cumulative[nearest] + t[nearest, cell] * seg_len[nearest]
        return rows, cols, chainage

    def _line_path_cells(self, line):
        """Connected in-bounds cell path along *line* (nearest-cell snap fallback)."""
        try:
            coords = list(line.coords)
        except (NotImplementedError, AttributeError):
            return []
        return line_cells(coords, self.transform, self.shape)

    def _record_berm(self, ew, berm):
        """Carry the built berm onto *ew*: its crest, and how tall it stands.

        A crest is one elevation and the ground under it is not, so the bank's height
        varies along its run — on this design by as much as 0.60 m to 1.73 m on a single
        swale. A mean would report 0.98 m for that and describe neither end, so the
        range travels with it and the panel prints all three.

        Measured only where a bank was actually built. Where the ground already stands
        above the crest the burn adds nothing (it fills, never cuts), and folding those
        zeros into a "height" would understate the structure that does exist.
        """
        mask, crest = berm if berm is not None else (None, None)
        if mask is None or crest is None or not mask.any():
            ew.berm_crest_elevation = None
            ew.berm_height_m = None
            return
        try:
            built = np.clip(crest - self.original[mask], 0.0, None)
            built = built[built > 1e-3]
            if built.size == 0:
                ew.berm_crest_elevation = None
                ew.berm_height_m = None
                return
            low, mean, high = (float(built.min()), float(built.mean()),
                               float(built.max()))
            ew.berm_crest_elevation = float(crest)
            ew.berm_height_m = (low, mean, high)
        except (AttributeError, TypeError, ValueError):
            return
        w = berm_variation_warning(ew.name, low, high, mean)
        if w:
            self.warnings.append(w)

    def _contact_mask(self, line, ew):
        """Cells a **barrier** touches its water at — not the cells it is built from.

        For a cut, the hole and the water are the same cells. For a barrier they are not:
        ``_downstream_footprint`` deliberately places the wall on the far side of the
        drawn line (the inner-wall convention — the drawn line is the wet face), so the
        wall band and the pool it impounds share no cell at all. Attributing ponding by
        the wall band therefore credits a dam with nothing: on the Quail Island design,
        Dam 15 and Dam 40 both measured 0 m³ against four-figure capacities, and the
        pools they hold went to whichever basin happened to overlap them.

        The drawn line is where the water stands, so a symmetric band on it — at least
        one cell wide, and ``all_touched`` because this is a contact question rather than
        a volume one — is what the pool can be recognised by.
        """
        radius = max(_as_float(getattr(ew, "width", 0.0)) / 2.0, self.cell_size)
        try:
            return self._rasterize(line.buffer(radius), all_touched=True)
        except Exception:
            return np.zeros(self.shape, dtype=bool)

    def _record_mask(self, ew, mask, dem=None, spill=None):
        """Remember what *ew* claimed, and — for a cut — how much the trench itself holds.

        ``burned_cut`` is the excavation filled to the **bare** pour point, before any
        companion berm: ``Σ (spill − floor) × cell_area`` over the footprint. It answers
        one narrow question — *did the grid hold the section I drew?* — and it is free,
        because both terms are already in hand at the end of a burn.

        It is deliberately **not** the feature's storage. A footprint integrated to its own
        rim minimum cannot see water standing above natural ground: a one-cell ring reports
        the lowest adjacent cell, which on a contour swale is the *uphill* lip, and water
        crossing that lip runs into rising ground and cannot leave. Swale 5 on the Quail
        Island design ponds to 69.60 m while its ring minimum is 68.88 m, so this figure
        read 439 m³ against a pond of 1,095 m³. Storage is a question about a pond, and it
        is answered by flooding one — see :meth:`feature_storage`.
        """
        key = getattr(ew, "id", None) or getattr(ew, "name", None)
        if key is None or mask is None:
            return
        self.burned_masks[key] = mask
        if dem is not None and spill is not None and mask.any():
            held = np.clip(spill - dem[mask], 0.0, None)
            self.burned_cut[key] = float(held.sum()) * (self.cell_area)

    def _record_raised(self, ew, mask):
        """Remember the ground *ew* raised, separately from what it claimed.

        A dam's recorded mask is its contact band and a swale's is its trench, so
        ``mask & raised`` finds a dam's wall and finds **nothing at all** for a swale — the
        companion berm sits beside the trench, not in it. Keyed swale berms were therefore
        never checked for overtopping, while dams were, even though a keyed berm impounds
        water for exactly the same reason and fails the same way.

        Kept as a second record rather than widened into ``burned_masks``, because that mask
        is what ``attribute_ponding_volume`` and the verification measure against: widening
        it would move which pool belongs to which feature and with it every bermed swale's
        Δ. This follows the precedent a barrier already sets, where ``_contact_mask`` is
        held apart from the volumetric mask — contact, volume and *raised* are three
        different questions and each wants its own answer.
        """
        key = getattr(ew, "id", None) or getattr(ew, "name", None)
        if key is None or mask is None or not mask.any():
            return
        prev = self.burned_raised.get(key)
        self.burned_raised[key] = mask if prev is None else (prev | mask)

    def _warn_sub_cell(self, name, min_dimension):
        """Record a sub-cell advisory if *min_dimension* is narrower than one cell."""
        w = sub_cell_warning(name, min_dimension, self.cell_size)
        if w:
            self.warnings.append(w)

    def _downstream_footprint(self, line, thickness):
        """Wall footprint offset to the downstream (lower) side of *line* (§7).

        The drawn dam line is the inner (wet-side) wall; all wall thickness is added
        downstream. Offsets the line by ``thickness/2`` to each side, keeps the side
        whose ground is lower, and buffers it into a wall band. Falls back to a
        centred buffer if the offset can't be built (short/degenerate lines).

        Returns ``(band, centreline)``. The centreline is what the wall gets *sealed*
        along: the band is rasterised at its drawn width, on cell centres, and a band
        much under one and a half cells wide can come out as cells that touch only at
        their corners — a barrier D8 flow walks straight through. Burning the
        centreline's cell path as well closes those gaps without widening the wall,
        because the path runs down the middle of the band it is sealing.
        """
        half = thickness / 2.0
        try:
            left = line.parallel_offset(half, "left")
            right = line.parallel_offset(half, "right")
            left_mask = self._rasterize(left.buffer(half))
            right_mask = self._rasterize(right.buffer(half)) & ~left_mask
            left_mean = self._ground_mean(left_mask)
            right_mean = self._ground_mean(right_mask)
            if np.isinf(left_mean) and np.isinf(right_mean):
                return (line.buffer(half), line)
            chosen = left if left_mean <= right_mean else right
            return (chosen.buffer(half), chosen)
        except Exception:
            return (line.buffer(half), line)

    # ---------------------------------------------------------------- earthwork types

    def _burn_swale(self, dem, line, ew):
        """Excavate a swale to a **level invert** — it is storage, not a drain.

        A swale used to be cut at constant depth and then breached with
        ``enforce_monotonic_path``, which forced a strictly descending centreline.
        That guaranteed depression-filling would find an outlet, so a burned swale
        ponded essentially nothing however much capacity the panel credited it —
        most of the analytic-vs-terrain gap the Verify chip was reporting.

        **The cut is referenced to the natural pour point** — the lowest rim cell of the
        original ground — for two reasons. It is order-independent: read off the
        accumulating DEM instead, one rim cell that a neighbour has already trenched
        takes the whole floor down with it, and Swale 27 on the Quail Island design
        (which shares exactly one rim cell with Swale 25) was floored 0.66 m too deep and
        ponded 1.94 m for a 1.00 m design. And it makes *depth* mean what the user typed:
        referenced to the bermed rim instead, the trench deepens as its own berm grows,
        which is both circular — the berm is built from the spoil the trench yields — and
        a licence to over-excavate.

        The berm still goes in first and still raises the containing level; it is simply
        no longer allowed to move the floor. What it holds above natural ground is real
        storage, and it is large — but it is not measurable from this footprint, because a
        footprint has no rim high enough to contain it. :meth:`feature_storage` floods the
        finished feature and reports the pond; what is recorded here is the trench alone.
        """
        footprint = line.buffer(ew.buffer_radius_m)  # radius = top_width_m / 2
        mask = self._rasterize(footprint, all_touched=False)
        path_cells = self._line_path_cells(line)
        dem = dem.copy()

        if not mask.any():
            # Sub-cell: no cell centre is inside the buffer, so claim the path cells.
            mask = np.zeros(self.shape, dtype=bool)
            for rc in path_cells:
                mask[rc] = True

        berm = self._companion_berm(line, mask, ew) if ew.companion_berm else None
        if berm is not None:
            dem = self._add_companion_berm(dem, line, mask, ew, berm=berm)
            # The crest is an output of the burn, not an input to it — the spoil sets
            # it — so this is where the panel and the report can learn it, together with
            # how tall the bank that reaches it actually stands.
            self._record_berm(ew, berm)

        spill, _ = pour_point(self.original, mask, nodata=self.nodata)
        if spill is None:
            return dem
        relief = internal_relief(self.original, mask, nodata=self.nodata)
        dem = self._storage_invert(dem, footprint, mask, ew, spill,
                                   channel_batter_run(ew))

        # The bare pour point, deliberately: this records the trench, not the pond.
        self._record_mask(ew, mask, dem=dem, spill=spill)
        self._warn_sub_cell(ew.name, ew.bottom_width_m)
        self._warn_steep(ew, mask, relief, dem)
        return dem

    def _companion_berm(self, line, swale_mask, ew):
        """``(berm_mask, raise_height)`` for this swale's spoil berm, or ``(None, 0.0)``.

        Geometry only, and derived entirely from ``self.original`` — so it is the same
        answer whether it is being applied to the burn or consulted for the feature's
        spill datum, and it costs one computation rather than two.
        """
        berm_width = ew.top_width_m
        keyed = bool(getattr(ew, "key_into_banks", False))

        # Spoil first: the crest is whatever the excavated material reaches to, so the
        # cut has to be known before the bank can be placed. 0.75 accounts for
        # bulking/compaction losses.
        spill, _ = pour_point(self.original, swale_mask, nodata=self.nodata)
        if spill is None:
            return None, 0.0
        # The same taper the trench is cut with, or the bank is sized from earth the
        # excavation does not produce — a full-depth rectangle yields about half again
        # the spoil of the trapezoid that replaced it.
        reach = taper_reach(swale_mask, channel_batter_run(ew),
                            (self.cell_h, self.cell_size))
        floor = spill - ew.depth * (1.0 if reach is None else reach[swale_mask])
        cut_depths = np.clip(self.original[swale_mask] - floor, 0.0, None)
        # nansum: a hole in the footprint yields no spoil, because there is no ground
        # there to dig. A plain sum would return NaN for the whole feature, and NaN
        # passes the `spoil_m3 <= 0` guard in level_crest_from_spoil.
        spoil_m3 = float(np.nansum(cut_depths)) * (self.cell_area) * 0.75

        try:
            offset = ew.buffer_radius_m + berm_width / 2
            left_line = line.parallel_offset(offset, "left")
            right_line = line.parallel_offset(offset, "right")
            left_zone = left_line.buffer(berm_width / 2)
            right_zone = right_line.buffer(berm_width / 2)
        except Exception:
            return None, 0.0

        # Centre-based, like every other volumetric burn: the bank is a quantity of
        # earth spread over a band, and all_touched claims cells the band only brushes.
        # It also let the two offset bands overlap each other and the trench, so the
        # side that lost the tie was trimmed and the same spoil went up over a narrower
        # strip — the crest came out ~50% above the height the drawn dimensions predict,
        # on flat ground, where terrain explains none of it.
        left_mask = self._rasterize(left_zone, all_touched=False) & ~swale_mask
        right_mask = (self._rasterize(right_zone, all_touched=False)
                      & ~swale_mask & ~left_mask)

        if left_mask.any() and right_mask.any():
            berm_mask = (left_mask
                         if self._ground_mean(left_mask) < self._ground_mean(right_mask)
                         else right_mask)
        elif left_mask.any():
            berm_mask = left_mask
        elif right_mask.any():
            berm_mask = right_mask
        else:
            return None, 0.0

        if keyed:
            berm_mask = self._key_berm_into_banks(line, berm_mask, swale_mask, ew,
                                                  berm_width)

        crest = level_crest_from_spoil(self.original[berm_mask], self.cell_area,
                                       spoil_m3)
        if crest is None:
            return None, 0.0
        return berm_mask, crest

    def _key_berm_into_banks(self, line, berm_mask, swale_mask, ew, berm_width):
        """Close the berm around both ends of the swale, so the pool cannot go round it.

        A bank along the downhill side only is open at its ends. Water fills the trench,
        runs to whichever end is lower, and leaves — so the berm impounds nothing it is
        credited with, however tall it is built.

        **Not** ``extend_to_abutments``, which is the dam version of this idea: walk
        outward along the wall's own bearing until the ground rises to the crest. That
        works for a wall thrown across a valley, and does nothing for a swale, because a
        swale is laid *along* a contour — the ground off each end is at the same
        elevation as the ground under the line, so the walk runs its full 250 m and keys
        into nothing. Measured on planar 5% ground, keying that way changed the ponded
        volume by 0 m³.

        A return at each end does work: a cap of bank around each endpoint, reaching from
        the downhill berm across the end of the trench to its uphill lip, so the pool is
        enclosed on three sides and held by the fourth (the hillside itself).
        """
        try:
            from shapely.geometry import Point
            reach = ew.buffer_radius_m + berm_width
            coords = list(line.coords)
            caps = Point(coords[0]).buffer(reach).union(Point(coords[-1]).buffer(reach))
            wrapped = self._rasterize(caps) & ~swale_mask
        except Exception:
            return berm_mask
        return berm_mask | wrapped if wrapped.any() else berm_mask

    def _add_companion_berm(self, dem, line, swale_mask, ew, berm=None):
        """Build this swale's companion berm on *dem*, to a **level crest**.

        *berm* is a precomputed ``(mask, crest_elevation)`` from
        :meth:`_companion_berm`, so the datum pass and the burn pass share one
        derivation.

        Levelled, not raised. ``dem[mask] += height`` put the crest on the same slope as
        the ground under it, so the bank was highest where the water was shallowest and
        the pool simply ran out of its low end — the burn credited a berm with impounding
        water that a berm of that shape cannot hold. Measured on the Quail Island design,
        the raised bank ran 1.44–3.53 m tall against a declared 1.22 m, and it still
        impounded almost nothing on planar ground. ``np.maximum`` keeps it a fill: ground
        already above the crest stays where it is.
        """
        berm_mask, crest = (
            berm if berm is not None else self._companion_berm(line, swale_mask, ew))
        if berm_mask is None:
            return dem
        dem = dem.copy()
        dem[berm_mask] = np.maximum(dem[berm_mask], crest)
        # A keyed berm is a barrier, and the overtopping check can only find it here: the
        # feature's recorded mask is the trench, and the bank is beside it.
        self._record_raised(ew, berm_mask)
        return dem

    def _burn_berm(self, dem, line, ew):
        # Barrier: raise a flow-blocking ridge (never a cut). Incise-free — the
        # footprint band where resolvable, the nearest-cell path when sub-cell.
        footprint = line.buffer(ew.width / 2)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        if mask.any():
            dem[mask] += ew.depth
            self._record_raised(ew, mask)
        else:
            for rc in self._line_path_cells(line):
                dem[rc] += ew.depth
        self._record_mask(ew, self._contact_mask(line, ew))
        self._warn_sub_cell(ew.name, ew.width)
        return dem

    def _burn_basin(self, dem, polygon, ew):
        """Excavate a basin to a **level floor** at ``pour point − depth``.

        Previously ``dem[mask] -= depth`` — a translation, which kept the original
        ground slope, so the depression-filled pond was a wedge spilling at the
        lowest rim rather than a prism. That is why a basin could read "full · 581 m³"
        on the panel while the simulated pool was a small blocky corner: on 10% ground
        a nominal 600 m³ basin actually held about 210 m³.

        Battered walls (``batter_run_m``) are burned as nested steps, so the grid
        represents as much of the batter as its cell size allows.

        The datum comes from the **original** ground, not the running array: a basin
        whose rim clips a neighbouring trench would otherwise be floored to that trench's
        depth and pond well past its design. Basin 39 read 4.45 m of water for a 1.50 m
        design that way.
        """
        mask = self._rasterize(polygon, all_touched=False)
        dem = dem.copy()

        if not mask.any():
            # A sub-cell polygon (or a sliver crossing no cell centre) used to burn
            # nothing at all, silently. Claim its centroid cell instead.
            try:
                c = polygon.centroid
                row, col = xy_to_rc(self.transform, c.x, c.y)
                if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                    mask[row, col] = True
            except Exception:
                return dem
            if not mask.any():
                return dem

        # Basins carried no sub-cell advisory at all, so a footprint smaller than a
        # cell claimed its full analytic volume with nothing to flag it.
        self._warn_sub_cell(ew.name, min_dimension(polygon))

        spill, _ = pour_point(self.original, mask, nodata=self.nodata)
        if spill is None:
            return dem
        relief = internal_relief(self.original, mask, nodata=self.nodata)

        dem = self._storage_invert(dem, polygon, mask, ew, spill,
                                   _as_float(getattr(ew, "batter_run_m", 0.0)))

        self._record_mask(ew, mask, dem=dem, spill=spill)
        self._warn_steep(ew, mask, relief, dem)
        return dem

    def _storage_invert(self, dem, footprint, mask, ew, spill, batter_run):
        """Cut *footprint* as the section it was drawn as — battered, not squared off.

        A drawn swale carries ``batter_run_m = 0`` (the field is basin-only), so every
        linear feature took ``level_invert`` and was cut as a **rectangle at full depth**
        however its cross-section was specified. Against a drawn 3.0 m / 1.0 m / 1.0 m
        section that is 3.0 m²/m of trench for a 2.0 m²/m design — half again as much
        earth, and half again as much storage credited to the terrain model. Passing the
        batter the widths already imply makes the burn cut what the user asked for.

        The taper comes from a distance transform rather than nested erosions, so it is
        the true trapezoid at any batter run and does not collapse when the steps would
        be narrower than a cell. See :func:`~terrainflow_assessment.modules.burn_strategy.
        tapered_invert`.
        """
        if batter_run and batter_run > 0:
            return tapered_invert(dem, mask, ew.depth, batter_run, spill,
                                  cell_size=(self.cell_h, self.cell_size))
        return level_invert(dem, mask, ew.depth, spill)

    def _warn_steep(self, ew, mask, relief, burned_dem):
        """Record the over-excavation advisory, quantified in real cubic metres."""
        depth = _as_float(getattr(ew, "depth", 0.0))
        try:
            # nansum for the same reason as the spoil above: cells with no ground
            # contribute no excavation, rather than making the whole figure NaN.
            cut = float(np.nansum(
                np.clip(self.original[mask] - burned_dem[mask], 0.0, None)
            )) * (self.cell_area)
            storage = float(mask.sum()) * (self.cell_area) * depth
        except Exception:
            cut = storage = None
        w = steep_ground_warning(ew.name, relief, depth, cut, storage)
        if w:
            self.warnings.append(w)

    def _burn_dam(self, dem, line, ew):
        if ew.crest_elevation is None:
            return self._burn_berm(dem, line, ew)
        # Inner-wall convention (§7): wall thickness sits downstream of the drawn
        # line, raised to the crest.
        #
        # Rasterised on cell centres, like every other volumetric burn. ``all_touched``
        # claims every cell the band so much as brushes, which on a diagonal alignment
        # burned a 2.0 m wall 4.2 m thick — see :meth:`_rasterize`.
        footprint, centreline = self._downstream_footprint(line, ew.width)
        mask = self._rasterize(footprint, all_touched=False)
        dem = dem.copy()
        if mask.any():
            dem[mask] = np.maximum(dem[mask], ew.crest_elevation)
        # Always, not just when the band claimed nothing: it is the seal as well as the
        # sub-cell fallback, and a wall with a corner-only join in it is not a wall.
        for rc in self._line_path_cells(centreline):
            dem[rc] = max(dem[rc], ew.crest_elevation)
        # Keying in is ground, not a what-if. It used to be applied only in
        # :meth:`_keyed_dam_dem`, so a keyed dam's *capacity* was measured against a
        # wall that reached its abutments while every raster — the ponding layer, the
        # streams, the verification burn — got the short one as drawn. The pond then
        # filled to the natural saddle and ran round the ends of a dam the panel
        # reported 64% full.
        if getattr(ew, "key_into_banks", False):
            keyed, reach = self._key_dam_ends(dem, line, ew.crest_elevation)
            if keyed:
                self.warnings.append(
                    f"{ew.name}: crest sits above the natural bank — the wall is keyed "
                    f"{reach:.0f} m into each abutment. It has to be built into higher "
                    f"ground or water escapes around the ends."
                )
        self._record_mask(ew, self._contact_mask(line, ew))
        # The wall band itself, which is what the pond stands against. Recorded here as
        # well as via the contact mask so a dam and a keyed swale berm are found by the
        # same question rather than by two different ones.
        if mask.any():
            self._record_raised(ew, mask)
        self._warn_sub_cell(ew.name, ew.width)
        return dem

    def _burn_diversion(self, dem, line, ew):
        dem = dem.copy()
        coords = list(line.coords)
        if len(coords) < 2:
            return dem

        # The grade datum, taken at the first vertex that sits on mapped ground rather
        # than blindly at the first. A line starting in a nodata hole used to read the
        # sentinel as an elevation and grade the entire channel away from about
        # -10,000 m, burning a trench that deep along its whole length.
        start_elev = None
        for x0, y0 in coords:
            r0, c0 = xy_to_rc(self.transform, x0, y0)
            row0 = max(0, min(self.shape[0] - 1, r0))
            col0 = max(0, min(self.shape[1] - 1, c0))
            z0 = float(dem[row0, col0])
            if np.isfinite(z0):
                start_elev = z0
                break
        if start_elev is None:
            self.warnings.append(
                f"{ew.name}: no point along this drain has an elevation to grade from "
                f"— every vertex falls on ground the DEM does not cover. The drain was "
                f"left unburned rather than cut to a guessed level."
            )
            return dem

        cum_dist = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            cum_dist.append(cum_dist[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        total_length = cum_dist[-1]
        if total_length == 0:
            return dem

        gradient_frac = ew.gradient_pct / 100.0

        # The whole channel in one pass. This used to walk the line at three samples
        # per cell and, for *each* sample, buffer a point, rasterise it over the full
        # raster and run a full-array minimum — on the working DEM that is billions of
        # cell-touches for one drain, to burn a band a couple of cells wide.
        #
        # The band is the same shape either way: the samples overlapped heavily and
        # their union is the buffered line, which the mask comment below already noted.
        # So buffer once, rasterise once, and give each cell in the band its own invert
        # from how far along the line it sits.
        band = self._rasterize(LineString(coords).buffer(ew.width / 2.0))
        if not band.any():
            # Sub-cell channel: the buffer rasterised to nothing. The centreline's cell
            # path keeps the graded invert carving at least one connected cell, which is
            # what the per-sample nearest-cell snap did. Only when the band is empty —
            # adding it unconditionally would widen every channel by its centreline and
            # change which cells the breach below sees. Off-extent cells are not in the
            # path, so a drain off the DEM stays a no-op rather than burning an edge.
            for r, c in self._line_path_cells(line):
                band[r, c] = True

        if band.any():
            rows, cols, chainage = self._band_chainage(band, coords)
            burn = start_elev - chainage * gradient_frac - ew.depth
            # np.minimum, so a cell already lower keeps its level and a hole stays a
            # hole — NaN propagates rather than the drain inventing ground to cut.
            dem[rows, cols] = np.minimum(dem[rows, cols], burn)

        # A diversion IS a conveyance, so it gets the monotonic breach that swales
        # no longer do: the graded invert plus nearest-cell snapping can leave
        # one-cell humps that break connectivity and pond the drain.
        dem = enforce_monotonic_path(dem, self._line_path_cells(line))

        # The whole alignment, for the verification's benefit. Built here rather than
        # accumulated through the per-sample loop above: the samples overlap heavily and
        # a union of them is the same band.
        self._record_mask(ew, self._contact_mask(line, ew))
        # Bed (bottom) width of the trapezoidal channel drives the sub-cell check.
        self._warn_sub_cell(ew.name, max(0.05, ew.width - 2 * ew.depth))
        return dem

    def get_ponding_layer(self, modified_dem, transform=None):
        """
        Calculate ponding depth where water pools in the modified DEM.

        Compares modified DEM against a depression-filled version.
        Returns float32 array of ponding depth in metres (0 = no ponding).
        Auto-downsamples large DEMs to stay within memory limits.
        ``transform`` overrides the burner's own affine — pass the windowed
        transform when flooding a cropped sub-DEM (dam stage-storage).
        """
        import os
        import tempfile

        from pysheds.grid import Grid
        from rasterio.transform import Affine

        base_transform = transform if transform is not None else self.transform
        rows, cols = modified_dem.shape
        n_cells = rows * cols
        scale = 1.0

        # Resolution-aware cap: the cell cap is a memory guard, not a resolution
        # choice. When it trips we coarsen a *copy* to stay within memory, but the
        # DEM is never upsampled below native res (deferred Strategy B) — surface the
        # degrade as a warning rather than letting it happen silently (§3).
        cap_warning = ponding_resolution_warning(n_cells, _MAX_PONDING_CELLS)
        if cap_warning:
            self.warnings.append(cap_warning)

        if n_cells > _MAX_PONDING_CELLS:
            from scipy.ndimage import zoom as _zoom
            scale = (_MAX_PONDING_CELLS / n_cells) ** 0.5
            work_dem = _zoom(modified_dem, scale, order=1).astype("float32")
        else:
            work_dem = modified_dem

        # A hole survives the downsample as a hole. `order=1` propagates NaN into the
        # cells bordering one, so a ring of real terrain is lost around each hole; that
        # is the deliberate trade. Interpolating the raw sentinel instead — which is
        # what happened while the array carried -9999 — produced intermediate values
        # like -4974.5 that match no declared nodata, so pysheds saw ordinary ground a
        # kilometre down, `fill_depressions` filled the phantom pit, and baseline
        # ponding and per-feature storage came back orders of magnitude too large.
        hole = ~np.isfinite(work_dem)

        scaled_transform = Affine(
            base_transform.a / scale, base_transform.b, base_transform.c,
            base_transform.d, base_transform.e / scale, base_transform.f,
        )

        # The temp raster is what pysheds reads, so its holes have to be the sentinel
        # it is told to expect — NaN written under a numeric nodata tag is masked by
        # neither convention.
        write_nodata = (self.nodata if self.nodata is not None
                        and np.isfinite(self.nodata) else -9999.0)
        write_dem = (np.where(hole, write_nodata, work_dem).astype("float32")
                     if hole.any() else work_dem)

        tmp = tempfile.mktemp(suffix=".tif")
        try:
            with rasterio.open(
                tmp, "w", driver="GTiff", dtype="float32",
                crs=self.crs, transform=scaled_transform,
                width=work_dem.shape[1], height=work_dem.shape[0],
                count=1, nodata=write_nodata,
            ) as dst:
                dst.write(write_dem, 1)

            grid = Grid.from_raster(tmp)
            dem_raster = grid.read_raster(tmp)

            try:
                dem_raster = grid.fill_pits(dem_raster)
            except MemoryError:
                pass

            try:
                depression_filled = grid.fill_depressions(dem_raster)
            except MemoryError:
                _log.error("Ponding analysis failed (MemoryError) — returning empty layer.")
                return np.zeros_like(modified_dem)

            ponding = np.array(depression_filled, dtype="float32") - work_dem
            ponding = np.clip(ponding, 0, None)
            # No ground, no pond. Also keeps NaN out of the upsample below, which
            # would otherwise spread it a second time on the way back to full size.
            ponding[hole] = 0.0
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        if scale < 1.0:
            from scipy.ndimage import zoom as _zoom
            ponding = _zoom(
                ponding,
                (rows / ponding.shape[0], cols / ponding.shape[1]),
                order=1,
            )
            ponding = np.clip(ponding.astype("float32"), 0, None)

        return ponding

    def _key_dam_ends(self, dem, line, crest):
        """Extend the dam wall into the banks so it holds to its crest (idealised).

        A dam drawn only across the visible channel leaks around its ends: the pond's
        pour point becomes the natural abutment saddle, not the crest, so raising the
        crest above the saddle adds no storage (the depression-fill plateau). For the
        *analytical design tier* we key the wall into higher ground — stepping outward
        from each drawn endpoint along the line's bearing and raising cells to the crest
        until the natural terrain already reaches the crest (or we run off the DEM).
        Mutates *dem* in place; returns ``(keyed, max_reach_m)`` — ``keyed`` is True when
        either bank sat below the crest (so the wall had to be keyed in).
        """
        try:
            coords = list(line.coords)
        except (NotImplementedError, AttributeError):
            return (False, 0.0)
        if len(coords) < 2:
            return (False, 0.0)

        step = self.cell_size * 0.5  # half-cell keeps the raised path 4-connected
        max_reach = (self.shape[0] + self.shape[1]) * self.cell_size
        keyed = False
        max_used = 0.0
        for (ex, ey), (ix, iy) in ((coords[0], coords[1]), (coords[-1], coords[-2])):
            dx, dy = ex - ix, ey - iy  # outward bearing (inner → end)
            norm = (dx * dx + dy * dy) ** 0.5
            if norm == 0:
                continue
            dx, dy = dx / norm, dy / norm
            row, col = xy_to_rc(self.transform, ex, ey)
            # A hole at the abutment is not "below the crest": NaN fails the comparison,
            # so `keyed` stays False rather than being asserted off a -9999 that reads
            # ten kilometres down. You cannot key a wall into ground that is not there.
            # The walk below is unaffected — it never stopped on a hole either way.
            if 0 <= row < self.shape[0] and 0 <= col < self.shape[1] \
                    and self.original[row, col] < crest:
                keyed = True
            reach, x, y = 0.0, ex, ey
            while reach < max_reach:
                x += dx * step
                y += dy * step
                reach += step
                row, col = xy_to_rc(self.transform, x, y)
                if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                    break
                if self.original[row, col] >= crest:
                    break  # keyed into ground already above the crest
                dem[row, col] = max(dem[row, col], crest)
            max_used = max(max_used, reach)
        return (keyed, max_used)

    def _keyed_dam_dem(self, dam):
        """DEM with the dam raised to its crest and keyed into the banks (analytical)."""
        crest = dam.crest_elevation
        dem = self.original.copy()
        line = self._to_shapely(dam.geometry)
        if line is None:
            return dem
        # Same wall ``_burn_dam`` builds — drawn width on cell centres, sealed along the
        # band's centreline — so the estimate and the burn describe one structure.
        footprint, centreline = self._downstream_footprint(line, dam.width)
        mask = self._rasterize(footprint, all_touched=False)
        if mask.any():
            dem[mask] = np.maximum(dem[mask], crest)
        for rc in self._line_path_cells(centreline):
            dem[rc] = max(dem[rc], crest)
        keyed, reach = self._key_dam_ends(dem, line, crest)
        if keyed:
            self.warnings.append(
                f"{getattr(dam, 'name', 'Dam')}: crest sits above the natural bank — "
                f"keyed {reach:.0f} m into each abutment for the storage estimate; the "
                f"dam must be built into higher ground or water escapes around the ends."
            )
        return dem

    def _isolated_burn(self, ew, keep_warnings=False):
        """*(dem, mask)* for **this feature alone** on the original ground.

        Burning one feature into a fresh copy is what makes the storage figure
        order-independent: it is what this earthwork does to this hillside, not what it
        happens to do given whichever neighbours were burned before it.

        ``burn_earthworks`` resets the burner's per-feature records, so the site-wide
        burn's masks and trench volumes are snapshotted and put back. Warnings are
        discarded by default — the isolated pass regenerates advisories the real burn has
        already raised, and reporting a sub-cell swale twice helps nobody.
        """
        masks, cuts, warns = self.burned_masks, self.burned_cut, self.warnings
        raised = self.burned_raised
        try:
            dem = self.burn_earthworks([ew])
            key = getattr(ew, "id", None) or getattr(ew, "name", None)
            mask = self.burned_masks.get(key)
            if keep_warnings:
                warns = self.warnings
        finally:
            self.burned_masks, self.burned_cut, self.warnings = masks, cuts, warns
            self.burned_raised = raised
        return dem, mask

    def feature_storage(self, ew, baseline_ponding=None, isolated_dem=None,
                        isolated_mask=None, keep_warnings=False):
        """What *ew* impounds on this terrain, measured by flooding it in isolation.

        Returns a :class:`FeatureStorage`. This is the answer to "how much water does this
        earthwork hold", and it has to be a flood rather than an integral over the
        footprint, because a footprint cannot contain the pond. A companion berm keyed into
        the banks holds water **above natural ground**, standing both deeper than the trench
        and further up the hill than the trench reaches: on Swale 5 of the Quail Island
        design the footprint holds 439 m³ measured to its own rim and the pond is 1,095 m³,
        of which 334 m³ lies outside the footprint entirely. Integrating the footprint
        answers a question about a hole; this answers the question about the water.

        The pond is attributed **by region, not by window**. Flooding a crop and summing it
        would credit the feature with any natural hollow that happened to share the crop, so
        the new ponding is labelled and only the regions touching the feature's own burn
        mask are counted.

        ``baseline_ponding`` is the pre-earthwork ponding on the same grid — pass the cached
        baseline layer to save a flood pass. ``isolated_dem`` overrides the burn (the dam
        path passes its idealised keyed variant); ``isolated_mask`` goes with it.
        """
        from rasterio.transform import Affine
        from scipy.ndimage import binary_dilation
        from scipy.ndimage import label as _label

        if isolated_dem is None:
            isolated_dem, isolated_mask = self._isolated_burn(
                ew, keep_warnings=keep_warnings)
        if isolated_mask is None:
            key = getattr(ew, "id", None) or getattr(ew, "name", None)
            isolated_mask = self.burned_masks.get(key)
        if isolated_mask is None or not isolated_mask.any():
            return FeatureStorage(0.0, None, None, 0.0, 0.0)

        cell_area = self.cell_area
        rows, cols = self.shape
        r_lo, r_hi, c_lo, c_hi = self._feature_cell_bounds(ew)
        pad = _DAM_WINDOW_PAD_CELLS
        while True:
            r0, r1 = max(0, r_lo - pad), min(rows - 1, r_hi + pad)
            c0, c1 = max(0, c_lo - pad), min(cols - 1, c_hi + pad)
            win = (slice(r0, r1 + 1), slice(c0, c1 + 1))
            full_window = r0 == 0 and c0 == 0 and r1 == rows - 1 and c1 == cols - 1
            sub_t = self.transform * Affine.translation(c0, r0)

            pond = self.get_ponding_layer(isolated_dem[win], transform=sub_t)
            if baseline_ponding is not None:
                base = np.clip(
                    np.asarray(baseline_ponding, dtype="float64")[win], 0.0, None)
            else:
                base = self.get_ponding_layer(self.original[win], transform=sub_t)

            new_pond = np.clip(pond.astype("float64") - base, 0.0, None)
            labels, _ = _label(new_pond > _POND_MIN_DEPTH_M)
            own = set(np.unique(labels[isolated_mask[win]])) - {0}
            region = np.isin(labels, list(own)) if own else np.zeros_like(labels, bool)

            # Only *this feature's* pond has to be inside the window. A neighbouring
            # hollow clipped by the crop is not evidence that the crop is too small.
            if full_window or not region.any() or not _pond_touches_edge(
                    np.where(region, new_pond, 0.0)):
                break
            pad *= 2

        if not region.any():
            return FeatureStorage(0.0, None, None, 0.0, 0.0)

        volume = float(new_pond[region].sum() * cell_area)
        surface = isolated_dem[win][region] + new_pond[region]
        level = float(np.median(surface))

        # The part standing proud of natural ground — water the bank is holding rather
        # than the trench — and how deep it stands against the bank that holds it. Both
        # come free from arrays already in hand, and stage 3 warns on them.
        ground = self.original[win]
        above = np.clip(level - ground[region], 0.0, None)
        above_m3 = float(above.sum() * cell_area)
        raised = (isolated_dem[win] > ground + 1e-6) & binary_dilation(region)
        retained = float(np.clip(level - ground[raised], 0.0, None).max()) \
            if raised.any() else 0.0

        return FeatureStorage(volume, level, region, above_m3, retained)

    def feature_storage_m3(self, ew, baseline_ponding=None):
        """Just the volume from :meth:`feature_storage` — the common case."""
        return self.feature_storage(ew, baseline_ponding=baseline_ponding).volume_m3

    def dam_stage_storage(self, dam, baseline_ponding=None, key_into_banks=False):
        """Impounded volume (m³) of a dam = the new ponding it creates behind its crest.

        A thin wrapper over :meth:`feature_storage`, which is the same measurement
        generalised: burn the feature alone, flood a padded window, subtract the natural
        ponding, and count the pond it makes. This is the honest storage the dam holds — a
        short wall leaks around its ends, so raising the crest above the natural abutment
        saddle adds nothing (the signal to extend the dam). Pass ``key_into_banks=True`` for
        the *idealised* estimate, which extends the wall into higher ground
        (see :meth:`_keyed_dam_dem`) so it fills to the crest. Returns 0.0 for a dam
        without a crest.

        The parameter is now only a what-if: ``_burn_dam`` reads ``dam.key_into_banks``
        itself, so on a dam that carries the flag both paths key in and agree — which is
        the point, the capacity and the ponding raster have to describe one wall. Passing
        False for such a dam therefore does **not** buy an as-drawn measurement.
        """
        if getattr(dam, "crest_elevation", None) is None:
            return 0.0

        self.warnings = []
        if key_into_banks:
            # ``_keyed_dam_dem`` raises the abutment advisory the caller then surfaces, so
            # this path builds its own DEM and hands it over rather than re-burning.
            dem = self._keyed_dam_dem(dam)
            mask = self._contact_mask(self._to_shapely(dam.geometry), dam)
            return self.feature_storage(dam, baseline_ponding=baseline_ponding,
                                        isolated_dem=dem, isolated_mask=mask).volume_m3
        return self.feature_storage(dam, baseline_ponding=baseline_ponding,
                                    keep_warnings=True).volume_m3

    def _feature_cell_bounds(self, ew):
        """(row_lo, row_hi, col_lo, col_hi) of the feature geometry, clamped to the DEM."""
        rows, cols = self.shape
        line = self._to_shapely(ew.geometry)
        if line is None:
            return 0, rows - 1, 0, cols - 1
        minx, miny, maxx, maxy = line.bounds
        # transform.e is negative (north-up): larger y → smaller row, so the
        # two corners come back in an order that depends on the sign — sorted below.
        r_lo, c_lo = xy_to_rc(self.transform, minx, maxy)
        r_hi, c_hi = xy_to_rc(self.transform, maxx, miny)
        r_lo, r_hi = sorted((r_lo, r_hi))
        c_lo, c_hi = sorted((c_lo, c_hi))
        return (
            max(0, min(rows - 1, r_lo)), max(0, min(rows - 1, r_hi)),
            max(0, min(cols - 1, c_lo)), max(0, min(cols - 1, c_hi)),
        )
