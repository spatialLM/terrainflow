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
sill_limited_head_m           — the head the *width* is solved at: capped by the sill
adoptable_spillway_width      — that requirement made buildable, or refused
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
    SPILLWAY_MAX_REACH_M,
    berm_variation_warning,
    enforce_monotonic_path,
    level_invert,
    line_cells,
    notch_pool,
    ponding_resolution_warning,
    rasterisable_capacity,
    spillway_burn_width,
    spillway_notch,
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


class StageStorage(NamedTuple):
    """How much water this feature holds at any level — the stage–storage curve.

    ``levels_m``    ascending water-surface elevations, the last one being the spill level
    ``volumes_m3``  cumulative volume held at each of those levels
    ``area_m2``     wetted area at the top level, used only to extrapolate above it

    Built by :meth:`DEMBurner.feature_storage` from arrays it already has in hand — one
    sort and a cumulative sum over the pond it just flooded, no second pass. It has to be
    built there because ``region`` is in the flood window's frame and no caller can
    integrate it.

    Sampled rather than exact. The full curve is one point per pond cell, which on a
    forty-feature design is tens of megabytes retained on the model for a readout that is
    printed to the nearest cubic metre; the samples are taken at exact volumes and
    interpolated between, and the top sample is placed exactly at the spill level so
    ``volume_at(level_m)`` reproduces ``volume_m3`` and not something near it.

    The datum is the pond's own bed **net of any natural ponding**, the same basis
    ``volume_m3`` is measured on, so the two answer one question and not two.
    """

    levels_m: object
    volumes_m3: object
    area_m2: float

    def volume_at(self, elevation):
        """Volume (m³) held with the water surface at *elevation*.

        Zero at or below the lowest bed cell. Above the spill level the pond is no
        longer a pond — it is leaving — so this extrapolates on the top wetted area and
        the answer is a *what the hole would hold*, not a prediction.
        """
        if elevation is None:
            return None
        z = float(elevation)
        levels, volumes = self.levels_m, self.volumes_m3
        if len(levels) == 0:
            return 0.0
        if z <= float(levels[0]):
            return 0.0
        if z >= float(levels[-1]):
            return float(volumes[-1]) + (z - float(levels[-1])) * float(self.area_m2)
        return float(np.interp(z, levels, volumes))

    def level_at(self, volume_m3):
        """The water-surface elevation at which this pond holds *volume_m3*.

        :meth:`volume_at` read the other way round, and it is the question the event
        readout asks: the balance says how much water is in the feature, and what the
        user needs to know is whether that water reaches the sill. Monotone by
        construction, so it is one interpolation and never an iteration.

        Above the spill level it extrapolates on the top wetted area, matching
        :meth:`volume_at` — a feature the balance credits with more than its pond can
        hold is overflowing, and saying so with a level above the rim is the honest
        reading of a figure that is already past the model's edge.
        """
        if volume_m3 is None:
            return None
        v = max(0.0, float(volume_m3))
        levels, volumes = self.levels_m, self.volumes_m3
        if len(levels) == 0:
            return None
        if v <= float(volumes[0]):
            return float(levels[0])
        if v >= float(volumes[-1]):
            area = float(self.area_m2)
            if area <= 0:
                return float(levels[-1])
            return float(levels[-1]) + (v - float(volumes[-1])) / area
        return float(np.interp(v, volumes, levels))


class FeatureStorage(NamedTuple):
    """What one earthwork impounds, measured by flooding it alone on the original ground.

    ``volume_m3``        the pond it creates, over and above what ponds there naturally
    ``level_m``          the water surface it fills to — the level it actually spills at
    ``region``           bool mask of the pond, in the flood window's frame
    ``above_ground_m3``  the part standing proud of natural ground (the bank's work)
    ``retained_depth_m`` how deep the water stands against the bank that holds it
    ``excavation_m3``    the earth it takes out of this hillside — see :meth:`feature_storage`
    ``stage_storage``    the volume at any level below the spill level (:class:`StageStorage`)

    ``above_ground_m3`` and ``retained_depth_m`` are what separate a swale from a small
    dam, and neither is visible in a volume alone: a 1,095 m³ pond is unremarkable if it
    sits in a hollow and is a retaining structure if 1.0 m of it stands against a spoil
    bank.
    """

    volume_m3: float
    level_m: Optional[float]
    region: object
    above_ground_m3: float
    retained_depth_m: float
    excavation_m3: float = 0.0
    stage_storage: Optional[StageStorage] = None


#: Points on a retained stage–storage curve. Twenty-five centimetres of pond depth
#: sampled at 128 levels is under 2 mm a step, which is finer than the DEM can place a
#: water surface anyway; the cost is two 128-float arrays per feature.
_STAGE_SAMPLES = 128


def build_stage_storage(depths, level_m, cell_area, samples=_STAGE_SAMPLES):
    """Stage–storage curve for a pond of *depths* standing at *level_m*.

    *depths* is the water depth at each pond cell, so ``level_m - depth`` is that cell's
    effective bed — natural ground where there was none, the pre-existing water surface
    where there was. Working from depths rather than from the DEM is what makes the curve
    integrate to the *new* volume the caller reports rather than to the total standing
    water, which are different numbers wherever the site ponds naturally.

    Mirrors ``reporting.level_for_volume``, which solves the inverse by the same sorted
    array method; the two should stay recognisable as one technique.
    """
    depths = np.asarray(depths, dtype="float64").ravel()
    depths = depths[np.isfinite(depths) & (depths > 0.0)]
    if depths.size == 0 or level_m is None:
        return None

    bed = np.sort(float(level_m) - depths)
    n = bed.size
    # V(bed[k]) = sum_{i<k} (bed[k] - bed[i]) — accumulated one step at a time, because
    # each step raises the water over exactly the cells already wet.
    cum = np.empty(n, dtype="float64")
    cum[0] = 0.0
    if n > 1:
        np.cumsum(np.arange(1, n, dtype="float64") * np.diff(bed), out=cum[1:])
    cum *= float(cell_area)

    if n <= samples:
        levels, volumes = bed, cum
    else:
        # Even in *level*, not in cell index: the readout is asked for volumes at
        # elevations, and a pond's cells cluster round its bed.
        levels = np.linspace(bed[0], bed[-1], samples)
        idx = np.searchsorted(bed, levels, side="right") - 1
        idx = np.clip(idx, 0, n - 1)
        volumes = cum[idx] + (idx + 1) * (levels - bed[idx]) * float(cell_area)

    # The spill level itself is always the last sample, exactly, so the curve and the
    # volume the caller publishes agree at the one elevation both are read at.
    top = float(level_m)
    if top > float(levels[-1]) + 1e-12:
        k = n - 1
        top_volume = cum[k] + n * (top - bed[k]) * float(cell_area)
        levels = np.append(levels, top)
        volumes = np.append(volumes, top_volume)

    return StageStorage(levels, volumes, float(n) * float(cell_area))


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


def default_sill_depth_m(ew_type):
    """The depth a fresh sill on *ew_type* opens at — its head plus its freeboard.

    The same number the properties dialog seeds, expressed as a depth rather than as an
    elevation. ``_seed_spillway`` takes the top of :func:`spillway_datum`'s band, which is
    ``containment − head − freeboard``, so the drop below containment is ``head +
    freeboard`` — 0.30 m on a swale, 0.60 m on a dam or basin.

    It exists so the review table can seed its depth editor without re-deriving policy: a
    second expression of the same rule is a second place for it to drift, and the two
    surfaces disagreeing about what depth a feature starts at is exactly the class of
    quiet divergence :func:`bind_crest` exists to prevent.

    Type policy only, so it knows nothing about a per-feature ``freeboard_m`` override —
    which is correct where it is used, on a feature that has no spillway to carry one.
    """
    freeboard, head, _band = spillway_policy(ew_type)
    return max(0.0, float(head)) + max(0.0, float(freeboard))


def sill_limited_head_m(head_m, sill_depth_m):
    """The depth of flow the **width** is solved at: the design head, capped by the sill.

    The design head is what a type wants to pass, and it is deliberately held fixed so the
    freeboard readout can go negative and say the notch is too shallow. It is not, however,
    a depth this sill can run at. Water standing deeper than the notch is over the
    containing ground, not over the weir -- so ``min(design head, sill depth)`` is the most
    nappe the sill can actually offer, and it is the H that belongs in ``L = Q / (C*H^1.5)``.

    Sizing the width off the design head instead left the requirement frozen: 0.39 m at a
    1.00 m sill and 0.39 m at a 0.20 m one, when the shallow notch has barely half the depth
    to pass the same 92 L/s through and needs the best part of twice the width. The one
    control the user has could not move the one number it should move.

    **Only the width is sized this way.** The freeboard readout, the validity sentences,
    the stored ``head_m`` and :func:`effective_head_m`'s adequacy test all stay on the
    design head, so a sill too shallow for its storm still reads as one rather than quietly
    redefining its way into compliance.

    A *sill_depth_m* of ``None`` returns the head unchanged -- no datum, no cap. That is
    what every caller wants where the containment level is unknown, and it is what keeps
    the two tiers of :class:`EarthworksController` from disagreeing about whether the cap
    applies at all. A negative depth (a crest standing above its containment, which
    :func:`spillway_validity` reports as a fault) floors at zero rather than being fed into
    ``H^1.5``.
    """
    if head_m is None:
        return None
    if sill_depth_m is None:
        return head_m
    return min(float(head_m), max(0.0, float(sill_depth_m)))


def adoptable_spillway_width(required_m, cell_size=None, feature_length_m=None):
    """The width an auto sill is actually built at, or ``None`` where it cannot be built.

    Two constraints on a requirement before it becomes a number the terrain is cut to.
    The first is rasterisability, and it is :func:`spillway_burn_width`'s: a weir is burned
    to whole DEM cells. The second is the feature -- once the width is solved at the sill
    depth
    (:func:`sill_limited_head_m`) rather than at the design head, a very shallow notch
    asks for a very wide weir, and it asks without bound: 2.01 m at a 0.10 m sill, 22.43 m
    at 0.02 m, 63.45 m at 0.01 m. Nothing downstream clamps it. ``_scaled_bar`` re-cuts the
    sill bar to whatever width it is handed, so a requirement wider than the feature would
    burn a notch through the ground that is holding the water in.

    **Refused rather than clamped, and the distinction matters.** A clamped width would sit
    permanently and silently short of its own requirement, firing
    :func:`spillway_validity`'s shortfall sentence and its does-it-fit sentence together
    for the same feature forever. Returning ``None`` instead leaves the caller holding the
    last width that could actually be built, while the *requirement* is still reported in
    full and the fit check still says in words that the feature cannot carry it. It is the
    same idiom as the zero-head guard: a figure that says nothing is not written through.

    This is a sanity bound and not a construction ceiling. Shapely reports a polygon's
    perimeter as its length, so a basin's bound is generous, and a long swale's is the
    whole alignment -- and the bar is grown about the sill point, so one of exactly feature
    length still runs off the ends unless the sill sits at the midpoint. It catches the
    nonsense case. Where the notch can honestly be cut is a burn-tier question.

    *feature_length_m* is ``None`` from the dialog and ``0.0`` from ``Earthwork.length_m``
    on empty geometry; both mean "no bound".
    """
    from terrainflow_assessment.modules.burn_strategy import spillway_burn_width

    if required_m is None or float(required_m) <= 0:
        return None
    built = (spillway_burn_width(required_m, cell_size) if cell_size
             else float(required_m))
    if feature_length_m and float(feature_length_m) > 0:
        if float(required_m) > float(feature_length_m):
            return None
    return built


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

    **Three numbers describe one crest**, and the dialog binds all three (see
    :func:`bind_crest`). ``crest_elevation`` is absolute and is the authoritative one —
    it is what gets set out on the ground and what a burn would cut to; the other two
    are views of it against a datum, and either datum can move under a saved design.

    ``drop_below_rim_m`` measures down from the **containment level**: the lowest ground
    that actually holds this feature's water, which is where it would spill if no
    spillway were built. That is not always undisturbed ground — on a swale with a
    companion berm it is the berm crest, and on a dam it is the wall.

    ``height_above_floor_m`` measures **up from the feature's own floor**, and it is the
    only one of the three a builder can set out with a staff standing in the trench. It
    is also the one that survives a falling alignment: the containment level is a single
    global minimum over the whole footprint, usually at one end of a contour swale, so a
    notch sited mid-run and referenced to it is referenced to ground a hundred metres
    away. The floor is under the sill.

    Because the crest is authoritative, the other two are **re-derived whenever the
    datums are recomputed** — see :func:`rebase_spillway`, which the design-restore path
    runs so a stored drop cannot outlive the datum it was measured against.

    ``auto`` means a crest seeded from the ground is still the tool's rather than the
    user's, so placing the spillway on the map may re-read it from the DEM. It does
    **not** re-derive the crest when the DEM or the footprint changes: the only reader
    is ``_on_spillway_placed``, and a saved crest is otherwise re-clamped into the band
    only when the dialog next opens.
    """

    def __init__(self, crest_elevation=None, drop_below_rim_m=None, head_m=0.30,
                 width_m=0.0, point_wkt=None, auto=True, width_auto=True,
                 freeboard_m=None, height_above_floor_m=None,
                 width_required_m=None):
        self.crest_elevation = crest_elevation
        self.drop_below_rim_m = drop_below_rim_m
        # Crest measured up from the feature's floor. ``None`` means "not known on this
        # feature" — there is no floor for a dam (its invert is the ground under the
        # wall, not a cut), and a design saved before this field existed has none until
        # ``rebase_spillway`` runs against a DEM.
        self.height_above_floor_m = height_above_floor_m
        self.head_m = head_m
        # Clear height demanded below the rim. ``None`` inherits the feature type's
        # policy (see spillway_policy) rather than freezing today's number into the
        # saved design — the same "None means take the site/type default" convention
        # Earthwork.soil_name uses. A stored value is a deliberate user override.
        self.freeboard_m = freeboard_m
        self.width_m = width_m        # width as BUILT (or tracking, while width_auto)
        # What the design flow needs at this head, **before** the grid rounds it up.
        # Derived and never serialised: it is a function of the storm and the catchment,
        # both of which move under a saved design. It is kept beside the built width so
        # the rounding note can quote both figures without recomputing the requirement in
        # the sentence that describes it.
        self.width_required_m = width_required_m
        # Whether the built width tracks the computed requirement. Separate from
        # ``auto`` (which tracks the crest against the rim) because a user who has
        # committed to a dug width has not thereby fixed the crest, or vice versa.
        self.width_auto = width_auto
        self.point_wkt = point_wkt    # placed location, or None for "not sited yet"
        self.auto = auto

    # ``from_dict`` probes the data dict per field, so adding one here reads an older
    # document without a version gate. The bump on SCHEMA_VERSION is for the other
    # direction: an older build re-saving this design iterates its own shorter tuple
    # and drops the field silently, and the version is what lets that be noticed.
    _SERIAL_FIELDS = (
        "crest_elevation", "drop_below_rim_m", "height_above_floor_m", "head_m",
        "width_m", "width_auto", "point_wkt", "auto", "freeboard_m",
    )

    def to_dict(self):
        """Plain-data form.

        **An auto width is not written.** ``width_auto`` means the built width tracks
        the requirement, and the requirement is a function of the storm, the catchment
        and the routing — none of which the design file pins down. Storing the number
        anyway meant that opening an old project rewrote a width the user never chose,
        silently, on the first live recompute; once the grid rounds that width the change
        becomes *visible* and unexplained. So the flag is stored and the figure is
        derived, which is what the flag always claimed.

        A **committed** width (``width_auto`` False) is stored, rounded, exactly as
        before — that is a decision, and decisions are saved.
        """
        data = {f: getattr(self, f, None) for f in self._SERIAL_FIELDS}
        if self.width_auto:
            data.pop("width_m", None)
        return data

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return None
        sp = cls()
        for field in cls._SERIAL_FIELDS:
            if field in data:
                setattr(sp, field, data[field])
        # An auto width is absent from a document this build wrote and *present* in one
        # an older build wrote. Neither is a chosen figure, and the restore path's own
        # `_recompute_live_assessment` puts the derived one back before anything renders
        # — but until it runs, `width_m` has to be a number the map label, the sill bar
        # and the review row can render rather than a None they would each crash on.
        if sp.width_auto and not sp.width_m:
            sp.width_m = 0.0
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


def spillway_datum(containment_elevation, invert_elevation=None, head_m=0.30,
                   min_freeboard_m=SPILLWAY_MIN_FREEBOARD_M):
    """Crest elevations physically available on this feature — ``(lowest, highest)``.

    The ceiling is not the containment level itself but ``containment − head −
    freeboard``: the crest has to sit low enough that a full design nappe still clears
    the ground that holds the water in. Raising the head therefore lowers the highest
    usable crest, which is exactly the trade-off the dialog needs to show.

    *containment_elevation* is the level water is actually held to — the measured spill
    level where a burn has been run, else the companion berm's crest where one was
    built, else the bare ring minimum. It is **not** necessarily undisturbed ground, and
    that distinction is the point: a bermed swale holding to 69.60 m against a ring
    minimum of 68.88 m had its crest clamped 0.72 m too low while this was the ring, and
    the storage went with it.

    The floor is the feature's invert — a crest there stores nothing, which is
    degenerate rather than invalid, so it is the bound rather than an error.

    Returns ``(None, None)`` when the containment level is unknown. ``highest < lowest``
    is a meaningful answer: the feature is too shallow to pass that head at all.
    """
    if containment_elevation is None:
        return (None, None)
    rim = float(containment_elevation)
    highest = rim - max(0.0, float(head_m)) - max(0.0, float(min_freeboard_m))
    lowest = float(invert_elevation) if invert_elevation is not None else highest
    return (lowest, highest)


def bind_crest(rim_elevation, crest=None, drop=None, band=None,
               invert_elevation=None, height=None):
    """Resolve the crest / drop / height triple from whichever one the user changed.

    Returns ``(crest, drop, height)``:

    * ``crest``  — absolute elevation, the authoritative value
    * ``drop``   — how far it sits below *rim_elevation*, the **containment** level
    * ``height`` — how far it stands above *invert_elevation*, the feature's floor

    Give any one of ``crest`` / ``drop`` / ``height`` and the other two are derived.
    Passing more than one lets the absolute crest win, then the drop, then the height —
    the same "the more anchored value wins" order the two-way binding used.

    They are exact inverses of one another, so a round trip through any control cannot
    drift. A partner whose datum is unknown comes back ``None`` rather than guessed:
    without a DEM there is no containment level, and a dam has no floor to stand on.

    *band* is an optional ``(lowest, highest)`` from :func:`spillway_datum`. When
    supplied the crest is clamped into it **before** the partners are computed, so the
    three controls never disagree after a clamp — the failure mode that makes
    hand-written bindings creep apart. Clamping once, here, is the whole reason this is
    one function rather than three assignments at three call sites.
    """
    rim = None if rim_elevation is None else float(rim_elevation)
    invert = None if invert_elevation is None else float(invert_elevation)

    if crest is not None:
        value = float(crest)
    elif drop is not None and rim is not None:
        value = rim - float(drop)
    elif height is not None and invert is not None:
        value = invert + float(height)
    else:
        # Nothing resolvable — hand the inputs straight back rather than inventing a
        # crest from a datum that is not there.
        return (crest, drop, height)

    if band is not None:
        lo, hi = band
        # An inverted band (too shallow for this head) has no satisfiable value;
        # clamping to either end would fabricate one, so leave the crest alone and
        # let spillway_validity say why.
        if lo is not None and hi is not None and hi >= lo:
            value = max(lo, min(hi, value))

    return (value,
            None if rim is None else rim - value,
            None if invert is None else value - invert)


def rebase_spillway(spillway, containment_elevation, invert_elevation=None):
    """Re-derive a spillway's relative figures against datums measured *now*.

    ``crest_elevation`` is the value that must not move. The other two are measurements
    of it against ground, and both datums can change under a design that has been sitting
    on disk — the containment level moved when it stopped being a bare ring minimum, and
    ``height_above_floor_m`` did not exist at all before this build. A stored
    ``drop_below_rim_m`` left alone would then describe a rim nothing computes any more,
    while still looking like a setting-out figure.

    So: keep the crest, recompute the pair, and mutate in place. Returns *spillway* for
    convenience. A spillway with no crest yet is left entirely alone — there is nothing
    authoritative to re-base against, and a stored drop is then the only thing the user
    chose.
    """
    if spillway is None or getattr(spillway, "crest_elevation", None) is None:
        return spillway
    crest, drop, height = bind_crest(
        containment_elevation, crest=spillway.crest_elevation,
        invert_elevation=invert_elevation)
    spillway.crest_elevation = crest
    spillway.drop_below_rim_m = drop
    spillway.height_above_floor_m = height
    return spillway


def spillway_validity(crest_elevation, containment_elevation, invert_elevation=None,
                      head_m=0.30, min_freeboard_m=SPILLWAY_MIN_FREEBOARD_M,
                      width_m=None, required_width_m=None,
                      standard_freeboard_m=None, typical_head_m=None,
                      feature_length_m=None):
    """Plain-language problems with a proposed spillway; empty list means fine.

    Every message quotes the numbers it is objecting to, because "invalid" on its
    own gives the user nothing to act on.

    *containment_elevation* is the level this feature's water is actually held to — see
    :func:`spillway_datum`. It is deliberately not "the lowest natural ground": a crest
    standing above the ring minimum but below a measured berm crest is a legitimate
    design, and refusing it here would mark every bermed swale failed. The clearance
    against bare ground is worth saying and is said by :func:`spillway_notes`, which is
    a note rather than a problem because ``_spillway_row`` fails a row on any problem
    at all.

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
    if crest_elevation is None or containment_elevation is None:
        return problems

    crest = float(crest_elevation)
    rim = float(containment_elevation)
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
            f"Only {rim - crest:.2f} m between the crest and the containing "
            f"ground, but "
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


# Containment provenance, in the order :func:`spillway_datum`'s caller prefers them.
# Public because three surfaces render the same distinction and none of them should
# spell it themselves: the dialog, the Spillways review and the report.
CONTAINMENT_MEASURED = "measured"     # the pond's own spill level, off a burn
CONTAINMENT_BERM = "berm"             # the companion berm's crest, as built
CONTAINMENT_WALL = "wall"             # a dam: the wall crest the user specified
CONTAINMENT_LIP = "lip"               # bare ground: the ring minimum round the footprint


def spillway_notes(crest_elevation, lip_elevation=None, containment_elevation=None,
                   containment_source=None, berm_crest_elevation=None,
                   built_width_m=None, required_width_m=None):
    """Things worth saying about a crest that are **not** faults. Empty list means none.

    A separate channel from :func:`spillway_validity` on purpose. ``_spillway_row`` sets
    a row's state to ``"fail"`` on any non-empty ``problems``, so anything routed through
    that list is an accusation. These are the opposite: a crest above the bare ring
    minimum but under a measured berm crest is exactly what a bermed swale is *for*, and
    it used to be refused.

    One producer, several surfaces — the dialog, the review table and the report all
    render this list rather than each composing its own sentence about the same fact.

    *built_width_m* / *required_width_m* add the width-rounding sentence. The terrain
    model can only cut whole cells, so a weir is burned at the next cell up from what the
    flow needs. That is worth saying, and it is emphatically **not** a fault: the sill is
    at the same level either way, so it changes nothing the feature holds.
    """
    notes = []
    if crest_elevation is None:
        return notes
    crest = float(crest_elevation)

    # The clearance that used to be the fail. Said in the direction the user can act on:
    # what is holding the water up there, and how it was known.
    if lip_elevation is not None and crest > float(lip_elevation) + _ELEV_EPS:
        lip = float(lip_elevation)
        line = (f"Crest {crest:.2f} m stands {crest - lip:.2f} m above natural ground "
                f"({lip:.2f} m).")
        if containment_source == CONTAINMENT_MEASURED and containment_elevation is not None:
            line += (f" The last analysis measured this pond holding to "
                     f"{float(containment_elevation):.2f} m, so the water is held by what "
                     f"was built rather than by the hillside.")
        elif containment_source == CONTAINMENT_BERM and berm_crest_elevation is not None:
            line += (f" The companion berm was built to {float(berm_crest_elevation):.2f} m, "
                     f"which is what holds it — so this is a sill in made ground.")
        elif containment_source == CONTAINMENT_WALL:
            line += " The wall is the containment here, not the valley floor."
        else:
            line += (" Nothing measured is holding it there yet — run Re-analyse with "
                     "Earthworks to check what the built feature actually contains.")
        notes.append(line)

    # The rounding, stated as what it is. Deliberately not "the extra width lowers the
    # head": true of the weir equation, and it reads as though widening moved the water
    # level *in the feature*, which it does not. Width is horizontal; the level water
    # leaves at is vertical, and it is the crest.
    if built_width_m is not None and required_width_m is not None:
        built, need = float(built_width_m), float(required_width_m)
        if need > 0 and built > need + 0.005:
            notes.append(
                f"{built:.1f} m built — rounded up from the {need:.1f} m the flow needs, "
                f"so the terrain model can cut it. Water still leaves at the same level, "
                f"so this changes nothing the feature holds; a wider sill only runs the "
                f"overflow shallower."
            )

    return notes


# ---------------------------------------------------------------------------
# Spillway links - a diversion drain that starts where another feature spills
# ---------------------------------------------------------------------------
#
# A diversion drain normally takes its grade datum from the ground under its own first
# vertex, which is a guess about where the water it carries arrives. Where that water
# comes over a designed spillway the level is not a guess at all: it is the crest, and
# it is already on the design. The link says so.
#
# Stored on the **drain** as ``"<source id>:<kind>:<end>"``:
#
# * **source id** - the feature whose spillway supplies the level. An id and not a
#   boolean, because a flag cannot say *which* spillway, and every feature carries two.
# * **kind** - ``outflow`` or ``inflow``. Only an outflow is a source of water. An inlet
#   is where water *arrives*, so a drain attached to one is delivering rather than
#   taking and its start level would be at its far end, graded backwards; that case is
#   not built, and :func:`resolve_spillway_links` reports it rather than inventing it.
#   The token is here because the model really does carry two spillways per feature, and
#   a link that could not name which would be ambiguous the day the second case lands.
# * **end** - which end of *this drain* is attached: ``start`` (its first vertex) or
#   ``end`` (its last).
#
# **Why the end is recorded rather than the alignment reversed.** ``_burn_diversion``
# grades down from ``coords[0]``, so a drain linked at its far end has to be graded from
# the other direction, and there are two ways to arrange that. Reversing the drawn
# geometry on link is the smaller change to the burn and much the larger change to
# everything else: it mutates a geometry the user drew, it desynchronises
# ``source_contour_coords`` from the vertices it is supposed to describe, and unlinking
# would leave the alignment reversed - silently, with no undo stack anywhere in this
# plugin to put it back, which is the argument that already won for the spillway-removal
# confirmation. So the link carries the end and the burn reverses its own *working copy*
# of the coordinates, which leaves chainage, the path cells and the monotonic breach
# reading exactly the code they read today. Grading from the wrong end is not a visible
# failure: it produces a drain running uphill from an entirely plausible-looking level.

SPILLWAY_LINK_KINDS = ("outflow", "inflow")
SPILLWAY_LINK_ENDS = ("start", "end")


def format_spillway_link(source_id, kind="outflow", end="start"):
    """The stored form of a link, or ``None`` when there is nothing to store."""
    if not source_id:
        return None
    if kind not in SPILLWAY_LINK_KINDS:
        kind = "outflow"
    if end not in SPILLWAY_LINK_ENDS:
        end = "start"
    return f"{source_id}:{kind}:{end}"


def parse_spillway_link(value):
    """``(source_id, kind, end)`` for a stored link, or ``None`` if it is not one.

    A two-token value - the form a hand-edited file is most likely to hold - reads as
    ``start``, which is the end ``_burn_diversion`` has always graded from. An
    under-specified link therefore degrades to today's behaviour rather than to the
    opposite end of the drain.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.split(":")
    source_id = parts[0].strip()
    if not source_id:
        return None
    kind = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "outflow"
    end = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "start"
    if kind not in SPILLWAY_LINK_KINDS:
        return None
    if end not in SPILLWAY_LINK_ENDS:
        end = "start"
    return source_id, kind, end


def resolve_spillway_links(earthworks):
    """``({drain id: start datum}, [(drain name, why)])`` over *earthworks*.

    **Resolved at read time**, mirroring ``overflow_target_id`` -> ``resolve_targets``: a
    link that no longer names a live spillway falls back to the drain's own ground
    sample and is reported. Links are deliberately *not* cleared when the source is
    deleted - that is a second and silently different failure mode, in which the design
    quietly stops meaning what it said and there is nothing left to report.

    The datum is the source spillway's ``crest_elevation``, an absolute already carried
    on the design. That is what makes this independent of burn order, and it is the
    reason the level is not read off the burned surface: the notch is cut as a
    **post-pass** (see :meth:`DEMBurner.burn_earthworks`), so at the moment
    ``_burn_diversion`` runs, the source's spillway is not in the array yet. Reading the
    surface would put back exactly the order dependence this exists to remove.

    Every rejection carries its own reason, because they are different faults: a source
    that was deleted, one that is switched off, one whose spillway was cleared and one
    whose link points at an inlet are four different things for the user to do.
    """
    by_id = {}
    for ew in earthworks or []:
        key = getattr(ew, "id", None)
        if key is not None and key not in by_id:
            by_id[key] = ew

    inverts, dangling = {}, []
    for ew in earthworks or []:
        link = parse_spillway_link(getattr(ew, "spillway_link_id", None))
        if link is None:
            continue
        name = getattr(ew, "name", "This drain")
        source_id, kind, _end = link
        source = by_id.get(source_id)
        source_name = getattr(source, "name", "its source")
        if source is ew:
            dangling.append((name, "it is linked to itself"))
            continue
        if source is None:
            dangling.append(
                (name, "the feature it took its level from is no longer in the design"))
            continue
        if not getattr(source, "enabled", True):
            dangling.append((name, f"{source_name} is switched off"))
            continue
        if kind != "outflow":
            dangling.append(
                (name, f"it is linked to {source_name}'s inlet, which is where water "
                       f"arrives rather than where it leaves"))
            continue
        spillway = getattr(source, "outflow_spillway", None)
        crest = None if spillway is None else getattr(spillway, "crest_elevation", None)
        if crest is None:
            dangling.append((name, f"{source_name} no longer has a spillway crest"))
            continue
        try:
            inverts[getattr(ew, "id", None)] = float(crest)
        except (TypeError, ValueError):
            dangling.append((name, f"{source_name}'s crest is not a level"))
    inverts.pop(None, None)
    return inverts, dangling


def spillway_link_cycle(earthworks, extra=None):
    """Ids whose spillway links close a loop, or ``[]``.

    *extra* is ``(drain_id, source_id)`` for a link the user is proposing but which is
    not on the model yet, so the refusal can name the loop **before** it is made.

    A drain carries at most one link, so this is a functional graph and
    :func:`~terrainflow_assessment.modules.flow_graph.topological_order` is exactly the
    right tool - the same one ``on_connection_made`` uses for the overflow graph, and for
    the same reason: a loop is not merely unroutable, it is a drain that starts where it
    ends. It is refused here rather than assumed impossible, because nothing on the model
    stops a diversion from carrying a spillway of its own.

    Self-links are **not** reported by this function - ``topological_order`` skips an
    edge to its own node by construction. They are refused separately, at the point the
    link is made.
    """
    from terrainflow_assessment.modules.flow_graph import topological_order

    edges = {}
    for ew in earthworks or []:
        key = getattr(ew, "id", None)
        if key is None:
            continue
        link = parse_spillway_link(getattr(ew, "spillway_link_id", None))
        if link is not None:
            edges[key] = link[0]
    if extra:
        drain_id, source_id = extra
        if drain_id is not None:
            edges[drain_id] = source_id
    if not edges:
        return []
    _order, broken = topological_order(edges)
    return broken


def burn_order(earthworks):
    """*earthworks*, reordered so a linked drain's source is burned before the drain.

    The datum itself does not need this - it is an absolute off the design, not something
    read from the running array - but the drain's *cut* is applied with ``np.minimum``
    against whatever is already there, and its monotonic breach walks the surface as it
    stands. Burning the source first makes that reading the finished one, and removes a
    draw-order dependence that has been latent since diversions existed.

    **Stable and minimal.** A source is moved ahead only where it currently sits behind
    one of its drains; everything else keeps the order the user put it in. A design with
    no links therefore burns in exactly the order it burned before, which is what stops
    this from moving any published number. A cycle - refused when a link is made, but
    still possible in a hand-edited file - leaves its members in their original order
    rather than raising.
    """
    import heapq

    items = list(earthworks or [])
    position = {}
    for i, ew in enumerate(items):
        key = getattr(ew, "id", None)
        if key is not None and key not in position:
            position[key] = i

    blockers = {}          # position -> positions that must burn first
    unblocks = {}          # position -> positions waiting on it
    for i, ew in enumerate(items):
        link = parse_spillway_link(getattr(ew, "spillway_link_id", None))
        if link is None:
            continue
        src = position.get(link[0])
        if src is None or src == i:
            continue
        blockers.setdefault(i, set()).add(src)
        unblocks.setdefault(src, set()).add(i)
    if not blockers:
        return items

    remaining = {i: set(v) for i, v in blockers.items()}
    ready = [i for i in range(len(items)) if not remaining.get(i)]
    heapq.heapify(ready)
    out, placed = [], set()
    while ready:
        i = heapq.heappop(ready)
        out.append(items[i])
        placed.add(i)
        # ``j`` is in ``unblocks[i]`` only because ``i`` is in ``blockers[j]``, so it
        # always has an entry here and is pushed exactly once — when its last blocker
        # is discarded.
        for j in sorted(unblocks.get(i, ())):
            remaining[j].discard(i)
            if not remaining[j]:
                heapq.heappush(ready, j)
    # Cycle members, in the order they were given. Nothing here can be satisfied, so
    # holding the original order is the only answer that is not arbitrary.
    out.extend(items[i] for i in range(len(items)) if i not in placed)
    return out


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

    def __init__(self, ew_type, geometry, name, *, dims=None):
        self.type = ew_type          # 'swale' | 'berm' | 'basin' | 'dam' | 'diversion'
        self.geometry = geometry     # QgsGeometry
        self.name = name
        self.id = uuid.uuid4().hex   # stable identity (for overflow linkage; survives reorder)
        # Registry-seeded sizing defaults (the registry is the one-file home of
        # per-type policy); unknown types fall back to the historical 0.5 / 2.0.
        # Bottom width is the canonical stored cross-section field; side_slope is derived
        # from it (see the side_slope property). Seeded from the type's default batter so a
        # fresh feature reproduces its historical slope (channels default 1:1 → bottom = 1.0 m).
        #
        # ``dims`` is the user's standard cross-section (core.registry.earthwork_defaults),
        # resolved by the controller and passed in rather than read from here: this module
        # cannot see QgsSettings, and a parameter is what keeps ``from_dict`` out of it.
        # A restored feature passes nothing, so a saved design never reloads at whatever
        # the person opening it happens to prefer.
        if dims is not None:
            self.depth = dims.depth
            self.top_width_m = dims.top_width_m
            self.bottom_width_m = dims.bottom_width_m
        else:
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
        # The earth this takes out of the hillside, off the same isolated burn. Not the
        # same question as `capacity_m3` and not derivable from it: a level floor under
        # ground that rises away from the pour point digs well past the drawn section,
        # and that over-dig is most of what a contractor prices. None, never 0.0, until
        # a DEM has been burned — "moves no earth" is a claim, and an unmeasured feature
        # is not making it. Not serialised, for `terrain_capacity_m3`'s reason above.
        self.excavation_m3 = None
        # Where the finished pond was measured to let go — FeatureStorage.level_m from
        # the same flood. This is the honest containment datum once anything has been
        # built: the analytic ring minimum describes the hillside before the spoil went
        # anywhere, and on a bermed swale that is 0.72 m and 656 m³ out. None before any
        # measurement, and not serialised for the reason above it.
        self.terrain_spill_level_m = None
        # How far this feature's sill sits below that containment level, measured on the
        # last build of the spillway review. The width a sill needs is solved at the head
        # the sill can pass rather than at the design head (`sill_limited_head_m`), and
        # the two places that solve it — the drag-tier auto-width pass and the undersized
        # -spillway warning — deliberately do no DEM work, so this is the measurement they
        # read. `None` means "not measured", which correctly means "do not cap".
        # Derived; never serialised, for the reason above it.
        self.measured_sill_depth_m = None
        # What this feature holds at any level below that one — a StageStorage, off the
        # same flood. It costs one sort and a running sum on a pond already flooded, and
        # it is the only thing that can answer "what does a sill here give up" without
        # re-flooding on every spin of a crest control. Derived; never serialised.
        self.stage_storage = None
        # What it holds **brim-full**, with no spillway — the top of that same curve, and
        # the denominator of "% full". `terrain_capacity_m3` is the volume to the sill
        # once one is sited, so dividing by it would pin every spillwayed feature at 100%
        # exactly when the spillway starts doing its job. Derived; never serialised.
        self.containment_capacity_m3 = None
        # The three levels a spillway is judged by, all measured, all derived (see the
        # Spillways review). `burned_sill_elevation_m` is the highest level water has to
        # clear on its way through the notch, off the burned surface;
        # `actual_spill_level_m` is where the finished pond was found to let go once the
        # whole site was burned. They disagree when the notch did not do what it claimed.
        self.burned_sill_elevation_m = None
        self.actual_spill_level_m = None
        # Diversion only: the spillway this drain starts at, as
        # "<source id>:<kind>:<end>" — see the Spillway links section above for why it
        # is an id rather than a flag, and why the end of the drain travels with it
        # instead of the alignment being reversed. A link is a decision the user made,
        # so unlike everything else in this run of fields it *is* serialised.
        self.spillway_link_id = None
        # The level that link resolves to: the source spillway's crest. It is the datum
        # `_burn_diversion` grades **down from**, standing in for the ground sample it
        # would otherwise take at the drain's first vertex — so the drain's bed sits one
        # depth below it, exactly as it sits one depth below sampled ground. Not the bed
        # level itself, despite the name.
        #
        # Derived and never serialised, the same rule `terrain_capacity_m3` follows: it
        # is a function of another feature's crest, and a level cached in a project file
        # outlives the design that produced it. Re-derived by
        # `_refresh_spillway_link_inverts` before anything reads it.
        self.invert_start_m = None

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

    def clear_terrain_measurements(self):
        """Drop every figure measured off a DEM. Call when the terrain changes.

        These are the fields ``__init__`` declares as *derived and never serialised*,
        and the reason each gives for not being serialised — "a terrain number cached
        in a project file outlives the terrain that produced it" — is exactly as true
        of one cached on a live object across a DEM swap. ``invalidate_results()``
        clears the state-side results but cannot reach these, because they live on the
        ``Earthwork`` objects rather than on ``PluginState``; the design-file Open path
        re-measured afterwards and so was covered by accident, the DEM picker did not
        and was not. The visible symptom was `_spillway_datums` preferring
        ``terrain_spill_level_m`` — measured on the *old* terrain — as the containment
        ceiling, and the dialog labelling it "measured".

        Listed here rather than in the controller because this is where the fields are
        declared: a tenth derived field added above should be added here, and reading
        both lists side by side is what makes that obvious. Distinct from
        ``EarthworksController._clear_measured_levels``, which nulls the four
        *containment* fields for the narrower "there is no measurement" case and is
        called from five places that must keep the rest.
        """
        # The containment family: level, curve, brim volume, sill depth.
        self.terrain_spill_level_m = None
        self.stage_storage = None
        self.containment_capacity_m3 = None
        self.measured_sill_depth_m = None
        # What the isolated flood measured: the pond, its diagnostics, and the dig.
        self.terrain_capacity_m3 = None
        self.impounded_above_ground_m3 = None
        self.retained_depth_m = None
        self.excavation_m3 = None
        # What the *site* burn measured: the companion berm as built, and the two
        # spillway levels read off the burned surface.
        self.berm_crest_elevation = None
        self.berm_height_m = None
        self.burned_sill_elevation_m = None
        self.actual_spill_level_m = None

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
        # A user decision, so it is stored. It is also the reason SCHEMA_VERSION went
        # to 3: `from_dict` probes per field so an older document simply has none, but
        # an older build re-saving one of these designs iterates its own shorter tuple
        # and drops the link silently — and a drain that quietly goes back to guessing
        # its start level looks exactly like a drain that was never linked.
        "spillway_link_id",
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
    Calculate storage capacity of an earthwork — **the whole drawn shape, brim-full.**

    Swale — trapezoidal cross-section × length.
    Basin — battered-wall inset-prism volume (vertical when ``batter_run`` is
    None/0 — identical to the historical prism).
    Berm / Dam / Diversion — no storage, returns (0.0, 0.0).

    There is no freeboard allowance in this figure, and that is deliberate. It used to
    return the section less a blanket 20%, so the one number the plugin called capacity
    was a design *policy* wearing the name of a geometry — and it was the figure printed
    under "Capacity (geometric)" beside a measured pond, where the discount was large
    enough to flip the sign of the comparison: a swale holding 11% less than it was drawn
    to hold read as holding 11% more. Freeboard on a real feature is the height its
    spillway leaves between the design nappe and the crest, which is per-feature, in
    metres, and lives in ``spillway_policy`` / ``effective_freeboard_m``. A single
    site-wide fraction was never that, and subtracting one here only made the honest
    figure unavailable to everything downstream.

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

        volume_m3 = cross_section * length

    elif ew_type == "basin":
        # Battered walls via the shared inset-prism primitive. NOTE: the burn tier
        # (_burn_basin) still carves a vertical drop this phase; the analytic/burned
        # divergence for battered basins is surfaced by the verification tier.
        area_m2 = shapely_area(geometry)
        perimeter_m = shapely_length(geometry)
        z = (batter_run / depth) if batter_run and depth > 0 else 0.0
        volume_m3 = basin_volume_battered(area_m2, perimeter_m, depth, z).volume
    else:
        return 0.0, 0.0

    return round(volume_m3, 2), round(volume_m3 * 1000, 1)


def capacity_breakdown(ew, cell_size=1.0, n_cells=None, terrain_storage_m3=None,
                       cut_m3=None, cell_area=None, excavation_m3=None):
    """Split a feature's capacity into the numbers the Verify stage compares.

    A single "capacity" figure conflated unrelated gaps, which is why a verification
    delta of −38% was uninterpretable — it mixed a burn error, a terrain effect and a
    blanket design allowance into one percentage. The allowance is gone entirely now;
    the rest are separated here.

    ``geometric``   the exact drawn shape: "if I dug precisely this, how big is it?"
    ``rasterisable`` **what this feature impounds on this terrain**, measured by flooding
                    its own burn in isolation (``DEMBurner.feature_storage``)
    ``cut_m3``                the trench as the burn cut it, brim-full, no berm
    ``excavation_m3``         the earth that comes out of this hillside
    ``resolution_penalty_m3`` cut ← section (did the grid hold the section you drew?)
    ``impoundment_m3``        rasterisable ← section (what berm and hillside add)
    ``section_m3`` / ``berm_credit_m3``   what ``geometric`` is made of

    **``cut_m3`` and ``excavation_m3`` are not the same question and the report needs both.**
    ``cut_m3`` is ``Σ (spill − floor)`` — the trench filled to its own bare pour point, which
    is exactly the right yardstick for ``resolution_penalty_m3`` because it asks only whether
    the grid could hold the drawn section. ``excavation_m3`` is ``Σ (original − burned)``:
    the earth actually moved, which on falling ground is larger, because the burn cuts to a
    level invert and the ground rises away from the pour point. The two differ by that
    over-dig and by nothing else on flat ground, where they are equal.

    They were the same number until ``cut_m3`` was printed under a column headed *Cut
    (measured)* on the page that says to price the job from it — a trench void quoted as an
    excavation, understating a 10% cross-slope swale by 23%, and contradicting the site
    total in the table directly beneath it. The direction is not fixed: a basin dug into a
    natural hollow sits below its own rim, so there ``cut_m3`` reads the *higher* of the two.

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
    the geometry. ``excavation_m3`` likewise: absent, the key is ``None`` and the column
    degrades to an em dash rather than to a zero that would read as "moves no earth".
    """
    geometric, _ = calculate_capacity(
        ew.type, ew.geometry, ew.depth, ew.width,
        getattr(ew, "companion_berm", False),
        bottom_width=getattr(ew, "bottom_width_m", None),
        batter_run=getattr(ew, "batter_run_m", None),
    )
    # What the feature carries. For a channel or a basin this is the same figure
    # ``calculate_capacity`` just returned; for a dam it is the flood
    # ``_compute_dam_capacity`` cached, which is the only capacity a wall has.
    cached = getattr(ew, "capacity_m3", 0.0) or 0.0

    # The same figure with the berm credit withheld — the trench alone, which is what
    # the grid columns describe.
    section, _ = calculate_capacity(
        ew.type, ew.geometry, ew.depth, ew.width, False,
        bottom_width=getattr(ew, "bottom_width_m", None),
        batter_run=getattr(ew, "batter_run_m", None),
    )

    # Barrier-impounded storage — a dam. There is no drawn cross-section to rasterise:
    # the shape of the water is the shape of the valley, and the analytic capacity is
    # *already* a flooded-volume computation over the same grid the burn uses. Passing
    # a dam through the trapezoid path yielded a geometric of 0 and a rasterisable of
    # whatever a channel of that width would hold, so a dam that verified perfectly
    # (2,148 m³ measured against 2,148 m³ designed) reported Δ +1712%.
    barrier = geometric <= 0 and cached > 0
    if barrier:
        # A dam has no drawn cross-section, but it does have a pond, and since that pond
        # is now measured the same way every other feature's is, it can be reported.
        raster = float(terrain_storage_m3) if terrain_storage_m3 is not None else cached
        return {
            "geometric": round(cached, 2),
            "rasterisable": round(raster, 2),
            # Both cut figures stay absent for a wall. A dam's contact band is the drawn
            # line, not the footprint of the wall standing on it, so any excavation
            # integrated over it would be a number about the wrong shape.
            "cut_m3": None,
            "excavation_m3": None,
            "resolution_penalty_m3": 0.0,
            "impoundment_m3": 0.0,
            "section_m3": round(cached, 2),
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
        "geometric": round(geometric, 2),
        "rasterisable": round(raster, 2),
        "cut_m3": round(float(cut_m3), 2) if cut_m3 is not None else None,
        "excavation_m3": (round(float(excavation_m3), 2)
                          if excavation_m3 is not None else None),
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

    Summed with ``nansum``, as every other quantity taken off these surfaces is.
    Holes are NaN in ``self.original`` and in everything derived from it, so a plain
    ``.sum()`` returns NaN for the **whole site** the moment the DEM has one nodata
    cell — which a DEM clipped to a boundary always does. That NaN then reached the
    report's volume rounding, and ``int(round(nan))`` is what put "Export failed:
    cannot convert float NaN to integer" on the screen in place of a document. A
    cell with no elevation moved no earth.
    """
    orig = np.asarray(original, dtype="float64")
    burn = np.asarray(burned, dtype="float64")
    diff = burn - orig
    return {
        "cut_m3": float(np.nansum(np.clip(-diff, 0.0, None))) * cell_area_m2,
        "fill_m3": float(np.nansum(np.clip(diff, 0.0, None))) * cell_area_m2,
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
        # {earthwork id: bool mask} for the cells a designed spillway lowered. A
        # **fourth** record rather than a widening of ``burned_masks``: that mask is what
        # the pool attribution and the verification measure against, so widening it would
        # move which pool belongs to which feature and with it every bermed swale's Δ.
        # Contact, volume, raised and *notched* are four different questions.
        self.burned_notches = {}
        # {earthwork id: m} — the **as-burned sill**: the highest level water crossing
        # the notch has to clear, on the surface as it stands afterwards. Recorded even
        # where nothing was cut, because a refusal's own figure is what says how far the
        # bank still stands above the crest that was designed.
        self.burned_sills = {}

    def burn_earthworks(self, earthworks, sills=None):
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

        *sills* is ``{earthwork id: crest-bar WKT}`` — where each designed spillway's
        crest actually lies on its feature. The snap that produces it stays in the
        controller, because ``plan_geometry``'s docstring forbids a second implementation
        of "nearest point on this alignment", and this method receives only ``Earthwork``
        objects. A sill the controller could not resolve is simply absent from the dict,
        and ``_cut_spillway`` never sees the feature; the controller warns about that
        rather than this method inventing a location.

        **The notches are cut as a second pass, after the type dispatch.** Fills are
        ``np.maximum`` and ``_burn_berm`` is additive, so a notch cut inside a ``_burn_*``
        is plugged by that feature's own companion berm or by a later feature that
        overlaps it. A post-pass is the only place where *the notch is the last thing to
        touch these cells* is true by construction — and it stays order-independent all
        the same, because the level cut to is an absolute carried on the design rather
        than anything read off the running array.

        The type dispatch itself runs in :func:`burn_order`, so a diversion drain that
        takes its start level from another feature's spillway is cut **after** that
        feature. Its *datum* does not need that — it is an absolute too — but its cut is
        an ``np.minimum`` against the running array and its breach walks the surface as
        it stands, so burning the source first makes both read the finished ground. The
        reorder is stable and minimal: a design with no links burns in exactly the order
        it was given, which is what keeps this from moving any existing number.
        """
        modified = self.original.copy()
        self.warnings = []
        self.burned_masks = {}
        self.burned_cut = {}
        self.burned_raised = {}
        self.burned_notches = {}
        self.burned_sills = {}
        _dispatch = {
            "swale":     self._burn_swale,
            "berm":      self._burn_berm,
            "basin":     self._burn_basin,
            "dam":       self._burn_dam,
            "diversion": self._burn_diversion,
        }
        burned = []
        for ew in burn_order(earthworks):
            if not ew.enabled:
                continue
            shapely_geom = self._to_shapely(ew.geometry)
            if shapely_geom is None:
                continue
            burn_fn = _dispatch.get(ew.type)
            if burn_fn is None:
                continue
            modified = burn_fn(modified, shapely_geom, ew)
            burned.append(ew)

        for ew in burned:
            modified = self._cut_spillway(modified, ew, (sills or {}).get(
                getattr(ew, "id", None) or getattr(ew, "name", None)))
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

    # ------------------------------------------------------------- spillway notch

    def _cut_spillway(self, dem, ew, sill_wkt):
        """Cut *ew*'s **outflow** spillway into *dem*, or say why it could not be.

        The one place a designed spillway becomes terrain. Everything downstream then
        follows with no plumbing at all, because ``find_impoundments`` measures a
        hollow's storage as ``Σ(filled − ground)`` and has no idea what made it: lower
        the level water can leave at, and retention, exit cells, Surface Runoff, exit
        volumes and the ponding layers all move with the cut.

        **Outflow only.** Every earthwork can also carry an ``inflow_spillway`` with its
        own crest, and an inlet is not a weir — it is a protected entry. Notching one
        would cut a hole in the bank at the point water arrives and drain the pond
        through its own inlet. ``_refresh_auto_spillway_widths`` already draws this line.

        *sill_wkt* is the crest bar the controller snapped to the feature. ``None`` means
        no spillway is sited (ordinary) or the recorded point is no longer on the feature
        (a fault the controller reports); either way there is nothing to cut here.

        Returns the DEM — the input untouched wherever the notch was refused, and every
        refusal is recorded in :attr:`warnings` with the figure that explains it.
        """
        spillway = getattr(ew, "outflow_spillway", None)
        crest = None if spillway is None else getattr(spillway, "crest_elevation", None)
        if not sill_wkt or crest is None:
            return dem
        key = getattr(ew, "id", None) or getattr(ew, "name", None)
        try:
            from shapely import wkt as _wkt
            bar = _wkt.loads(sill_wkt)
        except Exception:
            return dem
        try:
            coords = list(bar.coords)
        except (AttributeError, NotImplementedError):
            return dem
        if len(coords) < 2:
            return dem

        # The **rounded** width, widened about the bar's own centre so the cut stays
        # where the user sited it. A sill narrower than a cell cannot be cut as one.
        burn_width = spillway_burn_width(getattr(spillway, "width_m", 0.0),
                                         (self.cell_h, self.cell_size))
        bar = self._scaled_bar(coords, burn_width)
        crest_cells = self._bar_cells(bar)
        if crest_cells[0].size == 0:
            return dem

        step = self._downhill_step(bar)
        if step is None:
            return dem

        footprint = self.burned_masks.get(key)
        if footprint is None or footprint.shape != self.shape:
            footprint = np.zeros(self.shape, dtype=bool)
            footprint[crest_cells] = True
        floor = self._burned_floor(dem, footprint)
        window = self._notch_window(crest_cells, footprint)
        pool = notch_pool(dem, footprint, crest, window=window)

        cut = spillway_notch(dem, crest_cells, crest, step,
                             max_reach_m=SPILLWAY_MAX_REACH_M,
                             cell_size=max(self.cell_size, self.cell_h),
                             pool=pool, floor_elev=floor)

        if key is not None and cut.sill_elev is not None:
            self.burned_sills[key] = float(cut.sill_elev)

        name = getattr(ew, "name", "This feature")
        if floor is not None and float(crest) <= float(floor):
            self.warnings.append(
                f"{name}: the spillway crest at {float(crest):.2f} m sits at or below "
                f"the floor the burn actually cut ({float(floor):.2f} m), so a notch "
                f"there would empty the feature. No notch was cut — raise the crest, or "
                f"deepen the feature under it."
            )
            return dem
        if cut.into_pool:
            self.warnings.append(
                f"{name}: the spillway notch found ground below its crest, but that "
                f"ground is still inside this feature's own pond — the bank wraps round "
                f"it. Nothing was cut, because the water would leave where it already "
                f"leaves. Move the sill to a point where the bank has ground below the "
                f"crest on its far side."
            )
            return dem
        if not cut.daylit:
            self.warnings.append(
                f"{name}: the spillway at {float(crest):.2f} m does not daylight — the "
                f"ground stays above the crest for the whole {cut.reach_m:.0f} m tried, "
                f"so the notch would discharge into rising ground. Nothing was cut. "
                f"Site the sill where the bank falls away."
            )
            return dem

        mask = np.zeros(self.shape, dtype=bool)
        mask[cut.cells] = True
        if key is not None:
            self.burned_notches[key] = mask
        return cut.dem

    def _scaled_bar(self, coords, width_m):
        """The crest bar re-cut to *width_m* about its own centre.

        The controller draws the bar at the *built* width, which is what the map should
        show; the burn needs it at the rounded width. Scaling here rather than asking for
        a second bar keeps one snap and one geometry.
        """
        (x1, y1), (x2, y2) = coords[0][:2], coords[-1][:2]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0:
            return LineString([(cx, cy), (cx, cy)])
        wanted = float(width_m or 0.0)
        half = (wanted if wanted > 0 else length) / 2.0
        ux, uy = dx / length, dy / length
        return LineString([(cx - ux * half, cy - uy * half),
                           (cx + ux * half, cy + uy * half)])

    def _bar_cells(self, bar):
        """``(rows, cols)`` for the crest bar.

        ``all_touched=True``, and here that is the right question: this is *did we lose
        the sill?*, not *how much earth came out?* — the same distinction
        :meth:`_rasterize` draws. A bar shorter than a cell, or one lying diagonally
        between centres, still has to claim cells or the notch is not cut at all, so the
        ``line_cells`` fallback backs it up.
        """
        mask = np.zeros(self.shape, dtype=bool)
        try:
            band = self._rasterize(bar, all_touched=True)
            if band is not None and band.any():
                mask |= band
        except Exception:
            pass
        if not mask.any():
            for rc in self._line_path_cells(bar):
                mask[rc] = True
        return np.nonzero(mask)

    def _downhill_step(self, bar):
        """``(drow, dcol)`` unit step from the crest bar toward the **lower** side.

        Both normals to the bar are tried on the original ground — never on the running
        array, where this feature's own companion berm or wall would make the built side
        look like the high one and send the notch inward. The band is a few cells deep so
        the answer is about the hillside rather than about one cell.
        """
        try:
            coords = list(bar.coords)
            (x1, y1), (x2, y2) = coords[0][:2], coords[-1][:2]
        except Exception:
            return None
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0:
            return None
        # Normal in map space, converted to grid space. transform.e is negative on a
        # north-up grid, so a step north is a step to a *smaller* row.
        nx, ny = -dy / length, dx / length
        drow = ny / self.transform.e if self.transform.e else 0.0
        dcol = nx / self.transform.a if self.transform.a else 0.0
        norm = (drow * drow + dcol * dcol) ** 0.5
        if norm == 0:
            return None
        drow, dcol = drow / norm, dcol / norm

        reach = 3
        rows0, cols0 = self._bar_cells(bar)
        if rows0.size == 0:
            return None
        means = []
        for sign in (1.0, -1.0):
            side = np.zeros(self.shape, dtype=bool)
            for k in range(1, reach + 1):
                rr = rows0 + int(round(sign * k * drow))
                cc = cols0 + int(round(sign * k * dcol))
                ok = ((rr >= 0) & (rr < self.shape[0])
                      & (cc >= 0) & (cc < self.shape[1]))
                if ok.any():
                    side[rr[ok], cc[ok]] = True
            means.append(self._ground_mean(side))
        if np.isinf(means[0]) and np.isinf(means[1]):
            return None
        sign = 1.0 if means[0] <= means[1] else -1.0
        return (sign * drow, sign * dcol)

    def _burned_floor(self, dem, footprint):
        """The bed of whatever this feature holds water in, or ``None``.

        Measured off the cut rather than off the analytic ``rim − depth``: a footprint
        too narrow to hold its batter never reaches full depth (``tapered_invert``), so
        the analytic figure would refuse a crest that is in fact well above the floor.

        The **lower** of the burned surface and the original ground, because the two
        kinds of feature put their bed in different places. A cut has its floor in the
        burned array — the burn dug it. A barrier does not: its recorded mask is the
        drawn line, the burn *raises* that line to the crest, and reading the floor off
        the burned array would return the top of the wall and refuse every spillway on
        every dam. Taking the minimum of both is the pond bed either way.
        """
        if footprint is None or not footprint.any():
            return None
        vals = np.minimum(dem[footprint], self.original[footprint])
        vals = vals[np.isfinite(vals)]
        return float(vals.min()) if vals.size else None

    def _notch_window(self, crest_cells, footprint):
        """A crop around the feature and its crest, padded by the march's own cap.

        The pool flood only has to answer *is the far end of this notch still inside the
        pond*, which is a local question. Flooding the whole grid to answer it would cost
        a full-array labelling per spillway on every burn.
        """
        rows = [crest_cells[0].min(), crest_cells[0].max()]
        cols = [crest_cells[1].min(), crest_cells[1].max()]
        if footprint is not None and footprint.any():
            fr, fc = np.nonzero(footprint)
            rows += [fr.min(), fr.max()]
            cols += [fc.min(), fc.max()]
        pad = int(round(SPILLWAY_MAX_REACH_M / max(self.cell_size, self.cell_h))) + 2
        r0 = max(0, int(min(rows)) - pad)
        r1 = min(self.shape[0] - 1, int(max(rows)) + pad)
        c0 = max(0, int(min(cols)) - pad)
        c1 = min(self.shape[1] - 1, int(max(cols)) + pad)
        return (slice(r0, r1 + 1), slice(c0, c1 + 1))

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

    @staticmethod
    def _link_datum(ew):
        """*ew*'s linked start level as a finite float, or ``None``.

        ``invert_start_m`` is resolved out in the controller and put on the feature, so
        the burner never has to know what a link is — it receives ``Earthwork`` objects
        and reads one derived number, the same arrangement ``sills=`` uses for the snap.
        A link that could not be resolved leaves the field ``None`` and the drain grades
        from the ground under its own alignment, which is what it did before it was
        linked; the controller reports that rather than this method guessing at it.
        """
        value = getattr(ew, "invert_start_m", None)
        if value is None:
            return None
        try:
            level = float(value)
        except (TypeError, ValueError):
            return None
        return level if np.isfinite(level) else None

    def _burn_diversion(self, dem, line, ew):
        dem = dem.copy()
        coords = list(line.coords)
        if len(coords) < 2:
            return dem

        # **The grade datum.** Where this drain is linked to a spillway it is that
        # spillway's crest — an absolute carried on the design, which is why it does not
        # matter that the notch itself is cut as a post-pass and is not in this array
        # yet. The drain's bed comes out one depth below it, exactly as it comes out one
        # depth below sampled ground, so nothing else in this method changes.
        #
        # A drain linked at its **far** end is graded from that end, and the reversal is
        # local to this method: the stored geometry keeps the vertex order the user drew
        # (see the Spillway links section for why). Reversing the working copy rather
        # than special-casing the grade means chainage, the path cells and the monotonic
        # breach all go on reading a line whose first vertex is the graded start.
        start_elev = self._link_datum(ew)
        if start_elev is not None:
            link = parse_spillway_link(getattr(ew, "spillway_link_id", None))
            if link is not None and link[2] == "end":
                coords = coords[::-1]
                line = LineString(coords)
        else:
            # No link: the first vertex that sits on mapped ground rather than blindly
            # the first. A line starting in a nodata hole used to read the sentinel as
            # an elevation and grade the entire channel away from about -10,000 m,
            # burning a trench that deep along its whole length.
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
            # The batter, from the two widths, exactly as a swale's — the registry gives
            # a diversion `default_side_slope` and derives its bottom width, so it is
            # specified as a trapezoid and `calculate_cut_volume` prices it as one. This
            # burn cut it as a full-depth rectangle across the whole band: the same
            # mismatch `_storage_invert` was rewritten to fix for swales, left standing
            # here because a graded invert could not use `tapered_invert`, whose floor is
            # one elevation. It does not need to — the taper is a per-cell fraction of
            # depth and the grade is a per-cell datum, so they simply multiply.
            reach = taper_reach(band, channel_batter_run(ew),
                                (self.cell_h, self.cell_size))
            depth = ew.depth if reach is None else ew.depth * reach[rows, cols]
            burn = start_elev - chainage * gradient_frac - depth
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

        from rasterio.transform import Affine

        from terrainflow_assessment.modules.pysheds_compat import Grid

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

    def _keyed_dam_dem(self, dam, sills=None):
        """DEM with the dam raised to its crest and keyed into the banks (analytical).

        **This path bypasses :meth:`burn_earthworks` entirely**, so the spillway notch has
        to be cut here by hand or a keyed dam is the one type the whole change is invisible
        on — and a keyed dam is the case the spillway spec is written about. The cut goes
        through the same :meth:`_cut_spillway` the post-pass calls, so the two cannot drift:
        one function, two callers.

        The mask it records is discarded with the rest of this idealised surface — this is
        an estimate of a wall the user drew shorter, and the site burn is what the map,
        the streams and the verification are measured off.
        """
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
        # The notch, on the same surface. `_cut_spillway` reads `burned_masks` for the
        # feature's footprint and none was recorded here, so the contact band is supplied
        # — the drawn line is where this dam's water stands, which is what the pool test
        # and the burned-floor check both want.
        key = getattr(dam, "id", None) or getattr(dam, "name", None)
        sill = (sills or {}).get(key)
        if sill:
            masks, notches = self.burned_masks, self.burned_notches
            sill_levels = self.burned_sills
            try:
                self.burned_masks = dict(masks)
                self.burned_masks[key] = self._contact_mask(line, dam)
                self.burned_notches, self.burned_sills = {}, {}
                dem = self._cut_spillway(dem, dam, sill)
            finally:
                self.burned_masks, self.burned_notches = masks, notches
                self.burned_sills = sill_levels
        return dem

    def _isolated_burn(self, ew, keep_warnings=False, sills=None):
        """*(dem, mask)* for **this feature alone** on the original ground.

        Burning one feature into a fresh copy is what makes the storage figure
        order-independent: it is what this earthwork does to this hillside, not what it
        happens to do given whichever neighbours were burned before it.

        ``burn_earthworks`` resets the burner's per-feature records, so the site-wide
        burn's masks and trench volumes are snapshotted and put back. Warnings are
        discarded by default — the isolated pass regenerates advisories the real burn has
        already raised, and reporting a sub-cell swale twice helps nobody.

        ``burned_notches`` is on that list for the same reason the other three are, and
        it matters more: these isolated burns run **after** the site burn, once per
        feature, so without the restore the dict would end up holding whichever feature
        was measured last — and that dict is what the overtopping check subtracts its
        barrier crest against.

        *sills* is passed straight through, so the isolated measurement sees the same
        notch the site burn cut. Without it every per-feature capacity would be measured
        on a feature with its spillway plugged, which is the figure this whole change
        exists to stop reporting.
        """
        masks, cuts, warns = self.burned_masks, self.burned_cut, self.warnings
        raised, notches = self.burned_raised, self.burned_notches
        sill_levels = self.burned_sills
        try:
            dem = self.burn_earthworks([ew], sills=sills)
            key = getattr(ew, "id", None) or getattr(ew, "name", None)
            mask = self.burned_masks.get(key)
            if keep_warnings:
                warns = self.warnings
        finally:
            self.burned_masks, self.burned_cut, self.warnings = masks, cuts, warns
            self.burned_raised, self.burned_notches = raised, notches
            self.burned_sills = sill_levels
        return dem, mask

    def feature_storage(self, ew, baseline_ponding=None, isolated_dem=None,
                        isolated_mask=None, keep_warnings=False, sills=None):
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

        **``excavation_m3`` — the earth this feature takes out, and why it is measured here.**
        ``Σ (original − isolated) × cell_area`` wherever the isolated burn lowered ground.
        It is the same expression :meth:`_warn_steep` computes for its advisory, and it is
        the figure the report's *Cut (measured)* column wants: it carries the level-floor
        over-dig, which on falling ground is most of the difference between the drawn section
        and the earth a contractor moves.

        It is measured on the **isolated** burn and over the **whole grid**, and both halves
        of that matter. The site burn lowers one shared array feature by feature with
        ``np.minimum``, so a feature whose footprint overlaps a neighbour's finished trench
        writes nothing there and would be credited with earth the neighbour moved — the
        attribution :func:`burn_quantities` refuses to invent. Burning alone removes the
        ordering entirely. Whole-grid rather than over ``burned_masks`` because a recorded
        mask is not always the cut: a diversion drain records its contact band and cuts a
        separately rasterised set of cells, so integrating the mask would miss earth it moves.

        **``stage_storage`` — the volume at any level, not only at the top.** The pond is
        already flooded and its depths are already in hand, so the cumulative curve is a
        sort and a running sum away, and it is the only thing that can answer "what does
        a sill *here* give up" without re-flooding on every spin of a crest control. It
        is derived and is never serialised, the same rule ``terrain_capacity_m3`` follows.

        For a dam ``isolated_dem`` is the caller's idealised keyed wall rather than the burn,
        so the figure describes that idealisation; nothing reads it, because a dam has no
        drawn section to compare against and :func:`capacity_breakdown` reports no cut for one.
        """
        from rasterio.transform import Affine
        from scipy.ndimage import binary_dilation
        from scipy.ndimage import label as _label

        if isolated_dem is None:
            isolated_dem, isolated_mask = self._isolated_burn(
                ew, keep_warnings=keep_warnings, sills=sills)
        if isolated_mask is None:
            key = getattr(ew, "id", None) or getattr(ew, "name", None)
            isolated_mask = self.burned_masks.get(key)
        if isolated_mask is None or not isolated_mask.any():
            return FeatureStorage(0.0, None, None, 0.0, 0.0)

        cell_area = self.cell_area
        # Both idioms are :func:`burn_quantities`', and for its reasons. nansum, not sum:
        # a DEM clipped to a boundary always has nodata, one NaN would take the whole
        # total with it, and a cell with no elevation moved no earth. float64, not the
        # DEM's float32: accumulating twenty thousand cells in float32 costs the last
        # four digits, which is enough to stop this agreeing with the site total it is
        # supposed to sum toward.
        excavation_m3 = float(np.nansum(np.clip(
            self.original.astype("float64") - np.asarray(isolated_dem, dtype="float64"),
            0.0, None))) * cell_area
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
            # It holds nothing, but it was still dug: a cut-off drain on falling ground
            # ponds nowhere and is the most earth-moving feature on some sites.
            return FeatureStorage(0.0, None, None, 0.0, 0.0, excavation_m3)

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

        # The curve comes off the same arrays, so it costs one sort and one cumulative
        # sum over a pond that has already been flooded. That is the whole reason the
        # live "what this sill gives up" readout does not need a second pass.
        stage = build_stage_storage(new_pond[region], level, cell_area)

        return FeatureStorage(volume, level, region, above_m3, retained, excavation_m3,
                              stage)

    def feature_storage_m3(self, ew, baseline_ponding=None, sills=None):
        """Just the volume from :meth:`feature_storage` — the common case."""
        return self.feature_storage(ew, baseline_ponding=baseline_ponding,
                                    sills=sills).volume_m3

    def dam_stage_storage(self, dam, baseline_ponding=None, key_into_banks=False,
                          sills=None):
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

        return self.dam_storage(dam, baseline_ponding=baseline_ponding,
                                key_into_banks=key_into_banks,
                                sills=sills).volume_m3

    def dam_storage(self, dam, baseline_ponding=None, key_into_banks=False,
                    sills=None):
        """The full :class:`FeatureStorage` behind :meth:`dam_stage_storage`.

        Same measurement; this one does not throw the rest of it away. The name has been
        promising a curve since it was written and returning a single volume, and the
        curve is what the live "what this sill gives up" readout reads — so a dam, which
        is the type most likely to carry a spillway, would otherwise be the one type
        without one.

        Returns an empty measurement for a dam with no crest, which is what
        :meth:`dam_stage_storage`'s ``0.0`` meant.
        """
        if getattr(dam, "crest_elevation", None) is None:
            return FeatureStorage(0.0, None, None, 0.0, 0.0)

        self.warnings = []
        if key_into_banks:
            # ``_keyed_dam_dem`` raises the abutment advisory the caller then surfaces, so
            # this path builds its own DEM and hands it over rather than re-burning.
            dem = self._keyed_dam_dem(dam, sills=sills)
            mask = self._contact_mask(self._to_shapely(dam.geometry), dam)
            return self.feature_storage(dam, baseline_ponding=baseline_ponding,
                                        isolated_dem=dem, isolated_mask=mask)
        return self.feature_storage(dam, baseline_ponding=baseline_ponding,
                                    keep_warnings=True, sills=sills)

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
