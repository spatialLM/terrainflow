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
calculate_spillway_width      — broad-crested weir sizing
Spillway             — designed overflow point (crest, head, width, location)
spillway_datum       — crest elevations a feature can physically offer
bind_crest           — two-way crest ↔ drop-below-rim binding
spillway_validity    — plain-language problems with a proposed spillway
berm_height_estimate          — companion berm height from swale volume
"""

import json
import logging
import uuid

import numpy as np
import rasterio
from shapely.geometry import shape as shapely_shape

from terrainflow_assessment.core.registry.earthwork_types import get_type
from terrainflow_assessment.core.sizing import (
    basin_volume_battered,
    manning_flow,
    trapezoid_section,
)
from terrainflow_assessment.modules.burn_strategy import (
    battered_invert,
    enforce_monotonic_path,
    level_invert,
    line_cells,
    ponding_resolution_warning,
    rasterisable_capacity,
    steep_ground_warning,
    sub_cell_warning,
)
from terrainflow_assessment.modules.footprint import (
    internal_relief,
    min_dimension,
    pour_point,
    rasterize_footprint,
)
from terrainflow_assessment.qgis.adapters.geom import shapely_area, shapely_length

_log = logging.getLogger(__name__)

_MAX_PONDING_CELLS = 4_000_000  # ~2000 × 2000
_DAM_WINDOW_PAD_CELLS = 64      # initial crop padding for the windowed dam flood


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
# spillway and the top of the settled embankment". The previous 0.15 m was half that.
SPILLWAY_MIN_FREEBOARD_M = 0.30

# Head range that broad-crested weir practice treats as ordinary. Outside it the
# weir formula still holds; it simply stops being a routine design.
SPILLWAY_TYPICAL_HEAD_M = (0.20, 0.50)

# Elevation comparisons are made to the millimetre. Without this a crest clamped
# to exactly the highest value spillway_datum offers reports as *insufficient*,
# because rim − head − freeboard does not reconstruct head + freeboard in binary
# floating point. Sub-millimetre setting-out is meaningless on a DEM anyway.
_ELEV_EPS = 0.001


class Spillway:
    """Where a feature is *designed* to overflow, and how wide that has to be.

    Two numbers describe one crest, and the dialog binds them both ways.
    ``crest_elevation`` is absolute; ``drop_below_rim_m`` is how far it sits below
    the rim — the lowest containing ground, which is where the feature would spill
    if nothing were built. The rim is the datum because it is the elevation the DEM
    actually supplies: an absolute crest typed without reference to it is
    unanchored, and a drop is meaningless without it.

    ``auto`` means the crest tracks the rim as the DEM or the footprint changes,
    rather than staying where it was first computed.
    """

    def __init__(self, crest_elevation=None, drop_below_rim_m=None, head_m=0.30,
                 width_m=0.0, point_wkt=None, auto=True, width_auto=True):
        self.crest_elevation = crest_elevation
        self.drop_below_rim_m = drop_below_rim_m
        self.head_m = head_m
        self.width_m = width_m        # width as BUILT (or tracking, while width_auto)
        # Whether the built width tracks the computed requirement. Separate from
        # ``auto`` (which tracks the crest against the rim) because a user who has
        # committed to a dug width has not thereby fixed the crest, or vice versa.
        self.width_auto = width_auto
        self.point_wkt = point_wkt    # placed location, or None for "not sited yet"
        self.auto = auto

    _SERIAL_FIELDS = (
        "crest_elevation", "drop_below_rim_m", "head_m", "width_m",
        "width_auto", "point_wkt", "auto",
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
                      width_m=None, required_width_m=None):
    """Plain-language problems with a proposed spillway; empty list means fine.

    Every message quotes the numbers it is objecting to, because "invalid" on its
    own gives the user nothing to act on.
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
    elif rim - crest < head + freeboard - _ELEV_EPS:
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

    lo, hi = SPILLWAY_TYPICAL_HEAD_M
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
        self.key_into_banks = False  # dam only: opt-in idealised (keyed) storage estimate
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
            # The berm is a 1:1 triangle of height h — base 2h, area h² — which is
            # exactly what calculate_fill_volume builds and what berm_height_estimate
            # derives from the spoil. The water it impounds behind it is that same
            # triangle, so the capacity credit is h², not h×top_width/2. The old form
            # exceeded the berm's own section by T/(2h) — a third larger at the
            # registry defaults, claiming ~14% more swale capacity than the berm
            # actually has material to hold back.
            berm_height = (cross_section * 0.75) ** 0.5
            cross_section += berm_height * berm_height

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


def capacity_breakdown(ew, cell_size=1.0, n_cells=None):
    """Split a feature's capacity into the four numbers the Verify stage compares.

    A single "capacity" figure conflated three unrelated gaps, which is why a
    verification delta of −38% was uninterpretable — it mixed a burn error, a
    resolution artefact and a design allowance into one percentage.

    ``design``      geometric × freeboard — the headline the panel plans on
    ``geometric``   the exact drawn shape: "if I dug precisely this, how big is it?"
    ``rasterisable`` that shape after rasterising to this cell size
    ``freeboard_m3``          design ← geometric (an allowance you chose)
    ``resolution_penalty_m3`` rasterisable ← geometric (the grid's distortion)

    Verification should compare measured ponding against ``rasterisable``: a non-zero
    delta there is a genuine burn problem and nothing else. ``n_cells`` is the burned
    footprint's cell count; without it the rasterisable figure falls back to the
    geometric one (nothing measured, so nothing claimed).
    """
    geometric, _ = calculate_capacity(
        ew.type, ew.geometry, ew.depth, ew.width,
        getattr(ew, "companion_berm", False),
        bottom_width=getattr(ew, "bottom_width_m", None),
        batter_run=getattr(ew, "batter_run_m", None),
    )
    geometric = geometric / FREEBOARD if FREEBOARD > 0 else geometric
    design = getattr(ew, "capacity_m3", 0.0) or 0.0

    # Barrier-impounded storage — a dam. There is no drawn cross-section to rasterise:
    # the shape of the water is the shape of the valley, and the analytic capacity is
    # *already* a flooded-volume computation over the same grid the burn uses. Passing
    # a dam through the trapezoid path yielded a geometric of 0 and a rasterisable of
    # whatever a channel of that width would hold, so a dam that verified perfectly
    # (2,148 m³ measured against 2,148 m³ designed) reported Δ +1712%.
    barrier = geometric <= 0 and design > 0
    if barrier:
        return {
            "design": round(design, 2),
            "geometric": round(design, 2),
            "rasterisable": round(design, 2),
            "freeboard_m3": 0.0,
            "resolution_penalty_m3": 0.0,
            "barrier_impounded": True,
        }

    if n_cells:
        raster = rasterisable_capacity(
            n_cells, cell_size ** 2, ew.depth,
            ew.width, getattr(ew, "bottom_width_m", ew.width), cell_size,
            batter_run=_as_float(getattr(ew, "batter_run_m", 0.0)),
        )
    else:
        raster = geometric

    return {
        "design": round(design, 2),
        "geometric": round(geometric, 2),
        "rasterisable": round(raster, 2),
        "freeboard_m3": round(geometric - design, 2),
        "resolution_penalty_m3": round(raster - geometric, 2),
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
        length = shapely_length(geometry)
        top_width = width
        bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
        swale_cs = trapezoid_section(top_width, bottom_width, depth).area
        berm_height = (swale_cs * 0.75) ** 0.5
        berm_cs = berm_height * berm_height  # triangular
        return round(berm_cs * length, 2)

    if ew_type == "dam":
        length = shapely_length(geometry)
        return round(width * depth * length, 2) if depth and width else 0.0

    return 0.0


def berm_height_estimate(depth, width, bottom_width=None):
    """Estimate companion berm height from swale excavation (75% compaction).

    ``width`` is the declared top width of the swale (metres). ``bottom_width`` is the
    trapezoid bottom width; ``None`` derives it from 1:1 side slopes (historical behaviour).
    """
    top_width = width
    bottom_width = _resolve_bottom_width(bottom_width, top_width, depth)
    cross_section = trapezoid_section(top_width, bottom_width, depth).area
    return round((cross_section * 0.75) ** 0.5, 2)


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

    def __init__(self, dem_path):
        with rasterio.open(dem_path) as src:
            self.original = src.read(1).astype("float32")
            self.transform = src.transform
            self.crs = src.crs
            self.nodata = src.nodata
            self.shape = self.original.shape
        self.cell_size = abs(self.transform.a)
        # Non-fatal advisories raised during the last burn / ponding pass (sub-cell
        # features, resolution-cap degrade). The controller surfaces these to the
        # QGIS message bar — Strategy C is honest about what it approximates.
        self.warnings = []

    def burn_earthworks(self, earthworks):
        """
        Apply all enabled earthworks to a copy of the original DEM.
        Returns modified DEM as float32 numpy array.
        """
        modified = self.original.copy()
        self.warnings = []
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
        """Save a DEM array to GeoTIFF with LZW compression."""
        with rasterio.open(
            output_path, "w",
            driver="GTiff", dtype="float32",
            crs=self.crs, transform=self.transform,
            width=self.shape[1], height=self.shape[0],
            count=1, nodata=self.nodata, compress="lzw",
        ) as dst:
            dst.write(array, 1)

    # ------------------------------------------------------------------ helpers

    def _to_shapely(self, qgs_geometry):
        try:
            return shapely_shape(json.loads(qgs_geometry.asJson()))
        except Exception:
            return None

    def _rasterize(self, shapely_geom):
        """Footprint cell mask, matching every other rasterise call in the project.

        This was the only one omitting ``all_touched``, so it included a cell only
        when the cell *centre* fell inside the geometry — undersizing every footprint
        by up to one cell all round and stair-stepping its edges, while the analytic
        capacity used the exact polygon. Diagonal lines were worst affected.
        """
        return rasterize_footprint(shapely_geom, self.shape, self.transform,
                                   all_touched=True)

    def _line_path_cells(self, line):
        """Connected in-bounds cell path along *line* (nearest-cell snap fallback)."""
        try:
            coords = list(line.coords)
        except (NotImplementedError, AttributeError):
            return []
        return line_cells(coords, self.transform, self.shape)

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
        """
        half = thickness / 2.0
        try:
            left = line.parallel_offset(half, "left")
            right = line.parallel_offset(half, "right")
            left_mask = self._rasterize(left.buffer(half))
            right_mask = self._rasterize(right.buffer(half)) & ~left_mask
            left_mean = float(np.mean(self.original[left_mask])) if left_mask.any() else np.inf
            right_mean = (
                float(np.mean(self.original[right_mask])) if right_mask.any() else np.inf
            )
            if np.isinf(left_mean) and np.isinf(right_mean):
                return line.buffer(half)
            chosen = left if left_mean <= right_mean else right
            return chosen.buffer(half)
        except Exception:
            return line.buffer(half)

    # ---------------------------------------------------------------- earthwork types

    def _burn_swale(self, dem, line, ew):
        """Excavate a swale to a **level invert** — it is storage, not a drain.

        A swale used to be cut at constant depth and then breached with
        ``enforce_monotonic_path``, which forced a strictly descending centreline.
        That guaranteed depression-filling would find an outlet, so a burned swale
        ponded essentially nothing however much capacity the panel credited it —
        most of the analytic-vs-terrain gap the Verify chip was reporting.

        Order matters: the companion berm goes in **first**, because it raises the
        spill level and is most of the swale's real capacity. Taking the pour point
        before building it would reference the floor to ground the berm then buries.
        """
        footprint = line.buffer(ew.buffer_radius_m)  # radius = top_width_m / 2
        mask = self._rasterize(footprint)
        path_cells = self._line_path_cells(line)
        dem = dem.copy()

        if not mask.any():
            # Sub-cell: no cell centre is inside the buffer, so claim the path cells.
            mask = np.zeros(self.shape, dtype=bool)
            for rc in path_cells:
                mask[rc] = True

        if ew.companion_berm:
            dem = self._add_companion_berm(dem, line, mask, ew)

        spill, _ = pour_point(dem, mask, nodata=self.nodata)
        if spill is None:
            return dem
        relief = internal_relief(self.original, mask)
        dem = level_invert(dem, mask, ew.depth, spill)

        self._warn_sub_cell(ew.name, ew.bottom_width_m)
        self._warn_steep(ew, mask, relief, dem)
        return dem

    def _add_companion_berm(self, dem, line, swale_mask, ew):
        berm_width = ew.top_width_m
        try:
            left_line = line.parallel_offset(ew.buffer_radius_m + berm_width / 2, "left")
            right_line = line.parallel_offset(ew.buffer_radius_m + berm_width / 2, "right")
            left_zone = left_line.buffer(berm_width / 2)
            right_zone = right_line.buffer(berm_width / 2)
        except Exception:
            return dem

        # With all_touched the offset bands can now overlap each other and the
        # swale trench; keep them disjoint so the berm is not raised over the cut.
        left_mask = self._rasterize(left_zone) & ~swale_mask
        right_mask = self._rasterize(right_zone) & ~swale_mask & ~left_mask

        if left_mask.any() and right_mask.any():
            left_mean = float(np.mean(self.original[left_mask]))
            right_mean = float(np.mean(self.original[right_mask]))
            berm_mask = left_mask if left_mean < right_mean else right_mask
        elif left_mask.any():
            berm_mask = left_mask
        elif right_mask.any():
            berm_mask = right_mask
        else:
            return dem

        # Height from the volume actually excavated, not (cells × nominal depth):
        # with a level invert the cut varies across the footprint, so the old ratio
        # no longer conserved anything. 0.75 accounts for bulking/compaction losses.
        n_berm = int(np.sum(berm_mask))
        spill, _ = pour_point(self.original, swale_mask, nodata=self.nodata)
        if spill is not None and n_berm > 0:
            floor = spill - ew.depth
            cut_depths = np.clip(self.original[swale_mask] - floor, 0.0, None)
            raise_height = float(cut_depths.sum()) * 0.75 / n_berm
        else:
            raise_height = ew.depth

        dem = dem.copy()
        dem[berm_mask] += raise_height
        return dem

    def _burn_berm(self, dem, line, ew):
        # Barrier: raise a flow-blocking ridge (never a cut). Incise-free — the
        # footprint band where resolvable, the nearest-cell path when sub-cell.
        footprint = line.buffer(ew.width / 2)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        if mask.any():
            dem[mask] += ew.depth
        else:
            for rc in self._line_path_cells(line):
                dem[rc] += ew.depth
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
        """
        mask = self._rasterize(polygon)
        dem = dem.copy()

        if not mask.any():
            # A sub-cell polygon (or a sliver crossing no cell centre) used to burn
            # nothing at all, silently. Claim its centroid cell instead.
            try:
                c = polygon.centroid
                col = int((c.x - self.transform.c) / self.transform.a)
                row = int((c.y - self.transform.f) / self.transform.e)
                if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                    mask[row, col] = True
            except Exception:
                return dem
            if not mask.any():
                return dem

        # Basins carried no sub-cell advisory at all, so a footprint smaller than a
        # cell claimed its full analytic volume with nothing to flag it.
        self._warn_sub_cell(ew.name, min_dimension(polygon))

        spill, _ = pour_point(dem, mask, nodata=self.nodata)
        if spill is None:
            return dem
        relief = internal_relief(self.original, mask)

        batter = _as_float(getattr(ew, "batter_run_m", 0.0))
        if batter > 0:
            dem = battered_invert(dem, self._batter_steps(polygon, ew.depth, batter), spill)
        else:
            dem = level_invert(dem, mask, ew.depth, spill)

        self._warn_steep(ew, mask, relief, dem)
        return dem

    def _batter_steps(self, polygon, depth, batter_run, n_steps=3):
        """Nested (mask, depth) pairs approximating battered walls on the grid.

        Shrinks the footprint inward by the batter run in equal slices; each inner
        slice is cut deeper. With fewer than ~3 cells across, the erosions vanish and
        this degrades to a single full-depth mask — the honest grid limit, which
        ``rasterisable_capacity`` reports rather than hides.
        """
        steps = []
        for i in range(1, n_steps + 1):
            frac = i / n_steps
            # The opening is widest at the rim and narrows with depth: the inset grows
            # WITH the depth fraction. (Inverting these two gives an inverted bowl —
            # a footprint that widens as it deepens.)
            inset = batter_run * frac
            try:
                shrunk = polygon.buffer(-inset) if inset > 0 else polygon
            except Exception:
                shrunk = polygon
            if shrunk.is_empty:
                continue
            mask = self._rasterize(shrunk)
            if mask.any():
                steps.append((mask, depth * frac))
        if not steps:
            steps = [(self._rasterize(polygon), depth)]
        return steps

    def _warn_steep(self, ew, mask, relief, burned_dem):
        """Record the over-excavation advisory, quantified in real cubic metres."""
        depth = _as_float(getattr(ew, "depth", 0.0))
        try:
            cut = float(np.sum(
                np.clip(self.original[mask] - burned_dem[mask], 0.0, None)
            )) * (self.cell_size ** 2)
            storage = float(mask.sum()) * (self.cell_size ** 2) * depth
        except Exception:
            cut = storage = None
        w = steep_ground_warning(ew.name, relief, depth, cut, storage)
        if w:
            self.warnings.append(w)

    def _burn_dam(self, dem, line, ew):
        if ew.crest_elevation is None:
            return self._burn_berm(dem, line, ew)
        # Inner-wall convention (§7): wall thickness sits downstream of the drawn
        # line; raise the wall band (or the nearest-cell path) up to the crest.
        footprint = self._downstream_footprint(line, ew.width)
        mask = self._rasterize(footprint)
        dem = dem.copy()
        if mask.any():
            dem[mask] = np.maximum(dem[mask], ew.crest_elevation)
        else:
            for rc in self._line_path_cells(line):
                dem[rc] = max(dem[rc], ew.crest_elevation)
        self._warn_sub_cell(ew.name, ew.width)
        return dem

    def _burn_diversion(self, dem, line, ew):
        dem = dem.copy()
        coords = list(line.coords)
        if len(coords) < 2:
            return dem

        x0, y0 = coords[0]
        col0 = int((x0 - self.transform.c) / self.transform.a)
        row0 = int((y0 - self.transform.f) / self.transform.e)
        row0 = max(0, min(self.shape[0] - 1, row0))
        col0 = max(0, min(self.shape[1] - 1, col0))
        start_elev = float(dem[row0, col0])

        cum_dist = [0.0]
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i - 1][0]
            dy = coords[i][1] - coords[i - 1][1]
            cum_dist.append(cum_dist[-1] + (dx ** 2 + dy ** 2) ** 0.5)
        total_length = cum_dist[-1]
        if total_length == 0:
            return dem

        gradient_frac = ew.gradient_pct / 100.0

        for seg_i in range(len(coords) - 1):
            x1, y1 = coords[seg_i]
            x2, y2 = coords[seg_i + 1]
            seg_dist = cum_dist[seg_i + 1] - cum_dist[seg_i]
            if seg_dist == 0:
                continue
            n_steps = max(2, int(seg_dist / self.cell_size * 3))
            for step in range(n_steps + 1):
                t = step / n_steps
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                dist_along = cum_dist[seg_i] + t * seg_dist
                target_floor = start_elev - dist_along * gradient_frac

                from shapely.geometry import Point
                pt_geom = Point(x, y).buffer(ew.width / 2)
                cell_mask = self._rasterize(pt_geom)
                burn_elev = target_floor - ew.depth
                if cell_mask.any():
                    dem[cell_mask] = np.minimum(dem[cell_mask], burn_elev)
                else:
                    # Sub-cell channel: the buffer rasterised empty. Snap to the
                    # nearest cell so the graded invert still carves ≥ 1 cell — but
                    # only when the sample lies within the DEM (an off-extent point
                    # stays a no-op, never a spurious edge-cell burn).
                    col = int((x - self.transform.c) / self.transform.a)
                    row = int((y - self.transform.f) / self.transform.e)
                    if 0 <= row < self.shape[0] and 0 <= col < self.shape[1]:
                        dem[row, col] = min(float(dem[row, col]), burn_elev)

        # A diversion IS a conveyance, so it gets the monotonic breach that swales
        # no longer do: the graded invert plus nearest-cell snapping can leave
        # one-cell humps that break connectivity and pond the drain.
        dem = enforce_monotonic_path(dem, self._line_path_cells(line))

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

        scaled_transform = Affine(
            base_transform.a / scale, base_transform.b, base_transform.c,
            base_transform.d, base_transform.e / scale, base_transform.f,
        )

        tmp = tempfile.mktemp(suffix=".tif")
        try:
            with rasterio.open(
                tmp, "w", driver="GTiff", dtype="float32",
                crs=self.crs, transform=scaled_transform,
                width=work_dem.shape[1], height=work_dem.shape[0],
                count=1, nodata=self.nodata,
            ) as dst:
                dst.write(work_dem, 1)

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
            row = int((ey - self.transform.f) / self.transform.e)
            col = int((ex - self.transform.c) / self.transform.a)
            if 0 <= row < self.shape[0] and 0 <= col < self.shape[1] \
                    and self.original[row, col] < crest:
                keyed = True
            reach, x, y = 0.0, ex, ey
            while reach < max_reach:
                x += dx * step
                y += dy * step
                reach += step
                row = int((y - self.transform.f) / self.transform.e)
                col = int((x - self.transform.c) / self.transform.a)
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
        footprint = self._downstream_footprint(line, dam.width)
        mask = self._rasterize(footprint)
        if mask.any():
            dem[mask] = np.maximum(dem[mask], crest)
        else:
            for rc in self._line_path_cells(line):
                dem[rc] = max(dem[rc], crest)
        keyed, reach = self._key_dam_ends(dem, line, crest)
        if keyed:
            self.warnings.append(
                f"{getattr(dam, 'name', 'Dam')}: crest sits above the natural bank — "
                f"keyed {reach:.0f} m into each abutment for the storage estimate; the "
                f"dam must be built into higher ground or water escapes around the ends."
            )
        return dem

    def dam_stage_storage(self, dam, baseline_ponding=None, key_into_banks=False):
        """Impounded volume (m³) of a dam = the new ponding it creates behind its crest.

        Burns the dam to its crest *as drawn* and depression-fills via
        ``get_ponding_layer``; the volume is ``Σ max(dammed − baseline, 0) × cell_area``.
        This is the honest storage the dam holds — a short wall leaks around its ends, so
        raising the crest above the natural abutment saddle adds nothing (the signal to
        extend the dam). Pass ``key_into_banks=True`` for the opt-in *idealised* estimate,
        which virtually extends the wall into higher ground (see :meth:`_keyed_dam_dem`) so
        it fills to the crest — a what-if only; it never redraws the dam or the verify-burn.
        ``baseline_ponding`` is the pre-dam ponding array (same shape as the DEM) — pass the
        cached baseline layer to save a flood pass, or leave None to compute it from the bare
        DEM. Returns 0.0 for a dam without a crest.
        """
        if getattr(dam, "crest_elevation", None) is None:
            return 0.0

        self.warnings = []
        dammed = self._keyed_dam_dem(dam) if key_into_banks else self.burn_earthworks([dam])
        cell_area = abs(self.transform.a * self.transform.e)

        # Windowed flood: a dam's pond is local, so flood a crop around the dam
        # instead of the whole DEM (the reshape-tool release felt slow on real
        # rasters). If the new ponding touches the window edge the pond may be
        # clipped — double the padding and retry, falling back to the full DEM.
        from rasterio.transform import Affine

        rows, cols = self.shape
        r_lo, r_hi, c_lo, c_hi = self._dam_cell_bounds(dam)
        pad = _DAM_WINDOW_PAD_CELLS
        while True:
            r0, r1 = max(0, r_lo - pad), min(rows - 1, r_hi + pad)
            c0, c1 = max(0, c_lo - pad), min(cols - 1, c_hi + pad)
            full_window = r0 == 0 and c0 == 0 and r1 == rows - 1 and c1 == cols - 1
            sub_t = self.transform * Affine.translation(c0, r0)

            dam_pond = self.get_ponding_layer(dammed[r0:r1 + 1, c0:c1 + 1], transform=sub_t)
            if baseline_ponding is not None:
                base_pond = np.clip(
                    np.asarray(baseline_ponding, dtype="float64")[r0:r1 + 1, c0:c1 + 1],
                    0.0, None,
                )
            else:
                base_pond = self.get_ponding_layer(
                    self.original[r0:r1 + 1, c0:c1 + 1], transform=sub_t
                )

            new_pond = np.clip(dam_pond.astype("float64") - base_pond, 0.0, None)
            if full_window or not _pond_touches_edge(new_pond):
                return float(new_pond.sum() * cell_area)
            pad *= 2

    def _dam_cell_bounds(self, dam):
        """(row_lo, row_hi, col_lo, col_hi) of the dam geometry, clamped to the DEM."""
        rows, cols = self.shape
        line = self._to_shapely(dam.geometry)
        if line is None:
            return 0, rows - 1, 0, cols - 1
        minx, miny, maxx, maxy = line.bounds
        c_lo = int((minx - self.transform.c) / self.transform.a)
        c_hi = int((maxx - self.transform.c) / self.transform.a)
        # transform.e is negative (north-up): larger y → smaller row
        r_lo = int((maxy - self.transform.f) / self.transform.e)
        r_hi = int((miny - self.transform.f) / self.transform.e)
        r_lo, r_hi = sorted((r_lo, r_hi))
        c_lo, c_hi = sorted((c_lo, c_hi))
        return (
            max(0, min(rows - 1, r_lo)), max(0, min(rows - 1, r_hi)),
            max(0, min(cols - 1, c_lo)), max(0, min(cols - 1, c_hi)),
        )
