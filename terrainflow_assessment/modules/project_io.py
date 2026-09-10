"""project_io.py — the portable design document (`.tfd`).

A design used to survive only inside a QGIS project file, and only partially: the
earthworks round-tripped (see :mod:`earthwork_design`) but nothing carried the *inputs*
that sized them, and every path was absolute — so a project opened on another machine
resolved no DEM, no boundary, and no storm.

This module owns the plain-data document that fixes that. It is deliberately pure: no
QGIS, no filesystem, no widgets. Callers hand it values and get a dict or a JSON string
back, which is what lets the whole schema sit under the test-coverage gate and be
exercised without a QGIS runtime.

Two rules shape the design:

**Inputs only, never derived state.** Analysis results, layer IDs, flow caches, contour
features, keypoints and simulation frames all recompute from the inputs here. Persisting
them would bloat the file and, worse, would make the known staleness bug portable — a
saved ponding raster shared to another machine is a stale answer that looks authoritative.

**Degrade, never raise.** A document written by a newer build, or one that lost a key,
loads what it can and falls back to defaults for the rest. Losing one field beats
refusing to open a design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# 2 (2026-08-19): ``Spillway.height_above_floor_m``. The field itself needs no version
# gate — ``Spillway.from_dict`` probes per field, so an older document simply has none.
# The bump is for the *other* direction: an older build re-saving one of these designs
# iterates its own shorter field tuple and drops the height silently, and the version is
# what lets ``is_from_newer_build`` say so before that happens.
#
# 3 (2026-08-19): ``Earthwork.spillway_link_id`` — which spillway a diversion drain takes
# its start level from. Same mechanics, and the bump matters more here: a height can be
# re-derived from the crest, whereas a link is a *decision*, recoverable from nothing.
# An older build re-saving the design would drop it, and the drain would go back to
# grading from the ground under its own alignment — which is a plausible-looking level,
# and so a change nobody would notice. (The Stage B round did **not** bump for the auto
# width, and said why: that figure is derived either way, so nothing a user chose is
# lost in either direction.)
SCHEMA_VERSION = 3

# Recognised DEM carriage modes. "embedded" means the archive holds a (buffered) clip of
# the DEM; "reference" means it holds only a fingerprint and the DEM must be located.
DEM_MODES = ("embedded", "reference")

# Public because the panel maps them back onto combo positions when restoring a design:
# one tuple shared with validation beats two lists drifting apart. SIZING_BASIS_VALUES is
# order-significant — it mirrors the sizing-basis combo's item order.
ROUTING_VALUES = ("d8", "dinf")
SIZING_BASIS_VALUES = ("coefficient", "rainfall", "runoff")
# TR-55 hydrologic condition of the pasture. Order-significant — it mirrors the ground
# condition combo's item order, least to most runoff.
GROUND_CONDITION_VALUES = ("good", "fair", "poor")


# Every analysis input the panel exposes, as ``name: (type, default)``. This is the
# single source of truth for what a design file carries — the panel's collect/apply pair
# and this module's validation both read it, so a new input cannot be added to one and
# silently forgotten by the other.
#
# Grouped by what they drive, though the document stores them flat: nesting would buy
# nothing and would make forward-compatible defaulting fiddlier.
INPUT_FIELDS = {
    # Site identity
    "site_name": (str, "Unnamed Site"),
    # Storm / baseline. 65 mm is what the spin box has opened on since the initial
    # commit; this table read 120.0 — the same fault recorded below, found a third time
    # and the widest of the three, because the design storm scales every runoff depth,
    # every feature's inflow and every spillway in the report.
    "rainfall_mm": (float, 65.0),
    "duration_hr": (float, 24.0),
    "soil_name": (str, ""),
    # Pasture hydrologic condition. Defaults to "good", which is what every design
    # written before this field existed implicitly assumed, so an older file reopens
    # with the curve numbers it was designed against.
    "ground_condition": (str, "good"),
    "cn": (int, 61),
    "moisture": (str, ""),
    # "dinf" is the combo's opening entry and the one it labels recommended. This read
    # "d8", which is not merely a different answer: pysheds' d8 accumulation calls
    # ``np.in1d``, which NumPy 2 removed, so a file restoring as d8 can raise outright
    # rather than quietly route the water differently.
    "routing": (str, "dinf"),
    "stream_threshold_ha": (float, 5.0),
    "exit_flow_ls": (float, 0.0),
    # Sizing basis (see STRETCH_GOALS "runoff basis and infiltration policy")
    "sizing_basis": (str, "coefficient"),
    "runoff_coefficient": (float, 0.5),
    "earthwork_soil_name": (str, ""),
    "peak_intensity_mm_hr": (float, 0.0),
    # Off by default, and deliberately so — soakage is derived from a soil-texture
    # lookup rather than a percolation test, so crediting it to capture would inflate
    # every feature's performance on a term the site has not verified.
    "count_infiltration": (bool, False),
    # Analysis tuning — not needed to rebuild the design (earthworks carry their own
    # dimensions), but restoring it is what makes a reopened file feel like the session
    # you left rather than a fresh one.
    # Every default here must equal the panel spin box it restores. Two of them did
    # not — ``simple_contour_interval_m`` was 5.0 against the panel's 1.0 and
    # ``max_slope_deg`` was 15.0 against the panel's 18.0 — and because
    # ``normalise_inputs`` fills every absent key from this table, reopening a file
    # saved before those keys existed silently re-answered the contour analysis at a
    # different interval and a different slope cutoff from the one on screen. This is
    # the same fault recorded for the swale trio below, found twice; the parity check
    # in ``tests_qgis/checks_design_file`` now walks ``panel._FIELD_WIDGETS`` and
    # asserts the whole table, so a third occurrence fails the suite instead of
    # shipping.
    "contour_interval_m": (float, 1.0),
    "simple_contour_interval_m": (float, 1.0),
    "max_slope_deg": (float, 18.0),
    "min_contour_length_m": (float, 50.0),
    "min_catchment_ha": (float, 0.5),
    # Keyline settings. Inputs only — the generated keylines themselves are derived
    # from the DEM plus these four numbers and are never serialised, under the same
    # rule that keeps a stage-storage curve and an auto spillway width out of the file.
    # Every default here must equal its spin box; the parity check walks them.
    "keypoint_count": (int, 5),
    "keyline_runs": (int, 3),
    "keyline_spacing_m": (float, 5.0),
    "keyline_max_grade_n": (int, 500),
    "keyline_max_valleys": (int, 8),
    # These three must match the panel spin boxes they restore, and they did not: the
    # pair was written as 0.6/2.0 while the criteria boxes were 0.3/0.6.
    # ``normalise_inputs`` fills every absent key from these defaults, so a file saved
    # before the keys existed came back with a different cross-section from the one the
    # session was using, and the next "Find Best Swale Segments" answered a different
    # question without saying so.
    #
    # All three now sit on ``core/registry``'s swale — 2.0 m top, 1.0 m floor, 0.5 m
    # deep, which works out at a 1:1 batter. A swale is built to three tape
    # measurements and dug with a flat bottom; the old criteria pair described a V-drain
    # of 0.09 m², which asked for tens of kilometres of swale on any real catchment.
    #
    # The batter is **not** stored. It is ``(top − bottom) / 2·depth`` and saving a
    # derived value beside the three it derives from is how a reloaded file comes back
    # describing a section that never existed.
    "swale_depth_m": (float, 0.5),
    "swale_width_m": (float, 2.0),
    "swale_bottom_width_m": (float, 1.0),
}

# Inputs whose value must come from a fixed set. An unrecognised value falls back to the
# default rather than propagating into an analysis run, because these are read as enums
# downstream (routing picks the flow algorithm; sizing_basis picks the runoff model;
# ground_condition picks a curve-number table) and a bogus string would fail somewhere
# far less obvious — or, worse, silently select the wrong table.
_ENUM_FIELDS = {
    "routing": ROUTING_VALUES,
    "sizing_basis": SIZING_BASIS_VALUES,
    "ground_condition": GROUND_CONDITION_VALUES,
}

# The three site polygons, carried as geometry rather than layer paths so they survive
# the trip to another machine.
AREA_KEYS = ("boundary", "analysis", "earthworks")


def _coerce(name, value):
    """Best-effort coercion of *value* to the declared type for *name*.

    Falls back to the declared default when the value is missing, ``None``, of an
    unconvertible type, or outside an enum's allowed set.
    """
    expected, default = INPUT_FIELDS[name]
    if value is None:
        return default
    try:
        # bool before int: bool is an int subclass, so an int branch would catch it first.
        if expected is bool:
            if isinstance(value, str):
                # A hand-edited file may carry "false", which is truthy to bool().
                lowered = value.strip().lower()
                if lowered in ("false", "0", "no", ""):
                    return False
                if lowered in ("true", "1", "yes"):
                    return True
                return default
            coerced = bool(value)
        elif expected is str:
            coerced = str(value)
        elif expected is int:
            coerced = int(round(float(value)))
        elif expected is float:
            coerced = float(value)
        else:  # pragma: no cover — no other declared types today
            coerced = expected(value)
    except (TypeError, ValueError):
        return default

    allowed = _ENUM_FIELDS.get(name)
    if allowed is not None and coerced not in allowed:
        return default
    return coerced


def default_inputs():
    """The full input set at its defaults — the baseline a partial document falls back to."""
    return {name: spec[1] for name, spec in INPUT_FIELDS.items()}


def normalise_inputs(raw):
    """Coerce and complete an arbitrary mapping into a full, valid input set.

    Unknown keys are dropped rather than carried: they are either a typo or a field from
    a newer schema this build cannot act on, and keeping them would imply otherwise.
    """
    raw = raw or {}
    return {name: _coerce(name, raw.get(name)) for name in INPUT_FIELDS}


@dataclass
class DemReference:
    """How a design file carries its DEM.

    In ``reference`` mode only the fingerprint travels and the DEM must be found on the
    opening machine; in ``embedded`` mode the archive holds a clip. The fingerprint is
    kept in *both* modes so an embedded clip can still be checked against the DEM it came
    from, and so a reference-mode locate can be verified rather than trusted.
    """

    mode: str = "reference"
    # Archive-relative name of the embedded raster; None in reference mode.
    filename: str | None = None
    # Original path, purely informational — shown to the user when asking them to locate
    # a missing DEM. Never trusted for resolution.
    original_path: str | None = None
    # Identity of the DEM: content hash plus the grid properties that would change the
    # analysis even if the pixels matched.
    fingerprint: str | None = None
    cell_size_m: float | None = None
    crs: str | None = None
    extent: list | None = None          # [xmin, ymin, xmax, ymax]
    width: int | None = None
    height: int | None = None
    # True when `filename` is a clip rather than the whole DEM. Recorded so the
    # difference is never silent: baseline re-runs on the clipped extent, so exit points
    # that fell outside it are gone.
    is_clip: bool = False
    clip_buffer_m: float = 0.0

    def to_dict(self):
        return {
            "mode": self.mode,
            "filename": self.filename,
            "original_path": self.original_path,
            "fingerprint": self.fingerprint,
            "cell_size_m": self.cell_size_m,
            "crs": self.crs,
            "extent": list(self.extent) if self.extent else None,
            "width": self.width,
            "height": self.height,
            "is_clip": bool(self.is_clip),
            "clip_buffer_m": float(self.clip_buffer_m or 0.0),
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        mode = data.get("mode")
        if mode not in DEM_MODES:
            # An unknown mode is safer read as "reference": it makes the opener resolve
            # and verify a DEM rather than trust an archive member that may not exist.
            mode = "reference"

        extent = data.get("extent")
        try:
            extent = [float(v) for v in extent] if extent else None
            if extent is not None and len(extent) != 4:
                extent = None
        except (TypeError, ValueError):
            extent = None

        def _num(key, cast):
            try:
                value = data.get(key)
                return cast(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        return cls(
            mode=mode,
            filename=data.get("filename"),
            original_path=data.get("original_path"),
            fingerprint=data.get("fingerprint"),
            cell_size_m=_num("cell_size_m", float),
            crs=data.get("crs"),
            extent=extent,
            width=_num("width", int),
            height=_num("height", int),
            is_clip=bool(data.get("is_clip", False)),
            clip_buffer_m=_num("clip_buffer_m", float) or 0.0,
        )

    def matches(self, other):
        """Whether *other* describes the same DEM as this reference.

        Compared on the content hash when both have one, because that is the only field
        that cannot coincide by accident. Grid properties alone are not enough — two
        different rasters over the same site share them.
        """
        if not isinstance(other, DemReference):
            return False
        if self.fingerprint and other.fingerprint:
            return self.fingerprint == other.fingerprint
        return False


@dataclass
class DesignDocument:
    """The whole saveable design: inputs, site polygons, DEM reference, earthworks.

    ``earthworks`` holds the payload produced by
    :meth:`~terrainflow_assessment.modules.earthwork_design.EarthworkManager.to_json`,
    parsed. Delegating rather than re-describing earthwork fields here keeps one schema
    for them — the manager's ``_SERIAL_FIELDS`` stays the only place that list lives.
    """

    inputs: dict = field(default_factory=default_inputs)
    # key → {"wkt": str, "crs": str | None}; absent keys mean the user never set that area.
    areas: dict = field(default_factory=dict)
    dem: DemReference = field(default_factory=DemReference)
    earthworks: dict = field(default_factory=dict)
    # IDF table JSON (see modules.rainfall_idf). None until the user enters one — its
    # absence is a real state, not a default.
    idf: str | None = None
    # Version this document was *read* from. Equals SCHEMA_VERSION for anything this
    # build wrote; higher means the file came from a newer build and may have lost
    # fields on load, which the caller should surface.
    source_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(cls, inputs, areas=None, dem=None, earthworks_json=None, idf=None):
        """Assemble a document from live session values.

        *earthworks_json* is the string from ``EarthworkManager.to_json()``; it is parsed
        so the result is one JSON document rather than JSON nested inside a JSON string.
        """
        return cls(
            inputs=normalise_inputs(inputs),
            areas=_clean_areas(areas),
            dem=dem if isinstance(dem, DemReference) else DemReference.from_dict(dem),
            earthworks=_parse_earthworks(earthworks_json),
            idf=idf or None,
        )

    # ------------------------------------------------------------------ serialisation

    def to_dict(self):
        return {
            "format": "terrainflow-design",
            "version": SCHEMA_VERSION,
            "inputs": dict(self.inputs),
            "areas": _clean_areas(self.areas),
            "dem": self.dem.to_dict(),
            "earthworks": dict(self.earthworks) if self.earthworks else {},
            "idf": self.idf,
        }

    def to_json(self, indent=2):
        """JSON text for the archive's ``design.json``.

        Indented and key-sorted: these files land in git as test fixtures, and a stable
        byte order makes a real change visible in a diff instead of drowning in churn.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data):
        """Rebuild from :meth:`to_dict`, filling anything missing with defaults."""
        data = data if isinstance(data, dict) else {}
        try:
            version = int(data.get("version", SCHEMA_VERSION))
        except (TypeError, ValueError):
            version = SCHEMA_VERSION

        return cls(
            inputs=normalise_inputs(data.get("inputs")),
            areas=_clean_areas(data.get("areas")),
            dem=DemReference.from_dict(data.get("dem")),
            earthworks=_parse_earthworks(data.get("earthworks")),
            idf=data.get("idf") or None,
            source_version=version,
        )

    @classmethod
    def from_json(cls, text):
        """Rebuild from JSON text. Returns ``None`` only when nothing can be read at all.

        A blank or malformed file is genuinely unopenable and must be reported, which is
        why this is the one path that yields ``None`` rather than defaults — silently
        presenting an empty design as a successful load would be worse.
        """
        if not text:
            return None
        try:
            payload = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return cls.from_dict(payload)

    # ------------------------------------------------------------------ queries

    def earthworks_json(self):
        """The earthwork payload as a string for ``EarthworkManager.from_json``."""
        return json.dumps(self.earthworks or {})

    def earthwork_count(self):
        items = (self.earthworks or {}).get("earthworks")
        return len(items) if isinstance(items, list) else 0

    def is_from_newer_build(self):
        """Whether this document came from a schema this build does not fully understand."""
        return self.source_version > SCHEMA_VERSION


def _parse_earthworks(value):
    """Normalise an earthwork payload given as a JSON string, a dict, or nothing."""
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_areas(areas):
    """Keep only recognised area keys that actually carry geometry."""
    if not isinstance(areas, dict):
        return {}
    cleaned = {}
    for key in AREA_KEYS:
        entry = areas.get(key)
        if not isinstance(entry, dict):
            continue
        wkt = entry.get("wkt")
        if not wkt:
            continue
        cleaned[key] = {"wkt": str(wkt), "crs": entry.get("crs")}
    return cleaned
