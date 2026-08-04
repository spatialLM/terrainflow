"""
rainfall_idf.py — intensity-duration-frequency data, entered by the user.

The rational method needs the rainfall intensity for a storm whose duration equals the
catchment's time of concentration, at a chosen return period. That is not derivable
from a DEM: it is regional rainfall statistics. In New Zealand it comes from NIWA's
High Intensity Rainfall Design System (HIRDS v4, https://hirds.niwa.co.nz), which
returns a depth-duration-frequency table for any location.

So the plugin cannot compute this and does not pretend to. It accepts the table the
user looked up, and does the two things it legitimately can: interpolate to the
duration their catchment actually responds at, and convert depth to intensity.

Interpolation is **log-log** across duration. Depth-duration curves are close to
straight on log-log axes over the range HIRDS reports, and linear interpolation
between, say, the 10-minute and 60-minute rows would understate everything in
between — the region where small catchments live. Outside the entered range the
value is clamped rather than extrapolated: an IDF curve fitted to 10 minutes says
nothing trustworthy about 2 minutes, and a silent extrapolation there would be
invented data presented as a lookup.

Pure: no QGIS, no I/O beyond parsing text the caller supplies.
"""

import json
import math

# HIRDS reports these durations by default. Offered as the table's starting rows so a
# user can paste a column of depths straight down beside them.
HIRDS_DURATIONS_MIN = (10.0, 20.0, 30.0, 60.0, 120.0, 360.0, 720.0, 1440.0)

# Annual recurrence intervals HIRDS reports, in years.
HIRDS_ARI_YEARS = (2, 5, 10, 20, 50, 100)

# The 2-year 24-hour depth is a separate input to TR-55's sheet-flow travel time. It
# is a row of the same table (ARI 2, duration 1440 min), so entering the table once
# supplies it — no second question.
SHEET_FLOW_ARI_YEARS = 2
SHEET_FLOW_DURATION_MIN = 1440.0


class IDFTable:
    """Depth-duration-frequency data for one site, as looked up by the user.

    Stored as ``{ari_years: {duration_min: depth_mm}}``. Sparse is fine — a user who
    only entered the 50-year row gets 50-year answers and nothing else, which is
    honest, rather than a curve fitted through one point.
    """

    def __init__(self, depths=None, source="", site=""):
        self.depths = {}
        for ari, rows in (depths or {}).items():
            clean = {float(d): float(mm) for d, mm in (rows or {}).items()
                     if d and mm is not None and float(mm) > 0}
            if clean:
                self.depths[int(ari)] = clean
        self.source = source or ""     # e.g. "HIRDS v4"
        self.site = site or ""         # free text: coordinates, place name, run date

    # ------------------------------------------------------------------ queries

    def has_data(self):
        return bool(self.depths)

    def available_aris(self):
        return sorted(self.depths)

    def depth_mm(self, duration_min, ari_years):
        """Rainfall depth (mm) for a duration and return period, log-log interpolated.

        Returns None when that return period has no data. Durations outside the
        entered range are clamped to the nearest end — see the module note on why
        extrapolation is refused.
        """
        rows = self.depths.get(int(ari_years))
        if not rows or duration_min is None or duration_min <= 0:
            return None

        points = sorted(rows.items())
        if len(points) == 1:
            return points[0][1]

        d = float(duration_min)
        if d <= points[0][0]:
            return points[0][1]
        if d >= points[-1][0]:
            return points[-1][1]

        for (d0, p0), (d1, p1) in zip(points, points[1:]):
            if d0 <= d <= d1:
                if d1 == d0 or p0 <= 0 or p1 <= 0:
                    return p0
                # Straight line in log-log space through the bracketing pair.
                t = (math.log(d) - math.log(d0)) / (math.log(d1) - math.log(d0))
                return math.exp(math.log(p0) + t * (math.log(p1) - math.log(p0)))
        return points[-1][1]

    def intensity_mm_hr(self, duration_min, ari_years):
        """Average intensity (mm/hr) over a storm of *duration_min* — depth / duration.

        This is the quantity the rational method wants: the mean intensity over a
        storm whose duration equals the time of concentration, not an instantaneous
        rate within it.
        """
        depth = self.depth_mm(duration_min, ari_years)
        if depth is None or duration_min is None or duration_min <= 0:
            return None
        return depth * 60.0 / float(duration_min)

    def sheet_flow_p2_mm(self):
        """2-year 24-hour depth (mm) for TR-55 sheet flow, or None if not entered."""
        return self.depth_mm(SHEET_FLOW_DURATION_MIN, SHEET_FLOW_ARI_YEARS)

    def is_extrapolated(self, duration_min):
        """True when *duration_min* falls outside the entered range, so the answer
        was clamped. The caller should say so rather than quoting it plainly."""
        if not self.depths or duration_min is None:
            return False
        durations = {d for rows in self.depths.values() for d in rows}
        if not durations:
            return False
        return not (min(durations) <= float(duration_min) <= max(durations))

    # ------------------------------------------------------------------ transport

    def to_dict(self):
        return {
            "source": self.source,
            "site": self.site,
            "depths": {str(a): {str(d): v for d, v in rows.items()}
                       for a, rows in self.depths.items()},
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            return cls()
        return cls(
            depths={int(a): {float(d): float(v) for d, v in rows.items()}
                    for a, rows in (data.get("depths") or {}).items()},
            source=data.get("source", ""),
            site=data.get("site", ""),
        )

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text):
        if not text:
            return cls()
        try:
            return cls.from_dict(json.loads(text))
        except (TypeError, ValueError):
            return cls()


def parse_hirds_text(text):
    """Best-effort parse of a table pasted out of HIRDS — ``(IDFTable, [problems])``.

    HIRDS exports and on-screen tables vary, so this is forgiving by design: it looks
    for a header row naming the return periods and reads duration/depth rows beneath.
    Anything it cannot read is reported rather than guessed at, because a silently
    misread rainfall depth would propagate into every spillway on the site.

    Accepts comma, tab or whitespace separation. Durations may be given as ``10m``,
    ``10 min``, ``1h``, ``24 hr`` or a bare number (assumed minutes).
    """
    problems = []
    if not text or not text.strip():
        return IDFTable(), ["Nothing to read."]

    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    aris = None
    depths = {}

    for lineno, line in enumerate(lines, start=1):
        fields = _split_fields(line)
        if not fields:
            continue

        if aris is None:
            header = _parse_ari_header(fields)
            if header:
                aris = header
                continue
            problems.append(
                f"Line {lineno}: expected a header naming the return periods "
                f"(e.g. 'duration, 2, 5, 10, 20, 50, 100')."
            )
            continue

        duration = _parse_duration(fields[0])
        if duration is None:
            problems.append(f"Line {lineno}: could not read '{fields[0]}' as a duration.")
            continue

        values = fields[1:]
        if len(values) < len(aris):
            problems.append(
                f"Line {lineno}: {len(values)} depths for {len(aris)} return periods."
            )
        elif len(values) > len(aris):
            # The signature of thousands separators in comma-separated text: '1,180.0'
            # splits into '1' and '180.0', and the row silently becomes 1 mm. NZ's
            # wettest 24-hour depths do exceed 1000 mm, so this is reachable, and a
            # depth read as 1 mm would size every spillway below it at nothing.
            problems.append(
                f"Line {lineno}: {len(values)} values for {len(aris)} return periods — "
                f"if depths use a thousands separator (1,180.0), remove it or paste "
                f"the table tab-separated."
            )
        for ari, raw in zip(aris, values):
            value = _parse_number(raw)
            if value is None or value <= 0:
                continue
            depths.setdefault(ari, {})[duration] = value

    if aris is None:
        problems.append("No header row found, so the return periods are unknown.")
    if not depths:
        problems.append("No usable depths were found.")
    return IDFTable(depths=depths, source="HIRDS v4"), problems


def _split_fields(line):
    # Tabs are tested first on purpose. A tab-separated line is unambiguous even when
    # its numbers contain commas as thousands separators; splitting on the comma
    # first would tear '1,180.0' in half and read the row as 1 mm.
    if "\t" in line:
        return [f.strip() for f in line.split("\t") if f.strip()]
    if "," in line:
        return [f.strip() for f in line.split(",") if f.strip()]
    return line.split()


def _parse_ari_header(fields):
    """Return periods from a header row, or None when it is not a header."""
    values = []
    for raw in fields[1:]:
        n = _parse_number(raw)
        if n is None or n <= 0 or n != int(n):
            return None
        values.append(int(n))
    return values or None


def _parse_duration(raw):
    """Duration in minutes from '10', '10m', '10 min', '1h', '24 hr', '1440'."""
    text = str(raw).strip().lower().replace(" ", "")
    multiplier = 1.0
    for suffix, mult in (("mins", 1.0), ("min", 1.0), ("hrs", 60.0), ("hr", 60.0),
                         ("hours", 60.0), ("hour", 60.0), ("h", 60.0), ("m", 1.0)):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            multiplier = mult
            break
    value = _parse_number(text)
    if value is None or value <= 0:
        return None
    return value * multiplier


def _parse_number(raw):
    try:
        return float(str(raw).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
