"""
_symbols.py — canvas symbology for the drawn earthworks.

Every symbol the Design stage puts on the map is built here, in one place, so the
visual grammar can be read as a whole rather than reconstructed from a 3,000-line
controller. `earthworks.py` holds the QGIS wiring; this module holds the look.

The governing rule, and the reason the module exists:

    **Structure is drawn in metres. Identity and direction are drawn in millimetres.**

A band that answers *how big is it really* has to be `RenderMetersInMapUnits` and
data-defined from ``width_m``, so it tracks the true ground footprint at every zoom.
A mark that answers *what is it, and which way does water go* has to be in
millimetres, so it stays legible at 1:200 and 1:20,000 alike. Mixing the two units
inside one symbol layer is what made the old berm dash scale with the band and the
old diversion arrow outweigh the drain it decorated.

Same split `baseline.py` already makes for exit points.

Every embellishment is best-effort: on any failure the builders degrade to a plain
coloured symbol so layer creation never breaks on a QGIS API difference.
"""

from __future__ import annotations

from qgis.core import (
    QgsFillSymbol,
    QgsLineSymbol,
    QgsPalLayerSettings,
    QgsProperty,
    QgsSimpleFillSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsSymbolLayer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsUnitTypes,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QFont

# Greyed-out stand-in for a disabled earthwork. Not in the registry because it is a
# UI state, not a feature type.
DISABLED_COLOUR = "#9aa4a2"


def band_max(layer, band=1):
    """The band maximum a ramp stretches to. Never zero, never negative."""
    try:
        stats = layer.dataProvider().bandStatistics(band)
        maximum = stats.maximumValue or 1.0
    except Exception:
        maximum = 1.0
    return maximum if maximum > 0 else 1.0


def apply_raster_ramp(layer, stops, max_value=None, min_value=None,
                      fade_from=0.0, absolute=False):
    """Paint ``layer`` with a ``core.registry.map_palette`` ramp.

    ``stops`` are ``(fraction_of_max, (r, g, b, a), label)``. Baseline and the
    simulation both draw the same quantities — captured water, the channel
    network — and each used to carry its own copy of the stops, so a change on
    one side left two views of one thing in different colours.

    ``absolute=True`` says the first element of each stop is **already a value in the
    band's own units**, not a fraction, and lays the stops down untouched. Aspect is the
    case that needs it: ``ASPECT_CLASSES`` declares compass degrees, and scaling those by
    the band maximum put the nine stops at −360, 0, 16200 … 113400 against data that only
    ever spans [−1, 360]. Every real value then landed inside the first stop's colour and
    the whole map drew as one flat wash — measured at **0.32 % of the ramp occupied**.
    A palette whose values mean something absolute has no business being multiplied by
    whatever the brightest cell happens to be.

    ``min_value`` moves the **bottom of the colour ramp** to an absolute value:
    the stops are laid out over ``[min_value, top]`` rather than ``[0, top]``,
    and one extra stop is prepended at ``fade_from`` carrying the lowest colour
    at zero alpha. Between the two the shader interpolates alpha, so the layer
    comes up out of nothing over that range instead of switching on at an edge —
    a threshold drawn as an edge reads as water *stopping* there. Values below
    ``fade_from`` clamp onto that transparent stop, which is what makes one stop
    enough. See ``map_palette.SURFACE_RUNOFF_FADE_TOP_M3``, the only caller.

    A ``min_value`` at or above ``top`` is ignored: blanking or flattening a
    whole layer is never the useful reading of a display floor.
    """
    from qgis.core import (
        QgsColorRampShader,
        QgsRasterShader,
        QgsSingleBandPseudoColorRenderer,
    )

    if absolute:
        items = [
            QgsColorRampShader.ColorRampItem(float(value), QColor(*rgba), label)
            for value, rgba, label in stops
        ]
        floor = 0.0
    else:
        top = band_max(layer) if max_value is None else max_value
        if not top or top <= 0:
            top = 1.0
        try:
            floor = float(min_value or 0.0)
        except (TypeError, ValueError):
            floor = 0.0
        if floor <= 0.0 or floor >= top:
            floor = 0.0
        span = top - floor
        items = [
            QgsColorRampShader.ColorRampItem(floor + span * fraction,
                                             QColor(*rgba), label)
            for fraction, rgba, label in stops
        ]
    if floor and stops:
        clear = QColor(*stops[0][1])
        clear.setAlpha(0)
        items.insert(0, QgsColorRampShader.ColorRampItem(
            float(fade_from), clear, stops[0][2]))
    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(items)
    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    layer.setRenderer(
        QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))


def apply_shared_ramp(state, project, family, layer, stops,
                      max_value=None, min_value=None):
    """Paint ``layer`` on a ramp whose top every layer of ``family`` shares.

    A Baseline layer and its Earthworks counterpart answer the same question
    about the same ground, and the only reason to draw both is to read one
    against the other. That works only while a colour means one quantity on
    both — and it did not. Each layer stretched its stops over its own band
    maximum, so the same two metres of water was mid-blue before the design and
    navy after it purely because the deepest pond on the site had moved, and the
    "difference" the pair appeared to show was the ramp rescaling itself. The
    event pond was scaled to the full pond deliberately for exactly this reason;
    this is that argument applied to every pair rather than to one of them.

    A family's top is the largest maximum any live member has claimed. Raising it
    repaints the earlier members, because a shared scale only the last layer drawn
    is on is not shared. Members are held by id and pruned when they no longer
    resolve, so a cleared stage group releases its claim on the scale instead of
    propping it up with rasters nobody can see.

    ``state`` carries ``ramp_scales``; ``project`` resolves the ids.
    """
    from terrainflow_assessment.qgis.controllers._layers import resolve_layer

    entry = state.ramp_scales.setdefault(family, {"members": {}})
    claimed = band_max(layer) if max_value is None else max_value
    entry["members"][layer.id()] = float(claimed) if claimed and claimed > 0 else 1.0

    live = {}
    for layer_id, top in entry["members"].items():
        found = layer if layer_id == layer.id() else resolve_layer(project, layer_id)
        if found is not None:
            live[layer_id] = (found, top)
    entry["members"] = {lid: top for lid, (_l, top) in live.items()}

    shared = max((top for _l, top in live.values()), default=1.0)
    entry["top"] = shared
    for member, _top in live.values():
        apply_raster_ramp(member, stops, shared, min_value)
        if member is not layer:
            member.triggerRepaint()

# --- the metres/millimetres split, as numbers -------------------------------
#
# BAND_* is ground truth: the coloured band is exactly top_width_m, so measuring it
# off the screen gives the real footprint. BAND_MIN_MM stops it disappearing when
# that footprint goes sub-pixel — without it a 2 m swale renders 0 px at 1:15,000
# (measured), which is why a fixed-mm "pin" used to be drawn over the top.
#
# CASING_* is legibility, not dimension. It is the white outline that keeps a band
# readable over aerial imagery. It sits outside the band and is not read as
# earthwork, which is what makes the +0.6 m acceptable while the band itself stays
# exact.
#
# Note for anyone tempted by a screen-constant halo: `@map_scale` is NOT available
# in a symbol-layer data-defined expression on 3.40. It evaluates to NULL and QGIS
# silently falls back to the static width, which looks like it works. Verified by
# rendering a 99 m static fallback: `DD = 10` gives 10 m, `DD = @map_scale/1000`
# gives 99 m.
BAND_MIN_MM = 0.6
CASING_ADD_M = 0.6
CASING_MIN_MM = 1.8

BAND_EXPR = 'coalesce("width_m", 0)'
CASING_EXPR = f'coalesce("width_m", 0) + {CASING_ADD_M}'

# --- per-type signature ------------------------------------------------------
#
# Deliberately a local table rather than new fields on EarthworkTypeConfig: the
# registry is owned by other work, and colour/nominal width still come from it, so
# the panel's draw-button chips stay in step automatically.
#
# Intervals are millimetres, which is the whole point — marker cadence becomes a
# function of how long the line is *on screen*, not of how many vertices the draw
# tool happened to emit. That is what makes a signature survive zooming.
SIG_NONE = "none"
SIG_ARROW = "arrow"       # direction of flow along the line
SIG_TICKS = "ticks"       # barrier: hachures straddling the crest
SIG_CHEVRON = "chevron"   # barrier with a protected side: hachures on one side

_GRAMMAR = {
    # key         signature     interval mm   size mm
    "swale":     (SIG_NONE,     0.0,  0.0),
    "berm":      (SIG_CHEVRON,  4.5,  2.6),
    "dam":       (SIG_TICKS,    3.0,  2.2),
    "diversion": (SIG_ARROW,    8.0,  2.6),
    "basin":     (SIG_NONE,     0.0,  0.0),
}


def signature_for(key):
    return _GRAMMAR.get(key, (SIG_NONE, 0.0, 0.0))


# Label priorities. When PAL runs out of room it drops the low numbers first, so
# identity ("Swale 1") outlives detail ("outflow · 70.07 m").
PRIORITY_EARTHWORK = 7
PRIORITY_STRESS = 5
PRIORITY_SPILLWAY = 4

# Above these denominators the text is noise rather than information. Geometry is
# never suppressed — only labels.
MAX_SCALE_POINT_LABEL = 5000.0


# ---------------------------------------------------------------- labels

def label_colour(hex_or_qcolour):
    """Darken a type colour until it is readable as text.

    The map's colour grammar ties a feature to its panel chip, so labels keep the
    type hue rather than going flat black. But #00BCD4 cyan at 9 pt over a white
    halo is roughly 1.5:1 contrast — the hue has to come down about 40% in
    luminance before it is text rather than decoration.
    """
    # Both arms of the conditional that used to be here were identical (ruff
    # RUF034). `QColor(QColor)` is already the copy constructor, so a hex string
    # and a QColor both arrive as a fresh QColor and neither is mutated.
    c = QColor(hex_or_qcolour)
    h, s, v, a = c.getHsv()
    return QColor.fromHsv(h, min(255, int(s * 1.1)), max(0, int(v * 0.6)), a)


def label_format(colour, size_pt=9.0, halo=True, bold=True):
    """The one place text formatting is built.

    ``QgsTextFormat.setFont()`` silently discards a QFont's point size — the trap
    already documented at ``baseline.py:472-474`` and the reason earthwork labels
    have been rendering at the 10 pt default instead of the intended 8. Size must
    go through ``setSize()`` + ``setSizeUnit()``. Routing every caller through
    here is what stops that recurring at the next call site.

    The halo is not optional in practice: over aerial imagery, unbuffered text
    dissolves into canopy.
    """
    fmt = QgsTextFormat()
    font = QFont()
    font.setBold(bold)
    fmt.setFont(font)
    fmt.setSize(float(size_pt))
    fmt.setSizeUnit(QgsUnitTypes.RenderPoints)
    fmt.setColor(QColor(colour) if not isinstance(colour, QColor) else colour)
    if halo:
        buf = QgsTextBufferSettings()
        buf.setEnabled(True)
        buf.setColor(QColor(255, 255, 255))
        buf.setSize(1.0)
        fmt.setBuffer(buf)
    return fmt


def _placement(name):
    """Resolve a placement enum across the 3.22 → 3.40 rename, newest spelling first."""
    from qgis.core import Qgis
    enum = getattr(Qgis, "LabelPlacement", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return getattr(QgsPalLayerSettings, name)


def line_label_settings(field_or_expr, fmt, is_expression=False,
                        priority=PRIORITY_EARTHWORK):
    """Labels that follow a line.

    The default placement is ``AroundPoint``, which is a *point* arrangement: on a
    LineString layer it yields no label at all. That single unset property is why
    no swale, berm, dam or diversion drain has ever carried a name on the map.
    ``Curved`` follows the alignment; ``Line`` is the straight-text equivalent.
    """
    s = QgsPalLayerSettings()
    s.fieldName = field_or_expr
    s.isExpression = is_expression
    s.enabled = True
    s.setFormat(fmt)
    s.placement = _placement("Curved")
    s.dist = 2.0            # lift the text off the band so it does not sit in the ink
    s.priority = priority

    # Let the text run past the ends of a short line rather than be dropped.
    # Without this, zooming out silently loses the name of whichever feature is
    # shortest on screen: at 1:10,000 an 80 m drain is ~30 px, far less than
    # "Diversion 5" needs, and PAL discards it while its longer neighbours keep
    # theirs. Losing labels in an order that tracks nothing the user can see is
    # worse than a little overhang.
    try:
        ls = s.lineSettings()
        ls.setOverrunDistance(25.0)
        ls.setOverrunDistanceUnit(QgsUnitTypes.RenderMillimeters)
        s.setLineSettings(ls)
    except Exception:
        pass
    return s


def polygon_label_settings(field_or_expr, fmt, is_expression=False,
                           priority=PRIORITY_EARTHWORK):
    """Labels inside a footprint. A basin already reads as an area, so horizontal
    text at the centroid is clearer than text bent around the ring."""
    s = QgsPalLayerSettings()
    s.fieldName = field_or_expr
    s.isExpression = is_expression
    s.enabled = True
    s.setFormat(fmt)
    s.placement = _placement("Horizontal")
    s.priority = priority
    return s


def earthwork_label_settings(cfg, colour_hex):
    """The drawn earthwork's name, e.g. "Swale 1".

    Type is carried by the symbol's colour and signature, so it is not repeated
    in the text — and neither is capacity. The label used to read
    "Swale 26 · 130 m³", which doubled its length for a number nobody reads off
    a map, and wherever features cluster (a dozen swales across one face) the
    result was a wall of overlapping text with the design invisible under it.
    Capacity is a column in "Water arriving at each feature", where it can be
    compared against the next feature's rather than hunted for.
    """
    expr = "\"name\""
    fmt = label_format(label_colour(colour_hex), size_pt=9.0)
    if cfg.geom_type == "Polygon":
        return polygon_label_settings(expr, fmt, is_expression=True)
    return line_label_settings(expr, fmt, is_expression=True)


def point_label_settings(field_or_expr, fmt, is_expression=False,
                         priority=PRIORITY_SPILLWAY, max_scale=None):
    """Labels anchored above a marker.

    Mirrors ``baseline.py``'s exit-point configuration — the one labelling setup in
    the plugin that has always demonstrably rendered.
    """
    s = QgsPalLayerSettings()
    s.fieldName = field_or_expr
    s.isExpression = is_expression
    s.enabled = True
    s.setFormat(fmt)
    s.placement = _placement("OverPoint")
    try:
        s.quadOffset = QgsPalLayerSettings.QuadrantAbove
    except AttributeError:
        pass
    s.yOffset = 2.0
    s.dist = 2.5
    s.priority = priority
    if max_scale:
        s.scaleVisibility = True
        # "minimum" is the smallest denominator (most zoomed in) at which the label
        # shows; "maximum" is the largest. Zoomed out past max_scale, text goes.
        s.minimumScale = float(max_scale)
        s.maximumScale = 0.0
    return s


# ---------------------------------------------------------------- earthwork symbols

def earthwork_symbol(cfg, enabled=True):
    """Rich per-type canvas symbol. Casing (white underlay) for legibility on any
    background, a per-type line signature (dam ticks / diversion flow arrows /
    berm + diversion dash patterns), and a greyed dashed variant when disabled.
    Every embellishment is best-effort — on any failure it degrades to a plain
    coloured symbol so layer creation never breaks."""
    base = QColor(cfg.style[1])
    try:
        main_w = float(cfg.style[2])
    except (TypeError, ValueError):
        main_w = 2.0
    colour = base if enabled else QColor(DISABLED_COLOUR)

    if cfg.geom_type == "Polygon":
        return earthwork_fill_symbol(colour, enabled, main_w)
    return earthwork_line_symbol(cfg.key, colour, enabled, main_w)


def earthwork_fill_symbol(colour, enabled, main_w):
    """Basin footprint. The polygon is already true to the ground, so this is the
    one type where dimensional fidelity comes free and the work is all contrast:
    at 17.6% alpha over aerial imagery the basin was the faintest thing on the map.
    A slightly stronger fill plus a white-cased edge fixes that without hiding the
    ponding raster underneath.
    """
    try:
        rgba = QColor(colour.red(), colour.green(), colour.blue(), 52 if enabled else 24)
        fill = QgsSimpleFillSymbolLayer(rgba)
        fill.setStrokeStyle(Qt.PenStyle.NoPen)      # the edge is its own layer

        # Stroke-only fill layers: a QgsFillSymbol holds fill layers, so an outline
        # is a fill layer with no brush rather than a line layer.
        casing = QgsSimpleFillSymbolLayer(QColor(0, 0, 0, 0))
        casing.setBrushStyle(Qt.BrushStyle.NoBrush)
        casing.setStrokeColor(QColor(255, 255, 255, 220))
        casing.setStrokeWidth(1.6)                  # millimetres
        casing.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)

        edge = QgsSimpleFillSymbolLayer(QColor(0, 0, 0, 0))
        edge.setBrushStyle(Qt.BrushStyle.NoBrush)
        edge.setStrokeColor(colour)
        edge.setStrokeWidth(0.7 if enabled else 0.5)
        edge.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if not enabled:
            edge.setStrokeStyle(Qt.PenStyle.DashLine)

        symbol = QgsFillSymbol([fill])
        symbol.appendSymbolLayer(casing)
        symbol.appendSymbolLayer(edge)
        return symbol
    except Exception:
        return QgsFillSymbol.createSimple(
            {"style": "no", "outline_color": colour.name(), "outline_width": str(main_w)}
        )


def _clamped(layer, min_mm):
    """Give a metres-in-map-units width a minimum on-screen size."""
    from qgis.core import QgsMapUnitScale
    try:
        scale = QgsMapUnitScale()
        scale.minSizeMMEnabled = True
        scale.minSizeMM = min_mm
        layer.setWidthMapUnitScale(scale)
    except Exception:
        pass
    return layer


def earthwork_line_symbol(key, colour, enabled, main_w):
    try:
        layers = []

        # --- casing: white outline, legibility only (see CASING_* above) ------
        casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 235))
        casing.setWidth(main_w + CASING_ADD_M)     # static fallback, metres
        casing.setWidthUnit(QgsUnitTypes.RenderMetersInMapUnits)
        casing.setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth,
            QgsProperty.fromExpression(CASING_EXPR),
        )
        casing.setPenCapStyle(Qt.PenCapStyle.RoundCap)
        casing.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
        _clamped(casing, CASING_MIN_MM)
        layers.append(casing)

        # --- band: the real ground width, always -----------------------------
        # Disabled features keep their true width too. "Off" is a colour
        # statement; shrinking the band would make it a dimensional lie, and a
        # disabled earthwork is still that many metres wide.
        band = QgsSimpleLineSymbolLayer(colour)
        band.setWidth(main_w)
        band.setWidthUnit(QgsUnitTypes.RenderMetersInMapUnits)
        band.setDataDefinedProperty(
            QgsSymbolLayer.PropertyStrokeWidth,
            QgsProperty.fromExpression(BAND_EXPR),
        )
        band.setPenCapStyle(Qt.PenCapStyle.RoundCap)
        band.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
        _clamped(band, BAND_MIN_MM)
        layers.append(band)

        # --- signature: millimetres, so it survives zoom ----------------------
        if enabled:
            sig, interval_mm, size_mm = signature_for(key)
            if sig != SIG_NONE:
                marker = _signature_layer(sig, colour, interval_mm, size_mm)
                if marker is not None:
                    layers.append(marker)
        else:
            # The one dash in the grammar, and it means exactly one thing: off.
            # It rides on a millimetre layer, never on the metres band — Qt scales
            # dash length with pen width, so a dash on the band is ground-
            # referenced and dissolves into detached capsules as you zoom.
            off = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 210))
            off.setWidth(0.9)                     # millimetres
            off.setPenStyle(Qt.PenStyle.DashLine)
            off.setPenCapStyle(Qt.PenCapStyle.FlatCap)
            layers.append(off)

        return QgsLineSymbol(layers)
    except Exception:
        return QgsLineSymbol.createSimple(
            {"color": colour.name(), "width": str(main_w),
             "capstyle": "round", "joinstyle": "round"}
        )


def _signature_layer(sig, colour, interval_mm, size_mm):
    """The per-type mark that rides on top of the band, in millimetres.

    Replaces QgsArrowSymbolLayer outright rather than repairing it. That class was
    being driven through setArrowHeadLength/setArrowHeadThickness, which do not
    exist on it (the real API is setHeadLength/setHeadThickness) — hasattr-guarded,
    so the calls silently no-opped and every arrowhead stayed at the 1.5 mm
    default. It also defaults isCurved=True, bowing the ribbon off the alignment,
    and its sub-symbol was the same colour as the band underneath it. Repairing
    the method names would have fixed one of those three.

    A marker line placed at a millimetre interval has none of those problems and
    gives a constant on-screen cadence for free. Returns None on failure, so a
    missing API degrades to a plain band rather than breaking the layer.
    """
    from qgis.core import (
        QgsHashedLineSymbolLayer,
        QgsMarkerLineSymbolLayer,
        QgsMarkerSymbol,
    )
    try:
        if sig in (SIG_TICKS, SIG_CHEVRON):
            # Hachures, the conventional embankment mark. Dam ticks straddle the
            # crest; berm ticks are offset so they fall on one side only, which is
            # what distinguishes "barrier" from "barrier with a protected side".
            #
            # Berms deliberately do NOT get a marker sitting on the band: white
            # markers along a green band read as a dashed line, and dashed already
            # means disabled. One cue, one meaning.
            hashed = QgsHashedLineSymbolLayer()
            _set_interval(hashed, interval_mm)
            hashed.setHashLength(size_mm)
            hashed.setHashLengthUnit(QgsUnitTypes.RenderMillimeters)
            hashed.setHashAngle(90)          # across the crest, not along it
            if sig == SIG_CHEVRON:
                hashed.setOffset(size_mm / 2.0)
                hashed.setOffsetUnit(QgsUnitTypes.RenderMillimeters)
            sub = QgsLineSymbol.createSimple({
                "color": "255,255,255,235", "width": "0.4",
                "capstyle": "flat",
            })
            hashed.setSubSymbol(sub)
            hashed.setAverageAngleLength(4.0)
            return hashed

        marker_line = QgsMarkerLineSymbolLayer()
        _set_interval(marker_line, interval_mm)
        marker_line.setRotateSymbols(True)
        # White fill with a coloured edge: an orange arrow on an orange band is
        # invisible, which is what the old ribbon did.
        sub = QgsMarkerSymbol.createSimple({
            "name": "filled_arrowhead", "size": str(size_mm),
            "color": "255,255,255,235",
            "outline_color": colour.name(), "outline_width": "0.3",
            "angle": "0",
        })
        marker_line.setSubSymbol(sub)
        return marker_line
    except Exception:
        return None


def spillway_symbol():
    """A spillway drawn as the weir it is: a crest bar at the built width, lying
    square across the feature, with a chevron showing which way water goes.

    The bar is in map metres, so a 0.5 m sill and a 6 m emergency weir stop
    drawing identically — the built width was previously carried only as label
    text and had no geometric expression at all. It is clamped at both ends: a
    narrow weir must stay findable when zoomed out, and a wide one must not
    swamp the feature it notches.

    The chevron stays in millimetres. Direction is identity, not dimension.
    """
    from qgis.core import (
        Qgis,
        QgsMapUnitScale,
        QgsMarkerLineSymbolLayer,
        QgsMarkerSymbol,
    )

    kind_colour = "CASE WHEN \"kind\" = 'inflow' THEN '#2e7d55' ELSE '#1273b5' END"

    casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 235))
    casing.setWidth(1.6)                       # millimetres
    casing.setPenCapStyle(Qt.PenCapStyle.FlatCap)

    bar = QgsSimpleLineSymbolLayer(QColor("#1273b5"))
    bar.setWidth(0.9)
    bar.setPenCapStyle(Qt.PenCapStyle.FlatCap)
    bar.setDataDefinedProperty(
        QgsSymbolLayer.PropertyStrokeColor, QgsProperty.fromExpression(kind_colour))
    try:
        scale = QgsMapUnitScale()
        scale.minSizeMMEnabled = True
        scale.minSizeMM = 0.9
        scale.maxSizeMMEnabled = True
        scale.maxSizeMM = 2.4
        bar.setWidthMapUnitScale(scale)
    except Exception:
        pass

    symbol = QgsLineSymbol([casing, bar])

    # Chevron at the middle of the bar, turned to face across it: out of the
    # feature for an outflow, into it for an inflow.
    try:
        chevron = QgsMarkerLineSymbolLayer()
        placement = getattr(
            getattr(Qgis, "MarkerLinePlacement", None), "CentralPoint", None)
        if placement is not None and hasattr(chevron, "setPlacements"):
            chevron.setPlacements(placement)
        chevron.setRotateSymbols(True)
        sub = QgsMarkerSymbol.createSimple({
            "name": "filled_arrowhead", "size": "3.2",
            "color": "#1273b5",
            "outline_color": "#ffffff", "outline_width": "0.4",
        })
        sub.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyFillColor, QgsProperty.fromExpression(kind_colour))
        sub.symbolLayer(0).setDataDefinedProperty(
            QgsSymbolLayer.PropertyAngle,
            QgsProperty.fromExpression(
                "CASE WHEN \"kind\" = 'inflow' THEN 180 ELSE 0 END"))
        chevron.setSubSymbol(sub)
        symbol.appendSymbolLayer(chevron)
    except Exception:
        pass
    return symbol


def burned_spillway_symbol():
    """The notch the burn cut, drawn as a band on the ground.

    Deliberately the same blue as :func:`spillway_symbol`'s outflow bar — this is that
    same spillway after the terrain was cut, not a different object. The outline is
    solid and the fill light, because the useful reading is *where the band is and how
    far it reaches through the bank*, and a heavy fill over a Verify-stage map already
    carrying ponding and overtopping would bury both.
    """
    from qgis.core import QgsFillSymbol

    from terrainflow_assessment.core.registry.map_palette import (
        SPILLWAY_BURNED_EDGE,
        SPILLWAY_BURNED_FILL,
    )

    return QgsFillSymbol.createSimple({
        "color": ",".join(str(c) for c in SPILLWAY_BURNED_FILL),
        "outline_color": ",".join(str(c) for c in SPILLWAY_BURNED_EDGE),
        "outline_width": "0.5",
    })


def connection_symbol():
    """An overflow link: an annotation, drawn like one.

    These were the heaviest ink on the map — up to 3 mm of saturated dark blue
    spanning the whole canvas, with Qt's dash length scaling off the pen width so
    a loaded link became a train of fat lozenges. They outweighed the structures
    they describe, and with no working arrowhead they read as linear *features*
    rather than as links between two things.

    So: fixed weight, receding colour, one clear arrowhead, and a white casing so
    it survives imagery. Volume no longer drives thickness — the Live Assessment
    panel states it in cubic metres, which is a better channel for a number than
    line width, and that encoding is exactly why these lines dominated.
    """
    from qgis.core import QgsSimpleLineSymbolLayer

    slate = QColor(58, 96, 140, 205)

    casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 200))
    casing.setWidth(1.7)                       # millimetres
    casing.setPenCapStyle(Qt.PenCapStyle.RoundCap)

    line = QgsSimpleLineSymbolLayer(slate)
    line.setWidth(0.9)
    line.setPenCapStyle(Qt.PenCapStyle.RoundCap)
    # Solid where the user drew the link, dashed where analysis inferred it, so
    # "I decided this" and "the ground decided this" are distinguishable. Both mm,
    # so the dash cannot scale itself apart when zooming.
    line.setDataDefinedProperty(
        QgsSymbolLayer.PropertyStrokeStyle,
        QgsProperty.fromExpression(
            'CASE WHEN "is_user_link" = 1 THEN \'solid\' ELSE \'dash\' END'),
    )

    symbol = QgsLineSymbol([casing, line])
    arrow = flow_arrow_layer(slate, size_mm=3.0, interval_mm=14.0)
    if arrow is not None:
        symbol.appendSymbolLayer(arrow)
    return symbol


def stress_symbol():
    """A stress point — where a feature overtops.

    A circle with a dark ring, not the amber triangle it used to be: at a glance
    that triangle was hard to tell from a spillway marker, and the two mean very
    different things. A stress point marks a station, not a dimension, so it
    stays fixed in millimetres.
    """
    from qgis.core import QgsMarkerSymbol

    return QgsMarkerSymbol.createSimple({
        "name": "circle", "size": "4.4",
        "color": "#e8a33d",
        "outline_color": "#7a4a05", "outline_width": "0.7",
    })


def flow_arrow_layer(colour, size_mm=2.6, interval_mm=8.0):
    """A repeating flow-direction arrow ribbon, in millimetres.

    The one arrow idiom in the plugin: earthwork signatures and overflow
    connections use the same mechanism, so direction always reads the same way.
    """
    return _signature_layer(SIG_ARROW, colour, interval_mm, size_mm)


def _set_interval(layer, interval_mm):
    """Interval placement, across the 3.22 → 3.40 enum rename."""
    from qgis.core import Qgis, QgsTemplatedLineSymbolLayerBase
    layer.setInterval(interval_mm)
    layer.setIntervalUnit(QgsUnitTypes.RenderMillimeters)
    placement = getattr(
        getattr(Qgis, "MarkerLinePlacement", None), "Interval", None
    )
    if placement is None:
        placement = QgsTemplatedLineSymbolLayerBase.Interval
    if hasattr(layer, "setPlacements"):
        layer.setPlacements(placement)
    else:
        layer.setPlacement(placement)
