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
    QgsProperty,
    QgsSimpleFillSymbolLayer,
    QgsSimpleLineSymbolLayer,
    QgsSymbolLayer,
    QgsUnitTypes,
)
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor

# Greyed-out stand-in for a disabled earthwork. Not in the registry because it is a
# UI state, not a feature type.
DISABLED_COLOUR = "#9aa4a2"


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
    try:
        rgba = QColor(colour.red(), colour.green(), colour.blue(), 45 if enabled else 22)
        fl = QgsSimpleFillSymbolLayer(rgba)
        fl.setStrokeColor(colour)
        fl.setStrokeWidth(0.7 if enabled else 0.4)
        if not enabled:
            fl.setStrokeStyle(Qt.PenStyle.DashLine)
        return QgsFillSymbol([fl])
    except Exception:
        return QgsFillSymbol.createSimple(
            {"style": "no", "outline_color": colour.name(), "outline_width": str(main_w)}
        )


def earthwork_line_symbol(key, colour, enabled, main_w):
    try:
        layers = []
        if enabled:
            # Casing (white underlay) scales with the real width so it always
            # wraps the coloured band; +0.6 m total ≈ 0.3 m of white each side.
            casing = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 235))
            casing.setWidth(main_w + 0.9)   # fallback only if width_m is NULL/0
            casing.setWidthUnit(QgsUnitTypes.RenderMetersInMapUnits)
            casing.setDataDefinedProperty(
                QgsSymbolLayer.PropertyStrokeWidth,
                QgsProperty.fromExpression('"width_m" + 0.6'),
            )
            casing.setPenCapStyle(Qt.PenCapStyle.RoundCap)
            casing.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
            layers.append(casing)

        # Main line: when enabled, its width is the earthwork's real ground
        # width (data-defined from the width_m attribute, in map metres) so the
        # band reads as the true dimension and scales with zoom. Disabled stays
        # a thin greyed mm dashed line — an "inactive" cue, not a true-scale band.
        main = QgsSimpleLineSymbolLayer(colour)
        main.setWidth(main_w if enabled else max(0.4, main_w * 0.6))
        main.setPenCapStyle(Qt.PenCapStyle.RoundCap)
        main.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if enabled:
            main.setWidthUnit(QgsUnitTypes.RenderMetersInMapUnits)
            main.setDataDefinedProperty(
                QgsSymbolLayer.PropertyStrokeWidth,
                QgsProperty.fromField("width_m"),
            )
        # Diversion keeps a solid base line — the flow-arrow ribbon below carries
        # the direction signal. Disabled + berm read as dashed.
        if not enabled or key == "berm":
            main.setPenStyle(Qt.PenStyle.DashLine)
        layers.append(main)

        if enabled and key == "diversion":
            # Flow-direction ribbon on top of the real-width base line — a
            # fixed-mm decorative accent (QgsArrowSymbolLayer follows the drawn
            # line, so direction is unambiguous). Skipped if unavailable.
            arrow = arrow_line_layer(colour, main_w)
            if arrow is not None:
                layers.append(arrow)

        if enabled and key == "dam":
            # Embankment/barrier look: short white dashes across the wall
            # (marker-free, so it always renders).
            hatch = QgsSimpleLineSymbolLayer(QColor(255, 255, 255, 235))
            hatch.setWidth(max(0.6, main_w * 0.55))
            hatch.setPenCapStyle(Qt.PenCapStyle.FlatCap)
            try:
                hatch.setUseCustomDashPattern(True)
                hatch.setCustomDashVector([1.4, 2.6])  # dash, gap (mm)
            except Exception:
                hatch.setPenStyle(Qt.PenStyle.DotLine)
            layers.append(hatch)

        if enabled:
            # Thin fixed-mm centreline so the feature stays visible when the
            # real-width band is sub-pixel at low zoom; it disappears into the
            # band once zoomed in.
            pin = QgsSimpleLineSymbolLayer(colour)
            pin.setWidth(0.5)   # millimetres — constant on screen
            pin.setPenCapStyle(Qt.PenCapStyle.RoundCap)
            pin.setPenJoinStyle(Qt.PenJoinStyle.RoundJoin)
            layers.append(pin)

        return QgsLineSymbol(layers)
    except Exception:
        return QgsLineSymbol.createSimple(
            {"color": colour.name(), "width": str(main_w),
             "capstyle": "round", "joinstyle": "round"}
        )


def arrow_line_layer(colour, main_w):
    """Flow-direction ribbon along a line — repeated arrowheads pointing in the
    drawn (downhill) direction, via QgsArrowSymbolLayer. Returns None on failure."""
    from qgis.core import QgsArrowSymbolLayer
    try:
        arrow = QgsArrowSymbolLayer()
        shaft = max(0.7, main_w * 0.55)
        for name, val in (
            ("setArrowWidth", shaft),
            ("setArrowStartWidth", shaft),
            ("setArrowHeadLength", 3.4),
            ("setArrowHeadThickness", 3.4),
        ):
            if hasattr(arrow, name):
                getattr(arrow, name)(val)
        if hasattr(arrow, "setIsRepeated"):
            arrow.setIsRepeated(True)   # multiple arrowheads down the line
        fill = QgsFillSymbol.createSimple(
            {"color": colour.name(), "outline_style": "no"}
        )
        arrow.setSubSymbol(fill)
        return arrow
    except Exception:
        return None
