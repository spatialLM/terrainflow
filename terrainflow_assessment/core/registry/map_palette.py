"""
map_palette.py — the colour ramps the raster overlays are drawn with.

Pure config, no Qt and no QGIS, so the panel's inline key, the layer renderers
and the report's map legend can all read the same stops. They could not before:
the surface-runoff ramp was declared in ``controllers/baseline.py`` and
hand-copied into ``panel.py`` with a comment admitting the two would drift, and
the water-captured and stream ramps each existed twice over — once for the
baseline layers and once for the simulation's. A key that describes a ramp the
map is no longer drawn with is worse than no key at all.

A stop is ``(fraction, (r, g, b, a), label)``. ``fraction`` multiplies the
band maximum, so the ramps stretch to whatever the run produced.

Two rules the stops here follow, because the report leans on both:

* **Darker means more water.** Every ramp runs light-to-dark with volume, so a
  reader who has learnt one map can read the next.
* **Alpha is for absence, not for magnitude.** Only the zero stop is
  transparent. Encoding volume as fading opacity — which the water-captured
  ramp used to do — makes a shallow pond indistinguishable from bare ground and
  leaves the whole layer looking like a smudge over the basemap.
"""

# --------------------------------------------------------------------------- surface runoff

#: Where the stops sit, per panel scale mode. Flow accumulation is heavily
#: skewed — a handful of channel cells carry orders of magnitude more than the
#: hillsides feeding them — so a linear stretch renders everything but the main
#: channels as near-nothing. ``log`` places the stops at decades of the maximum
#: so minor flow paths stay legible.
SURFACE_RUNOFF_SCALES = {
    "linear": (0.0, 0.15, 0.4, 0.7, 1.0),
    "quantile": (0.0, 0.02, 0.08, 0.25, 1.0),
    "log": (0.0, 1e-4, 1e-3, 1e-2, 1.0),
}
DEFAULT_SURFACE_RUNOFF_SCALE = "log"

#: Hue and weight per stop. Diffuse sheet flow covers nearly the whole site, so
#: at any real weight it reads as a wash over the map rather than an overlay on
#: it — and the point of the layer is to see what the flow is going *over*. The
#: gathering and channel stops keep their weight; those are the answer.
SURFACE_RUNOFF_COLOURS = (
    ((255, 255, 255, 0), "none"),          # nothing flows here
    ((226, 240, 250, 40), "diffuse"),      # off-white sheet flow, ~16%
    ((144, 196, 232, 140), "gathering"),
    ((48, 122, 190, 225), "concentrated"),
    ((8, 36, 110, 245), "channel"),        # concentrated channel
)

# --------------------------------------------------------------------------- streams

#: The thresholded network. Every non-zero cell is already a stream, so the ramp
#: jumps straight to a solid colour above zero rather than fading in.
#:
#: All three stops are dark and saturated. They used to open at a light blue,
#: which is legible over a DEM but disappears over pasture, bush shadow and bare
#: ground on an aerial — and a single-cell-wide raster line has no casing to
#: fall back on, so the colour is the whole of its contrast.
STREAMS = (
    (0.0, (0, 0, 0, 0), "none"),
    (0.001, (25, 80, 175, 255), "stream"),
    (0.4, (14, 52, 135, 255), "channel"),
    (1.0, (4, 18, 70, 255), "main"),
)

# --------------------------------------------------------------------------- water captured

#: Standing water, baseline and post-earthworks alike.
#:
#: Full opacity throughout. This ramp used to climb 0 → 160 → 220 in alpha, so
#: how much water a feature held was carried mostly by how transparent it was:
#: shallow ponding washed out into the basemap entirely and the layer read as a
#: stain rather than a measurement. Volume is carried by hue and value now —
#: light cyan where least, dark navy where most — and alpha does nothing except
#: keep dry ground out of the way.
WATER_CAPTURED = (
    (0.0, (168, 224, 240, 0), "dry"),
    (0.001, (168, 224, 240, 255), "shallow"),
    (0.35, (72, 160, 215, 255), "holding"),
    (0.7, (28, 96, 180, 255), "deep"),
    (1.0, (8, 36, 110, 255), "deepest"),
)


# --------------------------------------------------------------------------- event water line

#: The edge of the water *this event* delivers, drawn over the full pond.
#:
#: Amber, and the only warm line on the map. Every water colour here is blue, and the
#: event pond nests inside the full one drawn in the same ramp — so on hue alone the
#: two read as one pond with a darker middle. The line is what separates "the pond
#: could hold this" from "the storm fills it to here", and it has to survive being
#: half over navy water and half over pale ground, which no blue does.
EVENT_WATER_LINE = (255, 176, 46, 255)


# --------------------------------------------------------------------------- overtopping

#: The run of crest a pool goes over, drawn as a band on the crest itself.
#:
#: Red, and the only red on the water maps. Everything else here grades water by depth;
#: this is not a depth, it is the place a structure is being overtopped, and it has to
#: read as a hazard at a glance rather than as more water. Filled rather than outlined,
#: because the point it exists to make is that the spill happens along the *whole* band
#: and not at the single cell the flow routing draws.
OVERTOPPING_FILL = (192, 57, 43, 110)
OVERTOPPING_EDGE = (150, 30, 20, 255)


def surface_runoff_ramp(scale=DEFAULT_SURFACE_RUNOFF_SCALE):
    """Stops for the surface-runoff raster at the given panel scale mode."""
    fractions = SURFACE_RUNOFF_SCALES.get(scale,
                                          SURFACE_RUNOFF_SCALES["log"])
    return tuple((f, rgba, label)
                 for f, (rgba, label) in zip(fractions, SURFACE_RUNOFF_COLOURS))


def hex_of(rgba):
    """``(r, g, b, a)`` to ``#rrggbb``. Alpha is dropped, deliberately.

    A key shows the hue a stop stands for, not the blend it lands as over
    whatever happens to be underneath it — a swatch drawn at 16% alpha over a
    white panel is white.
    """
    r, g, b = rgba[0], rgba[1], rgba[2]
    return f"#{r:02X}{g:02X}{b:02X}"


def visible_stops(ramp):
    """The stops worth putting in a key — everything but the transparent one.

    Returns ``[(hex, label), ...]`` low to high.
    """
    return [(hex_of(rgba), label) for _f, rgba, label in ramp if rgba[3]]
