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
* **Alpha is for absence, not for magnitude.** Encoding volume as fading
  opacity — which the water-captured ramp used to do, and the surface-runoff
  ramp did until a field run-through — makes a shallow pond indistinguishable
  from bare ground and leaves the whole layer looking like a smudge over the
  basemap. Layer opacity was the compromise that replaced it, and it is gone
  too: it dilutes evenly rather than re-ordering the stops, but a 55% wash over
  an aerial still hands the basemap's own lightness range to a ramp whose whole
  low end is light.

  **Surface runoff takes one bounded exception**, and it is a fade-in rather
  than a magnitude encoding. Below :data:`SURFACE_RUNOFF_FADE_TOP_M3` the ramp
  is one colour, so alpha is the only channel left and nothing can be
  re-ordered against anything: the band exists to bring the map up out of
  nothing over the first couple of cubic metres instead of switching it on at a
  hard edge. Above that value every stop is opaque and the rule holds as
  written.
"""

# --------------------------------------------------------------------------- surface runoff

#: Where the stops sit, per panel scale mode. Flow accumulation is heavily
#: skewed — a handful of channel cells carry orders of magnitude more than the
#: hillsides feeding them — so a linear stretch renders everything but the main
#: channels as near-nothing. ``log`` places the stops at decades of the maximum
#: so minor flow paths stay legible.
#:
#: Fraction 0.0 is the *bottom of the colour ramp*, and for this one ramp that is
#: an absolute volume rather than the raster's zero — see
#: ``SURFACE_RUNOFF_FADE_TOP_M3``. The remaining fractions are unchanged from when
#: there were five stops; only the transparent one at the bottom has gone, because
#: the fade below now does that job over a range instead of at a point.
SURFACE_RUNOFF_SCALES = {
    "linear": (0.0, 0.4, 0.7, 1.0),
    "quantile": (0.0, 0.08, 0.25, 1.0),
    "log": (0.0, 1e-3, 1e-2, 1.0),
}
DEFAULT_SURFACE_RUNOFF_SCALE = "log"

#: The volume at which the ramp's lowest colour is fully opaque, in cubic metres.
#:
#: Everything about this ramp above this value is a fraction of whatever the run
#: produced. This one number is not, and deliberately: it is the bottom sliver of
#: a heavily skewed field, where a fraction of the maximum means nothing to a
#: reader and one cubic metre of water means something to everybody.
#:
#: Below it the ramp holds the "diffuse" colour and fades its alpha linearly to
#: nothing at 0 m³ — so 1 m³ draws at 50%, and a cell carrying only the rain that
#: landed on it (a quarter of a cubic metre on a 2 m grid in a 65 mm storm) is
#: barely on the map at all. That is the same end a hard floor at one cell's own
#: rainfall served, reached without an edge: a threshold drawn as an edge invites
#: the reading that water *stops* there, which is exactly the complaint this
#: layer has collected twice already.
SURFACE_RUNOFF_FADE_TOP_M3 = 2.0

#: Hue per stop. Every stop is fully opaque; weight is carried by value, light
#: cyan to dark blue, exactly as in ``WATER_CAPTURED`` and ``STREAMS``. The fade
#: below the bottom stop is not in here — it is one alpha over one colour, and
#: putting it in the tuple would put a second meaning in a column that carries
#: hue.
#:
#: These alphas used to climb 0 -> 40 -> 140 -> 225 -> 245, on the argument that
#: diffuse sheet flow covers nearly the whole site and at any real weight reads as
#: a wash over the map rather than an overlay on it. The problem was real; alpha
#: was the wrong instrument for it, for three reasons a field run-through made
#: plain:
#:
#: 1. Per-stop alpha does not dilute, it **re-orders**. The same value at 16%
#:    reads one colour over sunlit pasture and another over bush shadow, so two
#:    parts of one map cannot be compared. That is precisely the defect
#:    ``WATER_CAPTURED`` was converted to fix; it does not weaken because the
#:    field is throughflow rather than depth.
#: 2. **The key was lying.** ``hex_of`` drops alpha deliberately, so the panel
#:    key and the report legend drew "diffuse" solid while the map drew it at
#:    16%. Going opaque is what makes the legend and the map agree.
#: 3. The ramp is anchored on the band maximum and ``log`` opens at 1e-4 of it,
#:    so on a real site the "diffuse" stop lands somewhere around a few hundred
#:    upslope cells. Everything below that was drawn at under a fifth opacity —
#:    which is why runoff appeared to *stop* below a pond that retains its
#:    catchment, when in fact it was drawn and could not be seen.
#:
#: The low end is light cyan — the same cyan ``WATER_CAPTURED`` opens on, so the
#: shallowest water on one map and the faintest flow on the next are the same
#: colour. It was briefly white, which reads its distance from the background
#: quickly but has nowhere left to go underneath it: the fade below needs a
#: *colour* to fade, and white fading out over an aerial is indistinguishable
#: from white fading out over paper.
SURFACE_RUNOFF_COLOURS = (
    ((168, 224, 240, 255), "diffuse"),     # light cyan, sheet flow
    ((108, 179, 216, 255), "gathering"),
    ((48, 118, 172, 255), "concentrated"),
    ((16, 46, 82, 255), "channel"),        # concentrated channel
)


def surface_runoff_alpha(volume_m3, fade_top_m3=SURFACE_RUNOFF_FADE_TOP_M3):
    """Alpha (0–255) for a cell below the bottom of the colour ramp.

    Straight-line from nothing at 0 m³ to fully opaque at ``fade_top_m3``, which
    puts the midpoint at half — the shape the renderer produces from two stops
    and an interpolated shader, stated here so the panel key and anything that
    has to explain the map can read it off the same definition rather than
    reconstruct it.
    """
    if not fade_top_m3 or fade_top_m3 <= 0:
        return 255
    fraction = float(volume_m3) / float(fade_top_m3)
    return int(round(255 * min(max(fraction, 0.0), 1.0)))

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
#: Two reference states, two layers — "Overtopping (event)" and "Overtopping (full)",
#: named to match the pond pair beside them — and one colour between them. The band is
#: measured on the full pond, so it says "filled, this pool leaves over its own crest"
#: whatever the storm does; the event layer is the subset the modelled storm actually
#: reaches, and is drawn over the other.
#:
#: The distinction is carried by the layer name and by fill *pattern*, never by hue: it
#: is the same fault on the same crest, and a second colour would read as a second kind
#: of thing. The capacity bands are hatched so the two stay legible where they overlap,
#: and given more alpha because a diagonal hatch at 110 over an aerial is barely there —
#: the two have to carry comparable weight or the qualified one reads as the lesser
#: problem.
OVERTOPPING_FILL = (192, 57, 43, 110)           # solid — this event reaches the crest
OVERTOPPING_CAPACITY_FILL = (192, 57, 43, 205)  # hatched — only a full pond would
OVERTOPPING_EDGE = (150, 30, 20, 255)

#: The spillway notch **as it was actually cut into the terrain** — the Verify-stage
#: partner to the crest bar drawn on the Design stage.
#:
#: The same blue as the outflow spillway symbol, on purpose: it is the same structure,
#: seen after the burn rather than before it, and giving it its own hue would read as a
#: second kind of thing sitting on the same feature. It carries no magnitude — a notch
#: was cut here or it was not — so this is a flat fill and the alpha is for letting the
#: ground through, which is what alpha is for everywhere in this file.
SPILLWAY_BURNED_FILL = (18, 115, 181, 120)
SPILLWAY_BURNED_EDGE = (18, 115, 181, 255)


# --------------------------------------------------------------------------- terrain indices

# The terrain indices are the first rasters here that are not volumes of water, and two
# of them cannot be drawn under the two rules at the top of this file. Rather than bend
# those rules quietly, the exceptions are written out — the same way surface runoff's
# fade-in is — so the report legend and the panel key can be read against them.
#
# **A third rule, for signed quantities.** A diverging ramp is for a quantity whose
# **zero is a real boundary**, not merely its midpoint: convex ground and concave ground
# are different things, and the cell between them is neither. Such a ramp must be
# anchored **symmetrically** about zero, or the two tails re-order against each other
# and the eye reads an asymmetry the terrain does not have. Its centre is transparent,
# which keeps "alpha is for absence" intact — planar ground is the *absence* of
# curvature, not a small amount of it.
#
# **A fourth, for cyclic quantities.** Aspect has no magnitude at all: 359° and 1° are
# neighbours, so any light-to-dark ramp puts a hard seam across the map at north. It is
# drawn as **named classes**, not a ramp — which is also what the decision needs ("that
# face is north-facing"), and what the report legend can print as swatches with words.

#: Topographic wetness index. A water quantity, so the rules at the top apply as
#: written: light where dry, dark where wet, alpha only to keep dry ground out of the
#: way. Shares the ``WATER_CAPTURED`` hue family on purpose — the wetness map and the
#: ponding map answer the same question at different confidence, and a reader who has
#: learnt one should be able to read the other.
WETNESS_INDEX = (
    (0.0, (168, 224, 240, 0), "driest"),
    (0.25, (168, 224, 240, 255), "dry"),
    (0.55, (72, 160, 215, 255), "damp"),
    (0.8, (28, 96, 180, 255), "wet"),
    (1.0, (8, 36, 110, 255), "wettest"),
)

#: Stream power and sediment transport — the erosive pair.
#:
#: **Warm, and deliberately not blue.** This is not water arriving; it is soil leaving.
#: Drawing it in the water family would say "more water" about a quantity that means
#: "more erosion", and the two maps get read side by side. It is not the overtopping red
#: either — that is a hazard on one structure, and this is a gradient over the whole
#: site — so it runs through ochre to a dark red-brown rather than to a signal red.
EROSIVE_POWER = (
    (0.0, (250, 240, 200, 0), "none"),
    (0.2, (250, 232, 168, 255), "low"),
    (0.5, (221, 168, 83, 255), "moderate"),
    (0.78, (176, 106, 38, 255), "high"),
    (1.0, (105, 48, 18, 255), "severe"),
)

#: Plan and profile curvature — the file's first diverging ramp.
#:
#: Fractions run −1 → +1 and the renderer anchors them on ±p95 of |curvature|, not on
#: the raw min and max: one spike would otherwise pull a tail out and leave the other
#: flat. Concave (collecting — a hollow) is blue-green and convex (shedding — a nose) is
#: ochre; neither end is the water blue, because convex ground drawn in it would read as
#: standing water.
CURVATURE = (
    (-1.0, (13, 106, 110, 255), "concave"),
    (-0.35, (110, 178, 180, 200), "gathering"),
    (0.0, (245, 245, 240, 0), "planar"),
    (0.35, (214, 172, 106, 200), "shedding"),
    (1.0, (140, 88, 20, 255), "convex"),
)

#: Aspect, as eight named compass classes plus the flat sentinel.
#:
#: The value is each class's lower bound in compass degrees; ``-1`` is ground with no
#: aspect at all. Opposing faces are the contrast that matters on a NZ hill farm — a
#: north face dry and warm, a south face cool and damp — so those two are the strongest
#: pair here and the side slopes are deliberately muted between them.
#:
#: **These are degrees, not fractions, and the renderer has to be told so.** Every other
#: palette in this file is a fraction of the band maximum; this one is not. Passed
#: through the fractional path it multiplied each bound by the band max (~360) and laid
#: the stops at −360, 0, 16200 … 113400 against data that only ever spans [−1, 360] — so
#: every real value fell inside the first stop and the whole map drew as one flat wash,
#: measured at **0.32 % of the ramp occupied**. Its caller passes ``absolute=True``; see
#: ``qgis/controllers/_symbols.apply_raster_ramp``.
#:
#: **Aspect is circular, so the ramp closes.** North appears twice, at 0° and again at
#: 360°, because a face at 359° is north-facing and must not be drawn as the far end of a
#: linear scale. Without the closing stop, everything from 315° to 360° clamped flat onto
#: the NW colour.
ASPECT_CLASSES = (
    (-1.0, (235, 235, 232, 255), "flat"),
    (0.0, (214, 96, 45, 255), "N"),
    (45.0, (223, 152, 92, 255), "NE"),
    (90.0, (226, 205, 150, 255), "E"),
    (135.0, (176, 200, 160, 255), "SE"),
    (180.0, (58, 122, 156, 255), "S"),
    (225.0, (108, 158, 182, 255), "SW"),
    (270.0, (168, 196, 206, 255), "W"),
    (315.0, (208, 168, 120, 255), "NW"),
    (360.0, (214, 96, 45, 255), "N"),
)


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


# --------------------------------------------------------------------------- drawn areas

#: The three areas the user can draw or pick — outline only, high contrast, and
#: deliberately unlike the red a parcel layer usually comes in so each stands out
#: over whatever basemap is underneath.
#:
#: Here rather than in the controller that draws them because the report's map key
#: has to name the same colours. It named a different one: "Site boundary" printed
#: red while the map drew it bright blue, under a comment claiming a test asserted
#: the pair. There was no such test. There is now, and one definition to assert.
AREA_OUTLINES = {
    "boundary": (0, 162, 232, 255),      # bright blue
    "analysis": (255, 127, 14, 255),     # orange
    "earthworks": (148, 103, 189, 255),  # purple
}


def stop_colour(ramp, label):
    """The hex of the ramp stop with this label, for a key that draws a line.

    A watercourse is a line on the key and a graded raster on the map, so its
    swatch is one stop of the ramp rather than the whole gradient. Reading it out
    beats transcribing it: the transcription is what drifts.
    """
    for _f, rgba, name in ramp:
        if name == label:
            return hex_of(rgba)
    raise KeyError(f"{label!r} is not a stop of this ramp")
