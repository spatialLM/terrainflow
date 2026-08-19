"""
help_text.py — single source of truth for input/button tooltip copy.

The plugin is meant to be usable with limited prior knowledge, so most controls
carry an explanatory tooltip. Centralising the strings here (rather than inline in
panel.py) keeps the voice consistent, makes the copy reviewable in one place, and
lets dialogs/other tabs reuse the same wording.

Usage:
    from terrainflow_assessment.qgis import help_text as H
    widget.setToolTip(H.STREAM_THRESHOLD)

A few constants carry ``{name}`` fields and are applied with ``.format()``. They
are still copy — the sentence is fixed and only a figure varies — and they are
marked as templates where they are defined.

**What is deliberately not here.** ``verification_table._row_tooltip``, its
``_footer_text`` and ``spillway_table._row_tooltip`` build a paragraph per row
out of twenty-odd branches, choosing what to say from what the numbers did. The
sentence *is* the logic there, and lifting the wording out would leave the
widget naming constants nobody can read and this file holding sentences whose
conditions live elsewhere — less reviewable, not more, which is the opposite of
why this file exists.

Lives under qgis/ (the Qt/QGIS layer) as it is pure UI copy.
"""

# --------------------------------------------------------------------- Data Input
DRAW_BOUNDARY = "Draw the site boundary directly on the map."
DRAW_ANALYSIS_AREA = "Draw the analysis area directly on the map."
DRAW_EARTHWORKS_AREA = "Draw the earthworks area directly on the map."

ANALYSIS_AREA = (
    "Restrict Contour Analysis and Keypoint Analysis to this polygon.\n\n"
    "Leave blank to use the full DEM (or the site boundary if set)."
)
EARTHWORKS_AREA = (
    "Optional polygon defining where earthworks can be placed.\n\n"
    "When set, drawing tools will be constrained to this area and\n"
    "Optimal Swale Contours will only be generated within it.\n\n"
    "Useful for separating the project area from sensitive zones\n"
    "(wetlands, roads, existing structures) that must not be disturbed."
)

# --------------------------------------------------------------------- Baseline inputs
RAINFALL = (
    "Total rainfall depth for the storm event (mm).\n\n"
    "This is the cumulative rainfall over the full duration —\n"
    "the same figure reported in daily rainfall records or\n"
    "intensity-frequency-duration (IFD) tables as a daily total.\n\n"
    "Example: a 1-in-10-year, 24-hour storm in NZ hill country\n"
    "might be 80–120 mm total.\n\n"
    "The SCS model converts this total depth into runoff depth\n"
    "using the Curve Number and soil moisture condition."
)
DURATION = (
    "Storm duration in hours.\n\n"
    "Used to calculate peak flow rate at site exit points\n"
    "(volume ÷ duration = average flow rate).\n\n"
    "Set this to match the duration of your total rainfall figure —\n"
    "e.g. 24 hr if using a daily rainfall total from historical records."
)
SOIL_TYPE = (
    "Dominant soil texture of the site — with Ground condition below, it sets\n"
    "a starting Curve Number (runoff potential), and on its own it sets the\n"
    "infiltration rate used to size swales.\n\n"
    "Sandy soils soak up more water (less runoff); clay soils shed more\n"
    "(more runoff). Pick the texture that best matches your topsoil; you\n"
    "can override the Curve Number below if you know it."
)
GROUND_CONDITION = (
    "How well covered the ground is — the hydrologic condition of the pasture.\n\n"
    "  Good — over 75% ground cover, grazed lightly or not at all\n"
    "  Fair — 50-75% cover, grazed but not hard\n"
    "  Poor — under 50% cover, heavily grazed, or bare and compacted\n\n"
    "This matters more than the soil texture does. The same sandy loam is CN 49\n"
    "under good cover and CN 74 when it is hammered — a bigger swing than sand\n"
    "to clay. Thin cover means less to slow the water and a surface sealed by\n"
    "hooves, so far more of the rain runs off.\n\n"
    "Judge it in the paddock at the end of a dry spell, not from a green photo\n"
    "taken in spring. If in doubt on grazed country, Fair is the honest choice."
)
CURVE_NUMBER = (
    "SCS Curve Number — soil runoff potential.\n"
    "Higher = more runoff.\n\n"
    "Typical values (Normal moisture), by soil and ground condition:\n"
    "                   Good   Fair   Poor\n"
    "  Sand              39     49     68\n"
    "  Sandy loam        49     59     74\n"
    "  Loam              61     69     79\n"
    "  Clay loam         74     79     86\n"
    "  Clay              80     84     89\n\n"
    "Auto-filled from Soil Type and Ground condition above. Override if you\n"
    "know the site-specific CN (e.g. from land-use or measured data) — a typed\n"
    "value is kept, and saved and reloaded as typed."
)
MOISTURE = (
    "Antecedent moisture — how wet the ground already is when the storm\n"
    "hits (SCS AMC). It shifts the effective Curve Number.\n\n"
    "  dry    — soil dry, little recent rain (less runoff)\n"
    "  normal — average conditions (the usual choice)\n"
    "  wet    — soil already saturated from recent rain (most runoff)\n\n"
    "Use 'wet' for a conservative worst-case design."
)
STREAM_THRESHOLD = (
    "Minimum upstream catchment area for a flow path to be drawn as a channel\n"
    "(controls the blue stream layer only — site exit points use their own\n"
    "L/s threshold below).\n\n"
    "Lower = more channels shown.  Higher = major watercourses only.\n\n"
    "Channel types by contributing area:\n"
    "  Rills / erosion paths:   < 0.5 ha\n"
    "  Ephemeral / seasonal:    0.5 – 5 ha\n"
    "  Permanent stream:        5 – 20 ha\n"
    "  River:                   > 20 ha"
)
EXIT_FLOW = (
    "Minimum event-average flow for a site exit point to be shown, in litres\n"
    "per second (a physical, site-scale-independent criterion).\n\n"
    "An exit is where the drainage network crosses the site boundary. Lower\n"
    "this to reveal small crossings on a small holding; raise it to focus on\n"
    "the major outflows only.\n\n"
    "Guide:  0.1 L/s trickle  ·  0.5 L/s small channel  ·  5+ L/s main outflow."
)

# --------------------------------------------------------------------- Terrain Tools
QUERY_PONDING = (
    "Click on a blue zone in the 'Pond Capacity (full)' layer to select the\n"
    "entire connected pooling area and report its volume and surface area.\n\n"
    "That layer is what each hollow holds when brim-full, so this reports a\n"
    "capacity. 'Pond Capacity (event)' shows how far the storm actually fills it."
)
SLOPE_CLASS = (
    "Semi-transparent slope suitability overlay (calculated from DEM):\n"
    "  Green  (0–3°):    ideal — suitable for swales and basins\n"
    "  Yellow (3–8°):    moderate — suitable with care\n"
    "  Orange (8–13°):   challenging — consider companion berm\n"
    "  Orange (13–18°):  upper limit for swales\n"
    "  Red    (18–25°):  steep — berms or diversion drains recommended\n"
    "  Dark   (25–50°):  very steep — leave as vegetated slope\n"
    "  Black  (≥50°):    extreme — not workable"
)
SLOPE_VECTORS = (
    "Overlay a regular field of hachures — short strokes running downhill, drawn\n"
    "at a fixed ground spacing. Each stroke points the way water runs and thickens\n"
    "with the steepness of the ground, so you can read direction and gradient at a\n"
    "glance without the overlay covering the map underneath.\n\n"
    "Sampled from the DEM surface gradient. Requires: baseline analysis run."
)

# Rich-text body for the slope-class info dialog (F3).
SLOPE_CLASS_INFO = (
    "<b>Slope classes &amp; earthwork suitability</b>"
    "<p>The slope overlay grades the ground by steepness. What suits each class:</p>"
    "<table cellpadding='4'>"
    "<tr><td><b><font color='#2e7d32'>0–3°</font></b></td>"
    "<td>Ideal. Swales, basins and dams all work well; easy to build on-contour.</td></tr>"
    "<tr><td><b><font color='#c9a000'>3–8°</font></b></td>"
    "<td>Good. Swales are well suited; the usual range for keyline cultivation.</td></tr>"
    "<tr><td><b><font color='#e07b00'>8–13°</font></b></td>"
    "<td>Workable with care. Keep swales shorter, add a companion berm on the "
    "downhill side, and watch for concentrated flow.</td></tr>"
    "<tr><td><b><font color='#c62828'>13–18°</font></b></td>"
    "<td>Upper limit for most swales. Above this, water moves fast and cut/fill "
    "grows — prefer diversion drains or benched terraces.</td></tr>"
    "<tr><td><b><font color='#a11'>18–25°</font></b></td>"
    "<td>Too steep for swales. Use diversion drains, terraces, or keep as "
    "vegetated slope.</td></tr>"
    "<tr><td><b><font color='#7f1d1d'>25–50°</font></b></td>"
    "<td>Very steep. Earthworks are impractical and destabilising — best left "
    "under permanent vegetation.</td></tr>"
    "<tr><td><b><font color='#3A0000'>&ge; 50°</font></b></td>"
    "<td>Bluff or near-vertical face. Treat as unworkable ground.</td></tr>"
    "</table>"
    "<p><b>Rule of thumb:</b> a swale is usually recommended only up to about "
    "<b>15–18°</b>. The Analysis 'Max slope' filter defaults to 18° for this reason.</p>"
)
SIMPLE_CONTOUR_INTERVAL = "Contour interval (m)"
GENERATE_CONTOURS = (
    "Generate simple elevation contours from the DEM at the chosen interval.\n\n"
    "Useful for visualising terrain alongside the slope classification."
)

# --------------------------------------------------------------------- Contour analysis
CONTOUR_INTERVAL = (
    "Vertical spacing between candidate swale contours (m).\n\n"
    "Swales are level (on-contour), so this is the vertical drop between\n"
    "one swale and the next down the slope. Smaller = more, closely-spaced\n"
    "candidates (finer control, slower); larger = fewer, wider-spaced lines.\n\n"
    "0.5–1 m suits detailed design on a small holding; 2–5 m for a quick\n"
    "overview of a large block. This is a planning interval, not the built\n"
    "swale spacing — thin the results to the lines you'll actually build."
)
MAX_SLOPE = (
    "Drop candidate contours running across ground steeper than this (°).\n\n"
    "Swales get impractical and erosion-prone on steep slopes: earthworks\n"
    "are typically limited to gentle-to-moderate ground.\n\n"
    "  ≲ 8°   ideal for swales\n"
    "  8–15°  workable with care\n"
    "  15–18° upper limit for most swales (default 18°)\n"
    "  > 18°  consider keyline cultivation or terraces instead\n\n"
    "Lower this to keep only the easier ground."
)
USABLE_AREA = (
    "Optionally clip contour analysis to one of the polygon layers\n"
    "already selected in the Data section above.\n\n"
    "  None — analyse the full DEM extent\n"
    "  Analysis Area — use the Analysis Area polygon layer\n"
    "  Earthworks Area — use the Earthworks Area polygon layer"
)
MIN_CONTOUR_LENGTH = (
    "Exclude contours shorter than this length.\n\n"
    "Short enclosed contours (from shallow dips or small knolls)\n"
    "can rank highly because their accumulation is concentrated,\n"
    "but they are too short to place a meaningful swale.\n\n"
    "Set to 0 to include all contours."
)
ANALYSE_CONTOURS = (
    "Generate level contours across the site and rank them as candidate\n"
    "swale lines by how much runoff drains onto each.\n\n"
    "Uses the contour interval, max-slope filter, usable-area clip and\n"
    "min-length settings above. Runs like the baseline button: it fills as a\n"
    "progress bar, then shows ✓ when done.\n\n"
    "Requires: Baseline Analysis run first."
)
CONTOUR_LIST = (
    "Candidate swale contours, most inflow first.\n\n"
    "  Tick / untick a row to show or hide that contour on the map. An unticked\n"
    "  contour also drops out of 'Select Top Swales' and the inflow gradient, so\n"
    "  this is how you rule a line out rather than just hide it.\n\n"
    "  Click a row to highlight that contour on the map (and zoom-flash it when\n"
    "  you pick a single row).\n\n"
    "  It works the other way too: pick a contour on the canvas with QGIS's\n"
    "  Select Features tool and its row highlights here."
)
CONTOUR_LEGEND = (
    "Every 'how much water arrives here' view — the candidate contours, the\n"
    "inflow gradient and the peak-inflow overlay inside the swale segments —\n"
    "uses these four bands, in m³ over the event.\n\n"
    "Each band is both a colour (cyan → navy) and a line width (hairline →\n"
    "bold), with a white halo underneath. That is deliberate: over aerial\n"
    "imagery a colour step alone gets wiped out — a pale line disappears on\n"
    "sunlit grass, a dark one in tree shadow — whereas a thick line is obviously\n"
    "thicker than a thin one whatever is beneath it. Quiet stretches recede by\n"
    "being thin rather than by being faint.\n\n"
    "The overlay inside the swale segments carries the same bands in violet →\n"
    "deep purple: it sits on the segment's green core rather than on ground,\n"
    "and cyan → navy inside green is too small a step to read.\n\n"
    "Band edges come from the value, not from rank position. This replaced\n"
    "'top 5 / top 10 / rest', which only said where a contour sat in the queue:\n"
    "it drew a fixed five contours the same colour whether the fifth carried\n"
    "nearly as much as the first or a twentieth of it."
)
CONTOUR_LEGEND_EMPTY = "Inflow bands: run Analyse Contours"
TOP_N = (
    "How many of the top-ranked candidate swale contours to pull into a\n"
    "separate layer (ranked by peak inflow accumulation)."
)
TOP5_SWALES = (
    "Create a separate layer containing the top-N ranked candidate\n"
    "swale contours by peak inflow accumulation (set N on the left).\n\n"
    "Only ticked contours are considered — untick a row in the list above to\n"
    "keep it out of the running, and the top N is chosen from what remains.\n\n"
    "Requires: Analyse Contours run first."
)
INFLOW_BANDS = (
    "Band contours by the runoff that drains onto them along their length (m³\n"
    "over the event), on one shared scale keyed to the largest inflow in scope:\n"
    "a cyan hairline where little arrives, up to a bold navy line where the most\n"
    "does.\n\n"
    "Scope: the swales from 'Select Top Swales' once you have made that pick,\n"
    "otherwise every ticked contour. Narrowing the pick and pressing it again\n"
    "redraws the gradient over the shorter list.\n\n"
    "Because the scale is absolute and shared, a bold stretch on one contour\n"
    "carries genuinely more water than a thin stretch anywhere else — so you can\n"
    "compare swale-worthy spots across everything in scope.\n\n"
    "The candidate contour layer switches off while this is up: it draws the same\n"
    "lines, and two renderings of one line are harder to read than either.\n\n"
    "Requires: Analyse Contours + Baseline run first."
)
SEGMENT_GRADIENT = (
    "Draw the inflow bands *inside* each recommended segment, keeping the\n"
    "green/amber outline as a rim around them.\n\n"
    "The outline says a swale belongs on this stretch of contour; the bands say\n"
    "where along it the water actually concentrates — which is where the\n"
    "crossing, the deepest section and any overflow want to go.\n\n"
    "Same four bands and same m³ as the contour inflow gradient, drawn narrower\n"
    "so the green still shows. Follows the same scale setting.\n\n"
    "Coloured violet → deep purple rather than the gradient's cyan → navy: these\n"
    "bands are read against the segment's own green core instead of against the\n"
    "ground, and blue inside green is too small a step to pick out.\n\n"
    "Requires: Find Best Swale Segments run first."
)
INFLOW_SCALE = (
    "Where the four band edges fall. Applies to both the contour inflow gradient\n"
    "and the peak-inflow overlay inside the swale segments, so the two always\n"
    "answer 'how much' the same way.\n\n"
    "  Natural — edges land on the real gaps in the values (Jenks). The same\n"
    "  scheme the candidate contour list is banded on, so the list and the map\n"
    "  agree. Start here.\n\n"
    "  Log — even bands in log space; a band still means a fixed m³ range\n"
    "  (comparable across the map) but a long tail stays readable. Best when one\n"
    "  crossing dwarfs the rest.\n\n"
    "  Linear — even bands from 0 to the maximum m³. True absolute scale, but a\n"
    "  single extreme point flattens everything else into the bottom band.\n\n"
    "  Quantile — equal number of stretches per band; maximal separation, but a\n"
    "  band becomes a rank rather than an amount."
)
SEG_RANK_MODE = (
    "How swale segments are selected:\n\n"
    "  Min catchment above — only crossings draining at least the hectares set\n"
    "  below qualify (good for larger blocks with defined drainage lines).\n\n"
    "  Largest inflow — ignore the hectare threshold and surface the crossings\n"
    "  with the most inflow regardless of absolute size (use on a small holding\n"
    "  where no crossing reaches the catchment threshold)."
)
SEG_MAX_SLOPE = (
    "Optionally drop swale segments whose ground is steeper than this, sampled\n"
    "along the segment itself (independent of the whole-contour max-slope filter).\n\n"
    "Use it to prioritise segments on smoother, gentler gradient where a swale\n"
    "is easier to build and less erosion-prone."
)
SEGMENT_LIST = (
    "Recommended swale segments, best first. Each shows the inflow it must manage\n"
    "and whether a swale of your depth/width fits:\n"
    "  ✓ holds the design inflow within the available contour\n"
    "  ⚠ contour too short — the swale needs an overflow / companion feature.\n\n"
    "Click a row to zoom to it on the map."
)
CLEAR_ANALYSIS = (
    "Remove all analysis layers (contours, swale segments, keypoints, ridgelines,\n"
    "pond sites, keylines, inflow bands) and reset, so you can re-run cleanly\n"
    "without restarting the whole session."
)
MIN_CATCHMENT = (
    "Minimum contributing area above a contour crossing to qualify\n"
    "as a swale placement zone.\n\n"
    "Only flow paths draining at least this many hectares will produce\n"
    "a recommended segment.  Raise this to focus on major drainage lines;\n"
    "lower it to pick up smaller catchments too.\n\n"
    "Default 0.5 ha."
)
SWALE_DEPTH = (
    "Design depth of the swale cross-section (m).\n"
    "Drives the required swale length: the swale is sized to manage the\n"
    "inflow via trapezoidal trench storage plus infiltration over the storm\n"
    "(not to store the whole storm in the trench)."
)
SWALE_WIDTH = (
    "Design top width of the swale cross-section (m).\n"
    "With the depth and side slope this sets the trapezoidal capacity per\n"
    "metre used to size the required swale length."
)
SWALE_BOTTOM_WIDTH = (
    "Width of the swale FLOOR — the flat bottom the machine cuts (m).\n\n"
    "A swale is dug with a floor, not to a point. Together with the top width\n"
    "and the depth this fixes the trapezoid the segment sizing integrates, and\n"
    "the side slope below is what those three come out as.\n\n"
    "Wider floor, same top and depth: more section per metre, so a swale holds\n"
    "more and the recommended segments get shorter. It also flattens the\n"
    "batter, which is what you want in softer ground.\n\n"
    "It cannot exceed the top width. A drawn swale can override all three in\n"
    "its own properties dialog."
)
SWALE_SIDE_SLOPE = (
    "Wall batter, CALCULATED from the three dimensions above — you do not set\n"
    "it here.\n\n"
    "    side slope = (top width - bottom width) / (2 x depth)\n\n"
    "Shown as the angle from horizontal and as run:rise. 1:1 is a 45 degree\n"
    "wall; 1.5:1 or flatter is the usual recommendation in loam and softer,\n"
    "where a steeper face can slump. A vertical-sided trench reads 90 degrees.\n\n"
    "To flatten the batter, widen the top or the floor, or make it shallower.\n"
    "The earthwork properties dialog carries the soil-specific advisory for a\n"
    "swale you have actually drawn."
)
SWALE_SECTION_NOTE = (
    "The cross-sectional area these three dimensions give, and the volume it\n"
    "holds per metre of swale, brim-full.\n\n"
    "This is the number the segment sizing divides the inflow by, so it is the\n"
    "quickest check on a recommendation that looks too long: a swale that is\n"
    "sized as a V-drain rather than a floored trench holds a fraction as much\n"
    "and asks for kilometres."
)
FIND_SEGMENTS = (
    "Find swale placement zones on each candidate contour and size each\n"
    "segment to manage the incoming runoff over the design storm.\n\n"
    "Locates where drainage lines cross each contour (flow accumulation\n"
    "peaks), calculates inflow volume from the contributing catchment,\n"
    "then sets the swale length from trapezoidal trench storage plus\n"
    "infiltration (soil-dependent) over the storm duration.\n\n"
    "The segment is centred on the crossing point; segments flagged ⚠ are\n"
    "where the contour is too short to hold the design inflow.\n"
    "Results ranked globally by inflow volume (m³).\n\n"
    "Requires: Analyse Contours + Baseline Analysis run first."
)

# --------------------------------------------------------------------- Keypoint / keyline
KEYPOINT_COUNT = (
    "Number of keypoints to detect.\n"
    "Each keypoint is a valley inflection where slope eases from steep to gentle.\n"
    "Keypoints are spatially separated so they cover the full elevation range."
)
RUN_KEYPOINT = (
    "Analyse the DEM and flow accumulation to locate:\n\n"
    "  Keypoints — valley inflection points where slope transitions\n"
    "  from steep to gentle. This is where Yeomans' keyline begins.\n\n"
    "  Ridgelines — watershed divides that separate drainage basins.\n\n"
    "Requires: baseline analysis run."
)
RECOMMEND_PONDS = (
    "For each keypoint, find the optimal dam/pond location:\n"
    "the narrowest valley cross-section just downstream.\n\n"
    "Requires: keypoints found first."
)
KEYLINE_RUNS = (
    "Number of parallel cultivation guides to draw above and below the\n"
    "keyline (total lines = 2 × this + the keyline)."
)
KEYLINE_SPACING = (
    "Horizontal spacing between parallel cultivation guides\n"
    "(typically the plough/implement working width)."
)
KEYLINE_GRADE = (
    "Advisory guide grade as 1:N (e.g. 500 = 1:500). Recorded on each\n"
    "guide; the off-contour drift itself comes from the parallel geometry."
)
RUN_KEYLINE = (
    "Yeomans keyline design: find the keypoint on the primary valley, draw\n"
    "the on-contour keyline through it, and generate parallel cultivation\n"
    "guides above and below. The guides deliberately drift off-contour to\n"
    "move water from the wet valley toward the drier ridges.\n\n"
    "Requires: a DEM (baseline analysis recommended for best flow routing)."
)
DRAW_KEYLINE = (
    "Draw a keyline plough guide by hand. The status bar shows the ground\n"
    "slope under the cursor as a grade cue while you draw."
)
CONVERT_KEYLINE = (
    "Turn the current keyline (generated or drawn) into a swale in the\n"
    "Design stage. The swale stays reshape-locked to the keyline."
)
KEYPOINT_LIST = "Click a keypoint or pond site to zoom to it on the map."


# ---------------------------------------------------------------------------
# Design stage — water balance, routing and swale sizing
# ---------------------------------------------------------------------------

DIRECT_CATCHMENT = (
    "Runoff arriving here from this feature's own contributing area — the\n"
    "ground whose water reaches THIS feature before any other.\n\n"
    "Each cell of the site is traced downhill to whichever earthwork catches\n"
    "it first, so catchments never overlap: adding a swale upslope visibly\n"
    "shrinks the one below it, and no hillside is counted twice."
)
SWALE_HOLDS = (
    "What this swale can take over the event, at the length you drew:\n"
    "live storage in the trench (trapezoidal cross-section × length, brim-full)\n"
    "plus what soaks into its bed over the storm duration."
)
SWALE_DEFICIT = (
    "Whether the swale as drawn holds its event, and what would fix it.\n\n"
    "This leads instead of a 'recommended length' because a contour swale's\n"
    "inflow grows with its length — extend it and it intercepts proportionally\n"
    "more hillside, so the required length grows too and the shortfall never\n"
    "closes. Deepening, widening or routing the surplus downstream does close it."
)
SWALE_RECOMMENDED_LENGTH = (
    "For reference: the length that would hold this event at the current\n"
    "depth and width. Treat it as a rough guide — because inflow scales with\n"
    "length, simply extending the swale to this figure will not usually clear\n"
    "the shortfall on its own."
)
SITE_EXIT = (
    "Water leaving the site, measured in two parts:\n"
    "  • runoff that never reached any earthwork, and\n"
    "  • overflow that cascaded past the last feature in its chain.\n\n"
    "Both are measured from the routed flow paths, not inferred by subtraction."
)
AREA_OUTFLOW = (
    "Two figures per area, because they answer different questions.\n\n"
    "  • The total is every flow path crossing that boundary, measured cell by\n"
    "    cell. It does not change when you move the exit threshold.\n"
    "  • 'of which …' is the part running through the exits currently drawn on\n"
    "    the map, so it rises and falls with 'Show exits above (L/s)'.\n\n"
    "Water captured is what ponds naturally in hollows in the bare terrain,\n"
    "before any earthwork. On a large DEM the ponding raster is computed at a\n"
    "coarser resolution, so treat that figure as indicative."
)
TERMINAL_DEFICIT = (
    "Overflow escaping features that have nothing downstream of them.\n\n"
    "This is the extra storage, in m³, the design still needs — most usefully\n"
    "added upslope, where it can intercept the water before it concentrates."
)
CAPTURE_DENOMINATOR = (
    "Share of the whole design storm held on site.\n\n"
    "The denominator is the rainfall over the analysis area (or the site\n"
    "boundary, or the DEM) — not the catchment of a single outlet."
)
OVERFLOW_TARGET_AUTO = (
    "Where this feature's overflow goes once it is full.\n\n"
    "'Auto' follows the actual flow path downhill from the feature's low point\n"
    "until it meets another earthwork or leaves the site. Choosing a target\n"
    "instead forces the link — unless that would create a loop, which is\n"
    "refused with a warning."
)
STRESS_POINTS = (
    "Where a feature is predicted to overtop before it is nominally full.\n\n"
    "Inflow concentrates where drainage lines cross an alignment, so a swale\n"
    "with adequate total capacity can still spill part-way along. Each marker\n"
    "shows the distance along the feature and the surplus in m³."
)
THROUGHFLOW = (
    "Surface water that actually passes over each cell during the event, as a\n"
    "blue gradient — light cyan where flow is diffuse, dark blue where it\n"
    "concentrates.\n\n"
    "Finer-grained than the stream layer: it shows the whole surface, so you\n"
    "can see water gathering before it becomes a defined channel.\n\n"
    "The colour scale starts at 2 m³. Below that the layer fades out instead:\n"
    "2 m³ is solid, 1 m³ is half, 0 m³ is nothing. Every cell on the site\n"
    "carries at least the rain that fell on it, and drawing all of those solid\n"
    "painted the whole map with 'it rained here' — the fade keeps them on the\n"
    "map without letting them dominate it.\n\n"
    "Every hollow on the way has already taken what it can hold — a swale, a\n"
    "dam's pool or a natural depression — so a feature that captures its whole\n"
    "catchment shows nothing leaving it. Where a line does continue past one,\n"
    "that is the surplus after it filled, and it is the same water the panel\n"
    "counts as leaving.\n\n"
    "It is a volume for the whole event, not a rate: it says how much passes,\n"
    "never when. Storage is the measured pond on this terrain, so a hollow the\n"
    "DEM cannot resolve holds nothing here."
)
THROUGHFLOW_SCALE = (
    "How the colour range is stretched.\n"
    "  Log — compresses a skewed range so minor flow paths stay visible (default)\n"
    "  Linear — straight 0 → maximum\n"
    "  Quantile — equal cell count per colour band"
)
CATCHMENT_LAYER = (
    "Which earthwork catches the runoff from each part of the site, one colour\n"
    "per feature. Grey is ground that drains off site without meeting anything."
)
CONNECTIONS_LAYER = (
    "Arrows showing where each feature's overflow goes. Solid lines are links\n"
    "you set yourself; dashed lines were resolved by following the terrain.\n"
    "Thicker arrows carry more water."
)


SITE_SOIL = (
    "Soil texture across the site, used for infiltration rates when sizing\n"
    "earthworks. Individual features can override this in their properties\n"
    "dialog — useful where a basin sits in a clay hollow on otherwise loamy ground."
)
COUNT_INFILTRATION = (
    "Off (default): features must HOLD their water. Capture counts only what is\n"
    "impounded, and whatever soaks into the ground is reported separately as a\n"
    "buffer. This sizes conservatively — you build for the volume you can see.\n\n"
    "On: soakage counts toward capture, so features can be smaller.\n\n"
    "Off is the safer default because the infiltration model uses one steady-state\n"
    "rate per soil texture for the whole event, with no saturation limit: it will\n"
    "keep absorbing at that rate as long as water is present. Real ground varies\n"
    "field to field, wets up during a storm, and stops accepting water once the\n"
    "profile saturates."
)
FEATURE_SOIL = (
    "Soil under this feature. 'Site default' follows the Design tab setting.\n\n"
    "Override it where the ground genuinely differs — infiltration is the term\n"
    "most sensitive to soil, so a basin in heavy clay sized on loam rates will be\n"
    "undersized."
)


RUNOFF_METHOD = (
    "How runoff is estimated from the design storm. The depth this produces is what\n"
    "the whole assessment then works from — the analysis rasters (streams,\n"
    "surface runoff, exit points) and earthwork sizing alike.\n\n"
    "RUNOFF COEFFICIENT (default). The rational method: runoff = rainfall x C,\n"
    "with C taken from the Surface row below. Standard practice in water-harvesting\n"
    "design and the basis of Brad Lancaster's sizing calculations. Its merit is that\n"
    "the assumption is a single visible number you can question, or replace with one\n"
    "measured on this site.\n\n"
    "TOTAL RAINFALL. Every millimetre falling on a catchment is assumed to reach the\n"
    "feature (C = 1.0). This is not a prediction but an upper bound — no method here\n"
    "returns more, and it sits above Lancaster's coefficient for a metal roof. Use it\n"
    "when you want the runoff estimate ruled out as the thing that was wrong.\n\n"
    "SCS CURVE NUMBER. The USDA NRCS TR-55 method, deriving runoff from soil group,\n"
    "cover and antecedent moisture through the CN and Moisture Condition rows above.\n"
    "The most physically detailed of the three, and the one most sensitive to its\n"
    "inputs — it is worth using once the curve number reflects surveyed ground rather\n"
    "than the default.\n\n"
    "The three disagree substantially. On a 120 mm storm over CN 61 pasture they give\n"
    "roughly 60 mm (C = 0.50), 120 mm and 31 mm. That spread is real rather than a\n"
    "defect of any one method: runoff depends on antecedent moisture and rainfall\n"
    "intensity, which a design storm does not pin down. The same CN 61 ground already\n"
    "wet yields 64 mm rather than 31 mm. Choosing a method is therefore a judgement\n"
    "about margin, and the consequences are asymmetric — a feature sized low breaches\n"
    "and passes the problem downstream, while one sized high costs excavation.\n\n"
    "Changing this invalidates a completed baseline — the Baseline and Analysis stages\n"
    "are marked stale so the rasters are rebuilt against the depth you chose."
)


RUNOFF_SURFACE = (
    "Surface coefficients from Brad Lancaster, 'Rainwater Harvesting for Drylands\n"
    "and Beyond' — the rational method used throughout water-harvesting practice:\n\n"
    "    runoff = catchment area x rainfall x coefficient\n\n"
    "Each entry shows the typical value with Lancaster's published range beside it.\n"
    "Those ranges span 3-7x for a single surface; that spread is the honest state of\n"
    "the art rather than a defect of the table, and it is why the default sits\n"
    "toward the wet end.\n\n"
    "'Pasture / grass, wet ground' is a design default rather than one of his\n"
    "figures: above his grass coefficient because the storm that breaks an earthwork\n"
    "is usually the second one, arriving on ground already saturated.\n\n"
    "Pick 'Custom' to type a measured or locally-derived value."
)
AREA_SUBTOTALS = (
    "The same water balance, broken down by sub-catchment.\n\n"
    "A single site figure hides where the work is needed: 19% held overall might be\n"
    "80% on one catchment and nothing on the next, and those call for very different\n"
    "responses. Worst first, so the row worth acting on leads.\n\n"
    "Each row reports runoff GENERATED on that ground and where it ended up. A cell\n"
    "in one catchment may drain to a feature in another, so 'held' means it reached\n"
    "some earthwork, not necessarily one inside these bounds — which keeps the rows a\n"
    "true partition that sums to the site total.\n\n"
    "This is the same cell-exact labelling the headline uses, not a second\n"
    "calculation. Two disagreeing 'water leaving' figures in one panel would be worse\n"
    "than one."
)
INFLOW_PROFILE = (
    "Where along this feature its catchment actually arrives — one bar per reach.\n\n"
    "Runoff does not spread evenly. It concentrates where drainage lines cross the\n"
    "alignment, so a feature whose TOTAL capacity is adequate can still go over the\n"
    "side at one point. A flat profile means the total-vs-total verdict above is\n"
    "safe; a spike means it is not.\n\n"
    "The amber tick marks the first station where cumulative inflow outruns the\n"
    "storage in the reach traversed so far. It also appears on the map, in the\n"
    "'Stress points' layer, next to the problem rather than only in this dialog.\n\n"
    "Each catchment cell is attributed to its nearest point along the centreline.\n"
    "That is exact for a contour swale, where runoff runs perpendicular to the line,\n"
    "and looser the further the alignment departs from the contour."
)
EXIT_TABLE = (
    "Every place water crosses your boundary, largest first — the same crossings the\n"
    "red dots mark on the map, in one list you can read without clicking each one.\n\n"
    "AVERAGE RATE      the event volume spread over the storm duration, in L/s\n"
    "VOLUME OVER EVENT the water that leaves through this crossing, in m3\n\n"
    "The two are one measurement: the rate IS the volume divided by the duration. Both\n"
    "are shown because a culvert is sized by a rate and a dam is sized by a volume.\n\n"
    "This is NOT the peak flow a structure is sized against. That is a larger number,\n"
    "calculated at the time of concentration, and it is on the Spillways table.\n\n"
    "The rows will not add up to the 'Water leaving' total above them. Each crossing\n"
    "is reported at its busiest cell, and anything under your L/s threshold is left\n"
    "out entirely. The total above is measured cell by cell around the whole boundary\n"
    "and is the figure to quote."
)
DRAWN_VOLUME = (
    "The drawn cross-section over the drawn length, brim-full — the size of the hole,\n"
    "with nothing taken off it.\n\n"
    "It used to be this figure LESS a blanket 20% freeboard allowance. That allowance\n"
    "is gone: freeboard on a real feature is the height its spillway leaves between\n"
    "the design nappe and the crest, which is set per feature on the Spillways table\n"
    "from a peak flow a site-wide fraction knew nothing about.\n\n"
    "This same number is the GEOMETRIC figure in the Verify stage's table, and what\n"
    "the feature actually ponds once the design is burned into the terrain is MEASURED\n"
    "there beside it. On a keyed swale the measured pond is often much the larger:\n"
    "a companion berm holds water above natural ground, and no cross-section predicts\n"
    "that because it depends on the hillside."
)
VERIFICATION_TABLE = (
    "What each feature was drawn to hold, against what it holds on the actual\n"
    "ground. Three numbers because a single figure hid two unrelated gaps.\n\n"
    "GEOMETRIC   the drawn shape exactly. For a swale with a companion berm this is\n"
    "            the trench PLUS the berm's own section\n"
    "AT GRID     what this feature impounds on this hillside — its own cut and its own\n"
    "            bank, flooded on the DEM in isolation from every other feature\n"
    "MEASURED    the pond it ends up with once the whole design is built\n\n"
    "The first is CALCULATED, the last two are MEASURED. That is the important\n"
    "division, and it is why AT GRID is usually the larger.\n\n"
    "There was a fourth column, DESIGN — the drawn shape less a blanket 20% freeboard.\n"
    "Both it and the allowance behind it are gone. Freeboard on a real feature is set\n"
    "by its spillway, sized on the Spillways table from a peak flow that fraction knew\n"
    "nothing about, so a rule of thumb sat first in a row of figures meant to be\n"
    "compared and invited a comparison it could not support. GEOMETRIC is now the\n"
    "whole drawn section, brim-full.\n\n"
    "AT GRID is bigger than GEOMETRIC on most swales, and this is real, not an error.\n"
    "A companion berm keyed into its banks holds water ABOVE natural ground, standing\n"
    "deeper than the trench and reaching further up the slope than the trench does.\n"
    "No cross-section can predict that — it depends on the hillside — so the drawn\n"
    "figure understates a keyed swale, often by half. That is also why the live score\n"
    "sizes against AT GRID: against the drawn figure a keyed swale reads 'full' while\n"
    "most of its pond is still empty, and the design gets oversized.\n\n"
    "Delta is MEASURED against AT GRID and should sit near zero. Both are floods now,\n"
    "so a non-zero delta means the finished site ponds differently from the feature on\n"
    "its own — interaction with a neighbour, and nothing else.\n\n"
    "Where a feature impounds well above natural ground it is flagged as a retaining\n"
    "structure. It is holding water like a dam; give it a designed spillway and build\n"
    "the bank properly, rather than letting it overtop at its own lowest point.\n\n"
    "A row marked with a dagger is one where the grid could not hold the section you\n"
    "drew — a feature narrower than about three cells has no cell more than half a\n"
    "cell from its own edge, so it cannot reach full depth. That is a grid limit, and\n"
    "GEOMETRIC is the capacity to read on such a row.\n\n"
    "A feature narrower than one cell shows no measured figure. It is verified for\n"
    "placement and routing only — a storage volume read off the grid would measure\n"
    "the grid rather than the design.\n\n"
    "Two features whose pools run together are blank in the same two columns and get\n"
    "one line beneath the table instead. A basin and the dam on its lip hold a single\n"
    "sheet of water; dividing it between them would report the division, not the burn."
)
RAINFALL_DATA = (
    "Depth-duration-frequency statistics for this site — how much rain falls, over\n"
    "what duration, at what return period.\n\n"
    "This cannot be derived from terrain. It is regional rainfall climatology, and in\n"
    "New Zealand it is free from NIWA's High Intensity Rainfall Design System\n"
    "(HIRDS v4) at hirds.niwa.co.nz: enter a location, read off the table.\n\n"
    "With it the plugin can compute how long your catchment takes to respond and read\n"
    "the intensity for a storm of exactly that length, which is what the rational\n"
    "method actually asks for. Without it you supply a peak intensity by hand.\n\n"
    "Entering the table also supplies the 2-year 24-hour depth that TR-55's sheet-flow\n"
    "timing needs, so it is not a second lookup."
)
RAINFALL_DATA_TABLE = (
    "Rainfall depth in millimetres: rows are storm durations, columns are return\n"
    "periods (the '50 yr' column is the storm expected once in 50 years on average).\n\n"
    "Partial data is fine. Only the return periods you fill in become available, which\n"
    "is honest — a curve through one point is not a curve.\n\n"
    "Between entered durations the depth is interpolated on log-log axes, which is how\n"
    "depth-duration curves actually behave. Outside the entered range the value is\n"
    "held at the nearest end rather than extrapolated: data fitted from 10 minutes\n"
    "says nothing dependable about 2, and inventing it would look like a lookup."
)
TIME_OF_CONCENTRATION = (
    "How long runoff takes to travel from the most hydraulically distant point of this\n"
    "feature's catchment to the feature itself.\n\n"
    "It matters because peak flow happens when the WHOLE catchment is contributing at\n"
    "once, which is at t = Tc. A shorter storm is fiercer but only engages part of the\n"
    "catchment; a longer one engages all of it but is weaker. So Tc is the storm\n"
    "duration to read the IDF curve at.\n\n"
    "Computed by TR-55's segmental method over the longest flow path measured from the\n"
    "DEM: sheet flow for the first 30 m, then shallow concentrated flow, then channel\n"
    "flow. A few hectares of pasture typically gives 10-20 minutes — which is why\n"
    "reading intensity off a 24-hour storm is meaningless.\n\n"
    "Below 6 minutes the method is outside its stated calibration and the value is\n"
    "held at that floor."
)
CHANNEL_LENGTH = (
    "How much of the flow path runs in a defined watercourse rather than across open\n"
    "ground.\n\n"
    "Channel flow is roughly twice the speed of shallow concentrated flow, so it\n"
    "shortens the time of concentration and RAISES the design intensity. Leaving it at\n"
    "zero is therefore not the safe default: it overstates Tc, understates intensity,\n"
    "and undersizes the overflow.\n\n"
    "Set it if there is a gully, drain or watercourse along the path. For a small\n"
    "paddock catchment with no defined channel, zero is correct."
)
PEAK_INTENSITY = (
    "Peak rainfall intensity used to size overflow structures. Storage and overflow\n"
    "are sized on different questions: how much rain falls, and how fast.\n\n"
    "This cannot be derived from the storm depth and duration above. Dividing one by\n"
    "the other gives the event AVERAGE — 120 mm over 24 h averages 5 mm/hr, which is\n"
    "a daily mean, not anything a spillway will ever see. The same 120 mm falling in\n"
    "2 hours is identical storage and twelve times the peak flow.\n\n"
    "Peak flow follows the rational method, Q = C x i x A, where the fraction C comes\n"
    "from whichever runoff method is set above and A is the catchment measured for\n"
    "each feature. Only the intensity has to be supplied.\n\n"
    "Use Compare to see what each intensity implies before choosing. A local IDF or\n"
    "HIRDS figure for your site beats any default here."
)
DESIGN_INTENSITY_TABLE = (
    "Peak flow and required spillway width for a range of design intensities,\n"
    "costed against a real catchment on this site.\n\n"
    "The lower rows show what the storm depth implies if it fell over a shorter\n"
    "window. Those are still averages: a storm average is only a valid peak when the\n"
    "storm duration is close to how fast the catchment responds — minutes to about an\n"
    "hour for a few hectares — so the 24-hour row is there to be rejected, not chosen.\n\n"
    "The other runoff methods are shown for comparison but cannot be selected there.\n"
    "Sizing an overflow on a method the rest of the assessment is not using would put\n"
    "the spillway and the storage it protects on different assumptions. Change the\n"
    "method on the Baseline tab if you want it, and re-run."
)
SPILLWAY_DESIGN_FLOW_PEAK = (
    "Peak flow this structure has to pass: Q = C x i x A.\n\n"
    "  C  the instantaneous runoff fraction for the chosen method\n"
    "  i  the peak design intensity from the Baseline tab\n"
    "  A  this feature's measured direct catchment, plus any overflow routed in\n"
    "     from features upslope — once a feature is full, what arrives leaves\n\n"
    "On the SCS-CN method, C is the MARGINAL fraction dQ/dP, not the event average.\n"
    "The curve-number model sheds progressively more as a storm proceeds, once the\n"
    "initial abstraction is paid off, so its event average badly understates the rate\n"
    "at the peak: CN 61 under a 120 mm storm averages 0.26 but is 0.58 at the margin.\n\n"
    "It is evaluated at the full storm depth, which assumes the peak burst arrives at\n"
    "the end of the event on the wettest ground. That is the conservative corner\n"
    "rather than a neutral one — the same storm gives 0.27 half way through.\n\n"
    "Note this pairs SCS-CN runoff with a rational-method peak. TR-55's own peak\n"
    "discharge method uses unit hydrographs; this is an engineering approximation\n"
    "taken on the safe side, not a textbook procedure."
)
SPILLWAY_BUILT_WIDTH = (
    "The width you intend to build, against the minimum the design flow needs.\n\n"
    "Shown at the width the terrain model can actually cut: it works in whole DEM\n"
    "cells, so a weir the flow sizes at 1.4 m is burned 2.0 m wide on a 1 m model.\n"
    "That changes nothing the feature holds - water still leaves at the crest,\n"
    "which is a level and not a width - it only runs the overflow shallower.\n\n"
    "'auto' decides which half of the weir equation you are holding fixed. Ticked, the\n"
    "width follows the design head and keeps following it as the design changes —\n"
    "including when a feature drawn upslope starts routing its overflow through this\n"
    "one. Untick it to commit to a width, and the head becomes the consequence: the\n"
    "row above switches to the depth the water will actually run at, and the freeboard\n"
    "is checked against that rather than against what you asked for.\n\n"
    "Standard practice adds 20-30% to the computed minimum and armours the outlet.\n"
    "An under-armoured spillway scours, deepens, and drains the feature it exists to\n"
    "protect."
)
SPILLWAY_HARVESTING_C = (
    "This runoff coefficient was calibrated for water harvesting, not for sizing an\n"
    "overflow.\n\n"
    "Lancaster's figures describe how much can be captured from ordinary rain. A\n"
    "spillway answers a different question: what happens in the extreme event, on\n"
    "ground already saturated by earlier rain. His 'Grass / lawn' 0.18 sizes a 2.8 ha\n"
    "spillway at 0.20 m, where the same ground wet on SCS-CN needs 0.94 m — nearly\n"
    "five times wider.\n\n"
    "The storage sizing can reasonably use a harvesting coefficient. The overflow\n"
    "should not."
)
SPILLWAY_GROUP = (
    "Where this feature is designed to overflow when it fills.\n\n"
    "It will overflow regardless — the question is only whether that happens at a\n"
    "place you chose and armoured, or at whichever point of the rim happens to be\n"
    "lowest. An uncontrolled overflow concentrates on unprotected ground and is the\n"
    "usual way an earthwork fails: the breach starts at the outflow, not the wall.\n\n"
    "This records the structure — its crest, its width and where it sits. It does not\n"
    "set the order the system fills in: which feature spills into which comes from the\n"
    "overflow link you draw, or from the downhill path when you draw none.\n\n"
    "Leave the group unticked to record that no spillway is designed yet. The Spillways\n"
    "list on the Design stage still sizes one for you, so you can see what it would need\n"
    "before committing to it."
)
SPILLWAY_CREST = (
    "Absolute elevation of the spillway crest — the level water reaches before it\n"
    "starts leaving. This is the authoritative value: the other two crest rows are\n"
    "views of this one against a datum, and it is what would be cut into the terrain\n"
    "model.\n\n"
    "Note the reported storage is still measured to the full depth of the feature, not\n"
    "down to this line, so a crest set well below the spill level holds less than the\n"
    "capacity figure claims — which is what the storage row underneath says in cubic\n"
    "metres. What the crest does govern here is the freeboard budget: the gap between\n"
    "it and the level the feature is held to has to accommodate the design head and\n"
    "still leave a margin.\n\n"
    "Bound to both 'Height above floor' and 'Below spill level': changing any one\n"
    "updates the other two. The value is held within what the feature can offer — no\n"
    "higher than the spill level minus head minus freeboard, and no lower than the\n"
    "floor."
)
SPILLWAY_HEIGHT_ABOVE_FLOOR = (
    "How far the crest stands above the floor of the feature. This is the row to set\n"
    "out from: it is the one measurement a builder can take with a staff standing in\n"
    "the trench, and it is the one that does not depend on ground somewhere else.\n\n"
    "It is offered first because the other datum is a single lowest point over the\n"
    "whole footprint, and on a contour swale that point is usually at one of the ends.\n"
    "A notch cut half way along a falling swale and measured down from an end is\n"
    "measured from ground a hundred metres away, at an elevation it does not share.\n\n"
    "The floor here is the design invert — the level the burn cuts to. The floor the\n"
    "terrain model actually reaches can be higher where the footprint is too narrow to\n"
    "batter down to full depth, so on a narrow swale treat this as the intent rather\n"
    "than the built level.\n\n"
    "Not offered on a dam: there the feature has no cut floor, only the ground the wall\n"
    "stands on, and the height above that is the wall height rather than a sill level."
)
SPILLWAY_DROP = (
    "How far the crest sits below the level this feature is held to. The same crest as\n"
    "the rows above, measured down from the top rather than up from the bottom.\n\n"
    "The datum is the containment level, not necessarily natural ground: on a swale\n"
    "with a companion berm it is the berm crest, so this is the drop below what was\n"
    "built. Where the two differ the dialog shows both.\n\n"
    "Needs a DEM to be meaningful, so it is disabled until one is loaded."
)
SPILLWAY_CONTAINMENT = (
    "The level this feature's water is actually held to — where it would spill if you\n"
    "built no spillway at all. Everything on this page is measured against it.\n\n"
    "It is not always undisturbed ground, and the row says which it is:\n\n"
    "  measured off the last analysis — the level the built feature was found to pond\n"
    "    to when the terrain model was flooded. The best answer, and the one to prefer\n"
    "    once an earthworks re-analysis has run.\n"
    "  the companion berm as built — the crest the spoil bank reached in the last\n"
    "    burn. A measurement of the structure, not an estimate of one.\n"
    "  the wall crest you specified — a dam. The wall is the containment there; the\n"
    "    valley floor it is holding back is not.\n"
    "  the lowest natural ground round the footprint — nothing has been built or\n"
    "    measured yet, so this is the bare hillside.\n\n"
    "This used to be the ring minimum in every case, including on features whose whole\n"
    "purpose is to hold water above it. A swale bermed and keyed into its banks ponded\n"
    "to 69.60 m against a ring minimum of 68.88 m — 1,095 m³ against 439 m³ — and its\n"
    "crest was clamped to the lower figure, so the storage was given away before the\n"
    "user saw it."
)
SPILLWAY_LIP = (
    "The lowest natural ground on the ring of cells just outside the footprint —\n"
    "where water would escape if you dug the hole and built nothing else.\n\n"
    "Shown only when it differs from the level above, and then the gap between them is\n"
    "what the built structure is holding up. A crest above this line is not a fault: it\n"
    "is a sill in made ground, which is what a bermed swale is for. It is worth knowing\n"
    "because that ground has to be built and stay built.\n\n"
    "Once a spillway is sited, this is taken locally — from the ring within about a\n"
    "sill width of where you placed it — rather than from the whole footprint. On a\n"
    "swale that falls along its run those are different elevations, and the one under\n"
    "your sill is the one that describes it."
)
SPILLWAY_GIVE_UP = (
    "What this sill costs, in the units the decision is made in.\n\n"
    "The first figure is what the feature holds with the crest where it is now; the\n"
    "second is what it would hold with no spillway, filled to the level in the row at\n"
    "the top. The difference is the storage you are giving up in exchange for choosing\n"
    "where the water leaves.\n\n"
    "Both come from flooding this feature alone on the terrain model and reading the\n"
    "volume off at each level, so they are measured rather than derived from the drawn\n"
    "cross-section. That means there is nothing to read until an analysis has measured\n"
    "the feature — which the row says, rather than printing a zero.\n\n"
    "Giving something up is the point of a spillway, so this is not a warning. It is\n"
    "highlighted past about 40% because that is where the sill, rather than the\n"
    "excavation, has become the thing setting the size of the feature."
)
SPILLWAY_HEAD = (
    "How deep you intend the water to run over the crest at peak flow — a target, not\n"
    "a measurement.\n\n"
    "H is a depth of FLOW, not a dimension of the structure. Nothing about the sill\n"
    "is H metres anything; it is how far the water surface stands above the crest\n"
    "while the overflow is running, and it is zero once the storm is over.\n\n"
    "The weir equation Q = C x L x H^1.5 has one spare degree of freedom, so one of\n"
    "head and width has to be chosen and the other follows. While the width is on\n"
    "'auto' this is the one you choose: a lower head needs a wider sill to pass the\n"
    "same flow. Untick 'auto' and the roles swap — the width becomes yours and the\n"
    "head becomes whatever the flow makes it.\n\n"
    "Head is the sensible one to choose because everything else bears on it. It spends\n"
    "freeboard directly, since the water surface sits this far above the crest while\n"
    "the spillway works, so raising it lowers the highest crest the feature can accept.\n"
    "It also sets the exit velocity on its own — a broad crest passes critical depth,\n"
    "so how fast the water leaves depends on the head and not at all on the width,\n"
    "which is what an erosion-protection detail is written against.\n\n"
    "Typical values depend on what you are building. An earth dam or basin overflows\n"
    "at 0.20-0.50 m. A swale spills over a low sill in its own bank and wants much\n"
    "less, nearer 0.10-0.30 m — a deep spillway on a shallow swale spends the storage\n"
    "the swale exists to provide."
)
SPILLWAY_ACTUAL_HEAD = (
    "How deep the water will actually run over the width you have committed to.\n\n"
    "Shown instead of the target once 'auto' is off, because then the width is the\n"
    "input and the head is the consequence: H = (Q / (C x L))^(2/3).\n\n"
    "It is the honest way to read an undersized sill. 'Two metres short' is hard to\n"
    "act on; 'the water stands 18 cm deeper than you planned, leaving 2 cm of\n"
    "freeboard' is the same fact in the units that decide whether the structure\n"
    "survives.\n\n"
    "Head grows as the two-thirds power of flow, so doubling the flow raises it by\n"
    "about 1.6x while an auto width would have grown by 2x. Committing to a width is\n"
    "often the cheaper decision — provided the freeboard is there to absorb it."
)
SPILLWAY_FREEBOARD = (
    "Clear height kept between the design water surface and the lowest containing\n"
    "ground.\n\n"
    "This is the margin that makes the spillway the actual control rather than merely\n"
    "the intended one. If the water surface at design head reaches the surrounding\n"
    "rim, water escapes there too and the choice of where to overflow is given up.\n\n"
    "The default depends on what you are building, because the published figure is an\n"
    "embankment one. NRCS Conservation Practice Standard 378 requires 0.30 m between\n"
    "the design flow level and the top of a settled dam wall — a wall that would be\n"
    "breached if overtopped. A cut swale has no wall: its containment is undisturbed\n"
    "ground, and the failure is water leaving at an unarmoured point rather than a\n"
    "structure giving way. Holding a swale to the dam figure spends more depth than a\n"
    "shallow swale has.\n\n"
    "You can reduce it, including to zero, and the design will still be reported — but\n"
    "below the figure for this type it is called out, because it is a real reduction\n"
    "in margin and not a formality."
)
SPILLWAY_INLET = (
    "Where water enters this feature from upslope, if you have sited an inlet.\n\n"
    "A separate structure from the outflow, with a separate job. An outflow is a weir\n"
    "sized to pass a peak; an inlet is a protected entry that stops the incoming jet\n"
    "cutting the bank. It has no weir width, so none is shown — sizing it with the\n"
    "overflow formula would be confidently wrong.\n\n"
    "Place it from the map with 'Inflow Spillway'. Routed overflow is drawn to this\n"
    "point, so siting it makes the connection follow the ground rather than run\n"
    "centroid to centroid."
)
SPILLWAY_REVIEW_TABLE = (
    "Every feature that holds water, with the overflow it needs at the flow arriving\n"
    "now.\n\n"
    "These update as you design. Adding a feature upslope changes what reaches the\n"
    "ones below it twice over — it intercepts part of their catchment, and once full\n"
    "it passes its own peak on — so a sill sized last week can be undersized by a\n"
    "swale drawn today without anything else appearing to change.\n\n"
    "A feature with no spillway designed yet is still sized here, so you can see what\n"
    "it would need before deciding to build one. Use the arrows to site the outflow\n"
    "and the inlet on the ground.\n\n"
    "This list is the live one. The properties dialog freezes its figures when it\n"
    "opens, and for a feature you are still drawing it shows only that feature's own\n"
    "catchment, before any routing. Where they disagree, this is current."
)
SPILLWAY_WIDTH = (
    "Minimum crest width from the broad-crested weir formula:\n\n"
    "    Q = C x L x H^1.5,  so  L = Q / (C x H^1.5),  with C = 1.45\n\n"
    "C is the SI broad-crested coefficient from Brater & King for a crest 0.6 m or\n"
    "wider at ordinary head - which is what an earthen dam crest is.\n\n"
    "Widening a sill lowers the water depth over it, not the level it lets go at:\n"
    "1.4 m to 2.0 m takes a 0.30 m nappe to about 0.24 m, so the peak water surface\n"
    "sits 6 cm lower and freeboard gains that much. Stored volume is untouched.\n\n"
    "A minimum, not a recommendation. Standard practice adds 20-30% and protects\n"
    "the outlet against erosion — an under-armoured spillway scours, deepens, and\n"
    "drains the feature it was meant to protect."
)
SPILLWAY_LOCATION = (
    "Where the spillway sits on the ground. Place it from the map so its elevation\n"
    "comes from the DEM rather than from an assumption.\n\n"
    "Position matters as much as size: a spillway discharging onto a steep face or\n"
    "straight at the next feature's wall moves the erosion problem rather than\n"
    "solving it."
)
RUNOFF_COEFFICIENT = (
    "Fraction of rainfall leaving the catchment as surface flow. Everything else is\n"
    "intercepted by vegetation, held in surface hollows, or soaks in where it lands.\n\n"
    "This is the most consequential number in the whole assessment — it scales every\n"
    "inflow, and so every feature size, linearly. Sanity-check it against ground you\n"
    "have stood on: bare compacted earth sheds roughly half its rain, healthy pasture\n"
    "much less, and the same paddock sheds far more when already wet."
)
EXPORT_REPORT = (
    "Write the Site Water Plan as a PDF (or HTML) you can print, hand to a\n"
    "contractor, or send on.\n\n"
    "Needs a Baseline and nothing more. Sections that depend on a design, or on\n"
    "Re-analyse with Earthworks, say so on the page rather than quietly going\n"
    "missing — so the document always states how complete it is.\n\n"
    "Storage figures are arithmetic on the shapes you drew until Re-analyse has\n"
    "measured them against the burned terrain. It is a design estimate from a\n"
    "terrain model, not a survey and not an engineering certification."
)

# --------------------------------------------------------------------- Earthwork properties
# The per-feature dialog. Dams and swales each have a "key the banks into the
# hillside" option that does the same thing for a different shape, so the copy
# differs and the two constants are kept apart.
DAM_CREST_ELEVATION = (
    "Absolute elevation of the dam crest (top of the wall).\n\n"
    "Pre-filled from the highest ground the drawn line touches.\n"
    "All cells under the wall will be raised to this elevation,\n"
    "so the wall height varies with the valley shape beneath it.\n\n"
    "Water will pool behind the dam up to this level.\n"
    "Run Re-analyse with Earthworks to see retained volume."
)
DAM_KEY_BANKS = (
    "On (default): extend each end of the wall along its own bearing\n"
    "until the ground rises to the crest elevation, so water cannot flow\n"
    "around the ends. The drawn line is replaced by the wall that would\n"
    "actually have to be built — often noticeably longer — and both the\n"
    "capacity and the verification burn use that wall.\n\n"
    "You are told how far each end moved. If an end finds no ground at\n"
    "crest height within 250 m you get a warning: the design does not\n"
    "impound as drawn, and the crest is too high for this location.\n\n"
    "Off: keep the wall exactly as drawn. Water escapes around the ends\n"
    "if it stops short of high ground, and the reported storage is what\n"
    "the short wall actually holds."
)
DAM_WALL_VOLUME = (
    "Estimated volume of earthfill needed to construct the dam wall.\n\n"
    "Calculated as: sum along the wall of (crest − ground) × wall thickness × segment length.\n"
    "This is a rectangular cross-section approximation — add ~20% for side slopes."
)
DAM_MAX_HEIGHT = (
    "Height of the tallest point of the dam wall above the ground beneath it.\n\n"
    "Lower is better — a maximum height under 4–5 m is generally\n"
    "considered feasible for a farm dam without engineering certification.\n"
    "Higher walls require professional design and may need regulatory approval."
)
WALL_SLOPE = (
    "Wall batter as horizontal run per unit of depth (H:V).\n"
    "0 : 1 = vertical walls; 1 : 1 = 45°; flatter is more stable.\n\n"
    "The stored capacity accounts for the sloped walls. Note the DEM\n"
    "burn (Re-analyse) still carves vertical walls this phase — the\n"
    "verification comparison will surface the difference."
)
BOTTOM_WIDTH = (
    "Width of the channel floor, centred under the top width.\n"
    "Together with depth and top width this sets the side batter\n"
    "(shown below). A narrower bottom → steeper batter."
)
SIDE_SLOPE = (
    "Side batter angle from horizontal, derived from top/bottom width and\n"
    "depth. 45° = 1:1; a smaller angle is flatter/more stable; 90° = vertical."
)
CHANNEL_GRADIENT = (
    "Channel gradient — the fall in elevation per 100 m of drain length.\n\n"
    "Recommended range: 0.5–2.0 %\n"
    "  0.5 % — minimum to maintain flow, suits gentle slopes\n"
    "  1.0 % — standard design gradient\n"
    "  2.0 % — steep; use erosion protection (rock mulch / vegetation)\n"
    "  >2.0 % — significant erosion risk; consider drop structures\n\n"
    "Higher gradient → higher discharge capacity but greater erosion risk."
)
COMPANION_BERM = (
    "Excavated material is placed on the downhill side of the swale,\n"
    "forming a retaining berm. Volume is conserved — the spoil from the\n"
    "trench is spread along the bank at 75% compaction, and the bank is\n"
    "built to a LEVEL crest at whatever elevation that volume reaches.\n\n"
    "Level, not a constant height: a bank raised the same amount all along\n"
    "sloping ground has its crest on that slope, and the water leaves at\n"
    "the low end. The crest elevation is reported after Re-analyse with\n"
    "Earthworks, since it depends on how much earth the cut produced."
)
SWALE_KEY_BANKS = (
    "On (default): the berm wraps around both ends of the swale, running\n"
    "on until the ground rises to the crest, so water cannot flow around\n"
    "it. This is what makes a companion berm hold the water it is credited\n"
    "with — soil is added at the ends as well as along the downhill side.\n\n"
    "Off: the bank runs along the downhill side only. On ground that falls\n"
    "along the swale the pool escapes at the low end, and the berm adds\n"
    "little or nothing to what the trench holds by itself."
)
BERM_CREST = (
    "The level the bank was built to, from the volume of earth the trench\n"
    "produced. Water stands behind the berm up to this height.\n\n"
    "The height quoted under Calculated Capacity predicts the same bank on\n"
    "level ground; this is what the terrain model made of it."
)
SWALE_LENGTH = (
    "Total length of the swale as drawn on the map.\n"
    "Compare with the Recommended length below — if this swale\n"
    "is shorter, consider extending it or adjusting depth / width."
)
OVERFLOW_TARGET = (
    "Where this feature's overflow goes once it is full.\n\n"
    "Auto: the nearest feature downslope (elevation heuristic).\n"
    "A named target only receives water when it actually sits\n"
    "downslope of this feature — water can't flow uphill. An uphill\n"
    "choice is flagged below and its water goes downslope instead."
)
MANNINGS_CAPACITY = (
    "Peak discharge capacity using Manning's equation.\n"
    "Q = (1/n) × A × R^(2/3) × S^(1/2)\n"
    "Manning's n = 0.025 (compacted earthen channel)\n"
    "Trapezoidal cross-section, 1:1 side slopes."
)
MIN_DIMENSION = (
    "Narrowest dimension of the cross-section (the channel bottom width).\n"
    "If this falls below the DEM cell size the feature burns at 1-cell width\n"
    "(routing effect only) — you'll see a warning when you re-analyse."
)
FILL_RATIO = (
    "Ratio of design-storm inflow volume to depression capacity.\n"
    "Below 100%: depression absorbs the full storm event.\n"
    "Above 100%: overflow will occur — consider enlarging the earthwork."
)

# --------------------------------------------------------------------- Workbench panel
STORM_CHIP = "The design storm the score is computed against.\nClick to edit in Baseline."
SAVE_DESIGN = (
    "Save the DEM reference, every storm and sizing input, the site areas and "
    "all drawn earthworks to a single .tfd file."
)
OPEN_DESIGN = (
    "Open a .tfd design file. Inputs, areas and earthworks are restored "
    "immediately; you are then offered the baseline re-run that restores scoring."
)
# The ⓘ button beside the slope-class toggle. SLOPE_CLASS_INFO above is the body
# of the dialog it opens; this is the button's own tooltip.
SLOPE_CLASS_INFO_BTN = "What do the slope classes mean for earthworks?"
RESHAPE_EARTHWORK = (
    "Drag earthwork vertices on the map — the Live Assessment updates\n"
    "as you drag. Double-click a segment to insert a vertex; Del removes\n"
    "the highlighted vertex; right-click or Esc finishes."
)

# --------------------------------------------------------------------- Flow network view
NETWORK_WATER_HELD = "Water held (ponded) · fill"
NETWORK_SOAKED = (
    "Soaked into the ground over the event — captured, but not held "
    "as standing water, so it does not fill the feature."
)
NETWORK_SITE_EXIT = "Runoff not held by any feature"


# --------------------------------------------------------------------- Tool menu
# The draw tools' own copy lives on the earthwork registry, beside the type it
# describes. These are the three that belong to no type.
TOOL_DRAW_FALLBACK = "Draw a {label}."          # when a registry type carries no copy

TOOL_OUTFLOW_SPILLWAY = (
    "Click the map to site where the selected feature OVERFLOWS.\n"
    "Its crest is then read from the ground there rather than typed.\n"
    "This is the weir sized to pass the peak flow."
)
TOOL_INFLOW_SPILLWAY = (
    "Click the map to site where water ENTERS the selected feature\n"
    "from upslope. A separate structure with a separate job: an inlet is\n"
    "protected against the incoming jet cutting the bank, rather than\n"
    "sized to pass a peak.\n\n"
    "Routed overflow is drawn to this point, so placing it makes the\n"
    "connection follow the ground rather than run centroid to centroid."
)
TOOL_ROUTE_OVERFLOW = (
    "Click the feature that overflows, then the one it flows into.\n"
    "The link is drawn from the source's outflow spillway to the target's\n"
    "inflow spillway where both are placed.\n"
    "A link that would close a loop is refused."
)
TOOL_LINK_DRAIN = (
    "Click the END of a diversion drain, then the feature whose spillway\n"
    "feeds it. The drain is then cut from that crest instead of from the\n"
    "ground under its own line — which is a guess about where the water\n"
    "arrives.\n"
    "The end you click is the end it grades DOWN from.\n"
    "Run it again on the same pair to unlink."
)

# --------------------------------------------------------------------- Flow network
# Drawdown bands. Lancaster sizes earthworks so they "work, don't flood, and
# don't puddle"; this is the third constraint, and conventional practice is full
# drawdown inside 24-48 hours.
NETWORK_DRAIN_NEVER = (
    "No infiltration — this water has nowhere to go and will stand until it "
    "evaporates."
)
NETWORK_DRAIN_SLOW = (
    "Over 48 h to drain — too slow. Expect mosquito breeding, drowned "
    "plantings, and no freeboard left for the next storm."
)
NETWORK_DRAIN_MARGINAL = (
    "24-48 h to drain — acceptable, but little margin before the next storm."
)
NETWORK_DRAIN_GOOD = "Drains well within the conventional 24 h target."

NETWORK_CAPACITY_UNMEASURED = (
    "Storage capacity from the drawn cross-section. No terrain measurement "
    "yet — load a DEM to see what the ground actually holds here."
)
NETWORK_CAPACITY_MEASURED = (
    "{cap:,.0f} m³ is what this feature impounds on the actual ground, measured "
    "by flooding it on the DEM. The fill bar is against that figure.\n\n"
    "{drawn:,.0f} m³ is the drawn cross-section, brim-full — the number you "
    "can check by hand and the one a contractor builds to."
)
NETWORK_CAPACITY_ABOVE_GROUND = (
    "\n\nThe difference is water the bank holds above natural ground, and up "
    "the slope behind it. No cross-section can predict it; it depends on "
    "this hillside."
)

# --------------------------------------------------------------------- Verify table
VERIFY_TABLE_SUBHEAD = (
    "Geometric is calculated from your dimensions; At grid and Measured are "
    "flooded on the terrain. At grid is usually the larger — a keyed bank holds "
    "water above natural ground."
)
#: One per column of the verification table, in column order.
VERIFY_TABLE_HEADER_TIPS = (
    "The earthwork, as named on the map.",
    "CALCULATED. The drawn shape exactly, without the DEM. With a companion berm\n"
    "it is the trench plus the berm's own section. This is what a contractor\n"
    "builds to.",
    "MEASURED. What this feature impounds on this hillside — its own cut and its\n"
    "own bank, flooded on the DEM in isolation. Usually larger than Geometric,\n"
    "because a keyed bank holds water above natural ground and up the slope\n"
    "behind it. This is what the live score sizes against.",
    "MEASURED. The pond this feature ends up with once the whole design is built.",
    "Measured against At-grid. Both are floods, so a gap here means a neighbouring\n"
    "feature is changing where this one's water goes — nothing else.",
)

# --------------------------------------------------------------------- Ponding query
PONDING_VOLUME_HELD = "Total water volume in the connected depression area."
