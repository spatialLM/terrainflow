"""
help_text.py — single source of truth for input/button tooltip copy.

The plugin is meant to be usable with limited prior knowledge, so most controls
carry an explanatory tooltip. Centralising the strings here (rather than inline in
panel.py) keeps the voice consistent, makes the copy reviewable in one place, and
lets dialogs/other tabs reuse the same wording.

Usage:
    from terrainflow_assessment.qgis import help_text as H
    widget.setToolTip(H.STREAM_THRESHOLD)

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
    "Click on a blue zone in the 'Water Captured' layer to select the\n"
    "entire connected pooling area and report its volume and surface area.\n\n"
    "Baseline: shows natural low spots where water collects.\n"
    "Earthworks: shows water captured by your swales/basins."
)
SLOPE_CLASS = (
    "Semi-transparent slope suitability overlay (calculated from DEM):\n"
    "  Green  (0–3°):   ideal — suitable for swales and basins\n"
    "  Yellow (3–8°):   moderate — suitable with care\n"
    "  Orange (8–15°):  challenging — consider companion berm\n"
    "  Red    (>15°):   steep — berms or diversion drains recommended"
)
SLOPE_ARROWS = (
    "Overlay arrows showing the direction of steepest downslope at regular intervals.\n"
    "Generated from the DEM aspect — arrows point in the direction water would flow."
)
FLOW_LINES = (
    "Draw smooth downslope flow lines across the site — the paths water would\n"
    "take over the surface. Lines converge into valleys and drainage lines, so\n"
    "you can read how runoff gathers and where it concentrates.\n\n"
    "Coloured by slope steepness (green gentle → red steep), so each line shows\n"
    "both the path and how steep the ground it crosses is.\n\n"
    "Traced from the DEM surface gradient. Requires: baseline analysis run."
)
SLOPE_VECTORS = (
    "Overlay a regular field of downslope arrows. Each arrow points the way water\n"
    "runs and is coloured and sized by slope steepness (green gentle → red steep),\n"
    "so you can read direction and gradient at a glance.\n\n"
    "Complements the flow lines (which show the path). Requires: baseline run."
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
    "<tr><td><b><font color='#e07b00'>8–15°</font></b></td>"
    "<td>Workable with care. Keep swales shorter, add a companion berm on the "
    "downhill side, and watch for concentrated flow.</td></tr>"
    "<tr><td><b><font color='#c62828'>15–18°</font></b></td>"
    "<td>Upper limit for most swales. Above this, water moves fast and cut/fill "
    "grows — prefer diversion drains or benched terraces.</td></tr>"
    "<tr><td><b><font color='#7f1d1d'>&gt; 18°</font></b></td>"
    "<td>Too steep for swales. Use diversion drains, terraces, or keep as "
    "vegetated slope.</td></tr>"
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
TOP_N = (
    "How many of the top-ranked candidate swale contours to pull into a\n"
    "separate layer (ranked by peak inflow accumulation)."
)
TOP5_SWALES = (
    "Create a separate layer containing the top-N ranked candidate\n"
    "swale contours by peak inflow accumulation (set N on the left).\n\n"
    "Requires: Analyse Contours run first."
)
INFLOW_BANDS = (
    "Colour every contour by the runoff that drains onto it along its length,\n"
    "on one continuous low→high ramp keyed to the largest inflow found across\n"
    "all contours (m³ over the event).\n\n"
    "Because the ramp is absolute and shared, a hot (red) stretch on one contour\n"
    "carries genuinely more water than a cool (blue) stretch anywhere else — so\n"
    "you can compare swale-worthy spots across the whole site.\n\n"
    "Requires: Analyse Contours + Baseline run first."
)
INFLOW_SCALE = (
    "How the inflow gradient maps colour to value:\n\n"
    "  Log — colour by log(inflow); a colour still means a fixed m³ (comparable\n"
    "  across the map) but the huge range is compressed so small and large\n"
    "  stretches are all distinguishable. Best when one point dwarfs the rest.\n\n"
    "  Linear — colour straight from 0 to the maximum m³. True absolute scale,\n"
    "  but a single extreme point flattens everything else to one colour.\n\n"
    "  Quantile — equal number of stretches per colour; maximal separation, but\n"
    "  colour becomes a rank rather than an absolute value."
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
    "With the depth and soil batter this sets the trapezoidal capacity per\n"
    "metre used to size the required swale length."
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
    "live storage in the trench (trapezoidal cross-section × length × 0.8\n"
    "freeboard) plus what soaks into its bed over the storm duration."
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
AREA_SUBTOTALS = (
    "The same water balance restricted to one area, so a large site can be\n"
    "read block by block. Subtotals come from the same cell-by-cell routing as\n"
    "the site figure, so they always add up to it."
)
STRESS_POINTS = (
    "Where a feature is predicted to overtop before it is nominally full.\n\n"
    "Inflow concentrates where drainage lines cross an alignment, so a swale\n"
    "with adequate total capacity can still spill part-way along. Each marker\n"
    "shows the distance along the feature and the surplus in m³."
)
THROUGHFLOW = (
    "Total water that passes through each cell over the event, as a blue\n"
    "gradient — pale where flow is diffuse, dark where it concentrates.\n\n"
    "Finer-grained than the stream layer: it shows the whole surface, so you\n"
    "can see water gathering before it becomes a defined channel."
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
    "throughflow, exit points) and earthwork sizing alike.\n\n"
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
VERIFICATION_TABLE = (
    "What each feature was designed to hold, against what the burned terrain\n"
    "actually holds. Four numbers because a single figure hid three unrelated gaps.\n\n"
    "DESIGN      the drawn shape less the 0.8 freeboard allowance — what you plan on\n"
    "GEOMETRIC   the drawn shape exactly — the hole as specified\n"
    "AT GRID     that shape rasterised to the DEM — what this cell size can represent\n"
    "MEASURED    ponding read off the burned DEM — what the burn produced\n\n"
    "Only the last comparison tests anything. Delta is MEASURED against AT GRID, and\n"
    "it should sit near zero; a non-zero delta there is a burn problem and nothing\n"
    "else.\n\n"
    "The other two gaps are expected. Geometric to At-grid is the resolution penalty,\n"
    "and it is often POSITIVE: a 1 m cell cannot cut a battered side, so it burns the\n"
    "section square and ends up larger than drawn. Design to Geometric is the\n"
    "freeboard you chose, a fixed -20%.\n\n"
    "A feature narrower than one cell shows no measured figure. It is verified for\n"
    "placement and routing only — a storage volume read off the grid would measure\n"
    "the grid rather than the design."
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
    "Leave 'auto' ticked to track the requirement as the design changes — including\n"
    "when a new feature upslope starts routing overflow through this one. Untick it\n"
    "to commit to a width, and it will be flagged if it later falls short.\n\n"
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
    "Setting a spillway also fixes the order the system fills in, because the crest\n"
    "elevation decides which feature spills into which. Leave the group unticked to\n"
    "record that no spillway is designed yet."
)
SPILLWAY_RIM = (
    "The lowest ground containing this feature — its natural spill point, sampled\n"
    "from the DEM around the footprint.\n\n"
    "This is the datum the crest is measured against, because it is the elevation\n"
    "the terrain actually supplies. A crest typed without reference to it is a\n"
    "number with nothing behind it."
)
SPILLWAY_CREST = (
    "Absolute elevation of the spillway crest — the level water reaches before it\n"
    "starts leaving. Storage is measured to this line, so lowering the crest trades\n"
    "capacity for freeboard directly.\n\n"
    "Bound to 'Below rim': changing either updates the other. The value is held\n"
    "within what the feature can offer — no higher than rim minus head minus\n"
    "freeboard, and no lower than the floor."
)
SPILLWAY_DROP = (
    "How far the crest sits below the rim. The same crest as the row above,\n"
    "expressed the way it is usually set out in the field — you measure down from\n"
    "the surrounding ground, not up from a datum.\n\n"
    "Needs a DEM to be meaningful, so it is disabled until one is loaded."
)
SPILLWAY_HEAD = (
    "Depth of water flowing over the crest at peak flow. Typical design values are\n"
    "0.2-0.5 m.\n\n"
    "It sets the width through the weir formula — a lower head needs a wider\n"
    "spillway to pass the same flow — and it consumes freeboard, because the water\n"
    "surface sits this far above the crest while the spillway is working. Raising\n"
    "it therefore lowers the highest crest the feature can accept."
)
SPILLWAY_DESIGN_FLOW = (
    "Peak design flow = total storm inflow / storm duration.\n\n"
    "This is an average rate over the whole event, so a short intense burst will\n"
    "exceed it. Treat the resulting width as a floor rather than an answer, and note\n"
    "that it inherits whatever runoff method the Baseline tab is set to."
)
SPILLWAY_WIDTH = (
    "Minimum crest width from the broad-crested weir formula:\n\n"
    "    Q = C x L x H^1.5,  so  L = Q / (C x H^1.5),  with C = 1.45\n\n"
    "C is the SI broad-crested coefficient from Brater & King for a crest 0.6 m or\n"
    "wider at ordinary head - which is what an earthen dam crest is.\n\n"
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
