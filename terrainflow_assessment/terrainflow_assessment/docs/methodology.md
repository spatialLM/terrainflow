# TerrainFlow Assessment — Methodology

## Overview

TerrainFlow Assessment estimates surface water runoff, flow routing, and earthwork
effectiveness for a defined catchment area. The workflow is:

1. **Load DEM** → extract terrain metadata
2. **Baseline analysis** → flow direction, accumulation, stream network, exit points
3. **Contour & keypoint analysis** → identify candidate swale locations and pond sites
4. **Earthwork design** → place swales, berms, basins, dams, diversions
5. **DEM burning** → re-run flow analysis on modified terrain
6. **Fill simulation** → time-stepped water routing through earthworks
7. **Before/after reporting** → compare baseline vs post-intervention

---

## Flow Routing

Flow direction and accumulation are computed using [pysheds](https://mattbartos.com/pysheds/).

**Algorithm:**
1. Fill single-cell pits (DEM noise artefacts)
2. Breach depressions (preserves natural storage shapes)
3. Resolve flat areas
4. Compute D-infinity flow direction (recommended) or D8
5. Compute weighted flow accumulation (weighted by SCS runoff volume per cell)

**D-infinity vs D8:**
D-infinity distributes flow fractionally between the two steepest downslope cells,
producing smoother stream networks on gentle terrain. D8 routes all flow to a single
neighbour and can produce artificial straightness on uniform slopes.

---

## Rainfall-Runoff Model (SCS Curve Number)

Runoff depth is estimated using the USDA-NRCS SCS Curve Number method:

```
S  = (25400 / CN) - 254     # potential max retention (mm)
Ia = 0.2 × S                 # initial abstraction (mm)
Q  = (P - Ia)² / (P - Ia + S)  if P > Ia, else Q = 0
```

Where:
- P = gross rainfall (mm)
- CN = Curve Number (1–100, higher = more runoff)
- Q = runoff depth (mm)

**Antecedent moisture conditions:**
- Dry (AMC I): lower CN → less runoff (dry soil, high infiltration capacity)
- Normal (AMC II): standard design CN
- Wet (AMC III): higher CN → more runoff (saturated soil)

**Limitations:**
The SCS model assumes uniform rainfall over the catchment. Spatial variability
in rainfall is not modelled. The model is best suited for single-event design
storms, not continuous simulation.

---

## Contour Analysis

**Extraction:** Contours are generated at a user-specified interval using GDAL
(`gdal_contour`). If GDAL is unavailable, a scikit-image marching squares fallback
is used.

**Slope filter:** Contours where the mean terrain slope exceeds 18° (approximately
1:3 grade) are removed. Swales placed on slopes steeper than 18° risk erosion and
are not recommended.

**Flow crossing ranking:** The flow accumulation raster is sampled at N points along
each contour. The peak value (maximum upstream contributing area crossing the
contour) is used to rank contours. Higher-ranked contours intercept more runoff and
are better candidates for swale placement.

**Usable area:** The user can draw an inclusion polygon to restrict contour analysis
to a specific area. Contours outside the polygon are discarded.

---

## Earthwork Design

### Swale capacity
Trapezoidal cross-section (1:1 side slopes), with 0.8 freeboard factor:

```
bottom_width = width - 2 × depth
cross_section = ((bottom_width + top_width) / 2) × depth
capacity = cross_section × length × 0.8
```

With companion berm (volume-conserved, 75% compaction):

```
berm_height = sqrt(cross_section × 0.75)
additional = berm_height × top_width / 2
capacity += additional × length × 0.8
```

### Basin capacity
```
capacity = polygon_area × depth × 0.8
```

### Dam wall
Cells under the dam wall are set to the crest elevation (absolute, not relative).
Water ponds behind the wall up to the crest.

### Diversion drain (Manning's equation)
Peak discharge capacity:

```
Q = (1/n) × A × R^(2/3) × S^(1/2)
n = 0.025 (compacted earthen channel)
side slopes: 1:1
```

Where A = cross-sectional area, R = hydraulic radius, S = grade (m/m).

---

## DEM Burning

Earthworks are applied to a copy of the original DEM before re-analysis:

| Earthwork | Operation |
|-----------|-----------|
| Swale | Lower cells by `depth` within `buffer(width/2)` |
| Companion berm | Raise downhill parallel zone (volume conserved) |
| Berm | Raise cells by `depth` |
| Basin | Lower cells by `depth` within polygon |
| Dam | Set cells to `crest_elevation` |
| Diversion | Grade-controlled lowering: `target = start_elev - dist × gradient_pct/100` |

**Resolution note:** DEM burning accuracy depends on DEM cell size. At 1 m
resolution, swales narrower than 1–2 m may not be captured accurately. For
reliable results, DEM cell size should be ≤ swale width / 2.

---

## Fill Simulation

The simulation steps through a rainfall time series and routes water through
the earthwork network:

1. For each timestep: compute incremental SCS runoff depth
2. Distribute runoff to earthworks proportionally to their footprint area
3. Subtract infiltration losses (steady-state rate × area × timestep)
4. If stored volume > capacity: overflow to next downslope earthwork (by elevation)
5. If no downstream earthwork: add to site exit flow

**Cascading overflow:** Earthworks are processed from highest to lowest elevation.
When an earthwork overflows, the surplus is passed to the nearest lower-elevation
earthwork. This is a simplified routing model — actual cascade depends on the
downstream flow path, which may differ from the elevation-nearest earthwork.

**Infiltration model:** Steady-state infiltration rate × soil area × timestep.
This is conservative (ignores initial high infiltration rates). The Green-Ampt
model would give more accurate results but requires additional soil parameters.

---

## Cut/Fill Volume

| Earthwork | Cut volume | Fill volume |
|-----------|-----------|------------|
| Swale | Trapezoidal cross-section × length | Companion berm (if enabled) |
| Basin | Polygon area × depth | 0 |
| Diversion | Trapezoidal cross-section × length | 0 |
| Berm | 0 | Triangular cross-section × length |
| Dam | 0 | Width × depth × length (approximate) |

Net cut/fill = sum(cuts) − sum(fills). Positive = net cut (soil exported).
Negative = net fill (import required or balance from cut).

---

## Limitations and Caveats

1. **DEM resolution:** Results are only as accurate as the input DEM. A 1 m DEM
   is recommended. Coarser DEMs will miss small swales and underestimate flow
   convergence.

2. **Rectangular geometry:** DEM burning uses rectangular cross-sections at DEM
   cell size. Actual earthwork hydraulics (trapezoidal cross-section, roughness)
   are not modelled in the raster domain.

3. **Simplified infiltration:** Constant steady-state rate. Does not model:
   - Initial abstraction dynamics
   - Ponded infiltration vs lateral flow
   - Soil saturation over time

4. **Cascade routing:** Overflow is routed to the nearest lower-elevation
   earthwork, not necessarily the hydraulically connected one. Review the
   earthwork placement to confirm cascade logic is realistic.

5. **Single-event model:** The SCS model is calibrated for single design storms.
   Long-duration continuous simulation requires a different approach.

6. **Future improvements:**
   - Surveyed cross-sections for capacity calculation
   - Green-Ampt infiltration model
   - Actual flow-path-based cascade routing
   - Continuous rainfall simulation from station data
---

*TerrainFlow Assessment v0.1 — For design decisions, verify with a qualified engineer.*
