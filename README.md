TerrainFlow
A QGIS plugin for catchment-scale hydrological terrain analysis and earthworks design.
TerrainFlow enables engineers and land managers to model how water moves across a landscape — and how proposed earthworks change that movement — directly inside QGIS, using freely available DEM and LiDAR data, without requiring external hydrological software.
The core workflow is a before-and-after comparison: run a baseline flow analysis on the existing terrain, draw proposed earthworks (swales, dams, basins, diversion drains) onto the map, and immediately re-run the analysis on the modified DEM to quantify the effect on flow paths, catchment boundaries, and water retention.
Status: Active development. The core pipeline runs end-to-end — DEM in, hydrological outputs out, earthworks designed and compared. Current focus is on improving usability, refining output accuracy and adding test coverage. Contributions and feedback welcome.

What it does
Baseline flow analysis
Load a DEM and TerrainFlow runs a complete hydrological terrain analysis:
Output
Description
Flow accumulation raster
Maps where surface water concentrates across the landscape
Drainage network
Extracts the surface drainage network from flow accumulation thresholds
Catchment boundary
Delineates the contributing area draining to your site
Exit points
Identifies where water leaves the site boundary

Flow routing uses D-infinity (default) or D8 — D-infinity distributes flow fractionally between the two steepest downslope cells, producing smoother accumulation patterns on gentle terrain.

Earthworks design and before/after comparison
Draw proposed earthworks directly onto the map canvas. TerrainFlow burns each feature into a copy of the DEM and re-runs the full flow analysis, letting you compare:
Flow paths before and after intervention
Where water now ponds (and how much)
Whether earthworks achieve the intended hydrological effect
Supported earthwork types:
Type
Description
Swale
Cut channel along contour; optional volume-conserved companion berm placed on downhill side
Berm
Raised barrier; deflects or slows surface flow
Basin
Polygon detention area; water ponds within the excavated footprint
Dam
Wall set to absolute crest elevation; water pools behind up to the crest
Diversion drain
Graded channel designed to intercept and redirect flow; gradient and discharge calculated via Manning's equation

Ponding analysis
After earthworks are applied, TerrainFlow calculates where water pools in the modified landscape by comparing the modified DEM against a depression-filled version. Click any ponding zone with the Ponding Query Tool to get the total volume (m³ and litres) and area of that pond.
SCS Curve Number rainfall-runoff
Estimate runoff from a rainfall event using the USDA-NRCS SCS Curve Number method. Adjust for:
Soil type (sand through clay, with reference CN values)
Antecedent moisture condition (dry / normal / wet)
Storm intensity (presets or custom mm/hr)
CN-weighted flow accumulation produces runoff-aware stream thresholds and retention volume estimates.
Keyline design
Identifies the key hydrological features of the landscape for Yeomans-style keyline planning:
Keypoints — valley inflection points where slope eases from steep to gentle
Ridgelines — watershed divides (low-accumulation, topographically high cells)
Pond sites — recommended locations near each keypoint based on valley cross-section
Cultivation lines — keyline elevation and parallel cultivation lines above/below
Storm simulation
A time-stepped flow simulation partitions storm rainfall into discrete timesteps and computes weighted flow-accumulation for each step. Outputs include incremental (per-step) and cumulative rasters, with animated playback in QGIS.
Session management
Save and reload design sessions as .tflow files — preserving earthwork geometries, CN zones, analysis settings, and DEM path. Reload a session, re-run baseline, and continue where you left off.

Why TerrainFlow
Catchment hydrological analysis that includes earthworks design typically requires either expensive proprietary software (SWMM, HEC-HMS, TUFLOW) or a Python workflow running outside any GIS environment. For engineers and practitioners working at site or farm scale — where a rapid, spatially grounded assessment is needed before committing to detailed modelling — there is a gap.
TerrainFlow fills that gap: a practical, accessible tool that runs inside QGIS, uses freely available LiDAR data, supports interactive earthworks design, and shows you the hydrological effect of your interventions immediately — all within the same GIS environment you use for everything else.
It is not intended to replace detailed hydraulic modelling. It is designed to inform early-stage engineering judgement with real spatial data.

Installation
QGIS plugin manager installation is under development. Current installation is via the QGIS Plugin directory.
Requirements:
QGIS 3.x
Python 3.8+
pysheds
rasterio
numpy
scipy
shapely
geopandas
Install dependencies:
bash
pip install pysheds rasterio numpy scipy shapely geopandas
Install the plugin:
Clone or download this repository and copy the plugin/ folder into your QGIS plugins directory:
Linux/Mac: ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/terrainflow/
Windows: %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\terrainflow\
Then enable TerrainFlow in QGIS via Plugins → Manage and Install Plugins.

Usage
Open QGIS and enable the TerrainFlow panel (Plugins → TerrainFlow → Toggle Panel)
Load a DEM from file or select an existing raster layer
Optionally set a site boundary polygon to clip analysis to your area of interest
Click Run Baseline Analysis — TerrainFlow generates flow accumulation, drainage network, and catchment boundary as QGIS layers
Draw earthworks using the panel tools (swale, berm, basin, dam, diversion drain)
Click Run with Earthworks — TerrainFlow burns the earthworks into the DEM and re-runs the analysis
Compare baseline and post-earthworks layers side by side; use the Ponding Query Tool to inspect ponding volumes

Data sources
TerrainFlow is designed to work with freely available DEM data. For New Zealand:
LINZ LiDAR data — data.linz.govt.nz (high-resolution LiDAR DEMs for most of NZ)
LINZ aerial imagery — useful for land cover classification to inform CN values

Roadmap
DEM loading and preprocessing (pit filling, D-infinity and D8 routing)
Flow direction, flow accumulation, drainage network extraction
Catchment contributing area preview
Site boundary support and exit point detection
Full QGIS docked panel GUI
Interactive earthwork drawing tools (line, polygon, point)
DEM burning — swale, berm, basin, dam, diversion drain
Volume-conserved companion berm placement
Manning's equation discharge for diversion drains
SCS Curve Number rainfall-runoff with AMC adjustments
Ponding analysis and Ponding Query Tool
Keyline design (keypoints, ridgelines, pond sites, cultivation lines)
Time-stepped storm simulation with animated playback
Session save/load (.tflow files)
Refine output accuracy and edge case handling
Test suite and validation against benchmark catchments
QGIS Plugin Manager installation support
Full usage documentation

Demo
A short walkthrough of TerrainFlow running on a Canterbury catchment using LINZ LiDAR data is available here: [YouTube link — coming soon]


About
Developed by Liam Murphy — Civil & Environmental Engineer (BEng Civil & Environmental Engineering; MSc Sustainable Energy Engineering), Christchurch, New Zealand.
Built to provide a practical, accessible hydrological design tool for engineers and land managers working with QGIS — particularly for early-stage assessment of earthworks interventions on surface water behaviour.
liamurphynz@gmail.com

Licence
GNU Affero General Public License v3.0. See LICENSE for details.

