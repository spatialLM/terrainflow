# TerrainFlow QGIS Plugin

## Project layout
- Source: `plugin/` (top-level files only)
- Deployed: `C:/Users/liamm/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/plugin/`
- **After every change: copy the modified file to the deployed location**

## Gotcha — nested duplicate directories
The `plugin/` folder contains a deeply nested `plugin/plugin/plugin/...` structure (artifact from repeated folder copies). The real source files are only at depth 1. When searching, use `find . -maxdepth 3 | grep -v "/plugin/plugin/"` to avoid false matches.

## Key files
- `terrain_flow.py` — main plugin controller + AnalysisWorker thread (2192 lines)
- `terrain_flow_dialog.py` — docked panel UI
- `earthwork_properties_dialog.py` — swale/berm/basin properties popup
- `processing/flow_analysis.py` — pysheds flow direction/accumulation/catchments
- `processing/scs_runoff.py` — SCS Curve Number runoff model
- `processing/dem_burner.py` — burns earthworks into DEM, computes ponding
- `map_tools/` — draw_line, draw_polygon, place_point, ponding_query, select_contour

## Architecture notes
- Analysis runs in a QThread (AnalysisWorker) — never pass QGIS objects into the worker
- Stream raster stores acc values at stream cells (not binary) for gradient rendering
- combo_contour_layer in the UI should be checked before self._contour_layer in any contour function
