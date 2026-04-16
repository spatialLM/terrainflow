# Sample Catchment — Demo Data

Place a demo DEM (GeoTIFF, preferably 1 m resolution) and a site boundary
(GeoPackage polygon) in this folder to use for testing and presentations.

**Recommended test data sources (NZ):**
- LINZ Data Service: 1 m LiDAR DEMs (free download by region)
  https://data.linz.govt.nz/layer/53621-nz-8m-digital-elevation-model-2012/
- Regional councils often provide 1 m LiDAR for rural areas

**Naming convention (for the demo):**
- `dem.tif` — digital elevation model
- `boundary.gpkg` — site boundary polygon (NZTM2000 / EPSG:2193)

**Synthetic rainfall (80 mm / 12 hr):**
Use the "Uniform event" simulation mode with 80 mm, 12 hr, 60 min timestep.
