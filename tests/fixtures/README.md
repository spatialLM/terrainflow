# Test fixtures

Two DEMs, the same terrain at two scales. Both are open data; neither carries a design.

| File | Extent | Cells | Valid cells | Size |
|---|---|---|---|---|
| `quail_island_catchment.tif` | 400 x 400 | 160,000 | 160,000 | 444 KB |
| `quail_island_full.tif` | 1139 x 1016 | 1,157,224 | 702,134 | 1.5 MB |

Both are float32, 1 m cells, EPSG:2193 (NZGD2000 / New Zealand Transverse Mercator),
nodata `-9999.0`.

## Provenance

Both are extracts of **tile BX24** of the LINZ *New Zealand LiDAR 1m DEM* collection,
covering Ōtamahua / Quail Island in Lyttelton Harbour, Canterbury.

Source: [LINZ Data Service](https://data.linz.govt.nz) — licensed **CC BY 4.0**.
Attribution: *Sourced from the LINZ Data Service and licensed for reuse under CC BY 4.0.*

The clip is a **bit-exact window** of the full tile at column 182, row 335 — not a
resample, not a reprojection. `test_fixture_integrity.py` asserts this, which is what
lets a result measured on one be compared against the other.

## Which one to use

**`quail_island_catchment.tif` is the default and should stay that way.** It is what
`tests_qgis/checks_fixture_regression.py` hard-codes, and the screenshot baselines in
`tests_qgis/` were captured against the synthetic surface, so repointing the whole
suite with `--fixture=` makes every render read as changed. Do not pass `--fixture` on
a full run.

**`quail_island_full.tif` is for work the clip cannot represent**, and it needs opting
into explicitly:

- It is 7.2x the clip in cells and holds real ridge and gully structure across a whole
  catchment. The clip yields 2 ponds; the full tile is where terrain-shape work —
  skeletonisation, ridgelines, long flow paths — is actually exercised.
- **Keep it out of the pure suite.** A full `FlowAnalysis.run()` on it costs roughly
  5.4 s; `tests/` as a whole runs in about 40 s and should stay there. Metadata and
  windowed reads are cheap and fine.
- It is 39% nodata (sea). Anything that divides by cell count needs to say which count
  it means — the clip has no nodata at all, so a bug of that shape is invisible there.

## What is deliberately absent

The real **designs** these were measured against — 35 earthwork features apiece — are
client work and are not distributed. Only the terrain is here, because only the terrain
is open data. A test that needs features should build them, so that it states its own
preconditions rather than depending on someone's saved session.
