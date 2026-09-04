# Bracket 6 — Open-source geospatial

**~3 contacts, tracker Tier 4.** QGIS.org / the QGIS plugin community, Whitebox
Geospatial Inc., and Terraformation on its open-source technology side.

## Who
The maintainers and communities whose software this plugin sits inside or builds
on. Not users to be recruited — peers whose ecosystem he is about to publish into.

## What they measure
Whether the code is readable and maintained. Whether the author understands the
platform conventions. Whether this is another plugin abandoned after one release.

## What they distrust
Drive-by contributions. Announcements dressed as collaboration. Plugins that
duplicate existing tools without saying why.

## Facts that prove it, in order
1. `TF-02` AGPL-3.0-or-later, source public. First sentence — it changes how
   everything after it reads.
2. `TF-50` modular pure-Python `core/` with a separate `qgis/` adapter layer.
3. `TF-51` above 95% coverage on the pure-Python modules under a branch-coverage
   gate — **the calculation layer, not the UI.** Say it exactly that way.
4. `TF-52` QGIS 3.22+ LTR with the API longevity work done. To QGIS.org this is
   the sentence that says he intends to maintain it.
5. `TF-40`, `TF-41` standard methods, no novelty claimed.
6. `TF-72` Plugin Manager support not done — which for QGIS.org is the relevant
   open item and the natural thing to ask about.

## Whitebox Geospatial — handle deliberately
`TF-32` says the flow analysis matches WhiteboxTools almost cell for cell. Writing
to Whitebox Geospatial means telling the authors of WhiteboxTools that their
software was used as the reference standard.

That is a genuinely strong opener and a delicate one. Get it right:

- It is a **cross-check against their work**, not a comparison in which anything
  competes. WhiteboxTools is the reference; TerrainFlow is the thing being checked.
- Never imply equivalence, improvement, or that it does anything WhiteboxTools
  does better.
- "Almost cell for cell" is the wording. Not "identical", not "matches exactly".
- The honest ask is whether that comparison was done in a way they would consider
  fair — which is a real question, and a good one.

## Opener mechanism — the method statement
Same as bracket 5. State what it is, what it is licensed under, and link the repo.

> It is a QGIS plugin for water-harvesting earthworks design — D-infinity flow
> accumulation, earthworks burned into the DEM, re-run. AGPL, and the code is here.

## Limits to state
`TF-72`, `TF-73`, `TF-71`. The roadmap gaps are the most interesting part of the
email to this reader.

## The ask
`call` plus `react` — read the method, tell me what is wrong with it. For QGIS.org
specifically the honest question is about publishing conventions and Plugin
Manager submission, not about the hydrology.

## Never say
- "Novel", "first", or any novelty claim.
- Anything positioning the plugin against WhiteboxTools, pysheds, or SAGA.
- Marketing register of any kind. This is the least tolerant audience for it.
- That it is validated. `TF-71`.

## Signals
```yaml
strong:
  - open source
  - QGIS
  - plugin repository
  - OSGeo
  - GDAL
  - WhiteboxTools
  - geospatial software
  - AGPL
  - GitHub
weak:
  - GIS
  - remote sensing
  - developer
  - maintainer
  - community
```
