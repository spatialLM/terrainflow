# The claim bank

Every factual sentence in an email cites an ID from this file as `[f:TF-nn]`.
`tools/check.py` hard-fails an ID that is not here, and warns on a number that
cites nothing.

**If no fact fits what you want to say, the answer is to not say it.** Do not add
an entry here to license a sentence you already wrote — add it because it is true
and you could show it to someone. The refusal is the point: it is what stops the
pipeline inventing capability.

> **Rebuilt 2026-09-01 from the base email, which supersedes the repo README.**
> The README still lists validation as a roadmap item; it is out of date. The
> earlier version of this file said there was no validation and forbade
> mentioning any, which would have blocked the strongest paragraph in the email.
> Where the two sources disagree, the base email wins.

Status is part of the fact. `shipped` works today. `partial` works with known
gaps. `planned` is not real yet and may not be claimed in any tense but future.

---

## What it is

TF-01  `shipped`  Terrain Flow Design — TerrainFlow in the QGIS plugin
       repository — a free and open-source QGIS plugin for designing and
       validating water-harvesting earthworks: swales, basins, dams, berms and
       diversion drains.

TF-02  `shipped`  Free and open source under AGPL-3.0-or-later. Full source at
       https://github.com/spatialLM/terrainflow

TF-03  `shipped`  Pre-release. Feedback is being sought deliberately before
       release rather than after. This is the reason the email exists, and it
       expires when the tool ships.

TF-04  `shipped`  It runs inside QGIS, not beside it.

## The data it needs

TF-10  `shipped`  It works from public LiDAR where that exists, or coarser open
       elevation data anywhere it does not. **This is the fact that makes the
       tool relevant outside New Zealand** — lead with it for every recipient
       who is not NZ-based.

TF-11  `shipped`  In New Zealand specifically, LINZ LiDAR at data.linz.govt.nz.
       Only mention LINZ to NZ recipients; to anyone else it reads as a
       limitation rather than a convenience.

## What it does, end to end

TF-20  `shipped`  Runs a rainfall flow analysis on topographic data.

TF-21  `shipped`  In-built tools identify suitable earthwork locations.

TF-22  `shipped`  Earthworks are drawn in directly, with accurate dimensions and
       depths.

TF-23  `shipped`  It gives immediate feedback on earthwork sizing, and on how
       overflow cascades from one structure to the next.

TF-24  `shipped`  Re-run the flow analysis to see how the design performs before
       anything is built.

TF-25  `shipped`  It produces a design report showing what the tool actually
       outputs. Not attached to a first email — offered in a reply.

TF-26  `shipped`  Click any ponding zone and get its volume, in cubic metres and
       litres, and its area.

TF-27  `shipped`  Keyline design in the Yeomans sense: keypoints where valley
       slope eases, ridgelines, recommended pond sites, and the keyline with
       parallel cultivation lines.

TF-28  `shipped`  Time-stepped storm simulation with animated playback in QGIS.

TF-29  `shipped`  Sessions save and reload as `.tflow` files, preserving
       earthwork geometry, CN zones, settings and the DEM path.

## Validation — the strongest material in the email

**TF-30 to TF-32 may never appear without TF-33.** The qualifier is not a
disclaimer bolted on; it is what makes the claim credible to the people most
worth convincing.

TF-30  `shipped`  Tested against CAMELS-NZ, a dataset published specifically to
       validate hydrological models against real measurements.

TF-31  `shipped`  On a catchment with 34 years of measured flow, both the
       catchment area it delineates and the runoff it predicts came within a
       couple of percent of the published figures.

TF-32  `shipped`  Its flow analysis matches WhiteboxTools, a widely used
       open-source hydrology package, almost cell for cell.

TF-33  `shipped`  **That is one catchment so far, with more to be tested before
       release.** Always attached to any of TF-30, TF-31 or TF-32.

## Method

TF-40  `shipped`  Flow routing is D-infinity by default, D8 available.
       D-infinity splits flow fractionally between the two steepest downslope
       cells, which reads better on gentle terrain.

TF-41  `shipped`  Rainfall-runoff uses the USDA-NRCS SCS Curve Number method,
       adjustable for soil type, antecedent moisture and storm intensity.

TF-42  `shipped`  Diversion drain gradient and discharge use the Manning
       equation.

TF-43  `shipped`  A swale can place a volume-conserved companion berm on its
       downhill side — the spoil goes where spoil actually goes.

TF-44  `shipped`  Ponding is found by comparing the modified DEM against a
       depression-filled version.

## How it is built

TF-50  `shipped`  Modular pure-Python `core/` with a separate `qgis/` adapter
       layer.

TF-51  `shipped`  Above 95% test coverage on the pure-Python modules under
       pytest with a branch-coverage gate. **This is the calculation layer, not
       the QGIS UI layer** — say it that way or not at all.

TF-52  `shipped`  Targets QGIS 3.22+ LTR, with the API longevity work done.

TF-53  `shipped`  Depends on pysheds, rasterio, numpy, scipy, shapely and
       geopandas.

## Who made it

TF-60  `shipped`  Liam Murphy, a civil and environmental engineer from Ireland
       living in Christchurch, New Zealand, with a Master's in Renewable and
       Sustainable Engineering.

TF-61  `shipped`  He uses QGIS professionally.

TF-62  `shipped`  He started building it after volunteering on a rewilding
       project that lost several years of plantings to drought — wanting to test
       whether simple earthworks could have captured enough rainfall to change
       the outcome. Most land managers have no easy way to run that test before
       committing time and money to digging.

TF-63  `shipped`  One person, not a company. No team, no funding, and no users
       to point at yet.

TF-64  `shipped`  He completed Geoff Lawton's permaculture design course.
       *(Stated by Liam, 2026-09-01.)*

TF-65  `shipped`  He has followed Andrew Millison's videos for some time, and
       they were a large part of what led him to do that course. *(Stated by
       Liam, 2026-09-01. Millison only — do not generalise this to any other
       educator on the list.)*

TF-66  `shipped`  His experience with mapping software is both professional and
       academic. *(Stated by Liam, 2026-09-07, in the four emails he sent to
       Andrew Millison, Zaytuna Farm, Sam and Ben Missmer. Widens TF-61, which
       covers professional QGIS use only.)*

TF-67  `shipped`  He first wanted an easy way to design earthworks for land in
       Ireland while living abroad, and the scope expanded as soon as he started
       working on it. *(Stated by Liam, 2026-09-07, in the same four emails.
       This is the earlier motivation and TF-62's rewilding project is what
       widened it — the two are sequential, not alternatives.)*

TF-68  `shipped`  He completed that course in 2025. *(Written as "last year" by
       Liam, 2026-09-07, in the same four emails; recorded here as the absolute
       year so it does not rot. Dates TF-64 — cite the two together.)*

TF-69  `shipped`  He has followed Andrew Millison's videos for a few years.
       *(Stated by Liam, 2026-09-07, in the email he sent to Andrew Millison.
       Sharpens the "for some time" of TF-65; prefer this wording. Millison
       only, and the caution on TF-65 still applies.)*

### Biography is not free text

TF-60 to TF-69 are the **only** permitted claims about Liam. The gate can check
what is said about the software and about a recipient; it cannot check what is
said about him, so this is the one place where an invented sentence would reach a
real reader unchallenged. It has already happened twice — *"I have spent two
years building software"* and a claim about how long another site had been
documented — both fluent, both false, neither catchable.

So any first-person sentence about his history, study, practice or habits cites
an ID here or does not go in. If a draft needs a biographical fact that is not
listed, the answer is to ask him and add it with the date he said it, not to
write it and hope.

---

## The limits — cite these, do not hide them

To an engineer or an academic these are the most credible sentences available.
An unqualified claim from a stranger reads as marketing; the same claim with its
boundary stated reads as someone who has done the work.

TF-70  `shipped`  It is not a replacement for detailed hydraulic modelling. It
       informs early-stage judgement with real spatial data, before committing
       to a full model.

TF-71  `shipped`  **One catchment has been validated. The tool is not
       "validated".** Never drop the qualifier, never write "validated against
       CAMELS-NZ" without "on one catchment so far", and never imply a
       benchmarking programme that does not exist. This is the difference
       between a claim that earns a reply from a hydrologist and one that ends
       the conversation.

TF-72  `partial`  Installation is manual — copy the folder into the QGIS plugins
       directory. QGIS Plugin Manager support is not done.

TF-73  `partial`  Full usage documentation is not written.

TF-74  `partial`  Output accuracy refinement and edge-case handling are open
       items on the roadmap.

---

## Not facts

Things that are true and need no ID: that he built it, that he would like their
opinion, that he is offering half an hour, that he would rather hear what is
missing now than after release. Cite an ID for what the software does, not for
what he wants.
