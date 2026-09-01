# Bracket 5 — Swale and NFM academics

**~7 contacts, tracker Tier 5.** Named individual researchers: swale hydrology,
swale and stormwater design, natural flood management, flood risk and data
science. UK, USA, Sweden.

**This is the bracket the validation work was done for.** Everywhere else
CAMELS-NZ is supporting evidence; here it is the reason to write.

## Who
Individual academics who have published specifically on swale performance,
vegetated conveyance, or natural flood management. Not "academics" generally —
people with a named research output on the exact structure this tool designs.

Six of the seven have no address in the tracker and are resolved from published
faculty pages. That is `route: lookup`.

## What they measure
Method, and whether they could reproduce it. What the tool was compared against
and how closely. Whether the author has read what already exists.

## What they distrust
Claims of novelty. Software with no published basis. A tool that says "validated"
when it means "we ran it once". Being approached as an endorsement source.

## Facts that prove it, in order
1. `TF-30`, `TF-31` **open here.** Tested against CAMELS-NZ, a dataset published
   to validate hydrological models against real measurements; on a catchment with
   34 years of measured flow the delineated area and predicted runoff came within
   a couple of percent of the published figures.
2. `TF-33` immediately, in the same breath. One catchment so far, more before
   release. **This bracket will respect the qualifier far more than the claim.**
3. `TF-32` the WhiteboxTools cross-check, almost cell for cell.
4. `TF-40` D-infinity, `TF-41` SCS Curve Number — standard methods, named.
5. `TF-02` AGPL, source public, so they can check any of it.
6. `TF-74` the open roadmap items. To a researcher an honest gap is an invitation.

## Opener mechanism — the method statement
No hook. State what it does, what it was tested against, and link the repo. This
is the one bracket where an ordinary opener is worse than none.

> I have built a QGIS plugin for designing water-harvesting earthworks, and
> tested it against CAMELS-NZ on a catchment with 34 years of measured flow —
> one catchment so far.

The tailored opener should reference their specific published work on swales or
NFM, cited `web` from their faculty or group page. This bracket notices whether
the citation is real.

## Limits to state
`TF-71` above all, and `TF-74`. The single most useful sentence available to this
bracket is an honest statement of what has not been tested yet — it is also the
one most likely to produce a reply, because it gives them something to answer.

## The ask
Half an hour is on the table, but the lower-friction version is real here: read
the method and tell me what is wrong with it. Two families — `call` plus `react`.
Do not add `contribute` on a first email to someone who has never heard of this.

## Never say
- "Novel", "new method", "first tool to". It composes standard methods, which is
  the honest and the better claim.
- "Validated" unqualified. `TF-71`. This is the bracket that will notice first.
- Anything overselling `TF-51` coverage. It is the pure-Python layer. Say so.
- That their published work inspired the tool, unless it actually did.

## Signals
```yaml
strong:
  - swale
  - vegetated swale
  - natural flood management
  - NFM
  - stormwater control measure
  - bioretention
  - flood risk
  - catchment hydrology
  - urban drainage
  - runoff modelling
weak:
  - university
  - lecturer
  - professor
  - publications
  - doctoral
  - research group
```
