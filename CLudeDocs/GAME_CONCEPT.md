# Earthworks game — concept outline

*Not part of the plugin. A record of the design conversation of 4 August 2026, written up
21 August 2026 so the thinking isn't only in a chat transcript. Nothing here has been
built or committed to; the plugin ships first.*

---

## 1. The pitch

A level presents a landscape and a small homestead to protect. The player has a budget of
earthworks — swales, dams, diversion drains, basins — measured in **m³ of earth moved**.
They place what they can afford, press **Start rainfall**, and watch.

If the design is good, the water is caught and the homestead survives. If it isn't, swale
beds burst, dam walls breach, and the surge takes the homestead. Retry until it holds.

Poly Bridge's loop — constrained build → simulate → spectacular failure → retry — with
hydrology in place of statics. m³ of earth moved is a good budget currency because it is
the *real* constraint in earthworks: digger hours are money.

## 2. What makes it not a Poly Bridge reskin

**Legible failure.** Most physics-puzzle games fail you against arbitrary rules. This one
fails for the actual reasons dams and swales fail — undersized spillway, too much
catchment, wall too low. So after the flood you can show *why*:

> "This swale received 3× its capacity. Its catchment is the whole ridge above it."

That is the plugin's analysis output re-skinned as a post-mortem screen, and it converts
retry-frustration into learning. It is the single strongest idea in the concept.

**Real NZ terrain as levels.** The DEM pipeline already exists. A distinctive hook nobody
else has.

**The storm replay** — the homestead surviving by centimetres as the dam holds — is the
shareable moment. That is what actually sold Poly Bridge.

## 3. Division of labour: truth vs spectacle

The one architectural decision the conversation settled, and the one worth keeping.

| | Owns | Character |
|---|---|---|
| **Game engine** (new work) | Flowing water, erosion, buildings breaking, camera | Fast, fudged, juicy. Only has to *look* right. |
| **Hydrology model** (exists) | Catchment areas, how much water each earthwork receives, storage vs inflow, freeboard and spillway capacity | Decides *whether and where* failure triggers. The engine just performs it. |

The plugin computes steady-state hydrology — it answers "will it fail?" but not "show me
the failure." The washing-away homestead, the bursting swale, the breach propagating
downstream is a real-time heightfield fluid sim, and it is new work. The good news is that
it's a solved problem in games: the **virtual-pipes heightfield method** (GPU Gems 2) runs
fast on GPU, and both Timberborn and From Dust do it.

## 4. Why 3D, and which kind

2D fails this game specifically:

- **The core skill is reading terrain** — which ridge sheds water where, what catchment a
  swale intercepts. A side-on cross-section destroys exactly that: a dam in section is
  just a wall, and its whole meaning is the catchment behind it.
- **Top-down 2D is the QGIS plugin.** It works analytically but has no visceral impact,
  and it asks a casual player to read contour maps.
- **The emotional payoff needs the third dimension.** In top-down 2D a breach is "the blue
  polygon got bigger."

The reassuring part: this is the easiest kind of 3D there is. The world is a heightfield —
a grid of elevations, which is the DEM data already in hand — plus a water surface and a
handful of low-poly props. No character animation, no interiors, no complex collision, no
asset streaming. One terrain mesh, one water shader, an orbiting camera.

**Worth stealing from Timberborn:** a top-down map mode for the design phase, where
contour thinking happens naturally, then the camera drops to ground level for the storm.
*Design in plan view, fail in cinematic view.*

## 5. Art direction — Bad North

Chosen because its beauty comes from restraint rather than rendering tech.

| Ingredient | How it's made |
|---|---|
| Flat matte low-poly, no textures | Vertex colours or a tiny palette texture; stock material |
| Diorama framing — an island in a void | Orthographic or narrow-FOV camera + a backdrop colour. No skybox, no horizon, no distant terrain. *Reduces* the workload. |
| Soft pastel palette, gentle lighting | One directional light, soft shadows, ambient, maybe subtle SSAO |
| Clean silhouettes | Art-direction discipline, not a feature — this is where the difficulty actually lives |

Why it suits this game in particular:

- **The diorama IS the level.** A self-contained chunk of hillside with a homestead — one
  catchment, edges falling away to nothing — maps exactly onto the island-in-a-void
  framing. You never have to answer "what's beyond the level boundary?"
- **The terrain is the hero asset**, and the levels are literally landforms.
- **Stylised water is easier than realistic water** — a depth-to-colour ramp (shallow pale,
  deep saturated) with foam at edges and flow fronts, sitting on top of the heightfield sim.
- **No texturing pipeline**, so per-level art cost is sculpt a terrain and place props —
  sustainable for one person across 40+ levels.

The honest caveat: minimalism is unforgiving of taste errors. With six colours and a
landform on screen, an off palette or a mushy silhouette is glaring. Lock a palette, test
in-engine lighting from week one. Cheap to render, expensive to curate.

## 6. Engine — Godot

**For:**
- GDScript is near-Python; productive in days, and the ported model reads almost
  line-for-line
- Free and open source, no revenue cut, no terms that can change underneath
- Low-poly cartoony needs nothing Unity has that Godot lacks
- Compute shaders supported; virtual-pipes has open-source Godot demos to crib from
- `.tscn`/`.gd` are plain text — clean git diffs, and Claude Code can read and edit the
  whole project with ordinary file tools, no MCP required for most work

**Against:**
- You write the water sim yourself — weeks of shader work, the hardest component
- Smaller ecosystem, fewer answers when stuck
- Console ports need third-party outfits (ignorable until Steam success demands it)

**The Unity argument, honestly:** its Asset Store has ready-made heightfield fluid/erosion
packages (~$50–100) that buy out the single biggest technical risk, plus a proven path in
this exact genre (Poly Bridge and Timberborn are both Unity). Take it only if the water
shader intimidates enough to be worth learning C#.

**Track record check:** Godot's portfolio is now real — Slay the Spire 2 (ported off Unity
mid-development, ~575k peak concurrent), Brotato, Dome Keeper, Backpack Battles, Cassette
Beasts. It skews 2D; the 3D wins are Buckshot Roulette and Cruelty Squad. There is no
3D physics-puzzle hit of Poly Bridge's scale in Godot *yet*, but nothing in this scope
pushes Godot's known 3D limits.

## 7. What the plugin actually contributes

**The Python does not ship in the game.** Neither engine embeds Python well, and
rasterio/pysheds won't go in a game binary. Instead:

- **Python becomes the level-authoring pipeline** — pull real NZ DEMs, precompute flow
  networks and catchments, validate that a level is solvable, export a heightmap plus
  metadata per level. Runs on the developer's machine, not the player's.
- **Port the small runtime pieces** — D8 flow accumulation on a 256² grid and the
  sizing/failure thresholds are a few hundred lines, trivially fast in GDScript. The
  tested `modules/` become the **reference implementation** to verify the port against.

So the engine choice is genuinely free of the existing codebase — pick on ergonomics.

The deeper contribution isn't code, it's that the model is *validated*. Things like
`peak_runoff_fraction()` returning marginal dQ/dP rather than Q/P take a domain expert to
know, and silently halve a spillway when wrong. That has already been paid for.

## 8. Where Claude helps, and where it can't

Roughly **70% of the project by effort** sits in the strengths: the virtual-pipes solver's
numerical core (it is TerrainFlow's maths in different clothes), the D8 port, the
failure/sizing logic, the DEM level pipeline, all UI, and the offline Python tooling.

Human-in-the-loop, ranked:

1. **Art direction and taste — the real gap.** Whether a palette feels right or a
   silhouette reads across the room. Bad North's style makes this tractable but
   unforgiving; the project succeeds or fails on the developer's eye.
2. **Asset creation.** Terrain comes procedurally from DEMs; Blender Python can generate
   simple props; KayKit/Kenney packs cover prototyping. Beyond that, either learn basic
   low-poly modelling (days, not months) or commission the ~20 props the style needs.
3. **Game feel.** Camera moves, storm pacing, how a breach *lands*. Written by Claude,
   judged by playing it.
4. **Shader debugging.** Visual bugs need screenshots — which is the loop `tests_qgis/`
   already runs here: deterministic render, snapshot diff, look at the image.

**Scale, for calibration:** a shippable demo (5–10 levels, working storm sim, the failure
drama) is a few months part-time. A full Steam release — 40+ levels, sound, polish — is a
year-class project.

## 9. Sequencing

After TerrainFlow ships. Three things carry over: the tested hydrology modules as a
reference implementation, the CLAUDE.md-plus-layered-architecture working style (which
transfers verbatim — scenes/sim/UI instead of controllers/modules/panel), and the
visual-verification harness habit. Finishing the plugin first also pressure-tests the
maths the game inherits.

## 10. Open questions

- Does the water ledger create *interesting* decisions, or does one obvious swale
  placement dominate every level? Untested.
- How does a level communicate its budget and its target — "capture 80% of a 1-in-20
  event" — without becoming a spreadsheet?
- Should failure be binary (homestead lost) or graded (crops watered, % captured)?
- How much of real hydrology survives contact with fun, and where does the model have to
  lie to be readable?
