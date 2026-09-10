# User-set standard earthwork dimensions — implementation record

*Built 2026-09-10. Plan correlated from three independently developed plans, reviewed
cold, then reshaped once by the user in review — see §1. Section 8 records what was
excluded and why.*

---

## Context

A landowner's earthwork size is dictated by the implement they own — a tractor-mounted
trough that digs to a fixed depth and width, a technique repeated on every contour. Every
drawn feature used to be seeded from shipped constants in `core/registry/earthwork_types.py`
(a swale started 0.50 m deep, 2.00 m top width), so the user retyped their own numbers for
every feature.

The plugin now **learns the standard from use**: a "Save as my standard size" toggle in the
earthwork properties dialog, on by default until a standard exists. Draw the first swale at
1.60 × 0.35 and that becomes the size every later swale starts at. No settings screen, no
setup step.

Not in scope: inverse sizing (STRETCH_GOALS §2). This is the *constraint* half — declaring
what the machine can cut.

---

## 1. The correction that reshaped the plan

All three source plans built a separate "Standard dimensions…" preferences dialog. In
review the user proposed the toggle instead, and it is better: it puts the question at the
moment the user is already deciding the size, and it removes an entire dialog file, a panel
button, a panel signal, a `plugin.py` wiring line, a `checks_visual` render check and a
coverage-placement decision.

The auto-off rule needed one refinement to work. A naive *"changing the dimensions unticks
it"* fires backwards on the very first swale — the user changes 2.0 → 1.6, which is exactly
when the value is worth keeping. The rule that behaves correctly is conditional on a
standard already existing (see §3).

A second finding also stands: the Analysis stage's **Find Best Swale Segments** criteria are
three editable spin boxes (`panel.py:1114-1152`) carrying exactly this triple, and the revert
comment at `qgis/controllers/earthworks.py` forbidding a wire between them was **stale** — it
argued from criteria defaults describing a 0.6 m sub-cell swale, and those defaults have since
been realigned to the registry. The criteria grid is now *seeded* from the standard, and the
comment rewritten to say why the old reasoning no longer applies.

---

## 2. The four decisions

1. **Capture with a toggle in the properties dialog, not a settings screen.**
   `earthwork_properties_dialog.py` is already in the coverage `omit` list by exact filename,
   so this costs nothing on the 95% gate — where a new dialog file would have needed careful
   placement to avoid sinking it.
2. **Store in `QgsSettings`, per user** — one JSON blob under `TerrainFlow/earthwork_defaults`.
   Project scope fails outright: `readEntry` returns nothing for a project never saved, which
   is exactly when the first swale is drawn. The `.tfd` fails too — `normalise_inputs` runs on
   open, so a colleague's design would overwrite the reader's own machine dimensions.
3. **Layer over the registry, never mutate it.** `_REGISTRY` stays the record of what ships.
   `tests/test_project_io.py:179-186` is the canary: it breaks the instant anyone reaches for
   `register_type`, and it passed untouched throughout.
4. **Deliver by constructor parameter, not an ambient global** — `Earthwork.__init__(…, *, dims=None)`,
   where `dims=None` reproduces the old path verbatim. This gives `from_dict` its guarantee
   structurally: **a preference can never retroactively resize a saved design.**

**Stored fields: depth, top width, bottom width — never side slope.** The same choice
`modules/project_io` made for the design file and `panel.py` for the swale criteria: three
measurements you can take with a tape at the machine, one consequence you cannot. It also
means the `max(0.1, …)` floor is never reached by a user's own numbers.

**Sparse at the type level, not the field level.** A type whose whole triple matches the
shipped one is dropped; any other keeps all three numbers. Per-field sparseness would be a
bug — storing only a changed depth lets the resolver re-derive a bottom width the user never
saw. Dropping a matching type is also how a standard is **cleared**: draw one at the shipped
size with the toggle on.

---

## 3. The toggle

Re-evaluated live on `valueChanged`, compared at 2 dp:

| State | Toggle |
|---|---|
| No standard stored for this type | **On** — whatever they OK becomes the standard |
| Values match the stored standard | On, inert (writing the same thing is a no-op) |
| Values differ from the stored standard | **Off** — they tick it to promote this size |

- **Create only.** `_editing` is already `True` on create (the controller passes a constructed
  `Earthwork`), so the dialog takes an explicit `is_new=True` — the existing flag cannot tell
  a fresh feature from an old one.
- **`clicked`, not `toggled`.** `clicked` fires for the user only, so an opinion they have
  expressed is not overwritten by the next keystroke in a spin box (`_save_standard_touched`).
- **The label carries the standard** — "Save as my standard size (now 1.60 m wide × 0.35 m
  deep)" — since there is no settings screen to read it from. Built from `settable_dims`, so a
  dam reads "2.00 m thick" and a basin only its depth.
- The success message ends *"Ones already drawn are unchanged."*

---

## 4. Architecture

**`core/registry/earthwork_defaults.py`** — pure, stdlib + `earthwork_types` only, 117
statements at 100% branch coverage.

```
DimensionDefaults (frozen)   # depth | top_width_m | bottom_width_m, each float|None
ResolvedDims (frozen)        # all three concrete
settable_dims · shipped_dims · resolve_dimensions · dims_match
sanitise · encode · decode
```

`shipped_dims` absorbs the `try/except KeyError` fallback that was duplicated in
`Earthwork.__init__` and the properties dialog, so the historical `0.5 / 2.0 / 1.0` triple
now exists in one place.

Flow, one way only:

```
QgsSettings ──► EarthworksController.load_earthwork_defaults()
                     │  decode()  (pure, sanitising)
                     ▼
              self._earthwork_defaults ──► _resolved_dims(ew_type)
                     ├──────────────► _provisional_catchment(top_width_m=…)
                     └──────────────► Earthwork(…, dims=…)
```

Resolving **once** and using it twice is what makes the catchment shown in the dialog and the
feature actually built agree by construction — they previously agreed only by coincidence,
both reading the same registry constant sixteen lines apart.

---

## 5. Files changed

**New:** `core/registry/earthwork_defaults.py`, `tests/core/test_earthwork_defaults.py`

**Modified:** `modules/earthwork_design.py` (constructor `dims=`, `from_dict` untouched),
`qgis/controllers/earthworks.py` (persistence pair, `_resolved_dims`,
`_remember_standard_dims`, `seed_swale_criteria_from_standard`, resolve-once in
`_on_geometry_drawn`, rewritten revert comment), `earthwork_properties_dialog.py` (toggle,
`is_new`/`standard_dims`, range widening), `panel.py` (`seed_swale_criteria`),
`qgis/plugin.py` (one call), `qgis/help_text.py` (`SAVE_AS_STANDARD`),
`tests/test_earthwork_design.py`, `tests_qgis/checks_earthworks.py`

**Not touched:** `core/registry/earthwork_types.py`, `modules/project_io.py`,
`pyproject.toml`, `SCHEMA_VERSION`, `_state.py`, `CLAUDE.md`

---

## 6. The traps, and how each was handled

| Trap | Handling |
|---|---|
| `from_dict` re-seeds before overwriting | Structural — no `dims` passed, so shipped values, as before. Pinned at both tiers. |
| `_editing` is `True` even on create | Explicit `is_new=True`; the existing flag is not reused. |
| `setRange` before `setValue` silently clamps | Widened to `setRange(min(lo,seed), max(hi,seed))` for depth, width and bottom width. Without this the dialog claws the user's own standard back inside the advisory range and the feature defeats itself. Also fixes the pre-existing case of a feature saved before a range was tightened being silently rewritten on OK. |
| Bottom width clamps at 0.1 | Neutralised by storing it explicitly. |
| Parallel defaults in `project_io` | Untouched; its test stayed green — the dividend of not mutating the registry. |
| **`QgsSettings` persists across checks in one process** | The harness isolates the org/app name but not check-to-check. Every new check restores the prior value in a `finally` via `_set_standard`. |
| **`test_tooltip_copy_lives_in_help_text`** is parametrised over every package `.py` | `H.SAVE_AS_STANDARD` is mandatory, not optional. |

**Known and deliberately not fixed:** `earthwork_properties_dialog.py` seeds the bottom width
through `min(seed_bottom, spin_width.value())`. A stored bottom wider than the top is silently
clamped there while the toggle path only warns. Pre-existing seeding heuristic; recorded so it
is not discovered as a surprise.

---

## 7. Verification

- `python -m pytest tests/` — **2747 passed**, 2 xfailed, 5 xpassed. **No existing test was
  edited**, which was the acceptance criterion: `dims=None` reproduces the old path exactly.
- `pytest --cov` on the new module — **117 statements, 36 branches, 100%**.
- `ruff check terrainflow_assessment/` — clean.
- `.\run_qgis_tests.ps1 checks_earthworks` — **68 passed, 0 failed**, including seven new
  checks: a drawn swale uses the standard; the dialog opens at it and unticks on departure;
  the toggle is on for a first feature; the round trip persists and reloads; a saved design
  ignores the reader's standard; the provisional catchment uses the same width; a basin
  ignores a width entry.
- Full `.\run_qgis_tests.ps1` before commit — required, since `core/registry/*`, `panel.py`
  and `qgis/plugin.py` are all on CLAUDE.md's "no scope" list.

**The acceptance test worth keeping:** parametrised over all five types,
`resolve_dimensions(cfg, None)` equals what the constructor has always produced — including
basin's `max(0.1, 0.0) = 0.1` and dam's 2.0. If that passes, a user who never sets a standard
has not had a single number moved.

---

## 8. Excluded, and why

**A separate "Standard dimensions…" preferences dialog** — replaced by the toggle (§1). The
cost is that there is no single screen listing all five standards; resetting is drawing one at
the shipped size with the toggle on, which the tooltip states.

**Module-global preference store in `core/`** *(plan A)* — needs a compensating re-seed inside
`from_dict`; the constructor parameter closes that structurally. The leaked `"terrace"` in
`tests/core/test_earthwork_types.py:56-71` is in-repo proof of what a mutable module global
costs.

**Three separate `depth=`/`top_width=`/`side_slope=` kwargs** *(plan B)* — one frozen
`ResolvedDims` travels as a unit, so the three cannot be passed inconsistently.

**Storing side slope** *(plan B)* — contradicts the reasoning already in `project_io.py` and
`panel.py` about not storing a derived value beside the numbers producing it, and reintroduces
the 0.1 m clamp.

**One settings key per field** *(plan B)* — no single "unset" state, no single degrade path;
abandons the one-JSON-blob precedent set by the IDF table.

**Backfilling `bottom_width_m` in `from_dict`** *(plan A)* — real bug: a pre-bottom-width
payload loads at the wrong batter. Fixing it changes how existing projects load, which is the
one thing this feature promised not to do. **Still open — file separately.**

**Per-type opt-in checkbox** *(plan C)* — moot; the toggle answers the same question at the
moment of use.

**Dropping out-of-advisory-range values** *(plan C)* — partially rejected. `sanitise` drops
non-finite, zero and negative; a value merely outside `depth_range` is kept and flagged.
Overriding a user's declared machine limit because it sits outside a guideline envelope would
defeat the feature, and contradicts *"Advisory only — never a silent clamp."*

**An architecture guard test for root-level dialogs** *(plan A)* — good idea, unmotivated once
no dialog is added.

**`PluginState` field** *(plan C)* — only the earthworks controller needs it.

**`gradient_pct`, `crest_elevation`, `batter_run_m`, `companion_berm`** — these describe where
the feature *sits* (site grade, absolute elevation), not what the implement *cuts*.
`settable_dims` is one function, so a v2 is cheap.

**Per-project standards, named machinery presets, a live wire to the criteria grid** — out of
scope. The resolver shape makes per-project a single extra lookup with no call-site change.
