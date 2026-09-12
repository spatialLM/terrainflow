"""
test_architecture.py — the layering rules in CLAUDE.md, as executable checks.

Every rule here is one CLAUDE.md states in prose. Prose is not enforceable: each
of these was broken (or one rename away from being broken) at the time the file
was written, and none of it showed up as a test failure. Cheap deterministic
checks catch the *rule-shaped* mistakes so review attention is left for the
judgement-shaped ones.

Source is **parsed, never imported** — no QGIS mock is involved, nothing here
depends on ``conftest``, and the checks run in milliseconds. Adding a rule means
adding a function; there is deliberately no shared abstraction to learn.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "terrainflow_assessment"
TESTS_QGIS = REPO / "tests_qgis"
CLAUDE_MD = REPO / "CLAUDE.md"
MODULES = PKG / "modules"
CONTROLLERS = PKG / "qgis" / "controllers"
STATE_PY = CONTROLLERS / "_state.py"
LIFECYCLE_PY = PKG / "qgis" / "workers" / "_lifecycle.py"


def _py_files(root):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _import_time_nodes(tree):
    """Every node that executes when the module is imported.

    Recurses through ``if``/``try``/class bodies — all of which run at import —
    but stops at function boundaries, because a deferred import inside a function
    is the sanctioned way to reach an optional dependency.
    """
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield node
        yield from _import_time_nodes(node)


def _import_time_modules(tree):
    """``(module_name, lineno)`` for each name imported at import time."""
    for node in _import_time_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            yield base, node.lineno
            # `from terrainflow_assessment import panel` — the module being
            # imported is the *name*, not node.module.
            for alias in node.names:
                yield (f"{base}.{alias.name}" if base else alias.name), node.lineno


# --------------------------------------------------------------- 1. modules/ purity

def test_modules_do_not_import_qgis_at_import_time():
    """``modules/`` is the pure core — that is what makes ``tests/`` meaningful.

    A deferred import inside a function is fine and is used deliberately (see
    ``EarthworkDesign.from_dict``, which defaults to ``QgsGeometry.fromWkt`` only
    when the caller supplies no geometry factory). A *module-level* import would
    make the whole file unimportable without QGIS, taking the pure tests with it.
    """
    offenders = []
    for path in _py_files(MODULES):
        for name, lineno in _import_time_modules(_parse(path)):
            if name == "qgis" or name.startswith("qgis."):
                offenders.append(f"{path.relative_to(PKG)}:{lineno} imports {name}")
    assert not offenders, (
        "modules/ must import qgis only inside a function:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------- 2. layers land in the layer tree

def test_controllers_do_not_call_add_map_layer():
    """"Never call ``addMapLayer()`` from a controller" — CLAUDE.md.

    ``_groups.py`` owns registration so every layer is filed under the site's
    stage group and added collapsed. A controller that calls ``addMapLayer``
    itself either drops the layer loose at the top of the legend, or — worse,
    because it looks correct — reimplements the grouping inline and then drifts
    from it.
    """
    offenders = []
    for path in _py_files(CONTROLLERS):
        if path.name == "_groups.py":
            continue                       # the one file allowed to say it
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "addMapLayer" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(PKG)}:{i}: {line.strip()}")
    assert not offenders, (
        "controllers must place layers via self.place() / _groups.py:\n  "
        + "\n  ".join(offenders)
    )


def test_controllers_do_not_import_the_panel():
    """Controllers receive the panel by injection; they must not import it.

    The dependency runs panel → plugin → controller. An import the other way
    closes the loop and makes the controllers unloadable without the UI.
    """
    offenders = []
    for path in _py_files(CONTROLLERS):
        for name, lineno in _import_time_modules(_parse(path)):
            if name == "panel" or name.endswith(".panel"):
                offenders.append(f"{path.relative_to(PKG)}:{lineno} imports {name}")
    assert not offenders, (
        "controllers must take the panel as a constructor argument:\n  "
        + "\n  ".join(offenders)
    )


# ------------------------------------------------------------ 3. PluginState fields

def _state_fields():
    """``[(name, annotation_source, lineno)]`` for PluginState's declared fields."""
    tree = _parse(STATE_PY)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "PluginState")
    return [(n.target.id, ast.unparse(n.annotation), n.lineno)
            for n in cls.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]


def test_plugin_state_declares_each_field_once():
    """A field declared twice is silently the second one — and reads as two.

    ``stress_points_layer_id`` was declared on two lines with different comments,
    which is harmless to the dataclass and misleading to anyone reading it.
    """
    seen, duplicates = {}, []
    for name, _annotation, lineno in _state_fields():
        if name in seen:
            duplicates.append(f"{name} (lines {seen[name]} and {lineno})")
        seen[name] = lineno
    assert not duplicates, "PluginState declares a field twice:\n  " + "\n  ".join(duplicates)


def test_plugin_state_holds_layer_ids_not_layers():
    """"Never store a raw layer object on ``_state``" — CLAUDE.md.

    A stored layer outlives the layer: once QGIS deletes it, the attribute is a
    dead "wrapped C/C++ object" that raises on touch. An id resolves to ``None``
    through ``_layers.resolve_layer`` instead.

    Enforced on the *name*, because every field on the dataclass is typed
    ``Any``: a field mentioning a layer must be an id, and must be annotated as
    one, so the convention cannot be honoured in name only.
    """
    bad_name, bad_type = [], []
    for name, annotation, lineno in _state_fields():
        if "layer" not in name:
            continue
        where = f"{STATE_PY.name}:{lineno} {name}: {annotation}"
        if name.endswith("_layer_id"):
            if annotation != "str | None":
                bad_type.append(f"{where} — expected `str | None`")
        elif name.endswith("_layer_ids"):
            if annotation not in ("list[str]", "dict"):
                bad_type.append(f"{where} — expected `list[str]` or `dict`")
        else:
            bad_name.append(where)
    assert not bad_name, (
        "PluginState fields naming a layer must end _layer_id / _layer_ids:\n  "
        + "\n  ".join(bad_name)
    )
    assert not bad_type, "PluginState layer fields must be typed as ids:\n  " + "\n  ".join(bad_type)


def test_every_worker_slot_is_joined_on_unload():
    """``WORKER_SLOTS`` must name every ``PluginState`` field that holds a thread.

    ``join_workers`` walks that tuple and nothing else, so a slot missing from it
    is a thread ``unload`` never waits for: the scratch directory is removed under
    a worker still writing into it, and the ``QThread`` loses its last reference —
    "Destroyed while thread is still running", then abort. The constant's own
    comment names that failure, and ``terrain_worker`` was missing from it anyway.

    Enforced on the field *name*, which is the convention the slots already follow:
    a field ending ``_worker`` holds a worker, so declaring one is enough to be
    joined and adding a controller cannot silently skip the step.
    """
    tree = _parse(LIFECYCLE_PY)
    slots = next(
        (ast.literal_eval(n.value) for n in tree.body
         if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "WORKER_SLOTS" for t in n.targets)),
        None,
    )
    assert slots is not None, f"WORKER_SLOTS is not a literal assignment in {LIFECYCLE_PY.name}"

    declared = {name for name, _annotation, _lineno in _state_fields()
                if name.endswith("_worker")}
    missing = sorted(declared - set(slots))
    unknown = sorted(set(slots) - declared)
    assert not missing, (
        "PluginState declares a worker slot that unload never joins:\n  "
        + "\n  ".join(missing)
    )
    assert not unknown, (
        "WORKER_SLOTS names a field PluginState does not declare:\n  "
        + "\n  ".join(unknown)
    )


# ------------------------------------------------------------------- 4. tooltip copy

def _tooltip_literals(tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setToolTip"
                and node.args):
            continue
        arg = node.args[0]
        if isinstance(arg, ast.JoinedStr) or (
                isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            yield node.lineno


def test_help_text_declares_each_constant_once():
    """A constant assigned twice is silently the second one.

    ``AREA_SUBTOTALS`` was declared in two sections a hundred lines apart. The
    first was a superseded draft that nothing could reach, so editing it — the
    obvious thing to do when the wording needs a change — would have had no
    effect on the tooltip at all.
    """
    tree = _parse(PKG / "qgis" / "help_text.py")
    seen, duplicates = {}, []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in seen:
                duplicates.append(f"{name} (lines {seen[name]} and {node.lineno})")
            seen[name] = node.lineno
    assert not duplicates, "help_text.py declares a constant twice:\n  " + "\n  ".join(duplicates)


@pytest.mark.parametrize("path", _py_files(PKG), ids=lambda p: p.name)
def test_tooltip_copy_lives_in_help_text(path):
    """Tooltip copy is centralised in ``help_text.py`` as UPPER_SNAKE constants.

    Parametrised per file so a new offender names itself. Computed copy is fine
    — ``setToolTip(_capacity_tooltip(node))`` builds a sentence from live figures
    and has no fixed string to centralise; only a literal is a copy decision made
    in the wrong file.
    """
    offenders = [f"{path.relative_to(PKG)}:{lineno}" for lineno in _tooltip_literals(_parse(path))]
    assert not offenders, (
        "use setToolTip(H.NAME) with the copy in qgis/help_text.py:\n  "
        + "\n  ".join(offenders)
    )


# ------------------------------------------------------------------ 5. what ships

def test_powershell_scripts_are_ascii_and_bom_less():
    """PowerShell 5.1 reads a ``.ps1`` as ANSI, not UTF-8.

    A UTF-8 em dash arrives as mojibake and the script dies with a parse error
    naming neither the character nor the encoding — and ``deploy.ps1`` is the
    only route the plugin has into QGIS, so the failure blocks every manual test.
    A BOM is the same class of problem.

    ``.sh`` is deliberately not covered: bash reads UTF-8 correctly, so the
    hazard is specific to the Windows ANSI codepage.
    """
    offenders = []
    for path in sorted(REPO.rglob("*.ps1")):
        if ".git" in path.parts:
            continue
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            offenders.append(f"{path.name}: starts with a UTF-8 BOM")
        for offset, byte in enumerate(raw):
            if byte > 0x7F:
                line = raw[:offset].count(b"\n") + 1
                # ascii() rather than repr(): this message is most likely to be
                # read on the cp1252 console the rule exists for, and a message
                # that cannot itself be printed helps nobody.
                char = ascii(raw[offset:offset + 4].decode("utf-8", "replace")[0])
                offenders.append(f"{path.name}:{line}: non-ASCII {char} (0x{byte:02x})")
                break
    assert not offenders, "PowerShell 5.1 cannot parse these:\n  " + "\n  ".join(offenders)


def test_test_code_stays_out_of_the_deployed_package():
    """Only ``terrainflow_assessment/`` is deployed or zipped.

    ``tests_qgis/`` boots a real QgsApplication and drives the panel through
    synthetic mouse events. Inside the package it would ship to users, and
    ``deploy.ps1`` would copy it into the QGIS profile.
    """
    strays = sorted(p.relative_to(PKG)
                    for pattern in ("test_*.py", "checks_*.py", "conftest.py")
                    for p in PKG.rglob(pattern) if "__pycache__" not in p.parts)
    assert not strays, (
        "test code must live beside the package, not inside it:\n  "
        + "\n  ".join(str(p) for p in strays)
    )


def test_the_package_is_not_nested_inside_itself():
    """``terrainflow_assessment/terrainflow_assessment/`` is a packaging accident.

    ``.gitignore`` guards it, which stops it being *committed* — not created, and
    not deployed. ``deploy.ps1`` copies the working tree, so an ignored nested
    copy reaches QGIS and shadows the real one.
    """
    nested = PKG / PKG.name
    assert not nested.exists(), f"nested package copy at {nested.relative_to(REPO)}"


# ------------------------------------------------- 6. the QGIS scoping table stays true

def _checks_modules_on_disk():
    return {p.stem for p in TESTS_QGIS.glob("checks_*.py")}


def _checks_modules_named_in_claude_md():
    # `checks_*.py` in prose is a glob, not a module, and does not match: the character
    # class stops at the `*`. Matching the bare stem anywhere in the file rather than
    # parsing the markdown table means reformatting the table cannot break this test.
    return set(re.findall(r"checks_[a-z_]+", CLAUDE_MD.read_text(encoding="utf-8")))


def test_claude_md_scoping_table_names_every_checks_module():
    """CLAUDE.md's "Touching -> Run" table is how a scoped QGIS run gets chosen.

    A module missing from it is invisible: the table is the only thing saying which
    ~30 s run covers a given source file, so an unlisted module is one nobody ever
    runs on purpose, and its checks are paid for only in the 8-minute full run.
    """
    missing = sorted(_checks_modules_on_disk() - _checks_modules_named_in_claude_md())
    assert not missing, (
        "these tests_qgis/checks_*.py modules have no row in CLAUDE.md's scoping "
        "table:\n  " + "\n  ".join(missing)
    )


def test_claude_md_scoping_table_names_no_module_that_is_gone():
    """The other direction: a rename leaves the old name behind, pointing nowhere.

    A row naming a module that no longer exists is worse than a missing row — it reads
    as a working command and selects nothing, and "No checks matched." is easy to skim
    past as a pass.
    """
    phantom = sorted(_checks_modules_named_in_claude_md() - _checks_modules_on_disk())
    assert not phantom, (
        "CLAUDE.md names these checks modules, but they are not on disk:\n  "
        + "\n  ".join(phantom)
    )


# ------------------------------------------------- help copy reaches a surface

HELP_TEXT_PY = PKG / "qgis" / "help_text.py"

#: Copy that is deliberately not attached to one widget. Each entry says why, and
#: the point of the list is that adding to it is a decision rather than a drift.
_HELP_NOT_ATTACHED = {
    # Formatted at the call site with a label, not used as a constant by name.
    "TOOL_DRAW_FALLBACK",
}


def _help_constants():
    """Every UPPER_SNAKE string constant defined in ``help_text.py``."""
    tree = _parse(HELP_TEXT_PY)
    names = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and re.fullmatch(r"[A-Z][A-Z0-9_]*",
                                                             target.id):
                names.add(target.id)
    return names


def _names_referenced_outside_help_text():
    """Every identifier used anywhere in the package except ``help_text.py`` itself.

    Read off the source text rather than the AST because the constants are reached
    as ``H.NAME`` attributes, and an attribute access carries no binding to resolve.
    """
    used = set()
    word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    for path in _py_files(PKG):
        if path == HELP_TEXT_PY:
            continue
        used.update(word.findall(path.read_text(encoding="utf-8")))
    return used


def test_every_help_text_constant_reaches_a_widget():
    """Q-19. Tooltip copy nobody can read is worse than none.

    Seven constants here were orphans, and three of them described features that had
    shipped with no tooltip at all: the Stress points layer, the Overflow connections
    layer and the channel-length figure in the intensity dialog. The copy was written,
    reviewed and then simply never attached, and nothing said so.

    Worse, an orphan can be *more correct than the one on screen*. ``OVERFLOW_TARGET_
    AUTO`` described the flow-path walk that ``resolve_targets`` actually does, while
    the ``OVERFLOW_TARGET`` tooltip the combo box shows still described the elevation
    heuristic that walk replaced. Two versions of the same sentence, one of them wired
    and wrong.
    """
    defined = _help_constants()
    used = _names_referenced_outside_help_text()
    orphans = sorted(defined - used - _HELP_NOT_ATTACHED)
    assert not orphans, (
        "these help_text constants are referenced nowhere in the package — attach "
        "them to the widget they describe, or delete them:\n  "
        + "\n  ".join(orphans)
    )


# ------------------------------------------------- T-3: rules that had no check

CORE = PKG / "core"
MAP_TOOLS = PKG / "map_tools"
EARTHWORK_DESIGN_PY = MODULES / "earthwork_design.py"
SYMBOLS_PY = CONTROLLERS / "_symbols.py"


def test_core_does_not_import_qgis_at_import_time():
    """The same rule as ``modules/``, over the tier the scan was not walking.

    ``core/`` is the sizing and registry layer and is imported by ``modules/``, so a
    module-level ``import qgis`` there makes the pure tier unimportable exactly as it
    would in ``modules/`` — and the scan looked only at ``modules/``, so it would
    have found nothing to say about it.
    """
    offenders = []
    for path in _py_files(CORE):
        for name, lineno in _import_time_modules(_parse(path)):
            if name == "qgis" or name.startswith("qgis."):
                offenders.append(f"{path.relative_to(PKG)}:{lineno} imports {name}")
    assert not offenders, (
        "core/ must import qgis only inside a function:\n  " + "\n  ".join(offenders)
    )


def test_only_the_shared_ramp_helper_applies_a_raster_ramp():
    """"Never call ``apply_raster_ramp`` directly" — CLAUDE.md.

    A Baseline layer and its Earthworks counterpart have to be scaled against each
    other or a colour means a different depth on each, and the before/after toggle
    compares two different pictures. ``apply_shared_ramp`` is what holds the family
    top; a direct call scales a layer against itself.

    Q-10 was the live violation this would have caught — the simulation frame's
    stream layer, scaled alone.

    Two files may call it. ``_symbols.py`` defines it and ``apply_shared_ramp`` is
    built on it; ``terrain.py``'s layers have no Baseline/Earthworks counterpart to
    share a top with, so there is no family for them to join.
    """
    allowed = {SYMBOLS_PY.resolve(), (CONTROLLERS / "terrain.py").resolve()}
    offenders = []
    for path in _py_files(PKG):
        if path.resolve() in allowed:
            continue
        for node in ast.walk(_parse(path)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "apply_raster_ramp"):
                offenders.append(f"{path.relative_to(PKG)}:{node.lineno}")
    assert not offenders, (
        "these call apply_raster_ramp directly instead of apply_shared_ramp, so "
        "their layer is scaled against itself:\n  " + "\n  ".join(offenders)
    )


#: ``_rasterize`` calls that may leave ``all_touched`` to the default, by the method
#: they sit in. Both are questions about *position*, not about how much earth moved.
_ALL_TOUCHED_EXEMPT = {
    # Which side of the line is downhill. A band one cell wider on both sides
    # changes no volume — it is a comparison of two means.
    "_downstream_footprint",
}


def test_every_rasterize_call_states_all_touched():
    """M-1. ``_rasterize`` defaults to ``all_touched=True``, and for a volume that
    is wrong: it claims every cell the geometry *touches*, so a 2.0 m diversion on a
    1 m DEM cuts a band about 1.3 m wider than drawn and the measured cut — the
    column the report tells a contractor to price the job on — over-reads by tens of
    percent.

    The module's own docstring said so while three burns still took the default.
    A default nobody can see is not a decision, so every call has to make one; the
    exemptions are listed above with the reason each is a position question rather
    than a volume one.
    """
    tree = _parse(EARTHWORK_DESIGN_PY)
    enclosing = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                enclosing.setdefault(id(node), fn.name)

    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_rasterize"):
            continue
        owner = enclosing.get(id(node), "<module>")
        if owner in _ALL_TOUCHED_EXEMPT:
            continue
        if not any(kw.arg == "all_touched" for kw in node.keywords):
            offenders.append(f"earthwork_design.py:{node.lineno} in {owner}()")
    assert not offenders, (
        "these _rasterize calls take the all_touched default, so whether they claim "
        "a band wider than the section drawn is invisible at the call site:\n  "
        + "\n  ".join(offenders)
    )


def _feature_ish(node):
    """Is *node* a ``<something>.name`` attribute read?"""
    return isinstance(node, ast.Attribute) and node.attr == "name"


def test_no_per_feature_dict_is_keyed_on_a_bare_name():
    """Q-7 / M-3. Two features can carry the same name, and routinely do.

    The default name counter reproduces a deleted feature's name — draw "Swale 3",
    delete it, draw another, and there are two. A dict keyed on the name then has
    one entry where there should be two, and which of the pair survives is whichever
    was written last: one feature's fill timeline drawn from the other's data, and a
    freeboard advisory reporting the wrong feature's spillway.

    The fix everywhere was ``key = ew.id or ew.name`` — a name is a *label*, and the
    id is the identity. That is still allowed, and is what this permits: the key may
    *fall back* to a name, it may not *be* one.
    """
    offenders = []
    for root in (MODULES, CONTROLLERS):
        for path in _py_files(root):
            tree = _parse(path)
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for t in targets:
                    if isinstance(t, ast.Subscript) and _feature_ish(t.slice):
                        offenders.append(
                            f"{path.relative_to(PKG)}:{node.lineno}: "
                            f"{ast.unparse(t)} = ...")
                if isinstance(node, ast.DictComp) and _feature_ish(node.key):
                    offenders.append(
                        f"{path.relative_to(PKG)}:{node.lineno}: "
                        f"{{{ast.unparse(node.key)}: ...}}")
    assert not offenders, (
        "these key a dict on a feature's name, which two features can share — use "
        "`x.id or x.name`:\n  " + "\n  ".join(offenders)
    )


# ------------------------------------------------- counts that move every commit

#: Where a stale count was found, and where one would be read as authoritative.
_COUNT_QUOTING_FILES = (
    "CLAUDE.md",
    "tests_qgis/README.md",
    "tests_qgis/_harness.py",
    "tests_qgis/checks_robustness.py",
)

#: "2,590 tests", "200 checks", "the 259 checks". Durations are deliberately not
#: matched: they are what a reader plans around and they move slowly.
_COUNT_QUOTE = re.compile(r"\b\d[\d,]*\s+(?:tests|checks)\b", re.IGNORECASE)


def test_the_docs_do_not_quote_a_test_or_check_count():
    """H-3 / H-4 / H-9. Every count these files quoted was wrong, in both directions.

    CLAUDE.md said "~2,590 tests" against 2,974 and "200 checks" against 306;
    ``_harness.py`` and ``checks_robustness.py`` both said "the 259 checks"; the
    tests_qgis README described a ``checks_slow.py`` that no longer exists. None of
    it was load-bearing and all of it was quietly false, because a figure that
    changes on most commits is one nobody re-measures.

    So the rule is: **do not quote one.** "It takes under a minute" and "8-10 min"
    are what a reader actually needs, and they stay true. If a count ever genuinely
    earns its place, derive it at read time rather than typing it.
    """
    offenders = []
    for rel in _COUNT_QUOTING_FILES:
        path = REPO / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for hit in _COUNT_QUOTE.finditer(line):
                offenders.append(f"{rel}:{i}: {hit.group(0)!r} in {line.strip()[:70]}")
    assert not offenders, (
        "these quote a test/check count, which goes stale on the next commit and "
        "is then read as authoritative:\n  " + "\n  ".join(offenders)
    )
