"""p_coverage — what the suites do not touch, and what the tooltips promise.

Step E of the defect-documentation campaign. Two jobs, both of which produce rows for
`ANALYSIS_DEFECTS.md` §4 rather than findings:

**1. Coverage.** For each subject the campaign named as absent from both findings and
coverage, report whether either suite *calls* it — not merely imports or mentions it. The
distinction matters: a name appearing in an import line is how a function looks covered
while nothing has ever run it, which is how `landform_tpi` and `landform_classes` came to
be the corrected implementations with no production caller. `not exercised` is printed as
an honest third state, per the register's own convention.

**2. `help_text.py` against the constants it quotes.** 1,475 lines of user-facing claims
with no test of any kind. The cheap, sound check is not "parse every number in the prose"
— most of them are geography, not constants — but an explicit table of the places the copy
states a *default* or a *threshold* that exists as a named value elsewhere. Each pair is
asserted; anything not in the table is reported as unchecked so the gap is visible rather
than implied. A tooltip that has drifted from its constant is a lie told in the UI, and
this file has no other guard at all.

Run::

    $env:QT_QPA_PLATFORM = 'offscreen'
    & 'F:\\bin\\python-qgis-ltr.bat' tests_qgis\\probes\\p_coverage.py
"""

import re

import _probe

#: `(subject, module path)` for every name Step E listed. The module path is where the
#: definition lives, so "called" can be told from "named in an unrelated string".
SUBJECTS = [
    ("strahler_order", "terrainflow_assessment/modules/flow_graph.py"),
    ("stream_links", "terrainflow_assessment/modules/flow_graph.py"),
    ("topographic_wetness_index", "terrainflow_assessment/modules/terrain_indices.py"),
    ("stream_power_index", "terrainflow_assessment/modules/terrain_indices.py"),
    ("sediment_transport_index", "terrainflow_assessment/modules/terrain_indices.py"),
    ("landform_classes", "terrainflow_assessment/modules/terrain_indices.py"),
    ("landform_tpi", "terrainflow_assessment/modules/terrain_indices.py"),
    ("slope_statistics", "terrainflow_assessment/modules/terrain_indices.py"),
    ("specific_catchment_area", "terrainflow_assessment/modules/terrain_indices.py"),
    ("terrace_vertical_interval", "terrainflow_assessment/core/sizing/advisories.py"),
    ("bulking_factor", "terrainflow_assessment/core/sizing/advisories.py"),
    ("compaction_factor", "terrainflow_assessment/core/sizing/advisories.py"),
    ("capture_spacing", "terrainflow_assessment/core/sizing/advisories.py"),
    ("spacing_advisory", "terrainflow_assessment/core/sizing/advisories.py"),
    ("embankment_volume", "terrainflow_assessment/modules/impoundment_sites.py"),
    ("transect_cells", "terrainflow_assessment/modules/impoundment_sites.py"),
    ("flow_bearing", "terrainflow_assessment/modules/impoundment_sites.py"),
    ("UsableAreaDisjoint", "terrainflow_assessment/modules/contour_analysis.py"),
    ("clip_to_usable_area", "terrainflow_assessment/modules/contour_analysis.py"),
    ("classify_contour_inflow", "terrainflow_assessment/modules/contour_analysis.py"),
    ("find_swale_segments", "terrainflow_assessment/modules/contour_analysis.py"),
    ("recommend_swale_length", "terrainflow_assessment/modules/swale_design.py"),
    ("haul_regions", "terrainflow_assessment/modules/mass_haul.py"),
    ("allocate_haul", "terrainflow_assessment/modules/mass_haul.py"),
    ("earthwork_balance", "terrainflow_assessment/modules/mass_haul.py"),
    ("burn_quantities", "terrainflow_assessment/modules/earthwork_design.py"),
    ("curvature", "terrainflow_assessment/modules/terrain_indices.py"),
    ("horn_gradient", "terrainflow_assessment/modules/dem_loader.py"),
    ("aspect_degrees", "terrainflow_assessment/modules/dem_loader.py"),
    ("find_ridgelines", "terrainflow_assessment/modules/keypoint_analysis.py"),
]

#: Panel properties Step E named. Read from `panel.py` rather than from a live widget,
#: because building one needs the whole Qt stack for a question about source.
PANEL_PROPERTIES = ("keyline_max_grade_n", "keyline_max_valleys", "set_spacing_advice")

#: `(help-text constant, the phrase it states, the module constant, its value)`.
#: Curated by reading, because most numbers in this file are geography — "5-20 ha is a
#: permanent stream" is a description of catchments, not a setting anybody can change.
#: Every pair here is a claim the copy makes about a value the code holds.
HELP_TEXT_PAIRS = [
    {
        "help_constant": "MIN_CATCHMENT",
        "claim": r"Default\s+([\d.]+)\s*ha",
        "source": "terrainflow_assessment/panel.py",
        "source_pattern": r"_min_catchment_ha_spin\.setValue\(([\d.]+)\)",
        "what": "the swale placement threshold's default, in hectares",
    },
    {
        "help_constant": "THROUGHFLOW",
        "claim": r"colour scale starts at\s+([\d.]+)\s*m",
        "source": "terrainflow_assessment/core/registry/map_palette.py",
        "source_pattern": r"SURFACE_RUNOFF_FADE_TOP_M3\s*=\s*([\d.]+)",
        "what": "the surface-runoff fade top, in cubic metres",
    },
    {
        "help_constant": "MAX_SLOPE",
        "claim": r"default\s+([\d.]+)\s*\u00b0",
        "source": "terrainflow_assessment/modules/project_io.py",
        "source_pattern": r'"max_slope_deg":\s*\(float,\s*([\d.]+)\)',
        "what": "the analysis max-slope filter's default, in degrees",
    },
    {
        "help_constant": "COMPANION_BERM",
        "claim": r"spread along the bank at\s+(\d+)\s*%\s*compaction",
        "source": "terrainflow_assessment/modules/earthwork_design.py",
        "source_pattern": r"spoil_m3 = float\(np\.nansum\(cut_depths\)\) \* \(self\.cell_area\) \* ([\d.]+)",
        "what": "the spoil compaction factor the berm is built from, as a percentage",
        "scale": 100.0,
    },
    {
        "help_constant": "KEYLINE_GRADE",
        "claim": r"steeper than\s+1:N",
        "source": "terrainflow_assessment/panel.py",
        "source_pattern": r"_keyline_grade_spin\.setValue\((\d+)\)",
        "what": "the keyline drift limit — the copy names it as 1:N, not as a number, "
                "so there is nothing here that can drift; recorded to show it was asked",
        "no_number_in_copy": True,
    },
]


# ------------------------------------------------------------------- coverage


def called_in(text, name):
    """Is *name* **called** in *text*, as opposed to imported or mentioned?

    `name(` after a word boundary, excluding the definition itself and excluding the
    `from x import name` form. Crude by design: a false positive here is visible in the
    line it quotes, whereas a regex clever enough to parse call graphs would be a second
    thing to be wrong.
    """
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "from ", "import ", "#")):
            continue
        if re.search(rf"\b{re.escape(name)}\s*\(", line):
            hits.append((i, stripped[:120]))
    return hits


def stage_coverage(ev):
    with ev.stage("coverage_rows") as rec:
        pure = _read_tree(_probe.REPO / "tests")
        qgis = _read_tree(_probe.REPO / "tests_qgis")
        prod = _read_tree(_probe.REPO / "terrainflow_assessment")

        rows = {}
        for name, home in SUBJECTS:
            pure_hits = _hits_across(pure, name)
            qgis_hits = _hits_across(qgis, name)
            # The defining module is NOT skipped. `called_in` already ignores `def`
            # lines, and excluding the home file hid every function called only by its
            # own siblings — `specific_catchment_area` feeds the three index functions
            # beside it and looked unwired, which is a different and much less
            # interesting claim than the one it is.
            prod_hits = _hits_across(prod, name)
            outside = [h for h in prod_hits if h[0] != home]
            rows[name] = {
                "defined_in": home,
                "pure_suite_call_sites": len(pure_hits),
                "qgis_suite_call_sites": len(qgis_hits),
                "production_call_sites": len(prod_hits),
                "production_call_sites_outside_its_module": len(outside),
                "production_callers": sorted({f for f, _l, _t in outside})[:4],
                "state": _state(len(pure_hits) + len(qgis_hits), len(prod_hits)),
            }
        rec["subjects"] = rows

        untested = [n for n, r in rows.items() if r["state"].startswith("not exercised")]
        unwired = [n for n, r in rows.items() if r["production_call_sites"] == 0]
        internal = [n for n, r in rows.items()
                    if r["production_call_sites"]
                    and not r["production_call_sites_outside_its_module"]]
        rec["not_exercised_by_either_suite"] = untested
        rec["no_production_caller"] = unwired
        rec["called_only_from_within_their_own_module"] = internal
        ev.note(
            f"Coverage over {len(rows)} Step E subjects: "
            f"{len(untested)} are not exercised by either suite "
            f"({', '.join(untested) or 'none'}); {len(unwired)} have no production "
            f"caller at all ({', '.join(unwired) or 'none'}); {len(internal)} are called "
            f"only from inside their own module "
            f"({', '.join(internal) or 'none'}).")

    with ev.stage("coverage_panel_properties") as rec:
        panel = (_probe.REPO / "terrainflow_assessment" / "panel.py").read_text(
            encoding="utf-8")
        pure = _read_tree(_probe.REPO / "tests")
        qgis = _read_tree(_probe.REPO / "tests_qgis")
        rows = {}
        for name in PANEL_PROPERTIES:
            rows[name] = {
                "defined_in_panel": bool(re.search(rf"\bdef {re.escape(name)}\b", panel)),
                "pure_suite_references": sum(
                    text.count(name) for text in pure.values()),
                "qgis_suite_references": sum(
                    text.count(name) for text in qgis.values()),
            }
            rows[name]["state"] = (
                "exercised" if rows[name]["pure_suite_references"]
                + rows[name]["qgis_suite_references"] else "not exercised")
        rec["properties"] = rows
        ev.note(
            "Coverage, panel properties: " + "; ".join(
                f"{k} {v['state']} "
                f"({v['pure_suite_references']} pure / {v['qgis_suite_references']} qgis "
                f"references)" for k, v in rows.items()))


def _read_tree(root):
    out = {}
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            out[str(path.relative_to(_probe.REPO)).replace("\\", "/")] = path.read_text(
                encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return out


def _hits_across(tree, name):
    out = []
    for rel, text in tree.items():
        for line, snippet in called_in(text, name):
            out.append((rel, line, snippet))
    return out


def _state(test_calls, production_calls):
    if test_calls and production_calls:
        return "exercised, wired"
    if test_calls and not production_calls:
        return "exercised, no production caller — library behaviour only"
    if production_calls and not test_calls:
        return "not exercised by either suite, but shipped and called"
    return "not exercised by either suite, and no production caller"


# ------------------------------------------------------------------ help text


def stage_help_text(ev):
    with ev.stage("help_text_vs_constants") as rec:
        help_path = _probe.REPO / "terrainflow_assessment" / "qgis" / "help_text.py"
        help_src = help_path.read_text(encoding="utf-8")
        rec["help_text_lines"] = len(help_src.splitlines())

        constants = _help_constants(help_src)
        rec["help_constants_defined"] = len(constants)

        results, mismatches = [], []
        for pair in HELP_TEXT_PAIRS:
            copy = constants.get(pair["help_constant"])
            source_src = (_probe.REPO / pair["source"]).read_text(encoding="utf-8")
            source_match = re.search(pair["source_pattern"], source_src)
            entry = {
                "help_constant": pair["help_constant"],
                "what": pair["what"],
                "source": pair["source"],
                "constant_value": source_match.group(1) if source_match else None,
            }
            if copy is None:
                entry["verdict"] = "help constant not found"
            elif pair.get("no_number_in_copy"):
                entry["copy_states"] = None
                entry["verdict"] = (
                    "no number in the copy to compare"
                    if re.search(pair["claim"], copy) else "phrase not found in the copy")
            else:
                claim = re.search(pair["claim"], copy)
                entry["copy_states"] = claim.group(1) if claim else None
                if claim is None or source_match is None:
                    entry["verdict"] = "could not read one side"
                else:
                    scale = pair.get("scale", 1.0)
                    entry["agrees"] = float(claim.group(1)) == float(
                        source_match.group(1)) * scale
                    entry["verdict"] = "agrees" if entry["agrees"] else "DISAGREES"
                    if not entry["agrees"]:
                        mismatches.append(entry)
            results.append(entry)

        rec["pairs"] = results
        rec["mismatches"] = mismatches
        rec["pairs_checked"] = len(
            [e for e in results if e.get("agrees") is not None])
        rec["constants_unchecked"] = len(constants) - len(
            {p["help_constant"] for p in HELP_TEXT_PAIRS})
        ev.note(
            f"help_text.py: {rec['help_text_lines']:,} lines defining "
            f"{rec['help_constants_defined']} tooltip constants, with no test of any "
            f"kind in either suite. {rec['pairs_checked']} numeric claims were checked "
            f"against the constant each quotes; {len(mismatches)} disagree "
            f"({', '.join(m['help_constant'] for m in mismatches) or 'none'}). "
            f"{rec['constants_unchecked']} constants remain unchecked — they state "
            f"geography, method or advice rather than a value the code holds, and no "
            f"mechanical rule separates those from the ones that do.")


def _help_constants(src):
    """`{NAME: text}` for every module-level string constant in `help_text.py`.

    The body is taken by counting parentheses, not by a non-greedy regex. A regex stops
    at the first `)` that ends a line, and this file's copy is full of parenthetical
    asides — which truncated `RUN_KEYLINE` partway through on the first run and reported
    its claim as unreadable when it was simply out of view.
    """
    out = {}
    for match in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*(\(|\"|')", src, re.MULTILINE):
        name, opener = match.group(1), match.group(2)
        start = match.end(2) - 1
        if opener == "(":
            depth, i = 0, start
            while i < len(src):
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = src[start:i + 1]
        else:
            body = src[start:src.find("\n", start)]
        pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        if not pieces:
            continue
        out[name] = "".join(pieces).replace("\\n", "\n")
    return out


# --------------------------------------------------------------------- driver


def main():
    _probe.banner("p_coverage — Step E coverage rows and the help-text check", "n/a")
    ev = _probe.Evidence("p_coverage", ["STEP-E"], None)
    stage_coverage(ev)
    stage_help_text(ev)
    ev.write()


if __name__ == "__main__":
    main()
