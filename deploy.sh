#!/bin/bash
set -euo pipefail

PLUGIN_NAME="terrainflow_assessment"
SRC="$(dirname "$0")/${PLUGIN_NAME}"

# The profile to deploy into, from the environment rather than from a name typed
# once on a machine that no longer exists. This read `/c/Users/liamm/...`, which is
# not this user: `mkdir -p` below happily created the whole phantom tree, both
# sanity gates counted the files it had just copied there and passed, "Done"
# printed, and QGIS went on running the previous build. A deploy that reports
# success and deploys nothing is worse than one that fails. deploy.ps1 has always
# read $env:APPDATA; this is the same thing in the shell's spelling.
APPDATA_DIR="${APPDATA:-}"
if [ -z "$APPDATA_DIR" ]; then
    echo "APPDATA is not set, so there is no QGIS profile to deploy into." >&2
    exit 1
fi
# MSYS/Git Bash leaves APPDATA in Windows form (C:\Users\...); cygpath converts it
# when available, and the manual substitution covers a plain POSIX shell.
if command -v cygpath >/dev/null 2>&1; then
    APPDATA_DIR="$(cygpath -u "$APPDATA_DIR")"
else
    APPDATA_DIR="/$(echo "$APPDATA_DIR" | sed -e 's|\\|/|g' -e 's|^\([A-Za-z]\):|\1|')"
fi

PLUGINS_DIR="${APPDATA_DIR}/QGIS/QGIS3/profiles/default/python/plugins"
DEST="${PLUGINS_DIR}/${PLUGIN_NAME}"

# Refuse rather than create. The plugins directory exists on any machine that has
# run QGIS once; if it is absent, either QGIS has never run here or APPDATA points
# somewhere unexpected — and in both cases making the directory would produce a
# deploy nobody can load, reported as a success.
if [ ! -d "$PLUGINS_DIR" ]; then
    echo "No QGIS plugins directory at:" >&2
    echo "    $PLUGINS_DIR" >&2
    echo "Run QGIS once to create the default profile, or check APPDATA." >&2
    exit 1
fi

# Build artefacts that must never reach the profile. See deploy.ps1 for the full
# reasoning: run_qgis_tests.ps1 leaves cpython-312 bytecode in the source tree,
# cp -r preserves mtimes, and Python then trusts that bytecode over the sources
# beside it — so a .py that later arrives with an older timestamp runs stale code
# inside QGIS while the source on disk says something else.
PRUNE_DIRS=(__pycache__ .pytest_cache)
PRUNE_FILES=('*.pyc' '*.pyo' '*.aux.xml' 'symbology-style.db')

echo "Deploying TerrainFlow Assessment plugin..."
[ -d "$SRC" ] || { echo "Source folder not found: $SRC" >&2; exit 1; }

rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "${SRC}/." "$DEST/"

pruned=0
for name in "${PRUNE_DIRS[@]}"; do
    while IFS= read -r -d '' dir; do
        pruned=$(( pruned + $(find "$dir" -type f | wc -l) ))
        rm -rf "$dir"
    done < <(find "$DEST" -type d -name "$name" -print0)
done
for pattern in "${PRUNE_FILES[@]}"; do
    while IFS= read -r -d '' file; do
        rm -f "$file"
        pruned=$(( pruned + 1 ))
    done < <(find "$DEST" -type f -name "$pattern" -print0)
done

# A half-copied plugin is worse than a failed deploy, because QGIS will load it.
leftover=$(find "$DEST" \( -name '*.pyc' -o -name '*.pyo' \) -type f | wc -l)
if [ "$leftover" -gt 0 ]; then
    echo "Bytecode survived pruning ($leftover file(s)) — deploy aborted." >&2
    exit 1
fi
files=$(find "$DEST" -type f | wc -l)
if [ "$files" -lt 50 ]; then
    echo "Only $files files copied — the deploy looks incomplete." >&2
    exit 1
fi

echo "  $files files deployed, $pruned build artefact(s) pruned."
echo "Done. Reload the plugin in QGIS (disable + re-enable in Plugin Manager)."
