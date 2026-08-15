#!/bin/bash
set -euo pipefail

PLUGIN_NAME="terrainflow_assessment"
SRC="$(dirname "$0")/${PLUGIN_NAME}"
DEST="/c/Users/liamm/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/${PLUGIN_NAME}"

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
