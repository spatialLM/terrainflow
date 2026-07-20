#!/bin/bash
set -euo pipefail

PLUGIN_NAME="terrainflow_assessment"
SRC="$(dirname "$0")/terrainflow_assessment"
DEST="/c/Users/liamm/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/${PLUGIN_NAME}"

echo "Deploying TerrainFlow Assessment plugin..."
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "${SRC}/." "$DEST/"
echo "Done. Reload the plugin in QGIS (disable + re-enable in Plugin Manager)."
