#!/bin/bash
DEST="/c/Users/liamm/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/plugin"

echo "Deploying TerrainFlow plugin..."
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$(dirname "$0")/plugin/." "$DEST/"
echo "Done. Reload the plugin in QGIS (disable + re-enable in Plugin Manager)."
