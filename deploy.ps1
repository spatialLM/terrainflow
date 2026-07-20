#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$PluginName = 'terrainflow_assessment'
$Src = Join-Path $PSScriptRoot 'terrainflow_assessment'
$Dest = Join-Path $env:APPDATA "QGIS\QGIS3\profiles\default\python\plugins\$PluginName"

Write-Host "Deploying TerrainFlow Assessment plugin..."
if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
Copy-Item $Src $Dest -Recurse
Write-Host "Done. Reload the plugin in QGIS (disable + re-enable in Plugin Manager)."
