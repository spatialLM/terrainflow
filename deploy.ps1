#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$PluginName = 'terrainflow_assessment'
$Src = Join-Path $PSScriptRoot $PluginName
$Dest = Join-Path $env:APPDATA "QGIS\QGIS3\profiles\default\python\plugins\$PluginName"

# Build artefacts that must never reach the profile.
#
# Compiled bytecode is the one that bites. run_qgis_tests.ps1 drives QGIS's own
# Python over the source tree, so the repo accumulates cpython-312 .pyc files -
# the very interpreter QGIS runs. Copy-Item preserves mtimes, so Python accepts
# them as valid for the sources beside them and loads them in preference. Ship
# them and any .py that later arrives with an older timestamp (a revert, a
# checkout, a restored file) silently runs stale bytecode inside QGIS, with the
# source on disk saying something else entirely.
#
# Deploying only what git tracks would be tidier but wrong: work in progress is
# untracked by definition, and that is exactly what this script exists to push.
$PruneDirs = @('__pycache__', '.pytest_cache')
$PruneFiles = @('*.pyc', '*.pyo', '*.aux.xml', 'symbology-style.db')

Write-Host "Deploying TerrainFlow Assessment plugin..."
if (-not (Test-Path $Src)) { throw "Source folder not found: $Src" }

if (Test-Path $Dest) { Remove-Item $Dest -Recurse -Force }
Copy-Item $Src $Dest -Recurse

$pruned = 0
foreach ($name in $PruneDirs) {
    Get-ChildItem $Dest -Recurse -Directory -Filter $name -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            $pruned += (Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue |
                        Measure-Object).Count
            Remove-Item $_.FullName -Recurse -Force
        }
}
foreach ($pattern in $PruneFiles) {
    Get-ChildItem $Dest -Recurse -File -Filter $pattern -Force -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_.FullName -Force; $pruned++ }
}

# A half-copied plugin is worse than a failed deploy, because QGIS will load it.
# Say what actually landed, and refuse to report success if bytecode survived.
$leftover = @(Get-ChildItem $Dest -Recurse -File -Force -ErrorAction SilentlyContinue |
              Where-Object { $_.Extension -in '.pyc', '.pyo' })
if ($leftover.Count -gt 0) {
    throw "Bytecode survived pruning ($($leftover.Count) file(s)) - deploy aborted, plugin may be stale."
}
$files = (Get-ChildItem $Dest -Recurse -File -Force | Measure-Object).Count
if ($files -lt 50) { throw "Only $files files copied - the deploy looks incomplete." }

Write-Host "  $files files deployed, $pruned build artefact(s) pruned."
Write-Host "Done. Reload the plugin in QGIS (disable + re-enable in Plugin Manager)."
