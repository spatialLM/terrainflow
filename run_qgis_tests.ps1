#Requires -Version 5.1
# Runs the real-QGIS smoke checks in tests_qgis/ under QGIS's bundled Python.
# These exercise the qgis/ controller layer that `pytest tests/` mocks out.
#
#   .\run_qgis_tests.ps1              # everything
#   .\run_qgis_tests.ps1 baseline     # only checks matching "baseline"
#   .\run_qgis_tests.ps1 -Snapshot    # store this run's screenshots as the baseline
#
# Visual-diff workflow: -Snapshot before a change, then run again after it, and the
# summary lists exactly which screenshots moved. A changed image is reported, never
# treated as a failure.
#
# Override the interpreter with $env:TERRAINFLOW_QGIS_PYTHON if QGIS lives elsewhere.
param([switch]$Snapshot)

$ErrorActionPreference = 'Stop'

function Find-QgisPython {
    if ($env:TERRAINFLOW_QGIS_PYTHON) {
        if (-not (Test-Path $env:TERRAINFLOW_QGIS_PYTHON)) {
            throw "TERRAINFLOW_QGIS_PYTHON is set but does not exist: $env:TERRAINFLOW_QGIS_PYTHON"
        }
        return $env:TERRAINFLOW_QGIS_PYTHON
    }

    $roots = @('C:\Program Files', 'C:\Program Files (x86)', 'C:\OSGeo4W')
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $installs = Get-ChildItem $root -Directory -Filter 'QGIS *' -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending
        foreach ($install in $installs) {
            foreach ($name in @('python-qgis-ltr.bat', 'python-qgis.bat')) {
                $candidate = Join-Path $install.FullName "bin\$name"
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }
    throw "Could not find a QGIS Python launcher. Set `$env:TERRAINFLOW_QGIS_PYTHON to bin\python-qgis-ltr.bat."
}

$QgisPython = Find-QgisPython
$Runner = Join-Path $PSScriptRoot 'tests_qgis\run_all.py'

$runnerArgs = @($args)
if ($Snapshot) { $runnerArgs += '--snapshot' }

Write-Host "QGIS Python: $QgisPython"
& $QgisPython $Runner @runnerArgs
exit $LASTEXITCODE
