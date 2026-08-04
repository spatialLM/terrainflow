#Requires -Version 5.1
# Runs the real-QGIS smoke checks in tests_qgis/ under QGIS's bundled Python.
# These exercise the qgis/ controller layer that `pytest tests/` mocks out.
#
#   .\run_qgis_tests.ps1              # everything
#   .\run_qgis_tests.ps1 baseline     # only checks matching "baseline"
#   .\run_qgis_tests.ps1 -Snapshot    # run, then store the screenshots as the baseline
#   .\run_qgis_tests.ps1 -Accept      # accept the screenshots already on disk (no re-run)
#   .\run_qgis_tests.ps1 -Prompt      # run, then ASK whether to accept any changes
#
# Visual-diff workflow: the summary lists exactly which screenshots moved. A changed
# image is reported, never treated as a failure. Use -Prompt if you would otherwise
# forget to accept them; the reminder escalates on its own after two stale runs.
#
# Override the interpreter with $env:TERRAINFLOW_QGIS_PYTHON if QGIS lives elsewhere.
param(
    [switch]$Snapshot,
    [switch]$Accept,
    [switch]$Prompt
)

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

if ($Accept) {
    # Accept what is already on disk; nothing is re-run.
    & $QgisPython $Runner '--accept'
    exit $LASTEXITCODE
}

& $QgisPython $Runner @runnerArgs
$exitCode = $LASTEXITCODE

if ($Prompt -and -not $Snapshot) {
    $stateFile = Join-Path $PSScriptRoot 'tests_qgis\_shots_baseline\diff_state.json'
    if (Test-Path $stateFile) {
        $state = Get-Content $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $names = @($state.names)
        if ($names.Count -gt 0) {
            Write-Host ""
            foreach ($n in $names) { Write-Host "  changed: $n" }
            $answer = Read-Host "Accept these $($names.Count) image(s) as the new baseline? [y/N]"
            if ($answer -match '^\s*(y|yes)\s*$') {
                & $QgisPython $Runner '--accept'
            } else {
                Write-Host "Left as-is. Accept later with: .\run_qgis_tests.ps1 -Accept"
            }
        }
    }
}

exit $exitCode
