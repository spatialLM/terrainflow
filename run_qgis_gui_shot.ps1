#Requires -Version 5.1
# Loads the plugin in GENUINE QGIS (real iface, real dock, real theme), screenshots
# it, and quits. Complements run_qgis_tests.ps1, which is headless and reliable;
# this one trades reliability for fidelity.
#
#   .\run_qgis_gui_shot.ps1                    # launch, shoot, quit
#   .\run_qgis_gui_shot.ps1 -Keep              # leave QGIS open to poke at by hand
#   .\run_qgis_gui_shot.ps1 -TimeoutSec 600    # allow a slower machine
#
# A QGIS window WILL appear on screen for the duration. It runs under a throwaway
# profile ("TerrainFlowTest") with -noplugins, so it cannot disturb your working
# QGIS profile or load your deployed copy of the plugin alongside this one.
#
# Unattended-safe: if QGIS hangs or never reaches a usable state, the whole process
# tree is killed at -TimeoutSec and the script exits non-zero.
param(
    [switch]$Keep,
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = 'Stop'

function Find-QgisGui {
    if ($env:TERRAINFLOW_QGIS_GUI) {
        if (-not (Test-Path $env:TERRAINFLOW_QGIS_GUI)) {
            throw "TERRAINFLOW_QGIS_GUI does not exist: $env:TERRAINFLOW_QGIS_GUI"
        }
        return $env:TERRAINFLOW_QGIS_GUI
    }
    # An OSGeo4W install put at a drive root has bin\qgis-ltr.bat directly under it,
    # with no "QGIS x.y" folder in between -- which is the shape on this machine
    # (F:\bin\qgis-ltr.bat). Scanning only C:\Program Files finds nothing there and
    # the throw below then blames a missing QGIS rather than a missed layout.
    $drives = Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue |
              Where-Object { $_.Root -match '^[A-Za-z]:\\$' }
    foreach ($drive in $drives) {
        foreach ($name in @('qgis-ltr.bat', 'qgis.bat')) {
            $candidate = Join-Path $drive.Root "bin\$name"
            if (Test-Path $candidate) { return $candidate }
        }
    }

    $roots = @()
    foreach ($drive in $drives) {
        $roots += (Join-Path $drive.Root 'Program Files')
        $roots += (Join-Path $drive.Root 'Program Files (x86)')
        $roots += (Join-Path $drive.Root 'OSGeo4W')
        $roots += (Join-Path $drive.Root 'OSGeo4W64')
    }
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        foreach ($name in @('qgis-ltr.bat', 'qgis.bat')) {
            $direct = Join-Path $root "bin\$name"
            if (Test-Path $direct) { return $direct }
        }
        $installs = Get-ChildItem $root -Directory -Filter 'QGIS *' -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending
        foreach ($install in $installs) {
            foreach ($name in @('qgis-ltr.bat', 'qgis.bat')) {
                $candidate = Join-Path $install.FullName "bin\$name"
                if (Test-Path $candidate) { return $candidate }
            }
        }
    }
    throw "Could not find a QGIS launcher. Set `$env:TERRAINFLOW_QGIS_GUI to bin\qgis-ltr.bat."
}

$QgisGui = Find-QgisGui
$Script  = Join-Path $PSScriptRoot 'tests_qgis\launch_in_qgis.py'
$Report  = Join-Path $PSScriptRoot 'tests_qgis\_shots\gui_report.txt'

if (Test-Path $Report) { Remove-Item $Report -Force }
if ($Keep) { $env:TFA_GUI_KEEP = '1' } else { $env:TFA_GUI_KEEP = '0' }
$env:TFA_TESTS_DIR = Join-Path $PSScriptRoot 'tests_qgis'

Write-Host "QGIS GUI: $QgisGui"
Write-Host "A QGIS window will open; it closes itself when done."

$qgisArgs = @(
    '--profile', 'TerrainFlowTest',
    '--noplugins',
    '--nologo',
    '--code', $Script
)

# Start-Process joins -ArgumentList with spaces and does NOT quote the parts, so a
# script path containing a space (this repo lives under "Terrain Flow Design")
# arrives at QGIS as three arguments and --code gets a truncated path. QGIS swallows
# that silently: it starts normally, never runs launch_in_qgis.py, and the run dies
# at the watchdog with no report and nothing to explain it.
# The call operator (&) quotes properly, so only the Start-Process form needs this --
# which is why -Keep worked while unattended runs did not.
$qgisArgsQuoted = $qgisArgs | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
}

function Get-QgisPids {
    @(Get-Process -Name 'qgis*' -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty Id)
}

if ($Keep) {
    # Interactive use: no watchdog, the window is meant to stay open.
    # The pipe is what makes this block -- qgis-ltr.bat starts its child detached
    # and exits immediately, so only holding its stdout handle waits for QGIS.
    & $QgisGui @qgisArgs | Out-Null
} else {
    # Any qgis process already running belongs to the user. Record those first so
    # the watchdog can only ever kill the instance this script started -- killing
    # every qgis* process would take their working session down with it.
    $preExisting = Get-QgisPids
    if ($preExisting.Count -gt 0) {
        Write-Host "Note: $($preExisting.Count) QGIS process(es) already running; they will not be touched."
    }

    Start-Process -FilePath $QgisGui -ArgumentList $qgisArgsQuoted | Out-Null

    $ourPids = @()
    $appearBy = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $appearBy) {
        $ourPids = @(Get-QgisPids | Where-Object { $preExisting -notcontains $_ })
        if ($ourPids.Count -gt 0) { break }
        Start-Sleep -Milliseconds 300
    }
    if ($ourPids.Count -eq 0) {
        Write-Warning "QGIS never started - no new qgis process appeared within 60s."
        exit 1
    }

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $alive = @($ourPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        if ($alive.Count -eq 0) { break }
        Start-Sleep -Milliseconds 500
    }

    $alive = @($ourPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    if ($alive.Count -gt 0) {
        Write-Warning "QGIS did not exit within ${TimeoutSec}s - killing it."
        foreach ($stuck in $alive) { & taskkill /T /F /PID $stuck 2>&1 | Out-Null }
        if (Test-Path $Report) {
            Write-Host "`n--- gui_report.txt (partial) ---"
            Get-Content $Report -Encoding UTF8
        } else {
            # launch_in_qgis.py writes a breadcrumb report the moment it loads, so no
            # report AT ALL means --code never reached the file -- not that the run
            # hung. Say so, or this presents as a timing problem for the next hour.
            Write-Warning @"
No report was written at all, so --code never reached launch_in_qgis.py.
launch_in_qgis.py writes a breadcrumb the moment it loads, so its absence means
the script was never executed - check the path is correct and quoted:
  $Script
"@
        }
        exit 1
    }
}

if (Test-Path $Report) {
    Write-Host "`n--- gui_report.txt ---"
    Get-Content $Report -Encoding UTF8
    $status = (Select-String -Path $Report -Pattern '^status:').Line
    if ($status -ne 'status: ok') { exit 1 }
    exit 0
}

Write-Warning "No report written - QGIS may not have run the --code script."
exit 1
