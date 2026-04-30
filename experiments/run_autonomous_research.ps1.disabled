param(
    [int]$Cycles = 0,
    [string]$Frontier = "",
    [string]$Python = "",
    [int]$RestartDelaySeconds = 15
)

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    param([string]$Requested)

    if ($Requested) {
        return $Requested
    }

    foreach ($Candidate in @("python3.11.exe", "python.exe")) {
        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Command) {
            return $Command.Source
        }
    }

    throw "Could not resolve a usable Python interpreter. Pass -Python explicitly."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$OutDir = Join-Path $ProjectRoot "research_runs"
$Journal = Join-Path $OutDir "journal.json"
$Dashboard = Join-Path $OutDir "dashboard.html"
$Status = Join-Path $OutDir "status.json"
$StopFile = Join-Path $OutDir "STOP"
$Python = Resolve-PythonCommand -Requested $Python

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$CmdArgs = @(
    "experiments/autonomous_research.py",
    "--out", $OutDir,
    "--journal", $Journal,
    "--dashboard", $Dashboard,
    "--status", $Status,
    "--stop-file", $StopFile,
    "--cycles", $Cycles
)
if ($Frontier) {
    $CmdArgs += @("--frontier", $Frontier)
}

Write-Host "=========================================="
Write-Host "  AUTONOMOUS RESEARCH - 24/7 MODE"
Write-Host "  Output: $OutDir"
Write-Host "  Cycles: $(if ($Cycles -eq 0) { 'infinite' } else { $Cycles })"
Write-Host "  Frontier: $(if ($Frontier) { $Frontier } else { 'all (UCB1 selection)' })"
Write-Host "  Stop file: $StopFile"
Write-Host "  Python: $Python"
Write-Host "=========================================="
Write-Host ""
Write-Host "Create the stop file to end the loop cleanly:"
Write-Host "  New-Item -ItemType File -Force -Path '$StopFile' | Out-Null"
Write-Host ""

while ($true) {
    $StartedAt = (Get-Date).ToUniversalTime().ToString("o")
    Write-Host "[$StartedAt] Starting autonomous research..."

    & $Python @CmdArgs
    $ExitCode = $LASTEXITCODE

    try {
        & $Python "experiments/frontiers/research_dashboard.py" "--journal" $Journal "--output" $Dashboard *> $null
    } catch {
    }

    if ($ExitCode -eq 0) {
        Write-Host "[$((Get-Date).ToUniversalTime().ToString('o'))] Research exited cleanly."
        break
    }

    Write-Host "[$((Get-Date).ToUniversalTime().ToString('o'))] Research crashed (exit=$ExitCode). Restarting in $RestartDelaySeconds seconds..."
    Start-Sleep -Seconds $RestartDelaySeconds
}

Write-Host "Dashboard: $Dashboard"
Write-Host "Status: $Status"
