param(
    [int]$Cycles = 0,
    [string]$Python = "",
    [string]$Model = "",
    [int]$RestartDelaySeconds = 20
)

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    param([string]$Requested)

    if ($Requested) {
        return @($Requested)
    }

    $Launcher = Get-Command py.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($Launcher) {
        foreach ($Version in @("-3.11", "-3.10", "-3.13")) {
            try {
                & $Launcher.Source $Version -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
                if ($LASTEXITCODE -eq 0) {
                    return @($Launcher.Source, $Version)
                }
            } catch {
            }
        }
    }

    foreach ($Candidate in @("python3.11.exe", "python3.10.exe", "python.exe")) {
        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Command -and $Command.Source -notlike "*WindowsApps*") {
            return @($Command.Source)
        }
    }

    throw "Could not resolve a usable Python interpreter. Pass -Python explicitly."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$OutDir = Join-Path $ProjectRoot "research_runs"
$Journal = Join-Path $OutDir "journal.json"
$LlmJournal = Join-Path $OutDir "codex_lab_journal.json"
$Dashboard = Join-Path $OutDir "dashboard.html"
$Status = Join-Path $OutDir "codex_lab_status.json"
$StopFile = Join-Path $OutDir "CODEX_STOP"
$PythonCommand = Resolve-PythonCommand -Requested $Python

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$CmdArgs = @(
    "experiments/codex_autonomous_lab.py",
    "--out", $OutDir,
    "--journal", $Journal,
    "--llm-journal", $LlmJournal,
    "--dashboard", $Dashboard,
    "--status", $Status,
    "--stop-file", $StopFile,
    "--cycles", $Cycles
)
if ($Model) {
    $CmdArgs += @("--model", $Model)
}

Write-Host "=========================================="
Write-Host "  CODEX AUTONOMOUS LAB"
Write-Host "  Output: $OutDir"
Write-Host "  Cycles: $(if ($Cycles -eq 0) { 'infinite' } else { $Cycles })"
Write-Host "  Stop file: $StopFile"
Write-Host "  Python: $($PythonCommand -join ' ')"
Write-Host "=========================================="
Write-Host ""
Write-Host "Create the stop file to end the loop cleanly:"
Write-Host "  New-Item -ItemType File -Force -Path '$StopFile' | Out-Null"
Write-Host ""

while ($true) {
    Write-Host "[$((Get-Date).ToUniversalTime().ToString('o'))] Starting Codex autonomous lab..."
    $PythonExe = $PythonCommand[0]
    $PythonPrefixArgs = @()
    if ($PythonCommand.Length -gt 1) {
        $PythonPrefixArgs = $PythonCommand[1..($PythonCommand.Length - 1)]
    }
    & $PythonExe @PythonPrefixArgs @CmdArgs
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        Write-Host "[$((Get-Date).ToUniversalTime().ToString('o'))] Codex autonomous lab exited cleanly."
        break
    }

    Write-Host "[$((Get-Date).ToUniversalTime().ToString('o'))] Codex autonomous lab crashed (exit=$ExitCode). Restarting in $RestartDelaySeconds seconds..."
    Start-Sleep -Seconds $RestartDelaySeconds
}

Write-Host "Dashboard: $Dashboard"
Write-Host "Status: $Status"
Write-Host "Codex Journal: $LlmJournal"
