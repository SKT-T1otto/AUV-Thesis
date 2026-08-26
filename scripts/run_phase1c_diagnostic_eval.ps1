[CmdletBinding()]
param(
    [string]$Config = "configs/chapter3/bser_phase1c_diagnostic_eval.json",
    [string[]]$Checkpoint,
    [string]$CheckpointDir,
    [string]$CheckpointPattern = "phase1c_episode_*.pt",
    [string]$OutputDir,
    [int]$Episodes,
    [int]$Workers,
    [string]$Device,
    [string[]]$Modes
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
    $env:PYTHONPATH = $RepoRoot
} else {
    $env:PYTHONPATH = "$RepoRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
}

$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $CondaCommand) {
    $ConfiguredConda = $env:CRK_CONDA_EXE
    if ([string]::IsNullOrWhiteSpace($ConfiguredConda) -or
        -not (Test-Path -LiteralPath $ConfiguredConda -PathType Leaf)) {
        throw "conda was not found; install it or set CRK_CONDA_EXE to the conda executable"
    }
    $CondaExecutable = (Resolve-Path -LiteralPath $ConfiguredConda).Path
} else {
    $CondaExecutable = $CondaCommand.Source
}

$Arguments = @(
    "run", "--no-capture-output", "-n", "AUV", "python", "-B", "-m",
    "chapter3_bser.experiments.phase1c_bser_rmaddpg.evaluate_phase1c_checkpoints",
    "--config", $Config
)
foreach ($Path in $Checkpoint) {
    if (-not [string]::IsNullOrWhiteSpace($Path)) { $Arguments += @("--checkpoint", $Path) }
}
if (-not [string]::IsNullOrWhiteSpace($CheckpointDir)) { $Arguments += @("--checkpoint-dir", $CheckpointDir) }
if (-not [string]::IsNullOrWhiteSpace($CheckpointPattern)) { $Arguments += @("--checkpoint-pattern", $CheckpointPattern) }
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { $Arguments += @("--output-dir", $OutputDir) }
if ($PSBoundParameters.ContainsKey("Episodes")) { $Arguments += @("--episodes", $Episodes) }
if ($PSBoundParameters.ContainsKey("Workers")) { $Arguments += @("--workers", $Workers) }
if (-not [string]::IsNullOrWhiteSpace($Device)) { $Arguments += @("--device", $Device) }
if ($Modes -and $Modes.Count -gt 0) { $Arguments += "--modes"; $Arguments += $Modes }

$DiagnosticRoot = if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $RepoRoot $OutputDir
} else {
    Join-Path $RepoRoot "outputs\chapter3\phase1c_bser_rmaddpg\diagnostics_v1"
}
$LogDirectory = Join-Path $DiagnosticRoot "logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ConsoleLog = Join-Path $LogDirectory "console_$Timestamp.log"

& $CondaExecutable @Arguments 2>&1 | Tee-Object -FilePath $ConsoleLog -Append
$ExitCode = $LASTEXITCODE
Write-Host "Phase 1C diagnostic console log: $ConsoleLog"
exit $ExitCode
