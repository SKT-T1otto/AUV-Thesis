[CmdletBinding()]
param(
    [string]$Config = "configs/chapter3/bser_phase1c_v2_train.json",
    [int]$Seed,
    [int]$Episodes,
    [int]$MaxSteps,
    [int]$Workers,
    [string]$Device,
    [string]$Resume,
    [switch]$DryRun,
    [string]$OutputDir
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

$TrainingArguments = @(
    "run", "--no-capture-output", "-n", "AUV", "python", "-B", "-m",
    "chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2",
    "--config", $Config
)
if ($PSBoundParameters.ContainsKey("Seed")) { $TrainingArguments += @("--seed", $Seed) }
if ($PSBoundParameters.ContainsKey("Episodes")) { $TrainingArguments += @("--episodes", $Episodes) }
if ($PSBoundParameters.ContainsKey("MaxSteps")) { $TrainingArguments += @("--max-steps", $MaxSteps) }
if ($PSBoundParameters.ContainsKey("Workers")) { $TrainingArguments += @("--workers", $Workers) }
if (-not [string]::IsNullOrWhiteSpace($Device)) { $TrainingArguments += @("--device", $Device) }
if (-not [string]::IsNullOrWhiteSpace($Resume)) { $TrainingArguments += @("--resume", $Resume) }
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { $TrainingArguments += @("--output-dir", $OutputDir) }
if ($DryRun) { $TrainingArguments += "--dry-run" }

$BaseLogDirectory = if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    Join-Path $RepoRoot $OutputDir
} else {
    Join-Path $RepoRoot "outputs\chapter3\phase1c_bser_rmaddpg_v2\training"
}
if ($DryRun) { $BaseLogDirectory = Join-Path $BaseLogDirectory "dry_run" }
$TrainingLogDirectory = Join-Path $BaseLogDirectory "logs"
New-Item -ItemType Directory -Path $TrainingLogDirectory -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ConsoleLog = Join-Path $TrainingLogDirectory "console_$Timestamp.log"

& $CondaExecutable @TrainingArguments 2>&1 | Tee-Object -FilePath $ConsoleLog -Append
$TrainingExitCode = $LASTEXITCODE
Write-Host "Phase 1C-v2 console log: $ConsoleLog"
exit $TrainingExitCode
