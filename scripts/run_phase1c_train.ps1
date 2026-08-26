[CmdletBinding()]
param(
    [int]$Seed,
    [int]$Episodes,
    [string]$Resume,
    [switch]$DryRun
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
    "run",
    "--no-capture-output",
    "-n",
    "AUV",
    "python",
    "-m",
    "chapter3_bser.experiments.phase1c_bser_rmaddpg.train_phase1c",
    "--config",
    "configs/chapter3/bser_phase1c_train.json"
)

if ($PSBoundParameters.ContainsKey("Seed")) {
    $TrainingArguments += @("--seed", $Seed)
}
if ($PSBoundParameters.ContainsKey("Episodes")) {
    $TrainingArguments += @("--episodes", $Episodes)
}
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $TrainingArguments += @("--resume", $Resume)
}
if ($DryRun) {
    $TrainingArguments += "--dry-run"
}

$TrainingLogDirectory = Join-Path $RepoRoot "outputs\chapter3\phase1c_bser_rmaddpg\training\logs"
New-Item -ItemType Directory -Path $TrainingLogDirectory -Force | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ConsoleLog = Join-Path $TrainingLogDirectory "console_$Timestamp.log"

& $CondaExecutable @TrainingArguments 2>&1 |
    Tee-Object -FilePath $ConsoleLog -Append
$TrainingExitCode = $LASTEXITCODE
Write-Host "Phase 1C console log: $ConsoleLog"
exit $TrainingExitCode
