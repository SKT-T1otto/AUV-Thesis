[CmdletBinding()]
param(
    [string]$Checkpoint,
    [ValidateSet(1, 10)][int]$Episodes = 1,
    [string]$Config,
    [int]$Workers = 1,
    [string]$OutputRoot,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Checkpoint)) { throw 'Checkpoint must be explicitly supplied with -Checkpoint.' }
$CheckpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($Config)) { $Config = Join-Path $RepoRoot 'configs\chapter3\bser_phase1c_prrac_eval.json' }
$ConfigPath = (Resolve-Path -LiteralPath $Config).Path
$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
if (-not [string]::IsNullOrWhiteSpace($env:CRK_CONDA_EXE)) {
    $CondaExecutable = (Resolve-Path -LiteralPath $env:CRK_CONDA_EXE).Path
} elseif (-not [string]::IsNullOrWhiteSpace($env:CONDA_EXE)) {
    $CondaExecutable = (Resolve-Path -LiteralPath $env:CONDA_EXE).Path
} elseif ($null -ne $CondaCommand -and $CondaCommand.CommandType -eq 'Function') {
    $CondaExecutable = $CondaCommand.Name
} elseif ($null -ne $CondaCommand -and [System.IO.Path]::GetExtension($CondaCommand.Source) -in @('.exe', '.bat', '.cmd')) {
    $CondaExecutable = $CondaCommand.Source
} else { throw 'A usable conda command was not found; set CRK_CONDA_EXE to the actual conda.exe.' }
$PairArguments = @('run', '--no-capture-output', '-n', 'AUV', 'python', '-B', '-m',
    'chapter3_bser.experiments.phase1c_prrac.paired_evaluation', 'run',
    '--checkpoint', $CheckpointPath, '--config', $ConfigPath, '--episodes', "$Episodes", '--workers', "$Workers")
if (-not [string]::IsNullOrWhiteSpace($OutputRoot)) { $PairArguments += @('--output-root', [System.IO.Path]::GetFullPath($OutputRoot)) }
if ($PrepareOnly) { $PairArguments += '--prepare-only' }
$env:MPLBACKEND = 'Agg'
if ([string]::IsNullOrWhiteSpace($env:OMP_NUM_THREADS)) { $env:OMP_NUM_THREADS = '1' }
if ([string]::IsNullOrWhiteSpace($env:MKL_NUM_THREADS)) { $env:MKL_NUM_THREADS = '1' }
Push-Location -LiteralPath $RepoRoot
try { & $CondaExecutable @PairArguments; $PairExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $PairExitCode
