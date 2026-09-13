[CmdletBinding()]
param(
    [string]$CondaEnv = $(if ($env:CRK_CONDA_ENV) { $env:CRK_CONDA_ENV } else { 'AUV' }),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArguments
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $RepoRoot
$CondaExecutable = $env:CRK_CONDA_EXE
if (-not $CondaExecutable) { $CondaExecutable = $env:CONDA_EXE }
if (-not $CondaExecutable) {
    $CondaExecutable = (Get-Command conda.exe, conda.bat -ErrorAction SilentlyContinue | Select-Object -First 1).Source
}
if (-not $CondaExecutable) { throw 'Set CRK_CONDA_EXE to your conda executable' }
$env:MPLBACKEND = 'Agg'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
# conda run initializes the selected environment without depending on shell activation.
& $CondaExecutable run --no-capture-output -n $CondaEnv python -B -m chapter3_bser.experiments.phase1c_prrac.collision_terminal_cli @RunArguments
exit $LASTEXITCODE
