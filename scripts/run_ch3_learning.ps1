[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$CondaEnv = $(if ($env:CRK_CONDA_ENV) { $env:CRK_CONDA_ENV } else { 'AUV' }),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArguments
)
$ErrorActionPreference = 'Stop'
# OpenMP runtime compatibility (explicit user opt-in only).
[Console]::Error.WriteLine("OPENMP_RUNTIME_ENV KMP_DUPLICATE_LIB_OK={0}", $(if ($null -eq $env:KMP_DUPLICATE_LIB_OK) { '<unset>' } else { $env:KMP_DUPLICATE_LIB_OK }))
if ($env:KMP_DUPLICATE_LIB_OK -and $env:KMP_DUPLICATE_LIB_OK -notmatch '^(FALSE|0|NO|OFF)$') {
    [Console]::Error.WriteLine('WARNING: OpenMP duplicate-runtime bypass explicitly requested; this unsafe workaround may crash or silently produce incorrect results. Numerical reliability is not guaranteed.')
}
# End OpenMP runtime compatibility.
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
& $CondaExecutable run --no-capture-output -n $CondaEnv python -B -m chapter3_bser.experiments.hgr.cli @RunArguments
exit $LASTEXITCODE
