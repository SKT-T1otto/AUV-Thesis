[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvaluatorArguments
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $CondaCommand) {
    $ConfiguredConda = $env:CRK_CONDA_EXE
    if ([string]::IsNullOrWhiteSpace($ConfiguredConda) -or
        -not (Test-Path -LiteralPath $ConfiguredConda -PathType Leaf)) {
        throw "conda was not found; install it or set CRK_CONDA_EXE"
    }
    $CondaExecutable = (Resolve-Path -LiteralPath $ConfiguredConda).Path
} else {
    $CondaExecutable = $CondaCommand.Source
}

$env:MPLBACKEND = "Agg"
if ([string]::IsNullOrWhiteSpace($env:OMP_NUM_THREADS)) { $env:OMP_NUM_THREADS = "1" }
if ([string]::IsNullOrWhiteSpace($env:MKL_NUM_THREADS)) { $env:MKL_NUM_THREADS = "1" }

$Arguments = @(
    "run", "--no-capture-output", "-n", "AUV", "python", "-B", "-m",
    "chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints"
) + $EvaluatorArguments

& $CondaExecutable @Arguments
exit $LASTEXITCODE
