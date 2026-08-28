[CmdletBinding()]
param(
    [switch]$Formal,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$Arguments = @("run", "--no-capture-output", "-n", "AUV", "python", "-B", "-m", "chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac", "--config", "configs/chapter3/bser_phase1c_prrac_s1_train.json")
if (-not $Formal) { $Arguments += "--dry-run" }
$Arguments += $RemainingArguments
& conda @Arguments
exit $LASTEXITCODE
