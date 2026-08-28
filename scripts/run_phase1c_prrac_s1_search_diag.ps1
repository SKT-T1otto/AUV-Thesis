[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [ValidateSet("legacy", "native")][string]$RuntimeOrigin = "legacy",
    [int]$Workers = 4,
    [int]$Episodes = 50
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$Config = if ($RuntimeOrigin -eq "native") { "configs/chapter3/bser_phase1c_prrac_s1_search_diag_native.json" } else { "configs/chapter3/bser_phase1c_prrac_s1_search_diag_legacy.json" }
& conda run --no-capture-output -n AUV python -B -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints --config $Config --checkpoint $Checkpoint --output-dir $OutputDir --workers $Workers --episodes $Episodes --modes full_prrac searcher_residual_off --disable-failure-trace
exit $LASTEXITCODE
