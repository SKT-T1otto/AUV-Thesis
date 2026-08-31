[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$Workers = 4,
    [int]$Episodes = 10,
    [switch]$Formal
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $CondaCommand) {
    $ConfiguredConda = $env:CRK_CONDA_EXE
    if ([string]::IsNullOrWhiteSpace($ConfiguredConda) -or -not (Test-Path -LiteralPath $ConfiguredConda -PathType Leaf)) { throw "conda was not found; install it or set CRK_CONDA_EXE" }
    $CondaExecutable = (Resolve-Path -LiteralPath $ConfiguredConda).Path
} else { $CondaExecutable = $CondaCommand.Source }
$env:MPLBACKEND = "Agg"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$Arguments = @("run", "--no-capture-output", "-n", "AUV", "python", "-B", "-m", "chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints", "--config", "configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json", "--checkpoint", $Checkpoint, "--output-dir", $OutputDir, "--workers", "$Workers", "--modes", "full_prrac", "--execution-variants", "B1_ATOMIC_LAST_VALID", "--search-recovery-variants", "S2A_C0_BASELINE", "S2A_C1_ROUTE_REFRESH", "S2A_C2_EGRESS_ROUTE")
if (-not $Formal) { $Arguments += @("--episodes", "$Episodes") }
& $CondaExecutable @Arguments
exit $LASTEXITCODE
