[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$OutputDir = "outputs/chapter3/phase1c_prrac/s2a1_local_connector_ablation",
    [int]$Workers = 4,
    [int]$Episodes = 10,
    [switch]$Formal,
    [string]$ScenarioIdFile = ""
)
$ErrorActionPreference = "Stop"
if ($Formal -and -not [string]::IsNullOrWhiteSpace($ScenarioIdFile)) { throw "--formal cannot be combined with --scenario-id-file" }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
if ($null -eq $CondaCommand) {
    $ConfiguredConda = $env:CRK_CONDA_EXE
    if ([string]::IsNullOrWhiteSpace($ConfiguredConda) -or -not (Test-Path -LiteralPath $ConfiguredConda -PathType Leaf)) { throw "conda was not found; install it or set CRK_CONDA_EXE" }
    $CondaExecutable = (Resolve-Path -LiteralPath $ConfiguredConda).Path
} else { $CondaExecutable = $CondaCommand.Source }
$env:MPLBACKEND = "Agg"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$Arguments = @("run", "--no-capture-output", "-n", "AUV", "python", "-B", "-m", "chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints", "--config", "configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json", "--checkpoint", $Checkpoint, "--output-dir", $OutputDir, "--workers", "$Workers", "--modes", "full_prrac", "--execution-variants", "B1_ATOMIC_LAST_VALID", "--search-recovery-variants", "S2A1_C0_BASELINE", "S2A1_C1_FORCED_REFRESH", "S2A1_C2_LOCAL_CONNECTOR")
if ($Formal) { $Arguments += "--formal" }
elseif (-not [string]::IsNullOrWhiteSpace($ScenarioIdFile)) { $Arguments += @("--scenario-id-file", $ScenarioIdFile) }
else { $Arguments += @("--episodes", "$Episodes") }
& $CondaExecutable @Arguments
exit $LASTEXITCODE
