[CmdletBinding()]
param(
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
$Launcher = Join-Path $PSScriptRoot "run_phase1c_v2_train.ps1"
$Arguments = @{
    Config = "configs/chapter3/bser_phase1c_v2_1_train.json"
    OutputDir = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
        "outputs/chapter3/phase1c_bser_rmaddpg_v2_1/training"
    } else {
        $OutputDir
    }
}
foreach ($Name in "Seed", "Episodes", "MaxSteps", "Workers") {
    if ($PSBoundParameters.ContainsKey($Name)) {
        $Arguments[$Name] = $PSBoundParameters[$Name]
    }
}
foreach ($Name in "Device", "Resume") {
    if (-not [string]::IsNullOrWhiteSpace($PSBoundParameters[$Name])) {
        $Arguments[$Name] = $PSBoundParameters[$Name]
    }
}
if ($DryRun) { $Arguments["DryRun"] = $true }

& $Launcher @Arguments
exit $LASTEXITCODE
