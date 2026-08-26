# Phase 1C Short Training Guide

## PowerShell

From the repository root:

```powershell
.\scripts\run_phase1c_train.ps1
```

The script resolves the repository root, sets `PYTHONPATH`, and uses the
`AUV` conda environment through `conda run`.

Optional overrides:

```powershell
.\scripts\run_phase1c_train.ps1 -Seed 2729 -Episodes 1000
```

## BAT

Double-click `scripts\run_phase1c_train.bat`, or run:

```bat
scripts\run_phase1c_train.bat
```

The BAT file delegates to the PowerShell script. It pauses only when the
training command reports an error.

## Output directory

All artifacts are isolated under:

```text
outputs/chapter3/phase1c_bser_rmaddpg/training/
|-- checkpoints/
|-- logs/
`-- metrics/
```

The runner refuses to overwrite an existing run by default. Select a new
`--output-dir` or resume from a compatible checkpoint.

## Stop and resume

Stop training from the terminal with `Ctrl+C`. Resume from the most recent
completed checkpoint:

```powershell
.\scripts\run_phase1c_train.ps1 `
  -Resume "outputs/chapter3/phase1c_bser_rmaddpg/training/checkpoints/phase1c_episode_0050.pt"
```

Direct Python form:

```powershell
python -m chapter3_bser.experiments.phase1c_bser_rmaddpg.train_phase1c `
  --config configs/chapter3/bser_phase1c_train.json `
  --resume outputs/chapter3/phase1c_bser_rmaddpg/training/checkpoints/phase1c_episode_0050.pt
```

Each checkpoint restores actor, twin critics, target networks, optimizers,
replay buffer, completed episode, global/update steps, sampling counters, and
episode metrics. Resume is rejected if the method, dimensions, BSER integration
version, or resolved config hash differs.

## Dry run

The dry-run command is opt-in and is never launched automatically:

```powershell
.\scripts\run_phase1c_train.ps1 -DryRun
```
