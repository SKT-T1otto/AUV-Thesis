# PRRAC training guide

PRRAC is an independent Phase 1C architecture. It keeps the frozen BSER
high-level planner and environment waypoint prior, while replacing the
low-level residual policy and critics with phase-routed components.

The implementation is not compatible with Phase 1C-v1 or Phase 1C-v2
checkpoints. Start PRRAC from a new initialization and use a new output
directory. Do not treat a dry-run or pilot as evidence of convergence or
formal thesis performance.

Run the user-controlled dry-run:

```powershell
.\scripts\run_phase1c_prrac_train.ps1 -DryRun
```


Run the user-controlled 100-episode pilot:

```powershell
.\scripts\run_phase1c_prrac_train.ps1 `
  -Seed 2729 `
  -Episodes 100 `
  -MaxSteps 400 `
  -Workers 4 `
  -Device cpu `
  -OutputDir "outputs\chapter3\phase1c_prrac\pilot_ep100_seed2729"
```


CUDA is opt-in through `-Device cuda`. Resume accepts only checkpoints with
schema `bser.phase1c.prrac.training_state.v1` whose architecture and resolved
configuration hashes match exactly.
