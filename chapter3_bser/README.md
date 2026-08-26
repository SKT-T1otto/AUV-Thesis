# Chapter 3 BSER

Belief-guided Submodular Search-Execution Reallocation (BSER) is the Chapter 3
high-level allocation and guidance method. The directory contains both the
historical offline formulation and the later online/learning integration.

## Implementation phases

- **Phase 1A / 1A.1 - offline allocation.** Reachable search and executor
  standby candidates feed a nonnegative monotone submodular objective under a
  partition matroid. Exact, greedy, lazy-greedy, search-only, and random
  baselines support corrected offline validation.
- **Phase 1B - online reallocation.** Public planning state and mission events
  drive online BSER allocation without changing the Phase 1A objective.
- **Phase 1B.1 - corrected pilot.** Corrected online pilot and consistency
  checks establish the event-triggered execution path.
- **Phase 1B.2 / 1B.2.1 - partial allocation and execution consistency.** Path
  tracking, waypoint consistency, executor target locking, and
  public-information replanning are integrated while preserving the frozen
  Phase 1B experiment semantics.
- **Phase 1B.3a - diagnosis.** Event/candidate failure diagnostics and baseline
  reproduction verify behavior preservation.
- **Phase 1C - residual-control architectures.** The earlier BSER-RMADDPG
  integration is retained as historical work. The independent PRRAC
  architecture and trainer are implemented with the same frozen 28D/3D/124D
  contracts.

The Phase 1A description above is historical scope, not the repository's
current implementation limit.

## Current PRRAC status

The repository is in post-rebuild source repair and verification. PRRAC has
not run a dry-run, 100-episode pilot, or formal experiment;
`performance_passed` is not established. Chapter 4 RCAG has not begun.

## Historical BSER-RMADDPG status

Method `ch3_bser_rmaddpg_phase1c` has an independent runtime, trainer,
configuration, PowerShell/BAT launchers, metrics, checkpoint metadata, and
resume support. Implementation availability is distinct from experiment
completion.

Historical status: **WIP / short training interrupted / previously
resume-ready**. This is not the current experiment or current resume target.

- Profile: `M20_MOVING_UNKNOWN_MULTI`
- Seed: `2729`
- Planned run: 1000 episodes x 400 maximum steps
- Workers: 4; `training_update=true`; checkpoint interval: 50 episodes
- Last recorded episode: 128
- Last complete recovery checkpoint: episode 100
- Episodes 101-128 are not represented in that checkpoint
- The 1000-episode run and formal comparisons are not complete

The interruption was caused by
`RuntimeError: current_pos is not a legal reachable planner point` during an
unknown-map reset. `GuidedEnv` now applies a Phase-1C-only initial endpoint
guard: an invalid legacy endpoint temporarily holds its current position until
BSER guidance is installed. It does not skip a scenario or episode and does not
modify the shared path planner or Phase 1B behavior.

The trainer records `initial_planner_endpoint_fallback_count`; worker failures
and complete console output are retained under the training log directory.

## Latest recorded local verification

- Phase 1C guidance: 3 tests OK.
- Observation 28D contract and Phase 1B path tracking: PASS (2 tests).
- Episode 100 checkpoint: readable; actor, critic, optimizer, replay buffer,
  and training counters can be restored.

These are local verification records, not GitHub CI results. No final training
summary or convergence claim exists yet.

## Historical recovery command

This command is retained for provenance and must not be started automatically
or presented as the current PRRAC workflow:

```powershell
.\scripts\run_phase1c_train.ps1 `
  -Resume "outputs/chapter3/phase1c_bser_rmaddpg/training/checkpoints/phase1c_episode_0100.pt"
```
