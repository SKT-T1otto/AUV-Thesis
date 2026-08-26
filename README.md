# CRK-Thesis-v2

Self-contained production repository for the multi-agent AUV thesis. The
shared mission environment and reinforcement-learning infrastructure live in
`core`; chapter-specific work remains in its chapter namespace.

## Repository boundaries

- `core`: shared executable environment, scenarios, MADDPG, and replay code.
- `chapter3_bser`: Chapter 3 BSER implementation and experiments.
- `chapter4_rcag`: Chapter 4 placeholder; RCAG is not implemented here.
- `chapter5_vsgc`: Chapter 5 placeholder; VSGC is not implemented here.
- sibling CH3/CH4/CH5 repositories: read-only historical references, never
  runtime dependencies.

## Chapter 3 status

Chapter 3 has progressed beyond the historical Phase 0B-2 baseline:

- Phase 1A: offline candidates, monotone submodular objective, greedy/exact
  solvers, and corrected offline validation.
- Phase 1B through Phase 1B.3a: online event-triggered reallocation, partial
  BSER, path/waypoint consistency, public-information replanning, executor
  target consistency, and behavior-preserving diagnosis.
- Phase 1C: BSER high-level guidance connected to RMADDPG residual control by
  an isolated trainer and output namespace.

Phase 1C is **WIP / short training interrupted / resume-ready**. This does not
mean that Phase 1C, the 1000-episode run, convergence analysis, or formal thesis
comparisons are complete.

## Current Phase 1C run

- Method: `ch3_bser_rmaddpg_phase1c`
- Profile: `M20_MOVING_UNKNOWN_MULTI`
- Seed: `2729`
- Plan: `1000` episodes, at most `400` steps per episode
- Workers: `4`
- Updates: `training_update=true`
- Checkpoint interval: `50` episodes

The short training process exited after recording episode 128. The last full
checkpoint is episode 100; episodes 101-128 are metrics-only and are not part
of that checkpoint. The run has not produced the normal final
`training_summary.json`, `training_log.json`, or formal curves.

The interruption was traced to an unknown-map reset endpoint error:
`RuntimeError: current_pos is not a legal reachable planner point`. A
Phase-1C-only reset guard now protects legacy initial waypoint setup before
BSER guidance is installed. It does not skip scenarios or episodes and does
not modify the shared path planner or frozen Phase 1B behavior.

Local recovery, when explicitly initiated by the user, starts from episode
100:

```powershell
.\scripts\run_phase1c_train.ps1 `
  -Resume "outputs/chapter3/phase1c_bser_rmaddpg/training/checkpoints/phase1c_episode_0100.pt"
```

Codex must not run this command automatically.

## Frozen environment contract

- Agents: 4
- Role order: `search_fast`, `search_balanced`, `search_precise`, `executor`
- Observation dimensions: `[28, 28, 28, 28]`
- Action dimensions: `[3, 3, 3, 3]`
- Centralized critic input: `124`
- Canonical profile: `M20_MOVING_UNKNOWN_MULTI`

## Latest recorded local verification

These are recorded local results, not GitHub CI results.

Passed in the latest recorded local verification:

- Phase 1C guidance: 3 tests OK.
- Observation 28D contract and Phase 1B path tracking: PASS (2 tests).
- Episode 100 checkpoint: locally verified readable and capable of restoring
  actor, critic, optimizer, replay buffer, and training counters.

These statements are historical local verification records. They do not imply
that the current working tree has been revalidated after every later
documentation or metadata edit.

## Repository provenance policy

`docs/provenance/ch3_to_core_migration_manifest.json` is the frozen Phase 0B-2
historical migration baseline. Its recorded hashes must not be rewritten to
hide later repository evolution.

The current repository contains one explicitly reviewed post-baseline evolution:

- `core/registry/experiment_registry.py` registers the independent
  `ch3_bser_rmaddpg_phase1c` runtime.
- The seven legacy Chapter-3 active experiment modes remain unchanged.
- `tests/test_repository_metadata.py` preserves the Phase 0B-2 historical hash,
  pins the reviewed current hash, and verifies the registry semantic contract.
- Any additional provenance drift is treated as an error until explicitly
  reviewed.

## Pre-resume verification

Before resuming the interrupted Phase 1C run, run:

```powershell
conda run --no-capture-output -n AUV python -B -m unittest tests.test_phase1c_guidance -v
conda run --no-capture-output -n AUV python -B -m unittest tests.test_observation_28d_contract tests.test_phase1b2_path_tracking -v
conda run --no-capture-output -n AUV python -B -m unittest tests.test_repository_metadata -v
```

All required verification commands must pass in the current working tree before
the resume command is treated as ready for execution. Do not describe these
commands as GitHub CI; they are local verification commands.

## Outputs and retention

Phase 1C local artifacts are isolated under:

```text
outputs/chapter3/phase1c_bser_rmaddpg/training/
|-- checkpoints/
|-- logs/
`-- metrics/
```

Do not commit checkpoints (`.pt`, `.pth`, `.ckpt`), models, raw long-run data,
or generated long-training outputs. Do not delete locally retained outputs.
Compact manifests, documentation, tests, and explicitly selected acceptance
evidence may be committed.
