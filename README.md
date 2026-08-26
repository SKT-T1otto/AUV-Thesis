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
- Phase 1C: the earlier BSER-RMADDPG integration is retained as historical
  work, and the independent PRRAC architecture and trainer are implemented in
  the Chapter 3 namespace.

The current engineering state is **post-rebuild source repair and
verification**. The PRRAC architecture is implemented, while Phase 1C
performance, convergence analysis, and formal thesis comparisons remain WIP.

## PRRAC experiment status

- PRRAC dry-run: not run.
- PRRAC 100-episode pilot: not run.
- PRRAC 300/1000-episode experiments: not run.
- `performance_passed`: not established.
- Chapter 4 RCAG: not begun.

No PRRAC training result is claimed by the implementation and source-repair
work.

## Historical Phase 1C run

- Method: `ch3_bser_rmaddpg_phase1c`
- Profile: `M20_MOVING_UNKNOWN_MULTI`
- Seed: `2729`
- Plan: `1000` episodes, at most `400` steps per episode
- Workers: `4`
- Updates: `training_update=true`
- Checkpoint interval: `50` episodes

This is retained historical evidence, not the current experiment and not the
current resume target. The old run was previously described as resume-ready.
Its short training process exited after recording episode 128. The last full
checkpoint is episode 100; episodes 101-128 are metrics-only and are not part
of that checkpoint. The run has not produced the normal final
`training_summary.json`, `training_log.json`, or formal curves.

The interruption was traced to an unknown-map reset endpoint error:
`RuntimeError: current_pos is not a legal reachable planner point`. A
Phase-1C-only reset guard now protects legacy initial waypoint setup before
BSER guidance is installed. It does not skip scenarios or episodes and does
not modify the shared path planner or frozen Phase 1B behavior.

The historical recovery command started from episode 100:

```powershell
.\scripts\run_phase1c_train.ps1 `
  -Resume "outputs/chapter3/phase1c_bser_rmaddpg/training/checkpoints/phase1c_episode_0100.pt"
```

Do not run this historical command automatically or present it as the current
PRRAC workflow.

## Frozen environment contract

- Agents: 4
- Role order: `search_fast`, `search_balanced`, `search_precise`, `executor`
- Observation dimensions: `[28, 28, 28, 28]`
- Action dimensions: `[3, 3, 3, 3]`
- Centralized critic input: `124`
- Canonical profile: `M20_MOVING_UNKNOWN_MULTI`

## Post-rebuild source verification

These are latest recorded local verification results, not GitHub CI results.

- Static compile and PRRAC PowerShell parser: PASS.
- PRRAC unit suite: 16 tests PASS.
- Frozen guidance/observation/path/reward/replay/metadata regressions: 18 tests
  PASS.
- Restored Phase 1A/1B regressions: 35 tests PASS after exact source/hash
  repair and the explicitly pinned registry evolution.
- PRRAC checkpoint verification loads the saved state into a new learner and
  a new replay object; it is not inferred from checkpoint-file presence.

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
