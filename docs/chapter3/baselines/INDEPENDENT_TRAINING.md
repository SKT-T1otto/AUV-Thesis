# Independent B2/B3 training

The B2/B3 framework now has independent replay-based MADDPG implementations.
The historical HGR-family `stochastic_direct_mc` and
`direct_boundary_corrected` implementations remain untouched. Their checkpoints
are not compatible with these baseline trainers.

Source-gate activation is pending explicit approval of
`docs/provenance/baseline_maddpg_evolution.json`. Until that record is approved
and installed, the framework correctly refuses training/evaluation preflight.
No historical source manifest is rewritten.

## New implementation files

Under `chapter3_bser/experiments/baselines/`:

- `__init__.py`
- `common/__init__.py`, `common/model.py`, `common/runtime.py`,
  `common/train.py`, `common/checkpoint.py`
- `direct_mc/__init__.py`, `direct_mc/runtime.py`, `direct_mc/train.py`,
  `direct_mc/checkpoint.py`
- `direct_boundary/__init__.py`, `direct_boundary/runtime.py`,
  `direct_boundary/train.py`, `direct_boundary/checkpoint.py`

Also added: `tools/ch3_baselines/learned_evaluation.py` and
`tests/test_ch3_baseline_training.py`.

## B2

`DirectMCTrainer` / `B2Trainer` constructs four core MADDPG agents and a core
`CH3ReplayBuffer`. Each mission installs the existing BSER guidance, observes
four raw 28D local observations, emits four 3D commands, and stores the complete
transition with the existing team reward. Uniform replay samples feed the
existing 124D centralized critics and actor updates, followed by target updates.
The core implementation, including its twin critics and target-Q clipping,
is reused unchanged. No separate textbook MADDPG implementation is introduced.

The neural expert, HGR estimator, reconstructed gradient, phase-specific losses,
success-prioritized replay and residual-action regularizer are absent. The
environment's existing physical prior and action mapping remain enabled because
they belong to the common task contract.

## B3

`DirectBoundaryTrainer` / `B3Trainer` follows the same rollout and optimization
path. Each actor additionally receives an 18D boundary feature vector through a
trainable encoder that conditions its 28D input representation. Features include
the detection latch, executor target-knowledge/handoff state, real
executor/searcher relative positions and velocities, locally known target delta,
role, completion state and hold progress.

Detection is represented by persistent searcher target knowledge, rather than a
transient one-step event. Unknown target coordinates remain masked for each
actor. B3 uses joint-observation side information for relative agent state; this
is its explicit information-conditioning intervention. Raw environment and
replay observations stay 28D, actions stay 3D, and critics stay 124D. Replay actor
updates use current-state boundary features; target policies use next-state
features. Encoder parameters participate in actor optimization and target updates.

Neither B2 nor B3 imports or calls an HGR model, estimator, trainer or runtime.

## Commands

Run from the repository root in the AUV Python environment after source-gate
approval. Preflight constructs no model and writes no output:

```sh
python -B -m tools.ch3_baselines.run_training --baseline B2_direct_mc --check-only
python -B -m tools.ch3_baselines.run_training --baseline B3_direct_boundary --check-only
```

Manual training commands (these were not executed as formal experiments):

```sh
python -B -m tools.ch3_baselines.run_training --baseline B2_direct_mc --episodes 1000
python -B -m tools.ch3_baselines.run_training --baseline B3_direct_boundary --episodes 1000
```

Default seed is 2729, with checkpoints every 100 completed episodes and a final
checkpoint. Output directories must be new or empty:

```text
outputs/chapter3/baselines/collision_terminal/B2_direct_mc_train_seed2729_v1/
outputs/chapter3/baselines/collision_terminal/B3_direct_boundary_train_seed2729_v1/
```

Use the same frozen validation manifest for both evaluations. Replace
`VALIDATION_MANIFEST.json` below with the actual manifest containing at least
100 validation scenarios; the evaluator never generates evaluation scenes:

```sh
python -B -m tools.ch3_baselines.run_baseline --baseline B2_direct_mc --checkpoint outputs/chapter3/baselines/collision_terminal/B2_direct_mc_train_seed2729_v1/checkpoints/B2_direct_mc_final.pt --manifest VALIDATION_MANIFEST.json --episodes 100 --seed 12729 --output-dir outputs/chapter3/baselines/collision_terminal/B2_eval100_seed12729_v1
python -B -m tools.ch3_baselines.run_baseline --baseline B3_direct_boundary --checkpoint outputs/chapter3/baselines/collision_terminal/B3_direct_boundary_train_seed2729_v1/checkpoints/B3_direct_boundary_final.pt --manifest VALIDATION_MANIFEST.json --episodes 100 --seed 12729 --output-dir outputs/chapter3/baselines/collision_terminal/B3_eval100_seed12729_v1
```

Evaluation uses deterministic actors without exploration or optimizer updates.
It checks the independent checkpoint schema, baseline/method/algorithm identity,
configuration hash, complete production inventory, model state, optimizer state,
and completed-episode counters. HGR and cross-baseline relabeling are rejected.
Checkpoints contain real actor/critic/target parameters and Adam state. Replay is
not serialized; automatic resume is deliberately unsupported.

Training records per-episode outcomes, team returns, replay size, environment
steps and actual optimizer counts, plus per-agent actor/critic losses. A step
budget finishes an already-started episode before stopping and explicitly
reports any overshoot; a partial requested episode count is not marked complete.

## Verification scope

Latest recorded local checks belong in `verification/`; they are not CI or
thesis performance evidence. The new mechanism tests use temporary four-step
missions with small replay/network settings to exercise real learning and
checkpoint round trips. Existing B0/B1 smoke tests retain their full common
task contract. No 1000-episode run, convergence claim, Phase 1C completion or
formal comparison is established by these checks.
