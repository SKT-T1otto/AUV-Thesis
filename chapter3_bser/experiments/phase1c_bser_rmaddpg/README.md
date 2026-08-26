# Phase 1C BSER-RMADDPG Preflight

This package is the independent, opt-in runtime for method
`ch3_bser_rmaddpg_phase1c`.

It performs rollout-only preflight validation:

- online BSER runs as the high-level event-triggered controller;
- `BSERControlContextV1` and `GuidedEnv` install navigation guidance;
- a randomly initialized RMADDPG actor emits residual actions;
- no replay update, optimizer step, checkpoint save, or training occurs.

Default protocol:

```text
seed=2729
episodes=10
max_steps=400
training_update=false
profile=M20_MOVING_UNKNOWN_MULTI
```

Run from the repository root:

```text
python -m chapter3_bser.experiments.phase1c_bser_rmaddpg.run_phase1c
```

Outputs are isolated under `outputs/chapter3/phase1c_bser_rmaddpg/`:

- `checkpoint/NO_TRAINING.txt`
- `logs/step_metrics.jsonl`
- `logs/failures.json`
- `metrics/episode_metrics.csv`
- `metrics/preflight_summary.json`
- `configs/default_disabled_config.json`
- `configs/resolved_preflight_config.json`

The source config remains disabled by default. Only the in-memory/resolved
preflight copy enables guidance, while `training_enabled` and
`training_update` remain false.

## Short-training entry

The explicit training config is `configs/chapter3/bser_phase1c_train.json`.
It is isolated from the default-disabled integration config and is locked to
one short-training seed. The configured command is:

```text
python -m chapter3_bser.experiments.phase1c_bser_rmaddpg.train_phase1c
```

The bounded training dry-run entry is:

```text
python -m chapter3_bser.experiments.phase1c_bser_rmaddpg.train_phase1c --dry-run
```

Training artifacts are written only below
`outputs/chapter3/phase1c_bser_rmaddpg/training/`.
