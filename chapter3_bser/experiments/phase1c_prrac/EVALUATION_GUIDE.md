# PRRAC deterministic checkpoint evaluation

This evaluator is read-only. It accepts only
`bser.phase1c.prrac.training_state.v1` checkpoints, constructs a fresh
`PRRACMADDPG`, loads the checkpoint with `weights_only=True`, and performs
deterministic rollouts with `explore=false`. It never creates replay storage or
calls an optimizer, learner update, target update, training entry point, or
checkpoint save operation.

## Fixed protocol

- Profile: `M20_MOVING_UNKNOWN_MULTI`
- Split: `validation`
- Scenario generator seed: `1729`
- Episodes: `100`
- Maximum steps: `400`
- Default mode: `full_prrac`
- Environment chain: `MissionCoreEnv -> GuidedEnv -> Phase1CV2TrainingEnv -> PRRACTrainingEnv`

`evaluation_manifest.json` freezes the generated scenarios and its SHA-256.
Every checkpoint and mode receives the same ordered scenario list. Evaluation
workers use the spawn multiprocessing context and exchange only NumPy arrays or
ordinary pickleable Python data.

## Manual invocation

Linux:

```bash
scripts/linux/run_phase1c_prrac_eval.sh \
  --checkpoint outputs/chapter3/phase1c_prrac/training/checkpoints/phase1c_prrac_episode_0600.pt \
  --output-dir outputs/chapter3/phase1c_prrac/evaluation_v1
```

PowerShell:

```powershell
scripts\run_phase1c_prrac_eval.ps1 `
  --checkpoint outputs\chapter3\phase1c_prrac\training\checkpoints\phase1c_prrac_episode_0600.pt `
  --output-dir outputs\chapter3\phase1c_prrac\evaluation_v1
```

The default configuration contains no checkpoint, so invoking a launcher with
no checkpoint arguments fails without starting an evaluation.

Use `--resume-evaluation` only with the same output directory and identical
manifest inputs. Completion keys include the absolute checkpoint path,
checkpoint config hash, checkpoint episode, evaluation mode, and manifest
hash. Existing completed combinations are not rerun.

## Modes and information boundary

`full_prrac` is the formal checkpoint-selection mode. The residual-off modes
are selected-checkpoint fault decomposition only. The
`oracle_current_target_diagnostic` mode is privileged and diagnostic-only. It
refreshes the next Actor observation from public guidance before installing
the true-target tracking override. True target data can appear only in failure
trace output and is never appended to the 28D Actor observation, Router,
Experts, Gate, Critic, or replay.

The evaluator reports a recommended checkpoint using the preregistered
lexicographic rule, but `performance_passed` remains `null`. Evaluation results
do not establish thesis performance or training completion.
