# core

The sole shared executable production package. It contains the Chapter-3
compatible mission environment, configuration, moving-target dynamics, unknown
mapping, online planning, fixed reliable communication, MADDPG, replay,
scenario generation, runtime/training helpers, evaluation boundaries, and
provenance support.

Public entry points include `core.env.MissionCoreEnv`, the deterministic
scenario APIs under `core.scenarios`, and training/runtime components under
`core.algorithms`, `core.replay`, and `core.runtime`.

`evaluation` and `wrappers` remain shared extension boundaries. CH4 disturbance
and CH5 acoustic-communication implementations have not been migrated.
