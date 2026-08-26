# env

Canonical shared mission environment. `core.env.MissionCoreEnv` is the stable
facade over `core.env.uav_env.UAVEnv`; task-state views and the frozen 28-D
observation contract are exposed from this package. Environment semantics are
frozen by the Phase 0B-2 E0 evidence and contract tests.
