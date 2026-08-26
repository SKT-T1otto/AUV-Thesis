# Phase 0B-1 architecture decisions

1. CH3 is the sole semantic authority. CH4 and CH5 are not merged into MissionCoreEnv.
2. Compatibility comes first: 27 required CH3 source files are copied byte-for-byte into `legacy_adapters/ch3_snapshot`; no data, checkpoint, log, evaluation, diagnostic, smoke, cache, or result file is copied.
3. MissionCoreEnv is a thin delegating wrapper. Dynamics, rewards, observations, target motion, belief/occupancy updates, planning, found/handoff/completion events, and fixed reliable communication remain in the snapshot.
4. Public state access uses frozen dataclasses containing copied tuples/scalars, so callers cannot mutate legacy tensors through the stable API.
5. The local observation remains exactly 28 dimensions. Privileged state is a separate accessor and no CH4/CH5 39D communication vector is introduced.
6. M20_MOVING_UNKNOWN_MULTI is the canonical thesis profile; M00, M10, and M90 are lower/medium/oracle boundaries.
7. CH3's manually narrowed `UAVEnv.__init__.__signature__` omits actual unknown-map keyword parameters. The new config adapter reads the real function code arguments plus the public signatures; CH3 is left unchanged.
8. E0 is migration equivalence only. It does not compare algorithm quality, success-rate superiority, or checkpoints.
