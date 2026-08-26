# Phase 0B-1 summary

Final status: **PASS_CH3_E0_EQUIVALENCE**

Git: initialized on branch `main`. The three planned local commits were created for the accepted baseline, compatible mission core, and E0 equivalence delivery; the post-commit worktree is validated separately as clean.

Migrated CH3 authority source files: **27**. The byte-identical snapshot includes environment/config/runtime, target motion, map/planner, metrics, communication, algorithm/runtime dependencies, registry, provenance utilities, training import closure, and the scenario builder. Historical data, checkpoints, logs, results, CH4, and CH5 were not migrated.

The local observation contract is `[28,28,28,28]` in canonical role order `search_fast`, `search_balanced`, `search_precise`, `executor`; actions are four continuous 3D residual accelerations in [-1,1], and reward shape is `[4]`.

Profiles: M00 clear lower boundary; M10 single-obstacle medium case; M20 unknown multi-obstacle canonical thesis profile; M90 known-map oracle upper boundary.

E0 completed/passed: **60/60 of 60**. Maximum floating-point difference: **0.0**. Task-event mismatches: **0**.

Legacy accepted snapshot read-only: **true** (CH3/CH4/CH5 final full-baseline check).

`allow_bser_phase1a = true`
