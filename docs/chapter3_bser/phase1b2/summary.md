# BSER Phase 1B.2 summary

Status: **PASS_BSER_PHASE1B2**. The execution-consistent interface gate passed and Phase 1C is eligible. No Phase 1C training was started by this task.

## Implemented closure

- Added a stateful `PathTracker` so planned path points are consumed in order before the final waypoint.
- Made `WaypointManager.current_assignment` the canonical controller, detector, and diagnostic waypoint source for Phase 1B.2.
- Redefined `EXECUTOR_INVALID` against the installed executor assignment, public reachability, and relative planning-cost increase; belief-peak changes no longer invalidate the assignment.
- Replaced the affected-agent highest-belief partial shortcut with local candidate regeneration followed by the frozen `solve_joint_greedy()` and original Phase 1A.1 objective.

## Formal pilot decision

The formal CPU pilot completed all 80/80 condition-episodes (5 seeds × 4 episode indices × 4 methods), with 0 execution failures, no checkpoint loading, no training, and no oracle access.

| Required gate | Result | Pass |
|---|---:|---:|
| Corrected success rate ≥ 0.15 | 0.20 | yes |
| Executor invalid decreases | 7,495 → 1,336 (-82.2%) | yes |
| Waypoint stale decreases | 7,677 → 740 (-90.4%) | yes |
| Accepted replans do not increase | 187 → 148 (-20.9%) | yes |
| Rejected replans do not increase | 7,638 → 7,528 (-1.4%) | yes |
| Mean replans ≤ 20 | 7.4 | yes |
| New tests pass | 5/5 | yes |

Under the same execution-consistent protocol, Phase 1B.2 improved success from 0.10 to 0.20 and found rate from 0.40 to 0.55 versus Event-BSER Phase 1B.1. Mean truncated completion improved from 394.7 to 383.8 steps, path-tracking error from 3.333 to 2.445, and total switch distance from 1,953.27 to 1,712.04.

The earlier Phase 1B.1 delivery reported static success 0.20 and corrected success 0.00. In this Phase 1B.2 execution-consistent rerun, static success is 0.15 and corrected success is 0.20. This separates the path-execution change from the event-mechanism comparison; the same-protocol Phase 1B.1 row remains the primary causal baseline.

## Engineering validation and scope

- New Phase 1B.2 tests: 5/5 passed in 0.092 s.
- Complete active suite: 83/83 passed in 412.140 s; 4 superseded legacy evidence tests were excluded by the existing runner.
- Frozen `objective.py`, `greedy_solver.py`, `candidate_generator.py`, and `exact_solver.py`: 0 diff.
- `core/`, Chapter 4, and Chapter 5 production code: 0 diff.
- `docs2/phase1b1` was preserved; all new experiment output is under `docs2/phase1b2`.

## Residual risks for Phase 1C monitoring

- Optimizer invocations increased from 1,401 to 1,696 (+21.1%) even though accepted and rejected replans both decreased. This is compute overhead, not additional executed replanning.
- Corrected cumulative collision flags were 1,558 versus 0 for the same-protocol Phase 1B.1 event row. Collision semantics were not changed; Phase 1C should monitor this metric explicitly.
- The highest corrected tail recorded 283 executor-invalid detections and 293 optimizer invocations in a seed-2733 episode, while accepting only 7 replans. Hysteresis limited execution churn, but event-query overhead remains.

These risks do not violate the user-specified Phase 1B.2 minimum gates. They should be treated as monitored entry conditions, not as permission to change frozen BSER theory or environment semantics.
