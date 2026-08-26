# BSER Phase 1B.3A diagnostic findings

Status: **PASS_PHASE1B3A_DIAGNOSIS**.

This is a diagnosis-only result over the local Phase 1B.2.1 working version. It does not implement Phase 1B.3B, authorize Phase 1C training, or authorize a new 80-condition experiment. No algorithm mechanism, configured threshold, cooldown, action/control semantics, or Phase 1A.1 theory code was changed.

## Case completion and behavior preservation

All 5/5 preregistered `Event-BSER-phase1b2_corrected` condition-episodes completed. `failure_cases.csv` is empty. Every frozen behavior field matches `baseline_cases.csv`; float comparisons used absolute tolerance `1e-9`.

| Case | Success | Found | Executor arrival | Completion | Collisions | Executor invalid | Waypoint stale (boolean) | Optimizer | Accepted | Rejected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2729/2 | false | - | - | - | 0 | 383 | 399 | 37 | 1 | 399 |
| 2731/1 | true | 147 | 156 | 363 | 0 | 361 | 139 | 21 | 3 | 360 |
| 2731/3 | false | - | - | - | 152 | 398 | 388 | 22 | 2 | 398 |
| 2732/3 | true | 294 | 295 | 376 | 0 | 361 | 279 | 26 | 6 | 365 |
| 2733/0 | false | 1 | - | - | 0 | 399 | 0 | 21 | 2 | 398 |

## Executor invalid semantics

The five cases contain 1,902 `EXECUTOR_INVALID` events. The exclusive primary partition is:

- `INSTALLED_ASSIGNMENT_INVALID`: 694 (36.4879%).
- `HARD_ROUTE_INVALID`: 1,183 (62.1977%).
- `SOFT_RESPONSE_DEGRADED_ONLY`: 25 (1.3144%).
- `DIAGNOSTIC_INCONSISTENCY`: 0 (0%).

The overlapping flags are: current query unreachable 1,856 (97.5815%), installed assignment marked unreachable 694, nonfinite installed/current cost 1,877, newly occupied installed-path cell 40, component change 40, and reachable-query relative cost increase above the existing 15% threshold 25. Thus the high event volume in these five cases is not primarily soft cost degradation. The two successful episodes still contain 361 invalid events each: 2731/1 has 215 installed-invalid, 140 hard, and 6 soft; 2732/3 has 81 installed-invalid and 280 hard. An invalid event is therefore not equivalent to mission failure.

## Waypoint stale semantics

The detector emitted 1,205 boolean `WAYPOINT_STALE` step-events. Per-searcher expansion produced 3,562 event instances:

- `FINAL_WAYPOINT_REACHED`: 0 (0%).
- `HARD_PATH_INVALID`: 3,562 (100%).
- `NO_ACTIVE_ASSIGNMENT`: 0 (0%).
- `DIAGNOSTIC_INCONSISTENCY`: 0 (0%).

All 3,562 instances have an unreachable final-waypoint query; 342 also contain a newly occupied cell on the remaining installed path, and 47 also have the current local tracking point within the existing tracking threshold. No final-waypoint-arrival instance was observed, so repeated final-waypoint-arrival triggers are zero.

## Mission phase distribution

- `SEARCH`: executor hard 1,180, executor soft 24, waypoint hard 3,562.
- `WAIT_PUBLIC_HANDOFF`: executor soft 1.
- `EXECUTE_PUBLIC_TARGET`: executor installed-invalid 692, executor hard 3.
- `DONE`: executor installed-invalid 2.

All waypoint-stale instances occur during `SEARCH`.

## Duplicate event keys

Event keys are counted within each episode because `assignment_version` restarts at 1 per episode. Keys are diagnostic only and suppress no events.

- Executor: 1,807/1,902 repeated same-key events (95.0053%); 1,807 consecutive repeats; maximum consecutive run 241.
- Waypoint: 3,370/3,562 repeated same-key agent-events (94.6098%); 3,370 consecutive repeats; maximum consecutive run 20.

## Replanning rejection and public target audit

Across the five cases there are 127 optimizer invocations, 14 accepted replans, and 1,920 actual decision rejections. `ATOMIC_REJECT_MISSING_LOCAL_CANDIDATES` occurs 63 times, all during `SEARCH` with `allocation_scope=executor_only`.

There are three public-target receipt events. Across 695 `EXECUTE_PUBLIC_TARGET` post-decision steps, the executor source is `PUBLIC_HANDOFF_TARGET` in all 695. Public-target lock violations are 0, and standby sources after handoff are 0.

Case 2733/0 found the target at step 1 and received the public target, but never arrived or completed. It emitted 399 executor-invalid events: 398 installed-assignment-invalid and one hard-route-invalid. Its rejection totals are 378 cooldown and 20 no-assignment-change decisions. The failure is an unavailable installed public route, not a missing handoff or target-source switch.

## Collision-tail case 2731/3

All 152 collision rows (steps 245-400) belong to agent 1. The tracker remains on local target `[17.0,7.0,5.5]` with `next_index=4` and six remaining path points. Remaining path length is 17.0533-17.1755 and cross-track error is 4.0597-4.0794. None of the 152 collision rows contains a newly occupied remaining-path cell. However, the final-waypoint query is unreachable throughout the stale diagnostics, so the remaining geometric segment is not newly obstacle-invalidated but the route is not currently reachable. The collision tail is the agent stuck far from one unchanged local tracking point.

## Tests, integrity, and next-stage rules

Specialized tests: 12/12 passed. Active suite: 99/99 passed, with four declared superseded historical tests excluded; failures and errors are zero. CH3, CH4, and CH5 tree hashes match the pre-task snapshot exactly.

Preregistered rules 2, 4, 5, and 7 trigger recommendations for Phase 1B.3B: versioned executor-event deduplication; a separate `SEARCH_WAYPOINT_UNREACHABLE` event; executor-standby replanning independent of search candidates; and non-failure wording for reachable soft degradation. Rules 1, 3, and 6 do not trigger.

No Phase 1B.3B mechanism was implemented. No objective, candidate set, joint greedy selection, standby-point generation, threshold, cooldown, state-refresh interval, action, reward, observation, target, obstacle, mission-success, handoff, or Phase 1A.1 theory code was changed.
