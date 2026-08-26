# BSER Phase 1B summary

Status: **PASS_BSER_PHASE1B_ONLINE**.

Phase 1B implements `OnlineBSERController`, information-bounded event detection, executor reassignment, selective waypoint replacement, and replanning hysteresis around the frozen Phase 1A.1 optimizer. It changes no `core/` or Phase 1A.1 asset and performs no training.

## E2 results

The formal run completed 100/100 real condition-episodes with zero runtime failures. No oracle target data was used.

| Method | Success | Completion time (success only) | Target found time (found only) | Replans/episode | Mean replan interval | Executor arrival (observed only) |
|---|---:|---:|---:|---:|---:|---:|
| No-BSER-static | 0.75 | 199.40 | 98.82 | 0.0 | n/a | 8.00 |
| Periodic-BSER | 0.70 | 221.14 | 136.12 | 13.4 | 20.00 | 51.63 |
| Event-BSER | 0.60 | 226.75 | 125.41 | 8.0 | 29.48 | 70.33 |

The frozen E2 run does **not** demonstrate a task-level advantage for Event-BSER over the static comparison; the results are reported without retuning or deleting cases. The online-mechanism result is nevertheless clear: the module executes deterministically, responds to events, and hysteresis prevents near-stepwise replanning.

## Ablations

| Ablation | Success | Completion time (success only) | Replans/episode | Mean replan interval |
|---|---:|---:|---:|---:|
| no belief trigger | 0.60 | 238.33 | 8.0 | 30.69 |
| no obstacle trigger | 0.70 | 264.71 | 4.1 | 47.57 |
| no target trigger | 0.60 | 223.50 | 7.2 | 32.58 |
| no hysteresis | 0.70 | 212.86 | 268.3 | 1.00 |

Across all conditions, failed handoff count was zero. Formal deterministic summary SHA-256: `9874e3c3a0a84567c21a964e74567b3b58f4cce925f975e1d7c79a0b0b64743a`.

## Delivery validation

- Branch: `chapter3-bser-phase1b`.
- Implementation/evidence commit: `4063a2382f984841410fd8651ce3e5755b93a3fe`.
- Added files: 49.
- Changed core files: 0.
- Phase 1A.1 assets changed: no.
- Event enum values: 7 (six semantic events plus the periodic control event).
- Local full suite: 69/69 passed in 268.519 seconds.
- Remote clean-clone full suite: 69/69 passed in 333.885 seconds.
- Remote clone: `C:\tmp\CH_BSER_PHASE1B_REMOTE_VERIFY_4063a23`.
- Remote E2: 100/100 completed, zero failures, summary SHA identical.
- Eight deterministic result files were byte-identical; all 100 episode rows were identical after excluding `runtime_seconds`.

All Phase 1B gates are satisfied. Entry to Phase 1C is allowed; Phase 1C was not started.
