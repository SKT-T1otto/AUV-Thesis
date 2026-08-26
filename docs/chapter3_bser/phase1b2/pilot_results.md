# BSER Phase 1B.2 pilot results

Protocol: `M20_MOVING_UNKNOWN_MULTI`, seeds 2729–2733, episode indices 0–3, `max_steps=400`, four methods, 80/80 condition-episodes, 0 failures. Path tracking was enabled for every method so the event-mechanism comparison uses a common execution layer.

## Task and execution metrics

| Method | Success | Found | Completion (success only) | Completion (truncated) | Invalid | Stale | Collision | Path error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-BSER-static | 0.15 | 0.45 | 261.33 | 379.20 | 0 | 0 | 489 | 4.328 |
| Periodic-BSER | 0.20 | 0.45 | 321.25 | 384.25 | 0 | 0 | 687 | 2.336 |
| Event-BSER-phase1b1 | 0.10 | 0.40 | 347.00 | 394.70 | 7,495 | 7,677 | 0 | 3.333 |
| Event-BSER-phase1b2_corrected | 0.20 | 0.55 | 319.00 | 383.80 | 1,336 | 740 | 1,558 | 2.445 |

## Planning and stability metrics

| Method | Accepted | Rejected | Optimizer calls | Mean replans | Switches | Switch distance |
|---|---:|---:|---:|---:|---:|---:|
| No-BSER-static | 0 | 0 | 0 | 0.00 | 0 | 0.00 |
| Periodic-BSER | 383 | 0 | 383 | 19.15 | 746 | 3,796.70 |
| Event-BSER-phase1b1 | 187 | 7,638 | 1,401 | 9.35 | 213 | 1,953.27 |
| Event-BSER-phase1b2_corrected | 148 | 7,528 | 1,696 | 7.40 | 168 | 1,712.04 |

## Mechanism metrics

| Method | Route-impact ratio | Partial attempts | Partial accepted | Partial success rate |
|---|---:|---:|---:|---:|
| No-BSER-static | 0.000 | 0 | 0 | 0.000 |
| Periodic-BSER | 0.000 | 0 | 0 | 0.000 |
| Event-BSER-phase1b1 | 0.553 | 1,230 | 65 | 0.0528 |
| Event-BSER-phase1b2_corrected | 0.649 | 1,593 | 76 | 0.0477 |

## Same-protocol Phase 1B.1 comparison

| Metric | Phase 1B.1 | Phase 1B.2 | Change |
|---|---:|---:|---:|
| Success rate | 0.10 | 0.20 | +0.10 |
| Found rate | 0.40 | 0.55 | +0.15 |
| Truncated completion | 394.70 | 383.80 | -10.90 |
| Executor invalid | 7,495 | 1,336 | -82.2% |
| Waypoint stale | 7,677 | 740 | -90.4% |
| Accepted replans | 187 | 148 | -20.9% |
| Rejected replans | 7,638 | 7,528 | -1.4% |
| Optimizer calls | 1,401 | 1,696 | +21.1% |
| Mean replans | 9.35 | 7.40 | -20.9% |
| Path error | 3.333 | 2.445 | -26.6% |
| Switch distance | 1,953.27 | 1,712.04 | -12.4% |

All user-specified minimum gates pass. Optimizer-call overhead and collision accumulation remain explicit Phase 1C monitoring risks.
