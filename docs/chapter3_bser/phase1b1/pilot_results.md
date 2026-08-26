# BSER Phase 1B.1 pilot results

The formal pilot used seeds 2729–2733, episode indices 0–3, four methods, `M20_MOVING_UNKNOWN_MULTI`, and `max_steps=400`. All 80 condition-episodes completed without training, checkpoint loading, or oracle access.

| Method | Success rate | Truncated completion | Truncated target found | Truncated post-found delay | Mean replans | Mean interval | Switches | Switch distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No-BSER-static | 0.20 | 386.85 | 255.15 | 364.85 | 0.00 | 400.00 | 0 | 0.00 |
| Periodic-BSER | 0.15 | 371.20 | 286.90 | 324.10 | 18.55 | 20.00 | 760 | 4132.40 |
| Event-BSER-phase1b_v1 | 0.00 | 400.00 | 315.25 | 364.75 | 9.80 | 40.91 | 407 | 2499.34 |
| Event-BSER-phase1b1_corrected | 0.00 | 400.00 | 311.80 | 388.20 | 8.20 | 49.74 | 160 | 1728.08 |

Corrected invoked the optimizer 1,472 times, accepted 164 replans, and recorded 7,746 rejected decisions. Of 181 obstacle events, 83 affected an active route (45.86%); 25 were explicitly rejected as `OBSTACLE_OFF_ROUTE`. Corrected route-source observations were `BELIEF_PEAK_FALLBACK` 1,181, `NO_VALID_ROUTE` 1,789, `standby:y01` 12, `standby:y02` 1,799, and `standby:y03` 3,219.

Five of six performance gates passed. `success_noninferiority` failed because 0.00 is below 0.15. Final status is `PASS_BSER_PHASE1B1_DIAGNOSTIC_ONLY`; expanded validation is not recommended.
