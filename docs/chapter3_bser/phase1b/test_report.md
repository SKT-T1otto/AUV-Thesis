# Phase 1B test report

## Local validation

- Nine requested Phase 1B test modules: 9/9 passed.
- Full inherited and current suite: 69/69 passed in 268.519 seconds.
- Formal E2: 100/100 condition-episodes recorded, zero failures.
- Core diff from `chapter3-bser-phase1a1-e1v2-verified-v1`: zero files.
- Frozen Phase 1A.1 asset diff: zero files.
- New Phase 1B source scan for `get_target_state`, `unwrapped`, `target_position`, and `target_truth`: zero matches.
- Formal training: not run.

The tests cover deterministic event detection, belief/obstacle/target transitions, hysteresis, suppression of overplanning, online allocation, selective waypoint changes, executor reassignment, and repeated-state allocation hash equality. The full suite also preserves the prior E1 and core contract checks.

## Remote clean-clone reproduction

- Clone path: `C:\tmp\CH_BSER_PHASE1B_REMOTE_VERIFY_4063a23`.
- Checked-out commit: `4063a2382f984841410fd8651ce3e5755b93a3fe`.
- Initial clone status: clean.
- Full suite: 69/69 passed in 333.885 seconds.
- Independent formal E2: 100/100 condition-episodes, zero failures.
- E2 deterministic summary SHA-256: `9874e3c3a0a84567c21a964e74567b3b58f4cce925f975e1d7c79a0b0b64743a` (identical to local).
- Eight deterministic result files were byte-identical by SHA-256.
- `episode_metrics.csv` matched for all 100 rows after excluding the intentionally non-deterministic `runtime_seconds` field.

Remote reproducibility passed.
