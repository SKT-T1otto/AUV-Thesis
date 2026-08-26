# BSER-E2 experiment protocol

E2 validates online reallocation logic in `M20_MOVING_UNKNOWN_MULTI` without training.

## Frozen setup

- Seeds: 1729, 1730, 1731, 1732, 1733.
- Total: exactly 100 real condition-episodes.
- Maximum steps: 400 per episode.
- Workers: 4 CPU processes.
- Formal training: disabled.
- Target oracle: disabled.
- Threshold adaptation: disabled.
- Failed episodes: retained in `failure_cases.csv`.

## Conditions

| Condition | Episodes |
|---|---:|
| No-BSER-static | 20 |
| Periodic-BSER | 20 |
| Event-BSER | 20 |
| Event-BSER-no-belief | 10 |
| Event-BSER-no-obstacle | 10 |
| Event-BSER-no-target | 10 |
| Event-BSER-no-hysteresis | 10 |

The phrase “100 episodes” is interpreted as 100 total real condition-episodes across the complete method comparison and ablation suite. No trajectory is copied or relabelled to fill another condition. Every condition covers all five fixed seeds.

The same deterministic fixed-action adapter and episode-index dither rule are used for every condition. The primary metrics are success rate, success-only completion time, found-only target-found time, replan count, replanning interval, waypoint switches, observed-only executor arrival time, and failed handoff count. Conditional denominators are explicit in the CSV column names.

Machine-readable protocol and raw results are under `experiments/chapter3/phase1b_online/`.
