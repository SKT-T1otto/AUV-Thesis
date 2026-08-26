# Phase 1B event model

`EventDetector` is deterministic for identical pairs of `PlanningStateView` snapshots and uses no environment private state or target ground truth.

The enum contains seven values. Six are semantic online events and the seventh is the periodic comparison control:

| Event | Detection rule | Effect |
|---|---|---|
| `BELIEF_SHIFT` | `0.5 * L1(b_t,b_{t-1}) + 0.5 * abs(H_t-H_{t-1}) > 0.15` | Ordinary replan candidate |
| `OBSTACLE_DISCOVERED` | Newly occupied probability mass exceeds `0.5` | Critical immediate replan |
| `TARGET_FOUND` | `False -> True` transition of public task state | Freeze search and reassign executor |
| `TARGET_LOST` | `True -> False` transition | Ordinary replan candidate |
| `EXECUTOR_INVALID` | Belief-peak route is unreachable under `TravelCostService` | Ordinary replan candidate |
| `WAYPOINT_STALE` | Search waypoint absent, unreachable, or within stale-distance tolerance | Ordinary replan candidate |
| `PERIODIC_REFRESH` | Configured step modulus | Periodic baseline/control event |

Diagnostics record previous/current entropy, belief L1 distance, combined belief score, newly occupied cell count, new obstacle probability mass, positive risk change, executor reachability, and stale searcher IDs.

Obstacle discovery compares only `OccupancyBeliefView` snapshots. Target-found executor reassignment uses the current belief peak or handoff-compatible public state; it never reads the true target position.

All thresholds are declared in `configs/chapter3/bser_phase1b.json`. They were frozen before E2 and were not adapted to its results.
