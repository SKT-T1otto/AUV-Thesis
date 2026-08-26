# Online BSER algorithm

## Initialization

`OnlineBSERController.initialize(state)` runs one frozen Phase 1A.1 greedy allocation, initializes the snapshot cache, records the initial replan step, and emits the initial changed waypoints.

## Per-step update

For every environment step:

1. Compare the previous and current `PlanningStateView` snapshots.
2. If no event is present, retain the current allocation without invoking the optimizer.
3. If only ordinary events occur during cooldown, retain the current allocation.
4. Otherwise compute a proposed allocation with the Phase 1A.1 candidate generator, objective, and greedy solver.
5. Apply critical-event or gain/cooldown policy.
6. If accepted, atomically replace the immutable allocation and emit only waypoints that changed.

On `TARGET_FOUND`, search assignments are cleared and the executor is routed to the current target-belief peak with `TravelCostService`. Ground-truth target position is never an input.

`OnlineAllocation` contains search assignments, executor assignment, objective value, expected detection probability, response time, trigger reason, and solver status. Its SHA-256 is derived from a canonical serialization, so identical states and configuration produce identical allocation hashes.

The controller is deliberately high level: low-level control remains outside BSER. The E2 harness uses one fixed, deterministic legal-action adapter for every comparison condition.
