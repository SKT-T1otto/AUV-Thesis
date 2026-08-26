# Phase 1B architecture

Phase 1B adds an information-bounded, high-level online replanning layer around the frozen Phase 1A.1 optimizer. It does not modify `core/`, produce low-level velocity/acceleration commands, or train a policy.

The online cycle is:

1. `OnlinePlanningStateProvider` obtains a `PlanningStateView` through the public environment contract.
2. `EventDetector` compares the cached and current snapshots.
3. `ReplanningPolicy` applies critical-event bypass, cooldown, and minimum-gain hysteresis.
4. `BSEROnlineAllocator` calls the frozen Phase 1A.1 candidate generator, objective, and greedy solver.
5. `WaypointManager` emits only changed high-level search waypoints.
6. `ExecutionManager` emits an executor standby/target-region route using `TravelCostService` and the belief peak.

`OnlineBSERController.initialize(state)` returns an `InitialBSERAllocation`. `step(state)` returns a `BSERActionAssignment` containing the event diagnostics, replan decision, immutable allocation, and waypoint changes. The controller never emits an environment action.

The E2-only `action_adapter.py` converts high-level assignments to fixed legal actions so the online logic can be exercised without introducing or training a controller. This adapter is not part of the BSER decision model.

## Frozen boundaries

- `core/`: zero changed files relative to `chapter3-bser-phase1a1-e1v2-verified-v1`.
- Phase 1A.1 configs, reports, and E1 outputs: zero changed files.
- CH4 and CH5: untouched.
- Reward, observation, action, success, target-motion, occupancy, belief, and A* semantics: unchanged.
