# BSER Phase 1B.1 mechanism changes

Phase 1B.1 adds the opt-in `phase1b1_corrected` mechanism while retaining `phase1b_v1` unchanged as a selectable compatibility path.

- Online decisions receive an immutable `OnlineMissionContext` constructed from `get_task_state()`, `get_search_execution_state()`, and `PlanningStateView` only.
- `EXECUTOR_TARGET_RECEIVED` represents the public false-to-true executor handoff transition. `TARGET_FOUND` no longer fabricates an executor route before public handoff.
- Executor target priority is public handoff target, current valid route, then reachable belief fallback.
- Obstacle events are filtered by active path/corridor impact before an optimizer call.
- Route-invalidating events replan only affected agents. A partial result is installed atomically or rejected in full.
- Event-specific cooldowns, relative objective gain, and a 1.0 m waypoint switch floor replace the single global corrected-policy gate.
- Step diagnostics record event, impact, scope, optimizer call, acceptance, rejection, changed agents, target source, and switch distance.

No reward, observation, action, mission-success, target-motion, disturbance, communication, Chapter 4, or Chapter 5 semantics are changed.
