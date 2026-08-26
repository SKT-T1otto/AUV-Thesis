# Information boundary v2

Algorithm modules receive only `PlanningStateView`, observable belief/occupancy state, agent state, roles, revisions, knowledge mode, and an immutable planning graph. Scenario identifiers and `obstacle_layout_id` live only in `PlanningSnapshotMetadata` and experiment output. `chapter3_bser` does not access `env.unwrapped` or planner internals. The trusted core adapter is the only planner-semantic extraction boundary.
