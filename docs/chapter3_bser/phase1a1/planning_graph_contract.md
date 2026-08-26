# Immutable planning graph contract

`core.mapping.planning_graph` is the trusted adapter around the frozen planner. It snapshots valid cells, authoritative role-specific edges, planning costs, physical travel times, components, endpoints, and deterministic tie semantics into immutable arrays and tuples. Planner cache and diagnostics are isolated and restored. Unknown-map extraction never reads obstacle truth.
