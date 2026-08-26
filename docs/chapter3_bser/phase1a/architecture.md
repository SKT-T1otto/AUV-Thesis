# Phase 1A architecture

Phase 1A adds two independent public services without changing any of the 40
pre-existing core Python files. `planning_state.py` copies the current public
mission/agent state and the planner-maintained belief and occupancy arrays into
frozen dataclasses. `travel_cost_service.py` performs deterministic, pure A*
and Dijkstra queries over that snapshot; it does not call the mutable legacy
planner cache.

The chapter-specific `chapter3_bser` package consumes only the immutable view.
Candidate construction feeds a fixed detection model and an objective context;
exact, standard greedy, and lazy greedy solvers share that context. Baselines
reuse the same belief and detection model. The E1 experiment runner is the only
layer that reads the frozen E0 reset manifests, and reset data is given directly
to the environment rather than exposed to BSER.

This is an offline finite allocation architecture. It does not inject actions,
change waypoints, trigger online reallocation, or update RMADDPG.
