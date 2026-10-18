# D2 second-round exact performance source review — 2026-10-10

The user's second-round Windows optimization request authorizes these exact
performance changes and a new source identity. The starting commit is
`c468663acf05fa11b41eb068169ae44cc79a2376`; the initial working tree was clean.
No historical manifest, formal output, checkpoint or experiment definition is
rewritten. This is bounded local verification, not CI or thesis acceptance.

## Retained implementation

The 100-step profile identifies the environment planner's _core_cell_path
(27.92 s inclusive), _edge_time (14.55 s) and _physical_edge_time (8.15 s)
as the relevant repeated work. TravelCostService A* uses only 0.34 s.

The only retained algorithm-runtime edits are in OnlineUnknownMapTaskPlanner:
CPU float32/float64 grid and occupancy tensors are read through live NumPy
views. The views are obtained anew per call, so both in-place mutation and
replacement tensors are observed. Native-precision displacement subtraction
produces the same values supplied to the EXISTING bounded physical-time cache;
its original Torch norm, role arithmetic and returned float values are retained.
Occupancy reads produce the same Python floats as Torch scalar .item().
The probability mean, entropy, risk and planning-cost expression is unchanged.
CUDA and other dtypes retain the original Torch indexing path. No new cache,
rounding, cost approximation or probability snapshot is introduced.

The base-graph candidate was tested and reverted: only 3 of 59 graph builds
could reuse its full content, and paired rollout throughput improved 0.86%.
planning_graph.py is restored byte-for-byte; topology, components, graph hash,
endpoints, TravelCostService and all A* loops remain the original implementation.

## Diagnostic-only changes

The benchmark reports topology and component timings in addition to graph,
adjacency, endpoint, filter, query and A* inclusive/self timings. Opt-in
cProfile supports bounded skip/measurement windows, heuristic and connector
timers, and A* expansion/repeat diagnostics. The adjacency observer delegates
the original rows without changing neighbor order. Detailed instrumentation
is excluded from throughput runs and is never imported by training. Existing
bounded cache statistics are read only by the diagnostic command.

## Historical provenance

The new exact successor follows the frozen `d2_performance_v1_evolution.json`.
Every parent profile and before hash is validated. The existing framework
gate dispatches to that successor; no source directory, hash mismatch or
historical record is exempted. All 27 Phase 0B-2 records stay present and pinned.
Regression coverage includes corruption of each changed/added path, the
review document, parent bytes, unused profiles and manifest scope.

## Experiment boundary

Reward, networks, optimizer, replay, warmup, update cadence, RNG consumption,
28D/3D/124D contracts, 400-step horizon, collision/team protocols, moving
targets, safety frequency, refresh triggers, route requests, recovery and
failure semantics stay unchanged. Exact before/after trajectories, rewards,
model/optimizer hashes and summaries are required before accepting a change.
Linux remains untouched; its final CPU1 A/B test is left to the user. No
400000-step experiment, HGR long training, commit or push is part of this work.

Timing evidence and the phase retention decision are recorded separately.
