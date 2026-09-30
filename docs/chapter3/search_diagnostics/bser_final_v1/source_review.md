# BSER final v1 source review

Authorization: the 2026-09-29 user requested implementation of the previously
discussed F0--F6 experiment and manual Windows launch interfaces. The user will
run complete evaluations and decide on retention after inspecting results.
No training, checkpoint restore, commit, push or formal evaluation is authorized
as an automatic verification step.

## Scope and preservation

All new executable logic is opt-in under
`chapter3_bser/experiments/bser_final_v1/`. The sole prior-source edit is the exact
successor dispatch in `safe_search_v1/provenance.py`. The frozen D source,
SafeSearch v1/v2 algorithms, original manifests and all 27 migration records
remain byte-identical. The successor manifest validates exact additions and
before/after hashes for all three checkout profiles. Tests copy and check the
new seal as well as every earlier seal; no old validation is skipped.

F0 calls the original EffectRuntime D2 allocator, controller, V4 bridge and
physics. Read-only solve timings are collected around its allocation methods.
F1--F6 use those same candidates, partial-replan constraints, events, cooldowns,
waypoint stabilization, zero 4x3 residuals and post-Found execution. The 28D
observations and 124D critic contracts are untouched. No core file is changed.

## Algorithm review

* Each eligible solve constructs one original D2 candidate context and greedy
  search/then-standby reference. The reference and every alternative are
  previewed through the original pure WaypointManager stabilization. D2 standby
  re-selection after stabilization is reproduced. Direct recovery is kept
  distinct because its original call chain does not stabilize a second time.
* Forecasts clone chapter guidance/tracker data, not the simulator. A stub owner
  isolates SafetyBridge route-repair caches and counters. Only public map,
  kinematics, public mission flags and white-listed vehicle constants are read.
  Search and executor prior strengths are read separately.
* The 20-step model keeps the map, public belief and high-level path fixed.
  It follows the original 0.75 tracker, native current-position Hold and zero
  residual dynamics. Initial flow estimate is zero; later XY velocity residuals
  estimate disturbance. Saturation makes this only an approximation; nonfinite
  or over-limit estimates are rejected. Nominal and four XY perturbations are
  screening samples, not a robust physical guarantee. No R2/R4 braking or Hold
  repair is activated.
* Detection uses cumulative minimum distance to predicted future swept segments,
  equivalent to a per-agent temporal maximum of the existing Gaussian kernel.
  Initial coverage is subtracted. Independent temporal re-observation trials
  are not multiplied. Across searchers the original complement-product union
  is retained. Near score is the mean new cumulative covered belief over 20 ticks.
* Dynamic response weights incremental new detection at tick t by response time
  from the executor's predicted location at t. Search and migration run in
  parallel. A public geometry-checked connector adapter uses the same linear
  connector metric as the public graph extractor, then the unchanged graph
  TravelCostService. At most 128 connector candidates are inspected; incomplete
  searches cause model rejection, not a physical-unreachable assertion.
* The response graph has the common public safety filter. F4 uses the same
  graph/connector service, but the old full-path static standby formula. This
  isolates the response scoring model. F2 instead retains D2's native sequential
  standby selection and has no response score in its acceptance/ranking.
* Public risk is per-agent maximum over forecast samples of known blocked
  transition count, unknown midpoint-weighted travel length, and maximum
  departure from remaining route. Hold departure measures drift from its start.
  Relative guards forbid cross-agent or cross-component risk compensation.
  They do not certify safety of D2 or absolute collision avoidance.
* F1/F4/F5/F6 require original full-path score and near score no lower than D2.
  F3 removes only those two guards. The legacy full-path guard intentionally
  preserves D2's original metric; only temporal scoring excludes traversed path.
  All response-enabled arms also require nondecreasing response and at least
  3% near gain or 5% response gain. F2 requires 3% near gain. Denominators at or
  below 1e-10 fall back. Response, then near, then full score rank accepted joint
  alternatives; ties use candidate IDs. F2 ranks near, then full.
* Enumeration covers the current candidate pool, at most 256 combinations; it
  is not a global continuous optimum. F6 uses a single coordinate-greedy sweep
  per standby with the same scores/guards, seeded from that event's D2 search
  selection, so solver comparison does not handicap the greedy initialization.
  Work ceilings are 24 independent route forecasts and 80 response queries.
  Budget exhaustion discards prefix winners and returns D2, independent of CPU
  speed. Forecast/response caches are scoped to one public planning event.

## Evidence and operating controls

The runner defaults to plan-only, never loads a policy, and executes only with
`--execute`. Output is restricted to a new subdirectory of runs/. Resume verifies
all source/input identities, actual terminal results, trace continuity and raw
audit hashes. Program failure is not converted to collision or timeout. Existing
retained runs and outputs are never rewritten. Fresh confirmation preparation
uses a distinct generator seed, validates no overlap with the original 100 by
ID, seed and content, and freezes source identity and population before launch.
This does not assert independence from every historical training dataset.

Primary metrics: Found, pre-Found collision, failure-penalized found steps.
Comparisons pair whole scenarios. Conditional Found/response/success times stay
secondary. Development and confirmation labels are separate. Seven comparisons
remain exploratory and no automatic efficacy/retention claim is produced.
Local verification is recorded separately; a passing fixture is not evidence
of improved Found performance or real-time deployment suitability.
