# Phase S2-A Search Collision Recovery

> This document freezes the v1 experiment. The independent v2 recovery is
> documented in `S2A1_LOCAL_CONNECTOR_RECOVERY_GUIDE.md`; v1 semantics and
> output provenance remain unchanged.

## Scientific hypothesis

S2-A tests whether a public-information-only, post-collision navigation overlay reduces pre-found Searcher collision stalls and improves Found without changing the learned residual policy. The existing collision/failure association is not itself causal; causal interpretation requires the paired C0/C1/C2 runs on one checkpoint and one frozen scenario manifest.

## Variants

- `S2A_C0_BASELINE` is an exact Native B1 Full PRRAC control path. It creates no recovery controller and changes no guidance, action, allocation, path tracker, decision reason, or Executor state.
- `S2A_C1_ROUTE_REFRESH` performs one path refresh to the unchanged semantic SearchAssignment waypoint on each collision edge. A continuous collision streak cannot retrigger it until one collision-free SEARCH transition rearms the detector.
- `S2A_C2_EGRESS_ROUTE` first performs C1. It escalates only when that refresh is unreachable or the next SEARCH transition still collides, then selects a deterministic reachable known-free egress cell and rejoins the latest base search guidance.

Semantic candidate/waypoint identity remains separate from a temporary navigation endpoint. Neither C1 nor C2 changes raw/applied Actor actions, suppresses residuals, touches the Executor, edits `allocation_sha256`, or enters the trainer.

## Information boundary

Recovery reads only `PlanningStateView` public grid/occupancy/graph/agents, public base guidance/SearchAssignment semantics, public collision flags, `TravelCostService`, and `PathTracker`. It does not read true obstacles, true target position, oracle validity, reward, success, future trajectories, evaluation outcomes, or scenario-specific exception lists.

## Run and regression gate

The launchers default to 10 scenarios and require explicit `--formal` for the configured 100 scenarios. They evaluate `full_prrac`, Native `B1_ATOMIC_LAST_VALID`, and C0/C1/C2; they never train or resume a checkpoint.

Before interpreting C1/C2, C0 must reproduce the fixed Ep100 Val100 baseline: 100 rows on the expected manifest, Found 48/100, Success 26/100, and the frozen Native B1 provenance. Run `validate_s2a_baseline_regression` after the formal evaluation. A FAIL preserves all results but prohibits interpreting C1/C2.

## Outputs and statistics

The evaluator writes recovery episode/summary CSVs and JSON, three pairwise comparisons, C0-defined baseline collision/no-collision strata, a deterministic failure funnel, failure traces, and nine fixed-order plots. Exact McNemar tests cover Found, Success, Contact, and pre-found collision episodes. Continuous paired differences are always `right - left`.

Baseline strata are defined exclusively from C0 pre-found collision status; using C1/C2 post-treatment collision status for stratification is prohibited. The no-collision stratum separately checks preservation of Found and Success.

Even a positive C2 result still requires S2-B before any training integration. This phase contains no Executor collision/start-connector recovery, terminal intercept controller, residual projection, objective changes, candidate-pool changes, or formal training.

## Single-line launch examples

Windows smoke: `powershell -ExecutionPolicy Bypass -File scripts/run_phase1c_prrac_s2a_collision_ablation.ps1 -Checkpoint C:\path\episode_100.pt -OutputDir C:\path\s2a_smoke -Workers 4 -Episodes 10`

Linux formal: `bash scripts/linux/run_phase1c_prrac_s2a_collision_ablation.sh --checkpoint /path/episode_100.pt --output-dir /path/s2a_formal --workers 4 --formal`
