# Phase S2-A.1: Public Local Start-Connector Recovery

S2-A.1 is an evaluation-only Searcher guidance overlay. It does not change the
PRRAC network, observation/action contracts, Actor residuals, Native B1
Executor, reward, replay, training, allocation identity, or post-found control.

The v1 variants and schema remain available unchanged. S2-A.1 uses
`bser.phase1c.prrac.search_collision_recovery.v2` and the variants
`S2A1_C0_BASELINE`, `S2A1_C1_FORCED_REFRESH`, and
`S2A1_C2_LOCAL_CONNECTOR`. The recovery schema and config hash are part of the
completed-combination key, so v1 results cannot be resumed as v2 results.

On a pre-found SEARCH collision edge, C1 requests one forced snapshot from the
public `OnlinePlanningStateProvider` and queries the original semantic waypoint
without changing its candidate identity. C2 first performs that same C1 query.
If it is unreachable, or refreshed guidance remains ineffective at the next
collision transition, C2 builds a continuous local connector using only the
public grid and occupancy belief. Candidate order is last collision-free anchor,
known-free valid graph cells, non-occupied valid cells, then deterministic
reverse-direction endpoints. Every segment is sampled at no more than half the
minimum positive grid spacing and is rejected if public occupied cells appear
after the first collision sample.

At the local endpoint, C2 forces another public refresh and queries the original
semantic waypoint. Failed endpoints are excluded deterministically. If no local
plan exists, the attempt ends as `RECOVERY_FAILED_PASS_THROUGH`; a plan-less
state is never counted as effective recovery time. Found or mission completion
immediately clears all Search recovery state.

Activation diagnostics compare base and overlay path hashes, tracking targets,
and final waypoints on each recovery step. The output separates event entry,
planning attempt, generated plan, active plan, changed guidance, endpoint reach,
and graph reconnect. Detailed rows are written to
`search_collision_recovery_activation_steps.csv` under artifact revision
`s2a1.activation_artifact.v1`; resume requires that artifact and deduplicates it
by checkpoint, variant, scenario, step, agent, and attempt. Targeted scenario
lists selected from historical C0 collision rows are diagnostic-only and are
rejected in formal mode. Their requested/generated counts remain visible while
the manifest, progress, summaries, and resolved episode count use the final
selected count.

Use `python scripts/validate_phase1c_prrac_s2a1_activation.py --output-dir
<evaluation-output>` for the strict cross-artifact protocol check. The older
`--summary-csv` mode remains a deliberately weaker activation-summary check.

No checkpoint evaluation is launched by repository tests or launchers unless a
user explicitly invokes a launcher with a checkpoint path.
