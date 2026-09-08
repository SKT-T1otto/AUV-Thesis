"""Offline canonical episode analysis of discovery timing and Found geometry.

No runtime imports, training, checkpoint loading or historical writes. The
shared offline helper is a sibling script, imported normally without sys.path
injection. Direct script execution and package imports are both supported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

if __package__:
    from . import analyze_found_executor_distance as geometry
else:
    import analyze_found_executor_distance as geometry


SCHEMA = "ch3.found_timing_failure_attribution.v1"
TIMING_FIELDS = "scenario_id episode_index seed found success contact found_step remaining_steps".split()
GEOMETRY_FIELDS = "scenario_id found_step success executor_distance_at_found".split()
BIN_FIELDS = "bin found_count success_count success_rate".split()
PROBABILITY_FIELDS = "dimension bin distance_min distance_max found_count success_count success_probability population".split()
DOMINANCE_RULE = (
    "Descriptive comparison within all Found episodes. A factor is supported only if both its single-variable "
    "and combined-model beta bootstrap and Wald 95% intervals lie strictly below zero, all required models "
    "converge, and at least 90% of bootstrap fits are valid. Fewer than 5000 repetitions, incomplete data, "
    "or unavailable fits prevent a dominant-factor selection. Unsupported does not establish absence of association."
)


def build_episodes(rows):
    """Keep all original episodes, with explicit NA for discovery in Not Found."""
    ids, indices, result = set(), set(), []
    for source in rows:
        sid = source["scenario_id"]
        index = geometry.integer(source.get("episode_index"))
        if not sid or sid in ids or index is None or index in indices:
            raise ValueError("one row per unique scenario and episode_index is required")
        ids.add(sid)
        indices.add(index)
        if source["evaluation_mode"] != "full_prrac":
            raise ValueError("only full_prrac episodes belong to the canonical population")
        found, success = geometry.boolean(source["found"]), geometry.boolean(source["success"])
        if success and not found:
            raise ValueError("Success without Found")
        horizon = geometry.integer(source["max_steps"])
        if horizon != 400:
            raise ValueError("fixed timing bins require the reviewed 400-step horizon")
        step = geometry.integer(source.get("found_step")) if found else None
        end = geometry.integer(source.get("episode_length"))
        if step is not None and not 1 <= step <= horizon:
            raise ValueError(f"invalid Found step: {sid}")
        if end is not None and (not 0 <= end <= horizon or (step is not None and end < step)):
            raise ValueError(f"invalid episode_length: {sid}")
        contact_raw = source.get("contact_episode")
        contact = None if contact_raw is None or str(contact_raw).strip().lower() in geometry.MISSING else geometry.boolean(contact_raw)
        first_contact = geometry.integer(source.get("first_contact_step")) if found else None
        if first_contact is not None and (contact is False or first_contact < 1 or first_contact > (end or horizon)
                                          or (step is not None and first_contact < step)):
            raise ValueError(f"first_contact_step inconsistent: {sid}")
        remaining = None if step is None else horizon-step
        recorded_remaining = geometry.integer(source.get("remaining_steps_after_found")) if found else None
        if step is not None and recorded_remaining is not None and remaining != recorded_remaining:
            raise ValueError("remaining_steps_after_found mismatch")
        distance = geometry.number(source.get(geometry.DISTANCE_FIELD)) if found else None
        if distance is not None and distance < 0:
            raise ValueError("negative physical distance")
        delay = None if first_contact is None or step is None else first_contact-step
        recorded_delay = geometry.integer(source.get("found_to_first_contact_steps")) if found else None
        if delay is not None and recorded_delay is not None and delay != recorded_delay:
            raise ValueError("found_to_first_contact_steps mismatch")
        result.append(dict(scenario_id=sid, episode_index=index, seed=geometry.integer(source["scenario_seed"]),
                           found=found, success=success, contact=contact, found_step=step,
                           remaining_steps=remaining, executor_distance_at_found=distance, episode_length=end,
                           max_steps=horizon, first_contact_step=first_contact, found_to_contact_steps=delay))
    return result


def population_counts(episodes):
    total = len(episodes)
    masks = {
        "A_not_found": [r for r in episodes if not r["found"]],
        "B_found_but_failed": [r for r in episodes if r["found"] and not r["success"]],
        "C_found_and_success": [r for r in episodes if r["found"] and r["success"]],
    }
    return {name: dict(count=len(rows), percentage=100*len(rows)/total, denominator=total) for name, rows in masks.items()}


def timing_bin(step):
    if not 0 <= step <= 400 or step != int(step):
        raise ValueError("Found timing bin requires integer step in [0,400]")
    return "0-100" if step <= 100 else ("101-200" if step <= 200 else ("201-300" if step <= 300 else "301-400"))


def timing_bins(episodes):
    found = [r for r in episodes if r["found"]]
    if any(r["found_step"] is None for r in found):
        raise ValueError("missing Found step; no partial timing bins")
    result = []
    for label in ("0-100", "101-200", "201-300", "301-400"):
        rows = [r for r in found if timing_bin(r["found_step"]) == label]
        successes = sum(r["success"] for r in rows)
        result.append(dict(bin=label, found_count=len(rows), success_count=successes,
                           success_rate=None if not rows else successes/len(rows)))
    return result


def model_inputs(found, timing_complete, geometry_complete):
    specs = {}
    if timing_complete:
        specs["found_step_model"] = (np.array([[r["found_step"]] for r in found], float), ["found_step"])
    if geometry_complete:
        specs["distance_model"] = (np.array([[r["executor_distance_at_found"]] for r in found], float), ["distance"])
    if timing_complete and geometry_complete:
        specs["combined_model"] = (np.array([[r["found_step"], r["executor_distance_at_found"]] for r in found], float), ["found_step", "distance"])
    return specs


def estimate(found, *, seed, repetitions, timing_complete, geometry_complete):
    """Same scenario resamples for three models and both mean/median differences."""
    specs = model_inputs(found, timing_complete, geometry_complete)
    y = np.array([r["success"] for r in found], bool)
    fits = {key: geometry.logistic(matrix, y) for key, (matrix, _) in specs.items()}
    fit_status = {key: Counter() for key in specs}
    fit_draws = {key: [] for key in specs}
    metrics = {}
    for name, complete in (("found_step", timing_complete), ("executor_distance_at_found", geometry_complete)):
        if complete:
            metrics[name] = np.array([r[name] for r in found], float)
    differences = {key: {"mean": [], "median": []} for key in metrics}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        indices = rng.integers(0, len(found), len(found))
        by = y[indices]
        both = by.any() and not by.all()
        for name, x in metrics.items():
            bx = x[indices]
            for stat, function in (("mean", np.mean), ("median", np.median)):
                differences[name][stat].append(float(function(bx[~by])-function(bx[by])) if both else None)
        for name, (matrix, _) in specs.items():
            fit = geometry.logistic(matrix[indices], by)
            fit_status[name][fit["status"]] += 1
            if fit["status"] == "ok":
                fit_draws[name].append(fit["beta"])
    groups = {}
    for name, x in metrics.items():
        success, failure = geometry.describe(x[y]), geometry.describe(x[~y])
        groups[name] = dict(status="complete", success=success, failure=failure,
                           success_mean=success["mean"], failure_mean=failure["mean"])
        for stat, function in (("mean", np.mean), ("median", np.median)):
            groups[name][stat+"_difference"] = dict(
                estimate=float(function(x[~y])-function(x[y])) if y.any() and not y.all() else None,
                **geometry.interval(differences[name][stat], repetitions))
        groups[name]["difference"] = groups[name]["mean_difference"]["estimate"]
    models = {}
    for key in ("found_step_model", "distance_model", "combined_model"):
        if key not in specs:
            models[key] = dict(status="not_fitted_missing_required_predictor", n=len(found))
            continue
        _, terms = specs[key]
        models[key] = dict(geometry.model_report(fits[key], terms, fit_draws[key], repetitions, fit_status[key]),
                           n=len(found), predictor_order=terms, formula="Success ~ "+" + ".join(terms))
    models["found_step_OR"] = models["found_step_model"].get("terms", {}).get("found_step", {}).get("odds_ratio_per_10steps")
    models["distance_OR"] = models["distance_model"].get("terms", {}).get("distance", {}).get("odds_ratio_per_1m")
    return groups, models


def dominant_factor(models, repetitions):
    labels = {
        (True, False): ("Evidence favors early discovery limitation", "discovery_timing"),
        (False, True): ("Evidence favors found-state geometry limitation", "found_state_geometry"),
        (True, True): ("Both discovery timing and handoff geometry contribute", "discovery_timing_and_found_state_geometry"),
        (False, False): ("No dominant factor identified", None),
    }
    def supported(model, term):
        if model.get("status") != "ok":
            return False
        values = model["terms"][term]
        boot = values["beta_bootstrap"]
        return (boot["valid_repetitions"] >= .9*repetitions and boot["ci95"] is not None
                and boot["ci95"][1] < 0 and values["beta_wald_ci95"][1] < 0)
    sufficient = repetitions >= 5000 and all(
        models[key].get("status") == "ok"
        and models[key].get("bootstrap_fit_status_counts", {}).get("ok", 0) >= .9*repetitions
        for key in ("found_step_model", "distance_model", "combined_model")
    )
    time = sufficient and supported(models["found_step_model"], "found_step") and supported(models["combined_model"], "found_step")
    distance = sufficient and supported(models["distance_model"], "distance") and supported(models["combined_model"], "distance")
    label, factor = labels[time, distance]
    return dict(interpretation_type="descriptive", conclusion=label, associated_factor=factor,
                population="Found episodes only", timing_supported=time, geometry_supported=distance,
                comparison_assessable=sufficient, rule=DOMINANCE_RULE)


def contact_budget(found):
    """Observed exposure and contact times, never imputed times for noncontacts."""
    contacts = [r for r in found if r["contact"] is True]
    noncontacts = [r for r in found if r["contact"] is False]
    durations = [r["found_to_contact_steps"] for r in contacts if r["found_to_contact_steps"] is not None]
    censored = [r for r in noncontacts if r["episode_length"] == r["max_steps"]]
    exposures = [r["episode_length"]-r["found_step"] for r in noncontacts
                 if r["episode_length"] is not None and r["found_step"] is not None]
    return dict(
        contact_success_equality=geometry.contact_equality(found),
        observed_found_to_contact_steps=geometry.describe(durations),
        contact_with_missing_duration_scenario_ids=[r["scenario_id"] for r in contacts if r["found_to_contact_steps"] is None],
        remaining_steps_by_outcome={name: geometry.describe([r["remaining_steps"] for r in found
                                                             if r["success"] == success and r["remaining_steps"] is not None])
                                   for name, success in (("success", True), ("found_but_failed", False))},
        found_without_contact_count=len(noncontacts),
        no_contact_at_horizon_count=len(censored),
        no_contact_at_horizon_scenario_ids=[r["scenario_id"] for r in censored],
        no_contact_observed_exposure_steps=geometry.describe(exposures),
        no_contact_missing_exposure_scenario_ids=[r["scenario_id"] for r in noncontacts
                                                  if r["episode_length"] is None or r["found_step"] is None],
        no_contact_ended_before_horizon_scenario_ids=[r["scenario_id"] for r in noncontacts
                                                     if r["episode_length"] is not None and r["episode_length"] < r["max_steps"]],
        interpretation="No-contact episodes at the horizon have right-censored contact times. Their required completion time is unknown; observed successful contact times cannot establish an individual failed episode's sufficient time budget.",
    )


def audit(source, expected_episodes):
    rows, paths, provenance = geometry.audit_inputs(source, expected_episodes)
    manifest = json.loads((source/"evaluation_manifest.json").read_text(encoding="utf-8-sig"))
    config = json.loads((source/"resolved_evaluation_config.json").read_text(encoding="utf-8-sig"))
    seed = geometry.integer(manifest.get("scenario_seed"))
    if seed is None or seed != geometry.integer(config.get("scenario_seed")):
        raise ValueError("manifest/config scenario generator seed mismatch")
    identity = {(r["checkpoint"], geometry.integer(r.get("checkpoint_episode")), r.get("checkpoint_config_hash"),
                 r.get("checkpoint_runtime_revision")) for r in rows}
    if len(identity) != 1:
        raise ValueError("mixed checkpoint identity metadata")
    checkpoint, episode, digest, revision = next(iter(identity))
    if episode is None or not digest or revision != config.get("checkpoint_runtime_revision"):
        raise ValueError("missing/inconsistent checkpoint identity metadata")
    metadata_path = source/"checkpoint_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if len(metadata) != 1 or metadata[0]["checkpoint"] != checkpoint or metadata[0]["completed_episode"] != episode:
            raise ValueError("checkpoint metadata identity mismatch")
        meta = metadata[0]["metadata"]
        if meta["config_hash"] != digest or meta["execution_runtime_revision"] != revision:
            raise ValueError("checkpoint metadata hash/revision mismatch")
        for key in ("observation_dim", "action_dim", "critic_dim"):
            if meta[key] != config[key]:
                raise ValueError("checkpoint metadata dimension mismatch")
        paths.append(metadata_path)
    provenance.update(generator_seed=seed, checkpoint_episode=episode, checkpoint_config_hash=digest,
                      checkpoint_metadata_verified=metadata_path.is_file(),
                      checkpoint_identity_scope="Recorded path, episode, configuration hash and runtime revision; no checkpoint bytes loaded.")
    return rows, paths, provenance


def analyze(source_dir, output_dir, *, bootstrap_seed=1729, bootstrap_repetitions=5000, expected_episodes=100):
    source, output = Path(source_dir).resolve(strict=True), Path(output_dir).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from historical inputs")
    if output.exists():
        raise FileExistsError("output exists; choose a new directory, never overwrite")
    if bootstrap_repetitions < 2:
        raise ValueError("at least two bootstrap repetitions; formal analysis requires 5000 or more")
    if expected_episodes < 1:
        raise ValueError("expected episode count must be positive")
    repo = Path(__file__).resolve().parents[1]
    git = geometry.git_identity(repo)
    snapshot = lambda: {str(p): geometry.sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    original_hashes = snapshot()
    raw, paths, provenance = audit(source, expected_episodes)
    episodes = build_episodes(raw)
    found = [r for r in episodes if r["found"]]
    counts = population_counts(episodes)
    missingness = {key: dict(count=sum(r[key] is None for r in found),
                            scenario_ids=[r["scenario_id"] for r in found if r[key] is None])
                   for key in ("found_step", "executor_distance_at_found", "contact", "episode_length")}
    timing_complete = bool(found) and not missingness["found_step"]["count"]
    geometry_complete = bool(found) and not missingness["executor_distance_at_found"]["count"]
    groups, models = estimate(found, seed=bootstrap_seed, repetitions=bootstrap_repetitions,
                              timing_complete=timing_complete, geometry_complete=geometry_complete) if found else ({}, {})
    if not found:
        models = {key: dict(status="not_fitted_no_found_episodes") for key in ("found_step_model", "distance_model", "combined_model")}
        models.update(found_step_OR=None, distance_OR=None)
    time_bins = timing_bins(episodes) if timing_complete else []
    distance_bins, quartile_method = geometry.quartiles([r["executor_distance_at_found"] for r in found],
                                                       [r["success"] for r in found]) if geometry_complete else ([], None)
    probability_rows = [dict(dimension="found_step_bin", bin=r["bin"], found_count=r["found_count"],
                             success_count=r["success_count"], success_probability=r["success_rate"], population="Found episodes") for r in time_bins]
    probability_rows += [dict(dimension="found_geometry_quartile", bin=r["bin"], distance_min=r["distance_min"], distance_max=r["distance_max"],
                              found_count=r["n"], success_count=r["success_count"], success_probability=r["success_rate"], population="Found episodes") for r in distance_bins]
    total = len(episodes)
    not_found = counts["A_not_found"]["count"]
    found_failed = counts["B_found_but_failed"]["count"]
    failures = not_found+found_failed
    summary = dict(
        schema_version=SCHEMA, status="complete" if timing_complete and geometry_complete else "partial_missing_required_data",
        interpretation_type="descriptive", provenance=provenance, total_episodes=total, found_count=len(found),
        populations=counts, not_found_ratio=not_found/total, found_failure_ratio=found_failed/total,
        ratio_denominators=dict(not_found_ratio="all episodes", found_failure_ratio="all episodes",
                                failure_given_found_ratio="Found episodes", not_found_share_of_all_failures="all failed episodes"),
        failure_given_found_ratio=None if not found else found_failed/len(found),
        not_found_share_of_all_failures=None if not failures else not_found/failures,
        found_failure_share_of_all_failures=None if not failures else found_failed/failures,
        missingness=missingness,
        found_step=groups.get("found_step", dict(status="stopped_missing_found_step", success_mean=None, failure_mean=None, difference=None)),
        distance=groups.get("executor_distance_at_found", dict(status="stopped_missing_distance", success_mean=None, failure_mean=None, difference=None)),
        logistic=models, dominant_failure_factor=dominant_factor(models, bootstrap_repetitions),
        model_units=dict(found_step="beta per 1 completed step; OR per 10 steps", distance="beta and OR per 1 metre",
                         standardization="center and population SD inside MLE, coefficients converted to raw units"),
        timing_bins=time_bins, geometry_quartiles=distance_bins, geometry_quartile_method=quartile_method,
        probability_definition="Empirical P(Success | Found, bin), no Not Found episodes in conditional denominators; empty bins are NA.",
        contact_budget=contact_budget(found), found_state_semantics=geometry.FOUND_SEMANTICS,
        bootstrap=dict(seed=bootstrap_seed, repetitions=bootstrap_repetitions, unit="episode/scenario",
                       formal_repetition_requirement_met=bootstrap_repetitions >= 5000,
                       method="unstratified paired scenario percentile 95% intervals; identical resamples across all models and statistics"),
        limitations=[
            "A/B/C are disjoint observed outcomes. Timing and geometry are overlapping associations within Found; they do not partition failed episodes into separate mechanisms.",
            "Remaining steps = 400 - found_step is an exact re-expression of discovery timing, not an independent predictor.",
            "No additional time-to-contact required for a noncontact episode is observed. Horizon censoring cannot establish that more time would produce Contact or Success.",
            "Single checkpoint and canonical scenario population; intervals describe within-sample uncertainty, not replication across checkpoints or seeds.",
            "Original evaluation code commit is unavailable in resolved config. Reviewed runtime revision and local source hashes are recorded.",
            "No step-level samples, trace backtracking, artificial missing-value zeros, or final-distance substitutes.",
        ],
    )
    if snapshot() != original_hashes:
        raise RuntimeError("historical input files changed during analysis")
    output.mkdir(parents=True)
    files = []
    for name, rows, fields in (
        ("found_timing_episode.csv", episodes, TIMING_FIELDS),
        ("found_timing_bins.csv", time_bins, BIN_FIELDS),
        ("found_geometry_episode.csv", found, GEOMETRY_FIELDS),
        ("success_probability_summary.csv", probability_rows, PROBABILITY_FIELDS),
    ):
        path = output/name
        geometry.write_csv(path, rows, fields)
        files.append(path)
    summary_path = output/"failure_attribution_summary.json"
    geometry.write_json(summary_path, summary)
    files.append(summary_path)
    sources = (*geometry.SOURCE_FILES, "scripts/analyze_found_timing_failure_attribution.py")
    manifest = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), git=git,
                    inputs=[dict(path=str(p), sha256=original_hashes[str(p)]) for p in paths],
                    input_bundle_sha256=original_hashes, input_bundle_unchanged=True,
                    source_evaluation_manifest_sha256=geometry.sha256(source/"evaluation_manifest.json"),
                    source_code_sha256={name: geometry.sha256(repo/name) for name in sources},
                    source_identity=provenance, expected_episodes=expected_episodes,
                    population_definition="All canonical full_prrac episodes for A/B/C; all unique Found episodes for timing, distance and Logistic, with no missing-value row deletion.",
                    found_state_semantics=geometry.FOUND_SEMANTICS,
                    geometry_source_field=geometry.DISTANCE_FIELD, horizon=400,
                    bootstrap=summary["bootstrap"], dominant_factor_rule=DOMINANCE_RULE,
                    software=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                    output_sha256={p.name: geometry.sha256(p) for p in files},
                    manifest_hash_policy="analysis_manifest.json excluded from its own hash map to avoid circular self-hash")
    geometry.write_json(output/"analysis_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", default="found_timing_failure_attribution_v1")
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--expected-episodes", type=int, default=100)
    args = parser.parse_args()
    summary = analyze(args.source_dir, args.output_dir, bootstrap_seed=args.bootstrap_seed,
                      bootstrap_repetitions=args.bootstrap_repetitions, expected_episodes=args.expected_episodes)
    print(json.dumps({key: summary[key] for key in ("status", "populations", "dominant_failure_factor")}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
