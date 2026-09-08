"""Offline, scenario-level Found geometry analysis (NumPy only; no runtime imports).

The reviewed evaluator stores a post-step physical 3D distance. It does not
store the two positions, so XY/dz/velocity cannot be recovered from that scalar.
No final distances, initial states or post-hoc traces are accepted as substitutes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np


SCHEMA = "ch3.found_executor_distance.v1"
REPORT_SCHEMA = "bser.phase1c.prrac.evaluation_report.v2"
REVISION = "dynamic_public_intercept_v3_atomic_continuity"
DISTANCE_FIELD = "executor_distance_to_target_at_found"
X_FIELD = "executor_target_distance_3d_at_found"
FOUND_SEMANTICS = {
    "state": "post_env_step",
    "step": "one-based completed physical step, first task_after.target_found == true",
    "event": "swept detection publishes Found after agent dynamics and target advance",
    "distance": "norm(float64(runtime._agent_pos[3]) - runtime.target_state.position)",
    "within_step": "end-of-step state at publication; not interpolated swept closest-approach tau",
    "latching": "ExecutionEpisodeDiagnostics.observe_step writes once when found_step is None",
}
TIMING_SEMANTICS = {
    "handoff_step": "derived from found_step: _publish_detection assigns both to step_count (publication)",
    "executor_received_target_step": "episode.executor_target_received_step from runtime.executor_received_target_step",
    "receive": "_advance_fixed_handoff executes before dynamics of entering delivery_step",
    "source_handoff_delay": "legacy CSV handoff_delay is delivery_step - found_step, NOT publication delay",
    "handoff_delay_steps": "publication handoff_step - found_step",
    "executor_receive_delay": "delivery step label - found_step; not elapsed complete physical steps",
    "executor_distance_at_handoff": "legacy alias of post-step distance on first observed receipt; NOT pre-dynamics delivery geometry",
}
SOURCE_FILES = (
    "chapter3_bser/experiments/phase1c_common/execution_diagnostics.py",
    "chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/training_env.py",
    "chapter3_bser/experiments/phase1c_prrac/training_env.py",
    "chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py",
    "core/env/uav_env.py", "core/env/mission_env.py",
    "scripts/analyze_searcher_residual_effect.py",
    "scripts/analyze_found_executor_distance.py",
)
FIELDS = """scenario_id episode_index scenario_seed found success contact found_step remaining_steps
executor_x_at_found executor_y_at_found executor_z_at_found target_x_at_found target_y_at_found target_z_at_found
executor_target_distance_3d_at_found executor_target_distance_xy_at_found executor_target_abs_dz_at_found
target_speed_at_found handoff_step executor_received_target_step handoff_delay_steps executor_receive_delay
first_contact_step found_to_contact_steps source_handoff_delay distance_source handoff_step_source""".split()
MISSING = {"", "na", "n/a", "null", "none", "nan"}


def number(value):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite numeric input")
    return value


def integer(value):
    value = number(value)
    if value is not None and value != int(value):
        raise ValueError("non-integral step/index/seed")
    return None if value is None else int(value)


def boolean(value):
    if str(value).strip().lower() in ("true", "1", "1.0"):
        return True
    if str(value).strip().lower() in ("false", "0", "0.0"):
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing/duplicate headers: {path}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"malformed CSV: {path}")
        return rows


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(executor, target):
    executor, target = np.asarray(executor, dtype=float), np.asarray(target, dtype=float)
    if executor.shape != (3,) or target.shape != (3,) or not np.isfinite([executor, target]).all():
        raise ValueError("physical positions must be finite 3D vectors")
    delta = executor - target
    return float(np.linalg.norm(delta)), float(np.linalg.norm(delta[:2])), float(abs(delta[2]))


def timing(found_step, received_step, *, semantics):
    # Never infer timing from a field name in an unknown schema.
    if semantics != TIMING_SEMANTICS or found_step is None:
        return dict(handoff_step=None, handoff_delay_steps=None, executor_receive_delay=None)
    if received_step is not None and received_step <= found_step:
        raise ValueError("receipt must follow Found for reviewed fixed-handoff runtime")
    return dict(handoff_step=found_step, handoff_delay_steps=0,
                executor_receive_delay=None if received_step is None else received_step-found_step)


def prepare_population(rows):
    """One row per full_prrac Found scenario; keep missing primary values visible."""
    population = []
    for source in rows:
        if source["evaluation_mode"] != "full_prrac" or not boolean(source["found"]):
            continue
        row = dict.fromkeys(FIELDS)
        row.update(scenario_id=source["scenario_id"], episode_index=integer(source.get("episode_index")),
                   scenario_seed=integer(source.get("scenario_seed")), found=True,
                   success=boolean(source["success"]),
                   contact=None if str(source.get("contact_episode", "")).lower() in MISSING else boolean(source["contact_episode"]),
                   found_step=integer(source.get("found_step")))
        step, limit = row["found_step"], integer(source.get("max_steps"))
        if step is not None and (limit is None or not 1 <= step <= limit):
            raise ValueError(f"invalid Found step: {row['scenario_id']}")
        row["remaining_steps"] = None if step is None else limit-step
        recorded_remaining = integer(source.get("remaining_steps_after_found"))
        if recorded_remaining is not None and recorded_remaining != row["remaining_steps"]:
            raise ValueError("remaining_steps_after_found mismatch")
        row[X_FIELD] = number(source.get(DISTANCE_FIELD))
        if row[X_FIELD] is not None and row[X_FIELD] < 0:
            raise ValueError("negative physical distance")
        row["distance_source"] = DISTANCE_FIELD if row[X_FIELD] is not None else None
        # This reviewed schema has no authoritative Found coordinate/velocity columns.
        # Refuse to manufacture XY, dz, coordinates or speed from scalar distance.
        received = integer(source.get("executor_target_received_step"))
        row["executor_received_target_step"] = received
        row.update(timing(step, received, semantics=TIMING_SEMANTICS))
        row["handoff_step_source"] = "derived_runtime_publish_equals_found" if step is not None else None
        row["source_handoff_delay"] = number(source.get("handoff_delay"))
        for name in ("handoff_delay", "found_to_target_received_steps"):
            observed = number(source.get(name))
            if observed is not None and observed != row["executor_receive_delay"]:
                raise ValueError(f"{name} disagrees with reviewed receipt semantics")
        first_contact = integer(source.get("first_contact_step"))
        if row["contact"] is False and first_contact is not None:
            raise ValueError("contact false but first_contact_step exists")
        if first_contact is not None and (step is None or not step <= first_contact <= limit):
            raise ValueError("invalid first_contact_step")
        row["first_contact_step"] = first_contact
        row["found_to_contact_steps"] = None if first_contact is None else first_contact-step
        recorded = integer(source.get("found_to_first_contact_steps"))
        if recorded is not None and recorded != row["found_to_contact_steps"]:
            raise ValueError("found_to_first_contact_steps mismatch")
        population.append(row)
    return population


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict.fromkeys(("mean", "std", "median", "Q1", "Q3", "min", "max"), None) | {"n": 0}
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x)>1 else None,
                median=float(np.median(x)), Q1=float(np.quantile(x, .25)), Q3=float(np.quantile(x, .75)),
                min=float(x.min()), max=float(x.max()))


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x-x.mean(), y-y.mean()
    denominator = np.linalg.norm(x)*np.linalg.norm(y)
    return None if denominator == 0 else float(np.dot(x, y)/denominator)


def ranks(x):
    x = np.asarray(x)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2)[inverse]


def auc(x, y):
    """ROC AUC of -distance, average ranks for ties."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    return float((ranks(-np.asarray(x))[y].sum()-n1*(n1+1)/2)/(n1*n0))


def logistic(predictors, y):
    """Unpenalized Bernoulli MLE, standardized Newton steps with backtracking.

    Reject singular, separating or nonconverged fits; never return clipped ORs.
    Diverging separating directions lack a finite MLE and are reported as such.
    """
    raw, y = np.asarray(predictors, float), np.asarray(y, float)
    if raw.ndim == 1:
        raw = raw[:, None]
    if len(np.unique(y)) != 2:
        return {"status": "single_outcome"}
    scale, center = raw.std(axis=0), raw.mean(axis=0)
    if np.any(scale < 1e-12):
        return {"status": "constant_predictor"}
    design = np.column_stack([np.ones(len(y)), (raw-center)/scale])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        return {"status": "rank_deficient"}
    for column in raw.T:
        yes, no = column[y == 1], column[y == 0]
        if yes.min() >= no.max() or no.min() >= yes.max():
            return {"status": "complete_or_quasi_separation"}
    beta = np.zeros(design.shape[1])
    beta[0] = math.log(y.mean()/(1-y.mean()))
    for iteration in range(100):
        eta = design @ beta
        p = np.exp(-np.logaddexp(0, -eta))
        hessian = design.T @ ((p*(1-p))[:, None]*design)
        condition = float(np.linalg.cond(hessian))
        if not math.isfinite(condition) or condition > 1e12 or np.linalg.norm(beta) > 100:
            return {"status": "separation_or_numerical_instability"}
        try:
            delta = np.linalg.solve(hessian, design.T@(y-p))
        except np.linalg.LinAlgError:
            return {"status": "singular_information"}
        likelihood = float(np.sum(y*eta-np.logaddexp(0, eta)))
        rate = 1.0
        while rate >= 2**-20:
            trial = beta+rate*delta
            trial_eta = design@trial
            new_likelihood = float(np.sum(y*trial_eta-np.logaddexp(0, trial_eta)))
            if new_likelihood >= likelihood-1e-12:
                break
            rate *= .5
        if rate < 2**-20:
            return {"status": "line_search_failed"}
        beta = trial
        if np.max(np.abs(rate*delta)) < 1e-8:
            eta = design@beta
            p = np.exp(-np.logaddexp(0, -eta))
            hessian = design.T@((p*(1-p))[:, None]*design)
            slopes = beta[1:]/scale
            return dict(status="ok", beta=slopes.tolist(), standardized_beta=beta[1:].tolist(),
                        intercept=float(beta[0]-np.dot(slopes, center)),
                        standard_error=(np.sqrt(np.diag(np.linalg.inv(hessian)))[1:]/scale).tolist(),
                        predictor_mean=center.tolist(), predictor_sd=scale.tolist(), iterations=iteration+1,
                        information_condition_number=float(np.linalg.cond(hessian)))
    return {"status": "nonconverged_possible_separation"}


def interval(values, attempted):
    valid = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    # Explicit coverage prevents degenerate resamples being silently omitted.
    return dict(ci95=None if len(valid)<2 else np.quantile(valid, [.025, .975]).tolist(),
                valid_repetitions=len(valid), invalid_repetitions=attempted-len(valid),
                method="scenario pairs percentile bootstrap",
                stability="insufficient_valid_replicates" if len(valid)<.9*attempted else "adequate_valid_replicates")


def odds(value):
    try:
        return math.exp(value)
    except OverflowError:
        return None


def model_report(fit, names, draws, repetitions, failures):
    result = dict(fit, bootstrap_fit_status_counts=dict(failures))
    if fit["status"] != "ok":
        return result
    terms = {}
    for i, name in enumerate(names):
        beta, se = fit["beta"][i], fit["standard_error"][i]
        ci = interval([draw[i] for draw in draws], repetitions)
        scales = (1, 5) if name == "distance" else ((10,) if name == "found_step" else (1,))
        term = dict(beta=beta, beta_bootstrap=ci, beta_wald_ci95=[beta-1.96*se, beta+1.96*se])
        for increment in scales:
            unit = "m" if name == "distance" else ("steps" if name == "found_step" else "m_per_s")
            term[f"odds_ratio_per_{increment}{unit}"] = dict(
                estimate=odds(beta*increment),
                bootstrap_ci95=None if ci["ci95"] is None else [odds(v*increment) for v in ci["ci95"]],
                wald_ci95=[odds((beta-1.96*se)*increment), odds((beta+1.96*se)*increment)])
        terms[name] = term
    result["terms"] = terms
    return result


def quartiles(x, y):
    """Value quantiles with duplicate cuts collapsed; tied distances stay together."""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    edges = np.unique(np.quantile(x, [0, .25, .5, .75, 1]))
    labels = np.searchsorted(edges[1:-1], x, side="left")
    bins = []
    for label in np.unique(labels):
        mask = labels == label
        bins.append(dict(bin=f"Q{len(bins)+1}", distance_min=float(x[mask].min()),
                         distance_max=float(x[mask].max()), n=int(mask.sum()),
                         success_count=int(y[mask].sum()), success_rate=float(y[mask].mean())))
    return bins, dict(method="linear value quantiles; right-closed intervals; duplicate edges and empty bins collapsed",
                      edges=edges.tolist(), observed_bins=len(bins))


def outcome_statistics(x, steps, y, *, seed, repetitions, speed=None):
    x, steps, y = np.asarray(x, float), np.asarray(steps, float), np.asarray(y, bool)
    grouped = dict(success=describe(x[y]), found_but_failed=describe(x[~y]))
    if not y.any() or y.all():
        return dict(status="single_outcome", distance_statistics=grouped)
    predictors = {"unadjusted_logistic": (x[:, None], ["distance"]),
                  "adjusted_logistic": (np.column_stack([x, steps]), ["distance", "found_step"])}
    if speed is not None:
        predictors["optional_extended_logistic"] = (np.column_stack([x, steps, speed]), ["distance", "found_step", "target_speed"])
    fits = {key: logistic(matrix, y) for key, (matrix, _) in predictors.items()}
    draws = {key: [] for key in predictors}
    failures = {key: Counter() for key in predictors}
    boot = {key: [] for key in ("mean_difference_failed_minus_success", "point_biserial", "spearman", "auc")}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        index = rng.integers(0, len(x), len(x))
        bx, by = x[index], y[index]
        valid = by.any() and not by.all()
        boot["mean_difference_failed_minus_success"].append(float(bx[~by].mean()-bx[by].mean()) if valid else None)
        boot["point_biserial"].append(correlation(bx, by) if valid else None)
        boot["spearman"].append(correlation(ranks(bx), ranks(by)) if valid else None)
        boot["auc"].append(auc(bx, by))
        for key, (matrix, _) in predictors.items():
            fit = logistic(matrix[index], by)
            failures[key][fit["status"]] += 1
            if fit["status"] == "ok":
                draws[key].append(fit["beta"])
    intervals = {key: interval(value, repetitions) for key, value in boot.items()}
    bins, bin_method = quartiles(x, y)
    result = dict(status="complete", distance_statistics=grouped,
                  mean_difference_failed_minus_success=dict(estimate=float(x[~y].mean()-x[y].mean()), **intervals["mean_difference_failed_minus_success"]),
                  median_difference_failed_minus_success=float(np.median(x[~y])-np.median(x[y])),
                  point_biserial=dict(r=correlation(x, y), **intervals["point_biserial"]),
                  spearman=dict(r=correlation(ranks(x), ranks(y)), **intervals["spearman"]),
                  distance_found_step_correlation=correlation(x, steps),
                  auc=dict(estimate=auc(x, y), score="-distance_3d_at_found", **intervals["auc"]),
                  quartile_success_rates=bins, quartile_method=bin_method, bootstrap_intervals=intervals)
    for key, (_, names) in predictors.items():
        result[key] = model_report(fits[key], names, draws[key], repetitions, failures[key])
    return result


def contact_equality(population):
    missing = [r["scenario_id"] for r in population if r["contact"] is None]
    discordant = [r["scenario_id"] for r in population if r["contact"] is not None and r["contact"] != r["success"]]
    return dict(equal_on_all_found=None if missing or not population else not discordant,
                missing_scenario_ids=missing, discordant_scenario_ids=discordant,
                contact_count=sum(r["contact"] is True for r in population))


def git_identity(root):
    def call(*args):
        result = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args],
                                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    return dict(commit=call("rev-parse", "HEAD"), status=call("status", "--porcelain=v1", "--untracked-files=all"))


def audit_inputs(source, expected_episodes):
    paths = [source/name for name in ("episode_evaluation.csv", "evaluation_manifest.json", "resolved_evaluation_config.json")]
    rows = read_csv(paths[0])
    manifest, config = [json.loads(p.read_text(encoding="utf-8-sig")) for p in paths[1:]]
    if len(rows) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} original episodes, got {len(rows)}")
    for doc in (manifest, config):
        if integer(doc.get("evaluation_episodes")) != len(rows):
            raise ValueError("manifest/config episode count mismatch")
        if boolean(doc.get("diagnostic_only", False)):
            raise ValueError("post-hoc diagnostic population forbidden")
    ids = [row["scenario_id"] for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("missing/duplicate scenario_id; scenarios are sampling units")
    scenarios = {s["scenario_id"]: s for s in manifest["scenarios"]}
    if len(scenarios) != len(manifest["scenarios"]) or set(scenarios) != set(ids):
        raise ValueError("evaluation_manifest scenario population mismatch")
    if set(config.get("resolved_scenario_ids", [])) != set(ids):
        raise ValueError("resolved config scenario population mismatch")
    if config.get("modes") != ["full_prrac"]:
        raise ValueError("only canonical full_prrac source accepted")
    for key, value in (("observation_dim", 28), ("action_dim", 3), ("critic_dim", 124),
                       ("report_schema", REPORT_SCHEMA), ("evaluation_runtime_revision", REVISION),
                       ("execution_variant", "B1_ATOMIC_LAST_VALID"), ("runtime_integration_mode", "native")):
        if config.get(key) != value:
            raise ValueError(f"unreviewed config {key}: {config.get(key)!r}")
    if config.get("search_value_guidance", {}).get("enabled") is not False:
        raise ValueError("search value guidance must be OFF")
    if config.get("resolved_search_recovery_variants") != ["S2A1_C2_LOCAL_CONNECTOR"]:
        raise ValueError("expected C2 recovery")
    checkpoints = set()
    for row in rows:
        sid = row["scenario_id"]
        for key, expected in (("evaluation_mode", "full_prrac"), ("report_schema", REPORT_SCHEMA),
                              ("evaluation_runtime_revision", REVISION), ("execution_variant", "B1_ATOMIC_LAST_VALID"),
                              ("search_recovery_variant", "S2A1_C2_LOCAL_CONNECTOR"), ("runtime_integration_mode", "native"),
                              ("manifest_sha256", manifest["manifest_sha256"])):
            if row.get(key) != expected:
                raise ValueError(f"row {sid}: {key} mismatch")
        if config["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("manifest identity mismatch")
        for key in ("diagnostic_only", "privileged_oracle", "explore", "training_update", "searcher_residual_off_enabled"):
            if boolean(row[key]):
                raise ValueError(f"row {sid}: forbidden {key}")
        guidance = json.loads(row["search_value_guidance"])
        if guidance.get("enabled") is not False:
            raise ValueError("episode search_value_guidance is not OFF")
        if integer(row["scenario_seed"]) != integer(scenarios[sid]["scenario_seed"]):
            raise ValueError("scenario_seed mismatch")
        if integer(row["max_steps"]) != integer(config["max_steps"]) or integer(row["max_steps"]) != integer(scenarios[sid]["max_steps"]):
            raise ValueError("max_steps mismatch")
        if boolean(row["success"]) and not boolean(row["found"]):
            raise ValueError("Success without Found")
        checkpoints.add(row["checkpoint"])
    if checkpoints != set(config["resolved_checkpoint_paths"]) or len(checkpoints) != 1:
        raise ValueError("checkpoint identity mismatch")
    return rows, paths, dict(original_output_dir=config.get("resolved_output_dir"),
                            scenario_manifest_identity=manifest["manifest_sha256"], checkpoint=next(iter(checkpoints)),
                            protocol="OFF + B1_ATOMIC_LAST_VALID + S2A1_C2_LOCAL_CONNECTOR + full_prrac",
                            historical_evaluation_git_commit=config.get("git_commit"),
                            note="Current source semantics audited; original evaluation code commit not independently attested when absent.")


def write_csv(path, rows, fields):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: "NA" if row.get(key) is None else row[key] for key in fields} for row in rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def analyze(source_dir, output_dir, *, bootstrap_seed=1729, bootstrap_repetitions=5000, expected_episodes=100):
    source, output = Path(source_dir).resolve(strict=True), Path(output_dir).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from historical input directory")
    if output.exists():
        raise FileExistsError("output directory already exists; choose a new directory, never overwrite")
    if bootstrap_repetitions < 2:
        raise ValueError("at least two bootstrap repetitions required (formal default: 5000)")
    repo = Path(__file__).resolve().parents[1]
    git = git_identity(repo)
    # Inventory and hash every file in the source bundle, even unused artifacts.
    original_hashes = {str(p): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    rows, paths, provenance = audit_inputs(source, expected_episodes)
    population = prepare_population(rows)
    missing = {key: dict(count=sum(r[key] is None for r in population),
                        scenario_ids=[r["scenario_id"] for r in population if r[key] is None])
               for key in FIELDS if key not in ("scenario_id", "found", "success")}
    counts = dict(total_episodes=len(rows), found_count=len(population),
                  success_count_total=sum(boolean(r["success"]) for r in rows),
                  success_count_among_found=sum(r["success"] for r in population),
                  found_but_failed_count=sum(not r["success"] for r in population))
    equality = contact_equality(population)
    summary = dict(schema_version=SCHEMA, provenance=provenance, episode_counts=counts, missingness=missing,
                   found_state_semantics=FOUND_SEMANTICS, contact_success_equality=equality,
                   primary_distance_complete=not missing[X_FIELD]["count"],
                   optional_extended_logistic=dict(status="not_fitted", reason="No audited Found-time velocity or speed stored; all Found speeds NA."),
                   limitations=["Observational association conditional on Found in one fixed canonical 100-scenario evaluation; no causal or external stability claim.",
                                "Scalar physical 3D distances are complete only if missingness gate passes; coordinates, XY/dz and speed were not serialized.",
                                "No initial/final state substitution, no trace interpolation, no rerun, no post-hoc 15-case sample.",
                                "Bootstrap uses whole scenarios with replacement; invalid/single-class/separating replicates are counted explicitly.",
                                "Original evaluation code commit is not recorded in the supplied resolved config; source revision labels agree, audited local source hashes are recorded."])
    timing_summary = {}
    for key in ("handoff_delay_steps", "executor_receive_delay", "source_handoff_delay"):
        values = [r[key] for r in population if r[key] is not None]
        timing_summary[key] = describe(values)
        timing_summary[key]["missing_count"] = len(population)-len(values)
        if values and len(set(values)) == 1:
            timing_summary[key]["interpretation"] = "not explanatory because no episode-level variance"
    summary["handoff_timing_summary"] = dict(semantics=TIMING_SEMANTICS, **timing_summary)
    bins = []
    if missing[X_FIELD]["count"] or missing["found_step"]["count"] or not population:
        summary.update(status="blocked_incomplete_primary_population",
                       blocked_reason="All Found episodes must have audited d_found and found_step; no samples dropped.",
                       minimum_instrumentation="At first post-step Found publication, latch scenario ID, completed step, _agent_pos[3], target_state.position and target_state.velocity, then serialize in episode diagnostics. Do not rerun automatically.")
    else:
        x, steps, y = ([r[key] for r in population] for key in (X_FIELD, "found_step", "success"))
        summary.update(outcome_statistics(x, steps, y, seed=bootstrap_seed, repetitions=bootstrap_repetitions))
        bins = summary.get("quartile_success_rates", [])
        if equality["equal_on_all_found"] is True:
            summary["contact_auxiliary"] = dict(status="not_refitted", reason="Contact == Success for every Found episode.")
        elif equality["missing_scenario_ids"]:
            summary["contact_auxiliary"] = dict(status="not_fitted", reason="Incomplete Contact labels; no silent complete-case analysis.")
        else:
            c = [r["contact"] for r in population]
            summary["contact_auxiliary"] = dict(distance_to_contact=outcome_statistics(x, steps, c, seed=bootstrap_seed, repetitions=bootstrap_repetitions),
                contact_to_success=[dict(contact=flag, n=sum(r["contact"] == flag for r in population),
                                        success_count=sum(r["contact"] == flag and r["success"] for r in population),
                                        success_rate=None if not any(r["contact"] == flag for r in population) else
                                        sum(r["contact"] == flag and r["success"] for r in population)/sum(r["contact"] == flag for r in population)) for flag in (False, True)])
    after_hashes = {str(p): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    if after_hashes != original_hashes:
        raise RuntimeError("historical input bundle changed during analysis")
    output.mkdir(parents=True)
    episode_path, bins_path, summary_path = [output/f"found_executor_distance_{suffix}" for suffix in ("episode.csv", "bins.csv", "summary.json")]
    write_csv(episode_path, population, FIELDS)
    write_csv(bins_path, bins, "bin distance_min distance_max n success_count success_rate".split())
    write_json(summary_path, summary)
    manifest = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(), git=git,
                    input_files=[dict(path=str(p), sha256=original_hashes[str(p)]) for p in paths],
                    input_bundle_sha256=original_hashes, input_bundle_unchanged=True,
                    source_evaluation_manifest_sha256=sha256(source/"evaluation_manifest.json"),
                    source_code_sha256={name: sha256(repo/name) for name in SOURCE_FILES},
                    analysis_population_definition="All unique full_prrac Found==true scenarios from the complete original episode_evaluation.csv; no trace samples.",
                    distance_definition=dict(source_field=DISTANCE_FIELD, physical_definition=FOUND_SEMANTICS["distance"], units="metres"),
                    found_state_semantics=FOUND_SEMANTICS, timing_semantics=TIMING_SEMANTICS,
                    bootstrap_seed=bootstrap_seed, bootstrap_repetitions=bootstrap_repetitions,
                    bootstrap_unit="one original episode/scenario; unstratified paired resampling; same indices for all models",
                    software=dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform()),
                    output_sha256={p.name: sha256(p) for p in (episode_path, bins_path, summary_path)},
                    manifest_hash_policy="Manifest excluded from its own output hash map to avoid circular self-hash.")
    write_json(output/"found_executor_distance_manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--expected-episodes", type=int, default=100)
    args = parser.parse_args()
    summary = analyze(args.source_dir, args.output_dir, bootstrap_seed=args.bootstrap_seed,
                      bootstrap_repetitions=args.bootstrap_repetitions, expected_episodes=args.expected_episodes)
    print(json.dumps({key: summary.get(key) for key in ("status", "episode_counts", "primary_distance_complete")}, indent=2))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
