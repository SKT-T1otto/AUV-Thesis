"""Spawn-safe orchestration of the existing evaluator, not a second simulator."""

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
import traceback

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from . import MAX_STEPS, PROFILE
from .provenance import (atomic_json, digest, file_hash, json_value, read_json,
                         scenario_identity, scenarios_from_manifest)
from .runtime_audit import BoundaryProbeComplete, CUnavailable, InvalidRoot, RuntimeAudit

C2 = "S2A1_C2_LOCAL_CONNECTOR"
NATIVE = "dynamic_public_intercept_v3_atomic_continuity"
OUTCOME_FIELDS = ("found", "contact_episode", "success", "episode_length", "reward",
                  "collision_episode", "searcher_collision_count_pre_found")


def arguments(description, argv=None, *, prediction=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--training-config", type=Path)
    source.add_argument("--training-manifest", type=Path)
    parser.add_argument("--historical-off-output", type=Path, required=True)
    parser.add_argument("--historical-on-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="D1: first two of frozen 30; D2: first selected scene; still 400-step cutoff")
    mode.add_argument("--diagnostic", action="store_true", help="all predeclared diagnostic scenarios; not a formal thesis test")
    if prediction:
        parser.add_argument("--manifest", type=Path, help="existing frozen 30-scene manifest; otherwise generate with seed 51729")
        parser.add_argument("--save-features", action="store_true", help="write compressed raw 34D feature dictionary, excluded from bundle")
    args = parser.parse_args(argv)
    for name in ("checkpoint", "training_config", "training_manifest"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"missing input file: {path}")
    return args


def key(row):
    return str(row["scenario_id"]), int(row["scenario_seed"])


def historical_inputs(off_path, on_path, model_identity, payload):
    """Strict provenance joins. Historical aggregate counts never select by outcome."""
    histories = {}
    for mode, path in (("OFF", off_path), ("ON", on_path)):
        path = Path(path)
        config = evaluator._load_config(path / "resolved_evaluation_config.json")
        for name, expected in (("profile", PROFILE), ("max_steps", MAX_STEPS), ("runtime_integration_mode", "native"),
                               ("explore", False), ("training_update", False), ("device", "cpu")):
            if config.get(name) != expected:
                raise ValueError(f"historical {mode} {name}: expected {expected}")
        if config.get("modes") != ["full_prrac"] or config.get("search_recovery_variants") != [C2]:
            raise ValueError(f"historical {mode} must contain exactly full_prrac / C2")
        if config.get("search_value_decision", {}).get("enabled", False):
            raise ValueError("another Search Value decision channel is enabled")
        guidance = config.get("search_value_guidance", {})
        if bool(guidance.get("enabled")) != (mode == "ON"):
            raise ValueError("historical OFF/ON guidance settings disagree")
        if mode == "ON" and any(float(guidance.get(k, v)) != v for k, v in (("weight", .1), ("clip_min", 0.), ("clip_max", 1.))):
            raise ValueError("historical ON is not the fixed weight=.1, clip=[0,1] algorithm")
        manifest = read_json(path / "evaluation_manifest.json")
        if manifest.get("manifest_sha256") != evaluator._hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
            raise ValueError("historical manifest hash mismatch")
        scenarios = scenarios_from_manifest(manifest)
        rows = evaluator._read_csv(path / "episode_evaluation.csv")
        by_key = {key(r): r for r in rows}
        if len(by_key) != len(rows) or set(by_key) != {key(s) for s in scenarios}:
            raise ValueError("historical episode rows do not join manifest exactly once")
        for row in rows:
            if any(row.get(k) != v for k, v in (("execution_variant", "B1_ATOMIC_LAST_VALID"),
                      ("search_recovery_variant", C2), ("evaluation_mode", "full_prrac"),
                      ("runtime_integration_mode", "native"), ("manifest_sha256", manifest["manifest_sha256"]),
                      ("checkpoint_config_hash", payload["metadata"]["config_hash"]),
                      ("checkpoint_episode", int(payload["completed_episode"])))):
                raise ValueError(f"historical {mode} episode provenance mismatch: {key(row)}")
            # Paths may differ across hosts, but identity cannot be inferred merely
            # from a matching basename. Record the historical SHA availability below.
            if Path(str(row["checkpoint"]).replace("\\", "/")).name != Path(model_identity["checkpoint_path"]).name:
                raise ValueError("historical checkpoint name differs from requested checkpoint")
        histories[mode] = dict(config=config, manifest=manifest, scenarios=scenarios, rows=by_key,
                                source_hashes={name: file_hash(path/name) for name in
                                    ("resolved_evaluation_config.json", "evaluation_manifest.json", "episode_evaluation.csv")})
    if [scenario_identity(s) for s in histories["OFF"]["scenarios"]] != [scenario_identity(s) for s in histories["ON"]["scenarios"]]:
        raise ValueError("OFF/ON scenario identities or manifest order differ")
    metrics_path = Path(on_path) / "search_value_guidance_metrics.json"
    metrics = read_json(metrics_path)
    by_id = {s["scenario_id"]: s for s in histories["ON"]["scenarios"]}
    if len(by_id) != len(histories["ON"]["scenarios"]):
        raise ValueError("ambiguous scenario IDs in historical metrics join")
    entries, selected = {}, []
    for entry in metrics["episodes"]:
        identifier = entry["scenario_id"]
        if identifier not in by_id or identifier in entries:
            raise ValueError("missing or duplicate historical ON metrics scene")
        scenario = by_id[identifier]
        row = histories["ON"]["rows"][key(scenario)]
        if any(entry.get(k) != row.get(k) for k in ("checkpoint", "evaluation_mode", "execution_variant", "search_recovery_variant", "search_value_guidance")):
            raise ValueError("ON accepted-change metrics do not match episode CSV")
        entries[identifier] = entry
        if int(entry["search_value_guidance"]["accepted_search_change_count"]) > 0:
            selected.append(key(scenario))
    if set(entries) != set(by_id):
        raise ValueError("ON metrics do not cover manifest")
    histories["selected"] = [s for s in histories["ON"]["scenarios"] if key(s) in selected]
    histories["checkpoint_verification"] = dict(
        requested_sha256=model_identity["checkpoint_sha256"],
        historical_check="checkpoint basename, episode, schema/runtime and training config hash",
        historical_sha256_verified=False,
        limitation="historical episode protocol did not store checkpoint file SHA256; deterministic outcome/prefix reproduction is additionally required")
    histories["metrics_sha256"] = file_hash(metrics_path)
    return histories


def make_job(model, identity, payload, config, scenario, index, manifest):
    info = evaluator._checkpoint_info(Path(identity["checkpoint_path"]), payload, "full_prrac", "B1_ATOMIC_LAST_VALID",
        manifest_sha256=manifest.get("manifest_sha256", digest(manifest)), evaluation_runtime_revision=NATIVE,
        runtime_integration_mode="native", search_recovery_variant=C2,
        search_recovery_config_hash=config.get("search_collision_recovery_config_hash", ""),
        search_recovery_schema=config["search_collision_recovery"]["schema"],
        search_diagnostics_hash=config.get("search_continuity_diagnostics_hash", ""),
        execution_overlay_config_hash=evaluator._hash(config["execution_continuity"]))
    return dict(**model, checkpoint_info=info, config=copy.deepcopy(config), scenario=copy.deepcopy(scenario),
                episode_index=index, device="cpu", failure_trace=dict(enabled=False, only_found_failures=False, max_traces=0))


def execute_work(work):
    """Only arrays/primitive job data cross spawn. A fresh runtime every invocation."""
    job, options = work["job"], work.get("options", {})
    if work.get("no_hook"):
        return dict(unit=work["unit"], status="completed", payload=evaluator._evaluate_episode_job(job))
    audit = RuntimeAudit(job, **options)
    try:
        evaluator._evaluate_episode_job(job, audit=audit)
        status = "mismatch_root_not_reached" if options.get("intervention") is not None and audit.root_fingerprint is None else "completed"
    except BoundaryProbeComplete:
        audit.finish(None)
        status = "boundary_probe"
    except CUnavailable:
        audit.finish(None)
        status = "C_UNAVAILABLE"
    except InvalidRoot:
        audit.finish(None)
        status = "mismatch_no_search_proposal"
    return dict(unit=work["unit"], status=status, audit=audit.export())


def run_work(work, workers, progress, *, on_result=None, worker=execute_work):
    results = {}
    if work:
        progress.write(stage="replaying", scenario=work[0].get("scenario_id"), branch=work[0].get("branch"))
    def completed(item, result):
        results[item["unit"]] = result
        if on_result:
            on_result(result)
        progress.write(stage="replaying", unit=item["unit"], scenario=item.get("scenario_id"), branch=item.get("branch"))
    if workers == 1:
        for item in work:
            completed(item, worker(item))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(worker, item): item for item in work}
            for future in as_completed(futures):
                completed(futures[future], future.result())
    return [results[item["unit"]] for item in work]


def task_signature(payload):
    row = payload["episode"]
    return {k: row.get(k) for k in OUTCOME_FIELDS}


def no_op_check(control, observed, bare):
    steps_equal = control["steps"] == observed["steps"]
    boundaries_equal = control["boundaries"] == observed["boundaries"]
    terminal_equal = control["terminal_fingerprint"] == observed["terminal_fingerprint"]
    task_equal = task_signature(control["payload"]) == task_signature(observed["payload"]) == task_signature(bare)
    episode_metrics_equal = control["payload"]["episode"] == observed["payload"]["episode"] == bare["episode"]
    return dict(passed=steps_equal and task_equal and boundaries_equal and terminal_equal and episode_metrics_equal,
                actions_states_guidance_rng_equal=steps_equal, boundaries_equal=boundaries_equal, terminal_equal=terminal_equal,
                complete_episode_metrics_equal=episode_metrics_equal,
                task_equal=task_equal, step_count=len(control["steps"]),
                bare_runner_has_no_hook=True, control_trace_sha256=digest(control["steps"]),
                observed_trace_sha256=digest(observed["steps"]))


def record_exception(output, progress):
    error = traceback.format_exc()
    atomic_json(Path(output)/"error.json", dict(status="failed", traceback=error))
    progress.write(stage="error", status="failed", error=error)
    return 1
