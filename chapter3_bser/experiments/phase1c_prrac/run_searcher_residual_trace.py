"""Explicitly invoked fixed-case trace diagnostic; never generates new scenarios."""

import argparse
import copy
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import traceback

from . import evaluate_prrac_checkpoints as evaluator
from .searcher_residual_trace import SCHEMA, TRACE_DEFINITIONS, SearcherResidualTrace

MODES = ("full_prrac", "searcher_residual_off")
HELP = "full_success_searcher_off_fail"
HURT = "full_fail_searcher_off_success"
C2 = "S2A1_C2_LOCAL_CONNECTOR"
BEHAVIOR_KEYS = ("base_candidate", "profile", "split", "scenario_seed", "max_steps", "checkpoint_runtime_revision",
                 "evaluation_runtime_revision", "runtime_integration_mode", "execution_runtime", "execution_continuity",
                 "search_collision_recovery", "search_continuity_diagnostics", "search_value_guidance", "search_value_decision",
                 "observation_dim", "action_dim", "critic_dim", "architecture_version", "method")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def indexed(rows):
    result = {}
    for row in rows:
        sid = str(row.get("scenario_id", ""))
        if not sid or sid in result:
            raise ValueError(f"missing/duplicate scenario_id: {sid}")
        result[sid] = row
    return result


def behavior(config):
    value = {k: copy.deepcopy(config.get(k)) for k in BEHAVIOR_KEYS}
    # Same resolution as original evaluator CLI's C2 selection, not a new policy.
    value["search_collision_recovery"]["variants"] = [C2]
    value["search_collision_recovery"]["enabled"] = True
    return value


def validate_sources(source_root, cases_csv, config_path, checkpoint, max_per_transition=None):
    root, checkpoint = Path(source_root).resolve(), Path(checkpoint).resolve()
    requested = evaluator._load_config(Path(config_path))
    histories = {}
    for mode in MODES:
        folder = root/mode
        manifest = read_json(folder/"evaluation_manifest.json")
        if manifest.get("manifest_sha256") != evaluator._hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
            raise ValueError("source manifest hash mismatch")
        scenarios = manifest["scenarios"]
        by_id = indexed(scenarios)
        if len(by_id) != 100 or manifest.get("evaluation_episodes") != 100:
            raise ValueError("source must be complete canonical 100-scenario manifest")
        rows = indexed(evaluator._read_csv(folder/"episode_evaluation.csv"))
        if set(rows) != set(by_id):
            raise ValueError("episode scenario set does not equal complete manifest")
        raw_config = read_json(folder/"resolved_evaluation_config.json")
        if raw_config.get("resolved_config_hash") != evaluator._hash({k: v for k, v in raw_config.items() if k != "resolved_config_hash"}):
            raise ValueError("source resolved config hash mismatch")
        config = evaluator._load_config(folder/"resolved_evaluation_config.json")
        for key in ("checkpoint_config_hash", "checkpoint_episode", "checkpoint_runtime_revision", "evaluation_runtime_revision",
                    "execution_overlay_config_hash", "search_collision_recovery_config_hash"):
            values = {row.get(key) for row in rows.values()}
            if len(values) != 1 or None in values or "" in values:
                raise ValueError(f"missing/mixed source checkpoint/runtime provenance: {key}")
        for key, expected in (("profile", "M20_MOVING_UNKNOWN_MULTI"), ("max_steps", 400), ("device", "cpu"),
                              ("explore", False), ("training_update", False), ("runtime_integration_mode", "native"),
                              ("modes", [mode]), ("execution_variants", ["B1_ATOMIC_LAST_VALID"]), ("search_recovery_variants", [C2])):
            if config.get(key) != expected:
                raise ValueError(f"source {mode} {key} mismatch")
        if config["scenario_seed"] != manifest["scenario_seed"] or behavior(config) != behavior(requested):
            raise ValueError("requested config differs from historical behavioral config/seed")
        if requested.get("explore") is not False or requested.get("training_update") is not False or requested.get("device") != "cpu":
            raise ValueError("requested config must be CPU, explore=false, training_update=false")
        for sid, row in rows.items():
            if row.get("scenario_seed") != by_id[sid]["scenario_seed"]:
                raise ValueError(f"scenario seed mismatch: {sid}")
            if by_id[sid].get("max_steps") != 400:
                raise ValueError("source scenario max_steps mismatch")
            for key, expected in (("evaluation_mode", mode), ("manifest_sha256", manifest["manifest_sha256"]),
                                  ("max_steps", 400), ("execution_variant", "B1_ATOMIC_LAST_VALID"),
                                  ("search_recovery_variant", C2), ("runtime_integration_mode", "native"),
                                  ("explore", False), ("training_update", False)):
                if row.get(key) != expected:
                    raise ValueError(f"source episode {key} mismatch: {sid}")
            if Path(str(row.get("checkpoint", ""))).resolve() != checkpoint:
                raise ValueError("checkpoint path differs from historical experiment (no basename fallback)")
        histories[mode] = dict(config=config, manifest=manifest, rows=rows)
    if histories[MODES[0]]["manifest"] != histories[MODES[1]]["manifest"]:
        raise ValueError("full/searcher-off manifests differ")
    cases = indexed(evaluator._read_csv(Path(cases_csv)))
    counts = {group: sum(r.get("transition_type") == group for r in cases.values()) for group in (HELP, HURT)}
    if len(cases) != 15 or counts != {HELP: 5, HURT: 10}:
        raise ValueError("cases must contain exactly 5 help + 10 hurt, even in smoke")
    if not set(cases).issubset(histories[MODES[0]]["rows"]):
        raise ValueError("case scenario is not in original manifest")
    discordant = {}
    for sid in histories[MODES[0]]["rows"]:
        left, right = (histories[mode]["rows"][sid] for mode in MODES)
        for key in ("checkpoint", "checkpoint_config_hash", "checkpoint_episode", "checkpoint_runtime_revision",
                    "evaluation_runtime_revision", "execution_overlay_config_hash", "search_collision_recovery_config_hash"):
            if left.get(key) != right.get(key):
                raise ValueError(f"paired historical {key} mismatch")
        if left["success"] != right["success"]:
            discordant[sid] = HELP if left["success"] else HURT
    if discordant != {sid: row["transition_type"] for sid, row in cases.items()}:
        raise ValueError("cases do not exactly reproduce historical Success discordance")
    if max_per_transition is not None and max_per_transition < 1:
        raise ValueError("max-per-transition must be positive")
    selected, taken = [], {HELP: 0, HURT: 0}
    for index, scenario in enumerate(histories[MODES[0]]["manifest"]["scenarios"]):
        sid = str(scenario["scenario_id"])
        if sid in cases:
            group = cases[sid]["transition_type"]
            if max_per_transition is None or taken[group] < max_per_transition:
                selected.append(dict(scenario=copy.deepcopy(scenario), episode_index=index, transition_type=group))
                taken[group] += 1
    return histories, selected


def trace_worker(work):
    job = work["job"]
    sink = SearcherResidualTrace(job["scenario"], job["checkpoint_info"]["evaluation_mode"], work["transition_type"])
    payload = evaluator._evaluate_episode_job(job, searcher_trace=sink)
    return dict(episode=payload["episode"], action=sink.action_rows, mission=sink.mission_rows)


def collect_jobs(jobs, workers, worker=trace_worker, on_result=None):
    results = []
    def accept(result):
        results.append(result)
        if on_result is not None:
            on_result(result)
    if workers == 1:
        for job in jobs:
            accept(worker(job))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(worker, job) for job in jobs]):
                accept(future.result())
    return sorted(results, key=lambda r: (str(r["episode"]["scenario_id"]), r["episode"]["evaluation_mode"]))


def write_results(output, results):
    """Parent only. Rewrites atomic snapshots in stable key order after each run."""
    for mode in MODES:
        selected = [r for r in results if r["episode"]["evaluation_mode"] == mode]
        if not selected:
            continue
        folder = Path(output)/mode
        for kind, filename in (("episode", "episode_evaluation.csv"), ("action", "searcher_action_trace.csv"), ("mission", "mission_step_trace.csv")):
            rows = [r[kind] for r in selected] if kind == "episode" else [row for r in selected for row in r[kind]]
            rows.sort(key=lambda r: (str(r["scenario_id"]), mode, r.get("step", -1), r.get("agent_id", -1)))
            evaluator._write_csv(folder/filename, [{k: "NA" if v is None else v for k, v in row.items()} for row in rows])


def run(args):
    output = args.output_dir.resolve()
    source = args.source_root.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from historical source")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("output exists and is nonempty; resume unsupported")
    input_hashes = {str(path.resolve()): sha(path) for path in (args.cases_csv, args.config, args.checkpoint)}
    histories, selected = validate_sources(source, args.cases_csv, args.config, args.checkpoint, args.max_per_transition)
    config = histories[MODES[1]]["config"]
    checkpoint_sha = sha(args.checkpoint)
    learner, payload = evaluator.load_prrac_checkpoint(args.checkpoint, device="cpu", config=config)
    metadata, state = payload["metadata"], payload["prrac_training_state"]
    jobs = []
    for entry in selected:
        for mode in MODES:
            original = histories[mode]["rows"][str(entry["scenario"]["scenario_id"])]
            for key, value in (("checkpoint_config_hash", metadata["config_hash"]),
                               ("checkpoint_episode", payload["completed_episode"]),
                               ("checkpoint_runtime_revision", metadata["execution_runtime_revision"])):
                if original.get(key) != value:
                    raise ValueError(f"loaded checkpoint {key} differs from history")
            run_config = histories[mode]["config"]
            info = evaluator._checkpoint_info(args.checkpoint, payload, mode, "B1_ATOMIC_LAST_VALID",
                manifest_sha256=histories[mode]["manifest"]["manifest_sha256"], runtime_integration_mode="native",
                evaluation_runtime_revision=run_config["evaluation_runtime_revision"],
                execution_overlay_config_hash=evaluator._hash(run_config["execution_continuity"]),
                search_diagnostics_hash=run_config["search_continuity_diagnostics_hash"], search_recovery_variant=C2,
                search_recovery_config_hash=run_config["search_collision_recovery_config_hash"],
                search_recovery_schema=run_config["search_collision_recovery"]["schema"])
            for key in ("evaluation_runtime_revision", "search_collision_recovery_config_hash", "execution_overlay_config_hash"):
                if info[key] != original.get(key):
                    raise ValueError(f"runtime provenance differs from history: {key}")
            job = dict(scenario=entry["scenario"], episode_index=entry["episode_index"], config=copy.deepcopy(run_config),
                       checkpoint_info=info, architecture=metadata["architecture"], loss=metadata["loss"],
                       gamma=state["gamma"], tau=state["tau"], reward=metadata["reward"], policy_snapshot=learner.policy_snapshot(),
                       search_value_snapshot=learner.search_value_snapshot() if run_config.get("search_value_guidance", {}).get("enabled") else None,
                       search_value_config=learner.search_value_config, device="cpu",
                       failure_trace=dict(enabled=bool(run_config["failure_trace"]["enabled"]),
                           only_found_failures=bool(run_config["failure_trace"]["only_found_failures"]),
                           max_traces=int(run_config["failure_trace"]["max_traces_per_checkpoint_mode"]),
                           selector=str(run_config["failure_trace"].get("selector", "found_failures"))))
            if evaluator._contains_tensor(job):
                raise ValueError("worker payload contains live tensors")
            jobs.append(dict(job=job, transition_type=entry["transition_type"]))
    def git(*arguments):
        return subprocess.check_output(["git", "-c", f"safe.directory={evaluator.ROOT.as_posix()}", *arguments], cwd=evaluator.ROOT, text=True).strip()
    manifest = dict(schema=SCHEMA, git_commit=git("rev-parse", "HEAD"), git_status_short=git("status", "--short"),
        checkpoint_path=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
        config_path=str(args.config.resolve()), config_sha256=sha(args.config), source_experiment_root=str(source),
        source_full_manifest_hash=histories[MODES[0]]["manifest"]["manifest_sha256"],
        source_searcher_off_manifest_hash=histories[MODES[1]]["manifest"]["manifest_sha256"],
        source_files={str(source/mode/name): sha(source/mode/name) for mode in MODES for name in
                      ("evaluation_manifest.json", "resolved_evaluation_config.json", "episode_evaluation.csv")},
        cases_csv_sha256=sha(args.cases_csv), cases_csv_path=str(args.cases_csv.resolve()),
        selected=[dict(scenario_id=e["scenario"]["scenario_id"], scenario_seed=e["scenario"]["scenario_seed"],
                       canonical_episode_index=e["episode_index"], transition_type=e["transition_type"], scenario_sha256=evaluator._hash(e["scenario"])) for e in selected],
        canonical_count=100, expected_help_count=5, expected_hurt_count=10, modes=list(MODES), max_steps=400,
        workers=args.workers, device="cpu", training_update=False, explore=False, definitions=TRACE_DEFINITIONS,
        smoke=args.max_per_transition is not None, scenario_count=len(selected), episode_runs_expected=len(jobs),
        historical_checkpoint_sha_verified=False, historical_checkpoint_limitation="historical protocol stores path/episode/config identity, not checkpoint byte hash; outcome reproduction is checked",
        status="running")
    output.mkdir(parents=True, exist_ok=True)
    evaluator._write_json(output/"trace_manifest.json", manifest)
    evaluator._write_csv(output/"selected_scenarios.csv", manifest["selected"])
    results = []
    try:
        def completed(result):
            results.append(result)
            write_results(output, results)
            evaluator._write_json(output/"progress.json", dict(completed=len(results), total=len(jobs), percent=100*len(results)/len(jobs)))
        collect_jobs(jobs, args.workers, on_result=completed)
        mismatches = []
        for result in results:
            row = result["episode"]
            original = histories[row["evaluation_mode"]]["rows"][str(row["scenario_id"]) ]
            for key in ("found", "contact_episode", "success", "episode_length", "found_step", "searcher_collision_count_pre_found"):
                if row.get(key) != original.get(key):
                    mismatches.append(dict(scenario_id=row["scenario_id"], mode=row["evaluation_mode"], field=key,
                                           original=original.get(key), replay=row.get(key)))
        if sha(args.checkpoint) != checkpoint_sha or any(sha(path) != value for path, value in {**manifest["source_files"], **input_hashes}.items()):
            raise RuntimeError("read-only input identity changed during run")
        manifest.update(status="completed" if not mismatches else "completed_with_mismatches", historical_reproduction_mismatches=mismatches,
                        trace_files={path.relative_to(output).as_posix(): sha(path) for mode in MODES for path in sorted((output/mode).glob("*.csv"))})
        evaluator._write_json(output/"trace_manifest.json", manifest)
        return 0 if not mismatches else 2
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
        evaluator._write_json(output/"trace_manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "cases-csv", "checkpoint", "config", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--max-per-transition", type=int)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        parser.exit(1, f"trace diagnostic failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
