"""Validated historical-anchor replay; explicit CLI only, no training path."""

import argparse
import copy
from pathlib import Path
import subprocess
import traceback

from scripts import analyze_searcher_residual_branch_sensitivity as analysis
from . import evaluate_prrac_checkpoints as evaluator
from . import run_searcher_residual_trace as source_trace
from .residual_branch_diagnostic import ResidualBranchDiagnostic


def alpha_grid(values, smoke):
    values = [float(v) for value in values for v in str(value).split(",")]
    if len(set(values)) != len(values) or len(values) < 2 or any(not 0 <= a <= 1 for a in values) or 1. not in values:
        raise ValueError("alpha grid requires distinct finite values in [0,1], including 1")
    values.sort()
    if not smoke and values != list(analysis.ALPHAS):
        raise ValueError("formal grid must be 0,0.25,0.5,0.75,1; custom alpha grids are smoke only")
    return values


def validate_inputs(args):
    trace_root = args.trace_root.resolve()
    manifest = source_trace.read_json(trace_root/"trace_manifest.json")
    expected = dict(schema="bser.searcher_residual_trace.v1", status="completed", smoke=False,
                    scenario_count=15, episode_runs_expected=30, max_steps=400, device="cpu", explore=False, training_update=False)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("requires completed canonical formal 15-scenario trace manifest")
    if Path(manifest["checkpoint_path"]).resolve() != args.checkpoint.resolve() or source_trace.sha(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint path/SHA differs from original trace")
    if source_trace.sha(args.config) != manifest["config_sha256"]:
        raise ValueError("config SHA differs from original trace")
    if Path(manifest["source_experiment_root"]).resolve() != args.source_root.resolve():
        raise ValueError("historical experiment path differs from trace source")
    histories, selected = source_trace.validate_sources(args.source_root, args.paired_first_divergence,
        args.config, args.checkpoint, max_per_transition=None)
    for key in ("search_value_guidance", "search_value_decision"):
        if histories[source_trace.MODES[1]]["config"].get(key, {}).get("enabled", False):
            raise ValueError("requires original Search Value OFF config")
    identities = [dict(scenario_id=e["scenario"]["scenario_id"], scenario_seed=e["scenario"]["scenario_seed"],
        canonical_episode_index=e["episode_index"], transition_type=e["transition_type"], scenario_sha256=evaluator._hash(e["scenario"])) for e in selected]
    if manifest["selected"] != identities:
        raise ValueError("selected scenarios/order/seeds/content differ from historical canonical subset")
    for mode, key in (("full_prrac", "source_full_manifest_hash"), ("searcher_residual_off", "source_searcher_off_manifest_hash")):
        if manifest.get(key) != histories[mode]["manifest"]["manifest_sha256"]:
            raise ValueError("trace canonical manifest identity mismatch")
    input_hashes = {str(p.resolve()): source_trace.sha(p) for p in
                    (args.checkpoint, args.config, args.paired_first_divergence, trace_root/"trace_manifest.json")}
    data = {}
    for mode in source_trace.MODES:
        for filename in ("evaluation_manifest.json", "resolved_evaluation_config.json", "episode_evaluation.csv"):
            path = (args.source_root/mode/filename).resolve()
            if manifest["source_files"].get(str(path)) != source_trace.sha(path):
                raise ValueError("historical source hash differs from trace provenance")
            input_hashes[str(path)] = source_trace.sha(path)
        bundle = {}
        for kind, filename in (("action", "searcher_action_trace.csv"), ("mission", "mission_step_trace.csv"), ("episode", "episode_evaluation.csv")):
            path = trace_root/mode/filename
            if manifest["trace_files"].get(path.relative_to(trace_root).as_posix()) != source_trace.sha(path):
                raise ValueError("historical trace hash mismatch")
            input_hashes[str(path)] = source_trace.sha(path)
            rows = analysis.trace.read_csv(path)
            grouped = {r["scenario_id"]: [] for r in identities}
            identity_by_id = {r["scenario_id"]: r for r in identities}
            for row in rows:
                sid = str(row["scenario_id"])
                if sid not in grouped or row.get("mode", row.get("evaluation_mode")) != mode or row["scenario_seed"] != identity_by_id[sid]["scenario_seed"]:
                    raise ValueError("trace scenario/seed/mode mismatch")
                if kind != "episode" and row["transition_type"] != identity_by_id[sid]["transition_type"]:
                    raise ValueError("trace transition mismatch")
                grouped[sid].append(row)
            for sid, items in grouped.items():
                if not items:
                    raise ValueError(f"missing trace rows: {sid}/{kind}")
                if kind == "action":
                    index = analysis.rows_index(items)
                    if set(index) != {(t, a) for t in range(max(t for t, _ in index)+1) for a in range(3)}:
                        raise ValueError("trace step/agent gap")
                if kind == "mission":
                    items.sort(key=lambda r: r["step"])
                    if [r["step"] for r in items] != list(range(len(items))) or any(r["state_step"] != r["step"]+1 for r in items):
                        raise ValueError("mission trace step mismatch")
                if kind == "episode" and len(items) != 1:
                    raise ValueError("duplicate episode trace")
            bundle[kind] = grouped
        data[mode] = bundle
    divergences = source_trace.indexed(analysis.trace.read_csv(args.paired_first_divergence))
    anchors = []
    for entry in selected:
        sid = entry["scenario"]["scenario_id"]
        selection = analysis.select_anchor(divergences[sid], data["full_prrac"]["action"][sid], data["searcher_residual_off"]["action"][sid])
        if selection["anchor_step"]+analysis.HORIZON > 400:
            raise ValueError("insufficient historical time budget for ten intervention steps")
        anchors.append(dict(entry, selection=selection))
    # Validate all 15 before any optional smoke slicing.
    if args.max_per_transition is not None:
        if args.max_per_transition < 1:
            raise ValueError("max-per-transition must be positive")
        taken = {source_trace.HELP: 0, source_trace.HURT: 0}
        subset = []
        for entry in anchors:
            group = entry["transition_type"]
            if taken[group] < args.max_per_transition:
                subset.append(entry)
                taken[group] += 1
        anchors = subset
    return histories, anchors, data, input_hashes, manifest


def make_jobs(args, histories, anchors, alphas):
    config = histories["full_prrac"]["config"]
    learner, payload = evaluator.load_prrac_checkpoint(args.checkpoint, device="cpu", config=config)
    metadata, state = payload["metadata"], payload["prrac_training_state"]
    snapshot = learner.policy_snapshot()
    jobs = []
    for entry in anchors:
        original = histories["full_prrac"]["rows"][entry["scenario"]["scenario_id"]]
        for key, value in (("checkpoint_config_hash", metadata["config_hash"]), ("checkpoint_episode", payload["completed_episode"]),
                           ("checkpoint_runtime_revision", metadata["execution_runtime_revision"])):
            if original[key] != value:
                raise ValueError(f"loaded checkpoint disagrees with historical {key}")
        info = evaluator._checkpoint_info(args.checkpoint, payload, "full_prrac", "B1_ATOMIC_LAST_VALID",
            manifest_sha256=histories["full_prrac"]["manifest"]["manifest_sha256"], runtime_integration_mode="native",
            evaluation_runtime_revision=config["evaluation_runtime_revision"],
            execution_overlay_config_hash=evaluator._hash(config["execution_continuity"]),
            search_diagnostics_hash=config["search_continuity_diagnostics_hash"], search_recovery_variant=source_trace.C2,
            search_recovery_config_hash=config["search_collision_recovery_config_hash"], search_recovery_schema=config["search_collision_recovery"]["schema"])
        for key in ("execution_overlay_config_hash", "search_collision_recovery_config_hash", "evaluation_runtime_revision"):
            if info[key] != original[key]:
                raise ValueError(f"runtime config differs from history: {key}")
        job = dict(scenario=entry["scenario"], episode_index=entry["episode_index"], config=copy.deepcopy(config),
            checkpoint_info=info, architecture=metadata["architecture"], loss=metadata["loss"], reward=metadata["reward"],
            gamma=state["gamma"], tau=state["tau"], policy_snapshot=snapshot, search_value_snapshot=None,
            search_value_config=learner.search_value_config, device="cpu",
            failure_trace=dict(enabled=bool(config["failure_trace"]["enabled"]), only_found_failures=bool(config["failure_trace"]["only_found_failures"]),
                max_traces=int(config["failure_trace"]["max_traces_per_checkpoint_mode"]), selector=config["failure_trace"].get("selector", "found_failures")))
        for alpha in alphas:
            jobs.append(dict(job=job, selection=entry["selection"], alpha=alpha))
    if evaluator._contains_tensor(jobs):
        raise ValueError("worker job contains live tensors")
    return jobs


def branch_worker(work):
    sink = ResidualBranchDiagnostic(work["job"]["scenario"], work["selection"], work["alpha"])
    evaluator._evaluate_episode_job(work["job"], searcher_trace=sink, branch_diagnostic=sink)
    result = sink.export()
    # Compatibility with the existing spawn/parent collector; not an episode metric.
    result["episode"] = dict(scenario_id=result["scenario_id"], evaluation_mode="full_prrac")
    return result


def write_csv(path, rows):
    import json
    evaluator._write_csv(path, [{k: "NA" if v is None else json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows])


def run(args):
    output = analysis.nonempty_output(args.output_dir, [args.source_root, args.trace_root])
    for path in (args.checkpoint, args.config, args.paired_first_divergence):
        if output == path.resolve() or output in path.resolve().parents:
            raise ValueError("output contains protected input file")
    alphas = alpha_grid(args.alphas, args.max_per_transition is not None)
    histories, anchors, data, hashes, original_manifest = validate_inputs(args)
    jobs = make_jobs(args, histories, anchors, alphas)
    def git(*arguments):
        return subprocess.check_output(["git", "-c", f"safe.directory={evaluator.ROOT.as_posix()}", *arguments], cwd=evaluator.ROOT, text=True).strip()
    manifest = dict(schema=analysis.SCHEMA, git_commit=git("rev-parse", "HEAD"), git_status_short=git("status", "--short"),
        checkpoint_path=str(args.checkpoint.resolve()), checkpoint_sha256=source_trace.sha(args.checkpoint),
        config_path=str(args.config.resolve()), config_sha256=source_trace.sha(args.config),
        source_historical_root=str(args.source_root.resolve()), source_trace_root=str(args.trace_root.resolve()),
        trace_manifest_sha256=source_trace.sha(args.trace_root/"trace_manifest.json"),
        paired_first_divergence_sha256=source_trace.sha(args.paired_first_divergence), input_sha256=hashes,
        selected=[dict(e["selection"], scenario_seed=e["scenario"]["scenario_seed"], canonical_episode_index=e["episode_index"], scenario_sha256=evaluator._hash(e["scenario"])) for e in anchors],
        alphas=alphas, horizon=analysis.HORIZON, intervention_scope="first_branch_agents", workers=args.workers, device="cpu",
        scenario_count=len(anchors), branch_runs_expected=len(jobs), max_steps=400,
        execution_variant="B1_ATOMIC_LAST_VALID", search_recovery_variant=source_trace.C2, runtime_integration_mode="native",
        source_manifest_sha256=histories["full_prrac"]["manifest"]["manifest_sha256"],
        anchor_definition="max(0,min(valid first_navigation_target_difference_step,first_waypoint_cursor_difference_step)-1); all earliest tie agents",
        same_state_hash_definition="deterministic step-zero full_prrac replay; strict existing runtime_fingerprint + explicit boundary/observations/raw/full applied commands and original pure waypoint prior preview; per-component inventory/exclusions retained",
        intervention_definition="after original mode/continuity adapter, before env.step; alpha multiplies selected Searcher commands only for [anchor,anchor+10); original physics/clip unchanged",
        immediate_definition="state at anchor+1, after the first intervention transition; endpoint at anchor+10 is state-only, not an 11th action",
        position_separation_at_horizon_definition="maximum same-agent R0/alpha Euclidean distance over three Searchers at post-tenth-transition endpoint",
        control_replay_tolerance=analysis.TOLERANCE, branch_comparison_tolerance=analysis.BRANCH_TOLERANCE,
        training_update=False, explore=False, smoke=args.max_per_transition is not None,
        limitations=analysis.LIMITATIONS, status="running")
    output.mkdir(parents=True, exist_ok=True)
    evaluator._write_json(output/"branch_sensitivity_manifest.json", manifest)
    write_csv(output/"selected_anchors.csv", manifest["selected"])
    results = []
    try:
        def accept(result):
            results.append(result)
            evaluator._write_json(output/"progress.json", dict(completed=len(results), total=len(jobs)))
        source_trace.collect_jobs(jobs, args.workers, worker=branch_worker, on_result=accept)
        scenario_rows, step_rows, checks = [], [], []
        for entry in anchors:
            sid = entry["scenario"]["scenario_id"]
            rows, steps, check = analysis.summarize_scenario([r for r in results if r["scenario_id"] == sid],
                data["full_prrac"]["action"][sid], data["full_prrac"]["mission"][sid], expected_alphas=alphas)
            scenario_rows.extend(rows)
            step_rows.extend(steps)
            checks.append(check)
        if any(source_trace.sha(path) != value for path, value in hashes.items()):
            raise RuntimeError("protected input changed during diagnostic")
        scenario_rows.sort(key=lambda r: (str(r["scenario_id"]), r["alpha"]))
        step_rows.sort(key=lambda r: (str(r["scenario_id"]), r["alpha"], r["step"], r["agent_id"]))
        write_csv(output/"branch_sensitivity_scenario.csv", scenario_rows)
        write_csv(output/"branch_sensitivity_step_trace.csv", step_rows)
        manifest.update(status="completed" if all(c["interpretation_allowed"] for c in checks) else "failed",
            scenario_checks=checks, anchor_fingerprints=[dict(scenario_id=r["scenario_id"], alpha=r["alpha"], fingerprint=r["anchor_fingerprint"]) for r in sorted(results, key=lambda r: (r["scenario_id"], r["alpha"]))],
            output_sha256={name: source_trace.sha(output/name) for name in ("selected_anchors.csv", "branch_sensitivity_scenario.csv", "branch_sensitivity_step_trace.csv")})
        evaluator._write_json(output/"branch_sensitivity_manifest.json", manifest)
        return 0 if manifest["status"] == "completed" else 2
    except Exception:
        manifest.update(status="failed", error=traceback.format_exc())
        evaluator._write_json(output/"branch_sensitivity_manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "trace-root", "paired-first-divergence", "checkpoint", "config", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--max-per-transition", type=int)
    parser.add_argument("--alphas", nargs="+", default=list(analysis.ALPHAS))
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        parser.exit(1, f"branch diagnostic failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
