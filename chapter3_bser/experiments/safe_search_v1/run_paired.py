"""Serial, checkpoint-free B0/B1 safe-search ablations on an existing manifest.

The default is a bounded mechanism experiment. The original 400-step task is
never shortened: a surviving prefix is budget_cutoff, not a task timeout.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASELINES = ("B0_search_prior", "B1_bser_prior")
VARIANTS = ("V0", "V1", "V2", "V3", "V4")
SUPPORTED_VARIANTS = (*VARIANTS, "V5")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    def reject(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def parse_variants(value):
    selected = value.split(",") if isinstance(value, str) else list(value)
    if not selected or len(set(selected)) != len(selected) or any(v not in SUPPORTED_VARIANTS for v in selected):
        raise ValueError("variants must be unique comma-separated V0,V1,V2,V3,V4,V5")
    return selected


def select_indices(value, count):
    if value == "all":
        return list(range(count))
    try:
        selected = [int(v) for v in value.split(",")]
    except (ValueError, AttributeError) as exc:
        raise ValueError("episode indices must be all or comma-separated original zero-based indices") from exc
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= count for i in selected):
        raise ValueError("episode indices must be unique and inside the original manifest")
    return selected


def validate_request(*, baseline, variants, steps, full_episodes, seed, verify_v0):
    if baseline not in BASELINES:
        raise ValueError("only checkpoint-free B0/B1 are supported")
    variants = parse_variants(variants)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 400:
        raise ValueError("steps must be an integer in 1..400")
    if steps > 100 and not full_episodes:
        raise ValueError("more than 100 physical steps requires explicit --full-episodes")
    if full_episodes and steps != 400:
        raise ValueError("--full-episodes requires --steps 400")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must fit an unsigned 32-bit innovation seed")
    if verify_v0 and "V0" not in variants:
        raise ValueError("--verify-v0 requires V0 in --variants")
    return variants


def validate_manifest(manifest, config, indices, seed):
    """Validate the entire source manifest, not only the selected subset."""
    scenes = manifest.get("scenarios")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain a nonempty scenarios list")
    seen = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("scenario must be an object")
        name, scenario_seed = scene.get("scenario_id"), scene.get("scenario_seed")
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError("missing or duplicate scenario identity")
        seen.add(name)
        if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int) or not 0 <= scenario_seed < 2**63:
            raise ValueError("invalid scenario seed")
        if scene.get("scenario_split") != "validation" or scene.get("scenario_role", "validation") != "validation":
            raise ValueError("safe-search comparison requires validation scenarios")
        if scene.get("scenario_profile") != config["profile"]:
            raise ValueError("scenario profile differs from runtime config")
        if type(scene.get("max_steps")) is not int or scene["max_steps"] < config["max_steps"]:
            raise ValueError("scenario horizon is shorter than the unchanged task horizon")
    selected = select_indices(indices, len(scenes))
    if seed + max(selected) >= 2**32:
        raise ValueError("seed + original episode index exceeds innovation seed range")
    return [(i, copy.deepcopy(scenes[i])) for i in selected]


def output_directory(output, inputs, root=ROOT):
    output = Path(output).resolve()
    root = Path(root).resolve()
    protected = [root / name for name in ("outputs", "3090结果", ".git", "core", "chapter3_bser", "configs", "tests", "tools", "scripts")]
    if output == root or any(output == p or p in output.parents for p in protected):
        raise ValueError("output cannot replace source code or retained experiment evidence")
    if any(output == Path(p).resolve() or output in Path(p).resolve().parents for p in inputs):
        raise ValueError("output cannot contain an input file")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("use a new or empty output directory")
    return output


def episode_metrics(runtime, observer, *, full_episodes, coverage=None):
    result = runtime.env.get_episode_result()
    found_step = runtime.found_step
    collision_step = result.get("first_collision_step")
    pre_collision = bool(collision_step is not None and
                         (found_step is None or collision_step <= found_step))
    telemetry = observer.result() if observer is not None else None
    agents = (telemetry or {}).get("agents", {})
    stalls = sum(agents.get(str(i), {}).get("motion_stall_proxy_steps", 0) for i in (0, 1, 2))
    holds = sum(agents.get(str(i), {}).get("hold_steps", 0) for i in (0, 1, 2))
    exposure = (telemetry or {}).get("counts", {}).get("pre_found_exposure_steps")
    coverage_summary = coverage.result() if coverage is not None else None
    return dict(physical_steps=runtime.step, terminal=bool(runtime.terminal),
        stop_reason=result.get("termination_reason") if runtime.terminal else "budget_cutoff",
        found_within_budget=found_step is not None, found_step=found_step,
        pre_found_collision=pre_collision,
        full_episode_completed=bool(runtime.terminal), full_episode_requested=full_episodes,
        pre_found_exposure_steps=exposure, searcher_motion_stall_proxy_agent_steps=stalls,
        searcher_hold_agent_steps=holds,
        searcher_motion_stall_proxy_fraction=None if not exposure else stalls / (3 * exposure),
        searcher_hold_fraction=None if not exposure else holds / (3 * exposure),
        effective_search_steps=(coverage_summary or {}).get("effective_observation_steps"),
        search_coverage=coverage_summary, episode_result=result, telemetry=telemetry,
        controller=runtime.controller_diagnostics())


def execute_episode(factory, output, *, steps, observed, deps, full_episodes=False):
    """One runtime only; injectable read-only dependencies permit bounded unit tests."""
    runtime = None
    observer = deps["TransitionObserver"]() if observed else None
    tap = deps["QueryTap"](deps["TravelCostService"]) if observed else None
    signatures = []
    started = time.perf_counter()
    try:
        with ExitStack() as stack:
            if observed:
                stack.enter_context(tap)
            runtime = factory()
            coverage_factory = deps.get("CoverageObserver") if observed else None
            coverage = coverage_factory(runtime) if coverage_factory is not None else None
            if coverage is not None:
                stack.enter_context(coverage)
            signatures.append(deps["physical_signature"](runtime, None))
            if observed:
                write_json(output / "initial_queries.json", tap.drain())
            with (output / "step_trace.jsonl").open("x", encoding="utf-8") if observed else nullcontext() as handle:
                for _ in range(steps):
                    if runtime.terminal:
                        break
                    before = deps["capture_runtime"](runtime) if observed else None
                    record = runtime.advance()
                    if observed:
                        after = deps["capture_runtime"](runtime)
                        terminal_result = runtime.env.get_episode_result() if runtime.terminal else None
                        row = observer.observe(before, after, tap.drain(), terminal_result)
                        if coverage is not None:
                            row["search_coverage"] = coverage.drain()
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        handle.flush()
                    signatures.append(deps["physical_signature"](runtime, record))
                    if runtime.step % 20 == 0:
                        print(f"SAFE_SEARCH_PROGRESS observed={observed} step={runtime.step}", flush=True)
            metrics = episode_metrics(runtime, observer, full_episodes=full_episodes, coverage=coverage)
            metrics["wall_seconds"] = time.perf_counter() - started
            if runtime.terminal:
                metrics["complete_episode_row"] = deps["complete_episode_row"](runtime, metrics["wall_seconds"])
            else:
                metrics["complete_episode_row"] = None
            return signatures, metrics
    finally:
        if runtime is not None:
            runtime.close()


def summarize(rows, variants, expected, full_episodes):
    by_variant = {}
    for variant in variants:
        arm = [r for r in rows if r["variant"] == variant]
        complete = len(arm) == expected and all(r["terminal"] for r in arm)
        exposure = sum(r["pre_found_exposure_steps"] or 0 for r in arm)
        stalls = sum(r["searcher_motion_stall_proxy_agent_steps"] for r in arm)
        holds = sum(r["searcher_hold_agent_steps"] for r in arm)
        effective = ([r.get("effective_search_steps") for r in arm])
        effective_total = (sum(effective) if effective and all(x is not None for x in effective) else None)
        found = sum(r["found_within_budget"] for r in arm)
        collisions = sum(r["pre_found_collision"] for r in arm)
        by_variant[variant] = dict(n_expected=expected, n_recorded=len(arm),
            n_terminal=sum(r["terminal"] for r in arm),
            n_budget_cutoff=sum(r["stop_reason"] == "budget_cutoff" for r in arm),
            found_within_budget_count=found, pre_found_collision_count=collisions,
            found_within_budget_fraction=found / len(arm) if arm else None,
            found_rate=found / expected if complete and full_episodes else None,
            pre_found_collision_rate=collisions / expected if complete and full_episodes else None,
            full_episode_evaluation_complete=bool(complete and full_episodes),
            searcher_motion_stall_proxy_fraction=stalls / (3 * exposure) if exposure else None,
            searcher_hold_fraction=holds / (3 * exposure) if exposure else None,
            pre_found_exposure_steps=exposure, effective_search_steps=effective_total,
            total_physical_steps=sum(r["physical_steps"] for r in arm),
            total_wall_seconds=sum(r["wall_seconds"] for r in arm))
    reference = {r["original_episode_index"]: r for r in rows if r["variant"] == "V0"}
    paired = {}
    for variant in variants:
        if variant == "V0":
            continue
        comparisons = []
        for r in rows:
            if r["variant"] != variant or r["original_episode_index"] not in reference:
                continue
            base = reference[r["original_episode_index"]]
            comparisons.append(dict(original_episode_index=r["original_episode_index"],
                scenario_id=r["scenario_id"], found_within_budget_delta=int(r["found_within_budget"]) - int(base["found_within_budget"]),
                pre_found_collision_delta=int(r["pre_found_collision"]) - int(base["pre_found_collision"]),
                pre_found_exposure_steps_delta=r["pre_found_exposure_steps"] - base["pre_found_exposure_steps"],
                searcher_motion_stall_proxy_agent_steps_delta=r["searcher_motion_stall_proxy_agent_steps"] - base["searcher_motion_stall_proxy_agent_steps"]))
        paired[variant] = dict(reference="V0", pairs=comparisons,
            found_within_budget_mean_delta=sum(p["found_within_budget_delta"] for p in comparisons) / len(comparisons) if comparisons else None,
            pre_found_collision_mean_delta=sum(p["pre_found_collision_delta"] for p in comparisons) / len(comparisons) if comparisons else None)
    return dict(schema="ch3.safe_search.paired_summary.v1", variants=by_variant, paired_vs_v0=paired,
        all_requested_runs_recorded=len(rows) == expected * len(variants),
        training=False, checkpoint_loaded=False, performance_passed=None,
        caveats=["Budget cutoffs are not task timeouts; Found rates require complete 400-step-horizon evaluation.",
                 "Motion stall and hold are movement proxies. Optional effective search counts actual observation-footprint new/revisit steps, not detection probability; unavailable coverage stays null.",
                 "Manifest validation labels do not establish independence from prior training or tuning."])


def load_dependencies(config, scenario, index, seed, baseline, *, search_coverage=False):
    from core.mapping.travel_cost_service import TravelCostService
    from scripts.search_diagnostic_observer import QueryTap, TransitionObserver
    from .telemetry import capture_runtime
    from scripts.run_search_diagnostic_probe import physical_signature
    from tools.ch3_baselines.evaluate import episode_row
    from tools.ch3_baselines.registry import method_spec
    spec = method_spec(baseline)
    def complete_episode_row(runtime, wall):
        row = episode_row(runtime, config, scenario, index, seed, wall)
        row.update(reference_baseline=baseline, method=spec["method"],
                   reference_runtime_method=spec["runtime_method"])
        return row
    dependencies = dict(TravelCostService=TravelCostService, QueryTap=QueryTap,
                TransitionObserver=TransitionObserver, capture_runtime=capture_runtime,
                physical_signature=physical_signature, complete_episode_row=complete_episode_row)
    if search_coverage:
        from .search_coverage import CoverageObserver
        dependencies["CoverageObserver"] = CoverageObserver
    return dependencies


def run_experiment(manifest_path, output_dir, *, config_path=ROOT / "configs/chapter3/hgr_train.json",
                   baseline="B0_search_prior", variants="V0,V1,V2,V3,V4", episode_indices="all",
                   steps=100, seed=12729, full_episodes=False, verify_v0=False, search_coverage=False):
    variants = validate_request(baseline=baseline, variants=variants, steps=steps,
        full_episodes=full_episodes, seed=seed, verify_v0=verify_v0)
    manifest_path, config_path = Path(manifest_path).resolve(), Path(config_path).resolve()
    output = output_directory(output_dir, (manifest_path, config_path))
    raw_config, manifest = read_json(config_path), read_json(manifest_path)
    if raw_config.get("max_steps") != 400:
        raise ValueError("safe-search v1 preserves the 400-step task horizon")
    selected = validate_manifest(manifest, raw_config, episode_indices, seed)
    from chapter3_bser.experiments.hgr.train import validate_config
    from tools.ch3_baselines.registry import task_conditions, method_spec
    from .runtime import make_runtime
    from .provenance import framework_sources, verify_sources
    from tools.ch3_baselines.basic_search_prior import BasicSearchPriorRuntime
    from tools.ch3_baselines.bser_prior import BSERPriorRuntime
    import numpy as np
    import torch
    config = validate_config(raw_config)
    conditions = task_conditions(config, episodes=len(manifest["scenarios"]), seed=seed)
    spec = method_spec(baseline)
    sources = framework_sources()
    input_hashes = {str(p): file_hash(p) for p in (config_path, manifest_path)}
    immutable_inputs = digest(dict(config=config, selected=selected))
    def verify():
        verify_sources(sources)
        if any(file_hash(p) != value for p, value in input_hashes.items()):
            raise RuntimeError("input file changed during experiment")
        if immutable_inputs != digest(dict(config=config, selected=selected)):
            raise RuntimeError("in-memory scenario/config changed during experiment")
    identity = dict(schema="ch3.safe_search.paired_identity.v1", baseline=baseline, variants=variants,
        seed=seed, seed_rule="seed + original zero-based manifest index; never subset index",
        steps=steps, task_horizon=400, full_episodes=full_episodes,
        search_coverage_enabled=bool(search_coverage),
        training=False, checkpoint_loaded=False, residual_source="zeros_4x3",
        formal_thesis_evaluation=False, input_sha256=input_hashes, sources_before=sources,
        common_task_conditions=conditions, baseline_spec=spec,
        selected=[dict(original_episode_index=i, scenario_id=s["scenario_id"], scenario_seed=s["scenario_seed"],
            environment_innovation_seed=seed+i, scenario_sha256=digest(s)) for i, s in selected],
        environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
            torch_num_threads=1, omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            mkl_num_threads=os.environ.get("MKL_NUM_THREADS")))
    output_directory(output, (manifest_path, config_path))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "identity.json", identity)
    write_json(output / "resolved_config.json", config)
    write_json(output / "evaluation_manifest.json", manifest)
    rows = []
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for index, scenario in selected:
            deps = load_dependencies(config, scenario, index, seed+index, baseline, search_coverage=search_coverage)
            reference_signatures = None
            if verify_v0:
                verify()
                cls = BasicSearchPriorRuntime if baseline == "B0_search_prior" else BSERPriorRuntime
                reference_signatures, _ = execute_episode(lambda: cls(copy.deepcopy(config), copy.deepcopy(scenario),
                    seed=seed+index, episode_id=index), output, steps=steps, observed=False, deps=deps,
                    full_episodes=full_episodes)
                verify()
            for variant in variants:
                verify()
                arm_dir = output / f"episode_{index:04d}" / variant
                arm_dir.mkdir(parents=True, exist_ok=False)
                signatures, row = execute_episode(lambda: make_runtime(copy.deepcopy(config), copy.deepcopy(scenario),
                    baseline=baseline, variant=variant, seed=seed+index, episode_id=index),
                    arm_dir, steps=steps, observed=True, deps=deps, full_episodes=full_episodes)
                verify()
                equal = signatures == reference_signatures if variant == "V0" and verify_v0 else None
                if equal is False:
                    raise RuntimeError("V0 or diagnostic observer changed physical trajectory/reward/guidance/RNG")
                row.update(variant=variant, baseline=baseline, original_episode_index=index,
                    scenario_id=scenario["scenario_id"], scenario_seed=scenario["scenario_seed"],
                    environment_innovation_seed=seed+index, signature_sha256=digest(signatures),
                    v0_matches_original_at_every_step=equal)
                write_json(arm_dir / "summary.json", row)
                rows.append(row)
                write_json(output / "episodes.json", rows)
                write_json(output / "summary.json", summarize(rows, variants, len(selected), full_episodes))
                print(f"SAFE_SEARCH_RESULT index={index} variant={variant} steps={row['physical_steps']} "
                      f"stop={row['stop_reason']} found={row['found_within_budget']} v0_equal={equal}", flush=True)
        verify()
        final = summarize(rows, variants, len(selected), full_episodes)
        final["source_and_input_verification_passed"] = True
        identity["sources_after"] = framework_sources()
        identity["source_and_input_verification_passed"] = True
        write_json(output / "identity.json", identity)
        write_json(output / "summary.json", final)
        return final
    except BaseException as exc:
        write_json(output / "failure.json", dict(exception_type=type(exc).__name__, message=str(exc),
            completed_runs=len(rows), experiment_complete=False))
        raise
    finally:
        torch.set_num_threads(old_threads)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--config", type=Path, default=ROOT / "configs/chapter3/hgr_train.json")
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--baseline", choices=BASELINES, default="B0_search_prior")
    result.add_argument("--variants", default="V0,V1,V2,V3,V4")
    result.add_argument("--episode-indices", default="all")
    result.add_argument("--steps", type=int, default=100)
    result.add_argument("--seed", type=int, default=12729)
    result.add_argument("--full-episodes", action="store_true")
    result.add_argument("--verify-v0", action="store_true")
    result.add_argument("--search-coverage", action="store_true",
                        help="Read-only actual target-belief observation-footprint diagnostics")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    run_experiment(args.manifest, args.output_dir, config_path=args.config, baseline=args.baseline,
        variants=args.variants, episode_indices=args.episode_indices, steps=args.steps, seed=args.seed,
        full_episodes=args.full_episodes, verify_v0=args.verify_v0, search_coverage=args.search_coverage)


if __name__ == "__main__":
    main()
