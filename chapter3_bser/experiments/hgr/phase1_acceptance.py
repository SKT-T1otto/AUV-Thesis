"""NEW manual-only bounded runtime acceptance with fresh actors, no training."""
import argparse
import copy
import multiprocessing as mp
from pathlib import Path
import traceback

import torch

from chapter3_bser.models.hgr.phase1 import (
    PairNoise, REVISION, STREAM_REVISION, phase1_options, named_seed, isolated_global_rng,
)
from chapter3_bser.models.hgr.policy import HandoffPolicy
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from .runtime import collect_trajectory, continue_branch, DecisionSnapshot, seed_innovations
from .train import load_config, validated_output, write_json
from .provenance import source_identity, checkout_identity

ROOT = Path(__file__).resolve().parents[3]
PREFIX_COUNT = 2
PAIR_COUNT = 6
HORIZON = 400
MAX_ENVIRONMENT_STEPS = (PREFIX_COUNT + 2 * PAIR_COUNT) * HORIZON
SCENARIO_GENERATOR_SEED = 20260924


def plan(config):
    return dict(runner="new_phase1_manual_acceptance_v1", phase1_revision=REVISION,
                random_source_revision=STREAM_REVISION, profile=config["profile"], horizon=HORIZON,
                scenario_generator_seed=SCENARIO_GENERATOR_SEED, scenario_split="train",
                prefix_count=PREFIX_COUNT, pair_count_including_replays_and_spawn=PAIR_COUNT,
                maximum_environment_steps=MAX_ENVIRONMENT_STEPS,
                boundary_selection="first natural handoff in predeclared prefix order",
                queries=["policy_crn", "exact replay of query 0", "spawn replay of query 0",
                         "swap policy AND source terms of query 0",
                         "independent policy/environment", "fresh policy_crn"],
                policy="fresh initialization; new phi.log_std += 0.01; no optimizer",
                training=False, checkpoint_restore=False, performance_claims_supported=False)


def _spawn_pair(queue, snapshot, old, new, old_noise, new_noise):
    try:
        torch.set_num_threads(1)
        with isolated_global_rng():
            a = continue_branch(snapshot, old, pair_noise=old_noise)
            queue.put(("partial", a))
            b = continue_branch(snapshot, new, pair_noise=new_noise)
            queue.put(("done", b))
    except BaseException:
        queue.put(("error", traceback.format_exc()))


def _query(snapshot, old, new, noises, *, spawned, accept_result):
    if not spawned:
        a = continue_branch(snapshot, old, pair_noise=noises[0]); accept_result(a)
        b = continue_branch(snapshot, new, pair_noise=noises[1]); accept_result(b)
        return a, b
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_pair, args=(queue, snapshot, old, new, *noises))
    process.start()
    results = []
    try:
        # A timeout fails the diagnostic; it never silently truncates a return.
        for _ in range(2):
            status, result = queue.get(timeout=600)
            if status == "error":
                raise RuntimeError("spawn continuation failed:\n" + result)
            results.append(result); accept_result(result)
        process.join(timeout=30)
        if process.exitcode != 0:
            raise RuntimeError(f"spawn worker exit status {process.exitcode}")
        return tuple(results)
    finally:
        if process.is_alive():
            process.terminate(); process.join()
        queue.close()


def _same_trace(left, right):
    keys = ("G_plus", "steps", "terminal_step", "termination_reason", "trajectory_trace_sha256")
    if any(left[key] != right[key] for key in keys):
        raise AssertionError("identical policy/state/innovations did not reproduce the complete branch trace")


def run(config, output):
    if phase1_options(config) is None:
        raise ValueError("manual phase1 acceptance requires an explicit phase1 config")
    output = validated_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="RUNNING", plan=plan(config), source_identity=source_identity(),
                  checkout=checkout_identity(), prefix_results=[], branch_results=[], checks={},
                  recorded_environment_steps=0, failed_call_steps_unknown=False,
                  true_runtime_acceptance="NOT_RUN", efficiency_benefit="NOT_RUN")
    write_json(output/"plan.json", report["plan"])
    write_json(output/"config.json", config)

    def persist():
        write_json(output/"acceptance.json", report)

    def accept_result(result):
        if (result["original_deadline"] != HORIZON
                or result["terminal_step"] > HORIZON
                or result["steps"] != result["terminal_step"]-result["initial_task_step"]):
            raise AssertionError("branch clock/deadline mismatch")
        report["branch_results"].append(result)
        report["recorded_environment_steps"] += result["steps"]
        persist()

    persist()
    try:
        torch.set_num_threads(1)
        with isolated_global_rng():
            _, init_seed = named_seed(config["seed"], 0, "manual/policy_initialization")
            seed_innovations(init_seed, cpu_only=True)
            old = HandoffPolicy(config["policy"])
            new = copy.deepcopy(old)
            with torch.no_grad():
                new.phi.log_std.add_(.01)
        scenarios = build_scenario_manifests(
            count=PREFIX_COUNT, generator_seed=SCENARIO_GENERATOR_SEED, split="train",
            profiles=[config["profile"]])[config["profile"]]["scenarios"]
        write_json(output/"predeclared_scenarios.json", scenarios)
        selected = None
        for index, scenario in enumerate(scenarios):
            stream_id, seed = named_seed(config["seed"], 0, "manual/prefix", index)
            trajectory = collect_trajectory(config, scenario, old, seed=seed, episode_id=index,
                                            stop_at_boundary=True)
            report["recorded_environment_steps"] += len(trajectory["records"])
            report["prefix_results"].append(dict(index=index, stream_id=stream_id, seed=seed,
                                                  tau=trajectory["tau"], **trajectory["summary"]))
            if selected is None and trajectory["snapshot"] is not None:
                selected = trajectory["snapshot"]
            persist()
        if selected is None:
            report.update(status="NOT_EXERCISED", true_runtime_acceptance="BLOCKED",
                          blocker="No natural reliable handoff in the two predeclared tasks; no retry or injection.")
            persist()
            return 2
        selected.save(output/"fresh_decision_snapshot.pkl")
        selected = DecisionSnapshot.load(output/"fresh_decision_snapshot.pkl")
        base = tuple(PairNoise.make(config["seed"], 0, "manual/branch", 0, role, "policy_crn")
                     for role in ("old", "new"))
        pairs = []
        for query in range(PAIR_COUNT):
            if query <= 2:
                policies, noises = (old,new), base
            elif query == 3:
                # Replay antisymmetry, not a fresh independent correction draw.
                policies, noises = (new,old), (base[1],base[0])
            else:
                mode = "independent" if query == 4 else "policy_crn"
                policies = (old,new)
                noises = tuple(PairNoise.make(config["seed"],0,"manual/branch",query,role,mode)
                               for role in ("old","new"))
            a,b = _query(selected,*policies,noises,spawned=query==2,accept_result=accept_result)
            pairs.append((a,b))
            if query in (1,2):
                _same_trace(pairs[0][0],a); _same_trace(pairs[0][1],b)
            elif query == 3:
                _same_trace(pairs[0][1],a); _same_trace(pairs[0][0],b)
                if b["G_plus"]-a["G_plus"] != -(pairs[0][1]["G_plus"]-pairs[0][0]["G_plus"]):
                    raise AssertionError("paired replay antisymmetry failed")
        if report["recorded_environment_steps"] > MAX_ENVIRONMENT_STEPS:
            raise AssertionError("declared environment-step bound exceeded")
        report["checks"] = dict(boundary_clock_restore="PASS", original_deadline="PASS",
                                actual_branch_termination="PASS", serial_replay_trace="PASS",
                                spawn_replay_trace="PASS", swap_with_sources="PASS",
                                coupling_scope="policy_only_environment_independent",
                                environmental_component_alignment="NOT_CLAIMED",
                                conditional_variance_reduction="NOT_RUN", complete_gradient_efficiency="NOT_RUN")
        report.update(status="PASS",true_runtime_acceptance="PASS")
        persist()
        return 0
    except BaseException:
        report.update(status="FAIL",true_runtime_acceptance="FAIL", error=traceback.format_exc(),
                      failed_call_steps_unknown=True,
                      cost_note="Recorded successful calls only; a failed in-flight call may have additional steps.")
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/chapter3/hgr_phase1_zero.json"))
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Predeclared worst-case budget: {MAX_ENVIRONMENT_STEPS} environment steps, including replay/spawn.",flush=True)
    status = run(config,args.output)
    print(f"Acceptance exit code: {status}. See {Path(args.output).resolve()/'acceptance.json'}",flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
