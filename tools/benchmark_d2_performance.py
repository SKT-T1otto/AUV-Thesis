"""Opt-in bounded D2 timing and semantic signatures; never a formal experiment.

Run as ``python -m tools.benchmark_d2_performance --output runs/...``.
Instrumentation is installed only inside this command, not in training imports.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import copy
import functools
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENES = "configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json"


def benchmark_model_builder():
    """The same CPU benchmark also runs on the frozen pre-repair checkout.

    Copy only this tool into that checkout; no new production module is needed.
    """
    try:
        from chapter3_bser.experiments.d2_performance.compute import build_model
    except ModuleNotFoundError as exc:
        if exc.name not in ("chapter3_bser.experiments.d2_performance",
                            "chapter3_bser.experiments.d2_performance.compute"):
            raise
        from chapter3_bser.experiments.baselines.common.model import build_model
    return build_model


class Timings:
    """Nested inclusive/exclusive wall timers, with no RNG or simulator writes."""
    def __init__(self, detailed=False):
        self.detailed = detailed
        self.rows = defaultdict(lambda: dict(calls=0, seconds=0.0, self_seconds=0.0))
        self.stack = []
        self.patches = ExitStack()

    def wrap(self, function, label):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            frame = [time.perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                row = self.rows[label]
                row["calls"] += 1
                row["seconds"] += elapsed
                row["self_seconds"] += elapsed - frame[1]
                if self.stack:
                    self.stack[-1][1] += elapsed
        return timed

    def method(self, module, cls, name, label):
        owner = getattr(importlib.import_module(module), cls)
        self.patches.enter_context(patch.object(owner, name, self.wrap(getattr(owner, name), label)))

    def function(self, module, name, label):
        original = getattr(importlib.import_module(module), name)
        wrapped = self.wrap(original, label)
        # Capture every already-imported alias, including the provider's import.
        for mod in tuple(sys.modules.values()):
            if mod is None or not getattr(mod, "__name__", "").startswith(("core.", "chapter3_bser.")):
                continue
            for key, value in tuple(vars(mod).items()):
                if value is original:
                    self.patches.enter_context(patch.object(mod, key, wrapped))

    def __enter__(self):
        # Import the runtime first so all its function aliases can be instrumented.
        import chapter3_bser.experiments.baselines.common.train
        import chapter3_bser.experiments.d2_v1.assembly
        import chapter3_bser.experiments.d2_v1.controller
        import chapter3_bser.experiments.safe_search_v1.safe_guidance
        prefix = "chapter3_bser.experiments."
        methods = [
            ("core.algorithms.maddpg", "MADDPG", "step", "A_rollout_forward"),
            (prefix + "phase1c_prrac.training_env", "PRRACTrainingEnv", "step", "B_env_step"),
            ("chapter3_bser.controllers.state_provider", "OnlinePlanningStateProvider", "snapshot", "C_snapshot"),
            (prefix + "safe_search_v1.runtime", "SearchController", "step", "F_controller"),
            (prefix + "safe_search_v1.public_geometry", "PublicGeometry", "__init__", "G_geometry_construct"),
            (prefix + "safe_search_v1.public_geometry", "PublicGeometry", "path_free", "G_path_free"),
            (prefix + "safe_search_v1.public_geometry", "PublicGeometry", "segment_free", "G_segment_free"),
            ("core.mapping.travel_cost_service", "TravelCostService", "query", "H_route_query"),
            ("core.mapping.travel_cost_service", "TravelCostService", "_astar", "H_astar"),
            (prefix + "safe_search_v1.safe_guidance", "SafetyBridge", "compile_guidance", "I_compile_guidance"),
            ("core.replay.ch3_buffer", "CH3ReplayBuffer", "sample", "J_replay_sample"),
            ("core.algorithms.maddpg", "MADDPG", "update", "K_model_update"),
            ("core.algorithms.maddpg", "MADDPG", "update_all_targets", "K_target_update"),
            (prefix + "baselines.common.train", "BaselineTrainer", "save", "M_checkpoint_save_validate"),
        ]
        if self.detailed:
            methods += [
            ("core.env.uav_env", "UAVEnv", "_sense_and_update_shared_map", "B_sense_map"),
            ("core.env.uav_env", "UAVEnv", "_refresh_intercept", "B_intercept"),
            ("core.env.uav_env", "UAVEnv", "_get_obs", "B_observations"),
            ("core.mapping.path_planner", "OnlineUnknownMapTaskPlanner", "integrate_obstacle_scan", "B_integrate_scan"),
            ("core.mapping.path_planner", "OnlineUnknownMapTaskPlanner", "predict_belief_motion", "B_belief_motion"),
            ("core.mapping.path_planner", "OnlineUnknownMapTaskPlanner", "_segment_is_free_np", "E_planner_segment"),
            ]
        for args in methods:
            self.method(*args)
        for args in [
            ("core.mapping.planning_state", "extract_planning_state", "D_extract_planning_state"),
            ("core.mapping.planning_graph", "build_planning_graph", "E_build_planning_graph"),
            ("core.mapping.planning_graph", "_build_endpoint", "E_endpoint"),
            ("core.mapping.planning_graph", "_build_role_adjacency", "E_adjacency"),
            ("core.mapping.planning_graph", "_graph_hash", "E_graph_hash"),
            (prefix + "safe_search_v1.public_geometry", "filter_planning_state", "G_filter_graph"),
            (prefix + "baselines.common.checkpoint", "write_json", "L_json_write"),
            (prefix + "baselines.common.checkpoint", "load_checkpoint", "M_checkpoint_load_validate"),
        ]:
            self.function(*args)
        try:
            module = importlib.import_module(prefix + "d2_performance.logging")
        except ModuleNotFoundError:
            pass
        else:
            self.method(module.__name__, "JsonlJournal", "append", "L_jsonl_append")
            self.method(module.__name__, "JsonlJournal", "materialize", "L_json_materialize")
        return self

    def __exit__(self, *exc):
        return self.patches.__exit__(*exc)

    def snapshot(self):
        return copy.deepcopy(dict(self.rows))


def difference(after, before):
    return {key: {field: value - before.get(key, {}).get(field, 0) for field, value in row.items()}
            for key, row in after.items()}


def record(timing, before, started, steps, **extra):
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    return dict(steps=steps, wall_seconds=wall, env_steps_per_second=steps / wall,
                timings=difference(timing.snapshot(), before), **extra)


def update_signature(hashed, transition):
    for key in ("observations", "actions", "rewards", "next_observations", "dones"):
        value = np.asarray(transition[key])
        hashed.update(key.encode())
        hashed.update(str(value.dtype).encode())
        hashed.update(value.tobytes())


def run(args):
    from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
    build_model = benchmark_model_builder()
    from chapter3_bser.experiments.baselines.common.checkpoint import fresh_source_identity, state_digest
    from chapter3_bser.experiments.baselines.direct_mc.train import DirectMCTrainer
    from chapter3_bser.experiments.baselines.direct_boundary.train import DirectBoundaryTrainer
    config_name = "direct_mc_train.json" if args.arm == "B2" else "direct_boundary_train.json"
    config = json.loads((ROOT / "configs/chapter3/d2_v1" / config_name).read_text())
    config.update(seed=args.seed, total_main_trajectories=2, checkpoint_interval=100)
    if args.device != "cpu" or args.threads != 1:
        config["performance"] = dict(learner_device=args.device, cpu_threads=args.threads)
    scenes_path = ROOT / args.manifest
    scenes = json.loads(scenes_path.read_text())["scenarios"]
    selected = [scenes[i] for i in args.scenes]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise ValueError("explicit CUDA benchmark requested but CUDA is unavailable")
    result = dict(schema="ch3.d2.performance.benchmark.v1", formal_experiment=False,
                  label=args.label, arm=args.arm, seed=args.seed, threads=args.threads, device=args.device,
                  platform=platform.platform(), processor=platform.processor(), torch=torch.__version__,
                  cuda_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  source=fresh_source_identity(), manifest=args.manifest,
                  manifest_sha256=hashlib.sha256(scenes_path.read_bytes()).hexdigest(), cases=[])
    def save():
        (output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with Timings() as timing:
        for index, scene in zip(args.scenes, selected):
            # Cold initialization is timed separately. Five real warmup steps
            # precede the requested 100-step window; task horizon stays 400.
            torch.manual_seed(args.seed)
            model = build_model(config, device=args.device)
            model.prep_rollouts(device=args.device)
            before, started = timing.snapshot(), time.perf_counter()
            runtime = BaselineMissionRuntime(config, scene, seed=args.seed, episode_id=1)
            result["cases"].append(record(timing, before, started, 0, name="initialize", scene=index))
            trace = hashlib.sha256()
            rewards = []
            try:
                for _ in range(5):
                    if runtime.terminal: break
                    transition = runtime.advance(model, explore=True)
                    update_signature(trace, transition)
                    rewards.append(transition["rewards"].tolist())
                first = runtime.step
                before, started = timing.snapshot(), time.perf_counter()
                while not runtime.terminal and runtime.step < min(first + 100, 400):
                    transition = runtime.advance(model, explore=True)
                    update_signature(trace, transition)
                    rewards.append(transition["rewards"].tolist())
                warm = record(timing, before, started, runtime.step - first, name="warm100", scene=index)
                result["cases"].append(warm)
                while not runtime.terminal and runtime.step < 400:
                    transition = runtime.advance(model, explore=True)
                    update_signature(trace, transition)
                    rewards.append(transition["rewards"].tolist())
                result["cases"].append(record(timing, before, started, runtime.step - first,
                    name="episode_after_5_warmup", scene=index, actual_length=runtime.step,
                    summary=runtime.summary(), transition_sha256=trace.hexdigest(), rewards=rewards))
                print(args.label, index, "warm100", round(warm["env_steps_per_second"], 3), flush=True)
            finally:
                runtime.close()
            save()
        trainer_cls = DirectMCTrainer if args.arm == "B2" else DirectBoundaryTrainer
        trainer = trainer_cls(config, output / "collision_terminal" / "training")
        original_advance = BaselineMissionRuntime.advance
        trace = hashlib.sha256()
        rewards = []
        def observed(runtime, *a, **kw):
            transition = original_advance(runtime, *a, **kw)
            update_signature(trace, transition)
            rewards.append(transition["rewards"].tolist())
            return transition
        # Existing development M20 scenes, used only for bounded computational
        # verification. No relabeling or publication as formal training data.
        trainer._scenario = lambda episode: (copy.deepcopy(selected[(episode-1) % len(selected)]), args.seed + episode)
        with patch.object(BaselineMissionRuntime, "advance", observed):
            for episode in range(args.training_episodes):
                before, started = timing.snapshot(), time.perf_counter()
                row = trainer.run_episode()
                result["cases"].append(record(timing, before, started, row["episode_steps"],
                    name="training_episode", episode=episode+1, summary=row,
                    transition_sha256=trace.hexdigest(), rewards=copy.deepcopy(rewards),
                    model_sha256=state_digest(trainer.model.training_state_dict())))
                save()
        before, started = timing.snapshot(), time.perf_counter()
        trainer.save(final=True)
        result["cases"].append(record(timing, before, started, 0, name="checkpoint"))
        result["global_step"] = trainer.global_step
        result["optimizer_updates"] = trainer.optimizer_updates
    save()
    return result


def environment_profile(args):
    """Ten real steps with cProfile: diagnose env internals, not throughput."""
    import cProfile
    import pstats
    from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
    build_model = benchmark_model_builder()
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/chapter3/d2_v1/direct_mc_train.json").read_text())
    scene = json.loads((ROOT / args.manifest).read_text())["scenarios"][args.scenes[0]]
    torch.manual_seed(args.seed)
    model = build_model(config)
    runtime = BaselineMissionRuntime(config, scene, seed=args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    profiler = cProfile.Profile()
    try:
        with Timings(detailed=True) as timings:
            profiler.enable()
            for _ in range(10):
                if runtime.terminal:
                    break
                runtime.advance(model, explore=True)
            profiler.disable()
        (output / "timings.json").write_text(json.dumps(timings.snapshot(), indent=2), encoding="utf-8")
        with (output / "cprofile.txt").open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(65)
        profiler.dump_stats(str(output / "profile.pstats"))
    finally:
        runtime.close()


def compute_profile(args):
    """Fixed synthetic replay microbenchmark, separate from real episodes."""
    build_model = benchmark_model_builder()
    from core.replay.ch3_buffer import CH3ReplayBuffer
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs/chapter3/d2_v1/direct_mc_train.json").read_text())
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was explicitly requested but is unavailable")
    model = build_model(config, device=args.device)
    replay = CH3ReplayBuffer(512, 4, [28]*4, [3]*4, success_priority=1., alpha=0., beta_start=0.)
    rng = np.random.default_rng(args.seed)
    for _ in range(256):
        replay.push(rng.normal(size=(4, 28)), rng.uniform(-1, 1, (4, 3)),
                    rng.normal(size=4), rng.normal(size=(4, 28)), [False]*4, [False]*4)
    rows = []
    for index in range(23):
        if args.device == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        sample = replay.sample(config["rl"]["batch_size"], norm_rews=False, device=args.device)
        if args.device == "cuda": torch.cuda.synchronize()
        sampled = time.perf_counter()
        for agent in range(4):
            model.update(sample, agent)
        model.update_all_targets()
        if args.device == "cuda": torch.cuda.synchronize()
        if index >= 3:
            rows.append(dict(sample_seconds=sampled-started, update_seconds=time.perf_counter()-sampled))
    value = dict(synthetic_fixed_replay=True, batch_size=128, agents=4, measured_updates=len(rows),
                 warmup_updates=3, threads=args.threads, device=args.device,
                 gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                 mean_sample_seconds=float(np.mean([r["sample_seconds"] for r in rows])),
                 mean_update_seconds=float(np.mean([r["update_seconds"] for r in rows])), rows=rows)
    (output / "compute.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in value.items() if k != "rows"}), flush=True)


def logging_profile(args):
    """Compare a 190k-row history rewrite to 200 new journal records."""
    from chapter3_bser.experiments.baselines.common.checkpoint import write_json
    from chapter3_bser.experiments.d2_performance.logging import JsonlJournal
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    row = dict(global_step=380000, optimizer_update=190000,
               actor_loss_by_agent=[.1284312, -.2399375, .215378, -.094329],
               critic_loss_by_agent=[.621389, .487113, .513372, .087139])
    history = [dict(row, optimizer_update=i+1) for i in range(190000)]
    journal = JsonlJournal(output / "training_metrics.jsonl")
    # Seed the old history outside the timed window: a running journal already
    # contains these bytes. Production uses append throughout its lifetime.
    with journal.path.open("ab") as stream:
        for record in history:
            stream.write((json.dumps(record, separators=(",", ":"))+"\n").encode())
    journal.count = len(history)
    old, new = [], []
    for _ in range(3):
        records = [dict(row, optimizer_update=len(history)+i+1) for i in range(200)]
        history.extend(records)
        started = time.perf_counter()
        write_json(output / "training_metrics.json", history)
        old.append(time.perf_counter()-started)
        started = time.perf_counter()
        for record in records:
            journal.append(record)
        new.append(time.perf_counter()-started)
    value = dict(history_rows=190000, appended_rows_per_episode=200, repeats=3,
                 legacy_pretty_json_bytes=(output / "training_metrics.json").stat().st_size,
                 before_seconds=old, after_seconds=new, speedup=float(np.mean(old)/np.mean(new)))
    (output / "logging.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="current")
    parser.add_argument("--arm", choices=("B2", "B3"), default="B2")
    parser.add_argument("--manifest", default=DEFAULT_SCENES)
    parser.add_argument("--scenes", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seed", type=int, default=2729)
    parser.add_argument("--training-episodes", type=int, choices=(1, 2), default=2,
                        help="bounded real training episodes; formal config/cadence is unchanged")
    parser.add_argument("--threads", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile-environment", action="store_true")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--logging-only", action="store_true")
    args = parser.parse_args()
    if len(args.scenes) > 3:
        parser.error("bounded benchmark permits at most three scenes")
    if sum((args.profile_environment, args.compute_only, args.logging_only)) > 1:
        parser.error("choose only one diagnostic mode")
    if args.profile_environment:
        environment_profile(args)
    elif args.compute_only:
        compute_profile(args)
    elif args.logging_only:
        logging_profile(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
