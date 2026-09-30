"""One full-task evaluator and read-only search diagnostics for all four arms."""
from __future__ import annotations

from pathlib import Path
import time
import traceback

from .plan import read, sha, write


def evaluate(root, plan, job, checkpoint=None):
    import numpy as np
    import torch
    from chapter3_bser.experiments.hgr.runtime import MissionRuntime
    from chapter3_bser.experiments.baselines.common.runtime import BaselineMissionRuntime
    from chapter3_bser.experiments.d2_v1.assembly import diagnostics
    from chapter3_bser.experiments.d2_v1.provenance import framework_sources
    from tools.ch3_baselines.evaluate import episode_row, summary
    from tools.ch3_baselines.bser_prior import ZeroResidualSource
    root = Path(root)
    config = read(root / job["config"])
    scenarios = read(root / "inputs/evaluation.json")["scenarios"]
    output, arm = root / job["evaluation"], job["arm"]
    if output.exists():
        raise FileExistsError("evaluation requires a new directory: " + str(output))
    checkpoint_sha = None if checkpoint is None else sha(checkpoint)
    if arm in ("D2_B2", "D2_B3"):
        from chapter3_bser.experiments.baselines.common.checkpoint import load_checkpoint, load_model, state_digest
        payload = load_checkpoint(checkpoint, expected_baseline=config["baseline"])
        policy = load_model(payload)
        fingerprint = lambda: state_digest(policy.training_state_dict())
        mode = "deterministic"
    elif arm == "D2_HGR":
        from chapter3_bser.experiments.hgr.train import load_checkpoint
        from chapter3_bser.experiments.hgr.provenance import require_source_match, fresh_source_identity
        from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash
        payload = load_checkpoint(checkpoint)
        require_source_match(payload["source_identity"], fresh_source_identity(), context="D2 suite evaluation")
        with torch.random.fork_rng():
            policy = HandoffPolicy(config["policy"])
        policy.load_state_dict(payload["policy"], strict=True)
        policy.eval()
        policy.requires_grad_(False)
        fingerprint = lambda: weights_hash(policy)
        mode = plan["settings"]["hgr_policy_mode"]
    elif arm == "D2" and checkpoint is None:
        policy, payload = ZeroResidualSource(), None
        fingerprint = lambda: "zero_residual_no_trainable_parameters"
        mode = "zero_residual"
    else:
        raise ValueError("invalid arm/checkpoint combination")
    if payload is not None and payload["config"] != config:
        raise ValueError("checkpoint is not from this prepared training job")
    frozen_policy = fingerprint()
    settings = plan["settings"]
    source_sha = plan["source_sha256"]
    def verify():
        if framework_sources()["inventory"]["sha256"] != source_sha:
            raise ValueError("source changed during evaluation")
        if fingerprint() != frozen_policy or (checkpoint is not None and sha(checkpoint) != checkpoint_sha):
            raise ValueError("evaluation changed policy/checkpoint")
        if any(sha(root / name) != expected for name, expected in plan["assets"].items()):
            raise ValueError("prepared inputs changed during evaluation")
    verify()
    output.mkdir(parents=True)
    write(output / "identity.json", dict(plan_sha256=plan["sha256"], arm=arm,
          training_seed=job["training_seed"], policy_mode=mode, policy_sha256=frozen_policy,
          checkpoint_sha256=checkpoint_sha, source_sha256=source_sha,
          scenario_sha256=plan["assets"]["inputs/evaluation.json"],
          evaluation_seed=settings["evaluation_seed"], optimizer_updates=0))
    rows, runtime, steps = [], None, 0
    started = time.perf_counter()
    def save(status, final=False):
        result = summary(rows, len(scenarios), finalized=final,
                         wall_seconds=time.perf_counter()-started, actual_steps=steps)
        result.update(method=arm, arm=arm, plan_sha256=plan["sha256"], policy_mode=mode)
        write(output / "episodes.json", rows)
        write(output / "summary.json", result)
        write(output / "progress.json", dict(status=status, completed=len(rows), expected=len(scenarios),
                                            actual_environment_steps=steps))
        return result
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        save("running")
        for index, scenario in enumerate(scenarios):
            begin = time.perf_counter()
            seed = settings["evaluation_seed"] + index
            runtime_class = BaselineMissionRuntime if arm in ("D2_B2", "D2_B3") else MissionRuntime
            runtime = runtime_class(config, scenario, seed=seed, episode_id=index)
            exposure = stagnant = stagnant_agents = 0
            while not runtime.terminal:
                searching = not runtime.env.get_task_state().target_found
                before = np.asarray(runtime.env.get_agent_state().positions).copy()
                before_step = runtime.step
                try:
                    with torch.no_grad():
                        if arm in ("D2_B2", "D2_B3"):
                            runtime.advance(policy, explore=False)
                        else:
                            runtime.advance(policy, deterministic=mode != "stochastic")
                finally:
                    steps += runtime.step - before_step
                if searching:
                    distance = np.linalg.norm(np.asarray(runtime.env.get_agent_state().positions)[:3]-before[:3], axis=1)
                    stopped = distance <= settings["stagnation_distance_per_step"]
                    exposure += 1
                    stagnant += int(stopped.all())
                    stagnant_agents += int(stopped.sum())
            row = episode_row(runtime, config, scenario, index, seed, time.perf_counter()-begin)
            collision, found = row["first_collision_step"], row["found_step"]
            row.update(method=arm, arm=arm, training_seed=job["training_seed"], policy_mode=mode,
                       plan_sha256=plan["sha256"], pre_found_steps=exposure,
                       pre_found_stagnant_steps=stagnant, pre_found_stagnant_agent_steps=stagnant_agents,
                       pre_found_moving_steps=exposure-stagnant,
                       pre_found_collision=collision is not None and (found is None or collision <= found),
                       found_penalized_steps=found if row["found"] and found is not None else config["max_steps"],
                       d2_diagnostics=diagnostics(runtime))
            runtime.close()
            runtime = None
            verify()
            rows.append(row)
            save("running")
            print(f"{job['name']} [{index+1}/{len(scenarios)}] found={row['found']} "
                  f"found_step={found} outcome={row['failure_stage']}", flush=True)
        return save("complete", True)
    except (Exception, KeyboardInterrupt) as exc:
        write(output / "failure.json", dict(message=str(exc), exception_type=type(exc).__name__,
                                            traceback=traceback.format_exc(), completed=len(rows)))
        save("failed")
        raise
    finally:
        if runtime is not None:
            runtime.close()
        torch.set_num_threads(previous_threads)
