"""Independent synchronous-rollout trainer for Phase 1C BSER-RMADDPG."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_bser_rmaddpg import METHOD
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.algorithms.maddpg import MADDPG
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.replay.ch3_buffer import CH3ReplayBuffer
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c_train.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "chapter3" / "phase1c_bser_rmaddpg" / "training"
)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _public_context(env: GuidedEnv, state) -> OnlineMissionContext:
    return OnlineMissionContext.from_public_views(
        env.get_task_state(), env.get_search_execution_state(), state
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema") != "bser.phase1c.training.v1":
        raise ValueError("invalid Phase 1C training configuration")
    if config.get("method") != METHOD:
        raise ValueError("Phase 1C training method mismatch")
    if config.get("phase1c_enable_bser_guidance") is not True:
        raise ValueError("training entry must explicitly enable BSER guidance")
    if config.get("guidance_enabled") is not True:
        raise ValueError("training entry must set guidance_enabled=true")
    if config.get("training_enabled") is not True:
        raise ValueError("training entry must explicitly enable training")
    if config.get("training_update") is not True:
        raise ValueError("Phase 1C short training requires updates")
    if int(config.get("seed", -1)) != 2729:
        raise ValueError("short training is locked to seed 2729")
    if config.get("profile") != "M20_MOVING_UNKNOWN_MULTI":
        raise ValueError("short training profile changed")
    if [int(config[key]) for key in ("observation_dim", "action_dim", "critic_dim")] != [28, 3, 124]:
        raise ValueError("Phase 1C network dimensions changed")
    return config


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _policy_snapshot(maddpg: MADDPG):
    return tuple(
        {
            key: value.detach().cpu().clone()
            for key, value in agent.policy.state_dict().items()
        }
        for agent in maddpg.agents
    )


def _collect_episode(job: dict[str, Any]):
    """Collect one episode with a fixed actor snapshot; never update weights."""

    torch.set_num_threads(1)
    episode_index = int(job["episode_index"])
    scenario = copy.deepcopy(job["scenario"])
    scenario_seed = int(scenario["scenario_seed"])
    _seed_all(scenario_seed)
    env_config = build_ch3_config(job["base_candidate"], job["profile"])
    base_env = MissionCoreEnv(
        **environment_kwargs_from_config(
            env_config,
            device="cpu",
            max_steps=int(job["max_steps"]),
            return_numpy=False,
        )
    )
    env = GuidedEnv(base_env, enabled=True)
    started = time.perf_counter()
    try:
        env.reset(scenario=scenario)
        phase1b_config = load_phase1b2_config()
        provider = OnlinePlanningStateProvider(
            env,
            refresh_interval=int(phase1b_config["online"]["state_refresh_interval"]),
        )
        state = provider.initialize()
        mission_context = _public_context(env, state)
        controller = OnlineBSERController(phase1b_config)
        initialized = controller.initialize(state, mission_context)
        bridge = RMADDPGGuidanceBridge()
        guidance = bridge.compile_guidance(
            initialized.allocation,
            state,
            mission_context,
            decision_reason="INITIALIZE",
        )
        env.install_guidance(guidance)
        observations = env.refresh_observation_after_guidance()

        rl = job["rl"]
        actor = MADDPG.init_from_env(
            env.unwrapped,
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
            lr_actor=float(rl["lr_actor"]),
            lr_critic=float(rl["lr_critic"]),
            hidden_dim=int(rl["hidden_dim"]),
            residual_action_reg=float(rl["residual_action_reg"]),
        )
        for agent, state_dict in zip(actor.agents, job["policy_states"]):
            agent.policy.load_state_dict(state_dict)
        actor.prep_rollouts(device="cpu")
        actor.reset_noise()

        transitions = []
        allocation_versions = {guidance.allocation_version}
        event_count = 0
        optimizer_invocations = 0
        accepted_replans = 0
        rejected_replans = 0
        collision_count = 0
        action_norms = []
        prior_norms = []
        residual_norms = []
        residual_ratios = []
        reward_total = 0.0
        found_step = None
        success_step = None
        final_step = 0

        for _ in range(int(job["max_steps"])):
            with torch.no_grad():
                parts = actor.step(observations, explore=True)
                actions = torch.stack(
                    [value.squeeze(0) for value in parts]
                ).to(env.unwrapped.device)
            action_norms.append(float(torch.norm(actions, dim=1).mean().item()))
            _, rewards, dones = env.step(actions)

            state = provider.snapshot(force=False)
            mission_context = _public_context(env, state)
            result = controller.step(state, mission_context)
            guidance = bridge.compile_guidance(
                result.allocation,
                state,
                mission_context,
                decision_reason=result.decision_reason,
            )
            env.install_guidance(guidance)
            next_observations = env.refresh_observation_after_guidance()
            allocation_versions.add(guidance.allocation_version)

            diagnostics = result.diagnostics
            attempted = bool(
                diagnostics is not None
                and (
                    diagnostics.optimizer_invoked
                    or diagnostics.allocation_scope != "none"
                )
            )
            event_count += len(result.events)
            optimizer_invocations += int(
                diagnostics is not None and diagnostics.optimizer_invoked
            )
            accepted_replans += int(bool(result.replanned))
            rejected_replans += int(attempted and not result.replanned)
            collision_count += int(env.unwrapped._collision_flags.sum().item())
            prior_norms.append(float(env.unwrapped.last_prior_term_norm))
            residual_norms.append(float(env.unwrapped.last_residual_term_norm))
            residual_ratios.append(
                float(env.unwrapped.last_residual_contribution_ratio)
            )
            reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
            reward_total += float(reward_tensor.sum().item())
            task = env.get_task_state()
            final_step = int(task.step)
            if task.target_found and found_step is None:
                found_step = final_step
            if task.mission_complete and success_step is None:
                success_step = final_step

            transitions.append((
                tuple(value.detach().cpu().clone() for value in observations),
                actions.detach().cpu().clone(),
                reward_tensor.detach().cpu().clone(),
                tuple(value.detach().cpu().clone() for value in next_observations),
                tuple(bool(value) for value in dones),
                tuple(bool(task.mission_complete) for _ in range(4)),
            ))
            observations = next_observations
            if all(bool(value) for value in dones):
                break

        def mean(values):
            return float(statistics.fmean(values)) if values else 0.0

        task = env.get_task_state()
        metrics = {
            "method": METHOD,
            "episode": episode_index + 1,
            "episode_index": episode_index,
            "scenario_seed": scenario_seed,
            "success": bool(task.mission_complete),
            "found": bool(task.target_found),
            "collision": int(collision_count),
            "episode_length": int(final_step),
            "found_step": "" if found_step is None else int(found_step),
            "success_step": "" if success_step is None else int(success_step),
            "event_count": int(event_count),
            "optimizer_invocation_count": int(optimizer_invocations),
            "accepted_replans": int(accepted_replans),
            "rejected_replans": int(rejected_replans),
            "allocation_version_count": len(allocation_versions),
            "reward": float(reward_total),
            "action_norm": mean(action_norms),
            "prior_norm": mean(prior_norms),
            "residual_norm": mean(residual_norms),
            "residual_ratio": mean(residual_ratios),
            "initial_planner_endpoint_fallback_count": int(
                env.initial_endpoint_fallback_count
            ),
            "wall_seconds": float(time.perf_counter() - started),
        }
        return metrics, transitions
    finally:
        env.close()


def _build_learner(config: dict[str, Any]):
    env_config = build_ch3_config(config["base_candidate"], config["profile"])
    env = MissionCoreEnv(
        **environment_kwargs_from_config(
            env_config,
            device="cpu",
            max_steps=int(config["max_steps"]),
            return_numpy=False,
        )
    )
    rl = config["rl"]
    try:
        maddpg = MADDPG.init_from_env(
            env.unwrapped,
            gamma=float(rl["gamma"]),
            tau=float(rl["tau"]),
            lr_actor=float(rl["lr_actor"]),
            lr_critic=float(rl["lr_critic"]),
            hidden_dim=int(rl["hidden_dim"]),
            residual_action_reg=float(rl["residual_action_reg"]),
        )
    finally:
        env.close()
    replay = CH3ReplayBuffer(
        max_steps=int(rl["replay_size"]),
        num_agents=4,
        obs_dims=(28, 28, 28, 28),
        ac_dims=(3, 3, 3, 3),
        storage_device="cpu",
    )
    return maddpg, replay


def _apply_transitions(
    maddpg: MADDPG,
    replay: CH3ReplayBuffer,
    transitions,
    rl: dict[str, Any],
    *,
    global_step: int,
    update_step: int,
    device: str,
):
    actor_losses = []
    critic_losses = []
    q_values = []
    replay_sample_count = 0
    optimizer_update_count = 0
    for obs, actions, rewards, next_obs, dones, success_flags in transitions:
        replay.push(obs, actions, rewards, next_obs, dones, success_flags)
        global_step += 1
        if (
            global_step < int(rl["warmup_steps"])
            or global_step % int(rl["update_frequency"]) != 0
            or len(replay) < int(rl["batch_size"])
        ):
            continue
        maddpg.prep_training(device=device)
        for _ in range(int(rl["updates_per_train"])):
            sample = replay.sample(
                int(rl["batch_size"]), norm_rews=False, device=device
            )
            replay_sample_count += 1
            with torch.no_grad():
                joint = torch.cat((*sample[0], *sample[1]), dim=1)
                q_values.append(float(statistics.fmean(
                    float(agent.critic1(joint).mean().item())
                    for agent in maddpg.agents
                )))
            errors = []
            update_actor = update_step % int(rl["policy_delay"]) == 0
            for agent_i in range(4):
                if update_actor:
                    critic_loss, actor_loss, error = maddpg.update(sample, agent_i)
                    actor_losses.append(float(actor_loss))
                else:
                    critic_loss, error = maddpg.update_critic_only(sample, agent_i)
                critic_losses.append(float(critic_loss))
                errors.append(error.detach())
                optimizer_update_count += 1
            replay.update_priorities(
                sample[6], torch.stack(errors).mean(dim=0), sample[7]
            )
            maddpg.update_all_targets(compute_diff=False)
            update_step += 1
        maddpg.prep_rollouts(device=device)

    def mean(values):
        return float(statistics.fmean(values)) if values else 0.0

    return {
        "global_step": global_step,
        "update_step": update_step,
        "actor_loss": mean(actor_losses),
        "critic_loss": mean(critic_losses),
        "q_value": mean(q_values),
        "replay_sample_count": replay_sample_count,
        "optimizer_update_count": optimizer_update_count,
    }


def _checkpoint_metadata(config: dict[str, Any], episode: int):
    return {
        "schema": "bser.phase1c.checkpoint.v1",
        "method": METHOD,
        "episode": int(episode),
        "seed": int(config["seed"]),
        "profile": str(config["profile"]),
        "observation_dim": 28,
        "action_dim": 3,
        "critic_dim": 124,
        "bser_integration_version": str(config["bser_integration_version"]),
        "config_hash": _config_hash(config),
        "training_update": True,
    }


def _save_checkpoint(
    maddpg: MADDPG,
    replay: CH3ReplayBuffer,
    directory: Path,
    config: dict[str, Any],
    episode: int,
    *,
    global_step: int,
    update_step: int,
    replay_sample_count: int,
    optimizer_update_count: int,
    episode_rows: list[dict[str, Any]],
) -> Path:
    path = directory / f"phase1c_episode_{int(episode):04d}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "schema": "bser.phase1c.training_state.v1",
        "metadata": _checkpoint_metadata(config, episode),
        "maddpg_training_state": maddpg.training_state_dict(),
        "replay_state": replay.state_dict(),
        "completed_episode": int(episode),
        "global_step": int(global_step),
        "update_step": int(update_step),
        "replay_sample_count": int(replay_sample_count),
        "optimizer_update_count": int(optimizer_update_count),
        "episode_metrics": [dict(row) for row in episode_rows],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    metadata = loaded.get("metadata", {})
    required = {
        "method": METHOD,
        "observation_dim": 28,
        "action_dim": 3,
        "critic_dim": 124,
        "bser_integration_version": str(config["bser_integration_version"]),
        "config_hash": _config_hash(config),
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise RuntimeError("saved checkpoint metadata validation failed")
    return path


def _load_checkpoint(
    path: Path,
    maddpg: MADDPG,
    replay: CH3ReplayBuffer,
    config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema") != "bser.phase1c.training_state.v1":
        raise ValueError("unsupported Phase 1C training checkpoint schema")
    metadata = payload.get("metadata", {})
    if metadata.get("method") != METHOD:
        raise ValueError("resume checkpoint method mismatch")
    if metadata.get("config_hash") != _config_hash(config):
        raise ValueError("resume checkpoint config hash mismatch")
    for key, expected in (
        ("observation_dim", 28),
        ("action_dim", 3),
        ("critic_dim", 124),
    ):
        if int(metadata.get(key, -1)) != expected:
            raise ValueError(f"resume checkpoint {key} mismatch")
    if metadata.get("bser_integration_version") != config["bser_integration_version"]:
        raise ValueError("resume checkpoint BSER integration version mismatch")
    maddpg.load_training_state_dict(payload["maddpg_training_state"])
    replay.load_state_dict(payload["replay_state"])
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rolling(values, window):
    result = []
    for index in range(len(values)):
        start = max(0, index + 1 - int(window))
        result.append(float(statistics.fmean(values[start : index + 1])))
    return result


def _plot_curves(rows: list[dict[str, Any]], metrics_dir: Path, window: int) -> None:
    episodes = [int(row["episode_index"]) + 1 for row in rows]

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, [float(row["actor_loss"]) for row in rows], label="actor")
    axis.plot(episodes, [float(row["critic_loss"]) for row in rows], label="critic")
    axis.set(xlabel="Episode", ylabel="Loss", title="Phase 1C training losses")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(metrics_dir / "loss_curves.png", dpi=180); plt.close(figure)

    rewards = [float(row["reward"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, rewards, alpha=0.35, label="episode")
    axis.plot(episodes, _rolling(rewards, window), label=f"rolling-{window}")
    axis.set(xlabel="Episode", ylabel="Reward", title="Phase 1C reward")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(metrics_dir / "reward_curve.png", dpi=180); plt.close(figure)

    success = _rolling([float(bool(row["success"])) for row in rows], window)
    found = _rolling([float(bool(row["found"])) for row in rows], window)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, success, label="success")
    axis.plot(episodes, found, label="found")
    axis.set(xlabel="Episode", ylabel="Rolling rate", title="Success / found trend", ylim=(0, 1))
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(metrics_dir / "success_found_trend.png", dpi=180); plt.close(figure)

    ratios = [float(row["residual_ratio"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(episodes, ratios, alpha=0.35, label="episode")
    axis.plot(episodes, _rolling(ratios, window), label=f"rolling-{window}")
    axis.set(xlabel="Episode", ylabel="Residual ratio", title="Residual contribution")
    axis.grid(alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(metrics_dir / "residual_ratio_curve.png", dpi=180); plt.close(figure)


def run_training(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    dry_run: bool = False,
    seed_override: int | None = None,
    resume: Path | None = None,
    episodes_override: int | None = None,
    max_steps_override: int | None = None,
    workers_override: int | None = None,
    batch_size_override: int | None = None,
    warmup_steps_override: int | None = None,
) -> dict[str, Any]:
    assert_registered_ch3_method(METHOD)
    config = _load_config(config_path)
    config = copy.deepcopy(config)
    if seed_override is not None:
        config["seed"] = int(seed_override)
    if dry_run:
        config["episodes"] = int(config["dry_run"]["episodes"])
        config["max_steps"] = int(config["dry_run"]["max_steps"])
    if episodes_override is not None:
        config["episodes"] = int(episodes_override)
    if max_steps_override is not None:
        config["max_steps"] = int(max_steps_override)
    if workers_override is not None:
        config["workers"] = int(workers_override)
    if batch_size_override is not None:
        config["rl"]["batch_size"] = int(batch_size_override)
    if warmup_steps_override is not None:
        config["rl"]["warmup_steps"] = int(warmup_steps_override)
    if int(config["episodes"]) <= 0 or int(config["max_steps"]) <= 0:
        raise ValueError("episodes and max_steps must be positive")

    configured_output = Path(config["output_dir"])
    if not configured_output.is_absolute():
        configured_output = ROOT / configured_output
    base_output = configured_output if output_dir is None else Path(output_dir)
    output = base_output / "dry_run" if dry_run else base_output
    directories = {
        name: output / name for name in ("checkpoints", "logs", "metrics")
    }
    protected_outputs = (
        output / "resolved_training_config.json",
        directories["metrics"] / "episode_metrics.csv",
        directories["metrics"] / "training_summary.json",
    )
    existing_checkpoints = tuple(directories["checkpoints"].glob("*.pt"))
    if resume is None and (
        any(path.exists() for path in protected_outputs) or existing_checkpoints
    ):
        raise FileExistsError(
            f"training output already exists; use --resume or a new output directory: {output}"
        )
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_training_config.json"
    if not resolved_path.exists():
        _write_json(resolved_path, config)

    _seed_all(int(config["seed"]))
    maddpg, replay = _build_learner(config)
    maddpg.prep_rollouts(device=config["device"])
    rows = []
    checkpoints = []
    global_step = 0
    update_step = 0
    replay_sample_count = 0
    optimizer_update_count = 0
    start_episode = 0
    if resume is not None:
        payload = _load_checkpoint(resume, maddpg, replay, config)
        start_episode = int(payload["completed_episode"])
        global_step = int(payload["global_step"])
        update_step = int(payload["update_step"])
        replay_sample_count = int(payload["replay_sample_count"])
        optimizer_update_count = int(payload["optimizer_update_count"])
        rows = [dict(row) for row in payload.get("episode_metrics", [])]
        checkpoints = [str(Path(resume).resolve())]
        if len(rows) != start_episode:
            raise ValueError("checkpoint episode metrics do not match completed episode")
    if start_episode >= int(config["episodes"]):
        raise ValueError("resume checkpoint already reached configured episode count")
    initial_actor = tuple(
        parameter.detach().cpu().clone()
        for agent in maddpg.agents
        for parameter in agent.policy.parameters()
    )

    manifest = build_scenario_manifests(
        count=int(config["episodes"]),
        generator_seed=int(config["seed"]),
        split="train",
        profiles=(str(config["profile"]),),
    )[str(config["profile"])]
    scenarios = list(manifest["scenarios"])
    started = time.perf_counter()
    remaining_episodes = int(config["episodes"]) - start_episode
    workers = max(1, min(int(config["workers"]), remaining_episodes))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_start in range(start_episode, int(config["episodes"]), workers):
            batch_indices = range(
                batch_start,
                min(batch_start + workers, int(config["episodes"])),
            )
            policy_states = _policy_snapshot(maddpg)
            jobs = [
                {
                    "episode_index": index,
                    "scenario": scenarios[index],
                    "base_candidate": config["base_candidate"],
                    "profile": config["profile"],
                    "max_steps": config["max_steps"],
                    "rl": config["rl"],
                    "policy_states": policy_states,
                }
                for index in batch_indices
            ]
            try:
                results = list(executor.map(_collect_episode, jobs))
            except Exception as exc:
                failure_path = directories["logs"] / (
                    f"worker_failure_episode_{batch_start + 1:04d}.json"
                )
                _write_json(failure_path, {
                    "schema": "bser.phase1c.worker_failure.v1",
                    "method": METHOD,
                    "batch_first_episode": batch_start + 1,
                    "batch_last_episode": max(batch_indices) + 1,
                    "scenarios": [
                        {
                            "episode": int(job["episode_index"]) + 1,
                            "scenario_id": str(job["scenario"].get("scenario_id")),
                            "scenario_seed": int(job["scenario"]["scenario_seed"]),
                        }
                        for job in jobs
                    ],
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                })
                raise
            for episode_metrics, transitions in sorted(
                results, key=lambda value: int(value[0]["episode_index"])
            ):
                update = _apply_transitions(
                    maddpg,
                    replay,
                    transitions,
                    config["rl"],
                    global_step=global_step,
                    update_step=update_step,
                    device=str(config["device"]),
                )
                global_step = int(update["global_step"])
                update_step = int(update["update_step"])
                replay_sample_count += int(update["replay_sample_count"])
                optimizer_update_count += int(update["optimizer_update_count"])
                episode_metrics.update({
                    "actor_loss": float(update["actor_loss"]),
                    "critic_loss": float(update["critic_loss"]),
                    "q_value": float(update["q_value"]),
                    "replay_size": len(replay),
                    "replay_sample_count": int(update["replay_sample_count"]),
                    "optimizer_update_count": int(update["optimizer_update_count"]),
                })
                rows.append(episode_metrics)
                episode_number = int(episode_metrics["episode_index"]) + 1
                if (
                    episode_number % int(config["checkpoint_interval"]) == 0
                    or episode_number == int(config["episodes"])
                ):
                    path = _save_checkpoint(
                        maddpg,
                        replay,
                        directories["checkpoints"],
                        config,
                        episode_number,
                        global_step=global_step,
                        update_step=update_step,
                        replay_sample_count=replay_sample_count,
                        optimizer_update_count=optimizer_update_count,
                        episode_rows=rows,
                    )
                    checkpoints.append(str(path.resolve()))
            _write_csv(directories["metrics"] / "episode_metrics.csv", rows)

    changed_actor_tensors = sum(
        int(not torch.equal(before, after.detach().cpu()))
        for before, after in zip(
            initial_actor,
            (
                parameter
                for agent in maddpg.agents
                for parameter in agent.policy.parameters()
            ),
        )
    )
    _plot_curves(rows, directories["metrics"], int(config["rolling_window"]))
    _write_json(directories["checkpoints"] / "checkpoint_list.json", checkpoints)
    summary = {
        "schema": "bser.phase1c.short_training.summary.v1",
        "method": METHOD,
        "dry_run": bool(dry_run),
        "episodes": int(config["episodes"]),
        "max_steps": int(config["max_steps"]),
        "workers": workers,
        "training_update": True,
        "resumed_from": None if resume is None else str(Path(resume).resolve()),
        "start_episode": start_episode,
        "completed_episode_count": len(rows),
        "global_step": global_step,
        "replay_size": len(replay),
        "replay_sample_count": replay_sample_count,
        "optimizer_update_count": optimizer_update_count,
        "parameter_update_count": changed_actor_tensors,
        "checkpoint_count": len(checkpoints),
        "checkpoint_load_verified": bool(checkpoints),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "found_rate": sum(bool(row["found"]) for row in rows) / len(rows),
        "collision_count": sum(int(row["collision"]) for row in rows),
        "mean_reward": float(statistics.fmean(float(row["reward"]) for row in rows)),
        "mean_actor_loss": float(statistics.fmean(float(row["actor_loss"]) for row in rows)),
        "mean_critic_loss": float(statistics.fmean(float(row["critic_loss"]) for row in rows)),
        "mean_q_value": float(statistics.fmean(float(row["q_value"]) for row in rows)),
        "mean_residual_ratio": float(statistics.fmean(float(row["residual_ratio"]) for row in rows)),
        "wall_seconds": float(time.perf_counter() - started),
        "passed": bool(
            len(rows) == int(config["episodes"])
            and replay_sample_count > 0
            and optimizer_update_count > 0
            and changed_actor_tensors > 0
            and checkpoints
        ),
    }
    _write_json(directories["metrics"] / "training_summary.json", summary)
    _write_json(directories["logs"] / "training_log.json", {
        "summary": summary,
        "checkpoint_metadata": _checkpoint_metadata(config, len(rows)),
    })
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int)
    args = parser.parse_args(argv)
    summary = run_training(
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        seed_override=args.seed,
        resume=args.resume,
        episodes_override=args.episodes,
        max_steps_override=args.max_steps,
        workers_override=args.workers,
        batch_size_override=args.batch_size,
        warmup_steps_override=args.warmup_steps,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
