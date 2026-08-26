"""Run the independent, no-update Phase 1C BSER-RMADDPG preflight."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import json
from pathlib import Path
import random
import time
import traceback
from typing import Any

import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_bser_rmaddpg import METHOD
from chapter3_bser.experiments.phase1c_bser_rmaddpg.metrics import (
    EpisodeMetrics,
    summarize_preflight,
)
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import load_phase1b2_config
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.mission_context import OnlineMissionContext
from core.algorithms.maddpg import MADDPG
from core.config.ch3_config import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "chapter3" / "bser_phase1c.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "chapter3" / "phase1c_bser_rmaddpg"


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
    if config.get("schema") != "bser.phase1c.integration.v1":
        raise ValueError("invalid Phase 1C integration config schema")
    if config.get("enabled") is not False:
        raise ValueError("Phase 1C must remain disabled by default")
    if config.get("phase1c_enable_bser_guidance") is not False:
        raise ValueError("BSER guidance must remain disabled by default")
    if config.get("training_enabled") is not False:
        raise ValueError("Phase 1C preflight must not enable training")
    preflight = dict(config.get("preflight", {}))
    if preflight.get("training_update") is not False:
        raise ValueError("Phase 1C preflight requires training_update=false")
    return config


def _episode(job: dict[str, Any]):
    torch.set_num_threads(1)
    episode_index = int(job["episode_index"])
    scenario = copy.deepcopy(job["scenario"])
    scenario_seed = int(scenario["scenario_seed"])
    _seed_all(scenario_seed)
    started = time.perf_counter()

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
    step_rows: list[dict[str, Any]] = []
    try:
        env.reset(scenario=scenario)
        provider = OnlinePlanningStateProvider(
            env,
            refresh_interval=int(load_phase1b2_config()["online"]["state_refresh_interval"]),
        )
        state = provider.initialize()
        mission_context = _public_context(env, state)
        controller = OnlineBSERController(load_phase1b2_config())
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

        maddpg = MADDPG.init_from_env(
            env.unwrapped,
            gamma=float(env_config["gamma"]),
            tau=float(env_config["tau"]),
            lr_actor=float(env_config["lr_actor"]),
            lr_critic=float(env_config["lr_critic"]),
            hidden_dim=int(env_config["hidden_dim"]),
            residual_action_reg=float(env_config["residual_action_reg"]),
        )
        maddpg.prep_rollouts(device="cpu")
        initial_params = tuple(
            parameter.detach().clone()
            for agent in maddpg.agents
            for parameter in agent.policy.parameters()
        )

        accumulator = EpisodeMetrics()
        initial_version = guidance.allocation_version
        final_version = initial_version
        found_step = None
        success_step = None
        final_step = 0

        for step in range(1, int(job["max_steps"]) + 1):
            with torch.no_grad():
                action_parts = maddpg.step(observations, explore=False)
                actions = torch.stack(
                    [value.squeeze(0) for value in action_parts]
                ).to(env.unwrapped.device)
            action_norm = float(torch.norm(actions, dim=1).mean().item())
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
            final_version = guidance.allocation_version

            diagnostics = result.diagnostics
            replan_attempted = bool(
                diagnostics is not None
                and (
                    diagnostics.optimizer_invoked
                    or diagnostics.allocation_scope != "none"
                )
            )
            event_values = tuple(event.value for event in result.events)
            collision_step = int(
                env.unwrapped._collision_flags.detach().sum().item()
            )
            reward_value = float(
                torch.as_tensor(rewards, dtype=torch.float32).sum().item()
            )
            task = env.get_task_state()
            final_step = int(task.step)
            if task.target_found and found_step is None:
                found_step = final_step
            if task.mission_complete and success_step is None:
                success_step = final_step

            accumulator.record_step(
                allocation_version=guidance.allocation_version,
                events=event_values,
                replan_attempted=replan_attempted,
                accepted_replan=bool(result.replanned),
                action_norm=action_norm,
                prior_norm=float(env.unwrapped.last_prior_term_norm),
                residual_norm=float(env.unwrapped.last_residual_term_norm),
                residual_ratio=float(
                    env.unwrapped.last_residual_contribution_ratio
                ),
                collision_count=collision_step,
                reward=reward_value,
            )
            step_rows.append({
                "method": METHOD,
                "episode_index": episode_index,
                "scenario_seed": scenario_seed,
                "step": final_step,
                "allocation_version": guidance.allocation_version,
                "events": list(event_values),
                "event_count": len(event_values),
                "replan_attempted": replan_attempted,
                "accepted_replan": bool(result.replanned),
                "decision_reason": result.decision_reason,
                "action_norm": action_norm,
                "prior_norm": float(env.unwrapped.last_prior_term_norm),
                "residual_norm": float(env.unwrapped.last_residual_term_norm),
                "residual_ratio": float(
                    env.unwrapped.last_residual_contribution_ratio
                ),
                "found": bool(task.target_found),
                "success": bool(task.mission_complete),
                "collision_count": collision_step,
            })
            observations = next_observations
            if all(bool(value) for value in dones):
                break

        parameter_update_count = sum(
            int(not torch.equal(before, after.detach()))
            for before, after in zip(
                initial_params,
                (
                    parameter
                    for agent in maddpg.agents
                    for parameter in agent.policy.parameters()
                ),
            )
        )
        if parameter_update_count != 0:
            raise RuntimeError("preflight changed RMADDPG actor parameters")

        task = env.get_task_state()
        row = accumulator.finalize(
            method=METHOD,
            episode_index=episode_index,
            scenario_seed=scenario_seed,
            steps=final_step,
            found=bool(task.target_found),
            success=bool(task.mission_complete),
            found_step=found_step,
            success_step=success_step,
            initial_allocation_version=initial_version,
            final_allocation_version=final_version,
        )
        row.update({
            "observation_dim": 28,
            "action_dim": 3,
            "critic_input_dim": 124,
            "training_update": False,
            "parameter_update_count": parameter_update_count,
            "wall_seconds": float(time.perf_counter() - started),
        })
        return row, step_rows
    finally:
        env.close()


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )


def _prepare_output(output: Path) -> dict[str, Path]:
    directories = {
        name: output / name for name in ("checkpoint", "logs", "metrics", "configs")
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    (directories["checkpoint"] / "NO_TRAINING.txt").write_text(
        "Phase 1C preflight uses no optimizer updates and creates no checkpoint.\n",
        encoding="utf-8",
    )
    return directories


def run_preflight(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path = DEFAULT_OUTPUT,
    workers: int = 4,
    episodes_override: int | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    assert_registered_ch3_method(METHOD)
    source_config = _load_config(config_path)
    specification = dict(source_config["preflight"])
    episodes = int(
        specification["episodes"]
        if episodes_override is None
        else episodes_override
    )
    max_steps = int(
        specification["max_steps"]
        if max_steps_override is None
        else max_steps_override
    )
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("preflight episodes and max_steps must be positive")
    output = Path(output_dir)
    directories = _prepare_output(output)

    resolved_config = copy.deepcopy(source_config)
    resolved_config["enabled"] = True
    resolved_config["phase1c_enable_bser_guidance"] = True
    resolved_config["training_enabled"] = False
    resolved_config["preflight"].update({
        "episodes": episodes,
        "max_steps": max_steps,
        "training_update": False,
        "method": METHOD,
        "workers": max(1, min(int(workers), episodes)),
    })
    _write_json(directories["configs"] / "resolved_preflight_config.json", resolved_config)
    _write_json(directories["configs"] / "default_disabled_config.json", source_config)

    manifest = build_scenario_manifests(
        count=episodes,
        generator_seed=int(specification["seed"]),
        split="validation",
        profiles=(str(specification["profile"]),),
    )[str(specification["profile"])]
    scenarios = list(manifest["scenarios"])
    if len(scenarios) != episodes:
        raise RuntimeError("scenario generator returned the wrong episode count")

    jobs = [
        {
            "episode_index": index,
            "scenario": scenario,
            "base_candidate": str(specification["base_candidate"]),
            "profile": str(specification["profile"]),
            "max_steps": max_steps,
        }
        for index, scenario in enumerate(scenarios)
    ]
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    worker_count = max(1, min(int(workers), episodes))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_episode, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                row, episode_steps = future.result()
                rows.append(row)
                step_rows.extend(episode_steps)
            except Exception as error:
                failures.append({
                    "episode_index": int(job["episode_index"]),
                    "scenario_seed": int(job["scenario"]["scenario_seed"]),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                })

    rows.sort(key=lambda row: int(row["episode_index"]))
    step_rows.sort(
        key=lambda row: (int(row["episode_index"]), int(row["step"]))
    )
    summary = summarize_preflight(
        rows,
        expected_episodes=episodes,
        failures=failures,
        method=METHOD,
    )
    summary.update({
        "seed": int(specification["seed"]),
        "episodes": episodes,
        "max_steps": max_steps,
        "profile": str(specification["profile"]),
        "workers": worker_count,
        "wall_seconds": float(time.perf_counter() - started),
        "default_phase1c_enabled": False,
        "preflight_guidance_enabled": True,
    })
    _write_csv(directories["metrics"] / "episode_metrics.csv", rows)
    _write_json(directories["metrics"] / "preflight_summary.json", summary)
    _write_jsonl(directories["logs"] / "step_metrics.jsonl", step_rows)
    _write_json(directories["logs"] / "failures.json", failures)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    summary = run_preflight(
        config_path=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        episodes_override=args.episodes,
        max_steps_override=args.max_steps,
    )
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
