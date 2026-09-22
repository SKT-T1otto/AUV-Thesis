"""Plain replay-based MADDPG training shared only by the independent B2/B3 baselines."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from core.replay.ch3_buffer import CH3ReplayBuffer
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from chapter3_bser.experiments.reward_objective import objective_identity
from core.env.task_protocol import protocol_identity
from .checkpoint import (CHECKPOINT_SCHEMA, IMPLEMENTATION_VERSION, ROOT, digest,
                         fresh_source_identity, load_checkpoint, require_source_match,
                         state_digest, validate_config, validated_output, write_json)
from .model import build_model
from .runtime import BaselineMissionRuntime


class BaselineTrainer:
    """One complete mission per episode; ordinary actor/critic replay updates.

    Construction starts from freshly initialized weights. Loading checkpoints is
    evaluation-only: no implicit restoration or automatic long-run continuation.
    """
    baseline = None

    def __init__(self, config, output):
        self.config = validate_config(config, self.baseline)
        self.output = validated_output(self.config, output)
        self.config["output_dir"] = str(self.output)
        self.source_identity = fresh_source_identity()
        self.completed_main = 0
        self.global_step = 0
        self.optimizer_updates = 0
        self.episodes = []
        self.update_metrics = []
        self.scenarios = self._load_scenarios()
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"] % (2 ** 32))
        torch.manual_seed(self.config["seed"])
        self.model = build_model(self.config, env=None, device="cpu")
        rl = self.config["rl"]
        self.replay = CH3ReplayBuffer(rl["replay_size"], 4, [28] * 4, [3] * 4,
                                      success_priority=1.0, alpha=0.0, beta_start=0.0)
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "config.json", self.config)

    def _load_scenarios(self):
        if not self.config.get("scenario_manifest"):
            return None
        path = Path(self.config["scenario_manifest"])
        path = path if path.is_absolute() else ROOT / path
        if self.config.get("scenario_manifest_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("training scenario manifest identity mismatch")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        scenarios = manifest.get("scenarios", [])
        if (not scenarios or manifest.get("scenario_profile", self.config["profile"]) != self.config["profile"]
                or manifest.get("scenario_split", "train") not in ("train", "smoke_train")
                or manifest.get("scenario_role", "train") not in ("train", "smoke_train")):
            raise ValueError("training requires a nonempty matching training manifest")
        identifiers = []
        for scenario in scenarios:
            identifiers.append(scenario.get("scenario_id"))
            if (not scenario.get("scenario_id") or type(scenario.get("scenario_seed")) is not int
                    or scenario.get("scenario_profile") != self.config["profile"]
                    or scenario.get("scenario_split") not in ("train", "smoke_train")
                    or scenario.get("scenario_role") != scenario.get("scenario_split")
                    or scenario.get("max_steps") != self.config["max_steps"]
                    or scenario.get("protocol") != "CH3_UNKNOWN_MAP_V1"):
                raise ValueError("training scenario identity/profile/role/horizon/protocol mismatch")
        if len(set(identifiers)) != len(identifiers) or manifest.get("scenario_count", len(scenarios)) != len(scenarios):
            raise ValueError("training manifest has duplicate scenario IDs or a count mismatch")
        return scenarios

    def _scenario(self, episode):
        # Scenario generation is independent of replay/exploration random draws.
        seed = self.config["seed"] + episode
        if self.scenarios is None:
            manifests = build_scenario_manifests(count=1, generator_seed=seed, split="train",
                                                profiles=[self.config["profile"]])
            scenario = manifests[self.config["profile"]]["scenarios"][0]
            scenario["max_steps"] = self.config["max_steps"]
        else:
            index = int(np.random.default_rng(seed).integers(len(self.scenarios)))
            scenario = self.scenarios[index]
        return copy.deepcopy(scenario), seed

    def _update(self):
        rl = self.config["rl"]
        if (len(self.replay) < rl["batch_size"] or self.global_step < rl["warmup_steps"]
                or self.global_step % rl["update_frequency"]):
            return
        self.model.prep_training(device="cpu")
        for _ in range(rl["updates_per_train"]):
            # Uniform replay and untouched team rewards keep the direct baseline
            # free of success prioritization, reward normalization and correction.
            sample = self.replay.sample(rl["batch_size"], norm_rews=False, device="cpu")
            actor_losses, critic_losses = [], []
            for agent_index in range(4):
                critic_loss, actor_loss, _ = self.model.update(sample, agent_index)
                if not np.isfinite([critic_loss, actor_loss]).all():
                    raise FloatingPointError("nonfinite MADDPG training loss")
                actor_losses.append(float(actor_loss))
                critic_losses.append(float(critic_loss))
            self.model.update_all_targets()
            self.optimizer_updates += 1
            self.update_metrics.append(dict(global_step=self.global_step, optimizer_update=self.optimizer_updates,
                                            actor_loss_by_agent=actor_losses, critic_loss_by_agent=critic_losses))
        self.model.prep_rollouts(device="cpu")

    def run_episode(self):
        episode = self.completed_main + 1
        scenario, seed = self._scenario(episode)
        runtime = BaselineMissionRuntime(self.config, scenario, seed=seed, episode_id=episode)
        started = time.perf_counter()
        updates_before = self.optimizer_updates
        first_step = self.global_step
        try:
            self.model.prep_rollouts(device="cpu")
            self.model.reset_noise()
            while not runtime.terminal and runtime.step < self.config["max_steps"]:
                transition = runtime.advance(self.model, explore=True)
                self.replay.push(transition["observations"], transition["actions"], transition["rewards"],
                                 transition["next_observations"], transition["dones"],
                                 transition.get("success_flags", [False] * 4))
                self.global_step += 1
                self._update()
            summary = runtime.summary()
        finally:
            runtime.close()
        self.completed_main += 1
        row = dict(summary)
        row.update(episode=episode, baseline=self.config["baseline"], algorithm=self.config["algorithm"],
                   method=self.config["method"], scenario_id=scenario.get("scenario_id"),
                   scenario_seed=scenario.get("scenario_seed"), environment_innovation_seed=seed,
                   episode_steps=self.global_step - first_step, global_step=self.global_step,
                   optimizer_updates=self.optimizer_updates - updates_before,
                   replay_size=len(self.replay), return_scope="full_mission", trajectory_complete=True,
                   wall_seconds=time.perf_counter() - started)
        self.episodes.append(row)
        write_json(self.output / "episodes.json", self.episodes)
        write_json(self.output / "training_metrics.json", self.update_metrics)
        print(f"[{self.config['baseline']}] episode={episode}/{self.config['total_main_trajectories']} "
              f"steps={row['episode_steps']} total_steps={self.global_step} updates={self.optimizer_updates}", flush=True)
        return row

    def save(self, *, final=False):
        if self.model is None or self.completed_main < 1:
            raise ValueError("checkpoint requires a complete episode and initialized model")
        require_source_match(self.source_identity, context="checkpoint save")
        directory = self.output / "checkpoints"
        directory.mkdir(exist_ok=True)
        suffix = "final" if final else f"episode_{self.completed_main:06d}"
        path = directory / f"{self.config['baseline']}_{suffix}.pt"
        if path.exists():
            raise FileExistsError(path)
        state = self.model.training_state_dict()
        payload = dict(schema=CHECKPOINT_SCHEMA, implementation_version=IMPLEMENTATION_VERSION,
                       source_identity=self.source_identity, config=self.config, config_hash=digest(self.config),
                       baseline=self.config["baseline"], algorithm=self.config["algorithm"], method=self.config["method"],
                       architecture_version=self.config["architecture_version"], episode_count=self.completed_main,
                       completed_main_trajectories=self.completed_main, episode_complete=True,
                       actual_total_environment_steps=self.global_step, optimizer_updates=self.optimizer_updates,
                       actor_optimizer_steps=self.optimizer_updates * 4, critic_optimizer_steps=self.optimizer_updates * 8,
                       model_training_state=state, model_state_sha256=state_digest(state),
                       episodes=copy.deepcopy(self.episodes), replay_transition_count=len(self.replay),
                       replay_sampling="uniform", replay_reward_normalization=False,
                       training_initialization="from_scratch", resume_supported=False,
                       torch_rng=torch.get_rng_state(), **objective_identity(self.config), **protocol_identity(self.config))
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        load_checkpoint(path, expected_baseline=self.config["baseline"])
        return path

    def run(self):
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            return self._run()
        finally:
            torch.set_num_threads(previous_threads)

    def _run(self):
        started = time.perf_counter()
        while self.completed_main < self.config["total_main_trajectories"]:
            budget = self.config.get("max_total_environment_steps")
            if budget is not None and self.global_step >= budget:
                break
            self.run_episode()
            if self.completed_main % self.config["checkpoint_interval"] == 0:
                self.save()
        latest = self.save(final=True) if self.completed_main else None
        budget = self.config.get("max_total_environment_steps")
        summary = dict(baseline=self.config["baseline"], algorithm=self.config["algorithm"], method=self.config["method"],
                       completed_main_trajectories=self.completed_main, episode_count=self.completed_main,
                       requested_main_trajectories=self.config["total_main_trajectories"],
                       actual_total_environment_steps=self.global_step, optimizer_updates=self.optimizer_updates,
                       actor_optimizer_steps=self.optimizer_updates * 4, critic_optimizer_steps=self.optimizer_updates * 8,
                       replay_transition_count=len(self.replay), replay_sampling="uniform",
                       latest_checkpoint=None if latest is None else str(latest),
                       source_sha256=self.source_identity["sha256"], config_hash=digest(self.config),
                       wall_seconds=time.perf_counter() - started,
                       budget_policy="finish_started_episode_then_stop",
                       budget_overshoot_steps=max(0, self.global_step - (budget or self.global_step)),
                       formal_experiments_completed=False, performance_claims_supported=False,
                       **objective_identity(self.config), **protocol_identity(self.config))
        write_json(self.output / "summary.json", summary)
        return summary
