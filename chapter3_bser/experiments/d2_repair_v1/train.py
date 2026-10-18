"""Independent, fresh-only trainer; long execution always requires --execute."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from chapter3_bser.experiments.baselines.common.checkpoint import (
    digest, fresh_source_identity, require_source_match, state_digest, validated_output, write_json,
    _finite_state,
)
from chapter3_bser.experiments.baselines.common.train import BaselineTrainer
from chapter3_bser.experiments.d2_performance.logging import JsonlJournal
from .config import VERSION, scenario_fingerprint, validate
from .model import build_model
from .replay import ExecutionReplay
from .runtime import RepairedMissionRuntime


class Trainer:
    _load_scenarios = BaselineTrainer._load_scenarios
    _scenario = BaselineTrainer._scenario

    def __init__(self, config, output):
        self.config = validate(config)
        from .provenance import verify_reviewed_changes
        verify_reviewed_changes()
        self.output = validated_output(self.config, output)
        self.config["output_dir"] = str(self.output)
        self.source = fresh_source_identity()
        self.scenarios = self._load_scenarios()
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"] % 2**32)
        torch.manual_seed(self.config["seed"])
        self.model = build_model(self.config)
        self.replay = ExecutionReplay(self.config["rl"]["replay_size"])
        self.global_step = self.completed_main = self.optimizer_updates = 0
        self.training_scenario_ids = set()
        self.training_scenario_seeds = set()
        self.training_scenario_fingerprints = set()
        self.output.mkdir(parents=True, exist_ok=True)
        self.episodes = JsonlJournal(self.output / "episodes.jsonl")
        self.metrics = JsonlJournal(self.output / "training_metrics.jsonl")
        write_json(self.output / "config.json", self.config)

    def _update(self):
        rl = self.config["rl"]
        if (len(self.replay) < rl["batch_size"] or self.global_step < rl["warmup_steps"]
                or self.global_step % rl["update_frequency"]):
            return
        self.model.prep_training(device="cpu")
        for _ in range(rl["updates_per_train"]):
            sample = self.replay.sample(rl["batch_size"])
            updates = []
            for i in range(4):
                critic, actor, _ = self.model.update(sample, i)
                updates.append(dict(agent=i, actor_updated=self.model.last_actor_updated,
                                    critic_loss=critic, actor_loss=actor, **self.model.last_target_diagnostics))
            self.model.update_all_targets()
            self.optimizer_updates += 1
            self.metrics.append(dict(global_step=self.global_step, update=self.optimizer_updates, agents=updates))
        self.model.prep_rollouts(device="cpu")

    def run_episode(self):
        scenario, seed = self._scenario(self.completed_main + 1)
        self.training_scenario_ids.add(scenario["scenario_id"])
        self.training_scenario_seeds.add(int(scenario["scenario_seed"]))
        self.training_scenario_fingerprints.add(scenario_fingerprint(scenario))
        runtime = RepairedMissionRuntime(self.config, scenario, seed=seed, episode_id=self.completed_main + 1)
        try:
            self.model.prep_rollouts(device="cpu")
            self.model.reset_noise()
            while not runtime.terminal:
                self.replay.push_transition(runtime.advance(self.model, explore=True))
                self.global_step += 1
                self._update()
            row = runtime.summary()
        finally:
            runtime.close()
        self.completed_main += 1
        row.update(episode=self.completed_main, global_step=self.global_step,
                   method=self.config["method"], trajectory_complete=True,
                   environment_innovation_seed=seed, max_steps=self.config["max_steps"])
        self.episodes.append(row)
        return row

    def save(self, label):
        require_source_match(self.source, context="repair checkpoint save")
        state = self.model.training_state_dict()
        payload = dict(schema=VERSION, config=copy.deepcopy(self.config), config_hash=digest(self.config),
                       source_identity=self.source, model_state=state, model_state_sha256=state_digest(state),
                       global_step=self.global_step, episodes=self.completed_main, optimizer_updates=self.optimizer_updates,
                       actor_optimizer_steps=list(self.model.actor_updates), critic_optimizer_steps=8*self.optimizer_updates,
                       training_scenario_ids=sorted(self.training_scenario_ids),
                       training_scenario_seeds=sorted(self.training_scenario_seeds),
                       training_scenario_fingerprints=sorted(self.training_scenario_fingerprints), resume_supported=False)
        path = self.output / f"checkpoint_{label}.pt"
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def run(self):
        threads = torch.get_num_threads()
        started = time.perf_counter()
        try:
            torch.set_num_threads(1)
            budget = self.config.get("max_total_environment_steps")
            while self.completed_main < self.config["total_main_trajectories"]:
                if budget is not None and self.global_step >= budget:
                    break
                self.run_episode()
                print(f"[{self.config['method']}] episode={self.completed_main} steps={self.global_step}", flush=True)
                if self.completed_main % self.config["checkpoint_interval"] == 0:
                    self.save(f"episode_{self.completed_main:06d}")
            checkpoint = self.save("final") if self.completed_main else None
            self.episodes.materialize()
            self.metrics.materialize()
            result = dict(episodes=self.completed_main, environment_steps=self.global_step,
                          optimizer_updates=self.optimizer_updates, actor_optimizer_steps=self.model.actor_updates,
                          critic_optimizer_steps=8*self.optimizer_updates, budget_policy="finish_started_episode",
                          latest_checkpoint=None if checkpoint is None else str(checkpoint),
                          wall_seconds=time.perf_counter()-started, performance_passed=False,
                          formal_experiments_completed=False)
            write_json(self.output / "summary.json", result)
            return result
        finally:
            torch.set_num_threads(threads)


def load_for_evaluation(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != VERSION or payload.get("config_hash") != digest(payload["config"]):
        raise ValueError("not a valid independent repair checkpoint")
    config = validate(payload["config"])
    require_source_match(payload["source_identity"], context="repair evaluation")
    state = payload["model_state"]
    if payload.get("model_state_sha256") != state_digest(state) or not _finite_state(state):
        raise ValueError("repair checkpoint state hash mismatch")
    with torch.random.fork_rng():
        model = build_model(config)
        model.load_training_state_dict(state)
    if model.niter != payload["optimizer_updates"] or model.actor_updates != payload["actor_optimizer_steps"]:
        raise ValueError("repair checkpoint optimizer count mismatch")
    if payload["critic_optimizer_steps"] != 8*model.niter:
        raise ValueError("repair checkpoint critic step count mismatch")
    for i,agent in enumerate(model.agents):
        for optimizer,count in ((agent.policy_optimizer, model.actor_updates[i]),
                                (agent.critic1_optimizer, model.niter), (agent.critic2_optimizer, model.niter)):
            parameters = [p for group in optimizer.param_groups for p in group["params"]]
            if set(optimizer.state) != (set(parameters) if count else set()):
                raise ValueError("repair optimizer state inventory mismatch")
            if any(int(item["step"]) != count for item in optimizer.state.values()):
                raise ValueError("repair optimizer per-parameter step mismatch")
        for net in (agent.policy, agent.target_policy, agent.critic1, agent.critic2, agent.target_critic1, agent.target_critic2):
            if any(not bool(torch.isfinite(p).all()) for p in net.parameters()):
                raise ValueError("nonfinite checkpoint parameters")
    model.prep_rollouts(device="cpu")
    return model, config, payload


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    c = validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if not args.execute:
        print(json.dumps(dict(executed=False, method=c["method"], gamma=c["rl"]["gamma"],
              repairs=c["repairs"], episodes=c["total_main_trajectories"]), indent=2))
        return
    print(json.dumps(Trainer(c, args.output).run(), indent=2))


if __name__ == "__main__":
    main()
