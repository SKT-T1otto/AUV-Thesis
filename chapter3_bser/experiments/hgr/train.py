"""Complete-cycle production training; no replay, PPO, critic or reward shaping."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from chapter3_bser.experiments.reward_objective import TEAM, objective_identity
from chapter3_bser.models.hgr import ARCHITECTURE_VERSION, CHECKPOINT_SCHEMA, METHODS
from chapter3_bser.models.hgr.policy import HandoffPolicy, weights_hash
from chapter3_bser.models.hgr.estimator import BoundaryPredictor, update_prefix, update_suffix
from chapter3_bser.experiments.hgr.runtime import FEATURE_DIM, collect_trajectory, continue_branch, seed_innovations
from core.env.task_protocol import STRICT, protocol_identity, validate_task_config, require_protocol_output
from core.registry.experiment_registry import assert_registered_ch3_method
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from .provenance import source_identity, IMPLEMENTATION_VERSION, validate_source_identity

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/chapter3/hgr_train.json"
COST_FIELDS = ("main_prefix_steps", "old_reference_suffix_steps", "main_current_suffix_steps",
               "suffix_policy_training_steps", "pilot_prefix_steps", "pilot_old_suffix_steps",
               "pilot_new_suffix_steps", "correction_old_suffix_steps", "correction_new_suffix_steps")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def immutable_config(config):
    return {k: v for k, v in config.items() if k not in ("output_dir", "total_main_trajectories", "max_total_environment_steps")}


def validated_output(config, output):
    """Read-only preflight, shared by direct APIs and every HGR-family entry."""
    path = Path(output).resolve()
    try:
        require_protocol_output(config, path)
    except ValueError as exc:
        raise ValueError(f'{exc}: {path}. Example: outputs/chapter3/hgr/collision_terminal/new_run') from exc
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'use a new or empty output directory: {path}')
    return path


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_config(config):
    config = copy.deepcopy(dict(config))
    if config.get("schema") != "hgr.training.v1":
        raise ValueError("unknown HGR training config schema")
    validate_task_config(config)
    if protocol_identity(config)["task_protocol"] != STRICT or objective_identity(config)["reward_objective"] != TEAM:
        raise ValueError("HGR-family methods require collision_terminal_v1 and team_mean_v1")
    if config.get("algorithm") not in METHODS:
        raise ValueError("unknown stochastic method")
    if config.get("method") != "ch3_" + config["algorithm"]:
        raise ValueError("registered method and algorithm mismatch")
    assert_registered_ch3_method(config["method"])
    if config.get("architecture_version") != ARCHITECTURE_VERSION or config.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("HGR architecture or checkpoint schema mismatch")
    if config.get("observation_dim") != 28 or config.get("action_dim") != 3:
        raise ValueError("HGR local observation/action contract mismatch")
    if not config["reward"].get("freeze_searchers_after_found"):
        raise ValueError("HGR v1 requires the existing Searcher freeze behavior")
    if not 0 < config["rl"]["gamma"] <= 1:
        raise ValueError("invalid shared gamma")
    for name in ("main_prefix_batch_size", "suffix_training_episodes_per_cycle", "pilot_prefix_episodes_per_cycle", "correction_draws_per_cycle", "checkpoint_interval", "total_main_trajectories", "max_steps"):
        if isinstance(config.get(name), bool) or not isinstance(config.get(name), int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.get("ablation", "none") not in ("none", "predictor_zero", "remove_correction", "remove_prediction"):
        raise ValueError("unknown diagnostic ablation")
    if config.get("search_value", {}).get("enabled") or config.get("search_value_decision", {}).get("enabled") or config.get("search_value_guidance", {}).get("enabled"):
        raise ValueError("HGR v1 does not use the historical search-value route")
    config.update(objective_identity(config))
    return config


class Trainer:
    def __init__(self, config, output, *, resume=None, mean_initialization=None):
        if resume and mean_initialization:
            raise ValueError("resume and mean initialization are mutually exclusive")
        config = validate_config(config)
        self.config = copy.deepcopy(config)
        self.run_mode = "same_method_resume" if resume else "cross_architecture_mean_initialization" if mean_initialization else "from_scratch"
        self.resume_from = None if resume is None else str(Path(resume).resolve())
        self.source_identity = source_identity()
        self.output = validated_output(config, output)
        self.output.mkdir(parents=True, exist_ok=True)
        seed_innovations(config["seed"])
        torch.set_num_threads(1)
        self.policy = HandoffPolicy(config["policy"])
        self.predictor = BoundaryPredictor(FEATURE_DIM, config["predictor"]["hidden_dim"])
        self.prefix_optimizer = torch.optim.SGD(self.policy.theta_minus.parameters(), lr=config["prefix_lr"])
        self.suffix_optimizer = torch.optim.SGD(self.policy.phi.parameters(), lr=config["suffix_lr"])
        self.predictor_optimizer_state = None
        self.completed_main = self.cycle = self.stream_counter = 0
        self.index_rng = np.random.default_rng(config["seed"])
        self.costs = {name: 0 for name in (*COST_FIELDS, "snapshot_restore_count", "predictor_updates", "prefix_actor_updates", "suffix_actor_updates")}
        self.cycles, self.episodes, self.branches = [], [], []
        self.initialization = dict(mode="from_scratch")
        self.scenarios = None
        if config.get("scenario_manifest"):
            manifest_path = Path(config["scenario_manifest"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.scenarios = manifest["scenarios"]
            if not self.scenarios or any(s.get("scenario_split", "train") != "train" for s in self.scenarios):
                raise ValueError("training requires a nonempty training manifest")
            if config.get("scenario_manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
                raise ValueError("training scenario manifest identity mismatch")
        if resume:
            self._resume(resume)
        elif mean_initialization:
            from chapter3_bser.experiments.phase1c_prrac.checkpoint_transfer import source_training_config, file_sha256
            from chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints import load_prrac_checkpoint
            payload = torch.load(mean_initialization, map_location="cpu", weights_only=True)
            source_training_config(Path(mean_initialization), payload)
            source, _ = load_prrac_checkpoint(Path(mean_initialization), config={**config, "allow_objective_transfer": True, "allow_protocol_transfer": True})
            mapping = self.policy.import_prrac_means([a.actor for a in source.agents])
            self.initialization = dict(mode="cross_architecture_mean_initialization", source=str(Path(mean_initialization).resolve()),
                                       source_sha256=file_sha256(mean_initialization), mapping=mapping,
                                       source_objective=objective_identity(payload["metadata"]), source_protocol=protocol_identity(payload["metadata"]))
        write_json(self.output / "config.json", config)

    def stream(self, purpose):
        self.stream_counter += 1
        identity = f"seed:{self.config['seed']}/stream:{self.stream_counter}/{purpose}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") & ((1 << 63) - 1)
        return identity, seed

    def scenario(self, purpose):
        identity, seed = self.stream(purpose + "/scenario")
        if self.scenarios is not None:
            # Every dataset uses the same uniform initial distribution. A
            # purpose-dependent round-robin would silently change that objective.
            index = int(np.random.default_rng(seed).integers(len(self.scenarios)))
            scenario = copy.deepcopy(self.scenarios[index])
        else:
            manifests = build_scenario_manifests(count=1, generator_seed=seed, split="train", profiles=[self.config["profile"]])
            scenario = manifests[self.config["profile"]]["scenarios"][0]
        return identity, seed, scenario

    def collect(self, purpose, policy, *, stop=False):
        dataset_id, _, scenario = self.scenario(purpose)
        stream_id, seed = self.stream(purpose + "/trajectory")
        trajectory = collect_trajectory(self.config, scenario, policy, seed=seed,
                                         episode_id=self.stream_counter, stop_at_boundary=stop)
        trajectory["dataset_id"] = dataset_id
        trajectory["stream_id"] = stream_id
        row = dict(trajectory["summary"])
        row.update(dataset_id=dataset_id, stream_id=stream_id,
                   purpose=purpose, cycle=self.cycle + 1, method=self.config["method"],
                   reward_objective=TEAM, gamma=self.config["rl"]["gamma"],
                   trajectory_complete=not stop or trajectory["tau"] is None)
        row["return_scope"] = "full_mission" if row["trajectory_complete"] else "prefix_only"
        self.episodes.append(row)
        return trajectory

    def branch(self, trajectory, policy, purpose):
        stream_id, seed = self.stream(purpose)
        result = continue_branch(trajectory["snapshot"], policy, seed=seed, stream_id=stream_id)
        self.costs["snapshot_restore_count"] += 1
        self.costs[purpose + "_steps"] += result["steps"]
        return result

    def paired_label(self, trajectory, old, new, purpose):
        new_result = self.branch(trajectory, new, purpose + "_new_suffix")
        old_result = self.branch(trajectory, old, purpose + "_old_suffix") if self.config["algorithm"] == "hgr" else None
        label = new_result["G_plus"] - old_result["G_plus"] if old_result is not None else new_result["G_plus"]
        row = dict(cycle=self.cycle + 1, dataset_id=trajectory["dataset_id"], purpose=purpose,
                   snapshot_hash=trajectory["snapshot"].sha256, tau=trajectory["tau"],
                   remaining_task_steps=self.config["max_steps"] - trajectory["tau"],
                   legal_features=trajectory["features"].tolist(), old=old_result, new=new_result,
                   G_plus_old=None if old_result is None else old_result["G_plus"],
                   G_plus_new=new_result["G_plus"], delta_hat=label if old_result is not None else None,
                   label=label)
        self.branches.append(row)
        return label, row

    def run_cycle(self, n):
        start = time.perf_counter(); before_cost = dict(self.costs)
        theta_hash = weights_hash(self.policy.theta_minus)
        old = copy.deepcopy(self.policy)
        phi0_hash = weights_hash(old.phi)
        # Independent full initial-distribution data; suffix update includes
        # absolute discount and zero contributions from no-handoff trajectories.
        suffix_data = [self.collect("suffix_training", self.policy) for _ in range(self.config["suffix_training_episodes_per_cycle"])]
        self.costs["suffix_policy_training_steps"] += sum(len(x["records"]) for x in suffix_data)
        suffix_loss = update_suffix(self.policy, self.suffix_optimizer, suffix_data, self.config["rl"]["gamma"])
        self.costs["suffix_actor_updates"] += 1
        suffix_ids = [x["dataset_id"] for x in suffix_data]
        del suffix_data
        if weights_hash(self.policy.theta_minus) != theta_hash:
            raise RuntimeError("suffix update changed prefix behavior")
        phi1_hash = weights_hash(self.policy.phi)
        method = self.config["algorithm"]
        main_policy = self.policy if method == "stochastic_direct_mc" else old
        main = [self.collect("main", main_policy, stop=method == "direct_boundary_corrected") for _ in range(n)]
        for trajectory in main:
            tau = trajectory["tau"]
            prefix = len(trajectory["records"]) if tau is None else tau
            self.costs["main_prefix_steps"] += prefix
            self.costs["main_current_suffix_steps" if method == "stochastic_direct_mc" else "old_reference_suffix_steps"] += len(trajectory["records"]) - prefix
        predictions, draws, pilot_ids = [0.0] * n, [], []
        fit = dict(updates=0, mse=None, samples=0)
        ablation = self.config.get("ablation", "none")
        if method != "stochastic_direct_mc":
            pilots = [self.collect("pilot", old, stop=True) for _ in range(self.config["pilot_prefix_episodes_per_cycle"])]
            self.costs["pilot_prefix_steps"] += sum(len(x["records"]) for x in pilots)
            pilot_ids = [x["dataset_id"] for x in pilots]
            features, labels = [], []
            for trajectory in pilots:
                if trajectory["tau"] is not None:
                    label, _ = self.paired_label(trajectory, old, self.policy, "pilot")
                    features.append(trajectory["features"]); labels.append(label)
            if ablation != "predictor_zero":
                fit = self.predictor.fit(features, labels, updates=self.config["predictor"]["updates"], lr=self.config["predictor"]["lr"])
                self.predictor_optimizer_state = fit.pop("optimizer", None)
                self.costs["predictor_updates"] += fit["updates"]
                with torch.no_grad():
                    predictions = [0.0 if x["tau"] is None else float(self.predictor(x["features"])) for x in main]
            self.predictor.requires_grad_(False)
            predictor_hash = weights_hash(self.predictor)
            k = self.config["correction_draws_per_cycle"]
            indices = self.index_rng.choice(n, size=k, replace=True)
            for draw_number, index in enumerate(indices):
                trajectory = main[int(index)]
                label = 0.0
                if trajectory["tau"] is not None:
                    label, row = self.paired_label(trajectory, old, self.policy, "correction")
                    row.update(draw_number=draw_number, main_index=int(index), q=1/n,
                               prediction=predictions[index], residual=label - predictions[index],
                               residual_sign=int(np.sign(label - predictions[index])))
                draws.append((int(index), 1/n, label))
            if weights_hash(self.predictor) != predictor_hash:
                raise RuntimeError("formal labels changed the frozen predictor")
        else:
            predictor_hash = weights_hash(self.predictor)
        if weights_hash(self.policy.theta_minus) != theta_hash or weights_hash(old.phi) != phi0_hash or weights_hash(self.policy.phi) != phi1_hash:
            raise RuntimeError("policy version changed during formal collection")
        update = update_prefix(self.policy, self.prefix_optimizer, main, predictions, draws,
                               gamma=self.config["rl"]["gamma"], method=method, ablation=ablation)
        self.costs["prefix_actor_updates"] += 1
        if weights_hash(self.policy.phi) != phi1_hash:
            raise RuntimeError("prefix update changed frozen suffix")
        self.policy.assert_isolated()
        self.completed_main += n; self.cycle += 1
        handoffs = sum(x["tau"] is not None for x in main)
        row = dict(cycle=self.cycle, completed_main_trajectories=self.completed_main, N=n,
                   handoff_count=handoffs, no_handoff_count=n-handoffs, K=len(draws),
                   draws=[dict(index=i, q=q, label=label) for i, q, label in draws],
                   theta_hash=theta_hash, theta_after_hash=weights_hash(self.policy.theta_minus),
                   phi0_hash=phi0_hash, phi1_hash=phi1_hash, predictor_hash=predictor_hash,
                   predictions=predictions, predictor_fit=fit, suffix_loss=suffix_loss,
                   suffix_dataset_ids=suffix_ids, pilot_dataset_ids=pilot_ids,
                   main_dataset_ids=[x["dataset_id"] for x in main],
                   biased_ablation=ablation in ("remove_prediction", "remove_correction"),
                   hgr_boundary_mechanism_active=method == "hgr" and handoffs > 0,
                   **update, costs={key: self.costs[key] - before_cost[key] for key in self.costs},
                   wall_seconds=time.perf_counter() - start, complete=True)
        row["actual_total_environment_steps"] = sum(row["costs"][key] for key in COST_FIELDS)
        self.cycles.append(row)
        # Only compact evidence survives the update. No stale on-policy prefixes.
        del main, old
        write_json(self.output / "cycles.json", self.cycles)
        write_json(self.output / "episodes.json", self.episodes)
        write_json(self.output / "branches.json", self.branches)
        return row

    def save(self):
        if self.cycle < 1 or not self.cycles or not self.cycles[-1]["complete"]:
            raise ValueError("checkpoint requires a completed cycle")
        directory = self.output / "checkpoints"; directory.mkdir(exist_ok=True)
        path = directory / f"hgr_main_{self.completed_main:06d}_cycle_{self.cycle:06d}.pt"
        if path.exists():
            raise FileExistsError(path)
        payload = dict(schema=CHECKPOINT_SCHEMA, architecture_version=ARCHITECTURE_VERSION,
                       run_mode=self.run_mode, resume_from=self.resume_from,
                       implementation_version=IMPLEMENTATION_VERSION, source_identity=self.source_identity,
                       config=self.config, config_hash=digest(self.config), immutable_config_hash=digest(immutable_config(self.config)),
                       policy=self.policy.state_dict(), predictor=self.predictor.state_dict(),
                       prefix_optimizer=self.prefix_optimizer.state_dict(), suffix_optimizer=self.suffix_optimizer.state_dict(),
                       predictor_optimizer=self.predictor_optimizer_state, completed_main=self.completed_main,
                       cycle=self.cycle, stream_counter=self.stream_counter, index_rng=self.index_rng.bit_generator.state,
                       torch_rng=torch.get_rng_state(), costs=self.costs, cycles=self.cycles,
                       episodes=self.episodes, branches=self.branches, initialization=self.initialization,
                       cycle_complete=True, prefixes_valid=False, **objective_identity(self.config), **protocol_identity(self.config),
                       theta_hash=weights_hash(self.policy.theta_minus), phi_hash=weights_hash(self.policy.phi))
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary); temporary.replace(path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if loaded["config_hash"] != digest(self.config) or not loaded["cycle_complete"]:
            raise RuntimeError("checkpoint readback failed")
        return path

    def _resume(self, path):
        payload = load_checkpoint(path)
        if payload["source_identity"]["sha256"] != self.source_identity["sha256"]:
            raise ValueError("same-method resume production source mismatch")
        if payload["immutable_config_hash"] != digest(immutable_config(self.config)):
            raise ValueError("same-method resume requires matching objective, physics, architecture and immutable config")
        self.policy.load_state_dict(payload["policy"], strict=True)
        self.predictor.load_state_dict(payload["predictor"], strict=True)
        self.prefix_optimizer.load_state_dict(payload["prefix_optimizer"])
        self.suffix_optimizer.load_state_dict(payload["suffix_optimizer"])
        self.predictor_optimizer_state = payload["predictor_optimizer"]
        for name in ("completed_main", "cycle", "stream_counter", "costs", "cycles", "episodes", "branches", "initialization"):
            setattr(self, name, copy.deepcopy(payload[name]))
        self.index_rng.bit_generator.state = payload["index_rng"]
        torch.set_rng_state(payload["torch_rng"])
        self.policy.assert_isolated()

    def run(self):
        started = time.perf_counter(); latest = None
        while self.completed_main < self.config["total_main_trajectories"]:
            spent = sum(self.costs[key] for key in COST_FIELDS)
            budget = self.config.get("max_total_environment_steps")
            if budget is not None and spent >= budget:
                break
            # Start a complete cycle only below budget; complete all its original
            # horizons and explicitly report any overshoot. Never truncate labels.
            until_checkpoint = self.config["checkpoint_interval"] - self.completed_main % self.config["checkpoint_interval"]
            n = min(self.config["main_prefix_batch_size"], until_checkpoint,
                    self.config["total_main_trajectories"] - self.completed_main)
            self.run_cycle(n)
            latest = self.save()
        spent = sum(self.costs[key] for key in COST_FIELDS)
        summary = dict(method=self.config["method"], reward_objective=TEAM, gamma=self.config["rl"]["gamma"],
                       run_mode=self.run_mode, resume_from=self.resume_from,
                       implementation_version=IMPLEMENTATION_VERSION, source_sha256=self.source_identity["sha256"],
                       completed_main_trajectories=self.completed_main, completed_cycles=self.cycle,
                       actual_total_environment_steps=spent, costs=self.costs, wall_seconds=time.perf_counter()-started,
                       budget_policy="finish_started_cycle_then_stop", budget_overshoot_steps=max(0, spent-(self.config.get("max_total_environment_steps") or spent)),
                       latest_checkpoint=None if latest is None else str(latest), initialization=self.initialization,
                       formal_experiments_completed=False, performance_claims_supported=False)
        write_json(self.output / "summary.json", summary)
        return summary


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("not a same-architecture HGR-family checkpoint")
    if payload.get("implementation_version") != IMPLEMENTATION_VERSION or not payload.get("source_identity", {}).get("sha256"):
        raise ValueError("HGR checkpoint implementation/source version missing")
    validate_source_identity(payload['source_identity'])
    if not payload.get("cycle_complete") or payload.get("prefixes_valid") is not False:
        raise ValueError("checkpoint contains an incomplete cycle or stale prefixes")
    if payload.get("config_hash") != digest(payload["config"]):
        raise ValueError("checkpoint config hash mismatch")
    if payload.get('immutable_config_hash') != digest(immutable_config(payload['config'])):
        raise ValueError('checkpoint immutable config hash mismatch')
    validate_config(payload["config"])
    for name, expected in (("method", payload["config"]["method"]),
                           ("algorithm", payload["config"]["algorithm"]),
                           ("gamma", payload["config"]["rl"]["gamma"])):
        if name in payload and payload[name] != expected:
            raise ValueError(f"checkpoint {name} contradicts its training config")
    if objective_identity(payload) != objective_identity(payload["config"]) or protocol_identity(payload) != protocol_identity(payload["config"]):
        raise ValueError("checkpoint objective or physics identity mismatch")
    if (payload.get("cycle", 0) < 1 or not payload.get("cycles") or not payload["cycles"][-1]["complete"]
            or payload["completed_main"] != sum(c["N"] for c in payload["cycles"])
            or payload["costs"]["prefix_actor_updates"] != payload["cycle"]):
        raise ValueError("checkpoint complete-cycle counters mismatch")
    with torch.random.fork_rng():
        policy = HandoffPolicy(payload["config"]["policy"])
        policy.load_state_dict(payload["policy"], strict=True)
        if any(not bool(torch.isfinite(t).all()) for t in policy.state_dict().values()):
            raise ValueError("checkpoint has nonfinite policy state")
        if weights_hash(policy.theta_minus) != payload["theta_hash"] or weights_hash(policy.phi) != payload["phi_hash"]:
            raise ValueError("checkpoint policy hash mismatch")
    return payload


def run_training(config_path=DEFAULT_CONFIG, *, output_dir=None, resume=None, mean_initialization=None):
    config = load_config(config_path)
    return Trainer(config, output_dir or config["output_dir"], resume=resume, mean_initialization=mean_initialization).run()
