"""Independent MADDPG baseline identities and real model/optimizer checkpoints."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re

import torch

from chapter3_bser.experiments.reward_objective import TEAM, objective_identity
from core.env.task_protocol import STRICT, protocol_identity, require_protocol_output, validate_task_config

ROOT = Path(__file__).resolve().parents[4]
CONFIG_SCHEMA = "ch3.baseline.training.v1"
CHECKPOINT_SCHEMA = "ch3.baseline.maddpg.v1"
IMPLEMENTATION_VERSION = "ch3.baseline.offpolicy.v1"
IDENTITIES = {
    "B2_direct_mc": ("maddpg", "ch3_baseline_direct_mc", "ch3.baseline.maddpg.v1"),
    "B3_direct_boundary": ("direct_boundary_maddpg", "ch3_baseline_direct_boundary", "ch3.baseline.boundary_maddpg.v1"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_config(config, expected_baseline=None):
    config = copy.deepcopy(dict(config))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("not an independent baseline training config; an HGR checkpoint cannot be relabeled")
    baseline = config.get("baseline")
    if baseline not in IDENTITIES or (expected_baseline is not None and baseline != expected_baseline):
        raise ValueError("baseline identity mismatch")
    algorithm, method, architecture = IDENTITIES[baseline]
    for key, expected in (("algorithm", algorithm), ("method", method),
                          ("architecture_version", architecture), ("checkpoint_schema", CHECKPOINT_SCHEMA)):
        if config.get(key) != expected:
            raise ValueError(f"baseline {key} mismatch; relabeling weights is forbidden")
    forbidden = {"policy", "predictor", "prefix_lr", "suffix_lr", "main_prefix_batch_size",
                 "suffix_training_episodes_per_cycle", "pilot_prefix_episodes_per_cycle",
                 "correction_draws_per_cycle"}
    if forbidden.intersection(config):
        raise ValueError("independent MADDPG baselines do not accept HGR cycle/policy/predictor fields")
    validate_task_config(config)
    if protocol_identity(config)["task_protocol"] != STRICT or objective_identity(config)["reward_objective"] != TEAM:
        raise ValueError("baselines require collision_terminal_v1 and team_mean_v1")
    if config.get("observation_dim") != 28 or config.get("action_dim") != 3 or config.get("critic_input_dim", 124) != 124:
        raise ValueError("baseline 28D observation/3D action/124D critic contract mismatch")
    if config.get("base_candidate") != "ch3_v3_full_reference" or config.get("profile") != "M20_MOVING_UNKNOWN_MULTI":
        raise ValueError("baseline environment/profile mismatch")
    if config.get("ablation", "none") != "none" or config.get("early_discovery", {}).get("enabled"):
        raise ValueError("baseline training does not accept ablations")
    for key in ("search_value", "search_value_decision", "search_value_guidance"):
        if config.get(key, {}).get("enabled"):
            raise ValueError("baseline training does not use historical search-value guidance")
    reward = config.get("reward", {})
    if (not reward.get("freeze_searchers_after_found") or reward.get("executor_id") != 3
            or reward.get("searcher_ids") != [0, 1, 2]):
        raise ValueError("baseline reward/frozen searcher/role contract mismatch")
    for key in ("total_main_trajectories", "checkpoint_interval", "max_steps"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int) or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    budget = config.get("max_total_environment_steps")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise ValueError("max_total_environment_steps must be null or a positive integer")
    rl = config.get("rl", {})
    for key in ("batch_size", "replay_size", "update_frequency", "updates_per_train", "hidden_dim"):
        if isinstance(rl.get(key), bool) or not isinstance(rl.get(key), int) or rl[key] < 1:
            raise ValueError(f"rl.{key} must be a positive integer")
    if rl["replay_size"] < rl["batch_size"]:
        raise ValueError("replay capacity must be at least batch_size")
    if isinstance(rl.get("warmup_steps"), bool) or not isinstance(rl.get("warmup_steps"), int) or rl["warmup_steps"] < 0:
        raise ValueError("rl.warmup_steps must be a nonnegative integer")
    for key in ("gamma", "tau", "lr_actor", "lr_critic"):
        if not isinstance(rl.get(key), (int, float)) or not math.isfinite(rl[key]) or rl[key] <= 0:
            raise ValueError(f"rl.{key} must be finite and positive")
    if rl["gamma"] > 1 or rl["tau"] > 1 or rl.get("residual_action_reg") != 0:
        raise ValueError("invalid gamma/tau or nonzero expert residual regularization")
    if baseline == "B2_direct_mc" and "boundary_encoder" in config:
        raise ValueError("B2 has no boundary encoder")
    if baseline == "B3_direct_boundary":
        boundary = config.get("boundary_encoder", {})
        if (boundary.get("schema", "ch3.baseline.boundary_encoder.v1") != "ch3.baseline.boundary_encoder.v1"
                or type(boundary.get("hidden_dim", 32)) is not int or boundary.get("hidden_dim", 32) < 1):
            raise ValueError("invalid boundary encoder schema or hidden dimension")
    config.update(objective_identity(config))
    return config


def validated_output(config, output):
    path = Path(output).resolve()
    require_protocol_output(config, path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"use a new or empty output directory: {path}")
    return path


def fresh_source_identity():
    # This records the same full production inventory used by historical gates,
    # but has an independent implementation version and no HGR dependency.
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for package in ("core", "chapter3_bser") for p in sorted((ROOT / package).rglob("*.py"))}
    return dict(implementation_version=IMPLEMENTATION_VERSION,
                sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), files=files)


def validate_source_identity(identity):
    if not isinstance(identity, dict) or identity.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("invalid baseline production source identity version")
    files = identity.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("production source identity requires its complete file inventory")
    for name, value in files.items():
        if not isinstance(name, str):
            raise ValueError("invalid production source filename")
        path = PurePosixPath(name)
        if (name != path.as_posix() or path.is_absolute() or not path.parts or ".." in path.parts
                or path.parts[0] not in ("core", "chapter3_bser") or path.suffix != ".py"
                or not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None):
            raise ValueError(f"invalid production source record: {name!r}")
    if identity.get("sha256") != hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest():
        raise ValueError("production source inventory/aggregate hash mismatch")
    return identity


def require_source_match(saved, current=None, *, context="evaluation"):
    validate_source_identity(saved)
    current = fresh_source_identity() if current is None else validate_source_identity(current)
    if saved != current:
        changed = sorted(name for name in saved["files"].keys() | current["files"].keys()
                         if saved["files"].get(name) != current["files"].get(name))
        raise ValueError(f"{context} production source mismatch: " + ", ".join(changed[:12]))


def state_digest(state):
    """Hash tensors and optimizer state without serialization-container timestamps."""
    hashed = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            hashed.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            hashed.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                hashed.update(repr(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            hashed.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            hashed.update(repr(value).encode())
    visit(state)
    return hashed.hexdigest()


def _finite_state(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_model(payload):
    from .model import build_model
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not an independent baseline checkpoint; an HGR checkpoint cannot be relabeled")
    config = validate_config(payload["config"], payload.get("baseline"))
    for key in ("baseline", "algorithm", "method", "architecture_version"):
        if payload.get(key) != config[key]:
            raise ValueError(f"checkpoint {key} identity mismatch; relabeling weights is forbidden")
    if payload.get("config_hash") != digest(config):
        raise ValueError("checkpoint config hash mismatch")
    require_source_match(payload.get("source_identity"), context="model load")
    state = payload.get("model_training_state")
    if not isinstance(state, dict) or payload.get("model_state_sha256") != state_digest(state) or not _finite_state(state):
        raise ValueError("checkpoint model/optimizer state hash or finite-value check failed")
    with torch.random.fork_rng():
        model = build_model(config, env=None, device="cpu")
        model.load_training_state_dict(state)
    if int(model.niter) != payload.get("optimizer_updates"):
        raise ValueError("checkpoint optimizer/update count mismatch")
    required = {"policy", "target_policy", "critic1", "target_critic1", "critic2", "target_critic2",
                "policy_opt", "critic1_opt", "critic2_opt"}
    for agent, params in zip(model.agents, state["agent_params"]):
        if set(params) != required:
            raise ValueError("checkpoint is missing model or optimizer state")
        for name, optimizer in (("policy_opt", agent.policy_optimizer), ("critic1_opt", agent.critic1_optimizer),
                                ("critic2_opt", agent.critic2_optimizer)):
            saved = params[name]
            ids = [index for group in saved["param_groups"] for index in group["params"]]
            if len(ids) != len(set(ids)) or (payload["optimizer_updates"] > 0 and set(saved["state"]) != set(ids)):
                raise ValueError("checkpoint optimizer parameter/state inventory mismatch")
            for parameter, item in optimizer.state.items():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key not in item or item[key].shape != parameter.shape:
                        raise ValueError("checkpoint optimizer moment shape mismatch")
    model.prep_rollouts(device="cpu")
    return model


def load_checkpoint(path, expected_baseline=None):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("not an independent baseline checkpoint; an HGR checkpoint cannot be relabeled")
    config = validate_config(payload.get("config", {}), expected_baseline)
    if payload.get("config_hash") != digest(config) or payload.get("config") != config:
        raise ValueError("checkpoint config hash mismatch")
    for key in ("baseline", "algorithm", "method", "architecture_version"):
        if payload.get(key) != config[key]:
            raise ValueError(f"checkpoint {key} identity mismatch; relabeling weights is forbidden")
    if payload.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("checkpoint implementation version mismatch")
    require_source_match(payload.get("source_identity"), context="checkpoint load")
    count = payload.get("completed_main_trajectories")
    updates = payload.get("optimizer_updates")
    steps = payload.get("actual_total_environment_steps")
    if (not isinstance(count, int) or isinstance(count, bool) or count < 1
            or payload.get("episode_count") != count or not payload.get("episode_complete")
            or not isinstance(updates, int) or isinstance(updates, bool) or updates < 0
            or not isinstance(steps, int) or steps < count
            or len(payload.get("episodes", [])) != count):
        raise ValueError("checkpoint completed-episode counters mismatch")
    if (sum(row.get("episode_steps", 0) for row in payload["episodes"]) != steps
            or sum(row.get("optimizer_updates", 0) for row in payload["episodes"]) != updates
            or count > config["total_main_trajectories"]
            or payload.get("actor_optimizer_steps") != updates * 4
            or payload.get("critic_optimizer_steps") != updates * 8):
        raise ValueError("checkpoint environment-step/optimizer counters mismatch")
    previous_step = 0
    for index, row in enumerate(payload["episodes"], start=1):
        length = row.get("episode_steps")
        if (row.get("episode") != index or type(length) is not int or not 1 <= length <= config["max_steps"]
                or row.get("global_step") != previous_step + length
                or row.get("baseline") != config["baseline"] or row.get("algorithm") != config["algorithm"]
                or not row.get("trajectory_complete")):
            raise ValueError("checkpoint episode history mismatch")
        previous_step += length
    if objective_identity(payload) != objective_identity(config) or protocol_identity(payload) != protocol_identity(config):
        raise ValueError("checkpoint objective or task protocol mismatch")
    load_model(payload)
    return payload
