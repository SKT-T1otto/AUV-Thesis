"""Explicit actor import and source identity; never migrates replay or optimizers."""
import copy
import hashlib
import json
from pathlib import Path

import torch

from core.env.task_protocol import LEGACY, STRICT, protocol_identity
from chapter3_bser.experiments.reward_objective import objective_identity


def checkpoint_path(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix != ".pt" or path.name.startswith("."):
        raise ValueError("supply a completed .pt checkpoint, not a temporary file")
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_training_config(path, payload):
    from .train_phase1c_prrac import _config_hash
    value = payload.get("resolved_training_config")
    if value is None:
        sidecar = Path(path).parent.parent / "resolved_training_config.json"
        if not sidecar.is_file():
            raise ValueError("protocol transfer requires the original resolved_training_config.json beside the source run")
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    if _config_hash(value) != payload["metadata"]["config_hash"]:
        raise ValueError("source training config hash mismatch")
    if protocol_identity(value) != protocol_identity(payload["metadata"]):
        raise ValueError("source training protocol metadata mismatch")
    if objective_identity(value) != objective_identity(payload["metadata"]):
        raise ValueError("source training reward objective metadata mismatch")
    for key in ("architecture", "loss", "reward", "execution_runtime_revision", "observation_dim", "action_dim", "critic_dim"):
        if value.get(key) != payload["metadata"].get(key):
            raise ValueError(f"source training config/metadata {key} mismatch")
    return copy.deepcopy(value)


def import_actors(path, learner, config):
    from .evaluate_prrac_checkpoints import _validate_checkpoint_payload
    path = checkpoint_path(path)
    digest = file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_checkpoint_payload(payload, {**config, "allow_protocol_transfer": True, "allow_objective_transfer": True})
    source = source_training_config(path, payload)
    if protocol_identity(config)["task_protocol"] != STRICT:
        raise ValueError("actor warmstart entry requires collision_terminal_v1")
    if payload["metadata"]["architecture"] != dict(config["architecture"]):
        raise ValueError("actor warmstart architecture mismatch")
    states = payload["prrac_training_state"]["agents"]
    if len(states) != 4:
        raise ValueError("actor warmstart requires four actors")
    # Validate every actor before modifying any actor, including buffers.
    for agent, state in zip(learner.agents, states):
        expected, actual = agent.actor.state_dict(), state["actor"]
        if expected.keys() != actual.keys() or any(expected[k].shape != actual[k].shape for k in expected):
            raise ValueError("actor warmstart parameter keys/shapes mismatch")
        if any(not torch.isfinite(v).all() for v in actual.values()):
            raise ValueError("actor warmstart contains nonfinite parameters")
    if file_sha256(path) != digest:
        raise ValueError("source checkpoint changed during import")
    for agent, state in zip(learner.agents, states):
        agent.actor.load_state_dict(state["actor"], strict=True)
        agent.target_actor.load_state_dict(agent.actor.state_dict(), strict=True)
    rows = payload.get("episode_metrics", [])
    return {
        "mode": "actor_warmstart", "source_checkpoint": str(path),
        "source_checkpoint_sha256": digest,
        "source_training_protocol": protocol_identity(source)["task_protocol"],
        "source_reward_objective": objective_identity(source)["reward_objective"],
        "target_reward_objective": objective_identity(config)["reward_objective"],
        "source_completed_episode": payload.get("completed_episode"),
        "source_environment_steps": payload.get("global_step"),
        "source_training_updates": payload.get("update_step"),
        "source_wall_seconds": payload.get("training_wall_seconds"),
        "source_collector_wall_seconds": sum(r["wall_seconds"] for r in rows)
            if rows and all(r.get("wall_seconds") is not None for r in rows) else None,
    }
