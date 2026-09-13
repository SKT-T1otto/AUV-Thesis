from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from chapter3_bser.experiments.phase1c_prrac import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_SCHEMA,
    IMPLEMENTATION_VERSION,
    METHOD,
)
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG


ARCHITECTURE = {
    "num_stages": 3,
    "encoder_hidden_dim": 8,
    "expert_hidden_dim": 8,
    "critic_hidden_dim": 16,
    "router_temperature": 1.0,
    "gate_initial_mean": 0.75,
    "alignment_scale_init": 1.0,
}
LOSS = {
    "router_ce_coef": 0.05,
    "gate_conflict_coef": 0.01,
    "gate_entropy_coef": 0.001,
    "residual_action_reg": 0.01,
}


def evaluation_config(*, scenario_count: int = 2) -> dict[str, Any]:
    config = copy.deepcopy(evaluator._load_config(evaluator.DEFAULT_CONFIG))
    config["evaluation_episodes"] = int(scenario_count)
    config["max_steps"] = 1
    config["workers"] = 2
    config["failure_trace"]["enabled"] = False
    return config


def checkpoint_payload() -> dict[str, Any]:
    training_config = json.loads(
        (evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_train.json").read_text(
            encoding="utf-8"
        )
    )
    learner = PRRACMADDPG(
        architecture=ARCHITECTURE,
        loss=LOSS,
        gamma=0.95,
        tau=0.005,
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "metadata": {
            "method": METHOD,
            "implementation_version": IMPLEMENTATION_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "config_hash": "test-config-hash",
            "completed_episode": 12,
            "observation_dim": 28,
            "action_dim": 3,
            "critic_dim": 124,
            "architecture": copy.deepcopy(ARCHITECTURE),
            "loss": copy.deepcopy(LOSS),
            "reward": copy.deepcopy(training_config["reward"]),
            "execution_runtime_revision": "dynamic_public_intercept_v2_1",
        },
        "prrac_training_state": learner.training_state_dict(),
        "completed_episode": 12,
    }


def write_checkpoint(path: Path, payload: dict[str, Any] | None = None) -> Path:
    torch.save(checkpoint_payload() if payload is None else payload, path)
    return path


def worker_jobs(path: Path, count: int = 2) -> list[dict[str, Any]]:
    config = evaluation_config(scenario_count=count)
    learner, payload = evaluator.load_prrac_checkpoint(path, config=config)
    scenarios, _ = evaluator._build_evaluation_manifest(config)
    metadata = dict(payload["metadata"])
    state = dict(payload["prrac_training_state"])
    info = evaluator._checkpoint_info(path, payload, "full_prrac")
    snapshot = learner.policy_snapshot()
    trace = {
        "enabled": False,
        "only_found_failures": True,
        "max_traces": 0,
    }
    return [
        {
            "episode_index": index,
            "scenario": scenarios[index],
            "config": config,
            "checkpoint_info": info,
            "architecture": metadata["architecture"],
            "loss": metadata["loss"],
            "gamma": state["gamma"],
            "tau": state["tau"],
            "reward": metadata["reward"],
            "policy_snapshot": snapshot,
            "failure_trace": trace,
            "device": "cpu",
        }
        for index in range(count)
    ]

from chapter3_bser.experiments.phase1c_common import Phase1CTransitionMetadata, TransitionPhase
from chapter3_bser.experiments.phase1c_prrac.transition_protocol import PRRACTransitionMetadata
from chapter3_bser.models.prrac.stage_mapping import transition_phase_to_prrac_stage

def _transitions():
    phases = [TransitionPhase.PRE_FOUND] * 4 + [TransitionPhase.POST_FOUND] * 4 + [TransitionPhase.HOLD] * 4
    rows = []
    previous = TransitionPhase.PRE_FOUND
    for index, phase in enumerate(phases):
        found = phase != TransitionPhase.PRE_FOUND
        hold = phase == TransitionPhase.HOLD
        base = Phase1CTransitionMetadata.build(
            episode_id=0, episode_index=0, step=index + 1,
            task_found=found, executor_target_assigned=found,
            contact=hold, full_hold=hold, hold_counter=int(hold), mission_complete=False,
        )
        metadata = PRRACTransitionMetadata(
            base, transition_phase_to_prrac_stage(previous), transition_phase_to_prrac_stage(phase)
        )
        obs = tuple(torch.randn(28) for _ in range(4))
        actions = torch.tanh(torch.randn(4, 3))
        rewards = torch.randn(4)
        next_obs = tuple(value + 0.01 for value in obs)
        rows.append((obs, actions, rewards, next_obs, (False,) * 4, (False,) * 4, metadata))
        previous = phase
    return rows


class _ImmediateExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, function, jobs):
        del function
        results = []
        for job in jobs:
            info = dict(job["checkpoint_info"])
            results.append(
                {
                    "episode": {
                        **info,
                        "scenario_id": str(job["scenario"]["scenario_id"]),
                        "scenario_seed": int(job["scenario"]["scenario_seed"]),
                        "found": True,
                        "contact_episode": True,
                        "hold_episode": False,
                        "success": False,
                        "collision_episode": False,
                        "post_found_collision_count": 0,
                        "executor_invalid_count": 0,
                        "executor_invalid_assignment_unreachable_count": 0,
                        "executor_min_distance_to_target": 1.0,
                        "executor_final_distance_to_target": 2.0,
                        "executor_replan_count": 0,
                        "executor_residual_ratio_post_found": 0.1,
                        "handoff_delay": 1,
                        "found_to_success_steps": None,
                        "failure_stage": "FOUND_NO_CONTACT",
                        "router_confusion_matrix": [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                        "gate_mean": 0.5,
                        "gate_p10": 0.4,
                        "gate_p90": 0.6,
                        "alignment_negative_rate": 0.0,
                    },
                    "failure_trace": [],
                    "trace_index": None,
                }
            )
        return results
