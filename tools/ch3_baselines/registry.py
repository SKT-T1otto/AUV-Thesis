"""Paper identities mapped to immutable production method identities."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from chapter3_bser.experiments.hgr import evaluation as native_evaluation
from chapter3_bser.experiments.hgr.train import validate_config
from chapter3_bser.models.hgr import METHODS
from .provenance import ROOT

REGISTRY_PATH = ROOT / "configs/chapter3/baselines/baseline_registry.json"
DEFAULT_REFERENCE = ROOT / "configs/chapter3/hgr_train.json"
CONTRACTS = {
    "B0_search_prior": ("ch3_baseline_search_prior", "ch3_basic_search_prior_v1", "search_only", "none", None),
    "B1_bser_prior": ("ch3_baseline_bser_prior", "ch3_baseline_bser_prior", "bser_joint", "none", None),
    "B2_direct_mc": ("ch3_baseline_direct_mc", "ch3_stochastic_direct_mc", "bser_joint", "complete_mc_return", "stochastic_direct_mc"),
    "B3_direct_boundary": ("ch3_baseline_direct_boundary", "ch3_direct_boundary_corrected", "bser_joint", "current_boundary_value_correction", "direct_boundary_corrected"),
}


def load_registry():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if set(registry) != set(CONTRACTS):
        raise ValueError("baseline registry must contain exactly B0 through B3")
    for key, (method, native, planner, learning_mode, algorithm) in CONTRACTS.items():
        spec = registry[key]
        learning = algorithm is not None
        expected = dict(method=method, runtime_method=native, planner=planner,
                        learning_mode=learning_mode, algorithm=algorithm, learning=learning,
                        training_required=learning, controller="residual_policy" if learning else "prior_only",
                        residual_source="trained_stochastic_policy" if learning else "zeros_4x3", hgr=False)
        if any(spec.get(k) != v for k, v in expected.items()):
            raise ValueError(f"invalid method isolation contract: {key}")
        if learning and algorithm not in METHODS:
            raise ValueError(f"production algorithm missing: {algorithm}; do not substitute or copy HGR")
    return registry


def method_spec(baseline):
    registry = load_registry()
    if baseline not in registry:
        raise ValueError(f"unknown baseline: {baseline}")
    return copy.deepcopy(registry[baseline])


def task_conditions(config, *, episodes=1, seed=12729):
    config = validate_config(config)
    resolved = native_evaluation.resolved_config(config, episodes, seed, "stochastic")
    expected = dict(profile="M20_MOVING_UNKNOWN_MULTI", max_steps=400,
                    task_protocol="collision_terminal_v1", collision_detection_revision="segment_closed_aabb_v1",
                    terminal_reward_revision="team_failure_override_v1", collision_terminal_reward=-2.0,
                    reward_objective="team_mean_v1", gamma=0.95,
                    execution_runtime_revision="dynamic_public_intercept_v2_1")
    if any(resolved.get(k) != v for k, v in expected.items()):
        raise ValueError("baseline task/collision/reward/runtime contract mismatch")
    if config.get("base_candidate") != "ch3_v3_full_reference":
        raise ValueError("baseline comparison requires the full reference base candidate")
    if config.get("ablation", "none") != "none" or config.get("early_discovery", {}).get("enabled"):
        raise ValueError("independent baselines do not accept evaluation/training ablations")
    if config["reward"].get("executor_id") != 3 or config["reward"].get("searcher_ids") != [0, 1, 2]:
        raise ValueError("baseline role order mismatch")
    if not resolved["environment_config"].get("use_residual_prior"):
        raise ValueError("baseline requires the existing residual prior")
    conditions = {k: copy.deepcopy(resolved[k]) for k in (*expected, "base_candidate", "reward",
                  "source_reward_revision", "environment_config", "execution_runtime", "phase1b_config",
                  "phase1b_reference_config_sha256")}
    conditions.update(observation_dim=config["observation_dim"], action_dim=config["action_dim"])
    # B0 alone declares this standby intervention; it is not a shared task field.
    conditions["environment_config"].pop("pse_use_standby", None)
    return conditions


def load_reference(path=None):
    path = DEFAULT_REFERENCE if path is None else Path(path).resolve()
    config = validate_config(json.loads(path.read_text(encoding="utf-8")))
    if config["method"] != "ch3_hgr":
        raise ValueError("reference-training-config must identify HGR; baseline identity is separate")
    task_conditions(config)
    return path, config


def validate_method_config(spec, config, reference):
    config = validate_config(config)
    if spec["learning"] and (config["method"] != spec["runtime_method"] or config["algorithm"] != spec["algorithm"]):
        raise ValueError("checkpoint/config belongs to another method; relabeling weights is forbidden")
    if task_conditions(config) != task_conditions(reference):
        raise ValueError("baseline and reference task conditions differ")
    for key in ("architecture_version", "checkpoint_schema", "policy"):
        if config[key] != reference[key]:
            raise ValueError(f"baseline/reference network contract mismatch: {key}")
    return config


def training_config(baseline, *, reference_training_config=None, output_dir=None):
    spec = method_spec(baseline)
    if not spec["training_required"]:
        raise ValueError("B0/B1 are training-free and have no trainer")
    reference_path, reference = load_reference(reference_training_config)
    if reference_training_config is None:
        config = json.loads((ROOT / spec["training_config"]).read_text(encoding="utf-8"))
    else:
        config = copy.deepcopy(reference)
        config.update(algorithm=spec["algorithm"], method=spec["runtime_method"], output_dir=spec["default_training_output"])
    allowed = {"method", "algorithm", "output_dir"}
    if {k for k in config.keys() | reference.keys() if config.get(k) != reference.get(k)} - allowed:
        raise ValueError("training config may change only method, algorithm and output_dir from reference")
    if output_dir is not None:
        config["output_dir"] = str(Path(output_dir).resolve())
    validate_method_config(spec, config, reference)
    return spec, reference_path, config
