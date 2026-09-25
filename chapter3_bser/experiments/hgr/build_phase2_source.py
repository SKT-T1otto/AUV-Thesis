"""Prepare a Phase2 frozen source without training or simulator rollouts.

The configured policy_source_path must be a manually supplied policy-only export:
  schema: hgr.phase2.policy_pair.v1
  source_identity: the complete production identity at export time
  runtime_contract_sha256: sha256(canonical(runtime_contract(runtime_config)))
  old_policy, new_policy: {state, training_modes, behavior_sha256}, exactly as in
    the Phase2 frozen-source policy payloads (full HandoffPolicy state, including
    theta_minus.* and phi.*; no optimizer, replay, or Trainer checkpoint)
  preparation_environment_steps: actual environment steps spent preparing it
  origin: {kind: real_policy_export, description, theta_id, phi0_id, phi1_id}

Origin is a human provenance declaration, not a proof of training history. This
module never invents or perturbs policy weights, imports a training checkpoint,
or reassigns an old export's source identity. Missing/invalid policy exports
return SOURCE_NOT_AVAILABLE before scenario generation or output creation.

Output keeps hgr.phase2.frozen_source.v1 unchanged: the identical theta is stored
under both policy state dictionaries; scenario and policy audit data live in
each case. The 401-state target trace is a deterministic full-horizon projection
using the existing motion function, not an executed mission trajectory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from core.config.ch3_config import build_ch3_config
from core.env.target_motion import TargetState, simulate_target_trajectory
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from chapter3_bser.models.hgr.phase1 import (
    behavior_identity, canonical, identical_behavior, isolated_global_rng, runtime_contract,
)
from .phase2_gradient_efficiency import (
    FrozenSnapshotSource, SOURCE_SCHEMA, _policy_from_payload,
    load_config as load_evaluation_config, resolve_config, save_snapshot_source,
)
from .provenance import fresh_source_identity, require_source_match

ROOT = Path(__file__).resolve().parents[3]
BUILDER_SCHEMA = "hgr.phase2.source_builder.v1"
POLICY_PAIR_SCHEMA = "hgr.phase2.policy_pair.v1"
PROFILE = "M20_MOVING_UNKNOWN_MULTI"
HORIZON = 400
DEFAULT_CONFIG = ROOT / "configs/chapter3/hgr_phase2_source_builder.json"


class PolicySourceUnavailable(ValueError):
    """A genuine, compatible, nonidentical policy pair was not supplied."""


def _sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path=DEFAULT_CONFIG):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_config(config):
    config = copy.deepcopy(dict(config))
    expected = {"schema", "evaluation_config", "source_output_path", "scenario_seeds", "policy_source_path"}
    if set(config) != expected or config["schema"] != BUILDER_SCHEMA:
        raise ValueError("invalid Phase2 source builder configuration")
    seeds = config["scenario_seeds"]
    if (not isinstance(seeds, list) or len(seeds) != 10
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or len(set(seeds)) != 10):
        raise ValueError("scenario_seeds must contain exactly 10 distinct nonnegative int32 seeds")
    value = config["evaluation_config"]
    evaluation = resolve_config(load_evaluation_config(_path(value)) if isinstance(value, (str, Path)) else value)
    runtime = evaluation["runtime_config"]
    if (evaluation["scope"] != "fresh_main" or evaluation["snapshot_count"] != 10
            or runtime["profile"] != PROFILE or runtime["max_steps"] != HORIZON):
        raise ValueError("builder requires fresh_main, 10 scenarios, M20_MOVING_UNKNOWN_MULTI, H=400")
    if not isinstance(config["source_output_path"], (str, Path)) or not str(config["source_output_path"]).strip():
        raise ValueError("source_output_path is required")
    output = _path(config["source_output_path"])
    if output.suffix != ".pt":
        raise ValueError("source_output_path must be a .pt file")
    policy_path = config["policy_source_path"]
    if policy_path is not None and (not isinstance(policy_path, (str, Path)) or not str(policy_path).strip()):
        raise ValueError("policy_source_path must be a file path or null")
    config.update(source_output_path=output,
                  policy_source_path=None if policy_path is None else _path(policy_path))
    return config, runtime


def validate_policy_pair(old, new, runtime):
    """Exact theta equality and distinct phi identities; never repair either."""
    contract = runtime_contract(runtime)
    old_theta = behavior_identity(old.theta_minus, contract)
    new_theta = behavior_identity(new.theta_minus, contract)
    if not identical_behavior(old_theta, new_theta):
        raise PolicySourceUnavailable("theta behavior identity mismatch; weights/modes will not be copied or repaired")
    phi0 = behavior_identity(old.phi, contract)
    phi1 = behavior_identity(new.phi, contract)
    if identical_behavior(phi0, phi1):
        raise PolicySourceUnavailable("phi0 and phi1 must have different behavior identities")
    return dict(theta_behavior_sha256=old_theta.sha256,
                phi0_behavior_sha256=phi0.sha256, phi1_behavior_sha256=phi1.sha256)


def load_policy_pair(path, runtime):
    """Only the explicit policy-only schema; no checkpoint autodetection/fallback."""
    if path is None:
        raise PolicySourceUnavailable("policy_source_path is null; supply a real policy-only pair export")
    path = _path(path)
    if not path.is_file():
        raise PolicySourceUnavailable(f"policy source file is missing: {path}")
    try:
        raw = path.read_bytes()
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        expected = {"schema", "source_identity", "runtime_contract_sha256", "old_policy", "new_policy",
                    "preparation_environment_steps", "origin"}
        if not isinstance(payload, dict) or set(payload) != expected or payload["schema"] != POLICY_PAIR_SCHEMA:
            raise ValueError("requires hgr.phase2.policy_pair.v1; Trainer checkpoints and bare state_dicts are forbidden")
        require_source_match(payload["source_identity"], fresh_source_identity(), context="Phase2 policy-only export")
        if payload["runtime_contract_sha256"] != _sha(runtime_contract(runtime)):
            raise ValueError("policy source runtime contract mismatch")
        cost = payload["preparation_environment_steps"]
        if type(cost) is not int or cost < 0:
            raise ValueError("invalid preparation_environment_steps")
        origin = payload["origin"]
        if (not isinstance(origin, dict)
                or set(origin) != {"kind", "description", "theta_id", "phi0_id", "phi1_id"}
                or origin["kind"] != "real_policy_export"
                or any(not isinstance(v, str) or not v.strip() for v in origin.values())):
            raise ValueError("explicit real_policy_export origin metadata is required")
        for key in ("old_policy", "new_policy"):
            if not isinstance(payload[key], dict) or set(payload[key]) != {"state", "training_modes", "behavior_sha256"}:
                raise ValueError("invalid policy-only payload fields")
        # The existing factory strictly loads ALL supplied tensors and checks the
        # complete forward identity. Random initialization is never used as data.
        with isolated_global_rng():
            old = _policy_from_payload(payload["old_policy"], runtime)
            new = _policy_from_payload(payload["new_policy"], runtime)
        identities = validate_policy_pair(old, new, runtime)
        source = FrozenSnapshotSource(old, new, [], "fresh_main", payload["source_identity"], cost)
        audit = dict(schema=POLICY_PAIR_SCHEMA, path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                     origin=copy.deepcopy(origin), **identities)
        return source, audit
    except PolicySourceUnavailable:
        raise
    except Exception as exc:
        raise PolicySourceUnavailable(f"policy source rejected: {exc}") from exc


def _scenario_audit(scenario, runtime):
    """Project the existing motion from the state precision used by UAVEnv.reset."""
    if (scenario["scenario_profile"] != PROFILE or scenario["max_steps"] != HORIZON
            or scenario["target_motion_mode"] != "constant_velocity_reflect_v1"
            or scenario["obstacle_layout_id"] != "custom_aabb_v1"
            or scenario["obstacle_knowledge_mode"] != "online_unknown"):
        raise ValueError("generator returned an incompatible M20 scenario")
    agents = np.asarray(scenario["initial_agent_positions"], dtype=np.float64)
    if agents.shape != (4, 3) or not np.isfinite(agents).all():
        raise ValueError("invalid initial agent positions")
    obstacles = scenario["obstacles"]
    if not 2 <= len(obstacles) <= 4 or scenario["obstacle_layout_sha256"] != _sha(obstacles):
        raise ValueError("invalid obstacle layout/hash")
    # The source generator records rounded float64 values. Runtime reset stores
    # target position and obstacle geometry in float32, then motion uses float64.
    # Do not rewrite the generator's target_trajectory_sha256 with a new meaning.
    runtime_obstacles = [{key: np.asarray(box[key], dtype=np.float32).astype(np.float64).tolist()
                          for key in ("center", "size")} for box in obstacles]
    env_config = build_ch3_config(runtime["base_candidate"], PROFILE)
    if env_config["target_continues_after_detection"] is not True:
        raise ValueError("a fixed target trajectory requires target_continues_after_detection")
    dt = float(env_config.get("dt", .2))
    bounds = np.asarray(env_config.get("space_size", [20., 20., 8.]), dtype=np.float32).astype(np.float64)
    state = TargetState(np.asarray(scenario["target_initial_position"], dtype=np.float32),
                        scenario["target_initial_velocity"], 0, scenario["target_motion_mode"],
                        state_schema=scenario["target_state_schema"], obstacle_layout_id=scenario["obstacle_layout_id"])
    trace = simulate_target_trajectory(
        state, HORIZON, dt, bounds, runtime_obstacles,
        clearance=env_config["target_obstacle_clearance"],
        max_reflections=env_config["target_max_reflections_per_step"], max_prediction_steps=HORIZON)
    trajectory = dict(kind="deterministic_runtime_projection", sample_steps=list(range(HORIZON+1)),
                      dt=dt, max_steps=HORIZON, motion_mode=state.motion_mode,
                      positions=[s.position.tolist() for s in trace],
                      velocities=[s.velocity.tolist() for s in trace],
                      reflection_counts=[s.reflection_count for s in trace],
                      runtime_obstacle_layout_hash=_sha(runtime_obstacles),
                      initial_precision="float32 position/obstacles; float64 velocity",
                      bounds=bounds.tolist(), target_obstacle_clearance=env_config["target_obstacle_clearance"],
                      max_reflections_per_step=env_config["target_max_reflections_per_step"])
    # canonical rejects nonfinite audit values, too.
    return dict(runtime_profile=PROFILE, max_steps=HORIZON,
                scenario_id=scenario["scenario_id"], scenario_seed=scenario["scenario_seed"],
                initial_agent_positions=agents.tolist(), obstacle_layout_hash=_sha(obstacles),
                target_trajectory=trajectory, target_trajectory_hash=_sha(trajectory))


def build_cases(seeds, runtime):
    """Use one predeclared generator seed per case; never select by handoff."""
    cases = []
    with isolated_global_rng():
        for index, seed in enumerate(seeds):
            manifest = build_scenario_manifests(count=1, generator_seed=seed, split="validation", profiles=[PROFILE])
            scenarios = manifest[PROFILE]["scenarios"]
            if len(scenarios) != 1 or scenarios[0]["scenario_seed"] != seed:
                raise ValueError("generator did not preserve the predeclared scenario seed")
            scenario = copy.deepcopy(scenarios[0])
            original_id = scenario["scenario_id"]
            scenario["scenario_id"] = f"scene_{index:02d}"
            audit = _scenario_audit(scenario, runtime)
            audit["generator_scenario_id"] = original_id
            cases.append(dict(snapshot_id=scenario["scenario_id"], scenario=scenario, scenario_audit=audit))
    return cases


def build_phase2_source(config):
    config, runtime = _resolve_config(config)
    output = config["source_output_path"]
    if output.exists():
        raise FileExistsError(f"retained source will not be overwritten: {output}")
    try:
        source, policy_audit = load_policy_pair(config["policy_source_path"], runtime)
    except PolicySourceUnavailable as exc:
        return dict(status="SOURCE_NOT_AVAILABLE", reason=str(exc), source_output_path=str(output),
                    generated=False, scenario_count=0, environment_steps=0)
    source.cases = build_cases(config["scenario_seeds"], runtime)
    for case in source.cases:
        case["policy_source_audit"] = copy.deepcopy(policy_audit)
    identities = validate_policy_pair(source.old_policy, source.new_policy, runtime)
    require_source_match(source.source_identity, fresh_source_identity(), context="Phase2 builder end check")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_snapshot_source(output, source, runtime)
    return dict(status="READY", schema=SOURCE_SCHEMA, source_output_path=str(output), generated=True,
                scenario_count=len(source.cases), environment_steps=0,
                preparation_environment_steps=source.preparation_environment_steps, **identities)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    result = build_phase2_source(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["generated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
