"""Real simulator continuation with an explicit, versioned object-graph snapshot.

Snapshots contain simulator/controller state only. They never contain a policy,
optimizer or replay. Persistent references preserve wrapper/provider ownership;
the navigation closure is re-bound once without installing or advancing guidance.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import io
import pickle
import random
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch

from chapter3_bser.controllers.state_provider import OnlinePlanningStateProvider
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 import _make_base_env, _public_context
from chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.training_env import Phase1CV2TrainingEnv
from chapter3_bser.experiments.phase1c_prrac.runtime_factory import build_prrac_online_controller
from chapter3_bser.experiments.phase1c_prrac.training_env import PRRACTrainingEnv
from chapter3_bser.integration.guided_env import GuidedEnv
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from chapter3_bser.online.config import load_phase1b2_config, execution_runtime_config
from core.env.task_protocol import strict_terminal
from .provenance import source_identity
from chapter3_bser.models.hgr.phase1 import (
    phase1_options, runtime_contract, behavior_identity, identical_behavior,
    isolated_global_rng, audit_local_generators,
)

SNAPSHOT_SCHEMA = "hgr.full_decision_state.v1"
FEATURE_DIM = 112 + 12 + 12 + 12 + 5


def seed_innovations(seed, *, cpu_only=False):
    random.seed(seed)
    np.random.seed(np.random.SeedSequence(seed).generate_state(4))
    if cpu_only:
        torch.random.default_generator.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])


@dataclass(frozen=True)
class DecisionSnapshot:
    schema: str
    sha256: str
    payload: bytes
    step: int
    max_steps: int
    policy_identity: dict

    def validate(self):
        if self.schema != SNAPSHOT_SCHEMA or hashlib.sha256(self.payload).hexdigest() != self.sha256:
            raise ValueError("snapshot schema or bytes mismatch")
        if not 0 <= self.step < self.max_steps:
            raise ValueError("snapshot has no legal continuation horizon")

    def save(self, path):
        from pathlib import Path
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        self.validate()
        path.write_bytes(pickle.dumps(self, protocol=5))

    @classmethod
    def load(cls, path):
        from pathlib import Path
        value = pickle.loads(Path(path).read_bytes())
        if not isinstance(value, cls):
            raise ValueError("not a decision snapshot")
        value.validate()
        return value


class _GraphPickler(pickle.Pickler):
    def __init__(self, stream, roots):
        super().__init__(stream, protocol=5)
        self.references = {id(value): name for name, value in roots.items()}

    def persistent_id(self, value):
        name = self.references.get(id(value))
        return None if name is None else ("root", name)


class _GraphUnpickler(pickle.Unpickler):
    def __init__(self, stream, roots):
        super().__init__(stream)
        self.roots = roots

    def persistent_load(self, identity):
        kind, name = identity
        if kind != "root" or name not in self.roots:
            raise ValueError("unknown snapshot reference")
        return self.roots[name]


class MissionRuntime:
    def __init__(self, config, scenario, *, seed, episode_id=0):
        self.config = copy.deepcopy(config)
        self.scenario = copy.deepcopy(scenario)
        phase1 = bool(phase1_options(config))
        seed_innovations(seed, cpu_only=phase1)
        base = _make_base_env(config, device="cpu")
        guided = GuidedEnv(base, enabled=True)
        v2 = Phase1CV2TrainingEnv(guided, reward_config=config["reward"])
        self.env = PRRACTrainingEnv(v2, reward_objective_config=config, gamma=config["rl"]["gamma"])
        self.env.reset(scenario=scenario, episode_id=episode_id, episode_index=episode_id)
        # reset may use the scenario planner seed; future innovation seed is distinct.
        seed_innovations(seed, cpu_only=phase1)
        self.action_rng = torch.Generator().manual_seed(seed)
        phase_config = load_phase1b2_config()
        runtime_config = execution_runtime_config(config)
        phase_config["execution_runtime"] = copy.deepcopy(runtime_config)
        self.provider = OnlinePlanningStateProvider(
            self.env, refresh_interval=int(phase_config["online"]["state_refresh_interval"]),
            **{key: runtime_config[key] for key in ("refresh_on_executor_handoff", "refresh_on_public_target_shift", "public_target_update_distance", "public_target_update_min_steps")})
        self.state = self.provider.initialize()
        context = _public_context(self.env, self.state)
        self.controller = build_prrac_online_controller(phase_config, config)
        initialized = self.controller.initialize(self.state, context)
        self.bridge = RMADDPGGuidanceBridge()
        self.guidance = self.bridge.compile_guidance(initialized.allocation, self.state, context, decision_reason="INITIALIZE")
        self.env.install_guidance(self.guidance)
        self.observations = self.env.refresh_observation_after_guidance()
        self.found_step = None
        self.handoff_event_step = None
        self.handoff_decision_step = None
        self._record_events()

    @property
    def step(self):
        return int(self.env.get_task_state().step)

    @property
    def terminal(self):
        return strict_terminal(self.env)

    @property
    def suffix(self):
        # This is the Executor's legal local knowledge, not global Found.
        known = self.env.get_search_execution_state().target_known_by_agent[3]
        return bool(known) and not self.terminal

    def _record_events(self):
        task = self.env.get_task_state()
        if task.target_found and self.found_step is None:
            self.found_step = int(self.env.unwrapped.found_step)
        if task.executor_knows_target and self.handoff_event_step is None:
            self.handoff_event_step = task.handoff_step
        if self.suffix and self.handoff_decision_step is None:
            self.handoff_decision_step = self.step

    def features(self):
        """Only existing local observations and public views; no target truth."""
        agents = self.env.get_agent_state()
        mapping = self.env.get_mapping_state()
        values = np.concatenate([
            np.asarray([torch.as_tensor(o).cpu().numpy() for o in self.observations]).reshape(-1),
            np.asarray(agents.positions).reshape(-1), np.asarray(agents.velocities).reshape(-1),
            np.asarray(agents.navigation_targets).reshape(-1),
            np.asarray([(self.config["max_steps"] - self.step) / self.config["max_steps"],
                        mapping.occupancy_known_ratio, mapping.target_belief_entropy,
                        mapping.target_belief_peak, mapping.map_revision])]).astype(np.float32)
        if values.shape != (FEATURE_DIM,) or not np.isfinite(values).all():
            raise ValueError("invalid public boundary features")
        return values

    def advance(self, policy, *, deterministic=False, standard_noise=None):
        if self.terminal:
            raise RuntimeError("cannot step a terminal mission")
        before = self.step
        suffix = self.suffix
        phase1 = (self.config.get("phase1") or {}).get("enabled", False)
        if phase1 and self.handoff_decision_step is not None and not suffix:
            raise RuntimeError("executor knowledge regressed after reliable handoff")
        active = [not bool(x) for x in self.env.get_search_execution_state().agent_finished]
        observations = [torch.as_tensor(o).detach().cpu().numpy().copy() for o in self.observations]
        optional_noise = {} if standard_noise is None else dict(standard_noise=standard_noise)
        actions, latents = policy.actions(observations, suffix=suffix, active=active,
                                          generator=self.action_rng, deterministic=deterministic, **optional_noise)
        step_obs, rewards, dones = self.env.step(actions)
        if self.terminal:
            self.observations = step_obs
        else:
            self.state = self.provider.snapshot(force=False)
            context = _public_context(self.env, self.state)
            result = self.controller.step(self.state, context)
            self.env.observe_controller_result(result, controller=self.controller, state_provider=self.provider)
            self.guidance = self.bridge.compile_guidance(result.allocation, self.state, context, decision_reason=result.decision_reason)
            self.env.install_guidance(self.guidance)
            self.observations = self.env.refresh_observation_after_guidance()
        self._record_events()
        if self.step != before + 1 or self.terminal != all(bool(d) for d in dones):
            raise RuntimeError("mission clock/done contract mismatch")
        record = dict(t=before, observations=observations, actions=actions.cpu().numpy().copy(),
                    latents=latents, suffix=suffix, **copy.deepcopy(self.env.reward_accounting.last),
                    dones=list(dones), next_observations=[torch.as_tensor(o).cpu().numpy().copy() for o in self.observations])
        if phase1:
            for value in (record["observations"], record["next_observations"], record["actions"], record["team_reward"]):
                if not np.isfinite(np.asarray(value)).all():
                    raise ValueError("invalid raw runtime state/action/reward")
            for i, latent in enumerate(latents):
                if (latent is not None) != (active[i] and not deterministic):
                    raise RuntimeError("active-agent latent score coverage mismatch")
            state = self.env.get_agent_state()
            record["public_positions"] = np.asarray(state.positions).copy()
            record["public_velocities"] = np.asarray(state.velocities).copy()
        return record

    def snapshot(self, policy_identity):
        if self.terminal or not self.suffix:
            raise ValueError("snapshot requires a live reliable-handoff decision")
        roots = dict(runtime=self, env=self.env, v2=self.env.env, guided=self.env.env.env,
                     mission=self.env.env.env.env, physics=self.env.unwrapped,
                     provider=self.provider, controller=self.controller, bridge=self.bridge)
        states = {name: dict(vars(obj)) for name, obj in roots.items()}
        # Reconstitute only the navigation closure, never re-install guidance.
        states["physics"].pop("_update_nav_targets", None)
        states["guided"]["_original_update_nav_targets"] = None
        classes = {name: (type(obj).__module__, type(obj).__qualname__) for name, obj in roots.items()}
        stream = io.BytesIO()
        _GraphPickler(stream, roots).dump(dict(states=states, rng=rng_state()))
        payload = pickle.dumps(dict(classes=classes, graph=stream.getvalue(),
                                    source_sha256=source_identity()["sha256"],
                                    fields={name: sorted(state) for name, state in states.items()}), protocol=5)
        identity = dict(policy_identity)
        if phase1_options(self.config):
            identity["phase1_boundary_features_sha256"] = hashlib.sha256(self.features().tobytes()).hexdigest()
        snapshot = DecisionSnapshot(SNAPSHOT_SCHEMA, hashlib.sha256(payload).hexdigest(), payload,
                                    self.step, int(self.config["max_steps"]), identity)
        snapshot.validate()
        return snapshot

    @classmethod
    def restore(cls, snapshot, *, innovation_seed=None):
        snapshot.validate()
        header = pickle.loads(snapshot.payload)
        if header.get("source_sha256") != source_identity()["sha256"]:
            raise ValueError("snapshot production source version mismatch")
        roots = {}
        for name, (module, qualname) in header["classes"].items():
            if not module.startswith(("chapter3_bser.", "core.")) or "." in qualname:
                raise ValueError("unsupported snapshot component")
            component = getattr(importlib.import_module(module), qualname)
            roots[name] = component.__new__(component)
        data = _GraphUnpickler(io.BytesIO(header["graph"]), roots).load()
        for name, state in data["states"].items():
            if sorted(state) != header["fields"][name]:
                raise ValueError("snapshot component field inventory mismatch")
            roots[name].__dict__.update(state)
        runtime = roots["runtime"]
        guided = roots["guided"]
        guided._install_navigation_hook()
        if (runtime.env is not roots["env"] or runtime.provider.env is not runtime.env
                or guided._runtime is not roots["physics"] or runtime.bridge is not roots["bridge"]
                or runtime.step != snapshot.step or runtime.config["max_steps"] != snapshot.max_steps
                or runtime.env.unwrapped.max_steps != snapshot.max_steps
                or not runtime.suffix or runtime.terminal):
            raise ValueError("restored ownership, clock or boundary contract mismatch")
        restore_rng(data["rng"])
        if innovation_seed is not None:
            # Deterministic target/flow process state stays intact. Only future
            # random innovations and policy samples receive a fresh child stream.
            seed_innovations(innovation_seed, cpu_only=bool(phase1_options(runtime.config)))
            runtime.action_rng.manual_seed(innovation_seed)
        return runtime

    def close(self):
        self.env.close()


def collect_trajectory(config, scenario, policy, *, seed, episode_id=0, stop_at_boundary=False,
                       deterministic=False):
    options = phase1_options(config)
    with isolated_global_rng() if options else nullcontext():
        return _collect_trajectory(config, scenario, policy, seed=seed, episode_id=episode_id,
                                   stop_at_boundary=stop_at_boundary, deterministic=deterministic,
                                   phase1=bool(options))


def _collect_trajectory(config, scenario, policy, *, seed, episode_id, stop_at_boundary,
                        deterministic, phase1):
    from chapter3_bser.models.hgr.policy import weights_hash
    theta_hash = weights_hash(policy.theta_minus)
    identity = {} if not phase1 else dict(
        theta_behavior_sha256=behavior_identity(policy.theta_minus, runtime_contract(config)).sha256,
        suffix_behavior_sha256=behavior_identity(policy.phi, runtime_contract(config)).sha256)
    runtime = MissionRuntime(config, scenario, seed=seed, episode_id=episode_id)
    records, boundary, features = [], None, None
    try:
        if phase1:
            audit_local_generators(runtime)
        while not runtime.terminal:
            if runtime.suffix and boundary is None:
                boundary = runtime.snapshot(dict(theta=weights_hash(policy.theta_minus), phi=weights_hash(policy.phi)))
                features = runtime.features()
                if stop_at_boundary:
                    break
            records.append(runtime.advance(policy, deterministic=deterministic))
        summary = runtime.env.finalize_episode()
        summary.update(scenario_id=scenario["scenario_id"],
                       scenario_seed=scenario["scenario_seed"], actual_length=runtime.step,
                       found_step=runtime.found_step, handoff_event_step=runtime.handoff_event_step,
                       handoff_decision_step=runtime.handoff_decision_step,
                       remaining_task_steps=config["max_steps"] - runtime.step)
        return dict(records=records, rewards=[r["team_reward"] for r in records],
                    theta_hash=theta_hash, consumed=False,
                    tau=None if boundary is None else boundary.step, snapshot=boundary,
                    features=features, summary=summary,
                    **({**identity, "trajectory_complete": runtime.terminal} if phase1 else {}))
    finally:
        runtime.close()


def continue_branch(snapshot, policy, *, seed=None, stream_id="restore_check", keep_records=False,
                    pair_noise=None):
    if pair_noise is not None and seed is not None:
        raise ValueError("pair_noise owns the branch innovation seeds")
    with isolated_global_rng() if pair_noise is not None else nullcontext():
        return _continue_branch(snapshot, policy, seed=seed, stream_id=stream_id,
                                keep_records=keep_records, pair_noise=pair_noise)


def _continue_branch(snapshot, policy, *, seed, stream_id, keep_records, pair_noise):
    if pair_noise is not None:
        seed = pair_noise.environment_seed
        stream_id = pair_noise.public()["trace_id"]
    runtime = MissionRuntime.restore(snapshot, innovation_seed=seed)
    records, value, discount = [], 0.0, 1.0
    gamma = runtime.config["rl"]["gamma"]
    try:
        noise = None
        trace = hashlib.sha256()
        extra = {}
        if pair_noise is not None:
            if phase1_options(runtime.config) is None:
                raise ValueError("addressed branch noise requires an opt-in phase1 snapshot")
            boundary_hash = hashlib.sha256(runtime.features().tobytes()).hexdigest()
            if boundary_hash != snapshot.policy_identity.get("phase1_boundary_features_sha256"):
                raise ValueError("restored legal boundary features differ from the captured decision")
            noise = pair_noise.table(snapshot.step, snapshot.max_steps)
            identity = behavior_identity(policy.phi, runtime_contract(runtime.config))
            extra = dict(**pair_noise.public(), rng_inventory=audit_local_generators(runtime),
                         suffix_behavior=identity.public(), original_deadline=snapshot.max_steps,
                         initial_task_step=runtime.step, restored_boundary_features_sha256=boundary_hash)
        while not runtime.terminal:
            if noise is None:
                record = runtime.advance(policy)
            else:
                record = runtime.advance(policy, standard_noise=noise[runtime.step - snapshot.step])
                # Stable primitive-array trace, independent of pickle object IDs.
                for key in ("t", "suffix", "actions", "observations", "next_observations",
                            "team_reward", "dones", "public_positions", "public_velocities"):
                    if key in record:
                        array = np.asarray(record[key])
                        trace.update(key.encode())
                        trace.update(str((array.dtype, array.shape)).encode())
                        trace.update(array.tobytes())
            value += discount * record["team_reward"]; discount *= gamma
            if keep_records:
                records.append(record)
        from chapter3_bser.models.hgr.policy import weights_hash
        if pair_noise is not None:
            if not np.isfinite(value):
                raise ValueError("nonfinite formal branch return")
            if not identical_behavior(identity, behavior_identity(policy.phi, runtime_contract(runtime.config))):
                raise RuntimeError("suffix behavior changed during branch")
            extra["trajectory_trace_sha256"] = trace.hexdigest()
        return dict(G_plus=value, steps=runtime.step - snapshot.step,
                    termination_reason=runtime.env.get_episode_result()["termination_reason"],
                    policy_hash=weights_hash(policy.phi), snapshot_hash=snapshot.sha256,
                    random_stream_id=stream_id, random_seed=seed, gamma=gamma,
                    terminal_step=runtime.step, records=records, **extra)
    finally:
        runtime.close()
