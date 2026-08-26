"""Build the E1-v2 snapshot protocol with metadata isolated from algorithms."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterator, Optional

import numpy as np
import torch

from core.config.ch3_config import build_ch3_config
from core.env import MissionCoreEnv, environment_kwargs_from_config
from core.mapping.planning_state import PlanningStateView, extract_planning_state, planning_state_sha256


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = ROOT / "configs" / "scenarios" / "e0_equivalence"
PROFILES = ("M00_MOVING_CLEAR", "M10_MOVING_UNKNOWN_SINGLE", "M20_MOVING_UNKNOWN_MULTI", "M90_MOVING_KNOWN_ORACLE")
TRACES = ("zero_action", "seeded_random_low_amplitude", "deterministic_scripted")
ENVIRONMENT_MAX_STEPS = 400


@dataclass(frozen=True)
class PlanningSnapshotMetadata:
    profile: str
    scenario_id: str
    scenario_seed: int
    action_trace: str
    requested_snapshot_step: int
    realized_step: int
    obstacle_layout_id: str
    termination_step: Optional[int]
    termination_reason: Optional[str]
    target_found_step: Optional[int]
    mission_complete_step: Optional[int]


@dataclass(frozen=True)
class SnapshotInstance:
    instance_id: str
    state: PlanningStateView
    metadata: PlanningSnapshotMetadata
    unique_state_sha256: str

    @property
    def profile(self): return self.metadata.profile
    @property
    def scenario_id(self): return self.metadata.scenario_id
    @property
    def scenario_seed(self): return self.metadata.scenario_seed
    @property
    def action_trace(self): return self.metadata.action_trace
    @property
    def snapshot_step(self): return self.metadata.requested_snapshot_step


@dataclass(frozen=True)
class SkippedSnapshot:
    instance_id: str
    metadata: PlanningSnapshotMetadata
    reason: str

    @property
    def profile(self): return self.metadata.profile
    @property
    def scenario_id(self): return self.metadata.scenario_id
    @property
    def scenario_seed(self): return self.metadata.scenario_seed
    @property
    def action_trace(self): return self.metadata.action_trace
    @property
    def snapshot_step(self): return self.metadata.requested_snapshot_step


def action_trace(name: str, scenario_seed: int, profile_index: int, max_steps: int = 50) -> np.ndarray:
    if name == "zero_action":
        return np.zeros((max_steps, 4, 3), dtype=np.float32)
    if name == "seeded_random_low_amplitude":
        rng = np.random.default_rng(scenario_seed * 1009 + profile_index * 9176 + 43)
        return rng.uniform(-0.25, 0.25, size=(max_steps, 4, 3)).astype(np.float32)
    step = np.arange(max_steps, dtype=np.float64)[:, None, None]
    agent = np.arange(4, dtype=np.float64)[None, :, None]
    dim = np.arange(3, dtype=np.float64)[None, None, :]
    values = 0.20 * np.sin(0.17 * step + 0.41 * agent + 0.73 * dim) + 0.04 * np.cos(0.11 * step + 0.29 * agent - 0.37 * dim)
    return values.astype(np.float32)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)


def _metadata(profile, scenario_id, seed, trace_name, requested, realized, layout_id,
              termination_step, termination_reason, target_found_step, mission_complete_step):
    return PlanningSnapshotMetadata(
        profile=profile,
        scenario_id=scenario_id,
        scenario_seed=seed,
        action_trace=trace_name,
        requested_snapshot_step=int(requested),
        realized_step=int(realized),
        obstacle_layout_id=layout_id,
        termination_step=termination_step,
        termination_reason=termination_reason,
        target_found_step=target_found_step,
        mission_complete_step=mission_complete_step,
    )


def iter_e1_snapshots(snapshot_steps=(0, 10, 25, 50), *, smoke=False) -> Iterator[object]:
    requested = tuple(sorted({int(value) for value in snapshot_steps}))
    if requested != (0, 10, 25, 50):
        raise ValueError("E1-v2 snapshot protocol must be exactly (0, 10, 25, 50)")
    for profile_index, profile in enumerate(PROFILES[:1] if smoke else PROFILES):
        manifest = json.loads((SCENARIO_ROOT / f"{profile}.json").read_text(encoding="utf-8"))
        entries = manifest["scenarios"][:1] if smoke else manifest["scenarios"]
        config = build_ch3_config("ch3_v3_full_reference", profile)
        env = MissionCoreEnv(**environment_kwargs_from_config(
            config, device="cpu", max_steps=ENVIRONMENT_MAX_STEPS, return_numpy=False
        ))
        try:
            for reset_entry in entries:
                seed = int(reset_entry["scenario_seed"])
                scenario_id = str(reset_entry["scenario_id"])
                for trace_name in (TRACES[:1] if smoke else TRACES):
                    _seed_all(seed)
                    env.reset(scenario=copy.deepcopy(reset_entry))
                    identity = env.get_scenario_identity()
                    layout_id = str(identity["obstacle_layout_id"])
                    actions = action_trace(trace_name, seed, profile_index, max(requested))
                    termination_step = None
                    termination_reason = None
                    target_found_step = None
                    mission_complete_step = None
                    realized_step = 0
                    for protocol_step in range(max(requested) + 1):
                        if protocol_step in requested:
                            instance_id = f"{profile}|{scenario_id}|{trace_name}|step_{protocol_step:03d}"
                            task = env.get_task_state()
                            if task.target_found and target_found_step is None:
                                target_found_step = realized_step
                            if task.mission_complete and mission_complete_step is None:
                                mission_complete_step = realized_step
                            metadata = _metadata(
                                profile, scenario_id, seed, trace_name, protocol_step, realized_step,
                                layout_id, termination_step, termination_reason,
                                target_found_step, mission_complete_step,
                            )
                            if task.mission_complete:
                                yield SkippedSnapshot(instance_id, metadata, "SKIPPED_MISSION_COMPLETE")
                            elif task.target_found:
                                yield SkippedSnapshot(instance_id, metadata, "SKIPPED_TARGET_ALREADY_FOUND")
                            elif termination_step is not None:
                                yield SkippedSnapshot(instance_id, metadata, "SKIPPED_NATURAL_ENV_TERMINATION")
                            else:
                                view = extract_planning_state(env)
                                if int(view.step) != protocol_step:
                                    raise RuntimeError(
                                        f"snapshot step mismatch: requested={protocol_step}, realized={view.step}"
                                    )
                                yield SnapshotInstance(instance_id, view, metadata, planning_state_sha256(view))
                        if protocol_step == max(requested) or termination_step is not None:
                            continue
                        _seed_all(seed * 10_000 + protocol_step)
                        _, _, done = env.step(torch.as_tensor(actions[protocol_step], dtype=torch.float32))
                        realized_step += 1
                        task = env.get_task_state()
                        if task.target_found and target_found_step is None:
                            target_found_step = realized_step
                        if task.mission_complete and mission_complete_step is None:
                            mission_complete_step = realized_step
                        if all(bool(value) for value in done):
                            termination_step = realized_step
                            if task.mission_complete:
                                termination_reason = "MISSION_COMPLETE"
                            elif task.target_found:
                                termination_reason = "TARGET_FOUND"
                            else:
                                termination_reason = "NATURAL_ENV_TERMINATION"
        finally:
            env.close()
