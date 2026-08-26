"""Validated Phase 1B configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
import copy
from typing import Any, Dict


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "chapter3" / "bser_phase1b.json"
CORRECTED_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "chapter3" / "bser_phase1b1.json"
EXECUTION_RUNTIME_DEFAULTS = {
    "dynamic_public_target_enabled": False,
    "public_target_update_distance": 0.75,
    "public_target_update_min_steps": 20,
    "defer_stale_endpoint_invalid": False,
    "refresh_on_executor_handoff": False,
    "refresh_on_public_target_shift": False,
}


def execution_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(EXECUTION_RUNTIME_DEFAULTS)
    runtime.update(dict(config.get("execution_runtime", {})))
    if float(runtime["public_target_update_distance"]) < 0.0:
        raise ValueError("public_target_update_distance must be non-negative")
    if int(runtime["public_target_update_min_steps"]) < 0:
        raise ValueError("public_target_update_min_steps must be non-negative")
    return runtime


def load_phase1b_config(path: Path | None = None) -> Dict[str, Any]:
    value = json.loads((Path(path) if path else DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if value.get("schema") not in {
        "bser.phase1b.config.v1",
        "bser.phase1b1.config.v1",
        "bser.phase1b2.config.v1",
    } or value.get("formal_training") is not False:
        raise ValueError("invalid BSER Phase 1B configuration")
    if float(value["events"]["belief_shift_threshold"]) != 0.15:
        raise ValueError("frozen belief shift threshold changed")
    version = value.get("mechanism_version", "phase1b_v1")
    if version not in {"phase1b_v1", "phase1b1_corrected", "phase1b2_corrected"}:
        raise ValueError(f"unsupported mechanism_version: {version}")
    if version == "phase1b_v1" and int(value["hysteresis"]["cooldown_steps"]) != 20:
        raise ValueError("frozen cooldown changed")
    if value.get("thresholds_tuned_from_results") is not False:
        raise ValueError("Phase 1B thresholds must not be tuned from results")
    execution_runtime_config(value)
    return value


def load_phase1b1_config(path: Path | None = None) -> Dict[str, Any]:
    value = load_phase1b_config(path or CORRECTED_CONFIG)
    if value.get("mechanism_version") != "phase1b1_corrected":
        raise ValueError("Phase 1B.1 configuration must select corrected mechanism")
    return value


def load_phase1b2_config() -> Dict[str, Any]:
    value = copy.deepcopy(load_phase1b1_config())
    value["schema"] = "bser.phase1b2.config.v1"
    value["mechanism_version"] = "phase1b2_corrected"
    value["execution"] = {
        "path_tracking_threshold": float(value["events"]["waypoint_stale_distance"]),
        "executor_cost_increase_threshold": float(
            value["route_impact"]["planning_cost_relative_threshold"]
        ),
    }
    # Executor-route retry is aligned with the planning-state refresh cadence.
    # This is a protocol correction, not a result-tuned threshold.
    value["hysteresis"]["executor_invalid_cooldown_steps"] = int(
        value["online"]["state_refresh_interval"]
    )
    value["experiment"]["methods"] = [
        "No-BSER-static",
        "Periodic-BSER",
        "Event-BSER-phase1b1",
        "Event-BSER-phase1b2_corrected",
    ]
    return value
