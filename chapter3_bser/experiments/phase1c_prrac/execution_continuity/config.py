"""Frozen variant registry and validation for execution continuity."""

from __future__ import annotations

from typing import Any, Mapping

from .types import ExecutionVariant


CHECKPOINT_RUNTIME_REVISION = "dynamic_public_intercept_v2_1"
OVERLAY_RUNTIME_REVISION = "dynamic_public_intercept_v3_reachable_proxy"
OVERLAY_SCHEMA = "bser.phase1c.prrac.execution_continuity.v1"
EXECUTION_ABLATION_SCHEMA = "bser.phase1c.prrac.execution_ablation.v1"
VARIANT_ORDER = tuple(ExecutionVariant)


def parse_execution_variant(value: str | ExecutionVariant) -> ExecutionVariant:
    try:
        return value if isinstance(value, ExecutionVariant) else ExecutionVariant(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in VARIANT_ORDER)
        raise ValueError(f"unsupported execution variant {value!r}; expected one of {allowed}") from exc


def overlay_enabled(variant: str | ExecutionVariant) -> bool:
    return parse_execution_variant(variant) is not ExecutionVariant.B0_LEGACY_V2_1


def overlay_config(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = dict(config.get("execution_runtime", {}))
    state_refresh_interval = int(
        dict(config.get("execution_continuity", {})).get(
            "state_refresh_interval", config.get("state_refresh_interval", 20)
        )
    )
    if state_refresh_interval <= 0:
        raise ValueError("execution-continuity state_refresh_interval must be positive")
    value = {
        "schema": OVERLAY_SCHEMA,
        "checkpoint_runtime_revision": CHECKPOINT_RUNTIME_REVISION,
        "evaluation_runtime_revision": OVERLAY_RUNTIME_REVISION,
        "state_refresh_interval": state_refresh_interval,
        "public_target_update_distance": float(runtime.get("public_target_update_distance", 0.75)),
        "public_target_update_min_steps": int(runtime.get("public_target_update_min_steps", 20)),
        "executor_cost_increase_threshold": float(
            dict(config.get("execution_continuity", {})).get(
                "executor_cost_increase_threshold", 0.15
            )
        ),
    }
    if value["public_target_update_distance"] < 0.0:
        raise ValueError("public_target_update_distance must be non-negative")
    if value["public_target_update_min_steps"] < 0:
        raise ValueError("public_target_update_min_steps must be non-negative")
    if value["executor_cost_increase_threshold"] < 0.0:
        raise ValueError("executor_cost_increase_threshold must be non-negative")
    return value


__all__ = (
    "CHECKPOINT_RUNTIME_REVISION",
    "EXECUTION_ABLATION_SCHEMA",
    "OVERLAY_RUNTIME_REVISION",
    "OVERLAY_SCHEMA",
    "VARIANT_ORDER",
    "overlay_config",
    "overlay_enabled",
    "parse_execution_variant",
)
