"""Configuration and hashing for the S2-A evaluation overlay."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .types import SearchRecoveryVariant
from .types_v2 import SearchRecoveryVariantV2


SEARCH_COLLISION_RECOVERY_SCHEMA = "bser.phase1c.prrac.search_collision_recovery.v1"
SEARCH_COLLISION_RECOVERY_SCHEMA_V2 = "bser.phase1c.prrac.search_collision_recovery.v2"


def parse_search_recovery_variant(value: str | SearchRecoveryVariant | SearchRecoveryVariantV2):
    if isinstance(value, (SearchRecoveryVariant, SearchRecoveryVariantV2)):
        return value
    for variant_type in (SearchRecoveryVariant, SearchRecoveryVariantV2):
        try:
            return variant_type(str(value))
        except ValueError:
            pass
    raise ValueError(f"unregistered search recovery variant: {value!r}")


def search_collision_recovery_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = dict((config or {}).get("search_collision_recovery", {}))
    schema = str(source.get("schema", SEARCH_COLLISION_RECOVERY_SCHEMA))
    if schema not in {SEARCH_COLLISION_RECOVERY_SCHEMA, SEARCH_COLLISION_RECOVERY_SCHEMA_V2}:
        raise ValueError("invalid search collision recovery schema")
    if schema == SEARCH_COLLISION_RECOVERY_SCHEMA_V2:
        return _search_collision_recovery_config_v2(source)
    result = {
        "schema": SEARCH_COLLISION_RECOVERY_SCHEMA,
        "enabled": bool(source.get("enabled", False)),
        "variants": list(source.get("variants", [SearchRecoveryVariant.S2A_C0_BASELINE.value])),
        "trigger": str(source.get("trigger", "collision_edge")),
        "collision_rearm": str(source.get("collision_rearm", "one_collision_free_search_transition")),
        "c2_escalation": str(source.get("c2_escalation", "collision_after_route_refresh_or_refresh_unreachable")),
        "egress_candidate_policy": str(source.get("egress_candidate_policy", "nearest_hop_known_free_v1")),
        "reuse_project_path_tracker_threshold": bool(source.get("reuse_project_path_tracker_threshold", True)),
        "modify_actions": bool(source.get("modify_actions", False)),
        "modify_executor": bool(source.get("modify_executor", False)),
    }
    variants = tuple(parse_search_recovery_variant(value) for value in result["variants"])
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("search recovery variants must be a non-empty unique registered list")
    result["variants"] = [item.value for item in variants]
    expected = {
        "trigger": "collision_edge",
        "collision_rearm": "one_collision_free_search_transition",
        "c2_escalation": "collision_after_route_refresh_or_refresh_unreachable",
        "egress_candidate_policy": "nearest_hop_known_free_v1",
        "reuse_project_path_tracker_threshold": True,
        "modify_actions": False,
        "modify_executor": False,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(f"unsupported search collision recovery {key}: {result[key]!r}")
    return result


def _search_collision_recovery_config_v2(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema": SEARCH_COLLISION_RECOVERY_SCHEMA_V2,
        "enabled": bool(source.get("enabled", False)),
        "variants": list(source.get("variants", [SearchRecoveryVariantV2.S2A1_C0_BASELINE.value])),
        "trigger": str(source.get("trigger", "collision_edge")),
        "collision_rearm": str(source.get("collision_rearm", "one_collision_free_search_transition")),
        "public_refresh": str(source.get("public_refresh", "force_on_recovery_edge")),
        "local_connector_policy": str(source.get("local_connector_policy", "public_tiered_deterministic_v2")),
        "segment_sample_spacing_fraction": float(source.get("segment_sample_spacing_fraction", 0.5)),
        "activation_diagnostics_schema": str(source.get("activation_diagnostics_schema", "bser.phase1c.prrac.search_collision_recovery.activation.v2")),
        "reuse_project_path_tracker_threshold": bool(source.get("reuse_project_path_tracker_threshold", True)),
        "modify_actions": bool(source.get("modify_actions", False)),
        "modify_executor": bool(source.get("modify_executor", False)),
    }
    variants = tuple(parse_search_recovery_variant(value) for value in result["variants"])
    allowed = {
        SearchRecoveryVariantV2.S2A1_C0_BASELINE,
        SearchRecoveryVariantV2.S2A1_C1_FORCED_REFRESH,
        SearchRecoveryVariantV2.S2A1_C2_LOCAL_CONNECTOR,
    }
    if not variants or len(set(variants)) != len(variants) or any(item not in allowed for item in variants):
        raise ValueError("v2 search recovery variants must be a non-empty unique S2-A.1 list")
    result["variants"] = [item.value for item in variants]
    expected = {
        "trigger": "collision_edge",
        "collision_rearm": "one_collision_free_search_transition",
        "public_refresh": "force_on_recovery_edge",
        "local_connector_policy": "public_tiered_deterministic_v2",
        "segment_sample_spacing_fraction": 0.5,
        "activation_diagnostics_schema": "bser.phase1c.prrac.search_collision_recovery.activation.v2",
        "reuse_project_path_tracker_threshold": True,
        "modify_actions": False,
        "modify_executor": False,
    }
    for key, value in expected.items():
        if result[key] != value:
            raise ValueError(f"unsupported S2-A.1 search collision recovery {key}: {result[key]!r}")
    return result


def search_collision_recovery_config_hash(value: Mapping[str, Any]) -> str:
    body = search_collision_recovery_config({"search_collision_recovery": dict(value)})
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


__all__ = (
    "SEARCH_COLLISION_RECOVERY_SCHEMA",
    "SEARCH_COLLISION_RECOVERY_SCHEMA_V2",
    "parse_search_recovery_variant",
    "search_collision_recovery_config",
    "search_collision_recovery_config_hash",
)
