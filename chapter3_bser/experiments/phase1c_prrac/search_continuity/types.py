"""Versioned contracts for read-only SEARCH continuity diagnostics."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SEARCH_CONTINUITY_SCHEMA = "bser.phase1c.prrac.search_continuity.v2"
DEFAULT_SEARCH_CONTINUITY_CONFIG = {
    "schema": SEARCH_CONTINUITY_SCHEMA,
    "enabled": True,
}


def search_continuity_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(DEFAULT_SEARCH_CONTINUITY_CONFIG)
    value.update(dict(config.get("search_continuity_diagnostics", {})))
    if value.get("schema") != SEARCH_CONTINUITY_SCHEMA:
        raise ValueError("invalid search continuity diagnostics schema")
    if not isinstance(value.get("enabled"), bool):
        raise ValueError("search continuity diagnostics enabled must be bool")
    return value


def search_continuity_config_hash(config: Mapping[str, Any]) -> str:
    value = search_continuity_config(config)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def nullable_rate(numerator: int | float, denominator: int | float):
    return None if not denominator else float(numerator) / float(denominator)


__all__ = (
    "DEFAULT_SEARCH_CONTINUITY_CONFIG",
    "SEARCH_CONTINUITY_SCHEMA",
    "nullable_rate",
    "search_continuity_config",
    "search_continuity_config_hash",
)
