"""Shared SEARCH continuity diagnostics and evaluation aggregation."""

from .action_adapter import apply_residual_mode
from .aggregation import (
    aggregate_search_continuity,
    paired_searcher_residual_comparisons,
    search_failure_funnel,
)
from .diagnostics import SearchContinuityDiagnostics
from .types import (
    SEARCH_CONTINUITY_SCHEMA,
    search_continuity_config,
    search_continuity_config_hash,
)

__all__ = (
    "SEARCH_CONTINUITY_SCHEMA",
    "SearchContinuityDiagnostics",
    "aggregate_search_continuity",
    "apply_residual_mode",
    "paired_searcher_residual_comparisons",
    "search_continuity_config",
    "search_continuity_config_hash",
    "search_failure_funnel",
)
